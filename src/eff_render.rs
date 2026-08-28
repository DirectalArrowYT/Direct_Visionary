//! GPU side of the effect runtime: drawing particles into the model viewport.
//!
//! Deliberately its own pass rather than anything routed through `ssbh_wgpu`. That renderer
//! draws fighter models — lit, depth-written, PBR — and particles are none of those things:
//! they are unlit camera-facing quads, usually additive, that read depth without writing it so
//! they neither occlude each other nor punch holes in the fighter. Trying to express that
//! through the model path would mean fighting every one of its assumptions.
//!
//! ## What this can and cannot reproduce
//!
//! Smash's own particle shaders ship as `BNSH` — compiled Maxwell GPU binaries. They can be
//! parsed but not executed off a Switch, so the combiner here is a reimplementation, not the
//! game's code. For the shapes effects actually take — a texture masked and tinted by the
//! emitter's colour keys, blended additively or by alpha — that matches. It will not match an
//! effect that relies on a bespoke shader program.
//!
//! The textures are the other half of that. Every effect texture sampled from the real dump is
//! `BC5Unorm`, a **two-channel** format: decoded to RGBA it carries red and green with blue at
//! zero. So the texture is not an RGB image to be drawn as-is — red is the mask, and the colour
//! comes from the emitter. A shader that multiplied the sampled RGB by the tint would render
//! every effect with no blue in it.

use wgpu::util::DeviceExt;

/// One particle, as the GPU consumes it. Kept to a compact instance layout because a single
/// move can have several thousand alive at once and the buffer is rewritten every frame.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleInstance {
    /// World position of the particle's centre.
    pub position: [f32; 3],
    /// Half-extent in world units. One value: these are square billboards.
    pub size: f32,
    /// Straight (non-premultiplied) tint. Alpha is applied by the blend state, so an additive
    /// particle still fades out through it rather than needing a separate opacity path.
    pub color: [f32; 4],
    /// Rotation about the view axis, radians.
    pub rotation: f32,
    pub _padding: [f32; 3],
}

/// A run of particles sharing one texture and one blend mode — the unit of a draw call.
pub struct ParticleBatch {
    pub texture: TextureKey,
    pub additive: bool,
    pub instances: Vec<ParticleInstance>,
}

/// Identifies an uploaded texture: which `.eff` it came from and its index in that file's pool.
/// Effects from the fighter file and from `ef_common` routinely share pool indices, so the file
/// has to be part of the key or one would silently render with the other's texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub file: std::path::PathBuf,
    pub index: usize,
}

struct UploadedTexture {
    bind_group: wgpu::BindGroup,
}

pub struct ParticleRenderer {
    pipeline_alpha: wgpu::RenderPipeline,
    pipeline_additive: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    textures: std::collections::HashMap<TextureKey, UploadedTexture>,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    /// (offset, count, texture, additive) for each batch queued this frame.
    draws: Vec<(u32, u32, TextureKey, bool)>,
}

/// Camera data the particle shader needs. A separate, smaller uniform than the model
/// renderer's: billboards need the view axes to face the camera and the view-projection to
/// place themselves, and nothing else.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleCamera {
    view_projection: [[f32; 4]; 4],
    /// Camera right and up in world space, so the vertex stage can build a facing quad without
    /// inverting a matrix per vertex.
    camera_right: [f32; 4],
    camera_up: [f32; 4],
}

const SHADER: &str = r#"
struct Camera {
    view_projection: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var particle_texture: texture_2d<f32>;
@group(1) @binding(1) var particle_sampler: sampler;

struct Instance {
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) rotation: f32,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, instance: Instance) -> VertexOut {
    // Two triangles as a strip: (-1,-1) (1,-1) (-1,1) (1,1).
    let corner = vec2<f32>(
        select(-1.0, 1.0, (vertex_index & 1u) == 1u),
        select(-1.0, 1.0, (vertex_index & 2u) == 2u),
    );

    let s = sin(instance.rotation);
    let c = cos(instance.rotation);
    let spun = vec2<f32>(corner.x * c - corner.y * s, corner.x * s + corner.y * c);

    let offset = camera.camera_right.xyz * spun.x * instance.size
               + camera.camera_up.xyz * spun.y * instance.size;

    var out: VertexOut;
    out.clip_position = camera.view_projection * vec4<f32>(instance.position + offset, 1.0);
    out.uv = corner * 0.5 + 0.5;
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let sampled = textureSample(particle_texture, particle_sampler, in.uv);
    // BC5 gives two channels: red carries the particle's shape. The colour is the emitter's,
    // not the texture's — sampling RGB here would render every effect missing its blue.
    let mask = sampled.r;
    return vec4<f32>(in.color.rgb * mask, in.color.a * mask);
}
"#;

impl ParticleRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle camera layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle camera"),
            size: std::mem::size_of::<ParticleCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&texture_layout)],
            immediate_size: 0,
        });

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 32,
                    shader_location: 3,
                },
            ],
        };

        // Alpha and additive differ only in the colour blend factor, so they are the same
        // pipeline twice rather than a branch in the shader: the blend state is fixed function
        // and cannot be chosen per draw.
        let make_pipeline = |label: &str, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[instance_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                // No depth attachment: this pass runs over egui's colour target, which has
                // none. Particles are therefore drawn over the model rather than intersecting
                // it — correct for the overwhelming majority of effects, which sit in front of
                // the fighter, and the reason a depth-aware pass is a later step rather than
                // this one.
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline_alpha = make_pipeline(
            "particle alpha",
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
        );
        let pipeline_additive = make_pipeline(
            "particle additive",
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("particle sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let instance_capacity = 4096;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle instances"),
            size: (instance_capacity * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_alpha,
            pipeline_additive,
            camera_bind_group,
            camera_buffer,
            texture_layout,
            sampler,
            textures: std::collections::HashMap::new(),
            instance_buffer,
            instance_capacity,
            draws: Vec::new(),
        }
    }

    /// Whether a texture is already on the GPU, so the caller can skip decoding it again.
    pub fn has_texture(&self, key: &TextureKey) -> bool {
        self.textures.contains_key(key)
    }

    /// Upload one decoded effect texture. Idempotent: uploading a key twice keeps the first.
    pub fn upload_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TextureKey,
        image: &image::RgbaImage,
    ) {
        if self.textures.contains_key(&key) {
            return;
        }
        let size = wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("particle texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Unorm rather than Srgb: the sampled channel is a mask, not colour, so it
                // must not go through an sRGB curve on the way in.
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            image.as_raw(),
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle texture bind group"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.textures.insert(key, UploadedTexture { bind_group });
    }

    /// Stage this frame's particles. Call once per frame, before `draw`.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        camera_right: glam::Vec3,
        camera_up: glam::Vec3,
        batches: &[ParticleBatch],
    ) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&ParticleCamera {
                view_projection: view_projection.to_cols_array_2d(),
                camera_right: camera_right.extend(0.0).to_array(),
                camera_up: camera_up.extend(0.0).to_array(),
            }),
        );

        self.draws.clear();
        let total: usize = batches.iter().map(|batch| batch.instances.len()).sum();
        if total == 0 {
            return;
        }
        if total > self.instance_capacity {
            self.instance_capacity = total.next_power_of_two();
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle instances"),
                size: (self.instance_capacity * std::mem::size_of::<ParticleInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let mut packed: Vec<ParticleInstance> = Vec::with_capacity(total);
        for batch in batches {
            if batch.instances.is_empty() || !self.textures.contains_key(&batch.texture) {
                continue;
            }
            let offset = packed.len() as u32;
            packed.extend_from_slice(&batch.instances);
            self.draws.push((
                offset,
                batch.instances.len() as u32,
                batch.texture.clone(),
                batch.additive,
            ));
        }
        if !packed.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&packed));
        }
    }

    /// Draw what `prepare` staged. Safe to call with nothing staged.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.draws.is_empty() {
            return;
        }
        render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for (offset, count, key, additive) in &self.draws {
            let Some(texture) = self.textures.get(key) else {
                continue;
            };
            render_pass.set_pipeline(if *additive {
                &self.pipeline_additive
            } else {
                &self.pipeline_alpha
            });
            render_pass.set_bind_group(1, &texture.bind_group, &[]);
            render_pass.draw(0..4, *offset..(*offset + *count));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instance layout is declared twice — once as a Rust struct and once as byte offsets
    /// in the vertex attributes — and nothing checks them against each other at compile time.
    /// A mismatch does not fail to build or to run: it renders particles at wrong positions or
    /// wrong colours, which reads as a simulation bug and is nearly impossible to trace back
    /// to a struct offset.
    #[test]
    fn the_instance_layout_matches_the_shader_attribute_offsets() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(offset_of!(ParticleInstance, position), 0);
        assert_eq!(offset_of!(ParticleInstance, size), 12);
        assert_eq!(offset_of!(ParticleInstance, color), 16);
        assert_eq!(offset_of!(ParticleInstance, rotation), 32);
        // 48, not 36: the padding exists so the stride stays 16-byte aligned. Dropping it
        // would silently misalign every instance after the first.
        assert_eq!(size_of::<ParticleInstance>(), 48);
        assert_eq!(align_of::<ParticleInstance>(), 4);
    }

    #[test]
    fn the_camera_uniform_is_sized_as_the_shader_expects() {
        // mat4 (64) + two vec4 (32).
        assert_eq!(std::mem::size_of::<ParticleCamera>(), 96);
    }
}
