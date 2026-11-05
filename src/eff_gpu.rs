//! Drawing effects with the game's own shaders.
//!
//! Every emitter names one shader variation of its file (`shader_references.shader_index`): a Maxwell vertex and
//! fragment program in the file's BNSH. `eff_render` approximates what those programs do; this runs them. The programs
//! are translated to WGSL by Visionary's own translator (`mx`, through eff_shader_cache; older caches hold
//! tools/eft_shaders.py's SPIR-V, which still loads), with this binding layout:
//!
//! * set 0: the vertex stage's constant buffer `c<n>` at binding n, the fragment stage's at 20 + n, each a read-only
//!   storage buffer of `uint`; the alpha test / point size block at 19 (vertex) and 39 (fragment).
//! * set 1: fragment texture unit u at binding u, vertex at 32 + u; each texture's sampler 64 above it.
//! * vertex input: generic attribute n at location n, always read as a `vec4`.
//!
//! The constant buffers are the game's own memory, byte for byte. What the shaders read, by NVN slot (`c<slot + 3>`):
//! 0 (c3), 1 (c4), 5 (c8) the camera, 6 (c9) the emitter block (header 0x50 + the emitter's static data),
//! 7 (c10), 8 (c11), 10 (c13), 11 (c14), 12 (c15), 13 (c16). c1 is the program's own constant data, c2 the driver's
//! texture table. `eff_capture` holds them as the game wrote them; matching each one from Visionary's own state is
//! the remaining work, and a capture is what every such input is checked against.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use wgpu::util::DeviceExt;

pub const VERTEX: u32 = 0;
pub const FRAGMENT: u32 = 1;

const FRAGMENT_CBUF_BASE: u32 = 20;
const MXPC_BINDING: [u32; 2] = [19, 39];
const SAMPLER_SPLIT: u32 = 64;
/// c2, the driver's texture table (handles, then texture sizes from word 96): no translated effect shader reads it,
/// but the fragment stages declare it.
const TEXTURE_TABLE_BYTES: usize = 896;

/// A texture unit one stage samples, and the binding of its texture (the sampler is `SAMPLER_SPLIT` above).
#[derive(Clone, Copy, Debug)]
pub struct SamplerSlot {
    pub unit: u32,
    pub binding: u32,
}

/// What one stage of a variation reads.
#[derive(Clone, Debug)]
pub struct StageInfo {
    /// Bit n: the stage reads constant buffer c<n>.
    pub cbufs: u32,
    /// Bit n: the stage reads generic input n (vertex) or interpolant n (fragment).
    pub inputs: u32,
    pub samplers: Vec<SamplerSlot>,
    /// c1: the program's constant data, from its code section.
    pub constants: Vec<u8>,
    /// The program's reflection (the BNSH's names): vertex inputs by name -> location (`sysLocalPosAttr` ...),
    /// samplers by name -> texture unit (`sysTextureSampler0`, `sysDepthBufferTexture` ...), and constant buffers by
    /// name -> NVN slot (`sysEmitterDynamicUniformBlock` ...; the shader reads slot n as c(n + 3)).
    pub reflection: Reflection,
    code_hash: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Reflection {
    pub inputs: Vec<(String, i32)>,
    pub samplers: Vec<(String, i32)>,
    pub constant_buffers: Vec<(String, i32)>,
}

impl Reflection {
    fn read(path: &Path) -> Self {
        let Ok(text) = std::fs::read(path) else { return Self::default() };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&text) else { return Self::default() };
        let list = |key: &str| -> Vec<(String, i32)> {
            json[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|p| Some((p[0].as_str()?.to_string(), p[1].as_i64()? as i32)))
                        .collect()
                })
                .unwrap_or_default()
        };
        Self { inputs: list("inputs"), samplers: list("samplers"), constant_buffers: list("constant_buffers") }
    }

    pub fn input(&self, name: &str) -> Option<u32> {
        self.inputs.iter().find(|(n, _)| n == name).and_then(|&(_, l)| u32::try_from(l).ok())
    }

    pub fn sampler(&self, name: &str) -> Option<u32> {
        self.samplers.iter().find(|(n, _)| n == name).and_then(|&(_, u)| u32::try_from(u).ok())
    }

    /// The shader's c-index of a named constant buffer.
    pub fn constant_buffer(&self, name: &str) -> Option<u32> {
        self.constant_buffers.iter().find(|(n, _)| n == name).and_then(|&(_, s)| u32::try_from(s + 3).ok())
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn le_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// The translated shaders of one effect file: Visionary's shader dump (`v<i>_s<stage>.code` / `.ctl`, test
/// `dump_shader_variations`) with its translation beside it: `wgsl/` (mx) or eft_shaders.py's `spv/`.
pub struct ShaderLibrary {
    dir: PathBuf,
    /// Visionary's own translation (`wgsl/`, eff_shader_cache) rather than eft_shaders.py's SPIR-V (`spv/`).
    wgsl: bool,
    stages: HashMap<(u32, u32), StageInfo>,
    by_code: HashMap<(u32, u64), Vec<u32>>,
}

impl ShaderLibrary {
    pub fn open(dir: &Path) -> Result<Self> {
        let wgsl = dir.join("wgsl").join("index.json").exists();
        let index_path = dir.join(if wgsl { "wgsl" } else { "spv" }).join("index.json");
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&index_path).with_context(|| format!("reading {}", index_path.display()))?,
        )?;
        let variations = index["variations"].as_object().ok_or_else(|| anyhow!("{}: no variations", index_path.display()))?;
        let mut stages = HashMap::new();
        let mut by_code: HashMap<(u32, u64), Vec<u32>> = HashMap::new();
        for (key, v) in variations {
            let variation: u32 = key.parse()?;
            for (stage, name, file_stage) in [(VERTEX, "vs", 0), (FRAGMENT, "fs", 4)] {
                let s = &v[name];
                if s["ok"].as_bool() != Some(true) {
                    continue;
                }
                let base = dir.join(format!("v{variation:04}_s{file_stage}"));
                // A stage whose dump is missing is left out (the variation is then not drawable), as is one the
                // translation failed on.
                let (Ok(code), Ok(ctl)) = (std::fs::read(base.with_extension("code")), std::fs::read(base.with_extension("ctl"))) else {
                    continue;
                };
                let (len, c_size, c_off) = (
                    le_u32(&ctl, 0x6f8).unwrap_or(0) as usize,
                    le_u32(&ctl, 0x6fc).unwrap_or(0) as usize,
                    le_u32(&ctl, 0x700).unwrap_or(0) as usize,
                );
                // The program as the game binds it: `len` bytes from the program header (+0x30 of the section) on,
                // header included. A captured program is matched by exactly these bytes (`Program::bytes`).
                let program = code.get(0x30..0x30 + len).ok_or_else(|| anyhow!("{}: short code", base.display()))?;
                let info = StageInfo {
                    cbufs: s["cbufs"].as_u64().unwrap_or(0) as u32,
                    inputs: s["in"].as_u64().unwrap_or(0) as u32,
                    samplers: s["samplers"]
                        .as_array()
                        .map(|list| {
                            list.iter()
                                .map(|x| SamplerSlot {
                                    unit: x["unit"].as_u64().unwrap_or(0) as u32,
                                    binding: x["binding"].as_u64().unwrap_or(0) as u32,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    constants: code.get(c_off..c_off + c_size).map(|c| c.to_vec()).unwrap_or_default(),
                    reflection: Reflection::read(&dir.join(format!("v{variation:04}_s{file_stage}.reflect.json"))),
                    code_hash: fnv(program),
                };
                by_code.entry((stage, info.code_hash)).or_default().push(variation);
                stages.insert((variation, stage), info);
            }
        }
        Ok(Self { dir: dir.to_path_buf(), wgsl, stages, by_code })
    }

    pub fn stage(&self, variation: u32, stage: u32) -> Option<&StageInfo> {
        self.stages.get(&(variation, stage))
    }

    /// A variation whose two programs are these (as `eff_capture::Program::bytes` gives them). Programs are shared
    /// between variations, so this is the first of the equivalent ones.
    pub fn variation_for_programs(&self, vs: &[u8], fs: &[u8]) -> Option<u32> {
        let vs = self.by_code.get(&(VERTEX, fnv(vs)))?;
        let fs = self.by_code.get(&(FRAGMENT, fnv(fs)))?;
        vs.iter().copied().filter(|v| fs.contains(v)).min()
    }

    /// Is this program (as `eff_capture::Program::bytes` gives it) the given stage of the given variation?
    pub fn is_program(&self, variation: u32, stage: u32, program: &[u8]) -> bool {
        self.stages.get(&(variation, stage)).is_some_and(|s| s.code_hash == fnv(program))
    }

    fn source(&self, variation: u32, stage: u32) -> Result<wgpu::ShaderSource<'static>> {
        if self.wgsl {
            let path = self.dir.join("wgsl").join(format!("v{variation:04}_s{}.wgsl", if stage == VERTEX { 0 } else { 4 }));
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            return Ok(wgpu::ShaderSource::Wgsl(text.into()));
        }
        Ok(wgpu::ShaderSource::SpirV(self.spirv(variation, stage)?.into()))
    }

    fn spirv(&self, variation: u32, stage: u32) -> Result<Vec<u32>> {
        let path = self.dir.join("spv").join(format!("v{variation:04}_s{}.spv", if stage == VERTEX { 0 } else { 4 }));
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() % 4 != 0 {
            bail!("{} is not SPIR-V", path.display());
        }
        Ok(bytes.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect())
    }
}

/// One vertex input, already widened to `vec4`s (see [`decode_attribute`]).
#[derive(Clone, Debug)]
pub struct VertexInput {
    pub location: u32,
    pub per_instance: bool,
    pub data: Vec<[f32; 4]>,
}

/// A sampler in NVN's terms (as `eff_capture::TexBinding` records it).
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SamplerKey {
    /// 0 nearest, 1 linear, then nearest / linear with nearest mips (2, 3) or linear mips (4, 5).
    pub min: u8,
    pub mag: u8,
    /// 1 repeat, 6 mirrored repeat, anything else clamps.
    pub wrap_s: u8,
    pub wrap_t: u8,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
}

impl Default for SamplerKey {
    fn default() -> Self {
        Self { min: 5, mag: 1, wrap_s: 1, wrap_t: 1, lod_min_bits: 0f32.to_bits(), lod_max_bits: 1000f32.to_bits() }
    }
}

pub struct TextureInput<'a> {
    pub stage: u32,
    pub unit: u32,
    pub view: &'a wgpu::TextureView,
    pub sampler: SamplerKey,
}

/// The fixed-function state of a draw, in the OpenGL enums the capture records.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DrawState {
    /// NVN primitive: 0 points, 1 lines, 3 line strip, 4 triangles, 5 triangle strip.
    pub prim: u32,
    pub indexed: bool,
    /// src rgb, dst rgb, equation rgb, src alpha, dst alpha, equation alpha.
    pub blend: Option<[u32; 6]>,
    pub depth_test: Option<u32>,
    pub depth_write: bool,
    /// render_state display_side: 0 both sides, 1 / 2 one side only (the other culled).
    pub display_side: u8,
}

pub struct DrawInputs<'a> {
    pub variation: u32,
    /// (stage, n, contents of c<n>). c1 and c2 come from the library; anything else not listed reads zeros.
    pub cbufs: Vec<(u32, u32, &'a [u8])>,
    pub vertices: Vec<VertexInput>,
    pub indices: Option<Vec<u32>>,
    pub state: DrawState,
    pub alpha_func: i32,
    pub alpha_ref: f32,
    pub count: u32,
    pub instances: u32,
    pub first: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
    pub textures: Vec<TextureInput<'a>>,
}

/// A draw with every GPU resource it needs, ready to record into a pass.
pub struct PreparedDraw {
    pipeline: wgpu::RenderPipeline,
    groups: [wgpu::BindGroup; 2],
    vertex_buffers: Vec<wgpu::Buffer>,
    index: Option<wgpu::Buffer>,
    count: u32,
    instances: u32,
    first: u32,
    base_vertex: i32,
    base_instance: u32,
}

impl PreparedDraw {
    pub fn encode(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[0], &[]);
        pass.set_bind_group(1, &self.groups[1], &[]);
        for (slot, buffer) in self.vertex_buffers.iter().enumerate() {
            pass.set_vertex_buffer(slot as u32, buffer.slice(..));
        }
        let instances = self.base_instance..self.base_instance + self.instances;
        match &self.index {
            Some(index) => {
                pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(self.first..self.first + self.count, self.base_vertex, instances);
            }
            None => pass.draw(self.first..self.first + self.count, instances),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PipelineKey {
    variation: u32,
    /// Per vertex input location, in buffer order: (location, per instance).
    vertex_layout: Vec<(u32, bool)>,
    state: DrawState,
    color: wgpu::TextureFormat,
    depth: Option<wgpu::TextureFormat>,
}

struct Layouts {
    groups: [wgpu::BindGroupLayout; 2],
    pipeline: wgpu::PipelineLayout,
}

pub struct GameEffectRenderer {
    pub library: ShaderLibrary,
    modules: HashMap<(u32, u32), wgpu::ShaderModule>,
    layouts: HashMap<u32, Layouts>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    /// Bound to every unit a draw does not supply: white, so an unbound depth texture reads "far".
    fallback: wgpu::TextureView,
}

fn blend_factor(gl: u32) -> wgpu::BlendFactor {
    use wgpu::BlendFactor as F;
    match gl {
        0 => F::Zero,
        1 => F::One,
        0x300 => F::Src,
        0x301 => F::OneMinusSrc,
        0x302 => F::SrcAlpha,
        0x303 => F::OneMinusSrcAlpha,
        0x304 => F::DstAlpha,
        0x305 => F::OneMinusDstAlpha,
        0x306 => F::Dst,
        0x307 => F::OneMinusDst,
        0x308 => F::SrcAlphaSaturated,
        0x8001 => F::Constant,
        0x8002 => F::OneMinusConstant,
        _ => F::One,
    }
}

fn blend_operation(gl: u32) -> wgpu::BlendOperation {
    use wgpu::BlendOperation as O;
    match gl {
        0x800A => O::Subtract,
        0x800B => O::ReverseSubtract,
        0x8007 => O::Min,
        0x8008 => O::Max,
        _ => O::Add,
    }
}

fn blend_component(src: u32, dst: u32, eq: u32) -> wgpu::BlendComponent {
    let operation = blend_operation(eq);
    // Min and max ignore the factors in OpenGL; wgpu requires them to be One.
    if matches!(operation, wgpu::BlendOperation::Min | wgpu::BlendOperation::Max) {
        return wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation };
    }
    wgpu::BlendComponent { src_factor: blend_factor(src), dst_factor: blend_factor(dst), operation }
}

fn compare(gl: u32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match gl {
        0x200 => C::Never,
        0x201 => C::Less,
        0x202 => C::Equal,
        0x203 => C::LessEqual,
        0x204 => C::Greater,
        0x205 => C::NotEqual,
        0x206 => C::GreaterEqual,
        _ => C::Always,
    }
}

fn topology(prim: u32) -> Result<wgpu::PrimitiveTopology> {
    use wgpu::PrimitiveTopology as T;
    Ok(match prim {
        0 => T::PointList,
        1 => T::LineList,
        3 => T::LineStrip,
        4 => T::TriangleList,
        5 => T::TriangleStrip,
        other => bail!("NVN primitive {other} has no wgpu topology"),
    })
}

fn padded(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out.resize(bytes.len().div_ceil(16).max(1) * 16, 0);
    out
}

impl GameEffectRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, library: ShaderLibrary) -> Self {
        let fallback = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("game effect fallback texture"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            )
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            library,
            modules: HashMap::new(),
            layouts: HashMap::new(),
            pipelines: HashMap::new(),
            samplers: HashMap::new(),
            fallback,
        }
    }

    fn module(&mut self, device: &wgpu::Device, variation: u32, stage: u32) -> Result<wgpu::ShaderModule> {
        if let Some(m) = self.modules.get(&(variation, stage)) {
            return Ok(m.clone());
        }
        let source = self.library.source(variation, stage)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("effect shader v{variation} s{stage}")),
            source,
        });
        self.modules.insert((variation, stage), module.clone());
        Ok(module)
    }

    fn layouts(&mut self, device: &wgpu::Device, variation: u32) -> Result<&Layouts> {
        if !self.layouts.contains_key(&variation) {
            let mut buffers = Vec::new();
            let mut textures = Vec::new();
            for stage in [VERTEX, FRAGMENT] {
                let info = self.library.stage(variation, stage).ok_or_else(|| anyhow!("variation {variation} has no stage {stage}"))?;
                let visibility = if stage == VERTEX { wgpu::ShaderStages::VERTEX } else { wgpu::ShaderStages::FRAGMENT };
                let base = if stage == VERTEX { 0 } else { FRAGMENT_CBUF_BASE };
                let storage = wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                };
                for n in (0..18).filter(|n| info.cbufs >> n & 1 != 0) {
                    buffers.push(wgpu::BindGroupLayoutEntry { binding: base + n, visibility, ty: storage, count: None });
                }
                buffers.push(wgpu::BindGroupLayoutEntry { binding: MXPC_BINDING[stage as usize], visibility, ty: storage, count: None });
                for slot in &info.samplers {
                    textures.push(wgpu::BindGroupLayoutEntry {
                        binding: slot.binding,
                        visibility,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });
                    textures.push(wgpu::BindGroupLayoutEntry {
                        binding: slot.binding + SAMPLER_SPLIT,
                        visibility,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    });
                }
            }
            let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("game effect buffers"), entries: &buffers });
            let group1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("game effect textures"), entries: &textures });
            let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("game effect pipeline layout"),
                bind_group_layouts: &[Some(&group0), Some(&group1)],
                immediate_size: 0,
            });
            self.layouts.insert(variation, Layouts { groups: [group0, group1], pipeline });
        }
        Ok(&self.layouts[&variation])
    }

    fn sampler(&mut self, device: &wgpu::Device, key: SamplerKey) -> wgpu::Sampler {
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                let filter = |linear: bool| if linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
                let wrap = |w: u8| match w {
                    1 => wgpu::AddressMode::Repeat,
                    6 => wgpu::AddressMode::MirrorRepeat,
                    _ => wgpu::AddressMode::ClampToEdge,
                };
                let mipmapped = key.min >= 2;
                let lod_min = f32::from_bits(key.lod_min_bits).clamp(0.0, 32.0);
                let lod_max = f32::from_bits(key.lod_max_bits).clamp(lod_min, 32.0);
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("game effect sampler"),
                    address_mode_u: wrap(key.wrap_s),
                    address_mode_v: wrap(key.wrap_t),
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: filter(key.mag != 0),
                    min_filter: filter(key.min % 2 == 1),
                    mipmap_filter: if key.min >= 4 { wgpu::MipmapFilterMode::Linear } else { wgpu::MipmapFilterMode::Nearest },
                    lod_min_clamp: if mipmapped { lod_min } else { 0.0 },
                    lod_max_clamp: if mipmapped { lod_max } else { 0.0 },
                    ..Default::default()
                })
            })
            .clone()
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        inputs: &DrawInputs<'_>,
        color: wgpu::TextureFormat,
        depth: Option<wgpu::TextureFormat>,
    ) -> Result<PreparedDraw> {
        let variation = inputs.variation;
        let vs_inputs = self.library.stage(variation, VERTEX).ok_or_else(|| anyhow!("variation {variation}: no vertex stage"))?.inputs;

        // Vertex buffers: one per input location the vertex stage reads; a location the draw does not supply reads
        // (0, 0, 0, 1), OpenGL's value for a disabled attribute, from a single instance-rate element.
        let mut vertex_layout = Vec::new();
        let mut vertex_buffers = Vec::new();
        for location in (0..32).filter(|l| vs_inputs >> l & 1 != 0) {
            let supplied = inputs.vertices.iter().find(|v| v.location == location);
            let (data, per_instance): (Vec<[f32; 4]>, bool) = match supplied {
                Some(v) if !v.data.is_empty() => (v.data.clone(), v.per_instance),
                _ => (vec![[0.0, 0.0, 0.0, 1.0]], true),
            };
            let stride_zero = supplied.is_none_or(|v| v.data.is_empty());
            vertex_layout.push((location, per_instance));
            vertex_buffers.push((device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("game effect vertex input"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::VERTEX,
            }), stride_zero));
        }
        let key = PipelineKey { variation, vertex_layout: vertex_layout.clone(), state: inputs.state, color, depth };
        let pipeline = match self.pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let vs = self.module(device, variation, VERTEX)?;
                let fs = self.module(device, variation, FRAGMENT)?;
                let strides: Vec<bool> = vertex_buffers.iter().map(|(_, zero)| *zero).collect();
                let attributes: Vec<[wgpu::VertexAttribute; 1]> = vertex_layout
                    .iter()
                    .map(|&(location, _)| [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: location }])
                    .collect();
                let buffers: Vec<wgpu::VertexBufferLayout> = vertex_layout
                    .iter()
                    .zip(&attributes)
                    .zip(&strides)
                    .map(|((&(_, per_instance), attrs), &zero)| wgpu::VertexBufferLayout {
                        array_stride: if zero { 0 } else { 16 },
                        step_mode: if per_instance { wgpu::VertexStepMode::Instance } else { wgpu::VertexStepMode::Vertex },
                        attributes: attrs,
                    })
                    .collect();
                let state = inputs.state;
                let topology = topology(state.prim)?;
                let blend = state.blend.map(|b| wgpu::BlendState {
                    color: blend_component(b[0], b[1], b[2]),
                    alpha: blend_component(b[3], b[4], b[5]),
                });
                let layout = &self.layouts(device, variation)?.pipeline;
                let p = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!("game effect v{variation}")),
                    layout: Some(layout),
                    vertex: wgpu::VertexState { module: &vs, entry_point: Some("main"), buffers: &buffers, compilation_options: Default::default() },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs,
                        entry_point: Some("main"),
                        targets: &[Some(wgpu::ColorTargetState { format: color, blend, write_mask: wgpu::ColorWrites::ALL })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology,
                        strip_index_format: (state.indexed && topology.is_strip()).then_some(wgpu::IndexFormat::Uint32),
                        // display_side 1 draws the front faces, 2 the back faces (front = counter-clockwise, as the
                        // game's GL winding; checked against the captures, where a mesh emitter's other side showed).
                        cull_mode: match state.display_side {
                            1 => Some(wgpu::Face::Back),
                            2 => Some(wgpu::Face::Front),
                            _ => None,
                        },
                        ..Default::default()
                    },
                    depth_stencil: depth.map(|format| wgpu::DepthStencilState {
                        format,
                        depth_write_enabled: Some(state.depth_write && state.depth_test.is_some()),
                        depth_compare: Some(state.depth_test.map(compare).unwrap_or(wgpu::CompareFunction::Always)),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
                self.pipelines.insert(key, p.clone());
                p
            }
        };

        // Set 0: constant buffers and the alpha test.
        let mut buffers: Vec<(u32, wgpu::Buffer)> = Vec::new();
        for stage in [VERTEX, FRAGMENT] {
            let info = self.library.stage(variation, stage).unwrap();
            let base = if stage == VERTEX { 0 } else { FRAGMENT_CBUF_BASE };
            for n in (0..18).filter(|n| info.cbufs >> n & 1 != 0) {
                let contents = match n {
                    1 => padded(&info.constants),
                    2 => vec![0u8; TEXTURE_TABLE_BYTES],
                    _ => inputs
                        .cbufs
                        .iter()
                        .find(|(s, i, _)| *s == stage && *i == n)
                        .map(|(_, _, b)| padded(b))
                        .unwrap_or_else(|| vec![0u8; 256]),
                };
                buffers.push((base + n, device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("game effect constant buffer"),
                    contents: &contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })));
            }
            let mxpc: [u32; 4] = [inputs.alpha_func as u32, inputs.alpha_ref.to_bits(), 1f32.to_bits(), 0];
            buffers.push((MXPC_BINDING[stage as usize], device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("game effect alpha test"),
                contents: bytemuck::cast_slice(&mxpc),
                usage: wgpu::BufferUsages::STORAGE,
            })));
        }
        // Set 1: textures and samplers.
        let mut views: Vec<(u32, wgpu::TextureView, wgpu::Sampler)> = Vec::new();
        for stage in [VERTEX, FRAGMENT] {
            let slots = self.library.stage(variation, stage).unwrap().samplers.clone();
            for slot in slots {
                let given = inputs.textures.iter().find(|t| t.stage == stage && t.unit == slot.unit);
                let view = given.map(|t| t.view.clone()).unwrap_or_else(|| self.fallback.clone());
                let sampler = self.sampler(device, given.map(|t| t.sampler).unwrap_or_default());
                views.push((slot.binding, view, sampler));
            }
        }
        let layouts = self.layouts(device, variation)?;
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("game effect buffers"),
            layout: &layouts.groups[0],
            entries: &buffers
                .iter()
                .map(|(binding, b)| wgpu::BindGroupEntry { binding: *binding, resource: b.as_entire_binding() })
                .collect::<Vec<_>>(),
        });
        let mut entries = Vec::new();
        for (binding, view, sampler) in &views {
            entries.push(wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::TextureView(view) });
            entries.push(wgpu::BindGroupEntry { binding: binding + SAMPLER_SPLIT, resource: wgpu::BindingResource::Sampler(sampler) });
        }
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("game effect textures"), layout: &layouts.groups[1], entries: &entries });
        let index = inputs.indices.as_ref().map(|indices| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("game effect indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        });
        Ok(PreparedDraw {
            pipeline,
            groups: [group0, group1],
            vertex_buffers: vertex_buffers.into_iter().map(|(b, _)| b).collect(),
            index,
            count: inputs.count,
            instances: inputs.instances.max(1),
            first: inputs.first,
            base_vertex: inputs.base_vertex,
            base_instance: inputs.base_instance,
        })
    }
}

/// The camera constant buffer: the shaders' c8 (NVN slot 5, 320 bytes), for a right-handed view (looking down -z)
/// and a projection to 0..1 depth (the game's is `Mat4::perspective_infinite_rh` with near 2).
///
/// Laid out as the game writes it, every matrix row-major: +0 view, +64 projection, +128 projection * view, +192 the
/// view's rotation inverted (camera to world, no translation), +256 view row 2 (the camera's back axis in world space),
/// +272 the camera position, +288 near, far, near * far, far. +304 is copied from `tail` (a capture has
/// (1, 1, 0.01, 0); not yet known).
pub fn camera_block(view: glam::Mat4, proj: glam::Mat4, near: f32, far: f32, tail: [f32; 4]) -> Vec<u8> {
    let mut f = Vec::with_capacity(80);
    let rows = |m: glam::Mat4, f: &mut Vec<f32>| {
        for r in 0..4 {
            f.extend_from_slice(&m.row(r).to_array());
        }
    };
    rows(view, &mut f);
    rows(proj, &mut f);
    rows(proj * view, &mut f);
    let rotation = glam::Mat4::from_mat3(glam::Mat3::from_mat4(view).transpose());
    rows(rotation, &mut f);
    f.extend_from_slice(&view.row(2).truncate().extend(0.0).to_array());
    f.extend_from_slice(&view.inverse().w_axis.truncate().extend(0.0).to_array());
    f.extend_from_slice(&[near, far, near * far, far]);
    f.extend_from_slice(&tail);
    bytemuck::cast_slice(&f).to_vec()
}

/// A texture as the game's shaders sample it (texture_import::decode_for_sampling), on the GPU with every mip.
pub fn upload_sampled(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &crate::texture_import::SampledTexture,
    label: &str,
) -> wgpu::TextureView {
    let (w, h) = texture.levels[0].dimensions();
    let data: Vec<u8> = texture.levels.iter().flat_map(|l| l.as_raw().iter().copied()).collect();
    device
        .create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: texture.levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if texture.srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        )
        .create_view(&Default::default())
}

/// One element of an NVN vertex format as the shader sees it (missing components read 0, 0, 0, 1).
///
/// Formats 1..=0x30 are (n - 1) / 12 + 1 components of kind (n - 1) % 12: 8-bit unorm, snorm, uint, sint; 16-bit
/// float, unorm, snorm, uint, sint; 32-bit float, uint, sint (the recomp's `comp_fmt`). 0x69 is A2B10G10R10 snorm.
pub fn decode_attribute(format: u32, bytes: &[u8]) -> Option<[f32; 4]> {
    let mut out = [0.0, 0.0, 0.0, 1.0];
    match format {
        1..=0x30 => {
            let comps = ((format - 1) / 12 + 1) as usize;
            let kind = (format - 1) % 12;
            let width = match kind {
                0..=3 => 1,
                4..=8 => 2,
                _ => 4,
            };
            let raw = bytes.get(..comps * width)?;
            for (c, value) in out.iter_mut().enumerate().take(comps) {
                let b = &raw[c * width..(c + 1) * width];
                *value = match kind {
                    0 => b[0] as f32 / 255.0,
                    1 => (b[0] as i8 as f32 / 127.0).max(-1.0),
                    2 => b[0] as f32,
                    3 => b[0] as i8 as f32,
                    4 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
                    5 => u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0,
                    6 => (i16::from_le_bytes([b[0], b[1]]) as f32 / 32767.0).max(-1.0),
                    7 => u16::from_le_bytes([b[0], b[1]]) as f32,
                    8 => i16::from_le_bytes([b[0], b[1]]) as f32,
                    9 => f32::from_le_bytes(b.try_into().ok()?),
                    10 => u32::from_le_bytes(b.try_into().ok()?) as f32,
                    _ => i32::from_le_bytes(b.try_into().ok()?) as f32,
                };
            }
        }
        0x69 => {
            let v = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
            let signed = |bits: u32, width: u32| -> f32 {
                let shift = 32 - width;
                let x = ((bits << shift) as i32) >> shift;
                (x as f32 / ((1 << (width - 1)) - 1) as f32).max(-1.0)
            };
            out = [signed(v & 0x3ff, 10), signed(v >> 10 & 0x3ff, 10), signed(v >> 20 & 0x3ff, 10), signed(v >> 30, 2)];
        }
        _ => return None,
    }
    Some(out)
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10 & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    sign * match exp {
        0 => mant * 2f32.powi(-24),
        31 => if mant == 0.0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// A captured draw's vertex inputs, widened to `vec4`s. An attribute in a format this cannot decode is left out (it
/// then reads (0, 0, 0, 1)), and named in the returned list of problems.
pub fn vertex_inputs_from_capture(draw: &crate::eff_capture::Draw) -> (Vec<VertexInput>, Vec<String>) {
    let mut out = Vec::new();
    let mut problems = Vec::new();
    for attr in &draw.attrs {
        let Some(stream) = draw.streams.get(&attr.stream) else {
            problems.push(format!("attribute {} reads stream {}, which was not captured", attr.location, attr.stream));
            continue;
        };
        if stream.stride == 0 || stream.divisor > 1 {
            problems.push(format!("attribute {}: stream stride {} divisor {} not supported", attr.location, stream.stride, stream.divisor));
            continue;
        }
        let mut data = Vec::new();
        let mut at = attr.offset as usize;
        while at < stream.bytes.len() {
            match decode_attribute(attr.format, &stream.bytes[at..]) {
                Some(v) => data.push(v),
                None => break,
            }
            at += stream.stride as usize;
        }
        if data.is_empty() {
            problems.push(format!("attribute {}: format {:#x} not decoded", attr.location, attr.format));
            continue;
        }
        out.push(VertexInput { location: attr.location, per_instance: stream.divisor == 1, data });
    }
    (out, problems)
}

/// A captured draw's indices, widened to u32.
pub fn indices_from_capture(draw: &crate::eff_capture::Draw) -> Option<Vec<u32>> {
    let bytes = draw.index.as_ref()?;
    Some(match draw.index_type {
        0 => bytes.iter().map(|&b| b as u32).collect(),
        1 => bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as u32).collect(),
        _ => bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect(),
    })
}

impl DrawState {
    pub fn from_capture(draw: &crate::eff_capture::Draw) -> Self {
        Self {
            prim: draw.prim,
            indexed: draw.kind == 1,
            blend: draw.blend.then_some([draw.src_rgb, draw.dst_rgb, draw.eq_rgb, draw.src_a, draw.dst_a, draw.eq_a]),
            depth_test: draw.depth_test.then_some(draw.depth_func),
            depth_write: draw.depth_mask,
            // which side is the emitter's (display_side); the capture only says whether one is culled
            display_side: 0,
        }
    }
}

impl SamplerKey {
    pub fn from_capture(t: &crate::eff_capture::TexBinding) -> Self {
        Self {
            min: t.min_filter,
            mag: t.mag_filter,
            wrap_s: t.wrap_s,
            wrap_t: t.wrap_t,
            lod_min_bits: t.lod_min.to_bits(),
            lod_max_bits: t.lod_max.to_bits(),
        }
    }
}

#[cfg(test)]
mod replay {
    //! Replay captured frames through the game's shaders, headless, and hold them against the game's own picture.
    //!
    //! VISIONARY_REPLAY_FILE   a capture (eftcap.bin), or VISIONARY_REPLAY_DIR + VISIONARY_REPLAY_ENTRY for a
    //!                         split_cycle.py file (programs from progs.bin in that folder)
    //! VISIONARY_REPLAY_FRAME  a frame, or "grabs" for every frame with framebuffer grabs (default: the busiest frame)
    //! VISIONARY_REPLAY_LOOK   <distance>: replace the game's camera by one looking at the effect from that far
    //! VISIONARY_SHADERS       the effect file's shader dump, with its wgsl/ (or eft_shaders.py's spv/)
    //! VISIONARY_EFF           the effect file (for textures)
    //! VISIONARY_REPLAY_OUT    output folder
    //! VISIONARY_REPLAY_LOG    the run's game log: frames are named by the eftcycle entry being shown
    //! VISIONARY_REPLAY_GENERATE  rebuild each draw from Visionary's own data (eff_game) instead of replaying it
    //!
    //! With grabs, each target is drawn over the game's own colour and depth from before its effect draws, and the
    //! result is compared with the game's colour after them: f<frame>_fbo<n>.png is game | replay | difference (x4).
    use super::*;
    /// Where the fighter stood for the frame being compared, from the capture log (x, y, facing).
    static REPLAY_FIGHTER: std::sync::Mutex<Option<Vec<f32>>> = std::sync::Mutex::new(None);
    use crate::eff_capture::{self, Draw, Program, Shot};

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }

    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    fn gpu() -> Gpu {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .unwrap();
        println!("adapter: {:?}", adapter.get_info().name);
        // Soft particles sample the scene depth; an R32Float texture needs this to be filterable.
        let features = adapter.features() & wgpu::Features::FLOAT32_FILTERABLE;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("replay"),
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu: {e}")));
        Gpu { device, queue }
    }

    /// The effect file's emitters and textures.
    struct EffFile {
        descriptors: Vec<effect_library::ptcl_file::TextureDescriptor>,
        pool: Vec<u8>,
        emitters: Vec<effect_library::Emitter>,
        views: HashMap<usize, wgpu::TextureView>,
        /// Visionary's own reading of each emitter, in the same order (for the simulation).
        defs: Vec<crate::effects::EmitterDef>,
        /// The emitter set each emitter belongs to, by name.
        set_names: Vec<String>,
        slots: crate::eff_sim::Slots,
        /// Generated emitter block (c9) -> emitter: a draw's emitter, exactly, now that c9 is solved.
        by_block: HashMap<Vec<u8>, usize>,
    }

    fn flatten(list: &[effect_library::Emitter], out: &mut Vec<effect_library::Emitter>) {
        for e in list {
            out.push(e.clone());
            flatten(&e.children, out);
        }
    }

    impl EffFile {
        fn open(path: &str) -> Self {
            let eff = effect_library::NamcoEffectFile::load(&std::fs::read(path).unwrap()).unwrap();
            let ptcl = eff.ptcl_file.as_ref().unwrap();
            let mut emitters = Vec::new();
            for set in &ptcl.emitter_list.emitter_sets {
                flatten(&set.emitters, &mut emitters);
            }
            let (defs, set_names): (Vec<crate::effects::EmitterDef>, Vec<String>) = crate::effects::load_effect(Path::new(path))
                .unwrap()
                .ptcl
                .emitter_sets
                .into_iter()
                .flat_map(|set| {
                    let name = set.name;
                    set.emitters.into_iter().map(move |e| (e, name.clone()))
                })
                .unzip();
            assert_eq!(defs.len(), emitters.len(), "Visionary's emitter list and the file's differ");
            let by_block = emitters
                .iter()
                .enumerate()
                .filter_map(|(k, e)| Some((crate::eff_block::emitter_block(e)?, k)))
                .collect();
            Self {
                descriptors: ptcl.texture_info.as_ref().map(|i| i.descriptors.clone()).unwrap_or_default(),
                pool: ptcl.texture_info.as_ref().and_then(|i| i.binary_data.clone()).unwrap_or_default(),
                emitters,
                views: HashMap::new(),
                defs,
                set_names,
                slots: crate::eff_sim::Slots::new(),
                by_block,
            }
        }

        /// The emitter a draw belongs to. Names repeat across a file (ef_common has many "smoke1"s), so among the
        /// emitters of that name the one whose stored data agrees most with the draw's emitter block wins: the block
        /// is the stored emitter data from byte 0 (name at +0x10), with the fields the runtime fills in changed.
        /// Returns the emitter, its score and the runner-up's.
        /// The file's emitter of that name nearest the draw: the one whose first texture has the size the game
        /// bound (`texture0`, width x height; same-named emitters of different sets often differ only there),
        /// then the most bytes of the captured emitter block in common.
        fn emitter(&self, name: &str, block: Option<&[u8]>, texture0: Option<(u32, u32)>) -> Option<(&effect_library::Emitter, usize, usize)> {
            let texture_fits = |e: &effect_library::Emitter| -> bool {
                let (Some(want), Some(s)) = (texture0, e.data.sampler0.as_ref()) else { return true };
                let Some(index) = self.descriptors.iter().position(|t| t.id == s.texture_id) else { return true };
                crate::texture_import::describe(&self.pool, index, &self.descriptors[index].name)
                    .map_or(true, |d| (d.width, d.height) == want)
            };
            let score = |e: &effect_library::Emitter| -> usize {
                let bytes = match (block, e.binary_data.as_ref().or(e.cached_binary.as_ref())) {
                    (Some(block), Some(bin)) => block.iter().zip(bin).filter(|(a, b)| a == b).count(),
                    _ => 0,
                };
                // a texture of the wrong size puts the emitter behind every one of the right size
                bytes + if texture_fits(e) { 0x10000 } else { 0 }
            };
            let mut scored: Vec<_> = self.emitters.iter().filter(|e| e.data.display_name() == name).map(|e| (e, score(e))).collect();
            if std::env::var("VISIONARY_REPLAY_CANDIDATES").is_ok() {
                for (e, s) in &scored {
                    let tex: Vec<String> = [&e.data.sampler0, &e.data.sampler1, &e.data.sampler2]
                        .iter()
                        .filter_map(|x| x.as_ref())
                        .filter_map(|x| self.descriptors.iter().position(|t| t.id == x.texture_id))
                        .map(|i| format!("{} {:?}", self.descriptors[i].name, crate::texture_import::describe(&self.pool, i, &self.descriptors[i].name).map(|d| (d.width, d.height, d.format))))
                        .collect();
                    println!("    candidate {name}: score {s:#x} textures {tex:?} (want {texture0:?})");
                }
                if let Some(want) = texture0 {
                    let fits: Vec<String> = (0..self.descriptors.len())
                        .filter_map(|i| {
                            let d = crate::texture_import::describe(&self.pool, i, &self.descriptors[i].name).ok()?;
                            ((d.width, d.height) == want).then(|| format!("{} {} {}", self.descriptors[i].name, d.format, d.mipmaps))
                        })
                        .collect();
                    println!("    textures of that size: {fits:?}");
                }
            }
            scored.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
            let runner_up = scored.get(1).map(|(_, s)| *s).unwrap_or(0);
            scored.first().map(|&(e, s)| (e, s, runner_up))
        }

        fn view(&mut self, gpu: &Gpu, index: usize) -> Option<wgpu::TextureView> {
            if let Some(v) = self.views.get(&index) {
                return Some(v.clone());
            }
            let name = self.descriptors.get(index)?.name.clone();
            let t = crate::texture_import::decode_for_sampling(&self.pool, index, &name)
                .map_err(|e| println!("  texture {name}: {e}"))
                .ok()?;
            let (w, h) = t.levels[0].dimensions();
            let data: Vec<u8> = t.levels.iter().flat_map(|l| l.as_raw().iter().copied()).collect();
            let view = gpu
                .device
                .create_texture_with_data(
                    &gpu.queue,
                    &wgpu::TextureDescriptor {
                        label: Some(&name),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: t.levels.len() as u32,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: if t.srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &data,
                )
                .create_view(&Default::default());
            self.views.insert(index, view.clone());
            Some(view)
        }
    }

    const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn texture(
        gpu: &Gpu,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        data: Option<&[u8]>,
    ) -> wgpu::Texture {
        let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: usage | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        if let Some(data) = data {
            let bpp = format.block_copy_size(None).unwrap();
            gpu.queue.write_texture(
                t.as_image_copy(),
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpp), rows_per_image: None },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        t
    }

    /// Writes a depth image (R32Float) into a depth attachment: copies into depth formats are not allowed.
    fn load_depth(gpu: &Gpu, source: &wgpu::TextureView, target: &wgpu::TextureView) {
        const WGSL: &str = r#"
            @group(0) @binding(0) var src: texture_2d<f32>;
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
                return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
            }
            struct Out { @builtin(frag_depth) depth: f32 }
            @fragment fn fs(@builtin(position) pos: vec4<f32>) -> Out {
                var o: Out;
                o.depth = textureLoad(src, vec2<i32>(pos.xy), 0).r;
                return o;
            }
        "#;
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth load"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth load"),
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) }],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth load"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: target,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    /// The game's own colour targets are UNORM, so a blend clamps at 1 after every draw: the full-size scene is
    /// 10 bits a channel (every captured value lies on k/1023) and the half-size target 8 (k/255).
    /// VISIONARY_REPLAY_FLOAT draws into half floats instead, unclamped.
    fn target_format(w: u32) -> wgpu::TextureFormat {
        if std::env::var_os("VISIONARY_REPLAY_FLOAT").is_some() {
            COLOR
        } else if w >= 1280 {
            wgpu::TextureFormat::Rgb10a2Unorm
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        }
    }

    /// A captured picture (RGBA half floats) in `format`'s bytes.
    fn pack_halves(format: wgpu::TextureFormat, pixels: &[u8]) -> Vec<u8> {
        let unorm = |v: f32, max: f32| (v.clamp(0.0, 1.0) * max).round() as u32;
        let texels = pixels.chunks_exact(8).map(|p| -> [f32; 4] {
            std::array::from_fn(|c| half_to_f32(u16::from_le_bytes([p[2 * c], p[2 * c + 1]])))
        });
        match format {
            wgpu::TextureFormat::Rgb10a2Unorm => texels
                .flat_map(|t| {
                    (unorm(t[0], 1023.0) | unorm(t[1], 1023.0) << 10 | unorm(t[2], 1023.0) << 20 | unorm(t[3], 3.0) << 30)
                        .to_le_bytes()
                })
                .collect(),
            wgpu::TextureFormat::Rgba8Unorm => texels.flat_map(|t| t.map(|v| unorm(v, 255.0) as u8)).collect(),
            _ => pixels.to_vec(),
        }
    }

    fn read_back(gpu: &Gpu, texture: &wgpu::Texture, format: wgpu::TextureFormat, w: u32, h: u32) -> Vec<[f32; 4]> {
        let bpp = format.block_copy_size(None).unwrap();
        let row = (w * bpp).div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = buffer.slice(..).get_mapped_range();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let at = (y * row + x * bpp) as usize;
                out.push(match format {
                    wgpu::TextureFormat::Rgb10a2Unorm => {
                        let v = u32::from_le_bytes(mapped[at..at + 4].try_into().unwrap());
                        [
                            (v & 1023) as f32 / 1023.0,
                            (v >> 10 & 1023) as f32 / 1023.0,
                            (v >> 20 & 1023) as f32 / 1023.0,
                            (v >> 30) as f32 / 3.0,
                        ]
                    }
                    wgpu::TextureFormat::Rgba8Unorm => std::array::from_fn(|c| mapped[at + c] as f32 / 255.0),
                    _ => std::array::from_fn(|c| half_to_f32(u16::from_le_bytes([mapped[at + 2 * c], mapped[at + 2 * c + 1]]))),
                });
            }
        }
        out
    }

    fn halves(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(8)
            .map(|p| std::array::from_fn(|c| half_to_f32(u16::from_le_bytes([p[2 * c], p[2 * c + 1]]))))
            .collect()
    }

    fn srgb8(v: f32) -> u8 {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u8
    }

    struct Frame<'a> {
        number: i64,
        draws: Vec<&'a Draw>,
        /// Each with the number of the frame's draws recorded before it: a capture writes a target's "before"
        /// picture, then the effect draws, then its "after" picture, so the draws of one batch lie between them.
        shots: Vec<(&'a Shot, usize)>,
    }

    /// One target of one frame, compared with the game: effect pixels (changed in the game's picture or ours), their
    /// mean error, and the share of them off by more than 0.05.
    struct Compared {
        changed: usize,
        mean: f32,
        share: f32,
    }

    /// Draws one frame's effect draws, per target, over the game's own "before" picture where there is one.
    fn replay_frame(
        gpu: &Gpu,
        sets: &mut [(GameEffectRenderer, EffFile)],
        programs: &HashMap<(i32, u32), Program>,
        frame: &Frame,
        look: Option<f32>,
        out_dir: &Path,
    ) -> Vec<Compared> {
        let mut report = Vec::new();
        let mut fbos: Vec<(u32, i32, i32)> = frame.draws.iter().map(|d| (d.fbo, d.fbo_w, d.fbo_h)).collect();
        fbos.sort();
        fbos.dedup();
        let generate = std::env::var("VISIONARY_REPLAY_GENERATE").is_ok();
        let game_random = std::env::var("VISIONARY_REPLAY_GAME_RANDOM").is_ok();
        let look_camera = look.and_then(|distance| {
            let target = frame
                .draws
                .iter()
                .filter(|d| {
                    let (Some(vs), Some(fs)) = (programs.get(&(d.vs, 0)), programs.get(&(d.fs, 1))) else { return false };
                    sets.iter().any(|(r, _)| r.library.variation_for_programs(vs.bytes(), fs.bytes()).is_some())
                })
                .find_map(|d| d.cbuf(0, 10))
                .map(|c10| {
                    let f: &[f32] = bytemuck::cast_slice(&c10[..112]);
                    glam::Vec3::new(f[19], f[23], f[27])
                })?;
            let captured = frame.draws.iter().find_map(|d| d.cbuf(0, 8))?;
            let floats: &[f32] = bytemuck::cast_slice(captured);
            let proj = glam::Mat4::from_cols_array(floats[16..32].try_into().unwrap()).transpose();
            let view = glam::Mat4::look_at_rh(target + glam::Vec3::new(0.0, 0.0, distance), target, glam::Vec3::Y);
            Some(camera_block(view, proj, floats[72], floats[73], floats[76..80].try_into().unwrap()))
        });
        // One batch of effect draws per "before" picture: a target can take effects, then other drawing (the fighter,
        // a blurred copy of the scene drawn into the half-size target), then effects again. Captures that took one
        // "before" per target and frame pair it with the last "after", over all of that target's draws.
        let mut batches: Vec<(u32, u32, u32, usize, Option<&Shot>, Option<&Shot>, Option<&Shot>, std::ops::Range<usize>)> = Vec::new();
        for &(fbo, w, h) in &fbos {
            let (w, h) = (w as u32, h as u32);
            let sized = |s: &Shot| s.width == w && s.height == h && look.is_none();
            let is = |i: usize, phase: u32, kind: u32| {
                let (s, _) = frame.shots[i];
                s.fbo == fbo && s.phase == phase && s.kind == kind
            };
            let befores: Vec<usize> = (0..frame.shots.len()).filter(|&i| is(i, 0, 0)).collect();
            if befores.is_empty() {
                batches.push((fbo, w, h, 0, None, None, None, 0..frame.draws.len()));
                continue;
            }
            for (k, &b) in befores.iter().enumerate() {
                let next = befores.get(k + 1).copied().unwrap_or(frame.shots.len());
                let after = (b..next).rev().find(|&i| is(i, 1, 0));
                let depth = (b..next).find(|&i| is(i, 0, 1));
                let end = match (after, befores.get(k + 1)) {
                    (Some(a), _) => frame.shots[a].1,
                    (None, Some(&n)) => frame.shots[n].1,
                    (None, None) => frame.draws.len(),
                };
                let pick = |i: Option<usize>| i.map(|i| frame.shots[i].0).filter(|s| sized(s));
                batches.push((fbo, w, h, k, pick(Some(b)), pick(depth), pick(after), frame.shots[b].1..end));
            }
        }
        for (fbo, w, h, batch, before, before_depth, after, range) in batches {
            if before.is_none() || std::env::var("VISIONARY_REPLAY_SHOTS").is_ok() {
                let have: Vec<String> = frame
                    .shots
                    .iter()
                    .filter(|(s, _)| s.fbo == fbo)
                    .map(|(s, at)| {
                        let mean = if s.kind == 0 {
                            let p = halves(&s.pixels);
                            p.iter().map(|q| q[0] + q[1] + q[2]).sum::<f32>() / (3.0 * p.len().max(1) as f32)
                        } else {
                            0.0
                        };
                        format!("after draw {at}: phase {} kind {} {}x{} mean {mean:.3}", s.phase, s.kind, s.width, s.height)
                    })
                    .collect();
                let what = if before.is_none() { "no usable 'before' picture" } else { "pictures" };
                println!("  fbo {fbo} batch {batch} ({w}x{h}), draws {range:?}: {what}; shots of it: {have:?}");
            }
            if std::env::var("VISIONARY_REPLAY_TEXTURES").is_ok() {
                if let Some(s) = &before_depth {
                    let v: &[f32] = bytemuck::cast_slice(&s.pixels);
                    let (mn, mx) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                    let ones = v.iter().filter(|&&x| x >= 1.0).count();
                    println!("  fbo {fbo} depth {}x{} ({} values, {} bytes) min {mn} max {mx}, {ones} at 1", s.width, s.height, v.len(), s.pixels.len());
                }
            }
            let depth_image = before_depth.map(|s| {
                texture(gpu, w, h, wgpu::TextureFormat::R32Float, wgpu::TextureUsages::TEXTURE_BINDING, Some(&s.pixels))
                    .create_view(&Default::default())
            });
            let mut prepared = Vec::new();
            let mut names = Vec::new();
            for d in frame.draws[range.clone()].iter().filter(|d| d.fbo == fbo) {
                let name = d.emitter_name().unwrap_or_default();
                // VISIONARY_REPLAY_SKIP=<names,..>: leave these emitters' draws out (which one a difference comes from)
                if std::env::var("VISIONARY_REPLAY_SKIP").is_ok_and(|l| l.split(',').any(|n| n.trim() == name)) {
                    continue;
                }
                let (Some(vs), Some(fs)) = (programs.get(&(d.vs, 0)), programs.get(&(d.fs, 1))) else {
                    println!("  draw {} {name}: program {}/{} not captured", d.draw, d.vs, d.fs);
                    continue;
                };
                // Another effect file's emitter (the stage's own effects) is not in this library: it is left out, and
                // stays in the game's "after" picture as a difference.
                // the effect file whose shaders these are (a fighter's draws mix its own file's and ef_common's)
                let Some(set) = sets.iter().position(|(r, _)| r.library.variation_for_programs(vs.bytes(), fs.bytes()).is_some()) else {
                    println!("  draw {} {name}: shaders in none of the libraries (left out)", d.draw);
                    continue;
                };
                let (renderer, eff) = &mut sets[set];
                let Some(variation) = renderer.library.variation_for_programs(vs.bytes(), fs.bytes()) else {
                    continue;
                };
                let block = d.ubo.get(&(0, eff_capture::EMITTER_BLOCK_SLOT)).map(|b| b.as_slice());
                let mut views: Vec<(u32, wgpu::TextureView)> = Vec::new();
                let mut cull_side = 0u8;
                let texture0 = d.textures.iter().find(|t| t.stage == FRAGMENT && t.unit == 0 && t.width > 0).map(|t| (t.width as u32, t.height as u32));
                match eff.emitter(&name, block, texture0) {
                    Some((e, score, runner_up)) => {
                        let (score, runner_up) = (score & 0xffff, if runner_up >> 16 == score >> 16 { runner_up & 0xffff } else { 0 });
                        if score < runner_up + 0x40 {
                            println!("  draw {} {name}: emitter choice is close ({score:#x} vs {runner_up:#x} of 0x750 bytes)", d.draw);
                        }
                        if std::env::var("VISIONARY_REPLAY_BLOCKDIFF").is_ok() {
                            // the byte ranges where the block differs from the stored emitter (runtime-filled fields)
                            let bin = e.binary_data.as_ref().or(e.cached_binary.as_ref()).cloned().unwrap_or_default();
                            let blk = block.unwrap_or(&[]);
                            let mut ranges: Vec<(usize, usize)> = Vec::new();
                            for i in 0..blk.len().min(bin.len()) {
                                if blk[i] != bin[i] {
                                    match ranges.last_mut() {
                                        Some(r) if r.1 + 8 >= i => r.1 = i,
                                        _ => ranges.push((i, i)),
                                    }
                                }
                            }
                            let text: Vec<String> = ranges.iter().map(|(a, b)| format!("{a:#x}-{b:#x}")).collect();
                            println!("  draw {} {name}: block differs at {}", d.draw, text.join(" "));
                        }
                        if d.cull {
                            let bin = e.binary_data.as_ref().or(e.cached_binary.as_ref());
                            cull_side = bin.and_then(|b| b.get(crate::eff_game::DISPLAY_SIDE).copied()).unwrap_or(0);
                        }
                        let samplers = [e.data.sampler0.clone(), e.data.sampler1.clone(), e.data.sampler2.clone()];
                        for (unit, sampler) in samplers.iter().enumerate() {
                            let Some(s) = sampler else { continue };
                            let Some(index) = eff.descriptors.iter().position(|t| t.id == s.texture_id) else { continue };
                            if std::env::var("VISIONARY_REPLAY_TEXTURES").is_ok() {
                                let t = &eff.descriptors[index];
                                let desc = crate::texture_import::describe(&eff.pool, index, &t.name);
                                let sampled = crate::texture_import::decode_for_sampling(&eff.pool, index, &t.name);
                                let texels = sampled.as_ref().ok().map(|s| {
                                    let l = &s.levels[0];
                                    let (w, h) = l.dimensions();
                                    [(w / 2, h / 2), (w / 4, h / 2), (w / 2, h / 4), (w / 3, h / 3)].map(|(x, y)| l.get_pixel(x, y).0)
                                });
                                println!("  draw {} {name}: sampler{unit} ours {} {:?} sel {:?} texels {:?}", d.draw, t.name, desc, crate::texture_import::channel_selector(&eff.pool, index), texels);
                                if let (Ok(s), Ok(dir)) = (&sampled, std::env::var("VISIONARY_REPLAY_OUT")) {
                                    let l = &s.levels[0];
                                    let (w, h) = l.dimensions();
                                    let mut both = image::RgbaImage::new(w * 2, h);
                                    for (x, y, p) in l.enumerate_pixels() {
                                        both.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], 255]));
                                        both.put_pixel(w + x, y, image::Rgba([p[3], p[3], p[3], 255]));
                                    }
                                    let _ = both.save(std::path::Path::new(&dir).join(format!("tex_{}.png", t.name)));
                                }
                            }
                            if let Some(view) = eff.view(gpu, index) {
                                views.push((unit as u32, view));
                            }
                        }
                    }
                    None => println!("  draw {} {name}: no emitter of that name in the file", d.draw),
                }
                // The scene depth, where a unit samples a depth texture (NVN format 53) the size of this target.
                if let Some(depth) = &depth_image {
                    for t in d.textures.iter().filter(|t| t.stage == FRAGMENT && t.format == 53 && t.width as u32 == w && t.height as u32 == h) {
                        views.push((t.unit, depth.clone()));
                    }
                }
                if std::env::var("VISIONARY_REPLAY_TEXTURES").is_ok() {
                    println!("  draw {} {name}: variation {variation} state {:?} prim {} instances {} base {} count {} depth image {}", d.draw, DrawState::from_capture(d), d.prim, d.instances, d.base_instance, d.count, depth_image.is_some());
                    for t in &d.textures {
                        println!("  draw {} {name}: game unit {}/{} {:?}", d.draw, t.stage, t.unit, t);
                    }
                }
                let textures: Vec<TextureInput> = d
                    .textures
                    .iter()
                    .filter_map(|t| {
                        let (_, view) = views.iter().find(|(u, _)| *u == t.unit && t.stage == FRAGMENT)?;
                        Some(TextureInput { stage: t.stage, unit: t.unit, view, sampler: SamplerKey::from_capture(t) })
                    })
                    .collect();
                let (vertices, problems) = vertex_inputs_from_capture(d);
                if std::env::var("VISIONARY_REPLAY_TEXTURES").is_ok() {
                    for a in &d.attrs {
                        let st = d.streams.get(&a.stream);
                        println!("  draw {} {name}: attr loc {} fmt {:#x} off {} stream {} stride {:?} divisor {:?} bytes {:?}", d.draw, a.location, a.format, a.offset, a.stream, st.map(|s| s.stride), st.map(|s| s.divisor), st.map(|s| s.bytes.len()));
                    }
                    let idx = indices_from_capture(d).unwrap_or_default();
                    println!("  draw {} {name}: indices {} max {:?} first {:?}", d.draw, idx.len(), idx.iter().max(), &idx[..idx.len().min(12)]);
                    for v in &vertices {
                        let mut mn = [f32::MAX; 4];
                        let mut mx = [f32::MIN; 4];
                        for p in &v.data { for c in 0..4 { mn[c] = mn[c].min(p[c]); mx[c] = mx[c].max(p[c]); } }
                        println!("  draw {} {name}: loc {} n {} inst {} min {mn:.3?} max {mx:.3?} first {:.3?}", d.draw, v.location, v.data.len(), v.per_instance, v.data.first());
                    }
                }
                for p in problems {
                    println!("  draw {} {name}: {p}", d.draw);
                }
                let camera = look_camera.as_deref();
                let mut cbufs: Vec<(u32, u32, &[u8])> = d
                    .ubo
                    .iter()
                    .map(|(&(stage, slot), bytes)| match camera {
                        Some(camera) if slot + 3 == 8 => (stage, 8, camera),
                        _ => (stage, slot + 3, bytes.as_slice()),
                    })
                    .collect();
                let (mut variation, mut vertices, mut state, mut instances) = (variation, vertices, DrawState { display_side: cull_side, ..DrawState::from_capture(d) }, d.instances);
                let mut per_particle_base = 0u32;
                // VISIONARY_REPLAY_GENERATE: the draw rebuilt from Visionary's own data (eff_game). Kept from the
                // capture: the camera, and the effect instance's placement, clock, colour and particle scale (c10's
                // matrices and scalars), which belong to the script that spawned it, not to the file.
                let generated = generate
                    .then(|| block.and_then(|b| eff.by_block.get(b.get(..crate::eff_block::BLOCK_LEN)?)).copied())
                    .flatten()
                    .and_then(|k| {
                        let e = &eff.emitters[k];
                        let stored = e.binary_data.as_ref().or(e.cached_binary.as_ref())?.clone();
                        let v = e.data.shader_references.shader_index as u32;
                        let reflection = renderer.library.stage(v, VERTEX)?.reflection.clone();
                        let c10: Vec<f32> = bytemuck::cast_slice(d.cbuf(0, 10)?.get(..192)?).to_vec();
                        let sim = crate::eff_sim::EmitterSim::read(&eff.defs[k], &eff.slots);
                        let mut spawns = crate::eff_sim::spawns(&sim, c10[8], sim.seed(0x5eed << 32, (k as u64) << 8));
                        // VISIONARY_REPLAY_GAME_RANDOM: the game's own per-particle randoms (sysRandomAttr) in place of
                        // ours, when the counts agree, so what is left of the error is not down to the dice.
                        if game_random && spawns.len() == d.instances as usize {
                            let names = renderer.library.stage(variation, VERTEX).map(|s| s.reflection.clone()).unwrap_or_default();
                            if let Some(l) = names.input("sysRandomAttr") {
                                let inputs = vertex_inputs_from_capture(d).0;
                                if let Some(data) = inputs.iter().find(|v| v.location == l && v.per_instance).map(|v| &v.data) {
                                    for (spawn, r) in spawns.iter_mut().zip(data) {
                                        spawn.random = *r;
                                    }
                                }
                            }
                        }
                        let rows = |a: &[f32]| glam::Mat4::from_cols_array(a.try_into().unwrap()).transpose();
                        let dynamic = crate::eff_game::Dynamic {
                            colour0: c10[0..4].try_into().unwrap(),
                            colour1: c10[4..8].try_into().unwrap(),
                            time: c10[8],
                            particles: spawns.len() as u32,
                            particle_scale: [c10[10], c10[11]],
                            set_scale: c10[12..16].try_into().unwrap(),
                            matrix: rows(&c10[16..32]),
                            matrix_rt: rows(&c10[32..48]),
                        };
                        Some((v, stored, reflection, spawns, dynamic, crate::eff_block::emitter_block(e)?))
                    });
                let generated_buffers = generated.as_ref().map(|(_, _, _, _, dynamic, c9)| {
                    (
                        c9.clone(),
                        crate::eff_game::dynamic_block(dynamic),
                        bytemuck::cast_slice(&crate::eff_game::RESERVED_PARAM).to_vec(),
                        bytemuck::cast_slice(&crate::eff_game::CUSTOM_BLOCK2).to_vec(),
                    )
                });
                if let (Some((v, stored, reflection, spawns, dynamic, _)), Some((c9, c10, c13, c16))) = (&generated, &generated_buffers) {
                    // The simulation against the game's own particles, by the attributes' names: count, and the
                    // mean of each attribute's spread (|xyz|) and of its w.
                    let game_inputs = vertex_inputs_from_capture(d).0;
                    let names = &renderer.library.stage(variation, VERTEX).map(|s| s.reflection.clone()).unwrap_or_default();
                    let ours = crate::eff_game::particle_inputs(reflection, spawns, dynamic.matrix, &[]);
                    let summary = |inputs: &[VertexInput], reflection: &Reflection, name: &str| -> Option<(f32, f32)> {
                        let l = reflection.input(name)?;
                        let data = &inputs.iter().find(|v| v.location == l && v.per_instance)?.data;
                        let n = data.len().max(1) as f32;
                        let xyz = data.iter().map(|p| glam::Vec3::new(p[0], p[1], p[2]).length()).sum::<f32>() / n;
                        let w = data.iter().map(|p| p[3]).sum::<f32>() / n;
                        Some((xyz, w))
                    };
                    let set_name = block.and_then(|b| eff.by_block.get(b.get(..crate::eff_block::BLOCK_LEN)?)).map(|&k| eff.set_names[k].as_str()).unwrap_or("?");
                    let mut line = format!("  draw {} {set_name}/{name}: particles game {} ours {}", d.draw, d.instances, spawns.len());
                    if d.base_instance != 0 || std::env::var_os("VISIONARY_REPLAY_PARTICLES").is_some() {
                        line += &format!(" (base {}, buffer {})", d.base_instance, game_inputs.iter().filter(|v| v.per_instance).map(|v| v.data.len()).max().unwrap_or(0));
                    }
                    for (label, attr) in [("pos/life", "sysLocalPosAttr"), ("vel/birth", "sysLocalVecAttr"), ("scale", "sysScaleAttr")] {
                        if let (Some(g), Some(o)) = (summary(&game_inputs, names, attr), summary(&ours, reflection, attr)) {
                            line += &format!(" | {label} game {:.2}/{:.2} ours {:.2}/{:.2}", g.0, g.1, o.0, o.1);
                        }
                    }
                    // VISIONARY_REPLAY_PARTICLES: every particle's life and birth (the w of pos and vel), both sides.
                    if std::env::var_os("VISIONARY_REPLAY_PARTICLES").is_some() {
                        let ws = |inputs: &[VertexInput], reflection: &Reflection, name: &str| -> Vec<f32> {
                            reflection
                                .input(name)
                                .and_then(|l| inputs.iter().find(|v| v.location == l && v.per_instance))
                                .map(|v| v.data.iter().map(|p| p[3]).collect())
                                .unwrap_or_default()
                        };
                        let full = |inputs: &[VertexInput], reflection: &Reflection, name: &str| -> Vec<[f32; 4]> {
                            reflection
                                .input(name)
                                .and_then(|l| inputs.iter().find(|v| v.location == l && v.per_instance))
                                .map(|v| v.data.iter().take(4).copied().collect())
                                .unwrap_or_default()
                        };
                        for attr in ["sysLocalPosAttr", "sysLocalVecAttr", "sysScaleAttr", "sysRandomAttr"] {
                            println!(
                                "    {name} {attr} game {:.3?} ours {:.3?}",
                                full(&game_inputs, names, attr),
                                full(&ours, reflection, attr)
                            );
                        }
                        println!(
                            "    {name} life/birth game {:?} / {:?} ours {:?} / {:?}",
                            ws(&game_inputs, names, "sysLocalPosAttr"),
                            ws(&game_inputs, names, "sysLocalVecAttr"),
                            ws(&ours, reflection, "sysLocalPosAttr"),
                            ws(&ours, reflection, "sysLocalVecAttr"),
                        );
                    }
                    // The first particle's rotation and randoms, as they are (they pick spin direction, UV flips ...).
                    let first = |inputs: &[VertexInput], reflection: &Reflection, name: &str| -> Option<[f32; 4]> {
                        let l = reflection.input(name)?;
                        inputs.iter().find(|v| v.location == l && v.per_instance)?.data.first().copied()
                    };
                    for (label, attr) in [("rot", "sysInitRotateAttr"), ("rand", "sysRandomAttr")] {
                        if let (Some(g), Some(o)) = (first(&game_inputs, names, attr), first(&ours, reflection, attr)) {
                            line += &format!(" | {label} game {g:.2?} ours {o:.2?}");
                        }
                    }
                    // Where a generated input differs from the game's own (c10 is the capture's, so not listed).
                    let mut differs = Vec::new();
                    let generated_state = crate::eff_game::draw_state(stored, d.prim, d.kind == 1);
                    if generated_state != state {
                        differs.push(format!("state game {state:?} ours {generated_state:?}"));
                    }
                    for (n, ours) in [(9u32, c9.as_slice()), (10, c10.as_slice()), (13, c13.as_slice()), (16, c16.as_slice())] {
                        if let Some(game) = d.cbuf(0, n) {
                            if game.get(..ours.len()) != Some(ours) {
                                let g: &[f32] = bytemuck::cast_slice(&game[..ours.len().min(game.len()) / 4 * 4]);
                                differs.push(format!("c{n} game {:?}", &g[..g.len().min(24)]));
                            }
                        }
                    }
                    if !differs.is_empty() {
                        line += &format!(" | DIFFERS: {}", differs.join("; "));
                    }
                    // The game's placement of the effect: its emitter matrix without the emitter's own transform, and
                    // (VISIONARY_REPLAY_FIGHTER = "x,y,lr") without the fighter's position and facing, so it reads as
                    // the spawn call's offset, rotation (degrees, X then Y then Z) and size on the top bone.
                    let (local, _) = crate::eff_game::emitter_local(stored);
                    let mut placement = dynamic.matrix * local.inverse();
                    let fighter = REPLAY_FIGHTER.lock().unwrap().clone().or_else(|| {
                        std::env::var("VISIONARY_REPLAY_FIGHTER")
                            .ok()
                            .map(|f| f.split(',').map(|x| x.trim().parse().unwrap_or(0.0)).collect::<Vec<f32>>())
                    });
                    if let Some(v) = fighter.filter(|v| v.len() >= 3) {
                        let top = glam::Mat4::from_translation(glam::Vec3::new(v[0], v[1], 0.0))
                            * glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2 * v[2]);
                        placement = top.inverse() * placement;
                    }
                    let (scale, rotation, translation) = placement.to_scale_rotation_translation();
                    let (rx, ry, rz) = rotation.to_euler(glam::EulerRot::ZYX);
                    line += &format!(
                        " | placement t ({:.1} {:.1} {:.1}) r ({:.1} {:.1} {:.1}) s {:.2} time {:.0}",
                        translation.x,
                        translation.y,
                        translation.z,
                        rz.to_degrees(),
                        ry.to_degrees(),
                        rx.to_degrees(),
                        scale.x,
                        dynamic.time
                    );
                    println!("{line}");
                    variation = *v;
                    // VISIONARY_REPLAY_GAME_CBUFS=<n,..>: keep the game's own c<n> (to find which generated one differs)
                    let keep: Vec<u32> = std::env::var("VISIONARY_REPLAY_GAME_CBUFS")
                        .map(|l| l.split(',').filter_map(|n| n.trim().parse().ok()).collect())
                        .unwrap_or_default();
                    for (_, n, bytes) in cbufs.iter_mut() {
                        if keep.contains(n) {
                            continue;
                        }
                        *bytes = match *n {
                            9 => c9.as_slice(),
                            10 => c10.as_slice(),
                            13 => c13.as_slice(),
                            16 => c16.as_slice(),
                            _ => *bytes,
                        };
                    }
                    vertices.retain(|v| !v.per_instance); // a mesh emitter's own vertices stay
                    vertices.extend(crate::eff_game::particle_inputs(reflection, spawns, dynamic.matrix, &[]));
                    // VISIONARY_REPLAY_GAME_INPUTS=<names,..>: these per-particle inputs from the capture instead (when
                    // the particle counts agree), to tell which generated input a difference comes from.
                    if let Ok(list) = std::env::var("VISIONARY_REPLAY_GAME_INPUTS") {
                        let game = vertex_inputs_from_capture(d).0;
                        if d.instances as usize == spawns.len() {
                            for name in list.split(',') {
                                let Some(l) = reflection.input(name) else { continue };
                                if let (Some(ours), Some(theirs)) = (
                                    vertices.iter_mut().find(|v| v.location == l && v.per_instance),
                                    game.iter().find(|v| v.location == l && v.per_instance),
                                ) {
                                    ours.data = theirs.data.clone();
                                }
                            }
                        }
                    }
                    state = crate::eff_game::draw_state(stored, d.prim, d.kind == 1);
                    instances = spawns.len() as u32;
                    // A sorted emitter (sort_type) is drawn a particle at a time, back to front, each draw picking its
                    // particle by base instance: ours are drawn the same way, so each is drawn once and in that order.
                    if d.instances < instances {
                        let base = d.base_instance.min(instances - 1);
                        instances = d.instances.min(instances - base);
                        per_particle_base = base;
                    }
                }
                let inputs = DrawInputs {
                    variation,
                    cbufs,
                    vertices,
                    indices: if d.kind == 1 { indices_from_capture(d) } else { None },
                    state,
                    alpha_func: d.alpha_func,
                    alpha_ref: d.alpha_ref,
                    count: d.count,
                    instances,
                    first: if d.kind == 1 { 0 } else { d.first.max(0) as u32 },
                    base_vertex: d.base_vertex,
                    // generated particles are their own buffers, from instance 0
                    base_instance: if generated.is_some() { per_particle_base } else { d.base_instance },
                    textures,
                };
                match renderer.prepare(&gpu.device, &inputs, target_format(w), Some(DEPTH)) {
                    Ok(p) => {
                        prepared.push(p);
                        names.push(name);
                    }
                    Err(e) => println!("  draw {} {name}: {e:#}", d.draw),
                }
            }
            if prepared.is_empty() {
                continue;
            }
            let format = target_format(w);
            let packed = before.map(|s| pack_halves(format, &s.pixels));
            let target = texture(
                gpu,
                w,
                h,
                format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                packed.as_deref(),
            );
            let depth_target = texture(gpu, w, h, DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT, None);
            let depth_view = depth_target.create_view(&Default::default());
            if let Some(source) = &depth_image {
                load_depth(gpu, source, &depth_view);
            }
            let view = target.create_view(&Default::default());
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("replay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: match before {
                                Some(_) => wgpu::LoadOp::Load,
                                None => wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.06, a: 0.0 }),
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if depth_image.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(1.0) },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                for p in &prepared {
                    p.encode(&mut pass);
                }
            }
            gpu.queue.submit([encoder.finish()]);
            let ours = read_back(gpu, &target, format, w, h);
            let game = after.map(|s| halves(&s.pixels));
            let base = before.map(|s| halves(&s.pixels));
            // game | replay | difference x4, at half size
            let panels = if game.is_some() { 3 } else { 1 };
            let (hw, hh) = (w / 2, h / 2);
            let mut png = image::RgbaImage::new(hw * panels, hh);
            let (mut changed, mut sum, mut off) = (0usize, 0f32, 0usize);
            // VISIONARY_REPLAY_WORST: print the worst pixels' raw values (the PNG clips at 1).
            let worst_wanted = std::env::var("VISIONARY_REPLAY_WORST").is_ok();
            let mut worst: Vec<(f32, u32, u32, [f32; 4], [f32; 4], [f32; 4])> = Vec::new();
            let px = |p: [f32; 4]| image::Rgba([srgb8(p[0]), srgb8(p[1]), srgb8(p[2]), 255]);
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    let r = ours[i];
                    if let (Some(game), Some(base)) = (&game, &base) {
                        let g = game[i];
                        let moved = |p: [f32; 4]| (0..3).any(|c| (p[c] - base[i][c]).abs() > 1.0 / 255.0);
                        // Only where this replay drew: the stage's own effects (another effect file) are in the
                        // game's picture and not in ours.
                        if moved(r) {
                            changed += 1;
                            let e = (0..3).map(|c| (g[c] - r[c]).abs()).fold(0.0, f32::max);
                            sum += e;
                            off += (e > 0.05) as usize;
                            if worst_wanted && e > 0.05 {
                                worst.push((e, x, y, g, r, base[i]));
                            }
                        }
                    }
                    if x % 2 == 0 && y % 2 == 0 && x / 2 < hw && y / 2 < hh {
                        match &game {
                            Some(game) => {
                                let g = game[i];
                                png.put_pixel(x / 2, y / 2, px(g));
                                png.put_pixel(hw + x / 2, y / 2, px(r));
                                let e = (0..3).map(|c| (g[c] - r[c]).abs() * 4.0).fold(0.0, f32::max);
                                png.put_pixel(2 * hw + x / 2, y / 2, image::Rgba([srgb8(e), srgb8(e * 0.25), srgb8(e * 0.25), 255]));
                            }
                            None => png.put_pixel(x / 2, y / 2, px(r)),
                        }
                    }
                }
            }
            // VISIONARY_REPLAY_QUANT: how much of the game's picture sits on a k/1023 or k/255 grid (a UNORM target).
            if std::env::var("VISIONARY_REPLAY_QUANT").is_ok() {
                if let Some(game) = &game {
                    let on = |v: f32, n: f32| ((v * n).round() / n - v).abs() <= v / 2048.0 + 1e-7;
                    let (mut all, mut q10, mut q8, mut above) = (0usize, 0usize, 0usize, 0usize);
                    for p in game.iter().step_by(7) {
                        for &v in &p[..3] {
                            if v <= 0.0 || v >= 0.125 {
                                above += (v > 1.0) as usize;
                                continue;
                            }
                            all += 1;
                            q10 += on(v, 1023.0) as usize;
                            q8 += on(v, 255.0) as usize;
                        }
                    }
                    println!(
                        "  fbo {fbo} {w}x{h}: {all} values, on k/1023 {:.1}%, on k/255 {:.1}%, above 1: {above}",
                        100.0 * q10 as f32 / all.max(1) as f32,
                        100.0 * q8 as f32 / all.max(1) as f32
                    );
                }
            }
            if worst_wanted && !worst.is_empty() {
                worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                let n = worst.len();
                let mean = |f: fn(&(f32, u32, u32, [f32; 4], [f32; 4], [f32; 4])) -> [f32; 4]| {
                    let mut m = [0f32; 4];
                    for w in &worst {
                        for c in 0..4 {
                            m[c] += f(w)[c] / n as f32;
                        }
                    }
                    m
                };
                println!("  {n} pixels off: mean game {:?} ours {:?} before {:?}", mean(|w| w.3), mean(|w| w.4), mean(|w| w.5));
                for (e, x, y, g, r, b) in worst.iter().step_by((n / 6).max(1)).take(6) {
                    println!("    ({x},{y}) e {e:.3} game {g:?} ours {r:?} before {b:?}");
                }
            }
            let path = out_dir.join(if batch == 0 { format!("f{}_fbo{fbo}.png", frame.number) } else { format!("f{}_fbo{fbo}_b{batch}.png", frame.number) });
            png.save(&path).unwrap();
            let mean = if changed > 0 { sum / changed as f32 } else { 0.0 };
            let share = if changed > 0 { off as f32 / changed as f32 } else { 0.0 };
            names.dedup();
            println!(
                "frame {} fbo {fbo}: {} draws ({}){} -> {}",
                frame.number,
                prepared.len(),
                names.join(" "),
                if game.is_some() {
                    format!(", {changed} effect pixels, mean error {mean:.4}, {:.1}% off by > 0.05", share * 100.0)
                } else {
                    String::new()
                },
                path.display()
            );
            if game.is_some() {
                report.push(Compared { changed, mean, share });
            }
        }
        report
    }

    #[test]
    fn replay_captured_frame() {
        let file = std::env::var("VISIONARY_REPLAY_FILE").ok();
        let dir = std::env::var("VISIONARY_REPLAY_DIR").ok();
        if file.is_none() && dir.is_none() {
            return;
        }
        let shaders = std::env::var("VISIONARY_SHADERS").unwrap();
        let out_dir = PathBuf::from(std::env::var("VISIONARY_REPLAY_OUT").unwrap());
        std::fs::create_dir_all(&out_dir).unwrap();
        let look = std::env::var("VISIONARY_REPLAY_LOOK").ok().and_then(|d| d.parse::<f32>().ok());
        // "grabs": every frame with framebuffer grabs; a number: that frame; unset: the busiest frame (split files).
        let want = std::env::var("VISIONARY_REPLAY_FRAME").unwrap_or_else(|_| "grabs".into());

        // eftcycle's "start <i> <name>" lines, placed by SSBU_FRAMELOG's frame marks: the entry shown at a frame.
        // movecap's lines also carry where the fighter stood: "start <i> <name> at <x> <y> lr <lr>".
        let mut starts: Vec<(i64, String, Option<Vec<f32>>)> = Vec::new();
        if let Ok(log) = std::env::var("VISIONARY_REPLAY_LOG") {
            let mut mark = 0i64;
            for line in String::from_utf8_lossy(&std::fs::read(log).unwrap()).lines() {
                if let Some(n) = line.strip_prefix("frame-mark: ") {
                    mark = n.trim().parse().unwrap_or(mark);
                } else if let Some(at) = line.find("eftcycle: start ").or_else(|| line.find("movecap: start ")) {
                    // "<plugin>: start <index> <name> ..."
                    let after = &line[at..];
                    let rest: Vec<&str> = after[after.find("start ").unwrap() + 6..].split_whitespace().collect();
                    if rest.len() >= 2 {
                        let at = (rest.len() >= 7 && rest[2] == "at" && rest[5] == "lr")
                            .then(|| [rest[3], rest[4], rest[6]].iter().map(|x| x.parse().unwrap_or(0.0)).collect());
                        starts.push((mark, rest[1].to_string(), at));
                    }
                }
            }
        }
        let entry_at = |frame: i64| {
            let start = starts.iter().rev().find(|(f, _, _)| *f <= frame);
            *REPLAY_FIGHTER.lock().unwrap() = start.and_then(|(_, _, at)| at.clone());
            start.map(|(_, n, _)| n.clone()).unwrap_or_default()
        };

        let gpu = gpu();
        // VISIONARY_SHADERS / VISIONARY_EFF: one or more, separated by ';', in the same order
        let effs = std::env::var("VISIONARY_EFF").unwrap();
        let mut sets: Vec<(GameEffectRenderer, EffFile)> = shaders
            .split(';')
            .zip(effs.split(';'))
            .map(|(dir, eff)| {
                let library = ShaderLibrary::open(Path::new(dir)).unwrap();
                (GameEffectRenderer::new(&gpu.device, &gpu.queue, library), EffFile::open(eff))
            })
            .collect();
        let mut all: Vec<Compared> = Vec::new();
        let mut camera_checked = false;
        let mut process = |number: i64, draws: &[Draw], shots: &[(Shot, usize)], programs: &HashMap<(i32, u32), Program>| {
            if !camera_checked {
                if let Some(captured) = draws.iter().find_map(|d| d.cbuf(0, 8)) {
                    // camera_block must rebuild the game's own camera.
                    let floats: &[f32] = bytemuck::cast_slice(captured);
                    let view = glam::Mat4::from_cols_array(floats[0..16].try_into().unwrap()).transpose();
                    let proj = glam::Mat4::from_cols_array(floats[16..32].try_into().unwrap()).transpose();
                    let rebuilt = camera_block(view, proj, floats[72], floats[73], floats[76..80].try_into().unwrap());
                    let rebuilt: &[f32] = bytemuck::cast_slice(&rebuilt);
                    let worst = floats.iter().zip(rebuilt).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0f32, f32::max);
                    println!("camera_block vs the captured camera: worst relative difference {worst:.2e}");
                    assert!(worst < 1e-3, "camera_block does not rebuild the captured camera");
                    camera_checked = true;
                }
            }
            if !starts.is_empty() {
                println!("frame {number}: {}", entry_at(number));
            }
            let frame = Frame { number, draws: draws.iter().collect(), shots: shots.iter().map(|(s, at)| (s, *at)).collect() };
            all.extend(replay_frame(&gpu, &mut sets, programs, &frame, look, &out_dir));
        };
        let wanted = |frame: i64, has_shots: bool| match want.as_str() {
            "grabs" => has_shots,
            n => n.parse::<i64>().ok() == Some(frame),
        };

        if let Some(file) = file {
            let mut programs = HashMap::new();
            let (mut frame, mut draws, mut shots): (i64, Vec<Draw>, Vec<(Shot, usize)>) = (i64::MIN, Vec::new(), Vec::new());
            let mut flush = |frame: i64, draws: &mut Vec<Draw>, shots: &mut Vec<(Shot, usize)>, programs: &HashMap<(i32, u32), Program>| {
                if !draws.is_empty() && wanted(frame, !shots.is_empty()) {
                    process(frame, draws, shots, programs);
                }
                draws.clear();
                shots.clear();
            };
            eff_capture::stream(Path::new(&file), |record| match record {
                eff_capture::Record::Program(p) => {
                    programs.insert((p.serial, p.stage), p);
                }
                eff_capture::Record::Draw(d) => {
                    if d.frame != frame {
                        flush(frame, &mut draws, &mut shots, &programs);
                        frame = d.frame;
                    }
                    draws.push(d);
                }
                eff_capture::Record::Shot(s) => {
                    if s.frame != frame {
                        flush(frame, &mut draws, &mut shots, &programs);
                        frame = s.frame;
                    }
                    shots.push((s, draws.len()));
                }
            })
            .unwrap();
            flush(frame, &mut draws, &mut shots, &programs);
        } else {
            let dir = PathBuf::from(dir.unwrap());
            let entry = std::env::var("VISIONARY_REPLAY_ENTRY").unwrap();
            let bytes = std::fs::read(dir.join(format!("{entry}.bin"))).unwrap();
            let programs = eff_capture::read_programs(&std::fs::read(dir.join("progs.bin")).unwrap());
            let draws = eff_capture::read_draws(&bytes);
            let number = want.parse::<i64>().unwrap_or_else(|_| {
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for d in &draws {
                    *counts.entry(d.frame).or_default() += 1;
                }
                counts.into_iter().max_by_key(|&(f, n)| (n, -f)).map(|(f, _)| f).unwrap_or(0)
            });
            let frame_draws: Vec<Draw> = draws.into_iter().filter(|d| d.frame == number).collect();
            process(number, &frame_draws, &[], &programs);
        }

        let px: usize = all.iter().map(|c| c.changed).sum();
        if px > 0 {
            let mean: f32 = all.iter().map(|c| c.mean * c.changed as f32).sum::<f32>() / px as f32;
            let off: f32 = all.iter().map(|c| c.share * c.changed as f32).sum::<f32>() / px as f32;
            println!("{} targets compared: {px} effect pixels, mean error {mean:.4}, {:.1}% off by > 0.05", all.len(), off * 100.0);
        }
    }
}
