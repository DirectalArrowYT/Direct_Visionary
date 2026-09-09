//! Geometry for effects that draw a mesh instead of a quad.
//!
//! About half of all emitters do — 352 of 714 in `ef_edge.eff`, 650 of 1348 in `ef_common.eff`
//! — and their models are exactly what they look like in a model viewer: discs, rings, arcs,
//! cones. `p_cmnsmashflash_ring1` is the smash-flash ring; `p_edgesword_arc1` is a torus. Drawn
//! as a camera-facing quad instead, a shockwave ring becomes a square, which no amount of
//! sprite-sheet or blend work can fix.
//!
//! An emitter's `primitive_id` is a NAME HASH matched against the file's primitive descriptors
//! by id, not an index into them. Reading it as an index, or looking in `ptcl.primitives` (the
//! raw section list) instead of `primitive_info.descriptors`, both make a file look like it has
//! no geometry at all.
//!
//! ## The vertex layout
//!
//! Measured across the pool rather than assumed, because a wrong stride or format does not
//! error — it produces a cloud of scattered triangles. Every primitive measured in `ef_edge`
//! uses the same shape: one vertex buffer, stride 24, one shape, `u16` indices, triangles.
//!
//! ```text
//! _p0  position  0x1805  offset 0    3x f32
//! _n0  normal    0x0e02  offset 12   packed, 4 bytes
//! _c0  colour    0x0b01  offset 16   4x u8 unorm
//! _u0  uv        0x1205 / 0x1201 / 0x0901  offset 20
//! ```
//!
//! Only position and uv are read. The normal is unused (particles are unlit) and the vertex
//! colour is left to the emitter's own colour keys, which is where a particle's colour actually
//! comes from.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One primitive, flattened to what the GPU needs.
#[derive(Debug, Clone, Default)]
pub struct EffectMesh {
    pub name: String,
    /// Interleaved position (3 floats) and uv (2 floats).
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u16>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// Identifies one primitive: which `.eff` holds it and its descriptor index within that file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshKey {
    pub file: PathBuf,
    pub descriptor: usize,
}

/// Attribute formats seen on effect primitives, by the encoding in the low byte.
///
/// Only the ones the pool actually uses are handled. An unknown format yields no UV rather than
/// a guessed one: a wrong guess puts the whole mesh on one texel of the sheet, which reads as a
/// flat coloured shape and looks like a shader bug rather than a parsing one.
fn read_uv(bytes: &[u8], format: u16) -> Option<[f32; 2]> {
    match format {
        // 16_16 float (half2).
        0x1205 => {
            let u = half_to_f32(u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]));
            let v = half_to_f32(u16::from_le_bytes([*bytes.get(2)?, *bytes.get(3)?]));
            Some([u, v])
        }
        // 16_16 unorm.
        0x1201 => {
            let u = u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]) as f32 / 65535.0;
            let v = u16::from_le_bytes([*bytes.get(2)?, *bytes.get(3)?]) as f32 / 65535.0;
            Some([u, v])
        }
        // 8_8 unorm.
        0x0901 => Some([
            *bytes.first()? as f32 / 255.0,
            *bytes.get(1)? as f32 / 255.0,
        ]),
        _ => None,
    }
}

/// IEEE half to f32. Written out rather than pulled in as a dependency for one conversion.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let value = match exponent {
        0 if mantissa == 0 => sign << 31,
        // Subnormal: renormalise into a float32 exponent.
        0 => {
            let mut e = -1i32;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let exp = (127 - 15 + e) as u32;
            (sign << 31) | (exp << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (mantissa << 13),
        _ => (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

/// Pull one primitive out of an `.eff`'s BFRES pool.
pub fn extract(blob: &[u8], descriptor: usize) -> Option<EffectMesh> {
    let single = effect_library::bfres::ResFile::export_single_model(blob, descriptor).ok()?;
    let (name, model) = effect_library::bfres::ResFile::parse_model_export(single).ok()?;

    let buffer = model.vertex_buffers.first()?;
    let position = buffer.attributes.get("_p0")?;
    let uv = buffer.attributes.get("_u0");
    let stride = *buffer.buffer_strides.first()? as usize;
    let data = buffer.buffers.first()?;
    if stride == 0 {
        return None;
    }

    let mut vertices = Vec::with_capacity(buffer.vertex_count as usize);
    for index in 0..buffer.vertex_count as usize {
        let base = index * stride;
        let at = base + position.offset as usize;
        let slice = data.get(at..at + 12)?;
        let read = |n: usize| {
            f32::from_le_bytes([slice[n], slice[n + 1], slice[n + 2], slice[n + 3]])
        };
        let uv_value = uv
            .and_then(|attr| {
                let at = base + attr.offset as usize;
                data.get(at..(at + 4).min(data.len()))
                    .and_then(|bytes| read_uv(bytes, attr.format))
            })
            .unwrap_or([0.0, 0.0]);
        vertices.push(MeshVertex {
            position: [read(0), read(4), read(8)],
            uv: uv_value,
        });
    }

    // Every measured primitive is one shape of triangles with u16 indices, but the fields are
    // read rather than assumed: a mesh that is not triangles would otherwise be drawn as though
    // it were, silently.
    let mut indices = Vec::new();
    for shape in model.shapes.values() {
        for mesh in &shape.meshes {
            if mesh.primitive_type != 3 || mesh.index_format != 1 {
                continue;
            }
            for pair in mesh.index_data.chunks_exact(2) {
                indices.push(u16::from_le_bytes([pair[0], pair[1]]));
            }
        }
    }
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    Some(EffectMesh {
        name,
        vertices,
        indices,
    })
}

/// Caches extracted primitives, and the BFRES pool each `.eff` carries.
#[derive(Default)]
pub struct MeshLibrary {
    pools: HashMap<PathBuf, Option<Vec<u8>>>,
    meshes: HashMap<MeshKey, Option<EffectMesh>>,
    /// `primitive_id` to descriptor index, per file. The id is a name hash, so this is a
    /// lookup rather than arithmetic.
    ids: HashMap<PathBuf, HashMap<u64, usize>>,
}

impl MeshLibrary {
    fn pool(&mut self, path: &Path) -> Option<&Vec<u8>> {
        if !self.pools.contains_key(path) {
            let parsed = std::fs::read(path)
                .ok()
                .and_then(|bytes| effect_library::NamcoEffectFile::load(&bytes).ok());
            let ids = parsed
                .as_ref()
                .and_then(|raw| raw.ptcl_file.as_ref())
                .and_then(|ptcl| ptcl.primitive_info.as_ref())
                .map(|info| {
                    info.descriptors
                        .iter()
                        .enumerate()
                        .map(|(index, descriptor)| (descriptor.id, index))
                        .collect::<HashMap<u64, usize>>()
                })
                .unwrap_or_default();
            let blob = parsed
                .and_then(|raw| raw.ptcl_file)
                .and_then(|ptcl| ptcl.primitive_info)
                .and_then(|info| info.binary_data);
            self.ids.insert(path.to_path_buf(), ids);
            self.pools.insert(path.to_path_buf(), blob);
        }
        self.pools.get(path).and_then(|slot| slot.as_ref())
    }

    /// Descriptor index for a `primitive_id`, or `None` when this file has no such primitive.
    pub fn descriptor_for(&mut self, path: &Path, primitive_id: u64) -> Option<usize> {
        if primitive_id == 0 || primitive_id == u64::MAX {
            return None;
        }
        self.pool(path);
        self.ids.get(path)?.get(&primitive_id).copied()
    }

    /// The geometry for one primitive, extracting and caching it on first use.
    pub fn mesh(&mut self, key: &MeshKey) -> Option<&EffectMesh> {
        if !self.meshes.contains_key(key) {
            let mesh = self
                .pool(&key.file)
                .cloned()
                .and_then(|blob| extract(&blob, key.descriptor));
            if mesh.is_none() {
                eprintln!(
                    "[eff] primitive {} of {} could not be read",
                    key.descriptor,
                    key.file.display()
                );
            }
            self.meshes.insert(key.clone(), mesh);
        }
        self.meshes.get(key).and_then(|slot| slot.as_ref())
    }

    pub fn clear(&mut self) {
        self.pools.clear();
        self.meshes.clear();
        self.ids.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Option<PathBuf> {
        std::env::var_os("VISIONARY_EFF_ROOT").map(PathBuf::from)
    }

    #[test]
    fn half_floats_convert() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert!((half_to_f32(0x3555) - 0.333).abs() < 0.001);
    }

    #[test]
    fn unknown_uv_formats_yield_nothing_rather_than_a_guess() {
        // A guessed UV puts the whole mesh on one texel, which reads as a shader bug.
        assert!(read_uv(&[0, 0, 0, 0], 0xFFFF).is_none());
        assert_eq!(read_uv(&[0xff, 0xff, 0x00, 0x00], 0x1201), Some([1.0, 0.0]));
        assert_eq!(read_uv(&[0xff, 0x80, 0, 0], 0x0901), Some([1.0, 0.5019608]));
    }

    /// Real primitives out of a real pool. The layout is undocumented, so this is the check
    /// that the stride/offset/format reading is right — and a wrong read does not error, it
    /// produces scattered triangles, so the assertions are on the geometry being sane rather
    /// than merely present.
    #[test]
    fn primitives_extract_with_sane_geometry() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let path = root.join("effect/fighter/edge/ef_edge.eff");
        let mut library = MeshLibrary::default();
        let Some(blob) = library.pool(&path).cloned() else {
            panic!("ef_edge has no primitive pool");
        };

        let mut extracted = 0usize;
        for descriptor in 0..12usize {
            let Some(mesh) = extract(&blob, descriptor) else {
                continue;
            };
            extracted += 1;

            assert!(!mesh.vertices.is_empty());
            assert!(!mesh.indices.is_empty());
            assert_eq!(
                mesh.indices.len() % 3,
                0,
                "{}: index count is not whole triangles",
                mesh.name
            );
            let limit = mesh.vertices.len() as u16;
            assert!(
                mesh.indices.iter().all(|index| *index < limit),
                "{}: an index points past the vertex buffer — the stride is wrong",
                mesh.name
            );

            // Effect primitives are small, origin-centred shapes. A wrong stride reads
            // neighbouring fields as position and scatters vertices over a huge range, so the
            // extent is the check that the offsets line up.
            let extent = mesh.vertices.iter().fold(0.0f32, |worst, vertex| {
                worst
                    .max(vertex.position[0].abs())
                    .max(vertex.position[1].abs())
                    .max(vertex.position[2].abs())
            });
            assert!(
                extent.is_finite() && extent > 0.0 && extent < 1000.0,
                "{}: vertices span {extent}, which is not a primitive shape",
                mesh.name
            );
            let uv_ok = mesh
                .vertices
                .iter()
                .all(|vertex| vertex.uv.iter().all(|value| value.is_finite()));
            assert!(uv_ok, "{}: non-finite UV", mesh.name);

            if extracted <= 4 {
                println!(
                    "  '{}' {} verts, {} tris, extent {extent:.3}",
                    mesh.name,
                    mesh.vertices.len(),
                    mesh.indices.len() / 3
                );
            }
        }
        assert!(extracted > 0, "no primitive extracted");
        println!("  extracted {extracted} of 12 sampled");
    }

    /// The id is a name hash, so the lookup has to go through the descriptor table. Treating it
    /// as an index is the mistake that made the whole corpus look like billboards.
    #[test]
    fn primitive_ids_resolve_through_the_descriptor_table() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let path = root.join("effect/fighter/edge/ef_edge.eff");
        let mut library = MeshLibrary::default();
        library.pool(&path);
        let ids = library.ids.get(&path).cloned().unwrap_or_default();
        assert!(!ids.is_empty(), "no primitive descriptors");
        println!("  {} descriptors", ids.len());

        // Every real id resolves, and each to a distinct slot.
        let mut seen = std::collections::BTreeSet::new();
        for (id, index) in &ids {
            assert_eq!(library.descriptor_for(&path, *id), Some(*index));
            assert!(seen.insert(*index), "two ids share descriptor {index}");
        }
        // The sentinels and an id nothing owns resolve to nothing rather than to slot 0.
        assert_eq!(library.descriptor_for(&path, 0), None);
        assert_eq!(library.descriptor_for(&path, u64::MAX), None);
        assert_eq!(library.descriptor_for(&path, 1), None);

        // And the geometry behind a real id loads.
        let (id, index) = ids.iter().next().map(|(a, b)| (*a, *b)).unwrap();
        let key = MeshKey {
            file: path.clone(),
            descriptor: index,
        };
        let mesh = library.mesh(&key).expect("mesh for a real id");
        println!(
            "  id {id:#x} -> descriptor {index} -> '{}' ({} verts)",
            mesh.name,
            mesh.vertices.len()
        );
    }
}
