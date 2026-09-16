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

/// Pull one primitive out of an `.eff`'s BFRES pool.
///
/// Decoded through [`crate::eff_mesh_io::read_model`], which reads every vertex format the
/// corpus uses. Reading positions as f32 and UVs from three formats, as this once did, drew the
/// half-float primitives as scattered triangles and left the 16-bit snorm UVs -- 51 primitives
/// across ef_common, ef_edge and the MHA carrier -- sampling one texel.
pub fn extract(blob: &[u8], descriptor: usize) -> Option<EffectMesh> {
    let (name, mesh) = crate::eff_mesh_io::read_primitive(blob, descriptor).ok()?;
    let vertices: Vec<MeshVertex> = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, position)| MeshVertex {
            position: *position,
            uv: mesh.uv0.get(i).copied().unwrap_or([0.0, 0.0]),
        })
        .collect();
    let indices: Vec<u16> = mesh.indices.iter().map(|i| *i as u16).collect();
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
    fn unknown_formats_yield_nothing_rather_than_a_guess() {
        // A guessed UV puts the whole mesh on one texel, which reads as a shader bug.
        use crate::eff_mesh_io::decode;
        assert!(decode(&[0, 0, 0, 0], 0xFFFF).is_none());
        assert_eq!(decode(&[0xff, 0xff, 0x00, 0x00], 0x1201).map(|v| [v[0], v[1]]), Some([1.0, 0.0]));
        assert_eq!(decode(&[0xff, 0x80], 0x0901).map(|v| [v[0], v[1]]), Some([1.0, 0.5019608]));
        // 16-bit snorm, which the preview once could not read at all.
        assert_eq!(decode(&[0xff, 0x7f, 0x01, 0x80], 0x1202).map(|v| [v[0], v[1]]), Some([1.0, -1.0]));
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

    /// Survey of how effect primitives are laid out, so a writer matches what the game ships.
    ///
    /// `VISIONARY_MESH_SURVEY` = comma separated .eff paths.
    #[test]
    fn survey_primitive_layouts() {
        let Ok(list) = std::env::var("VISIONARY_MESH_SURVEY") else {
            return;
        };
        let mut layouts: std::collections::BTreeMap<String, usize> = Default::default();
        for path in list.split(',') {
            let bytes = std::fs::read(path).expect("eff reads");
            let raw = effect_library::NamcoEffectFile::load(&bytes).expect("eff parses");
            let Some(info) = raw.ptcl_file.and_then(|p| p.primitive_info) else { continue };
            let Some(blob) = info.binary_data else { continue };
            for index in 0..info.descriptors.len() {
                let Ok(single) = effect_library::bfres::ResFile::export_single_model(&blob, index) else {
                    *layouts.entry("unreadable".into()).or_default() += 1;
                    continue;
                };
                let (_, model) = effect_library::bfres::ResFile::parse_model_export(single).unwrap();
                let mut key = format!("vb{} shapes{} mats{} bones{}", model.vertex_buffers.len(), model.shapes.len(), model.materials.len(), model.skeleton.bones.len());
                for vb in &model.vertex_buffers {
                    key += &format!(" | bufs{} strides{:?} skin{}", vb.buffers.len(), vb.buffer_strides, vb.vertex_skin_count);
                    for (name, a) in &vb.attributes {
                        key += &format!(" {name}@{}:{}:{:#06x}", a.buffer_index, a.offset, a.format);
                    }
                }
                for shape in model.shapes.values() {
                    key += &format!(" | meshes{} lods", shape.meshes.len());
                    for m in &shape.meshes {
                        key += &format!(" prim{} idx{} subs{}", m.primitive_type, m.index_format, m.sub_meshes.len());
                    }
                }
                *layouts.entry(key).or_default() += 1;
            }
        }
        for (k, n) in layouts {
            println!("{n:5}  {k}");
        }
    }
}
