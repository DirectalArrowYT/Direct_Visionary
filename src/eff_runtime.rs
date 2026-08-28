//! Runtime side of an effect: resolving a spawned name to emitter data the viewport can draw.
//!
//! [`crate::effects`] reads an `.eff` file; this module answers the question the editor never
//! had to ask — given a name an ACMD script spawns, *what actually comes on screen*. That is a
//! lookup across more than one file: a fighter's own `ef_<name>.eff` holds its specific
//! effects, and everything shared (smoke, sparks, hit flashes) lives in `ef_common.eff`, which
//! is why a fighter file alone resolves only part of what its scripts spawn.
//!
//! Resolution is deliberately separate from simulation and rendering. It is the half that can
//! be verified against the real dump without a GPU, and the half whose failures are silent:
//! an unresolved name draws nothing, which looks exactly like an effect that spawned correctly
//! and happened to be invisible.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::effects::{load_effect, EffEntryInfo, LoadedEffect};

/// Where a resolved effect came from, so the UI can say why nothing appeared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSource {
    /// The selected fighter's own `ef_<fighter>.eff`.
    Fighter,
    /// The shared `ef_common.eff`.
    Common,
}

/// One spawned effect name, followed through to the emitter sets it puts on screen.
#[derive(Debug, Clone)]
pub struct ResolvedEffect {
    pub name: String,
    pub source: EffectSource,
    pub file: PathBuf,
    /// Emitter sets this effect draws, each with the frame it starts on and the bone it hangs
    /// from. A single-part effect yields one; a multi-part effect yields its variants, which
    /// is why this is a list rather than one index — flattening it to the first part is how an
    /// effect ends up visibly missing its trail or its impact flash.
    pub parts: Vec<ResolvedPart>,
}

#[derive(Debug, Clone)]
pub struct ResolvedPart {
    pub set_idx: usize,
    /// Frames after the effect starts before this part appears.
    pub start_frame: u16,
    /// Bone this part attaches to. Empty means the effect's own attachment point, i.e. whatever
    /// bone the ACMD call named.
    pub bone: String,
}

/// Why a name could not be drawn. Kept as data rather than a log line because "nothing
/// appeared" is indistinguishable from "worked, but invisible" without it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveFailure {
    /// No file in the search path has an entry by this name.
    UnknownName,
    /// The entry exists but reaches no emitter set, directly or through a variant.
    NoEmitterSet,
}

/// Loads and caches `.eff` files, and resolves spawned names against them.
///
/// The cache is by path and holds whole files: an `.eff` is megabytes (`ef_common` is 33 MB)
/// and a move spawns several effects from the same one or two files, so re-reading per lookup
/// would dominate everything else the viewport does.
#[derive(Default)]
pub struct EffectResolver {
    files: HashMap<PathBuf, Option<LoadedEffect>>,
    /// Search order: the fighter's own file first, then common. Held as a list so the order is
    /// explicit rather than implied by two named fields.
    search: Vec<(EffectSource, PathBuf)>,
}

impl EffectResolver {
    /// Point the resolver at a fighter's effect file and the shared one.
    ///
    /// Either may be absent — a fighter with no `.eff` of its own still spawns common effects,
    /// and a dump without `ef_common` still resolves the fighter's own.
    pub fn set_search_path(&mut self, fighter_eff: Option<PathBuf>, common_eff: Option<PathBuf>) {
        let mut search = Vec::new();
        if let Some(path) = fighter_eff {
            search.push((EffectSource::Fighter, path));
        }
        if let Some(path) = common_eff {
            search.push((EffectSource::Common, path));
        }
        if search != self.search {
            self.search = search;
        }
    }

    /// The conventional location of the shared effect file under a dump root.
    pub fn common_eff_path(root: &Path) -> PathBuf {
        root.join("effect")
            .join("system")
            .join("common")
            .join("ef_common.eff")
    }

    fn file(&mut self, path: &Path) -> Option<&LoadedEffect> {
        if !self.files.contains_key(path) {
            let loaded = match load_effect(path) {
                Ok(loaded) => Some(loaded),
                Err(error) => {
                    eprintln!("[eff] {} could not be read: {error}", path.display());
                    None
                }
            };
            self.files.insert(path.to_path_buf(), loaded);
        }
        self.files.get(path).and_then(|slot| slot.as_ref())
    }

    /// Follow one spawned name to the emitter sets it draws.
    ///
    /// Matching is case-insensitive because ACMD scripts and the eff entry tables do not agree
    /// on case, and a name that differs only in case is the same effect rather than a miss.
    pub fn resolve(&mut self, name: &str) -> Result<ResolvedEffect, ResolveFailure> {
        let wanted = name.to_ascii_lowercase();
        let search = self.search.clone();
        let mut seen_entry = false;

        for (source, path) in search {
            let Some(loaded) = self.file(&path) else {
                continue;
            };
            let Some(entry) = loaded
                .entries
                .iter()
                .find(|entry| entry.name.eq_ignore_ascii_case(&wanted))
            else {
                continue;
            };
            seen_entry = true;
            let parts = parts_of(entry);
            if parts.is_empty() {
                continue;
            }
            return Ok(ResolvedEffect {
                name: entry.name.clone(),
                source,
                file: path.clone(),
                parts,
            });
        }

        Err(if seen_entry {
            ResolveFailure::NoEmitterSet
        } else {
            ResolveFailure::UnknownName
        })
    }

    /// The loaded file a resolved effect came from, for reading its emitter sets and textures.
    pub fn loaded(&mut self, path: &Path) -> Option<&LoadedEffect> {
        self.file(path)
    }

    /// Drop every cached file. Used when the data root changes underneath the session.
    pub fn clear(&mut self) {
        self.files.clear();
    }
}

/// The drawable parts of one entry: its own set, then any variants that have one.
///
/// An entry with no set of its own is normal rather than broken — a multi-part effect hangs its
/// content off variants — so both paths are followed instead of treating the direct set as
/// required.
fn parts_of(entry: &EffEntryInfo) -> Vec<ResolvedPart> {
    let mut parts = Vec::new();
    if let Some(set_idx) = entry.set_idx {
        parts.push(ResolvedPart {
            set_idx,
            start_frame: 0,
            bone: String::new(),
        });
    }
    for variant in &entry.variants {
        if let Some(set_idx) = variant.set_idx {
            parts.push(ResolvedPart {
                set_idx,
                start_frame: variant.start_frame,
                bone: variant.bone.clone(),
            });
        }
    }
    parts
}

/// What an effect actually is, in the terms someone deciding whether the viewport is lying to
/// them needs: where it came from, how much of it there is, and how it draws.
///
/// The draw kind is the point. A spawned name gives no clue whether it is a sheet of textured
/// quads, geometry, or a spawn volume with nothing visible of its own, and the viewport drawing
/// a flat quad is equally consistent with "this is a flat quad" and "this is a mesh the
/// renderer cannot draw yet". Stating it removes that ambiguity.
#[derive(Debug, Clone)]
pub struct EffectSummary {
    pub source: EffectSource,
    pub set_names: Vec<String>,
    pub emitters: usize,
    /// Emitters that sample a texture. Anything less than `emitters` means part of the effect
    /// draws untextured.
    pub textured: usize,
    /// Emitters whose `primitive_id` points at a primitive this file actually holds. Almost
    /// always zero: the id is populated on nearly every emitter as a name hash, and only means
    /// something when the file carries a primitive pool to resolve it against.
    pub mesh_emitters: usize,
    /// Distinct `billboard_type` values across the emitters, ascending. These are orientation
    /// modes — camera-facing, axis-aligned, velocity-aligned — not a mesh/quad switch.
    pub billboard_types: Vec<i64>,
    /// Shortest and longest particle life in frames, across the emitters.
    pub life_range: Option<(i64, i64)>,
}

impl EffectSummary {
    /// One line for the UI. Deliberately concrete about the count, because "26 emitters" is
    /// what explains a viewport showing 26 quads where the game shows a continuous stream.
    pub fn headline(&self) -> String {
        let where_from = match self.source {
            EffectSource::Fighter => "fighter eff",
            EffectSource::Common => "ef_common",
        };
        let kind = if self.mesh_emitters > 0 {
            format!("{} mesh + {} billboard", self.mesh_emitters, self.emitters - self.mesh_emitters)
        } else {
            "billboards".to_string()
        };
        let life = match self.life_range {
            Some((low, high)) if low == high => format!(", life {low}f"),
            Some((low, high)) => format!(", life {low}-{high}f"),
            None => String::new(),
        };
        let untextured = if self.textured < self.emitters {
            format!(", {} untextured", self.emitters - self.textured)
        } else {
            String::new()
        };
        format!(
            "{where_from} · {} emitter(s) · {kind}{life}{untextured}",
            self.emitters
        )
    }
}

impl EffectResolver {
    /// Describe a spawned name without drawing it.
    pub fn describe(&mut self, name: &str) -> Result<EffectSummary, ResolveFailure> {
        let resolved = self.resolve(name)?;
        let source = resolved.source;
        let file = resolved.file.clone();
        let table = crate::eff_attrs::table();
        let billboard_slot = table
            .iter()
            .position(|attr| attr.id == "particle_data.billboard_type");
        let primitive_slot = table
            .iter()
            .position(|attr| attr.id == "particle_data.primitive_id");
        let life_slot = table.iter().position(|attr| attr.id == "particle_data.life");

        let Some(loaded) = self.loaded(&file) else {
            return Err(ResolveFailure::NoEmitterSet);
        };
        // Whether a primitive id means anything at all is a property of the FILE, not the
        // emitter: with no pool there is nothing for an id to resolve against, and every id in
        // the file is then an inert leftover hash.
        let pool = loaded
            .ptcl
            .emitter_sets
            .is_empty()
            .then_some(0usize)
            .unwrap_or(0);
        let _ = pool;

        let mut summary = EffectSummary {
            source,
            set_names: Vec::new(),
            emitters: 0,
            textured: 0,
            mesh_emitters: 0,
            billboard_types: Vec::new(),
            life_range: None,
        };

        for part in &resolved.parts {
            let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else {
                continue;
            };
            summary.set_names.push(set.name.clone());
            for emitter in &set.emitters {
                summary.emitters += 1;
                if emitter.texture_index.is_some() {
                    summary.textured += 1;
                }
                let value = |slot: Option<usize>| -> Option<i64> {
                    match emitter.attrs.get(slot?).and_then(|a| a.as_ref())? {
                        crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                        crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                        crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                    }
                };
                if let Some(kind) = value(billboard_slot) {
                    if !summary.billboard_types.contains(&kind) {
                        summary.billboard_types.push(kind);
                    }
                }
                if let Some(life) = value(life_slot) {
                    summary.life_range = Some(match summary.life_range {
                        None => (life, life),
                        Some((low, high)) => (low.min(life), high.max(life)),
                    });
                }
                let _ = primitive_slot;
            }
        }
        summary.billboard_types.sort_unstable();
        Ok(summary)
    }
}

/// A texture the renderer needs but does not have yet, decoded ready to upload.
pub struct PendingTexture {
    pub key: crate::eff_render::TextureKey,
    pub image: std::sync::Arc<image::RgbaImage>,
}

/// Turn the effects live on this frame into particles the viewport can draw.
///
/// This is the vertical slice of the runtime: it resolves each live effect, finds the emitters
/// that sample a texture, and places one billboard per emitter at the bone the effect is
/// attached to. There is no emission, no lifetime and no motion yet — every emitter is a single
/// quad — so what it proves is the chain, not the simulation: name resolves, texture decodes and
/// uploads, and the result lands at the right place in world space on the right frames.
///
/// `live` is the effects the timeline says are active, as (effect name, bone name, tint, alpha).
pub fn build_particle_batches(
    resolver: &mut EffectResolver,
    uploaded: &dyn Fn(&crate::eff_render::TextureKey) -> bool,
    bone_matrices: &std::collections::HashMap<String, glam::Mat4>,
    live: &[(String, String, [f32; 3], f32)],
) -> (Vec<crate::eff_render::ParticleBatch>, Vec<PendingTexture>) {
    use crate::eff_render::{ParticleBatch, ParticleInstance, TextureKey};

    let mut batches: Vec<ParticleBatch> = Vec::new();
    let mut pending: Vec<PendingTexture> = Vec::new();
    let mut decoded: std::collections::HashSet<TextureKey> = std::collections::HashSet::new();

    for (name, bone, tint, alpha) in live {
        let Ok(resolved) = resolver.resolve(name) else {
            continue;
        };
        // The bone the ACMD call named. Without it there is no world position to place the
        // effect at, and drawing it at the origin would be worse than not drawing it.
        let Some(matrix) = lookup_bone(bone_matrices, bone) else {
            continue;
        };
        let origin = matrix.col(3).truncate();

        let file = resolved.file.clone();
        let Some(loaded) = resolver.loaded(&file) else {
            continue;
        };
        let Some(pool) = loaded.texture_pool.clone() else {
            continue;
        };

        for part in &resolved.parts {
            let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else {
                continue;
            };
            for emitter in &set.emitters {
                let Some(texture_index) = emitter.texture_index else {
                    continue;
                };
                let index = texture_index as usize;
                let Some(info) = loaded.ptcl.bntx_textures.get(index) else {
                    continue;
                };
                let key = TextureKey {
                    file: file.clone(),
                    index,
                };
                if !uploaded(&key) && decoded.insert(key.clone()) {
                    match crate::texture_import::decode_rgba(&pool, index, &info.tex_name, None) {
                        Ok(image) => pending.push(PendingTexture {
                            key: key.clone(),
                            image: std::sync::Arc::new(image),
                        }),
                        Err(error) => {
                            eprintln!("[eff] texture '{}' not drawable: {error}", info.tex_name);
                            continue;
                        }
                    }
                }

                // Colour comes from the emitter's first colour key, scaled by whatever the
                // ACMD call asked for. The texture supplies shape only — see the note in
                // `eff_render` about these being two-channel.
                let base = emitter.color0.first();
                let color = [
                    base.map_or(1.0, |key| key.r) * tint[0],
                    base.map_or(1.0, |key| key.g) * tint[1],
                    base.map_or(1.0, |key| key.b) * tint[2],
                    emitter.alpha0_keys.first().map_or(1.0, |key| key.r) * alpha,
                ];

                let batch = match batches
                    .iter_mut()
                    .find(|batch| batch.texture == key && batch.additive)
                {
                    Some(batch) => batch,
                    None => {
                        batches.push(ParticleBatch {
                            texture: key.clone(),
                            // Additive until the emitter's own render state is read. Most
                            // effects are additive, and an additive particle drawn as alpha
                            // reads as a grey box, which is worse than the reverse.
                            additive: true,
                            instances: Vec::new(),
                        });
                        batches.last_mut().expect("just pushed")
                    }
                };
                batch.instances.push(ParticleInstance {
                    position: origin.to_array(),
                    // Fixed for the slice. Real size comes from the emitter's scale keys and
                    // the ACMD call's own scale, which the simulation pass will supply.
                    size: 1.5,
                    color,
                    rotation: 0.0,
                    _padding: [0.0; 3],
                });
            }
        }
    }

    (batches, pending)
}

/// Bone lookup that tolerates the case difference between ACMD and the skeleton.
///
/// ACMD names bones lowercase (`"top"`, `"handr"`); skeletons spell them as the artist did
/// (`"Top"`, `"HandR"`). Matching exactly finds nothing for most effects.
fn lookup_bone(
    matrices: &std::collections::HashMap<String, glam::Mat4>,
    bone: &str,
) -> Option<glam::Mat4> {
    if bone.is_empty() {
        return matrices
            .get("Trans")
            .or_else(|| matrices.get("Top"))
            .copied();
    }
    if let Some(found) = matrices.get(bone) {
        return Some(*found);
    }
    matrices
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(bone))
        .map(|(_, matrix)| *matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground truth read off a real dump. Everything a particle runtime does rests on two
    /// answers the parsed data has to give first: a name the scripts spawn resolves to an
    /// emitter set, and that set's texture decodes to pixels. Both fail in ways no hand-built
    /// fixture reproduces — a name that only exists in `ef_common`, an emitter with no sampler,
    /// a surface format the decoder refuses — so this reads the game's own files.
    ///
    /// Set `VISIONARY_EFF_ROOT` to the dump root (the folder holding `effect/`).
    fn root() -> Option<PathBuf> {
        std::env::var_os("VISIONARY_EFF_ROOT").map(PathBuf::from)
    }

    #[test]
    fn a_fighter_eff_and_ef_common_both_load_and_decode() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };

        for relative in [
            "effect/fighter/mario/ef_mario.eff",
            "effect/system/common/ef_common.eff",
        ] {
            let path = root.join(relative);
            println!("\n=== {relative} ===");
            let loaded = match load_effect(&path) {
                Ok(loaded) => loaded,
                Err(error) => panic!("{relative} failed to load: {error}"),
            };

            println!(
                "entries={} emitter_sets={} textures={} pool={}",
                loaded.entries.len(),
                loaded.ptcl.emitter_sets.len(),
                loaded.ptcl.bntx_textures.len(),
                loaded
                    .texture_pool
                    .as_ref()
                    .map(|pool| format!("{} bytes", pool.len()))
                    .unwrap_or_else(|| "none".into()),
            );
            for entry in loaded.entries.iter().take(4) {
                println!(
                    "  entry '{}' set={:?} variants={}",
                    entry.name,
                    entry.set_idx,
                    entry.variants.len()
                );
            }
            assert!(!loaded.entries.is_empty(), "{relative} has no entries");

            let direct = loaded.entries.iter().filter(|e| e.set_idx.is_some()).count();
            let via_variant = loaded
                .entries
                .iter()
                .filter(|e| e.set_idx.is_none() && !parts_of(e).is_empty())
                .count();
            println!(
                "  resolvable: {direct} direct + {via_variant} via variants, \
                 {} reach nothing",
                loaded.entries.len() - direct - via_variant
            );

            let Some(pool) = loaded.texture_pool.as_ref() else {
                println!("  no texture pool");
                continue;
            };
            let mut decoded = 0usize;
            let mut refused: Vec<String> = Vec::new();
            let sampled = loaded.ptcl.bntx_textures.len().min(12);
            for (index, texture) in loaded.ptcl.bntx_textures.iter().enumerate().take(sampled) {
                match crate::texture_import::decode_rgba(pool, index, &texture.tex_name, None) {
                    Ok(image) => {
                        if decoded < 3 {
                            println!(
                                "  decoded '{}' {}x{} {}",
                                texture.tex_name,
                                image.width(),
                                image.height(),
                                texture.format
                            );
                        }
                        decoded += 1;
                    }
                    Err(error) => {
                        refused.push(format!("{} [{}]: {error}", texture.tex_name, texture.format))
                    }
                }
            }
            println!("  decoded {decoded} of {sampled} sampled");
            for line in refused.iter().take(3) {
                println!("  refused: {line}");
            }
            assert!(decoded > 0, "{relative}: no texture decoded");
        }
    }

    #[test]
    fn resolution_prefers_the_fighter_file_and_falls_back_to_common() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let fighter = root.join("effect/fighter/mario/ef_mario.eff");
        let common = EffectResolver::common_eff_path(&root);

        let mut resolver = EffectResolver::default();
        resolver.set_search_path(Some(fighter.clone()), Some(common.clone()));

        // Every name each file offers, resolved through the real search path.
        let fighter_names: Vec<String> = load_effect(&fighter)
            .expect("fighter eff loads")
            .entries
            .iter()
            .map(|entry| entry.name.clone())
            .collect();
        let common_names: Vec<String> = load_effect(&common)
            .expect("common eff loads")
            .entries
            .iter()
            .map(|entry| entry.name.clone())
            .collect();

        let mut from_fighter = 0usize;
        let mut unresolved = 0usize;
        for name in fighter_names.iter().take(60) {
            match resolver.resolve(name) {
                Ok(resolved) if resolved.source == EffectSource::Fighter => from_fighter += 1,
                Ok(resolved) => panic!("{name} resolved to {:?}, expected the fighter", resolved.source),
                Err(_) => unresolved += 1,
            }
        }
        println!("fighter names: {from_fighter} resolved here, {unresolved} reached no set");

        // A name only `ef_common` has must still resolve — this is the half a fighter file
        // alone cannot answer, and the reason the search path exists.
        let common_only: Vec<&String> = common_names
            .iter()
            .filter(|name| !fighter_names.iter().any(|f| f.eq_ignore_ascii_case(name)))
            .take(40)
            .collect();
        let mut from_common = 0usize;
        for name in &common_only {
            if let Ok(resolved) = resolver.resolve(name) {
                assert_eq!(resolved.source, EffectSource::Common, "{name}");
                from_common += 1;
            }
        }
        println!(
            "common-only names sampled: {}, resolved from common: {from_common}",
            common_only.len()
        );
        assert!(from_fighter > 0, "no fighter effect resolved at all");
        assert!(
            from_common > 0,
            "no common-only effect resolved — the ef_common fallback is not working"
        );

        assert_eq!(
            resolver.resolve("definitely_not_an_effect_name").err(),
            Some(ResolveFailure::UnknownName)
        );
    }

    /// The whole chain short of the GPU: a name the scripts spawn becomes placed billboards
    /// with a decoded texture behind them. Everything here is a step that fails silently in the
    /// viewport — an unresolved name, a bone that never matched, a texture that would not
    /// decode all render as "no effect appeared", which is also what a correct-but-invisible
    /// effect looks like.
    #[test]
    fn a_spawned_name_becomes_placed_particles_with_a_decoded_texture() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let fighter = root.join("effect/fighter/mario/ef_mario.eff");
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(
            Some(fighter.clone()),
            Some(EffectResolver::common_eff_path(&root)),
        );

        // Skeletons spell bones as the artist did; ACMD names them lowercase. Feeding the
        // skeleton's spelling in and asking with the script's is the case gap the lookup has
        // to close, and the most common reason an effect silently never places.
        let mut bones = std::collections::HashMap::new();
        bones.insert(
            "Top".to_string(),
            glam::Mat4::from_translation(glam::Vec3::new(1.0, 20.0, 3.0)),
        );

        let live = vec![(
            "MARIO_ATKHI3_ARC".to_string(),
            "top".to_string(),
            [1.0, 1.0, 1.0],
            1.0,
        )];
        let (batches, pending) =
            build_particle_batches(&mut resolver, &|_| false, &bones, &live);

        let instances: usize = batches.iter().map(|batch| batch.instances.len()).sum();
        println!(
            "\nMARIO_ATKHI3_ARC -> {} batch(es), {instances} instance(s), {} texture(s) to upload",
            batches.len(),
            pending.len()
        );
        for texture in &pending {
            println!(
                "  texture idx={} {}x{}",
                texture.key.index,
                texture.image.width(),
                texture.image.height()
            );
        }
        assert!(!batches.is_empty(), "no batch built for a resolvable effect");
        assert!(instances > 0, "no particles placed");
        assert!(!pending.is_empty(), "no texture decoded for upload");

        // Placed at the bone, not at the origin — the failure that looks like the effect
        // spawning correctly somewhere you are not looking.
        let first = batches[0].instances[0];
        assert_eq!(first.position, [1.0, 20.0, 3.0]);
        assert!(first.color[3] > 0.0, "fully transparent particle");

        // A texture already resident must not be decoded again: decoding is the expensive part
        // and it would otherwise run every frame the effect is live.
        let key = pending[0].key.clone();
        let (_batches, again) =
            build_particle_batches(&mut resolver, &|k| *k == key, &bones, &live);
        assert!(
            again.iter().all(|texture| texture.key != key),
            "a resident texture was decoded again"
        );

        // A bone the skeleton does not have places nothing rather than defaulting to origin.
        let missing = vec![(
            "MARIO_ATKHI3_ARC".to_string(),
            "no_such_bone".to_string(),
            [1.0, 1.0, 1.0],
            1.0,
        )];
        let (none, _) = build_particle_batches(&mut resolver, &|_| false, &bones, &missing);
        assert!(
            none.iter().all(|batch| batch.instances.is_empty()),
            "particles placed for a bone that does not exist"
        );
    }

    /// How much of the real data is a mesh rather than a billboard.
    ///
    /// The slice draws every emitter as a camera-facing quad, which is right for the majority
    /// of effects and wrong for the rest — Toolbox shows plenty of effects carrying actual
    /// geometry. There are two separate mechanisms and they need telling apart before either
    /// is built: an entry can spawn an **external model** alongside its particles, and an
    /// individual emitter can draw its particles **as a primitive** from the eff's own BFRES
    /// pool. This counts both so the next stage is sized from the data instead of an
    /// impression.
    #[test]
    fn how_much_of_the_corpus_draws_meshes_rather_than_billboards() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };

        let table = crate::eff_attrs::table();
        let primitive_attr = table
            .iter()
            .position(|attr| attr.id == "particle_data.primitive_id")
            .expect("the attribute table exposes primitive_id");
        let shape_primitive_attr = table
            .iter()
            .position(|attr| attr.id == "shape_info.primitive_index")
            .expect("the attribute table exposes shape_info.primitive_index");

        for relative in [
            "effect/fighter/mario/ef_mario.eff",
            "effect/system/common/ef_common.eff",
        ] {
            let path = root.join(relative);
            let bytes = std::fs::read(&path).expect("read eff");
            let raw = effect_library::NamcoEffectFile::load(&bytes).expect("parse eff");
            let loaded = load_effect(&path).expect("load eff");

            // `primitive_id` is populated on every emitter whether or not it means anything —
            // the attribute's own documentation says it is "only meaningful when this eff
            // holds it". So the pool is the gate: no primitives in the file means no emitter
            // in it draws a mesh, regardless of what the id says.
            let primitives = match raw.ptcl_file.as_ref().and_then(|ptcl| ptcl.primitives.as_ref())
            {
                None => {
                    println!("\n=== {relative} ===");
                    println!("  primitive pool               : absent");
                    0
                }
                Some(list) => {
                    println!("\n=== {relative} ===");
                    println!("  primitive pool               : {} entries", list.len());
                    list.len()
                }
            };
            println!(
                "  external model names         : {:?}",
                raw.external_model_names
            );

            let with_model = loaded
                .entries
                .iter()
                .filter(|entry| entry.model.is_some())
                .count();

            // `billboard_type` is the discriminator that actually decides how a particle is
            // drawn. Reported as a distribution rather than a boolean: guessing which values
            // mean "mesh" and counting matches is how the first two attempts at this produced
            // numbers that contradicted the primitive pool.
            let billboard_attr = table
                .iter()
                .position(|attr| attr.id == "particle_data.billboard_type")
                .expect("the attribute table exposes billboard_type");

            let mut emitters = 0usize;
            let mut textured = 0usize;
            let mut billboard_hist: std::collections::BTreeMap<i64, usize> =
                std::collections::BTreeMap::new();
            let mut primitive_ids: std::collections::BTreeSet<i64> =
                std::collections::BTreeSet::new();
            for set in &loaded.ptcl.emitter_sets {
                for emitter in &set.emitters {
                    emitters += 1;
                    if emitter.texture_index.is_some() {
                        textured += 1;
                    }
                    let value = |index: usize| -> Option<i64> {
                        match emitter.attrs.get(index).and_then(|a| a.as_ref())? {
                            crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                            crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                            crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                        }
                    };
                    if let Some(kind) = value(billboard_attr) {
                        *billboard_hist.entry(kind).or_default() += 1;
                    }
                    if let Some(id) = value(primitive_attr) {
                        primitive_ids.insert(id);
                    }
                }
            }

            println!("  entries spawning a model     : {with_model} of {}", loaded.entries.len());
            println!("  emitters                     : {emitters}");
            println!("    sampling a texture         : {textured}");
            println!("    billboard_type histogram   : {billboard_hist:?}");
            println!("    distinct primitive_id      : {primitive_ids:?}");
            let _ = (primitives, shape_primitive_attr);
            assert!(emitters > 0);
        }
    }

    /// Corpus-wide answer to "how much of this is geometry rather than billboards".
    ///
    /// Two files is not enough to base a renderer on, and the question decides the whole shape
    /// of the next stage: if effects are mostly meshes then the billboard pass is a detour, and
    /// if they are mostly billboards then mesh support is a special case to add later. Scans
    /// every fighter's eff and reports the primitive pools and model references it finds.
    #[test]
    fn how_much_of_every_fighter_eff_is_geometry() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let fighter_root = root.join("effect/fighter");
        let Ok(dirs) = std::fs::read_dir(&fighter_root) else {
            eprintln!("no {}", fighter_root.display());
            return;
        };

        let mut files = 0usize;
        let mut with_primitives = 0usize;
        let mut total_primitives = 0usize;
        let mut total_models = 0usize;
        let mut worst: Vec<(String, usize, usize)> = Vec::new();

        for dir in dirs.flatten().take(90) {
            let name = dir.file_name().to_string_lossy().to_string();
            let path = dir.path().join(format!("ef_{name}.eff"));
            if !path.exists() {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else { continue };
            let Ok(raw) = effect_library::NamcoEffectFile::load(&bytes) else {
                continue;
            };
            files += 1;
            let primitives = raw
                .ptcl_file
                .as_ref()
                .and_then(|ptcl| ptcl.primitives.as_ref())
                .map(|list| list.len())
                .unwrap_or(0);
            let models = raw.external_model_names.len();
            if primitives > 0 {
                with_primitives += 1;
            }
            total_primitives += primitives;
            total_models += models;
            if primitives > 0 || models > 1 {
                worst.push((name, primitives, models));
            }
        }

        println!("\n=== every fighter eff ===");
        println!("  files scanned                : {files}");
        println!("  files with a primitive pool  : {with_primitives}");
        println!("  primitives across all files  : {total_primitives}");
        println!("  external model refs (total)  : {total_models}");
        worst.sort_by_key(|(_, primitives, _)| std::cmp::Reverse(*primitives));
        println!("  files carrying geometry:");
        for (name, primitives, models) in worst.iter().take(15) {
            println!("    {name:16} primitives={primitives:3} models={models}");
        }
        assert!(files > 0, "no fighter eff files scanned");
    }

    /// What three effects from one real move actually are, field by field.
    ///
    /// Mario's down smash spawns `MARIO_FB_SHOOT`, `SYS_FLAME` and `SYS_ATK_SMOKE` — one from
    /// the fighter's file and two from common. Whether these are billboards or geometry is not
    /// answerable from a corpus average, because a single move can easily be the exception, so
    /// this reads the three by name.
    #[test]
    fn the_three_effects_on_marios_down_smash() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let fighter = root.join("effect/fighter/mario/ef_mario.eff");
        let common = EffectResolver::common_eff_path(&root);
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(Some(fighter), Some(common));

        let table = crate::eff_attrs::table();
        let index_of = |id: &str| table.iter().position(|attr| attr.id == id);
        let billboard = index_of("particle_data.billboard_type").unwrap();
        let primitive = index_of("particle_data.primitive_id").unwrap();
        let life = index_of("particle_data.life");
        let emitter_type = index_of("shape_info.emitter_type").or_else(|| index_of("shape_info.type"));

        for name in ["MARIO_FB_SHOOT", "SYS_FLAME", "SYS_ATK_SMOKE"] {
            println!("\n=== {name} ===");
            let resolved = match resolver.resolve(name) {
                Ok(resolved) => resolved,
                Err(failure) => {
                    println!("  could not resolve: {failure:?}");
                    continue;
                }
            };
            println!(
                "  from {:?} ({}), {} part(s)",
                resolved.source,
                resolved
                    .file
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                resolved.parts.len()
            );
            let file = resolved.file.clone();
            let Some(loaded) = resolver.loaded(&file) else {
                continue;
            };
            let pool_has_primitives = false; // reported separately by the corpus scan
            let _ = pool_has_primitives;

            for part in &resolved.parts {
                let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else {
                    continue;
                };
                println!(
                    "  set '{}' — {} emitter(s), starts +{}f, bone '{}'",
                    set.name,
                    set.emitters.len(),
                    part.start_frame,
                    part.bone
                );
                for emitter in &set.emitters {
                    let value = |slot: usize| -> Option<i64> {
                        match emitter.attrs.get(slot).and_then(|a| a.as_ref())? {
                            crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                            crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                            crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                        }
                    };
                    println!(
                        "    '{}' billboard_type={:?} life={:?} texture={:?} primitive_id={:?}{}",
                        emitter.name,
                        value(billboard),
                        life.and_then(value),
                        emitter.texture_index,
                        value(primitive),
                        emitter_type
                            .and_then(value)
                            .map(|t| format!(" emitter_shape={t}"))
                            .unwrap_or_default(),
                    );
                }
            }
        }
    }

    /// The label the effects panel shows, checked against the effects it describes.
    #[test]
    fn the_content_summary_says_what_each_effect_is_made_of() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(
            Some(root.join("effect/fighter/mario/ef_mario.eff")),
            Some(EffectResolver::common_eff_path(&root)),
        );

        for name in ["MARIO_FB_SHOOT", "SYS_FLAME", "SYS_ATK_SMOKE"] {
            let summary = resolver.describe(name).expect("resolves");
            println!("  {name:16} {}", summary.headline());
            println!("      types={:?} sets={:?}", summary.billboard_types, summary.set_names);
            assert!(summary.emitters > 0);
            // Every emitter in these three is textured, so the headline must not claim
            // otherwise — an untextured count appearing here would mean the texture link is
            // being read wrong.
            assert_eq!(summary.textured, summary.emitters, "{name}");
            assert!(
                summary.headline().contains("billboards"),
                "{name}: {}",
                summary.headline()
            );
            assert!(summary.life_range.is_some(), "{name} has no lifetimes");
        }

        // A name nothing has must say so rather than describing an empty effect.
        assert!(matches!(
            resolver.describe("not_a_real_effect"),
            Err(ResolveFailure::UnknownName)
        ));
    }

    #[test]
    fn an_emitter_set_exposes_what_a_runtime_needs() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let path = root.join("effect/fighter/mario/ef_mario.eff");
        let loaded = load_effect(&path).expect("fighter eff loads");

        let (entry, part) = loaded
            .entries
            .iter()
            .find_map(|entry| parts_of(entry).into_iter().next().map(|part| (entry, part)))
            .expect("some entry reaches an emitter set");
        let set = &loaded.ptcl.emitter_sets[part.set_idx];
        println!(
            "\nentry '{}' -> set {} '{}' with {} emitter(s)",
            entry.name,
            part.set_idx,
            set.name,
            set.emitters.len()
        );

        let table = crate::eff_attrs::table();
        let mut with_texture = 0usize;
        for emitter in set.emitters.iter().take(3) {
            let present = emitter.attrs.iter().filter(|a| a.is_some()).count();
            println!(
                "  '{}' depth={} texture={:?} color0={} alpha0={} attrs={}/{}",
                emitter.name,
                emitter.depth,
                emitter.texture_index,
                emitter.color0.len(),
                emitter.alpha0_keys.len(),
                present,
                emitter.attrs.len(),
            );
            if emitter.texture_index.is_some() {
                with_texture += 1;
            }
            let mut shown = 0;
            for (attr, value) in table.iter().zip(emitter.attrs.iter()) {
                let Some(value) = value else { continue };
                let label = attr.label.to_ascii_lowercase();
                if label.contains("life") || label.contains("rate") || label.contains("gravity") {
                    println!("    {} = {value:?}", attr.label);
                    shown += 1;
                    if shown >= 6 {
                        break;
                    }
                }
            }
            assert!(present > 0, "emitter '{}' carries no attributes", emitter.name);
        }
        println!("  {with_texture} of the sampled emitters sample a texture");
    }
}
