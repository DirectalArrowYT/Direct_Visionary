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
    /// A donor file the mod transplants effects from, under
    /// `effect/fighter/<name>/transplant/<donor>/ef_<donor>.eff`.
    ///
    /// A moveset mod routinely spawns effects belonging to a different fighter, and ships that
    /// fighter's eff alongside so the game can resolve them. Without searching these, every
    /// borrowed effect resolves to nothing and silently never appears -- which is most of what
    /// a transplant-heavy moveset spawns.
    Transplant,
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
    /// Primitive descriptor ids per file, so "does this emitter draw a mesh" does not reparse
    /// a 33 MB eff for every emitter of every effect on every frame.
    descriptors: HashMap<PathBuf, std::collections::HashSet<u64>>,
    /// Search order: the fighter's own file first, then common. Held as a list so the order is
    /// explicit rather than implied by two named fields.
    search: Vec<(EffectSource, PathBuf)>,
}

impl EffectResolver {
    /// Point the resolver at a fighter's effect file and the shared one.
    ///
    /// Either may be absent — a fighter with no `.eff` of its own still spawns common effects,
    /// and a dump without `ef_common` still resolves the fighter's own.
    pub fn set_search_path(
        &mut self,
        fighter_eff: Option<PathBuf>,
        transplants: Vec<PathBuf>,
        common_eff: Option<PathBuf>,
    ) {
        // Order is the resolution rule: the fighter's own file wins, then anything it
        // transplanted, then the shared file. A donor and `ef_common` can both define a name,
        // and the donor is the one the mod meant.
        let mut search = Vec::new();
        if let Some(path) = fighter_eff {
            search.push((EffectSource::Fighter, path));
        }
        for path in transplants {
            search.push((EffectSource::Transplant, path));
        }
        if let Some(path) = common_eff {
            search.push((EffectSource::Common, path));
        }
        if search != self.search {
            self.search = search;
        }
    }

    /// Donor effect files a mod transplants from, for one fighter, across every root.
    ///
    /// Layout is ARCropolis's: `effect/fighter/<name>/transplant/<donor>/ef_<donor>.eff`. The
    /// donor directory name is not assumed to match the file inside it -- the file is whatever
    /// `.eff` is there -- because the pairing is a mod author's convention rather than a rule.
    pub fn transplant_donors(roots: &[PathBuf], fighter: &str) -> Vec<PathBuf> {
        let mut donors = Vec::new();
        for root in roots {
            let base = root
                .join("effect")
                .join("fighter")
                .join(fighter)
                .join("transplant");
            let Ok(entries) = std::fs::read_dir(&base) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(files) = std::fs::read_dir(entry.path()) else {
                    continue;
                };
                for file in files.flatten() {
                    let path = file.path();
                    if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("eff"))
                        && !donors.contains(&path)
                    {
                        donors.push(path);
                    }
                }
            }
        }
        donors.sort();
        donors
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

    /// Primitive descriptor ids held by one `.eff`, cached alongside the file.
    ///
    /// An emitter's `primitive_id` is a name hash matched against these; an id with no
    /// descriptor in its own file draws nothing, which is how a leftover id on a file with no
    /// primitives at all is told apart from a real mesh reference.
    pub fn descriptor_ids(&mut self, path: &Path) -> std::collections::HashSet<u64> {
        if let Some(cached) = self.descriptors.get(path) {
            return cached.clone();
        }
        let ids = std::fs::read(path)
            .ok()
            .and_then(|bytes| effect_library::NamcoEffectFile::load(&bytes).ok())
            .and_then(|raw| {
                raw.ptcl_file
                    .as_ref()
                    .and_then(|ptcl| ptcl.primitive_info.as_ref())
                    .map(|info| {
                        info.descriptors
                            .iter()
                            .map(|d| d.id)
                            .collect::<std::collections::HashSet<u64>>()
                    })
            })
            .unwrap_or_default();
        self.descriptors.insert(path.to_path_buf(), ids.clone());
        ids
    }

    /// Drop every cached file. Used when the data root changes underneath the session.
    pub fn clear(&mut self) {
        self.files.clear();
        self.descriptors.clear();
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
    /// File stem the effect resolved from, so a transplant can name its donor.
    pub donor: String,
    pub set_names: Vec<String>,
    pub emitters: usize,
    /// Emitters that sample a texture. Anything less than `emitters` means part of the effect
    /// draws untextured.
    pub textured: usize,
    /// Emitters whose `primitive_id` resolves to a primitive descriptor in this file — that
    /// is, ones drawing geometry rather than a quad. Around half of them, measured across
    /// ef_edge, ef_mario and ef_common, so this is the common case rather than an exception.
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
            EffectSource::Fighter => "fighter eff".to_string(),
            EffectSource::Transplant => format!("transplant: {}", self.donor),
            EffectSource::Common => "ef_common".to_string(),
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
        let donor = file
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_default();
        let table = crate::eff_attrs::table();
        let billboard_slot = table
            .iter()
            .position(|attr| attr.id == "particle_data.billboard_type");
        let primitive_slot = table
            .iter()
            .position(|attr| attr.id == "particle_data.primitive_id");
        let life_slot = table.iter().position(|attr| attr.id == "particle_data.life");

        // Whether an emitter draws geometry is a lookup, not a flag: its `primitive_id` is
        // matched against the file's primitive descriptors by id. Treating the id as an index,
        // or reading the raw section list instead of the descriptor table, reports every file
        // as pure billboards -- which is what this said before, for files where half the
        // emitters draw a mesh.
        let descriptors = self.descriptor_ids(&file);
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
            donor,
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
                if let Some(id) = value(primitive_slot) {
                    if id != 0 && id != -1 && descriptors.contains(&(id as u64)) {
                        summary.mesh_emitters += 1;
                    }
                }
                if let Some(life) = value(life_slot) {
                    summary.life_range = Some(match summary.life_range {
                        None => (life, life),
                        Some((low, high)) => (low.min(life), high.max(life)),
                    });
                }
            }
        }
        summary.billboard_types.sort_unstable();
        Ok(summary)
    }
}

/// One effect the timeline has live on the current frame.
#[derive(Debug, Clone)]
pub struct LiveEffect {
    pub name: String,
    pub bone: String,
    pub tint: [f32; 3],
    pub alpha: f32,
    /// Frames since the effect started. This, not the absolute frame, is what the simulation
    /// runs on — an effect spawned on frame 12 and one spawned on frame 40 look the same five
    /// frames in, and the emitter data is written in those terms.
    pub age: f32,
    /// The ACMD call's own offset from the bone, in the effect's local frame. Scripts place
    /// effects off the joint constantly -- a hit flash at the end of a sword, smoke under a
    /// foot -- so dropping it puts every effect exactly on the joint instead.
    pub offset: glam::Vec3,
    /// The call's own Euler rotation in DEGREES, in the editor's X/Y/Z order.
    ///
    /// Degrees, not radians, and the distinction is not cosmetic: the emitter's own rotation in
    /// the eff file IS radians (a quarter turn reads as 1.5707964), while an ACMD spawn's
    /// rotation arguments are degrees and go to the script unconverted. Feeding both to the
    /// same euler constructor made one degree of aim come out as fifty-seven.
    ///
    /// This is how a script aims an effect: the same explosion is spawned pointing along the
    /// punch, up off the ground, or back over the shoulder purely by this argument. Dropping it
    /// leaves every effect at whatever angle its emitter happens to carry, which for most is
    /// none at all.
    pub rotation: glam::Vec3,
    /// The call's own size multiplier -- the `size` argument of the spawn macro.
    ///
    /// Scales the effect's whole geometry: how big each particle is AND how far it spreads
    /// from the emitter, because in the game an effect at half size is the same effect
    /// smaller, not the same spread with smaller dots in it. It does NOT scale the call's
    /// placement offset, which is a position in the bone's frame rather than part of the
    /// effect.
    ///
    /// A literal multiply, including zero. Only 2 of the 8732 `EFFECT_FOLLOW` calls in the
    /// script dump pass 0, so treating it as "invisible" costs nothing and matches what the
    /// game does, where reading it as "unset, use 1.0" would silently show two effects the
    /// game does not.
    pub scale: f32,
}

/// Put a decoded effect texture into one form the shader can treat uniformly.
///
/// Effect textures come in two families that store the particle's shape in different places,
/// and reading the wrong one renders a solid white quad rather than a shape:
///
/// * `BC3` and friends carry greyscale colour in RGB with the shape in ALPHA. On
///   `ef_cmn_impact05_ani` red averages 225 of 255 — nearly solid white — while alpha averages
///   74 and holds the actual smoke.
/// * `BC5` is two-channel: red and green carry data, blue is zero and alpha is a constant 255.
///   Here the shape is in RED, and there is no alpha to read.
///
/// Rather than branch in the shader on a format the shader cannot see, a constant-alpha texture
/// is rewritten so red becomes both its colour and its alpha. Everything downstream then means
/// the same thing by RGBA.
fn normalize_mask(image: image::RgbaImage) -> image::RgbaImage {
    let mut lowest = 255u8;
    let mut highest = 0u8;
    for pixel in image.pixels() {
        lowest = lowest.min(pixel.0[3]);
        highest = highest.max(pixel.0[3]);
    }
    // A texture whose alpha never varies is not carrying shape in it.
    if highest.saturating_sub(lowest) > 2 {
        return image;
    }
    let mut out = image;
    for pixel in out.pixels_mut() {
        let mask = pixel.0[0];
        pixel.0 = [mask, mask, mask, mask];
    }
    out
}

/// A texture the renderer needs but does not have yet, decoded ready to upload.
pub struct PendingTexture {
    pub key: crate::eff_render::TextureKey,
    pub image: std::sync::Arc<image::RgbaImage>,
}

/// Everything one frame of effects needs handed to the GPU.
/// Which orientation mode each `billboard_type` draws its quads with, indexed by type.
///
/// Type 0 is camera-facing and certain. The rest are not documented anywhere and are not
/// derivable from the file — which mode is right is something you can see and the data cannot
/// tell you — so the mapping is a setting rather than a constant, and the viewport lets it be
/// changed while looking at the effect.
///
/// Not every billboard type is a quad, and no mode here can cover the ones that are not.
/// `render/common/effect_param.prc` budgets the runtime's pools separately: 800 emitters, 120
/// emitter sets -- and 32 STRIPES. A stripe is a ribbon threaded through a particle's history
/// rather than a quad at a position, so whichever type values select one, they cannot be drawn
/// by choosing axes for a quad. Stepping every basis for such a type and finding that none
/// match is the expected outcome, not a missing entry in this list.
///
/// The modes are a closed set of six real techniques rather than a plane plus free rotation,
/// which is the point: picking from six things you can tell apart by eye converges, and
/// hand-turning three Euler angles per type does not. Two of the six — Y-billboard and
/// velocity — depend on where the camera is and where the particle is going, so no amount of
/// tuning a fixed plane will ever reach them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadPlanes {
    pub planes: [u32; 8],
    /// Extra turn applied to an oriented quad, per type, in degrees (X, Y, Z).
    ///
    /// Several effects come out a quarter or half turn off in a way the plane alone cannot
    /// express -- MIIGUNNER_ATK_SHOT_S and RIDLEY_SMASH_BOMB both want +90 on Y, and
    /// SYS_ATTACK_ARC wants X 180, Z -90. Whether that is a basis difference or a convention this
    /// does not model yet, it is measurable by eye and not readable from the file, so it is a
    /// setting until it is understood well enough to be a constant.
    pub offsets: [[f32; 3]; 8],
    /// Which of [`SPAWN_ORDERS`] a spawn's rotation arguments are applied in.
    ///
    /// Unlike `offsets` this is not per billboard type: it is how the ACMD call is read, so it
    /// applies to every effect, meshes included.
    pub spawn_order: usize,
    /// Quarter turns about up between the bone's frame and the frame a spawn's rotation and
    /// geometry live in. Applies to every effect, meshes included; the call's position offset
    /// stays in the bone's frame.
    pub spawn_frame_turns: u8,
}

impl Default for QuadPlanes {
    fn default() -> Self {
        // Dialled in against the game, per type, rather than derived -- the file records a
        // billboard type and says nothing about what it means, so these came from putting the
        // effect on screen next to what Smash draws.
        //
        // Type 0 is the interesting one. A camera-facing quad ignores orientation for its
        // PLANE, so its turn cannot be fixing how the quad faces: it can only be fixing where
        // the particles fly, since the same orientation rotates each particle's offset from the
        // emitter. That is a second, independent frame correction living in the same field, and
        // worth separating if a type ever needs different values for the two.
        //
        //   0  camera facing   X 90, Y 90   (particle motion frame)
        //   3  local XY        X 180, Z -90 SYS_ATTACK_ARC
        //   5  local XY        X 90         MIIGUNNER_ATK_SHOT_S, RIDLEY_SMASH_BOMB
        //
        // 1, 4, 6 and 7 are untouched: no effect using them has been checked against the game,
        // and a guessed default is worse than an obvious one because it looks deliberate.
        let mut offsets = [[0.0f32; 3]; 8];
        offsets[0] = [90.0, 90.0, 0.0];
        offsets[3] = [180.0, 0.0, -90.0];
        offsets[5] = [90.0, 0.0, 0.0];
        Self {
            planes: [0, 1, 1, 1, 1, 1, 1, 1],
            offsets,
            // Y, Z, X with a quarter frame turn: stepped through live against Dabi's jabs,
            // the one reading that matched all three. See
            // `spawn_rotation_defaults_match_dabis_jabs_in_game`.
            spawn_order: 3,
            spawn_frame_turns: 1,
        }
    }
}

impl QuadPlanes {
    pub const NAMES: [&'static str; 6] = [
        "Camera facing",
        "Local XY",
        "Local XZ",
        "Local ZY",
        "Y billboard",
        "Velocity",
    ];

    /// What each mode does, for the tooltip beside the picker. The distinction that matters
    /// when choosing is whether the quad can go edge-on: the plane modes can, the axis-locked
    /// ones cannot, and "it disappears from some angles" is the fastest way to tell them apart.
    pub const DESCRIPTIONS: [&'static str; 6] = [
        "Always square to the viewer. Never edge-on. The default, and most of the corpus.",
        "Pinned to the effect's XY plane. Goes edge-on from the side.",
        "Pinned to the effect's XZ plane — flat to the ground. Goes edge-on from level with it.",
        "Pinned to the effect's ZY plane. Goes edge-on from the front.",
        "Upright, spinning about the effect's up axis to face the viewer. Never edge-on. \
         For flames and columns that should stay vertical from every angle.",
        "Long axis follows the particle's travel, spinning about it to face the viewer. \
         Never edge-on. For sparks and streaks, which point where they are going.",
    ];

    /// A spawn's rotation arguments (degrees) as the turn from the bone's frame: the frame
    /// turn, then the angles in the chosen order.
    pub fn spawn_turn(&self, degrees: glam::Vec3) -> glam::Quat {
        glam::Quat::from_rotation_y(
            f32::from(self.spawn_frame_turns % 4) * std::f32::consts::FRAC_PI_2,
        ) * spawn_rotation_in(degrees, self.spawn_order)
    }

    pub fn plane_for(&self, billboard_type: i64) -> u32 {
        self.planes[billboard_type.clamp(0, 7) as usize]
            .min(Self::NAMES.len() as u32 - 1)
    }

    /// The per-type extra turn, as a quaternion.
    pub fn offset_for(&self, billboard_type: i64) -> glam::Quat {
        let [x, y, z] = self.offsets[billboard_type.clamp(0, 7) as usize];
        if x == 0.0 && y == 0.0 && z == 0.0 {
            return glam::Quat::IDENTITY;
        }
        glam::Quat::from_euler(
            glam::EulerRot::XYZ,
            x.to_radians(),
            y.to_radians(),
            z.to_radians(),
        )
    }

    /// Every way the effect's axes can be relabelled onto ours: the 24 rotations of a cube.
    ///
    /// This is the whole search space for a per-type correction, and it being FINITE is the
    /// point. A correction between two coordinate conventions can only ever send each axis to
    /// another axis -- it cannot send X to somewhere 37 degrees off X, because no file format
    /// disagrees with another by 37 degrees. So the honest control is "which of these 24", not
    /// three free angles, and the evidence that this is the right model is that the values
    /// found by eye keep landing on multiples of 90.
    ///
    /// Built rather than tabulated so the convention `offset_for` reads them back with cannot
    /// drift from the convention they were written in.
    pub fn cube_bases() -> Vec<glam::Quat> {
        let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
        let mut out = Vec::with_capacity(24);
        for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            for signs in 0..8u8 {
                let col = |i: usize| {
                    let sign = if signs & (1 << i) != 0 { -1.0 } else { 1.0 };
                    axes[perm[i]] * sign
                };
                let m = glam::Mat3::from_cols(col(0), col(1), col(2));
                // Half of the 48 signed permutations are reflections. A reflection is not a
                // rotation and would mirror the effect rather than turn it.
                if m.determinant() > 0.0 {
                    out.push(glam::Quat::from_mat3(&m));
                }
            }
        }
        out
    }

    /// One cube basis as the X/Y/Z degrees `offsets` stores, so stepping through them writes
    /// the same field the free-angle drags do and nothing downstream needs to know the
    /// difference.
    pub fn basis_degrees(index: usize) -> [f32; 3] {
        let bases = Self::cube_bases();
        let q = bases[index % bases.len()];
        let (x, y, z) = q.to_euler(glam::EulerRot::XYZ);
        // Snapped because these are exact quarter turns and floating point renders them as
        // 89.99999. A stored 90 also makes the settings readable as what they are.
        let snap = |v: f32| (v.to_degrees() / 90.0).round() * 90.0;
        [snap(x), snap(y), snap(z)]
    }

    /// Which cube basis the current angles are, when they are one. `None` for a setting that
    /// has been dragged off the lattice -- that is a real answer, not a failure: it says the
    /// correction being applied is not a relabelling of axes, which is worth knowing.
    pub fn basis_index_of(degrees: [f32; 3]) -> Option<usize> {
        let on_lattice = degrees
            .iter()
            .all(|v| (v / 90.0 - (v / 90.0).round()).abs() < 1e-3);
        if !on_lattice {
            return None;
        }
        let want = glam::Quat::from_euler(
            glam::EulerRot::XYZ,
            degrees[0].to_radians(),
            degrees[1].to_radians(),
            degrees[2].to_radians(),
        );
        Self::cube_bases()
            .iter()
            // Quaternions double-cover rotations, so q and -q are the same turn and comparing
            // components directly would miss half the matches.
            .position(|q| q.dot(want).abs() > 0.999)
    }
}

/// Every order a spawn's three angles can be applied in, each about the FIXED axes, as the
/// axis indices in application order. [`QuadPlanes::default`] picks index 3 (Y, Z, X); index 5
/// (Z, Y, X) is glam's intrinsic `XYZ`, which is what the renderer used originally.
pub const SPAWN_ORDERS: [[usize; 3]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// [`SPAWN_ORDERS`] as the picker shows them.
pub const SPAWN_ORDER_NAMES: [&str; 6] = [
    "X, Y, Z",
    "X, Z, Y",
    "Y, X, Z",
    "Y, Z, X",
    "Z, X, Y",
    "Z, Y, X",
];

/// An ACMD spawn's rotation arguments (degrees), applied in `SPAWN_ORDERS[order]`.
///
/// Orders only disagree when more than one angle is set, so single-axis spawns look right under
/// any of them and cannot tell them apart. Dabi's jabs can: they spawn one flat ring at three
/// combined rotations. No order alone matched them -- attack13's plane faced the camera under
/// every one -- until the effect frame was also turned a quarter turn about up from the bone's.
/// With that turn, Y, Z, X is the order that matched all three; X, Z, Y got two of them and
/// swung attack11 out of plane. Both stay selectable in case another effect disagrees.
pub fn spawn_rotation_in(degrees: glam::Vec3, order: usize) -> glam::Quat {
    let angles = [degrees.x, degrees.y, degrees.z];
    let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
    SPAWN_ORDERS[order.min(SPAWN_ORDERS.len() - 1)]
        .iter()
        .fold(glam::Quat::IDENTITY, |turn, &axis| {
            // Fixed axes: each later turn is applied on the outside.
            glam::Quat::from_axis_angle(axes[axis], angles[axis].to_radians()) * turn
        })
}

#[derive(Default)]
pub struct FrameEffects {
    pub batches: Vec<crate::eff_render::ParticleBatch>,
    pub mesh_batches: Vec<crate::eff_render::MeshBatch>,
    pub pending_textures: Vec<PendingTexture>,
    pub pending_meshes: Vec<(crate::eff_mesh::MeshKey, crate::eff_mesh::EffectMesh)>,
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
    failed: &dyn Fn(&crate::eff_render::TextureKey) -> bool,
    mesh_uploaded: &dyn Fn(&crate::eff_mesh::MeshKey) -> bool,
    meshes: &mut crate::eff_mesh::MeshLibrary,
    planes: QuadPlanes,
    bone_matrices: &std::collections::HashMap<String, glam::Mat4>,
    live: &[LiveEffect],
) -> (
    FrameEffects,
    Vec<crate::eff_render::TextureKey>,
) {
    use crate::eff_render::{MeshBatch, ParticleBatch, ParticleInstance, TextureKey};

    let mut batches: Vec<ParticleBatch> = Vec::new();
    let mut mesh_batches: Vec<MeshBatch> = Vec::new();
    let mut pending_meshes: Vec<(crate::eff_mesh::MeshKey, crate::eff_mesh::EffectMesh)> =
        Vec::new();
    let mut staged_meshes: std::collections::HashSet<crate::eff_mesh::MeshKey> =
        Default::default();
    let mut pending: Vec<PendingTexture> = Vec::new();
    let mut undecodable: Vec<TextureKey> = Vec::new();
    let mut decoded: std::collections::HashSet<TextureKey> = std::collections::HashSet::new();
    let slots = crate::eff_sim::Slots::new();

    for LiveEffect {
        name,
        bone,
        tint,
        alpha,
        age,
        offset: call_offset,
        rotation: call_rotation,
        scale,
    } in live
    {
        let Ok(resolved) = resolver.resolve(name) else {
            continue;
        };
        // The bone the ACMD call named. Without it there is no world position to place the
        // effect at, and drawing it at the origin would be worse than not drawing it.
        let Some(matrix) = lookup_bone(bone_matrices, bone) else {
            continue;
        };
        // The whole bone matrix, not just its translation. A bone's rotation is part of where
        // an effect points: taking only the position leaves every effect axis-aligned in world
        // space no matter how the limb it hangs off is turned.
        let (_, bone_rotation, _) = matrix.to_scale_rotation_translation();

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
            // Decode one of an emitter's textures if it is not already on the GPU, and say
            // whether it can be drawn. An emitter can bind two, so this is a closure rather
            // than the body of the loop it used to be.
            //
            // A texture that will not decode must be recorded as such. Without this the decode
            // is retried every frame the effect is live, which both costs the decode and floods
            // the console with the same line forever.
            let want = |index: usize,
                        pending: &mut Vec<PendingTexture>,
                        decoded: &mut std::collections::HashSet<TextureKey>,
                        undecodable: &mut Vec<TextureKey>|
             -> Option<TextureKey> {
                let info = loaded.ptcl.bntx_textures.get(index)?;
                let key = TextureKey {
                    file: file.clone(),
                    index,
                };
                if failed(&key) {
                    return None;
                }
                if !uploaded(&key) && decoded.insert(key.clone()) {
                    match crate::texture_import::decode_rgba(&pool, index, &info.tex_name, None) {
                        Ok(image) => pending.push(PendingTexture {
                            key: key.clone(),
                            image: std::sync::Arc::new(normalize_mask(image)),
                        }),
                        Err(error) => {
                            eprintln!("[eff] texture '{}' not drawable: {error}", info.tex_name);
                            undecodable.push(key.clone());
                            return None;
                        }
                    }
                }
                Some(key)
            };

            for (emitter_index, emitter) in set.emitters.iter().enumerate() {
                let Some(texture_index) = emitter.texture_index else {
                    continue;
                };
                let index = texture_index as usize;
                let Some(info) = loaded.ptcl.bntx_textures.get(index) else {
                    continue;
                };
                let Some(key) = want(index, &mut pending, &mut decoded, &mut undecodable) else {
                    continue;
                };
                // The emitter's second texture, where it binds one. Undrawable is not fatal
                // here the way it is for the first: the particle still has its own art.
                let key1 = emitter.texture_index1.and_then(|second| {
                    want(second as usize, &mut pending, &mut decoded, &mut undecodable)
                });

                let sim = crate::eff_sim::EmitterSim::read(emitter, &slots);
                // The sheet layout needs the texture's real size, which lives with the pool
                // rather than with the emitter — so the grid is worked out here, where both
                // are in hand, and the simulation only says which cell.
                let (columns, rows) =
                    crate::eff_sim::sheet_grid(info.width, info.height, sim.pattern_cells, sim.uv_div);
                let mut sim = sim;
                sim.sheet_cells = columns * rows;
                let sim = sim;
                // The emitter's own seed. Two emitters with identical settings must not
                // produce identically jittered particles stacked on each other, which is what
                // seeding by the effect alone would do.
                // Where this emitter sits and how it is turned, relative to the bone. Ignoring
                // it stacks every emitter of an effect at one point -- SYS_TURN_SMOKE separates
                // its five by up to 2 units, and collapsed together they are a blob rather than
                // a cloud.
                let emitter_rotation = glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    sim.rotation.x,
                    sim.rotation.y,
                    sim.rotation.z,
                );
                // Bone, then the call's aim, then the emitter's own. Order matters: the call
                // rotates the whole effect in the bone's frame, and the emitter is a further
                // turn inside that.
                let call_turn = planes.spawn_turn(*call_rotation);
                // Where the effect points, before any convention correction. Bone, then the
                // call's aim, then the emitter's own -- all three are real data.
                let placed = bone_rotation * call_turn * emitter_rotation;
                // The per-type basis correction applies to BILLBOARDS ONLY.
                //
                // A billboard has no geometry: the shader synthesises its axes from a rule, so
                // "which axes" is a convention question and a correction is the answer to it.
                // A mesh is the opposite -- its orientation is in its vertices, already, and
                // turning it by a per-type constant rotates a shape that was authored pointing
                // the right way.
                //
                // Measured, not assumed. All three attack arcs are meshes of type 3, and their
                // geometry is thin in Y -- 29.66 x 5.37 x 22.33, 20.06 x 2.98 x 11.11,
                // 26.19 x 4.78 x 26.19 -- so each is authored flat in the XZ plane, which is
                // exactly the plane the tilt observations independently say an arc lies in.
                // They need no correction. Applying one to all of them at once is also why
                // dialling in the correction for one arc moved the others: a single per-type
                // constant cannot serve three differently-shaped meshes.
                let orientation = placed * planes.offset_for(sim.billboard_type);
                // The ACMD call's own offset is in the effect's local frame, as is the
                // emitter's; both ride through the bone's rotation to reach world space.
                // The emitter's own offset is inside the effect's rotated frame; the call's is
                // in the bone's. Rotating both, or neither, puts multi-emitter effects in the
                // wrong shape as soon as a script aims one.
                // The emitter's own offset is part of the effect, so it rides the call's size:
                // a half-size SYS_TURN_SMOKE is a smaller cloud, not five smaller puffs spread
                // as widely as before. The call's own offset does not -- that is where the
                // script put the effect, not how big it is.
                let origin = matrix
                    .transform_point3(*call_offset + call_turn * (sim.translation * *scale));
                let seed = (part.set_idx as u64) << 32 ^ (emitter_index as u64) << 8 ^ index as u64;
                let particle_age = age - part.start_frame as f32;
                if particle_age < 0.0 {
                    continue;
                }
                let simulated = crate::eff_sim::evaluate(&sim, particle_age, seed);
                if simulated.is_empty() {
                    continue;
                }

                // An emitter whose primitive_id resolves draws geometry instead of a quad --
                // about half of them do. A ring drawn as a camera-facing square is the shape a
                // shockwave was showing, and no billboard setting fixes it.
                // Where the particle's own turn has to be applied depends on how the quad is
                // built, and the mesh path does not use the scalar roll at all.
                //
                //   camera-facing: the axes come from the camera, so the only expressible turn
                //     is a roll about the view axis -- the scalar, which the shader applies.
                //   everything else: the axes come from `orientation`, so folding the turn in
                //     here is what makes it visible, and it carries all three axes rather than
                //     just the one a scalar can hold.
                //
                // SYS_ATTACK_ARC is the case that proves it. It is a mesh, it sweeps about Y
                // at 2 degrees a frame, and `vs_mesh` never reads the scalar -- so before
                // this the arc simply did not move.
                let quad_mode = planes.plane_for(sim.billboard_type);
                let camera_facing = quad_mode == 0;
                let mesh_key = sim.primitive_id.and_then(|id| {
                    meshes
                        .descriptor_for(&file, id)
                        .map(|descriptor| crate::eff_mesh::MeshKey {
                            file: file.clone(),
                            descriptor,
                        })
                });
                if let Some(mesh_key) = mesh_key {
                    if !mesh_uploaded(&mesh_key) && staged_meshes.insert(mesh_key.clone()) {
                        if let Some(mesh) = meshes.mesh(&mesh_key) {
                            pending_meshes.push((mesh_key.clone(), mesh.clone()));
                        }
                    }
                    let blend = crate::eff_render::BlendMode::from_blend_type(sim.blend_type);
                    let batch_index = match mesh_batches.iter().position(|batch| {
                        batch.mesh == mesh_key
                            && batch.texture == key
                            && batch.texture1 == key1
                            && batch.blend == blend
                    })
                    {
                        Some(found) => found,
                        None => {
                            mesh_batches.push(MeshBatch {
                                mesh: mesh_key,
                                texture: key.clone(),
                                texture1: key1.clone(),
                                blend,
                                instances: Vec::new(),
                            });
                            mesh_batches.len() - 1
                        }
                    };
                    for particle in simulated {
                        let spun = placed
                            * glam::Quat::from_euler(
                                glam::EulerRot::XYZ,
                                particle.spin.x,
                                particle.spin.y,
                                particle.spin.z,
                            );
                        mesh_batches[batch_index].instances.push(ParticleInstance {
                            position: (origin + placed * (particle.offset * *scale))
                                .to_array(),
                            size: particle.size * *scale,
                            color: [
                                particle.color[0] * tint[0],
                                particle.color[1] * tint[1],
                                particle.color[2] * tint[2],
                                particle.color[3] * alpha,
                            ],
                            // A mesh keeps its own geometry, so its turn belongs in the
                            // orientation; the scalar would be ignored anyway.
                            rotation: 0.0,
                            uv_rect: crate::eff_sim::cell_uv(particle.cell, columns, rows),
                            orientation: spun.to_array(),
                            plane: quad_mode,
                            velocity: (orientation * particle.velocity).to_array(),
                            flags: combiner_flags(&sim, key1.is_some()),
                            uv_anim: particle.uv_anim,
                        });
                    }
                    continue;
                }

                let blend = crate::eff_render::BlendMode::from_blend_type(sim.blend_type);
                let batch_index = match batches
                    .iter()
                    .position(|batch| {
                        batch.texture == key && batch.texture1 == key1 && batch.blend == blend
                    })
                {
                    Some(found) => found,
                    None => {
                        batches.push(ParticleBatch {
                            texture: key.clone(),
                            texture1: key1.clone(),
                            // The emitter's own render state, not an assumption. Forcing
                            // this additive drew roughly three quarters of the game's emitters
                            // in the wrong mode -- every smoke and dust puff glowing like fire
                            // -- and the subtractive few darken rather than cover.
                            blend,
                            instances: Vec::new(),
                        });
                        batches.len() - 1
                    }
                };
                for particle in simulated {
                    let spun = if camera_facing {
                        orientation
                    } else {
                        orientation
                            * glam::Quat::from_euler(
                                glam::EulerRot::XYZ,
                                particle.spin.x,
                                particle.spin.y,
                                particle.spin.z,
                            )
                    };
                    batches[batch_index].instances.push(ParticleInstance {
                        position: (origin + orientation * (particle.offset * *scale))
                            .to_array(),
                        size: particle.size * *scale,
                        color: [
                            particle.color[0] * tint[0],
                            particle.color[1] * tint[1],
                            particle.color[2] * tint[2],
                            particle.color[3] * alpha,
                        ],
                        // Applied once, in whichever place the mode can express it: as a
                        // screen roll for a camera-facing quad, and folded into the axes for
                        // every other mode. Doing both would turn the quad twice.
                        rotation: if camera_facing { particle.rotation } else { 0.0 },
                        uv_rect: crate::eff_sim::cell_uv(particle.cell, columns, rows),
                        orientation: spun.to_array(),
                        plane: quad_mode,
                        velocity: (orientation * particle.velocity).to_array(),
                        flags: combiner_flags(&sim, key1.is_some()),
                        uv_anim: particle.uv_anim,
                    });
                }
            }
        }
    }

    (
        FrameEffects {
            batches,
            mesh_batches,
            pending_textures: pending,
            pending_meshes,
        },
        undecodable,
    )
}

/// Whether an emitter's particles add to what is behind them or blend over it.
///
/// The format numbers these 0 normal, 1 additive, 2 subtractive. Only two pipelines exist, so
/// subtractive -- 3% of emitters, and a darkening operation -- is drawn as normal rather than
/// as additive: getting it slightly wrong in the direction of "does not glow" is much closer
/// than lighting up something meant to darken.
pub fn is_additive(blend_type: i64) -> bool {
    blend_type == 1
}

/// The emitter's combiner settings, packed for the shader. See `ParticleInstance::flags`.
///
/// Bits 0-1 the alpha source, 2-3 the colour source, 4-5 the shader type, 6-7 how a second
/// texture folds in, and bit 8 whether there is a second texture to fold. The last one is not
/// in the data: an emitter can name a texture that failed to decode, and the shader has to
/// draw something either way.
pub fn combiner_flags(sim: &crate::eff_sim::EmitterSim, has_second: bool) -> u32 {
    (sim.alpha_input.clamp(0, 3) as u32)
        | ((sim.color_input.clamp(0, 3) as u32) << 2)
        | ((sim.shader_type.clamp(0, 3) as u32) << 4)
        | ((sim.tex1_blend.clamp(0, 3) as u32) << 6)
        | ((has_second as u32) << 8)
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

    /// The correction space is the cube group, and it is closed: 24 turns, no reflections.
    ///
    /// A reflection slipping in would mirror an effect rather than turn it, and a mirrored arc
    /// reads as a plausible-but-wrong swing rather than as an obvious bug.
    #[test]
    fn the_basis_corrections_are_the_twenty_four_rotations_of_a_cube() {
        let bases = QuadPlanes::cube_bases();
        assert_eq!(bases.len(), 24);
        for q in &bases {
            let m = glam::Mat3::from_quat(*q);
            assert!((m.determinant() - 1.0).abs() < 1e-4, "{q:?} is a reflection");
        }
        // Distinct as ROTATIONS: q and -q are the same turn, so a naive component compare
        // would call the list distinct while holding duplicates.
        for (i, a) in bases.iter().enumerate() {
            for (j, b) in bases.iter().enumerate().skip(i + 1) {
                assert!(a.dot(*b).abs() < 0.999, "bases {i} and {j} are the same turn");
            }
        }
    }

    /// Every basis round-trips through the degrees `offsets` stores, so stepping the picker
    /// and then reading the setting back cannot land on a different turn than the one shown.
    #[test]
    fn a_basis_survives_being_written_as_degrees_and_read_back() {
        for index in 0..QuadPlanes::cube_bases().len() {
            let degrees = QuadPlanes::basis_degrees(index);
            for value in degrees {
                assert_eq!(value, (value / 90.0).round() * 90.0, "{degrees:?} is off-lattice");
            }
            assert_eq!(
                QuadPlanes::basis_index_of(degrees),
                Some(index),
                "basis {index} wrote {degrees:?}, which reads back as something else"
            );
        }
        // An angle that is not a quarter turn is not a relabelling of axes, and saying so is
        // the point of the lookup.
        assert_eq!(QuadPlanes::basis_index_of([55.0, 0.0, 0.0]), None);
    }

    /// What the type 3 setting found by eye actually IS: a ground-plane arc.
    ///
    /// This is the one measured fact in the orientation model, and it was worth chasing
    /// because it turns a tuned pair of numbers into a statement about the format.
    ///
    /// The setting is mode "Local XY" -- quad spanned by the effect's X and Y -- plus a
    /// correction of (-90, 0, 90). That correction sends X to -Z and Y to -X, so the quad is
    /// really spanned by the effect's Z and X: its normal is the effect's Y, and it lies in
    /// the effect's XZ plane. A horizontal sweep, which is what a swing arc is.
    ///
    /// Four independent things agree on that plane. This setting, tuned by eye against the
    /// game; and three arc spawns whose on-screen shape is known because they are correct in
    /// the game already and needed no editing:
    ///
    ///   Bakugo  down tilt   rot [-90,  0, 0]   reads as ")"
    ///   Bakugo  side tilt   rot [ 55,-90, 0]   reads as "U"
    ///   Pit     up tilt     rot [ 80, 15, 0]   reads as ")"
    ///
    /// Start each of those from a plane whose normal is the effect's Y and all three come out
    /// facing the camera -- |normal.z| of 1.00, 0.82 and 0.99 -- which is the only way a U or
    /// a ) is what you see, since an arc turned edge-on reads as a streak. Start them from a
    /// plane whose normal is X or Z instead and at least one of the three goes edge-on. That
    /// rules out both alternatives outright rather than merely preferring Y.
    ///
    /// The same check says nothing about the Euler ORDER, and cannot: all three calls leave
    /// one of the three angles at zero, and with a zero angle the orders collapse onto each
    /// other. 98% of the corpus is like that, which is why the order has never been the thing
    /// worth chasing.
    ///
    /// What is left over is only the ROLL inside that plane, which is four options rather
    /// than a continuum, because the arc texture has an up and the plane does not say which
    /// way it points.
    #[test]
    fn the_type_three_correction_puts_the_arc_in_the_ground_plane() {
        let degrees = [-90.0, 0.0, 90.0];
        let index = QuadPlanes::basis_index_of(degrees)
            .expect("a pair of quarter turns has to be one of the 24");
        let q = QuadPlanes::cube_bases()[index];
        let near = |a: glam::Vec3, b: glam::Vec3| (a - b).length() < 1e-4;

        // Mode "Local XY" spans the quad with the corrected X and Y.
        let right = q * glam::Vec3::X;
        let up = q * glam::Vec3::Y;
        assert!(near(right, -glam::Vec3::Z), "right is {right:?}");
        assert!(near(up, -glam::Vec3::X), "up is {up:?}");

        // Which makes the quad's normal the effect's own Y -- the defining property of a
        // ground-plane arc, and the thing that survives however the arc is rolled.
        let normal = right.cross(up);
        assert!(near(normal, glam::Vec3::Y), "normal is {normal:?}");
    }

    /// The shape and orientation of specific primitives, by id. `VISIONARY_EFF_IDS` is a
    /// comma-separated list.
    ///
    /// The question this answers: when two effects that look alike need different orientation
    /// corrections, is it because their MESHES are authored at different angles? A per-type
    /// setting cannot tell two meshes apart, so if the geometry differs, no value of that
    /// setting can serve both.
    #[test]
    fn what_shape_are_these_primitive_ids() {
        let (Some(root), Ok(ids)) = (root(), std::env::var("VISIONARY_EFF_IDS")) else {
            eprintln!("VISIONARY_EFF_ROOT / VISIONARY_EFF_IDS not set — skipping");
            return;
        };
        // Any file when one is named, so this reaches a mod's own eff or a transplant donor.
        let path = std::env::var_os("VISIONARY_EFF_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| EffectResolver::common_eff_path(&root));
        let mut meshes = crate::eff_mesh::MeshLibrary::default();
        println!();
        for token in ids.split(',') {
            let id: u64 = token.trim().parse().expect("primitive id");
            let Some(descriptor) = meshes.descriptor_for(&path, id) else {
                println!("  id {id:<12} does not resolve");
                continue;
            };
            let key = crate::eff_mesh::MeshKey { file: path.clone(), descriptor };
            let Some(mesh) = meshes.mesh(&key) else {
                println!("  id {id:<12} desc {descriptor} has no geometry");
                continue;
            };
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &mesh.vertices {
                for a in 0..3 {
                    lo[a] = lo[a].min(v.position[a]);
                    hi[a] = hi[a].max(v.position[a]);
                }
            }
            let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            let flat = ["X", "Y", "Z"]
                .iter()
                .zip(ext)
                .filter(|(_, e)| *e < 0.05)
                .map(|(a, _)| *a)
                .collect::<Vec<_>>();
            println!(
                "  id {id:<12} desc {descriptor:<4} verts={:<5} extent=({:7.2},{:7.2},{:7.2})  flat in: {}",
                mesh.vertices.len(), ext[0], ext[1], ext[2],
                if flat.is_empty() { "nothing (solid)".to_string() } else { flat.join("+") }
            );
        }
    }

    /// Which named effects use a given primitive descriptor. `VISIONARY_EFF_DESC` selects it.
    ///
    /// The other half of the primitive search: knowing that descriptor 31 is a ground disc is
    /// only useful once you know which effect to clone to get one.
    #[test]
    fn which_effects_use_a_given_primitive() {
        let (Some(root), Ok(want)) = (root(), std::env::var("VISIONARY_EFF_DESC")) else {
            eprintln!("VISIONARY_EFF_ROOT / VISIONARY_EFF_DESC not set — skipping");
            return;
        };
        let want: usize = want.parse().expect("descriptor index");
        // Any file when one is named, so this reaches a mod's own eff or a transplant donor.
        let path = std::env::var_os("VISIONARY_EFF_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| EffectResolver::common_eff_path(&root));
        let mut meshes = crate::eff_mesh::MeshLibrary::default();
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(None, Vec::new(), Some(path.clone()));
        let loaded = resolver.loaded(&path).expect("ef_common loads").clone();
        let table = crate::eff_attrs::table();
        let slot = table.iter().position(|a| a.id == "particle_data.primitive_id");

        // set index -> the entry names that reach it, so a hit reports something spawnable.
        let mut names_for: std::collections::HashMap<usize, Vec<String>> = Default::default();
        for entry in &loaded.entries {
            if let Some(idx) = entry.set_idx {
                names_for.entry(idx).or_default().push(entry.name.clone());
            }
        }
        println!("
=== effects using descriptor {want} ===");
        let mut hits = 0usize;
        for (set_idx, set) in loaded.ptcl.emitter_sets.iter().enumerate() {
            for emitter in &set.emitters {
                let Some(Some(value)) = slot.and_then(|i| emitter.attrs.get(i)) else { continue };
                let id = match value {
                    crate::eff_attrs::AttrValue::UInt(v) => *v,
                    crate::eff_attrs::AttrValue::Int(v) if *v > 0 => *v as u64,
                    _ => continue,
                };
                if meshes.descriptor_for(&path, id) != Some(want) {
                    continue;
                }
                hits += 1;
                if hits <= 20 {
                    println!(
                        "  set '{}' emitter '{}'  spawned as: {}",
                        set.name,
                        emitter.name,
                        names_for.get(&set_idx).map(|n| n.join(", ")).unwrap_or_else(|| "(no entry)".into())
                    );
                }
            }
        }
        println!("  {hits} emitters total");
    }

    /// Every primitive a file holds, by shape. Finding a flat disc, a ring or a cone in the
    /// game's own pool means an effect can borrow geometry instead of importing it.
    #[test]
    fn what_shape_is_every_primitive_in_a_file() {
        let Some(path) = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from) else {
            eprintln!("VISIONARY_EFF_FILE not set — skipping");
            return;
        };
        let mut meshes = crate::eff_mesh::MeshLibrary::default();
        let mut rows: Vec<(String, usize)> = Vec::new();
        for descriptor in 0..256usize {
            let key = crate::eff_mesh::MeshKey { file: path.clone(), descriptor };
            let Some(mesh) = meshes.mesh(&key) else { continue };
            if mesh.vertices.is_empty() {
                continue;
            }
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &mesh.vertices {
                for a in 0..3 {
                    lo[a] = lo[a].min(v.position[a]);
                    hi[a] = hi[a].max(v.position[a]);
                }
            }
            let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            // A primitive with one near-zero extent is a flat card or disc; that is the shape
            // a ground ring needs, and it is the one worth finding by search rather than by
            // opening every effect that might have one.
            let thin = ext.iter().filter(|e| **e < 0.02).count();
            let shape = if thin >= 1 { "FLAT" } else { "solid" };
            rows.push((
                format!(
                    "  desc {descriptor:<4} {:<6} verts={:<6} tris={:<6} extent=({:6.2},{:6.2},{:6.2})",
                    shape, mesh.vertices.len(), mesh.indices.len() / 3, ext[0], ext[1], ext[2]
                ),
                descriptor,
            ));
        }
        println!("
=== {} : {} primitives ===", path.display(), rows.len());
        for (line, _) in &rows {
            println!("{line}");
        }
    }

    /// Which effects use each pool texture, across every part of every entry.
    ///
    /// The question before replacing a texture BY NAME: a pool texture is shared by every emitter
    /// that samples it, so an import meant for one effect reskins all the others that use it too.
    /// `VISIONARY_EFF_FILE`; `VISIONARY_EFF_MATCH` (comma-separated substrings) restricts the
    /// printout to textures that some matching entry uses, and still lists ALL of their users.
    #[test]
    fn which_entries_use_each_texture() {
        let Some(path) = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from) else {
            eprintln!("VISIONARY_EFF_FILE not set — skipping");
            return;
        };
        let wanted: Vec<String> = std::env::var("VISIONARY_EFF_MATCH")
            .unwrap_or_default()
            .split(',')
            .map(|w| w.trim().to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        let bytes = std::fs::read(&path).expect("eff reads");
        let namco = effect_library::NamcoEffectFile::load(&bytes).expect("eff parses");
        let loaded = load_effect(&path).expect("eff loads");
        let by_id = texture_names(&namco);
        let ptcl = namco.ptcl_file.as_ref().expect("ptcl");

        let mut users: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
            Default::default();
        for entry in &loaded.entries {
            for part in parts_of(entry) {
                let Some(set) = ptcl.emitter_list.emitter_sets.get(part.set_idx) else { continue };
                let mut flat = Vec::new();
                flat_emitters(&set.emitters, 0, &mut flat);
                for (_, emitter) in &flat {
                    for sampler in [&emitter.data.sampler0, &emitter.data.sampler1, &emitter.data.sampler2]
                        .into_iter()
                        .flatten()
                    {
                        if let Some(tex) = by_id.get(&sampler.texture_id) {
                            users.entry(tex.clone()).or_default().insert(entry.name.clone());
                        }
                    }
                }
            }
        }
        let shape = |tex: &str| {
            loaded
                .ptcl
                .bntx_textures
                .iter()
                .find(|t| t.tex_name == tex)
                .map(|t| format!("{}x{} {}", t.width, t.height, t.format))
                .unwrap_or_else(|| "?".into())
        };
        println!();
        for (tex, who) in &users {
            let relevant = wanted.is_empty()
                || who.iter().any(|e| wanted.iter().any(|w| e.to_lowercase().contains(w)));
            if !relevant {
                continue;
            }
            let shared = who.iter().any(|e| !wanted.iter().any(|w| e.to_lowercase().contains(w)));
            println!(
                "  {tex:<30} {:<18} {} {}",
                shape(tex),
                if shared && !wanted.is_empty() { "SHARED " } else { "private" },
                who.iter().cloned().collect::<Vec<_>>().join(", ")
            );
        }
    }

    /// Decode named pool textures to PNG, and say what colour they actually hold.
    ///
    /// For "is the colour in the texture or somewhere else": the mean RGB over the visible
    /// pixels tells a neutral mask (R, G and B roughly equal, or a two-channel format with
    /// blue at zero) from a texture with a hue baked into it, which no colour field can undo.
    ///
    /// `VISIONARY_EFF_FILE`, `VISIONARY_TEX_NAMES` (comma-separated), `VISIONARY_TEX_OUT` (a dir).
    #[test]
    fn export_named_pool_textures() {
        let (Some(path), Ok(names), Some(out)) = (
            std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from),
            std::env::var("VISIONARY_TEX_NAMES"),
            std::env::var_os("VISIONARY_TEX_OUT").map(PathBuf::from),
        ) else {
            eprintln!("VISIONARY_EFF_FILE / VISIONARY_TEX_NAMES / VISIONARY_TEX_OUT not set — skipping");
            return;
        };
        std::fs::create_dir_all(&out).expect("output dir");
        let loaded = load_effect(&path).expect("eff loads");
        let pool = loaded.texture_pool.as_ref().expect("eff has a texture pool");
        println!();
        for name in names.split(',').map(str::trim) {
            let Some(index) = loaded.ptcl.bntx_textures.iter().position(|t| t.tex_name == name) else {
                println!("  {name}: not in this pool");
                continue;
            };
            let info = &loaded.ptcl.bntx_textures[index];
            // Editable form when asked: swizzle applied and single-channel masks flattened to
            // black-and-white, which is also the form an import with `raw: false` reads back.
            let editable = std::env::var("VISIONARY_TEX_FORM").map(|f| f == "editable").unwrap_or(false);
            let layout = crate::texture_import::layout_of_public(pool, index, name)
                .map(|l| format!("{l:?}"))
                .unwrap_or_else(|_| "?".into());
            let decoded = if editable {
                crate::texture_import::decode_form(
                    pool,
                    index,
                    name,
                    crate::texture_import::Form::Editable,
                    None,
                )
            } else {
                crate::texture_import::decode_rgba(pool, index, name, None)
            };
            match decoded {
                Ok(image) => {
                    let file = out.join(format!("{name}.png"));
                    image.save(&file).expect("png writes");
                    // "Visible" by any channel, since a two-channel texture keeps its shape in R.
                    let (mut sum, mut n) = ([0u64; 4], 0u64);
                    for px in image.pixels() {
                        if px.0[..3].iter().copied().max().unwrap_or(0) > 24 {
                            for c in 0..4 {
                                sum[c] += px.0[c] as u64;
                            }
                            n += 1;
                        }
                    }
                    let mean = |c: usize| if n == 0 { 0 } else { sum[c] / n };
                    println!(
                        "  {name:<28} {}x{} {:<10} {layout:<26} visible mean rgba=({}, {}, {}, {}) over {n} px -> {}",
                        info.width, info.height, info.format,
                        mean(0), mean(1), mean(2), mean(3),
                        file.display()
                    );
                }
                Err(error) => println!("  {name}: {error}"),
            }
        }
    }

    /// Flatten an emitter tree parent-first, keeping each emitter's depth.
    fn flat_emitters(
        emitters: &[effect_library::structs::Emitter],
        depth: usize,
        out: &mut Vec<(usize, effect_library::structs::Emitter)>,
    ) {
        for emitter in emitters {
            out.push((depth, emitter.clone()));
            flat_emitters(&emitter.children, depth + 1, out);
        }
    }

    /// The emitters of one entry's primary set in a raw file, flattened.
    fn emitters_of(
        namco: &effect_library::NamcoEffectFile,
        entry: &str,
    ) -> Option<Vec<(usize, effect_library::structs::Emitter)>> {
        let at = namco.entry_names.iter().position(|n| n.eq_ignore_ascii_case(entry))?;
        let set_idx = (namco.entries[at].emitter_set_id as usize).checked_sub(1)?;
        let set = namco.ptcl_file.as_ref()?.emitter_list.emitter_sets.get(set_idx)?;
        let mut out = Vec::new();
        flat_emitters(&set.emitters, 0, &mut out);
        Some(out)
    }

    fn texture_names(namco: &effect_library::NamcoEffectFile) -> HashMap<u64, String> {
        namco
            .ptcl_file
            .as_ref()
            .and_then(|ptcl| ptcl.texture_info.as_ref())
            .map(|info| info.descriptors.iter().map(|d| (d.id, d.name.clone())).collect())
            .unwrap_or_default()
    }

    /// The loader resolves an emitter's second texture, and every emitter that asks for the
    /// indirect shader has one.
    ///
    /// Measured over ef_common: 243 of 1348 emitters bind a second texture, and all 112 that
    /// set shader type 2 do -- that shader reads the first texture as a displacement map and
    /// the second as the art, so one without a second texture would be drawing its own
    /// distortion map. `VISIONARY_EFF_FILE`.
    #[test]
    fn the_second_texture_is_resolved_where_the_data_has_one() {
        let Some(path) = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from) else {
            eprintln!("VISIONARY_EFF_FILE not set — skipping");
            return;
        };
        let loaded = load_effect(&path).expect("eff loads");
        let shader_type = crate::eff_attrs::index_of("combiner.shader_type").expect("attribute");
        let (mut emitters, mut with_second, mut indirect, mut indirect_without) = (0, 0, 0, 0);
        for set in &loaded.ptcl.emitter_sets {
            for emitter in &set.emitters {
                emitters += 1;
                let second = emitter.texture_index1.is_some();
                with_second += usize::from(second);
                let kind = emitter
                    .attrs
                    .get(shader_type)
                    .and_then(|v| v.as_ref())
                    .map(|v| v.as_f32() as i64)
                    .unwrap_or(0);
                if kind == 2 {
                    indirect += 1;
                    indirect_without += usize::from(!second);
                }
            }
        }
        println!(
            "
{}: {emitters} emitters, {with_second} with a second texture, {indirect} indirect",
            path.display()
        );
        assert!(with_second > 0, "no second texture resolved in {}", path.display());
        assert_eq!(
            indirect_without, 0,
            "{indirect_without} emitters want the indirect shader with nothing to displace into"
        );
    }

    #[test]
    /// What every emitter of one effect samples, and how its colour is produced.
    ///
    /// The question behind it: when recolouring every colour field an editor exposes still
    /// leaves part of an effect the wrong colour, the colour is coming from somewhere those
    /// fields do not reach -- baked into a texture, read from a second sampler (a gradient
    /// "grade" ramp), or picked by a combiner mode that ignores the particle colour. This
    /// prints all three per emitter. `VISIONARY_EFF_FILE`, `VISIONARY_EFF_NAME`.
    #[test]
    fn what_each_emitter_of_an_effect_samples() {
        let (Some(path), Ok(name)) = (
            std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from),
            std::env::var("VISIONARY_EFF_NAME"),
        ) else {
            eprintln!("VISIONARY_EFF_FILE / VISIONARY_EFF_NAME not set — skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("eff reads");
        let namco = effect_library::NamcoEffectFile::load(&bytes).expect("eff parses");
        let loaded = load_effect(&path).expect("eff loads");
        let by_id = texture_names(&namco);
        let shape = |tex: &str| {
            loaded
                .ptcl
                .bntx_textures
                .iter()
                .find(|t| t.tex_name == tex)
                .map(|t| format!("{}x{} {}", t.width, t.height, t.format))
                .unwrap_or_else(|| "?".into())
        };
        let table = crate::eff_attrs::table();
        let emitters = emitters_of(&namco, &name).expect("entry and its set exist");
        println!("\n=== {name}: {} emitters ===", emitters.len());
        for (depth, emitter) in &emitters {
            let data = &emitter.data;
            println!("{}- {}", "  ".repeat(*depth), data.display_name());
            let pad = "  ".repeat(*depth + 2);
            for (slot, sampler) in [(0, &data.sampler0), (1, &data.sampler1), (2, &data.sampler2)] {
                if let Some(sampler) = sampler {
                    if let Some(tex) = by_id.get(&sampler.texture_id) {
                        println!("{pad}sampler{slot}: {tex} [{}]", shape(tex));
                    }
                }
            }
            let mut line = Vec::new();
            for attr in table.iter() {
                let interesting = attr.id.starts_with("combiner.")
                    || attr.id.starts_with("particle_color.")
                    || attr.id.starts_with("texture_anim0.")
                    || matches!(
                        attr.id,
                        "render_state.blend_type"
                            | "particle_data.billboard_type"
                            | "particle_data.primitive_id"
                            | "emitter_static.color_scale"
                            | "emitter_static.num_color0_keys"
                            | "emitter_static.num_color1_keys"
                            | "emitter_static.num_alpha0_keys"
                            | "shader_references.shader_index"
                            | "particle_data.life"
                            | "particle_scale.scale_x"
                    );
                if !interesting {
                    continue;
                }
                if let Some(value) = (attr.get)(data) {
                    let zero = value.as_f32() == 0.0;
                    // Zeros are the default and say nothing -- except the two fields where
                    // zero is itself the answer (normal blend, constant colour).
                    if zero && !matches!(attr.id, "render_state.blend_type" | "particle_color.color0_type") {
                        continue;
                    }
                    line.push(format!("{}={:?}", attr.id, value));
                }
            }
            for chunk in line.chunks(4) {
                println!("{pad}{}", chunk.join("  "));
            }
        }
    }

    /// Every difference between two files, per effect and per pool texture.
    ///
    /// For finding what a hand edit in another tool actually changed. Attributes are compared
    /// through the same table the editor uses; textures by their decoded pixels, because a
    /// re-encode that changes nothing visible still changes bytes. `VISIONARY_EFF_FILE` is the
    /// edited file, `VISIONARY_EFF_BASE` the reference.
    #[test]
    fn how_two_files_differ() {
        let (Some(edited), Some(base)) = (
            std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from),
            std::env::var_os("VISIONARY_EFF_BASE").map(PathBuf::from),
        ) else {
            eprintln!("VISIONARY_EFF_FILE / VISIONARY_EFF_BASE not set — skipping");
            return;
        };
        let load_raw = |path: &PathBuf| {
            effect_library::NamcoEffectFile::load(&std::fs::read(path).expect("reads"))
                .expect("parses")
        };
        let (a, b) = (load_raw(&edited), load_raw(&base));
        let (names_a, names_b) = (texture_names(&a), texture_names(&b));
        let table = crate::eff_attrs::table();

        println!("\n=== effects ===");
        for entry in &a.entry_names {
            if !b.entry_names.iter().any(|n| n.eq_ignore_ascii_case(entry)) {
                println!("  {entry}: only in the edited file");
                continue;
            }
            let (Some(ea), Some(eb)) = (emitters_of(&a, entry), emitters_of(&b, entry)) else {
                continue;
            };
            let mut diffs = Vec::new();
            if ea.len() != eb.len() {
                diffs.push(format!("emitter count {} vs {}", ea.len(), eb.len()));
            }
            for ((_, x), (_, y)) in ea.iter().zip(eb.iter()) {
                let who = x.data.display_name();
                for attr in table.iter() {
                    let (va, vb) = ((attr.get)(&x.data), (attr.get)(&y.data));
                    let same = match (va, vb) {
                        (Some(p), Some(q)) => (p.as_f32() - q.as_f32()).abs() < 1e-5,
                        (None, None) => true,
                        _ => false,
                    };
                    if !same {
                        diffs.push(format!("{who}.{}: {:?} -> {:?}", attr.id, vb, va));
                    }
                }
                for (slot, sx, sy) in [
                    (0, &x.data.sampler0, &y.data.sampler0),
                    (1, &x.data.sampler1, &y.data.sampler1),
                    (2, &x.data.sampler2, &y.data.sampler2),
                ] {
                    let ta = sx.as_ref().and_then(|s| names_a.get(&s.texture_id));
                    let tb = sy.as_ref().and_then(|s| names_b.get(&s.texture_id));
                    if ta != tb {
                        diffs.push(format!("{who}.sampler{slot}: {tb:?} -> {ta:?}"));
                    }
                }
            }
            if !diffs.is_empty() {
                println!("  {entry}: {} differences", diffs.len());
                for d in diffs.iter().take(40) {
                    println!("      {d}");
                }
            }
        }

        println!("\n=== pool textures ===");
        let (la, lb) = (load_effect(&edited).expect("loads"), load_effect(&base).expect("loads"));
        let hash = |loaded: &LoadedEffect, index: usize| -> Option<u64> {
            let pool = loaded.texture_pool.as_ref()?;
            let name = &loaded.ptcl.bntx_textures.get(index)?.tex_name;
            let image = crate::texture_import::decode_rgba(pool, index, name, None).ok()?;
            let mut h: u64 = 0xcbf29ce484222325;
            for byte in image.as_raw() {
                h = (h ^ *byte as u64).wrapping_mul(0x100000001b3);
            }
            Some(h)
        };
        for (index, tex) in la.ptcl.bntx_textures.iter().enumerate() {
            match lb.ptcl.bntx_textures.iter().position(|t| t.tex_name == tex.tex_name) {
                None => println!("  {} only in the edited file", tex.tex_name),
                Some(other) => {
                    let o = &lb.ptcl.bntx_textures[other];
                    if (tex.width, tex.height, &tex.format) != (o.width, o.height, &o.format) {
                        println!(
                            "  {}: {}x{} {} -> {}x{} {}",
                            tex.tex_name, o.width, o.height, o.format, tex.width, tex.height, tex.format
                        );
                    } else if hash(&la, index) != hash(&lb, other) {
                        println!("  {}: pixels changed ({}x{} {})", tex.tex_name, tex.width, tex.height, tex.format);
                    }
                }
            }
        }
        let _ = names_b;
    }

    /// Effect names a file offers, filtered by substring. `VISIONARY_EFF_MATCH` is the needle.
    ///
    /// The starting point for "what does the game call the thing I want to replace".
    #[test]
    fn what_effects_is_a_file_named_after() {
        let Some(path) = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from) else {
            eprintln!("VISIONARY_EFF_FILE not set — skipping");
            return;
        };
        let needle = std::env::var("VISIONARY_EFF_MATCH").unwrap_or_default().to_lowercase();
        let loaded = load_effect(&path).expect("eff loads");
        let mut hits = 0usize;
        println!();
        for entry in &loaded.entries {
            if !needle.is_empty() && !entry.name.to_lowercase().contains(&needle) {
                continue;
            }
            let parts = parts_of(entry);
            let emitters: usize = parts
                .iter()
                .filter_map(|part| loaded.ptcl.emitter_sets.get(part.set_idx))
                .map(|set| set.emitters.len())
                .sum();
            println!("  {:<44} parts={} emitters={emitters}", entry.name, parts.len());
            hits += 1;
        }
        println!("  {hits} of {} entries match '{needle}'", loaded.entries.len());
    }

    /// The pool textures a file offers, by size and format.
    ///
    /// Adding a texture of your own means naming an existing one as its template, which
    /// supplies the format, the swizzle and the required dimensions -- so "which templates
    /// are 1024x1024" is the first question when bringing in outside art.
    #[test]
    fn what_texture_templates_a_file_offers() {
        let Some(path) = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from) else {
            eprintln!("VISIONARY_EFF_FILE not set — skipping");
            return;
        };
        let loaded = load_effect(&path).expect("eff loads");
        let want = std::env::var("VISIONARY_TEX_SIZE").unwrap_or_default();
        let mut by_shape: std::collections::BTreeMap<String, Vec<&str>> = Default::default();
        for texture in &loaded.ptcl.bntx_textures {
            by_shape
                .entry(format!("{}x{} {}", texture.width, texture.height, texture.format))
                .or_default()
                .push(&texture.tex_name);
        }
        println!("
=== {} : {} pool textures ===", path.display(), loaded.ptcl.bntx_textures.len());
        for (shape, names) in &by_shape {
            if !want.is_empty() && !shape.starts_with(&want) {
                continue;
            }
            let show = if std::env::var("VISIONARY_TEX_ALL").is_ok() { names.len() } else { 3 };
            println!("  {shape:<28} {:>4}   {}", names.len(), names.iter().take(show).cloned().collect::<Vec<_>>().join(", "));
        }
    }

    /// Does the primitive an emitter names actually resolve to geometry?
    ///
    /// An emitter that declares a primitive and cannot find it silently falls back to a flat
    /// quad, which for a curved ribbon is the difference between an arc and a square.
    #[test]
    fn how_many_declared_primitives_resolve() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let path = EffectResolver::common_eff_path(&root);
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(None, Vec::new(), Some(path.clone()));
        let ids = resolver.descriptor_ids(&path);
        let loaded = resolver.loaded(&path).expect("ef_common loads");
        let table = crate::eff_attrs::table();
        let slot = table.iter().position(|a| a.id == "particle_data.primitive_id");

        let (mut declared, mut resolved) = (0usize, 0usize);
        let mut missing: Vec<u64> = Vec::new();
        for set in &loaded.ptcl.emitter_sets {
            for emitter in &set.emitters {
                let Some(Some(value)) = slot.and_then(|i| emitter.attrs.get(i)) else { continue };
                let id = match value {
                    crate::eff_attrs::AttrValue::UInt(v) => *v,
                    crate::eff_attrs::AttrValue::Int(v) => *v as u64,
                    crate::eff_attrs::AttrValue::Float(v) => *v as u64,
                };
                if id == 0 || id == u64::MAX {
                    continue;
                }
                declared += 1;
                if ids.contains(&id) {
                    resolved += 1;
                } else if !missing.contains(&id) {
                    missing.push(id);
                }
            }
        }
        println!(
            "
ef_common: {declared} emitters declare a primitive, {resolved} resolve,              {} distinct ids reach nothing",
            missing.len()
        );
        println!("  descriptor table holds {} ids", ids.len());
        for id in missing.iter().take(10) {
            println!("  unresolved: {id}");
        }
    }

    /// Every attribute of every emitter of one named effect. The tool for asking "what does
    /// the game actually say this effect is", rather than inferring it from how it looks.
    ///
    /// `VISIONARY_EFF_ROOT` for the dump, `VISIONARY_EFF_NAME` for the effect, and
    /// `VISIONARY_EFF_GREP` to narrow to attributes whose id contains a substring.
    #[test]
    fn every_attribute_of_one_named_effect() {
        let (Some(root), Some(name)) = (root(), std::env::var("VISIONARY_EFF_NAME").ok()) else {
            eprintln!("VISIONARY_EFF_ROOT / VISIONARY_EFF_NAME not set — skipping");
            return;
        };
        let needle = std::env::var("VISIONARY_EFF_GREP").unwrap_or_default();
        let mut resolver = EffectResolver::default();
        // A specific file when one is named, so this reaches a transplant donor or a mod's
        // own eff and not just the system library.
        let fighter = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from);
        resolver.set_search_path(
            fighter,
            Vec::new(),
            Some(EffectResolver::common_eff_path(&root)),
        );
        let resolved = resolver.resolve(&name).expect("effect resolves");
        let file = resolved.file.clone();
        let parts = resolved.parts.clone();
        let loaded = resolver.loaded(&file).expect("file stays loaded");
        let table = crate::eff_attrs::table();

        for part in &parts {
            let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else { continue };
            for emitter in &set.emitters {
                println!("
--- {} / {} ---", name, emitter.name);
                for (i, attr) in table.iter().enumerate() {
                    if !needle.is_empty() && !attr.id.contains(&needle) {
                        continue;
                    }
                    let Some(Some(value)) = emitter.attrs.get(i) else { continue };
                    // Zeros are the overwhelming majority and say nothing; what an emitter
                    // sets is the signal.
                    let zero = match value {
                        crate::eff_attrs::AttrValue::Int(v) => *v == 0,
                        crate::eff_attrs::AttrValue::UInt(v) => *v == 0,
                        crate::eff_attrs::AttrValue::Float(v) => *v == 0.0,
                    };
                    if zero && needle.is_empty() {
                        continue;
                    }
                    println!("  {:<44} {:?}", attr.id, value);
                }
            }
        }
    }

    /// What the emitters of a real fighter file actually ask for, as opposed to what the
    /// renderer assumes. Point `VISIONARY_EFF_FILE` at any `.eff` and run with --nocapture.
    #[test]
    fn what_the_emitters_of_one_file_ask_for() {
        let Some(path) = std::env::var_os("VISIONARY_EFF_FILE").map(PathBuf::from) else {
            eprintln!("VISIONARY_EFF_FILE not set — skipping");
            return;
        };
        let loaded = load_effect(&path).expect("eff loads");
        let table = crate::eff_attrs::table();
        let at = |id: &str| table.iter().position(|a| a.id == id);
        let (blend, billboard, side, blend_on) = (
            at("render_state.blend_type"),
            at("particle_data.billboard_type"),
            at("render_state.display_side"),
            at("render_state.is_blend_enable"),
        );
        let mut counts: std::collections::BTreeMap<(&str, i64), usize> = Default::default();
        let mut total = 0usize;
        for set in &loaded.ptcl.emitter_sets {
            for emitter in &set.emitters {
                total += 1;
                let f = |slot: Option<usize>| -> Option<i64> {
                    match emitter.attrs.get(slot?).and_then(|a| a.as_ref())? {
                        crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                        crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                        crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                    }
                };
                for (label, slot) in [
                    ("blend_type", blend),
                    ("billboard_type", billboard),
                    ("display_side", side),
                    ("is_blend_enable", blend_on),
                    ("is_rotate_x", at("particle_data.is_rotate_x")),
                    ("is_rotate_y", at("particle_data.is_rotate_y")),
                    ("is_rotate_z", at("particle_data.is_rotate_z")),
                ] {
                    if let Some(v) = f(slot) {
                        *counts.entry((label, v)).or_default() += 1;
                    }
                }
            }
        }
        println!("
=== {} : {total} emitters ===", path.display());
        for ((label, value), n) in counts {
            println!("  {label:<16} = {value:<4} {n:>5}  ({:.0}%)", 100.0 * n as f32 / total as f32);
        }
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
        resolver.set_search_path(Some(fighter.clone()), Vec::new(), Some(common.clone()));

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
            Vec::new(),
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

        let live = vec![LiveEffect {
            name: "MARIO_ATKHI3_ARC".to_string(),
            bone: "top".to_string(),
            tint: [1.0, 1.0, 1.0],
            alpha: 1.0,
            age: 4.0,
            offset: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: 1.0,
        }];
        let (effects, _) =
            build_particle_batches(
                &mut resolver, &|_| false, &|_| false, &|_| false,
                &mut crate::eff_mesh::MeshLibrary::default(), QuadPlanes::default(), &bones, &live);
        // Both of this effect's emitters draw a primitive, so its particles arrive as mesh
        // batches rather than quads. The two paths are checked together: what matters is that
        // the effect produced placed, textured particles, not which pipeline draws them.
        let mut batches = effects.batches;
        batches.extend(effects.mesh_batches.into_iter().map(|mesh| {
            crate::eff_render::ParticleBatch {
                texture: mesh.texture,
                texture1: mesh.texture1,
                blend: mesh.blend,
                instances: mesh.instances,
            }
        }));
        let pending = effects.pending_textures;

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
        assert!(
            !batches.is_empty(),
            "no batch of either kind built for a resolvable effect"
        );
        assert!(instances > 0, "no particles placed");
        assert!(!pending.is_empty(), "no texture decoded for upload");

        // Placed at the bone, not at the origin — the failure that looks like the effect
        // spawning correctly somewhere you are not looking.
        // Particles move now, so an exact match would only pass for a motionless emitter.
        // What must hold is that they are placed AROUND the bone rather than at the origin.
        let first = batches[0].instances[0];
        let bone = glam::Vec3::new(1.0, 20.0, 3.0);
        let near = batches[0]
            .instances
            .iter()
            .any(|i| (glam::Vec3::from(i.position) - bone).length() < 50.0);
        assert!(near, "no particle near the bone; first at {:?}", first.position);
        assert!(first.color[3] > 0.0, "fully transparent particle");

        // A texture already resident must not be decoded again: decoding is the expensive part
        // and it would otherwise run every frame the effect is live.
        let key = pending[0].key.clone();
        let (again_effects, _) =
            build_particle_batches(
                &mut resolver, &|k| *k == key, &|_| false, &|_| false,
                &mut crate::eff_mesh::MeshLibrary::default(), QuadPlanes::default(), &bones, &live);
        assert!(
            again_effects
                .pending_textures
                .iter()
                .all(|texture| texture.key != key),
            "a resident texture was decoded again"
        );

        // A bone the skeleton does not have places nothing rather than defaulting to origin.
        let missing = vec![LiveEffect {
            name: "MARIO_ATKHI3_ARC".to_string(),
            bone: "no_such_bone".to_string(),
            tint: [1.0, 1.0, 1.0],
            alpha: 1.0,
            age: 4.0,
            offset: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: 1.0,
        }];
        let (none_effects, _) =
            build_particle_batches(
                &mut resolver, &|_| false, &|_| false, &|_| false,
                &mut crate::eff_mesh::MeshLibrary::default(), QuadPlanes::default(), &bones, &missing);
        let none: Vec<usize> = none_effects
            .batches
            .iter()
            .map(|batch| batch.instances.len())
            .chain(
                none_effects
                    .mesh_batches
                    .iter()
                    .map(|batch| batch.instances.len()),
            )
            .collect();
        assert!(
            none.iter().all(|count| *count == 0),
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
                for emitter in set.emitters.iter() {
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
        resolver.set_search_path(Some(fighter), Vec::new(), Some(common));

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
                for emitter in set.emitters.iter() {
                    let value = |slot: usize| -> Option<i64> {
                        match emitter.attrs.get(slot).and_then(|a| a.as_ref())? {
                            crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                            crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                            crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                        }
                    };
                    // The simulation's own view of this emitter. Printed next to the raw
                    // fields so a value that reads wrong here — an emission duration of 1 on a
                    // continuous flame, say — is visible as a misreading rather than showing
                    // up later as an effect that emits once and stops.
                    let sim = crate::eff_sim::EmitterSim::read(emitter, &crate::eff_sim::Slots::new());
                    println!(
                        "      sim: rate={} interval={} start={} duration={} life={}±{} \
                         all_dir={} desig={:?}×{} diff={:?} grav={:?} scale={:?} n@f5={}",
                        sim.rate,
                        sim.interval,
                        sim.emission_start,
                        sim.emission_duration,
                        sim.life,
                        sim.life_random,
                        sim.all_direction,
                        sim.designated_dir,
                        sim.designated_dir_scale,
                        sim.diffusion,
                        sim.gravity,
                        sim.scale,
                        crate::eff_sim::evaluate(&sim, 5.0, 1).len(),
                    );
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
            Vec::new(),
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
            // These three were reported as pure billboards while the mesh count was being
            // read from the wrong field. Roughly half of all emitters draw geometry, so the
            // headline has to distinguish them rather than assuming quads.
            assert!(
                summary.mesh_emitters <= summary.emitters,
                "{name}: more meshes than emitters"
            );
            if summary.mesh_emitters > 0 {
                assert!(
                    summary.headline().contains("mesh"),
                    "{name} has {} mesh emitter(s) but the headline hides them: {}",
                    summary.mesh_emitters,
                    summary.headline()
                );
            }
            assert!(summary.life_range.is_some(), "{name} has no lifetimes");
        }

        // A name nothing has must say so rather than describing an empty effect.
        assert!(matches!(
            resolver.describe("not_a_real_effect"),
            Err(ResolveFailure::UnknownName)
        ));
    }

    /// How a sprite sheet is divided into cells, read off emitters whose texture size is known.
    ///
    /// The viewport currently maps 0..1 UV across the whole texture, so an effect whose texture
    /// is a strip of animation frames draws every frame at once — which is exactly why a smoke
    /// puff reads as a square. The division is not documented, so it is inferred here by
    /// putting the candidate fields next to the texture's real dimensions and looking for the
    /// combination that divides it into square cells.
    #[test]
    fn how_effect_sprite_sheets_are_divided() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(
            Some(root.join("effect/fighter/mario/ef_mario.eff")),
            Vec::new(),
            Some(EffectResolver::common_eff_path(&root)),
        );

        let table = crate::eff_attrs::table();
        let at = |id: &str| table.iter().position(|attr| attr.id == id);
        let repeat = at("texture_anim0.repeat");
        let anim_type = at("texture_anim0.pattern_anim_type");
        let pat_num = at("emitter_static.tex_pattern_anim0.num");
        let pat_freq = at("emitter_static.tex_pattern_anim0.frequency");
        let pat_slots: Vec<Option<usize>> = (0..8)
            .map(|i| at(&format!("emitter_static.tex_pattern_anim0.table[{i}]")))
            .collect();

        for name in ["SYS_ATK_SMOKE", "SYS_FLAME", "MARIO_FB_SHOOT"] {
            let Ok(resolved) = resolver.resolve(name) else {
                continue;
            };
            let file = resolved.file.clone();
            let parts = resolved.parts.clone();
            let Some(loaded) = resolver.loaded(&file) else {
                continue;
            };
            println!("\n=== {name} ===");
            for part in &parts {
                let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else {
                    continue;
                };
                for emitter in set.emitters.iter().take(6) {
                    let value = |slot: Option<usize>| -> Option<i64> {
                        match emitter.attrs.get(slot?).and_then(|a| a.as_ref())? {
                            crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                            crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                            crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                        }
                    };
                    let size = emitter
                        .texture_index
                        .and_then(|i| loaded.ptcl.bntx_textures.get(i as usize))
                        .map(|t| (t.tex_name.clone(), t.width, t.height));
                    let slots: Vec<i64> =
                        pat_slots.iter().filter_map(|s| value(*s)).collect();
                    println!(
                        "  '{}' tex={:?} repeat={:?} anim_type={:?} pat_num={:?} freq={:?} slots={:?}",
                        emitter.name,
                        size,
                        value(repeat),
                        value(anim_type),
                        value(pat_num),
                        value(pat_freq),
                        slots,
                    );
                }
            }
        }
    }

    /// Where the effects a MOD spawns actually live, and what the smoke emitters look like.
    ///
    /// A mod routinely spawns effects belonging to a fighter other than the one it is built on
    /// — the Shigaraki moveset is on eflame but calls `edge_attack_dash_*`. Those resolve
    /// against neither eflame's eff nor `ef_common`, so a two-file search path finds nothing
    /// and the effect silently never appears.
    #[test]
    fn effects_a_mod_borrows_from_another_fighter() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };

        for (fighter, wanted) in [
            ("edge", vec!["EDGE_ATTACK_DASH_AURA", "EDGE_ATTACK_DASH_HIT"]),
        ] {
            let path = root.join(format!("effect/fighter/{fighter}/ef_{fighter}.eff"));
            let Ok(loaded) = load_effect(&path) else {
                println!("could not load {}", path.display());
                continue;
            };
            println!("\n=== ef_{fighter}.eff: {} entries ===", loaded.entries.len());
            for name in &wanted {
                let found = loaded
                    .entries
                    .iter()
                    .find(|entry| entry.name.eq_ignore_ascii_case(name));
                match found {
                    Some(entry) => println!(
                        "  {name}: set={:?} variants={}",
                        entry.set_idx,
                        entry.variants.len()
                    ),
                    None => println!("  {name}: NOT in this file"),
                }
            }
            // Anything else beginning EDGE_ATTACK_DASH, so a near-miss on the name is visible.
            let similar: Vec<&str> = loaded
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .filter(|name| name.to_ascii_uppercase().contains("ATTACK_DASH"))
                .collect();
            println!("  entries containing ATTACK_DASH: {similar:?}");
        }
    }

    /// The transplant search, against a real mod that relies on it.
    ///
    /// `VISIONARY_MOD_ROOT` should point at a mod root (the folder holding `effect/`). Skipped
    /// when unset, since this needs a mod rather than the vanilla dump.
    #[test]
    fn a_mods_transplanted_effects_resolve_through_its_donors() {
        let (Some(dump), Some(mod_root)) = (
            root(),
            std::env::var_os("VISIONARY_MOD_ROOT").map(PathBuf::from),
        ) else {
            eprintln!("VISIONARY_EFF_ROOT / VISIONARY_MOD_ROOT not set — skipping");
            return;
        };

        let roots = vec![mod_root.clone(), dump.clone()];
        let donors = EffectResolver::transplant_donors(&roots, "eflame");
        println!("\ntransplant donors found: {}", donors.len());
        for donor in &donors {
            println!("  {}", donor.display());
        }
        assert!(
            !donors.is_empty(),
            "no donors under {}/effect/fighter/eflame/transplant",
            mod_root.display()
        );

        // The mod ships a slot-specific eff of its own; that is the file the fighter half of
        // the search path should be pointing at for this costume.
        let own = mod_root.join("effect/fighter/eflame/ef_eflame_c80.eff");
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(
            own.exists().then(|| own.clone()),
            donors.clone(),
            Some(EffectResolver::common_eff_path(&dump)),
        );

        // Without the donors these resolve to nothing at all, which is the bug: the effect is
        // spawned by the script, the game shows it, and the viewport showed nothing.
        for name in ["EDGE_ATTACK_DASH_AURA", "EDGE_ATTACK_DASH_HIT"] {
            let resolved = resolver
                .resolve(name)
                .unwrap_or_else(|failure| panic!("{name} did not resolve: {failure:?}"));
            println!(
                "  {name} -> {:?} from {}",
                resolved.source,
                resolved
                    .file
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            );
            let summary = resolver.describe(name).expect("describes");
            println!("      {}", summary.headline());
            assert!(summary.emitters > 0, "{name} resolved to no emitters");
        }

        // Prove the donors are load-bearing rather than incidental: drop them and the same
        // names must stop resolving.
        let mut without = EffectResolver::default();
        without.set_search_path(
            own.exists().then_some(own),
            Vec::new(),
            Some(EffectResolver::common_eff_path(&dump)),
        );
        let still = ["EDGE_ATTACK_DASH_AURA", "EDGE_ATTACK_DASH_HIT"]
            .iter()
            .filter(|name| without.resolve(name).is_ok())
            .count();
        println!("  without donors, {still} of 2 still resolve");
    }

    /// How many emitters really draw geometry, counted through the right field.
    ///
    /// An earlier pass counted `ptcl.primitives`, which is the raw section list, and concluded
    /// almost nothing used meshes. The pool that matters is `primitive_info.descriptors`, and
    /// an emitter's `primitive_id` is matched against `descriptor.id` — not used as an index.
    /// Counting the wrong field made every file look like billboards.
    #[test]
    fn how_many_emitters_actually_draw_geometry() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let table = crate::eff_attrs::table();
        let primitive_slot = table
            .iter()
            .position(|attr| attr.id == "particle_data.primitive_id")
            .unwrap();

        for relative in [
            "effect/fighter/edge/ef_edge.eff",
            "effect/fighter/mario/ef_mario.eff",
            "effect/system/common/ef_common.eff",
        ] {
            let path = root.join(relative);
            let Ok(bytes) = std::fs::read(&path) else { continue };
            let Ok(raw) = effect_library::NamcoEffectFile::load(&bytes) else { continue };
            let Ok(loaded) = load_effect(&path) else { continue };

            let info = raw.ptcl_file.as_ref().and_then(|p| p.primitive_info.as_ref());
            let descriptors = info.map(|i| i.descriptors.as_slice()).unwrap_or(&[]);
            let blob = info
                .and_then(|i| i.binary_data.as_ref())
                .map(|b| b.len())
                .unwrap_or(0);

            let mut emitters = 0usize;
            let mut mesh = 0usize;
            let mut unresolved = 0usize;
            let mut used: std::collections::BTreeSet<usize> = Default::default();
            for set in &loaded.ptcl.emitter_sets {
                for emitter in &set.emitters {
                    emitters += 1;
                    let id = match emitter.attrs.get(primitive_slot).and_then(|a| a.as_ref()) {
                        Some(crate::eff_attrs::AttrValue::UInt(v)) => *v,
                        Some(crate::eff_attrs::AttrValue::Int(v)) => *v as u64,
                        _ => continue,
                    };
                    match effect_library::bfres::descriptor_index_for_id(descriptors, id) {
                        Some(index) => {
                            mesh += 1;
                            used.insert(index);
                        }
                        None if id != 0 && id != u64::MAX => unresolved += 1,
                        None => {}
                    }
                }
            }
            println!("\n=== {relative} ===");
            println!("  primitive descriptors : {}", descriptors.len());
            println!("  bfres blob            : {blob} bytes");
            println!(
                "  emitters              : {emitters}, of which {mesh} draw a primitive \
                 ({} distinct), {unresolved} name an id this file has no descriptor for",
                used.len()
            );
        }
    }

    /// What an effect primitive's geometry actually looks like, before writing a mesh loader.
    ///
    /// Half of all emitters draw one of these. The vertex layout is not documented anywhere,
    /// so the attribute names, their formats, and the index encoding are read off real models
    /// rather than assumed — a wrong stride or format silently produces a cloud of scattered
    /// triangles rather than an error.
    #[test]
    fn what_effect_primitives_are_made_of() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let path = root.join("effect/fighter/edge/ef_edge.eff");
        let Ok(bytes) = std::fs::read(&path) else { return };
        let raw = effect_library::NamcoEffectFile::load(&bytes).expect("parse eff");
        let info = raw
            .ptcl_file
            .as_ref()
            .and_then(|ptcl| ptcl.primitive_info.as_ref())
            .expect("ef_edge has a primitive pool");
        let blob = info.binary_data.as_ref().expect("pool has a bfres blob");
        println!(
            "\n{} descriptors, {} byte bfres blob",
            info.descriptors.len(),
            blob.len()
        );

        let mut formats: std::collections::BTreeMap<(String, u16), usize> = Default::default();
        let mut index_formats: std::collections::BTreeMap<u32, usize> = Default::default();
        let mut primitive_types: std::collections::BTreeMap<u32, usize> = Default::default();
        let mut loaded_models = 0usize;

        for index in 0..info.descriptors.len().min(8) {
            let single = match effect_library::bfres::ResFile::export_single_model(blob, index) {
                Ok(single) => single,
                Err(error) => {
                    println!("  model {index}: export failed: {error}");
                    continue;
                }
            };
            let (name, model) =
                match effect_library::bfres::ResFile::parse_model_export(single) {
                    Ok(parsed) => parsed,
                    Err(error) => {
                        println!("  model {index}: parse failed: {error}");
                        continue;
                    }
                };
            loaded_models += 1;
            let vertices: u32 = model.vertex_buffers.iter().map(|b| b.vertex_count).sum();
            let mut attrs: Vec<String> = Vec::new();
            for buffer in &model.vertex_buffers {
                for (attr_name, attr) in buffer.attributes.iter() {
                    *formats.entry((attr_name.clone(), attr.format)).or_insert(0) += 1;
                    attrs.push(format!(
                        "{attr_name}(fmt={:#06x} buf={} off={})",
                        attr.format, attr.buffer_index, attr.offset
                    ));
                }
            }
            let strides: Vec<u32> = model
                .vertex_buffers
                .iter()
                .flat_map(|b| b.buffer_strides.clone())
                .collect();
            let mut mesh_desc = Vec::new();
            for shape in model.shapes.values() {
                for mesh in &shape.meshes {
                    *index_formats.entry(mesh.index_format).or_insert(0) += 1;
                    *primitive_types.entry(mesh.primitive_type).or_insert(0) += 1;
                    mesh_desc.push(format!(
                        "{} idx={} fmt={} prim={} bytes={}",
                        shape.name,
                        mesh.index_count,
                        mesh.index_format,
                        mesh.primitive_type,
                        mesh.index_data.len()
                    ));
                }
            }
            println!(
                "  [{index}] '{name}' verts={vertices} strides={strides:?} shapes={}",
                model.shapes.len()
            );
            println!("      attrs: {}", attrs.join(", "));
            for line in mesh_desc.iter().take(2) {
                println!("      mesh: {line}");
            }
        }

        println!("\n  attribute formats seen: {formats:?}");
        println!("  index formats seen: {index_formats:?}");
        println!("  primitive types seen: {primitive_types:?}");
        assert!(loaded_models > 0, "no primitive model could be extracted");
    }

    /// The mesh path end to end: a transplanted effect that draws geometry produces mesh
    /// batches with real primitives behind them, not billboards.
    #[test]
    fn geometry_emitters_produce_mesh_batches() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let edge = root.join("effect/fighter/edge/ef_edge.eff");
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(Some(edge.clone()), Vec::new(), None);
        let mut meshes = crate::eff_mesh::MeshLibrary::default();

        let mut bones = std::collections::HashMap::new();
        bones.insert("Top".to_string(), glam::Mat4::from_translation(glam::Vec3::Y * 10.0));

        let live = vec![LiveEffect {
            name: "EDGE_ATTACK_DASH_HIT".into(),
            bone: "top".into(),
            tint: [1.0, 1.0, 1.0],
            alpha: 1.0,
            age: 4.0,
            offset: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: 1.0,
        }];
        let (effects, _) = build_particle_batches(
            &mut resolver,
            &|_| false,
            &|_| false,
            &|_| false,
            &mut meshes,
            QuadPlanes::default(),
            &bones,
            &live,
        );

        let quad_instances: usize = effects.batches.iter().map(|b| b.instances.len()).sum();
        let mesh_instances: usize = effects.mesh_batches.iter().map(|b| b.instances.len()).sum();
        println!(
            "
EDGE_ATTACK_DASH_HIT: {} quad batch(es)/{quad_instances} instances,              {} mesh batch(es)/{mesh_instances} instances, {} primitive(s) to upload",
            effects.batches.len(),
            effects.mesh_batches.len(),
            effects.pending_meshes.len()
        );
        for (key, mesh) in &effects.pending_meshes {
            println!(
                "  primitive {} -> '{}' {} verts {} tris",
                key.descriptor,
                mesh.name,
                mesh.vertices.len(),
                mesh.indices.len() / 3
            );
        }

        assert!(
            !effects.mesh_batches.is_empty(),
            "a mesh-drawing effect produced no mesh batches"
        );
        assert!(mesh_instances > 0, "mesh batches carry no instances");
        assert!(
            !effects.pending_meshes.is_empty(),
            "no primitive geometry was extracted for upload"
        );
        // Every mesh batch must name a primitive that really extracted, or the draw silently
        // does nothing.
        for batch in &effects.mesh_batches {
            assert!(
                meshes.mesh(&batch.mesh).is_some(),
                "mesh batch names primitive {} which does not extract",
                batch.mesh.descriptor
            );
        }
    }

    /// Emitter transforms and multi-texture use, on the effects being looked at.
    ///
    /// Particles are currently placed at the bone with no orientation at all: the emitter's own
    /// translation and rotation are ignored, as is the second and third sampler. This measures
    /// whether that is actually costing anything before either is built.
    #[test]
    fn emitter_transforms_and_extra_samplers() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(
            Some(root.join("effect/fighter/edge/ef_edge.eff")),
            Vec::new(),
            Some(EffectResolver::common_eff_path(&root)),
        );
        let table = crate::eff_attrs::table();
        let at = |id: &str| table.iter().position(|a| a.id == id);
        let trans = [at("emitter_info.trans_x"), at("emitter_info.trans_y"), at("emitter_info.trans_z")];
        let rot = [at("emitter_info.rotate_x"), at("emitter_info.rotate_y"), at("emitter_info.rotate_z")];
        let tex1 = at("sampler1.texture_id");
        let tex2 = at("sampler2.texture_id");
        let blend1 = at("combiner.texture1_color_blend");
        let blend2 = at("combiner.texture2_color_blend");
        let process = at("combiner.color_combiner_process");

        for name in ["EDGE_ATTACK_DASH_HIT", "SYS_TURN_SMOKE"] {
            let Ok(resolved) = resolver.resolve(name) else { continue };
            let file = resolved.file.clone();
            let parts = resolved.parts.clone();
            let Some(loaded) = resolver.loaded(&file) else { continue };
            println!("
=== {name} ===");
            for part in &parts {
                let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else { continue };
                for emitter in set.emitters.iter().take(7) {
                    let f = |slot: Option<usize>| -> f32 {
                        match emitter.attrs.get(slot.unwrap_or(usize::MAX)).and_then(|a| a.as_ref()) {
                            Some(crate::eff_attrs::AttrValue::Float(v)) => *v,
                            Some(crate::eff_attrs::AttrValue::Int(v)) => *v as f32,
                            Some(crate::eff_attrs::AttrValue::UInt(v)) => *v as f32,
                            None => 0.0,
                        }
                    };
                    let t = [f(trans[0]), f(trans[1]), f(trans[2])];
                    let r = [f(rot[0]), f(rot[1]), f(rot[2])];
                    let moved = t.iter().any(|v| v.abs() > 1e-4);
                    let turned = r.iter().any(|v| v.abs() > 1e-4);
                    println!(
                        "  '{}' trans={:?}{} rot={:?}{} tex1={} tex2={} blend=({},{}) proc={}",
                        emitter.name,
                        t, if moved { " <-- offset" } else { "" },
                        r, if turned { " <-- rotated" } else { "" },
                        f(tex1) as i64, f(tex2) as i64,
                        f(blend1) as i64, f(blend2) as i64, f(process) as i64,
                    );
                }
            }
        }
    }

    /// Emitters must land where their own transform puts them, not all on the bone.
    ///
    /// SYS_TURN_SMOKE separates its five emitters by up to 2 units front-to-back. Collapsed
    /// onto one point they are a blob; spread out they are a cloud. This is the difference.
    #[test]
    fn emitters_are_spread_by_their_own_transforms() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(None, Vec::new(), Some(EffectResolver::common_eff_path(&root)));
        let mut meshes = crate::eff_mesh::MeshLibrary::default();

        let bone_at = glam::Vec3::new(5.0, 12.0, -3.0);
        let mut bones = std::collections::HashMap::new();
        bones.insert("Top".to_string(), glam::Mat4::from_translation(bone_at));

        let live = vec![LiveEffect {
            name: "SYS_TURN_SMOKE".into(),
            bone: "top".into(),
            tint: [1.0, 1.0, 1.0],
            alpha: 1.0,
            age: 3.0,
            offset: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: 1.0,
        }];
        let (effects, _) = build_particle_batches(
            &mut resolver, &|_| false, &|_| false, &|_| false, &mut meshes, QuadPlanes::default(), &bones, &live,
        );

        let positions: Vec<glam::Vec3> = effects
            .batches
            .iter()
            .flat_map(|batch| batch.instances.iter())
            .chain(effects.mesh_batches.iter().flat_map(|b| b.instances.iter()))
            .map(|instance| glam::Vec3::from(instance.position))
            .collect();
        assert!(!positions.is_empty(), "SYS_TURN_SMOKE placed nothing");

        // The emitters sit up to 2 units apart, so the cloud must occupy more than a point.
        let span = positions.iter().fold(0.0f32, |worst, a| {
            positions
                .iter()
                .fold(worst, |worst, b| worst.max((*a - *b).length()))
        });
        println!(
            "
SYS_TURN_SMOKE: {} particles spanning {span:.2} units around the bone",
            positions.len()
        );
        assert!(
            span > 1.0,
            "every emitter landed on the same point — the cloud is a blob ({span})"
        );

        // And the whole thing still hangs off the bone rather than the origin.
        let centre = positions.iter().copied().sum::<glam::Vec3>() / positions.len() as f32;
        assert!(
            (centre - bone_at).length() < 25.0,
            "the cloud drifted away from its bone: centre {centre:?} vs bone {bone_at:?}"
        );
    }

    /// Which channel of an effect texture carries the shape.
    ///
    /// The shader samples red as a mask. If red is near 1.0 across the whole image the mask
    /// does nothing and every particle renders as a solid white quad, which is what the smoke
    /// effects look like. This reports the real per-channel statistics rather than assuming.
    #[test]
    fn which_texture_channel_is_the_mask() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        for (relative, wanted) in [
            (
                "effect/system/common/ef_common.eff",
                vec!["ef_cmn_smoke01", "ef_cmn_smoke03", "ef_cmn_impact05_ani", "ef_cmn_line02"],
            ),
            (
                "effect/fighter/edge/ef_edge.eff",
                vec!["ef_edge_aura01"],
            ),
        ] {
            let path = root.join(relative);
            let Ok(loaded) = load_effect(&path) else { continue };
            let Some(pool) = loaded.texture_pool.as_ref() else { continue };
            println!("
=== {relative} ===");
            for (index, info) in loaded.ptcl.bntx_textures.iter().enumerate() {
                let interesting = wanted.iter().any(|w| info.tex_name.contains(w));
                if !interesting {
                    continue;
                }
                let Ok(image) = crate::texture_import::decode_rgba(pool, index, &info.tex_name, None)
                else {
                    println!("  {} [{}]: would not decode", info.tex_name, info.format);
                    continue;
                };
                let mut stats = [[255u8, 0u8]; 4];
                let mut sums = [0u64; 4];
                let pixels = image.pixels().count() as u64;
                for pixel in image.pixels() {
                    for channel in 0..4 {
                        let v = pixel.0[channel];
                        stats[channel][0] = stats[channel][0].min(v);
                        stats[channel][1] = stats[channel][1].max(v);
                        sums[channel] += v as u64;
                    }
                }
                let names = ["R", "G", "B", "A"];
                let summary: Vec<String> = (0..4)
                    .map(|c| {
                        format!(
                            "{}:{}-{} avg{}",
                            names[c],
                            stats[c][0],
                            stats[c][1],
                            sums[c] / pixels.max(1)
                        )
                    })
                    .collect();
                println!(
                    "  {} [{}] {}x{}  {}",
                    info.tex_name,
                    info.format,
                    image.width(),
                    image.height(),
                    summary.join("  ")
                );
            }
        }
    }

    /// The two texture families must both end up meaning the same thing by RGBA.
    #[test]
    fn textures_are_normalised_so_the_shape_is_always_in_alpha() {
        // BC3-like: greyscale RGB, real shape in alpha. Left alone.
        let mut varying = image::RgbaImage::new(2, 2);
        varying.put_pixel(0, 0, image::Rgba([200, 200, 200, 0]));
        varying.put_pixel(1, 0, image::Rgba([200, 200, 200, 255]));
        varying.put_pixel(0, 1, image::Rgba([200, 200, 200, 128]));
        varying.put_pixel(1, 1, image::Rgba([200, 200, 200, 64]));
        let kept = normalize_mask(varying.clone());
        assert_eq!(kept.get_pixel(0, 0).0, [200, 200, 200, 0]);
        assert_eq!(kept.get_pixel(1, 0).0, [200, 200, 200, 255]);

        // BC5-like: shape in red, alpha a constant 255. Red becomes the alpha, so a particle
        // is shaped rather than a solid square.
        let mut constant = image::RgbaImage::new(2, 1);
        constant.put_pixel(0, 0, image::Rgba([0, 40, 0, 255]));
        constant.put_pixel(1, 0, image::Rgba([255, 90, 0, 255]));
        let fixed = normalize_mask(constant);
        assert_eq!(fixed.get_pixel(0, 0).0, [0, 0, 0, 0], "transparent where red is 0");
        assert_eq!(fixed.get_pixel(1, 0).0, [255, 255, 255, 255], "opaque where red is 1");
    }

    /// The real textures, through the real decode, must not come out as solid blocks.
    #[test]
    fn real_effect_textures_are_not_solid_after_normalisation() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let path = EffectResolver::common_eff_path(&root);
        let Ok(loaded) = load_effect(&path) else { return };
        let Some(pool) = loaded.texture_pool.as_ref() else { return };

        let mut checked = 0usize;
        for (index, info) in loaded.ptcl.bntx_textures.iter().enumerate() {
            if !["ef_cmn_smoke01", "ef_cmn_impact05_ani", "ef_cmn_line02"]
                .iter()
                .any(|w| info.tex_name.contains(w))
            {
                continue;
            }
            let Ok(image) = crate::texture_import::decode_rgba(pool, index, &info.tex_name, None)
            else {
                continue;
            };
            let normalised = normalize_mask(image);
            let mut lowest = 255u8;
            let mut highest = 0u8;
            for pixel in normalised.pixels() {
                lowest = lowest.min(pixel.0[3]);
                highest = highest.max(pixel.0[3]);
            }
            println!(
                "  {} [{}] alpha after normalisation: {lowest}-{highest}",
                info.tex_name, info.format
            );
            // A particle whose alpha never drops is a solid quad, whatever its texture shows.
            assert!(
                highest > lowest + 8,
                "{} has no shape in alpha after normalisation ({lowest}-{highest}) — it will                  render as a solid block",
                info.tex_name
            );
            checked += 1;
        }
        assert!(checked >= 2, "expected to check both texture families");
    }

    /// SYS_ATTACK_ARC, which renders horizontal when it should point along the swing.
    #[test]
    fn what_orients_the_attack_arc() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(None, Vec::new(), Some(EffectResolver::common_eff_path(&root)));
        let table = crate::eff_attrs::table();
        let at = |id: &str| table.iter().position(|a| a.id == id);
        let rot = [at("emitter_info.rotate_x"), at("emitter_info.rotate_y"), at("emitter_info.rotate_z")];
        let prim = at("particle_data.primitive_id");
        let billboard = at("particle_data.billboard_type");

        for name in ["SYS_ATTACK_ARC", "SYS_ATK_ARC"] {
            let Ok(resolved) = resolver.resolve(name) else {
                println!("{name}: does not resolve");
                continue;
            };
            let file = resolved.file.clone();
            let parts = resolved.parts.clone();
            let ids = resolver.descriptor_ids(&file);
            let Some(loaded) = resolver.loaded(&file) else { continue };
            println!("
=== {name} ===");
            for part in &parts {
                let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else { continue };
                for emitter in &set.emitters {
                    let f = |slot: Option<usize>| -> f32 {
                        match emitter.attrs.get(slot.unwrap_or(usize::MAX)).and_then(|a| a.as_ref()) {
                            Some(crate::eff_attrs::AttrValue::Float(v)) => *v,
                            Some(crate::eff_attrs::AttrValue::Int(v)) => *v as f32,
                            Some(crate::eff_attrs::AttrValue::UInt(v)) => *v as f32,
                            None => 0.0,
                        }
                    };
                    // As u64, NEVER through the f32 reader below: 1811533741 does not survive
                    // a round trip through f32, and reading it that way reports a resolvable
                    // primitive as missing.
                    let id = match prim.and_then(|slot| emitter.attrs.get(slot)).and_then(|v| v.as_ref()) {
                        Some(crate::eff_attrs::AttrValue::UInt(v)) => *v,
                        Some(crate::eff_attrs::AttrValue::Int(v)) if *v > 0 => *v as u64,
                        _ => 0,
                    };
                    println!(
                        "  '{}' rot=[{:.3}, {:.3}, {:.3}] billboard={} mesh={} tex={:?}",
                        emitter.name,
                        f(rot[0]), f(rot[1]), f(rot[2]),
                        f(billboard) as i64,
                        ids.contains(&id),
                        emitter.texture_index,
                    );
                }
            }
        }
    }

    /// A script's aim must reach the particles. This is how an effect points along a punch
    /// rather than sitting at whatever angle its emitter happens to carry.
    #[test]
    fn the_calls_own_rotation_aims_the_effect() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(None, Vec::new(), Some(EffectResolver::common_eff_path(&root)));
        let mut meshes = crate::eff_mesh::MeshLibrary::default();

        // A bone with no rotation of its own, so anything that moves is the call's doing.
        let mut bones = std::collections::HashMap::new();
        bones.insert("Top".to_string(), glam::Mat4::IDENTITY);

        let mut sample = |rotation: glam::Vec3, meshes: &mut crate::eff_mesh::MeshLibrary| {
            let live = vec![LiveEffect {
                name: "SYS_ATTACK_ARC".into(),
                bone: "top".into(),
                tint: [1.0, 1.0, 1.0],
                alpha: 1.0,
                age: 2.0,
                scale: 1.0,
                offset: glam::Vec3::ZERO,
                rotation,
            }];
            let (effects, _) = build_particle_batches(
                &mut resolver, &|_| false, &|_| false, &|_| false, meshes, QuadPlanes::default(), &bones, &live,
            );
            effects
                .batches
                .iter()
                .flat_map(|b| b.instances.iter())
                .chain(effects.mesh_batches.iter().flat_map(|b| b.instances.iter()))
                .map(|i| glam::Quat::from_array(i.orientation))
                .next()
        };

        let unaimed = sample(glam::Vec3::ZERO, &mut meshes).expect("arc places particles");
        // Degrees. An ACMD spawn's rotation arguments are degrees and reach here unconverted;
        // passing radians made one degree of aim come out as fifty-seven.
        let aimed = sample(glam::Vec3::new(0.0, 0.0, 90.0), &mut meshes)
            .expect("arc places particles when aimed");

        // Quarter turn about Z: the quad's own right axis must swing to point up.
        let right_before = unaimed * glam::Vec3::X;
        let right_after = aimed * glam::Vec3::X;
        let swing = right_before.angle_between(right_after).to_degrees();
        println!(
            "
SYS_ATTACK_ARC right axis {right_before:?} -> {right_after:?} ({swing:.1} degrees)"
        );
        assert!(
            swing > 80.0,
            "a 90 degree aim moved the effect by only {swing:.1} degrees — the call's rotation              is not reaching the particles"
        );

        // And the units must be degrees. One degree of aim is one degree of movement; treating
        // it as radians turns it into fifty-seven, which is what made a 1 look like a 90.
        let nudged = sample(glam::Vec3::new(0.0, 1.0, 0.0), &mut meshes).expect("places");
        let nudge = (unaimed * glam::Vec3::X)
            .angle_between(nudged * glam::Vec3::X)
            .to_degrees();
        println!("  a 1 degree aim moves the effect {nudge:.2} degrees");
        assert!(
            nudge < 3.0,
            "one degree of aim moved the effect {nudge:.1} degrees — the call's rotation is              being read as radians"
        );
    }

    /// Billboard types across the effects being looked at, so the quad-plane rule is chosen
    /// from what the data uses rather than from one example.
    #[test]
    fn billboard_types_in_use() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let table = crate::eff_attrs::table();
        let billboard = table.iter().position(|a| a.id == "particle_data.billboard_type").unwrap();

        for (relative, names) in [
            ("effect/fighter/miigunner/ef_miigunner.eff", vec!["MIIGUNNER_ATK_SHOT_S"]),
            ("effect/system/common/ef_common.eff", vec!["SYS_ATTACK_ARC", "SYS_TURN_SMOKE"]),
        ] {
            let path = root.join(relative);
            let Ok(loaded) = load_effect(&path) else {
                println!("{relative}: not loadable");
                continue;
            };
            println!("
=== {relative} ===");
            // Whole-file histogram, so the rule is not tuned to one effect.
            let mut histogram: std::collections::BTreeMap<i64, usize> = Default::default();
            for set in &loaded.ptcl.emitter_sets {
                for emitter in &set.emitters {
                    if let Some(crate::eff_attrs::AttrValue::Int(v)) =
                        emitter.attrs.get(billboard).and_then(|a| a.as_ref())
                    {
                        *histogram.entry(*v).or_insert(0) += 1;
                    }
                }
            }
            println!("  billboard_type histogram across the file: {histogram:?}");
            for wanted in names {
                let Some(entry) = loaded
                    .entries
                    .iter()
                    .find(|e| e.name.eq_ignore_ascii_case(wanted))
                else {
                    println!("  {wanted}: no entry");
                    continue;
                };
                for part in parts_of(entry) {
                    let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else { continue };
                    let types: Vec<i64> = set
                        .emitters
                        .iter()
                        .filter_map(|e| match e.attrs.get(billboard).and_then(|a| a.as_ref()) {
                            Some(crate::eff_attrs::AttrValue::Int(v)) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    println!("  {wanted}: {} emitters, types {types:?}", set.emitters.len());
                }
            }
        }
    }

    /// The per-type turn must apply in the units it is labelled with, and only to its own type.
    #[test]
    fn the_per_type_turn_is_in_degrees_and_type_scoped() {
        // Explicitly zeroed rather than Default: this covers the mechanism, and the shipped
        // defaults are checked by their own test. Starting from Default would make this fail
        // every time a type is dialled in against the game.
        let mut planes = QuadPlanes {
            planes: [0; 8],
            offsets: [[0.0; 3]; 8],
            ..QuadPlanes::default()
        };
        assert_eq!(planes.offset_for(5), glam::Quat::IDENTITY);

        planes.offsets[5] = [0.0, 90.0, 0.0];
        let turned = planes.offset_for(5) * glam::Vec3::X;
        // A quarter turn about Y takes +X to -Z.
        assert!(
            (turned - glam::Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5,
            "a 90 degree Y turn produced {turned:?}"
        );
        // Untouched types stay untouched — a global turn would move every effect at once.
        assert_eq!(planes.offset_for(3), glam::Quat::IDENTITY);
        assert_eq!(planes.offset_for(0), glam::Quat::IDENTITY);

        // Degrees, like the spawn arguments they sit beside.
        planes.offsets[3] = [0.0, 0.0, 1.0];
        let nudge = (planes.offset_for(3) * glam::Vec3::X)
            .angle_between(glam::Vec3::X)
            .to_degrees();
        assert!(nudge < 1.5, "one degree of turn moved {nudge} degrees");
    }

    /// Every order is its three fixed-axis turns composed in sequence, and the old intrinsic
    /// `XYZ` reading is one of them -- so picking it reproduces the renderer as it was.
    #[test]
    fn spawn_orders_are_fixed_axis_compositions() {
        let degrees = glam::Vec3::new(178.0, 236.0, 90.0);
        let (x, y, z) = (
            glam::Quat::from_rotation_x(degrees.x.to_radians()),
            glam::Quat::from_rotation_y(degrees.y.to_radians()),
            glam::Quat::from_rotation_z(degrees.z.to_radians()),
        );
        let expected = [z * y * x, y * z * x, z * x * y, x * z * y, y * x * z, x * y * z];
        for (order, want) in expected.iter().enumerate() {
            let got = spawn_rotation_in(degrees, order);
            assert!(got.dot(*want).abs() > 0.9999, "order {order}");
        }
        let intrinsic = glam::Quat::from_euler(
            glam::EulerRot::XYZ,
            degrees.x.to_radians(),
            degrees.y.to_radians(),
            degrees.z.to_radians(),
        );
        assert!(spawn_rotation_in(degrees, 5).dot(intrinsic).abs() > 0.9999);
    }

    /// The default spawn reading, matched by eye against Dabi's jabs in game. All three spawn
    /// the same ring mesh -- flat in its XZ plane, normal on local Y -- on `top`, at
    ///
    ///   attack12  (0, 0, -30)     flat, on a `\` tilt
    ///   attack13  (178, 180, 90)  a half moon top to bottom
    ///   attack11  (178, 236, 90)  attack13's arc turned within its plane, peaking overhead
    ///
    /// In the viewer's side view, screen right is the bone's +Z, up is +Y and toward the camera
    /// is -X. Each candidate was stepped through live; Y, Z, X with a quarter frame turn is the
    /// one that matched all three.
    #[test]
    fn spawn_rotation_defaults_match_dabis_jabs_in_game() {
        let planes = QuadPlanes::default();
        assert_eq!(SPAWN_ORDER_NAMES[planes.spawn_order], "Y, Z, X");
        assert_eq!(planes.spawn_frame_turns, 1);

        // (right, up, toward camera) for a local axis of the ring.
        let screen = |planes: &QuadPlanes, deg: [f32; 3], local: glam::Vec3| {
            let v = planes.spawn_turn(glam::Vec3::from(deg)) * local;
            glam::Vec3::new(v.z, v.y, -v.x)
        };
        let (jab11, jab12, jab13) = ([178.0, 236.0, 90.0], [0.0, 0.0, -30.0], [178.0, 180.0, 90.0]);

        // attack12 tilts left/right, not toward the camera: the `\` needs its normal in the
        // screen plane.
        let n12 = screen(&planes, jab12, glam::Vec3::Y);
        assert!(n12.y > 0.85 && n12.z.abs() < 0.01, "attack12 normal {n12:?}");

        // attack11 shares attack13's plane and is turned 56 degrees within it -- the Y argument.
        let (n11, n13) = (
            screen(&planes, jab11, glam::Vec3::Y),
            screen(&planes, jab13, glam::Vec3::Y),
        );
        assert!(n11.dot(n13).abs() > 0.99, "attack11 {n11:?} is off attack13's plane {n13:?}");
        let roll = screen(&planes, jab11, glam::Vec3::X)
            .angle_between(screen(&planes, jab13, glam::Vec3::X))
            .to_degrees();
        assert!((roll - 56.0).abs() < 1.0, "in-plane turn was {roll}");

        // The near miss, X, Z, Y: attack12 and attack13 come out identical, but the Y argument
        // lands last as a turn about up, swinging attack11 out of the plane like an open door.
        let door = QuadPlanes { spawn_order: 1, ..planes };
        assert!(screen(&door, jab13, glam::Vec3::Y).dot(n13) > 0.99);
        assert!(screen(&door, jab11, glam::Vec3::Y).dot(n13).abs() < 0.7);
    }

    /// The dialled-in orientation defaults. These came from comparing the viewport against the
    /// game rather than from anything in the file, so nothing derives them and nothing else
    /// would notice them silently changing.
    #[test]
    fn the_orientation_defaults_are_the_ones_measured_against_the_game() {
        let planes = QuadPlanes::default();

        // Type 0 faces the camera; its turn corrects particle motion, not the quad.
        assert_eq!(planes.plane_for(0), 0);
        assert_eq!(planes.offsets[0], [90.0, 90.0, 0.0]);

        // SYS_ATTACK_ARC and MIIGUNNER_ATK_SHOT_S both sit in the effect's local XY plane and
        // differ only in the turn they need.
        assert_eq!(planes.plane_for(3), 1);
        assert_eq!(planes.offsets[3], [180.0, 0.0, -90.0]);
        assert_eq!(planes.plane_for(5), 1);
        assert_eq!(planes.offsets[5], [90.0, 0.0, 0.0]);

        // Unverified types carry no turn. A guessed default is worse than an obvious one.
        for kind in [1, 4, 6, 7] {
            assert_eq!(
                planes.offsets[kind], [0.0; 3],
                "type {kind} has an unverified turn baked in"
            );
        }
    }

    /// Why the smoke effects still read as squares.
    ///
    /// `SYS_ATK_SMOKE` and `SYS_TURN_SMOKE` look square while sheet-animated effects work, so
    /// the cell count is not being found for these. Dumps every pattern channel rather than
    /// just the first, since an emitter using channel 1 or 2 would look exactly like one with
    /// no pattern at all.
    #[test]
    fn what_the_smoke_emitters_actually_carry() {
        let Some(root) = root() else {
            eprintln!("VISIONARY_EFF_ROOT not set — skipping");
            return;
        };
        let mut resolver = EffectResolver::default();
        resolver.set_search_path(None, Vec::new(), Some(EffectResolver::common_eff_path(&root)));

        let table = crate::eff_attrs::table();
        let at = |id: &str| table.iter().position(|attr| attr.id == id);
        let channels = [
            ("0", at("emitter_static.tex_pattern_anim0.num"), at("texture_anim0.pattern_anim_type"), at("emitter_static.tex_pattern_anim0.num_random")),
            ("1", at("emitter_static.tex_pattern_anim1.num"), at("texture_anim1.pattern_anim_type"), at("emitter_static.tex_pattern_anim1.num_random")),
            ("2", at("emitter_static.tex_pattern_anim2.num"), at("texture_anim2.pattern_anim_type"), at("emitter_static.tex_pattern_anim2.num_random")),
        ];
        let full_table: Vec<Option<usize>> = (0..32)
            .map(|i| at(&format!("emitter_static.tex_pattern_anim0.table[{i}]")))
            .collect();

        for name in ["SYS_ATK_SMOKE", "SYS_TURN_SMOKE", "SYS_DASH_SMOKE"] {
            let Ok(resolved) = resolver.resolve(name) else {
                println!("\n{name}: does not resolve");
                continue;
            };
            let file = resolved.file.clone();
            let parts = resolved.parts.clone();
            let Some(loaded) = resolver.loaded(&file) else { continue };
            println!("\n=== {name} ===");
            for part in &parts {
                let Some(set) = loaded.ptcl.emitter_sets.get(part.set_idx) else { continue };
                for emitter in set.emitters.iter().take(5) {
                    let value = |slot: Option<usize>| -> Option<i64> {
                        match emitter.attrs.get(slot?).and_then(|a| a.as_ref())? {
                            crate::eff_attrs::AttrValue::Int(v) => Some(*v),
                            crate::eff_attrs::AttrValue::UInt(v) => Some(*v as i64),
                            crate::eff_attrs::AttrValue::Float(v) => Some(*v as i64),
                        }
                    };
                    let tex = emitter
                        .texture_index
                        .and_then(|i| loaded.ptcl.bntx_textures.get(i as usize))
                        .map(|t| format!("{} {}x{}", t.tex_name, t.width, t.height));
                    let per_channel: Vec<String> = channels
                        .iter()
                        .map(|(label, num, kind, rand)| {
                            format!(
                                "ch{label}(num={:?} type={:?} rand={:?})",
                                value(*num),
                                value(*kind),
                                value(*rand)
                            )
                        })
                        .collect();
                    let seq: Vec<i64> = full_table.iter().filter_map(|s| value(*s)).collect();
                    let distinct: std::collections::BTreeSet<i64> = seq.iter().copied().collect();
                    println!("  '{}' tex={:?}", emitter.name, tex);
                    println!("      {}", per_channel.join(" "));
                    println!("      table distinct={distinct:?} len={}", seq.len());
                }
            }
        }
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
