//! Particle simulation, evaluated closed-form at any frame.
//!
//! The editor scrubs. A conventional particle system steps state forward a frame at a time and
//! cannot answer "what does frame 30 look like" without having run frames 0..29 first, so
//! dragging the playhead backwards either shows the wrong thing or forces a replay from the
//! start of the move. Everything here is therefore a pure function of the particle's age:
//! position is `p0 + v0·t + ½g·t²`, and every random quantity comes from hashing the particle's
//! identity rather than from a running generator. Scrub anywhere, get the same answer, in the
//! same time.
//!
//! That constraint rules out anything path-dependent — collision, turbulence that integrates,
//! drag applied per step — and those are the parts this deliberately does not model. What it
//! does cover is what the overwhelming majority of Smash effects are made of: an emitter
//! producing particles at a rate for a duration, each living a fixed life, moving under an
//! initial velocity and gravity, changing size and colour across that life.

use crate::eff_attrs::AttrValue;
use crate::effects::{ColorKey, EmitterDef};

/// Deterministic per-particle randomness.
///
/// Not a sequence: a hash. The nth particle's jitter has to be the same value whether the
/// viewport arrived at this frame by playing forward, scrubbing backwards, or opening the move
/// cold, and a stateful generator gives three different answers.
fn hashed(seed: u64, salt: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // 24 bits is plenty for jitter and keeps the value exactly representable.
    ((x >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Symmetric jitter in −1..1.
fn hashed_signed(seed: u64, salt: u64) -> f32 {
    hashed(seed, salt) * 2.0 - 1.0
}

/// The emitter fields the simulation reads, pulled out of the attribute table once.
///
/// Resolved by attribute id rather than by index: the table is generated and its ordering is
/// not a stable interface, so an index captured here would silently start reading a different
/// field the next time a row is inserted.
#[derive(Debug, Clone)]
pub struct EmitterSim {
    pub rate: f32,
    pub rate_random: f32,
    pub interval: f32,
    pub emission_start: f32,
    pub emission_duration: f32,
    pub position_random: f32,
    pub gravity: glam::Vec3,
    pub life: f32,
    pub life_random: f32,
    pub all_direction: f32,
    pub designated_dir: glam::Vec3,
    pub designated_dir_scale: f32,
    pub diffusion: glam::Vec3,
    pub velocity_random: f32,
    pub scale: glam::Vec3,
    /// Particle size over its own life: (frame, xyz). This is the PARTICLE's scale, distinct
    /// from `scale` above which is the emitter's — reading the emitter's as the particle's
    /// gives every particle in the game the same size, because emitters are nearly all 1.0.
    pub scale_keys: Vec<(f32, glam::Vec3)>,
    pub billboard_type: i64,
    /// How the emitter's particles blend with what is behind them.
    ///
    /// 0 is normal alpha, 1 additive, 2 subtractive -- the format's own numbering, with 0 the
    /// default as a format's zero usually is. Across two real fighter files this splits about
    /// 73 / 25 / 3, so an assumption that everything is additive is wrong for roughly three
    /// quarters of every emitter in the game, and wrong in the direction that makes smoke and
    /// dust glow like fire.
    pub blend_type: i64,
    /// The particle's own roll about the quad's normal, and how it changes.
    ///
    /// `init` is where the particle starts, `init_rand` a symmetric spread on that, `add` the
    /// per-frame velocity and `add_rand` a spread on that. Radians, as the emitter's own
    /// rotation is.
    ///
    /// Reading these is what tells a spinning smoke puff apart from an attack arc. Both are
    /// one quad with a texture on it; the puff carries a full turn of `init_rand` so every
    /// one in the cloud sits differently, and the arc carries zeros because the crescent is
    /// painted at the angle it is meant to be seen at. Substituting a random roll for the
    /// data made the puffs look right and left the arc at a random angle every time -- and no
    /// orientation setting could correct it, because the error was per-particle.
    pub rotate_init: glam::Vec3,
    pub rotate_init_rand: glam::Vec3,
    pub rotate_add: glam::Vec3,
    pub rotate_add_rand: glam::Vec3,
    /// Which axes the emitter actually animates, from `particle_data.is_rotate_*`.
    ///
    /// Not decoration. Across ef_common Z is enabled on 49% of emitters, Y on 14% and X on
    /// 5%, so an implementation that assumes the roll is always about Z covers about half the
    /// game and silently drops the rest -- including SYS_ATTACK_ARC, which sweeps about Y.
    pub rotate_enabled: [bool; 3],
    /// Per-frame multiplier on rotation velocity -- a resistance, not a rate. The arc carries
    /// 0.99 and its cylinder 0.98, so the sweep slows as it goes.
    ///
    /// Zero means the emitter never set it, not "stop instantly": treating an unset field as
    /// total damping would freeze every rotation in the game.
    pub rotate_regist: f32,
    /// Number of cells the emitter's texture is divided into. Zero or one means the texture is
    /// a single image and the particle uses all of it.
    pub pattern_cells: u32,
    /// Cell index to show at each step of the pattern animation.
    pub pattern_table: Vec<i32>,
    /// Frames each pattern step lasts.
    pub pattern_frequency: f32,
    /// Cells the emitter's texture divides into, worked out from the texture's real size. Set
    /// by the caller, which is the side that knows the texture; the emitter data alone does
    /// not say.
    pub sheet_cells: u32,
    /// Name hash of the primitive this emitter draws, when it draws one. Matched against the
    /// file's descriptor table -- it is not an index, and treating it as one finds the wrong
    /// mesh or none.
    pub primitive_id: Option<u64>,
    /// The emitter's own offset from the effect's origin, and its orientation.
    ///
    /// Ignored until now, which stacked every emitter of an effect at one point: SYS_TURN_SMOKE
    /// separates its five emitters by up to 2 units front-to-back, and collapsing them together
    /// turns a cloud into a single blob.
    pub translation: glam::Vec3,
    /// Euler rotation, radians. EDGE_ATTACK_DASH_HIT turns two of its emitters 90 degrees.
    pub rotation: glam::Vec3,
    pub color0: Vec<ColorKey>,
    pub alpha0: Vec<ColorKey>,
}

fn attr(emitter: &EmitterDef, slots: &Slots, which: Option<usize>) -> Option<f32> {
    let _ = slots;
    match emitter.attrs.get(which?).and_then(|value| value.as_ref())? {
        AttrValue::Int(v) => Some(*v as f32),
        AttrValue::UInt(v) => Some(*v as f32),
        AttrValue::Float(v) => Some(*v),
    }
}

/// Attribute-table positions, looked up once per load rather than per emitter.
pub struct Slots {
    rate: Option<usize>,
    rate_random: Option<usize>,
    interval: Option<usize>,
    emission_start: Option<usize>,
    emission_duration: Option<usize>,
    position_random: Option<usize>,
    gravity_scale: Option<usize>,
    gravity_dir: [Option<usize>; 3],
    life: Option<usize>,
    life_random: Option<usize>,
    all_direction: Option<usize>,
    designated_dir: [Option<usize>; 3],
    designated_dir_scale: Option<usize>,
    diffusion: [Option<usize>; 3],
    velocity_random: Option<usize>,
    scale: [Option<usize>; 3],
    primitive_id: Option<usize>,
    translation: [Option<usize>; 3],
    rotation: [Option<usize>; 3],
    pattern_cells: Option<usize>,
    pattern_frequency: Option<usize>,
    pattern_table: Vec<Option<usize>>,
    num_scale_keys: Option<usize>,
    scale_keys: Vec<[Option<usize>; 4]>,
    billboard_type: Option<usize>,
    blend_type: Option<usize>,
    rotate_init: [Option<usize>; 3],
    rotate_init_rand: [Option<usize>; 3],
    rotate_add: [Option<usize>; 3],
    rotate_add_rand: [Option<usize>; 3],
    is_rotate: [Option<usize>; 3],
    rotate_regist: Option<usize>,
}

impl Slots {
    pub fn new() -> Self {
        let table = crate::eff_attrs::table();
        let at = |id: &str| table.iter().position(|attr| attr.id == id);
        Self {
            rate: at("emission.rate"),
            rate_random: at("emission.rate_random"),
            interval: at("emission.interval"),
            emission_start: at("emission.start"),
            emission_duration: at("emission.duration"),
            position_random: at("emission.position_random"),
            gravity_scale: at("emission.gravity_scale"),
            gravity_dir: [
                at("emission.gravity_dir_x"),
                at("emission.gravity_dir_y"),
                at("emission.gravity_dir_z"),
            ],
            life: at("particle_data.life"),
            life_random: at("particle_data.life_random"),
            all_direction: at("particle_velocity.all_direction"),
            designated_dir: [
                at("particle_velocity.designated_dir_x"),
                at("particle_velocity.designated_dir_y"),
                at("particle_velocity.designated_dir_z"),
            ],
            designated_dir_scale: at("particle_velocity.designated_dir_scale"),
            diffusion: [
                at("particle_velocity.diffusion_x"),
                at("particle_velocity.diffusion_y"),
                at("particle_velocity.diffusion_z"),
            ],
            velocity_random: at("particle_velocity.vel_random"),
            scale: [
                at("emitter_info.scale_x"),
                at("emitter_info.scale_y"),
                at("emitter_info.scale_z"),
            ],
            primitive_id: at("particle_data.primitive_id"),
            translation: [
                at("emitter_info.trans_x"),
                at("emitter_info.trans_y"),
                at("emitter_info.trans_z"),
            ],
            rotation: [
                at("emitter_info.rotate_x"),
                at("emitter_info.rotate_y"),
                at("emitter_info.rotate_z"),
            ],
            pattern_cells: at("emitter_static.tex_pattern_anim0.num"),
            pattern_frequency: at("emitter_static.tex_pattern_anim0.frequency"),
            pattern_table: (0..32)
                .map(|i| at(&format!("emitter_static.tex_pattern_anim0.table[{i}]")))
                .collect(),
            num_scale_keys: at("emitter_static.num_scale_keys"),
            scale_keys: (0..8)
                .map(|i| {
                    [
                        at(&format!("emitter_static.scale_anim.keys[{i}].x")),
                        at(&format!("emitter_static.scale_anim.keys[{i}].y")),
                        at(&format!("emitter_static.scale_anim.keys[{i}].z")),
                        at(&format!("emitter_static.scale_anim.keys[{i}].time")),
                    ]
                })
                .collect(),
            billboard_type: at("particle_data.billboard_type"),
            blend_type: at("render_state.blend_type"),
            rotate_init: [
                at("emitter_static.rotate_init_x"),
                at("emitter_static.rotate_init_y"),
                at("emitter_static.rotate_init_z"),
            ],
            rotate_init_rand: [
                at("emitter_static.rotate_init_rand_x"),
                at("emitter_static.rotate_init_rand_y"),
                at("emitter_static.rotate_init_rand_z"),
            ],
            rotate_add: [
                at("emitter_static.rotate_add_x"),
                at("emitter_static.rotate_add_y"),
                at("emitter_static.rotate_add_z"),
            ],
            rotate_add_rand: [
                at("emitter_static.rotate_add_rand_x"),
                at("emitter_static.rotate_add_rand_y"),
                at("emitter_static.rotate_add_rand_z"),
            ],
            is_rotate: [
                at("particle_data.is_rotate_x"),
                at("particle_data.is_rotate_y"),
                at("particle_data.is_rotate_z"),
            ],
            rotate_regist: at("emitter_static.rotate_regist"),
        }
    }
}

impl Default for Slots {
    fn default() -> Self {
        Self::new()
    }
}

impl EmitterSim {
    pub fn read(emitter: &EmitterDef, slots: &Slots) -> Self {
        let get = |which: Option<usize>| attr(emitter, slots, which);
        let vec3 = |which: &[Option<usize>; 3]| {
            glam::Vec3::new(
                get(which[0]).unwrap_or(0.0),
                get(which[1]).unwrap_or(0.0),
                get(which[2]).unwrap_or(0.0),
            )
        };
        let gravity_scale = get(slots.gravity_scale).unwrap_or(0.0);
        Self {
            // A rate of zero would emit nothing at all, which is almost never what the data
            // means — it means the field was not populated for this emitter.
            primitive_id: slots.primitive_id.and_then(|slot| {
                match emitter.attrs.get(slot).and_then(|value| value.as_ref()) {
                    Some(AttrValue::UInt(v)) => Some(*v),
                    Some(AttrValue::Int(v)) if *v > 0 => Some(*v as u64),
                    _ => None,
                }
            }),
            translation: vec3(&slots.translation),
            rotation: vec3(&slots.rotation),
            rate: get(slots.rate).unwrap_or(1.0).max(0.0),
            rate_random: get(slots.rate_random).unwrap_or(0.0),
            interval: get(slots.interval).unwrap_or(0.0).max(0.0),
            emission_start: get(slots.emission_start).unwrap_or(0.0),
            emission_duration: get(slots.emission_duration).unwrap_or(0.0),
            position_random: get(slots.position_random).unwrap_or(0.0),
            gravity: vec3(&slots.gravity_dir) * gravity_scale,
            life: get(slots.life).unwrap_or(20.0).max(1.0),
            life_random: get(slots.life_random).unwrap_or(0.0),
            all_direction: get(slots.all_direction).unwrap_or(0.0),
            designated_dir: vec3(&slots.designated_dir),
            designated_dir_scale: get(slots.designated_dir_scale).unwrap_or(0.0),
            diffusion: vec3(&slots.diffusion),
            velocity_random: get(slots.velocity_random).unwrap_or(0.0),
            scale: {
                let scale = vec3(&slots.scale);
                // An all-zero scale draws nothing. Treated as "unset" rather than "invisible",
                // because an emitter that renders nothing at all is not a shape the data uses.
                if scale.length_squared() <= f32::EPSILON {
                    glam::Vec3::ONE
                } else {
                    scale
                }
            },
            scale_keys: {
                let count = get(slots.num_scale_keys).unwrap_or(0.0).max(0.0) as usize;
                slots
                    .scale_keys
                    .iter()
                    .take(count.min(8))
                    .filter_map(|key| {
                        Some((
                            get(key[3])?,
                            glam::Vec3::new(get(key[0])?, get(key[1])?, get(key[2])?),
                        ))
                    })
                    .collect()
            },
            pattern_cells: get(slots.pattern_cells).unwrap_or(0.0).max(0.0) as u32,
            pattern_frequency: get(slots.pattern_frequency).unwrap_or(1.0).max(1.0),
            sheet_cells: 0,
            pattern_table: slots
                .pattern_table
                .iter()
                .filter_map(|slot| get(*slot).map(|v| v as i32))
                .collect(),
            billboard_type: emitter
                .attrs
                .get(slots.billboard_type.unwrap_or(usize::MAX))
                .and_then(|value| value.as_ref())
                .map(|value| match value {
                    AttrValue::Int(v) => *v,
                    AttrValue::UInt(v) => *v as i64,
                    AttrValue::Float(v) => *v as i64,
                })
                .unwrap_or(0),
            blend_type: emitter
                .attrs
                .get(slots.blend_type.unwrap_or(usize::MAX))
                .and_then(|value| value.as_ref())
                .map(|value| match value {
                    AttrValue::Int(v) => *v,
                    AttrValue::UInt(v) => *v as i64,
                    AttrValue::Float(v) => *v as i64,
                })
                .unwrap_or(0),
            rotate_init: vec3(&slots.rotate_init),
            rotate_init_rand: vec3(&slots.rotate_init_rand),
            rotate_add: vec3(&slots.rotate_add),
            rotate_add_rand: vec3(&slots.rotate_add_rand),
            rotate_enabled: [
                get(slots.is_rotate[0]).unwrap_or(0.0) != 0.0,
                get(slots.is_rotate[1]).unwrap_or(0.0) != 0.0,
                get(slots.is_rotate[2]).unwrap_or(0.0) != 0.0,
            ],
            rotate_regist: get(slots.rotate_regist).unwrap_or(0.0),
            color0: emitter.color0.clone(),
            alpha0: emitter.alpha0_keys.clone(),
        }
    }

    /// How many frames this emitter keeps producing for. Zero duration means "as long as the
    /// effect runs", which is how a continuous emitter (a flame, a trail) is expressed.
    fn emission_span(&self) -> f32 {
        if self.emission_duration > 0.0 {
            self.emission_duration
        } else {
            f32::INFINITY
        }
    }

    /// Frames between births. `interval` wins when set; otherwise the rate is per frame.
    fn birth_step(&self) -> f32 {
        if self.interval > 0.0 {
            self.interval
        } else if self.rate > 0.0 {
            1.0 / self.rate
        } else {
            1.0
        }
    }
}

/// One particle, evaluated.
#[derive(Debug, Clone, Copy)]
pub struct SimParticle {
    pub offset: glam::Vec3,
    pub size: f32,
    pub color: [f32; 4],
    pub rotation: f32,
    /// Which cell of the emitter's sprite sheet this particle is showing right now.
    pub cell: u32,
    /// The particle's own turn, per axis, in radians at this age.
    ///
    /// Three axes rather than one because the emitter says which axis it animates and it is
    /// not always the same one -- an arc sweeps about Y, a spark tumbles about Z. Which of
    /// these is visible depends on how the quad is built, so the choice is made where the
    /// billboard mode is known rather than here.
    pub spin: glam::Vec3,
    /// Where the particle is heading right now, in the effect's own frame.
    ///
    /// The derivative of `offset`, so it already carries gravity: a spark thrown up and falling
    /// back points up early in its life and down late in it, which is the whole reason a
    /// velocity-oriented billboard looks like a spark and not a square. Unused by the plane
    /// modes, which take their axes from the emitter instead.
    pub velocity: glam::Vec3,
}

/// Sample a keyframe list at a normalised age.
///
/// The lists are sparse — often a single key — so an empty list means "no animation", not
/// "black". Returning the default for that case is what keeps a one-key emitter visible.
fn sample_keys(keys: &[ColorKey], age: f32, life: f32, default: [f32; 4]) -> [f32; 4] {
    if keys.is_empty() {
        return default;
    }
    if keys.len() == 1 {
        let key = keys[0];
        return [key.r, key.g, key.b, key.a];
    }
    // Keys are in frames along the particle's own life.
    let t = age.clamp(0.0, life);
    let mut previous = keys[0];
    for key in keys {
        if key.frame >= t {
            let span = key.frame - previous.frame;
            let mix = if span > f32::EPSILON {
                (t - previous.frame) / span
            } else {
                0.0
            };
            return [
                previous.r + (key.r - previous.r) * mix,
                previous.g + (key.g - previous.g) * mix,
                previous.b + (key.b - previous.b) * mix,
                previous.a + (key.a - previous.a) * mix,
            ];
        }
        previous = *key;
    }
    [previous.r, previous.g, previous.b, previous.a]
}

/// Particle size at an age, from the scale curve.
///
/// An emitter with no scale keys is not a zero-sized particle -- it is a particle whose size
/// does not animate, so the emitter's own scale carries it.
fn sample_scale(keys: &[(f32, glam::Vec3)], age: f32, life: f32) -> glam::Vec3 {
    if keys.is_empty() {
        return glam::Vec3::ONE;
    }
    if keys.len() == 1 {
        return keys[0].1;
    }
    let t = age.clamp(0.0, life);
    let mut previous = keys[0];
    for key in keys {
        if key.0 >= t {
            let span = key.0 - previous.0;
            let mix = if span > f32::EPSILON {
                (t - previous.0) / span
            } else {
                0.0
            };
            return previous.1 + (key.1 - previous.1) * mix;
        }
        previous = *key;
    }
    previous.1
}

/// How a sprite sheet of `cells` frames is laid out across a `width`×`height` texture.
///
/// The format stores the cell COUNT but not the grid, so the grid has to be inferred. The
/// constraint that pins it down is that cells are square: effect sheets are grids of equal
/// square frames, so the right layout is the smallest grid whose cells come out square and
/// which has room for every frame. Checked against real textures — `ef_cmn_fire00` at 512²
/// with 16 cells is 4×4 of 128px, `ef_cmn_impact11` at 256×128 with 5 is 4×2 of 64px, and
/// `ef_cmn_fireimpact04` at 256×768 with 8 is 2×6 of 128px — none of which a naive "N across"
/// or fixed 4×4 would get right.
///
/// A grid can hold more cells than the animation uses (12 frames in a 4×4 sheet is common), so
/// the fit is `columns * rows >= cells`, not equality.
pub fn sheet_grid(width: u32, height: u32, cells: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (1, 1);
    }
    if cells <= 1 {
        // No declared cell count. Most emitters that animate a sheet still leave `num` at 0 --
        // the smoke effects do -- so falling back to "use the whole texture" draws a 256x128
        // strip squashed onto a square quad, which is the square a foot-dust puff was showing.
        //
        // Effect sheet cells are square, so a non-square texture is a strip of them and its
        // own proportions give the grid. A square texture stays whole: without a cell count
        // there is nothing to say it is subdivided.
        let smaller = width.min(height);
        if smaller == 0 || width % smaller != 0 || height % smaller != 0 {
            return (1, 1);
        }
        let (columns, rows) = (width / smaller, height / smaller);
        // A strip of more than 16 would mean a very long sheet; more likely the texture is not
        // a sheet at all and the proportions are a coincidence.
        if columns * rows > 16 {
            return (1, 1);
        }
        return (columns, rows);
    }
    // Square cells means width/columns == height/rows, so the grid's aspect matches the
    // texture's. Walk column counts and keep the first that fits.
    let mut best = (cells.max(1), 1);
    for columns in 1..=64u32 {
        // Cells sit on whole pixels. Without this, a 128px sheet of 5 frames "fits" a 3×3 grid
        // of 42.67px cells — square, large enough, and not how any sheet is actually cut.
        if width % columns != 0 {
            continue;
        }
        let cell_width = width / columns;
        if cell_width == 0 || height % cell_width != 0 {
            continue;
        }
        let rows = height / cell_width;
        if rows > 0 && columns * rows >= cells {
            best = (columns, rows);
            break;
        }
    }
    best
}

/// UV rectangle for one cell: (offset_u, offset_v, scale_u, scale_v).
pub fn cell_uv(cell: u32, columns: u32, rows: u32) -> [f32; 4] {
    if columns <= 1 && rows <= 1 {
        return [0.0, 0.0, 1.0, 1.0];
    }
    let index = cell % (columns * rows).max(1);
    let column = index % columns;
    let row = index / columns;
    [
        column as f32 / columns as f32,
        row as f32 / rows as f32,
        1.0 / columns as f32,
        1.0 / rows as f32,
    ]
}

/// Cap on particles produced by one emitter in one evaluation.
///
/// A continuous emitter with a high rate, evaluated late in a long move, describes an unbounded
/// number of particles; the viewport has to stay interactive while scrubbing, and beyond a few
/// hundred quads per emitter nothing further is distinguishable on screen.
const MAX_PER_EMITTER: usize = 256;

/// Every particle this emitter has alive at `age_frames` since the effect started.
///
/// `seed` distinguishes emitters so two emitters with identical settings do not produce
/// identically jittered particles stacked on top of each other.
pub fn evaluate(sim: &EmitterSim, age_frames: f32, seed: u64) -> Vec<SimParticle> {
    let mut particles = Vec::new();
    if age_frames < sim.emission_start {
        return particles;
    }

    let step = sim.birth_step();
    let span = sim.emission_span();
    let since_start = age_frames - sim.emission_start;

    // Walk births backwards from the newest: the particles alive now are the most recent ones,
    // so the cap drops the oldest rather than never reaching the newest.
    let newest = (since_start / step).floor() as i64;
    for index in (0..=newest).rev() {
        if particles.len() >= MAX_PER_EMITTER {
            break;
        }
        let birth = index as f32 * step;
        if birth > span {
            continue;
        }
        let age = since_start - birth;
        if age < 0.0 {
            continue;
        }
        let id = index as u64;
        // `life_random` is a PERCENTAGE of the particle's life, not a number of frames. Read
        // as frames it produced lives from -6 to 34 on a 14-frame particle, which is where
        // the corpus values (±20, ±30, ±50 against lives of 6 to 20) stop making sense.
        let spread = 1.0 + (sim.life_random / 100.0) * hashed_signed(seed, id ^ 0x11);
        let life = (sim.life * spread).max(1.0);
        if age >= life {
            // Births are ordered, so everything older than this is dead too.
            break;
        }

        // Initial velocity: an outward burst, a designated direction, and a per-axis spread.
        // All three appear together in the data, so all three are summed rather than picked
        // between.
        let dir = glam::Vec3::new(
            hashed_signed(seed, id ^ 0x21),
            hashed_signed(seed, id ^ 0x22),
            hashed_signed(seed, id ^ 0x23),
        )
        .normalize_or_zero();
        let jitter = 1.0 + sim.velocity_random * hashed_signed(seed, id ^ 0x31);
        let velocity = (dir * sim.all_direction
            + sim.designated_dir * sim.designated_dir_scale
            + sim.diffusion
                * glam::Vec3::new(
                    hashed_signed(seed, id ^ 0x41),
                    hashed_signed(seed, id ^ 0x42),
                    hashed_signed(seed, id ^ 0x43),
                ))
            * jitter;

        let spawn = glam::Vec3::new(
            hashed_signed(seed, id ^ 0x51),
            hashed_signed(seed, id ^ 0x52),
            hashed_signed(seed, id ^ 0x53),
        ) * sim.position_random;

        // Closed form, so any frame costs the same as any other.
        let offset = spawn + velocity * age + sim.gravity * (0.5 * age * age);

        let fraction = (age / life).clamp(0.0, 1.0);
        let color = sample_keys(&sim.color0, age, life, [1.0, 1.0, 1.0, 1.0]);
        let alpha = sample_keys(&sim.alpha0, age, life, [1.0, 1.0, 1.0, 1.0])[0];

        // Fade the last quarter of life. The data expresses fades through flags and curves this
        // does not read yet, and a particle that pops out of existence at full brightness is
        // the single most obviously wrong thing on screen.
        let fade = if fraction > 0.75 {
            1.0 - (fraction - 0.75) / 0.25
        } else {
            1.0
        };

        // Which frame of the sheet this particle is on. The table is the sequence of cells;
        // stepping it by age is what makes a smoke puff billow instead of showing every frame
        // of its animation at once.
        // Two ways a particle picks its cell. An emitter with a declared cell count animates
        // through its table as it ages. One without -- the smoke family, which leaves `num` at
        // 0 -- is picking a variant per particle instead, so every puff in a cloud is not the
        // same drawing; that one is chosen by the particle's own hash so it stays put while
        // the playhead moves.
        let cell = if sim.pattern_cells > 1 && !sim.pattern_table.is_empty() {
            let step = (age / sim.pattern_frequency.max(1.0)) as usize;
            let entry = sim.pattern_table[step.min(sim.pattern_table.len() - 1)];
            entry.max(0) as u32 % sim.pattern_cells
        } else if sim.sheet_cells > 1 {
            (hashed(seed, id ^ 0x71) * sim.sheet_cells as f32) as u32 % sim.sheet_cells
        } else {
            0
        };

        // The turn the emitter actually asks for, per axis, rather than a random one.
        //
        // A cloud still looks like a cloud through this, because a puff emitter carries a full
        // turn in `rotate_init_rand_*` and gets its scatter from the data. An arc carries zeros
        // for its initial angle and a velocity about Y, so it sweeps instead of sitting at a
        // different random angle every time the frame is evaluated.
        let mut spin = glam::Vec3::ZERO;
        for axis in 0..3 {
            if !sim.rotate_enabled[axis] {
                continue;
            }
            let start = sim.rotate_init[axis]
                + sim.rotate_init_rand[axis] * hashed_signed(seed, id ^ (0x61 + axis as u64));
            let rate = sim.rotate_add[axis]
                + sim.rotate_add_rand[axis] * hashed_signed(seed, id ^ (0x71 + axis as u64));
            // `rotate_regist` damps the velocity each frame, so the total turn is a geometric
            // series rather than rate*age. Outside (0, 1) it is either unset or not damping,
            // and the undamped sum is the right reading of both.
            let turned = if sim.rotate_regist > 0.0 && sim.rotate_regist < 1.0 {
                let r = sim.rotate_regist;
                rate * (1.0 - r.powf(age)) / (1.0 - r)
            } else {
                rate * age
            };
            spin[axis] = start + turned;
        }

        let curve = sample_scale(&sim.scale_keys, age, life);
        particles.push(SimParticle {
            offset,
            // Emitter scale multiplies the particle's own curve, which is how the data is
            // laid out: the curve is the shape of the size over life, the emitter scales it.
            size: (curve.x.max(curve.y) * sim.scale.x.max(sim.scale.y)).max(0.01),
            color: [color[0], color[1], color[2], alpha * fade],
            rotation: spin.z,
            spin,
            cell,
            // d/dt of `offset`. Taking the birth velocity instead would point every particle
            // of a falling burst upwards for its whole life.
            velocity: velocity + sim.gravity * age,
        });
    }
    particles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emitter() -> EmitterSim {
        EmitterSim {
            rate: 2.0,
            rate_random: 0.0,
            interval: 0.0,
            emission_start: 0.0,
            emission_duration: 0.0,
            position_random: 0.0,
            gravity: glam::Vec3::new(0.0, -1.0, 0.0),
            life: 10.0,
            life_random: 0.0,
            all_direction: 0.0,
            designated_dir: glam::Vec3::Y,
            designated_dir_scale: 2.0,
            diffusion: glam::Vec3::ZERO,
            velocity_random: 0.0,
            scale: glam::Vec3::ONE,
            scale_keys: Vec::new(),
            billboard_type: 0,
            blend_type: 0,
            // No roll: the fixture is about emission and motion, and a spin would only make
            // its assertions harder to read.
            rotate_init: glam::Vec3::ZERO,
            rotate_init_rand: glam::Vec3::ZERO,
            rotate_add: glam::Vec3::ZERO,
            rotate_add_rand: glam::Vec3::ZERO,
            rotate_enabled: [false; 3],
            rotate_regist: 0.0,
            pattern_cells: 0,
            pattern_table: Vec::new(),
            pattern_frequency: 1.0,
            sheet_cells: 0,
            primitive_id: None,
            translation: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            color0: Vec::new(),
            alpha0: Vec::new(),
        }
    }

    /// The whole reason the simulation is closed-form. An editor scrubs, and a stepped system
    /// answers "what does frame 30 look like" differently depending on how the playhead got
    /// there — or refuses to answer at all without replaying from the start.
    #[test]
    fn evaluating_a_frame_does_not_depend_on_how_the_playhead_reached_it() {
        let sim = emitter();
        let direct = evaluate(&sim, 30.0, 7);
        // Same frame, reached after evaluating a scatter of others including later ones.
        for frame in [0.0, 5.0, 60.0, 12.0, 99.0, 1.0] {
            let _ = evaluate(&sim, frame, 7);
        }
        let after_scrubbing = evaluate(&sim, 30.0, 7);
        assert_eq!(direct.len(), after_scrubbing.len());
        for (a, b) in direct.iter().zip(&after_scrubbing) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.color, b.color);
            assert_eq!(a.rotation, b.rotation);
        }
    }

    #[test]
    fn particles_are_born_over_time_and_die_at_their_life() {
        let sim = emitter();
        // Rate 2/frame, life 10 -> at most 20 alive once the stream is saturated.
        assert!(evaluate(&sim, 0.0, 1).len() <= 1);
        let early = evaluate(&sim, 3.0, 1).len();
        let saturated = evaluate(&sim, 30.0, 1).len();
        assert!(early > 0, "nothing emitted after 3 frames");
        assert!(saturated > early, "the stream never grew: {early} -> {saturated}");
        assert!(
            saturated <= 21,
            "particles outliving their life: {saturated} alive at rate 2 life 10"
        );
    }

    /// Emission is capped, but the cap must keep the NEWEST particles. Dropping from the front
    /// would show a stream permanently frozen at its oldest, and the freshly emitted particles
    /// — the ones at the emitter, where the eye goes — would never appear.
    #[test]
    fn the_particle_cap_keeps_the_newest_rather_than_the_oldest() {
        let mut sim = emitter();
        sim.rate = 500.0;
        sim.life = 1000.0;
        let particles = evaluate(&sim, 100.0, 1);
        assert_eq!(particles.len(), MAX_PER_EMITTER);
        // Under constant downward gravity from a fixed spawn, an older particle has fallen
        // further. The youngest kept should be near the origin.
        let lowest = particles
            .iter()
            .map(|particle| particle.offset.y)
            .fold(f32::INFINITY, f32::min);
        let highest = particles
            .iter()
            .map(|particle| particle.offset.y)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            highest > lowest,
            "every kept particle is the same age — the cap is not keeping a window"
        );
        assert!(
            highest > -1.0,
            "the newest particles were dropped: highest y {highest}"
        );
    }

    /// The blend mode comes from the emitter, not from an assumption about the corpus.
    ///
    /// Measured on two real fighter files (ef_kamui, ef_ridley): blend_type is 0 for about
    /// 73% of emitters, 1 for 25% and 2 for 3%. Drawing everything additive -- which is what
    /// this did -- therefore put roughly three quarters of the game's emitters in the wrong
    /// mode, and in the direction that makes smoke and dust glow.
    #[test]
    fn only_the_emitters_that_ask_for_additive_get_it() {
        assert!(!super::super::eff_runtime::is_additive(0), "0 is normal alpha");
        assert!(super::super::eff_runtime::is_additive(1), "1 is additive");
        // Subtractive darkens. With no subtractive pipeline, normal is far closer than
        // additive, which would light up the very thing meant to dim.
        assert!(!super::super::eff_runtime::is_additive(2), "2 is subtractive");
    }

    /// The arc sweeps about Y, and the axis comes from the data.
    ///
    /// SYS_ATTACK_ARC sets `is_rotate_y`, `rotate_add_y` of 0.0349 rad/frame and a
    /// `rotate_regist` of 0.99, and leaves X and Z untouched. An implementation that reads
    /// only Z gives it no motion at all -- which is what "the arc's shape never matches"
    /// turned out to be.
    #[test]
    fn the_axis_a_particle_turns_about_is_the_one_the_emitter_enables() {
        let mut sim = emitter();
        sim.life = 40.0;
        sim.rate = 1.0;
        // The arc's own numbers.
        sim.rotate_enabled = [false, true, false];
        sim.rotate_add = glam::Vec3::new(0.0, 0.034906585, 0.0);
        sim.rotate_regist = 0.99;

        let particles = evaluate(&sim, 8.0, 7);
        let newest = particles.first().expect("a particle exists");
        let oldest = particles.last().expect("several ages exist");
        // X and Z are not enabled, so nothing may leak into them.
        assert_eq!(newest.spin.x, 0.0);
        assert_eq!(newest.spin.z, 0.0);
        assert!(oldest.spin.y > newest.spin.y, "the sweep must accumulate with age");

        // Damped, so the turn is a geometric series and falls short of rate*age.
        let age = 8.0f32;
        let undamped = 0.034906585 * age;
        let damped = particles
            .iter()
            .map(|p| p.spin.y)
            .fold(0.0f32, f32::max);
        assert!(damped < undamped, "regist 0.99 must slow the sweep: {damped} vs {undamped}");
        assert!(damped > undamped * 0.9, "0.99 is light damping, not a stop: {damped}");
    }

    /// An emitter that asks for no roll gets no roll.
    ///
    /// This is the whole difference between an attack arc and a smoke puff, and it used to be
    /// wrong in the direction that hides: every particle was given a uniformly random turn,
    /// which is indistinguishable from correct on a radial puff and ruins a crescent. The arc
    /// then sat at a different angle every time it was evaluated, which reads as "the shape is
    /// wrong" rather than as "the roll is random" -- and no orientation setting could fix it,
    /// because the error was per-particle rather than per-effect.
    #[test]
    fn a_particle_rolls_the_way_its_emitter_says_rather_than_at_random() {
        let mut sim = emitter();
        sim.life = 40.0;
        sim.rate = 1.0;

        // Zeros throughout: the arc case. Every particle must sit flat.
        for particle in evaluate(&sim, 6.0, 99) {
            assert_eq!(particle.rotation, 0.0, "an unrotated emitter produced a roll");
        }

        // A quarter turn asked for, and no randomness: every particle takes exactly it.
        sim.rotate_enabled = [false, false, true];
        sim.rotate_init = glam::Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        for particle in evaluate(&sim, 6.0, 99) {
            assert!((particle.rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        }

        // Rotation velocity accumulates over the particle's own age, so a long-lived particle
        // has turned further than a freshly born one.
        sim.rotate_init = glam::Vec3::ZERO;
        sim.rotate_add = glam::Vec3::new(0.0, 0.0, 0.1);
        sim.rotate_enabled = [false, false, true];
        let rolls: Vec<f32> = evaluate(&sim, 6.0, 99).iter().map(|p| p.rotation).collect();
        assert!(rolls.len() > 1, "need several ages to compare");
        let (newest, oldest) = (rolls[0], rolls[rolls.len() - 1]);
        assert!(
            oldest > newest,
            "the older particle should have turned further: {oldest} vs {newest}"
        );

        // And the puff case still scatters, because the DATA asks it to.
        sim.rotate_add = glam::Vec3::ZERO;
        sim.rotate_init_rand = glam::Vec3::new(0.0, 0.0, std::f32::consts::PI);
        let spread: Vec<f32> = evaluate(&sim, 6.0, 99).iter().map(|p| p.rotation).collect();
        assert!(
            spread.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-3),
            "an emitter asking for random roll must still scatter: {spread:?}"
        );
    }

    #[test]
    fn gravity_and_velocity_move_a_particle_the_way_the_closed_form_says() {
        let mut sim = emitter();
        sim.rate = 1.0;
        sim.designated_dir_scale = 0.0;
        sim.gravity = glam::Vec3::new(0.0, -2.0, 0.0);
        // The particle born at t=0 is the oldest; at age 5 it should have fallen ½·2·25 = 25.
        let particles = evaluate(&sim, 5.0, 3);
        let oldest = particles
            .iter()
            .map(|particle| particle.offset.y)
            .fold(f32::INFINITY, f32::min);
        assert!((oldest + 25.0).abs() < 0.001, "fell to {oldest}, expected -25");
    }

    #[test]
    fn a_single_colour_key_is_used_rather_than_treated_as_no_colour() {
        let keys = vec![ColorKey {
            frame: 0.0,
            r: 0.25,
            g: 0.5,
            b: 0.75,
            a: 1.0,
        }];
        assert_eq!(
            sample_keys(&keys, 5.0, 10.0, [1.0; 4]),
            [0.25, 0.5, 0.75, 1.0]
        );
        // No keys means no animation, not black.
        assert_eq!(sample_keys(&[], 5.0, 10.0, [1.0; 4]), [1.0; 4]);
    }

    #[test]
    fn colour_interpolates_between_keys_across_the_life() {
        let keys = vec![
            ColorKey { frame: 0.0, r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            ColorKey { frame: 10.0, r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
        ];
        let middle = sample_keys(&keys, 5.0, 10.0, [1.0; 4]);
        assert!((middle[0] - 0.5).abs() < 0.001, "got {middle:?}");
    }

    /// Every case here is a real texture from the dump with its real cell count. The grid is
    /// not stored anywhere, so this inference is the whole basis for showing one frame of a
    /// sheet instead of all of them, and getting it wrong shows a slice of four smoke puffs
    /// rather than one.
    #[test]
    fn sheet_grids_are_inferred_from_real_textures() {
        // (width, height, cells, expected grid)
        for (width, height, cells, expected) in [
            (512, 512, 16, (4, 4)),   // ef_cmn_fire00
            (128, 128, 16, (4, 4)),   // ef_cmn_bomb_indirect00
            (1024, 1024, 12, (4, 4)), // ef_cmn_fireimpact00 — 12 frames in a 16-cell sheet
            (256, 768, 8, (2, 6)),    // ef_cmn_fireimpact04 — non-square texture
            (256, 128, 5, (4, 2)),    // ef_cmn_impact11
            (128, 128, 5, (4, 4)),    // ef_cmn_wind00
            (256, 256, 6, (4, 4)),    // ef_cmn_impact05_ani
        ] {
            let grid = sheet_grid(width, height, cells);
            assert_eq!(
                grid, expected,
                "{width}x{height} with {cells} cells gave {grid:?}"
            );
            let (columns, rows) = grid;
            assert!(
                columns * rows >= cells,
                "{width}x{height}: grid {grid:?} cannot hold {cells} cells"
            );
            // Square cells is the property the inference rests on.
            let cell_w = width as f32 / columns as f32;
            let cell_h = height as f32 / rows as f32;
            assert!(
                (cell_w - cell_h).abs() < 0.51,
                "{width}x{height} grid {grid:?} gives non-square cells {cell_w}x{cell_h}"
            );
        }

        assert_eq!(cell_uv(0, 1, 1), [0.0, 0.0, 1.0, 1.0]);
    }

    /// Emitters that animate a sheet but leave the cell count at zero — the whole smoke family
    /// does — must still be divided, or a 256x128 strip is drawn squashed onto a square quad
    /// and a foot-dust puff reads as a square.
    #[test]
    fn an_undeclared_sheet_is_divided_by_the_textures_own_proportions() {
        // ef_cmn_smoke01 / smoke03: two square frames side by side.
        assert_eq!(sheet_grid(256, 128, 0), (2, 1));
        // ef_cmn_smoke04.
        assert_eq!(sheet_grid(384, 128, 0), (3, 1));
        // ef_cmn_fireimpact04: a vertical strip.
        assert_eq!(sheet_grid(256, 768, 0), (1, 3));

        // A square texture with no declared count stays whole — nothing says it is subdivided,
        // and guessing would slice single images into quarters.
        assert_eq!(sheet_grid(256, 256, 0), (1, 1));
        assert_eq!(sheet_grid(133, 133, 0), (1, 1)); // ef_cmn_smoke02
        // Proportions that do not divide evenly are not a sheet.
        assert_eq!(sheet_grid(133, 100, 0), (1, 1));
        // An implausibly long strip is more likely a coincidence than a 32-frame sheet.
        assert_eq!(sheet_grid(2048, 64, 0), (1, 1));

        // A declared count still wins over the proportions.
        assert_eq!(sheet_grid(256, 128, 5), (4, 2));
    }

    /// With no declared pattern, particles pick a cell each rather than all showing the same
    /// one — a cloud of identical puffs reads as wrong as a squashed one. The choice must be
    /// stable per particle so scrubbing does not reshuffle the cloud.
    #[test]
    fn particles_on_an_undeclared_sheet_vary_and_stay_put() {
        let mut sim = emitter();
        sim.rate = 4.0;
        sim.life = 20.0;
        sim.pattern_cells = 0;
        sim.sheet_cells = 2;

        let first = evaluate(&sim, 10.0, 9);
        let cells: std::collections::BTreeSet<u32> =
            first.iter().map(|particle| particle.cell).collect();
        assert!(
            cells.len() > 1,
            "every particle picked the same cell: {cells:?}"
        );
        assert!(
            cells.iter().all(|cell| *cell < 2),
            "cell outside the sheet: {cells:?}"
        );

        // Same frame again after scrubbing elsewhere: identical assignment.
        let _ = evaluate(&sim, 3.0, 9);
        let _ = evaluate(&sim, 40.0, 9);
        let again = evaluate(&sim, 10.0, 9);
        let before: Vec<u32> = first.iter().map(|p| p.cell).collect();
        let after: Vec<u32> = again.iter().map(|p| p.cell).collect();
        assert_eq!(before, after, "the cloud reshuffled when the playhead moved");
    }

    #[test]
    fn a_cell_maps_to_its_own_corner_of_the_sheet() {
        // 4x4: cell 0 top-left, cell 5 is column 1 row 1.
        assert_eq!(cell_uv(0, 4, 4), [0.0, 0.0, 0.25, 0.25]);
        assert_eq!(cell_uv(5, 4, 4), [0.25, 0.25, 0.25, 0.25]);
        // Out of range wraps rather than sampling outside the sheet.
        assert_eq!(cell_uv(16, 4, 4), cell_uv(0, 4, 4));
    }

    /// A particle on a sheet must advance through its frames as it ages. Showing cell 0 for
    /// the whole life is the same visual bug as showing the whole sheet, just subtler.
    #[test]
    fn a_particle_walks_the_pattern_table_as_it_ages() {
        let mut sim = emitter();
        sim.rate = 1.0;
        sim.life = 8.0;
        sim.pattern_cells = 16;
        sim.pattern_frequency = 1.0;
        sim.pattern_table = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // The oldest particle at each moment is the one born at t=0, so its cell tracks age.
        let cell_at = |age: f32| {
            evaluate(&sim, age, 5)
                .into_iter()
                .map(|particle| particle.cell)
                .max()
                .unwrap_or(0)
        };
        assert_eq!(cell_at(0.0), 0);
        assert!(cell_at(3.0) > cell_at(1.0), "the sheet never advanced");
        // Past the end of the table it holds the last frame rather than wrapping to the start.
        assert_eq!(cell_at(20.0), cell_at(7.0));

        // No pattern means cell 0 throughout, not a walk through a table that isn't there.
        sim.pattern_cells = 0;
        assert!(evaluate(&sim, 5.0, 5).iter().all(|p| p.cell == 0));
    }

    #[test]
    fn emission_start_delays_the_first_particle() {
        let mut sim = emitter();
        sim.emission_start = 10.0;
        assert!(evaluate(&sim, 5.0, 1).is_empty());
        assert!(!evaluate(&sim, 12.0, 1).is_empty());
    }

    #[test]
    fn a_finite_duration_stops_emission_but_lets_the_last_particles_live_out() {
        let mut sim = emitter();
        sim.emission_duration = 5.0;
        sim.life = 10.0;
        let during = evaluate(&sim, 4.0, 1).len();
        let just_after = evaluate(&sim, 8.0, 1).len();
        let long_after = evaluate(&sim, 40.0, 1).len();
        assert!(during > 0);
        assert!(just_after > 0, "particles vanished the moment emission stopped");
        assert_eq!(long_after, 0, "emission never stopped");
    }
}
