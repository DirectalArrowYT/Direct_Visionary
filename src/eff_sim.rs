//! Particle simulation, evaluated closed-form at any frame.
//!
//! The editor scrubs. A conventional particle system steps state forward a frame at a time and
//! cannot answer "what does frame 30 look like" without having run frames 0..29 first, so
//! dragging the playhead backwards either shows the wrong thing or forces a replay from the
//! start of the move. Everything here is therefore a pure function of the particle's age, and
//! every random quantity comes from hashing the particle's identity rather than from a running
//! generator. Scrub anywhere, get the same answer, in the same time.
//!
//! The rules themselves -- when an emitter fires, how many particles a firing makes, what each
//! random percentage does, how drag and gravity step -- are the game's own, read out of the
//! particle library in the 13.0.3 executable (NintendoWare Vfx, see
//! `research/decomp/ssbu-re`). Where a comment gives an address, that is the function the rule
//! was read from. The game steps a frame at a time; the forms here are the sums of those steps,
//! so they agree with it on whole frames and interpolate between them.
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
    /// Particles per firing, not per frame: the emitter fires every `interval + 1` frames and
    /// each firing makes this many at once (0x8d340). Fractions carry over to the next firing.
    pub rate: f32,
    /// Percentage a firing's count may fall short by. One-sided, as every percentage here is.
    pub rate_random: f32,
    /// Frames skipped between firings. Zero fires every frame.
    pub interval: f32,
    /// Up to this many extra whole frames added to each gap (0x88750).
    pub interval_random: f32,
    pub emission_start: f32,
    pub emission_duration: f32,
    /// Whether the emitter stops after `emission_duration`. An emitter without this set keeps
    /// firing for as long as the effect lives, whatever its duration says. `None` when the file
    /// does not carry the flag, which falls back to "stops if it has a duration".
    pub is_one_time: Option<bool>,
    /// Radius of a random shell around the spawn point: a unit direction times this, not a
    /// box.
    pub position_random: f32,
    /// Whether gravity pulls along the world's axes rather than the emitter's own.
    pub world_gravity: bool,
    /// Per-frame multiplier on velocity. 1 is no drag.
    pub air_res: f32,
    /// Symmetric spread on how far a particle travels per frame for its velocity: each one
    /// moves at `1 +- this` of the common pace for its whole life.
    pub momentum_random: f32,
    pub gravity: glam::Vec3,
    pub life: f32,
    pub life_random: f32,
    pub all_direction: f32,
    pub designated_dir: glam::Vec3,
    pub designated_dir_scale: f32,
    pub diffusion: glam::Vec3,
    pub velocity_random: f32,
    /// Half-angle, in degrees, of a cone around the designated direction that the directed
    /// speed is spread over. Zero keeps every particle exactly on the direction.
    pub diffusion_dir_angle: f32,
    /// Speed outward from the emitter's Y axis, taken from where in the shape the particle
    /// was born.
    pub xz_diffusion: f32,
    pub scale: glam::Vec3,
    /// Particle size over its own life: (frame, xyz). This is the PARTICLE's scale, distinct
    /// from `scale` above which is the emitter's — reading the emitter's as the particle's
    /// gives every particle in the game the same size, because emitters are nearly all 1.0.
    pub scale_keys: Vec<(f32, glam::Vec3)>,
    /// How the emitter says its texture is cut into cells, if it says at all.
    ///
    /// This is the authority when it is set: an explosion's 4x4 sheet is declared here and
    /// nowhere else, and inferring a grid from a square texture's proportions returns "one
    /// cell", which draws the whole sheet on every particle.
    pub uv_div: (u32, u32),
    /// The texture's own animation inside the particle, from `tex_scroll_anim0`: an offset
    /// and a zoom, each with a per-frame rate.
    ///
    /// This is what sweeps an attack arc. SYS_ATTACK_ARC_D scrolls V by 0.27 and zooms it
    /// 1.4x, both moving every frame, which slides a window across a slash frame so the arc
    /// is drawn from one end to the other. Without it the frame sits still, whole, and tapers
    /// at both ends.
    pub uv_scroll: glam::Vec2,
    pub uv_scroll_add: glam::Vec2,
    pub uv_scale: glam::Vec2,
    pub uv_scale_add: glam::Vec2,
    /// The shape particles are born in, as the game's own table at 0x4f48c30 orders them:
    /// 0 point, 1 circle, 2 circle-same-divide, 3 filled circle, 4 sphere,
    /// 5 sphere-same-divide, 6 sphere-same-divide-64, 7 filled sphere, 8 cylinder,
    /// 9 filled cylinder, 10 box, 11 filled box, 12 line, 13 line-same-divide, 14 rectangle,
    /// 15 primitive.
    ///
    /// Half the corpus is a point and the rest is mostly circles and spheres. The "same
    /// divide" variants space their particles evenly around the sweep instead of scattering
    /// them, which is what makes a shockwave ring a ring rather than a smear.
    pub volume_type: i64,
    /// Half-extent of that shape per axis.
    pub volume_radius: glam::Vec3,
    /// How much of the shape is used, in radians: longitude around, latitude from the pole.
    /// 2pi and pi -- the whole thing -- on nearly every emitter, but an arc of a ring is a
    /// sweep of less.
    pub sweep_longitude: f32,
    pub sweep_latitude: f32,
    pub sweep_start: f32,
    /// How a sphere is cut down: 0 by longitude (a wedge), 1 by latitude (a cap, around
    /// `latitude_dir`).
    pub arc_type: i64,
    /// Which way a latitude cap points: +X, -X, +Y, -Y, +Z, -Z.
    pub latitude_dir: i64,
    /// For the line shapes: how long, and where its centre sits.
    pub line_length: f32,
    pub line_center: f32,
    /// How much of a filled shape's radius is filled, from the rim inwards: 1.0 fills it to
    /// the middle, 0.2 leaves a hole 0.8 of the radius wide, which is how a ring of fire keeps
    /// its hole.
    pub caliber_ratio: f32,
    /// Start the sweep at a random angle each firing instead of at `sweep_start`.
    pub sweep_start_random: bool,
    /// Angular jitter, radians either way, on the evenly divided circle's positions.
    pub surface_pos_rand: f32,
    /// How many positions the evenly divided circle and line have, and the percentage a firing
    /// may fall short of that. A firing makes `rate` particles for each position.
    pub num_divide_circle: (u32, f32),
    pub num_divide_line: (u32, f32),
    /// The particle's size in world units, before its curve and the emitter's scale.
    ///
    /// This is the one that carries the data: across ef_common it runs 0.1 to 300, while the
    /// emitter's own scale is 1.0 on all but two of 1348 emitters. Building a size without it
    /// draws every particle in the game about a unit across, whatever the effect asked for.
    pub particle_scale: glam::Vec3,
    /// Per-particle spread on that size, as a PERCENTAGE -- 30 means a particle lands
    /// anywhere in 0.7x to 1.3x. Better than half the corpus sets one, and without it a
    /// cloud is a stack of identically sized puffs.
    pub scale_random: glam::Vec3,
    pub billboard_type: i64,
    /// How the emitter's particles blend with what is behind them.
    ///
    /// 0 is normal alpha, 1 additive, 2 subtractive -- the format's own numbering, with 0 the
    /// default as a format's zero usually is. Across two real fighter files this splits about
    /// 73 / 25 / 3, so an assumption that everything is additive is wrong for roughly three
    /// quarters of every emitter in the game, and wrong in the direction that makes smoke and
    /// dust glow like fire.
    pub blend_type: i64,
    /// Brightness multiplier on the emitter's colour, straight out of the emitter's constant
    /// buffer (`emitter_static.color_scale`). Values run about 1.0 to 5.0 in the game's own
    /// effects, so ignoring it draws everything dimmer and flatter than it plays -- and the
    /// ones that clip to white in game, which is what makes a hit spark read as a flash,
    /// never clip here at all.
    pub color_scale: f32,
    /// Which texture channel carries the particle's shape: 0 the texture's alpha, 1 its red,
    /// 2 nothing (fully opaque).
    ///
    /// Confirmed in game rather than inferred: a decal set to 1 with black art disappeared,
    /// because its shape lives in alpha and its red is zero everywhere.
    pub alpha_input: i64,
    /// Where the particle's colour comes from: 0 the texture's RGB, 1 the emitter's colour
    /// alone, with the texture contributing only shape.
    pub color_input: i64,
    /// How the emitter's textures are meant to be read. 2 is the indirect path, where the
    /// first texture displaces the lookup into the second rather than being drawn itself --
    /// every emitter using it in ef_common binds a second texture, and it is how fire is built.
    pub shader_type: i64,
    /// How a plain second texture folds into the first: 0 multiply, 1 add, 2 subtract.
    pub tex1_blend: i64,
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
    interval_random: Option<usize>,
    emission_start: Option<usize>,
    emission_duration: Option<usize>,
    is_one_time: Option<usize>,
    is_world_gravity: Option<usize>,
    position_random: Option<usize>,
    air_res: Option<usize>,
    momentum_random: Option<usize>,
    diffusion_dir_angle: Option<usize>,
    xz_diffusion: Option<usize>,
    sweep_start_random: Option<usize>,
    surface_pos_rand: Option<usize>,
    num_divide_circle: [Option<usize>; 2],
    num_divide_line: [Option<usize>; 2],
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
    uv_div: [Option<usize>; 2],
    uv_scroll: [Option<usize>; 2],
    uv_scroll_add: [Option<usize>; 2],
    uv_scale: [Option<usize>; 2],
    uv_scale_add: [Option<usize>; 2],
    volume_type: Option<usize>,
    volume_radius: [Option<usize>; 3],
    volume_form_scale: [Option<usize>; 3],
    sweep_longitude: Option<usize>,
    sweep_latitude: Option<usize>,
    sweep_start: Option<usize>,
    arc_type: Option<usize>,
    latitude_dir: Option<usize>,
    line_length: Option<usize>,
    line_center: Option<usize>,
    caliber_ratio: Option<usize>,
    particle_scale: [Option<usize>; 3],
    scale_random: [Option<usize>; 3],
    primitive_id: Option<usize>,
    translation: [Option<usize>; 3],
    rotation: [Option<usize>; 3],
    pattern_cells: Option<usize>,
    pattern_frequency: Option<usize>,
    pattern_table: Vec<Option<usize>>,
    num_scale_keys: Option<usize>,
    scale_keys: Vec<[Option<usize>; 4]>,
    billboard_type: Option<usize>,
    color_scale: Option<usize>,
    alpha_input: Option<usize>,
    color_input: Option<usize>,
    shader_type: Option<usize>,
    tex1_blend: Option<usize>,
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
            interval_random: at("emission.interval_random"),
            is_one_time: at("emission.is_one_time"),
            is_world_gravity: at("emission.is_world_gravity"),
            air_res: at("emitter_static.air_res"),
            momentum_random: at("particle_data.momentum_random"),
            diffusion_dir_angle: at("particle_velocity.diffusion_dir_angle"),
            xz_diffusion: at("particle_velocity.xz_diffusion"),
            sweep_start_random: at("shape_info.sweep_start_random"),
            arc_type: at("shape_info.arc_type"),
            latitude_dir: at("shape_info.volume_latitude_dir"),
            surface_pos_rand: at("shape_info.volume_surface_pos_rand"),
            num_divide_circle: [
                at("shape_info.num_divide_circle"),
                at("shape_info.num_divide_circle_random"),
            ],
            num_divide_line: [
                at("shape_info.num_divide_line"),
                at("shape_info.num_divide_line_random"),
            ],
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
            uv_div: [
                at("emitter_static.tex_scroll_anim0.uv_div_x"),
                at("emitter_static.tex_scroll_anim0.uv_div_y"),
            ],
            uv_scroll: [
                at("emitter_static.tex_scroll_anim0.scroll_x"),
                at("emitter_static.tex_scroll_anim0.scroll_y"),
            ],
            uv_scroll_add: [
                at("emitter_static.tex_scroll_anim0.scroll_add_x"),
                at("emitter_static.tex_scroll_anim0.scroll_add_y"),
            ],
            uv_scale: [
                at("emitter_static.tex_scroll_anim0.scale_x"),
                at("emitter_static.tex_scroll_anim0.scale_y"),
            ],
            uv_scale_add: [
                at("emitter_static.tex_scroll_anim0.scale_add_x"),
                at("emitter_static.tex_scroll_anim0.scale_add_y"),
            ],
            volume_type: at("shape_info.volume_type"),
            volume_radius: [
                at("shape_info.volume_radius_x"),
                at("shape_info.volume_radius_y"),
                at("shape_info.volume_radius_z"),
            ],
            volume_form_scale: [
                at("shape_info.volume_form_scale_x"),
                at("shape_info.volume_form_scale_y"),
                at("shape_info.volume_form_scale_z"),
            ],
            sweep_longitude: at("shape_info.sweep_longitude"),
            sweep_latitude: at("shape_info.sweep_latitude"),
            sweep_start: at("shape_info.sweep_start"),
            line_length: at("shape_info.line_length"),
            line_center: at("shape_info.line_center"),
            caliber_ratio: at("shape_info.caliber_ratio"),
            particle_scale: [
                at("particle_scale.scale_x"),
                at("particle_scale.scale_y"),
                at("particle_scale.scale_z"),
            ],
            scale_random: [
                at("particle_scale.scale_random_x"),
                at("particle_scale.scale_random_y"),
                at("particle_scale.scale_random_z"),
            ],
            billboard_type: at("particle_data.billboard_type"),
            color_scale: at("emitter_static.color_scale"),
            alpha_input: at("combiner.tex_alpha0_input_type"),
            color_input: at("combiner.tex_color0_input_type"),
            shader_type: at("combiner.shader_type"),
            tex1_blend: at("combiner.texture1_color_blend"),
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
            interval_random: get(slots.interval_random).unwrap_or(0.0).max(0.0),
            emission_start: get(slots.emission_start).unwrap_or(0.0),
            emission_duration: get(slots.emission_duration).unwrap_or(0.0),
            is_one_time: get(slots.is_one_time).map(|v| v != 0.0),
            world_gravity: get(slots.is_world_gravity).unwrap_or(0.0) != 0.0,
            position_random: get(slots.position_random).unwrap_or(0.0),
            // Zero is a real setting -- a handful of emitters use it to throw a particle one
            // step and leave it to gravity -- so only a missing field means "no drag".
            air_res: get(slots.air_res).unwrap_or(1.0).max(0.0),
            momentum_random: get(slots.momentum_random).unwrap_or(0.0),
            diffusion_dir_angle: get(slots.diffusion_dir_angle).unwrap_or(0.0),
            xz_diffusion: get(slots.xz_diffusion).unwrap_or(0.0),
            sweep_start_random: get(slots.sweep_start_random).unwrap_or(0.0) != 0.0,
            arc_type: get(slots.arc_type).unwrap_or(0.0) as i64,
            // Unset is +Y, the pole the cap is built around.
            latitude_dir: get(slots.latitude_dir).unwrap_or(2.0) as i64,
            surface_pos_rand: get(slots.surface_pos_rand).unwrap_or(0.0),
            num_divide_circle: (
                get(slots.num_divide_circle[0]).unwrap_or(1.0).max(1.0) as u32,
                get(slots.num_divide_circle[1]).unwrap_or(0.0),
            ),
            num_divide_line: (
                get(slots.num_divide_line[0]).unwrap_or(1.0).max(1.0) as u32,
                get(slots.num_divide_line[1]).unwrap_or(0.0),
            ),
            gravity: vec3(&slots.gravity_dir) * gravity_scale,
            life: get(slots.life).unwrap_or(20.0).max(1.0),
            life_random: get(slots.life_random).unwrap_or(0.0),
            all_direction: get(slots.all_direction).unwrap_or(0.0),
            designated_dir: vec3(&slots.designated_dir),
            designated_dir_scale: get(slots.designated_dir_scale).unwrap_or(0.0),
            diffusion: vec3(&slots.diffusion),
            velocity_random: get(slots.velocity_random).unwrap_or(0.0),
            color_scale: get(slots.color_scale).unwrap_or(1.0).max(0.0),
            alpha_input: get(slots.alpha_input).unwrap_or(0.0) as i64,
            color_input: get(slots.color_input).unwrap_or(0.0) as i64,
            shader_type: get(slots.shader_type).unwrap_or(0.0) as i64,
            tex1_blend: get(slots.tex1_blend).unwrap_or(0.0) as i64,
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
            uv_scroll: glam::Vec2::new(
                get(slots.uv_scroll[0]).unwrap_or(0.0),
                get(slots.uv_scroll[1]).unwrap_or(0.0),
            ),
            uv_scroll_add: glam::Vec2::new(
                get(slots.uv_scroll_add[0]).unwrap_or(0.0),
                get(slots.uv_scroll_add[1]).unwrap_or(0.0),
            ),
            uv_scale: {
                let scale = glam::Vec2::new(
                    get(slots.uv_scale[0]).unwrap_or(1.0),
                    get(slots.uv_scale[1]).unwrap_or(1.0),
                );
                // A zero zoom is "unset", not "collapse the texture to one texel".
                glam::Vec2::new(
                    if scale.x.abs() <= f32::EPSILON { 1.0 } else { scale.x },
                    if scale.y.abs() <= f32::EPSILON { 1.0 } else { scale.y },
                )
            },
            uv_scale_add: glam::Vec2::new(
                get(slots.uv_scale_add[0]).unwrap_or(0.0),
                get(slots.uv_scale_add[1]).unwrap_or(0.0),
            ),
            uv_div: {
                let axis = |slot| get(slot).unwrap_or(1.0).round().clamp(1.0, 64.0) as u32;
                (axis(slots.uv_div[0]), axis(slots.uv_div[1]))
            },
            volume_type: get(slots.volume_type).unwrap_or(0.0) as i64,
            // The shape's own per-axis stretch multiplies its radius (0x8dc80).
            volume_radius: vec3(&slots.volume_radius)
                * glam::Vec3::new(
                    get(slots.volume_form_scale[0]).unwrap_or(1.0),
                    get(slots.volume_form_scale[1]).unwrap_or(1.0),
                    get(slots.volume_form_scale[2]).unwrap_or(1.0),
                ),
            sweep_longitude: get(slots.sweep_longitude).unwrap_or(std::f32::consts::TAU),
            sweep_latitude: get(slots.sweep_latitude).unwrap_or(std::f32::consts::PI),
            sweep_start: get(slots.sweep_start).unwrap_or(0.0),
            line_length: get(slots.line_length).unwrap_or(0.0),
            line_center: get(slots.line_center).unwrap_or(0.0),
            caliber_ratio: get(slots.caliber_ratio).unwrap_or(1.0),
            particle_scale: {
                let scale = vec3(&slots.particle_scale);
                // An unset base is 1, so the curve and emitter scale still decide the size,
                // rather than the particle collapsing to nothing.
                if scale.length_squared() <= f32::EPSILON {
                    glam::Vec3::ONE
                } else {
                    scale
                }
            },
            scale_random: vec3(&slots.scale_random),
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

    /// How many frames this emitter keeps firing for.
    ///
    /// Only a one-time emitter stops (0x8dc80): it fires while its clock is short of
    /// `start + duration`, so a duration of 1 is a single firing. Anything else runs until the
    /// effect is removed. A one-time emitter that has made nothing yet keeps trying past its
    /// end, so even a duration of 0 fires once -- see [`bursts`].
    fn emission_span(&self) -> f32 {
        match self.is_one_time {
            Some(true) => self.emission_duration.max(0.0),
            Some(false) => f32::INFINITY,
            None if self.emission_duration > 0.0 => self.emission_duration,
            None => f32::INFINITY,
        }
    }

    /// How many positions an evenly divided shape has on this firing. `None` for every other
    /// shape.
    fn divisions(&self, unit: f32) -> Option<u32> {
        let (count, random) = match self.volume_type {
            2 => self.num_divide_circle,
            13 => self.num_divide_line,
            _ => return None,
        };
        let short = (unit * random * 0.01 * count as f32) as u32;
        Some(count.saturating_sub(short).max(1))
    }
}

/// One firing of an emitter: when, and which particles it made.
struct Burst {
    /// Frames after the emitter's start.
    time: f32,
    /// Index of its first particle among all the emitter has made.
    first: u64,
    count: u32,
    /// Positions of an evenly divided shape on this firing.
    divisions: u32,
    /// The firing's own random number, shared by its particles (a random sweep start).
    unit: f32,
}

/// Every firing up to `until` frames after the emitter's start, oldest first.
///
/// The game's loop (0x8d340), run forward: when the gap has elapsed, add the rate -- less a
/// random percentage -- to a running total, emit its whole part, keep the fraction, and draw
/// the next gap. Walked from the start because both the fraction and a random gap depend on
/// everything before them; it is a few additions per firing.
fn bursts(sim: &EmitterSim, until: f32, seed: u64) -> Vec<Burst> {
    let span = sim.emission_span();
    let mut out = Vec::new();
    let mut time = 0.0f32;
    let mut saving = 0.0f32;
    let mut first = 0u64;
    let mut index = 0u64;
    // Past its end a one-time emitter still fires if it has not managed a particle yet,
    // which is what guarantees a rate below one, or a duration of zero, shows something.
    while time <= until && (time < span || out.is_empty()) && out.len() < MAX_BURSTS {
        let unit = hashed(seed, index ^ 0xb0_0000);
        saving += sim.rate * (1.0 - sim.rate_random / 100.0 * hashed(seed, index ^ 0xb1_0000));
        saving = saving.max(0.0);
        let whole = saving as u32;
        saving -= whole as f32;
        let divisions = sim.divisions(unit);
        let count = whole.saturating_mul(divisions.unwrap_or(1));
        if count > 0 {
            out.push(Burst {
                time,
                first,
                count,
                divisions: divisions.unwrap_or(1),
                unit,
            });
            first += count as u64;
        }
        let extra = (hashed(seed, index ^ 0xb2_0000) * sim.interval_random).floor();
        time += sim.interval + 1.0 + extra;
        index += 1;
    }
    out
}

/// Where a particle is after `age` frames, relative to its spawn point, and its velocity then.
///
/// The game's step (0x95370) is, each frame: move by the velocity, scale the velocity by the
/// drag, add gravity. Summed over `n` whole frames that is a geometric series, written out
/// here; between frames the two neighbouring sums are blended, which keeps a scrubbed
/// half-frame on the path the game's own frames lie on.
fn travel(v0: glam::Vec3, gravity: glam::Vec3, drag: f32, age: f32) -> (glam::Vec3, glam::Vec3) {
    let undamped = (drag - 1.0).abs() < 1e-6;
    let at = |n: f32| {
        if undamped {
            v0 * n + gravity * (n * (n - 1.0) * 0.5)
        } else {
            let sum = (1.0 - drag.powf(n)) / (1.0 - drag);
            v0 * sum + gravity * ((n - sum) / (1.0 - drag))
        }
    };
    let whole = age.max(0.0).floor();
    let part = age.max(0.0) - whole;
    let offset = at(whole).lerp(at(whole + 1.0), part);
    let velocity = if undamped {
        v0 + gravity * age
    } else {
        let decay = drag.powf(age);
        v0 * decay + gravity * ((1.0 - decay) / (1.0 - drag))
    };
    (offset, velocity)
}

/// One particle, evaluated.
#[derive(Debug, Clone, Copy)]
pub struct SimParticle {
    /// Where the particle is in the emitter's frame, from everything but gravity.
    pub offset: glam::Vec3,
    /// How far gravity has carried it. Kept apart because an emitter can ask for the world's
    /// down rather than its own, and only the caller knows which way that is.
    pub fallen: glam::Vec3,
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
    /// The texture's animation at this age: (scroll u, scroll v, zoom u, zoom v).
    pub uv_anim: [f32; 4],
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
pub fn sheet_grid(width: u32, height: u32, cells: u32, declared: (u32, u32)) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (1, 1);
    }
    // What the emitter says, where it says anything. Only a grid of one in both axes means
    // "not declared" -- every emitter carries this field, most of them at 1x1.
    if declared.0 > 1 || declared.1 > 1 {
        return (declared.0.max(1), declared.1.max(1));
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
/// Firings walked per evaluation. Past this an emitter has been running for hours.
const MAX_BURSTS: usize = 100_000;

/// A uniformly random unit vector.
fn unit_vector(seed: u64, id: u64, salt: u64) -> glam::Vec3 {
    let y = hashed_signed(seed, id ^ salt);
    let turn = hashed(seed, id ^ (salt + 1)) * std::f32::consts::TAU;
    let ring = (1.0 - y * y).max(0.0).sqrt();
    glam::Vec3::new(ring * turn.cos(), y, ring * turn.sin())
}

/// Where in the emitter's shape a particle is born, and the direction the shape throws it.
///
/// The game's shape functions return both: a position, and a velocity that is this direction
/// times the emitter's all-direction speed. `slot` is the particle's place within its firing,
/// which is what the evenly divided shapes space themselves by.
///
/// These are the game's own formulas (0x90c00 to 0x92e00), with two exceptions: the evenly
/// divided spheres place particles from a lookup table and are scattered here instead, and
/// the box, rectangle and primitive shapes are not modelled and spawn at the emitter.
fn volume_spawn(
    sim: &EmitterSim,
    burst: &Burst,
    slot: u32,
    id: u64,
    seed: u64,
) -> (glam::Vec3, glam::Vec3) {
    let unit = |salt: u64| hashed(seed, id ^ salt);
    let radius = sim.volume_radius;
    let start = if sim.sweep_start_random {
        burst.unit * std::f32::consts::TAU
    } else {
        sim.sweep_start
    };
    // The sweep is centred on its start, not begun there.
    let scattered = start + sim.sweep_longitude * (unit(0x81) - 0.5);
    // Angle 0 lies along +Z and the sweep turns toward +X.
    let on_circle = |angle: f32, reach: f32| {
        let dir = glam::Vec3::new(angle.sin(), 0.0, angle.cos());
        (
            glam::Vec3::new(dir.x * radius.x, 0.0, dir.z * radius.z) * reach,
            dir,
        )
    };

    match sim.volume_type {
        1 => on_circle(scattered, 1.0),
        2 => {
            // An arc puts a particle on each end, so it has one gap fewer than positions; a
            // full turn would stack its two ends, so it keeps every gap.
            let open = (sim.sweep_longitude - std::f32::consts::TAU).abs() > 1e-4;
            let gaps = if open && burst.divisions > 1 {
                burst.divisions - 1
            } else {
                burst.divisions
            };
            let angle = start + sim.sweep_longitude / gaps as f32 * slot as f32
                - sim.sweep_longitude * 0.5
                + sim.surface_pos_rand * hashed_signed(seed, id ^ 0x87);
            on_circle(angle, 1.0)
        }
        3 | 9 => {
            // Evenly over the ring between the hole and the rim.
            let inner = 1.0 - sim.caliber_ratio;
            let pick = unit(0x82);
            let reach = (pick + (1.0 - pick) * inner * inner).max(0.0).sqrt();
            let (mut position, dir) = on_circle(scattered, reach);
            // On an ellipse the throw leans toward the long axis.
            let mut thrown = dir * reach;
            if radius.x > radius.z {
                thrown.z *= radius.z / radius.x;
            } else if radius.z > 0.0 {
                thrown.x *= radius.x / radius.z;
            }
            if sim.volume_type == 9 {
                position.y = hashed_signed(seed, id ^ 0x86) * radius.y;
            }
            (position, thrown.normalize_or_zero())
        }
        8 => {
            let (mut position, dir) = on_circle(scattered, 1.0);
            position.y = hashed_signed(seed, id ^ 0x86) * radius.y;
            (position, dir)
        }
        4 | 5 | 6 | 7 => {
            // Height is drawn evenly, which is what covers a sphere evenly. A longitude cut
            // keeps the full height and narrows the turn; a latitude cut keeps the full turn
            // and stops the height at the cap's edge.
            let (turn, height) = if sim.arc_type == 1 {
                let edge = sim.sweep_latitude.cos();
                (
                    unit(0x81) * std::f32::consts::TAU,
                    1.0 - unit(0x83) * (1.0 - edge),
                )
            } else {
                (scattered, hashed_signed(seed, id ^ 0x83))
            };
            let ring = (1.0 - height * height).max(0.0).sqrt();
            let mut dir = glam::Vec3::new(ring * turn.sin(), height, ring * turn.cos());
            if sim.arc_type == 1 {
                let pole = match sim.latitude_dir {
                    0 => glam::Vec3::X,
                    1 => glam::Vec3::NEG_X,
                    3 => glam::Vec3::NEG_Y,
                    4 => glam::Vec3::Z,
                    5 => glam::Vec3::NEG_Z,
                    _ => glam::Vec3::Y,
                };
                dir = glam::Quat::from_rotation_arc(glam::Vec3::Y, pole) * dir;
            }
            // A filled sphere reaches from its hole to its rim (0x922c0).
            let reach = if sim.volume_type == 7 {
                unit(0x84).sqrt() * sim.caliber_ratio + 1.0 - sim.caliber_ratio
            } else {
                1.0
            };
            (dir * radius * reach, dir)
        }
        12 | 13 => {
            // Along Z. `line_center` slides the line: 0 centres it on the emitter.
            let fraction = if sim.volume_type == 13 {
                if burst.divisions > 1 {
                    slot as f32 / (burst.divisions - 1) as f32
                } else {
                    0.5
                }
            } else {
                unit(0x81)
            };
            let length = sim.line_length;
            let z = length * fraction - 0.5 * (length + length * sim.line_center);
            (glam::Vec3::new(0.0, 0.0, z), glam::Vec3::Z)
        }
        // A point, and the shapes not modelled: at the emitter, thrown in a random direction.
        _ => (glam::Vec3::ZERO, unit_vector(seed, id, 0x21)),
    }
}

/// Every particle this emitter has alive at `age_frames` since the effect started.
///
/// `seed` distinguishes emitters so two emitters with identical settings do not produce
/// identically jittered particles stacked on top of each other.
pub fn evaluate(sim: &EmitterSim, age_frames: f32, seed: u64) -> Vec<SimParticle> {
    let mut particles = Vec::new();
    if age_frames < sim.emission_start {
        return particles;
    }
    let since_start = age_frames - sim.emission_start;

    // Walk firings backwards from the newest: the particles alive now are the most recent
    // ones, so the cap drops the oldest rather than never reaching the newest.
    'bursts: for burst in bursts(sim, since_start, seed).iter().rev() {
        let age = since_start - burst.time;
        // Nothing lives longer than the unshortened life, and firings are in order, so every
        // older one is dead too.
        if age >= sim.life.max(1.0) {
            break;
        }
        for slot in (0..burst.count).rev() {
            if particles.len() >= MAX_PER_EMITTER {
                break 'bursts;
            }
            let id = burst.first + slot as u64;
            // A whole percentage of the life, taken off: 30 means anywhere from 70% to the
            // full life, in steps of 1% (0x908d8). Never added.
            let short = (hashed(seed, id ^ 0x11) * sim.life_random.max(0.0)).floor();
            let life = (sim.life * (1.0 - short / 100.0)).max(1.0);
            if age >= life {
                continue;
            }

            let (born, thrown) = volume_spawn(sim, burst, slot, id, seed);
            let mut velocity = thrown * sim.all_direction;
            if sim.xz_diffusion != 0.0 {
                // Away from the emitter's Y axis. A particle born on the axis has no "away",
                // so it gets a random one.
                let mut outward = glam::Vec3::new(born.x, 0.0, born.z);
                if outward.length_squared() <= f32::EPSILON {
                    outward = glam::Vec3::new(
                        hashed_signed(seed, id ^ 0x24),
                        0.0,
                        hashed_signed(seed, id ^ 0x25),
                    );
                }
                velocity += outward.normalize_or_zero() * sim.xz_diffusion;
            }
            // The directed part: straight along the designated direction, or spread over a
            // cone about it. The cone is sampled about +Y and turned onto the direction.
            let directed = if sim.diffusion_dir_angle == 0.0 {
                sim.designated_dir
            } else {
                let floor = 1.0 - sim.diffusion_dir_angle / 90.0;
                let turn = hashed(seed, id ^ 0x26) * std::f32::consts::TAU;
                let y = floor + (1.0 - floor) * hashed(seed, id ^ 0x27);
                let ring = (1.0 - y * y).max(0.0).sqrt();
                let sample = glam::Vec3::new(ring * turn.cos(), y, ring * turn.sin());
                let length = sim.designated_dir.length();
                if length > f32::EPSILON {
                    glam::Quat::from_rotation_arc(glam::Vec3::Y, sim.designated_dir / length)
                        * sample
                        * length
                } else {
                    glam::Vec3::ZERO
                }
            };
            velocity += directed * sim.designated_dir_scale;
            // A percentage taken off the speed, never added (0x8fd7c): 50 means anywhere from
            // half speed to full. It scales the shape's throw and the directed part together.
            velocity *= 1.0 - sim.velocity_random / 100.0 * hashed(seed, id ^ 0x31);
            // The per-axis spread goes on afterwards, at full strength.
            velocity += sim.diffusion
                * glam::Vec3::new(
                    hashed_signed(seed, id ^ 0x41),
                    hashed_signed(seed, id ^ 0x42),
                    hashed_signed(seed, id ^ 0x43),
                );

            let spawn = born + unit_vector(seed, id, 0x51) * sim.position_random;
            let pace = 1.0 + sim.momentum_random * hashed_signed(seed, id ^ 0x54);
            let (travelled, heading) = travel(velocity, glam::Vec3::ZERO, sim.air_res, age);
            let (fallen, falling) = travel(glam::Vec3::ZERO, sim.gravity, sim.air_res, age);
            let offset = spawn + travelled * pace;
            let fallen = fallen * pace;

            let fraction = (age / life).clamp(0.0, 1.0);
            let color = sample_keys(&sim.color0, age, life, [1.0, 1.0, 1.0, 1.0]);
            let alpha = sample_keys(&sim.alpha0, age, life, [1.0, 1.0, 1.0, 1.0])[0];

            // Fade the last quarter of life. The data expresses fades through flags and curves
            // this does not read yet, and a particle that pops out of existence at full
            // brightness is the single most obviously wrong thing on screen.
            let fade = if fraction > 0.75 {
                1.0 - (fraction - 0.75) / 0.25
            } else {
                1.0
            };

            // Two ways a particle picks its cell. An emitter with a declared cell count
            // animates through its table as it ages, which is what makes a smoke puff billow
            // instead of showing every frame of its animation at once. One without -- the
            // smoke family, which leaves `num` at 0 -- is picking a variant per particle
            // instead, so every puff in a cloud is not the same drawing; that one is chosen by
            // the particle's own hash so it stays put while the playhead moves.
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
            // A cloud still looks like a cloud through this, because a puff emitter carries a
            // full turn in `rotate_init_rand_*` and gets its scatter from the data. An arc
            // carries zeros for its initial angle and a velocity about Y, so it sweeps instead
            // of sitting at a different random angle every time the frame is evaluated.
            let mut spin = glam::Vec3::ZERO;
            for axis in 0..3 {
                if !sim.rotate_enabled[axis] {
                    continue;
                }
                let start = sim.rotate_init[axis]
                    + sim.rotate_init_rand[axis] * hashed_signed(seed, id ^ (0x61 + axis as u64));
                let rate = sim.rotate_add[axis]
                    + sim.rotate_add_rand[axis] * hashed_signed(seed, id ^ (0x71 + axis as u64));
                // `rotate_regist` damps the velocity each frame, so the total turn is a
                // geometric series rather than rate*age. Outside (0, 1) it is either unset or
                // not damping, and the undamped sum is the right reading of both.
                let turned = if sim.rotate_regist > 0.0 && sim.rotate_regist < 1.0 {
                    let r = sim.rotate_regist;
                    rate * (1.0 - r.powf(age)) / (1.0 - r)
                } else {
                    rate * age
                };
                spin[axis] = start + turned;
            }

            let curve = sample_scale(&sim.scale_keys, age, life);
            // A percentage taken off the size, never added (0x90750). One draw sizes both
            // axes when their percentages match, so a square stays square; when they differ
            // each axis draws its own.
            let shrink_x = 1.0 - sim.scale_random.x / 100.0 * hashed(seed, id ^ 0x91);
            let shrink_y = if sim.scale_random.x == sim.scale_random.y {
                shrink_x
            } else {
                1.0 - sim.scale_random.y / 100.0 * hashed(seed, id ^ 0x92)
            };
            // Base size, the curve over life, and the emitter's own scale on top, per axis.
            let width = curve.x * sim.particle_scale.x * shrink_x.max(0.0) * sim.scale.x;
            let height = curve.y * sim.particle_scale.y * shrink_y.max(0.0) * sim.scale.y;
            particles.push(SimParticle {
                offset,
                fallen,
                size: width.max(height).max(0.01),
                color: [
                    color[0] * sim.color_scale,
                    color[1] * sim.color_scale,
                    color[2] * sim.color_scale,
                    alpha * fade,
                ],
                rotation: spin.z,
                spin,
                cell,
                // Where it is heading now rather than where it was thrown: taking the birth
                // velocity would point every particle of a falling burst upwards for its
                // whole life.
                velocity: heading + falling,
                // Rates are per frame, so the animation at this age is its start plus rate x
                // age.
                uv_anim: {
                    let scroll = sim.uv_scroll + sim.uv_scroll_add * age;
                    let scale = sim.uv_scale + sim.uv_scale_add * age;
                    [scroll.x, scroll.y, scale.x, scale.y]
                },
            });
        }
    }
    particles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emitter() -> EmitterSim {
        EmitterSim {
            uv_div: (1, 1),
            uv_scroll: glam::Vec2::ZERO,
            uv_scroll_add: glam::Vec2::ZERO,
            uv_scale: glam::Vec2::ONE,
            uv_scale_add: glam::Vec2::ZERO,
            volume_type: 0,
            volume_radius: glam::Vec3::ZERO,
            sweep_longitude: std::f32::consts::TAU,
            sweep_latitude: std::f32::consts::PI,
            sweep_start: 0.0,
            line_length: 0.0,
            line_center: 0.0,
            caliber_ratio: 1.0,
            sweep_start_random: false,
            arc_type: 0,
            latitude_dir: 2,
            surface_pos_rand: 0.0,
            num_divide_circle: (1, 0.0),
            num_divide_line: (1, 0.0),
            is_one_time: None,
            world_gravity: false,
            interval_random: 0.0,
            air_res: 1.0,
            momentum_random: 0.0,
            diffusion_dir_angle: 0.0,
            xz_diffusion: 0.0,
            particle_scale: glam::Vec3::ONE,
            scale_random: glam::Vec3::ZERO,
            color_scale: 1.0,
            alpha_input: 0,
            color_input: 0,
            shader_type: 0,
            tex1_blend: 0,
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

    #[test]
    fn the_texture_animation_moves_with_the_particles_age() {
        // SYS_ATTACK_ARC_D's own values: V scrolls from 0.27 at -0.08 a frame and zooms from
        // 1.4 at -0.045 a frame. That motion is what sweeps the arc across its slash frame.
        let mut sim = emitter();
        sim.uv_scroll = glam::Vec2::new(0.0, 0.27);
        sim.uv_scroll_add = glam::Vec2::new(0.0, -0.08);
        sim.uv_scale = glam::Vec2::new(1.0, 1.4);
        sim.uv_scale_add = glam::Vec2::new(0.0, -0.045);
        let particles = evaluate(&sim, 5.0, 7);
        let oldest = particles
            .iter()
            .max_by(|a, b| a.uv_anim[1].partial_cmp(&b.uv_anim[1]).unwrap().reverse())
            .expect("particles");
        // The oldest particle has moved furthest: its scroll has dropped and its zoom shrunk.
        assert!(oldest.uv_anim[1] < 0.27, "scroll did not move: {:?}", oldest.uv_anim);
        assert!(oldest.uv_anim[3] < 1.4, "zoom did not move: {:?}", oldest.uv_anim);

        // An emitter without the animation carries the identity, so nothing it draws shifts.
        let still = evaluate(&emitter(), 5.0, 7);
        for particle in &still {
            assert_eq!(particle.uv_anim, [0.0, 0.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn a_declared_sheet_grid_beats_the_inferred_one() {
        // An explosion's sheet is square and its grid is declared, not implied: inferring from
        // proportions returns one cell, which draws the whole sheet on every particle.
        assert_eq!(sheet_grid(512, 512, 0, (4, 4)), (4, 4));
        // A declaration wins even where inference would have produced something plausible.
        assert_eq!(sheet_grid(256, 128, 0, (4, 4)), (4, 4));
        assert_eq!(sheet_grid(640, 128, 5, (5, 1)), (5, 1));
        // 1x1 is what nearly every emitter carries and means "nothing declared", so the
        // strips still get their proportions read.
        assert_eq!(sheet_grid(256, 128, 0, (1, 1)), (2, 1));
    }

    #[test]
    fn a_ring_emitter_puts_its_particles_on_the_ring() {
        // Type 1 is the circle: particles start on its edge, in the plane, at the radius the
        // data asks for. Every one used to start at the emitter's own point, so a shockwave
        // ring was a dot that expanded into a scatter.
        let mut sim = emitter();
        sim.volume_type = 1;
        sim.volume_radius = glam::Vec3::new(6.0, 0.0, 6.0);
        sim.position_random = 0.0;
        sim.all_direction = 0.0;
        sim.diffusion = glam::Vec3::ZERO;
        sim.gravity = glam::Vec3::ZERO;
        let particles = evaluate(&sim, 0.0, 7);
        assert!(!particles.is_empty(), "no particles born");
        for particle in &particles {
            let flat = glam::Vec2::new(particle.offset.x, particle.offset.z).length();
            assert!((flat - 6.0).abs() < 1e-3, "born {flat} from the centre, not 6");
            assert!(particle.offset.y.abs() < 1e-3, "born off the ring's plane");
        }
    }

    #[test]
    fn a_filled_sphere_keeps_its_particles_inside() {
        let mut sim = emitter();
        sim.volume_type = 7;
        sim.volume_radius = glam::Vec3::splat(4.0);
        sim.position_random = 0.0;
        sim.all_direction = 0.0;
        sim.diffusion = glam::Vec3::ZERO;
        sim.gravity = glam::Vec3::ZERO;
        let particles = evaluate(&sim, 0.0, 7);
        assert!(!particles.is_empty(), "no particles born");
        for particle in &particles {
            assert!(
                particle.offset.length() <= 4.0 + 1e-3,
                "born {} from the centre of a radius-4 sphere",
                particle.offset.length()
            );
        }
    }

    #[test]
    fn the_same_divide_shapes_space_their_particles_evenly() {
        // Type 2 is the circle that divides its sweep instead of scattering: each firing puts
        // `rate` particles on every one of its divisions, which is what a ring reads as.
        let mut sim = emitter();
        sim.volume_type = 2;
        sim.volume_radius = glam::Vec3::new(5.0, 0.0, 5.0);
        sim.num_divide_circle = (8, 0.0);
        sim.rate = 1.0;
        sim.interval = 100.0;
        sim.position_random = 0.0;
        sim.all_direction = 0.0;
        sim.diffusion = glam::Vec3::ZERO;
        sim.gravity = glam::Vec3::ZERO;
        let particles = evaluate(&sim, 1.0, 7);
        assert_eq!(particles.len(), 8, "one firing should fill all eight divisions");
        let mut angles: Vec<f32> = particles
            .iter()
            .map(|p| p.offset.z.atan2(p.offset.x).rem_euclid(std::f32::consts::TAU))
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Every angle should be a multiple of an eighth of a turn.
        let eighth = std::f32::consts::TAU / 8.0;
        for angle in &angles {
            let step = angle / eighth;
            assert!(
                (step - step.round()).abs() < 1e-3,
                "{angle} is not on an eighth of the ring"
            );
        }
    }

    #[test]
    fn a_particle_is_sized_by_its_own_scale_rather_than_its_emitters() {
        // particle_scale carries the size in world units -- 0.1 to 300 across ef_common --
        // while emitter_info.scale is 1.0 on all but two of its 1348 emitters. Sizing from the
        // emitter alone drew every particle in the game about a unit across.
        let mut sim = emitter();
        sim.particle_scale = glam::Vec3::ONE;
        let small = evaluate(&sim, 3.0, 7);
        sim.particle_scale = glam::Vec3::splat(8.0);
        let large = evaluate(&sim, 3.0, 7);
        assert!(!small.is_empty(), "no particles to compare");
        for (a, b) in small.iter().zip(&large) {
            assert!(
                (b.size - a.size * 8.0).abs() < 1e-4,
                "{} is not 8x {}",
                b.size,
                a.size
            );
        }
    }

    #[test]
    fn scale_random_spreads_sizes_by_a_percentage() {
        // 30 means 0.7x to 1x -- taken off, never added -- not 30 units and not 30x.
        let mut sim = emitter();
        sim.particle_scale = glam::Vec3::splat(10.0);
        sim.scale_random = glam::Vec3::splat(30.0);
        let particles = evaluate(&sim, 20.0, 7);
        assert!(particles.len() > 2, "need a few particles to see a spread");
        let sizes: Vec<f32> = particles.iter().map(|p| p.size).collect();
        let (low, high) = sizes.iter().fold((f32::MAX, 0.0f32), |(l, h), &s| (l.min(s), h.max(s)));
        assert!(high > low, "every particle came out the same size");
        for size in &sizes {
            assert!(
                (7.0..=10.0).contains(size),
                "{size} is outside the 30% the data asked for"
            );
        }
    }

    #[test]
    fn velocity_random_is_a_percentage_too() {
        // It reads 5 to 90 in the corpus, the same shape as life_random. Taken as a fraction
        // it multiplied a particle's speed by up to 91, which threw particles off screen in a
        // frame instead of varying the ones in a puff.
        let mut sim = emitter();
        sim.all_direction = 1.0;
        sim.velocity_random = 0.0;
        let steady = evaluate(&sim, 6.0, 7);
        sim.velocity_random = 90.0;
        let varied = evaluate(&sim, 6.0, 7);
        assert!(!steady.is_empty(), "no particles to measure");
        let fastest = |particles: &[SimParticle]| {
            particles.iter().map(|p| p.velocity.length()).fold(0.0f32, f32::max)
        };
        let (base, spread) = (fastest(&steady), fastest(&varied));
        // At 90% the quickest particle may reach about 1.9x the steady speed. Read as a
        // fraction it would have reached 91x.
        assert!(
            spread <= base * 2.0,
            "fastest went from {base} to {spread}: that is {:.0}x, not a percentage",
            spread / base.max(1e-6)
        );
        assert!(spread > base, "nothing varied at all");
    }

    #[test]
    fn the_emitters_colour_scale_multiplies_its_particles() {
        // The game multiplies by this on the way to the shader -- it is in the emitter's own
        // constant buffer -- so a preview that ignores it draws every effect dimmer than it
        // plays, and never reaches the clipping that makes a hit spark read as a flash.
        let mut sim = emitter();
        sim.color_scale = 1.0;
        let plain = evaluate(&sim, 3.0, 7);
        sim.color_scale = 2.5;
        let scaled = evaluate(&sim, 3.0, 7);
        assert_eq!(plain.len(), scaled.len());
        assert!(!plain.is_empty(), "no particles to compare");
        for (a, b) in plain.iter().zip(&scaled) {
            for channel in 0..3 {
                assert!(
                    (b.color[channel] - a.color[channel] * 2.5).abs() < 1e-5,
                    "channel {channel}: {} is not 2.5x {}",
                    b.color[channel],
                    a.color[channel]
                );
            }
            // Alpha is the particle's shape over time, not its brightness: the scale is a
            // colour multiplier and leaving alpha alone is what keeps a fade a fade.
            assert_eq!(a.color[3], b.color[3]);
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
        // Two a frame, life 10 -> exactly 20 alive once the stream is saturated.
        assert_eq!(evaluate(&sim, 0.0, 1).len(), 2, "the first firing is on the first frame");
        let early = evaluate(&sim, 3.0, 1).len();
        let saturated = evaluate(&sim, 30.0, 1).len();
        assert!(early > 0, "nothing emitted after 3 frames");
        assert!(saturated > early, "the stream never grew: {early} -> {saturated}");
        assert!(
            saturated <= 20,
            "particles outliving their life: {saturated} alive at rate 2 life 10"
        );
    }

    /// Emission is capped, but the cap must keep the NEWEST particles. Dropping from the front
    /// would show a stream permanently frozen at its oldest, and the freshly emitted particles
    /// — the ones at the emitter, where the eye goes — would never appear.
    #[test]
    fn the_particle_cap_keeps_the_newest_rather_than_the_oldest() {
        let mut sim = emitter();
        // A hundred a frame, so the cap spans three firings rather than part of one.
        sim.rate = 100.0;
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
        use crate::eff_render::BlendMode;
        assert_eq!(BlendMode::from_blend_type(0), BlendMode::Alpha, "0 is normal alpha");
        assert_eq!(BlendMode::from_blend_type(1), BlendMode::Additive, "1 is additive");
        // Subtractive darkens what is behind it. It used to fall back to normal alpha, which
        // was the closer of the two wrong answers; it now has a pipeline of its own.
        assert_eq!(BlendMode::from_blend_type(2), BlendMode::Subtract, "2 is subtractive");
        // Anything the format has not defined draws the way the format's zero does.
        assert_eq!(BlendMode::from_blend_type(7), BlendMode::Alpha, "unknown falls back to alpha");
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
        // The game moves by the velocity and then adds gravity to it, so a particle at rest
        // does not move on its first frame: after 5 frames it has fallen 0+2+4+6+8 = 20, not
        // the 25 that half g t squared gives.
        let particles = evaluate(&sim, 5.0, 3);
        let oldest = particles
            .iter()
            .map(|particle| particle.fallen.y)
            .fold(f32::INFINITY, f32::min);
        assert!((oldest + 20.0).abs() < 0.001, "fell to {oldest}, expected -20");
        // Gravity's share is reported apart from the rest, so the caller can aim it.
        assert!(particles.iter().all(|particle| particle.offset.y.abs() < 1e-6));
    }

    #[test]
    fn drag_shortens_how_far_a_particle_carries() {
        let mut sim = emitter();
        sim.rate = 1.0;
        sim.interval = 100.0;
        sim.gravity = glam::Vec3::ZERO;
        sim.designated_dir = glam::Vec3::X;
        sim.designated_dir_scale = 8.0;
        sim.air_res = 0.5;
        // 8 + 4 + 2 = 14 after three frames, and never past 16 however long it lives.
        let three = evaluate(&sim, 3.0, 3)[0].offset.x;
        assert!((three - 14.0).abs() < 1e-3, "travelled {three}, expected 14");
        sim.life = 100.0;
        let late = evaluate(&sim, 60.0, 3)[0].offset.x;
        assert!((late - 16.0).abs() < 1e-3, "travelled {late}, expected to settle at 16");
    }

    #[test]
    fn an_emitter_fires_its_whole_rate_every_interval_plus_one_frames() {
        let mut sim = emitter();
        sim.rate = 3.0;
        sim.interval = 4.0;
        sim.life = 1000.0;
        // Firings at 0, 5 and 10, three particles each.
        assert_eq!(evaluate(&sim, 0.0, 1).len(), 3);
        assert_eq!(evaluate(&sim, 4.0, 1).len(), 3);
        assert_eq!(evaluate(&sim, 5.0, 1).len(), 6);
        assert_eq!(evaluate(&sim, 10.0, 1).len(), 9);
    }

    #[test]
    fn a_fractional_rate_carries_over_between_firings() {
        let mut sim = emitter();
        sim.rate = 0.5;
        sim.life = 1000.0;
        // Half a particle a frame is one every second frame, not none.
        assert_eq!(evaluate(&sim, 0.0, 1).len(), 0);
        assert_eq!(evaluate(&sim, 1.0, 1).len(), 1);
        assert_eq!(evaluate(&sim, 9.0, 1).len(), 5);
    }

    #[test]
    fn only_a_one_time_emitter_stops_at_its_duration() {
        let mut sim = emitter();
        sim.rate = 1.0;
        sim.life = 1000.0;
        sim.emission_duration = 3.0;
        sim.is_one_time = Some(true);
        assert_eq!(evaluate(&sim, 50.0, 1).len(), 3, "frames 0, 1 and 2 fire; frame 3 does not");
        sim.is_one_time = Some(false);
        assert_eq!(evaluate(&sim, 50.0, 1).len(), 51, "a looping emitter ignores its duration");
        // The commonest emitter in the game: one-time, and too short to fire on its own.
        sim.is_one_time = Some(true);
        sim.emission_duration = 0.0;
        sim.rate = 4.0;
        assert_eq!(evaluate(&sim, 50.0, 1).len(), 4, "it still fires exactly once");
    }

    #[test]
    fn the_random_percentages_only_ever_take_away() {
        let mut sim = emitter();
        sim.rate = 40.0;
        sim.interval = 1000.0;
        sim.gravity = glam::Vec3::ZERO;
        sim.designated_dir = glam::Vec3::X;
        sim.designated_dir_scale = 10.0;
        sim.velocity_random = 50.0;
        sim.life = 100.0;
        sim.life_random = 50.0;
        // One frame in: each particle has moved exactly its own speed.
        let speeds: Vec<f32> = evaluate(&sim, 1.0, 9).iter().map(|p| p.offset.x).collect();
        assert_eq!(speeds.len(), 40);
        let (low, high) = speeds
            .iter()
            .fold((f32::MAX, 0.0f32), |(l, h), &s| (l.min(s), h.max(s)));
        assert!(low >= 5.0 - 1e-3 && high <= 10.0 + 1e-3, "speeds ran {low} to {high}");
        assert!(high - low > 2.0, "the spread is missing: {low} to {high}");
        // Lives run from half to full, so some are gone by 60 and all are by 100.
        let at_60 = evaluate(&sim, 60.0, 9).len();
        assert!(at_60 > 0 && at_60 < 40, "{at_60} of 40 alive at frame 60");
        assert_eq!(evaluate(&sim, 49.0, 9).len(), 40, "none may die before half life");
        assert!(evaluate(&sim, 100.0, 9).is_empty(), "none may outlive the full life");
    }

    #[test]
    fn a_fully_filled_circle_reaches_its_centre_and_a_thin_one_keeps_its_hole() {
        let mut sim = emitter();
        sim.volume_type = 3;
        sim.volume_radius = glam::Vec3::new(10.0, 0.0, 10.0);
        sim.rate = 200.0;
        sim.interval = 1000.0;
        sim.gravity = glam::Vec3::ZERO;
        sim.designated_dir_scale = 0.0;
        let reach = |sim: &EmitterSim| {
            evaluate(sim, 0.0, 4)
                .iter()
                .map(|p| p.offset.length())
                .fold((f32::MAX, 0.0f32), |(l, h), r| (l.min(r), h.max(r)))
        };
        sim.caliber_ratio = 1.0;
        let (inner, outer) = reach(&sim);
        assert!(inner < 3.0 && outer <= 10.0 + 1e-3, "a full disc ran {inner} to {outer}");
        sim.caliber_ratio = 0.2;
        let (inner, outer) = reach(&sim);
        assert!(inner >= 8.0 - 1e-3 && outer <= 10.0 + 1e-3, "a thin ring ran {inner} to {outer}");
    }

    #[test]
    fn a_sphere_is_covered_evenly_and_a_latitude_cut_leaves_a_cap() {
        let mut sim = emitter();
        sim.volume_type = 4;
        sim.volume_radius = glam::Vec3::splat(1.0);
        sim.rate = 250.0;
        sim.interval = 1000.0;
        sim.gravity = glam::Vec3::ZERO;
        sim.designated_dir_scale = 0.0;
        // Evenly covered means height is evenly spread: its mean size is a half. Scattering
        // the polar angle instead crowds the poles and gives about 0.64.
        let heights: Vec<f32> = evaluate(&sim, 0.0, 6).iter().map(|p| p.offset.y).collect();
        let mean = heights.iter().map(|y| y.abs()).sum::<f32>() / heights.len() as f32;
        assert!((mean - 0.5).abs() < 0.06, "mean height {mean}");
        // A 60 degree cap about -X: every particle at least half a radius out along -X.
        sim.arc_type = 1;
        sim.latitude_dir = 1;
        sim.sweep_latitude = std::f32::consts::FRAC_PI_3;
        for particle in evaluate(&sim, 0.0, 6) {
            assert!(particle.offset.x <= -0.5 + 1e-3, "{:?} is outside the cap", particle.offset);
        }
    }

    #[test]
    fn a_line_lies_along_z_with_a_particle_on_each_end() {
        let mut sim = emitter();
        sim.volume_type = 13;
        sim.line_length = 6.0;
        sim.num_divide_line = (4, 0.0);
        sim.rate = 1.0;
        sim.interval = 1000.0;
        sim.gravity = glam::Vec3::ZERO;
        sim.designated_dir_scale = 0.0;
        let mut along: Vec<f32> = evaluate(&sim, 0.0, 2)
            .iter()
            .map(|p| {
                assert!(p.offset.x.abs() < 1e-4 && p.offset.y.abs() < 1e-4);
                p.offset.z
            })
            .collect();
        along.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(along.len(), 4);
        for (got, want) in along.iter().zip([-3.0, -1.0, 1.0, 3.0]) {
            assert!((got - want).abs() < 1e-4, "{along:?}");
        }
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
            let grid = sheet_grid(width, height, cells, (1, 1));
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
        assert_eq!(sheet_grid(256, 128, 0, (1, 1)), (2, 1));
        // ef_cmn_smoke04.
        assert_eq!(sheet_grid(384, 128, 0, (1, 1)), (3, 1));
        // ef_cmn_fireimpact04: a vertical strip.
        assert_eq!(sheet_grid(256, 768, 0, (1, 1)), (1, 3));

        // A square texture with no declared count stays whole — nothing says it is subdivided,
        // and guessing would slice single images into quarters.
        assert_eq!(sheet_grid(256, 256, 0, (1, 1)), (1, 1));
        assert_eq!(sheet_grid(133, 133, 0, (1, 1)), (1, 1)); // ef_cmn_smoke02
        // Proportions that do not divide evenly are not a sheet.
        assert_eq!(sheet_grid(133, 100, 0, (1, 1)), (1, 1));
        // An implausibly long strip is more likely a coincidence than a 32-frame sheet.
        assert_eq!(sheet_grid(2048, 64, 0, (1, 1)), (1, 1));

        // A declared count still wins over the proportions.
        assert_eq!(sheet_grid(256, 128, 5, (1, 1)), (4, 2));
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
