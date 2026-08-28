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
    num_scale_keys: Option<usize>,
    scale_keys: Vec<[Option<usize>; 4]>,
    billboard_type: Option<usize>,
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

        let curve = sample_scale(&sim.scale_keys, age, life);
        particles.push(SimParticle {
            offset,
            // Emitter scale multiplies the particle's own curve, which is how the data is
            // laid out: the curve is the shape of the size over life, the emitter scales it.
            size: (curve.x.max(curve.y) * sim.scale.x.max(sim.scale.y)).max(0.01),
            color: [color[0], color[1], color[2], alpha * fade],
            rotation: hashed(seed, id ^ 0x61) * std::f32::consts::TAU,
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
