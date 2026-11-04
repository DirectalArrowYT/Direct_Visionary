//! Effect draws captured from the game running in the SSBU recompilation (tools/recomp_eftcap).
//!
//! Every record is one draw exactly as the game issued it: the draw shape, the blend / depth state, every constant
//! buffer of both stages in full, the vertex streams, the index data, and which texture units were bound. That is
//! ground truth for how an effect is drawn, so `eff_gpu` can replay a record through the game's own shaders and the
//! result can be held against the game's picture, and every input Visionary generates itself can be checked against
//! the record's.
//!
//! The file is a run of chunks, `[tag u32][size u32][payload]`, little endian; a draw is `DRAW`, its data chunks, `END `.
//! `PROG` chunks (program code) are written once per run, with the first draw that uses the program, so a per-effect
//! file cut out of a run (split_cycle.py) is read together with the run's collected programs (collect_progs.py).

use std::collections::{BTreeMap, HashMap};

/// One vertex attribute: generic input `location` reads `format` (an NVN vertex format) at `offset` in `stream`.
#[derive(Clone, Copy, Debug)]
pub struct Attr {
    pub location: u32,
    pub format: u32,
    pub stream: u32,
    pub offset: u32,
}

/// A vertex stream: `divisor` 0 advances per vertex, 1 per instance.
#[derive(Clone, Debug)]
pub struct Stream {
    pub stride: u32,
    pub divisor: u32,
    pub bytes: Vec<u8>,
}

/// A texture unit the draw had bound: what the game's texture was (format is an NVN texture format), and the sampler.
#[derive(Clone, Copy, Debug)]
pub struct TexBinding {
    pub stage: u32,
    pub unit: u32,
    pub format: i32,
    pub width: i32,
    pub height: i32,
    pub levels: i32,
    pub hash: u64,
    pub min_filter: u8,
    pub mag_filter: u8,
    pub wrap_s: u8,
    pub wrap_t: u8,
    pub lod_min: f32,
    pub lod_max: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Draw {
    pub frame: i64,
    pub draw: u32,
    /// 0 = arrays, 1 = indexed.
    pub kind: u32,
    /// NVN primitive: 4 triangles, 5 triangle strip.
    pub prim: u32,
    pub count: u32,
    pub instances: u32,
    pub first: i32,
    pub base_vertex: i32,
    pub base_instance: u32,
    /// log2 of the index size.
    pub index_type: u32,
    /// Program serials, keys into the run's programs.
    pub vs: i32,
    pub fs: i32,
    pub blend: bool,
    /// OpenGL blend factors and equations, as the recomp's OpenGL back end set them.
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub eq_rgb: u32,
    pub src_a: u32,
    pub dst_a: u32,
    pub eq_a: u32,
    pub depth_test: bool,
    pub depth_func: u32,
    pub depth_mask: bool,
    pub cull: bool,
    /// The shader's alpha test: 0 or >= 8 always, 1 never, then <, ==, <=, >, !=, >= against `alpha_ref`.
    pub alpha_func: i32,
    pub alpha_ref: f32,
    pub fbo: u32,
    pub fbo_w: i32,
    pub fbo_h: i32,
    /// Constant buffers by (stage, NVN slot); the shaders read slot n as c(n + 3). Stage 0 vertex, 1 fragment.
    pub ubo: BTreeMap<(u32, u32), Vec<u8>>,
    pub attrs: Vec<Attr>,
    pub streams: BTreeMap<u32, Stream>,
    pub index: Option<Vec<u8>>,
    pub textures: Vec<TexBinding>,
}

/// The effect library's emitter block (0x750 bytes): the shaders' c9, NVN slot 6.
pub const EMITTER_BLOCK_SLOT: u32 = 6;
const EMITTER_BLOCK_LEN: usize = 0x750;

impl Draw {
    /// The emitter this draw belongs to: the C string at +0x10 of the emitter block.
    pub fn emitter_name(&self) -> Option<String> {
        let block = self.ubo.get(&(0, EMITTER_BLOCK_SLOT))?;
        if block.len() < EMITTER_BLOCK_LEN {
            return None;
        }
        let raw = &block[0x10..0x50];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        Some(String::from_utf8_lossy(&raw[..end]).into_owned())
    }

    /// The constant buffer the shader reads as `c<n>` for `stage` (0 vertex, 1 fragment), where the capture has it.
    pub fn cbuf(&self, stage: u32, n: u32) -> Option<&[u8]> {
        n.checked_sub(3).and_then(|slot| self.ubo.get(&(stage, slot))).map(|b| b.as_slice())
    }
}

/// A program as the game bound it: the NVN control words and the code from the program header on. The capture
/// copies `ctl[2] + 0x50` bytes, but `ctl[2]` already counts the 0x50-byte header: [`Program::bytes`] is the program.
#[derive(Clone, Debug)]
pub struct Program {
    pub serial: i32,
    pub stage: u32,
    pub ctl: [u32; 8],
    pub code: Vec<u8>,
}

impl Program {
    pub fn bytes(&self) -> &[u8] {
        &self.code[..(self.ctl[2] as usize).min(self.code.len())]
    }
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn i32_at(b: &[u8], at: usize) -> i32 {
    u32_at(b, at) as i32
}

fn chunks(bytes: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at + 8 > bytes.len() {
            return None;
        }
        let tag = &bytes[at..at + 4];
        let n = u32_at(bytes, at + 4) as usize;
        let body = bytes.get(at + 8..at + 8 + n)?;
        at += 8 + n;
        Some((tag, body))
    })
}

/// A framebuffer grab: a target's colour (RGBA half floats) or depth (f32) just before a frame's first effect draw
/// into it (`phase` 0) or after its last (`phase` 1). Row 0 is the top of the picture.
#[derive(Clone, Debug)]
pub struct Shot {
    pub frame: i64,
    pub fbo: u32,
    pub phase: u32,
    /// 0 colour, 1 depth.
    pub kind: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// One complete record of a capture.
pub enum Record {
    Draw(Draw),
    Program(Program),
    Shot(Shot),
}

/// Assembles records from chunks: a draw is its `DRAW` chunk, its data chunks and `END `.
#[derive(Default)]
struct Parser {
    cur: Option<Draw>,
}

impl Parser {
    fn push(&mut self, tag: &[u8], body: &[u8]) -> Option<Record> {
        match tag {
            b"DRAW" => {
                let u = |i: usize| u32_at(body, 8 + 4 * i);
                self.cur = Some(Draw {
                    frame: i64::from_le_bytes(body[0..8].try_into().unwrap()),
                    draw: u(0),
                    kind: u(1),
                    prim: u(2),
                    count: u(3),
                    instances: u(4),
                    first: u(5) as i32,
                    base_vertex: u(6) as i32,
                    base_instance: u(7),
                    index_type: u(8),
                    vs: u(9) as i32,
                    fs: u(10) as i32,
                    blend: u(11) != 0,
                    src_rgb: u(12),
                    dst_rgb: u(13),
                    eq_rgb: u(14),
                    src_a: u(15),
                    dst_a: u(16),
                    eq_a: u(17),
                    depth_test: u(18) != 0,
                    depth_func: u(19),
                    depth_mask: u(20) != 0,
                    cull: u(21) != 0,
                    alpha_func: u(22) as i32,
                    alpha_ref: f32::from_bits(u(23)),
                    fbo: u(24),
                    fbo_w: u(25) as i32,
                    fbo_h: u(26) as i32,
                    ..Default::default()
                });
                None
            }
            b"END " => self.cur.take().map(Record::Draw),
            b"PROG" if body.len() >= 40 => {
                let mut ctl = [0u32; 8];
                for (i, c) in ctl.iter_mut().enumerate() {
                    *c = u32_at(body, 8 + 4 * i);
                }
                Some(Record::Program(Program { serial: i32_at(body, 0), stage: u32_at(body, 4), ctl, code: body[40..].to_vec() }))
            }
            b"SHOT" if body.len() >= 32 => Some(Record::Shot(Shot {
                frame: i64::from_le_bytes(body[0..8].try_into().unwrap()),
                fbo: u32_at(body, 8),
                phase: u32_at(body, 12),
                kind: u32_at(body, 16),
                width: u32_at(body, 20),
                height: u32_at(body, 24),
                pixels: body[32..].to_vec(),
            })),
            _ => {
                let d = self.cur.as_mut()?;
                match tag {
                    b"UBO " => {
                        d.ubo.insert((u32_at(body, 0), u32_at(body, 4)), body[8..].to_vec());
                    }
                    b"ATTR" => {
                        let raw = u32_at(body, 4);
                        d.attrs.push(Attr { location: u32_at(body, 0), format: raw & 0xff, stream: (raw >> 8) & 31, offset: raw >> 14 });
                    }
                    b"VBUF" => {
                        d.streams.insert(u32_at(body, 0), Stream { stride: u32_at(body, 4), divisor: u32_at(body, 8), bytes: body[12..].to_vec() });
                    }
                    b"IBUF" => d.index = Some(body.to_vec()),
                    b"TEX " => d.textures.push(TexBinding {
                        stage: u32_at(body, 0),
                        unit: u32_at(body, 4),
                        format: i32_at(body, 20),
                        width: i32_at(body, 24),
                        height: i32_at(body, 28),
                        levels: i32_at(body, 36),
                        hash: u64::from_le_bytes(body[40..48].try_into().unwrap()),
                        min_filter: body[56],
                        mag_filter: body[57],
                        wrap_s: body[58],
                        wrap_t: body[59],
                        lod_min: f32::from_bits(u32_at(body, 60)),
                        lod_max: f32::from_bits(u32_at(body, 64)),
                    }),
                    _ => {}
                }
                None
            }
        }
    }
}

fn records(bytes: &[u8]) -> impl Iterator<Item = Record> + '_ {
    let mut parser = Parser::default();
    chunks(bytes).filter_map(move |(tag, body)| parser.push(tag, body))
}

/// Every draw record of a capture file, in order.
pub fn read_draws(bytes: &[u8]) -> Vec<Draw> {
    records(bytes).filter_map(|r| if let Record::Draw(d) = r { Some(d) } else { None }).collect()
}

/// Every program of a capture (or of collect_progs.py's file), by (serial, stage).
pub fn read_programs(bytes: &[u8]) -> HashMap<(i32, u32), Program> {
    records(bytes)
        .filter_map(|r| if let Record::Program(p) = r { Some(((p.serial, p.stage), p)) } else { None })
        .collect()
}

/// Every framebuffer grab of a capture, in order. A target can have several "after" grabs in one frame (its effect
/// draws were interleaved with other draws); the last one is the finished effect pass.
pub fn read_shots(bytes: &[u8]) -> Vec<Shot> {
    records(bytes).filter_map(|r| if let Record::Shot(s) = r { Some(s) } else { None }).collect()
}

/// Every record of a capture file, read as it streams: a run with framebuffer grabs is tens of gigabytes. A file cut
/// short (a run still writing it) ends at its last whole chunk.
pub fn stream(path: &std::path::Path, mut each: impl FnMut(Record)) -> std::io::Result<()> {
    use std::io::Read;
    let mut file = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
    let mut parser = Parser::default();
    let mut head = [0u8; 8];
    let mut body = Vec::new();
    loop {
        if file.read_exact(&mut head).is_err() {
            return Ok(());
        }
        let n = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
        body.resize(n, 0);
        if file.read_exact(&mut body).is_err() {
            return Ok(());
        }
        if let Some(r) = parser.push(&head[..4], &body) {
            each(r);
        }
    }
}
