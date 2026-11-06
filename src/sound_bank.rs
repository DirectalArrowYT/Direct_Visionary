//! Hearing a move's `sound_` script: find each named sound in the game's banks and play it.
//!
//! A fighter's sounds live in `sound/bank/fighter/se_<name>[_cNN].nus3audio`, its voice in
//! `sound/bank/fighter_voice/vc_<name>[_cNN].nus3audio`, and the shared ones in
//! `sound/bank/common/se_common*.nus3audio`. A nus3audio stores every tone's name next to its
//! stream, so a script's label is found by name with no soundlabelinfo lookup. Streams are
//! IDSP (Nintendo DSP-ADPCM in Namco's container); this decodes them and hands the PCM to the
//! default audio device.
//!
//! Layouts follow the `nus3` tool (NUS3BANK-Editor): `nus3/nus3audio.py` and `nus3/dsp.py`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One decoded tone, interleaved f32.
#[derive(Clone)]
pub struct Pcm {
    pub channels: u16,
    pub rate: u32,
    pub samples: Arc<[f32]>,
}

/// A parsed `.nus3audio`: tone names and where each stream sits in the file.
pub struct Nus3Audio {
    pub names: Vec<String>,
    spans: Vec<(usize, usize)>,
    data: Vec<u8>,
}

fn u32_le(d: &[u8], at: usize) -> Option<u32> {
    d.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn u32_be(d: &[u8], at: usize) -> Option<u32> {
    d.get(at..at + 4).map(|b| u32::from_be_bytes(b.try_into().unwrap()))
}

fn i16_be(d: &[u8], at: usize) -> Option<i16> {
    d.get(at..at + 2).map(|b| i16::from_be_bytes(b.try_into().unwrap()))
}

impl Nus3Audio {
    pub fn parse(data: Vec<u8>) -> Result<Self, String> {
        if data.get(..4) != Some(b"NUS3") || data.get(8..16) != Some(b"AUDIINDX") {
            return Err("not a nus3audio file".into());
        }
        let index_size = u32_le(&data, 16).ok_or("truncated")? as usize;
        let count = u32_le(&data, 20).ok_or("truncated")? as usize;
        let mut sections: HashMap<[u8; 4], usize> = HashMap::new();
        let mut pos = 20 + index_size;
        while pos + 8 <= data.len() {
            let magic: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
            let size = u32_le(&data, pos + 4).unwrap() as usize;
            sections.insert(magic, pos + 8);
            pos += 8 + size;
        }
        let nmof = *sections.get(b"NMOF").ok_or("no NMOF section")?;
        let adof = *sections.get(b"ADOF").ok_or("no ADOF section")?;
        let mut names = Vec::with_capacity(count);
        let mut spans = Vec::with_capacity(count);
        for i in 0..count {
            let name_at = u32_le(&data, nmof + 4 * i).ok_or("truncated NMOF")? as usize;
            let end = data
                .get(name_at..)
                .and_then(|rest| rest.iter().position(|b| *b == 0))
                .ok_or("bad tone name")?;
            names.push(String::from_utf8_lossy(&data[name_at..name_at + end]).to_lowercase());
            let off = u32_le(&data, adof + 8 * i).ok_or("truncated ADOF")? as usize;
            let size = u32_le(&data, adof + 8 * i + 4).ok_or("truncated ADOF")? as usize;
            spans.push((off, size));
        }
        Ok(Self { names, spans, data })
    }

    pub fn stream(&self, index: usize) -> Option<&[u8]> {
        let (off, size) = *self.spans.get(index)?;
        self.data.get(off..off + size)
    }
}

/// Decode an IDSP stream (big-endian header, one 0x60 DSP header per channel, data
/// interleaved in `interleave`-byte blocks).
pub fn decode_idsp(data: &[u8]) -> Result<Pcm, String> {
    if data.get(..4) != Some(b"IDSP") {
        return Err(format!(
            "unsupported stream {:?}",
            String::from_utf8_lossy(data.get(..4).unwrap_or_default())
        ));
    }
    let field = |i: usize| u32_be(data, 4 + 4 * i).ok_or("truncated IDSP header");
    let channels = field(1)? as usize;
    let rate = field(2)?;
    let samples = field(3)? as usize;
    let interleave = field(6)? as usize;
    let header_at = field(7)? as usize;
    let header_size = field(8)? as usize;
    let data_at = field(9)? as usize;
    let data_size = field(10)? as usize;
    if channels == 0 || channels > 8 || rate == 0 {
        return Err("bad IDSP header".into());
    }
    let block = if interleave == 0 { data_size } else { interleave };
    let body = data.get(data_at..).ok_or("IDSP data past end")?;
    let mut out = vec![0f32; samples * channels];
    for c in 0..channels {
        let h = header_at + c * header_size;
        let mut coefs = [0i32; 16];
        for (k, coef) in coefs.iter_mut().enumerate() {
            *coef = i16_be(data, h + 0x1C + 2 * k).ok_or("truncated channel header")? as i32;
        }
        let mut hist1 = i16_be(data, h + 0x40).unwrap_or(0) as i32;
        let mut hist2 = i16_be(data, h + 0x42).unwrap_or(0) as i32;
        // The byte at position `p` of this channel's own (de-interleaved) data.
        let byte_at = |p: usize| -> u8 {
            let (blk, within) = if channels == 1 { (0, p) } else { (p / block, p % block) };
            let at = if channels == 1 { within } else { (blk * channels + c) * block + within };
            body.get(at).copied().unwrap_or(0)
        };
        let mut produced = 0;
        let mut frame = 0;
        while produced < samples {
            let ps = byte_at(frame * 8) as i32;
            let (pred, scale) = ((ps >> 4) as usize & 7, 1i32 << (ps & 0xF));
            let (c1, c2) = (coefs[pred * 2], coefs[pred * 2 + 1]);
            for i in 0..14 {
                if produced == samples {
                    break;
                }
                let byte = byte_at(frame * 8 + 1 + i / 2) as i32;
                let mut nib = if i % 2 == 0 { byte >> 4 } else { byte & 0xF };
                if nib >= 8 {
                    nib -= 16;
                }
                let s = (((nib * scale) << 11) + 1024 + c1 * hist1 + c2 * hist2) >> 11;
                let s = s.clamp(-32768, 32767);
                hist2 = hist1;
                hist1 = s;
                out[produced * channels + c] = s as f32 / 32768.0;
                produced += 1;
            }
            frame += 1;
        }
    }
    Ok(Pcm {
        channels: channels as u16,
        rate,
        samples: out.into(),
    })
}

/// The bank a file stem belongs to with any `_cNN` slot suffix removed.
fn bank_base(stem: &str) -> &str {
    let b = stem.as_bytes();
    if b.len() > 4
        && b[b.len() - 4] == b'_'
        && b[b.len() - 3] == b'c'
        && b[b.len() - 2].is_ascii_digit()
        && b[b.len() - 1].is_ascii_digit()
    {
        &stem[..stem.len() - 4]
    } else {
        stem
    }
}

/// Plays script sound labels from the banks of the selected costume.
#[derive(Default)]
pub struct SoundPlayer {
    roots: Vec<PathBuf>,
    slot: Option<String>,
    /// (stem, path) of every bank under the roots, roots in priority order.
    index: Option<Vec<(String, PathBuf)>>,
    banks: HashMap<PathBuf, Option<Nus3Audio>>,
    decoded: HashMap<(PathBuf, usize), Option<Pcm>>,
    sink: Option<rodio::MixerDeviceSink>,
    sink_failed: bool,
    pub volume: f32,
    rng: u32,
}

impl SoundPlayer {
    /// Point at the selected costume: `roots` mod roots first, `slot` like `"c88"`.
    pub fn set_context(&mut self, roots: &[PathBuf], slot: Option<String>) {
        if self.roots.as_slice() != roots || self.slot != slot {
            self.roots = roots.to_vec();
            self.slot = slot;
            self.index = None;
        }
    }

    fn index(&mut self) -> &[(String, PathBuf)] {
        if self.index.is_none() {
            let mut out = Vec::new();
            for root in &self.roots {
                for sub in ["fighter", "fighter_voice", "common"] {
                    let dir = root.join("sound").join("bank").join(sub);
                    let Ok(rd) = std::fs::read_dir(&dir) else {
                        continue;
                    };
                    let mut found: Vec<(String, PathBuf)> = rd
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| {
                            p.extension()
                                .is_some_and(|e| e.eq_ignore_ascii_case("nus3audio"))
                        })
                        .filter_map(|p| {
                            let stem = p.file_stem()?.to_string_lossy().to_lowercase();
                            Some((stem, p))
                        })
                        .collect();
                    found.sort();
                    out.extend(found);
                }
            }
            self.index = Some(out);
        }
        self.index.as_deref().unwrap_or_default()
    }

    fn bank(&mut self, path: &Path) -> Option<&Nus3Audio> {
        if !self.banks.contains_key(path) {
            let parsed = std::fs::read(path)
                .map_err(|e| e.to_string())
                .and_then(Nus3Audio::parse)
                .map_err(|e| eprintln!("[sound] {}: {e}", path.display()))
                .ok();
            self.banks.insert(path.to_path_buf(), parsed);
        }
        self.banks.get(path).and_then(Option::as_ref)
    }

    fn next_random(&mut self, n: usize) -> usize {
        // xorshift; only picks among a random group's variants.
        let mut x = self.rng.max(0x9E37_79B9);
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        x as usize % n.max(1)
    }

    /// The bank file and tone a script label plays.
    pub fn resolve(&mut self, label: &str) -> Option<(PathBuf, usize, String)> {
        let label = label.trim().to_lowercase();
        // `seq_<fighter>_rnd_<x>` sequences pick among the voice clips `vc_<fighter>_<x>NN`.
        let wanted = match label.strip_prefix("seq_") {
            Some(rest) => format!("vc_{}", rest.replacen("rnd_", "", 1)),
            None => label.clone(),
        };
        let slot = self.slot.clone();
        let index = self.index().to_vec();
        // Banks whose name prefixes the label: this slot's own file first, then the base bank,
        // and every shared common bank as the last resort.
        let mut candidates: Vec<PathBuf> = Vec::new();
        for want_slot in [true, false] {
            for (stem, path) in &index {
                let base = bank_base(stem);
                let is_slot = base != stem.as_str();
                let slot_ok = if want_slot {
                    is_slot && slot.as_deref().is_some_and(|s| stem.ends_with(&format!("_{s}")))
                } else {
                    !is_slot
                };
                if slot_ok && wanted.starts_with(&format!("{base}_")) {
                    candidates.push(path.clone());
                }
            }
        }
        for (stem, path) in &index {
            if stem.starts_with("se_common") && !candidates.contains(path) {
                candidates.push(path.clone());
            }
        }
        for path in candidates {
            let Some(bank) = self.bank(&path) else {
                continue;
            };
            if let Some(i) = bank.names.iter().position(|n| *n == wanted) {
                let name = bank.names[i].clone();
                return Some((path, i, name));
            }
            // A random group (`vc_pit_attack` → `vc_pit_attack01..03`).
            let group: Vec<usize> = bank
                .names
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    n.strip_prefix(wanted.as_str()).is_some_and(|rest| {
                        !rest.is_empty()
                            && rest.trim_start_matches('_').chars().all(|c| c.is_ascii_digit())
                    })
                })
                .map(|(i, _)| i)
                .collect();
            if !group.is_empty() {
                let names: Vec<String> = group
                    .iter()
                    .map(|i| bank.names[*i].clone())
                    .collect();
                let pick = self.next_random(group.len());
                return Some((path, group[pick], names[pick].clone()));
            }
        }
        None
    }

    fn pcm(&mut self, path: &Path, tone: usize) -> Option<Pcm> {
        let key = (path.to_path_buf(), tone);
        if !self.decoded.contains_key(&key) {
            let pcm = self
                .bank(path)
                .and_then(|b| b.stream(tone))
                .map(decode_idsp)
                .and_then(|r| r.map_err(|e| eprintln!("[sound] {}#{tone}: {e}", path.display())).ok());
            self.decoded.insert(key.clone(), pcm);
        }
        self.decoded.get(&key).cloned().flatten()
    }

    /// Play a script label. Returns the tone that played, or why nothing did.
    pub fn play(&mut self, label: &str) -> Result<String, String> {
        let (path, tone, name) = self
            .resolve(label)
            .ok_or_else(|| format!("{label}: not in any bank"))?;
        let pcm = self
            .pcm(&path, tone)
            .ok_or_else(|| format!("{name}: could not decode"))?;
        if self.sink.is_none() && !self.sink_failed {
            match rodio::DeviceSinkBuilder::open_default_sink() {
                Ok(mut sink) => {
                    sink.log_on_drop(false);
                    self.sink = Some(sink);
                }
                Err(e) => {
                    self.sink_failed = true;
                    eprintln!("[sound] no audio device: {e}");
                }
            }
        }
        let sink = self.sink.as_ref().ok_or("no audio output device")?;
        let (Some(channels), Some(rate)) = (
            std::num::NonZero::new(pcm.channels),
            std::num::NonZero::new(pcm.rate),
        ) else {
            return Err(format!("{name}: empty stream"));
        };
        let volume = if self.volume > 0.0 { self.volume } else { 1.0 };
        let samples: Vec<f32> = pcm.samples.iter().map(|s| s * volume).collect();
        sink.mixer()
            .add(rodio::buffer::SamplesBuffer::new(channels, rate, samples));
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_suffix() {
        assert_eq!(bank_base("se_pit_c88"), "se_pit");
        assert_eq!(bank_base("se_pit"), "se_pit");
        assert_eq!(bank_base("se_common_append_01"), "se_common_append_01");
    }

    /// `VISIONARY_SOUND_ROOT=<dump root> cargo test decode_dump_bank -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn decode_dump_bank() {
        let root = PathBuf::from(std::env::var("VISIONARY_SOUND_ROOT").unwrap());
        let mut player = SoundPlayer::default();
        player.set_context(&[root], None);
        for label in ["se_pit_attack100", "se_common_swing_05", "seq_pit_rnd_attack", "vc_pit_attack01"] {
            match player.resolve(label) {
                Some((path, tone, name)) => {
                    let pcm = player.pcm(&path, tone).expect("decodes");
                    let peak = pcm.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
                    println!(
                        "{label} -> {name} ({}) {} ch {} Hz {} samples peak {peak:.3}",
                        path.file_name().unwrap().to_string_lossy(),
                        pcm.channels,
                        pcm.rate,
                        pcm.samples.len()
                    );
                    assert!(peak > 0.01, "{label} decoded silent");
                }
                None => println!("{label} -> unresolved"),
            }
        }
    }
}
