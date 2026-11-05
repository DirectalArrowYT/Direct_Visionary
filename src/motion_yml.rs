//! `motion_list.yml` — the YAML form of `motion_list.bin` (as written by motion_list_rs /
//! yamlist), shipped by modded characters whose motions aren't in the vanilla label set.
//!
//! The binary list only stores hashes, so a custom motion kind like `special_n_custom` shows up
//! as `0x0f…` with no script attached. The YAML carries the names, which is everything needed:
//! each name becomes a label, and when the costume has no `motion_list.bin` the YAML *is* the
//! list. Parsing is deliberately lenient (missing fields default) so a hand-written file works.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hash40::Hash40;
use motion_lib::mlist::{Animation, MList, Motion};
use serde_yaml::Value;

/// File names accepted next to (or instead of) `motion_list.bin`.
pub const FILE_NAMES: [&str; 2] = ["motion_list.yml", "motion_list.yaml"];

/// The YAML list in `motion_dir`, if one is there.
pub fn find(motion_dir: &Path) -> Option<PathBuf> {
    FILE_NAMES
        .iter()
        .map(|name| motion_dir.join(name))
        .find(|path| path.is_file())
}

/// A parsed `motion_list.yml`: the list itself plus hash → name for every name it spells out.
pub struct MotionYml {
    pub list: MList,
    pub labels: HashMap<u64, String>,
}

pub fn load(path: &Path) -> Result<MotionYml, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse(text: &str) -> Result<MotionYml, String> {
    let root: Value = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    let mut labels = HashMap::new();
    let mut list = MList::default();
    if let Some(path) = root.get("motion_path").and_then(|v| hash(v, &mut labels)) {
        list.motion_path = path;
    }
    let entries = root
        .get("list")
        .and_then(Value::as_mapping)
        .ok_or("no `list:` mapping of motion kinds")?;
    for (key, body) in entries {
        let Some(kind) = hash(key, &mut labels) else {
            continue;
        };
        let mut motion = Motion::default();
        if let Some(script) = body.get("game_script").and_then(|v| hash(v, &mut labels)) {
            motion.game_script = script;
        }
        if let Some(flags) = body.get("flags") {
            if let Ok(flags) = serde_yaml::from_value(flags.clone()) {
                motion.flags = flags;
            }
        }
        if let Some(frames) = body.get("blend_frames").and_then(Value::as_u64) {
            motion.blend_frames = frames.min(u8::MAX as u64) as u8;
        }
        for anim in body
            .get("animations")
            .and_then(Value::as_sequence)
            .into_iter()
            .flatten()
        {
            // `- name: x.nuanmb` / `unk: 0`, or just `- x.nuanmb`.
            let (name, unk) = match anim.get("name") {
                Some(name) => (name, anim.get("unk").and_then(Value::as_u64).unwrap_or(0)),
                None => (anim, 0),
            };
            if let Some(name) = hash(name, &mut labels) {
                motion.animations.push(Animation {
                    name,
                    unk: unk.min(u8::MAX as u64) as u8,
                });
            }
        }
        motion.scripts = body
            .get("scripts")
            .and_then(Value::as_sequence)
            .into_iter()
            .flatten()
            .filter_map(|v| hash(v, &mut labels))
            .collect();
        if let Some(extra) = body.get("extra").filter(|v| !v.is_null()) {
            motion.extra = serde_yaml::from_value(extra.clone()).ok();
        }
        list.list.insert(kind, motion);
    }
    Ok(MotionYml { list, labels })
}

/// A hash40 field: `0x…` hex stays a hash, anything else is a name (hashed and remembered).
fn hash(value: &Value, labels: &mut HashMap<u64, String>) -> Option<Hash40> {
    let text = match value {
        Value::String(s) => s.trim(),
        Value::Number(n) => return n.as_u64().map(Hash40),
        _ => return None,
    };
    if text.is_empty() {
        return None;
    }
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).ok().map(Hash40);
    }
    let hash = hash40::hash40(text);
    labels.insert(hash.0, text.to_string());
    Some(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
motion_path: fighter/mario/motion/body/c00
list:
  wait:
    game_script: game_wait
    flags:
      turn: false
      loop: true
      move: false
      fix_trans: false
      fix_rot: false
      fix_scale: false
      unk_40: false
      unk_80: false
      unk_100: false
      unk_200: false
      unk_400: false
      unk_800: false
      unk_1000: false
      unk_2000: false
    blend_frames: 4
    animations:
    - name: a00wait1.nuanmb
      unk: 0
    scripts:
    - effect_wait
    - sound_wait
    extra: null
  special_n_custom:
    animations:
    - d00specialncustom.nuanmb
  0x0123456789:
    game_script: 0x0abcdef012
"#;

    #[test]
    fn parses_names_hex_and_partial_entries() {
        let parsed = parse(SAMPLE).unwrap();
        assert_eq!(parsed.list.list.len(), 3);
        let wait = &parsed.list.list[&hash40::hash40("wait")];
        assert_eq!(wait.game_script, hash40::hash40("game_wait"));
        assert!(wait.flags.r#loop);
        assert_eq!(wait.blend_frames, 4);
        assert_eq!(wait.animations[0].name, hash40::hash40("a00wait1.nuanmb"));
        assert_eq!(wait.scripts.len(), 2);

        let custom = &parsed.list.list[&hash40::hash40("special_n_custom")];
        assert_eq!(
            custom.animations[0].name,
            hash40::hash40("d00specialncustom.nuanmb")
        );
        assert!(parsed.list.list.contains_key(&Hash40(0x0123456789)));

        let custom_hash = hash40::hash40("special_n_custom").0;
        assert_eq!(parsed.labels[&custom_hash], "special_n_custom");
        assert_eq!(
            parsed.labels[&hash40::hash40("effect_wait").0],
            "effect_wait"
        );
        assert!(!parsed.labels.contains_key(&0x0123456789));
    }
}
