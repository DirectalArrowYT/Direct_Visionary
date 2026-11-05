//! Recover names for motion-list hashes that no label file knows — a modded character's own
//! motions in a plain `motion_list.bin`.
//!
//! A hash40 is `(length << 32) | crc32(name)`, so a hash pins the name's exact length and a
//! candidate can be checked in nanoseconds. The candidates come from the motion's animation
//! file, whose real filename is on disk: `d00specialncustom.nuanmb` is motion kind
//! `special_n_custom` with its underscores removed and a slot prefix added, and its ACMD scripts
//! are `game_specialncustom`, `effect_specialncustom`, … The length tells how many underscores
//! are missing, and every placement of them is tried.
//!
//! When even that fails, the animation's prefix letter still says which family the move is in
//! (`c03…` smash attacks, `d0x…` specials), which is enough to file it in the right section.

use crate::move_kinds::MoveGroup;

/// Script prefixes a motion's ACMD scripts use (`game_` + the animation stem).
const SCRIPT_PREFIXES: [&str; 4] = ["game_", "effect_", "sound_", "expression_"];

/// Most underscores ever inserted; vanilla motion kinds have at most five.
const MAX_UNDERSCORES: usize = 6;

/// `d00specialncustom.nuanmb` → `specialncustom` (lowercase, prefix and extension removed).
pub fn anim_stem(file_name: &str) -> Option<String> {
    let lower = file_name.to_ascii_lowercase();
    let stem = lower.strip_suffix(".nuanmb").unwrap_or(&lower);
    let bytes = stem.as_bytes();
    let rest = if bytes.len() > 3
        && bytes[0].is_ascii_lowercase()
        && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_digit()
    {
        &stem[3..]
    } else {
        stem
    };
    (!rest.is_empty()).then(|| rest.to_string())
}

/// The name of length/crc `target` that is `stem` with underscores inserted, if there is one.
pub fn with_underscores(stem: &str, target: u64) -> Option<String> {
    let want = (target >> 32) as usize;
    let n = stem.len();
    if want < n || !stem.is_ascii() {
        return None;
    }
    let k = want - n;
    if k == 0 {
        return (hash40::hash40(stem).0 == target).then(|| stem.to_string());
    }
    if k > MAX_UNDERSCORES || n < 2 || k > n - 1 {
        return None;
    }
    // Choose k of the n-1 gaps between characters, in increasing order.
    let mut gaps: Vec<usize> = (1..=k).collect();
    let mut buf = String::with_capacity(want);
    loop {
        buf.clear();
        let mut last = 0;
        for &gap in &gaps {
            buf.push_str(&stem[last..gap]);
            buf.push('_');
            last = gap;
        }
        buf.push_str(&stem[last..]);
        if hash40::hash40(&buf).0 == target {
            return Some(buf);
        }
        // Next combination.
        let mut i = k;
        loop {
            if i == 0 {
                return None;
            }
            i -= 1;
            if gaps[i] < n - 1 - (k - 1 - i) {
                gaps[i] += 1;
                for j in i + 1..k {
                    gaps[j] = gaps[j - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Stem variants worth trying: the stem itself and the stem without a trailing number
/// (`wait1` animates motion `wait`).
fn variants(stems: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for stem in stems {
        for v in [
            stem.clone(),
            stem.trim_end_matches(|c: char| c.is_ascii_digit()).to_string(),
            // Smash-attack charge animations play the `*_hold` motions.
            stem.replace("charge", "hold"),
        ] {
            if !v.is_empty() && !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

/// Words a motion kind adds to the animation it shares with another motion
/// (`attack_hi4_hold` plays `attackhi4charge`'s sibling, `fall_aerial` plays `fall`).
const EXTRA_WORDS: [&str; 34] = [
    "hold", "start", "end", "loop", "mini", "aerial", "air", "landing", "wait", "attack", "f",
    "b", "l", "r", "s", "hi", "lw", "n", "fall", "turn", "hit", "charge", "max", "2", "1", "3",
    "jump", "shoot", "cancel", "walk", "run", "squat", "dash", "special",
];

/// A motion kind's name from its animations' stems.
pub fn guess_kind(stems: &[String], target: u64) -> Option<String> {
    let variants = variants(stems);
    if let Some(name) = variants.iter().find_map(|stem| with_underscores(stem, target)) {
        return Some(name);
    }
    // One word added before or after the stem; the length check rejects most at once.
    let want = (target >> 32) as usize;
    for stem in &variants {
        for word in EXTRA_WORDS {
            if stem.len() + word.len() > want {
                continue;
            }
            for joined in [format!("{stem}{word}"), format!("{word}{stem}")] {
                if let Some(name) = with_underscores(&joined, target) {
                    return Some(name);
                }
            }
        }
    }
    None
}

/// A script name (`game_…`, `effect_…`) from the motion's stems and, when known, its kind.
pub fn guess_script(stems: &[String], kind: Option<&str>, target: u64) -> Option<String> {
    let mut bases = variants(stems);
    if let Some(kind) = kind {
        bases.push(kind.to_string());
        bases.push(kind.replace('_', ""));
    }
    for base in &bases {
        for prefix in SCRIPT_PREFIXES {
            let name = format!("{prefix}{base}");
            if hash40::hash40(&name).0 == target {
                return Some(name);
            }
        }
    }
    None
}

/// The family an unnamed motion belongs to, from its animation file name. Vanilla files are
/// consistent about it: `c00` jabs and dash attack, `c01`/`c02` tilts, `c03`/`c04` smashes,
/// `c05` aerials, `d` specials, `e` grabs and throws, and so on.
pub fn group_from_anim(file_name: &str) -> Option<MoveGroup> {
    let lower = file_name.to_ascii_lowercase();
    let stem = anim_stem(&lower)?;
    if stem.starts_with("final") {
        return Some(MoveGroup::FinalSmash);
    }
    if stem.starts_with("attackdash") {
        return Some(MoveGroup::DashAttack);
    }
    if stem.starts_with("landing") {
        return Some(MoveGroup::JumpLand);
    }
    let prefix = lower.get(..3)?;
    Some(match prefix {
        "c00" => MoveGroup::Jab,
        "c01" | "c02" => MoveGroup::Tilt,
        "c03" | "c04" => MoveGroup::Smash,
        "c05" => MoveGroup::Aerial,
        _ => match prefix.as_bytes()[0] {
            b'c' => MoveGroup::Tilt,
            b'd' => MoveGroup::Special,
            b'e' => MoveGroup::Grab,
            b'b' => MoveGroup::Defense,
            b'f' => MoveGroup::Damage,
            b'h' => MoveGroup::Item,
            b'j' => MoveGroup::Presentation,
            b'g' | b'i' | b'k' => MoveGroup::Situational,
            _ => return None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems() {
        assert_eq!(anim_stem("d00specialncustom.nuanmb").unwrap(), "specialncustom");
        assert_eq!(anim_stem("C03AttackS4S.nuanmb").unwrap(), "attacks4s");
        assert_eq!(anim_stem("custom.nuanmb").unwrap(), "custom");
    }

    #[test]
    fn recovers_kinds_and_scripts() {
        for (file, kind) in [
            ("d00specialncustom.nuanmb", "special_n_custom"),
            ("c00attack11.nuanmb", "attack_11"),
            ("c03attacks4s.nuanmb", "attack_s4_s"),
            ("d02specialairhifall.nuanmb", "special_air_hi_fall"),
            ("a00wait1.nuanmb", "wait"),
        ] {
            let stems = vec![anim_stem(file).unwrap()];
            assert_eq!(
                guess_kind(&stems, hash40::hash40(kind).0).as_deref(),
                Some(kind),
                "{file}"
            );
        }
        let stems = vec![anim_stem("d00specialncustom.nuanmb").unwrap()];
        let script = hash40::hash40("effect_specialncustom").0;
        assert_eq!(
            guess_script(&stems, None, script).as_deref(),
            Some("effect_specialncustom")
        );
        assert_eq!(guess_kind(&stems, hash40::hash40("something_else").0), None);
    }

    #[test]
    fn groups_by_prefix() {
        assert_eq!(group_from_anim("c03attacks4xx.nuanmb"), Some(MoveGroup::Smash));
        assert_eq!(group_from_anim("d04mystery.nuanmb"), Some(MoveGroup::Special));
        assert_eq!(group_from_anim("c00attackdashx.nuanmb"), Some(MoveGroup::DashAttack));
        assert_eq!(group_from_anim("plain.nuanmb"), None);
    }
}

#[cfg(test)]
mod corpus {
    use super::*;

    /// Name recovery over a real dump with no labels at all:
    /// `VISIONARY_GUESS_DIR=<…/fighter/mario/motion/body/c00> cargo test guess_corpus -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn guess_corpus() {
        let dir = std::path::PathBuf::from(std::env::var("VISIONARY_GUESS_DIR").unwrap());
        let list = motion_lib::open(dir.join("motion_list.bin")).unwrap();
        let mut files = std::collections::HashMap::new();
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            files.insert(hash40::hash40(&name).0, name);
        }
        let (mut with_anim, mut kinds, mut scripts, mut script_total, mut grouped) = (0, 0, 0, 0, 0);
        for (kind, motion) in &list.list {
            let names: Vec<&String> =
                motion.animations.iter().filter_map(|a| files.get(&a.name.0)).collect();
            let stems: Vec<String> = names.iter().filter_map(|n| anim_stem(n)).collect();
            if stems.is_empty() {
                continue;
            }
            with_anim += 1;
            let k = guess_kind(&stems, kind.0);
            if k.is_some() {
                kinds += 1;
            } else {
                println!("miss {:#x} {:?}", kind.0, stems);
            }
            if group_from_anim(names[0]).is_some() {
                grouped += 1;
            }
            for s in std::iter::once(&motion.game_script).chain(&motion.scripts) {
                if s.0 == 0 {
                    continue;
                }
                script_total += 1;
                if guess_script(&stems, k.as_deref(), s.0).is_some() {
                    scripts += 1;
                }
            }
        }
        println!(
            "motions {} with anim {with_anim}: kinds {kinds}, grouped {grouped}, scripts {scripts}/{script_total}",
            list.list.len()
        );
    }
}
