//! Weapons (articles) a move spawns, previewed next to the fighter.
//!
//! A weapon has its own model, motion list and ACMD (agent `<fighter>_<weapon>`), and the
//! game places it with status code Visionary does not run. So the preview works from what the
//! source does say: `ArticleModule::generate_article` lines in the move's scripts give the
//! weapon and its frame, and the files that install the weapon hint at which body bone it is
//! constrained to. Everything else is a per-spawn setting the user can adjust; settings are
//! preview-only and stored in the app's own storage, never in the mod.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use ssbh_data::{anim_data::AnimData, skel_data::SkelData};

/// One weapon instance a move spawns, with how the preview should place it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ArticleSpawn {
    /// Weapon folder name (`model/<article>`, `motion/<article>`).
    pub article: String,
    /// Body script frame it is generated on.
    pub frame: u32,
    /// The weapon's motion to play; empty draws the rest pose.
    pub motion: String,
    /// Body bone the weapon's root follows. Empty = stays where it spawned.
    pub attach_bone: String,
    /// Body bone a free weapon launches from (where it is on the spawn frame). Empty = `top`.
    pub start_bone: String,
    pub offset: [f32; 3],
    /// Degrees, X Y Z like the ACMD spawn macros.
    pub rotation: [f32; 3],
    /// Units per frame, for a weapon that is not attached.
    pub velocity: [f32; 3],
    /// Frames it lives. 0 = until its motion ends (a looping motion: until the move ends).
    pub life: u32,
    pub enabled: bool,
    /// Found in the move's script rather than added by hand.
    pub from_script: bool,
}

impl Default for ArticleSpawn {
    fn default() -> Self {
        Self {
            article: String::new(),
            frame: 1,
            motion: String::new(),
            attach_bone: String::new(),
            start_bone: String::new(),
            offset: [0.0; 3],
            rotation: [0.0; 3],
            velocity: [0.0; 3],
            life: 0,
            enabled: true,
            from_script: false,
        }
    }
}

/// An effect the mod's own code spawns (`EffectModule::req*`), which no effect script shows.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CodeEffect {
    pub name: String,
    /// Body script frame it is spawned on.
    pub frame: u32,
    /// Bone it is placed against; `top` for a world position at the fighter's feet.
    pub bone: String,
    pub offset: [f32; 3],
    /// Degrees, X Y Z.
    pub rotation: [f32; 3],
    pub scale: f32,
    /// Moves with its bone rather than staying where it was spawned.
    pub follows: bool,
    /// Frames it lives. 0 = until the move ends.
    pub life: u32,
    /// Copies around a horizontal circle (a ring of flames); 1 = a single effect.
    pub ring_count: u32,
    pub ring_radius: f32,
    pub enabled: bool,
    /// Where in the source it was found (`module::function`), empty when added by hand.
    pub source: String,
}

impl Default for CodeEffect {
    fn default() -> Self {
        Self {
            name: String::new(),
            frame: 1,
            bone: "top".into(),
            offset: [0.0; 3],
            rotation: [0.0; 3],
            scale: 1.0,
            follows: false,
            life: 0,
            ring_count: 1,
            ring_radius: 0.0,
            enabled: true,
            source: String::new(),
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct FighterPrefs {
    /// Move name → its spawns, once the user has changed anything for that move.
    moves: BTreeMap<String, Vec<ArticleSpawn>>,
    /// Move name → its code-spawned effects, once the user has changed any.
    effects: BTreeMap<String, Vec<CodeEffect>>,
}

/// A motion in a weapon's motion list.
pub struct ArticleMotion {
    pub hash: u64,
    pub name: String,
    pub anim_path: Option<PathBuf>,
    pub looping: bool,
    /// game_script followed by the other scripts.
    pub script_hashes: Vec<u64>,
}

/// Everything loaded for one weapon.
pub struct ArticleAssets {
    pub model_dir: Option<PathBuf>,
    pub skel: Option<Arc<SkelData>>,
    pub motions: Vec<ArticleMotion>,
    /// Script name → source text, from the linked project.
    pub scripts: HashMap<String, String>,
    /// Body bone names quoted in the files that install the weapon, first appearance first.
    pub bone_literals: Vec<String>,
    /// `const <NAME>: Vector3f = Vector3f{x, y, z}` with ROT in the name, in the same files.
    pub rot_offsets: Vec<(String, [f32; 3])>,
    /// The installer constrains the weapon to its owner (`set_model_constraint_*`, `LinkModule::link`).
    pub constrained: bool,
    /// Launch numbers from the fighter's `vl.prc` (`param_<weapon>`), for weapons the status
    /// code moves.
    pub params: Option<WeaponParams>,
}

/// What a weapon's `param_<weapon>` struct says about how it flies.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeaponParams {
    /// Units per frame along its launch angle.
    pub speed: Option<f32>,
    /// Degrees up from straight ahead.
    pub angle: f32,
    pub life: Option<u32>,
    /// Spawn point ahead of / above the fighter (`offset_x`, `offset_y`).
    pub offset: Option<[f32; 2]>,
}

impl WeaponParams {
    /// Read `param_<weapon>` from a fighter's `vl.prc`. Matches the struct name with or without
    /// underscores; a list of structs reads its first entry.
    pub fn read(vl_prc: &Path, weapon: &str) -> Option<Self> {
        let root = prc::open(vl_prc).ok()?;
        let want = norm(weapon);
        let names = [format!("param_{weapon}"), format!("param_{want}")];
        let hashes: Vec<u64> = names.iter().map(|n| hash40::hash40(n).0).collect();
        let st = root.0.iter().find_map(|(h, v)| {
            if !hashes.contains(&h.0) {
                return None;
            }
            match v {
                prc::ParamKind::Struct(s) => Some(s.clone()),
                prc::ParamKind::List(l) => match l.0.first() {
                    Some(prc::ParamKind::Struct(s)) => Some(s.clone()),
                    _ => None,
                },
                _ => None,
            }
        })?;
        let num = |key: &str| -> Option<f32> {
            let want = hash40::hash40(key).0;
            st.0.iter().find(|(h, _)| h.0 == want).and_then(|(_, v)| match v {
                prc::ParamKind::Float(f) => Some(*f),
                prc::ParamKind::I32(i) => Some(*i as f32),
                prc::ParamKind::U32(i) => Some(*i as f32),
                prc::ParamKind::I16(i) => Some(*i as f32),
                prc::ParamKind::U16(i) => Some(*i as f32),
                prc::ParamKind::I8(i) => Some(*i as f32),
                prc::ParamKind::U8(i) => Some(*i as f32),
                _ => None,
            })
        };
        let first = |keys: &[&str]| keys.iter().find_map(|k| num(k));
        // Uncharged / full-health spellings first; a charge range starts at its min.
        let speed = first(&["speed", "speed_x", "max_speed_x", "min_speed", "speed_min_x", "start_speed", "init_speed"]);
        let life = first(&["life", "max_life", "min_life", "life_max"]).map(|v| v.max(0.0) as u32);
        let offset = match (num("offset_x"), num("offset_y")) {
            (None, None) => None,
            (x, y) => Some([x.unwrap_or(0.0), y.unwrap_or(0.0)]),
        };
        let params = Self {
            speed,
            angle: num("angle").unwrap_or(0.0),
            life,
            offset,
        };
        (params != Self::default()).then_some(params)
    }

    /// Velocity in the preview's space: +Z ahead, +Y up.
    pub fn velocity(&self) -> Option<[f32; 3]> {
        let speed = self.speed?;
        let a = self.angle.to_radians();
        Some([0.0, speed * a.sin(), speed * a.cos()])
    }
}

impl ArticleAssets {
    pub fn motion(&self, name: &str) -> Option<&ArticleMotion> {
        let want = hash40::hash40(&name.to_ascii_lowercase()).0;
        self.motions.iter().find(|m| m.hash == want || m.name == name)
    }
}

/// Hitboxes and effects a weapon motion's scripts make.
#[derive(Default)]
pub struct ParsedScripts {
    pub hitboxes: Vec<crate::data::Hitbox>,
    pub effects: Vec<crate::data::EffectCall>,
}

/// Where the selected fighter's parts live.
#[derive(Clone)]
pub struct ArticleContext {
    pub fighter: String,
    /// `fighter/<name>/model`
    pub model_root: PathBuf,
    /// `fighter/<name>/motion`
    pub motion_root: PathBuf,
    pub slot: u8,
}

/// A spawn that is live on the current frame, resolved for drawing.
pub struct ArticleInstance {
    /// Index into [`ArticlePreview::spawns`].
    pub index: usize,
    pub article: String,
    pub model_dir: Option<PathBuf>,
    pub skel: Option<Arc<SkelData>>,
    pub anim: Option<Arc<AnimData>>,
    /// Zero-based frame into the weapon's animation.
    pub anim_frame: f32,
    /// One-based frame of the weapon's own scripts.
    pub script_frame: u32,
    pub root: glam::Mat4,
    /// The weapon's bones in world space, keyed like the body's (both spellings).
    pub bones: HashMap<String, glam::Mat4>,
    pub scripts: Arc<ParsedScripts>,
}

pub type BodyBones<'a> = &'a dyn Fn(f32) -> HashMap<String, glam::Mat4>;

#[derive(Default)]
pub struct ArticlePreview {
    pub ctx: Option<ArticleContext>,
    prefs: FighterPrefs,
    pub move_name: String,
    /// What the move's script spawns, with guessed settings.
    pub detected: Vec<ArticleSpawn>,
    /// `generate_article` calls whose weapon could not be named: (frame, argument text).
    pub unresolved: Vec<(u32, String)>,
    /// Weapon folders the fighter ships.
    pub available: Vec<String>,
    /// Body skeleton names, lowercase.
    body_bones: HashSet<String>,
    assets: HashMap<String, Arc<ArticleAssets>>,
    anims: HashMap<PathBuf, Option<Arc<AnimData>>>,
    parsed: HashMap<(String, u64), Arc<ParsedScripts>>,
    pub hide: bool,
    /// Code-spawned effects found by following the move's calls into the source.
    pub detected_effects: Vec<CodeEffect>,
}

fn prefs_path(fighter: &str) -> PathBuf {
    crate::scratch_dirs::app_storage_root()
        .join("article-preview")
        .join(format!("{fighter}.json"))
}

fn norm(name: &str) -> String {
    name.to_ascii_lowercase().replace('_', "")
}

impl ArticlePreview {
    /// Point the preview at a newly selected fighter (or none).
    pub fn set_fighter(&mut self, ctx: Option<ArticleContext>, body_bone_names: &[String]) {
        self.assets.clear();
        self.anims.clear();
        self.parsed.clear();
        self.detected.clear();
        self.unresolved.clear();
        self.move_name.clear();
        self.body_bones = body_bone_names.iter().map(|b| b.to_ascii_lowercase()).collect();
        self.available.clear();
        self.prefs = FighterPrefs::default();
        if let Some(ctx) = &ctx {
            self.prefs = std::fs::read_to_string(prefs_path(&ctx.fighter))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .unwrap_or_default();
            let mut names: Vec<String> = Vec::new();
            for root in [&ctx.model_root, &ctx.motion_root] {
                if let Ok(entries) = std::fs::read_dir(root) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if entry.path().is_dir() && name != "body" && !names.contains(&name) {
                            names.push(name);
                        }
                    }
                }
            }
            names.sort();
            self.available = names;
        }
        self.ctx = ctx;
    }

    /// Read a newly opened move's spawns from its script text.
    pub fn set_move(
        &mut self,
        move_name: &str,
        body: &str,
        index: Option<&crate::acmd_src::SourceIndex>,
        labels: &HashMap<u64, String>,
    ) {
        self.move_name = move_name.to_string();
        self.detected.clear();
        self.unresolved.clear();
        self.detected_effects.clear();
        if self.ctx.is_none() {
            return;
        }
        if let Some(index) = index {
            self.detected_effects = detect_code_effects(body, &index.files);
        }
        let calls = detect_generate_calls(body);
        // Motions the move sets on its weapons, which beat a guess from the move's name.
        let motion_changes: Vec<(String, String)> = detect_motion_changes(body)
            .into_iter()
            .filter_map(|(arg, motion)| Some((self.resolve_article(&arg)?, motion)))
            .collect();
        let effect_bones = detect_effect_bones(body);
        let mut per_frame: HashMap<(u32, String), usize> = HashMap::new();
        for (frame, arg) in calls {
            let Some(article) = self.resolve_article(&arg) else {
                self.unresolved.push((frame, arg));
                continue;
            };
            let nth = {
                let n = per_frame.entry((frame, article.clone())).or_default();
                *n += 1;
                *n - 1
            };
            self.ensure_assets(&article, index, labels);
            let mut spawn = ArticleSpawn {
                article: article.clone(),
                frame,
                from_script: true,
                ..Default::default()
            };
            self.auto_settings(&mut spawn, nth);
            // A free weapon with no spawn offset of its own leaves the hand the move's effect
            // marks on that frame (Mario's fireball: `mario_fb_shoot` on `havel`).
            if spawn.attach_bone.is_empty() && spawn.offset == [0.0; 3] {
                if let Some(b) = launch_bone(&effect_bones, frame, &self.body_bones) {
                    spawn.start_bone = b;
                }
            }
            if let Some((_, motion)) = motion_changes.iter().find(|(a, _)| *a == article) {
                if self.assets.get(&article).is_some_and(|a| a.motion(motion).is_some()) {
                    spawn.motion = motion.clone();
                }
            }
            self.detected.push(spawn);
        }
        // Saved spawns may name weapons the script does not.
        let saved: Vec<String> = self
            .prefs
            .moves
            .get(move_name)
            .map(|s| s.iter().map(|s| s.article.clone()).collect())
            .unwrap_or_default();
        for article in saved {
            self.ensure_assets(&article, index, labels);
        }
    }

    /// Guess motion, attach bone and rotation for a spawn from its assets. `nth` counts spawns
    /// of the same weapon on the same frame (two hands generating one weapon each).
    pub fn auto_settings(&self, spawn: &mut ArticleSpawn, nth: usize) {
        let Some(assets) = self.assets.get(&spawn.article) else {
            return;
        };
        spawn.motion = pick_motion(assets, &self.move_name).unwrap_or_default();
        // No source says how it moves: a vanilla weapon flies by its params.
        if assets.scripts.is_empty() && !assets.constrained {
            if let Some(params) = assets.params {
                if let Some(v) = params.velocity() {
                    spawn.velocity = v;
                }
                if let Some(life) = params.life {
                    spawn.life = life;
                }
                if let Some([x, y]) = params.offset {
                    spawn.offset = [0.0, y, x];
                }
            }
        }
        if assets.constrained && !assets.bone_literals.is_empty() {
            spawn.attach_bone = assets.bone_literals[nth % assets.bone_literals.len()].clone();
            // The preview faces right, so a right-facing offset wins.
            if let Some((_, rot)) = assets
                .rot_offsets
                .iter()
                .find(|(name, _)| name.ends_with('R'))
                .or_else(|| assets.rot_offsets.first())
            {
                spawn.rotation = *rot;
            }
        }
    }

    fn resolve_article(&self, arg: &str) -> Option<String> {
        let raw = if let Some(start) = arg.find('"') {
            let rest = &arg[start + 1..];
            rest[..rest.find('"')?].to_string()
        } else {
            let ident = arg
                .rsplit(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .find(|s| !s.is_empty())?;
            match ident.find("GENERATE_ARTICLE_") {
                Some(at) => ident[at + "GENERATE_ARTICLE_".len()..].to_string(),
                None => ident.to_string(),
            }
        };
        let raw = raw.to_ascii_lowercase();
        let exact = |raw: &str| {
            self.available
                .iter()
                .find(|a| **a == raw)
                .or_else(|| self.available.iter().find(|a| norm(a) == norm(raw)))
                .cloned()
        };
        exact(&raw)
            // `ITEM_BANANA` is the `banana` folder.
            .or_else(|| raw.strip_prefix("item_").and_then(|r| exact(r)))
            // `DAIKON` is `peachdaikon`: a folder that only adds the fighter's name in front.
            .or_else(|| {
                let fighter = self.ctx.as_ref().map(|c| norm(&c.fighter)).unwrap_or_default();
                let want = norm(&raw);
                let mut ends: Vec<&String> = self
                    .available
                    .iter()
                    .filter(|a| {
                        let a = norm(a);
                        a.ends_with(&want) && (a.len() == fighter.len() + want.len() && a.starts_with(&fighter))
                    })
                    .collect();
                if ends.is_empty() {
                    ends = self.available.iter().filter(|a| norm(a).ends_with(&want)).collect();
                }
                (ends.len() == 1 && want.len() >= 4).then(|| ends[0].clone())
            })
    }

    /// The spawns the current move previews: the saved list if the user changed it, else the
    /// detected one.
    pub fn spawns(&self) -> &[ArticleSpawn] {
        self.prefs
            .moves
            .get(&self.move_name)
            .map(Vec::as_slice)
            .unwrap_or(&self.detected)
    }

    pub fn is_customized(&self) -> bool {
        self.prefs.moves.contains_key(&self.move_name)
    }

    /// Editable spawns for the current move (copies the detected list on first edit).
    pub fn spawns_mut(&mut self) -> &mut Vec<ArticleSpawn> {
        let detected = self.detected.clone();
        self.prefs
            .moves
            .entry(self.move_name.clone())
            .or_insert(detected)
    }

    pub fn reset_move(&mut self) {
        self.prefs.moves.remove(&self.move_name);
        self.save();
    }

    /// The code-spawned effects the current move previews.
    pub fn code_effects(&self) -> &[CodeEffect] {
        self.prefs
            .effects
            .get(&self.move_name)
            .map(Vec::as_slice)
            .unwrap_or(&self.detected_effects)
    }

    pub fn code_effects_customized(&self) -> bool {
        self.prefs.effects.contains_key(&self.move_name)
    }

    pub fn code_effects_mut(&mut self) -> &mut Vec<CodeEffect> {
        let detected = self.detected_effects.clone();
        self.prefs
            .effects
            .entry(self.move_name.clone())
            .or_insert(detected)
    }

    pub fn reset_code_effects(&mut self) {
        self.prefs.effects.remove(&self.move_name);
        self.save();
    }

    /// The code-spawned effects alive on `frame`, ready for the effect renderer.
    pub fn live_code_effects(&self, frame: u32) -> Vec<crate::eff_runtime::LiveEffect> {
        let mut out = Vec::new();
        if self.hide {
            return out;
        }
        for fx in self.code_effects() {
            if !fx.enabled || fx.name.is_empty() || frame < fx.frame {
                continue;
            }
            if fx.life > 0 && frame >= fx.frame + fx.life {
                continue;
            }
            let count = fx.ring_count.max(1);
            for k in 0..count {
                let mut offset = glam::Vec3::from(fx.offset);
                if count > 1 {
                    let angle = std::f32::consts::TAU * k as f32 / count as f32;
                    offset += glam::Vec3::new(angle.cos(), 0.0, angle.sin()) * fx.ring_radius;
                }
                out.push(crate::eff_runtime::LiveEffect {
                    name: fx.name.clone(),
                    bone: if fx.bone.is_empty() { "top".into() } else { fx.bone.clone() },
                    tint: [1.0; 3],
                    alpha: 1.0,
                    age: (frame - fx.frame) as f32,
                    offset,
                    rotation: glam::Vec3::from(fx.rotation),
                    scale: fx.scale,
                    follows: fx.follows,
                    stopped: None,
                });
            }
        }
        out
    }

    pub fn save(&self) {
        let Some(ctx) = &self.ctx else {
            return;
        };
        let path = prefs_path(&ctx.fighter);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(&self.prefs) {
            if let Err(e) = std::fs::write(&path, text) {
                eprintln!("[articles] could not save {}: {e}", path.display());
            }
        }
    }

    pub fn assets(&self, article: &str) -> Option<&Arc<ArticleAssets>> {
        self.assets.get(article)
    }

    /// Load a weapon's model, motion list and scripts once.
    pub fn ensure_assets(
        &mut self,
        article: &str,
        index: Option<&crate::acmd_src::SourceIndex>,
        labels: &HashMap<u64, String>,
    ) {
        if self.assets.contains_key(article) {
            return;
        }
        let Some(ctx) = self.ctx.clone() else {
            return;
        };
        let assets = load_assets(&ctx, article, index, labels, &self.body_bones);
        println!(
            "[articles] {article}: model {}, {} motion(s), {} script(s), bones {:?}",
            assets
                .model_dir
                .as_ref()
                .map_or("none".to_string(), |d| d.display().to_string()),
            assets.motions.len(),
            assets.scripts.len(),
            assets.bone_literals,
        );
        self.assets.insert(article.to_string(), Arc::new(assets));
    }

    fn anim(&mut self, path: &Path) -> Option<Arc<AnimData>> {
        self.anims
            .entry(path.to_path_buf())
            .or_insert_with(|| AnimData::from_file(path).ok().map(Arc::new))
            .clone()
    }

    fn parsed(&mut self, article: &str, assets: &ArticleAssets, motion: Option<&ArticleMotion>) -> Arc<ParsedScripts> {
        let key = (article.to_string(), motion.map_or(0, |m| m.hash));
        if let Some(parsed) = self.parsed.get(&key) {
            return parsed.clone();
        }
        let mut body = String::new();
        if let Some(motion) = motion {
            let mut seen = HashSet::new();
            for (name, text) in &assets.scripts {
                let hash = hash40::hash40(&name.to_ascii_lowercase()).0;
                if motion.script_hashes.contains(&hash) && seen.insert(name.clone()) {
                    body.push_str(text);
                    body.push_str("\n\n");
                }
            }
            // A list without matching script hashes: fall back to the motion's own name.
            if body.is_empty() {
                for prefix in ["game_", "effect_"] {
                    if let Some(text) = assets.scripts.get(&format!("{prefix}{}", motion.name)) {
                        body.push_str(text);
                        body.push_str("\n\n");
                    }
                }
            }
            // No linked source for this weapon: a vanilla one's scripts come from the dump.
            if body.is_empty() && assets.scripts.is_empty() && !motion.name.starts_with("0x") {
                if let Some(ctx) = &self.ctx {
                    body = crate::acmd::fetch_weapon_script_cached(&ctx.fighter, article, &motion.name);
                }
            }
        }
        let parsed = if body.is_empty() {
            ParsedScripts::default()
        } else {
            ParsedScripts {
                hitboxes: crate::acmd::parse_acmd_script(&body).to_hitboxes(),
                effects: crate::acmd::parse_effect_script(&body).to_effect_calls(),
            }
        };
        let parsed = Arc::new(parsed);
        self.parsed.insert(key, parsed.clone());
        parsed
    }

    /// The spawns live on body frame `current_frame` (one-based), placed in the world.
    pub fn instances(&mut self, current_frame: u32, body_bones: BodyBones) -> Vec<ArticleInstance> {
        if self.hide || self.ctx.is_none() {
            return Vec::new();
        }
        let spawns: Vec<ArticleSpawn> = self.spawns().to_vec();
        let mut out = Vec::new();
        for (index, spawn) in spawns.iter().enumerate() {
            if !spawn.enabled || current_frame < spawn.frame.max(1) {
                continue;
            }
            let Some(assets) = self.assets.get(&spawn.article).cloned() else {
                continue;
            };
            let script_frame = current_frame - spawn.frame.max(1) + 1;
            let anim_frame = (script_frame - 1) as f32;
            if spawn.life > 0 && script_frame > spawn.life {
                continue;
            }
            let motion = (!spawn.motion.is_empty())
                .then(|| assets.motion(&spawn.motion))
                .flatten();
            let anim = motion
                .and_then(|m| m.anim_path.clone())
                .and_then(|p| self.anim(&p));
            if spawn.life == 0 {
                if let (Some(anim), Some(motion)) = (&anim, motion) {
                    if !motion.looping && anim_frame > anim.final_frame_index + 1.0 {
                        continue;
                    }
                }
            }
            let anim_frame = match (&anim, motion) {
                (Some(anim), Some(m)) if m.looping && anim.final_frame_index > 0.0 => {
                    anim_frame % (anim.final_frame_index + 1.0)
                }
                (Some(anim), _) => anim_frame.min(anim.final_frame_index),
                _ => anim_frame,
            };
            let scripts = self.parsed(&spawn.article, &assets, motion);
            let root = spawn_root(spawn, current_frame, body_bones);
            let bones = assets
                .skel
                .as_deref()
                .map(|skel| placed_bones(skel, anim.as_deref(), anim_frame, root))
                .unwrap_or_default();
            out.push(ArticleInstance {
                index,
                article: spawn.article.clone(),
                model_dir: assets.model_dir.clone(),
                skel: assets.skel.clone(),
                anim,
                anim_frame,
                script_frame,
                root,
                bones,
                scripts,
            });
        }
        out
    }

    /// An instance's bones `back` frames ago, for effect trails.
    pub fn bones_before(
        &self,
        inst: &ArticleInstance,
        current_frame: u32,
        back: f32,
        body_bones: BodyBones,
    ) -> HashMap<String, glam::Mat4> {
        let Some(spawn) = self.spawns().get(inst.index) else {
            return HashMap::new();
        };
        let Some(skel) = inst.skel.as_deref() else {
            return HashMap::new();
        };
        let frame = (current_frame as f32 - back).max(spawn.frame as f32);
        let root = spawn_root_at(spawn, frame, body_bones);
        placed_bones(skel, inst.anim.as_deref(), (inst.anim_frame - back).max(0.0), root)
    }
}

fn placed_bones(
    skel: &SkelData,
    anim: Option<&AnimData>,
    frame: f32,
    root: glam::Mat4,
) -> HashMap<String, glam::Mat4> {
    let mut bones = crate::renderer::compute_bone_world_matrices(skel, anim, &[], frame);
    for m in bones.values_mut() {
        *m = root * *m;
    }
    bones
}

/// ACMD rotation order: Rz * Ry * Rx, degrees.
pub fn euler_deg(r: [f32; 3]) -> glam::Mat4 {
    glam::Mat4::from_rotation_z(r[2].to_radians())
        * glam::Mat4::from_rotation_y(r[1].to_radians())
        * glam::Mat4::from_rotation_x(r[0].to_radians())
}

fn bone<'a>(bones: &'a HashMap<String, glam::Mat4>, name: &str) -> Option<&'a glam::Mat4> {
    bones.get(name).or_else(|| bones.get(&name.to_ascii_lowercase()))
}

fn spawn_root(spawn: &ArticleSpawn, current_frame: u32, body_bones: BodyBones) -> glam::Mat4 {
    spawn_root_at(spawn, current_frame as f32, body_bones)
}

/// Where a spawn's root is on (possibly fractional, one-based) body frame `frame`.
fn spawn_root_at(spawn: &ArticleSpawn, frame: f32, body_bones: BodyBones) -> glam::Mat4 {
    let offset = glam::Vec3::from(spawn.offset);
    let rot = euler_deg(spawn.rotation);
    if !spawn.attach_bone.is_empty() {
        let bones = body_bones((frame - 1.0).max(0.0));
        if let Some(m) = bone(&bones, &spawn.attach_bone) {
            let (_, r, t) = m.to_scale_rotation_translation();
            return glam::Mat4::from_rotation_translation(r, t)
                * glam::Mat4::from_translation(offset)
                * rot;
        }
    }
    let start = spawn.frame.max(1) as f32;
    let bones = body_bones(start - 1.0);
    let from = if spawn.start_bone.is_empty() { "top" } else { spawn.start_bone.as_str() };
    let base = bone(&bones, from)
        .or_else(|| bone(&bones, "top"))
        .map(|m| m.col(3).truncate())
        .unwrap_or(glam::Vec3::ZERO);
    let age = (frame - start).max(0.0);
    glam::Mat4::from_translation(base + offset + glam::Vec3::from(spawn.velocity) * age) * rot
}

/// Best motion for a weapon spawned by `move_name`.
fn pick_motion(assets: &ArticleAssets, move_name: &str) -> Option<String> {
    let want = norm(move_name);
    let names: Vec<&str> = assets.motions.iter().map(|m| m.name.as_str()).collect();
    if let Some(found) = names.iter().find(|n| norm(n) == want) {
        return Some(found.to_string());
    }
    let prefixed: Vec<&&str> = names
        .iter()
        .filter(|n| !want.is_empty() && norm(n).starts_with(&want))
        .collect();
    if let Some(found) = prefixed
        .iter()
        .find(|n| n.ends_with('r'))
        .or_else(|| prefixed.first())
    {
        return Some(found.to_string());
    }
    // A thrown weapon's launch motion, ahead of held/landing ones that sort first.
    for launch in ["fly", "shoot", "regular", "move", "start"] {
        if let Some(found) = names.iter().find(|n| **n == launch) {
            return Some(found.to_string());
        }
    }
    names.first().map(|n| n.to_string())
}

fn slot_dir(part: &Path, slot: u8) -> Option<PathBuf> {
    let preferred = part.join(format!("c{slot:02}"));
    if preferred.is_dir() {
        return Some(preferred);
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(part)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .and_then(crate::data::parse_costume_dir)
                    .is_some()
        })
        .collect();
    dirs.sort();
    dirs.into_iter().next()
}

fn load_assets(
    ctx: &ArticleContext,
    article: &str,
    index: Option<&crate::acmd_src::SourceIndex>,
    labels: &HashMap<u64, String>,
    body_bones: &HashSet<String>,
) -> ArticleAssets {
    let model_part = ctx.model_root.join(article);
    let model_dir = slot_dir(&model_part, ctx.slot).filter(|d| d.join("model.numshb").is_file());
    let skel = crate::data::find_part_skel(&model_part, ctx.slot)
        .and_then(|p| SkelData::from_file(&p).ok())
        .map(Arc::new);

    let agent = format!("{}_{article}", ctx.fighter);
    let mut scripts = index.map(|i| i.agent_scripts(&agent)).unwrap_or_default();
    let sources = index.map(|i| i.agent_sources(&agent)).unwrap_or_default();
    let mut bone_literals: Vec<String> = Vec::new();
    let mut rot_offsets = Vec::new();
    let mut constrained = false;
    for (_, text) in &sources {
        constrained |= text.contains("set_model_constraint") || text.contains("LinkModule::link(");
        for literal in string_literals(text) {
            let lower = literal.to_ascii_lowercase();
            if matches!(lower.as_str(), "top" | "root" | "trans" | "rot") {
                continue;
            }
            if body_bones.contains(&lower) && !bone_literals.contains(&lower) {
                bone_literals.push(lower);
            }
        }
        rot_offsets.extend(rot_consts(text));
    }

    let mut motions = Vec::new();
    let motion_part = ctx.motion_root.join(article);
    if let Some(motion_dir) = slot_dir(&motion_part, ctx.slot) {
        let files: HashMap<u64, PathBuf> = std::fs::read_dir(&motion_dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "nuanmb"))
                    .filter_map(|p| {
                        let name = p.file_name()?.to_str()?.to_ascii_lowercase();
                        Some((hash40::hash40(&name).0, p))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Names: the app's labels, then whatever the weapon's own scripts spell out.
        let mut names: HashMap<u64, String> = HashMap::new();
        for script in scripts.keys() {
            if let Some((_, rest)) = script.split_once('_') {
                names.insert(hash40::hash40(rest).0, rest.to_string());
            }
        }
        if let Some(yml) = crate::motion_yml::find(&motion_dir).and_then(|p| crate::motion_yml::load(&p).ok()) {
            names.extend(yml.labels);
        }
        match motion_lib::open(motion_dir.join("motion_list.bin")) {
            Ok(list) => {
                for (kind, motion) in &list.list {
                    let name = labels
                        .get(&kind.0)
                        .or_else(|| names.get(&kind.0))
                        .cloned()
                        .or_else(|| {
                            // The animation file name usually spells the motion.
                            motion.animations.first().and_then(|a| files.get(&a.name.0)).and_then(|p| {
                                let stem = p.file_stem()?.to_str()?.to_ascii_lowercase();
                                let guess = stem.get(3..)?.to_string();
                                (hash40::hash40(&guess).0 == kind.0).then_some(guess)
                            })
                        })
                        .unwrap_or_else(|| format!("{:#x}", kind.0));
                    let anim_path = motion
                        .animations
                        .iter()
                        .find_map(|a| files.get(&a.name.0).cloned());
                    motions.push(ArticleMotion {
                        hash: kind.0,
                        name,
                        anim_path,
                        looping: motion.flags.r#loop,
                        script_hashes: std::iter::once(motion.game_script.0)
                            .chain(motion.scripts.iter().map(|s| s.0))
                            .filter(|h| *h != 0)
                            .collect(),
                    });
                }
            }
            Err(e) => eprintln!("[articles] {article}: no motion list in {}: {e}", motion_dir.display()),
        }
    }
    scripts.retain(|_, text| !text.is_empty());
    let params = ctx
        .model_root
        .parent()
        .map(|f| f.join("param").join("vl.prc"))
        .filter(|p| p.is_file())
        .and_then(|p| WeaponParams::read(&p, article));
    ArticleAssets {
        params,
        model_dir,
        skel,
        motions,
        scripts,
        bone_literals,
        rot_offsets,
        constrained,
    }
}

fn string_literals(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('"') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('"') else { break };
        let lit = &after[..end];
        if !lit.is_empty() && lit.len() < 40 && lit.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            out.push(lit);
        }
        rest = &after[end + 1..];
    }
    out
}

/// `const NAME: Vector3f = Vector3f{x:1.0, y:2.0, z:3.0}` with ROT in the name.
fn rot_consts(text: &str) -> Vec<(String, [f32; 3])> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub const ").or_else(|| line.strip_prefix("const ")) else {
            continue;
        };
        let Some((name, rest)) = rest.split_once(':') else { continue };
        let name = name.trim();
        if !name.to_ascii_uppercase().contains("ROT") || !rest.contains("Vector3f") {
            continue;
        }
        let field = |key: &str| -> Option<f32> {
            let at = rest.find(&format!("{key}:"))? + 2;
            let tail = rest[at..].trim_start();
            let end = tail
                .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                .unwrap_or(tail.len());
            tail[..end].parse().ok()
        };
        if let (Some(x), Some(y), Some(z)) = (field("x"), field("y"), field("z")) {
            out.push((name.to_string(), [x, y, z]));
        }
    }
    out
}

/// `(frame, second argument)` for each `generate_article(` call in a script body, with the
/// frame tracked through `frame(...)` and `wait(...)`.
pub fn detect_generate_calls(body: &str) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    let mut frame = 1.0f32;
    for line in body.lines() {
        let code = line.split("//").next().unwrap_or("");
        if code.contains("fn ") && code.contains('(') {
            frame = 1.0;
        }
        if let Some(v) = call_number(code, "frame") {
            frame = v.max(1.0);
        }
        if let Some(v) = call_number(code, "wait") {
            frame += v;
        }
        if let Some(at) = code.find("generate_article").filter(|at| {
            // `generate_article_enable` etc. are not spawns.
            code[at + "generate_article".len()..].trim_start().starts_with('(')
        }) {
            let args = &code[at..];
            let Some(open) = args.find('(') else { continue };
            let args = &args[open + 1..];
            let parts: Vec<&str> = split_top_level(args);
            if let Some(arg) = parts.get(1) {
                out.push((frame.round() as u32, arg.trim().to_string()));
            }
        }
    }
    out
}

/// (frame, bone) of each effect macro in a script: `EFFECT*(agent, Hash40::new("name"),
/// Hash40::new("bone"), ...)`.
pub fn detect_effect_bones(body: &str) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    let mut frame = 1.0f32;
    for line in body.lines() {
        let code = line.split("//").next().unwrap_or("");
        if code.contains("fn ") && code.contains('(') {
            frame = 1.0;
        }
        if let Some(v) = call_number(code, "frame") {
            frame = v.max(1.0);
        }
        if let Some(v) = call_number(code, "wait") {
            frame += v;
        }
        let Some(at) = code.find("EFFECT") else { continue };
        let Some(open) = code[at..].find('(') else { continue };
        let args = split_top_level(&code[at + open + 1..]);
        let Some(bone) = args.get(2).and_then(|a| {
            let a = a.trim().strip_prefix("Hash40::new(\"")?;
            Some(a[..a.find('"')?].to_ascii_lowercase())
        }) else {
            continue;
        };
        out.push((frame.round() as u32, bone));
    }
    out
}

/// The hand-like body bone an effect marks closest to `frame` (within 4 frames).
fn launch_bone(effects: &[(u32, String)], frame: u32, body_bones: &HashSet<String>) -> Option<String> {
    effects
        .iter()
        .filter(|(f, b)| {
            f.abs_diff(frame) <= 4
                && (b.starts_with("have") || b.starts_with("hand") || b.starts_with("arm"))
                && (body_bones.is_empty() || body_bones.contains(b.as_str()))
        })
        .min_by_key(|(f, _)| f.abs_diff(frame))
        .map(|(_, b)| b.clone())
}

/// `ArticleModule::change_motion(boma, <weapon>, Hash40::new("<motion>"), ...)` calls:
/// (weapon argument text, motion name).
pub fn detect_motion_changes(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in body.lines() {
        let code = line.split("//").next().unwrap_or("");
        let Some(at) = code.find("ArticleModule::change_motion(") else { continue };
        let args = split_top_level(&code[at + "ArticleModule::change_motion(".len()..]);
        let (Some(weapon), Some(motion)) = (args.get(1), args.get(2)) else { continue };
        let Some(q) = motion.find('"') else { continue };
        let Some(end) = motion[q + 1..].find('"') else { continue };
        out.push((weapon.trim().to_string(), motion[q + 1..q + 1 + end].to_string()));
    }
    out
}

/// Last numeric argument of a bare `name(` call on this line.
fn call_number(code: &str, name: &str) -> Option<f32> {
    let pat = format!("{name}(");
    let mut search = 0;
    while let Some(rel) = code[search..].find(&pat) {
        let at = search + rel;
        search = at + pat.len();
        let before = code[..at].chars().last();
        if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let args = &code[at + pat.len()..];
        let parts = split_top_level(args);
        let last = parts.last()?.trim().trim_end_matches(';').trim_end_matches(')');
        return last.trim().parse().ok();
    }
    None
}

/// Split call arguments at top-level commas, stopping at the call's closing paren.
fn split_top_level(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in args.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                if depth == 0 {
                    out.push(&args[start..i]);
                    return out;
                }
                depth -= 1;
            }
            ',' if depth == 0 => {
                out.push(&args[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&args[start..]);
    out
}

/// Effects a move's script reaches through the mod's own functions.
///
/// Follows each `crate::module::function(` call in the body (at the frame the script is on) into
/// the project's source, then same-file helpers it calls, collecting `EffectModule::req*` calls.
/// Names, joints and sizes resolve through string/float consts and simple `let`s; a request in
/// a `for _ in 0..N` loop that places with `cos`/`sin` reads as a ring of N.
pub fn detect_code_effects(body: &str, files: &[PathBuf]) -> Vec<CodeEffect> {
    let mut out = Vec::new();
    let mut frame = 1.0f32;
    let mut texts: HashMap<PathBuf, Option<String>> = HashMap::new();
    for line in body.lines() {
        let code = line.split("//").next().unwrap_or("");
        if code.contains("fn ") && code.contains('(') {
            frame = 1.0;
        }
        if let Some(v) = call_number(code, "frame") {
            frame = v.max(1.0);
        }
        if let Some(v) = call_number(code, "wait") {
            frame += v;
        }
        let mut search = 0;
        while let Some(rel) = code[search..].find("crate::") {
            let at = search + rel;
            search = at + 7;
            let path: String = code[at + 7..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            if !code[at + 7 + path.len()..].starts_with('(') {
                continue;
            }
            let segs: Vec<&str> = path.split("::").filter(|s| !s.is_empty()).collect();
            let [modules @ .., func] = segs.as_slice() else { continue };
            let Some(module) = modules.last() else { continue };
            let Some(file) = module_file(files, module) else { continue };
            let text = texts
                .entry(file.clone())
                .or_insert_with(|| std::fs::read_to_string(&file).ok())
                .clone();
            let Some(text) = text else { continue };
            let mut seen = HashSet::new();
            collect_fn_effects(&text, module, func, frame.round() as u32, 0, &mut seen, &mut out);
        }
    }
    out
}

/// The source file for `crate::...::<module>`: `<module>.rs` or `<module>/mod.rs`.
fn module_file(files: &[PathBuf], module: &str) -> Option<PathBuf> {
    files
        .iter()
        .find(|f| {
            let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let parent = f
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");
            stem == module || (stem == "mod" && parent == module)
        })
        .cloned()
}

/// The text inside `fn <name>`'s braces.
fn fn_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("fn {name}");
    let mut search = 0;
    while let Some(rel) = text[search..].find(&pat) {
        let at = search + rel;
        search = at + pat.len();
        let after = &text[at + pat.len()..];
        if !(after.starts_with('(') || after.starts_with('<')) {
            continue;
        }
        let open = at + text[at..].find('{')?;
        return block_end(text, open).map(|end| &text[open + 1..end]);
    }
    None
}

/// Index of the `}` matching the `{` at `open`.
fn block_end(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

fn collect_fn_effects(
    text: &str,
    module: &str,
    func: &str,
    frame: u32,
    depth: u32,
    seen: &mut HashSet<String>,
    out: &mut Vec<CodeEffect>,
) {
    if depth > 3 || !seen.insert(func.to_string()) {
        return;
    }
    let Some(body) = fn_body(text, func) else { return };
    // `for _ in 0..N {` spans, to tell a ring from a single request.
    let mut loops: Vec<(usize, usize, u32)> = Vec::new();
    let mut search = 0;
    while let Some(rel) = body[search..].find("for ") {
        let at = search + rel;
        search = at + 4;
        let Some(open_rel) = body[at..].find('{') else { break };
        let header = &body[at..at + open_rel];
        let Some(range) = header.split(" in ").nth(1) else { continue };
        let Some(bound) = range.trim().strip_prefix("0..") else { continue };
        let Some(n) = eval_number(text, body, bound.trim()) else { continue };
        let open = at + open_rel;
        let end = block_end(body, open).unwrap_or(body.len());
        loops.push((open, end, n.max(1.0) as u32));
    }
    let mut search = 0;
    while let Some(rel) = body[search..].find("EffectModule::req") {
        let at = search + rel;
        search = at + 17;
        let kind: String = body[at + 17..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let Some(open) = body[at..].find('(') else { continue };
        let args = split_top_level(&body[at + open + 1..]);
        let (name_at, joint_at, scale_at) = match kind.as_str() {
            "" => (1, None, 4),
            "_on_joint" | "_follow" => (1, Some(2), 5),
            _ => continue,
        };
        let Some(name) = args.get(name_at).and_then(|a| hash_arg(text, a)) else { continue };
        let mut fx = CodeEffect {
            name,
            frame,
            source: format!("{module}::{func}"),
            follows: kind == "_follow",
            ..Default::default()
        };
        if let Some(bone) = joint_at.and_then(|j| args.get(j)).and_then(|a| hash_arg(text, a)) {
            fx.bone = bone;
        }
        if let Some(v) = args.get(scale_at).and_then(|a| eval_number(text, body, a.trim())) {
            fx.scale = v;
        }
        if let Some((_, _, n)) = loops.iter().find(|(s, e, _)| at > *s && at < *e) {
            if *n > 1 && body.contains(".cos()") {
                fx.ring_count = *n;
                fx.ring_radius = const_named(text, "RADIUS").unwrap_or(10.0);
            }
        }
        out.push(fx);
    }
    // Same-file helpers this function calls.
    let mut search = 0;
    while let Some(rel) = body[search..].find('(') {
        let at = search + rel;
        search = at + 1;
        let start = body[..at]
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map(|i| i + 1)
            .unwrap_or(0);
        let ident = &body[start..at];
        if ident.is_empty() || body[..start].ends_with(':') || body[..start].ends_with('.') {
            continue;
        }
        if text.contains(&format!("fn {ident}(")) {
            collect_fn_effects(text, module, ident, frame, depth + 1, seen, out);
        }
    }
}

/// `Hash40::new("x")` / `Hash40::new(CONST)` → the name.
fn hash_arg(text: &str, arg: &str) -> Option<String> {
    let inner = arg
        .trim()
        .strip_prefix("Hash40::new(")?
        .trim_end()
        .strip_suffix(')')?
        .trim();
    if let Some(lit) = inner.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Some(lit.to_string());
    }
    let pat = format!("const {inner}:");
    let at = text.find(&pat)?;
    let rest = &text[at..];
    let q = rest.find('"')?;
    let end = rest[q + 1..].find('"')?;
    Some(rest[q + 1..q + 1 + end].to_string())
}

/// A number literal, a `const`, or a `let` in the function (its first term, so a charge-scaled
/// size reads as its uncharged value).
fn eval_number(text: &str, body: &str, expr: &str) -> Option<f32> {
    let expr = expr.trim().trim_end_matches(';').trim();
    let first: String = expr
        .trim_start_matches('(')
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '-')
        .collect();
    let first = strip_num_suffix(&first);
    if let Ok(v) = first.parse::<f32>() {
        return Some(v);
    }
    let last = first.rsplit('.').next().unwrap_or(first);
    if let Some(v) = const_value(text, last) {
        return Some(v);
    }
    let pat = format!("let {last} = ");
    if let Some(at) = body.find(&pat) {
        let stmt = body[at + pat.len()..].split(';').next().unwrap_or("");
        let first_word: String = stmt
            .trim()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if first_word != last {
            return eval_number(text, body, stmt);
        }
    }
    None
}

fn strip_num_suffix(s: &str) -> &str {
    for suffix in ["_f32", "_usize", "_u32", "_i32"] {
        if let Some(v) = s.strip_suffix(suffix) {
            return v;
        }
    }
    s
}

fn const_value(text: &str, name: &str) -> Option<f32> {
    let pat = format!("const {name}:");
    let at = text.find(&pat)?;
    let rest = &text[at + pat.len()..];
    let eq = rest.find('=')?;
    let val = rest[eq + 1..].split(';').next()?.trim();
    strip_num_suffix(val).parse().ok()
}

/// The first numeric const whose name ends with `suffix`.
fn const_named(text: &str, suffix: &str) -> Option<f32> {
    let mut search = 0;
    while let Some(rel) = text[search..].find("const ") {
        let at = search + rel;
        search = at + 6;
        let name: String = text[at + 6..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.ends_with(suffix) {
            if let Some(v) = const_value(text, &name) {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZONE_SOURCE: &str = r#"
pub const RADIUS: f32 = 12.0;
const EDGE: &str = "clone_a_zone_edge";
const SPARK: &str = "clone_a_spark";
const COUNT: usize = 6;
pub unsafe fn cast(boma: *mut B) {
    let radius = RADIUS + (20.0 - RADIUS) * charge;
    let edge = EffectModule::req(boma, Hash40::new(EDGE), &pos, &rot, radius, 0, -1, false, 0);
    light(boma);
}
unsafe fn light(boma: *mut B) {
    for i in 0..COUNT {
        let angle = TAU * i as f32 / COUNT as f32;
        let at = Vector3f { x: r * angle.cos(), y: 0.0, z: r * angle.sin() };
        EffectModule::req(boma, Hash40::new(SPARK), &at, &rot, 0.5, 0, -1, false, 0);
    }
    EffectModule::req_on_joint(boma, Hash40::new("clone_a_glow"), Hash40::new("handr"), &a, &b, 2.0, &c, &d, false, 0, 0, 0);
}
"#;

    #[test]
    fn reads_vanilla_weapon_params() {
        // Needs a game dump; skipped without one.
        let vl = Path::new("D:/Documents/SmashArc/fighter/link/param/vl.prc");
        if !vl.is_file() {
            return;
        }
        let beam = WeaponParams::read(vl, "swordbeam").expect("param_swordbeam");
        assert_eq!(beam.speed, Some(3.4));
        assert_eq!(beam.life, Some(40));
        assert_eq!(beam.offset, Some([11.0, 6.0]));
        let boomerang = WeaponParams::read(vl, "boomerang").expect("param_boomerang");
        assert_eq!(boomerang.speed, Some(2.4));
        assert_eq!(boomerang.life, Some(129));
        assert!(WeaponParams::read(vl, "navy").is_none());
        let v = WeaponParams { speed: Some(2.0), angle: 90.0, ..Default::default() }.velocity().unwrap();
        assert!(v[1] > 1.99 && v[2].abs() < 1e-5);
    }

    #[test]
    fn vanilla_generate_constants_and_motions() {
        let body = "unsafe extern \"C\" fn game_attacks4(agent: &mut L2CAgentBase) {\n    frame(agent.lua_state_agent, 12.0);\n    if macros::is_excute(agent) {\n        ArticleModule::generate_article(agent.module_accessor, *FIGHTER_LINK_GENERATE_ARTICLE_SWORD_BEAM, false, -1);\n        ArticleModule::generate_article_enable(agent.module_accessor, *FIGHTER_LINK_GENERATE_ARTICLE_SWORD_BEAM, false, -1);\n        ArticleModule::change_motion(agent.module_accessor, *FIGHTER_LINK_GENERATE_ARTICLE_SWORD_BEAM, Hash40::new(\"fly\"), false, -1.0);\n    }\n}\n";
        let calls = detect_generate_calls(body);
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].0, 12);
        let preview = ArticlePreview {
            available: vec!["boomerang".into(), "swordbeam".into()],
            ..Default::default()
        };
        assert_eq!(preview.resolve_article(&calls[0].1).as_deref(), Some("swordbeam"));
        let preview = ArticlePreview {
            available: vec!["banana".into(), "kinopio".into(), "peachdaikon".into()],
            ..Default::default()
        };
        assert_eq!(preview.resolve_article("*FIGHTER_X_GENERATE_ARTICLE_ITEM_BANANA").as_deref(), Some("banana"));
        assert_eq!(preview.resolve_article("*FIGHTER_X_GENERATE_ARTICLE_DAIKON").as_deref(), Some("peachdaikon"));
        assert_eq!(preview.resolve_article("*FIGHTER_X_GENERATE_ARTICLE_STUFF"), None);
        let fx = "    frame(agent.lua_state_agent, 13.0);
    macros::EFFECT(agent, Hash40::new(\"x_shoot\"), Hash40::new(\"havel\"), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, true);
    macros::EFFECT(agent, Hash40::new(\"x_smoke\"), Hash40::new(\"top\"), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, true);
";
        let marks = detect_effect_bones(fx);
        assert_eq!(marks, vec![(13, "havel".to_string()), (13, "top".to_string())]);
        assert_eq!(launch_bone(&marks, 14, &HashSet::new()).as_deref(), Some("havel"));
        assert_eq!(launch_bone(&marks, 30, &HashSet::new()), None);
        let changes = detect_motion_changes(body);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].1, "fly");
    }

    #[test]
    fn follows_calls_into_code_effects() {
        let dir = std::env::temp_dir().join(format!("vis_code_fx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("zone.rs");
        std::fs::write(&file, ZONE_SOURCE).unwrap();
        let body = "unsafe extern \"C\" fn game_x(agent: &mut L2CAgentBase) {\n    frame(agent.lua_state_agent, 14.0);\n    if is_excute(agent) {\n        crate::zone::cast(boma, entry);\n    }\n}\n";
        let found = detect_code_effects(body, &[file]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(found[0].name, "clone_a_zone_edge");
        assert_eq!(found[0].frame, 14);
        assert_eq!(found[0].scale, 12.0);
        assert_eq!(found[1].name, "clone_a_spark");
        assert_eq!(found[1].ring_count, 6);
        assert_eq!(found[1].ring_radius, 12.0);
        assert_eq!(found[1].scale, 0.5);
        assert_eq!(found[2].bone, "handr");
        assert_eq!(found[2].scale, 2.0);
        assert_eq!(found[2].ring_count, 1);
    }


    #[test]
    fn finds_generate_calls_with_frames() {
        let body = r#"
unsafe extern "C" fn game_attack13(agent: &mut L2CAgentBase) {
    frame(agent.lua_state_agent, 1.0);
    if macros::is_excute(agent) {
        ArticleModule::generate_article(boma, crate::article_id("spike"), false, -1);
        ArticleModule::generate_article(boma, crate::article_id("spike"), false, -1);
    }
    frame(agent.lua_state_agent, 10.0);
    wait(agent.lua_state_agent, 4.0);
    ArticleModule::generate_article(agent.module_accessor, *FIGHTER_CLONE_A_GENERATE_ARTICLE_ORB, false, -1);
}"#;
        let calls = detect_generate_calls(body);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, 1);
        assert!(calls[0].1.contains("\"spike\""));
        assert_eq!(calls[2].0, 14);
        let mut p = ArticlePreview::default();
        p.available = vec!["orb".into(), "spike".into()];
        assert_eq!(p.resolve_article(&calls[0].1).as_deref(), Some("spike"));
        assert_eq!(p.resolve_article(&calls[2].1).as_deref(), Some("orb"));
    }

    #[test]
    fn reads_rotation_consts() {
        let text = "pub const CLONE_ROT_OFFSETR: Vector3f = Vector3f{x:-90.0, y:0.0, z:90.0}; //x";
        assert_eq!(rot_consts(text), vec![("CLONE_ROT_OFFSETR".to_string(), [-90.0, 0.0, 90.0])]);
    }
}
