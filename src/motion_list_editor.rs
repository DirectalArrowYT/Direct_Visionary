//! The Motion List window: the open move's `motion_list.bin` entry (flags, blend frames,
//! cancel frame, intangibility window, animations, scripts), edited in memory and written back.
//!
//! Where it saves is decided once per load (see `Targets`): a list read from the vanilla dump is
//! never written in place — it goes to the project workspace overlay — while a mod's own list
//! (and the `motion_list.yml` it is built from, when there is one) is rewritten where it lives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use egui::{RichText, Ui};
use hash40::Hash40;
use motion_lib::mlist::{Animation, Extra, Flags, MList, Motion};

/// What the move loader read, handed over with the move list.
pub struct Loaded {
    pub list: MList,
    /// The `.bin` the list came from, when it came from one.
    pub bin: Option<PathBuf>,
    /// The `motion_list.yml` that names it (beside the bin or at the moveset project root).
    pub yml: Option<PathBuf>,
}

/// Where Save writes. Both empty means saving is unavailable, and `why` says so.
#[derive(Default, Clone)]
pub struct Targets {
    pub bin: Option<PathBuf>,
    pub yml: Option<PathBuf>,
    pub why: Option<String>,
}

impl Targets {
    /// `vanilla` is the dump root; `overlay` is the workspace romfs path of this costume's
    /// `motion_list.bin`, when a project is open.
    pub fn decide(loaded: &Loaded, vanilla: Option<&Path>, overlay: Option<PathBuf>) -> Targets {
        let in_dump = |p: &Path| vanilla.is_some_and(|root| under(p, root));
        let mut t = Targets::default();
        match &loaded.bin {
            Some(bin) if in_dump(bin) => match overlay {
                Some(path) => t.bin = Some(path),
                None => {
                    t.why = Some(
                        "This list is from the game dump. Open or create a project to save an edited copy into it."
                            .into(),
                    )
                }
            },
            Some(bin) => t.bin = Some(bin.clone()),
            None => {}
        }
        if let Some(yml) = &loaded.yml {
            if !in_dump(yml) {
                t.yml = Some(yml.clone());
            }
        }
        if t.bin.is_none() && t.yml.is_none() && t.why.is_none() {
            t.why = Some("Nowhere writable to save this list.".into());
        }
        t
    }
}

fn under(path: &Path, root: &Path) -> bool {
    canonical(path).starts_with(canonical(root))
}

/// `canonicalize` for a path that may not exist yet: the nearest existing ancestor is
/// canonicalized and the rest appended, so both sides of a comparison get the same prefix.
fn canonical(path: &Path) -> PathBuf {
    let mut tail = Vec::new();
    let mut cur = path;
    loop {
        if let Ok(c) = cur.canonicalize() {
            return tail.iter().rev().fold(c, |acc: PathBuf, part| acc.join(part));
        }
        match (cur.parent(), cur.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name.to_os_string());
                cur = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// The list being edited, plus the copy on disk it was read from.
pub struct Doc {
    pub list: MList,
    saved: MList,
    pub read_from: Option<PathBuf>,
    pub targets: Targets,
    /// Text being typed into a hash field, by field id, until it is committed.
    bufs: HashMap<String, String>,
    bufs_kind: u64,
    pub status: Option<String>,
}

pub enum Action {
    None,
    Save,
}

impl Doc {
    pub fn new(loaded: Loaded, targets: Targets) -> Doc {
        Doc {
            saved: loaded.list.clone(),
            read_from: loaded.bin.clone().or(loaded.yml.clone()),
            list: loaded.list,
            targets,
            bufs: HashMap::new(),
            bufs_kind: 0,
            status: None,
        }
    }

    /// The extra block (cancel frame, intangibility) of motion `kind`, as edited.
    pub fn extra(&self, kind: u64) -> Option<&Extra> {
        self.list.list.get(&Hash40(kind))?.extra.as_ref()
    }

    pub fn dirty(&self) -> bool {
        self.list.motion_path != self.saved.motion_path || self.list.list != self.saved.list
    }

    /// Write every target. On success the written list becomes the saved state.
    pub fn save(&mut self, labels: &HashMap<u64, String>) -> Result<Vec<PathBuf>, String> {
        let mut written = Vec::new();
        if let Some(bin) = &self.targets.bin {
            if let Some(dir) = bin.parent() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
            // Through a temp file so a failed write never leaves half a list behind.
            let tmp = bin.with_extension("bin.tmp");
            motion_lib::save(&tmp, &self.list).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, bin).map_err(|e| format!("{}: {e}", bin.display()))?;
            written.push(bin.clone());
        }
        if let Some(yml) = &self.targets.yml {
            let text = crate::motion_yml::write(&self.list, labels);
            std::fs::write(yml, text).map_err(|e| format!("{}: {e}", yml.display()))?;
            written.push(yml.clone());
        }
        self.saved = self.list.clone();
        Ok(written)
    }

    pub fn revert(&mut self) {
        self.list = self.saved.clone();
        self.bufs.clear();
    }

    /// The window body for motion `kind` (`name` is its display name).
    pub fn ui(
        &mut self,
        ui: &mut Ui,
        kind: u64,
        name: &str,
        labels: &mut HashMap<u64, String>,
        current_frame: u32,
        total_frames: u32,
    ) -> Action {
        if self.bufs_kind != kind {
            self.bufs.clear();
            self.bufs_kind = kind;
        }
        let mut action = Action::None;

        ui.horizontal(|ui| {
            ui.label(RichText::new(name).strong());
            if self.dirty() {
                ui.label(RichText::new("● unsaved").color(ui.visuals().warn_fg_color));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let can_save = self.targets.bin.is_some() || self.targets.yml.is_some();
                let save = ui
                    .add_enabled(can_save && self.dirty(), egui::Button::new("Save"))
                    .on_hover_text(self.save_hover());
                if save.clicked() {
                    action = Action::Save;
                }
                if ui
                    .add_enabled(self.dirty(), egui::Button::new("Revert"))
                    .on_hover_text("Throw away every unsaved change to the list (all motions)")
                    .clicked()
                {
                    self.revert();
                }
            });
        });
        if let Some(why) = &self.targets.why {
            ui.label(RichText::new(why).small().color(ui.visuals().warn_fg_color));
        }
        if let Some(from) = &self.read_from {
            ui.label(RichText::new(format!("Read from {}", from.display())).small().weak());
        }
        if let Some(status) = &self.status {
            ui.label(RichText::new(status).small().weak());
        }
        ui.separator();

        let bufs = &mut self.bufs;
        let Some(m) = self.list.list.get_mut(&Hash40(kind)) else {
            ui.label("This motion is not in the list.");
            return action;
        };

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("mlist_main").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                ui.label("Game script").on_hover_text(
                    "The ACMD game script this motion runs. A name, or a 0x hash.",
                );
                hash_field(ui, bufs, "game_script", &mut m.game_script, labels);
                ui.end_row();

                ui.label("Blend frames").on_hover_text(
                    "Frames of interpolation from the previous animation into this one.",
                );
                ui.add(egui::DragValue::new(&mut m.blend_frames).range(0..=255));
                ui.end_row();
            });

            ui.add_space(6.0);
            ui.label(RichText::new("Flags").strong());
            egui::Grid::new("mlist_flags").num_columns(4).spacing([16.0, 4.0]).show(ui, |ui| {
                for (i, (field, value, hint)) in flag_fields_mut(&mut m.flags).into_iter().enumerate() {
                    ui.checkbox(value, field).on_hover_text(hint);
                    if i % 4 == 3 {
                        ui.end_row();
                    }
                }
            });

            ui.add_space(6.0);
            extra_ui(ui, m, current_frame, total_frames);

            ui.add_space(6.0);
            ui.label(RichText::new("Animations").strong());
            let mut remove = None;
            for (i, anim) in m.animations.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    hash_field(ui, bufs, &format!("anim{i}"), &mut anim.name, labels);
                    ui.label("unk");
                    ui.add(egui::DragValue::new(&mut anim.unk).range(0..=255));
                    if ui.small_button("✖").on_hover_text("Remove").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                m.animations.remove(i);
                bufs.retain(|k, _| !k.starts_with("anim"));
            }
            if ui.small_button("+ Animation").clicked() {
                m.animations.push(Animation::default());
            }

            ui.add_space(6.0);
            ui.label(RichText::new("Scripts").strong()).on_hover_text(
                "Effect, sound and expression scripts, in list order.",
            );
            let mut remove = None;
            for (i, script) in m.scripts.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    hash_field(ui, bufs, &format!("script{i}"), script, labels);
                    if ui.small_button("✖").on_hover_text("Remove").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                m.scripts.remove(i);
                bufs.retain(|k, _| !k.starts_with("script"));
            }
            if ui.small_button("+ Script").clicked() {
                m.scripts.push(Hash40(0));
            }
        });
        action
    }

    fn save_hover(&self) -> String {
        let mut lines = Vec::new();
        if let Some(bin) = &self.targets.bin {
            lines.push(format!("Writes {}", bin.display()));
        }
        if let Some(yml) = &self.targets.yml {
            lines.push(format!("Writes {} (comments in it are not kept)", yml.display()));
        }
        if lines.is_empty() {
            lines.push(self.targets.why.clone().unwrap_or_default());
        }
        lines.join("\n")
    }
}

fn extra_ui(ui: &mut Ui, m: &mut Motion, current_frame: u32, total_frames: u32) {
    let mut has = m.extra.is_some();
    if ui
        .checkbox(&mut has, RichText::new("Extra").strong())
        .on_hover_text(
            "The optional block with the cancel frame and intangibility window. Most motions have none.",
        )
        .changed()
    {
        m.extra = has.then(Extra::default);
    }
    let Some(e) = m.extra.as_mut() else {
        return;
    };
    let frame = current_frame.min(255) as u8;
    egui::Grid::new("mlist_extra").num_columns(3).spacing([12.0, 6.0]).show(ui, |ui| {
        ui.label("Cancel frame").on_hover_text(
            "The frame this motion can be cancelled out of (its FAF). 0 = no cancel frame.",
        );
        ui.add(egui::DragValue::new(&mut e.cancel_frame).range(0..=255));
        ui.horizontal(|ui| {
            if ui.small_button("= current").on_hover_text(format!("Set to frame {frame}")).clicked() {
                e.cancel_frame = frame;
            }
            if total_frames > 0 {
                ui.label(RichText::new(format!("of {total_frames}")).weak());
            }
        });
        ui.end_row();

        ui.label("Intangible from").on_hover_text("xlu_start: first frame of the intangibility window");
        ui.add(egui::DragValue::new(&mut e.xlu_start).range(0..=255));
        if ui.small_button("= current").on_hover_text(format!("Set to frame {frame}")).clicked() {
            e.xlu_start = frame;
        }
        ui.end_row();

        ui.label("Intangible to").on_hover_text("xlu_end: last frame of the intangibility window");
        ui.add(egui::DragValue::new(&mut e.xlu_end).range(0..=255));
        if ui.small_button("= current").on_hover_text(format!("Set to frame {frame}")).clicked() {
            e.xlu_end = frame;
        }
        ui.end_row();

        ui.label("No stop interp").on_hover_text("no_stop_intp");
        ui.checkbox(&mut e.no_stop_intp, "");
        ui.end_row();
    });
}

/// A hash40 text field: shows the name when one is known, commits on Enter or focus loss.
/// A name is hashed (and remembered as a label); `0x…` is taken as the hash itself.
fn hash_field(
    ui: &mut Ui,
    bufs: &mut HashMap<String, String>,
    id: &str,
    value: &mut Hash40,
    labels: &mut HashMap<u64, String>,
) {
    let shown = display(*value, labels);
    let buf = bufs.entry(id.to_string()).or_insert_with(|| shown.clone());
    let response = ui.add(egui::TextEdit::singleline(buf).desired_width(260.0));
    if response.lost_focus() {
        match parse_hash(buf) {
            Some((hash, name)) => {
                *value = hash;
                if let Some(name) = name {
                    labels.insert(hash.0, name);
                }
                *buf = display(hash, labels);
            }
            None => *buf = shown,
        }
    } else if !response.has_focus() && *buf != shown {
        // The value changed underneath (revert, undo): follow it.
        *buf = shown;
    }
}

fn display(h: Hash40, labels: &HashMap<u64, String>) -> String {
    match labels.get(&h.0) {
        Some(name) => name.clone(),
        None if h.0 == 0 => String::new(),
        None => format!("0x{:010x}", h.0),
    }
}

/// `0x…` → that hash; anything else → its hash40 and the name. Empty → 0.
pub fn parse_hash(text: &str) -> Option<(Hash40, Option<String>)> {
    let text = text.trim();
    if text.is_empty() {
        return Some((Hash40(0), None));
    }
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).ok().map(|h| (Hash40(h), None));
    }
    Some((hash40::hash40(text), Some(text.to_string())))
}

fn flag_fields_mut(f: &mut Flags) -> [(&'static str, &mut bool, &'static str); 14] {
    [
        ("turn", &mut f.turn, "Turns the fighter around over the motion"),
        ("loop", &mut f.r#loop, "Animation loops"),
        ("move", &mut f.r#move, "Root motion (Trans bone) moves the fighter"),
        ("fix_trans", &mut f.fix_trans, "fix_trans"),
        ("fix_rot", &mut f.fix_rot, "fix_rot"),
        ("fix_scale", &mut f.fix_scale, "fix_scale"),
        ("unk_40", &mut f.unk_40, "Unknown flag 0x40"),
        ("unk_80", &mut f.unk_80, "Unknown flag 0x80"),
        ("unk_100", &mut f.unk_100, "Unknown flag 0x100"),
        ("unk_200", &mut f.unk_200, "Unknown flag 0x200"),
        ("unk_400", &mut f.unk_400, "Unknown flag 0x400"),
        ("unk_800", &mut f.unk_800, "Unknown flag 0x800"),
        ("unk_1000", &mut f.unk_1000, "Unknown flag 0x1000"),
        ("unk_2000", &mut f.unk_2000, "Unknown flag 0x2000"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vis_mlist_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn dump_lists_go_to_the_overlay_and_mod_lists_stay_put() {
        let dir = tmp("targets");
        let dump = dir.join("dump");
        let modr = dir.join("mod");
        std::fs::create_dir_all(&dump).unwrap();
        std::fs::create_dir_all(&modr).unwrap();
        let loaded = |bin: PathBuf, yml: Option<PathBuf>| Loaded { list: MList::default(), bin: Some(bin), yml };

        let t = Targets::decide(&loaded(dump.join("motion_list.bin"), None), Some(&dump), None);
        assert!(t.bin.is_none() && t.why.is_some());
        let overlay = dir.join("ws/romfs/motion_list.bin");
        let t = Targets::decide(&loaded(dump.join("motion_list.bin"), None), Some(&dump), Some(overlay.clone()));
        assert_eq!(t.bin, Some(overlay));

        let yml = modr.join("motion_list.yml");
        let t = Targets::decide(&loaded(modr.join("motion_list.bin"), Some(yml.clone())), Some(&dump), None);
        assert_eq!(t.bin, Some(modr.join("motion_list.bin")));
        assert_eq!(t.yml, Some(yml));
    }

    #[test]
    fn save_writes_bin_and_yml_that_read_back() {
        let dir = tmp("save");
        let mut list = MList::default();
        let mut m = Motion::default();
        m.game_script = hash40::hash40("game_attack11");
        m.extra = Some(Extra { xlu_start: 1, xlu_end: 4, cancel_frame: 19, no_stop_intp: false });
        list.list.insert(hash40::hash40("attack_11"), m);
        let loaded = Loaded { list, bin: None, yml: None };
        let targets = Targets {
            bin: Some(dir.join("motion_list.bin")),
            yml: Some(dir.join("motion_list.yml")),
            why: None,
        };
        let mut doc = Doc::new(loaded, targets);
        doc.list.list[0].flags.r#move = true;
        assert!(doc.dirty());
        std::fs::create_dir_all(&dir).unwrap();
        let labels = HashMap::from([(hash40::hash40("attack_11").0, "attack_11".to_string())]);
        doc.save(&labels).unwrap();
        assert!(!doc.dirty());
        let bin = motion_lib::open(dir.join("motion_list.bin")).unwrap();
        assert_eq!(bin.list, doc.list.list);
        let yml = crate::motion_yml::load(&dir.join("motion_list.yml")).unwrap();
        assert_eq!(yml.list.list, doc.list.list);
        assert!(!dir.join("motion_list.bin.tmp").exists());
    }

    /// A whole vanilla list through both writers, when the configured dump is on this machine.
    #[test]
    fn a_dump_list_survives_both_writers() {
        let Some(root) = dirs::config_dir()
            .and_then(|d| std::fs::read_to_string(d.join("visionary").join("data_root")).ok())
        else {
            return;
        };
        let src = Path::new(root.trim()).join("fighter/mario/motion/body/c00/motion_list.bin");
        let Ok(list) = motion_lib::open(&src) else {
            return;
        };
        let dir = tmp("dump");
        let bin = dir.join("motion_list.bin");
        motion_lib::save(&bin, &list).unwrap();
        assert_eq!(std::fs::read(&bin).unwrap(), std::fs::read(&src).unwrap());
        let text = crate::motion_yml::write(&list, &HashMap::new());
        let back = crate::motion_yml::parse(&text).unwrap().list;
        assert_eq!(back.motion_path, list.motion_path);
        assert!(back.list.iter().eq(list.list.iter()));
    }

    #[test]
    fn hash_text() {
        assert_eq!(parse_hash("0x10").unwrap().0, Hash40(0x10));
        assert_eq!(parse_hash(" wait ").unwrap(), (hash40::hash40("wait"), Some("wait".into())));
        assert!(parse_hash("0xzz").is_none());
    }
}
