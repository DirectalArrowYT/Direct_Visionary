//! Game-version offsets. Every main offset in this plugin is written as its 13.0.4 value and goes
//! through `o()`, which gives the running build's counterpart:
//! - 13.0.4: unchanged.
//! - 13.0.5: read-only data, data and bss (at 0x39c8000 and above) moved by exactly one page (+0x1000).
//!   Code offsets come from `ver_table.rs`, found by instruction signature (PC-relative fields masked,
//!   exactly one match), regenerated with map_offsets.py whenever an offset is added.
//!   An offset missing from the table returns 0, so the hook or call at it is skipped or fails
//!   visibly instead of landing in the wrong function.

use core::sync::atomic::{AtomicU8, Ordering};

#[path = "ver_table.rs"]
mod table;

#[repr(C)]
struct DisplayVersion {
    name: [u8; 16],
}

extern "C" {
    #[link_name = "_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE"]
    fn get_display_version(out: *mut DisplayVersion);
}

/// 0 unknown, 1 = 13.0.4 (or anything else: offsets as written), 2 = 13.0.5
static VERSION: AtomicU8 = AtomicU8::new(0);

pub fn is_13_0_5() -> bool {
    let mut v = VERSION.load(Ordering::Relaxed);
    if v == 0 {
        let mut d = DisplayVersion { name: [0; 16] };
        unsafe { get_display_version(&mut d) };
        v = if d.name.starts_with(b"13.0.5\0") { 2 } else { 1 };
        VERSION.store(v, Ordering::Relaxed);
    }
    v == 2
}

/// the running build's offset for a 13.0.4 main offset
pub fn o(off: usize) -> usize {
    if !is_13_0_5() {
        return off;
    }
    if off >= 0x39c8000 {
        return off + 0x1000;
    }
    match table::CODE_13_0_5.binary_search_by_key(&off, |&(a, _)| a) {
        Ok(i) => table::CODE_13_0_5[i].1,
        Err(_) => {
            println!("[effect_viewer] no 13.0.5 counterpart for main+{off:#x} (add it with map_offsets.py)");
            0
        }
    }
}
