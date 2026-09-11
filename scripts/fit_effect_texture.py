#!/usr/bin/env python3
"""Resize an outside effect texture to the exact size a Smash pool texture needs.

Visionary's importer will not resize for you, and it is right not to: emitters carry UV
rects authored against the texture's size, so a silent rescale would move every sprite-sheet
cell. This does the resize explicitly, and preserves alpha, which is where most effect
textures keep their shape.

    python fit_effect_texture.py <in.png> <width> <height> [-o out.png] [--pad]

By default the image is stretched to the target size. --pad instead fits it inside the target
and centres it on transparency, which is what you want when the source is a different aspect
ratio and stretching would distort the shape.

Find the target size with:
    VISIONARY_TEX_ALL=1 VISIONARY_EFF_FILE=<eff> \
      cargo test --bin visionary what_texture_templates_a_file_offers -- --nocapture
"""
import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("needs Pillow:  pip install pillow")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("-o", "--out")
    ap.add_argument(
        "--pad",
        action="store_true",
        help="fit inside the target and centre on transparency instead of stretching",
    )
    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0:
        return print("width and height must be positive") or 1

    # RGBA throughout: an effect texture's shape usually lives in alpha, and dropping it on
    # the way through is the failure that renders as a solid square.
    image = Image.open(args.source).convert("RGBA")
    target = (args.width, args.height)

    if args.pad:
        scale = min(args.width / image.width, args.height / image.height)
        inner = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
        canvas = Image.new("RGBA", target, (0, 0, 0, 0))
        canvas.paste(inner, ((args.width - inner.width) // 2, (args.height - inner.height) // 2))
        out_image = canvas
    else:
        out_image = image.resize(target, Image.LANCZOS)

    out = args.out
    if not out:
        stem, ext = os.path.splitext(args.source)
        out = f"{stem}_{args.width}x{args.height}{ext}"
    out_image.save(out)
    print(f"{image.width}x{image.height} -> {args.width}x{args.height}  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
