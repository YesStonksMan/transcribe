"""Generate the Transcribe app icon as a .icns bundle.

Design: claymorphism, matching static/index.html — a single raised violet
clay squircle (no outer plate) with the Phosphor "waveform"
(bold, MIT) in white.

The icon is authored as HTML/CSS (same shadow language as the UI) and
rasterised by headless Chrome at 1024px, then downsampled for the .iconset.

Usage:
    python make_icon.py
    # Produces icon.icns + icon_preview.png at the project root.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
ICNS = ROOT / "icon.icns"
PREVIEW = ROOT / "icon_preview.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
M = 1024

# Phosphor waveform-bold, verbatim.
WAVEFORM = (
    '<svg viewBox="0 0 256 256" fill="none" stroke="#fff" stroke-linecap="round" stroke-width="24">'
    '<line x1="48" y1="96" x2="48" y2="160"/><line x1="88" y1="32" x2="88" y2="224"/>'
    '<line x1="128" y1="64" x2="128" y2="192"/><line x1="168" y1="96" x2="168" y2="160"/>'
    '<line x1="208" y1="80" x2="208" y2="176"/></svg>'
)

# macOS icon grid: 824px body centred in 1024 with room for the drop shadow.
HTML = f"""<!doctype html><html><head><style>
html,body{{margin:0;width:{M}px;height:{M}px;background:transparent;overflow:hidden;}}
.tile{{position:absolute;left:100px;top:92px;width:824px;height:824px;border-radius:186px;
  display:grid;place-items:center;
  background:linear-gradient(150deg,#a89ef9 0%,#7b6cf0 50%,#5b4bd8 100%);
  box-shadow:0 26px 40px rgba(70,50,190,.35),
    inset 22px 24px 44px rgba(255,255,255,.40),
    inset -28px -34px 52px rgba(40,24,140,.50);}}
.tile svg{{width:470px;height:470px;filter:drop-shadow(0 10px 12px rgba(40,24,140,.45));}}
</style></head><body><div class="tile">{WAVEFORM}</div></body></html>"""


def render_master() -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "icon.html"
        png = Path(tmp) / "icon.png"
        html.write_text(HTML, encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--default-background-color=00000000", "--force-device-scale-factor=1",
             f"--window-size={M},{M}", f"--screenshot={png}", html.as_uri()],
            check=True, capture_output=True, timeout=60,
        )
        return Image.open(png).convert("RGBA").copy()


def main() -> int:
    if not shutil.which("iconutil") or not Path(CHROME).exists():
        print("✗ needs macOS iconutil + Google Chrome", file=sys.stderr)
        return 1

    master = render_master()
    master.resize((512, 512), Image.LANCZOS).save(PREVIEW, "PNG")

    with tempfile.TemporaryDirectory(suffix=".iconset") as iconset:
        sizes = [
            (16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
            (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
            (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
            (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
            (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png"),
        ]
        for size, name in sizes:
            master.resize((size, size), Image.LANCZOS).save(Path(iconset) / name, "PNG")
        subprocess.run(["iconutil", "-c", "icns", "-o", str(ICNS), iconset], check=True)

    print(f"✓ {ICNS} ({ICNS.stat().st_size // 1024} KB) — preview: {PREVIEW.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
