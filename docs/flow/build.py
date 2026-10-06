# Builds the computation-flow page: a swimlane of who moves which bytes, the model's
# stages by unit, and the measured cycle split. Numbers are from the 2026-10 board run.
import base64, io, sys
from pathlib import Path
from PIL import Image

ROOT = Path("/home/armin/Public/tiny-tpu")
OUT = Path(sys.argv[1])

def data_uri(img, size=None):
    if size: img = img.resize(size, Image.BICUBIC)
    b = io.BytesIO(); img.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

img_in = data_uri(Image.open(ROOT / "assets/examples/demo01.jpg").convert("RGB"), (126, 126))
img_out = data_uri(Image.open(ROOT / "assets/results/onboard/demo01.fpga.png").convert("L"))

# ---------------------------------------------------------------- swimlane
L = {"host": 70, "ddr": 215, "cpu": 360, "dma": 495, "buf": 630, "arr": 770, "vec": 905}
heads = [("host", "Host PC", "Python, OpenOCD"), ("ddr", "DDR3", "512 MB on the board"),
         ("cpu", "CV32E40P", "driver in BRAM"), ("dma", "DMA", "in tiny-tpu"),
         ("buf", "Buffers", "ACT·WGT·CFG·OUT"), ("arr", "16×16 array", "GEMM int8→int32"),
         ("vec", "Vector unit", "softmax · LN · LUT")]
W, H = 980, 800
parts = []
def t(x, y, s, cls="lbl", anchor="middle"):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>')
def arrow(a, b, y, label, mover, dashed=False, sub=None):
    x1, x2 = L[a], L[b]
    d = 1 if x2 > x1 else -1
    x2a = x2 - d * 4
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2a}" y2="{y}" class="e {mover}"{dash} marker-end="url(#m-{mover})"/>')
    mx = (x1 + x2) / 2
    t(mx, y - 7, label, f"elbl {mover}-t")
    if sub: t(mx, y + 15, sub, "sub")
def note(lane, y, l1, l2=None, mover="host"):
    x = L[lane]; w = 126; h = 34 if l2 else 22
    parts.append(f'<rect x="{x - w/2}" y="{y - h/2}" width="{w}" height="{h}" rx="3" class="note {mover}-n"/>')
    if l2:
        t(x, y - 3, l1, "nlbl"); t(x, y + 11, l2, "nsub")
    else:
        t(x, y + 4, l1, "nlbl")
def band(y0, y1, label):
    parts.append(f'<rect x="8" y="{y0}" width="{W - 16}" height="{y1 - y0}" class="band"/>')
    t(W - 18, y0 + 15, label, "band-t", "end")

band(66, 196, "ONCE, PER PROGRAM")
band(206, 790, "PER PICTURE · 7.4 s")
for k, name, sub in heads:
    x = L[k]
    parts.append(f'<line x1="{x}" y1="56" x2="{x}" y2="{H - 14}" class="life"/>')
    parts.append(f'<rect x="{x - 62}" y="10" width="124" height="44" rx="4" class="head {k}-h"/>')
    t(x, 29, name, "hlbl"); t(x, 45, sub, "hsub")

note("host", 104, "export_tpu.py", "7,717 descriptors")
arrow("host", "ddr", 150, "① program 26.6 MB", "host", sub="JTAG system bus · 91 s")
note("cpu", 176, "checksum, zero", "13.8 MB arena", "cpu")

note("host", 236, "resize · im2col", "82 × 592 int8")
arrow("host", "ddr", 282, "② input 48.5 KB", "host", sub="into the arena")
arrow("host", "cpu", 318, "RUN_GO", "host", dashed=True)

parts.append(f'<rect x="150" y="336" width="{W - 162}" height="318" rx="6" class="loop"/>')
t(162, 352, "loop over 7,631 descriptors", "loop-t", "start")
arrow("ddr", "cpu", 380, "read descriptor", "cpu", sub="128 B")
arrow("ddr", "buf", 420, "③ operands A, B", "dma", sub="66.8 MB / picture · 36%")
arrow("ddr", "buf", 464, "④ im2col rows", "cpu", sub="head convs only · 21%")
arrow("ddr", "buf", 508, "⑤ bias · mult · shift, tables", "cpu", sub="2%")
arrow("cpu", "arr", 546, "start · poll done", "cpu", dashed=True)
arrow("buf", "arr", 580, "GEMM tile → OUT", "arr", sub="6,616 ops")
arrow("buf", "vec", 618, "vector op → OUT", "vec", sub="1,015 ops · both 25%")
arrow("buf", "ddr", 680, "⑦ result copy OUT → arena", "cpu", sub="17%")

arrow("cpu", "host", 716, "verdict", "cpu", dashed=True, sub="hostio ring")
arrow("ddr", "host", 756, "⑧ depth 254 KB", "host", sub="JTAG read-back")
note("host", 782, "× scale → PNG", None)

defs = "".join(f'<marker id="m-{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{m}-f"/></marker>'
               for m in ("host", "cpu", "dma", "arr", "vec"))
swim = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Swimlane of one run: the host loads the program and the input into DDR3 over JTAG; for each of 7,631 descriptors the CPU reads the descriptor, the DMA or the CPU fills the buffers, the array or vector unit computes, and the CPU copies the result back to DDR3; the host reads the depth map back.">'
        f'<defs>{defs}</defs>{"".join(parts)}</svg>')

# ---------------------------------------------------------------- cycle bars
cyc = {"encoder": [("dma", 113.8), ("arr", 67.8), ("cpu-g", 0.0), ("cpu", 32.1 + 6.4)],
       "head":    [("dma", 14.8), ("arr", 22.1), ("cpu-g", 75.2), ("cpu", 27.5 + 1.1)]}
names = {"dma": "DMA", "arr": "array + vector unit", "cpu-g": "CPU: im2col gather", "cpu": "CPU: copy + config"}
scale = 600 / 220.0
bp = []
y = 34
for stage, segs in cyc.items():
    tot = sum(v for _, v in segs)
    bp.append(f'<text x="0" y="{y + 15}" class="blbl">{stage}</text>')
    x = 90
    for k, v in segs:
        if v <= 0: continue
        w = v * scale
        bp.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" class="seg {k}-s"/>')
        if w > 46: bp.append(f'<text x="{x + w / 2:.1f}" y="{y + 15}" class="segl" text-anchor="middle">{v:.0f} M</text>')
        x += w
    bp.append(f'<text x="{x + 8:.1f}" y="{y + 15}" class="btot">{tot:.0f} M cycles · {tot / 50:.2f} s</text>')
    y += 40
for i, v in enumerate([0, 50, 100, 150, 200]):
    gx = 90 + v * scale
    bp.append(f'<line x1="{gx:.1f}" y1="26" x2="{gx:.1f}" y2="{y - 10}" class="grid"/>')
    bp.append(f'<text x="{gx:.1f}" y="18" class="tick" text-anchor="middle">{v} M</text>')
lx = 90
for k in ("dma", "arr", "cpu-g", "cpu"):
    bp.append(f'<rect x="{lx}" y="{y + 4}" width="12" height="12" class="seg {k}-s"/>')
    bp.append(f'<text x="{lx + 18}" y="{y + 14}" class="tick" text-anchor="start">{names[k]}</text>')
    lx += 34 + len(names[k]) * 6.6
bars = (f'<svg viewBox="0 0 900 {y + 26}" role="img" aria-label="Cycles per picture: encoder 220 million, mostly DMA; head 140 million, mostly the CPU gather.">{"".join(bp)}</svg>')

html = (Path(__file__).parent / "template.html").read_text()
html = html.replace("{{SWIM}}", swim).replace("{{BARS}}", bars).replace("{{IMG_IN}}", img_in).replace("{{IMG_OUT}}", img_out)
OUT.write_text(html)
print("wrote", OUT, len(html), "bytes")
