"""Create the slide 6 experiment-design figure as editable SVG and PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE, GOLD = "#2774AE", "#FFD100"
INK, SLATE, RULE = "#17212B", "#667085", "#E4E7EC"
DPI = 160
matplotlib.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})
fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 2560), ylim=(1440, 0))
ax.axis("off")
items, emphasis = [], []


def text(x, y, value, size=36, weight="normal", color=INK, bounds=None):
    item = ax.text(x, y, value, fontsize=size * 72 / DPI, color=color,
                   weight=weight, va="baseline", ha="left")
    items.append((item, bounds))
    return item


def rule(y, color=RULE, width=2):
    ax.plot([154, 2406], [y, y], color=color, lw=width * 72 / DPI,
            solid_capstyle="butt")


# Two lines describe the main variant; the external all-local case is below.
text(154, 124, "Llama 3.1 8B dimensions, BF16, one simulated GPU", 44, "bold")
text(154, 184, "32 layers: 24 local and 8 global, 256 generated tokens", 38)

TOP, HEADER, ROW = 230, 82, 134
LEFT, RIGHT = 154, 2406
COLS = (154, 610, 1465, 2406)
PADDING = 28
BOTTOM = TOP + HEADER + 6 * ROW
ax.add_patch(Rectangle((LEFT, TOP), RIGHT - LEFT, HEADER,
                       facecolor=BLUE, edgecolor="none"))
for index, label in enumerate(("Study", "Input changed", "Held fixed")):
    text(COLS[index] + PADDING, TOP + 55, label, 38, "bold", "white",
         (COLS[index], TOP, COLS[index + 1], TOP + HEADER))

rows = [
    ("Prompt length", ("512, 2,048, 8,192, 32,768", "tokens"),
     ("Batch 1, window 1,024,", "same main GPU profile")),
    ("Window size", ("512, 1,024, 2,048, 4,096", "positions"),
     ("Prompt 32,768, batch 1,", "same main GPU profile")),
    ("Batch size", ("1, 2, 4, 8, 16, 24, 32", "conversations"),
     ("Prompt 8,192, window 1,024,", "same main GPU profile")),
    ("GPU-like profile", ("A100, H100, H200",),
     ("Prompt 32,768, batch 1,", "window 1,024")),
    ("HBM bandwidth", ("0.5×, 1×, 2×, 4×",),
     ("Reference compute and capacity,", "same workload")),
    ("Capacity limit", ("40, 80, 141 GB",),
     ("Reference compute and bandwidth,", "prompt 8,192, window 1,024")),
]

for index, (study, changed, fixed) in enumerate(rows):
    y0 = TOP + HEADER + index * ROW
    y1 = y0 + ROW
    center = (y0 + y1) / 2
    text(COLS[0] + PADDING, center + 12, study, 36, "bold",
         bounds=(COLS[0], y0, COLS[1], y1))
    for column, lines in ((1, changed), (2, fixed)):
        baseline = center + 12 if len(lines) == 1 else center - 13
        for line_index, value in enumerate(lines):
            artist = text(COLS[column] + PADDING, baseline + 48 * line_index,
                          value, 36, bounds=(COLS[column], y0, COLS[column + 1], y1))
            if column == 1 and line_index == 0:
                emphasis.append((artist, baseline + 10))
    rule(y1)

# Output quantities are visibly separate from the input table.
text(154, 1182, "Outputs", 36, "bold", BLUE)
text(456, 1182, "History memory", 36)
text(1040, 1182, "Peak GPU memory", 36)
text(1720, 1182, "Generation per person", 36)

text(154, 1257,
     "Separate reference cases: all-local Mistral cache and published A100 baseline timing.",
     32)
text(154, 1325,
     "Controlled RAPID-LLM simulations. Change one input while holding the comparison settings fixed.",
     30, color=SLATE)

# Gold underlines identify the changed values without sacrificing contrast.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist, y in emphasis:
    box = artist.get_window_extent(renderer)
    ax.plot([box.x0, box.x1], [y, y], color=GOLD, lw=5 * 72 / DPI,
            solid_capstyle="butt")

# Check actual glyph extents, including cell containment and the 6% safe area.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
boxes = []
for artist, cell in items:
    b = artist.get_window_extent(renderer)
    box = (b.x0, 1440 - b.y1, b.x1, 1440 - b.y0)
    x0, y0, x1, y1 = box
    assert 153.6 <= x0 and x1 <= 2406.4, (artist.get_text(), "horizontal margin")
    assert 86.4 <= y0 and y1 <= 1353.6, (artist.get_text(), "vertical margin")
    if cell:
        a, b, c, d = cell
        assert a + 20 <= x0 and x1 <= c - 20, (artist.get_text(), "column overflow")
        assert b + 15 <= y0 and y1 <= d - 15, (artist.get_text(), "row overflow")
    for previous, (a, b, c, d) in boxes:
        assert min(x1, c) <= max(x0, a) or min(y1, d) <= max(y0, b), (
            previous, artist.get_text(), "text overlap")
    boxes.append((artist.get_text(), box))

for extension in ("svg", "png"):
    path = OUT / f"slide6_experiment_setup.{extension}"
    fig.savefig(path, dpi=DPI, facecolor="white", edgecolor="none")
    print(path)
print("Passed: 2560 × 1440; safe margins; cell containment; no text overlaps.")
