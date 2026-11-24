"""Plot slide 7 from saved full-precision results; export PNG and editable SVG."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
results = json.loads((ROOT / "runs/published_comparison/results.json").read_text())
reference = results["published"]
BLUE, GOLD = "#2774AE", "#FFD100"
INK, SLATE, RULE = "#17212B", "#667085", "#E4E7EC"
DPI = 160
PX = 72 / DPI
plt.rcParams.update({
    "font.family": "Arial", "svg.fonttype": "none",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": SLATE, "axes.linewidth": 2.5 * PX,
    "font.size": 32 * PX,
})
fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor="white")

def text(x, y, value, size=32, weight="normal", color=INK, **kwargs):
    return fig.text(x / 2560, 1 - y / 1440, value,
                    fontsize=size * PX, weight=weight, color=color,
                    va="top", linespacing=1.35, **kwargs)

def axis(x, width, maximum, ticks, ylabel):
    ax = fig.add_axes([x / 2560, (1440 - 950) / 1440, width / 2560, 550 / 1440])
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.set_ylim(0, maximum)
    ax.set_yticks(ticks)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, linewidth=2 * PX)
    ax.tick_params(axis="both", labelsize=32 * PX, length=0, pad=12 * PX)
    ax.set_ylabel(ylabel, fontsize=36 * PX, labelpad=23 * PX, linespacing=1.2)
    return ax

# Panel headings identify the evidence, without a slide headline.
text(154, 150, "History-cache reduction", size=46, weight="bold")
text(154, 232, "Mistral 7B, all-local", size=34)
text(154, 283, "W = 4,096, context = 32,768", size=32)
text(1130, 150, "Published A100 timing", size=46, weight="bold")
text(1130, 232, "Llama 3.1 8B Instruct · A100 SXM · Batch 1", size=34)

full = next(r for r in results["memory"] if r["window"] is None)
window = next(r for r in results["memory"] if r["window"] == 4096)
ratio = full["kv_resident_bytes"] / window["kv_resident_bytes"]
assert full["total_tokens"] == window["total_tokens"] == 32768
assert full["kv_resident_gib"] == 4 and window["kv_resident_gib"] == 0.5
assert ratio == reference["memory"]["reported_cache_reduction"] == 8
left = axis(295, 630, 10, np.arange(0, 11, 2), "Reduction factor\nversus full attention")
bars = left.bar([0, 1], [reference["memory"]["reported_cache_reduction"], ratio],
                width=0.56, color=[BLUE, GOLD], edgecolor=BLUE, linewidth=3 * PX)
left.set_xlim(-0.65, 1.65)
left.set_xticks([0, 1], ["Published paper", "RAPID-LLM"])
for bar in bars:
    left.annotate("8×", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  xytext=(0, 15 * PX), textcoords="offset points", ha="center",
                  va="bottom", fontsize=42 * PX, weight="bold", color=INK)

# Source order is matched by actual token counts rather than nominal shapes.
rows = reference["timing"]["rows"]
by_shape = {(r["prompt_tokens"], r["decode_tokens"]): r for r in results["timing"]}
published = np.array([r["wall_s"] for r in rows])
predicted = np.array([
    by_shape[(r["input_tokens"], r["output_tokens"])]["prefill_s"]
    + by_shape[(r["input_tokens"], r["output_tokens"])]["decode_s"]
    for r in rows
])
errors = (predicted / published - 1) * 100
mae, largest = np.mean(np.abs(errors)), np.max(np.abs(errors))
assert np.allclose(errors, results["timing_errors_percent"], rtol=1e-10, atol=1e-10)
assert round(mae, 1) == 9.7 and round(largest, 1) == 24.6
assert all(r["window"] is None for r in results["timing"])
right = axis(1335, 1070, 9, [0, 2, 4, 6, 8, 9], "End-to-end time (seconds)")
x, width = np.arange(3), 0.30
for positions, values, color in [(x - width / 2, published, BLUE),
                                  (x + width / 2, predicted, GOLD)]:
    bars = right.bar(positions, values, width=width, color=color,
                     edgecolor=BLUE, linewidth=3 * PX)
    for bar, value in zip(bars, values):
        right.annotate(f"{value:.2f}",
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 13 * PX), textcoords="offset points", ha="center",
                       va="bottom", fontsize=36 * PX, weight="bold", color=INK)
right.set_xlim(-0.58, 2.58)
right.set_xticks(x, [f'{r["input_tokens"]:,} input\n{r["output_tokens"]:,} output' for r in rows])
legend = fig.legend(
    handles=[Patch(facecolor=BLUE, edgecolor=BLUE, label="Published measurement"),
             Patch(facecolor=GOLD, edgecolor=BLUE, linewidth=3 * PX,
                   label="RAPID-LLM prediction")],
    loc="upper left", bbox_to_anchor=(1130 / 2560, 1 - 315 / 1440),
    ncol=2, frameon=False, fontsize=31 * PX,
    handlelength=1.2, handleheight=0.85, columnspacing=1.5, borderaxespad=0,
)
text(154, 1080, "Simulated cache: 4.00 GiB full,\n0.50 GiB window.", size=32)
text(1130, 1080, f"Mean absolute error: {mae:.1f}%", size=34, weight="bold")
text(1130, 1134, f"Largest error: {largest:.1f}%", size=34)
fig.add_artist(plt.Line2D([154 / 2560, 2406 / 2560], [1 - 1220 / 1440] * 2,
                         transform=fig.transFigure, color=RULE, linewidth=2 * PX))
text(154, 1252,
     "Cache comparison reproduces a published ratio. Timing compares the full-attention baseline.",
     size=32)
text(154, 1311,
     "Windowed speed predictions remain unverified by direct hardware measurements.", size=32)

# Validate visible text remains inside the requested 6% safe margin.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for item in [*fig.texts, legend]:
    box = item.get_window_extent(renderer)
    assert box.x0 >= 153 and box.x1 <= 2407, (item, box)
    assert box.y0 >= 85 and box.y1 <= 1355, (item, box)

stem = OUT / "slide7_published_comparison"
fig.savefig(stem.with_suffix(".png"), dpi=DPI, facecolor="white")
fig.savefig(stem.with_suffix(".svg"), facecolor="white")
plt.close(fig)
print(json.dumps({"png": str(stem.with_suffix(".png")),
                  "svg": str(stem.with_suffix(".svg")),
                  "predicted_seconds": predicted.tolist(),
                  "signed_errors_percent": errors.tolist(),
                  "mean_absolute_error_percent": float(mae)}, indent=2))
