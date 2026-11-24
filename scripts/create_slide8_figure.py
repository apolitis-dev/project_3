"""Create the slide 8 figure from full-precision main-experiment results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
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

results = json.loads((ROOT / "runs/main_experiment/results.json").read_text())
rows = [r for r in results if r["prompt_tokens"] == 32768 and r["batch_size"] == 1]
full = next(r for r in rows if r["window"] is None)
windows = [512, 1024, 2048, 4096]
variants = [next(r for r in rows if r["window"] == w) for w in windows]
assert len(rows) == 5
assert all(r["decode_tokens"] == 256 and r["total_tokens"] == 33024 and r["fits"]
           for r in rows)
assert all((r["local_layers"], r["global_layers"]) == (24, 8) for r in variants)
assert (full["local_layers"], full["global_layers"]) == (0, 32)
assert all(r["hardware"] == full["hardware"] for r in variants)
assert full["hardware"]["sw_param"]["precision"]["tensor_format"] == "bf16"

expected = np.array([
    [1.0546875, 24.353515625, 79.12396050285643],
    [1.1015625, 24.400390625, 78.9649691002804],
    [1.1953125, 24.494140625, 78.64889548302388],
    [1.3828125, 24.681640625, 78.02427886289433],
])
keys = ["kv_resident_gib", "peak_gib", "per_user_tok_s"]
actual = np.array([[r[k] for k in keys] for r in variants])
np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
np.testing.assert_allclose([full[k] for k in keys],
                           [4.03125, 27.306640625, 70.18565815748944],
                           rtol=1e-12, atol=0)
percentages = [100 * (1 - actual[0, i] / full[k]) for i, k in enumerate(keys[:2])]
percentages.append(100 * (actual[0, 2] / full[keys[2]] - 1))

fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor="white")

def text(x, y, value, size=32, weight="normal", color=INK, **kwargs):
    return fig.text(x / 2560, 1 - y / 1440, value, fontsize=size * PX,
                    weight=weight, color=color, va="top", linespacing=1.3, **kwargs)

panels = [
    (290, "History memory", "History-cache memory (GiB)", 5, [0, 1, 2, 3, 4, 5],
     f"{percentages[0]:.1f}% less history memory", "GiB", 1),
    (1080, "Total GPU memory", "Peak memory (GiB)", 33, [0, 10, 20, 30],
     f"{percentages[1]:.1f}% less total memory", "GiB", -1),
    (1870, "Generation per person", "Tokens/s per person", 100, [0, 20, 40, 60, 80, 100],
     f"{percentages[2]:.1f}% faster generation", "tokens/s", 1),
]
axes = []
for i, (x, heading, ylabel, maximum, ticks, result, unit, direction) in enumerate(panels):
    center = x + 265
    text(center, 150, heading, size=44, weight="bold", ha="center")
    text(center, 235, result, size=38, weight="bold", ha="center")
    text(center, 291, "Window 512 versus full attention", size=29, color=SLATE, ha="center")
    ax = fig.add_axes([x / 2560, (1440 - 975) / 1440, 530 / 2560, 530 / 1440])
    axes.append(ax)
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.set_xscale("log", base=2)
    ax.set_xlim(420, 5000)
    ax.xaxis.set_major_locator(FixedLocator(windows))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, maximum)
    ax.set_yticks(ticks)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, linewidth=2 * PX)
    ax.tick_params(axis="both", labelsize=32 * PX, length=0, pad=12 * PX)
    ax.set_ylabel(ylabel, fontsize=36 * PX, labelpad=22 * PX)
    ax.set_xlabel("Window size (positions)", fontsize=36 * PX, labelpad=22 * PX)
    text(center, 1085, "Log base 2 · sampled inputs", size=29, color=SLATE, ha="center")

    key = keys[i]
    ref = full[key]
    ax.axhline(ref, color=BLUE, linewidth=3.5 * PX, linestyle=(0, (8, 5)))
    ax.plot(windows, actual[:, i], color=GOLD, linewidth=4 * PX,
            marker="D", markersize=15 * PX, markerfacecolor=GOLD,
            markeredgecolor=BLUE, markeredgewidth=2.8 * PX, zorder=3)
    # Direct labels use opposite sides where the two series are close.
    full_above = i != 2
    ax.annotate(f"Full attention: {ref:.2f} {unit}", (480, ref),
                xytext=(0, (15 if full_above else -17) * PX),
                textcoords="offset points", ha="left",
                va="bottom" if full_above else "top", color=BLUE,
                fontsize=30 * PX,
                bbox=dict(facecolor="white", edgecolor="none", pad=2 * PX))
    ax.annotate(f"{actual[0, i]:.2f} {unit}", (512, actual[0, i]),
                xytext=(12 * PX, direction * 21 * PX), textcoords="offset points",
                ha="left", va="bottom" if direction > 0 else "top",
                fontsize=32 * PX, weight="bold", color=INK)

text(1280, 365, "Further shrinking the window gives modest extra gains.", size=34, ha="center")
legend = fig.legend(handles=[
    Line2D([], [], color=BLUE, linewidth=3.5 * PX, linestyle=(0, (8, 5)),
           label="Full attention · 32 global layers"),
    Line2D([], [], color=GOLD, linewidth=4 * PX, marker="D", markersize=15 * PX,
           markeredgecolor=BLUE, markeredgewidth=2.8 * PX,
           label="Windowed · 24 local / 8 global layers"),
], loc="upper center", bbox_to_anchor=(0.5, 1 - 1163 / 1440),
    ncol=2, frameon=False, fontsize=32 * PX, handlelength=2.7,
    columnspacing=2.5, borderaxespad=0)
fig.add_artist(Line2D([154 / 2560, 2406 / 2560], [1 - 1240 / 1440] * 2,
                      transform=fig.transFigure, color=RULE, linewidth=2 * PX))
text(154, 1270,
     "RAPID-LLM simulation. Prompt 32,768, generated tokens 256, batch 1, fixed 24-local/8-global pattern.",
     size=31)
text(154, 1318,
     "Main repository A100 profile · BF16 · Final available positions: 33,024", size=29, color=SLATE)

# Check visible text against the 6% safe margin before exporting.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [*fig.texts, legend]
for ax in axes:
    texts.extend([ax.xaxis.label, ax.yaxis.label, *ax.texts,
                  *ax.get_xticklabels(), *ax.get_yticklabels()])
for item in texts:
    box = item.get_window_extent(renderer)
    assert box.x0 >= 153 and box.x1 <= 2407, (item, box)
    assert box.y0 >= 85 and box.y1 <= 1355, (item, box)

stem = OUT / "slide8_window_size_tradeoffs"
fig.savefig(stem.with_suffix(".png"), dpi=DPI, facecolor="white", transparent=False)
fig.savefig(stem.with_suffix(".svg"), facecolor="white", transparent=False)
plt.close(fig)
print(json.dumps({"png": str(stem.with_suffix(".png")),
                  "svg": str(stem.with_suffix(".svg")),
                  "percentages_at_window_512": percentages}, indent=2))
