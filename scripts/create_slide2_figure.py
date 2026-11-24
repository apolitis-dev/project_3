"""Create the slide 2 figure from fixed, shared diagram geometry."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, GOLD = '#2774AE', '#FFD100'
INK, SLATE, RULE = '#17212B', '#667085', '#E4E7EC'
PALE, GRAY = '#E8F2F9', '#F1F3F6'
DPI = 160
matplotlib.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 2560)
ax.set_ylim(1440, 0)
ax.axis('off')

def text(x, y, s, size=32, weight='normal', color=INK, ha='left', va='baseline', **kw):
    return ax.text(x, y, s, fontsize=size*72/DPI, weight=weight, color=color,
                   ha=ha, va=va, linespacing=1.3, **kw)

def rect(x, y, w, h, fill='white', edge=BLUE, lw=3.5):
    p = Rectangle((x,y), w,h, facecolor=fill, edgecolor=edge, linewidth=lw*72/DPI)
    ax.add_patch(p)
    return p

def line(xs, ys, color=BLUE, lw=3.5, **kw):
    return ax.plot(xs, ys, color=color, lw=lw*72/DPI, solid_capstyle='butt', **kw)[0]

def arrow(x1, y1, x2, y2, lw=4, color=BLUE):
    p = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle='-|>',
                        mutation_scale=24*72/DPI, linewidth=lw*72/DPI,
                        color=color, shrinkA=0, shrinkB=0,
                        joinstyle='miter', capstyle='butt')
    ax.add_patch(p)
    return p

# Every row uses these exact dimensions; only the stored-token count changes.
MX, MW, MH = 510, 1896, 200
TX, PITCH, TW, TH = 550, 174, 134, 110
WEIGHTS_X, WEIGHTS_W = 2118, 264
row_geometry = []
for title, top, n in [('Short conversation', 290, 4), ('Longer conversation', 710, 8)]:
    text(154, top-122, title, size=44, weight='bold')
    text(154, top-50, 'Token positions', size=32, color=SLATE)
    text(154, top+15, 'GPU memory', size=28, color=SLATE)
    # Fixed outlines and unchanged weights regions.
    rect(MX, top, MW, MH)
    line([2090,2090], [top,top+MH], color=RULE, lw=3)
    rect(WEIGHTS_X, top+22, WEIGHTS_W, 156, fill=GRAY, edge=RULE, lw=2)
    text(WEIGHTS_X+WEIGHTS_W/2, top+87, 'Weights and', size=30, ha='center')
    text(WEIGHTS_X+WEIGHTS_W/2, top+128, 'working data', size=30, ha='center')
    centers = [TX+i*PITCH+TW/2 for i in range(n)]
    timeline_y = top-64
    arrow(centers[0], timeline_y, centers[-1]+88, timeline_y, lw=3)
    for i, cx in enumerate(centers):
        rect(cx-34, timeline_y-28, 68, 56, lw=3)
        text(cx, timeline_y+1, f't{i+1}', size=30, ha='center', va='center')
        line([cx,cx], [timeline_y+31,top+16], color=SLATE, lw=2, dashes=(4,4))
        x = TX+i*PITCH
        rect(x, top+22, TW, TH, fill=PALE)
        line([x,x+TW], [top+22+TH/2]*2, lw=3)
        text(cx, top+22+TH/4, 'K', size=34, ha='center', va='center')
        text(cx, top+22+3*TH/4, 'V', size=34, ha='center', va='center')
    text(TX, top+174, 'Conversation-history cache', size=33, weight='bold', color=BLUE)
    row_geometry.append({'count': n, 'outline': [MX,top,MW,MH], 'tile_size': [TW,TH],
                         'pitch': PITCH, 'centers': centers,
                         'weights': [WEIGHTS_X, top+22, WEIGHTS_W,156]})

# One short bracket spans K AND V, explaining the pair as one token.
line([540,526,526,540], [312,312,422,422], color=SLATE, lw=2.5)
line([470,526], [367,367], color=SLATE, lw=2.5)
text(154, 353, 'Saved information', size=30)
text(154, 393, 'for one token', size=30)

# Capacity annotation is attached to the boundary of the fixed GPU memory.
text(1560, 218, 'Capacity: how much fits', size=33)
line([1645,1645,1695], [233,260,290], color=SLATE, lw=2.5)

text(154, 980, 'More conversations create additional caches.', size=34)

# A separate primary READ path flows from history to the shared generation step.
line([1040,1040], [910,1235], lw=5)
arrow(1040,1235,1660,1235,lw=5)
text(1110, 1199, 'Read saved history', size=35, weight='bold', color=BLUE)
text(154, 1092, 'Bandwidth: how quickly data moves', size=33)
line([770,934,1040], [1106,1106,1150], color=SLATE, lw=2.5)

rect(1660,1170,746,130, lw=3.5)
text(2033,1235,'Generate the next token',size=38,weight='bold',ha='center',va='center')

# New pair sits OUTSIDE the existing cache, along an independent return path.
NX, NY = 1942, 986
rect(NX,NY,TW,TH,fill=GOLD)
line([NX,NX+TW],[NY+TH/2]*2,lw=3)
text(NX+TW/2,NY+TH/4,'K',size=34,ha='center',va='center')
text(NX+TW/2,NY+3*TH/4,'V',size=34,ha='center',va='center')
arrow(NX+TW/2,1170,NX+TW/2,NY+TH+2,lw=3)
arrow(NX+TW/2,NY-2,NX+TW/2,912,lw=3)
text(2120,1034,'Store the new',size=32)
text(2120,1077,'K and V',size=32)

text(154,1340,'Concept illustration. Tile counts are illustrative, not measured memory usage.',
     size=29,color=SLATE)

# Invariants guard the conceptual comparison rather than rendering details.
a,b = row_geometry
assert a['tile_size'] == b['tile_size'] and a['pitch'] == b['pitch']
assert a['centers'] == b['centers'][:4]
assert a['outline'][2:] == b['outline'][2:]
assert a['weights'][0] == b['weights'][0] and a['weights'][2:] == b['weights'][2:]
for ext in ['svg','png']:
    fig.savefig(OUT/f'slide2_conversation_history_gpu_memory.{ext}',dpi=DPI,
                facecolor='white',edgecolor='none')
print('Saved SVG and 2560 × 1440 PNG in',OUT)
