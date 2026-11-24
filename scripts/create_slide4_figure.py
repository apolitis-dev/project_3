"""Render the implemented single-GPU RAPID-LLM extension as a native diagram."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, GOLD = '#2774AE', '#FFD100'
INK, SLATE, RULE = '#17212B', '#667085', '#E4E7EC'
GRAY = '#F6F7F9'
DPI = 160
matplotlib.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 2560)
ax.set_ylim(1440, 0)
ax.axis('off')
text_items, block_items, connector_items = [], [], []

def text(x, y, s, size=32, weight='normal', color=INK, ha='left', va='baseline', block=None):
    t = ax.text(x, y, s, fontsize=size*72/DPI, weight=weight, color=color,
                ha=ha, va=va, linespacing=1.3)
    text_items.append((t, block))
    return t

def rect(x, y, w, h, fill='white', edge=SLATE, lw=3, name=None):
    p = Rectangle((x,y), w,h, facecolor=fill, edgecolor=edge, linewidth=lw*72/DPI)
    ax.add_patch(p)
    if name:
        block_items.append((name, (x,y,w,h)))
    return p

def line(points, color=BLUE, lw=3.5):
    xs, ys = zip(*points)
    return ax.plot(xs, ys, color=color, lw=lw*72/DPI,
                   solid_capstyle='butt', solid_joinstyle='miter')[0]

def path(points, name, lw=3.5):
    assert all(a[0]==b[0] or a[1]==b[1] for a,b in zip(points, points[1:])), name
    if len(points)>2:
        line(points[:-1], lw=lw)
    p = FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                        mutation_scale=22*72/DPI, linewidth=lw*72/DPI,
                        color=BLUE, shrinkA=0, shrinkB=0,
                        capstyle='butt', joinstyle='miter')
    ax.add_patch(p)
    connector_items.append((name, points))

def changed(x,y,w,h,name):
    rect(x,y,w,h,edge=BLUE,lw=3.5,name=name)
    line([(x+22,y+14),(x+88,y+14)], color=GOLD, lw=6)

# Understated key: component boundaries, rather than large badges, carry meaning.
rect(154,239,34,28,edge=BLUE,lw=3.5)
line([(160,246),(182,246)],color=GOLD,lw=4)
text(208,264,'Project changes',size=30)
rect(546,239,34,28,edge=SLATE,lw=3)
text(600,264,'Existing simulator',size=30)

# One lightly outlined model region contains both branches and their reunion.
rect(1000,330,960,800,edge=RULE,lw=2)
text(1040,385,'RAPID-LLM analytical model',size=42,weight='bold')
text(1040,431,'Shared layer-pattern helper: window_attention.py',size=28,color=SLATE)

# Document-like configuration inputs, with the new settings emphasized in gold.
rect(154,450,420,300,name='model_input')
text(184,507,'Model and workload',size=36,weight='bold',block='model_input')
text(184,560,'Prompt, batch,',size=32,block='model_input')
text(184,603,'generated tokens',size=32,block='model_input')
text(184,660,'Window size and',size=32,block='model_input')
text(184,703,'layer pattern',size=32,block='model_input')
line([(184,715),(433,715)],color=GOLD,lw=5)

rect(154,830,420,230,name='hardware_input')
text(184,881,'Hardware',size=36,weight='bold',block='hardware_input')
text(184,925,'configuration',size=36,weight='bold',block='hardware_input')
text(184,981,'Compute, memory capacity,',size=30,block='hardware_input')
text(184,1022,'bandwidth',size=30,block='hardware_input')

# Parsing gains attention settings; hardware characteristics enter the same path.
changed(650,640,280,270,'parser')
text(790,704,'Read',size=37,weight='bold',ha='center',block='parser')
text(790,751,'configuration',size=37,weight='bold',ha='center',block='parser')
text(790,798,'config.py',size=28,color=SLATE,ha='center',block='parser')
text(790,860,'Attention settings',size=28,ha='center',block='parser')
line([(679,875),(901,875)],color=GOLD,lw=4)

path([(574,600),(612,600),(612,705),(650,705)],'model to parser')
path([(574,945),(612,945),(612,845),(650,845)],'hardware to parser')

# A shared input spine fans out; its only two destinations are the work and memory blocks.
line([(930,775),(1045,775)])
line([(1045,580),(1045,955)])
path([(1045,580),(1100,580)],'configuration to attention')
path([(1045,955),(1100,955)],'configuration to memory')

# Upper branch: changed attention work feeds the reused operator timing model.
changed(1100,470,310,220,'attention')
text(1255,542,'Local/global',size=36,weight='bold',ha='center',block='attention')
text(1255,590,'attention work',size=36,weight='bold',ha='center',block='attention')
text(1255,655,'inference_timing.py',size=28,color=SLATE,ha='center',block='attention')

rect(1460,470,450,220,fill=GRAY,name='operator_cost')
text(1685,553,'Existing',size=36,weight='bold',ha='center',block='operator_cost')
text(1685,603,'operator-cost model',size=36,weight='bold',ha='center',block='operator_cost')
path([(1410,580),(1460,580)],'attention to operator costs')

# Lower branch: resident storage and peak usage are distinct accounting outputs.
changed(1100,820,420,270,'memory')
text(1310,868,'Resident cache and',size=34,weight='bold',ha='center',block='memory')
text(1310,913,'peak memory',size=34,weight='bold',ha='center',block='memory')
line([(1122,931),(1498,931)],color=RULE,lw=2)
text(1310,973,'Resident-cache accounting',size=29,ha='center',block='memory')
text(1310,1014,'Peak-memory estimation',size=29,ha='center',block='memory')
text(1310,1062,'memory_estimation.py',size=28,color=SLATE,ha='center',block='memory')

rect(1600,855,310,190,fill=GRAY,name='execute')
text(1755,916,'Execute and',size=36,weight='bold',ha='center',block='execute')
text(1755,964,'summarize',size=36,weight='bold',ha='center',block='execute')
text(1755,1015,'run_perf.py',size=28,color=SLATE,ha='center',block='execute')
path([(1685,690),(1685,775),(1755,775),(1755,855)],'costs to execution')
path([(1520,955),(1600,955)],'memory to summary')

# A single arrow targets a bracket grouping the three separate predictions.
text(2050,504,'Simulated',size=38,weight='bold')
text(2050,553,'predictions',size=38,weight='bold')
line([(2034,605),(2020,605),(2020,1020),(2034,1020)],color=SLATE,lw=2.5)
text(2050,657,'Peak GPU memory',size=33)
text(2050,807,'History-cache memory',size=33)
text(2050,957,'Generation tokens/s',size=33)
text(2050,1002,'per person',size=33)
path([(1910,955),(1980,955),(1980,760),(2020,760)],'summary to predictions')

text(154,1220,'Python configures runs and plots simulator outputs.',size=34)
text(154,1340,
     'Conceptual mapping of the implemented extension. Single-GPU inference with the analytical backend.',
     size=29,color=SLATE)

# Verify layout constraints in rendered coordinates before delivering the artifact.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
blocks=dict(block_items)
text_boxes=[]
for t,block in text_items:
    b=t.get_window_extent(renderer)
    x0,y0,x1,y1=b.x0,1440-b.y1,b.x1,1440-b.y0
    assert x0 >= 153.6-1 and x1 <= 2406.4+1, (t.get_text(), 'horizontal safe margin', (x0,x1))
    assert y0 >= 86.4-1 and y1 <= 1353.6+1, (t.get_text(), 'vertical safe margin', (y0,y1))
    if block:
        x,y,w,h=blocks[block]
        assert x+12 <= x0 and x1 <= x+w-12 and y+12 <= y0 and y1 <= y+h-12, (t.get_text(), 'block containment')
    text_boxes.append((t.get_text(),(x0,y0,x1,y1)))
for name,points in connector_items:
    for (a,b) in zip(points,points[1:]):
        for label,(x0,y0,x1,y1) in text_boxes:
            if a[0] == b[0]:
                intersects = x0-3 < a[0] < x1+3 and max(min(a[1],b[1]),y0-3) < min(max(a[1],b[1]),y1+3)
            else:
                intersects = y0-3 < a[1] < y1+3 and max(min(a[0],b[0]),x0-3) < min(max(a[0],b[0]),x1+3)
            assert not intersects, (name, 'connector intersects text', label)

for ext in ['svg','png']:
    fig.savefig(OUT/f'slide4_rapid_llm_extension.{ext}',dpi=DPI,facecolor='white',edgecolor='none')
print(json.dumps({'saved':['figures/slide4_rapid_llm_extension.svg','figures/slide4_rapid_llm_extension.png'],
                  'size':[2560,1440],'layout_checks':'passed'}))
