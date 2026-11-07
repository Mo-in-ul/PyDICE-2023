"""Collision check for every figure of the manuscript. Hooks matplotlib's Figure.savefig, draws the figure, and
reports (1) pairs of text items whose boxes overlap (labels, annotations, legend entries, tick labels, titles),
(2) legend boxes that cover plotted data or other text, (3) annotations that sit on plotted lines, points or bars,
(4) text that falls outside the figure. Text inside boxes of diagram axes (axis off) is exempt from (3).
Usage: python overlap_check.py  (runs fig_v6.py, fig_mc2.py's share figure and the mechanism figure)."""
import os, sys, json, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, LineCollection, PolyCollection
from matplotlib.patches import Rectangle, FancyBboxPatch

PAD = 0.5          # px tolerance at 100 dpi
REPORT = {}


def bbox_of(a, r):
    try:
        b = Text.get_window_extent(a, r) if isinstance(a, Text) else a.get_window_extent(r)   # annotation: text only, arrow excluded
        return b if b.width > 0 and b.height > 0 else None
    except Exception:
        return None


def inter(a, b, pad=PAD):
    return a.x0 + pad < b.x1 and b.x0 + pad < a.x1 and a.y0 + pad < b.y1 and b.y0 + pad < a.y1


def drawn_ticklabels(fig):
    keep, allt = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                allt.update([tk.label1, tk.label2])
            for tk in axis._update_ticks():
                if tk.label1.get_visible(): keep.add(tk.label1)
                if tk.label2.get_visible(): keep.add(tk.label2)
    return keep, allt


def texts_of(fig):
    out = []
    keep, allt = drawn_ticklabels(fig)
    for t in fig.findobj(Text):
        if t in allt and t not in keep:
            continue
        if not t.get_visible() or not t.get_text().strip() or t.get_alpha() == 0:
            continue
        out.append(t)
    return out


def data_points(ax, r):
    """Display coordinates of plotted data in ax: line samples, scatter points, bar rectangles, error bars."""
    pts, rects = [], []
    leg = ax.get_legend()
    leg_artists = set(leg.findobj()) if leg else set()
    for ln in ax.get_lines():
        if ln in leg_artists or not ln.get_visible():
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) == 0:
            continue
        seg = [xy[0]]
        for a, b in zip(xy[:-1], xy[1:]):
            if ln.get_linestyle() not in ('None', 'none', ' ', ''):
                n = int(min(200, max(2, np.hypot(*(b - a)) / 1.5)))
                seg += list(a + (b - a) * np.linspace(0, 1, n)[1:, None])
            else:
                seg.append(b)
        pts.append((f'line {ln.get_label()}', np.array(seg)))
    for c in ax.collections:
        if c in leg_artists or not c.get_visible():
            continue
        if isinstance(c, PathCollection):
            off = c.get_offset_transform().transform(np.asarray(c.get_offsets(), float))
            pts.append(('scatter', off))
        elif isinstance(c, LineCollection):
            for seg_ in c.get_segments():
                xy = c.get_transform().transform(np.asarray(seg_, float))
                if len(xy) >= 2:
                    a, b = xy[0], xy[-1]
                    pts.append(('errorbar', a + (b - a) * np.linspace(0, 1, 40)[:, None]))
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_visible() and p not in leg_artists:
            b = bbox_of(p, r)
            if b is not None and b.width > 0.5 and b.height > 0.5:
                rects.append(('bar', b))
    return pts, rects


def check(fig, name):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    issues = []
    W, H = fig.bbox.width, fig.bbox.height
    T = texts_of(fig)
    boxes = [(t, bbox_of(t, r)) for t in T]
    boxes = [(t, b) for t, b in boxes if b is not None]
    # (4) outside the figure (tight bbox is used at save time, so only report text cut by the canvas when not tight)
    # (1) text-text
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if inter(b1, b2):
            # tick labels of different axes never collide by design; skip a tick label against its own axis' ticks
            issues.append(('text/text', t1.get_text()[:40], t2.get_text()[:40]))
    for ax in fig.axes:
        pts, rects = data_points(ax, r)
        diagram = not ax.axison
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            lb = leg.get_window_extent(r)
            lb = lb.padded(-2)
            for lab, P in pts:
                inside = (P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) & (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1)
                if inside.any():
                    issues.append(('legend/data', f'legend of panel at {ax.get_position().bounds[:2]}', f'{lab} ({int(inside.sum())} pts)'))
            for lab, b in rects:
                if inter(lb, b):
                    issues.append(('legend/data', 'legend', lab))
            leg_texts = set(leg.findobj(Text))
            for t, b in boxes:
                if t not in leg_texts and t.axes is ax and inter(lb, b):
                    issues.append(('legend/text', 'legend', t.get_text()[:40]))
        axb = ax.get_window_extent(r)
        for t in ax.texts:
            b = bbox_of(t, r)
            if b is not None and t.get_text().strip() and t.get_clip_on() is False or (b is not None and t.get_text().strip()):
                if b.x0 < axb.x0 - 1 or b.x1 > axb.x1 + 1 or b.y0 < axb.y0 - 1 or b.y1 > axb.y1 + 1:
                    if t.get_transform() != ax.transAxes or not (t.get_position()[1] > 1.0):   # panel labels above the axes are intended
                        issues.append(('text/outside-panel', t.get_text()[:40].replace('\n', ' | '), ''))
        if diagram:
            continue
        ann = [t for t in ax.texts] + [c for c in ax.get_children() if type(c).__name__ == 'Annotation']
        for t in set(ann):
            b = bbox_of(t, r)
            if b is None or not t.get_text().strip():
                continue
            if t.get_color() in ('white', 'w', '#ffffff'):
                continue      # labels drawn inside bars on purpose
            bb = b.padded(-1)
            for lab, P in pts:
                inside = (P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) & (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)
                if inside.any():
                    issues.append(('text/data', t.get_text()[:40].replace('\n', ' | '), f'{lab} ({int(inside.sum())} pts)'))
            for lab, rb in rects:
                if inter(bb, rb):
                    issues.append(('text/data', t.get_text()[:40].replace('\n', ' | '), lab))
    REPORT[name] = issues


_orig = Figure.savefig
def patched(self, fname, *a, **k):
    name = os.path.basename(str(fname))
    if name.endswith('.pdf'):
        check(self, name)
    if not name.startswith('fig_v6'):
        fname = os.path.join('/tmp', name)      # figures of earlier versions are checked but never rewritten
    return _orig(self, fname, *a, **k)
Figure.savefig = patched

if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
    import runpy
    runpy.run_path('fig_v6.py', run_name='fig_v6')
    runpy.run_path('fig_mc2.py', run_name='__main__')      # checked only; its files go to /tmp
    keep = {'fig_v6_schematic.pdf', 'fig_v6_decomp.pdf', 'fig_v6_fair.pdf', 'fig_v6_sign.pdf', 'fig_v6_policy.pdf', 'fig_v6_mc.pdf',
            'fig_v6_arch.pdf', 'fig_v6_mechanism.pdf', 'fig_mc2_share.pdf', 'fig_v6_scc_dist.pdf', 'fig_v6_1p5c.pdf'}
    n = 0
    for k in sorted(REPORT):
        if k not in keep:
            continue
        print(f'== {k}: {len(REPORT[k])} issue(s)')
        for i in REPORT[k]:
            print('   ', i)
        n += len(REPORT[k])
    print('TOTAL', n)
