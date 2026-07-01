"""Reusable Plotly utilities for the data-science notebook.

Three seams, top to bottom:

    OP MODEL          ──▶  RENDER (scene → figure)  ──▶  DISPLAY (figure → HTML)
    matmul/scale/          figure(op, style,             show(fig, controls,
    softmax own their      animate) → go.Figure          loop) → HTML  (the one
    result + glyph +       backend = _render_3d|_2d      require.js-safe boundary:
    animation schedule                                   always include_plotlyjs=cdn)

A *static* figure is just the terminal frame of an op's schedule, so static and
animated, 2D and 3D, are all *modes over the same op* — not parallel functions.

Used from a notebook cell:
    glue("scores_plot", show(figure(matmul(Q, Kt))), display=False)
or inline in a ``{code-cell}`` whose final expression renders as the output:
    show(figure(matmul(Q, Kt)))

Importable because conf.py prepends ``source/_lib`` to PYTHONPATH (kernels run
in a subprocess and inherit os.environ).
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML

__all__ = [
    # palette / defaults
    "PALETTE", "VIEW", "PLOT_HEIGHT",
    # value type + ops
    "Matrix", "Op", "matmul", "scale", "softmax", "op",
    # render + display
    "figure", "show",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. theme — the single source of truth for geometry + colour
# ─────────────────────────────────────────────────────────────────────────────
ZGAP, GAP, SERIF = 1.5, 2.0, 0.22   # depth gap, h-gap between matrices, bracket feet
BRACKET_W, NUM_SIZE = 5, 14         # bracket thickness (px), element font size
BRACKET_OFFSET = -0.5               # nudge brackets to align under 3D perspective
EXTRA_PLANES, FADE = 2, 0.4         # ghost planes behind front slice + fade-to-white
PLOT_HEIGHT = 250
VIEW = dict(x=0, y=-1.7, z=0)       # locked 3D camera eye

CELL_PX = 44    # 2D heatmap: pixels per cell side (square baseline)
GAP_2D_PX = 40  # 2D heatmap: gap between subplots in px

# 10 generic colours — cycles when more than 10 matrices share a figure
PALETTE = [
    "#2C5C8A",  # 0 blue
    "#4F7A1A",  # 1 green
    "#A86A12",  # 2 amber
    "#7A4FA0",  # 3 purple
    "#1F7A6B",  # 4 teal-green
    "#B5651D",  # 5 sienna
    "#1F6F8B",  # 6 steel-blue
    "#8A2C5C",  # 7 wine
    "#6B8A1A",  # 8 olive
    "#5C1F7A",  # 9 deep-violet
]

NEUTRAL = "#CCCCCC"                 # 3D dimmed-cell colour
NEUTRAL_2D = "#F0F0F0"              # 2D heatmap empty/dim cell
S_ACTIVE, S_FILLED = "#FFD080", "#F0A040"   # 2D result: active (yellow) / filled (amber)


def _lighten(hex_color: str, t: float) -> str:
    """Blend a #rrggbb colour toward white by fraction t (0 unchanged, 1 white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * t) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _p(v: float) -> str:
    """Format a scalar as int when integral, else 2dp — used in titles."""
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def _color(m, idx: int) -> str:
    """Resolve a matrix colour: explicit value wins; otherwise cycle PALETTE."""
    return m.color if m.color is not None else PALETTE[idx % len(PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. value model — Matrix and Op
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Matrix:
    """A tensor to draw. Accepts (rows, cols) or (B, rows, cols); the front
    slice is rendered and EXTRA_PLANES ghost planes are drawn behind it."""
    data: np.ndarray
    name: str
    color: str | None = None
    shape: str | None = None
    planes: int | None = None   # ghost planes behind front; None → EXTRA_PLANES

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.front = self.data[0] if self.data.ndim == 3 else self.data
        self.rows, self.cols = self.front.shape
        dec = 0 if np.allclose(self.front, np.round(self.front)) else 2
        self.plain = [f"{v:.{dec}f}" for v in self.front.reshape(-1)]  # row-major
        if self.shape is None:
            self.shape = f"({self.rows}, {self.cols})"


@dataclass
class _Frame:
    """One animation step: a title + per-matrix list of cell *roles*
    ('full' | 'dim' | 'active' | 'empty'). Backends resolve roles to visuals,
    so the schedule stays backend-agnostic."""
    step: int
    title: str
    roles: list[list[str]]


@dataclass
class Op:
    """An operation = the matrices to display (operands + result), the glyphs
    between them, an optional prefix label, and an animation schedule. The
    result is always the last matrix."""
    matrices: list[Matrix]
    glyphs: list[str]
    prefix: str | None
    _states: list[_Frame] = field(default_factory=list)

    @property
    def result(self) -> Matrix:
        return self.matrices[-1]

    @property
    def array(self) -> np.ndarray:
        return self.result.front

    def states(self) -> list[_Frame]:
        return self._states


def _as_matrix(x) -> Matrix:
    """Coerce an operand: a Matrix passes through; an Op unwraps to its result
    (so ops chain like math). Raw arrays are rejected — operands need a colour
    and a name, so the caller must wrap them in Matrix."""
    if isinstance(x, Matrix):
        return x
    if isinstance(x, Op):
        return x.result
    raise TypeError(
        f"operand must be a Matrix or Op, got {type(x).__name__}; "
        "wrap arrays as Matrix(data, name, color)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. operations — each owns its result, glyphs and schedule
# ─────────────────────────────────────────────────────────────────────────────
def _terminal_states(matrices: list[Matrix]) -> list[_Frame]:
    """A single 'everything revealed' frame — the static / non-animated state."""
    roles = [["full"] * (m.rows * m.cols) for m in matrices]
    return [_Frame(0, "", roles)]


def _matmul_states(A: Matrix, B: Matrix, C: Matrix) -> list[_Frame]:
    """One frame per output cell: highlight row i of A and col j of B, reveal
    C[i,j]. The final frame reveals everything (the paused end state)."""
    m, n, k = C.rows, C.cols, A.cols
    total = m * n
    states: list[_Frame] = []
    for step in range(total + 1):
        done = step >= total
        ai = step // n if not done else -1
        aj = step % n if not done else -1
        roles_A = ["full" if (done or r == ai) else "dim"
                   for r in range(A.rows) for _ in range(A.cols)]
        roles_B = ["full" if (done or c == aj) else "dim"
                   for _ in range(B.rows) for c in range(B.cols)]
        roles_C = []
        for r in range(C.rows):
            for c in range(C.cols):
                idx = r * C.cols + c
                if done or idx < step:
                    roles_C.append("full")
                elif r == ai and c == aj:
                    roles_C.append("active")
                else:
                    roles_C.append("empty")
        if done:
            title = f"{C.name} = {A.name} @ {B.name}  —  press ▶ to replay"
        else:
            i, j = step // n, step % n
            dp = " + ".join(f"{_p(A.front[i, d])}×{_p(B.front[d, j])}" for d in range(k))
            title = f"Step {step + 1}/{total}:  {C.name}[{i},{j}] = {dp} = {_p(C.front[i, j])}"
        states.append(_Frame(step, title, [roles_A, roles_B, roles_C]))
    return states


def matmul(A, B, *, name="result", color: str | None = None, shape=None) -> Op:
    """A @ B, drawn as ``A @ B = result``. Animation walks one output cell per
    frame. Operands may be Matrix or Op (chained)."""
    A, B = _as_matrix(A), _as_matrix(B)
    res = Matrix(A.front @ B.front, name, color, shape)
    return Op([A, B, res], ["@", "="], None, _matmul_states(A, B, res))


def scale(A, by, *, glyph="÷ √dₖ =", name="scaled", color: str | None = None, shape=None) -> Op:
    """Element-wise A / by, drawn as ``A {glyph} result`` (static by default)."""
    A = _as_matrix(A)
    res = Matrix(A.front / by, name, color, shape)
    op_ = Op([A, res], [glyph], None, [])
    op_._states = _terminal_states(op_.matrices)
    return op_


def softmax(A, *, axis=-1, prefix="softmax", name="W", color=None, shape=None) -> Op:
    """Row-wise softmax, drawn as ``softmax  A = result`` (static by default)."""
    A = _as_matrix(A)
    x = A.front - A.front.max(axis=axis, keepdims=True)
    e = np.exp(x)
    res = Matrix(e / e.sum(axis=axis, keepdims=True), name, color, shape)
    op_ = Op([A, res], ["="], prefix, [])
    op_._states = _terminal_states(op_.matrices)
    return op_


def op(matrices, glyphs, *, prefix=None, states=None) -> Op:
    """Escape hatch: lay out arbitrary matrices with arbitrary glyphs between
    them. Pass ``states`` to supply a custom animation schedule."""
    mats = [_as_matrix(m) for m in matrices]
    return Op(mats, list(glyphs), prefix, states or _terminal_states(mats))


# ─────────────────────────────────────────────────────────────────────────────
# 4a. primitives — pure go.* trace builders (2D coords, shape-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def _bracket_trace(x0, rows, cols, color):
    x_lo, x_hi = x0 - 0.5, x0 + cols - 0.5
    z_lo, z_hi = -0.5 - BRACKET_OFFSET, rows - 0.5 - BRACKET_OFFSET
    bx, by, bz = [], [], []
    for pts in ([(x_lo + SERIF, z_hi), (x_lo, z_hi), (x_lo, z_lo), (x_lo + SERIF, z_lo)],
                [(x_hi - SERIF, z_hi), (x_hi, z_hi), (x_hi, z_lo), (x_hi - SERIF, z_lo)]):
        for px, pz in pts:
            bx.append(px); by.append(0); bz.append(pz)
        bx.append(None); by.append(None); bz.append(None)
    return go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                        line=dict(color=color, width=BRACKET_W),
                        hoverinfo="none", showlegend=False, opacity=0.85)


def _ghost_plane_traces(x0, rows, cols, color, *, planes=EXTRA_PLANES):
    x_lo, x_hi = x0 - 0.5, x0 + cols - 0.5
    z_lo, z_hi = -0.5 - BRACKET_OFFSET, rows - 0.5 - BRACKET_OFFSET
    out = []
    for b in range(1, planes + 1):
        y = b * ZGAP
        c = _lighten(color, FADE * b / EXTRA_PLANES)
        out.append(go.Mesh3d(
            x=[x_lo + 0.05, x_hi - 0.05, x_hi - 0.05, x_lo + 0.05], y=[y, y, y, y],
            z=[z_lo + 0.05, z_lo + 0.05, z_hi - 0.05, z_hi - 0.05],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=c, opacity=0.2, hoverinfo="none", showlegend=False))
    return out


def _grid_xyz(rows, cols, x0):
    tx, ty, tz = [], [], []
    for r in range(rows):
        for c in range(cols):
            tx.append(x0 + c); ty.append(0); tz.append(rows - 1 - r)
    return tx, ty, tz


def _label_traces(x0, rows, cols, name, shape, color):
    xc = x0 + (cols - 1) / 2
    return [
        go.Scatter3d(x=[xc], y=[0], z=[rows + 0.45], mode="text",
                     text=[f"<b>{name}</b>"], textfont=dict(size=16, color=color),
                     hoverinfo="none", showlegend=False),
        go.Scatter3d(x=[xc], y=[0], z=[rows - 0.05], mode="text",
                     text=[shape], textfont=dict(size=12, color=color),
                     hoverinfo="none", showlegend=False),
    ]


def _text_trace(x, z, text, size, color):
    return go.Scatter3d(x=[x], y=[0], z=[z], mode="text", text=[text],
                        textfont=dict(size=size, color=color),
                        hoverinfo="none", showlegend=False)


# role → visual resolvers
def _txt3d(plain, role, is_result):
    if not is_result:
        return f"<b>{plain}</b>"                 # operands always show their number
    if role == "full":
        return f"<b>{plain}</b>"
    if role == "active":
        return "○"
    return ""                                    # not yet computed


def _col3d(role, base):
    return base if role in ("full", "active") else NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# 4b. backends — scene → go.Figure
# ─────────────────────────────────────────────────────────────────────────────
def _render_3d(op_: Op, *, animate: bool, height: int, planes: int | None = None) -> go.Figure:
    states = op_.states()
    terminal = states[-1]
    result_idx = len(op_.matrices) - 1

    fig = go.Figure()
    num_idx = []          # number-trace index per matrix (these are animated)
    bounds = []
    x0 = GAP if op_.prefix else 0.0

    for mi, m in enumerate(op_.matrices):
        c = _color(m, mi)
        is_result = mi == result_idx
        roles = terminal.roles[mi]
        fig.add_trace(_bracket_trace(x0, m.rows, m.cols, c))
        tx, ty, tz = _grid_xyz(m.rows, m.cols, x0)
        text = [_txt3d(m.plain[k], roles[k], is_result) for k in range(len(roles))]
        color = c if is_result else [_col3d(roles[k], c) for k in range(len(roles))]
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="text", text=text,
                                   textfont=dict(size=NUM_SIZE, color=color),
                                   hoverinfo="none", showlegend=False))
        num_idx.append(len(fig.data) - 1)
        nplanes = planes if planes is not None else (m.planes if m.planes is not None else EXTRA_PLANES)
        for t in _ghost_plane_traces(x0, m.rows, m.cols, c, planes=nplanes):
            fig.add_trace(t)
        for t in _label_traces(x0, m.rows, m.cols, m.name, m.shape, c):
            fig.add_trace(t)
        bounds.append((x0 - 0.5, x0 + m.cols - 0.5))
        x0 += m.cols + GAP

    if op_.prefix:
        fig.add_trace(_text_trace((bounds[0][0] - GAP) / 2, 1.0, f"<b>{op_.prefix}</b>", 15, "#333"))
    for gi, g in enumerate(op_.glyphs):
        xmid = (bounds[gi][1] + bounds[gi + 1][0]) / 2
        fig.add_trace(_text_trace(xmid, 1.0, f"<b>{g}</b>", 18 if len(g) == 1 else 15, "#333"))

    _ml = _mr = 20

    fig.update_layout(
        template="plotly_white",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data", camera=dict(eye=VIEW)),
        showlegend=False, height=height)

    if not animate:
        fig.update_layout(margin=dict(l=_ml, r=_mr, t=0, b=0))
        return fig

    # animated: step text lives in hidden annotations (shown by show(steps=True)).
    # Each frame re-asserts scene_camera so PLAY keeps the WebGL camera locked.
    fig.update_layout(
        annotations=_step_ann(terminal.title),
        margin=dict(l=_ml, r=_mr, t=10, b=10))
    frames = []
    for st in states:
        data = []
        for mi, m in enumerate(op_.matrices):
            roles = st.roles[mi]
            c = _color(m, mi)
            if mi == result_idx:
                data.append(go.Scatter3d(text=[_txt3d(m.plain[k], roles[k], True)
                                               for k in range(len(roles))]))
            else:
                data.append(go.Scatter3d(textfont=dict(
                    color=[_col3d(roles[k], c) for k in range(len(roles))])))
        frames.append(go.Frame(name=str(st.step), data=data, traces=num_idx,
                               layout=go.Layout(annotations=_step_ann(st.title),
                                                scene_camera=dict(eye=VIEW))))
    fig.frames = frames
    return fig


_Z = {"full": 1.0, "active": 0.5, "dim": 0.0, "empty": 0.0}


def _heat_zt(m: Matrix, roles, is_result):
    """Build (z, text) for a heatmap, flipped so row 0 sits at the top."""
    z = np.array([_Z[r] for r in roles]).reshape(m.rows, m.cols)
    text = [(m.plain[k] if (not is_result or roles[k] == "full") else "")
            for k in range(len(roles))]
    text = np.array(text, dtype=object).reshape(m.rows, m.cols)
    return z[::-1].tolist(), text[::-1].tolist()


def _colorscale_2d(color: str, is_result):
    if is_result:
        return [[0.0, NEUTRAL_2D], [0.5, S_ACTIVE], [1.0, S_FILLED]]
    return [[0.0, NEUTRAL_2D], [1.0, _lighten(color, 0.6)]]


def _render_2d(op_: Op, *, animate: bool, height: int) -> go.Figure:
    states = op_.states()
    terminal = states[-1]
    n = len(op_.matrices)
    result_idx = n - 1
    max_rows = max(m.rows for m in op_.matrices)

    # Margins reserve space for labels (top) and the step text (bottom).
    ml = 60 if op_.prefix else 20
    mr, mt = 20, 46
    mb = 42 if animate else 12

    # HEIGHT IS THE CONSTRAINT. The plot area is what's left after the reserves;
    # the tallest matrix (max_rows) fills it, which fixes a uniform cell height.
    # Every matrix then uses that same cell size, so all cells are equal squares.
    avail_h = max(height - mt - mb, 20)
    cell_h  = avail_h / max_rows
    # Square by default; widen only if the widest number needs more room.
    longest = max((len(s) for m in op_.matrices for s in m.plain), default=1)
    cell_w  = max(cell_h, longest * 8.5 + 16)

    gap    = GAP_2D_PX
    plot_w = sum(m.cols for m in op_.matrices) * cell_w + max(n - 1, 0) * gap
    fig_w  = plot_w + ml + mr
    fig_h  = height

    # Per-matrix x/y axis domains (fractions of the plot area). Matrices are
    # CENTRE-aligned vertically, so every matrix's centre sits at y=0.5 and the
    # operators line up with all of them. Shorter matrices get the SAME cell
    # height as the tallest — they just occupy less vertical space.
    xdoms, ydoms = [], []
    x_px = 0.0
    for m in op_.matrices:
        w = m.cols * cell_w
        xdoms.append((x_px / plot_w, (x_px + w) / plot_w))
        x_px += w + gap
        h = m.rows / max_rows
        ydoms.append((0.5 - h / 2, 0.5 + h / 2))

    fig = go.Figure()
    axis_layout = {}
    for mi, m in enumerate(op_.matrices):
        a = mi + 1
        xref = "x" if a == 1 else f"x{a}"
        yref = "y" if a == 1 else f"y{a}"
        c = _color(m, mi)
        z, text = _heat_zt(m, terminal.roles[mi], mi == result_idx)
        fig.add_trace(go.Heatmap(z=z, text=text, texttemplate="<b>%{text}</b>",
                                 textfont={"size": NUM_SIZE, "color": "#222"},
                                 colorscale=_colorscale_2d(c, mi == result_idx),
                                 zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
                                 hoverinfo="skip", xaxis=xref, yaxis=yref))
        xkey = "xaxis" if a == 1 else f"xaxis{a}"
        ykey = "yaxis" if a == 1 else f"yaxis{a}"
        axis_layout[xkey] = dict(visible=False, domain=list(xdoms[mi]),
                                 range=[-0.5, m.cols - 0.5], anchor=yref)
        axis_layout[ykey] = dict(visible=False, domain=list(ydoms[mi]),
                                 range=[-0.5, m.rows - 0.5], anchor=xref)

    # ── labels: names, shapes, operators, prefix (paper coords) ───────────────
    # Font sizes match the 3D renderer: name 16, shape 11, numbers NUM_SIZE,
    # operators 18 (single glyph) / 15, prefix 15.
    op_y   = 0.5                  # vertical centre — aligns with every matrix
    step_y = -17 / avail_h        # in the bottom margin, below the cells
    # Centre the step text on the FIGURE (paper x=0.5 is the plot-area centre,
    # which drifts when a prefix widens the left margin).
    step_x = 0.5 + (mr - ml) / (2 * plot_w)

    for mi, m in enumerate(op_.matrices):
        c = _color(m, mi)
        xc = (xdoms[mi][0] + xdoms[mi][1]) / 2
        top = ydoms[mi][1]                       # this matrix's own top edge
        fig.add_annotation(text=f"<b>{m.name}</b>", x=xc, y=top + 20 / avail_h,
                           xref="paper", yref="paper",
                           xanchor="center", yanchor="bottom", showarrow=False,
                           font=dict(size=16, color=c))
        fig.add_annotation(text=m.shape, x=xc, y=top + 4 / avail_h,
                           xref="paper", yref="paper",
                           xanchor="center", yanchor="bottom", showarrow=False,
                           font=dict(size=12, color=c))

    for gi, g in enumerate(op_.glyphs):
        xmid = (xdoms[gi][1] + xdoms[gi + 1][0]) / 2
        fig.add_annotation(text=f"<b>{g}</b>", x=xmid, y=op_y,
                           xref="paper", yref="paper",
                           xanchor="center", yanchor="middle", showarrow=False,
                           font=dict(size=18 if len(g) == 1 else 15, color="#333"))

    if op_.prefix:
        fig.add_annotation(text=f"<b>{op_.prefix}</b>", x=0, y=op_y,
                           xref="paper", yref="paper",
                           xanchor="left", yanchor="middle", showarrow=False,
                           font=dict(size=15, color="#333"))

    # Serialise before update_layout: update_layout(annotations=...) merges into
    # existing Annotation objects by index, which would corrupt label text.
    label_anns = [a.to_plotly_json() for a in fig.layout.annotations]

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        dragmode=False,
        width=fig_w, height=fig_h,
        margin=dict(l=ml, r=mr, t=mt, b=mb),
        **axis_layout,

    )
    if animate:
        fig.add_annotation(**_step_ann(terminal.title, x=step_x, y=step_y,
                                       yanchor="middle")[0])

    if not animate:
        return fig

    frames = []
    for st in states:
        data = []
        for mi, m in enumerate(op_.matrices):
            z, text = _heat_zt(m, st.roles[mi], mi == result_idx)
            data.append(go.Heatmap(z=z, text=text))
        frames.append(go.Frame(name=str(st.step), data=data, traces=list(range(n)),
                               layout=go.Layout(annotations=(
                                   label_anns + _step_ann(st.title, x=step_x,
                                                           y=step_y, yanchor="middle")))))
    fig.frames = frames
    return fig


def figure(op_: Op, *, style="3d", animate=False, height=None, planes=None) -> go.Figure:
    """Render an op to a go.Figure. style='3d' (bracket/ghost-plane) or '2d'
    (heatmap subplots); animate=True walks the op's schedule, else draws the
    terminal frame. planes overrides the ghost-plane count for ALL matrices in
    this figure (None → each matrix's own .planes or EXTRA_PLANES). Pure —
    returns a Figure; call show() to display it."""
    height = height or PLOT_HEIGHT
    if style == "3d":
        return _render_3d(op_, animate=animate, height=height, planes=planes)
    if style == "2d":
        return _render_2d(op_, animate=animate, height=height)
    raise ValueError(f"style must be '3d' or '2d', got {style!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. display — the one require.js-safe boundary (always include_plotlyjs="cdn")
# ─────────────────────────────────────────────────────────────────────────────
_ID = [0]


def _next_id() -> str:
    _ID[0] += 1
    return f"dsviz-plot-{_ID[0]}"


def _step_ann(text, *, x=0.5, y=0.02, yanchor="bottom"):
    """Step text annotation — hidden by default; show(steps=True) makes it visible.
    Note: yref='paper' y is relative to the PLOT AREA (0=bottom edge, 1=top edge),
    so y<0 lands in the bottom margin (below the cells)."""
    return [dict(text=text, x=x, y=y, xref="paper", yref="paper",
                 xanchor="center", yanchor=yanchor, showarrow=False,
                 font=dict(size=13, color="black"), visible=False)]


def _toggle_button():
    """One button: args = play (1st click), args2 = pause (2nd click)."""
    return dict(label="▶ ⏸", method="animate",
                args=[None, dict(frame=dict(duration=900, redraw=True),
                                 transition=dict(duration=0), mode="immediate",
                                 fromcurrent=True)],
                args2=[[None], dict(frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0), mode="immediate")])


def _autoplay_js(duration=900, transition=200, terminal_pause=2000) -> str:
    # Loops: jump to frame 0, play the full sequence, hold on the terminal frame
    # for terminal_pause ms, repeat. Replaying via animate(null) alone fails — it
    # resumes from the current (last) frame — so we reset to frame 0 each cycle and
    # drive it on a timer (no plotly_animated, which the reset jump would re-fire).
    return f"""
    (function() {{
        var gd = document.getElementById('{{plot_id}}');
        var opts = {{frame: {{duration: {duration}, redraw: true}},
                     transition: {{duration: {transition}}}, mode: 'immediate'}};
        function loop() {{
            var names = gd._transitionData._frames.map(function(f) {{ return f.name; }});
            Plotly.animate(gd, [names[0]],
                           {{mode: 'immediate', frame: {{duration: 0, redraw: true}},
                             transition: {{duration: 0}}}});
            Plotly.animate(gd, names, opts);
        }}
        var nframes = gd._transitionData._frames.length;
        var cycleMs = nframes * ({duration} + {transition}) + {terminal_pause};
        loop();
        setInterval(loop, cycleMs);
    }})();
    """


def show(fig: go.Figure, *, div_id=None, controls="hover", loop=False,
         height=None, steps=False, modebar=False) -> HTML:
    """Turn a figure into a displayable HTML object — usable as a glue() value
    OR as a code-cell's final expression.

    controls : 'hover' (toggle button fades in on hover), 'always' (button
               always visible), or None (no button). Ignored for static figures.
    loop     : autoplay + restart every 10s (for the 2D animation).
    steps    : show per-frame step text as a bottom-center black annotation
               instead of the figure title (default False — hidden).
    modebar  : show Plotly's top toolbar (default False — hidden).
    """
    if height is not None:
        fig.update_layout(height=height)
    has_frames = bool(fig.frames)

    if has_frames and steps:
        # Only make frame annotations visible. For loop mode the initial figure
        # annotation (terminal text) stays invisible — if made visible it persists
        # as a second overlay on top of every frame's step annotation.
        for fr in fig.frames:
            if fr.layout and fr.layout.annotations:
                for ann in fr.layout.annotations:
                    ann.visible = True
        if not loop:
            # Non-loop (3D paused): initial annotation shows terminal text on load.
            for ann in fig.layout.annotations:
                ann.visible = True

    post = None
    css_rules = []
    is_2d = not any(isinstance(t, (go.Scatter3d, go.Mesh3d)) for t in fig.data)
    if is_2d:
        div_id = div_id or _next_id()
        # Kill the crosshair cursor (centering is handled by a flex wrapper below).
        css_rules.append(f"#{div_id} .nsewdrag,#{div_id} .drag{{cursor:default !important;}}")
    if has_frames and controls in ("hover", "always"):
        if is_2d:
            # Hang the toggle in the bottom margin, below the cells, right-aligned
            # (paper y=0 is the plot-area bottom; yanchor='top' drops it into mb).
            btn_pos = dict(x=1.0, xanchor="right", y=0.0, yanchor="top",
                           pad=dict(t=6, r=0))
        else:
            btn_pos = dict(x=0.98, xanchor="right", y=0.02, yanchor="bottom",
                           pad=dict(r=2, b=2))
        fig.update_layout(updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            buttons=[_toggle_button()], **btn_pos)])
        if controls == "hover":
            div_id = div_id or _next_id()
            css_rules.append(f"#{div_id} .updatemenu-container{{opacity:0;transition:opacity .2s ease;}}")
            css_rules.append(f"#{div_id}:hover .updatemenu-container{{opacity:1;}}")
    if has_frames and loop:
        post = _autoplay_js()
        # "press ▶ to replay" on the terminal frame is pointless in auto-loop mode;
        # keep the equation part, strip the replay instruction.
        if steps and fig.frames:
            for ann in fig.layout.annotations:
                ann.text = ann.text.split("  —  ")[0]
            for ann in fig.frames[-1].layout.annotations:
                ann.text = ann.text.split("  —  ")[0]

    style = f"<style>{''.join(css_rules)}</style>" if css_rules else ""
    config = None if modebar else {"displayModeBar": False}
    html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                       div_id=div_id, post_script=post, config=config)
    if is_2d:
        # Fixed-width 2D figures default to flush-left; a full-width flex wrapper
        # centres the figure on the page. (3D is responsive, so no wrapper needed.)
        html = f'<div style="display:flex;justify-content:center;width:100%">{html}</div>'
    return HTML(style + html)
