"""
Generate the simple explanatory diagrams for the SAARTHI "How it works" page.

Uses graphviz (flowcharts) + matplotlib (example charts) and writes SVGs into
frontend/public/how/ so Vite bundles them as static assets.

Run:  python scripts/make_explainer_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import graphviz

# --- SAARTHI palette -------------------------------------------------------
RISK = "#F59E0B"; RISK_D = "#D97706"
DANGER = "#DC2626"; DANGER_D = "#991B1B"
SAFE = "#0D9488"; SAFE_D = "#0F766E"
INK = "#0F172A"; INK_MUTE = "#334155"
PAPER = "#FFFFFF"; MIST = "#F1F5F9"; LINE = "#CBD5E1"

OUT = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "how")
os.makedirs(OUT, exist_ok=True)
FONT = "Helvetica"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def save_gv(dot: graphviz.Digraph, name: str):
    svg = dot.pipe(format="svg").decode("utf-8")
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)
    print("wrote", name)


def _node(dot, nid, title, sub, fill, fg="#FFFFFF"):
    label = (f'<<table border="0" cellborder="0" cellspacing="0">'
             f'<tr><td><font point-size="13"><b>{title}</b></font></td></tr>'
             f'<tr><td><font point-size="10">{sub}</font></td></tr></table>>')
    dot.node(nid, label=label, fillcolor=fill, fontcolor=fg)


# ---------------------------------------------------------------------------
# 1) The whole pipeline
# ---------------------------------------------------------------------------
def pipeline():
    g = graphviz.Digraph("pipeline")
    g.attr(rankdir="LR", bgcolor="transparent", nodesep="0.28", ranksep="0.5")
    g.attr("node", shape="box", style="filled,rounded", fontname=FONT,
           color="none", margin="0.16,0.10", height="0.1")
    g.attr("edge", color=LINE, penwidth="1.6", arrowsize="0.7")

    steps = [
        ("up", "1 · Upload", "your loan CSV", INK),
        ("map", "2 · Auto-map", "AI names the columns", INK_MUTE),
        ("train", "3 · Train", "model learns patterns", SAFE),
        ("pd", "4 · Predict", "default probability", SAFE_D),
        ("shap", "5 · Explain", "SHAP: which factors", RISK),
        ("llm", "6 · Describe", "AI writes plain English", RISK_D),
        ("judge", "7 · Verify", "judge AI checks it", DANGER),
        ("fix", "8 · Recommend", "the one move to fix it", SAFE),
        ("fair", "9 · Fairness", "no group bias", SAFE_D),
        ("dash", "10 · Dashboard", "officer sees it all", INK),
    ]
    for nid, t, s, c in steps:
        _node(g, nid, t, s, c)
    for a, b in zip([s[0] for s in steps], [s[0] for s in steps][1:]):
        g.edge(a, b)
    save_gv(g, "pipeline.svg")


# ---------------------------------------------------------------------------
# 2) Column mapping (messy -> fixed schema)
# ---------------------------------------------------------------------------
def mapping():
    g = graphviz.Digraph("mapping")
    g.attr(rankdir="LR", bgcolor="transparent", nodesep="0.18", ranksep="1.4")
    g.attr("node", shape="box", style="filled,rounded", fontname=FONT,
           color="none", margin="0.14,0.08", fontsize="12")
    g.attr("edge", color=LINE, penwidth="1.4", arrowsize="0.6")

    with g.subgraph(name="cluster_src") as c:
        c.attr(label="Your messy columns", labelloc="t", fontname=FONT,
               fontsize="12", color=LINE, style="rounded")
        for nid, txt in [("s1", "DisbursedAmount"), ("s2", "MIS_Status"),
                         ("s3", "NAICS"), ("s4", "ProprietorGender"),
                         ("s5", "State")]:
            c.node(nid, txt, fillcolor=MIST, fontcolor=INK)

    with g.subgraph(name="cluster_dst") as c:
        c.attr(label="Fixed SAARTHI schema", labelloc="t", fontname=FONT,
               fontsize="12", color=LINE, style="rounded")
        c.node("d1", "loan_amount", fillcolor="#CCFBF1", fontcolor=SAFE_D)
        c.node("d2", "target  (default?)", fillcolor="#CCFBF1", fontcolor=SAFE_D)
        c.node("d3", "sector", fillcolor="#CCFBF1", fontcolor=SAFE_D)
        c.node("d4", "gender  · PROTECTED", fillcolor="#FEE2E2", fontcolor=DANGER_D)
        c.node("d5", "region  · PROTECTED", fillcolor="#FEE2E2", fontcolor=DANGER_D)

    for a, b in [("s1", "d1"), ("s2", "d2"), ("s3", "d3"), ("s4", "d4"), ("s5", "d5")]:
        g.edge(a, b)
    save_gv(g, "mapping.svg")


# ---------------------------------------------------------------------------
# 3) Who does what (the golden rule)
# ---------------------------------------------------------------------------
def roles():
    g = graphviz.Digraph("roles")
    g.attr(rankdir="LR", bgcolor="transparent", nodesep="0.5", ranksep="0.6")
    g.attr("node", shape="box", style="filled,rounded", fontname=FONT,
           color="none", margin="0.2,0.14")
    g.attr("edge", color=LINE, penwidth="1.6", arrowsize="0.7")
    _node(g, "m", "ML MODEL", "gives the NUMBER<br/>(probability of default)", SAFE)
    _node(g, "a", "AI (LLM)", "writes the WORDS<br/>(reasons + the fix)", RISK)
    _node(g, "j", "JUDGE AI", "CHECKS the words match<br/>the model's evidence", INK)
    g.edge("m", "a", label="  drivers")
    g.edge("a", "j", label="  explanation")
    g.node("note", '<<b>The AI never invents the score.</b><br/>The model owns the math.>',
           shape="note", fillcolor="#FEF3C7", fontcolor=INK, fontname=FONT, fontsize="11")
    g.edge("j", "note", style="invis")
    save_gv(g, "roles.svg")


# ---------------------------------------------------------------------------
# 4) Faithfulness judge loop
# ---------------------------------------------------------------------------
def judge():
    g = graphviz.Digraph("judge")
    g.attr(rankdir="LR", bgcolor="transparent", nodesep="0.4", ranksep="0.55")
    g.attr("node", shape="box", style="filled,rounded", fontname=FONT,
           color="none", margin="0.18,0.12", fontsize="12")
    g.attr("edge", color=LINE, penwidth="1.6", arrowsize="0.7", fontname=FONT, fontsize="10")
    _node(g, "w", "AI writes", "the explanation", RISK_D)
    g.node("chk", "Does it match\nthe SHAP evidence?", shape="diamond",
           style="filled", fillcolor="#FEF3C7", fontcolor=INK, fontname=FONT, fontsize="11")
    _node(g, "ok", "✓ Verified", "badge shown to officer", SAFE)
    _node(g, "no", "Rewrite once", "then re-check", DANGER)
    g.edge("w", "chk")
    g.edge("chk", "ok", label="  yes", color=SAFE_D)
    g.edge("chk", "no", label="  no", color=DANGER_D)
    g.edge("no", "chk", style="dashed", color=DANGER_D)
    save_gv(g, "judge.svg")


# ---------------------------------------------------------------------------
# 5) SHAP drivers example (matplotlib)
# ---------------------------------------------------------------------------
def shap_example():
    feats = ["Credit score 530\n(low)", "DSCR 0.54\n(cash-flow tight)",
             "3 past late payments", "Collateral too low", "Sector: Manufacturing"]
    vals = [0.34, 0.28, 0.18, 0.09, -0.06]
    colors = [DANGER if v > 0 else SAFE for v in vals]
    y = np.arange(len(feats))[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color=INK, lw=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.012 if v > 0 else -0.012), yi, f"{v:+.2f}",
                va="center", ha="left" if v > 0 else "right",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=10, color=INK)
    ax.set_xlabel("← pulls risk DOWN      pushes risk UP →", fontsize=10, color=INK_MUTE)
    ax.set_title("Why is THIS loan risky?  (SHAP contribution of each factor)",
                 fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(-0.15, 0.42)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=MIST, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shap.svg"), transparent=True)
    plt.close(fig); print("wrote shap.svg")


# ---------------------------------------------------------------------------
# 6) Recourse before -> after (matplotlib)
# ---------------------------------------------------------------------------
def recourse():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.bar([0], [0.72], width=0.5, color=DANGER, zorder=3)
    ax.bar([2], [0.28], width=0.5, color=SAFE, zorder=3)
    ax.text(0, 0.72 + 0.03, "72%", ha="center", fontsize=15, fontweight="bold", color=DANGER_D)
    ax.text(2, 0.28 + 0.03, "28%", ha="center", fontsize=15, fontweight="bold", color=SAFE_D)
    ax.text(0, -0.08, "BEFORE", ha="center", fontsize=11, color=INK)
    ax.text(2, -0.08, "AFTER", ha="center", fontsize=11, color=INK)
    ax.add_patch(FancyArrowPatch((0.42, 0.5), (1.58, 0.4),
                 arrowstyle="-|>", mutation_scale=22, color=INK, lw=2))
    ax.text(1.0, 0.60, "Extend tenure 6 months\n+ add working-capital line",
            ha="center", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FEF3C7", ec=RISK))
    ax.axhline(0.5, color=LINE, lw=1, ls="--")
    ax.text(2.5, 0.5, "high-risk line", fontsize=8.5, color=INK_MUTE, va="center")
    ax.set_ylim(0, 0.9); ax.set_xlim(-0.6, 3.1)
    ax.set_title("The one move that lowers this loan's default risk",
                 fontsize=12, color=INK, fontweight="bold", loc="left", pad=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recourse.svg"), transparent=True)
    plt.close(fig); print("wrote recourse.svg")


# ---------------------------------------------------------------------------
# 7) 12-month risk curve (matplotlib)
# ---------------------------------------------------------------------------
def curve():
    m = np.arange(1, 13)
    pd_cum = 1 - np.exp(-((m / 9.5) ** 2.1)) * 0.9
    pd_cum = np.clip(pd_cum / pd_cum[-1] * 0.72, 0, 1)
    onset = int(m[np.argmax(pd_cum >= 0.20)])

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(m, pd_cum, color=RISK_D, lw=3, marker="o", ms=5, zorder=3)
    ax.fill_between(m, pd_cum, color=RISK, alpha=0.10, zorder=1)
    ax.axhline(0.20, color=DANGER, lw=1.4, ls="--")
    ax.text(12, 0.21, "alert line (20%)", ha="right", fontsize=9, color=DANGER_D)
    ax.axvline(onset, color=INK, lw=1.2, ls=":")
    ax.annotate(f"crosses the line\nin month {onset} → act now",
                xy=(onset, 0.20), xytext=(onset + 0.4, 0.5), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xticks(m)
    ax.set_xlabel("months ahead", fontsize=10, color=INK_MUTE)
    ax.set_ylabel("chance of default (cumulative)", fontsize=10, color=INK_MUTE)
    ax.set_title("See the risk building — up to 12 months early",
                 fontsize=12, color=INK, fontweight="bold", loc="left", pad=10)
    ax.set_ylim(0, 0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(LINE)
    ax.tick_params(length=0, labelsize=9)
    ax.grid(color=MIST, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curve.svg"), transparent=True)
    plt.close(fig); print("wrote curve.svg")


if __name__ == "__main__":
    pipeline(); mapping(); roles(); judge()
    shap_example(); recourse(); curve()
    print("\nAll diagrams written to", os.path.abspath(OUT))
