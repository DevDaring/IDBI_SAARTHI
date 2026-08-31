"""Generate simple, slide-friendly PNG diagrams for the submission deck."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RISK="#F59E0B"; RISK_D="#D97706"; SAFE="#0D9488"; SAFE_D="#0F766E"
INK="#0F172A"; MUTE="#334155"; LINE="#CBD5E1"; MIST="#F1F5F9"
OUT=os.path.join(os.path.dirname(__file__),"..","..","deck_figs")
OUT=os.path.abspath(OUT); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans"})


def box(ax, x, y, w, h, text, fc, tc="#FFFFFF", fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=0, facecolor=fc, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3, wrap=True)
    return (x+w/2, y+h/2)


def arrow(ax, p1, p2, color=MUTE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                 color=color, lw=2, zorder=1,
                 shrinkA=3, shrinkB=3))


def process_flow():
    fig, ax = plt.subplots(figsize=(10, 4.3)); ax.set_xlim(0,10); ax.set_ylim(0,4.3); ax.axis("off")
    # phase labels
    ax.text(0.1, 3.95, "PHASE 1 — SCORE", fontsize=10, fontweight="bold", color=SAFE_D)
    ax.text(0.1, 1.85, "PHASE 2 — EXPLAIN & ACT", fontsize=10, fontweight="bold", color=RISK_D)
    row1=["Upload\nloan CSV","Auto-map\ncolumns","Build\nfeatures","Train &\ncalibrate","Predict\n12-mo PD"]
    row2=["SHAP\nreasons","Plain-English\n(AI)","Faithfulness\njudge","Recourse\n(the fix)","Fairness\naudit","Dashboard"]
    # row 1
    n1=len(row1); w=1.62; gap=(10-0.2-n1*w)/(n1-1); y1=2.7
    cx=[]
    for i,t in enumerate(row1):
        x=0.1+i*(w+gap); c=box(ax,x,y1,w,1.0,t,SAFE,fs=10); cx.append((x,x+w,c))
    for i in range(n1-1): arrow(ax,(cx[i][1],y1+0.5),(cx[i+1][0],y1+0.5),SAFE_D)
    # connector down
    arrow(ax,(cx[-1][2][0],y1),(cx[-1][2][0],1.65),INK)
    ax.text(cx[-1][2][0]+0.15,2.05,"then", fontsize=8, color=MUTE)
    # row 2
    n2=len(row2); w2=1.44; gap2=(10-0.2-n2*w2)/(n2-1); y2=0.6
    cx2=[]
    for i,t in enumerate(row2):
        x=0.1+i*(w2+gap2); c=box(ax,x,y2,w2,1.0,t,RISK,fs=9.5); cx2.append((x,x+w2,c))
    for i in range(n2-1): arrow(ax,(cx2[i][1],y2+0.5),(cx2[i+1][0],y2+0.5),RISK_D)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT,"process_flow.png"), dpi=200, facecolor="white")
    plt.close(fig); print("wrote process_flow.png")


def architecture():
    fig, ax = plt.subplots(figsize=(10, 5.0)); ax.set_xlim(0,10); ax.set_ylim(0,5.0); ax.axis("off")
    box(ax,1.6,4.15,6.8,0.6,"Browser  ·  React + Vite single-page app",INK,fs=12)
    arrow(ax,(5,4.15),(5,3.85),MUTE); ax.text(5.12,3.98,"HTTPS (trusted TLS)",fontsize=8.5,color=MUTE)
    box(ax,1.6,3.2,6.8,0.58,"Caddy  ·  reverse proxy + auto-HTTPS  (:443)",SAFE_D,fs=12)
    arrow(ax,(5,3.2),(5,2.9),MUTE)
    box(ax,1.6,2.25,6.8,0.58,"Gunicorn  →  Flask API  ·  async job manager",SAFE,fs=12)
    arrow(ax,(3.6,2.25),(2.9,1.9),MUTE); arrow(ax,(6.4,2.25),(7.1,1.9),MUTE)
    box(ax,0.3,0.65,5.2,1.15,
        "ML Pipeline\ningest → map → features → train →\nsurvival → explain → judge →\nrecourse → fairness",
        RISK,fs=10)
    box(ax,5.8,0.65,3.9,1.15,
        "LLM Gateway\nDeepSeek · Mistral ·\nOpenRouter · Gemini\n(key rotation + fallback)",
        RISK_D,fs=10)
    ax.text(5.0,0.28,"The ML model owns the number · the LLM only explains · a judge verifies · protected attributes are audit-only",
            fontsize=8.5, color=MUTE, style="italic", ha="center")
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT,"architecture.png"), dpi=200, facecolor="white")
    plt.close(fig); print("wrote architecture.png")


if __name__=="__main__":
    process_flow(); architecture()
    print("figures in", OUT)
