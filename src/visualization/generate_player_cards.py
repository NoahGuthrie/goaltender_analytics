"""
Generate static PNG player cards for top goalies.
Professional, clean design — white background, simple bar charts, rink overlay.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import duckdb

# ── Design tokens ──────────────────────────────────────────
BG     = "#ffffff"
TEXT   = "#222222"
MUTED  = "#888888"
BORDER = "#dddddd"
POS    = "#1e8449"
NEG    = "#c0392b"
BLUE   = "#2471a3"
GOLD   = "#d4ac0d"

BAR_BG   = "#e8e8e8"
BAR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "pct", [NEG, GOLD, POS]
)

FONT = {"family": "sans-serif", "size": 11}
matplotlib.rc("font", **FONT)


def draw_rink_zone(ax):
    """Draw the defensive zone on an axes (x: 25→100, y: -42.5→42.5)."""
    ax.set_xlim(23, 102)
    ax.set_ylim(-44, 44)
    ax.set_aspect("equal")
    ax.axis("off")

    kw = dict(color=BORDER, linewidth=0.8)

    # Boards — straight sides
    ax.plot([25, 89], [-42.5, -42.5], **kw)
    ax.plot([25, 89], [42.5, 42.5], **kw)
    # End boards — curved
    from matplotlib.patches import FancyBboxPatch, Arc
    ax.plot([89, 100, 100, 89], [-42.5, -28, 28, 42.5], color=BORDER, lw=0.8)
    # Blue line
    ax.plot([25, 25], [-42.5, 42.5], color=BLUE, lw=2)
    # Goal line
    ax.axvline(89, color="#c0392b", lw=1, ls="--", alpha=0.5)
    # Net
    ax.add_patch(patches.Rectangle((89, -3), 4, 6, fill=False, edgecolor="#555", lw=1.2))
    # Crease
    crease = patches.Arc((89, 0), 12, 12, angle=0, theta1=90, theta2=270,
                          edgecolor="#c0392b", lw=1)
    ax.add_patch(crease)
    ax.add_patch(patches.Wedge((89, 0), 6, 90, 270, facecolor="lightblue", alpha=0.1, edgecolor="none"))
    # Faceoff circles
    for fy in [22, -22]:
        ax.add_patch(patches.Circle((69, fy), 15, fill=False, edgecolor="#c0392b", lw=0.5))
        ax.plot(69, fy, "o", color="#c0392b", ms=3)


def create_goalie_card(goalie_id, stats_df):
    row = stats_df[stats_df["goalie_id"] == goalie_id].iloc[0]
    name = row["Goalie"]

    fig = plt.figure(figsize=(12, 8), facecolor=BG, dpi=150)

    # ── LEFT: stats (55% width) ──
    ax = fig.add_axes([0.03, 0.05, 0.50, 0.90], facecolor=BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    ax.text(2, 95, name, fontsize=22, fontweight="bold", color=TEXT, va="top")
    ax.text(2, 89, "GSAx 2.0  ·  2025-26", fontsize=9, color=MUTED, va="top")
    ax.plot([2, 98], [86, 86], color=BORDER, lw=0.8)

    # Metrics with percentile bars
    metrics = [
        ("GSAx 2.0",     row["gsax_2_0_percentile"],     f'{row["gsax_2_0"]:+.1f}'),
        ("True Talent",   row["dsis_true_talent_gsax_per_shot_percentile"], f'{row["dsis_true_talent_gsax_per_shot"]:+.4f}'),
        ("Rebound Ctrl",  row.get("rci_percentile", 50), f'{row.get("rci", 0):.2f}' if pd.notna(row.get("rci")) else "N/A"),
        ("Traffic",       row["avg_traffic_percentile"],  f'{row["avg_traffic_percentile"]:.0f}th'),
        ("Shot Speed",    row["avg_speed_percentile"],    f'{row["avg_speed_percentile"]:.0f}th'),
        ("Lateral Demand",row["avg_movement_percentile"], f'{row["avg_movement_percentile"]:.0f}th'),
    ]

    y0 = 78
    for label, pct, val_str in metrics:
        ax.text(2, y0, label, fontsize=10, color=TEXT, va="center", fontweight="bold")
        ax.text(38, y0, val_str, fontsize=9, color=MUTED, va="center", ha="right")
        # bar background
        ax.add_patch(patches.Rectangle((40, y0 - 1.8), 48, 3.6, facecolor=BAR_BG, edgecolor="none"))
        # bar fill
        bar_w = (pct / 100) * 48
        bar_c = BAR_CMAP(pct / 100)
        ax.add_patch(patches.Rectangle((40, y0 - 1.8), bar_w, 3.6, facecolor=bar_c, edgecolor="none"))
        # percentile label
        ax.text(90, y0, f"{pct:.0f}%", fontsize=9, color=TEXT, va="center", fontweight="bold")
        y0 -= 10

    # Projection
    ax.plot([2, 98], [y0 + 3, y0 + 3], color=BORDER, lw=0.8)
    y0 -= 2
    proj = row["proj_1yr_talent_per_shot"]
    curr = row["dsis_true_talent_gsax_per_shot"]
    diff = proj - curr
    if abs(diff) < 0.0001:
        trend, tcol = "STABLE", MUTED
    elif diff > 0:
        trend, tcol = "TRENDING UP ↑", POS
    else:
        trend, tcol = "TRENDING DOWN ↓", NEG
    ax.text(2, y0, "1-Year Trajectory:", fontsize=10, color=TEXT, fontweight="bold", va="center")
    ax.text(35, y0, trend, fontsize=10, color=tcol, fontweight="bold", va="center")

    # Footer
    ax.text(50, 2, "Data: NHL API  ·  Model: CatBoost + PyMC DSIS + Kalman",
            fontsize=7, color="#bbb", ha="center", va="bottom")

    # ── RIGHT: shot map (45% width) ──
    ax_map = fig.add_axes([0.56, 0.12, 0.42, 0.78], facecolor=BG)
    draw_rink_zone(ax_map)

    # Plot shots
    try:
        conn = duckdb.connect()
        q = f"""
            SELECT adjusted_x, adjusted_y, is_goal
            FROM read_parquet('data/processed/scored_shots.parquet')
            WHERE goalie_in_net_id = {goalie_id} AND season = 20252026
            LIMIT 2000
        """
        shots = conn.execute(q).df()
        if not shots.empty:
            saves = shots[shots["is_goal"] == 0]
            goals = shots[shots["is_goal"] == 1]
            ax_map.scatter(saves["adjusted_x"], saves["adjusted_y"],
                           s=8, c=BLUE, alpha=0.12, linewidths=0)
            ax_map.scatter(goals["adjusted_x"], goals["adjusted_y"],
                           s=20, c=NEG, alpha=0.8, linewidths=0.3, edgecolors="#fff",
                           zorder=5)
        ax_map.text(62, 46, "Shots Faced — 2025-26", fontsize=9, color=TEXT,
                    ha="center", fontweight="bold")
        # legend
        ax_map.scatter([], [], s=8, c=BLUE, alpha=0.5, label="Save")
        ax_map.scatter([], [], s=20, c=NEG, label="Goal")
        ax_map.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08),
                      ncol=2, frameon=False, fontsize=8)
    except Exception as e:
        print(f"  Shot map failed for {goalie_id}: {e}")

    # Save
    out_dir = Path("outputs/cards")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = name.replace(" ", "_").lower() + "_card.png"
    path = out_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


if __name__ == "__main__":
    stats = pd.read_parquet("data/processed/card_stats.parquet")
    top = stats.sort_values("gsax_2_0", ascending=False).head(5)["goalie_id"].tolist()
    for gid in top:
        p = create_goalie_card(gid, stats)
        print(f"Generated: {p}")
