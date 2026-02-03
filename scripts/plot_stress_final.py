"""
Generate final stacked-area stress plots from the fixed stakes and
feedback datasets, with presentation-focused styling.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0

# Font sizes
AXIS_TICK_SIZE = 22
AXIS_LABEL_SIZE = 22
TITLE_SIZE = 24
LEGEND_SIZE = 22
MR_LABEL_SIZE = 22

# Palette A (blue + cream)
COLORS = {
    'strategic': '#F8CBA6',    # Apricot (bottom)
    'direct': '#FFE7CC',       # Light peach
    'failed': '#FFFBEB',       # Cream white
    'robust': '#ECF9FF',       # Light blue (top)
}

# Border color - gray
EDGE_COLOR = '#CCCCCC'

# Data
stakes_data = {
    'labels': ['High\n(Baseline)', 'Medium', 'Low'],
    'x': [0, 1, 2],
    'strategic': [33.55, 39.87, 58.82],
    'direct': [36.77, 39.87, 29.41],
    'failed': [3.87, 7.19, 8.50],
    'robust': [25.81, 13.07, 3.27],
}

feedback_data = {
    'labels': ['High', 'Medium\n(Baseline)', 'Low'],
    'x': [0, 1, 2],
    'strategic': [23.08, 33.55, 64.71],
    'direct': [41.03, 36.77, 33.99],
    'failed': [17.95, 3.87, 1.31],
    'robust': [17.95, 25.81, 0.00],
}

def _interp_extrapolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    y_new = np.interp(x_new, x, y)
    if len(x) < 2:
        return y_new
    left_mask = x_new < x[0]
    right_mask = x_new > x[-1]
    if np.any(left_mask):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        y_new[left_mask] = y[0] + slope * (x_new[left_mask] - x[0])
    if np.any(right_mask):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        y_new[right_mask] = y[-1] + slope * (x_new[right_mask] - x[-1])
    return y_new


def create_stacked_area(ax, data, title):
    """Create a stacked area chart."""
    x = np.array(data['x'])

    # Fill the full plot area.
    x_smooth = np.linspace(-0.5, 2.5, 100)

    # Compute cumulative values.
    y_strategic = np.array(data['strategic'])
    y_direct = y_strategic + np.array(data['direct'])
    y_failed = y_direct + np.array(data['failed'])
    y_robust = y_failed + np.array(data['robust'])

    # Interpolate.
    y1_smooth = _interp_extrapolate(x, y_strategic, x_smooth)
    y2_smooth = _interp_extrapolate(x, y_direct, x_smooth)
    y3_smooth = _interp_extrapolate(x, y_failed, x_smooth)
    y4_smooth = _interp_extrapolate(x, y_robust, x_smooth)

    # Draw stacked areas (gray borders).
    ax.fill_between(x_smooth, 0, y1_smooth, color=COLORS['strategic'], edgecolor=EDGE_COLOR, linewidth=1.0, label='Strategic')
    ax.fill_between(x_smooth, y1_smooth, y2_smooth, color=COLORS['direct'], edgecolor=EDGE_COLOR, linewidth=1.0, label='Direct')
    ax.fill_between(x_smooth, y2_smooth, y3_smooth, color=COLORS['failed'], edgecolor=EDGE_COLOR, linewidth=1.0, label='Failed')
    ax.fill_between(x_smooth, y3_smooth, y4_smooth, color=COLORS['robust'], edgecolor=EDGE_COLOR, linewidth=1.0, label='Robust')

    # ========== Emphasize MR boundary ==========
    # Compute MR values (boundary of Strategic + Direct).
    mr_values = [data['strategic'][i] + data['direct'][i] for i in range(len(x))]

    # Interpolate to extend the red line across the full plot.
    mr_smooth = _interp_extrapolate(x, np.array(mr_values), x_smooth)

    # Draw the red boundary line across the plot.
    ax.plot(x_smooth, mr_smooth,
            color='#8B0000',  # Dark red
            linewidth=2.5,
            zorder=4)

    # Draw MR points.
    for i in range(len(x)):
        mr_y = mr_values[i]
        ax.plot(x[i], mr_y, 'o',
                color='#8B0000',
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=1.5,
                zorder=5)

        # Annotate MR values (below points).
        ax.annotate(f'{mr_y:.1f}%',
                   xy=(x[i], mr_y),
                   xytext=(0, -18),
                   textcoords='offset points',
                   ha='center', va='top',
                   fontsize=MR_LABEL_SIZE, fontweight='bold',
                   color='#8B0000')

    # Set axes.
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], fontsize=AXIS_TICK_SIZE)
    ax.tick_params(axis='y', labelsize=AXIS_TICK_SIZE)
    ax.set_ylabel('Percentage (%)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=15)

    # Gridlines.
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#E5E5E5')
    ax.set_axisbelow(True)

    # Gray borders.
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stress final stacked areas.")
    parser.add_argument(
        "--outdir",
        default="outputs_misalign",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--stem",
        default="fig_stress_final",
        help="Output filename stem (without extension).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    create_stacked_area(axes[0], stakes_data, '(a) Effect of Stakes')
    create_stacked_area(axes[1], feedback_data, '(b) Effect of Feedback')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    png_path = outdir / f"{args.stem}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PNG: {png_path}")

    pdf_path = outdir / f"{args.stem}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF: {pdf_path}")

    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()
