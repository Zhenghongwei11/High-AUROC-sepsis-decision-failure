#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "figures"

# --- Style Configuration ---
COLORS = {
    "discovery": "#2C3E50",      # Dark Slate Blue
    "validation": "#95A5A6",     # Concrete Grey
    "transfer": "#E67E22",       # Carrot Orange (for stress test/transfer)
    "highlight": "#E74C3C",      # Alizarin Red (for failure/warning)
    "good": "#27AE60",           # Nephritis Green (for success)
    "neutral": "#ECF0F1",        # Clouds (backgrounds)
    "text_main": "#2C3E50",
    "text_light": "#7F8C8D",
}

def setup_theme() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.5,
    })

# --- Primitives ---

def draw_cylinder(ax, xy, width, height, color, label, sublabel=None):
    """Draws a database-like cylinder."""
    x, y = xy
    ellipse_height = height * 0.15
    
    # Bottom ellipse
    bottom = patches.Ellipse((x + width/2, y), width, ellipse_height, facecolor=color, edgecolor='none', zorder=2)
    # Body rectangle
    rect = patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='none', zorder=1)
    # Top ellipse
    top = patches.Ellipse((x + width/2, y + height), width, ellipse_height, facecolor=color, edgecolor='none', zorder=3)
    # Top rim (darker)
    top_rim = patches.Ellipse((x + width/2, y + height), width, ellipse_height, facecolor="none", edgecolor='white', linewidth=0.5, zorder=4, alpha=0.3)

    ax.add_patch(bottom)
    ax.add_patch(rect)
    ax.add_patch(top)
    ax.add_patch(top_rim)
    
    # Label
    ax.text(x + width/2, y + height/2 + 0.02, label, ha='center', va='center', color='white', fontweight='bold', fontsize=8.6, zorder=5)
    if sublabel:
        ax.text(x + width/2, y + height/2 - 0.03, sublabel, ha='center', va='center', color='white', fontsize=6.7, zorder=5, alpha=0.92)

def draw_pill(ax, xy, width, height, color, label, edgecolor=None, text_color='#2C3E50'):
    """Draws a pill-shaped button with configurable text color."""
    if edgecolor is None:
        edgecolor = color
    box = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.05", 
                                 facecolor=color, edgecolor=edgecolor, linewidth=1, zorder=2)
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, label, ha='center', va='center',
            color=text_color, fontsize=8.1, zorder=3, fontweight='bold')

def draw_arrow(ax, start, end, color="#7F8C8D", style="-|>", curve=0.0):
    """Draws a curved arrow."""
    arrow = patches.FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=15,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        zorder=1
    )
    ax.add_patch(arrow)

def draw_gaussian(ax, mean, std, height, color, label=None, linestyle='-'):
    """Draws a schematic Gaussian curve."""
    x = np.linspace(0, 10, 200)
    y = height * np.exp(-0.5 * ((x - mean) / std)**2)
    ax.plot(x, y, color=color, linewidth=2, linestyle=linestyle)
    if label:
        ax.text(mean, height + 0.05, label, ha='center', va='bottom', color=color, fontsize=8)
    return x, y

# --- Panel Renderers ---

def render_panel_a(ax):
    """Panel A: Cohorts & roles."""
    ax.set_title("A  Cohort sources and analytical roles", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Discovery Group
    draw_cylinder(ax, (0.08, 0.53), 0.22, 0.28, COLORS["discovery"], "GSE65682", "Discovery\nN = 359")
    
    # Validation Group
    draw_cylinder(ax, (0.47, 0.66), 0.17, 0.18, COLORS["validation"], "GSE95233", "External microarray\nN = 73")
    draw_cylinder(ax, (0.47, 0.38), 0.17, 0.18, COLORS["transfer"], "GSE154918", "RNA-seq transfer\nN = 91")
    draw_cylinder(ax, (0.47, 0.10), 0.17, 0.18, COLORS["highlight"], "GSE28750", "Inflammatory stress\nN = 41")

    # Connectors
    draw_arrow(ax, (0.33, 0.67), (0.45, 0.73), curve=-0.08)
    draw_arrow(ax, (0.33, 0.67), (0.45, 0.46), curve=0.00)
    draw_arrow(ax, (0.33, 0.67), (0.45, 0.19), curve=0.08)
    
    # Label moved and simplified for maximum clarity
    ax.text(0.79, 0.46, "External cohorts", ha='center', va='center', fontsize=9.2,
            color=COLORS["discovery"], fontweight='bold', rotation=-90)

def render_panel_b(ax):
    """Panel B: Benchmark structure."""
    ax.set_title("B  Benchmark structure", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Input Box
    draw_pill(ax, (0.05, 0.74), 0.28, 0.14, COLORS["neutral"], "Harmonized\nphenotypes")
    
    # Task A
    draw_pill(ax, (0.54, 0.79), 0.38, 0.11, "#D6EAF8", "Primary analysis\nSepsis vs healthy")
    ax.text(0.73, 0.775, "N = 275; 3 external readouts", ha='center', va='top', fontsize=6.8, color=COLORS["text_light"])
    
    # Task B
    draw_pill(ax, (0.54, 0.50), 0.38, 0.11, "#FAD7A0", "Secondary analysis\nSepsis vs inflammation")
    ax.text(0.73, 0.485, "N = 147; clinically harder comparator", ha='center', va='top', fontsize=6.8, color=COLORS["text_light"])
    
    # Task C
    draw_pill(ax, (0.54, 0.21), 0.38, 0.11, "#D2B4DE", "Nested readout\nMicroarray to RNA-seq")
    ax.text(0.73, 0.195, "Platform-specific transfer view", ha='center', va='top', fontsize=6.8, color=COLORS["text_light"])

    # Arrows
    draw_arrow(ax, (0.37, 0.82), (0.53, 0.86), curve=0)
    draw_arrow(ax, (0.37, 0.82), (0.53, 0.56), curve=0.1)
    draw_arrow(ax, (0.37, 0.82), (0.53, 0.26), curve=0.2)

def render_panel_c(ax):
    """Panel C: Workflow components."""
    ax.set_title("C  Workflow components", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Steps labels
    ax.text(0.15, 0.90, "Preprocessing", ha='center', fontweight='bold', fontsize=8.8)
    ax.text(0.50, 0.90, "Model", ha='center', fontweight='bold', fontsize=8.8)
    ax.text(0.84, 0.90, "Validation", ha='center', fontweight='bold', fontsize=8.8)

    # 1. Preprocessing Branches
    y_starts = [0.7, 0.5, 0.3]
    labels = ["Standard\nz-score", "Rank-based\nz-score", "Robust\nscaling"]
    colors = ["#F2F3F4", "#D5F5E3", "#D5F5E3"] # Grey for bad, Green for good
    
    for y, lbl, col in zip(y_starts, labels, colors):
        draw_pill(ax, (0.02, y), 0.26, 0.12, col, lbl)
        draw_arrow(ax, (0.30, y+0.06), (0.40, 0.55), curve=0) # Converge to model

    # 2. Model
    draw_pill(ax, (0.42, 0.45), 0.16, 0.2, COLORS["discovery"], "Logistic\nbaseline",
              edgecolor="none", text_color='white')
    ax.text(0.5, 0.42, "Interpretable linear model", ha='center', va='center', color='white', fontsize=6.8, zorder=4)

    # 3. Validation
    draw_arrow(ax, (0.60, 0.55), (0.70, 0.55), curve=0)
    # Narrower pill to make room for metrics
    draw_pill(ax, (0.65, 0.40), 0.18, 0.3, COLORS["neutral"], "External\nvalidation")
    
    # Metrics - Shifted further right to avoid overlap
    ax.text(0.90, 0.60, "AUROC", ha='center', fontsize=7.0)
    ax.text(0.90, 0.52, "Balanced accuracy", ha='center', fontsize=7.0)
    ax.text(0.90, 0.44, "Calibration and score scale", ha='center', fontsize=7.0)

def render_panel_d(ax):
    """Panel D: Threshold collapse under transfer."""
    ax.set_title("D  Threshold collapse under transfer", loc="left", color=COLORS["highlight"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Subplot 1: Ideal (Internal)
    ax.text(2.5, 4.45, "Internal development setting", ha='center', fontweight='bold', color=COLORS["good"], fontsize=8.5)
    # Distributions
    draw_gaussian(ax, 1.5, 0.6, 3, COLORS["validation"], linestyle='--') # Healthy
    draw_gaussian(ax, 3.5, 0.6, 3, COLORS["discovery"]) # Sepsis
    # Threshold
    ax.axvline(x=2.5, ymin=0.1, ymax=0.7, color='black', linestyle=':', linewidth=1)
    ax.text(2.5, 3.45, "Fixed threshold", ha='center', fontsize=6.9, backgroundcolor='white')
    ax.text(2.5, 0.50, "Usable decision boundary", ha='center', fontsize=7.6, color=COLORS["good"], fontweight='bold')

    # Separator
    ax.axvline(x=5, color='#BDC3C7', linewidth=1)

    # Subplot 2: Failure (External Z-score)
    ax.text(7.5, 4.45, "External transfer setting", ha='center', fontweight='bold', color=COLORS["highlight"], fontsize=8.5)
    # Shifted Distributions (Simulating mean shift)
    draw_gaussian(ax, 0.5, 0.6, 3, COLORS["validation"], linestyle='--') # Healthy (Shifted Left)
    draw_gaussian(ax, 2.0, 0.6, 3, COLORS["discovery"]) # Sepsis (Shifted Left)
    # The Locked Threshold (Still at relative 2.5 position from training, but data moved)
    # Visually, if data moves left, the old threshold (2.5) is now way to the right
    ax.axvline(x=2.5 + 5, ymin=0.1, ymax=0.7, color='black', linestyle=':', linewidth=1) # Threshold stays fixed
    ax.text(7.5, 3.45, "Same fixed threshold", ha='center', fontsize=6.9, backgroundcolor='white')
    
    # Annotation of failure
    ax.annotate("", xy=(2.5, 2), xytext=(4.5, 2), arrowprops=dict(arrowstyle="->", color=COLORS["highlight"]))
    ax.text(7.5, 1.55, "Scores shift left after transfer", ha='center', fontsize=7.6, color=COLORS["highlight"])
    ax.text(7.5, 0.48, "Decision failure despite preserved ranking", ha='center', fontsize=7.3, color=COLORS["highlight"], fontweight='bold')


def render() -> None:
    setup_theme()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 8.4), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1.2])

    ax_a = fig.add_subplot(gs[0, 0])
    render_panel_a(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    render_panel_b(ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    render_panel_c(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    render_panel_d(ax_d)

    # Final Polish
    fig.suptitle("Study design and analytical framework of the sepsis transcriptomic transportability benchmark", fontsize=13, fontweight="bold", color="#2C3E50", y=1.02)
    
    png_path = OUTDIR / "figure1_design_overview.png"
    pdf_path = OUTDIR / "figure1_design_overview.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Generated figures at:\n{png_path}\n{pdf_path}")
    plt.close(fig)

if __name__ == "__main__":
    render()
