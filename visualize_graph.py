import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import sys

# --- CONFIG ----

EC_Y_AXIS_LABEL = "# of Epithelial cells"
F_Y_AXIS_LABEL = "# of Fibroblasts"
M_Y_AXIS_LABEL = "# of Macrophages"
SECONDARY_AXIS_SCALE = 0.1  # Ratio of Secondary Axis (F/M) to Primary Axis (EC)

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate static graph of cell counts from simulation.')
parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='List of JSON simulation files to visualize')
parser.add_argument('--output', type=str, help='Optional manual output filename', default=None)

args = parser.parse_args()

# --- Load Data ---
simulations = []
print(f"Loading {len(args.files)} simulation files...")

raw_max_ec = 0
raw_max_fm = 0
max_duration = 0
base_config = None

for filepath in args.files:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

            # Validation (Basic check to ensure they are comparable)
            if base_config is None:
                base_config = data["config"]

            simulations.append({
                'filename': os.path.basename(filepath),
                'history': data['history'],
                'config': data['config']
            })

            # Calculate Raw Maxima
            for step in data['history']:
                e = len([x for x in step['e'] if not x['dying']])
                f_c = len([x for x in step['f'] if not x['dying']])
                m = len([x for x in step['m'] if not x['dying']])

                raw_max_ec = max(raw_max_ec, e)
                raw_max_fm = max(raw_max_fm, f_c, m)

            max_duration = max(max_duration, data['config']['SIM_DURATION'])

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

# --- Calculate Axis Limits with Fixed Ratio ---
# We need the Primary Axis (EC) to be large enough to hold raw_max_ec
# AND large enough that Primary * SCALE >= raw_max_fm (so the secondary data fits too)
min_required_ec_height = max(raw_max_ec, raw_max_fm / SECONDARY_AXIS_SCALE)

# Add padding (1.2x)
global_max_ec = int(min_required_ec_height * 1.2) if min_required_ec_height > 0 else 10

# Derive secondary limit strictly from primary limit
global_max_fm = round(global_max_ec * SECONDARY_AXIS_SCALE, 4)

print(f"Axis Scaling: EC Limit={global_max_ec}, F/M Limit={global_max_fm} (Ratio: {SECONDARY_AXIS_SCALE})")

# --- Setup Plotting ---
num_sims = len(simulations)
# Dynamic height: 3 inches per simulation
fig, axes = plt.subplots(num_sims, 1, figsize=(10, 3 * num_sims), sharex=True, constrained_layout=True)

# Ensure axes is a list even if 1 sim
if num_sims == 1:
    axes = [axes]

graph_colors = {'E': base_config['COLORS']['lob'], 'F': base_config['COLORS']['f'], 'M': base_config['COLORS']['m']}

# Extract SH Colors from first config (assuming consistency)
sh_colors_raw = simulations[0]['config']['SH_COLORS']
SH_COLORS = {int(k): v for k, v in sh_colors_raw.items()}

print("Generating plots...")

for i, sim in enumerate(simulations):
    ax = axes[i]
    history = sim['history']
    config = sim['config']

    times = [step['time'] for step in history]
    e_counts = [len([x for x in step['e'] if not x['dying']]) for step in history]
    f_counts = [len([x for x in step['f'] if not x['dying']]) for step in history]
    m_counts = [len([x for x in step['m'] if not x['dying']]) for step in history]

    # --- Background: SH Phases ---
    schedule = config['SCHEDULE']
    dt = config['DT']
    for start_step, end_step, level in schedule:
        start_time = start_step * dt
        end_time = end_step * dt
        ax.axvspan(start_time, end_time, color=SH_COLORS[int(level)], alpha=0.3, zorder=0)
        # Add label for SH level
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, global_max_ec * 0.95, f"SH={level}",
                ha='center', va='top', color='black', alpha=0.5, fontweight='bold', fontsize=9)

    # --- Primary Axis (Left) - EC ---
    ax.plot(times, e_counts, color=graph_colors['E'], label='EC', linewidth=2, zorder=3)
    ax.set_ylabel(EC_Y_AXIS_LABEL, fontweight='bold')
    ax.tick_params(axis='y')
    ax.set_ylim(0, global_max_ec)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    # --- Secondary Axis (Right) - F / M ---
    ax_right = ax.twinx()
    ax_right.plot(times, f_counts, color=graph_colors['F'], label='F', linewidth=2, zorder=3)
    ax_right.plot(times, m_counts, color=graph_colors['M'], label='M', linewidth=2, zorder=3)

    # Manual Legend/Labels for secondary axis
    ax_right.text(
        1.07, 0.5,
        F_Y_AXIS_LABEL,
        color="orange",
        rotation=90,
        va="center",
        ha="right",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    ax_right.text(
        1.07, 0.5,
        M_Y_AXIS_LABEL,
        color="green",
        rotation=90,
        va="center",
        ha="left",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    ax_right.set_ylim(0, global_max_fm)

    # --- Styling ---
    title = sim['filename'].replace('.json', '')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=10)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax_right.spines['top'].set_visible(False)

# X-Label for bottom plot
axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
axes[-1].set_xlim(0, max_duration)

# --- Saving ---
if args.output:
    outfile = args.output
else:
    # Generate filename based on cell counts
    descriptors = []
    for sim in simulations:
        conf = sim['config']
        if 'INITIAL_COUNTS' in conf:
            ic = conf['INITIAL_COUNTS']
            desc = f"E{ic.get('E','?')}_F{ic.get('F','?')}_M{ic.get('M','?')}"
        else:
            h0 = sim['history'][0]
            desc = f"E{len(h0['e'])}_F{len(h0['f'])}_M{len(h0['m'])}"
        descriptors.append(desc)

    name_str = "_vs_".join(descriptors)
    if len(name_str) > 100: name_str = name_str[:100] + "_etc"
    outfile = f"graph_{name_str}.png"

plt.savefig(outfile, dpi=150)
print(f"Done. Graph saved to: {outfile}")
