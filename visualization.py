import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import json
import time
import sys
import argparse
import os

# --- Config ---

EC_Y_AXIS_LABEL = "# of Epithelial cells"
F_Y_AXIS_LABEL = "# of Fibroblasts"
M_Y_AXIS_LABEL = "# of Macrophages"

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description='Visualize multiple cell simulations vertically.')
parser.add_argument('files', metavar='F', type=str, nargs='*',
                    help='List of JSON simulation files to visualize')
parser.add_argument('--release', action='store_true',
                    help='Render at full FPS. Default is fast preview (low FPS).')
parser.add_argument('--gif', action='store_true',
                    help='Output a GIF file.')
# parser.add_argument('--mp4', action='store_true',
#                     help='Output a MP4 file (MP4 is default).')
parser.add_argument('--output-mark', action='store', type=str,
                    help='Mark output filename end.')
parser.add_argument('--speedup', action='store_true',
                    help='Speed up final video playback by 1.5x.')

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

if not args.files:
    print("Error: No simulation files provided.")
    sys.exit(1)

# --- Load & Validate Data ---
simulations = []
print(f"Loading {len(args.files)} simulation files...")

base_config = None

for filepath in args.files:
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

            if base_config is None:
                base_config = data["config"]
            else:
                curr_config = data["config"]
                if abs(curr_config["SIM_DURATION"] - base_config["SIM_DURATION"]) > 0.001:
                    raise ValueError(f"Duration mismatch in {filepath}")
                if curr_config["SCHEDULE"] != base_config["SCHEDULE"]:
                    raise ValueError(f"Schedule mismatch in {filepath}")

            simulations.append({
                "filename": os.path.basename(filepath),
                "config": data["config"],
                "history": data["history"]
            })

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

print("Validation successful.")

# --- Shared Config ---
config = base_config
FPS = config["FPS"]
DT = config["DT"]
SIM_DURATION = config["SIM_DURATION"]
SCHEDULE = config["SCHEDULE"]
SH_COLORS = {int(k): v for k, v in config["SH_COLORS"].items()}
E_HEIGHT = config["E_HEIGHT"]
ECM_HEIGHT = config["ECM_HEIGHT"]
F_HEIGHT = config["F_HEIGHT"]
M_HEIGHT = config["M_HEIGHT"]

# --- Frame Skipping & Speed Control ---
speed_multiplier = 1.5 if args.speedup else 1.0

if args.release:
    render_step = 1
    # Increase the playback FPS if speedup is requested
    writer_fps = FPS * speed_multiplier
    print(f"Release mode: Full FPS (Playback Speed: {speed_multiplier}x).")
else:
    target_fps = 5
    render_step = max(1, int(FPS / target_fps))
    # Adjust writer FPS to maintain relative speed
    writer_fps = (FPS / render_step) * speed_multiplier
    print(f"Test mode: Every {render_step}th frame (Playback Speed: {speed_multiplier}x).")

# --- Geometry & Viewport ---
MAX_CIRCUMFERENCE = 8.0
MAX_RADIUS = MAX_CIRCUMFERENCE / np.pi

print("Pre-computing global limits...")
global_max_width = 0
global_max_ec_count = 0
global_max_fm_count = 0

for sim in simulations:
    for step in sim['history']:
        if step['ec_total_width'] > global_max_width:
            global_max_width = step['ec_total_width']

        # Count Maxima
        e_c = len([x for x in step['e'] if not x['dying']])
        f_c = len([x for x in step['f'] if not x['dying']])
        m_c = len([x for x in step['m'] if not x['dying']])

        global_max_ec_count = max(global_max_ec_count, e_c)
        global_max_fm_count = max(global_max_fm_count, f_c, m_c)

# Padding for graphs
global_max_ec_count = int(global_max_ec_count * 1.2) if global_max_ec_count > 0 else 10
global_max_fm_count = int(global_max_fm_count * 1.5) if global_max_fm_count > 0 else 5

# Spatial View Limits
if global_max_width > MAX_CIRCUMFERENCE:
    max_straight_len = (global_max_width - MAX_CIRCUMFERENCE) / 2
    max_r = MAX_RADIUS
else:
    max_straight_len = 0
    max_r = global_max_width / np.pi if global_max_width > 0 else 0.1

total_layer_height = E_HEIGHT + ECM_HEIGHT + F_HEIGHT + M_HEIGHT

# Updated Padding: Reduced to remove vertical whitespace
padding = 0.5
text_headroom = 2.5

view_x_min = -2.0
view_x_max = max_straight_len + max_r + total_layer_height + padding
view_y_limit = max_r + total_layer_height + padding + text_headroom

# --- Alignment Logic ---
spatial_width = view_x_max - view_x_min
dist_to_wall = 0 - view_x_min
wall_ratio = dist_to_wall / spatial_width

time_view_max = SIM_DURATION
time_view_min = - (SIM_DURATION * wall_ratio) / (1 - wall_ratio)

print(f"Alignment: Wall at ratio {wall_ratio:.2f}. Graph Axis: {time_view_min:.1f} to {time_view_max:.1f}")

# --- Figure Layout ---
num_sims = len(simulations)
data_height_per_sim = 2 * view_y_limit
fig_width_inch = 12.0

# Margin definitions (must match subplots_adjust below)
margin_left = 0.05
margin_right = 0.10 # 1.0 - 0.90
margin_top = 0.05   # 1.0 - 0.95
margin_bottom = 0.05

effective_width_ratio = 1.0 - (margin_left + margin_right)
effective_height_ratio = 1.0 - (margin_top + margin_bottom)

# Calculate Effective Width in Inches available for the axes
eff_width_inch = fig_width_inch * effective_width_ratio

# Calculate required Axis Height in Inches to maintain Aspect Ratio with Data
# aspect = Height / Width
req_axis_height_inch = eff_width_inch * (data_height_per_sim / spatial_width)

# Calculate the Figure-level height allocation needed to result in that Axis Height
# h_sim_fig * effective_height_ratio = req_axis_height_inch (approximately, for the subplot slot)
h_sim = req_axis_height_inch / effective_height_ratio

h_graph = 1.8
h_time = 1.0

ratios = []
for _ in range(num_sims):
    ratios.append(h_sim)
    ratios.append(h_graph)
ratios.append(h_time)

total_fig_height = sum(ratios)

fig, axes_array = plt.subplots(len(ratios), 1,
                               figsize=(fig_width_inch, total_fig_height),
                               gridspec_kw={'height_ratios': ratios})
# Updated Margins
fig.subplots_adjust(hspace=0, left=margin_left, right=1.0-margin_right, top=1.0-margin_top, bottom=margin_bottom)

# --- Pre-process Graph Data ---
graph_objs = []
graph_colors = {'E': config['COLORS']['lob'], 'F': config['COLORS']['f'], 'M': config['COLORS']['m']}
ax_timeline = axes_array[-1]

for i, sim in enumerate(simulations):
    ax_graph_idx = i * 2 + 1
    ax_graph = axes_array[ax_graph_idx]

    # Data
    times = [step['time'] for step in sim['history']]
    e_counts = [len([x for x in step['e'] if not x['dying']]) for step in sim['history']]
    f_counts = [len([x for x in step['f'] if not x['dying']]) for step in sim['history']]
    m_counts = [len([x for x in step['m'] if not x['dying']]) for step in sim['history']]

    # --- Primary Axis (Left - EC) ---
    ax_graph.plot(times, e_counts, color=graph_colors['E'], label='EC', linewidth=1.5)

    # Style: Move Spine to x=0 (The Black Vertical)
    ax_graph.spines['left'].set_position(('data', 0))
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)
    ax_graph.spines['bottom'].set_visible(False)

    ax_graph.set_ylabel(EC_Y_AXIS_LABEL, fontsize=10, fontweight='bold')
    ax_graph.yaxis.set_label_coords(0.00, 0.5)
    ax_graph.tick_params(axis='y')
    ax_graph.set_ylim(0, global_max_ec_count)

    # --- Secondary Axis (Right - F/M) ---
    ax_right = ax_graph.twinx()
    ax_right.plot(times, f_counts, color=graph_colors['F'], label='F', linewidth=1.5)
    ax_right.plot(times, m_counts, color=graph_colors['M'], label='M', linewidth=1.5)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    ax_right.text(
        1.07, 0.5,
        F_Y_AXIS_LABEL,
        color="orange",
        rotation=90,
        va="center",
        ha="right",
        transform=ax_graph.transAxes,
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
        transform=ax_graph.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    ax_right.tick_params(axis='y', labelcolor='black')
    ax_right.set_ylim(0, global_max_fm_count)

    # Combined styling
    ax_graph.set_xlim(time_view_min, time_view_max)
    ax_graph.grid(True, linestyle='--', alpha=0.3)
    ax_graph.set_xticks([])
    ax_right.set_xticks([])

    # Moving Marker
    marker = ax_graph.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)
    graph_objs.append({'marker': marker})

    # The Wall visual on graph
    ax_graph.axvline(x=0, color='black', linewidth=3, zorder=10)


# --- Helper Functions ---
def get_u_shape_coords(linear_p, layer_offset, total_width):
    if total_width <= MAX_CIRCUMFERENCE:
        R = total_width / np.pi
        if R == 0: R = 0.001
        angle = ((linear_p / total_width) * np.pi) - (np.pi / 2)
        r_eff = R + layer_offset
        return r_eff * np.cos(angle), r_eff * np.sin(angle)
    else:
        R = MAX_RADIUS
        S = (total_width - MAX_CIRCUMFERENCE) / 2
        if linear_p < S:
            return linear_p, -(R + layer_offset)
        elif linear_p > (S + MAX_CIRCUMFERENCE):
            dist_from_end = total_width - linear_p
            return dist_from_end, (R + layer_offset)
        else:
            p_curve = linear_p - S
            angle = ((p_curve / MAX_CIRCUMFERENCE) * np.pi) - (np.pi / 2)
            r_eff = R + layer_offset
            return S + (r_eff * np.cos(angle)), r_eff * np.sin(angle)

def create_layer_polygon(cell_data, layer_bottom, layer_top, total_width, color, alpha=1.0):
    p_start = cell_data['left'] + (total_width / 2)
    p_end = cell_data['right'] + (total_width / 2)
    steps = int(max(2, (p_end - p_start) / 0.1))
    p_values = np.linspace(p_start, p_end, steps)
    inner_verts = [get_u_shape_coords(p, layer_bottom, total_width) for p in p_values]
    outer_verts = [get_u_shape_coords(p, layer_top, total_width) for p in reversed(p_values)]
    verts = inner_verts + outer_verts
    return patches.Polygon(verts, closed=True, facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha, zorder=2)

def update_frame(frame_idx):
    current_time = simulations[0]['history'][frame_idx]['time']

    for i, sim in enumerate(simulations):
        ax_sim = axes_array[i * 2]
        ax_sim.clear()

        # Update Marker
        graph_objs[i]['marker'].set_xdata([current_time])

        step_data = sim['history'][frame_idx]
        sim_colors = sim['config']['COLORS']
        total_width = step_data['ec_total_width']
        if total_width < 0.1: total_width = 0.1

        # Layers
        r_ec_in = 0; r_ec_out = E_HEIGHT
        r_ecm_in = r_ec_out; r_ecm_out = r_ecm_in + ECM_HEIGHT
        r_fib_in = r_ecm_out; r_fib_out = r_fib_in + F_HEIGHT
        r_mac_in = r_fib_out; r_mac_out = r_mac_in + M_HEIGHT

        # Draw Cells
        for e in step_data['e']:
            conc = e.get('ecm', 0)
            width = e.get('width', 1)
            a = min(1.0, conc / width) if width > 0 else 0
            if a > 0.05:
                poly = create_layer_polygon(e, r_ecm_in, r_ecm_out, total_width, sim_colors['ecm'], alpha=a)
                poly.set_edgecolor('none')
                ax_sim.add_patch(poly)

        for e in step_data['e']: ax_sim.add_patch(create_layer_polygon(e, r_ec_in, r_ec_out, total_width, e['color']))
        for f in step_data['f']: ax_sim.add_patch(create_layer_polygon(f, r_fib_in, r_fib_out, total_width, f['color']))
        for m in step_data['m']: ax_sim.add_patch(create_layer_polygon(m, r_mac_in, r_mac_out, total_width, m['color']))

        # Wall
        ax_sim.plot([0, 0], [-view_y_limit, view_y_limit], color='black', linewidth=3)
        ax_sim.set_xlim(view_x_min, view_x_max)
        ax_sim.set_ylim(-view_y_limit, view_y_limit)
        ax_sim.set_aspect('equal')
        ax_sim.axis('off')

        # Text
        e_count = len([x for x in step_data['e'] if not x['dying']])
        f_count = len([x for x in step_data['f'] if not x['dying']])
        m_count = len([x for x in step_data['m'] if not x['dying']])
        fname = sim['filename'].split('.')[0]
        title_str = (f"{fname}\n"
                     f"EC: {e_count:<3d} | F: {f_count:<3d} | M: {m_count:<3d}")
        # Updated text position to account for tighter headroom
        ax_sim.text(0.2, view_y_limit - 0.5, title_str, family='monospace', fontsize=14, fontweight='bold', ha='left', va='top')

    # Timeline
    ax_timeline.clear()
    ax_timeline.set_xlim(time_view_min, time_view_max)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.axis('off')

    for start_step, end_step, level in SCHEDULE:
        start_t, end_t = start_step * DT, end_step * DT
        rect = plt.Rectangle((start_t, 0), end_t - start_t, 1, color=SH_COLORS[level], alpha=0.8)
        ax_timeline.add_patch(rect)
        ax_timeline.text((start_t + end_t)/2, 0.5, f"SH={level}", ha='center', va='center', fontweight='bold', color='white', fontsize=12)

    ax_timeline.axvline(x=0, color='black', linewidth=3)
    ax_timeline.axvline(x=current_time, color='black', linewidth=3, alpha=0.5)

    ax_timeline.text(current_time, 0.5, f" Time: {current_time:5.1f}s ",
                     ha='left', va='center', fontsize=14, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))

# --- Render Setup ---
total_frames = len(simulations[0]['history'])

# Smart Filename Generation
descriptors = []
for sim in simulations:
    init_counts = sim['config'].get('INITIAL_COUNTS')
    if init_counts:
         desc = f"E{init_counts.get('E', '?')}_F{init_counts.get('F', '?')}_M{init_counts.get('M', '?')}"
    else:
        h0 = sim['history'][0]
        e0 = len([x for x in h0['e'] if not x['dying']])
        f0 = len([x for x in h0['f'] if not x['dying']])
        m0 = len([x for x in h0['m'] if not x['dying']])
        desc = f"E{e0}_F{f0}_M{m0}"
    descriptors.append(desc)

sim_names = "_vs_".join(descriptors)
if len(sim_names) > 200:
    sim_names = sim_names[:200] + "_etc"

if args.output_mark:
    sim_names = sim_names[:-len(args.output_mark)] + args.output_mark

base_filename = f'vis_{sim_names}'
frames_to_render = list(range(0, total_frames, render_step))

print(f"Rendering {len(frames_to_render)} frames...")

# --- Writers Setup ---
writers = []

if args.gif:
    gif_out = base_filename + ".gif"
    print(f" -> Outputting GIF: {gif_out}")
    w_gif = PillowWriter(fps=writer_fps)
    w_gif.setup(fig, gif_out, dpi=100)
    writers.append(w_gif)
else:
    mp4_out = base_filename + ".mp4"
    print(f" -> Outputting MP4: {mp4_out}")
    w_mp4 = FFMpegWriter(fps=writer_fps)
    w_mp4.setup(fig, mp4_out, dpi=100)
    writers.append(w_mp4)

# --- Render Loop ---
ani = FuncAnimation(fig, update_frame, frames=frames_to_render)
start_time = time.time()
count = 0

for i in frames_to_render:
    update_frame(i)
    for w in writers:
        w.grab_frame()
    count += 1
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps_proc = count / elapsed
        bar_len = 30
        filled = int(bar_len * count / len(frames_to_render))
        bar = '█' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r|{bar}| Frame {count}/{len(frames_to_render)} | {fps_proc:.2f} frames/s |")
        sys.stdout.flush()

print("\nFinalizing files...")
for w in writers:
    w.finish()

print("Done!")
