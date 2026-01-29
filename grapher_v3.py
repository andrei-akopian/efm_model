import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse
import json

def plot_grouped_averages(file_pattern="*_seed*.csv", output_file="comparison_plot.svg", json_config=None):
    # 1. Find Files
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(files)} files.")

    # 2. Group Files
    groups = {}
    for f in files:
        filename = os.path.basename(f)
        try:
            group_name = filename.split('_seed')[0]
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(f)
        except IndexError:
            print(f"Skipping file with unexpected format: {f}")
            continue

    print(f"Identified groups: {list(groups.keys())}")

    # 3. Load Background Config (if provided)
    sh_data = None
    if json_config:
        try:
            with open(json_config, 'r') as f:
                data = json.load(f)
                config = data.get('config', {})

                raw_colors = config.get('SH_COLORS', {})
                # Convert keys to int because JSON stores dict keys as strings
                sh_colors = {int(k): v for k, v in raw_colors.items()}

                sh_data = {
                    'schedule': config.get('SCHEDULE'),
                    'dt': config.get('DT'),
                    'colors': sh_colors
                }
                print(f"Loaded SH schedule from {json_config}")
        except Exception as e:
            print(f"Warning: Could not load JSON config: {e}")

    # 4. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Style Configurations
    colors = {'EC': 'grey', 'F': 'orange', 'M': 'green'}
    linestyles = ['-', ':', '-.', '--']
    global_max_ec = 0

    # 5. Process and Plot Each Group
    for i, (group_name, file_list) in enumerate(sorted(groups.items())):
        dfs = []
        for file in file_list:
            try:
                df = pd.read_csv(file)
                if 'Time' in df.columns:
                    df = df.set_index('Time')
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not dfs:
            continue

        combined = pd.concat(dfs)
        by_time = combined.groupby(combined.index)
        mean_df = by_time.mean()
        std_df = by_time.std()

        if 'EC_Count' in mean_df:
            current_max = (mean_df['EC_Count'] + std_df['EC_Count']).max()
            global_max_ec = max(global_max_ec, current_max)

        time = mean_df.index
        style = linestyles[i % len(linestyles)]
        alpha_val = 0.15

        # Top Graph (EC)
        if 'EC_Count' in mean_df:
            c = colors['EC']
            ax1.plot(time, mean_df['EC_Count'], label=f"{group_name} EC",
                     color=c, linestyle=style, linewidth=2.5)
            ax1.fill_between(time, mean_df['EC_Count'] - std_df['EC_Count'],
                             mean_df['EC_Count'] + std_df['EC_Count'], color=c, alpha=alpha_val)

        # Bottom Graph (F & M)
        if 'F_Count' in mean_df:
            c = colors['F']
            ax2.plot(time, mean_df['F_Count'], label=f"{group_name} F",
                     color=c, linestyle=style, linewidth=2)
            ax2.fill_between(time, mean_df['F_Count'] - std_df['F_Count'],
                             mean_df['F_Count'] + std_df['F_Count'], color=c, alpha=alpha_val)

        if 'M_Count' in mean_df:
            c = colors['M']
            ax2.plot(time, mean_df['M_Count'], label=f"{group_name} M",
                     color=c, linestyle=style, linewidth=2)
            ax2.fill_between(time, mean_df['M_Count'] - std_df['M_Count'],
                             mean_df['M_Count'] + std_df['M_Count'], color=c, alpha=alpha_val)

    # 6. Scaling & Backgrounds
    ax1.set_xlim(left=0)
    ax2.set_xlim(left=0)

    y_limit_top = global_max_ec * 1.1 if global_max_ec > 0 else 10
    y_limit_bottom = y_limit_top / 10

    ax1.set_ylim(bottom=0, top=y_limit_top)
    ax2.set_ylim(bottom=0, top=y_limit_bottom)

    # Draw SH Indicators if JSON was provided
    if sh_data:
        schedule = sh_data['schedule']
        dt = sh_data['dt']
        sh_colors = sh_data['colors']

        for ax, y_lim in [(ax1, y_limit_top), (ax2, y_limit_bottom)]:
            for start_step, end_step, level in schedule:
                start_time = start_step * dt
                end_time = end_step * dt

                # Draw colored background span
                # zorder=0 pushes it to the back
                ax.axvspan(start_time, end_time, color=sh_colors.get(int(level), 'white'), alpha=0.2, zorder=0)

                # Add text label
                mid_time = (start_time + end_time) / 2
                ax.text(mid_time, y_lim * 0.95, f"SH={level}",
                        ha='center', va='top', color='black', alpha=0.5, fontweight='bold', fontsize=10)

    # 7. Styling & Saving
    ax1.set_ylabel("Epithelial Cells (EC)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10, frameon=True)
    ax1.set_title("Epithelial Layer Growth", fontsize=14, fontweight='bold')

    ax2.set_ylabel("Stromal Cells (F & M)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10, frameon=True)
    ax2.set_title("Stromal Growth", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format='svg')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average cell counts from seeded CSV simulation files.")
    parser.add_argument("--pattern", type=str, default="*_seed*.csv", help="Glob pattern to match CSV files")
    parser.add_argument("--output", type=str, default="average_comparison.svg", help="Output filename")
    parser.add_argument("--json", type=str, default=None, help="Path to a simulation JSON file to extract SH schedule and colors")

    args = parser.parse_args()

    plot_grouped_averages(args.pattern, args.output, args.json)
