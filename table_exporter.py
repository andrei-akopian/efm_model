import json
import csv
import argparse
import sys
import os

def export_simulation_to_csv(input_file, output_file=None):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        history = data.get('history', [])
        if not history:
            print(f"Warning: No history found in {input_file}")
            return

        if not output_file:
            output_file = os.path.splitext(input_file)[0] + ".csv"

        print(f"Exporting {len(history)} steps from '{input_file}' to '{output_file}'...")

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Time', 'EC_Count', 'F_Count', 'M_Count'])

            for step in history:
                t = step['time']

                e_count = len([c for c in step['e'] if not c.get('dying', False)])
                f_count = len([c for c in step['f'] if not c.get('dying', False)])
                m_count = len([c for c in step['m'] if not c.get('dying', False)])

                writer.writerow([f"{t:.2f}", e_count, f_count, m_count])

        print(f"Done. Saved to {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert simulation JSON to CSV (Time + 3 Cell Counts).')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='JSON simulation file(s) to convert')

    args = parser.parse_args()

    for file in args.files:
        export_simulation_to_csv(file)
