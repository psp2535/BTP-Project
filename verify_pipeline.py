import os
import subprocess
import pandas as pd
import json
import sys

def verify():
    print("====================================================")
    print("Starting Pipeline Verification for IM-VRM Baseline")
    print("====================================================")
    
    # 1. Run main.py using python subprocess with a small taxi count
    cmd = [sys.executable, "main.py", "--num-taxis", "5", "--step", "all"]
    print(f"Running pipeline command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline execution stdout output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Pipeline execution FAILED with return code", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        sys.exit(1)
        
    # 2. Check generated files
    expected_files = {
        "data/processed/clean_trips.csv": [
            "taxi_id", "timestamp", "longitude", "latitude", "trip_id", 
            "delta_dist_m", "delta_time_sec", "speed_kmh", 
            "fuel_light_l", "co2_light_g", "fuel_heavy_l", "co2_heavy_g"
        ],
        "data/processed/grid_congestion_stats.csv": [
            "grid_row", "grid_col", "hour", "avg_speed", "count", 
            "free_flow_speed", "speed_ratio", "congestion_level"
        ],
        "data/processed/cell_free_flow_speeds.csv": [
            "grid_row", "grid_col", "free_flow_speed"
        ],
        "data/processed/graph_edges.csv": [
            "grid_row", "grid_col", "next_row", "next_col", 
            "transition_count", "avg_speed_kmh", "avg_duration_sec"
        ],
        "data/processed/graph_nodes.json": None,
        "data/processed/synthetic_fleet.json": None
    }
    
    print("\nVerifying output files...")
    failed = False
    
    for filepath, columns in expected_files.items():
        if not os.path.exists(filepath):
            print(f"[-] ERROR: Expected file missing: {filepath}")
            failed = True
            continue
            
        print(f"[+] File exists: {filepath}")
        
        # Verify formats and schemas
        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
                if df.empty:
                    print(f"    [-] WARNING: {filepath} is empty!")
                    failed = True
                else:
                    print(f"    [+] {filepath} contains {len(df)} rows.")
                    
                # Check column headers
                if columns:
                    missing_cols = [col for col in columns if col not in df.columns]
                    if missing_cols:
                        print(f"    [-] ERROR: Missing columns in {filepath}: {missing_cols}")
                        failed = True
                    else:
                        print(f"    [+] {filepath} columns validated.")
                        
            elif filepath.endswith(".json"):
                with open(filepath, "r") as f:
                    data = json.load(f)
                if not data:
                    print(f"    [-] WARNING: {filepath} is empty!")
                    failed = True
                else:
                    if "synthetic_fleet.json" in filepath:
                        print(f"    [+] Fleet loaded: {len(data['fleet'])} vehicles, {len(data['depots'])} depots.")
                    else:
                        print(f"    [+] Graph nodes loaded: {len(data)} nodes.")
        except Exception as e:
            print(f"[-] ERROR Reading {filepath}: {str(e)}")
            failed = True

    print("====================================================")
    if failed:
        print("VERIFICATION FAILED: Some outputs were missing or invalid.")
        sys.exit(1)
    else:
        print("VERIFICATION SUCCESS: All baseline files and schemas are correct!")
        print("The repository structure is fully verified and ready for research.")
    print("====================================================")

if __name__ == "__main__":
    verify()
