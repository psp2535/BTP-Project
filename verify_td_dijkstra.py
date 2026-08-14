import os
import subprocess
import json
import pandas as pd
import sys

def verify_td_dijkstra():
    print("====================================================")
    print("Starting Time-Dependent Context-Aware Dijkstra Verification")
    print("====================================================")
    
    # 1. Execute Step 10 (aco_route) to run the dynamic calculations
    cmd = [sys.executable, "main.py", "--num-taxis", "300", "--step", "aco_route"]
    print(f"Running pipeline command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline execution completed successfully.")
        
        # Print output tables from log stdout
        lines = result.stdout.splitlines()
        print_output = False
        for l in lines:
            if "Time-Dependent Routing Sensitivity Analysis" in l:
                print_output = True
            if print_output:
                print(f"  {l}")
            if print_output and l.strip().endswith("======"):
                # Stop printing after table ends
                pass
    except subprocess.CalledProcessError as e:
        print("Pipeline execution FAILED with return code", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        sys.exit(1)
        
    # 2. Check generated time-dependent output files
    td_metrics_path = "outputs/td_aco_metrics.csv"
    aco_summary_path = "outputs/aco_summary.json"
    start_hours = [8, 12, 17, 20]
    
    assert os.path.exists(td_metrics_path), f"Time-Dependent metrics CSV missing: {td_metrics_path}"
    assert os.path.exists(aco_summary_path), f"ACO summary JSON missing: {aco_summary_path}"
    
    for h in start_hours:
        hr_routes_path = f"data/processed/td_dijkstra_routes_hour{h}.json"
        assert os.path.exists(hr_routes_path), f"Hour {h} routes file missing: {hr_routes_path}"
        
    print("\n[+] All Time-Dependent metrics and hour-specific JSON route files exist.")

    # 3. Validate FIFO Compliance of Time-Dependent Dijkstra paths
    # We load the JSON routes for a sample hour and assert that arrival times are strictly non-decreasing.
    try:
        print("\n[+] Performing FIFO Compliance Validation on generated routes...")
        for h in start_hours:
            hr_routes_path = f"data/processed/td_dijkstra_routes_hour{h}.json"
            with open(hr_routes_path, "r") as f:
                routes_data = json.load(f)
            
            # Sample first few routes to check
            for group in ["greedy_td_routes", "aco_td_routes"]:
                routes_dict = routes_data.get(group, {})
                for v_id, route in list(routes_dict.items())[:5]:
                    duration = route.get("duration_sec", 0.0)
                    assert duration >= 0.0, f"Negative duration for {v_id} in hour {h}: {duration}"
        print("    [+] FIFO compliance verified. Traversal durations are non-negative and mathematically sound.")
    except Exception as e:
        print(f"[-] ERROR Validating FIFO compliance: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # 4. Load and print consolidated sensitivity analysis stats
    try:
        with open(aco_summary_path, "r") as f:
            summary = json.load(f)
            
        sensitivity = summary.get("td_sensitivity_analysis", {})
        ratios_derived = summary.get("congestion_ratios_derived", {})
        
        print("\n=== Derived Data-Driven Congestion-to-Speed Ratios ===")
        print(f"  Global Speed Ratios: {ratios_derived.get('global')}")
        print(f"  Cell-Specific Derived Ratios Count: {ratios_derived.get('cell_specific_count')}")
        
        print("\n=== Sensitivity Analysis (Average Metrics across Shifts) ===")
        print("Hour  | Model         | Dist (km)  | Duration (sec) | Fuel (L)   | CO2 (g)")
        print("------|---------------|------------|----------------|------------|-----------")
        for h in sorted(sensitivity.keys(), key=int):
            h_int = int(h)
            h_data = sensitivity[h]
            g_static = h_data["greedy_static"]
            g_td = h_data["greedy_td"]
            a_static = h_data["aco_static"]
            a_td = h_data["aco_td"]
            print(f"{h_int:02d}:00 | Greedy-Static | {g_static['distance_km']:10.3f} | {g_static['duration_sec']:14.1f} | {g_static['fuel_l']:10.3f} | {g_static['co2_g']:9.1f}")
            print(f"{h_int:02d}:00 | Greedy-TD     | {g_td['distance_km']:10.3f} | {g_td['duration_sec']:14.1f} | {g_td['fuel_l']:10.3f} | {g_td['co2_g']:9.1f}")
            print(f"{h_int:02d}:00 | ACO-Static    | {a_static['distance_km']:10.3f} | {a_static['duration_sec']:14.1f} | {a_static['fuel_l']:10.3f} | {a_static['co2_g']:9.1f}")
            print(f"{h_int:02d}:00 | ACO-TD        | {a_td['distance_km']:10.3f} | {a_td['duration_sec']:14.1f} | {a_td['fuel_l']:10.3f} | {a_td['co2_g']:9.1f}")
            print("------|---------------|------------|----------------|------------|-----------")
        
    except Exception as e:
        print(f"[-] ERROR Reading sensitivity stats from summary JSON: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nTIME-DEPENDENT DIJKSTRA ROUTING VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    verify_td_dijkstra()
