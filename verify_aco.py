import os
import subprocess
import json
import pandas as pd
import sys

def verify_aco():
    print("====================================================")
    print("Starting ACO Route Selection Verification for Phase 4A (300 Taxis)")
    print("====================================================")
    
    # 1. Run main.py on 300 taxis with step aco_route using python subprocess
    cmd = [sys.executable, "main.py", "--num-taxis", "300", "--step", "aco_route"]
    print(f"Running pipeline command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline execution completed successfully.")
        # Print key lines from stdout logs
        lines = result.stdout.splitlines()
        for l in lines:
            if "STEP 10" in l or "Ant Colony" in l or "Comparative" in l or "Distance" in l or "Duration" in l or "Congestion" in l or "Fuel" in l or "CO2" in l or "Saved ACO" in l:
                print(f"  {l}")
    except subprocess.CalledProcessError as e:
        print("Pipeline execution FAILED with return code", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        sys.exit(1)
        
    # 2. Check generated ACO outputs
    aco_routes_path = "data/processed/aco_routes.json"
    aco_opt_routes_path = "data/processed/aco_optimized_routes.json"
    aco_metrics_path = "outputs/aco_metrics.csv"
    aco_traces_path = "outputs/aco_decision_traces.json"
    aco_summary_path = "outputs/aco_summary.json"
    pheromone_history_path = "outputs/pheromone_history.json"
    
    heatmap_plot_path = "outputs/pheromone_heatmap_demo.png"
    convergence_plot_path = "outputs/aco_convergence_demo.png"
    comparison_plot_path = "outputs/aco_route_comparison_demo.png"
    
    assert os.path.exists(aco_routes_path), f"ACO routes file missing: {aco_routes_path}"
    assert os.path.exists(aco_opt_routes_path), f"ACO optimized routes missing: {aco_opt_routes_path}"
    assert os.path.exists(aco_metrics_path), f"ACO metrics CSV missing: {aco_metrics_path}"
    assert os.path.exists(aco_traces_path), f"ACO convergence traces missing: {aco_traces_path}"
    assert os.path.exists(aco_summary_path), f"ACO summary JSON missing: {aco_summary_path}"
    assert os.path.exists(pheromone_history_path), f"Pheromone history JSON missing: {pheromone_history_path}"
    assert os.path.exists(heatmap_plot_path), f"Pheromone heatmap plot missing: {heatmap_plot_path}"
    assert os.path.exists(convergence_plot_path), f"ACO convergence plot missing: {convergence_plot_path}"
    assert os.path.exists(comparison_plot_path), f"ACO comparison plot missing: {comparison_plot_path}"
    
    print("\n[+] All ACO target files, metrics, and visualization plots found.")
    
    # 3. Load and validate comparative metrics CSV
    try:
        df_aco = pd.read_csv(aco_metrics_path)
        print(f"[+] Loaded ACO comparative metrics ({len(df_aco)} vehicles).")
        required_cols = [
            "vehicle_id", "vehicle_type", "depot_id",
            "aco_distance_km", "aco_duration_sec", "aco_avg_congestion", "aco_fuel_l", "aco_co2_g",
            "dijkstra_distance_km", "dijkstra_duration_sec", "dijkstra_avg_congestion", "dijkstra_fuel_l", "dijkstra_co2_g",
            "improvement_distance_km", "improvement_duration_sec", "improvement_congestion", "improvement_fuel_l", "improvement_co2_g",
            "reduction_distance_pct", "reduction_duration_pct", "reduction_congestion_pct", "reduction_fuel_pct", "reduction_co2_pct"
        ]
        for c in required_cols:
            assert c in df_aco.columns, f"Missing column in aco_metrics CSV: {c}"
            
        # Ensure values are numerically sound
        assert (df_aco["aco_distance_km"] >= 0).all(), "Negative distance values in ACO routes"
        assert (df_aco["dijkstra_fuel_l"] >= 0).all(), "Negative fuel values in Dijkstra-ACO routes"
        print("    [+] ACO metrics CSV columns and positive limits validated.")
    except Exception as e:
        print(f"[-] ERROR Validating ACO metrics CSV: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Load summary JSON and print comparisons
    try:
        with open(aco_summary_path, "r") as f:
            summary = json.load(f)
            
        print("\n=== ACO vs GREEDY Comparative Route Optimization Performance (300 Taxis) ===")
        avg = summary["averages"]
        greedy_d = avg["greedy_dijkstra_optimized"]
        aco_d = avg["aco_dijkstra_optimized"]
        red = avg["optimized_improvements_pct"]
        
        print(f"  Average Distance:   Greedy-Opt={greedy_d['distance_km']:.3f} km vs ACO-Opt={aco_d['distance_km']:.3f} km ({red['distance_reduction_pct']:.2f}% reduction)")
        print(f"  Average Duration:   Greedy-Opt={greedy_d['duration_sec']:.1f} s vs ACO-Opt={aco_d['duration_sec']:.1f} s ({red['duration_reduction_pct']:.2f}% reduction)")
        print(f"  Average Congestion: Greedy-Opt={greedy_d['avg_congestion']:.3f} vs ACO-Opt={aco_d['avg_congestion']:.3f} ({red['congestion_reduction_pct']:.2f}% reduction)")
        print(f"  Average Fuel:       Greedy-Opt={greedy_d['fuel_l']:.3f} L vs ACO-Opt={aco_d['fuel_l']:.3f} L ({red['fuel_reduction_pct']:.2f}% reduction)")
        print(f"  Average CO2:        Greedy-Opt={greedy_d['co2_g']:.1f} g vs ACO-Opt={aco_d['co2_g']:.1f} g ({red['co2_reduction_pct']:.2f}% reduction)")
        print("=========================================================================")
        
    except Exception as e:
        print(f"[-] ERROR Reading ACO summary JSON: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nANT COLONY OPTIMIZATION (ACO) ROUTE SELECTION VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    verify_aco()
