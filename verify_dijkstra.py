import os
import subprocess
import json
import pandas as pd
import sys

def verify_dijkstra():
    print("====================================================")
    print("Starting Dijkstra Route Optimization Verification for Phase 3B")
    print("====================================================")
    
    # 1. Run main.py route_opt step using python subprocess
    cmd = [sys.executable, "main.py", "--num-taxis", "50", "--step", "route_opt"]
    print(f"Running pipeline command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline execution stdout output:")
        lines = result.stdout.splitlines()
        # Find key logs
        for l in lines:
            if "STEP 9" in l or "Dijkstra" in l or "Summary" in l or "Avg" in l or "Bounds" in l:
                print(f"  {l}")
    except subprocess.CalledProcessError as e:
        print("Pipeline execution FAILED with return code", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        sys.exit(1)
        
    # 2. Check generated files
    opt_routes_path = "data/processed/optimized_routes.json"
    metrics_path = "outputs/dijkstra_metrics.csv"
    comparison_path = "outputs/route_comparison.csv"
    summary_path = "outputs/optimization_summary.json"
    
    demo_plot_path = "outputs/route_comparison_demo.png"
    heatmap_plot_path = "outputs/edge_cost_heatmap.png"
    
    assert os.path.exists(opt_routes_path), f"Optimized routes file missing: {opt_routes_path}"
    assert os.path.exists(metrics_path), f"Dijkstra metrics file missing: {metrics_path}"
    assert os.path.exists(comparison_path), f"Route comparison CSV missing: {comparison_path}"
    assert os.path.exists(summary_path), f"Optimization summary JSON missing: {summary_path}"
    assert os.path.exists(demo_plot_path), f"Route comparison plot missing: {demo_plot_path}"
    assert os.path.exists(heatmap_plot_path), f"Edge cost heatmap plot missing: {heatmap_plot_path}"
    
    print("\n[+] Verification target files and plots found.")
    
    # 3. Load and validate comparison CSV (per-route improvements)
    try:
        df_comp = pd.read_csv(comparison_path)
        print(f"[+] Loaded route comparisons successfully ({len(df_comp)} routes).")
        required_cols = [
            "vehicle_id", "vehicle_type", "depot_id",
            "greedy_distance_km", "dijkstra_distance_km", "improvement_distance_km", "reduction_distance_pct",
            "greedy_duration_sec", "dijkstra_duration_sec", "improvement_duration_sec", "reduction_duration_pct",
            "greedy_avg_congestion", "dijkstra_avg_congestion", "improvement_congestion", "reduction_congestion_pct",
            "greedy_fuel_l", "dijkstra_fuel_l", "improvement_fuel_l", "reduction_fuel_pct",
            "greedy_co2_g", "dijkstra_co2_g", "improvement_co2_g", "reduction_co2_pct"
        ]
        for c in required_cols:
            assert c in df_comp.columns, f"Missing column in comparison CSV: {c}"
            
        print("    [+] Route comparison CSV columns validated.")
        
        # Verify values are positive / valid
        assert (df_comp["greedy_distance_km"] >= 0).all(), "Negative distance values in Greedy baseline"
        assert (df_comp["dijkstra_distance_km"] >= 0).all(), "Negative distance values in Dijkstra routes"
        assert (df_comp["dijkstra_fuel_l"] >= 0).all(), "Negative fuel values in Dijkstra routes"
        
        print("    [+] Fuel consumption and travel distance values are physically consistent.")
    except Exception as e:
        print(f"[-] ERROR Validating route comparison: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Load summary report and print
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        print("\n=== IM-VRM Comparative Performance (Dijkstra vs Greedy) ===")
        avg_red = summary["averages"]["average_reductions"]
        print(f"  Average Travel Distance Reduction:  {avg_red['distance_reduction_pct']:.2f}%")
        print(f"  Average Travel Duration Reduction:  {avg_red['duration_reduction_pct']:.2f}%")
        print(f"  Average Route Congestion Reduction: {avg_red['congestion_reduction_pct']:.2f}%")
        print(f"  Average Fuel Consumption Reduction: {avg_red['fuel_reduction_pct']:.2f}%")
        print(f"  Average Carbon CO2 Emission Reduction:{avg_red['co2_reduction_pct']:.2f}%")
        
        print("\n  Normalization Bounds Used:")
        bounds = summary["graph_stats"]["normalization_bounds"]
        for k, v in bounds.items():
            print(f"    - {k:<12}: min={v['min']:.2f}, max={v['max']:.2f}")
            
        print("\n  Weighted Edge Cost Distribution:")
        dist = summary["graph_stats"]["cost_distribution"]
        for k, v in dist.items():
            print(f"    - {k:<8}: {v:.4f}")
        print("=========================================================")
        
    except Exception as e:
        print(f"[-] ERROR Reading optimization summary: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nDIJKSTRA ROUTE OPTIMIZATION VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    verify_dijkstra()
