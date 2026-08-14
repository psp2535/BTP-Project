import os
import subprocess
import json
import pandas as pd
import torch
import sys

def verify_deep_aco():
    print("====================================================")
    print("Starting DeepACO Route Optimization Verification")
    print("====================================================")
    
    # 1. Run main.py step all using python subprocess with a small taxi count
    # Running all steps ensures that intermediate inputs (gnn embeddings, graph) are updated
    cmd = [sys.executable, "main.py", "--num-taxis", "1000", "--step", "all"]
    print(f"Running pipeline command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline execution stdout output:")
        lines = result.stdout.splitlines()
        # Find key logs
        for l in lines:
            if "STEP 11" in l or "DeepACO" in l or "Evaluation" in l or "Metric" in l or "Avg" in l:
                print(f"  {l}")
    except subprocess.CalledProcessError as e:
        print("Pipeline execution FAILED with return code", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        sys.exit(1)
        
    # 2. Check generated files
    model_path = "data/processed/deep_aco_model.pt"
    routes_path = "data/processed/deep_aco_routes.json"
    comp_csv_path = "outputs/deep_aco_comparison.csv"
    
    assert os.path.exists(model_path), f"[-] ERROR: Model weights file missing: {model_path}"
    assert os.path.exists(routes_path), f"[-] ERROR: DeepACO routes JSON missing: {routes_path}"
    assert os.path.exists(comp_csv_path), f"[-] ERROR: Comparison CSV file missing: {comp_csv_path}"
    
    print("\n[+] Verification target files and output results found.")
    
    # 3. Load and validate weights
    try:
        weights = torch.load(model_path)
        print("[+] Loaded trained DeepACO model weights successfully.")
        # Check some layer parameters
        assert "net.0.weight" in weights, "Missing net layer weights"
        print("    [+] Model weight structure validated.")
    except Exception as e:
        print(f"[-] ERROR Validating PyTorch weights: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Load routes and validate structure
    try:
        with open(routes_path, "r") as f:
            routes = json.load(f)
        assert len(routes) > 0, "Routes output JSON is empty"
        print(f"[+] Successfully loaded DeepACO routes for {len(routes)} vehicles.")
        demo_veh = list(routes.keys())[0]
        demo_route = routes[demo_veh]
        assert "route_nodes" in demo_route, "Missing route nodes details"
        assert "total_distance_km" in demo_route, "Missing total distance metric"
        assert "fuel_l" in demo_route, "Missing fuel consumption metric"
        print("    [+] Route coordinates and target sequences validated.")
    except Exception as e:
        print(f"[-] ERROR Reading DeepACO routes JSON: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # 5. Load and validate comparison CSV
    try:
        df_comp = pd.read_csv(comp_csv_path)
        print(f"[+] Loaded route comparisons successfully ({len(df_comp)} rows).")
        required_cols = [
            "vehicle_id", "greedy_dist", "aco_dist", "deep_aco_dist",
            "greedy_time", "aco_time", "deep_aco_time",
            "greedy_fuel", "aco_fuel", "deep_aco_fuel",
            "greedy_co2", "aco_co2", "deep_aco_co2",
            "greedy_cong", "aco_cong", "deep_aco_cong",
            "greedy_cost", "aco_cost", "deep_aco_cost"
        ]
        for c in required_cols:
            assert c in df_comp.columns, f"Missing column in comparison CSV: {c}"
        
        # Verify values are positive / valid
        assert (df_comp["deep_aco_dist"] >= 0).all(), "Negative distance values in DeepACO routes"
        assert (df_comp["deep_aco_fuel"] >= 0).all(), "Negative fuel values in DeepACO routes"
        assert not df_comp.isna().any().any(), "Comparison table contains NaN values"
        
        print("    [+] Comparison table values validated.")
    except Exception as e:
        print(f"[-] ERROR Validating comparison CSV: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nDEEPACO ROUTE SELECTION VERIFICATION SUCCESSFUL!")
    print("====================================================")

if __name__ == "__main__":
    verify_deep_aco()
