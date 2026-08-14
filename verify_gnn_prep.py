import os
import subprocess
import torch
import json
import sys

def verify_gnn():
    print("====================================================")
    print("Starting GNN Graph Feature Verification for Phase 2A")
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
        
    # 2. Check generated GNN files
    pt_path = "data/processed/gnn_graph_data.pt"
    json_path = "data/processed/gnn_graph_stats.json"
    
    if not os.path.exists(pt_path):
        print(f"[-] ERROR: GNN graph tensors file missing: {pt_path}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(json_path):
        print(f"[-] ERROR: GNN graph stats report file missing: {json_path}", file=sys.stderr)
        sys.exit(1)
        
    print("\n[+] Verification target files found.")
    
    # 3. Load GNN Tensors and Validate Structure
    try:
        graph_data = torch.load(pt_path)
        print("[+] Loaded PyTorch graph data successfully.")
        
        # Verify keys
        required_keys = ["x", "edge_index", "edge_attr", "adj_matrix"]
        missing_keys = [k for k in required_keys if k not in graph_data]
        if missing_keys:
            print(f"[-] ERROR: Missing keys in graph data dictionary: {missing_keys}", file=sys.stderr)
            sys.exit(1)
            
        x = graph_data["x"]
        edge_index = graph_data["edge_index"]
        edge_attr = graph_data["edge_attr"]
        adj_matrix = graph_data["adj_matrix"]
        
        N = x.shape[0]
        E = edge_index.shape[1]
        
        print(f"    - Node Feature Matrix X shape: {list(x.shape)} (Expected: [{N}, 4])")
        print(f"    - Edge Index list shape: {list(edge_index.shape)} (Expected: [2, {E}])")
        print(f"    - Edge Feature Matrix E_attr shape: {list(edge_attr.shape)} (Expected: [{E}, 4])")
        print(f"    - Dense Adjacency Matrix shape: {list(adj_matrix.shape)} (Expected: [{N}, {N}])")
        
        # Shape validations
        assert x.shape == (N, 4), f"Node features shape mismatch: {x.shape}"
        assert edge_index.shape == (2, E), f"Edge index shape mismatch: {edge_index.shape}"
        assert edge_attr.shape == (E, 4), f"Edge features shape mismatch: {edge_attr.shape}"
        assert adj_matrix.shape == (N, N), f"Adjacency matrix shape mismatch: {adj_matrix.shape}"
        
        # Data types validations
        assert x.dtype == torch.float, "x must be a float tensor"
        assert edge_index.dtype == torch.long, "edge_index must be a long tensor"
        assert edge_attr.dtype == torch.float, "edge_attr must be a float tensor"
        assert adj_matrix.dtype == torch.float, "adj_matrix must be a float tensor"
        
        # Value bounds validations (sanity check)
        assert not torch.isnan(x).any(), "x contains NaN values"
        assert not torch.isnan(edge_attr).any(), "edge_attr contains NaN values"
        
        print("[+] PyTorch tensor shapes, dtypes, and value sanity verified.")
        
    except Exception as e:
        print(f"[-] ERROR Validating PyTorch tensors: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Load Stats report and Print
    try:
        with open(json_path, "r") as f:
            stats = json.load(f)
            
        print("\n=== Graph Connectivity & Feature Statistics ===")
        print(f"  Total Nodes (Grid Cells):       {stats['num_nodes']}")
        print(f"  Total Edges (Transitions):       {stats['num_edges']}")
        print(f"  Average Node Out-degree:       {stats['average_degree']:.3f}")
        print(f"  Weakly Connected Components:    {stats['weakly_connected_components']}")
        print(f"  Largest Component Node Count:   {stats['largest_component_size']}")
        print(f"  Isolated Node Count:            {stats['isolated_nodes']}")
        print("===============================================")
        
    except Exception as e:
        print(f"[-] ERROR Reading graph stats report: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nGNN GRAPH PREPARATION VERIFICATION SUCCESSFUL!")
    
if __name__ == "__main__":
    verify_gnn()
