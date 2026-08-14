import os
import subprocess
import torch
import json
import sys

def verify_embeddings():
    print("====================================================")
    print("Starting GNN Embedding Verification for Phase 2B")
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
        
    # 2. Check generated GNN embedding files
    embed_path = "data/processed/node_embeddings.pt"
    json_path = "outputs/embedding_statistics.json"
    plot_path = "outputs/embeddings_visualization.png"
    
    if not os.path.exists(embed_path):
        print(f"[-] ERROR: GNN embeddings file missing: {embed_path}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(json_path):
        print(f"[-] ERROR: GNN embedding stats report file missing: {json_path}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(plot_path):
        print(f"[-] ERROR: Embeddings visualization plot missing: {plot_path}", file=sys.stderr)
        sys.exit(1)
        
    print("\n[+] Verification target files found.")
    
    # 3. Load GNN Embeddings and Validate Structure
    try:
        embeddings = torch.load(embed_path)
        print("[+] Loaded PyTorch embeddings successfully.")
        
        N = embeddings.shape[0]
        D = embeddings.shape[1]
        
        print(f"    - Node Embedding Matrix shape: {list(embeddings.shape)} (Expected: [N, 64])")
        
        # Shape validations
        assert D == 64, f"Embeddings dimension mismatch: {D} (expected 64)"
        assert embeddings.dtype == torch.float, "embeddings must be a float tensor"
        assert not torch.isnan(embeddings).any(), "embeddings contain NaN values"
        
        print("[+] PyTorch embeddings shapes, dtypes, and value sanity verified.")
        
    except Exception as e:
        print(f"[-] ERROR Validating PyTorch embeddings: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Load Stats report and Print
    try:
        with open(json_path, "r") as f:
            stats = json.load(f)
            
        print("\n=== Node Embedding Diagnostics ===")
        print(f"  Embedding Shape:                 {stats['embedding_shape']}")
        print(f"  Contains NaN:                    {stats['has_nans']}")
        print(f"  L2 Norm mean:                    {stats['l2_norm_stats']['mean']:.4f}")
        print(f"  L2 Norm std:                     {stats['l2_norm_stats']['std']:.4f}")
        print(f"  L2 Norm min/max:                 [{stats['l2_norm_stats']['min']:.4f}, {stats['l2_norm_stats']['max']:.4f}]")
        print(f"  PCA Explained Variance Ratio:    {stats['pca_explained_variance_ratio']}")
        print("==================================")
        
    except Exception as e:
        print(f"[-] ERROR Reading embedding stats report: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nGNN EMBEDDING EXTRACTION VERIFICATION SUCCESSFUL!")
    
if __name__ == "__main__":
    verify_embeddings()
