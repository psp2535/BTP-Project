import os
import subprocess
import json
import pandas as pd
import sys

def verify_tcfmu():
    print("====================================================")
    print("Starting XGBoost TCFMu Verification for Phase 3A")
    print("====================================================")
    
    # 1. Run main.py using python subprocess with 50 taxis (to get a good sized dataset)
    cmd = [sys.executable, "main.py", "--num-taxis", "50", "--step", "all"]
    print(f"Running pipeline command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline execution stdout output:")
        # Show key logs about TCFMu step
        lines = result.stdout.splitlines()
        tcfmu_lines = [l for l in lines if "TCFMu" in l or "XGBoost" in l or "Class" in l]
        for l in tcfmu_lines:
            print(f"  {l}")
    except subprocess.CalledProcessError as e:
        print("Pipeline execution FAILED with return code", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        sys.exit(1)
        
    # 2. Check generated TCFMu files
    model_path = "outputs/xgboost_model.json"
    predictions_path = "data/processed/congestion_predictions.csv"
    metrics_path = "outputs/xgboost_metrics.json"
    
    cm_path = "outputs/confusion_matrix.png"
    feat_path = "outputs/feature_importance.png"
    dist_path = "outputs/congestion_class_distribution.png"
    
    assert os.path.exists(model_path), f"XGBoost model file missing: {model_path}"
    assert os.path.exists(predictions_path), f"Test predictions CSV missing: {predictions_path}"
    assert os.path.exists(metrics_path), f"Metrics JSON missing: {metrics_path}"
    
    assert os.path.exists(cm_path), f"Confusion matrix plot missing: {cm_path}"
    assert os.path.exists(feat_path), f"Feature importance plot missing: {feat_path}"
    assert os.path.exists(dist_path), f"Class distribution plot missing: {dist_path}"
    
    print("\n[+] Verification target files and plots found.")
    
    # 3. Load and validate predictions CSV
    try:
        df = pd.read_csv(predictions_path)
        print(f"[+] Loaded test predictions successfully ({len(df)} rows).")
        required_cols = ["grid_row", "grid_col", "hour", "true_congestion_level", "predicted_congestion_level"]
        for c in required_cols:
            assert c in df.columns, f"Missing column in predictions CSV: {c}"
            
        print("    [+] Predictions CSV columns validated.")
    except Exception as e:
        print(f"[-] ERROR Validating predictions CSV: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Load stats report and print
    try:
        with open(metrics_path, "r") as f:
            stats = json.load(f)
            
        print("\n=== XGBoost TCFMu Performance Metrics ===")
        print(f"  Accuracy Score:                  {stats['accuracy']:.4f}")
        print(f"  Macro Precision:                 {stats['precision_macro']:.4f}")
        print(f"  Macro Recall:                    {stats['recall_macro']:.4f}")
        print(f"  Macro F1-Score:                  {stats['f1_macro']:.4f}")
        print("\n  Confusion Matrix (Numerical):")
        labels = ["FreeFlow", "Moderate", "Congested"]
        for i, row in enumerate(stats['confusion_matrix']):
            print(f"    True {labels[i]:<10}: {row}")
            
        print("\n  Feature Importances:")
        for feat, val in sorted(stats['feature_importances'].items(), key=lambda item: item[1], reverse=True):
            print(f"    - {feat:<20}: {val:.4f}")
        print("=========================================")
        
    except Exception as e:
        print(f"[-] ERROR Reading metrics stats report: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\nXGBOOST TCFMu CONGESTION PREDICTION VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    verify_tcfmu()
