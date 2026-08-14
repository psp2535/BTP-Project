import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    os.makedirs("research/figures", exist_ok=True)
    
    # 1. Scalability Plot
    if os.path.exists("research/tables/scalability_results.csv"):
        df_scale = pd.read_csv("research/tables/scalability_results.csv")
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_scale, x="Fleet Size", y="Total Inference Time (s)", marker='o', linewidth=2.5, color='#2ecc71')
        plt.title('DeepACO Computational Scalability', fontsize=14, fontweight='bold')
        plt.xlabel('Fleet Size (Vehicles)', fontsize=12)
        plt.ylabel('Total Inference Latency (s)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("research/figures/scalability_plot.png", dpi=300)
        print("Saved scalability_plot.png")
        plt.close()

    # 2. Sensitivity Plot (Heatmap for Beta and q0 vs CO2)
    if os.path.exists("research/tables/sensitivity_results.csv"):
        df_sens = pd.read_csv("research/tables/sensitivity_results.csv")
        heatmap_data = df_sens.pivot(index="Beta (Heuristic Weight)", columns="q0 (Exploitation)", values="Avg CO2 Emissions (g)")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'CO2 Emissions (g)'})
        plt.title('Hyperparameter Sensitivity: $\\beta$ and $q_0$ vs $CO_2$', fontsize=14, fontweight='bold')
        plt.xlabel('$q_0$ (Exploitation Probability)', fontsize=12)
        plt.ylabel('$\\beta$ (Heuristic Weight)', fontsize=12)
        plt.tight_layout()
        plt.savefig("research/figures/sensitivity_heatmap.png", dpi=300)
        print("Saved sensitivity_heatmap.png")
        plt.close()

if __name__ == "__main__":
    generate_plots()
