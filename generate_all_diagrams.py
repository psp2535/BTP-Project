import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx

def save_fig(name):
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {name}")

def draw_workflow():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    steps = ["Raw T-Drive Data", "Data Cleaning &\nMap Matching", "Graph Construction\n& Feature Extraction", "GNN & TCFMu\nPredictions", "DeepACO\nOptimization", "JSON Route Output"]
    for i, step in enumerate(steps):
        box = patches.FancyBboxPatch((i*2, 1), 1.5, 1, boxstyle="round,pad=0.1", facecolor='#34495E', edgecolor='#2C3E50')
        ax.add_patch(box)
        plt.text(i*2+0.75, 1.5, step, color='white', ha='center', va='center', fontweight='bold', fontsize=10)
        if i < len(steps)-1:
            plt.annotate("", xy=(i*2+1.8, 1.5), xytext=(i*2+1.5, 1.5), arrowprops=dict(facecolor='black', width=2, headwidth=8))
    ax.set_xlim(0, len(steps)*2)
    ax.set_ylim(0, 3)
    save_fig('placeholder_workflow.png')

def draw_road_graph():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    G = nx.watts_strogatz_graph(30, 4, 0.1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, node_color='#3498DB', node_size=100, edge_color='#BDC3C7')
    # Highlight depots
    depots = [0, 5, 10]
    nx.draw_networkx_nodes(G, pos, nodelist=depots, node_color='#E74C3C', node_size=300, ax=ax, label="Depots")
    plt.title("Spatial Grid Graph (Mock Representation)", fontsize=14, fontweight='bold')
    save_fig('placeholder_road_graph.png')

def draw_traffic_prediction():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    box = patches.Rectangle((1, 2), 2, 2, facecolor='#ECF0F1', edgecolor='black')
    ax.add_patch(box)
    plt.text(2, 3, "Input Features:\n- Coordinates\n- Density\n- Historical Speed\n- Lag-1 State", ha='center', va='center')
    plt.annotate("", xy=(4, 3), xytext=(3, 3), arrowprops=dict(facecolor='black', width=2))
    box2 = patches.FancyBboxPatch((4, 2), 2, 2, boxstyle="round,pad=0.1", facecolor='#27AE60', edgecolor='black')
    ax.add_patch(box2)
    plt.text(5, 3, "TCFMu\n(XGBoost)", color='white', ha='center', va='center', fontweight='bold')
    plt.annotate("", xy=(7, 3), xytext=(6, 3), arrowprops=dict(facecolor='black', width=2))
    box3 = patches.Rectangle((7, 1.5), 2, 3, facecolor='#ECF0F1', edgecolor='black')
    ax.add_patch(box3)
    plt.text(8, 3, "Congestion State:\n1. Free-Flow (>0.7)\n2. Moderate (0.4-0.7)\n3. Congested (<0.4)", ha='center', va='center')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    save_fig('placeholder_traffic_prediction.png')

def draw_gnn_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    plt.text(5, 5, "Graph Convolutional Network (GCN)", ha='center', va='center', fontsize=14, fontweight='bold')
    plt.annotate("Input Node Features (4-dim)", xy=(3, 3), xytext=(1, 3), arrowprops=dict(facecolor='black', width=1))
    box1 = patches.FancyBboxPatch((3, 2), 2, 2, facecolor='#2980B9', edgecolor='black', boxstyle="round")
    ax.add_patch(box1)
    plt.text(4, 3, "GCN Layer 1\n(Agg + Combine)", color='white', ha='center', va='center')
    plt.annotate("", xy=(6, 3), xytext=(5, 3), arrowprops=dict(facecolor='black', width=1))
    box2 = patches.FancyBboxPatch((6, 2), 2, 2, facecolor='#2980B9', edgecolor='black', boxstyle="round")
    ax.add_patch(box2)
    plt.text(7, 3, "GCN Layer 2\n(Agg + Combine)", color='white', ha='center', va='center')
    plt.annotate("64-dim Latent Embedding", xy=(9.5, 3), xytext=(8, 3), arrowprops=dict(facecolor='black', width=1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    save_fig('placeholder_gnn_architecture.png')

def draw_deepaco_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    box1 = patches.Rectangle((1, 3), 3, 2, facecolor='#ECF0F1')
    ax.add_patch(box1)
    plt.text(2.5, 4, "State Input (137-dim):\n- GNN Embeddings (128)\n- Environment (5)\n- Vehicle Meta (4)", ha='center', va='center')
    
    plt.annotate("", xy=(5, 4), xytext=(4, 4), arrowprops=dict(facecolor='black', width=2))
    
    box2 = patches.FancyBboxPatch((5, 3), 2, 2, boxstyle="round", facecolor='#8E44AD')
    ax.add_patch(box2)
    plt.text(6, 4, "DeepACOHeuristicNet\n(MLP)", color='white', ha='center', va='center', fontweight='bold')
    
    plt.annotate("", xy=(8, 4), xytext=(7, 4), arrowprops=dict(facecolor='black', width=2))
    
    box3 = patches.Rectangle((8, 3.5), 2.5, 1, facecolor='#F39C12')
    ax.add_patch(box3)
    plt.text(9.25, 4, "Neural Heuristic Score ($H_{ij}$)", color='white', ha='center', va='center')
    
    plt.annotate("", xy=(9.25, 3.5), xytext=(9.25, 3), arrowprops=dict(facecolor='black', width=1))
    box4 = patches.Rectangle((7.5, 1), 3.5, 2, facecolor='#E74C3C')
    ax.add_patch(box4)
    plt.text(9.25, 2, "Transition Probability Matrix\n(Combined w/ Pheromones)", color='white', ha='center', va='center', fontweight='bold')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    save_fig('placeholder_deepaco_pipeline.png')

def draw_experimental_workflow():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    steps = ["T-Drive Dataset Split\n(Train/Val/Test)", "Unsupervised\nGNN Pretraining", "XGBoost Supervised\nTraining", "DeepACO REINFORCE\nPolicy Training", "Evaluation against\nClassical Baselines"]
    for i, step in enumerate(steps):
        box = patches.Rectangle((i*2.2, 1), 1.8, 1, facecolor='#7F8C8D')
        ax.add_patch(box)
        plt.text(i*2.2+0.9, 1.5, step, color='white', ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(steps)-1:
            plt.annotate("", xy=(i*2.2+2.1, 1.5), xytext=(i*2.2+1.8, 1.5), arrowprops=dict(facecolor='black', width=1))
    ax.set_xlim(0, len(steps)*2.2)
    ax.set_ylim(0, 3)
    save_fig('placeholder_experimental_workflow.png')

def draw_execution_time():
    fig, ax = plt.subplots(figsize=(8, 6))
    algos = ['Greedy', 'Classical ACO', 'DeepACO (Ours)']
    times = [0.018, 0.653, 0.606]
    bars = ax.bar(algos, times, color=['#E74C3C', '#F1C40F', '#2ECC71'])
    ax.set_ylabel('Execution Time per Vehicle (s)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold')
    save_fig('placeholder_execution_time.png')

if __name__ == '__main__':
    draw_workflow()
    draw_road_graph()
    draw_traffic_prediction()
    draw_gnn_architecture()
    draw_deepaco_pipeline()
    draw_experimental_workflow()
    draw_execution_time()
