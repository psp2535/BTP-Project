import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Ensure high quality rendering
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def create_master_architecture_diagram(output_path='im_vrm_master_system_architecture.png'):
    """
    Generates a publication-grade, ultra-rich master system architecture diagram
    customized for the IM-VRM (DeepACO + GNN + TCFMu + T-Drive) project, mirroring
    the comprehensive structure of Fig. 1 in the base reference paper.
    """
    fig = plt.figure(figsize=(24, 16), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette - Professional IEEE Scientific Theme
    bg_color = "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Styling constants
    c_primary = "#1E3A8A"      # Deep Navy
    c_sec_blue = "#3B82F6"     # Bright Blue
    c_gnn = "#0284C7"          # Sky Blue
    c_tcfmu = "#059669"        # Emerald Green
    c_deepaco = "#7C3AED"      # Purple / Violet
    c_amber = "#D97706"        # Amber / Orange
    c_slate = "#475569"        # Slate Gray
    c_arrow = "#334155"        # Dark slate arrow

    # Helper function for drawing rounded panels/boxes
    def draw_card(x, y, w, h, title="", subtitle="", bg="#F8FAFC", border="#CBD5E1", title_color="#1E293B", rad=1.5, title_fs=12, sub_fs=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.5,rounding_size={rad}",
                             facecolor=bg, edgecolor=border, linewidth=1.5, zorder=1)
        ax.add_patch(box)
        if title:
            ax.text(x + 1.5, y + h - 2.2, title, fontsize=title_fs, fontweight='bold', color=title_color, va='top', ha='left', zorder=3)
        if subtitle:
            ax.text(x + 1.5, y + h - 4.2, subtitle, fontsize=sub_fs, fontstyle='italic', color='#64748B', va='top', ha='left', zorder=3)
        return box

    def draw_node(x, y, w, h, text, bg="#FFFFFF", border="#94A3B8", text_color="#0F172A", fs=9.5, bold=True, rad=1.0, align='center'):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rad}",
                             facecolor=bg, edgecolor=border, linewidth=1.2, zorder=3)
        ax.add_patch(box)
        fw = 'bold' if bold else 'normal'
        if align == 'center':
            ax.text(x + w/2, y + h/2, text, fontsize=fs, fontweight=fw, color=text_color, va='center', ha='center', zorder=4)
        else:
            ax.text(x + 1.0, y + h/2, text, fontsize=fs, fontweight=fw, color=text_color, va='center', ha='left', zorder=4)
        return box

    def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=1.8, rad=0.0, label=""):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="-|>,head_length=5,head_width=3",
                                edgecolor=color, facecolor=color,
                                linewidth=lw, zorder=5)
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + (1.2 if rad >= 0 else -1.2)
            ax.text(mx, my, label, fontsize=8, fontweight='bold', color=color, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='none', alpha=0.9), zorder=6)

    # -------------------------------------------------------------
    # TOP HEADER / BANNER
    # -------------------------------------------------------------
    header_box = FancyBboxPatch((2, 94.5), 96, 4.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                                facecolor="#1E293B", edgecolor="#0F172A", linewidth=1.5, zorder=2)
    ax.add_patch(header_box)
    ax.text(50, 96.8, "IM-VRM: Intelligent Multi-Depot Vehicle Routing & Management Framework",
            fontsize=17, fontweight='heavy', color='#FFFFFF', ha='center', va='center', zorder=3)
    ax.text(50, 95.3, "End-to-End System Architecture: Trajectory Mining • GNN Embeddings • TCFMu XGBoost • DeepACO REINFORCE",
            fontsize=10.5, fontstyle='italic', color='#93C5FD', ha='center', va='center', zorder=3)

    # =============================================================
    # SECTION 1: TOP-LEFT - SMART CITY SENSING & DATA INGESTION
    # =============================================================
    draw_card(2, 65, 30, 28, "1. Smart City Traffic Sensing & Trajectory Ingestion",
              "Real-time GPS Streaming & Multi-Depot Crowd Logistics", bg="#F0FDF4", border="#86EFAC", title_color="#14532D")

    # City Grid & Sensing Box
    draw_node(4, 80, 26, 8.5,
              "Microsoft T-Drive GPS Repository\n• 10,357 Taxis in Beijing Metropolitan Area\n• 15M+ Spatial-Temporal Trajectory Records\n• GPS Bounding Box: Lat (39.5-40.3), Lon (116.0-116.8)",
              bg="#FFFFFF", border="#86EFAC", text_color="#166534", fs=8.5, align='center')

    # Multi-depot ride requests
    draw_node(4, 67, 12, 10.5,
              "Multi-Depot Requests\nN Delivery Rides:\n• Ride 1: [S1, {D1,1..D1,m}]\n• Ride 2: [S2, {D2,1..D2,k}]\n• Ride n: [Sn, {Dn,1..Dn,p}]",
              bg="#DCFCE7", border="#22C55E", text_color="#14532D", fs=8.0)

    # Fleet specs
    draw_node(17.5, 67, 12.5, 10.5,
              "Heterogeneous Fleet\n• Depots M >= 3 across city\n• Vehicle Class: Mass, Fuel\n• Base rate: 7.0 L/100km\n• Load penalty: lambda=0.05/100kg\n• CO2 factor: 2.3 kg/L",
              bg="#FEF3C7", border="#F59E0B", text_color="#92400E", fs=8.0)

    # =============================================================
    # SECTION 2: TOP-RIGHT - DATA PREPROCESSING & SPATIAL GRAPH
    # =============================================================
    draw_card(34, 65, 64, 28, "2. Spatial Discretization, Kinematics & Graph Modeling",
              "Map-Matching, Directed Graph Construction & Multi-Class Congestion Labeling", bg="#EFF6FF", border="#93C5FD", title_color="#1E3A8A")

    # Flow inside section 2
    draw_node(36, 80, 17, 8.5,
              "Spatial Grid Binning\n• 30 x 30 Uniform Grid (900 cells)\n• Centroid Geodetic Coordinates\n• Boundary: [Lat_min, Lon_min..\nLat_max, Lon_max]",
              bg="#FFFFFF", border="#93C5FD", text_color="#1E40AF", fs=8.2)

    draw_node(55.5, 80, 18, 8.5,
              "Kinematic Feature Pipeline\n• Haversine Delta d, Duration Delta t\n• Speed v_i = 3.6 * (Delta d / Delta t)\n• Anomaly & Outlier Filtering\n• Step Fuel F_L(v_i) = (F_adj/100)*d_i*eta(v_i)",
              bg="#FFFFFF", border="#93C5FD", text_color="#1E40AF", fs=8.0)

    draw_node(76, 80, 20, 8.5,
              "Spatial Transition Graph G = (V, E, W)\n• Nodes V: Active grid sectors (N approx 300)\n• Directed Edges E: Empirical transitions\n• Edge Weights W: Mean duration, distance,\nfuel consumption, and historical flow counts",
              bg="#DBEAFE", border="#3B82F6", text_color="#1E3A8A", fs=8.0)

    # Bottom of Section 2: Congestion State Labeling & Temporal Split
    draw_node(36, 67, 28, 10.5,
              "TCFMu Feature & Label Extraction\n• Free-flow baseline V_ff = P_90(V_hist)\n• Speed Ratio R_cong(v, h) = V_avg(v, h) / V_ff(v)\n• 3-Class Ground Truth:\n  - FreeFlow (R_cong >= 0.7) [Class 0]\n  - Moderate (0.4 <= R_cong < 0.7) [Class 1]\n  - Congested (R_cong < 0.4) [Class 2]",
              bg="#FFFFFF", border="#93C5FD", text_color="#1E40AF", fs=8.0)

    draw_node(66, 67, 30, 10.5,
              "Temporal Dataset Partitioning\n• Training Set: Hours 00:00 - 18:59 (80% chronological)\n• Evaluation Set: Hours 19:00 - 23:59 (Peak evening congestion)\n• Node Feature Matrix X in R^(N x 4) (x, y, density, speed)\n• Normalized Adjacency A_norm = D^(-1) * A",
              bg="#FFFFFF", border="#93C5FD", text_color="#1E40AF", fs=8.0)

    # Arrows in Top Rows
    draw_arrow(18, 80, 18, 77.5, color="#166534")
    draw_arrow(30, 84, 36, 84, color="#1E40AF", label="Raw Trajectories")
    draw_arrow(53, 84, 55.5, 84, color="#1E40AF")
    draw_arrow(73.5, 84, 76, 84, color="#1E40AF")
    draw_arrow(64.5, 74, 66, 74, color="#1E40AF")

    # =============================================================
    # SECTION 3: MIDDLE-LEFT - GNN SPATIAL EMBEDDER
    # =============================================================
    draw_card(2, 33, 44, 30, "3. Graph Neural Network (GNN) Spatial Embedding Module",
              "Self-Supervised Topological Representation Learning via Geometric Distance Reconstruction",
              bg="#F0F9FF", border="#7DD3FC", title_color="#0369A1")

    draw_node(4, 51, 19, 7.5,
              "Layer 1: Spatial Graph Conv\nm_v^(1) = Sum_{u in N(v)} (h_u^(0) W_neigh) / sqrt(d_u d_v)\nh_v^(1) = ReLU(h_v^(0) W_self + m_v^(1))",
              bg="#FFFFFF", border="#7DD3FC", text_color="#0369A1", fs=7.8)

    draw_node(25, 51, 19, 7.5,
              "Layer 2: Dense Embedding Conv\nm_v^(2) = Sum_{u in N(v)} (h_u^(1) W_neigh) / sqrt(d_u d_v)\nh_v^(2) = h_v^(1) W_self + m_v^(2) + b",
              bg="#FFFFFF", border="#7DD3FC", text_color="#0369A1", fs=7.8)

    draw_node(4, 35, 19, 13.5,
              "Self-Supervised Training\n• Objective (Distance Reconstruction Loss):\n  L_GNN = (1/|E|) * Sum_{(u,v)} (||h_u - h_v||_2 - d_true(u,v))^2\n• Adam Optimizer (lr=0.01, Epochs=100)\n• Reconstructs metric graph geodesics\n• Captures complex road network topology",
              bg="#E0F2FE", border="#0284C7", text_color="#0369A1", fs=8.0)

    draw_node(25, 35, 19, 13.5,
              "Dense Spatial Node Representations\n• Node Embedding Matrix H in R^(N x 64)\n• Latent vector h_i in R^64 for each grid sector v_i\n• Encodes multi-hop topological proximity\n• Seamlessly transfers to routing heuristic\n• Validation MSE converges to < 0.042",
              bg="#BAE6FD", border="#0284C7", text_color="#075985", fs=8.0)

    draw_arrow(23, 55, 25, 55, color="#0284C7")
    draw_arrow(13.5, 51, 13.5, 48.5, color="#0284C7")
    draw_arrow(23, 42, 25, 42, color="#0284C7", label="Learned h_v")

    # =============================================================
    # SECTION 4: MIDDLE-RIGHT - TCFMU PREDICTIVE CONGESTION FORECASTER
    # =============================================================
    draw_card(48, 33, 50, 30, "4. Traffic Congestion Forecasting Module (TCFMu)",
              "Extreme Gradient Boosting (XGBoost) Ensemble for Real-Time Edge Congestion Classification",
              bg="#ECFDF5", border="#6EE7B7", title_color="#065F46")

    draw_node(50, 50, 22, 8.5,
              "8-Dimensional Feature Matrix X_t\n• Spatial: Grid Row, Grid Column\n• Temporal: Operational Hour h in [0, 23]\n• Traffic Dynamics: Flow Count, Points Count\n• Network Topology: Transition In+Out Density\n• Historical State: Baseline Speed V_ff, Lag-1 State",
              bg="#FFFFFF", border="#6EE7B7", text_color="#065F46", fs=7.8)

    draw_node(74, 50, 22, 8.5,
              "XGBoost Regularized Objective\nL_XGB = -Sum_i Sum_{k=0}^2 y_ik log(p_ik) + Sum_j Omega(f_j)\n• Trees: 300, Max Depth: 5, Learning Rate: 0.5\n• Complexity Penalty Omega(f) = gamma*T + 0.5*lambda*||w||^2\n• Multiclass Softmax: p_k = exp(z_k) / Sum exp(z_m)",
              bg="#FFFFFF", border="#6EE7B7", text_color="#065F46", fs=7.8)

    draw_node(50, 35, 22, 12.5,
              "Model Training & Validation\n• Train on Hours 0-18 | Test on Hours 19-23\n• High Accuracy: 91.4%, Macro F1: 0.892\n• FreeFlow Recall: 94.2%, Congested F1: 0.887\n• High-throughput inference (<1ms/query)\n• Avoids deep spatial-temporal lag overhead",
              bg="#D1FAE5", border="#059669", text_color="#065F46", fs=8.0)

    draw_node(74, 35, 22, 12.5,
              "Dynamic Edge Weight Modulation\n• Live predicted congestion level R_cong(e_ij, t)\n• Speed modifier: v_mod = v_base * (1 - 0.35 * R_cong)\n• Modulated edge traversal time t_mod = d_ij / v_mod\n• Modulated fuel and carbon emission matrices\n• Real-time graph cost update prior to routing",
              bg="#A7F3D0", border="#059669", text_color="#064E3B", fs=8.0)

    draw_arrow(72, 54, 74, 54, color="#059669")
    draw_arrow(61, 50, 61, 47.5, color="#059669")
    draw_arrow(72, 41, 74, 41, color="#059669", label="Edge Multipliers")

    # Inter-layer arrows from 2 to 3 and 4
    draw_arrow(80, 65, 35, 63, color="#0284C7", rad=-0.2, label="Graph G=(V, E)")
    draw_arrow(50, 67, 58, 63, color="#059669", rad=0.1, label="Tabular Dataset")

    # =============================================================
    # SECTION 5: BOTTOM - DEEPACO NEURAL HEURISTIC & GREEN ROUTING
    # =============================================================
    draw_card(2, 2, 96, 29, "5. DeepACO Neural Policy Optimization & Multi-Depot Green Fleet Dispatch",
              "REINFORCE Policy Gradient Training, Neural-Guided Ant Colony Search & Multi-Objective Green Logistics Optimization",
              bg="#FAF5FF", border="#D8B4FE", title_color="#581C87")

    # Box 5A: 137-dim feature vector
    draw_node(4, 15, 20, 11.5,
              "137-Dim State Vector f_ij\nf_ij = [h_i || h_j || e_ij || v_k]\n• h_i in R^64: Source GNN embedding\n• h_j in R^64: Destination GNN embedding\n• e_ij in R^5: Transition (d, t, R_cong, F_L, E_CO2)\n• v_k in R^4: Fleet (mass, base_rate, lambda, rho_CO2)",
              bg="#FFFFFF", border="#D8B4FE", text_color="#581C87", fs=7.8)

    # Box 5B: DeepACOHeuristicNet MLP
    draw_node(26, 15, 20, 11.5,
              "DeepACOHeuristicNet (MLP)\n• Linear Layer 1: 137 -> 64 + ReLU\n• Linear Layer 2: 64 -> 32 + ReLU\n• Linear Layer 3: 32 -> 1\n• Output: H_ij = Softplus(MLP(f_ij)) + epsilon\n• Generates strictly positive heuristics H_ij > 0",
              bg="#FFFFFF", border="#D8B4FE", text_color="#581C87", fs=7.8)

    # Box 5C: REINFORCE Policy Training Loop
    draw_node(48, 15, 24, 11.5,
              "REINFORCE Policy Gradient Training\n• Transition Prob: P_ij = exp(beta * log H_ij) / Sum exp(beta * log H_iu)\n• Multi-Ant Rollout: M=10 ants sample routes pi_m\n• Green Cost: J(pi) = w1*d + w2*t + w3*Cong + w4*F_L\n• Policy Gradient Ascent with Baseline Subtraction:\n  Grad_theta J(theta) approx (1/M)*Sum (R(pi_m) - b) * Sum Grad log P_ij",
              bg="#F3E8FF", border="#9333EA", text_color="#581C87", fs=7.6)

    # Box 5D: Classical ACO Search with Neural Guidance
    draw_node(74, 15, 22, 11.5,
              "Ant Colony Combinatorial Search\n• State Transition Rule:\n  P_ij = ([tau_ij]^alpha * [H_ij]^beta) / Sum ([tau_iu]^alpha * [H_iu]^beta)\n• Local Pheromone Decay:\n  tau_ij <- (1 - rho_local)*tau_ij + rho_local*tau_0\n• Global Pheromone Reinforcement:\n  tau_ij <- (1 - rho_global)*tau_ij + Q / J(pi_best)",
              bg="#F3E8FF", border="#9333EA", text_color="#581C87", fs=7.6)

    # Bottom Sub-Row: Green Decision Matrix & Final Dispatch
    draw_node(4, 4, 30, 8.5,
              "Green Logistics Multi-Objective Criteria\nC1: Congestion Minimization (R_cong bottleneck avoidance)\nC2: Fuel Expenditure Reduction (F_L via kinematic efficiency)\nC3: Carbon Emission Mitigation (E_CO2 = F_L * rho_CO2)\nC4: Path Distance Minimization (d_ij in kilometers)\nC5: Travel Duration & Arrival Windows (t_ij in seconds)",
              bg="#FEF3C7", border="#F59E0B", text_color="#92400E", fs=7.8)

    draw_node(36, 4, 33, 8.5,
              "Multi-Depot Candidate Selection Matrix\n• Cross-fleet evaluation across depots D1, D2, D3\n• Capacity feasibility & delivery node partitioning\n• Neural heuristic matrix H in R^(K x K) evaluated in O(1)\n• Pareto optimal trade-off selection across constraints C1..C5",
              bg="#FEF9C3", border="#CA8A04", text_color="#854D0E", fs=7.8)

    draw_node(71, 4, 25, 8.5,
              "Optimal Fleet Route Dispatch\n• Vehicle Routes R1*, R2*, ..., Rn* deployed\n• -23.0% Carbon Emissions vs Greedy\n• -7.2% Inference Latency vs Classical ACO\n• -75% Convergence Iteration Reduction\n• Real-Time Dispatch to Urban Fleet API",
              bg="#DCFCE7", border="#22C55E", text_color="#14532D", fs=8.0)

    # Arrows in Section 5
    draw_arrow(24, 20.5, 26, 20.5, color="#7C3AED")
    draw_arrow(46, 20.5, 48, 20.5, color="#7C3AED")
    draw_arrow(72, 20.5, 74, 20.5, color="#7C3AED")
    draw_arrow(34, 8.2, 36, 8.2, color="#B45309")
    draw_arrow(69, 8.2, 71, 8.2, color="#15803D")

    # Cross arrows from GNN and TCFMu into DeepACO
    draw_arrow(34.5, 35, 14, 26.5, color="#0284C7", rad=-0.15, label="Embeddings h_v (64d)")
    draw_arrow(85, 35, 22, 26.5, color="#059669", rad=0.2, label="Predictions R_cong")
    draw_arrow(85, 15, 85, 12.5, color="#7C3AED")

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"[SUCCESS] Master Architecture Diagram saved to: {output_path}")

def create_flowcharts():
    """
    Generates high-resolution dedicated modular flowcharts for the paper.
    """
    # 1. End-to-End System Flowchart
    fig, ax = plt.subplots(figsize=(18, 5), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    
    stages = [
        ("Raw Trajectory\nIngestion", "10,357 Taxis\n15M+ GPS Points\nBeijing Bounding Box", "#3B82F6"),
        ("Spatial Discretization\n& Kinematics", "30x30 Grid Bins\nHaversine Distance\nSpeed & Fuel Calc", "#0EA5E9"),
        ("GNN Spatial\nEmbedding", "2-Layer GraphConv\nSelf-Supervised MSE\n64-dim Node Vectors", "#6366F1"),
        ("TCFMu Congestion\nForecasting", "8 Tabular Features\nXGBoost 3-Class\nFreeFlow/Mod/Cong", "#10B981"),
        ("DeepACO Policy\nTraining", "137-dim Input Vector\nREINFORCE Ascent\nSoftplus Heuristics", "#8B5CF6"),
        ("Multi-Depot Route\nOptimization", "Dynamic Pheromones\nGreen Constraints\nMulti-Fleet Dispatch", "#F59E0B")
    ]
    
    for i, (title, desc, col) in enumerate(stages):
        x = i * 3.0 + 0.2
        # Card
        box = FancyBboxPatch((x, 0.5), 2.5, 3.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                             facecolor='#F8FAFC', edgecolor=col, linewidth=2.0)
        ax.add_patch(box)
        # Header banner
        header = FancyBboxPatch((x, 3.2), 2.5, 1.1, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=col, edgecolor=col, linewidth=1.0)
        ax.add_patch(header)
        ax.text(x + 1.25, 3.75, f"Phase {i+1}", color='#FFFFFF', fontsize=9, fontweight='bold', ha='center', va='center')
        ax.text(x + 1.25, 3.4, title, color='#FFFFFF', fontsize=9, fontweight='bold', ha='center', va='center')
        ax.text(x + 1.25, 1.8, desc, color='#1E293B', fontsize=8.5, ha='center', va='center')
        
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x + 2.5, 2.4), (x + 2.95, 2.4),
                                    arrowstyle="-|>,head_length=4,head_width=3",
                                    color='#475569', linewidth=2.0)
            ax.add_patch(arrow)
            
    ax.set_xlim(0, len(stages) * 3.0)
    ax.set_ylim(0, 5)
    plt.title("IM-VRM Chronological End-to-End Operational Pipeline", fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.savefig('im_vrm_end_to_end_flowchart.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("[SUCCESS] Saved im_vrm_end_to_end_flowchart.png")

    # 2. GNN Embedding Flowchart
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    
    gnn_steps = [
        (0.5, 2, 2.8, 2.2, "Raw Graph Nodes v in V\n• Node Feature x_v in R^4\n• Coordinates (x, y)\n• Traffic count, Baseline speed", "#0284C7"),
        (4.0, 2, 2.8, 2.2, "GCN Layer 1 (4 -> 32)\n• Neighbor Aggregation\n  m_v^(1) = Sum (h_u W_neigh) / sqrt(d_u d_v)\n• ReLU Combination", "#0369A1"),
        (7.5, 2, 2.8, 2.2, "GCN Layer 2 (32 -> 64)\n• High-Order Topological Mix\n  h_v^(2) = h_v^(1) W_self + m_v^(2)\n• Latent Embeddings h_v", "#075985"),
        (11.0, 2, 2.8, 2.2, "Self-Supervised Loss\nL = (1/|E|) * Sum_{(u,v)} (||h_u - h_v||_2\n - d_true(u, v))^2\n• Adam Optimization (100 Ep)", "#0C4A6E")
    ]
    for x, y, w, h, text, col in gnn_steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.3",
                             facecolor='#F0F9FF', edgecolor=col, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, fontsize=8.5, color='#0F172A', fontweight='bold', ha='center', va='center')
        
    for i in range(len(gnn_steps)-1):
        x1 = gnn_steps[i][0] + gnn_steps[i][2]
        x2 = gnn_steps[i+1][0]
        arrow = FancyArrowPatch((x1, 3.1), (x2, 3.1), arrowstyle="-|>,head_length=4,head_width=3", color='#0284C7', linewidth=2.0)
        ax.add_patch(arrow)
        
    ax.set_xlim(0, 14.5)
    ax.set_ylim(1, 5)
    plt.title("Self-Supervised Graph Neural Network (GNN) Spatial Embedding Pipeline", fontsize=13, fontweight='bold', pad=12, color='#0F172A')
    plt.savefig('gnn_embedding_pipeline_flowchart.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("[SUCCESS] Saved gnn_embedding_pipeline_flowchart.png")

    # 3. TCFMu Flowchart
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    
    tcfmu_steps = [
        (0.5, 2, 2.8, 2.2, "Raw Speed & Density Logs\n• GPS instantaneous speeds\n• Active transition degrees\n• 90th-pct Free-Flow speed", "#059669"),
        (4.0, 2, 2.8, 2.2, "Lag-1 State & Features\n• Vector X_t in R^8\n• Temporal Hour h in [0,23]\n• Auto-regressive R_cong(t-1)", "#047857"),
        (7.5, 2, 2.8, 2.2, "XGBoost 300 Trees\n• Max Depth 5, eta=0.5\n• Multi:softprob objective\n• Split on gain - gamma", "#065F46"),
        (11.0, 2, 2.8, 2.2, "Congestion Regimes\n• FreeFlow (R_cong >= 0.7)\n• Moderate (0.4 <= R < 0.7)\n• Congested (R < 0.4)\n• Accuracy: 91.4%", "#064E3B")
    ]
    for x, y, w, h, text, col in tcfmu_steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.3",
                             facecolor='#ECFDF5', edgecolor=col, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, fontsize=8.5, color='#0F172A', fontweight='bold', ha='center', va='center')
        
    for i in range(len(tcfmu_steps)-1):
        x1 = tcfmu_steps[i][0] + tcfmu_steps[i][2]
        x2 = tcfmu_steps[i+1][0]
        arrow = FancyArrowPatch((x1, 3.1), (x2, 3.1), arrowstyle="-|>,head_length=4,head_width=3", color='#059669', linewidth=2.0)
        ax.add_patch(arrow)
        
    ax.set_xlim(0, 14.5)
    ax.set_ylim(1, 5)
    plt.title("Traffic Congestion Forecasting Module (TCFMu) Machine Learning Pipeline", fontsize=13, fontweight='bold', pad=12, color='#0F172A')
    plt.savefig('tcfmu_prediction_flowchart.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("[SUCCESS] Saved tcfmu_prediction_flowchart.png")

    # 4. DeepACO REINFORCE Flowchart
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    
    deepaco_steps = [
        (0.5, 2, 2.6, 2.4, "State Assembly f_ij\n• h_i, h_j in R^64 (GNN)\n• e_ij in R^5 (Edge)\n• v_k in R^4 (Vehicle)\nTotal 137-dim Tensor", "#7C3AED"),
        (3.7, 2, 2.6, 2.4, "Policy MLP Forward\n• 137 -> 64 -> 32 -> 1\n• Softplus Activation\n• Heuristic Score H_ij > 0\n• Logit: beta * log H_ij", "#6D28D9"),
        (6.9, 2, 2.6, 2.4, "Stochastic Ant Rollout\n• Prob: Categorical Softmax\n• M=10 Simulated Ants\n• Multi-Objective Cost J(pi)\n• Reward R(pi) = -J(pi)", "#5B21B6"),
        (10.1, 2, 2.6, 2.4, "REINFORCE Update\n• Baseline b = (1/M)*Sum R_m\n• Advantage A_m = R_m - b\n• Loss: -A_m * Sum log P_ij\n• Gradient Ascent on theta", "#4C1D95"),
        (13.3, 2, 2.6, 2.4, "Pheromone Routing\n• Hybrid Transition Rule:\n  P_ij proportional to [tau_ij]^alpha * [H_ij]^beta\n• Local rho_loc + Global rho_glob\n• Optimal Route Dispatch", "#3B0764")
    ]
    for x, y, w, h, text, col in deepaco_steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.3",
                             facecolor='#FAF5FF', edgecolor=col, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, fontsize=8.0, color='#0F172A', fontweight='bold', ha='center', va='center')
        
    for i in range(len(deepaco_steps)-1):
        x1 = deepaco_steps[i][0] + deepaco_steps[i][2]
        x2 = deepaco_steps[i+1][0]
        arrow = FancyArrowPatch((x1, 3.2), (x2, 3.2), arrowstyle="-|>,head_length=4,head_width=3", color='#7C3AED', linewidth=2.0)
        ax.add_patch(arrow)
        
    ax.set_xlim(0, 16.5)
    ax.set_ylim(1, 5.2)
    plt.title("DeepACO Neural Policy Network & REINFORCE Policy Gradient Training Loop", fontsize=13, fontweight='bold', pad=12, color='#0F172A')
    plt.savefig('deepaco_reinforce_flowchart.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("[SUCCESS] Saved deepaco_reinforce_flowchart.png")

    # 5. Multi-Depot Green Matrix Flowchart
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    
    matrix_box = FancyBboxPatch((0.5, 0.5), 11, 5, boxstyle="round,pad=0.3,rounding_size=0.4",
                                facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=2.0)
    ax.add_patch(matrix_box)
    ax.text(6.0, 5.0, "Multi-Depot Green Logistics Decision Matrix (Constraints C1 - C5)",
            fontsize=12, fontweight='bold', color='#92400E', ha='center', va='center')
    
    criteria = [
        ("C1: Minimum Congestion", "Preemptively routes around predicted XGBoost bottlenecks (R_cong < 0.4)", "#DC2626"),
        ("C2: Cost of Fuel (L)", "Calculates payload-adjusted velocity curves eta(v) to minimize consumption", "#D97706"),
        ("C3: Carbon Emission (g CO2)", "Multiplies fuel burn by stoichiometric factor rho_CO2 = 2300 g/L", "#059669"),
        ("C4: Travel Distance (km)", "Minimizes cumulative geodetic Haversine path distance across Beijing sectors", "#2563EB"),
        ("C5: Travel Duration & Windows", "Ensures timely delivery within specified shift and time window constraints", "#7C3AED")
    ]
    for idx, (title, desc, c_tag) in enumerate(criteria):
        y_pos = 4.1 - idx * 0.8
        tag_box = FancyBboxPatch((1.0, y_pos - 0.25), 3.2, 0.55, boxstyle="round,pad=0.1,rounding_size=0.2",
                                 facecolor=c_tag, edgecolor=c_tag)
        ax.add_patch(tag_box)
        ax.text(2.6, y_pos, title, fontsize=8.5, fontweight='bold', color='#FFFFFF', ha='center', va='center')
        ax.text(4.5, y_pos, desc, fontsize=8.5, color='#1E293B', ha='left', va='center')
        
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    plt.savefig('multi_depot_green_routing_matrix.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("[SUCCESS] Saved multi_depot_green_routing_matrix.png")

if __name__ == '__main__':
    print("Generating Master System Architecture Diagram & Flowcharts...")
    create_master_architecture_diagram()
    create_flowcharts()
    print("All diagrams generated successfully!")
