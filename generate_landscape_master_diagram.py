import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Set global matplotlib rendering styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def build_landscape_master_diagram(output_path='im_vrm_master_system_architecture.png'):
    """
    Generates an ultra-wide, high-definition (32x18 inch, 300 DPI) landscape
    master system architecture diagram covering every single detail of the IM-VRM project.
    """
    fig = plt.figure(figsize=(32, 18), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    bg_color = "#0B1120"  # Sleek modern Dark Slate / Navy background for ultra-premium aesthetic
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Color Palette Tokens
    c_card_bg = "#1E293B"      # Dark panel fill
    c_subcard_bg = "#0F172A"   # Darker sub-panel fill
    c_white = "#FFFFFF"
    c_light_slate = "#94A3B8"
    c_cyan = "#38BDF8"         # Ingestion / Graph
    c_blue = "#60A5FA"         # GNN Blue
    c_emerald = "#34D399"      # TCFMu Green
    c_purple = "#A78BFA"       # DeepACO Violet
    c_amber = "#FBBF24"        # Green Matrix / Constraints
    c_rose = "#F43F5E"         # Congestion / Bottleneck
    c_arrow = "#38BDF8"        # Signal flow connector

    # Helper function: Draw Container Card
    def draw_panel(x, y, w, h, title="", step_num="", subtitle="", accent="#38BDF8", bg="#1E293B", rad=1.2):
        # Outer Card
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={rad}",
                             facecolor=bg, edgecolor=accent, linewidth=2.0, zorder=1)
        ax.add_patch(box)
        
        # Header Badge
        if title:
            # Pill badge for step number
            if step_num:
                badge = FancyBboxPatch((x + 1.2, y + h - 2.8), 2.2, 1.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                                       facecolor=accent, edgecolor=accent, zorder=3)
                ax.add_patch(badge)
                ax.text(x + 2.3, y + h - 1.9, step_num, fontsize=11, fontweight='heavy', color="#0F172A", ha='center', va='center', zorder=4)
                
            ax.text(x + (3.8 if step_num else 1.2), y + h - 1.9, title, fontsize=12.5, fontweight='heavy', color=c_white, va='center', ha='left', zorder=3)
            
        if subtitle:
            ax.text(x + 1.2, y + h - 3.8, subtitle, fontsize=9.5, fontstyle='italic', color=c_light_slate, va='center', ha='left', zorder=3)
            
        return box

    # Helper function: Draw Content Sub-Box
    def draw_subbox(x, y, w, h, text, bg="#0F172A", border="#334155", text_color="#E2E8F0", fs=9.0, bold=False, rad=0.6, align='center', title=""):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.25,rounding_size={rad}",
                             facecolor=bg, edgecolor=border, linewidth=1.2, zorder=2)
        ax.add_patch(box)
        
        if title:
            ax.text(x + w/2, y + h - 1.0, title, fontsize=fs+0.5, fontweight='bold', color="#38BDF8", ha='center', va='top', zorder=4)
            y_offset = -0.5
        else:
            y_offset = 0.0

        fw = 'bold' if bold else 'normal'
        if align == 'center':
            ax.text(x + w/2, y + h/2 + y_offset, text, fontsize=fs, fontweight=fw, color=text_color, va='center', ha='center', zorder=4)
        else:
            ax.text(x + 0.8, y + h/2 + y_offset, text, fontsize=fs, fontweight=fw, color=text_color, va='center', ha='left', zorder=4)
        return box

    # Helper function: Curved Arrows with Glow
    def draw_flow_arrow(x1, y1, x2, y2, color="#38BDF8", lw=2.2, rad=0.0, label=""):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="-|>,head_length=6,head_width=4",
                                edgecolor=color, facecolor=color,
                                linewidth=lw, zorder=5)
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + (1.2 if rad >= 0 else -1.2)
            ax.text(mx, my, label, fontsize=8.5, fontweight='bold', color="#0F172A", ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='none', alpha=0.95), zorder=6)

    # -------------------------------------------------------------
    # TOP HEADER BANNER
    # -------------------------------------------------------------
    banner_box = FancyBboxPatch((1.5, 93.8), 97, 4.8, boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor="#0284C7", edgecolor="#38BDF8", linewidth=2.0, zorder=2)
    ax.add_patch(banner_box)
    ax.text(50, 96.6, "IM-VRM: Intelligent Multi-Depot Vehicle Routing & Management Framework",
            fontsize=20, fontweight='heavy', color='#FFFFFF', ha='center', va='center', zorder=3)
    ax.text(50, 94.8, "End-to-End System Architecture: GPS Trajectory Mining • Self-Supervised GNN • XGBoost TCFMu • DeepACO Policy Reinforcement Learning",
            fontsize=11.5, fontweight='medium', color='#E0F2FE', ha='center', va='center', zorder=3)

    # =============================================================
    # ROW 1 (TOP): INGESTION, GRAPH DISCRETIZATION & TEMPORAL SPLIT
    # =============================================================

    # 1. Smart City Sensing & Ingestion (x=1.5, y=63.5, w=30, h=29)
    draw_panel(1.5, 63.5, 30, 29, "Smart City Traffic Sensing & Ingestion", "1",
               "Microsoft T-Drive GPS Repository & Urban Fleets", accent=c_cyan)
    
    draw_subbox(2.8, 77.5, 27.4, 9.5,
                "Microsoft T-Drive GPS Stream\n• 10,357 Active Taxicabs in Beijing Metropolitan Area\n• 15,248,163 GPS Spatial-Temporal Trajectory Records\n• Geodetic Bounds: Lat [39.5, 40.3] | Lon [116.0, 116.8]\n• Continuous GPS Telemetry: <Timestamp, Taxi-ID, Lat, Lon>",
                bg=c_subcard_bg, border=c_cyan, text_color="#E2E8F0", fs=8.8, title="Trajectory Data Ingestion")
    
    draw_subbox(2.8, 65.0, 13.2, 11.0,
                "Multi-Depot Requests (N)\n• Customer Deliveries (C)\n• Depot Fleet Bases (D)\n• Ride 1: [S1 -> {D1,1..D1,m}]\n• Ride 2: [S2 -> {D2,1..D2,k}]\n• Time Windows: [a_i, b_i]",
                bg=c_subcard_bg, border="#475569", text_color="#E2E8F0", fs=8.2, title="Order Dispatch")
    
    draw_subbox(16.8, 65.0, 13.4, 11.0,
                "Heterogeneous Fleets (K)\n• Vehicle Types: Van, Truck\n• Base Fuel: 7.0 L/100km\n• Cargo Mass: Payload (kg)\n• Load Penalty: lambda=0.05\n• Carbon Factor: 2300 g/L",
                bg=c_subcard_bg, border="#475569", text_color="#E2E8F0", fs=8.2, title="Vehicle Classes")

    # 2. Spatial Discretization & Kinematics Engine (x=33.0, y=63.5, w=34, h=29)
    draw_panel(33.0, 63.5, 34, 29, "Spatial Discretization & Kinematic Graph Modeling", "2",
               "30x30 Grid Partitioning, Velocity Curves & Graph Construction", accent=c_cyan)
    
    draw_subbox(34.2, 77.5, 15.5, 9.5,
                "Spatial 30x30 Grid Binning\n• 900 Total Sector Cells\n• Cell Centroid Lat/Lon\n• Map-Matching Trajectories\n• Active Graph Nodes: |V|=294\n• Grid Mapping: (r, c) Index",
                bg=c_subcard_bg, border=c_cyan, text_color="#E2E8F0", fs=8.2, title="Grid Partition")
    
    draw_subbox(50.5, 77.5, 15.3, 9.5,
                "Kinematic Feature Pipeline\n• Haversine Dist: Delta d (m)\n• Traversal Duration: Delta t (s)\n• Speed: v_i = 3.6*(Delta d / Delta t)\n• Speed Filters: 0 < v_i <= 140\n• Acceleration Anomaly Removal",
                bg=c_subcard_bg, border=c_cyan, text_color="#E2E8F0", fs=8.2, title="Kinematic Processing")
    
    draw_subbox(34.2, 65.0, 31.6, 11.0,
                "Spatial Transition Graph Construction G = (V, E, W)\n• Directed Edges E: 1,428 Empirical Transition Corridors\n• Edge Weights W: [Distance d_ij, FreeFlow Speed v_ff, Base Fuel F_ij]\n• Velocity Efficiency Multiplier: eta(v) = 1.0 + 0.5*((45-v)/45)^2 + 0.3*((v-70)/30)^2\n• Step Fuel: F_L(v) = Idling if v < 0.5 km/h else (F_adj / 100) * d * eta(v)",
                bg=c_subcard_bg, border="#475569", text_color="#E2E8F0", fs=8.2, title="Graph & Non-Linear Fuel Model")

    # 3. Temporal Partitioning & Featurization (x=68.5, y=63.5, w=30, h=29)
    draw_panel(68.5, 63.5, 30, 29, "Temporal Partitioning & Congestion Labeling", "3",
               "3-Class Traffic Regimes & Normalized Laplacian", accent=c_cyan)
    
    draw_subbox(69.8, 77.5, 27.4, 9.5,
                "3-Class Congestion Regime Extraction\n• Free-Flow Baseline Speed: V_ff = 90th percentile historical speed\n• Congestion Ratio: R_cong(v, h) = V_avg(v, h) / V_ff(v)\n• Class 0 (FreeFlow): R_cong >= 0.70\n• Class 1 (Moderate): 0.40 <= R_cong < 0.70\n• Class 2 (Congested): R_cong < 0.40",
                bg=c_subcard_bg, border=c_cyan, text_color="#E2E8F0", fs=8.4, title="Ground Truth Congestion Labels")
    
    draw_subbox(69.8, 65.0, 13.2, 11.0,
                "Temporal Slicing\n• Train: Hours 00-18\n  (80% Chronological)\n• Test: Hours 19-23\n  (Peak Evening Rush)\n• Out-of-Time Eval",
                bg=c_subcard_bg, border="#475569", text_color="#E2E8F0", fs=8.2, title="Dataset Split")
    
    draw_subbox(83.8, 65.0, 13.4, 11.0,
                "Graph Matrices\n• Node Features: X in R^(N x 4)\n  [x, y, degree, speed]\n• Normalized Adjacency:\n  A_norm = D^(-1) * A\n• Graph Corridors: |E|=1,428",
                bg=c_subcard_bg, border="#475569", text_color="#E2E8F0", fs=8.2, title="Laplacian Matrices")

    # Connectors Row 1
    draw_flow_arrow(31.5, 78.0, 33.0, 78.0, color=c_cyan)
    draw_flow_arrow(67.0, 78.0, 68.5, 78.0, color=c_cyan)

    # =============================================================
    # ROW 2 (MIDDLE): GNN EMBEDDING & TCFMU CONGESTION FORECASTER
    # =============================================================

    # 4. Self-Supervised GNN (x=1.5, y=32.5, w=47.5, h=29.5)
    draw_panel(1.5, 32.5, 47.5, 29.5, "Self-Supervised Graph Neural Network (GNN) Spatial Embedder", "4",
               "2-Layer Graph Convolutional Network (GCN) for Multi-Hop Topological Representation Learning", accent=c_blue)
    
    draw_subbox(2.8, 47.0, 21.8, 10.0,
                "Layer 1: Spatial Message Passing (4 -> 32)\n• Neighbor Aggregation: m_v^(1) = Sum_{u in N(v)} (h_u W_neigh) / sqrt(d_u d_v)\n• Node Combination: h_v^(1) = ReLU(h_v^(0) W_self + m_v^(1) + b_1)\n• Captures 1st-order localized road topology",
                bg=c_subcard_bg, border=c_blue, text_color="#E2E8F0", fs=8.4, title="GCN Layer 1")
    
    draw_subbox(26.0, 47.0, 21.8, 10.0,
                "Layer 2: Dense Embedding Conv (32 -> 64)\n• High-Order Aggregation: m_v^(2) = Sum_{u in N(v)} (h_u^(1) W_neigh) / sqrt(d_u d_v)\n• Linear Combination: h_v^(2) = h_v^(1) W_self + m_v^(2) + b_2\n• Generates 64-dimensional latent embedding vector h_v",
                bg=c_subcard_bg, border=c_blue, text_color="#E2E8F0", fs=8.4, title="GCN Layer 2")
    
    draw_subbox(2.8, 34.0, 21.8, 11.5,
                "Self-Supervised Geometric Objective\n• Loss: L_GNN = (1/|E|) * Sum_{(u,v)} (||h_u - h_v||_2 - d_true(u,v))^2\n• Reconstructs true metric road distance from topological space\n• Adam Optimizer (lr=0.01, 100 Epochs)\n• Replaces manual coordinates with continuous spatial manifolds",
                bg=c_subcard_bg, border=c_blue, text_color="#E2E8F0", fs=8.4, title="Distance Reconstruction Loss")
    
    draw_subbox(26.0, 34.0, 21.8, 11.5,
                "Dense Node Embedding Matrix H in R^(N x 64)\n• Continuous latent representations h_i in R^64 for all 294 active sectors\n• Preserves multi-hop shortest paths & corridor connectivity\n• Seamlessly transfers topological inductive bias to routing policy\n• Validation MSE converges to < 0.042",
                bg=c_subcard_bg, border=c_blue, text_color="#E2E8F0", fs=8.4, title="Learned Spatial Embeddings")

    draw_flow_arrow(24.6, 52.0, 26.0, 52.0, color=c_blue)
    draw_flow_arrow(13.7, 47.0, 13.7, 45.5, color=c_blue)
    draw_flow_arrow(24.6, 39.5, 26.0, 39.5, color=c_blue)

    # 5. Traffic Congestion Forecasting Module (TCFMu) (x=51.0, y=32.5, w=47.5, h=29.5)
    draw_panel(51.0, 32.5, 47.5, 29.5, "Traffic Congestion Forecasting Module (TCFMu)", "5",
               "Extreme Gradient Boosting (XGBoost) Ensemble for Real-Time Edge Congestion State Prediction", accent=c_emerald)
    
    draw_subbox(52.3, 47.0, 21.8, 10.0,
                "8-Dimensional State Feature Matrix X_t\n• Spatial: Grid Row, Grid Column\n• Temporal: Operational Hour h in [0, 23]\n• Traffic Volume: Flow count, Point count\n• Topological: Transition In-degree + Out-degree\n• Historical: Baseline speed V_ff, Lag-1 State",
                bg=c_subcard_bg, border=c_emerald, text_color="#E2E8F0", fs=8.4, title="Feature Engineering")
    
    draw_subbox(75.5, 47.0, 21.8, 10.0,
                "Regularized XGBoost Multi-Class Ensemble\n• Objective: L_XGB = -Sum_i Sum_k y_ik log(p_ik) + Sum_m Omega(f_m)\n• 300 Decision Trees, Max Depth: 5, Learning Rate: 0.50\n• Multiclass Softmax: p_k = exp(z_k) / Sum exp(z_j)\n• Fast Inference: < 1ms per query",
                bg=c_subcard_bg, border=c_emerald, text_color="#E2E8F0", fs=8.4, title="XGBoost Classifier")
    
    draw_subbox(52.3, 34.0, 21.8, 11.5,
                "Model Performance & Validation\n• Train on Hours 00-18 | Evaluate on Hours 19-23\n• Classification Accuracy: 91.4%\n• Macro F1-Score: 0.892\n• FreeFlow Recall: 94.2% | Congested F1: 0.887\n• Real-Time Edge Speed Prediction: v_mod = v_base*(1 - 0.35*R_cong)",
                bg=c_subcard_bg, border=c_emerald, text_color="#E2E8F0", fs=8.4, title="Predictive Accuracy")
    
    draw_subbox(75.5, 34.0, 21.8, 11.5,
                "Dynamic Graph Cost Modulation\n• Live predicted congestion multipliers across all directed edges e_ij in E\n• Dynamic Traversal Time: t_mod = d_ij / v_mod\n• Dynamic Fuel Burn: F_mod(v_mod, Payload)\n• Real-Time Edge Tensor Update prior to DeepACO ant exploration",
                bg=c_subcard_bg, border=c_emerald, text_color="#E2E8F0", fs=8.4, title="Dynamic Edge Modulation")

    draw_flow_arrow(74.1, 52.0, 75.5, 52.0, color=c_emerald)
    draw_flow_arrow(63.2, 47.0, 63.2, 45.5, color=c_emerald)
    draw_flow_arrow(74.1, 39.5, 75.5, 39.5, color=c_emerald)

    # Inter-Row Connectors Row 1 -> Row 2
    draw_flow_arrow(25.0, 63.5, 25.0, 62.0, color=c_cyan, label="Graph Structure G")
    draw_flow_arrow(75.0, 63.5, 75.0, 62.0, color=c_cyan, label="Tabular Dataset X_t")

    # =============================================================
    # ROW 3 (BOTTOM): DEEPACO NEURAL POLICY, ACO & GREEN DISPATCH
    # =============================================================

    # 6. DeepACO State Tensor & MLP Policy (x=1.5, y=1.5, w=31.5, h=29.5)
    draw_panel(1.5, 1.5, 31.5, 29.5, "DeepACO Neural Policy Network", "6",
               "137-Dim State Vector & Positive Softplus Heuristics", accent=c_purple)
    
    draw_subbox(2.8, 16.5, 28.9, 10.0,
                "137-Dimensional Candidate State Tensor f_ij\n• h_i in R^64: GNN Spatial Embedding of current source node v_i\n• h_j in R^64: GNN Spatial Embedding of candidate next node v_j\n• e_ij in R^5: Edge Dynamics (Distance, Duration, Congestion, Fuel, CO2)\n• v_k in R^4: Vehicle Fleet Profile (Curb Mass, Base Rate, Lambda, CO2 Factor)",
                bg=c_subcard_bg, border=c_purple, text_color="#E2E8F0", fs=8.2, title="State Tensor Assembly")
    
    draw_subbox(2.8, 3.0, 28.9, 12.0,
                "DeepACOHeuristicNet (Multi-Layer Perceptron)\n• Linear Layer 1: 137 -> 64 + ReLU\n• Linear Layer 2: 64 -> 32 + ReLU\n• Linear Layer 3: 32 -> 1\n• Strictly Positive Heuristic Visibility:\n  H_ij = Softplus(MLP(f_ij; theta)) + epsilon = ln(1 + exp(z)) + 10^(-6)\n• Replaces static inverse distance (1/d_ij) with learned contextual visibility",
                bg=c_subcard_bg, border=c_purple, text_color="#E2E8F0", fs=8.2, title="Policy Network Architecture")

    # 7. REINFORCE Policy Gradient Training Loop (x=34.5, y=1.5, w=33.5, h=29.5)
    draw_panel(34.5, 1.5, 33.5, 29.5, "REINFORCE Policy Gradient Training Loop", "7",
               "Multi-Ant Trajectory Sampling, Reward Baseline & Gradient Ascent", accent=c_purple)
    
    draw_subbox(35.8, 16.5, 30.9, 10.0,
                "Stochastic Ant Exploration Rollouts\n• Categorical Softmax Action Selection: P_ij = (H_ij)^beta / Sum (H_iu)^beta\n• Batch of M=10 Concurrent Artificial Ants sample complete vehicle tours\n• Multi-Objective Cost: J(pi) = w1*d + w2*t + w3*Cong + w4*F_L\n• Scalar Reinforcement Reward: R(pi) = -J(pi)",
                bg=c_subcard_bg, border=c_purple, text_color="#E2E8F0", fs=8.2, title="Trajectory Rollout & Reward")
    
    draw_subbox(35.8, 3.0, 30.9, 12.0,
                "Policy Gradient Ascent with Baseline Subtraction\n• Empirical Mean Baseline: b = (1/M) * Sum_{m=1}^M R(pi_m)\n• Policy Gradient Estimator: Grad_theta J(theta) = (1/M) Sum (R_m - b) Sum Grad ln P_ij\n• Adam Optimizer (lr=0.005, 20 Epochs)\n• Trains policy network to prioritize green, low-congestion candidate edges",
                bg=c_subcard_bg, border=c_purple, text_color="#E2E8F0", fs=8.2, title="Policy Gradient Update")

    # 8. Ant Colony Search & Green Fleet Dispatch (x=69.5, y=1.5, w=29.0, h=29.5)
    draw_panel(69.5, 1.5, 29.0, 29.5, "Multi-Depot Green Fleet Dispatch", "8",
               "Dynamic Pheromones, Constraints C1-C5 & Optimal Routes", accent=c_amber)
    
    draw_subbox(70.8, 16.5, 26.4, 10.0,
                "Hybrid Ant Colony Decision Rule\n• Transition Prob: P_ij proportional to [tau_ij]^alpha * [H_ij]^beta\n• Pseudo-Random Proportional Action Selection (q_0 = 0.70)\n• Local Pheromone Decay: tau_ij <- (1 - rho_loc)*tau_ij + rho_loc*tau_0\n• Global Pheromone Update: tau_ij <- (1 - rho_glob)*tau_ij + Q / J_best\n• Dynamic Pheromone Bounds: [tau_min, tau_max] = [0.05, 5.0]",
                bg=c_subcard_bg, border=c_amber, text_color="#E2E8F0", fs=8.0, title="Ant Colony Optimization")
    
    draw_subbox(70.8, 3.0, 26.4, 12.0,
                "Green Multi-Objective Dispatch (C1 - C5)\n• C1: Congestion Minimization (Bottleneck Avoidance)\n• C2: Fuel Burn Optimization (Kinematic Velocity Curve)\n• C3: Carbon Emissions Mitigation (-23.0% vs Greedy)\n• C4: Travel Distance Minimization (Haversine Geo-Metric)\n• C5: Time Windows [a_i, b_i] & Fast Latency (0.606s/veh)\n• Final Multi-Depot Fleet Schedule Pi* = {pi_1*, ..., pi_K*}",
                bg=c_subcard_bg, border=c_amber, text_color="#E2E8F0", fs=8.0, title="Optimal Fleet Route Schedule")

    # Connectors Row 3
    draw_flow_arrow(33.0, 16.0, 34.5, 16.0, color=c_purple)
    draw_flow_arrow(68.0, 16.0, 69.5, 16.0, color=c_amber)

    # Inter-Row Connectors Row 2 -> Row 3
    draw_flow_arrow(25.0, 32.5, 17.0, 31.0, color=c_blue, rad=-0.1, label="GNN Embeddings h_v (64d)")
    draw_flow_arrow(75.0, 32.5, 25.0, 31.0, color=c_emerald, rad=0.15, label="Predicted Congestion R_cong")

    # Save Figure with ultra-high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"[SUCCESS] Landscape Master System Architecture Diagram generated and saved to: {output_path}")

if __name__ == '__main__':
    build_landscape_master_diagram()
