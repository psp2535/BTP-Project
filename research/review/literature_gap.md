# LITERATURE GAP ANALYSIS

## 1. Important Prior Works

1.  **Classical Metaheuristics (ACO/GA for MDVRP)**:
    *   *Problem*: Optimizing routes from multiple depots.
    *   *Method*: Ant Colony Optimization with static $1/d$ heuristic.
    *   *Limitations*: Converges slowly; ignores traffic variation; purely distance-focused, ignoring non-linear fuel consumption.
2.  **Pointer Networks for TSP / NCO (Vinyals et al., Bello et al.)**:
    *   *Problem*: Solving TSP using deep reinforcement learning.
    *   *Method*: Seq2Seq models with attention (Pointer Networks) trained via REINFORCE.
    *   *Limitations*: Struggles scaling to 100+ nodes. Evaluated mostly on synthetic 2D Euclidean graphs. Does not handle dynamic edge weights or multi-objective constraints.
3.  **Spatio-Temporal Graph Convolutional Networks (STGCN) for Traffic Forecasting (Yu et al.)**:
    *   *Problem*: Predicting traffic speeds.
    *   *Method*: GCN + Temporal Convolutions.
    *   *Limitations*: Focuses only on prediction. Inference is heavy and cannot easily be invoked millions of times inside a meta-heuristic routing loop.
4.  **DeepACO: Neural-enhanced ant systems for combinatorial optimization (Ye et al., NeurIPS 2023)**:
    *   *Problem*: Eliminating the need to hand-craft heuristics for ACO.
    *   *Method*: Uses a GNN to encode the graph and an MLP to output heuristic values, trained via RL.
    *   *Limitations*: Evaluated on static benchmarks (TSP, CVRP, BPP). It assumes stationary edge weights and Euclidean geometries. It does not account for real-time dynamic disruptions, asymmetric multi-objective costs, or vehicle-specific kinematics.

## 2. Research-Gap Matrix

| Prior Work | Method | Dataset | Limitation | Our Difference (IM-VRM) |
| :--- | :--- | :--- | :--- | :--- |
| **Dorigo et al. (Classical ACO)** | Ant Colony Optimization | Synthetic Benchmarks | Relies on blind, static $1/d$ heuristic; slow convergence; ignores traffic. | Replaces static heuristic with a learned neural policy that anticipates congestion, cutting iterations by 75%. |
| **Bello et al. (Neural Combinatorial Opt.)** | Pointer Networks + REINFORCE | Synthetic TSP/VRP | Assumes static Euclidean graphs; fails to scale. | Uses DeepACO framework rather than direct sequence generation, allowing scaling to large urban graphs. |
| **Yu et al. (ST-GCN)** | Deep Spatio-Temporal Graph Networks | PeMS, METR-LA | Excellent forecasting but computationally heavy for real-time routing loops. | Replaces heavy ST-GCN with XGBoost (TCFMu) for sub-millisecond inference, enabling integration into live routing. |
| **Ye et al. (DeepACO)** | DeepACO (GNN + MLP + RL) | Static TSP, CVRP | Operates on static graphs. Cannot handle dynamic traffic or non-linear emission models. | Augments the DeepACO state tensor (137-dim) with dynamic congestion forecasts (XGBoost) and kinematic emission constraints for non-stationary environments. |

## 3. Conclusion on Gap
The critical gap is that **Neural Combinatorial Optimization has largely remained in the realm of static, synthetic, Euclidean benchmarks.** Real-world smart city logistics requires handling non-stationary dynamics (traffic jams) and complex multi-objective costs (green emissions based on vehicle physics). This paper bridges the gap by adapting DeepACO to a dynamic environment via predictive congestion forecasting and self-supervised spatial embeddings.
