# RESEARCH QUESTIONS AND EXPERIMENT PLAN

## 1. Research Questions (RQs)
Based on the novelty audit and literature gap, the paper must answer the following precise research questions experimentally:

*   **RQ1 (Performance & Efficiency):** Does integrating neural heuristics (DeepACO) trained via policy gradients improve multi-objective green routing efficiency compared to exact (Time-Dependent Dijkstra) and classical meta-heuristic (ACO) solvers?
*   **RQ2 (Component Contribution):** To what extent do the self-supervised spatial embeddings (GNN) and dynamic congestion forecasts (TCFMu) independently contribute to the routing performance of the DeepACO policy?
*   **RQ3 (Robustness to Congestion):** How robust is the proposed neural policy when subjected to highly non-stationary traffic regimes (e.g., peak evening rush hour vs. off-peak hours)?
*   **RQ4 (Scalability):** Does the inference latency of the DeepACO policy scale linearly with increasing multi-depot fleet sizes, enabling real-time urban dispatch?

## 2. Experimental Rigor Audit
*Current Status*: The project evaluates Greedy, TD-Dijkstra, Classical ACO, and DeepACO. It includes an ablation study (removing GNN, TCFMu, REINFORCE) and a scalability test (50 to 300 vehicles).
*Missing Experiments*: 
1.  **Hyperparameter Sensitivity Analysis**: How sensitive is DeepACO to the $\beta$ (heuristic weight) and $q_0$ (exploitation rate) parameters? (This is partially in the paper, but needs a formal script).
2.  **Robustness across Temporal Shifts**: The paper mentions evaluating on evening rush hour (19:00-23:59), but a formal comparison matrix of performance across Morning Peak vs. Afternoon Off-Peak vs. Evening Peak is required to answer RQ3.

## 3. The Experimental Matrix

| Experiment | Hypothesis | Independent Var | Dependent Var | Dataset | Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Main Results (RQ1)** | DeepACO significantly reduces $CO_2$ and duration while remaining computationally tractable. | Routing Algorithm | Distance, Duration, $CO_2$, Execution Time | T-Drive (Test Set) | Greedy, TD-Dijkstra, Classical ACO |
| **Exp 2: Ablation Study (RQ2)** | GNN and TCFMu are critical for avoiding congested non-Euclidean bottlenecks. | DeepACO Components | Distance, Duration, $CO_2$ | T-Drive (Test Set) | Full DeepACO |
| **Exp 3: Temporal Robustness (RQ3)** | DeepACO outperforms classical methods most significantly during peak congestion hours. | Time of Day (Peak vs Off-Peak) | $CO_2$, Duration reduction % | T-Drive (Full Set) | Classical ACO |
| **Exp 4: Fleet Scalability (RQ4)** | DeepACO inference scales linearly, while exact solvers scale exponentially. | Fleet Size (50 to 300) | Inference Latency | T-Drive | TD-Dijkstra |
| **Exp 5: Parameter Sensitivity** | Proper balancing of $\beta$ and $q_0$ prevents premature convergence. | $\beta$, $q_0$ | $CO_2$, Iterations to converge | T-Drive | N/A |

### Classification of Experiments
1.  **Essential Experiments**: Exp 1, Exp 2, Exp 4. (Already implemented in `main.py`).
2.  **Strongly Recommended**: Exp 3, Exp 5. (Need standalone scripts).
3.  **Optional**: Cross-dataset validation on NYC Taxi Data. (Will document as a limitation due to data unavailability).
