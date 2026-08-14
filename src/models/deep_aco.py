import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.utils.helpers import haversine_distance
from src.features.metrics import compute_fuel_and_emissions
from src.models.greedy_routing import dijkstra_pathfinder

logger = logging.getLogger("IM-VRM")

def get_cell_centroid(row, col, bbox, grid_dims):
    """
    Compute geodetic coordinates for a grid cell centroid.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols
    lat = min_lat + (row + 0.5) * lat_step
    lon = min_lon + (col + 0.5) * lon_step
    return lat, lon

def evaluate_route_metrics_fast(path, edge_stats, congestion_map, sorted_nodes, vehicle_config, bbox, grid_dims):
    """
    Compute actual metrics for a route path segment.
    """
    if not path or len(path) < 2:
        return {
            "distance_km": 0.0,
            "duration_sec": 0.0,
            "avg_congestion": 0.0,
            "fuel_l": 0.0,
            "co2_g": 0.0
        }
        
    total_dist_m = 0.0
    total_dur_s = 0.0
    total_fuel_l = 0.0
    total_co2_g = 0.0
    congestions_visited = []
    
    for i in range(len(path) - 1):
        u_idx = path[i]
        v_idx = path[i+1]
        
        u_cell = sorted_nodes[u_idx]
        v_cell = sorted_nodes[v_idx]
        
        congestions_visited.append(congestion_map.get(v_cell, 0.0))
        dur = edge_stats.get((u_idx, v_idx), 0.0)
        
        lat_u, lon_u = get_cell_centroid(u_cell[0], u_cell[1], bbox, grid_dims)
        lat_v, lon_v = get_cell_centroid(v_cell[0], v_cell[1], bbox, grid_dims)
        dist = haversine_distance(lat_u, lon_u, lat_v, lon_v)
        
        total_dist_m += dist
        total_dur_s += dur
        
        speed = (dist / dur) * 3.6 if dur > 0 else 0.0
        
        fuel_step, co2_step = compute_fuel_and_emissions(
            np.array([speed]), 
            np.array([dist]), 
            np.array([dur]), 
            vehicle_config, 
            payload_kg=0.0
        )
        total_fuel_l += float(fuel_step[0])
        total_co2_g += float(co2_step[0])
        
    if not congestions_visited:
        congestions_visited = [congestion_map.get(sorted_nodes[path[0]], 0.0)]
    mean_cong = float(np.mean(congestions_visited))
    
    return {
        "distance_km": total_dist_m / 1000.0,
        "duration_sec": total_dur_s,
        "avg_congestion": mean_cong,
        "fuel_l": total_fuel_l,
        "co2_g": total_co2_g
    }

class DeepACOHeuristicNet(nn.Module):
    """
    Learned edge heuristic network for DeepACO.
    Maps edge features + source/target node embeddings to a positive scalar score.
    Input dimension: 64 (src emb) + 64 (dst emb) + 9 (edge & vehicle attributes) = 137
    """
    def __init__(self, input_dim=137, hidden_dim=64):
        super(DeepACOHeuristicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # Predict positive scalar scores using Softplus to prevent zeros/negatives
        return self.softplus(self.net(x)) + 1e-6

def get_edge_feature_vector(src_idx, dst_idx, embeddings_np, cost_cache, vehicle_config):
    """
    Assemble the 137-dimensional edge feature vector for candidate transition.
    """
    emb_src = embeddings_np[src_idx]
    emb_dst = embeddings_np[dst_idx]
    
    cache = cost_cache.get((src_idx, dst_idx))
    if cache is None:
        # Unreachable transition placeholder values
        dist_km = 999.0
        dur_min = 999.0
        cong = 2.0
        fuel_l = 999.0
        co2_g = 999.0
    else:
        dist_km = cache["distance_m"] / 1000.0
        dur_min = cache["duration_sec"] / 60.0
        cong = cache["congestion"]
        fuel_l = cache["fuel_l"]
        co2_g = cache["co2_g"]
        
    # Scale vehicle attributes
    mass = vehicle_config.get("mass_kg", 1500.0) / 1000.0
    base_rate = vehicle_config.get("base_fuel_rate_l_per_100km", 7.0)
    penalty = vehicle_config.get("load_penalty_factor", 0.05)
    co2_rate = vehicle_config.get("co2_g_per_liter", 2300.0) / 1000.0
    
    # 64 + 64 + 5 (edge) + 4 (vehicle) = 137 features
    feat = np.concatenate([
        emb_src, emb_dst,
        [dist_km, dur_min, cong, fuel_l, co2_g / 1000.0],
        [mass, base_rate, penalty, co2_rate]
    ])
    return feat

def train_deep_aco(fleet, depots, embeddings, adj_dict, N, predictions_df, sorted_nodes, edges_df, config, 
                   epochs=20, num_ants=10, lr=0.005, beta_param=2.0, reward_weights=None):
    """
    Train the DeepACOHeuristicNet policy network using policy gradient REINFORCE.
    """
    if reward_weights is None:
        reward_weights = {
            "w_distance": 0.30,
            "w_duration": 0.25,
            "w_congestion": 0.20,
            "w_fuel": 0.15,
            "w_co2": 0.10,
            "w_penalty": 100.0
        }
        
    logger.info("Initializing DeepACOHeuristicNet model.")
    policy_net = DeepACOHeuristicNet(input_dim=137)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    embeddings_np = embeddings.numpy()
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    # Pre-parse graph configurations
    bbox = config["preprocessing"].get("bbox", [39.5, 116.0, 40.3, 116.8])
    grid_dims = (config["spatial_grid"].get("num_rows", 30), config["spatial_grid"].get("num_cols", 30))
    
    # Cache static edge durations
    edge_stats = {}
    for _, edge in edges_df.iterrows():
        u = (int(edge["grid_row"]), int(edge["grid_col"]))
        v = (int(edge["next_row"]), int(edge["next_col"]))
        if u in node_to_idx and v in node_to_idx:
            edge_stats[(node_to_idx[u], node_to_idx[v])] = float(edge["avg_duration_sec"])
            
    # Cache congestion prediction average
    congestion_map = {}
    if not predictions_df.empty:
        df_avg = predictions_df.groupby(["grid_row", "grid_col"])["predicted_congestion_level"].mean().reset_index()
        for _, row in df_avg.iterrows():
            congestion_map[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["predicted_congestion_level"])
            
    logger.info("Pre-calculating pairwise transition cost caches across active routing nodes.")
    # Extract candidate depots and destinations from all fleet configurations
    visited_cells = set()
    for v in fleet:
        visited_cells.add((v["depot_grid_row"], v["depot_grid_col"]))
    
    # Include default depots
    for d in depots:
        visited_cells.add((d["grid_row"], d["grid_col"]))
        
    # We build cache for all pair nodes that vehicles could visit
    candidate_node_indices = sorted(list(set([node_to_idx[c] for c in visited_cells if c in node_to_idx])))
    
    cost_cache = {}
    for u_idx in candidate_node_indices:
        for v_idx in candidate_node_indices:
            if u_idx == v_idx:
                continue
            path, dur, dist_m = dijkstra_pathfinder(u_idx, v_idx, adj_dict, N)
            if path is not None:
                cost_cache[(u_idx, v_idx)] = {
                    "path": path,
                    "duration_sec": dur,
                    "distance_m": dist_m
                }
            else:
                cost_cache[(u_idx, v_idx)] = None
                
    # Build complete cache including all delivery nodes dynamically during training.
    # To optimize, we will build cache mapping between depot nodes and any selected delivery node
    
    loss_history = []
    reward_history = []
    
    rng = np.random.default_rng(seed=42)
    
    logger.info(f"Starting DeepACO REINFORCE Training: Epochs={epochs}, Ants={num_ants}, Batch={len(fleet)} vehicles.")
    
    for epoch in range(1, epochs + 1):
        policy_net.train()
        optimizer.zero_grad()
        
        epoch_losses = []
        epoch_rewards = []
        
        for vehicle in fleet:
            v_id = vehicle["vehicle_id"]
            depot_row = vehicle["depot_grid_row"]
            depot_col = vehicle["depot_grid_col"]
            vehicle_config = vehicle["config"]
            
            # Identify mapped depot idx
            depot_cell = (int(depot_row), int(depot_col))
            if depot_cell not in node_to_idx:
                continue
            depot_idx = node_to_idx[depot_cell]
            
            # Reconstruct unvisited target cells visited by this vehicle in baseline_routes
            # During training, we select 5 delivery nodes
            # To keep it deterministic and consistent with initial ACO evaluations,
            # we seed the generator based on vehicle_id name hash
            v_seed = int(v_id.split('_')[-1])
            v_rng = np.random.default_rng(seed=v_seed)
            
            available_nodes = [i for i in range(N) if i != depot_idx]
            if len(available_nodes) >= 5:
                delivery_indices = list(v_rng.choice(available_nodes, size=5, replace=False))
            else:
                delivery_indices = available_nodes
                
            # Build pairwise transition caches for these delivery nodes on the fly if needed
            aco_nodes = [depot_idx] + delivery_indices
            K = len(aco_nodes)
            
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    u_idx = aco_nodes[i]
                    v_idx = aco_nodes[j]
                    if (u_idx, v_idx) not in cost_cache:
                        path, dur, dist_m = dijkstra_pathfinder(u_idx, v_idx, adj_dict, N)
                        if path is not None:
                            cost_cache[(u_idx, v_idx)] = {
                                "path": path,
                                "duration_sec": dur,
                                "distance_m": dist_m
                            }
                        else:
                            cost_cache[(u_idx, v_idx)] = None
                            
            # Add metrics to cached entries
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    u_idx = aco_nodes[i]
                    v_idx = aco_nodes[j]
                    cache = cost_cache[(u_idx, v_idx)]
                    if cache is not None and "congestion" not in cache:
                        metrics = evaluate_route_metrics_fast(
                            cache["path"], edge_stats, congestion_map, sorted_nodes, vehicle_config, bbox, grid_dims
                        )
                        cache.update({
                            "congestion": metrics["avg_congestion"],
                            "fuel_l": metrics["fuel_l"],
                            "co2_g": metrics["co2_g"]
                        })
                        
            # Run $M$ ants to sample solutions
            ant_logs = []
            ant_costs = []
            
            for ant in range(num_ants):
                curr_node = depot_idx
                unvisited = set(delivery_indices)
                
                path_log_probs = []
                full_route_nodes = [curr_node]
                
                accum_dist_m = 0.0
                accum_dur_s = 0.0
                accum_fuel_l = 0.0
                accum_co2_g = 0.0
                congestions_visited = []
                unvisited_penalty = 0
                
                while unvisited:
                    candidates = list(unvisited)
                    valid_candidates = []
                    feature_vectors = []
                    
                    for cand in candidates:
                        cache = cost_cache.get((curr_node, cand))
                        if cache is None:
                            continue # Unreachable
                            
                        # Assemble 137d feature vector
                        feat = get_edge_feature_vector(curr_node, cand, embeddings_np, cost_cache, vehicle_config)
                        feature_vectors.append(feat)
                        valid_candidates.append(cand)
                        
                    if not valid_candidates:
                        # Ant is stuck
                        unvisited_penalty = len(unvisited)
                        break
                        
                    # Compute scores using our network
                    feat_tensor = torch.tensor(np.array(feature_vectors), dtype=torch.float)
                    scores = policy_net(feat_tensor).squeeze(-1) # [len(valid_candidates)]
                    
                    # Convert to probabilities using categorical distribution
                    # Prob proportional to score^beta
                    logits = beta_param * torch.log(scores + 1e-8)
                    probs = torch.softmax(logits, dim=0)
                    
                    # Sample next candidate target node
                    dist = torch.distributions.Categorical(probs=probs)
                    sampled_idx = dist.sample()
                    
                    selected_node = valid_candidates[int(sampled_idx.item())]
                    
                    # Store log-probability for gradient backpropagation
                    path_log_probs.append(dist.log_prob(sampled_idx))
                    
                    # Add path details
                    cache = cost_cache[(curr_node, selected_node)]
                    full_route_nodes.extend(cache["path"][1:])
                    
                    accum_dist_m += cache["distance_m"]
                    accum_dur_s += cache["duration_sec"]
                    accum_fuel_l += cache["fuel_l"]
                    accum_co2_g += cache["co2_g"]
                    congestions_visited.append(cache["congestion"])
                    
                    unvisited.remove(selected_node)
                    curr_node = selected_node
                    
                # Return to depot segment
                if curr_node != depot_idx:
                    cache = cost_cache.get((curr_node, depot_idx))
                    if cache is not None:
                        full_route_nodes.extend(cache["path"][1:])
                        accum_dist_m += cache["distance_m"]
                        accum_dur_s += cache["duration_sec"]
                        accum_fuel_l += cache["fuel_l"]
                        accum_co2_g += cache["co2_g"]
                        congestions_visited.append(cache["congestion"])
                    else:
                        unvisited_penalty += 1
                        
                # Compute multi-objective routing costs
                if not congestions_visited:
                    congestions_visited = [congestion_map.get(sorted_nodes[full_route_nodes[0]], 0.0)]
                mean_cong = float(np.mean(congestions_visited))
                
                # Metrics
                dist_km = accum_dist_m / 1000.0
                dur_sec = accum_dur_s
                fuel_l = accum_fuel_l
                co2_g = accum_co2_g
                
                # Cost criteria (same as in evaluation)
                cost = dist_km + dur_sec / 3600.0 + mean_cong * 10.0 + fuel_l * 5.0 + unvisited_penalty * 1000.0
                
                ant_costs.append(cost)
                ant_logs.append(path_log_probs)
                
            if not ant_costs:
                continue
                
            # REINFORCE policy gradient calculation
            rewards = -np.array(ant_costs)
            baseline = np.mean(rewards)
            
            epoch_rewards.append(baseline)
            
            # Loss = - E [ (reward - baseline) * log_prob ]
            for m in range(num_ants):
                log_probs_tensor = ant_logs[m]
                if not log_probs_tensor:
                    continue
                sum_log_prob = torch.stack(log_probs_tensor).sum()
                advantage = rewards[m] - baseline
                
                loss = -advantage * sum_log_prob
                epoch_losses.append(loss)
                
        if epoch_losses:
            total_loss = torch.stack(epoch_losses).mean()
            total_loss.backward()
            optimizer.step()
            
            loss_val = float(total_loss.item())
            mean_r = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            
            loss_history.append(loss_val)
            reward_history.append(mean_r)
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"  DeepACO Epoch {epoch:02d}/{epochs:02d} | Policy Loss: {loss_val:.6f} | Mean Reward: {mean_r:.4f}")
                
    return policy_net, loss_history, reward_history

def deep_aco_route_planner(depot_node_idx, delivery_node_indices, policy_net, embeddings, adj_dict, N, 
                           predictions_df, sorted_nodes, edges_df, vehicle_config, reward_weights=None):
    """
    Run the trained DeepACO policy network combined with classical ACO ant transitions.
    Outputs the optimized route coordinates and metrics.
    """
    policy_net.eval()
    
    embeddings_np = embeddings.numpy()
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    bbox = [39.5, 116.0, 40.3, 116.8]
    grid_dims = (30, 30)
    
    edge_stats = {}
    for _, edge in edges_df.iterrows():
        u = (int(edge["grid_row"]), int(edge["grid_col"]))
        v = (int(edge["next_row"]), int(edge["next_col"]))
        if u in node_to_idx and v in node_to_idx:
            edge_stats[(node_to_idx[u], node_to_idx[v])] = float(edge["avg_duration_sec"])
            
    congestion_map = {}
    if not predictions_df.empty:
        df_avg = predictions_df.groupby(["grid_row", "grid_col"])["predicted_congestion_level"].mean().reset_index()
        for _, row in df_avg.iterrows():
            congestion_map[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["predicted_congestion_level"])
            
    aco_nodes = [depot_node_idx] + list(delivery_node_indices)
    K = len(aco_nodes)
    node_to_pos = {node_idx: pos for pos, node_idx in enumerate(aco_nodes)}
    
    cost_cache = {}
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            u_idx = aco_nodes[i]
            v_idx = aco_nodes[j]
            path, dur, dist_m = dijkstra_pathfinder(u_idx, v_idx, adj_dict, N)
            if path is not None:
                metrics = evaluate_route_metrics_fast(
                    path, edge_stats, congestion_map, sorted_nodes, vehicle_config, bbox, grid_dims
                )
                cost_cache[(u_idx, v_idx)] = {
                    "path": path,
                    "duration_sec": dur,
                    "distance_m": dist_m,
                    "congestion": metrics["avg_congestion"],
                    "fuel_l": metrics["fuel_l"],
                    "co2_g": metrics["co2_g"]
                }
            else:
                cost_cache[(u_idx, v_idx)] = None
                
    # Precompute neural heuristics matrix H (K x K) using DeepACOHeuristicNet
    H = np.zeros((K, K), dtype=float)
    with torch.no_grad():
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                u_idx = aco_nodes[i]
                v_idx = aco_nodes[j]
                
                feat = get_edge_feature_vector(u_idx, v_idx, embeddings_np, cost_cache, vehicle_config)
                feat_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(0)
                score = float(policy_net(feat_tensor).item())
                H[i, j] = score
                
    # Classical ACO variables initialization
    Tau = np.ones((K, K), dtype=float)
    Tau_min, Tau_max = 0.05, 5.0
    
    alpha = 1.0 # pheromone weight
    beta = 2.0  # heuristic weight
    rho_global = 0.2
    rho_local = 0.1
    q0 = 0.7
    Q = 1.0
    
    best_overall_route = None
    best_overall_cost = float('inf')
    best_overall_metrics = {}
    best_overall_seq = []
    
    rng = np.random.default_rng(seed=42)
    
    # Run classical ACO iterations guided by learned heuristic matrix H
    for iteration in range(25):
        ant_routes = []
        ant_costs = []
        ant_seqs = []
        ant_metrics = []
        
        for ant in range(15):
            curr_node = depot_node_idx
            curr_pos = node_to_pos[curr_node]
            
            unvisited = set(delivery_node_indices)
            target_seq = [curr_node]
            full_path = [curr_node]
            
            accum_dist_m = 0.0
            accum_dur_s = 0.0
            accum_fuel_l = 0.0
            accum_co2_g = 0.0
            congestions_visited = []
            unvisited_penalty = 0
            
            while unvisited:
                candidates = list(unvisited)
                valid_candidates = []
                prob_weights = []
                
                for cand in candidates:
                    cache = cost_cache.get((curr_node, cand))
                    if cache is None:
                        continue
                    
                    cand_pos = node_to_pos[cand]
                    eta = H[curr_pos, cand_pos]
                    tau = Tau[curr_pos, cand_pos]
                    
                    weight = (tau ** alpha) * (eta ** beta)
                    
                    valid_candidates.append(cand)
                    prob_weights.append(weight)
                    
                if not valid_candidates:
                    unvisited_penalty = len(unvisited)
                    break
                    
                prob_weights = np.array(prob_weights)
                best_cand_idx = np.argmax(prob_weights)
                
                selected_node = None
                r = rng.uniform(0.0, 1.0)
                
                if r < q0:
                    selected_node = valid_candidates[best_cand_idx]
                else:
                    total_w = np.sum(prob_weights)
                    if total_w > 0:
                        probs = prob_weights / total_w
                        selected_node = rng.choice(valid_candidates, p=probs)
                    else:
                        selected_node = rng.choice(valid_candidates)
                        
                selected_pos = node_to_pos[selected_node]
                
                # Local pheromone update
                Tau[curr_pos, selected_pos] = (1.0 - rho_local) * Tau[curr_pos, selected_pos] + rho_local * 1.0
                Tau[curr_pos, selected_pos] = np.clip(Tau[curr_pos, selected_pos], Tau_min, Tau_max)
                
                cache = cost_cache[(curr_node, selected_node)]
                full_path.extend(cache["path"][1:])
                target_seq.append(selected_node)
                
                accum_dist_m += cache["distance_m"]
                accum_dur_s += cache["duration_sec"]
                accum_fuel_l += cache["fuel_l"]
                accum_co2_g += cache["co2_g"]
                congestions_visited.append(cache["congestion"])
                
                unvisited.remove(selected_node)
                curr_node = selected_node
                curr_pos = selected_pos
                
            # Return to depot segment
            if curr_node != depot_node_idx:
                cache = cost_cache.get((curr_node, depot_node_idx))
                if cache is not None:
                    full_path.extend(cache["path"][1:])
                    target_seq.append(depot_node_idx)
                    accum_dist_m += cache["distance_m"]
                    accum_dur_s += cache["duration_sec"]
                    accum_fuel_l += cache["fuel_l"]
                    accum_co2_g += cache["co2_g"]
                    congestions_visited.append(cache["congestion"])
                else:
                    unvisited_penalty += 1
                    
            if not congestions_visited:
                congestions_visited = [congestion_map.get(sorted_nodes[full_path[0]], 0.0)]
            mean_cong = float(np.mean(congestions_visited))
            
            # Metrics
            dist_km = accum_dist_m / 1000.0
            dur_sec = accum_dur_s
            fuel_l = accum_fuel_l
            co2_g = accum_co2_g
            
            # Cost criteria (same as in training)
            cost = dist_km + dur_sec / 3600.0 + mean_cong * 10.0 + fuel_l * 5.0 + unvisited_penalty * 1000.0
            
            metrics = {
                "distance_km": dist_km,
                "duration_sec": dur_sec,
                "avg_congestion": mean_cong,
                "fuel_l": fuel_l,
                "co2_g": co2_g
            }
            
            ant_routes.append(full_path)
            ant_costs.append(cost)
            ant_seqs.append(target_seq)
            ant_metrics.append(metrics)
            
        if not ant_costs:
            continue
            
        best_idx_iter = np.argmin(ant_costs)
        best_cost_iter = ant_costs[best_idx_iter]
        best_seq_iter = ant_seqs[best_idx_iter]
        
        # Global pheromone update (reinforce iteration best)
        Tau = (1.0 - rho_global) * Tau
        for i in range(len(best_seq_iter) - 1):
            u_pos = node_to_pos[best_seq_iter[i]]
            v_pos = node_to_pos[best_seq_iter[i+1]]
            Tau[u_pos, v_pos] += Q / best_cost_iter
        Tau = np.clip(Tau, Tau_min, Tau_max)
        
        if best_cost_iter < best_overall_cost:
            best_overall_cost = best_cost_iter
            best_overall_route = ant_routes[best_idx_iter]
            best_overall_metrics = ant_metrics[best_idx_iter]
            best_overall_seq = best_seq_iter
            
    route_cells = [sorted_nodes[idx] for idx in best_overall_route]
    
    # Reconstruct decision trace mapping standard structure
    decision_trace = []
    for i in range(len(best_overall_seq) - 1):
        u_idx = best_overall_seq[i]
        v_idx = best_overall_seq[i+1]
        cache = cost_cache.get((u_idx, v_idx))
        decision_trace.append({
            "from_node": int(u_idx),
            "selected_target": int(v_idx),
            "path_found": cache is not None,
            "path_nodes": [int(n) for n in cache["path"]] if cache is not None else []
        })
        
    return {
        "route_nodes": [int(n) for n in best_overall_route],
        "route_cells": [[int(c[0]), int(c[1])] for c in route_cells],
        "decision_trace": decision_trace,
        "unvisited_remaining": [int(n) for n in (set(delivery_node_indices) - set(best_overall_seq))],
        "total_duration_sec": float(best_overall_metrics["duration_sec"]),
        "total_distance_km": float(best_overall_metrics["distance_km"]),
        "avg_congestion": float(best_overall_metrics["avg_congestion"]),
        "fuel_l": float(best_overall_metrics["fuel_l"]),
        "co2_g": float(best_overall_metrics["co2_g"]),
        "objective_value": float(best_overall_cost)
    }
