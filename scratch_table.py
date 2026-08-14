import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/deep_aco_comparison.csv')
greedy = df[['greedy_dist', 'greedy_time', 'greedy_fuel', 'greedy_co2', 'greedy_cong']].mean()
deepaco = df[['deep_aco_dist', 'deep_aco_time', 'deep_aco_fuel', 'deep_aco_co2', 'deep_aco_cong']].mean()

dist_red = (greedy.greedy_dist - deepaco.deep_aco_dist) / greedy.greedy_dist * 100
time_red = (greedy.greedy_time - deepaco.deep_aco_time) / greedy.greedy_time * 100
cong_red = (greedy.greedy_cong - deepaco.deep_aco_cong) / greedy.greedy_cong * 100
fuel_red = (greedy.greedy_fuel - deepaco.deep_aco_fuel) / greedy.greedy_fuel * 100
co2_red = (greedy.greedy_co2 - deepaco.deep_aco_co2) / greedy.greedy_co2 * 100

data = [
    ["Distance Reduction", f"{dist_red:.2f}%"],
    ["Duration Reduction", f"{time_red:.2f}%"],
    ["Congestion Reduction", f"{cong_red:.2f}%"],
    ["Fuel Reduction", f"{fuel_red:.2f}%"],
    ["CO₂ Reduction", f"{co2_red:.2f}%"]
]

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

ax.axis('tight')
ax.axis('off')

# Title
plt.text(0.02, 1.1, "Improvement of DeepACO over Greedy", color='white', fontsize=18, fontweight='bold', transform=ax.transAxes)
plt.text(0.02, 1.0, "Metric", color='#d3d3d3', fontsize=14, fontweight='bold', transform=ax.transAxes)
plt.text(0.85, 1.0, "Improvement (%)", color='#d3d3d3', fontsize=14, fontweight='bold', transform=ax.transAxes)

# Horizontal Line
ax.plot([0.02, 0.98], [0.95, 0.95], color='#333333', lw=1, transform=ax.transAxes)

y_pos = 0.8
for row in data:
    plt.text(0.02, y_pos, row[0], color='white', fontsize=12, fontweight='bold', transform=ax.transAxes)
    plt.text(0.95, y_pos, row[1], color='white', fontsize=12, horizontalalignment='right', transform=ax.transAxes)
    
    # Separator line
    ax.plot([0.02, 0.98], [y_pos - 0.08, y_pos - 0.08], color='#222222', lw=1, transform=ax.transAxes)
    
    y_pos -= 0.18

plt.savefig('outputs/deepaco_improvement_table.png', facecolor='black', bbox_inches='tight', dpi=300)
print("Image saved to outputs/deepaco_improvement_table.png")
