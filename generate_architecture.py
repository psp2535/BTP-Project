import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Define layer boundaries
layer1 = patches.Rectangle((0.05, 0.05), 0.25, 0.9, linewidth=1, edgecolor='#d0d0d0', facecolor='#f8f9fa', linestyle='--', zorder=0)
layer2 = patches.Rectangle((0.35, 0.05), 0.3, 0.9, linewidth=1, edgecolor='#d0d0d0', facecolor='#f8f9fa', linestyle='--', zorder=0)
layer3 = patches.Rectangle((0.7, 0.05), 0.25, 0.9, linewidth=1, edgecolor='#d0d0d0', facecolor='#f8f9fa', linestyle='--', zorder=0)

ax.add_patch(layer1)
ax.add_patch(layer2)
ax.add_patch(layer3)

# Layer Titles
plt.text(0.175, 0.92, "Data Ingestion Layer", ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
plt.text(0.5, 0.92, "Predictive Modeling Layer", ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
plt.text(0.825, 0.92, "Optimization Layer", ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

# Define helper for drawing boxes
def draw_box(x, y, width, height, text, facecolor, edgecolor):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
                                 linewidth=2, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    plt.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12, fontweight='bold', color='white', zorder=3)

# Draw Nodes
# Layer 1
draw_box(0.08, 0.65, 0.19, 0.15, "Raw GPS Trajectories\n(Microsoft T-Drive)", "#2C3E50", "#1A252F")
draw_box(0.08, 0.35, 0.19, 0.15, "Data Processing &\nKinematics", "#34495E", "#2C3E50")

# Layer 2
draw_box(0.4, 0.65, 0.2, 0.15, "GNN Spatial Embedding\n(64-dim Topology)", "#2980B9", "#1F618D")
draw_box(0.4, 0.35, 0.2, 0.15, "TCFMu Congestion\nPredictor (XGBoost)", "#27AE60", "#1E8449")

# Layer 3
draw_box(0.73, 0.5, 0.19, 0.15, "DeepACO Routing\nEngine (REINFORCE)", "#8E44AD", "#6C3483")
draw_box(0.73, 0.2, 0.19, 0.12, "Optimal Route\nDispatch", "#E67E22", "#AF601A")

# Draw Arrows
arrow_props = dict(facecolor='#555555', edgecolor='#555555', shrinkA=0, shrinkB=0, width=2, headwidth=10, zorder=1)
curved_arrow1 = dict(facecolor='#555555', edgecolor='#555555', shrinkA=0, shrinkB=0, width=2, headwidth=10, connectionstyle="arc3,rad=-0.2", zorder=1)
curved_arrow2 = dict(facecolor='#555555', edgecolor='#555555', shrinkA=0, shrinkB=0, width=2, headwidth=10, connectionstyle="arc3,rad=0.2", zorder=1)

# L1 -> L1
plt.annotate("", xy=(0.175, 0.5), xytext=(0.175, 0.65), arrowprops=arrow_props)

# L1 -> L2
plt.annotate("", xy=(0.4, 0.725), xytext=(0.27, 0.425), arrowprops=curved_arrow1)
plt.annotate("", xy=(0.4, 0.425), xytext=(0.27, 0.425), arrowprops=arrow_props)

# L2 -> L3
plt.annotate("", xy=(0.73, 0.575), xytext=(0.6, 0.725), arrowprops=curved_arrow2)
plt.annotate("", xy=(0.73, 0.575), xytext=(0.6, 0.425), arrowprops=curved_arrow1)

# L3 -> L3
plt.annotate("", xy=(0.825, 0.32), xytext=(0.825, 0.5), arrowprops=arrow_props)

ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.savefig('placeholder_architecture.png', dpi=300, bbox_inches='tight')
print("Saved architecture diagram to placeholder_architecture.png")
