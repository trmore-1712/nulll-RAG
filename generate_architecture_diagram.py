import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#e0f7fa', fontsize=15):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.01, y-0.01), width, height, boxstyle="round,pad=0.02", 
                                   linewidth=0, facecolor='gray', alpha=0.3)
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", 
                                linewidth=1.5, edgecolor='#006064', facecolor=color)
    ax.add_patch(box)
    # Text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#006064')
    return x+width, y+height/2

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2.0, color='#455a64')) # Thicker arrow

def generate_diagram():
    fig, ax = plt.subplots(figsize=(14, 7)) # Larger figure
    ax.set_xlim(0, 13) # Expanded width
    ax.set_ylim(0, 7) # Expanded height
    ax.axis('off')

    # Phase 1: Indexing
    ax.text(2.0, 6.2, "Phase 1: Structure-Aware Indexing", fontsize=18, fontweight='bold', color='#1a237e')
    
    draw_box(ax, 0.5, 4.5, 2.0, 1.0, "PDF/JSON\nDocuments", fontsize=13)
    draw_arrow(ax, 2.5, 5.0, 3.0, 5.0)
    
    draw_box(ax, 3.0, 4.5, 2.5, 1.0, "Graph Builder\n(Nodes & Edges)", color='#c5cae9', fontsize=13)
    draw_arrow(ax, 5.5, 5.0, 6.0, 5.0)
    
    draw_box(ax, 6.0, 4.5, 2.5, 1.0, "Table-Text Graph\n(Heterogeneous)", color='#c5cae9', fontsize=13)

    # Phase 2: Retrieval
    ax.text(2.0, 3.5, "Phase 2: Hierarchical Retrieval", fontsize=18, fontweight='bold', color='#1a237e')
    
    draw_box(ax, 0.5, 2.0, 2.0, 1.0, "User Query", color='#fff9c4', fontsize=13)
    draw_arrow(ax, 2.5, 2.5, 3.0, 2.5)
    
    draw_box(ax, 3.0, 2.0, 2.5, 1.0, "Hierarchical Search\n(Section -> Para/Table)", color='#ffcc80', fontsize=13)
    draw_arrow(ax, 5.5, 2.5, 6.0, 2.5)
    
    draw_box(ax, 6.0, 2.0, 2.5, 1.0, "TTGNN Reranking\n(Structure Aware)", color='#ffcc80', fontsize=13)
    draw_arrow(ax, 8.5, 2.5, 9.0, 2.5)

    # Phase 3: Fusion
    ax.text(9.0, 6.2, "Phase 3: Reasoning", fontsize=18, fontweight='bold', color='#1a237e')
    
    # Connecting Arrows across phases
    draw_arrow(ax, 7.25, 4.5, 7.25, 3.0) # Graph to TTGNN (vertical)
    
    draw_box(ax, 9.0, 2.0, 2.5, 2.5, "Symbolic-Neural\nFusion Router", color='#dcedc8', fontsize=15)
    
    # Explicit Outputs from Router
    draw_arrow(ax, 11.5, 3.25, 12.0, 3.25)
    
    # Final Output Box
    draw_box(ax, 12.0, 2.75, 2.5, 1.0, "Final Answer\n(Grounded & Correct)", color='#b2dfdb', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('Fig0_Architecture.png', dpi=300, bbox_inches='tight')
    print("Diagram generated: Fig0_Architecture.png")

if __name__ == "__main__":
    generate_diagram()
