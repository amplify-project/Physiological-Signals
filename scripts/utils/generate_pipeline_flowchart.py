"""
Generate a flowchart diagram for the Binary Engagement Classification Pipeline
Saves as PNG for insertion into deliverable documents
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Binary Engagement Classification Pipeline', 
            fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(7, 9.0, 'TUS Audience Pose - Real-time Engagement Estimation', 
            fontsize=11, ha='center', va='center', style='italic', color='gray')
    
    # Colors
    input_color = '#E3F2FD'      # Light blue
    process_color = '#FFF3E0'    # Light orange
    ai_color = '#E8F5E9'         # Light green
    output_color = '#FCE4EC'     # Light pink
    border_color = '#1976D2'     # Blue border
    
    # Box dimensions
    box_width = 2.8
    box_height = 1.2
    
    # Stage positions (x, y)
    positions = {
        'camera': (1.5, 7),
        'yolo': (5.5, 7),
        'mediapipe': (9.5, 7),
        'buffer': (12, 5),
        'transformer': (7, 3.5),
        'aggregation': (3, 3.5),
        'redis': (3, 1.5),
        'ar': (7, 1.5),
        'display': (11, 1.5),
    }
    
    def draw_box(x, y, width, height, text, subtext, color, text_size=10):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='#333333', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.15, text, fontsize=text_size, fontweight='bold', 
                ha='center', va='center')
        if subtext:
            ax.text(x, y - 0.25, subtext, fontsize=8, ha='center', va='center', 
                    color='#555555')
    
    def draw_arrow(start, end, color='#333333'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Draw boxes
    # Input
    draw_box(*positions['camera'], 2.2, 1.0, '📹 Camera', 'RGB Video Feed\n(720p/1080p)', input_color)
    
    # Stage 1
    draw_box(*positions['yolo'], box_width, box_height, 'Stage 1: Person Detection', 
             'YOLO26n\n(up to ~20 people)', process_color)
    
    # Stage 2
    draw_box(*positions['mediapipe'], box_width, box_height, 'Stage 2: Pose Estimation', 
             'MediaPipe Holistic\n(543 keypoints × 3D)', process_color)
    
    # Buffer
    draw_box(*positions['buffer'], 2.2, 1.0, '⏱️ Temporal Buffer', 
             '64 frames\n(~2 seconds)', '#FFF9C4')
    
    # Stage 3
    draw_box(*positions['transformer'], box_width + 0.5, box_height + 0.3, 
             'Stage 3: Engagement Classification', 
             'Temporal Transformer\n(4 layers, 8 heads, 256d)\n82.75% accuracy', ai_color)
    
    # Stage 4
    draw_box(*positions['aggregation'], box_width, box_height, 'Stage 4: Aggregation', 
             'Crowd Average\n(mean of all persons)', process_color)
    
    # Outputs
    draw_box(*positions['redis'], 2.4, 1.0, '📡 Redis Pub/Sub', 
             'Channel: engagement_score\n1 Hz', output_color)
    draw_box(*positions['ar'], 2.4, 1.0, '🥽 AR Application', 
             'Unity Subscriber\n(0.0 - 1.0)', output_color)
    draw_box(*positions['display'], 2.4, 1.0, '🖥️ Visual Display', 
             'Red→Green Bar\nBounding Boxes', output_color)
    
    # Draw arrows
    # Main flow
    draw_arrow((2.6, 7), (4.0, 7))  # Camera → YOLO
    draw_arrow((6.9, 7), (8.0, 7))  # YOLO → MediaPipe
    draw_arrow((10.9, 7), (11.8, 5.6))  # MediaPipe → Buffer (angled)
    draw_arrow((12, 4.5), (9.0, 3.8))  # Buffer → Transformer (angled)
    draw_arrow((5.3, 3.5), (4.5, 3.5))  # Transformer → Aggregation
    draw_arrow((3, 2.9), (3, 2.0))  # Aggregation → Redis
    draw_arrow((4.2, 1.5), (5.8, 1.5))  # Redis → AR
    draw_arrow((8.2, 1.5), (9.8, 1.5))  # AR → Display
    
    # Per-person loop indicator
    ax.annotate('', xy=(9.5, 5.5), xytext=(5.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5, 
                               connectionstyle='arc3,rad=-0.3', linestyle='--'))
    ax.text(7.5, 5.2, 'Per person', fontsize=8, ha='center', color='#666666', style='italic')
    
    # Legend
    legend_y = 0.3
    legend_items = [
        (1.5, legend_y, input_color, 'Input'),
        (3.5, legend_y, process_color, 'Processing'),
        (5.5, legend_y, ai_color, 'AI Model'),
        (7.5, legend_y, output_color, 'Output'),
    ]
    for x, y, color, label in legend_items:
        box = FancyBboxPatch((x - 0.3, y - 0.15), 0.6, 0.3,
                             boxstyle="round,pad=0.02", facecolor=color, 
                             edgecolor='#333333', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.5, y, label, fontsize=9, ha='left', va='center')
    
    # Data annotations
    ax.text(3.3, 7.5, 'Video\nFrames', fontsize=7, ha='center', color='#666666')
    ax.text(7.5, 7.5, 'Bounding\nBoxes', fontsize=7, ha='center', color='#666666')
    ax.text(11.5, 6.3, '543×3\nKeypoints', fontsize=7, ha='center', color='#666666')
    ax.text(10.5, 4.5, '64 frames\nper person', fontsize=7, ha='center', color='#666666')
    ax.text(4.9, 3.9, '0.0-1.0\nscore', fontsize=7, ha='center', color='#666666')
    
    plt.tight_layout()
    
    # Save
    output_path = 'docs/Amplify Delivarables/D3.2/pipeline_flowchart.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ Flowchart saved to: {output_path}")
    
    # Also save a higher resolution version
    output_path_hires = 'docs/Amplify Delivarables/D3.2/pipeline_flowchart_hires.png'
    plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ High-res version saved to: {output_path_hires}")
    
    plt.show()
    
if __name__ == '__main__':
    create_pipeline_flowchart()
