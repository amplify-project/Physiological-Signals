#!/usr/bin/env python3
"""
Visualize the Binary Action Transformer architecture.

Creates a detailed diagram showing:
- Input layer (1629 features from MediaPipe Holistic)
- Embedding layer (1629 → 128)
- Positional encoding
- 2 Transformer encoder layers (4 attention heads each)
- Classification head (128 → 2)
- Parameter counts per layer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Calculate parameter counts
def calculate_params():
    """Calculate parameters for each layer."""
    params = {}
    
    # Embedding layer: Linear(1629, 128)
    params['embedding'] = 1629 * 128 + 128  # weights + bias
    
    # Positional encoding: No trainable parameters (fixed sinusoidal)
    params['pos_encoding'] = 0
    
    # Transformer Encoder Layer (×2)
    d_model = 128
    nhead = 4
    dim_feedforward = d_model * 4  # 512
    
    # Multi-head attention
    # Q, K, V projections: 3 × (d_model × d_model + d_model)
    params['mha_qkv'] = 3 * (d_model * d_model + d_model)
    # Output projection
    params['mha_out'] = d_model * d_model + d_model
    params['mha_total'] = params['mha_qkv'] + params['mha_out']
    
    # Layer norm 1
    params['ln1'] = 2 * d_model  # gamma + beta
    
    # Feed-forward network
    params['ffn_fc1'] = d_model * dim_feedforward + dim_feedforward
    params['ffn_fc2'] = dim_feedforward * d_model + d_model
    params['ffn_total'] = params['ffn_fc1'] + params['ffn_fc2']
    
    # Layer norm 2
    params['ln2'] = 2 * d_model
    
    # Total per encoder layer
    params['encoder_layer'] = params['mha_total'] + params['ln1'] + params['ffn_total'] + params['ln2']
    
    # Total for 2 layers
    params['transformer_total'] = 2 * params['encoder_layer']
    
    # Classification head: Linear(128, 2)
    params['classifier'] = 128 * 2 + 2
    
    # Grand total
    params['total'] = params['embedding'] + params['transformer_total'] + params['classifier']
    
    return params

def create_architecture_diagram():
    """Create detailed architecture visualization."""
    
    params = calculate_params()
    
    fig, ax = plt.subplots(figsize=(16, 28))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 32)
    ax.axis('off')
    
    # Title
    ax.text(5, 31, 'Binary Action Transformer Architecture', 
            ha='center', va='top', fontsize=22, fontweight='bold')
    ax.text(5, 30, 'Engagement vs Disengagement Classification', 
            ha='center', va='top', fontsize=14, style='italic', color='gray')
    
    y_pos = 28.5
    
    # Helper function to draw layer box
    def draw_layer(y, width, height, label, params_text, color, details=None):
        box = FancyBboxPatch((5-width/2, y-height), width, height,
                            boxstyle="round,pad=0.15", 
                            edgecolor='black', facecolor=color, linewidth=2.5)
        ax.add_patch(box)
        
        # Main label
        text_y = y - height/2
        if details:
            # If details exist, stack: label, params, details
            ax.text(5, text_y + 0.25, label, ha='center', va='center', 
                   fontsize=13, fontweight='bold')
            ax.text(5, text_y - 0.05, params_text, ha='center', va='center', 
                   fontsize=11, style='italic', color='darkblue')
            ax.text(5, text_y - 0.35, details, ha='center', va='center', 
                   fontsize=10, color='gray')
        else:
            # If no details, just label and params
            ax.text(5, text_y + 0.15, label, ha='center', va='center', 
                   fontsize=13, fontweight='bold')
            ax.text(5, text_y - 0.15, params_text, ha='center', va='center', 
                   fontsize=11, style='italic', color='darkblue')
        
        return y - height
    
    # Helper function to draw arrow
    def draw_arrow(y_start, y_end):
        arrow = FancyArrowPatch((5, y_start), (5, y_end),
                               arrowstyle='->', mutation_scale=25, 
                               linewidth=2.5, color='black')
        ax.add_patch(arrow)
    
    # INPUT LAYER
    y_pos = draw_layer(y_pos, 4.0, 1.3, 'Input Sequence',
                      'Shape: (batch, 300, 1629)',
                      '#E8F4F8',
                      '300 frames × 543 landmarks × 3 coords')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # EMBEDDING LAYER
    y_pos = draw_layer(y_pos, 4.0, 1.3, 'Linear Embedding',
                      f'{params["embedding"]:,} params',
                      '#D4E6F1',
                      '1629 → 128 (per frame)')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # POSITIONAL ENCODING
    y_pos = draw_layer(y_pos, 4.0, 1.1, 'Positional Encoding',
                      '0 params (fixed sinusoidal)',
                      '#AED6F1',
                      'Adds temporal position info')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.2
    
    # TRANSFORMER ENCODER LAYER 1
    ax.text(5, y_pos-0.3, 'Transformer Encoder Layer 1', 
           ha='center', va='top', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    y_pos -= 1.0
    
    # Multi-Head Attention
    y_pos = draw_layer(y_pos, 3.5, 1.5, 'Multi-Head Self-Attention',
                      f'{params["mha_total"]:,} params',
                      '#FFE6CC',
                      '4 heads × 32d per head')
    
    # Draw attention heads
    head_width = 0.6
    head_spacing = 0.8
    head_y = y_pos - 0.4
    for i in range(4):
        head_x = 5 - 1.7 + i * head_spacing
        head_box = FancyBboxPatch((head_x-head_width/2, head_y-0.35), 
                                 head_width, 0.35,
                                 boxstyle="round,pad=0.03",
                                 edgecolor='orange', facecolor='#FFEBCD', 
                                 linewidth=1.5)
        ax.add_patch(head_box)
        ax.text(head_x, head_y-0.175, f'H{i+1}', ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # Add & Norm 1
    y_pos = draw_layer(y_pos, 2.5, 0.8, 'Add & LayerNorm',
                      f'{params["ln1"]:,} params',
                      '#FFE6CC')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # Feed-Forward Network
    y_pos = draw_layer(y_pos, 3.5, 1.2, 'Feed-Forward Network',
                      f'{params["ffn_total"]:,} params',
                      '#FFE6CC',
                      '128 → 512 → 128')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # Add & Norm 2
    y_pos = draw_layer(y_pos, 2.5, 0.8, 'Add & LayerNorm',
                      f'{params["ln2"]:,} params',
                      '#FFE6CC')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.2
    
    # TRANSFORMER ENCODER LAYER 2
    ax.text(5, y_pos-0.3, 'Transformer Encoder Layer 2 (identical)', 
           ha='center', va='top', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    y_pos -= 1.0
    
    # Just show simplified version for layer 2
    y_pos = draw_layer(y_pos, 3.5, 1.8, 'Multi-Head Attention + FFN',
                      f'{params["encoder_layer"]:,} params',
                      '#FFE6CC',
                      'Same structure as Layer 1')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # GLOBAL AVERAGE POOLING
    y_pos = draw_layer(y_pos, 3.5, 1.1, 'Global Average Pooling',
                      '0 params',
                      '#E8DAEF',
                      '(batch, 300, 128) → (batch, 128)')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # DROPOUT
    y_pos = draw_layer(y_pos, 2.8, 0.8, 'Dropout (p=0.3)',
                      '0 params',
                      '#E8DAEF')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # CLASSIFICATION HEAD
    y_pos = draw_layer(y_pos, 3.5, 1.2, 'Classification Head',
                      f'{params["classifier"]:,} params',
                      '#D5F4E6',
                      'Linear: 128 → 2')
    draw_arrow(y_pos, y_pos-0.8)
    y_pos -= 1.0
    
    # OUTPUT
    y_pos = draw_layer(y_pos, 3.5, 1.1, 'Output (Softmax)',
                      'Shape: (batch, 2)',
                      '#ABEBC6',
                      '[Engagement, Disengagement]')
    
    # Summary box
    summary_y = 2.5
    summary_box = FancyBboxPatch((0.5, summary_y-2.0), 9, 1.9,
                                boxstyle="round,pad=0.15",
                                edgecolor='darkblue', facecolor='#F0F8FF',
                                linewidth=2.5, linestyle='--')
    ax.add_patch(summary_box)
    
    ax.text(5, summary_y-0.4, 'Model Summary', ha='center', va='top',
           fontsize=15, fontweight='bold')
    ax.text(5, summary_y-0.85, f'Total Parameters: {params["total"]:,} (~{params["total"]/1e6:.2f}M)',
           ha='center', va='top', fontsize=13)
    ax.text(5, summary_y-1.25, 
           f'Embedding: {params["embedding"]:,} | Transformer: {params["transformer_total"]:,} | Classifier: {params["classifier"]:,}',
           ha='center', va='top', fontsize=11, color='gray')
    ax.text(5, summary_y-1.65,
           'Architecture: 2 layers × 4 heads × 128d',
           ha='center', va='top', fontsize=10, style='italic', color='gray')
    ax.text(5, summary_y-1.90,
           'Regularization: Dropout 0.3, Weight Decay 0.01, Label Smoothing 0.1',
           ha='center', va='top', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    return fig, params

if __name__ == '__main__':
    print("Creating Binary Action Transformer architecture diagram...")
    fig, params = create_architecture_diagram()
    
    output_path = 'docs/binary_transformer_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Architecture diagram saved to: {output_path}")
    
    # Print parameter breakdown
    print("\n" + "="*60)
    print("PARAMETER BREAKDOWN")
    print("="*60)
    print(f"Embedding Layer:           {params['embedding']:>10,} params")
    print(f"Positional Encoding:       {params['pos_encoding']:>10,} params (fixed)")
    print(f"Transformer Encoder (×2):  {params['transformer_total']:>10,} params")
    print(f"  - Multi-Head Attention:  {params['mha_total']:>10,} params/layer")
    print(f"  - Feed-Forward Network:  {params['ffn_total']:>10,} params/layer")
    print(f"  - Layer Norms (×2):      {params['ln1'] + params['ln2']:>10,} params/layer")
    print(f"Classification Head:       {params['classifier']:>10,} params")
    print("-"*60)
    print(f"TOTAL PARAMETERS:          {params['total']:>10,} (~{params['total']/1e6:.2f}M)")
    print("="*60)
    
    # Show the plot
    plt.show()
