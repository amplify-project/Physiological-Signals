#!/usr/bin/env python3
"""
Interactive 3D Transformer Network Visualization

Creates an interactive 3D model showing:
- All nodes (neurons) in the transformer architecture
- Connection weights between layers
- Attention head structure
- Weight magnitudes as edge colors/thickness
- Interactive: rotate, pan, zoom, hover for details

Usage:
    python visualize_model_3d.py --model-path models/action_transformer_binary/best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path
import json


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ActionTransformer(nn.Module):
    def __init__(self, input_dim=1629, d_model=256, nhead=8, num_layers=4, num_classes=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


def extract_layer_weights(model):
    """Extract weights from each layer for visualization."""
    weights = {}
    
    # Input projection layer
    weights['input_proj'] = {
        'weight': model.input_proj.weight.detach().cpu().numpy(),
        'bias': model.input_proj.bias.detach().cpu().numpy(),
        'shape': model.input_proj.weight.shape,
        'name': 'Input Projection (1629→256)'
    }
    
    # Transformer layers
    for i, layer in enumerate(model.transformer.layers):
        layer_weights = {}
        
        # Multi-head attention
        mha = layer.self_attn
        layer_weights['attention_q'] = mha.in_proj_weight[:256].detach().cpu().numpy()
        layer_weights['attention_k'] = mha.in_proj_weight[256:512].detach().cpu().numpy()
        layer_weights['attention_v'] = mha.in_proj_weight[512:].detach().cpu().numpy()
        layer_weights['attention_out'] = mha.out_proj.weight.detach().cpu().numpy()
        
        # Feed-forward
        layer_weights['ffn_fc1'] = layer.linear1.weight.detach().cpu().numpy()
        layer_weights['ffn_fc2'] = layer.linear2.weight.detach().cpu().numpy()
        
        weights[f'transformer_layer_{i+1}'] = layer_weights
    
    # Classification head (Sequential: Linear -> ReLU -> Dropout -> Linear)
    weights['classifier'] = {
        'fc1': model.classifier[0].weight.detach().cpu().numpy(),  # First linear layer (256→128)
        'bias1': model.classifier[0].bias.detach().cpu().numpy(),
        'fc2': model.classifier[3].weight.detach().cpu().numpy(),  # Output linear layer (128→2)
        'bias2': model.classifier[3].bias.detach().cpu().numpy(),
        'shape1': model.classifier[0].weight.shape,
        'shape2': model.classifier[3].weight.shape,
        'name': 'Classifier Layers'
    }
    
    return weights


def create_layer_nodes(layer_size, layer_name, x_position, color):
    """Create nodes for a single layer in 3D space (left-to-right flow)."""
    # Arrange nodes in a grid pattern for visualization
    grid_size = int(np.ceil(np.sqrt(layer_size)))
    
    x_coords = []
    y_coords = []
    z_coords = []
    node_names = []
    
    for i in range(layer_size):
        x = x_position  # Horizontal position (left-to-right progression)
        y = (i % grid_size) - grid_size / 2  # Vertical spread
        z = (i // grid_size) - grid_size / 2  # Depth spread
        
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        node_names.append(f"{layer_name}_node_{i}")
    
    return {
        'x': x_coords,
        'y': y_coords,
        'z': z_coords,
        'names': node_names,
        'color': color,
        'layer_name': layer_name
    }


def create_connections(source_nodes, target_nodes, weights, max_connections=500, threshold_percentile=90):
    """
    Create connections between layers based on weight magnitude.
    Returns traces with varying thickness and opacity based on weight strength.
    """
    # Flatten weights if needed
    if len(weights.shape) > 2:
        weights = weights.reshape(weights.shape[0], -1)
    
    # Get weight magnitudes
    weight_magnitudes = np.abs(weights)
    
    # Only keep top connections to avoid visual clutter
    threshold = np.percentile(weight_magnitudes, threshold_percentile)
    strong_connections = np.where(weight_magnitudes >= threshold)
    
    # Limit total number of connections
    if len(strong_connections[0]) > max_connections:
        indices = np.random.choice(len(strong_connections[0]), max_connections, replace=False)
        strong_connections = (strong_connections[0][indices], strong_connections[1][indices])
    
    # Store connections with their weights for grouping by strength
    connections = []
    
    for src_idx, tgt_idx in zip(strong_connections[0], strong_connections[1]):
        # Safely get indices with bounds checking
        if src_idx < len(source_nodes['x']) and tgt_idx < len(target_nodes['x']):
            weight_val = weight_magnitudes[src_idx, tgt_idx]
            connections.append({
                'src_x': source_nodes['x'][src_idx],
                'src_y': source_nodes['y'][src_idx],
                'src_z': source_nodes['z'][src_idx],
                'tgt_x': target_nodes['x'][tgt_idx],
                'tgt_y': target_nodes['y'][tgt_idx],
                'tgt_z': target_nodes['z'][tgt_idx],
                'weight': weight_val
            })
    
    # Group connections into strength tiers for better visualization
    if len(connections) == 0:
        empty = {'x': [], 'y': [], 'z': [], 'color': 'rgba(0,0,0,0)', 'width': 1}
        return {
            'strong': empty,
            'medium': empty,
            'weak': empty
        }
    
    weights_array = np.array([c['weight'] for c in connections])
    max_weight = weights_array.max()
    min_weight = weights_array.min()
    
    # Create separate trace data for different weight tiers
    traces = []
    
    # Strong connections (top 10%)
    strong_threshold = np.percentile(weights_array, 90)
    strong_conns = [c for c in connections if c['weight'] >= strong_threshold]
    
    # Medium connections (50-90%)
    medium_threshold = np.percentile(weights_array, 50)
    medium_conns = [c for c in connections if medium_threshold <= c['weight'] < strong_threshold]
    
    # Weak connections (bottom 50%)
    weak_conns = [c for c in connections if c['weight'] < medium_threshold]
    
    def make_edges(conns, width, opacity, color_base):
        edge_x, edge_y, edge_z = [], [], []
        for c in conns:
            edge_x.extend([c['src_x'], c['tgt_x'], None])
            edge_y.extend([c['src_y'], c['tgt_y'], None])
            edge_z.extend([c['src_z'], c['tgt_z'], None])
        return {
            'x': edge_x,
            'y': edge_y,
            'z': edge_z,
            'color': f'rgba({color_base}, {opacity})',
            'width': width
        }
    
    # Return multiple edge sets with different styling
    return {
        'strong': make_edges(strong_conns, 3, 0.8, '255, 50, 50'),
        'medium': make_edges(medium_conns, 2, 0.4, '100, 100, 255'),
        'weak': make_edges(weak_conns, 1, 0.2, '150, 150, 150')
    }



def create_3d_network_visualization(model_path, output_path='docs/model_3d_network.html'):
    """Create interactive 3D network visualization."""
    
    print("Loading trained model...")
    device = torch.device('cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ActionTransformer(input_dim=1629, d_model=256, nhead=8, num_layers=4, num_classes=2, dropout=0.1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Extracting layer weights...")
    weights = extract_layer_weights(model)
    
    print("Creating 3D node positions...")
    
    # Define layer structure
    layers = [
        {'name': 'Input', 'size': 1629, 'x': 0, 'color': 'lightblue'},
        {'name': 'Input Projection', 'size': 256, 'x': 10, 'color': 'lightgreen'},
        {'name': 'Transformer L1 (MHA)', 'size': 256, 'x': 20, 'color': 'orange'},
        {'name': 'Transformer L1 (FFN)', 'size': 1024, 'x': 25, 'color': 'coral'},
        {'name': 'Transformer L1 (Out)', 'size': 256, 'x': 30, 'color': 'orange'},
        {'name': 'Transformer L2 (MHA)', 'size': 256, 'x': 40, 'color': 'orange'},
        {'name': 'Transformer L2 (FFN)', 'size': 1024, 'x': 45, 'color': 'coral'},
        {'name': 'Transformer L2 (Out)', 'size': 256, 'x': 50, 'color': 'orange'},
        {'name': 'Transformer L3 (MHA)', 'size': 256, 'x': 60, 'color': 'orange'},
        {'name': 'Transformer L3 (FFN)', 'size': 1024, 'x': 65, 'color': 'coral'},
        {'name': 'Transformer L3 (Out)', 'size': 256, 'x': 70, 'color': 'orange'},
        {'name': 'Transformer L4 (MHA)', 'size': 256, 'x': 80, 'color': 'orange'},
        {'name': 'Transformer L4 (FFN)', 'size': 1024, 'x': 85, 'color': 'coral'},
        {'name': 'Transformer L4 (Out)', 'size': 256, 'x': 90, 'color': 'orange'},
        {'name': 'Pooling', 'size': 256, 'x': 100, 'color': 'plum'},
        {'name': 'Classifier L1', 'size': 128, 'x': 110, 'color': 'yellow'},
        {'name': 'Classifier L2', 'size': 2, 'x': 120, 'color': 'lightcoral'}
    ]
    
    # Create nodes for each layer
    all_nodes = []
    for layer in layers:
        nodes = create_layer_nodes(layer['size'], layer['name'], layer['x'], layer['color'])
        all_nodes.append(nodes)
    
    print("Creating 3D visualization...")
    
    # Create figure
    fig = go.Figure()
    
    # Add nodes for each layer
    for nodes in all_nodes:
        fig.add_trace(go.Scatter3d(
            x=nodes['x'],
            y=nodes['y'],
            z=nodes['z'],
            mode='markers',
            name=nodes['layer_name'],
            marker=dict(
                size=3,
                color=nodes['color'],
                line=dict(color='black', width=0.5)
            ),
            text=nodes['names'],
            hovertemplate='%{text}<br>Layer: ' + nodes['layer_name'] + '<extra></extra>'
        ))
    
    # Helper function to add all three tiers of connections
    def add_connection_tiers(connections, name_prefix):
        """Add strong, medium, and weak connection tiers to the figure."""
        if connections['strong']['x']:
            fig.add_trace(go.Scatter3d(
                x=connections['strong']['x'], 
                y=connections['strong']['y'], 
                z=connections['strong']['z'],
                mode='lines',
                name=f'{name_prefix} (Strong)',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=3),
                hoverinfo='skip',
                showlegend=False
            ))
        
        if connections['medium']['x']:
            fig.add_trace(go.Scatter3d(
                x=connections['medium']['x'], 
                y=connections['medium']['y'], 
                z=connections['medium']['z'],
                mode='lines',
                name=f'{name_prefix} (Medium)',
                line=dict(color='rgba(0, 0, 255, 0.4)', width=2),
                hoverinfo='skip',
                showlegend=False
            ))
        
        if connections['weak']['x']:
            fig.add_trace(go.Scatter3d(
                x=connections['weak']['x'], 
                y=connections['weak']['y'], 
                z=connections['weak']['z'],
                mode='lines',
                name=f'{name_prefix} (Weak)',
                line=dict(color='rgba(128, 128, 128, 0.2)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Add connections between ALL layers
    print("Creating weighted connections...")
    
    # Input → Input Projection (1629 → 256)
    connections = create_connections(
        all_nodes[0], all_nodes[1],
        weights['input_proj']['weight'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'Input→Proj')
    
    # Input Projection → Transformer L1 MHA (256 → 256)
    connections = create_connections(
        all_nodes[1], all_nodes[2], 
        weights['transformer_layer_1']['attention_q'].T,
        max_connections=300
    )
    add_connection_tiers(connections, 'Proj→TF1 MHA')
    
    # Transformer L1: MHA → FFN (256 → 1024, expansion)
    connections = create_connections(
        all_nodes[2], all_nodes[3],
        weights['transformer_layer_1']['ffn_fc1'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF1 MHA→FFN')
    
    # Transformer L1: FFN → Out (1024 → 256, contraction)
    connections = create_connections(
        all_nodes[3], all_nodes[4],
        weights['transformer_layer_1']['ffn_fc2'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF1 FFN→Out')
    
    # Transformer L1 Out → L2 MHA (256 → 256)
    connections = create_connections(
        all_nodes[4], all_nodes[5],
        weights['transformer_layer_2']['attention_q'].T,
        max_connections=300
    )
    add_connection_tiers(connections, 'TF1→TF2')
    
    # Transformer L2: MHA → FFN (256 → 1024)
    connections = create_connections(
        all_nodes[5], all_nodes[6],
        weights['transformer_layer_2']['ffn_fc1'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF2 MHA→FFN')
    
    # Transformer L2: FFN → Out (1024 → 256)
    connections = create_connections(
        all_nodes[6], all_nodes[7],
        weights['transformer_layer_2']['ffn_fc2'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF2 FFN→Out')
    
    # Transformer L2 Out → L3 MHA (256 → 256)
    connections = create_connections(
        all_nodes[7], all_nodes[8],
        weights['transformer_layer_3']['attention_q'].T,
        max_connections=300
    )
    add_connection_tiers(connections, 'TF2→TF3')
    
    # Transformer L3: MHA → FFN (256 → 1024)
    connections = create_connections(
        all_nodes[8], all_nodes[9],
        weights['transformer_layer_3']['ffn_fc1'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF3 MHA→FFN')
    
    # Transformer L3: FFN → Out (1024 → 256)
    connections = create_connections(
        all_nodes[9], all_nodes[10],
        weights['transformer_layer_3']['ffn_fc2'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF3 FFN→Out')
    
    # Transformer L3 Out → L4 MHA (256 → 256)
    connections = create_connections(
        all_nodes[10], all_nodes[11],
        weights['transformer_layer_4']['attention_q'].T,
        max_connections=300
    )
    add_connection_tiers(connections, 'TF3→TF4')
    
    # Transformer L4: MHA → FFN (256 → 1024)
    connections = create_connections(
        all_nodes[11], all_nodes[12],
        weights['transformer_layer_4']['ffn_fc1'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF4 MHA→FFN')
    
    # Transformer L4: FFN → Out (1024 → 256)
    connections = create_connections(
        all_nodes[12], all_nodes[13],
        weights['transformer_layer_4']['ffn_fc2'].T,
        max_connections=400
    )
    add_connection_tiers(connections, 'TF4 FFN→Out')
    
    # Transformer L4 Out → Pooling (256 → 256, identity-like)
    connections = create_connections(
        all_nodes[13], all_nodes[14],
        np.eye(256),
        max_connections=200
    )
    add_connection_tiers(connections, 'TF4→Pool')
    
    # Pooling → Classifier L1 (256 → 128, contraction)
    connections = create_connections(
        all_nodes[14], all_nodes[15],
        weights['classifier']['fc1'].T,
        max_connections=300,
        threshold_percentile=0
    )
    add_connection_tiers(connections, 'Pool→C1')
    
    # Classifier L1 → Output (128 → 2, final contraction)
    connections = create_connections(
        all_nodes[15], all_nodes[16],
        weights['classifier']['fc2'].T,
        max_connections=256,
        threshold_percentile=0
    )
    add_connection_tiers(connections, 'C1→Output')
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Binary Engagement Classifier: Temporal Sequence Processing (300 Frames → 1 Prediction)<br>' +
                    '<sub>Architecture View: Feature dimensions shown | Each layer processes 300 time steps simultaneously</sub><br>' +
                    '<sub style="font-size:10px">10 seconds of body movement (30 FPS) → Pattern recognition → Engaged/Disengaged</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        scene=dict(
            xaxis=dict(
                title='Information Flow → (Spatial Features)', 
                showgrid=True, 
                gridcolor='rgba(200,200,200,0.3)',
                zeroline=False, 
                showticklabels=True,
                ticktext=['Raw Pose', 'Projection', 'Transform 1-4', 'Pool', 'Classify', 'Output'],
                tickvals=[0, 10, 50, 100, 110, 120]
            ),
            yaxis=dict(
                title='Neurone Distribution (Grid Layout)', 
                showgrid=True, 
                gridcolor='rgba(200,200,200,0.2)',
                zeroline=False, 
                showticklabels=False
            ),
            zaxis=dict(
                title='Neurone Distribution (Grid Layout)', 
                showgrid=True, 
                gridcolor='rgba(200,200,200,0.2)',
                zeroline=False, 
                showticklabels=False
            ),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            annotations=[
                dict(
                    showarrow=False,
                    x=0,
                    y=0,
                    z=-25,
                    text="⏱️ Hidden Temporal Dimension:<br>300 time steps per layer<br>(until pooling)",
                    xanchor="center",
                    xshift=0,
                    yshift=0,
                    font=dict(size=12, color='darkblue'),
                    bgcolor='rgba(173, 216, 230, 0.7)',
                    borderpad=5
                )
            ]
        ),
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest',
        annotations=[
            dict(
                text=(
                    '<b>⚠️ TEMPORAL ARCHITECTURE:</b><br>'
                    'This shows FEATURE dimensions, not temporal dimension<br><br>'
                    '<b>🕐 Each layer processes 300 time steps (frames) simultaneously</b><br><br>'
                    '<b>Input (1,629 features):</b> Per-frame: 543 keypoints × 3 coords<br>'
                    '  → Actual input: 300 frames × 1,629 = 489,300 values<br><br>'
                    '<b>Input Projection (256):</b> Compress to 256 features per frame<br>'
                    '  → Output: 300 × 256 = 76,800 values<br><br>'
                    '<b>Transformer Layers (×4):</b> Process temporal sequences<br>'
                    '  • <b>MHA (256):</b> Compare all 300 frames (90K comparisons/head)<br>'
                    '  • <b>FFN (1,024):</b> Expand features at each time step<br>'
                    '  • <b>Out (256):</b> Compress back to 256 per time step<br>'
                    '  → Each layer: 300 × 256 = 76,800 values flowing through<br><br>'
                    '<b>Pooling (256):</b> 300 time steps → 1 summary (average)<br>'
                    '  → Collapses temporal dimension: 76,800 → 256 values<br><br>'
                    '<b>Classifier (128):</b> No temporal processing (single vector)<br><br>'
                    '<b>Output (2):</b> One prediction per 10-second clip<br><br>'
                    '<b>Connection Colours:</b> Red = strong | Blue = medium | Grey = weak'
                ),
                xref='paper',
                yref='paper',
                x=1.0,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='darkblue',
                borderwidth=2,
                borderpad=10,
                align='left'
            )
        ]
    )
    
    # Save interactive HTML
    print(f"Saving interactive 3D visualization to {output_path}...")
    fig.write_html(output_path)
    
    print(f"\n✅ Interactive 3D network visualization created!")
    print(f"📂 Open in browser: {output_path}")
    print(f"\n🎮 Controls:")
    print(f"   - Click + Drag: Rotate")
    print(f"   - Scroll: Zoom in/out")
    print(f"   - Shift + Drag: Pan")
    print(f"   - Hover: Node details")
    print(f"   - Click legend: Toggle layers")
    
    # Also create weight distribution plots
    create_weight_distributions(weights, output_path.replace('.html', '_weights.html'))
    
    return fig


def create_weight_distributions(weights, output_path):
    """Create interactive plots showing weight distributions."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Input Projection Layer Weights',
            'Transformer Layer 1 Attention Weights',
            'Transformer Layer 1 FFN Weights',
            'Classifier Weights'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # Input projection weights
    emb_weights = weights['input_proj']['weight'].flatten()
    fig.add_trace(
        go.Histogram(x=emb_weights, nbinsx=50, name='Input Proj', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Transformer attention weights
    attn_weights = weights['transformer_layer_1']['attention_q'].flatten()
    fig.add_trace(
        go.Histogram(x=attn_weights, nbinsx=50, name='Attention Q', marker_color='orange'),
        row=1, col=2
    )
    
    # FFN weights
    ffn_weights = weights['transformer_layer_1']['ffn_fc1'].flatten()
    fig.add_trace(
        go.Histogram(x=ffn_weights, nbinsx=50, name='FFN FC1', marker_color='coral'),
        row=2, col=1
    )
    
    # Classifier weights (2D heatmap since it's small)
    classifier_weights = weights['classifier']['fc1']
    fig.add_trace(
        go.Heatmap(
            z=classifier_weights,
            colorscale='RdBu',
            zmid=0,
            text=np.round(classifier_weights, 3),
            texttemplate='%{text}',
            textfont={"size": 6},
            colorbar=dict(title='Weight')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Weight Distributions Across Layers',
        height=800,
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Weight Value', row=1, col=1)
    fig.update_xaxes(title_text='Weight Value', row=1, col=2)
    fig.update_xaxes(title_text='Weight Value', row=2, col=1)
    fig.update_xaxes(title_text='Neuron Index', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    fig.update_yaxes(title_text='Class', row=2, col=2)
    
    fig.write_html(output_path)
    print(f"✅ Weight distribution plots saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create 3D network visualization')
    parser.add_argument('--model-path', type=str, 
                       default='models/action_transformer_binary/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-path', type=str,
                       default='docs/model_3d_network.html',
                       help='Output HTML file path')
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        print(f"   This tool should be run AFTER training completes.")
        print(f"   The model will be saved during training.")
        exit(1)
    
    create_3d_network_visualization(args.model_path, args.output_path)
