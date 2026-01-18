import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_decision_boundary(model, X, y, resolution=80):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    with torch.no_grad():
        preds = model(torch.tensor(grid)).numpy().reshape(xx.shape)

    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=preds,
        colorscale=[[0, '#ef8354'], [0.5, '#f7f7f7'], [1, '#4f5d75']],
        showscale=True,
        colorbar=dict(title="Pred", thickness=15),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>pred: %{z:.2f}<extra></extra>'
    ))

    # Data points
    for cls, color, name in [(0, '#ef8354', 'Class 0'), (1, '#4f5d75', 'Class 1')]:
        mask = y == cls
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode='markers', name=name,
            marker=dict(size=8, color=color, line=dict(width=1.5, color='white')),
            hovertemplate=f'{name}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title="Decision Boundary",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(title="X₁", scaleanchor="y"),
        yaxis=dict(title="X₂"),
        legend=dict(orientation="h", y=1.12)
    )
    return fig


def plot_loss(train_loss, test_loss=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=train_loss, mode='lines', name='Train Loss',
        line=dict(color='#ef8354', width=2),
        hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}<extra>Train</extra>'
    ))

    if test_loss:
        fig.add_trace(go.Scatter(
            y=test_loss, mode='lines', name='Test Loss',
            line=dict(color='#4f5d75', width=2, dash='dash'),
            hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}<extra>Test</extra>'
        ))

    fig.update_layout(
        title="Loss Curve",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        legend=dict(orientation="h", y=1.15)
    )
    return fig


def plot_network_with_activations(model, X, n_layers, neurons):
    """Network diagram with neuron activations shown as colors"""

    # Get activations
    activations = model.get_neuron_activations(X)

    layer_sizes = [2] + [neurons] * n_layers + [1]

    # Get weights
    weights = [p.detach().numpy() for n, p in model.named_parameters() if 'weight' in n]

    fig = go.Figure()
    positions = {}

    # Calculate positions
    max_neurons = max(layer_sizes)
    x_spacing = 1.2

    for li, n_neurons in enumerate(layer_sizes):
        for ni in range(n_neurons):
            y_offset = (max_neurons - n_neurons) / 2
            positions[(li, ni)] = (li * x_spacing, max_neurons - 1 - ni - y_offset)

    # Draw edges with weight colors
    for li in range(len(layer_sizes) - 1):
        W = weights[li] if li < len(weights) else None
        for fi in range(layer_sizes[li]):
            for ti in range(layer_sizes[li + 1]):
                x0, y0 = positions[(li, fi)]
                x1, y1 = positions[(li + 1, ti)]

                w = W[ti, fi] if W is not None else 0
                w_norm = np.tanh(w)
                alpha = min(abs(w_norm) * 0.8 + 0.1, 1.0)

                if w_norm > 0:
                    color = f'rgba(79, 93, 117, {alpha})'
                else:
                    color = f'rgba(239, 131, 84, {alpha})'

                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode='lines',
                    line=dict(color=color, width=abs(w_norm) * 3 + 0.5),
                    hoverinfo='skip', showlegend=False
                ))

    # Draw nodes with activation colors
    for li, n_neurons in enumerate(layer_sizes):
        # Get activation values for this layer
        if li == 0:
            act_key = 'input'
            act_vals = np.mean(np.abs(activations.get(act_key, np.zeros((1, n_neurons)))), axis=0)
        elif li == len(layer_sizes) - 1:
            act_key = 'output'
            act_vals = np.mean(activations.get(act_key, np.zeros((1, 1))), axis=0)
        else:
            act_key = f'hidden_{li}'
            act_vals = np.mean(np.abs(activations.get(act_key, np.zeros((1, n_neurons)))), axis=0)

        for ni in range(n_neurons):
            x, y = positions[(li, ni)]

            # Color based on activation (normalized)
            if li == 0:
                color = '#2d3142'
                act_val = 0
            elif li == len(layer_sizes) - 1:
                act_val = float(act_vals[0]) if len(act_vals) > 0 else 0
                # Output: show prediction value as color intensity
                color = f'rgb({int(239 - act_val * 160)}, {int(131 - act_val * 38)}, {int(84 + act_val * 33)})'
            else:
                act_val = float(act_vals[ni]) if ni < len(act_vals) else 0
                # Normalize activation to [0, 1] for color
                act_norm = min(act_val / 2.0, 1.0)  # Assuming ReLU-like activations
                # Blend from gray to blue based on activation
                r = int(200 - act_norm * 121)
                g = int(200 - act_norm * 107)
                b = int(200 - act_norm * 83)
                color = f'rgb({r}, {g}, {b})'

            # Node label
            if li == 0:
                label = f'Input X{ni + 1}'
            elif li == len(layer_sizes) - 1:
                label = f'Output<br>Activation: {act_val:.2f}'
            else:
                label = f'Layer {li}, Neuron {ni + 1}<br>Activation: {act_val:.2f}'

            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=28, color=color, line=dict(width=2, color='white')),
                text="", showlegend=False,
                hovertemplate=f'{label}<extra></extra>'
            ))

    # Layer labels
    layer_names = ['Input'] + [f'Hidden {i + 1}' for i in range(n_layers)] + ['Output']
    for idx, name in enumerate(layer_names):
        fig.add_annotation(
            x=idx * x_spacing, y=max_neurons + 0.5,
            text=name, showarrow=False,
            font=dict(size=11, color='#666')
        )

    fig.update_layout(
        title="Network Architecture (hover for activations)",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.5, (len(layer_sizes) - 1) * x_spacing + 0.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, max_neurons + 1]),
        plot_bgcolor='white'
    )
    return fig


def plot_neuron_outputs(model, X, y, n_layers, neurons, resolution=50):
    """Visualize what each neuron in the first hidden layer has learned"""

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Get first hidden layer activations
    with torch.no_grad():
        x_tensor = torch.tensor(grid)
        # Forward through first layer only
        first_layer = model.net[0]
        first_activation = model.net[1]
        hidden_out = first_activation(first_layer(x_tensor)).numpy()

    # Create subplots for each neuron (max 8)
    n_show = min(neurons, 8)
    cols = min(n_show, 4)
    rows = (n_show + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Neuron {i + 1}' for i in range(n_show)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    for i in range(n_show):
        row = i // cols + 1
        col = i % cols + 1

        z = hidden_out[:, i].reshape(xx.shape)

        fig.add_trace(
            go.Heatmap(
                x=np.linspace(x_min, x_max, resolution),
                y=np.linspace(y_min, y_max, resolution),
                z=z,
                colorscale=[[0, '#f7f7f7'], [1, '#4f5d75']],
                showscale=False,
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>activation: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="First Hidden Layer - What Each Neuron Detects",
        height=250 * rows,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # Update axes
    for i in range(n_show):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    return fig