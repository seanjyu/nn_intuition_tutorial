import streamlit as st
from sklearn.model_selection import train_test_split
from data import generate_data
from network import FlexibleMLP, get_prediction_grid, get_data_points
from nn_visualizer.svelte_components import network_diagram, decision_boundary, loss_curve
import time

from visual import *

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.optimizer = None
    st.session_state.epoch = 0
    st.session_state.train_loss = []
    st.session_state.test_loss = []
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.config = None
    st.session_state.is_training = False
    st.session_state.prediction_grid = None
    st.session_state.data_points = None
    st.session_state.loss_history = []

st.title("Interactive Neural Network Trainer")

with st.container():
    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        st.header("Dataset")
        dataset_type = st.selectbox("Choose a pattern", ["Circle", "XOR"])
        noise = st.slider("Noise level", 0.0, 0.5, 0.1)
        n_samples = st.slider("Number of points", 100, 500, 200)
        test_ratio = st.slider("Test set ratio", 0.1, 0.4, 0.2)

    with input_col2:
        st.header("Network Architecture")
        n_hidden_layers = st.slider("Hidden layers", 1, 4, 2)
        neurons_per_layer = st.slider("Neurons per layer", 2, 8, 4)
        activation = st.selectbox("Activation function", ["ReLU", "Tanh", "Sigmoid"])

    with input_col3:
        st.markdown("**Training Settings**")
        learning_rate = st.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
        epochs_per_step = st.select_slider("Epochs per step", options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        max_epochs = st.select_slider("Max epochs", options=[500, 1000, 1500, 2000])

        st.markdown("**Controls**")
        if st.session_state.is_training:
            pause_btn = st.button("⏸ Pause", type="primary", use_container_width=True)
            if pause_btn:
                st.session_state.is_training = False
                st.rerun()
        else:
            play_btn = st.button("▶ Play", type="primary", use_container_width=True)
            if play_btn:
                st.session_state.is_training = True
                st.rerun()
        step_btn = st.button("⏭ Step", use_container_width=True)
        reset_button = st.button("↺ Reset", use_container_width=True)

current_config = {
    "dataset_type": dataset_type,
    "noise": noise,
    "n_samples": n_samples,
    "test_ratio": test_ratio,
    "n_hidden_layers": n_hidden_layers,
    "neurons_per_layer": neurons_per_layer,
    "activation": activation,
    "learning_rate": learning_rate,
    "epochs_per_step": epochs_per_step
}

config_changed = st.session_state.config != current_config

config = (dataset_type, noise, n_samples, n_hidden_layers, neurons_per_layer, activation, learning_rate)

# ==================== INITIALIZE MODEL ====================
if st.session_state.model is None or st.session_state.config != config or reset_button:
    # Stop training on reset/config change
    st.session_state.is_training = False

    # Generate data
    X, y = generate_data(dataset_type, n_samples, noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Create model
    st.session_state.model = FlexibleMLP(n_hidden_layers, neurons_per_layer, activation)
    st.session_state.optimizer = torch.optim.Adam(
        st.session_state.model.parameters(),
        lr=learning_rate
    )
    st.session_state.config = config
    st.session_state.epoch = 0
    st.session_state.loss = 0.0

    st.session_state.prediction_grid = None
    st.session_state.data_points = None
    st.session_state.loss_history = []
# Build layer sizes
layer_sizes = [2] + [neurons_per_layer] * n_hidden_layers + [1]

weights = []
for i in range(len(layer_sizes) - 1):
    # Shape: [to_neurons, from_neurons]
    w = (np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.5).tolist()
    weights.append(w)

@st.fragment
def visualization_section():
    with st.container():
        nn_viz_col1, nn_viz_col2, nn_viz_col3 = st.columns(3)

    with nn_viz_col1:

        progress = min(st.session_state.epoch / max_epochs, 1.0)
        st.progress(progress, text=f"Epoch {st.session_state.epoch} / {max_epochs}")

        st.markdown("**Network Architecture**")
        network_diagram(
            name="Test",
            layerSizes=layer_sizes,
            weights=weights,
            epoch=0,
            loss=0.5,
            key="nn_viz"
        )

    with nn_viz_col2:
        st.markdown("**Decision Boundary**")
        decision_boundary(
            predictions=st.session_state.prediction_grid,
            dataPoints=st.session_state.data_points,
            key="boundary_viz"
        )

    with nn_viz_col3:
        st.markdown("**Loss Curve**")
        loss_curve(
            lossHistory=st.session_state.loss_history,
            key="loss_curve"
        )
visualization_section()
st.divider()


def train_epochs(n_epochs):
    model = st.session_state.model
    optimizer = st.session_state.optimizer
    X_t = torch.tensor(st.session_state.X_train)
    y_t = torch.tensor(st.session_state.y_train).unsqueeze(1)
    criterion = torch.nn.BCELoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        st.session_state.loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

    st.session_state.epoch += n_epochs

    # Update visualization data
    st.session_state.prediction_grid = get_prediction_grid(model)
    st.session_state.data_points = get_data_points(
        st.session_state.X_train,
        st.session_state.y_train
    )

# Step button - single update
if step_btn:
    train_epochs(epochs_per_step)

# Auto-training when playing
if st.session_state.is_training and st.session_state.epoch < max_epochs:
    train_epochs(epochs_per_step)
    time.sleep(0.1)  # Small delay for visualization
    st.rerun()

# Stop training if max epochs reached
if st.session_state.epoch >= max_epochs and st.session_state.is_training:
    st.session_state.is_training = False
    st.toast(f"Training complete! Reached {max_epochs} epochs.")

