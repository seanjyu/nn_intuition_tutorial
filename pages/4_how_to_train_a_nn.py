import streamlit as st
from sklearn.model_selection import train_test_split
from data import generate_data
from network import FlexibleMLP, get_prediction_grid, get_data_points
from nn_visualizer.svelte_components import decision_boundary, loss_curve
import torch
import time

st.title("How To Train A Neural Network")
st.markdown(
    """
    So far, we have explored how neural networks compute their outputs. One important detail we haven't discussed yet is
    that neural networks require training data to learn from. This is a set of example inputs along with their correct 
    outputs. This section will focus on how the network uses these examples to gradually adjust its weights until it 
    can make accurate predictions.
    """
)

st.subheader("Steps to Neural Network Training")
st.markdown(
    """
    Training a neural network can be broken down into four main steps: initialization, forward pass, loss calculation, 
    and backpropagation. These steps are repeated many times until the model learns to make accurate predictions.
    
    1. Initialize the model
    First, a network architecture is chosen, including the number of layers and neurons per layer. The weights are then 
    set to small random values. These will be adjusted during training.
    2. Forward Pass
    The training inputs are passed through the model to get predictions. The data flows through each layer, with every 
    neuron performing its calculation and passing the result forward until the final output is produced.
    3. Calculate Loss
    A loss function compares the model's predictions to the correct outputs and produces a single number representing 
    how wrong the model was. A higher loss means worse predictions. A simple example is Mean Squared Error, which takes 
    the average of the squared differences between predictions and correct values. Squaring means larger errors are 
    penalized more heavily.
    
    4. Backpropagation
    Now that we have a value to know how wrong the model's prediction was, we need to adjust the weights and biases to 
    reduce the error. Backpropagation works backwards from the output, calculating the gradient for each weight. The 
    gradient tells us which direction to adjust the weight and by how much in order to reduce the loss. Each weight is 
    then moved in that direction. By repeating this process many times, the network gradually learns. This process is 
    commonly known as gradient descent.
    
    5. Repeat Until Done!
    The forward pass, loss calculation, and backpropagation steps are repeated many times until the model makes accurate predictions.
    
    In practice, training is measured in steps and epochs. A step is one update to the weights, which usually happens after the model sees a small batch of training examples. An epoch is one full pass through the entire training dataset. Training typically takes many epochs—the model sees the same data multiple times, getting a little better each time.
    """
)

# Prefix session state keys to avoid conflicts with other pages
if "train_model" not in st.session_state:
    st.session_state.train_model = None
    st.session_state.train_optimizer = None
    st.session_state.train_epoch = 0
    st.session_state.train_X = None
    st.session_state.train_y = None
    st.session_state.train_is_training = False
    st.session_state.train_prediction_grid = None
    st.session_state.train_data_points = None
    st.session_state.train_loss_history = []
    st.session_state.train_lr = 0.03

# Fixed architecture for this demo
N_HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 4
ACTIVATION = "ReLU"
DATASET = "XOR"
N_SAMPLES = 100
NOISE = 0.1
MAX_EPOCHS = 500


def initialize_model():
    """Reset everything and create a fresh model."""
    st.session_state.train_is_training = False

    # Generate data
    X, y = generate_data(DATASET, N_SAMPLES, NOISE)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    st.session_state.train_X = X_train
    st.session_state.train_y = y_train

    # Create model
    st.session_state.train_model = FlexibleMLP(N_HIDDEN_LAYERS, NEURONS_PER_LAYER, ACTIVATION)
    st.session_state.train_optimizer = torch.optim.Adam(
        st.session_state.train_model.parameters(),
        lr=st.session_state.train_lr
    )
    st.session_state.train_epoch = 0
    st.session_state.train_loss_history = []
    st.session_state.train_prediction_grid = get_prediction_grid(st.session_state.train_model)
    st.session_state.train_data_points = get_data_points(X_train, y_train)


# Initialize on first load
if st.session_state.train_model is None:
    initialize_model()


def train_step(n_epochs=10):
    """Run one training step."""
    model = st.session_state.train_model
    optimizer = st.session_state.train_optimizer
    X_t = torch.tensor(st.session_state.train_X)
    y_t = torch.tensor(st.session_state.train_y).unsqueeze(1)
    criterion = torch.nn.BCELoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        st.session_state.train_loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

    st.session_state.train_epoch += n_epochs
    st.session_state.train_prediction_grid = get_prediction_grid(model)


# ============== UI ==============
st.subheader("Try it - Training")
st.markdown("""
Watch a neural network learn in real-time. Click **Step** to run a few training iterations, 
or **Play** to train continuously. Notice how the loss decreases and the decision boundary 
evolves from random to accurate.
""")

# Controls
col_controls, col_viz1, col_viz2 = st.columns([1, 1.5, 1.5])

with col_controls:
    st.markdown("**Training Controls**")

    # Learning rate
    new_lr = st.select_slider(
        "Learning rate",
        options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        value=st.session_state.train_lr,
        key="train_lr_slider"
    )
    if new_lr != st.session_state.train_lr:
        st.session_state.train_lr = new_lr
        for param_group in st.session_state.train_optimizer.param_groups:
            param_group['lr'] = new_lr

    # Progress
    progress = min(st.session_state.train_epoch / MAX_EPOCHS, 1.0)
    st.progress(progress, text=f"Epoch {st.session_state.train_epoch} / {MAX_EPOCHS}")

    # Buttons
    if st.session_state.train_is_training:
        if st.button("⏸ Pause", type="primary", use_container_width=True):
            st.session_state.train_is_training = False
            st.rerun()
    else:
        if st.button("▶ Play", type="primary", use_container_width=True):
            st.session_state.train_is_training = True
            st.rerun()

    if st.button("⏭ Step", use_container_width=True):
        train_step(10)
        st.rerun()

    if st.button("↺ Reset", use_container_width=True):
        initialize_model()
        st.rerun()

with col_viz1:
    st.markdown("**Decision Boundary**")
    decision_boundary(
        predictions=st.session_state.train_prediction_grid,
        dataPoints=st.session_state.train_data_points,
        key="train_boundary"
    )

with col_viz2:
    st.markdown("**Loss Curve**")
    loss_curve(
        lossHistory=st.session_state.train_loss_history,
        key="train_loss"
    )

# Auto-training loop
if st.session_state.train_is_training and st.session_state.train_epoch < MAX_EPOCHS:
    train_step(10)
    time.sleep(0.1)
    st.rerun()

# Stop at max epochs
if st.session_state.train_epoch >= MAX_EPOCHS and st.session_state.train_is_training:
    st.session_state.train_is_training = False
    st.toast("Training complete!")

# Guided prompts
st.markdown("""
**Things to try:**
1. Click **Step** a few times—watch the loss go down
2. Try a high learning rate (0.3) vs low (0.001)—what's different?
3. Press **Reset** and **Play** to watch the full training process
""")

# st.subheader("References")
st.divider()
col_prev, mid_col, col_next = st.columns([1, 3, 1])

with col_prev:
    st.page_link("pages/3_why_a_single_neuron_is_not_enough.py", label="Previous", icon="⬅️", use_container_width=True)
with col_next:
    st.page_link("pages/5_does_it_actually_work.py", label="Next", icon = "➡️", use_container_width=True)


