import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from network import get_prediction_grid, get_data_points, FlexibleMLP
from nn_visualizer.svelte_components import decision_boundary

st.title("Does It Actually Work?")
st.markdown(
    """
    If you've read this far, you now understand the computations that a neural network does, the basic architecture, and
     how to train one. Let's say you train a model and achieve high accuracy, is your job done? Is the model ready to be
      used?

    Not quite. A model that performs well on training data often struggles with new data it hasn't seen before. This 
    section will cover the basics of training versus testing, and two common problems: underfitting and overfitting.
"""
)
st.subheader("Overfitting vs Underfitting")
st.markdown(
    """
    The phenomenon in which the model achieves a high training score but does poorly on new data is called overfitting. 
    This occurs because the model essentially memorizes the training data rather than learning a general pattern. 
    A memorized pattern only works if new data looks exactly like the training data, which is rarely the case.
    
    Underfitting is the opposite problem—the model performs poorly on both training and new data. This can happen when 
    the model is too simple to capture the patterns, or when training hasn't run long enough for the model to learn them
     yet. For example, trying to fit a complex pattern with a single neuron would result in underfitting.
    
    The goal is to find a balance between overfitting and underfitting, meaning that the model is not just memorizing 
    training data but learning the underlying patterns that apply to new data as well.
    
    Consider the example of 2 slightly overlapping classes:
    """
)
# Two overlapping blobs - true boundary is linear but with noisy overlap
np.random.seed(42)
n_samples = 100

# Blob 1 (class 0) - centered at (-0.5, 0)
X0 = np.random.randn(n_samples, 2) * 0.5 + [-0.5, 0]

# Blob 2 (class 1) - centered at (0.5, 0), overlaps with blob 1
X1 = np.random.randn(n_samples, 2) * 0.5 + [0.5, 0]

X = np.vstack([X0, X1]).astype(np.float32)
y = np.array([0] * n_samples + [1] * n_samples).astype(np.float32)

data_points = get_data_points(X, y)


# ============== TRAIN MODELS ==============

def train_model(model, X, y, epochs, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        loss.backward()
        optimizer.step()
    return model


# Underfit: trained for just a few epochs (hasn't learned yet)
torch.manual_seed(42)
model_underfit = FlexibleMLP(n_hidden_layers=1, neurons_per_layer=4, activation="ReLU")
model_underfit = train_model(model_underfit, X, y, epochs=5)

# Good fit: appropriate training
torch.manual_seed(42)
model_good = FlexibleMLP(n_hidden_layers=1, neurons_per_layer=4, activation="ReLU")
model_good = train_model(model_good, X, y, epochs=200)

# Overfit: complex model trained aggressively
torch.manual_seed(42)
model_overfit = FlexibleMLP(n_hidden_layers=3, neurons_per_layer=16, activation="ReLU")
model_overfit = train_model(model_overfit, X, y, epochs=3000, lr=0.01)

# ============== DISPLAY ==============
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Underfitting**")
    st.caption("Not trained enough")
    decision_boundary(
        predictions=get_prediction_grid(model_underfit),
        dataPoints=data_points,
        key="underfit"
    )

with col2:
    st.markdown("**Good Fit**")
    st.caption("Appropriate training")
    decision_boundary(
        predictions=get_prediction_grid(model_good),
        dataPoints=data_points,
        key="goodfit"
    )

with col3:
    st.markdown("**Overfitting**")
    st.caption("Model too complex")
    decision_boundary(
        predictions=get_prediction_grid(model_overfit),
        dataPoints=data_points,
        key="overfit"
    )

st.markdown("""
**Underfitting** (left): The model hasn't trained long enough to find the pattern.

**Good fit** (middle): A clean boundary that accepts some errors in the overlapping region—this is the best we can do with noisy data.

**Overfitting** (right): The boundary twists and curves to classify every training point correctly, even the noisy ones. This won't generalize well to new data.
""")
st.subheader("Training vs Test Data")
st.markdown(
    """
    So how do we know if our model is overfitting or underfitting? This is where the training and test split comes in.

Before training, we divide our data into two parts: a training set and a test set. The model learns from the training set, but the test set is kept hidden until training is complete. By comparing performance on both sets, we can diagnose problems. High training accuracy but low test accuracy suggests overfitting. Low accuracy on both suggests underfitting.
    """
)

# st.subheader("References")
st.divider()
col_prev, mid_col, col_next = st.columns([1, 3, 1])

with col_prev:
    st.page_link("pages/4_how_to_train_a_nn.py", label="Previous", icon="⬅️", use_container_width=True)
with col_next:
    st.page_link("pages/6_extra.py", label="Next", icon = "➡️", use_container_width=True)

