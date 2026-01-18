import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from network import get_prediction_grid, get_data_points
from nn_visualizer.svelte_components import decision_boundary, network_diagram

st.title("The Building Blocks - A Single Neuron")
st.markdown(
    """
    As stated in the previous section, typical neural networks are made from many neurons working together. The 
    neuron is a mathematical representation of how we think neurons work. Each neuron takes one or more inputs 
    and returns a single number as output. In this section, we will talk about the computation that each neuron does
     and the intuition behind it.
     """
)
st.subheader("Neuron Computation Step-by-Step")
st.markdown(
    """
    1) Inputs
    
    Each neuron receives one or more inputs. For example, if the problem is classifying points on an x-y grid, then 
    there will be two inputs: the x-coordinate and y-coordinate respectively. Note, however, that within a neural 
    network different neurons can have different numbers of inputs. We will come back to this point later.
    
    2) Weights
    
    The inputs of the neuron are multiplied by a set of numbers called 'weights'. For each input there is a single weight, and these weights are specific to each neuron. These weights control how much influence each input has.
    
    3) Sum Weights and Add Bias
    
    The weighted inputs are then added together, along with an additional number called the bias. The bias shifts the result up or down. The computation so far can be expressed as:
    
    `weighted_sum = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + bias`
    
    4) Activation Function
    
    Finally, the weighted sum is passed through an activation function, which determines the neuron's output. 
    A common example is the ReLU (Rectified Linear Unit) function, which outputs the weighted sum if it's positive and 
    zero otherwise.
    """
)

st.subheader("What does it actually do?")
st.markdown(
    """
    A single neuron essentially draws a straight line and classifies the input according to which side of the line the 
    input lies. The weights control the slope of this line and the bias can be thought of as shifting the line up or 
    down. This works well for simple problems, but as we will see in the next section, a single neuron is not enough for more 
    complex problems.
    """
)
st.subheader("Try it - A Single Neuron")

st.markdown("Adjust the weights and bias to see how a single neuron creates a linear decision boundary.")


# Single neuron model
class SingleNeuron(nn.Module):
    def __init__(self, w1, w2, bias):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            self.linear.weight[0, 0] = w1
            self.linear.weight[0, 1] = w2
            self.linear.bias[0] = bias

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Generate simple data points
np.random.seed(42)
n_points = 100
X = np.random.randn(n_points, 2).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)  # Simple diagonal split

# Controls
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Neuron Parameters**")
    w1 = st.slider("Weight 1 (w₁)", -3.0, 3.0, 1.0, 0.1)
    w2 = st.slider("Weight 2 (w₂)", -3.0, 3.0, 1.0, 0.1)
    bias = st.slider("Bias (b)", -3.0, 3.0, 0.0, 0.1)

    st.markdown("**Network Architecture**")
    # Single neuron: 2 inputs -> 1 output
    layer_sizes = [2, 1]
    weights = [[[w1, w2]]]  # Shape: [layer][to_neuron][from_neuron]

    network_diagram(
        name="Single Neuron",
        layerSizes=layer_sizes,
        weights=weights,
        epoch=0,
        loss=0,
        key="single_neuron_diagram"
    )

with col2:
    # Create model with current parameters
    model = SingleNeuron(w1, w2, bias)

    # Get visualization data using your existing helpers
    prediction_grid = get_prediction_grid(model)
    data_points = get_data_points(X, y)

    st.markdown("**Decision Boundary**")
    decision_boundary(
        predictions=prediction_grid,
        # dataPoints=data_points,
        key="single_neuron_boundary"
    )


# st.subheader("References")

# Navigation
st.divider()
col_prev, mid_col, col_next = st.columns([1, 3, 1])

with col_prev:
    st.page_link("pages/1_what_are_nn.py", label="Previous", icon="⬅️", use_container_width=True)

with col_next:
    st.page_link("pages/3_why_a_single_neuron_is_not_enough.py", label="Next", icon = "➡️", use_container_width=True)

