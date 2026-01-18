import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from network import get_prediction_grid, get_data_points
from nn_visualizer.svelte_components import network_diagram, decision_boundary

st.title("Why A Single Neuron Is Not Enough")
st.markdown(
    """
    In the previous section, we went over the computation that a single neuron does and the intuition behind it. However, 
    as you might have already noticed, a single neuron can only learn a linear relationship. In this section, we will go 
    over the implications of this and how neural networks solve this problem.
    """
)
st.subheader("The limit of a single neuron")
st.markdown(
    """
    A classic example of a problem a single neuron cannot solve is the XOR problem. Imagine four clusters of points 
    arranged in a square. The top-left and bottom-right clusters are red, while the top-right and bottom-left clusters 
    are blue. Try as you might, no single straight line can separate the red clusters from the blue ones.
    """
)
st.subheader("The Solution - Multiple Neurons and Hidden Layers")
# TODO Mention deep learning?
st.markdown(
    """
    The solution is to use several neurons together. There are two ways to do this: we can add more neurons in a single layer (width), or add more layers (depth).

    Placing multiple neurons in a single 'hidden layer' means they all receive the same inputs, but each has its own set of weights. The outputs of these neurons are then passed to the next layer, where they become inputs to another neuron that produces the final result.
    
    Multiple layers can be used to make the model learn more complex relationships. The output of one layer is fed directly to the next layer. Since each neuron outputs a single number, a neuron in the next layer receives one input for every neuron in the previous layer. For example, if the first hidden layer has 5 neurons, each neuron in the second layer will have 5 inputs. This is why, as mentioned in section 2, different neurons can have different numbers of inputs.
    
    By combining multiple neurons across one or more hidden layers, we create what is called a neural network. A single neuron on its own is not a neural network—it's the connections between many neurons working together that give the network its power.
    """
)

st.subheader("Some Intuition Behind the Neural Network")
st.markdown(
    """
    A single neuron can draw one line to divide the space. A hidden layer with multiple neurons can draw multiple lines. For example, a hidden layer with 3 neurons can draw 3 lines, dividing the space into several regions. Adding more layers allows the network to combine these regions into more complex shapes and patterns.
    """
)
st.subheader("Try It - Solving The XOR Problem")

st.markdown("""
Below are two networks side-by-side. On the left, try to solve XOR with a single neuron. 
On the right, use a network with 2 hidden neurons. Can you see why depth matters?
""")

np.random.seed(42)
n_per_cluster = 20
noise = 0.15

# Four corners
centers = [(-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8), (0.8, 0.8)]
labels = [1, 0, 0, 1]  # XOR pattern

X_list = []
y_list = []
for (cx, cy), label in zip(centers, labels):
    X_list.append(np.random.randn(n_per_cluster, 2) * noise + [cx, cy])
    y_list.append(np.full(n_per_cluster, label))

X = np.vstack(X_list).astype(np.float32)
y = np.concatenate(y_list).astype(np.float32)

xor_data_points = get_data_points(X, y)


class SingleNeuron(nn.Module):
    def __init__(self, w1, w2, bias):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight[0, 0] = w1
            self.linear.weight[0, 1] = w2
            self.linear.bias[0] = bias

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class TwoLayerNetwork(nn.Module):
    def __init__(self, hidden_weights, hidden_biases, output_weights, output_bias):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        with torch.no_grad():
            self.hidden.weight = nn.Parameter(torch.tensor(hidden_weights, dtype=torch.float32))
            self.hidden.bias = nn.Parameter(torch.tensor(hidden_biases, dtype=torch.float32))
            self.output.weight = nn.Parameter(torch.tensor(output_weights, dtype=torch.float32))
            self.output.bias = nn.Parameter(torch.tensor(output_bias, dtype=torch.float32))

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        return torch.sigmoid(self.output(x))


# ============== LAYOUT ==============
col_single, col_multi = st.columns(2)

# ---------- SINGLE NEURON ----------
with col_single:
    st.markdown("#### Single Neuron")

    w1 = st.slider("Weight 1", -5.0, 5.0, 1.0, 0.1, key="single_w1")
    w2 = st.slider("Weight 2", -5.0, 5.0, 1.0, 0.1, key="single_w2")
    bias = st.slider("Bias", -5.0, 5.0, 0.0, 0.1, key="single_bias")

    model_single = SingleNeuron(w1, w2, bias)
    pred_grid_single = get_prediction_grid(model_single)

    network_diagram(
        name="Single",
        layerSizes=[2, 1],
        weights=[[[w1, w2]]],
        epoch=0,
        loss=0,
        key="single_diagram"
    )

    decision_boundary(
        predictions=pred_grid_single,
        dataPoints=xor_data_points,
        key="single_boundary"
    )

    st.warning("No matter how you adjust the sliders, you cannot separate the corners!")

# ---------- TWO-LAYER NETWORK ----------
with col_multi:
    st.markdown("#### Two Hidden Neurons")

    # Preset solutions
    presets = {
        "Default": {
            "h1_w1": 1.0, "h1_w2": 1.0, "h1_b": 0.0,
            "h2_w1": 1.0, "h2_w2": 1.0, "h2_b": 0.0,
            "o_w1": 1.0, "o_w2": 1.0, "o_b": 0.0
        },
        "XOR Solution": {
            "h1_w1": 5.0, "h1_w2": 5.0, "h1_b": 2.5,
            "h2_w1": 5.0, "h2_w2": 5.0, "h2_b": -3.5,
            "o_w1": -5.0, "o_w2": 5.0, "o_b": 5.0
        }
    }

    # Track preset changes
    if "current_preset" not in st.session_state:
        st.session_state.current_preset = "Default"

    preset = st.selectbox("Load preset", list(presets.keys()), key="preset")

    # If preset changed, update all slider values
    if preset != st.session_state.current_preset:
        st.session_state.current_preset = preset
        for key, value in presets[preset].items():
            st.session_state[key] = value
        st.rerun()

    with st.expander("Hidden Neuron 1"):
        h1_w1 = st.slider("Weight from x", -10.0, 10.0, presets["Default"]["h1_w1"], 0.1, key="h1_w1")
        h1_w2 = st.slider("Weight from y", -10.0, 10.0, presets["Default"]["h1_w2"], 0.1, key="h1_w2")
        h1_b = st.slider("Bias", -10.0, 10.0, presets["Default"]["h1_b"], 0.1, key="h1_b")

    with st.expander("Hidden Neuron 2"):
        h2_w1 = st.slider("Weight from x", -10.0, 10.0, presets["Default"]["h2_w1"], 0.1, key="h2_w1")
        h2_w2 = st.slider("Weight from y", -10.0, 10.0, presets["Default"]["h2_w2"], 0.1, key="h2_w2")
        h2_b = st.slider("Bias", -10.0, 10.0, presets["Default"]["h2_b"], 0.1, key="h2_b")

    with st.expander("Output Neuron"):
        o_w1 = st.slider("Weight from H1", -10.0, 10.0, presets["Default"]["o_w1"], 0.1, key="o_w1")
        o_w2 = st.slider("Weight from H2", -10.0, 10.0, presets["Default"]["o_w2"], 0.1, key="o_w2")
        o_b = st.slider("Bias", -10.0, 10.0, presets["Default"]["o_b"], 0.1, key="o_b")

    hidden_w = [[h1_w1, h1_w2], [h2_w1, h2_w2]]
    hidden_b = [h1_b, h2_b]
    output_w = [[o_w1, o_w2]]
    output_b = [o_b]

    model_multi = TwoLayerNetwork(hidden_w, hidden_b, output_w, output_b)
    pred_grid_multi = get_prediction_grid(model_multi)

    network_diagram(
        name="Multi",
        layerSizes=[2, 2, 1],
        weights=[hidden_w, output_w],
        epoch=0,
        loss=0,
        key="multi_diagram"
    )

    decision_boundary(
        predictions=pred_grid_multi,
        dataPoints=xor_data_points,
        key="multi_boundary"
    )

    st.success("With two hidden neurons, we can separate the XOR pattern!")

st.divider()

st.subheader("The Solution - Multiple Neurons and Hidden Layers")
st.markdown("""
The solution is to use several neurons together. Placing multiple neurons in a 'hidden layer' means they all 
receive the same inputs, but each learns its own line. These outputs combine in the next layer to create 
complex decision boundaries.

**What's happening above:** Each hidden neuron draws one diagonal line. The output neuron combines these 
to create the XOR pattern.
""")

st.subheader("Importance of The Activation Function")
st.markdown(
    """
    A key factor that is often overlooked is the activation function. Without it, stacking multiple layers would just collapse into a single linear equation—no matter how many neurons or layers you add. The activation function is what allows each layer to introduce non-linearity, enabling the network to learn complex patterns.
    
    This covers the computational aspects of neural networks. The next section will discuss how these models are trained.
    """
)

# st.subheader("References")
st.divider()
col_prev, mid_col, col_next = st.columns([1, 3, 1])

with col_prev:
    st.page_link("pages/2_a_single_neuron.py", label="Previous", icon="⬅️", use_container_width=True)
with col_next:
    st.page_link("pages/4_how_to_train_a_nn.py", label="Next", icon = "➡️", use_container_width=True)


