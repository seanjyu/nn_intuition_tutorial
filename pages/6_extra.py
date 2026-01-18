import streamlit as st

st.title("Extra Challenges")

st.markdown(
    """
    This tutorial covered the fundamentals of neural networks, but there's much more to explore. The good news is that 
    these core ideas of neurons, layers, weights, and training apply to nearly all neural network architectures.
    """
)

st.subheader("Challenges")
st.markdown(
    """
    The interactive neural network trainer has various other inputs to be tested.
    
    Try these challenges using the neural network trainer:

    1) **Solve XOR**: Find the minimum number of neurons needed to correctly classify the XOR pattern.

    2) **Width vs Depth**: Try solving a problem with one wide layer (many neurons) versus multiple smaller layers. Which works better?

    3) **Overfitting in Action**: Train a very large network on a small dataset. What happens to the training accuracy vs test accuracy?

    4) **Underfitting in Action**: Train a single neuron on a complex pattern. Can you observe underfitting?

    5) **Learning Rate Experiment**: Try different learning rates. What happens when it's too high? Too low?

    6) **Find the Balance**: For a given dataset, find the smallest network that still generalizes well to the test set.
    """
)

