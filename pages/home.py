import streamlit as st

st.title("Interactive Tutorial on Building Neural Network Intuition")

st.markdown("""
Neural networks power many of today's most impressive technologies—from voice assistants to image recognition to language translation. But how do they actually work?

This interactive tutorial will help you build a deep intuition for neural networks through hands-on experimentation. Instead of starting with equations and theory, you'll explore visually, adjust parameters, and watch networks learn in real-time.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("What You'll Learn")
    st.markdown("""
    - **How a single neuron works** — the fundamental building block
    - **Why we need multiple neurons** — and what happens when we connect them
    - **How neural networks learn** — training, loss functions, and gradient descent
    - **The role of architecture choices** — layers, neurons, and activation functions
    - **Intuition through experimentation** — adjust weights, watch decision boundaries evolve
    """)

with col2:
    st.subheader("What You Won't Learn")
    st.markdown("""
    - Rigorous mathematical derivations
    - Production-level implementation details
    - Advanced architectures (CNNs, transformers, etc.)
    - Specific frameworks (PyTorch, TensorFlow)

    *This tutorial focuses on core concepts and intuition, not on making you a practitioner (yet!).*
    """)

st.divider()

st.subheader("Prerequisites")
st.markdown("""
None! This tutorial is designed for complete beginners. If you know what a function is (input → output), you're ready to start.
""")

st.subheader("How To Use This Tutorial")
st.markdown("""
Work through the sections in order using the sidebar. Each section builds on the previous one:

1. **What is a Neural Network?** — A brief overview
2. **The Building Blocks - A Single Neuron** — Understanding a single neuron  
3. **Why One Neuron Isn't Enough** — The XOR problem and hidden layers
4. **How to Train A Neural Network** — Explain the training loop of a neural network
5. **Does It Actually Work** - Overfitting and Underfitting
6. **Extra Challenges**
- **Interactive Neural Network Trainer** — Experiment freely with everything you've learned
""")

st.divider()

# Get started button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Get Started →", type="primary", use_container_width=True):
        st.switch_page("pages/1_what_are_nn.py")

st.divider()
