import streamlit as st

st.title("What are Neural Networks?")
st.markdown(
    """
    A neural network is a machine learning algorithm that tries to identify patterns given a set of example inputs and 
    outputs. For instance, a neural network could learn to recognize handwritten digits by being shown thousands of 
    examples of each number along with their correct labels. Today, neural networks power many everyday applications, 
    from voice assistants and language translation to photo filters and movie recommendations. 
    Generally, neural networks solve two classes of problems: classification and regression. Classification means 
    assigning a label to an input, e.g. deciding if an email is spam or not. Regression means predicting a number, 
    e.g. estimating a house price. This tutorial will focus on classification.
    """
)

st.subheader("Some History")
st.markdown(
    """ 
    Although the most basic form of the neural network was discovered in the 1970s$^{1}$, neural networks only rose to 
    popularity in the 2010s. This was in large part due to the hardware not being able to efficiently train the models, 
    causing funds to dwindle and optimism in the technology to fade. This period often referred to as the "AI winter"$^{2}$.
    However, over the years the combination of faster computer hardware and more efficient neural network training 
    techniques led to breakthroughs in the 2010s. Ever since, neural networks have been a mainstay in machine learning.
    """
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/1/MNIST_dataset_example.png", caption="The MNIST dataset: handwritten digits 0-9, one of the earliest classification problems to be solved by AI")


st.subheader("Where does the term 'Neural' come from?")
st.markdown(
    """
    The naming of the neural network comes from how our brains work. Neurons are tiny cells in our brains that process 
    information, and our brain has billions of them working together, which allows us to perform complex tasks. 
    The neural network is loosely based on this idea, as it is built from a bunch of artificial neurons that work 
    together. It is a common misconception that neural networks emulate how our brains work. In reality, they are only 
    inspired by how we think the brain works, similar to how early submarines were inspired by fish but use propellers 
    instead of fins."""
)
# TODO add a picture

st.subheader("References")
st.markdown(
    """
    1. McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The Bulletin of Mathematical Biophysics, 5(4), 115–133.
    2. Lighthill, J. (1973). "Artificial Intelligence: A General Survey."
    """
)

# Navigation
st.divider()
col_prev, mid_col, col_next = st.columns([1, 3, 1])

with col_prev:
    st.page_link("pages/home.py", label="Previous", icon="⬅️", use_container_width=True)

with col_next:
    st.page_link("pages/2_a_single_neuron.py", label="Next", icon = "➡️",  use_container_width=True)