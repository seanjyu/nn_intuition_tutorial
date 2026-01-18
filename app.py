import streamlit as st

st.set_page_config(
    page_title="Intuitive Understanding of Neural Networks",
    layout="wide",
)

# Define pages
home = st.Page("pages/home.py", title="tutorial", default=True)
interactive_trainer = st.Page("pages/interactive_nn_trainer.py", title="Interactive Neural Network Trainer")
one_what_are_nn = st.Page("pages/1_what_are_nn.py", title="1 - What Are the Neural Network")
two_a_single_neuron = st.Page("pages/2_a_single_neuron.py", title="2 - As a single neuron")
three_why_a_single_neuron_is_not_enough = st.Page("pages/3_why_a_single_neuron_is_not_enough.py", title="3 - Why a single neuron isn't enough")
four_how_training_works = st.Page("pages/4_how_to_train_a_nn.py", title="4 - How To Train A Neural Network")
five_overfitting = st.Page("pages/5_does_it_actually_work.py", title="5 - Does It Actually Work?")
six_extra = st.Page("pages/6_extra.py", title="6 - Extra Challenges")
# Create navigation and run
nav = st.navigation( [home,
                            one_what_are_nn,
                            two_a_single_neuron,
                            three_why_a_single_neuron_is_not_enough,
                            four_how_training_works,
                            five_overfitting,
                            six_extra,
                            interactive_trainer], position="hidden")

# Add persistent section links under Tutorial
with st.sidebar:
    st.page_link("pages/home.py", label="Home")
    st.page_link("pages/1_what_are_nn.py", label="1 - What Are Neural Networks")
    st.page_link("pages/2_a_single_neuron.py", label="2 - The Building Block - A Single Neuron")
    st.page_link("pages/3_why_a_single_neuron_is_not_enough.py", label="3 - Why A Single Neuron Is Not Enough")
    st.page_link("pages/4_how_to_train_a_nn.py", label="4 - How To Train A Neural Network")
    st.page_link("pages/5_does_it_actually_work.py", label="5 - Does It Actually Work?")
    st.page_link("pages/6_extra.py", label="6 - Extra Challenges")
    st.page_link("pages/interactive_nn_trainer.py", label="Interactive neural network")

nav.run()