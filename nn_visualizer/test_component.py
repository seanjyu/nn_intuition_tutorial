# nn_visualizer/test_component.py
import streamlit as st

st.set_page_config(page_title="Component Test", layout="centered")

# Import the component
from svelte_components import my_component

st.title("🧪 Testing Svelte Component")

# Sidebar controls
st.sidebar.header("Settings")
hidden_layers = st.sidebar.slider("Hidden Layers", 1, 6, 2)
neurons = st.sidebar.slider("Neurons per Layer", 2, 8, 4)
name = st.sidebar.text_input("Your Name", "Student")

st.write("---")

# Render the component
# The return value is whatever we set with Streamlit.setComponentValue()
click_count = my_component(
    name=name,
    hiddenLayers=hidden_layers,
    neuronsPerLayer=neurons,
    key="test"
)

st.write("---")

# Display what the component returned
if click_count is not None:
    st.success(f"🎉 Component returned: **{click_count}** clicks!")
else:
    st.info("Click the button in the component above!")

# Show that Python values update the component
st.write("### Try changing the sidebar values!")
st.write("The component should update automatically.")