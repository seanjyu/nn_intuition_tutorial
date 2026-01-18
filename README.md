# Interactive Tutorial on Building Neural Network Intuition

An interactive educational tool for building intuition about how neural networks work. Built with Streamlit and custom Svelte visualizations.

## Overview

This project provides a hands-on approach to understanding neural networks, starting from the basics of a single neuron and building up to training multi-layer networks on classification problems. Rather than focusing on equations and theory, the emphasis is on visual intuition and experimentation.

## Features

- **Interactive Visualizations**: Real-time decision boundary updates, network architecture diagrams, and loss curves
- **Step-by-Step Learning**: Progressive sections that build understanding from single neurons to deep networks
- **Hands-On Experimentation**: Adjust weights, biases, and network architecture to see immediate effects
- **Classic Problems**: Explore why single neurons fail on XOR and how hidden layers solve it

## Sections

1. **What Are Neural Networks**
   Brief introduction on what Neural Networks and short history of origin 
2. **The Building Blocks - A Single Neuron**  
   Understand what a single neuron computes and how it creates a linear decision boundary.
3. **Why A Single Neuron Is Not Enough**  
   Discover the limitations of linear classifiers through the XOR problem, and see how adding hidden layers enables complex patterns.
4. **How to Train A Neural Network**
5. **Does It Actually Work?**
6. **Extra Challenges and Additional Reading**
- **Interactive Trainer**  
   Train neural networks on different datasets (Circle, XOR) with customizable architecture and hyperparameters.

## Installation

### Prerequisites

- Python 3.12+
- Node.js (for Svelte components)

### Usage
## Run Locally
Install required 
```bash
# Clone the repository
git clone git@github.com:seanjyu/nn_intuition_tutorial.git
cd nn_intuition_tutorial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install and build Svelte components
cd nn_visualizer
npm install
npm run build
# Open a new terminal and cd to nn_intuition_tutorial folder
streamlit streamlit run app.py 
```
Then open your browser to `http://localhost:8501`.


## Technologies

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[PyTorch](https://pytorch.org/)** - Neural network implementation
- **[Svelte](https://svelte.dev/)** - Custom interactive visualizations
- **[scikit-learn](https://scikit-learn.org/)** - Basic Data utilities


[//]: # (## Screenshots)
[//]: # ()
[//]: # (<!-- TODO Add Screenshots -->)
[//]: # ()


## Acknowledgments
- TensorFlow Playground for the original inspiration
- [Streamlit Svelte Component Template](https://github.com/93degree/streamlit-component-svelte-template) - This repo was used as the basis for the svelte components