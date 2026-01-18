

import os
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "svelte_components",
        url="http://localhost:5000",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "public", "build")
    _component_func = components.declare_component("svelte_components", path=build_dir)


def network_diagram(
        name: str = "Student",
        layerSizes: list = None,
        weights: list = None,
        epoch: int = 0,
        loss: float = 0.0,
        animate: bool = True,
        height: int = 500,
        theme: dict = None,
        key: str = None
):
    """Display the neural network architecture diagram."""
    if layerSizes is None:
        layerSizes = [2, 4, 1]
    if weights is None:
        weights = []
    if theme is None:
        theme = {
            "base": "light",
            "primaryColor": "#ff4b4b",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#000000",
            "font": "sans-serif"
        }

    return _component_func(
        componentType="network",
        name=name,
        layerSizes=layerSizes,
        weights=weights,
        epoch=epoch,
        loss=loss,
        animate=animate,
        height=height,
        key=key,
        theme=theme,
        default=None
    )


def decision_boundary(
        predictions: list = None,
        dataPoints: list = None,
        height: int = 400,
        theme: dict = None,
        key: str = None
):
    """Display the decision boundary visualization."""
    if predictions is None:
        predictions = []
    if dataPoints is None:
        dataPoints = []
    if theme is None:
        theme = {
            "base": "light",
            "primaryColor": "#ff4b4b",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#000000",
            "font": "sans-serif"
        }


    return _component_func(
        componentType="boundary",
        predictions=predictions,
        dataPoints=dataPoints,
        height=height,
        key=key,
        theme=theme,
        default=None
    )

def loss_curve(
        lossHistory:list = None,
        height: int = 400,
        theme: dict = None,
        key: str = None
):
    if theme is None:
        theme = {
            "base": "light",
            "primaryColor": "#ff4b4b",
            "backgroundColor": "#0e1117",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#FFFFFF",
            "font": "sans-serif"
        }


    return _component_func(
          componentType="loss",
          lossHistory=lossHistory,
          height=height,
          key=key,
          theme=theme,
          default=None
      )