# ABOUTME: Streamlit custom component wrapper for the 3D blob UI
# ABOUTME: Bridges the standalone assets/index.html into the Streamlit app

import os
import streamlit.components.v1 as components

# Path to the assets/ directory at the project root
_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)

_component_func = components.declare_component("heathcliff_blob", path=_ASSETS_DIR)


def blob(state: str = "idle", height: int = 600, key: str | None = None):
    """Render the 3D Heathcliff blob.

    Args:
        state: One of "idle", "listening", "thinking", "speaking".
        height: Pixel height for the component iframe.
        key: Optional Streamlit widget key.
    """
    return _component_func(state=state, height=height, default=None, key=key)
