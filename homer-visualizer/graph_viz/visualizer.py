import os
import tempfile
from pyvis.network import Network
import streamlit as st
import networkx as nx


def visualize_graph(
    G: nx.Graph,
    height: str = "600px",
    width: str = "100%",
    notebook: bool = False
):
    """
    Render a NetworkX graph using PyVis in a Streamlit app.

    Parameters:
        G (nx.Graph): The graph to visualize.
        height (str): Height of the visualization canvas.
        width (str): Width of the visualization canvas.
        notebook (bool): If True, prepares output for notebook use (default False for Streamlit).
    """
    net = Network(height=height, width=width, directed=True, notebook=notebook)
    net.from_nx(G)

    net.set_options("""
    {
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.2
          }
        }
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "graph.html")
        net.save_graph(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            st.components.v1.html(html, height=int(height.replace("px", "")) + 50, scrolling=True)
