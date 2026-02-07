import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
import seaborn as sns

# ============================================
# 1. UPLOAD AND PROCESS JSON ARCHIVES
# ============================================

def load_ring_json_data(json_path):
    """
    Loads and processes the JSON file in GraphJSON format
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"File loaded successfully")
    print(f"Main keys: {list(data.keys())}")

    # Basic information
    print(f"Directed graph: {data.get('directed', False)}")
    print(f"Multigraph: {data.get('multigraph', False)}")

    return data

# ============================================
# 2. EXTRACT NODES AND EDGES
# ============================================

def extract_nodes_edges(data):
    """
    Extracts nodes and edges from GraphJSON format
    """
    # Extract nodes
    nodes = []
    if 'elements' in data and 'nodes' in data['elements']:
        for node in data['elements']['nodes']:
            node_info = node['data']
            nodes.append({
                'id': node_info['id'],
                'name': node_info.get('name', ''),
                'residue': f"{node_info['chain']}/{node_info['resi']}/{node_info['resn']}",
                'chain': node_info['chain'],
                'resi': node_info['resi'],
                'resn': node_info['resn'],
                'degree': node_info.get('degree', 0),
                'dssp': node_info.get('dssp', ''),
                'x_coord': node_info.get('x_coord', 0),
                'y_coord': node_info.get('y_coord', 0),
                'z_coord': node_info.get('z_coord', 0)
            })

    print(f"Number of nodes extracted: {len(nodes)}")

    # Extract edges (interactions)
    edges = []
    if 'elements' in data and 'edges' in data['elements']:
        for edge in data['elements']['edges']:
            edge_info = edge['data']
            edges.append({
                'source': edge_info['source'],
                'target': edge_info['target'],
                'interaction': edge_info.get('interaction', ''),
                'type': edge_info.get('type', ''),
                'distance': edge_info.get('distance', 0),
                'probability': edge_info.get('probability', 1.0),
                'energy': edge_info.get('energy', 0)
            })
    elif 'data' in data and len(data['data']) > 0:
        # Alternative format: edges in 'data'
        for edge in data['data']:
            edges.append({
                'source': edge['source'],
                'target': edge['target'],
                'interaction': edge.get('interaction', ''),
                'type': edge.get('type', ''),
                'distance': edge.get('distance', 0),
                'probability': edge.get('probability', 1.0),
                'energy': edge.get('energy', 0)
            })

    print(f"Number of edges (interactions) extracted: {len(edges)}")

    return nodes, edges

# ============================================
# 3. CREAR GRAFO CON NETWORKX
# ============================================

def create_graph_from_dataframes(df_nodos, df_aristas):
    """
    Crea un grafo de NetworkX a partir de DataFrames de nodos y aristas
    """
    G = nx.Graph()

    # Agregar nodos con atributos
    for _, nodo in df_nodos.iterrows():
        G.add_node(
            nodo['id'],
            residue=nodo['residue'],
            chain=nodo['chain'],
            resi=nodo['resi'],
            resn=nodo['resn'],
            degree=nodo['degree'],
            dssp=nodo['dssp'],
            x_coord=nodo['x_coord'],
            y_coord=nodo['y_coord'],
            z_coord=nodo['z_coord']
        )

    # Agregar aristas con atributos
    for idx, arista in df_aristas.iterrows():
        G.add_edge(
            arista['source'],
            arista['target'],
            interaction=arista['interaction'],
            type=arista['type'],
            distance=arista['distance'],
            probability=arista['probability'],
            energy=arista.get('energy', 0),
            id=idx
        )

    return G

