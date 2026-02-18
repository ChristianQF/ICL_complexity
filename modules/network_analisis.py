import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
from collections import Counter
import seaborn as sns
import powerlaw
import py3Dmol
from infomap import Infomap
import plotly.graph_objects as go
import warnings
from scipy.optimize import OptimizeWarning

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================
# 1. ANALYZES THE CLUSTERING COEFFICIENT
# ============================================

def analyze_clustering(G):
    """
    Analyzes the clustering coefficient of a graph.
    Args: G: NetworkX graph
    Returns: DataFrame with clustering per node and displays plots
    """
    # Calculate clustering per node
    node_clustering = nx.clustering(G)

    # Convert to DataFrame
    df_clustering = pd.DataFrame(
        list(node_clustering.items()),
        columns=['Node', 'Clustering_Coefficient']
    )

    # Calculate average clustering
    average_clustering = np.mean(list(node_clustering.values()))

    # Create figure with two subplots
    plt.subplots(1, 1, figsize=(6, 4))

    # Histogram
    plt.hist(list(node_clustering.values()), bins=20,
                 edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.title('Clustering Coefficient Distribution')
    plt.axvline(average_clustering, color='red', linestyle='--',
                   label=f'Average: {average_clustering:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Average Clustering Coefficient: {average_clustering:.4f}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    return df_clustering

# ============================================
# 2. ANALYZES DISTANCES IN A GRAPH
# ============================================

def analyze_distances(G):
    """
    Analyzes distances in a graph: average path length and diameter.
    Args: G: NetworkX graph (should be connected or the giant component will be analyzed)
    Returns: Dictionary with distance metrics
    """
    # If graph is not connected, use the giant component
    if not nx.is_connected(G):
        print("Graph not connected. Using giant component...")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        print(f"Nodes in giant component: {G.number_of_nodes()}")

    # Calculate all shortest distances
    print("Calculating distances... (may take time for large graphs)")
    paths = dict(nx.all_pairs_shortest_path_length(G))

    # Extract all distances (excluding distance 0 to itself)
    all_distances = []
    for source in paths:
        for target in paths[source]:
            if source != target:
                all_distances.append(paths[source][target])

    # Calculate metrics
    avg_path_length = np.mean(all_distances)
    max_distance = np.max(all_distances)  # Actual diameter
    distances_counter = Counter(all_distances)

    # Effective diameter (90th percentile)
    sorted_dists = sorted(all_distances)
    idx_90 = int(0.9 * len(sorted_dists))
    effective_diameter = sorted_dists[idx_90]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distance histogram
    dist_values = sorted(distances_counter.keys())
    dist_counts = [distances_counter[d] for d in dist_values]

    axes[0].bar(dist_values, dist_counts, edgecolor='black', alpha=0.7)
    axes[0].axvline(avg_path_length, color='red', linestyle='--',
                   label=f'Average: {avg_path_length:.2f}')
    axes[0].axvline(effective_diameter, color='green', linestyle='--',
                   label=f'Eff. diameter: {effective_diameter}')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Shortest Path Distance Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CDF Plot (Cumulative Distance)
    total_pairs = sum(dist_counts)
    cdf_values = np.cumsum(dist_counts) / total_pairs

    axes[1].plot(dist_values, cdf_values, 'bo-', linewidth=2, markersize=4)
    axes[1].axhline(0.9, color='green', linestyle='--', alpha=0.5)
    axes[1].axvline(effective_diameter, color='green', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Fraction of Node Pairs')
    axes[1].set_title('Cumulative Distance (CDF)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n=== DISTANCE RESULTS ===")
    print(f"Average path length: {avg_path_length:.4f}")
    print(f"Diameter (maximum distance): {max_distance}")
    print(f"Effective diameter (90%): {effective_diameter}")
    print(f"Number of node pairs: {len(all_distances)}")
    print(f"Most common distance: {distances_counter.most_common(1)[0][0]} " +
          f"(appears {distances_counter.most_common(1)[0][1]} times)")

    # Return metrics
    return {
        'avg_path_length': avg_path_length,
        'diameter': max_distance,
        'effective_diameter': effective_diameter,
        'distance_distribution': distances_counter,
        'all_distances': all_distances
    }

# ============================================
# 3. PLOTS OF THE DEGREE DISTRIBUTION
# ============================================

def plot_distribucion_grado(G):
    """
    Generates plots of the degree distribution
    """
    # Calcular grados
    grados = [d for n, d in G.degree()]
    grado_promedio = np.mean(grados)
    grado_max = np.max(grados)

    # Crear figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Histograma de grados
    axes[0].hist(grados, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(grado_promedio, color='red', linestyle='--',
                   label=f'Average: {grado_promedio:.2f}')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Distribución acumulativa
    grados_ordenados = np.sort(grados)
    probabilidad_acumulada = np.arange(1, len(grados_ordenados) + 1) / len(grados_ordenados)

    axes[1].plot(grados_ordenados, probabilidad_acumulada, 'b-', linewidth=2)
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Degree Distribution')
    axes[1].grid(True, alpha=0.3)

    # 3. Top 10 nodos con mayor grado
    top_grados = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    nodos_top = [n[0] for n in top_grados]
    valores_top = [n[1] for n in top_grados]

    axes[2].barh(nodos_top, valores_top, color='lightcoral')
    axes[2].set_xlabel('Degree')
    axes[2].set_title('Top 10 Nodes with the Highest Degree')
    axes[2].invert_yaxis()  # El mayor grado en la parte superior

    plt.tight_layout()
    plt.savefig('distribucion_grado.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Estadísticas adicionales
    print("\n=== DEGREE STATISTICS===")
    print(f"Average degree: {grado_promedio:.2f}")
    print(f"Highest degree: {grado_max}")
    print(f"Minimum degree: {np.min(grados)}")
    print(f"Standard deviation: {np.std(grados):.2f}")

    # Distribución de grados por residuo
    if G.number_of_nodes() > 0:
        grados_por_residuo = {}
        for nodo, attr in G.nodes(data=True):
            resn = attr.get('resn', 'UNK')
            if resn not in grados_por_residuo:
                grados_por_residuo[resn] = []
            grados_por_residuo[resn].append(G.degree(nodo))

        # Calcular promedio por tipo de residuo
        promedio_por_residuo = {resn: np.mean(grados) for resn, grados in grados_por_residuo.items()}

        # Top 10 residuos con mayor grado promedio
        top_residuos = sorted(promedio_por_residuo.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 residues with a higher average grade:")
        for resn, avg_degree in top_residuos:
            print(f"  {resn}: {avg_degree:.2f}")

    return grados

# ============================================
# 4. ANALYZE DEGREE STATISTICS
# ============================================

def analyze_degree_statistics(G):
    # 1. Get degree sequence
    degrees = [d for n, d in G.degree()]

    # 2. Distribution fitting with powerlaw
    # The xmin parameter indicates where the tail evaluation starts
    fit = powerlaw.Fit(degrees, discrete=True)

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # A. PDF plot (usually in log-log scale to see behavior)
    # Use logarithmic bins to reduce noise
    powerlaw.plot_pdf(degrees, ax=ax[0], color='b', marker='o', linestyle='None', label='Data (PDF)')
    ax[0].set_title("Degree PDF (Log-Log)")
    ax[0].set_xlabel("Degree (k)")
    ax[0].set_ylabel("P(k)")

    # B. CCDF plot and comparison of fits
    powerlaw.plot_ccdf(degrees, ax=ax[1], color='black', linewidth=2, label='Data (CCDF)')

    # Draw candidate fits
    fit.power_law.plot_ccdf(ax=ax[1], color='r', linestyle='--', label='Power Law Fit')
    fit.lognormal.plot_ccdf(ax=ax[1], color='g', linestyle='--', label='Log-Normal Fit')
    fit.exponential.plot_ccdf(ax=ax[1], color='orange', linestyle='--', label='Exponential Fit')

    ax[1].set_title("Degree CCDF and Candidate Fits")
    ax[1].set_xlabel("Degree (k)")
    ax[1].set_ylabel("P(K ≥ k)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # --- GOODNESS OF FIT COMPARISON ---
    print(f"Estimated Alpha value (Power Law): {fit.power_law.alpha:.2f}")
    print(f"xmin value (tail start): {fit.xmin}")

    # Compare Power Law vs other distributions
    # R is the log-likelihood ratio. If R > 0, the first model is better.
    # p is the significance value.

    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"\nPower Law vs Exponential comparison: R={R:.2f}, p-value={p:.4f}")

    R_log, p_log = fit.distribution_compare('power_law', 'lognormal')
    print(f"Power Law vs Log-Normal comparison: R={R_log:.2f}, p-value={p_log:.4f}")

# Execution:
# analyze_degree_statistics(G)

# ============================================
# 5. CENTRALITY ANALISIS
# ============================================

def analyze_centrality(G):
    """
    Calculates and visualizes centrality measures
    """
    print("\n=== CENTRALITY ANALYSIS ===")

    try:
        # Calculate different centrality measures
        print("Calculating betweenness centrality...")
        betweenness = nx.betweenness_centrality(G, normalized=True)

        print("Calculating closeness centrality...")
        closeness = nx.closeness_centrality(G)

        print("Calculating degree centrality...")
        degree_centrality = nx.degree_centrality(G)

        print("Calculating eigenvector centrality...")
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)

        # Create DataFrame with all measures
        centrality_df = pd.DataFrame({
            'Node': list(G.nodes()),
            'Betweenness': [betweenness[n] for n in G.nodes()],
            'Closeness': [closeness[n] for n in G.nodes()],
            'Degree_Centrality': [degree_centrality[n] for n in G.nodes()],
            'Eigenvector': [eigenvector[n] for n in G.nodes()]
        })

        # Add residue information
        residue_info = []
        for node in G.nodes():
            attr = G.nodes[node]
            residue_info.append(f"{attr.get('chain', '')}/{attr.get('resi', '')}/{attr.get('resn', '')}")

        centrality_df['Residue'] = residue_info

        # Centrality histograms
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes = axes.flatten()
        measures = ['Betweenness', 'Closeness', 'Degree_Centrality', 'Eigenvector']
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'plum']

        for idx, (measure, color) in enumerate(zip(measures, colors)):
            axes[idx].hist(centrality_df[measure], bins=20, alpha=0.7,
                          color=color, edgecolor='black')
            axes[idx].set_xlabel(f'{measure} Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {measure}')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('centrality_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Top nodes for each centrality measure
        print("\n=== TOP NODES BY CENTRALITY ===")

        for measure in measures:
            top_nodes = centrality_df.nlargest(5, measure)[['Node', 'Residue', measure]]
            print(f"\nTop 5 by {measure}:")
            print(top_nodes.to_string(index=False))

        return centrality_df

    except Exception as e:
        print(f"Error calculating centrality: {e}")
        print("Trying with basic measures...")

        # Only degree centrality if others fail
        degree_centrality = nx.degree_centrality(G)

        centrality_df = pd.DataFrame({
            'Node': list(G.nodes()),
            'Degree_Centrality': [degree_centrality[n] for n in G.nodes()]
        })

        return centrality_df

# ============================================
# 6. ANALYZE CENTRALITY RANKINGS
# ============================================

def analyze_centrality_rankings(G, top_n=50):
    print("Calculating centrality metrics... this may take a moment.")

    # 1. Centrality Calculations
    dict_centralities = {
        'Degree': nx.degree_centrality(G),
        'Closeness': nx.closeness_centrality(G),
        'Betweenness': nx.betweenness_centrality(G),
        'Eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
        'Katz': nx.katz_centrality(G, alpha=0.1, beta=1.0),
        'PageRank': nx.pagerank(G),
        'Subgraph': nx.subgraph_centrality(G)
    }

    # 2. Create DataFrame to compare rankings
    df_ranks = pd.DataFrame(index=G.nodes())

    for name, values in dict_centralities.items():
        # Save the value
        df_ranks[f'{name}_Val'] = pd.Series(values)
        # Create ranking (1 = most central)
        df_ranks[f'{name}_Rank'] = df_ranks[f'{name}_Val'].rank(ascending=False, method='min')

    # 3. Get Top 50 for each metric
    top_nodes_dict = {}
    for name in dict_centralities.keys():
        top_list = df_ranks.sort_values(by=f'{name}_Val', ascending=False).head(top_n).index.tolist()
        top_nodes_dict[name] = top_list

    df_top_50 = pd.DataFrame(top_nodes_dict)

    # 4. Export results
    df_top_50.to_csv("top_50_centralities.csv", index=False)
    print("Top 50 saved to 'top_50_centralities.csv'")

    return df_ranks, df_top_50

# Execution
# df_full, df_top50 = analyze_centrality_rankings(G)

# ====================================================
# 7. VISUALIZE CENTRALITY PERSISTENCE IN 3D STRUCTURE
# ====================================================

def visualize_persistence_3d(df_key_residues, pdb_path='pdb_path'):
#def visualize_persistence_3d(df_key_residues, pdb_id='6EDW'):
    # 1. Create viewer and load protein from PDB
    ##view = py3Dmol.view(query=f'pdb:{pdb_id}')

    # Upload from local archive
    view = py3Dmol.view()
    view.addModel(open(pdb_path, 'r').read(), 'pdb')

    # 2. Default style: Gray semi-transparent cartoon
    view.setStyle({'cartoon': {'color': '#e0e0e0', 'opacity': 0.6}})

    # 3. Define function to assign colors based on percentage
    # Use Blue (low) to Red (high) scale
    def get_color(percentage):
        if percentage == 100: return '#FF0000' # Red (Maximum consensus)
        if percentage >= 70:  return '#FF8C00' # Orange
        if percentage >= 50:  return '#FFFF00' # Yellow
        if percentage >= 30:  return '#00FF00' # Green
        return '#0000FF' # Blue (Specialists)

    # 4. Color Top residues according to their persistence
    # Node format is D/385/GLU -> [Chain, Number, AA]
    for _, row in df_key_residues.iterrows():
        parts = row['Node'].split('/')
        chain = parts[0]
        number = int(parts[1])
        percentage = row['Percentage']

        # Select specific residue
        selection = {'chain': chain, 'resi': number}

        # Apply style: Spheres (VDW) to highlight key residues
        color_hex = get_color(percentage)
        view.addStyle(selection, {'stick': {'colorscheme': f'{color_hex}raw', 'radius': 0.3}})
        view.addStyle(selection, {'sphere': {'colorscheme': f'{color_hex}raw', 'scale': 0.8}})

        # Optional: Add hover label
        view.addLabel(f"{parts[2]} {number} ({int(percentage)}%)",
                      {'fontSize': 10, 'fontColor': 'black', 'backgroundColor': 'white', 'backgroundOpacity': 0.5},
                      selection)

    # 5. Center and display
    view.zoomTo()
    #print(f"Visualizing structure {pdb_path}")
    print(f"Visualizing structure {pdb_path}")
    print("🔴 Red: 100% | 🟠 Orange: >70% | 🟡 Yellow: >50% | 🔵 Blue: <30%")
    return view.show()

# Execution:
# visualize_persistence_3d(df_key_residues, pdb_id='6EDW')
# visualize_persistence_3d(df_key_residues, pdb_path='pdb_path')

def visualize_persistence_3d_clean(df_key_residues, pdb_path='pdb_path'):
    # 1. Create the viewer
    #view = py3Dmol.view(query=f'pdb:{pdb_id}')
       #  view = py3Dmol.view(query=f'pdb:{pdb_id}')
    view = py3Dmol.view()
    view.removeAllModels() # Clean any residue from previous loading
    view.addModel(open(pdb_path, 'r').read(), 'pdb')

    # 2. Base style for entire protein (Light Gray)
    view.setStyle({'cartoon': {'color': '#D3D3D3', 'opacity': 0.7}})

    # Identify Top 10 for labels
    top_10_nodes = df_key_residues.head(10)['Node'].tolist()

    # 3. Define solid color scale
    def get_color_hex(percentage):
        if percentage == 100: return '0xFF0000' # Red
        if percentage >= 75:  return '0xFFA500' # Orange
        if percentage >= 50:  return '0xFFFF00' # Yellow
        return '0x0000FF' # Blue

    # 4. Iterate over ranking residues
    for idx, row in df_key_residues.iterrows():
        node = row['Node']
        parts = node.split('/')
        chain = parts[0]
        res_num = int(parts[1])
        aa_name = parts[2]
        percentage = row['Percentage']

        col_hex = get_color_hex(percentage)
        selection = {'chain': chain, 'resi': res_num}

        # Apply Stick and Sphere with forced color (no CPK)
        # Use 'color' instead of 'colorscheme' to avoid atomic standard
        view.addStyle(selection, {
            'stick': {'color': col_hex, 'radius': 0.2},
            'sphere': {'color': col_hex, 'scale': 0.7}
        })

        # 5. LABELS ONLY FOR TOP 10
        if node in top_10_nodes:
            view.addLabel(f"{aa_name}{res_num} ({int(percentage)}%)",
                          {
                              'fontSize': 12,
                              'fontColor': 'white',
                              'backgroundColor': col_hex,
                              'backgroundOpacity': 0.8,
                              'showBackground': True
                          },
                          selection)

    view.zoomTo()
    print(f"Visualizing {pdb_path}. Labels only shown for the 10 most persistent residues.")
    return view.show()

# Execution
# visualize_persistence_3d_clean(df_key_residues, pdb_id='6EDW')

import py3Dmol

def visualize_3d_persistence_pro(df_key_residues, pdb_path='pdb_path', show_labels=True):
    """
    Visualizes the protein coloring residues by persistence.
    - show_labels: True to view Top 10, False for a clean view.
    """
    # 1. Viewer configuration
   #  view = py3Dmol.view(query=f'pdb:{pdb_id}')
    view = py3Dmol.view()
    view.removeAllModels() # Clean any residue from previous loading
    view.addModel(open(pdb_path, 'r').read(), 'pdb')

    # 2. Base style (Gray Cartoon)
    view.setStyle({'cartoon': {'color': '#D3D3D3', 'opacity': 0.6}})

    # Color scale (Clean hexadecimals)
    colors_map = {
        100: '#FF0000', # Red
        75:  '#FFA500', # Orange
        50:  '#FFFF00', # Yellow
        0:   '#0000FF'  # Blue (for the rest of the top)
    }

    top_10_nodes = df_key_residues.head(10)['Node'].tolist()

    # 3. Residue mapping
    for idx, row in df_key_residues.iterrows():
        parts = row['Node'].split('/')
        chain = parts[0]
        res_num = int(parts[1])
        aa_name = parts[2]
        percentage = row['Percentage']

        # Determine color based on range
        if percentage == 100: color = colors_map[100]
        elif percentage >= 75: color = colors_map[75]
        elif percentage >= 50: color = colors_map[50]
        else: color = colors_map[0]

        selection = {'chain': chain, 'resi': res_num}

        # Force solid color in Stick and Sphere
        view.addStyle(selection, {
            'stick': {'color': color, 'radius': 0.25},
            'sphere': {'color': color, 'scale': 0.8}
        })

        # 4. Label logic
        if show_labels and (row['Node'] in top_10_nodes):
            view.addLabel(f"{aa_name}{res_num}",
                          {
                              'fontSize': 10,
                              'fontColor': 'black',
                              'backgroundColor': 'white',
                              'backgroundOpacity': 0.6
                          },
                          selection)

    view.zoomTo()
    # Bonus: Add stronger ambient light so colors stand out
    view.setClickable(True)

    status = "with labels (Top 10)" if show_labels else "without labels (clean view)"
    print(f"🔹 Showing {pdb_path} {status}")

    return view.show()

# --- EXECUTION MODES ---

# Option A: Clean view (Only colors)
# visualize_3d_persistence_pro(df_key_residues, pdb_path='pdb_path', show_labels=False)

# Option B: View with labels
# visualize_3d_persistence_pro(df_key_residues, pdb_path='pdb_path', show_labels=True)

# ====================================================
# 8. COMMUNITY ANALYSIS
# ====================================================

def analyze_and_visualize_infomap(G, csv_name="community_results.csv",  seed=None):
    if seed is None:
           	seed = SEED  # Using  global seed by default
    # 1. Infomap execution
    im = Infomap(f"--silent --seed {seed}")
    mapping = im.add_networkx_graph(G)
    im.run()

    # 2. Community dictionary and CSV export
    comm_dict = {mapping[node.node_id]: node.module_id for node in im.tree if node.is_leaf}
    pd.DataFrame(list(comm_dict.items()), columns=['Node', 'Community']).to_csv(csv_name, index=False)

    # 3. Identify the 10 largest communities
    count = Counter(comm_dict.values())
    top_10_ids = [community for community, _ in count.most_common(10)]

    # 4. Color assignment
    # Use a color map for top 10 and gray for the rest
    cmap = plt.cm.get_cmap('tab10', 10)
    color_map_top = {com_id: cmap(i) for i, com_id in enumerate(top_10_ids)}

    # List of colors for each node in G
    node_colors = [
        color_map_top[comm_dict[node]] if comm_dict[node] in top_10_ids else (0.9, 0.9, 0.9)
        for node in G.nodes()
    ]

    # 5. Visualization
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))

    # Draw edges with high transparency
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)

    # 6. Create Custom Legend (only Top 10)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Community {cid} ({count[cid]} nodes)',
                   markerfacecolor=color_map_top[cid], markersize=10)
        for cid in top_10_ids
    ]

    plt.legend(handles=legend_handles, title="Top 10 Communities", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Infomap: Focus on the 10 Main Communities\n(Total: {len(count)} communities)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return [[n for n, c in comm_dict.items() if c == i] for i in range(1, max(comm_dict.values()) + 1)]

# ====================================================
# 9. COMPARATIVE COMMUNITY ANALYSIS
# ====================================================

def comparative_community_analysis(G, csv_file_name="community_comparison.csv", seed=None):
    # Initialize DataFrame with graph nodes
    nodes = list(G.nodes())
    df_final = pd.DataFrame({'Node': nodes})

    # Define algorithms to execute
    # Each entry is: (Name, Algorithm_Function)

    if seed is None:
           	seed = SEED  # Using  global seed by default

    # 1. Infomap (separate logic as it's an external library)
    im = Infomap(f"--silent --seed {seed}")
    mapping = im.add_networkx_graph(G)
    im.run()
    infomap_dict = {mapping[node.node_id]: node.module_id for node in im.tree if node.is_leaf}
    df_final['Infomap'] = df_final['Node'].map(infomap_dict)

    # 2. Dictionary of NetworkX algorithms
    # Note: K-Clique may leave nodes without community, we'll mark them as 0
    methods = {
        'Modularity': lambda: nx.community.greedy_modularity_communities(G, weight='weight'),
        'Label_Propagation': lambda: nx.community.label_propagation_communities(G),
        'Girvan_Newman': lambda: next(nx.community.girvan_newman(G)), # Take first level
        'K_Clique_3': lambda: nx.community.k_clique_communities(G, 3)
    }

    # Execution and visualization
    for name, func in methods.items():
        try:
            community_sets = list(func())
            # Convert list of sets to dictionary {node: community_id}
            temp_dict = {}
            for i, cluster in enumerate(community_sets, 1):
                for node in cluster:
                    temp_dict[node] = i

            # Save to DataFrame (nodes not assigned in K-Clique remain as NaN/0)
            df_final[name] = df_final['Node'].map(temp_dict).fillna(0).astype(int)

            # Plot
            _plot_top_communities(G, temp_dict, name)

        except StopIteration:
            print(f"Error processing {name}")

    # Save the unified CSV
    df_final.to_csv(csv_file_name, index=False)
    print(f"Analysis complete. File saved as: {csv_file_name}")

    # Also plot Infomap which was processed outside the loop
    _plot_top_communities(G, infomap_dict, "Infomap")

    return df_final

def _plot_top_communities(G, comm_dict, title):
    """Internal function to maintain consistency in plots"""
    count = Counter(comm_dict.values())
    top_10_ids = [community for community, _ in count.most_common(10)]

    cmap = plt.cm.get_cmap('tab10', 10)
    color_map_top = {com_id: cmap(i) for i, com_id in enumerate(top_10_ids)}

    node_colors = [
        color_map_top[comm_dict[node]] if node in comm_dict and comm_dict[node] in top_10_ids
        else (0.9, 0.9, 0.9) for node in G.nodes()
    ]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'C{cid} ({count[cid]} n.)',
                   markerfacecolor=color_map_top[cid], markersize=8)
        for cid in top_10_ids
    ]

    plt.legend(handles=legend_handles, title="Top 10", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Algorithm: {title}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_local_pdb_communities(df_communities, method, pdb_path, n_communities=5):
    """
    Visualizes local PDB file coloring the N largest communities.

    Inputs:
    - df_communities: DataFrame (df_community_analisis_G_S01)
    - method: String ('Infomap', 'Modularity', etc.)
    - pdb_path: Local path to the .pdb file
    - n_communities: Number of largest communities to color
    """

    # 1. Identify the N largest communities for this method
    top_comms = df_communities[method].value_counts().nlargest(n_communities).index.tolist()

    # 2. Prepare color palette (using a Matplotlib colormap)
    cmap = plt.get_cmap('tab10') # Palette with 10 distinct colors
    hex_colors = [mcolors.to_hex(cmap(i)) for i in range(n_communities)]
    color_map = dict(zip(top_comms, hex_colors))

    # 3. Read the local PDB file
    try:
        with open(pdb_path, 'r') as f:
            pdb_data = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {pdb_path}")
        return

    # 4. Configure the viewer
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')

    # Base style: Transparent cartoon for non-top communities
    view.setStyle({'cartoon': {'color': '#f0f0f0', 'opacity': 0.2}})

    # 5. Color each community
    print(f"Coloring the {n_communities} largest communities for method: {method}")

    for i, comm_id in enumerate(top_comms):
        color = color_map[comm_id]
        # Filter nodes belonging to this community
        comm_nodes = df_communities[df_communities[method] == comm_id]['Node'].tolist()

        for node in comm_nodes:
            parts = node.split('/')
            chain = parts[0]
            res_num = int(parts[1])

            selection = {'chain': chain, 'resi': res_num}
            # Apply color to the community
            view.addStyle(selection, {'cartoon': {'color': color, 'opacity': 1.0}})

        print(f"   - Community {comm_id}: {color} ({len(comm_nodes)} residues)")

    view.zoomTo()
    return view.show()

# --- USAGE EXAMPLE ---
# path = "C:/users/documents/6EDW.pdb"  # Adjust your path
# visualize_local_pdb_communities(df_community_analisis_G_S01, 'Infomap', path, n_communities=4)

# =============================================================
# 10. CONSENSUS ANALYSIS BETWEEN COMMUNITY ANALYSIS ALGORITHMS
# =============================================================

def generate_consensus_matrix(df):
    # Extract only algorithm columns
    matrix_data = df.iloc[:, 1:].values
    n_nodes = matrix_data.shape[0]
    n_algorithms = matrix_data.shape[1]

    # Initialize co-occurrence matrix
    consensus = np.zeros((n_nodes, n_nodes))

    # Sum matches for each algorithm
    for i in range(n_algorithms):
        col = matrix_data[:, i].reshape(-1, 1)
        # Creates a boolean matrix where True means they belong to the same community
        consensus += (col == col.T)

    # Normalize (0 to 1) where 1 is total consensus across all methods
    consensus_norm = consensus / n_algorithms
    return consensus_norm

def extract_consensus_and_plot(df_analysis, threshold=1.0):
    """
    Analyzes node co-occurrence in communities across multiple methods.

    Inputs:
    - df_analysis: DataFrame with columns ['Node', 'Method1', 'Method2', ...]
    - threshold: Value from 0 to 1. 1.0 means nodes must match in ALL methods.
    """
    # --- 1. Preparation and Co-occurrence Calculation ---
    nodes = df_analysis['Node'].values
    matrix_data = df_analysis.iloc[:, 1:].values
    n_nodes = matrix_data.shape[0]
    n_algorithms = matrix_data.shape[1]

    # Vectorized calculation for better speed
    consensus = np.zeros((n_nodes, n_nodes))
    for i in range(n_algorithms):
        col = matrix_data[:, i].reshape(-1, 1)
        consensus += (col == col.T)

    consensus_norm = consensus / n_algorithms

    # --- 2. Extraction of Strong Relationships (Step 2) ---
    relationships = []
    # Use triu (upper triangle) to avoid duplicate pairs (A,B) and (B,A)
    rows, cols = np.where(np.triu(consensus_norm, k=1) >= threshold)

    for r, c in zip(rows, cols):
        relationships.append({
            'Node_A': nodes[r],
            'Node_B': nodes[c],
            'Consensus_Strength': consensus_norm[r, c]
        })

    df_consensus = pd.DataFrame(relationships)

    # --- 3. Generation of Consensus Graph (Step 3) ---
    G_cons = nx.Graph()
    if not df_consensus.empty:
        for _, row in df_consensus.iterrows():
            G_cons.add_edge(row['Node_A'], row['Node_B'], weight=row['Consensus_Strength'])

    # Visualization
    plt.figure(figsize=(12, 10))
    if len(G_cons) > 0:
        pos = nx.spring_layout(G_cons, k=0.15, seed=42)
        # Draw with clean style
        nx.draw_networkx_edges(G_cons, pos, alpha=0.2, edge_color='royalblue')
        nx.draw_networkx_nodes(G_cons, pos, node_size=30, node_color='darkblue', alpha=0.7)

        # Optional: Labels only if there are few nodes
        if len(G_cons) < 50:
            nx.draw_networkx_labels(G_cons, pos, font_size=8)

        plt.title(f"Consensus Graph (Threshold >= {threshold})\nNodes that algorithms consistently group together")
    else:
        plt.text(0.5, 0.5, "No relationships found with this threshold",
                 ha='center', va='center', fontsize=12)

    plt.axis('off')
    plt.show()

    print(f"Identified {len(df_consensus)} reliable relationships.")
    return df_consensus, G_cons

def plot_unit_count(df_multithreshold):
    """
    This function takes the df_multithreshold generated by the extract_consensus_and_plot function
    and summarizes the network complexity at each level.
    """
    # Count unique values per column (excluding 'Node')
    thresholds = df_multithreshold.columns[1:]
    counts = [df_multithreshold[u].nunique() for u in thresholds]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(thresholds, counts, color='skyblue', edgecolor='navy')
    plt.bar_label(bars, padding=3)
    plt.title("Evolution of the number of communities according to consensus threshold")
    plt.ylabel("Number of Communities")
    plt.xlabel("Consensus Level (Threshold)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def generate_multi_threshold_consensus_csv(df_analysis, thresholds=[1.0, 0.9, 0.8, 0.7, 0.6], output_name="multi_threshold_consensus.csv"):
    """
    Takes the algorithm comparison dataframe and creates a table of
    communities based on consensus for different rigor levels.
    """
    # 1. Extract data and prepare co-occurrence matrix
    nodes = df_analysis['Node'].values
    matrix_data = df_analysis.iloc[:, 1:].values
    n_nodes = matrix_data.shape[0]
    n_methods = matrix_data.shape[1]

    # Calculate how many times each node pair coincides in the same cluster
    consensus = np.zeros((n_nodes, n_nodes))
    for i in range(n_methods):
        col = matrix_data[:, i].reshape(-1, 1)
        consensus += (col == col.T)

    # Normalize (from 0.0 to 1.0)
    consensus_norm = consensus / n_methods

    # 2. Create output DataFrame
    df_multi_threshold = pd.DataFrame({'Node': nodes})

    # 3. For each threshold, find connected components (communities)
    for u in thresholds:
        G_temp = nx.Graph()
        G_temp.add_nodes_from(nodes)

        # Only create edges between nodes that coincide in at least 'u'% of cases
        rows, cols = np.where(np.triu(consensus_norm, k=1) >= u)
        for f, c in zip(rows, cols):
            G_temp.add_edge(nodes[f], nodes[c])

        # Extract resulting communities
        components = list(nx.connected_components(G_temp))

        # Map each node to its community ID at this specific threshold
        mapping = {}
        for idx, cluster in enumerate(components, 1):
            for node in cluster:
                mapping[node] = idx

        df_multi_threshold[f'threshold_{u}'] = df_multi_threshold['Node'].map(mapping)

    # 4. Save file
    df_multi_threshold.to_csv(output_name, index=False)
    print(f"Multi-threshold consensus file generated: {output_name}")

    return df_multi_threshold

# ================================================
# 11. GENERATES AN INTERACTIVE SANKEY DIAGRAM
# ================================================

def generate_filtered_sankey(df_multi_threshold, min_nodes=5):
    """
    Generates an interactive Sankey Diagram showing how nodes are grouped
    across different consensus thresholds.

    Inputs:
    - df_multi_threshold: DataFrame with columns ['Node', 'threshold_1.0', 'threshold_0.9', ...]
    - min_nodes: Communities with fewer than this number of nodes are grouped into 'Others'.
    """
    # 1. Prepare threshold column names
    threshold_cols = [c for c in df_multi_threshold.columns if c.startswith('threshold_')]
    df_plot = df_multi_threshold.copy()

    # 2. Filter noise: Group small communities
    for col in threshold_cols:
        counts = df_plot[col].value_counts()
        small = counts[counts < min_nodes].index
        # Mark as -1 communities that don't reach the minimum
        df_plot.loc[df_plot[col].isin(small), col] = -1

    # 3. Build Nodes and Flows for Plotly
    labels = []
    nodes_idx_map = {}

    # Create unique labels for each community at each level
    for col in threshold_cols:
        u_val = col.split('_')[1]
        unique_comms = sorted(df_plot[col].unique())
        for comm in unique_comms:
            node_name = f"{col}_C{comm}"
            nodes_idx_map[node_name] = len(labels)
            if comm == -1:
                labels.append(f"U{u_val} Miscellaneous")
            else:
                labels.append(f"U{u_val} C{comm}")

    sources, targets, values = [], [], []

    # Create connections between consecutive levels
    for i in range(len(threshold_cols) - 1):
        current = threshold_cols[i]
        next_col = threshold_cols[i+1]

        # Group flows
        flows = df_plot.groupby([current, next_col]).size().reset_index(name='count')

        for _, row in flows.iterrows():
            sources.append(nodes_idx_map[f"{current}_C{row[current]}"])
            targets.append(nodes_idx_map[f"{next_col}_C{row[next_col]}"])
            values.append(row['count'])

    # 4. Generate interactive chart
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15, thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = "royalblue"
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = "rgba(173, 216, 230, 0.4)" # Light blue transparent
        )
    )])

    fig.update_layout(
        title_text=f"Sankey Diagram: Community Stability (Minimum {min_nodes} nodes)",
        font_size=12,
        height=800
    )

    fig.show(renderer="notebook")

# --- HOW TO EXECUTE THE ENTIRE FLOW ---

# 1. Generate the multi-threshold dataframe (using the function I gave you earlier)
# df_multi_threshold = generate_multi_threshold_consensus_csv(df_analysis, thresholds=[1.0, 0.9, 0.8, 0.7, 0.6])

# 2. Execute Sankey passing that specific dataframe
# generate_filtered_sankey(df_multi_threshold, min_nodes=5)

# ================================================
# 12. NULL MODELS
# ================================================

def calculate_network_metrics(G):
    """Calculates metrics for items 2-5 and 7 for a given network."""
    # Degrees for PDF/CCDF and Alpha
    degrees = [d for n, d in G.degree() if d > 0]
    # Suprimir warnings específicos de powerlaw
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        alpha = fit.power_law.alpha

    # Centralities (averages)
    # Note: We use averages to be able to compare groups of networks
    bet = np.mean(list(nx.betweenness_centrality(G).values()))
    clo = np.mean(list(nx.closeness_centrality(G).values()))
    eig = np.mean(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
    pagerank = np.mean(list(nx.pagerank(G).values()))

    return {
        'Alpha': alpha,
        'Avg_Betweenness': bet,
        'Avg_Closeness': clo,
        'Avg_Eigenvector': eig,
        'Avg_PageRank': pagerank
    }

def null_models_analysis(G_real):
    n = G_real.number_of_nodes()
    m = G_real.number_of_edges()

    # Real Network Metrics
    res_real = calculate_network_metrics(G_real)

    er_results = []
    ba_results = []

    for i in range(10):
        # Erdős–Rényi: same n and m
        G_er = nx.gnm_random_graph(n, m)
        er_results.append(calculate_network_metrics(G_er))

        # Barabási–Albert: same n and m (m_ba is approx m/n)
        m_ba = max(1, int(m/n))
        G_ba = nx.barabasi_albert_graph(n, m_ba)
        # Adjust edges exactly if needed, but BA is structural
        ba_results.append(calculate_network_metrics(G_ba))

    # Tabulate results
    df_er = pd.DataFrame(er_results)
    df_ba = pd.DataFrame(ba_results)

    comparison = {
        'Metric': res_real.keys(),
        'Real': res_real.values(),
        'ER_Mean': df_er.mean(),
        'ER_Std': df_er.std(),
        'BA_Mean': df_ba.mean(),
        'BA_Std': df_ba.std()
    }

    return pd.DataFrame(comparison)

# Execution
# df_nulls = null_models_analysis(G)
# print(df_nulls)

def plot_null_model_comparison(G_real, n_simulations=10):
    n = G_real.number_of_nodes()
    m = G_real.number_of_edges()

    # 1. Get real network metrics
    real_metrics = calculate_network_metrics(G_real)
    metric_names = list(real_metrics.keys())

    # 2. Simulate networks and collect data
    data_list = []
    m_ba = max(1, int(m/n))

    for i in range(n_simulations):
        # ER
        G_er = nx.gnm_random_graph(n, m)
        metrics_er = calculate_network_metrics(G_er)
        for name in metric_names:
            data_list.append({'Metric': name, 'Value': metrics_er[name], 'Type': 'ER (Random)'})

        # BA
        G_ba = nx.barabasi_albert_graph(n, m_ba)
        metrics_ba = calculate_network_metrics(G_ba)
        for name in metric_names:
            data_list.append({'Metric': name, 'Value': metrics_ba[name], 'Type': 'BA (Scale-free)'})

    df_simulations = pd.DataFrame(data_list)

    # 3. Create subplots (one per metric)
    fig, axes = plt.subplots(1, len(metric_names), figsize=(20, 6))

    for i, name in enumerate(metric_names):
        # Filter data for current metric
        df_sub = df_simulations[df_simulations['Metric'] == name]

        # Draw Boxplot of null models
        sns.boxplot(
            data=df_sub,
            x='Type',
            y='Value',
            hue='Type',
            palette='Pastel1',
            width=0.5,
            ax=axes[i],
            legend=False
        )

        # Overlay real value as a prominent red point
        axes[i].plot(0.5, real_metrics[name], marker='D', color='red', markersize=10,
                     label='Real Network (Protein)', linestyle='None')

        axes[i].set_title(f'{name}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Value')
        if i == 0:
            axes[i].legend()

    plt.suptitle('Comparison: Protein Network vs. Null Models (ER and BA)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

# Execution
# plot_null_model_comparison(G)
