# Supplementary Material: Topological Analysis of ICL Allostery

This repository contains the source code, network datasets, and analytical pipelines used in the study: 
**"Topological Reorganization and Allosteric Signaling in Isocitrate Lyase: A Protein Structure Network Analysis"**.

## Overview

The scripts provided here allow for the reproduction of the Protein Structure Network (PSN) analysis of Isocitrate Lyase (ICL) in its various conformational states (PDB IDs: 6EDW, 6EDZ, and 6EE1). The workflow integrates graph theory, community detection, and resilience analysis to map allosteric communication pathways.

## Contents

* `/data`: Graph representations (Edge lists and Node attributes) derived from ICL crystal structures.
* `/scripts`: Python/Jupyter Notebooks for:
    * **Centrality Analysis:** Calculation of Degree, Betweenness, Closeness, and Eigenvector metrics.
    * **Community Detection:** Implementations of Infomap, Modularity, and Girvan-Newman algorithms.
    * **Consensus Mapping:** Generation of Complete Consensus Matrices and 3D hub visualization scripts.
    * **Percolation & Resilience:** Site percolation scripts for targeted attack vs. random failure simulations.
    * **Null Models:** Generation and comparison with Erdős–Rényi (ER) and Barabási–Albert (BA) networks.

## Requirements

The analysis was performed using Python 3.x and requires the following libraries:
* `NetworkX` (Network analysis)
* `Infomap` (Community detection)
* `Pandas` / `NumPy` (Data processing)
* `Matplotlib` / `Seaborn` (Visualization)
* `Py3Dmol` (3D structural mapping)

## Usage

1. Clone the repository: `git clone https://github.com/ChristianQF/ICL_complexity.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in directory `notebooks` to regenerate the figures presented in the manuscript.

## Citation

If you use this code or the generated datasets in your research, please cite:
> *Authors (Year). Topological Reorganization and Allosteric Signaling in Isocitrate Lyase: A Protein Structure Network Analysis. Journal Name (TBD).*

---
**Contact:** [csolisc@unmsm.edu.pe] for any inquiries regarding the implementation or data.
