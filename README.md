# Supplementary Material: Topological Analysis of ICL Allostery

This repository contains the source code, network datasets, and analytical pipelines used in the study: 
**"Topological Reorganization and Allosteric Signaling in Isocitrate Lyase: A Protein Structure Network Analysis"**.

![GitHub Logo](https://github.com/ChristianQF/chemoinformatics/blob/main/Cover_figure.png)

## Overview

The scripts provided here allow for the reproduction of the Protein Structure Network (PSN) analysis of Isocitrate Lyase (ICL) in its various conformational states (PDB IDs: 6EDW, 6EDZ, and 6EE1). The workflow integrates graph theory, community detection, and resilience analysis to map allosteric communication pathways.

## Abstract work
Allosteric regulation in Isocitrate Lyase (ICL) is a fundamental process for metabolic adaptation, yet the topological mechanisms governing signal transduction across its conformational landscape remain poorly understood. This study employs a Protein Structure Network (PSN) approach to characterize the intramolecular interaction networks of ICL in its Apo (6EDW), transition (6EDZ), and ligand-bound (6EE1) states. By integrating graph theory with biophysical parameters, we identified a transition from a diffuse, less efficient topology in the Apo state to a highly integrated "small-world" architecture upon Acetyl-CoA binding. Our results demonstrate that ligand binding induces a significant reduction in network diameter (~35%) and average path length, effectively optimizing internal communication routes. Degree distribution analysis revealed that the network follows a Log-Normal topology, reflecting physical packing constraints while facilitating the emergence of high-connectivity "super-hubs." Community detection via the Infomap algorithm further identified a reorganization of modular granularity, where the monolithic Apo-structure partitions into discrete functional units during the allosteric transition. Centrality consensus mapping pinpointed salient residues, specifically Arg193, Arg141, and Phe195, as critical gatekeepers of information flow. The biological significance of these hubs was validated through site percolation analysis, which revealed an extreme vulnerability to targeted attacks: the removal of the top 15% of central nodes triggers a global collapse of structural connectivity. Collectively, these findings provide a topological blueprint for ICL allostery, identifying specific residue clusters as primary targets for site-directed mutagenesis and the development of novel allosteric inhibitors.

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
> Solis-Calero, Christian (2026). Topological Reorganization and Allosteric Signaling in Isocitrate Lyase: A Protein Structure Network Analysis. (UNMSM, Lima Perú).*

---
**Contact:** [csolisc@unmsm.edu.pe] for any inquiries regarding the implementation or data.
