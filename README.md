# Protein Structure Network (PSN) Analysis: 6EDW-R

This project presents a comprehensive computational study of the **Protein Structure Network (PSN)** of the 6EDW-R complex. By integrating **Graph Theory** with **Biophysical Characterization**, we identify critical residues that act as rigid communication hubs, likely responsible for allosteric signal transduction.

---

## 🚀 Key Research Findings

* **Network Topology:** The protein network follows a **Log-Normal distribution** rather than a pure Power-Law, reflecting physical constraints in amino acid packing.
* **Rigid Communication Hubs:** Found a significant negative correlation (**Spearman ρ: -0.341**, *p < 0.001*) between **Betweenness Centrality** and **B-Factor**.
* **Allosteric Candidates:** Identified high-betweenness, low-flexibility residues—specifically **TRP-108 (Chain C)** and **TRP-457 (Chain D)**—as primary candidates for allosteric regulation.
* **Community Modularization:** Detected structural communities using *Infomap* and *Girvan-Newman* algorithms, identifying functional domains that operate as semi-independent units.

---

## 🛠 Features & Methodology

### 1. Network Construction & Centrality
* Built residue interaction networks using $C\alpha$ distances.
* Calculated 7 centrality metrics: *Degree, Betweenness, Closeness, Eigenvector, PageRank, Hubs, and Authorities*.
* Generated a **Consensus Persistence Ranking** to identify residues that remain important across all metrics.

### 2. Null Model Validation
* Generated 10 **Erdős–Rényi (ER)** and 10 **Barabási–Albert (BA)** networks as null models.
* Statistical comparison of Alpha ($\alpha$), Average Path Length, and Centrality distributions to prove the biological significance of the real network.

### 3. Structural & Biophysical Integration
* **SASA:** Calculated Solvent Accessible Surface Area.
* **Flexibility:** Analyzed B-Factors to map residue mobility.
* **Hidrophobicity:** Mapped Kyte-Doolittle scales to network nodes.

### 4. 3D Interactive Visualization
* Implemented **py3Dmol** for in-notebook 3D visualization.
* Color-coded residues by community membership and network persistence.

---

## 📊 Visualizations Included

1.  **Correlation Plots:** Betweenness vs. B-Factor (Rigidity-Communication analysis).
2.  **Histograms:** Node persistence across centrality measures.
3.  **Boxplots:** Real network metrics vs. ER and BA null models.
4.  **3D Structures:** Interactive PDB rendering of high-impact residues.

---

## 📦 Requirements

* `networkx`
* `biopython`
* `py3Dmol`
* `pandas` / `numpy` / `seaborn`
* `powerlaw`

---

## 📝 Author
*Project developed as part of an Advanced Structural Bioinformatics analysis.*
