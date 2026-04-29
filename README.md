# 🧬 DynaTCR: Dynamic Hard-Negative Ensemble Graph Learning for TCR–Epitope Binding Prediction

## 📌 Overview

"T-cell receptors (TCRs) recognize antigenic peptides presented by major histocompatibility complex (MHC) molecules and play a central role in adaptive immune responses. Accurate prediction of TCR–epitope binding (TEB) is critical for advancing immunotherapy, vaccine design, and immune profiling."

However, computational TEB prediction remains challenging due to:

* ⚠️ Limited labeled interaction data
* ⚠️ False-negative noise in unobserved pairs
* ⚠️ Over-smoothing in graph neural networks
* ⚠️ Poor generalization under strict evaluation protocols

To address these issues, we propose **DynaTCR**, a **dynamic hard-negative ensemble graph learning framework** that integrates protein language models, graph neural networks, and adaptive negative sampling strategies for robust TEB prediction.

---

## 🧩 Method Overview

DynaTCR consists of four core components:

---

### 1️⃣ Sequence Representation via Protein Language Model

* TCR and epitope sequences are encoded using:

  * 🧠 **Pretrained protein language model embeddings (e.g., ESM2.0)**
* Captures:

  * evolutionary signals
  * biochemical semantics
  * sequence-level contextual dependencies

---

### 2️⃣ Bipartite Graph Construction

* Constructs a **TCR–epitope bipartite interaction graph**
* Nodes:

  * TCR sequences
  * Epitope peptides
* Edges:

  * Known binding interactions

---

### 3️⃣ GR-VPA Graph Encoder (Key Architectural Contribution)

We design a **Graph Regularization + Variance-Preserving Aggregation (GR-VPA)** encoder:

* 📌 Stabilizes message passing in deep GNNs
* 📌 Preserves feature variance during aggregation
* 📌 Alleviates over-smoothing problem
* 📌 Enhances structural representation learning

Additionally, a:

* 🌐 **Global attention layer** is introduced

  * captures long-range dependencies in the interaction graph

---

### 4️⃣ Dynamic Hard-Negative Ensemble Learning

We propose a **dynamic ensemble training strategy**:

* 🔁 Iteratively updates **hard-negative samples**
* 🎯 Focuses on misclassified/unobserved pairs
* 📊 Trains multiple base learners
* 🔗 Ensemble fusion improves robustness

Key benefit:

> Significantly reduces false-negative predictions in TEB inference

---

## 📄 Abstract (DynaTCR)

T-cell receptors (TCRs) recognize antigenic peptides presented by major histocompatibility complex (MHC) molecules and are central to adaptive immunity. Computational prediction of TCR–epitope binding (TEB) can accelerate immunotherapy development, yet remains hampered by limited labeled data, false-negative noise in unobserved pairs, and over-smoothing in graph-based models.

We present **DynaTCR**, a dynamic graph ensemble learning framework for TEB prediction. DynaTCR encodes TCR and epitope sequences with protein language model embeddings and organizes them into a bipartite interaction graph. A **Graph Regularization–Variance Preserving Aggregation (GR-VPA)** encoder stabilizes message propagation and alleviates over-smoothing, while a global attention layer captures long-range dependencies.

Multiple base learners are trained with iteratively updated hard-negative samples to reduce false-negative predictions. Under the StrictTCR evaluation protocol on four public datasets, DynaTCR achieves **AUC improvements of 4.0–8.2%** over the strongest baseline and up to **15.8% improvement in AUPR**. On the most stringently curated dataset, DynaTCR reaches an **AUC of 95.1%**. Furthermore, on an independent structure-derived test set, it achieves the best performance (**72.6% AUC**) among all compared methods, demonstrating strong robustness and generalization ability.

---

## ⚙️ Requirements

```bash id="9xq2lv"
python==3.9.13
numpy==1.19.5
pandas==1.5.3
scikit-learn==1.2.1
torch==1.13.1
torch-geometric==2.0.4
```

---

## 📊 Data Preprocessing

Before training, run:

```bash id="k3n9zd"
python create_data.py
```

This script:

* Processes TCR–epitope interaction data
* Builds bipartite graph structures
* Generates training / validation / test splits

---

## 🚀 Usage

### Train & Evaluate Model

```bash id="m7v2pq"
python main.py
```

---

## 📈 Key Contributions

* 🧠 Protein language model-based TCR/epitope embedding
* 🌐 Bipartite graph modeling of immune interactions
* 🔗 GR-VPA encoder for stable message propagation
* ⚙️ Global attention for long-range dependency modeling
* 🎯 Dynamic hard-negative sampling strategy
* 📊 Ensemble learning for robust prediction
* 💡 Significant improvements under strict evaluation protocols

---

## 🔬 Experimental Highlights

* 📊 +4.0% ~ +8.2% AUC improvement over SOTA methods
* 📊 Up to +15.8% AUPR improvement
* 🧬 95.1% AUC on curated benchmark dataset
* 🧪 72.6% AUC on independent structure-derived test set
* 💪 Strong robustness under StrictTCR evaluation setting

---

## 🔗 Code & Data

👉 GitHub Repository:
[https://github.com/2014402680/TEB/](https://github.com/2014402680/TEB/)

---

## ⭐ Acknowledgements

If you find this work useful, please consider starring the repository ⭐

---
