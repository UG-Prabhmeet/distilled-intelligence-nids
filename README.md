# 📘 Distilled Intelligence for Network Intrusion Detection (BTP)

A research‑oriented pipeline that combines **deep learning** and **knowledge distillation** to detect network attacks using the UNSW‑NB15 dataset.  
This Bachelor’s Thesis project implements a teacher‑student training framework and provides tools for evaluation and real‑time simulation.

---

## 🎯 Objectives

1. **Establish High-Fidelity Baseline**: Train a 1D-CNN Teacher model to achieve an F1-Score above 0.95 on the UNSW-NB15 dataset.
2. **Dimensionality Reduction**: Utilize Random Forest Gini Impurity reduction to identify and select the top 15 most discriminative features.
3. **Knowledge Transfer**: Implement a custom Knowledge Distillation (KD) framework using a Composite Loss Function (Cross-Entropy + KL Divergence) to train compact student models.
4. **Edge Validation**: Evaluate and rank student models based on accuracy, size, and continuous average inference latency ($\tau_{avg}$) under simulated edge conditions.

---

## 🚀 Core Workflow (notebook steps)

> **Note:** the entire pipeline is implemented inside the Jupyter notebook; the repository contains no additional Python modules — files such as saved models or pickles are generated at runtime under the dataset path.

1. **Setup & EDA** – Logically concatenate 4 partitioned CSV files (~2.54M records). Conduct statistical assessments of class imbalance, attack taxonomy, and feature correlation.
2. **Preprocessing & Feature Selection:**  
   - Perform a 70/30 stratified split.
   - Apply Label Encoding to nominal features and Min-Max Scaling to numerical attributes.
   - Reduce the feature space from 49 to the top 15 features using Random Forest importance
3. **Teacher training** – Train a deep 1D-CNN featuring three stacked convolutional blocks with Batch Normalization, Max Pooling, and Dropout (0.5). Optimization is performed using Stratified 10-Fold Cross-Validation and the Adam optimizer.
4. **Distillation Pipeline** – 
   - Deploy five student architectures: R1 RobustTiny, R2 RobustSmall, R3 RobustMedium, R4 RobustDeep, and R5 WideFidelity.
   - Optimize students via soft target learning to mimic the teacher's probability distributions, fine-tuning temperature ($T$) and distillation weight ($\alpha$).
5. **Evaluation & Simulation** – 
   - Rank models using a weighted leaderboard balancing accuracy and efficiency.
   - Utilize a Discrete-Event Simulation environment to replay dataset records as sequential packet arrivals for real-time latency benchmarking.
---

## 🛠️ How to reproduce

1. **Clone repository.**
2. **Prepare the dataset** – Download UNSW-NB15 and place files in `dataset_path`.
3. **Install dependencies** – `pip install pandas numpy seaborn matplotlib scikit-learn tensorflow`.
4. **Run the notebook** – Execute cells sequentially to perform EDA, preprocessing, teacher training, and distillation.

---

## 📊 Important Results

The experimental results validate that Knowledge Distillation successfully bridges the gap between accuracy and efficiency:

- **Optimal Performer (Student R2):** The **RobustSmall** model emerged as the superior choice for deployment. It achieved an average latency ($\tau_{avg}$) of 87.59 ms, actually outperforming the Teacher model (96.94 ms) on the same hardware while maintaining comparable detection accuracy.
- **Extreme Compression (Student R1):** The **RobustTiny** model achieved a 94% reduction in parameter size, shrinking from 0.78 MB (Teacher) to just 0.05 MB.
- **High Fidelity Retention:** ROC analysis confirmed that the best student (R2) retained a high Area Under the Curve (AUC), ensuring minimal false negatives—critical for high-security environments.
- **Throughput:** Distilled models demonstrated stable performance in live traffic simulations, proving their suitability for resource-constrained edge nodes and IoT gateways.

---

## 🛡️ Originality Report

The overall similarity score of the Turnitin report is **11%**.

**Detailed Breakdown:**
- **Internet Sources:** 9%
- **Publications:** 5%
- **Submitted Works (Student Papers):** 7%

The report also indicates that 0% of the content was flagged for missing quotations or citations.

---

## 🧾 References

- UNSW-NB15 Dataset: Moustafa & Slay (2015). https://research.unsw.edu.au/projects/unsw-nb15-dataset
- 1D-CNN Baseline: Based on research by Vibhute et al. (2020). https://www.sciencedirect.com/science/article/pii/S1877050924008871
- Knowledge Distillation: Hinton et al. (2015). https://arxiv.org/abs/1503.02531

---

## 👥 Team & Supervisor

- **Prabhmeet Singh** (Roll: 2022UCM2305)
- **Aryan Jain** (Roll: 2022UCM2330)
- **Radhacharan** (Roll: 2022UCM2365)

> _Under the guidance of Dr. Anand Gupta_

---

## 🔖 Notes

- Paths are set for Google Drive; adjust `dataset_path` for local use.
- The notebook is self‑contained and modular – feel free to export cells to scripts.
- Saved assets (encoders, scaler, models) allow inference without rerunning training.

---