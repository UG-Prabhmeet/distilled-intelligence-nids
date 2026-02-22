# 📘 Distilled Intelligence for Network Intrusion Detection (BTP)

A research‑oriented pipeline that combines **deep learning** and **knowledge distillation** to detect network attacks using the UNSW‑NB15 dataset.  
This Bachelor’s Thesis project implements a teacher‑student training framework and provides tools for evaluation and real‑time simulation.

---

## 🎯 Objectives

1. Load & preprocess the UNSW‑NB15 network traffic dataset.
2. Train a large **teacher model** (Conv1D CNN) via 10‑fold cross‑validation.
3. Distil knowledge into several compact **student models** with varying architectures.
4. Evaluate accuracy, size, inference speed and deploy a streaming simulator to compare models in “real‑time” traffic.

---

## 🚀 Core Workflow (notebook steps)

> **Note:** the entire pipeline is implemented inside the Jupyter notebook; the repository contains no additional Python modules — files such as saved models or pickles are generated at runtime under the dataset path.

1. **Setup & EDA** – load features, merge CSVs, visualise class balance, correlation, key features.
2. **Preprocessing**  
   • Label‑encoding & scaling  
   • Feature selection via RandomForest  
   • Save reduced datasets and encoders.
3. **Teacher training** – Conv1D model trained with stratified 10‑fold CV; best weights saved for each fold.
4. **Distillation** – custom `Distiller` wraps teacher & student; 5 student configs trained across folds.
5. **Evaluation** – metrics (accuracy, recall, F1 etc.), ROC curves, confusion matrices, size/throughput comparison.
6. **Simulation** – `LiveTrafficSimulator` streams random packets from raw CSVs, preprocesses using saved assets, and compares live predictions/latency of teacher vs. students.

---

## 🛠️ How to reproduce

1. **Clone repository.**
2. **Prepare the dataset** – download UNSW‑NB15 (see link below) and place the CSVs and feature file under a directory; set `dataset_path` accordingly (default in the notebook is `/content/drive/MyDrive/BTP`).
3. **Install dependencies** (e.g., via pip):
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
    ```
    Colab already includes most packages.
4. **Run the notebook** – launch `nids-knowledge-distillation.ipynb` in Colab or VS Code and execute cells sequentially:
    - Section **1. Setup & EDA** through **6. Save Datasets**.
    - Train the teacher; then run the distillation cells one configuration at a time (each cell calls `run_kd_training(cfg)`).
    - Perform evaluation and, if desired, execute the real‑time simulator cells.
5. Feel free to adapt or export notebook code to standalone scripts; hyper‑parameters and architectures are defined near the top of the notebook.

---

## 📊 Results & Findings

- Teacher model achieves high recall but is large and slow.
- Distilled students trade a small drop in accuracy for significant reductions in size & inference latency.
- The “best” student (by recall and latency) is selected automatically in evaluation and simulation scripts.
- Full project report (`Grp20_ProjectReport.pdf`) details experiments, analysis and conclusions.

---

## 🧾 References

- [UNSW‑NB15 dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) (download required for preprocessing steps)
- Knowledge distillation literature (Hinton et al.)
- TensorFlow / Keras for model implementation
- Scikit‑learn for preprocessing and cross‑validation

---

## 🔖 Notes

- Paths are set for Google Drive; adjust `dataset_path` for local use.
- The notebook is self‑contained and modular – feel free to export cells to scripts.
- Saved assets (encoders, scaler, models) allow inference without rerunning training.

---

## 👥 Team & Supervisor

- **Prabhmeet Singh** (Roll: 2022UCM2305)
- **Aryan Jain** (Roll: 2022UCM2330)
- **Radhacharan** (Roll: 2022UCM2365)

> _Under the guidance of Dr. Anand Gupta_

---

✔️ **Ready for extension** – plug in new student architectures, try other datasets, or deploy the top model in a network appliance.
