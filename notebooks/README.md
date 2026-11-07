# Notebook

`coral_reef_final_pipeline.ipynb` is the complete, end-to-end pipeline used for every result in this repository:
setup → Kaggle download → taxonomy + grouped split → EDA → underwater colour correction → crop cache → GPU augmentation →
model zoo → training of all six models → point-crop evaluation (raw / +τ / +τ+TTA) → efficiency → sliding-window evaluation →
stride study → Grad-CAM → master comparison, saved weights and download bundle.

Open in Colab: https://colab.research.google.com/drive/1mITD_pCL3SODsILwt3VUXWFDiemTLxzA

Run with **Runtime → Change runtime type → T4 GPU**, then **Run all** (profile `full`, ≈1.5–2.5 h including the 18 GB download).
