# Coral reef final benchmark – results folder

- Best point-crop accuracy      : DINOv2-ViT-S/14 (86.70%)
- Best point-crop macro-F1      : DINOv2-ViT-S/14 (0.8312)
- Best sliding-window macro-F1  : DINOv2-ViT-S/14 (0.8049)
- Most accurate reef-cover map  : DINOv2-ViT-S/14 (MAE 3.11 pp)
- Fastest on GPU / CPU          : CoralCNN-SE (4.12 ms) / ShuffleNetV2 (17.5 ms)
- Smallest model                : ShuffleNetV2 (1.26 M params)
- Pareto-optimal (F1 vs GPU ms) : CoralCNN-SE, ResNet18, DINOv2-ViT-S/14
- Recommended trade-off model   : DINOv2-ViT-S/14
- Best per family (crop macro-F1): Transfer learning: ResNet18 (0.819) | CNN from scratch: CoralCNN-SE (0.800) | Vision Transformer: DINOv2-ViT-S/14 (0.831)
- Grad-CAM shown for            : ResNet18 (best TL by validation) · CoralCNN-SE · DINOv2-ViT-S/14
- Second XAI method             : Integrated Gradients + Occlusion for the same three models (faithfulness: ResNet18: best Ins−Del = Occlusion | CoralCNN-SE: best Ins−Del = Occlusion | DINOv2-ViT-S/14: best Ins−Del = Occlusion)
- Imbalance recipe (all models) : R0 · baseline (flip/rot90, mild affine + colour)  [ablation on ResNet18: val score R0 0.813, R1 0.746, R2 0.735, R3 0.758]
- τ objective                   : balanced  (Dead recall: MobileNetV2 0.66, ResNet18 0.71, EffNet-B0 0.67, ShuffleNetV2 0.72, CoralCNN (SE) 0.74, DINOv2 ViT-S/14 0.73)

## Folder map
- `figures/`  – EDA (`eda_*`), training (`train_*`, incl. `train_00_imbalance_recipe_ablation`), comparison (`result_*`, `latency_*`), sliding-window (`sw_*`; `sw_07_*` = class of every window, `sw_08_*` = per-quadrat composition),
                Grad-CAM (`gradcam_*`, `quadrat_report_*` = reference-style 4-panel quadrat reports), second XAI method (`xai_*`: Integrated Gradients + Occlusion) PNGs (200 dpi)
- `tables/`   – every table as report-ready PNG **and** CSV (`table_12_MASTER_comparison` is the headline table; `table_epochs_*` = per-epoch logs)
- `models/`   – `<model>.pt` (best-epoch weights `state_dict` + training history + config; reload with `load_and_predict_example.py`)
- `logs/`     – training histories (JSON) + `training_log_all_models.csv` + `imbalance_ablation.json`
- per-window / per-quadrat exports (in `tables/`): `sliding_window_per_window_predictions.csv.gz` (class + probabilities of EVERY sliding window, every model),
  `sw_per_image_composition_long.csv` / `_wide.csv` (%Healthy · %Dead · %Substrate per quadrat: expert vs every model)
- `coral_model_defs.py`, `load_and_predict_example.py` – reload & use the weights
- `results_master.json` – all metrics

## Protocol notes
- Split: grouped by quadrat image (70/15/15, seed 42); train crops class-balanced, val/test crops natural distribution.
- Headline numbers use ONE forward pass + prior-shift correction τ tuned on the validation set with objective `balanced` (`table_06` shows raw / +τ / +τ+TTA, `table_06b` compares objectives).
- Class imbalance: training crops are class-balanced; all models were trained with recipe **R0** (R0 · baseline (flip/rot90, mild affine + colour)) chosen on validation (`table_03_imbalance_recipe_ablation`).
- Sliding window: 224 px windows, stride 112 px, Gaussian blending (σ=0.35), scored at all expert points of 150 held-out quadrats.
- Latency: single 224×224 crop, fp32, CUDA-synchronised (T4); CPU column uses 2 vCPU.
