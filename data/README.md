# Dataset information

The images and annotations are **not stored in this repository** (≈ 18 GB). They are downloaded automatically by the notebook from Kaggle.

| Item | Value |
|---|---|
| Source | Kaggle – [`jxwleong/coral-reef-dataset`](https://www.kaggle.com/datasets/jxwleong/coral-reef-dataset) (public, CoralNet-style point annotations) |
| Main origin of the images | Moorea Coral Reef LTER (MCR-LTER) photo-quadrats, i.e. the *Moorea Labelled Corals* collection (fringing reef, 10 m and 17 m outer reef, 2008–2010) |
| Additional images | a smaller set of quadrats from other sources with sparser annotation (≈ 19 points / image; file names `i*.png`, `o*.png`) |
| Annotation format | `combined_annotations_remapped.csv` – one row per expert point: `Name` (image file), `Row`, `Column` (pixel coordinates), `Label` (raw CoralNet label) |
| Size after cleaning | **404,255 valid points on 2,455 quadrat images** (mean 164.7 points / image) |
| Task | 3-class benthic health: **Healthy Coral**, **Dead / Degraded**, **Substrate / CCA** |

## Download

The notebook does this for you (cell 2):

```python
import kagglehub
path = kagglehub.dataset_download("jxwleong/coral-reef-dataset")   # anonymous download; asks for kaggle.json only if it fails
```

To use a local copy instead, set `CORAL_DATASET_PATH=/path/to/coral-reef-dataset` before running the notebook.

Kaggle CLI alternative:

```bash
pip install kaggle
kaggle datasets download -d jxwleong/coral-reef-dataset -p data/raw --unzip
```

Please respect the licence and attribution terms given on the Kaggle dataset page and of the original MCR-LTER / CoralNet data providers.

## Label taxonomy (40 raw labels → 3 classes)

The full mapping is in [`label_mapping.csv`](label_mapping.csv).

| Health class | Raw labels |
|---|---|
| **Healthy Coral** (27) | porites, pocillopora, montipora, pavona, acropora, millepora, montastraea, leptastrea, goniastrea, fungia, astreopora, gardineroseris, herpolitha, lobophyllia, favia, platygyra, echinopora, cyphastrea, acanthastrea, psammocora, stylophora, favites, leptoseris, sandolitha, tuba, soft_coral, soft |
| **Dead / Degraded** (6) | dead_coral, broken_coral_rubble, turf, macroalgae, algae, green_fleshy_algae |
| **Substrate / CCA** (4) | crustose_coralline_algae, hard_substrate, rock, sand |
| *Ignored* (3) | off, bad, dark |

Algae are grouped with dead coral because algal overgrowth of coral framework is a primary indicator of reef degradation.

## Split and sampling

* **Grouped by quadrat image** (70 / 15 / 15, stratified by each image's majority class, seed 42), so no image contributes points to more than one split.
* Training crops are **class-balanced** (8,000 per class); validation and test crops keep the **natural reef distribution**.
* Natural class prior of the training split: Healthy 0.170 · Dead/Degraded 0.168 · Substrate/CCA 0.662 (used for prior-shift correction).

| Split | Quadrat images | Valid points | Crops used | Healthy | Dead/Degraded | Substrate/CCA | Distribution |
|---|---:|---:|---:|---:|---:|---:|---|
| Train | 1,718 | 285,305 | 24,000 | 8,000 | 8,000 | 8,000 | balanced |
| Val | 368 | 59,434 | 5,000 | 885 | 913 | 3,202 | natural |
| Test | 369 | 59,516 | 6,000 | 928 | 1,031 | 4,041 | natural |

(also in [`split_summary.csv`](split_summary.csv))

## Why point-centred crops?

Quadrats are mixed scenes: 68.1 % of images contain all three classes and only 24.8 % are ≥ 90 % pure, so a single label per image is wrong for most pixels. Every model is therefore trained on **224 × 224 crops centred on each expert point** (reflection-padded at the borders) after gray-world white balance + LAB-CLAHE underwater colour correction. See `results/tables/table_01_label_purity.csv` and `results/figures/eda_03_label_purity.png`.
