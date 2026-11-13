# RKEM-Net: Label-Free Continual Object Learning from Video

MS thesis code and results. RKEM-Net learns new object instances from video **without labels**. It keeps previously learned objects (avoiding catastrophic forgetting) and stays within a **fixed memory budget**.

## Method at a glance

Frozen ResNet18 encoder → label-free whitening (fitted on stage 1, then frozen) → two-timescale recurrent state (fast + slow) with a boundary gate → episode-level routing → dual memory (short-term candidates, long-term concepts) → sleep-like consolidation after each stage (budget B = 64) → causal concept ID for every frame.

- **RKEM-Net**: used when recording boundaries are known.
- **RKEM-Net (task-free)**: adds a label-free boundary detector (median − c·1.4826·MAD threshold, c = 1.0 chosen on the validation session) for streams without boundaries.

## Protocol

- **Data:** CORe50, 128×128 (50 objects, 10 categories, 11 sessions). Every 5th frame, at most 60 frames per recording.
- **Split:** train sessions 1, 2, 4, 5, 6, 8, 9; **validation session 11 (all tuning)**; **test sessions 3, 7, 10 (used once)**.
- **Protocols:** `six_objects` (5 look-alike plug adapters + 1 phone, 3 stages) and `ten_categories` (one object per category, 5 stages).
- **Runs:** 5 recording orders (seeds 11–55), paired comparisons, Wilcoxon signed-rank test.
- **Metrics:** matched accuracy (Hungarian matching), AUC_matched, ARI/NMI, ACC/BWT, and retention metrics RET_ACC / RET_F on previously seen training frames.

## Main results (CORe50 test, mean of 5 runs)

| Method | Protocol | ACC | AUC_matched | ARI | RET_F |
|---|---|---|---|---|---|
| GDM (Parisi et al., 2018) | ten_categories | 0.632 | 0.443 | 0.579 | 0.057 |
| **RKEM-Net** | ten_categories | **0.819** | **0.642** | **0.891** | **0.000** |
| GDM (Parisi et al., 2018) | six_objects | 0.390 | 0.441 | 0.185 | 0.222 |
| **RKEM-Net** | six_objects | **0.480** | **0.490** | **0.305** | **0.050** |

RKEM-Net beats GDM in 5 of 5 runs on every metric. Without recording boundaries, RKEM-Net (task-free) raises AUC_matched from 0.265 to 0.558 on ten_categories and from 0.304 to 0.437 on six_objects (5 of 5 runs). STAM (Smith et al., IJCAI 2021) could not finish the full protocol within 12 hours on raw pixels; its runtime is reported in `stam_timing.csv`.

## Repository contents

| File | Contents |
|---|---|
| `rkem-net-ms-thesis.ipynb` | Full notebook with all cell outputs (one clean "Save & Run All" run on Kaggle, T4 GPU) |
| `RKEM_v3_evidence.zip` | All result files (CSVs, retention matrices, figures) |

The last section of the notebook, *"Where each thesis result comes from"*, maps every thesis table and graph to its file.

### Naming used in the code

| In code | In thesis |
|---|---|
| v1 | Baseline |
| v2 (variants) | Ablation rows |
| v3 | RKEM-Net |
| v3.1 | RKEM-Net (task-free) |

## Reproduce

1. Open the notebook on Kaggle with a T4 GPU and the CORe50 128×128 dataset attached.
2. Run **Save & Run All**. Outputs are written to `/kaggle/working/rkem_v3/` and packed into `RKEM_v3_evidence.zip`.

**Tools:** Python, NumPy, SciPy, scikit-learn, pandas, PyTorch/torchvision.

## Citation

Sana Akhtar Naseer, *RKEM-Net: Label-Free Continual Object Learning from Video*, MS thesis, COMSATS University Islamabad, Lahore Campus. [Year]
