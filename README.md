# RKEM-Net: Label-Free Continual Object Learning from Video

**MS thesis, University of Central Punjab, Lahore (2023)** · Sana Akhtar Naseer

RKEM-Net learns to recognise new object instances from video **without any labels**. It keeps the objects it has already learned (no catastrophic forgetting) and stores everything in a **fixed memory budget**. Each frame receives a concept identity **causally**, from past and present frames only. The design pairs a fast-changing short-term memory with a stable long-term memory, and runs a sleep-like consolidation phase between learning stages.

---

## Highlights

- **Outperforms the published GDM dual-memory baseline** (Parisi et al., 2018) on CORe50 in **5 of 5 runs on every metric**, using the same features, protocol and validation-only tuning.
- **No measurable forgetting on ten_categories** (RET_F = 0.000), with accuracy 0.819 and ARI 0.891.
- **Online and label-free, yet above offline clustering.** RKEM-Net's ARI (0.891 and 0.305) exceeds offline k-means given the true number of objects (0.557 and 0.277).
- **Works without recording boundaries.** A one-parameter robust boundary detector raises AUC_matched from 0.265 to 0.558 (ten_categories) and from 0.304 to 0.437 (six_objects) when boundaries are not given.
- **Strict evaluation protocol.** All tuning was done on a validation session, the test sessions were used once, forgetting is measured as retention on previously seen frames, and results are paired comparisons over five recording orders.

---

## Method

```
frame ─► frozen ResNet18 ─► label-free whitening ─► two-timescale recurrent state ─► episode routing ─► dual memory ─► concept ID
                            (fit on stage 1,         fast f_t + slow s_t,            every 10 frames     STM: plastic candidates
                             then frozen)            boundary gate                                       LTM: consolidated concepts
                                                                                                              │
                                                                         after each stage: sleep consolidation ◄┘
                                                                         join · merge · promote · anchor · budget (B = 64)
```

| Component | Role |
|---|---|
| **Frozen encoder + whitening** | ResNet18 features, whitened with statistics from stage 1 only and then frozen. This removes *hubness* (a few prototypes becoming everyone's nearest neighbour) and keeps the feature space stationary, so stored memories keep their meaning. |
| **Two-timescale recurrent state** | A fast state tracks the current view and a slow state summarises the recording. They are combined as h_t = [√(1−ρ) f_t, √ρ s_t], so ρ trades current appearance against history. A similarity gate k(z_t, f_{t−1}) < β marks a change of object. |
| **Episode-level routing** | Every 10 frames, the current episode is scored against each long-term concept: ev(c) = (1−w)·mean_t max_j k(h_t, p_cj) + w·max_k k(d_e, key_ck). A confident match updates that concept; otherwise the episode goes to short-term memory. |
| **Dual memory** | Short-term memory (STM) holds plastic candidate concepts. Long-term memory (LTM) holds consolidated concepts whose prototypes slow down with use (η_j = η₀ / (1 + λ n_j)) and are tied to elastic anchors. |
| **Sleep consolidation** | After each stage: join STM candidates to LTM (affinity ≥ 0.7), merge the rest by average linkage (0.45), promote candidates seen in ≥ 2 episodes, anchor, and enforce the budget B = 64. |
| **Task-free extension** | For streams without boundaries, β = median − c·1.4826·MAD of gate similarities on one unlabelled stage-1 stream (c = 1.0, chosen on the validation session). |

---

## Experimental protocol

| | |
|---|---|
| **Dataset** | CORe50, 128×128 (50 objects, 10 categories, 11 sessions); every 5th frame, at most 60 frames per recording |
| **Split** | Train sessions 1, 2, 4, 5, 6, 8, 9 · **Validation session 11 (all tuning)** · **Test sessions 3, 7, 10 (used once)** |
| **Protocols** | `ten_categories`: one object per category, 5 stages · `six_objects`: 5 look-alike plug adapters + 1 phone, 3 stages |
| **Runs** | 5 recording orders (seeds 11–55), paired comparisons, Wilcoxon signed-rank test |
| **Metrics** | Matched accuracy (one-to-one Hungarian matching fitted on training frames), AUC_matched (mean over stages), ARI / NMI over all seen objects, ACC / BWT, and RET_ACC / RET_F (retention and forgetting on previously seen training frames) |

With n = 5 paired runs, the smallest two-sided Wilcoxon p-value attainable is 0.0625, so wins are reported as "better in 5 of 5 runs" rather than as p < 0.05.

---

## Results

All results are on the CORe50 test sessions, as the mean of 5 recording orders.

### Comparison with the published baseline

| Method | Protocol | ACC ↑ | AUC_matched ↑ | ARI ↑ | RET_F ↓ |
|---|---|---|---|---|---|
| GDM (Parisi et al., 2018) | ten_categories | 0.632 | 0.443 | 0.579 | 0.057 |
| **RKEM-Net** | ten_categories | **0.819** | **0.642** | **0.891** | **0.000** |
| GDM (Parisi et al., 2018) | six_objects | 0.390 | 0.441 | 0.185 | 0.222 |
| **RKEM-Net** | six_objects | **0.480** | **0.490** | **0.305** | **0.050** |

GDM was run from the authors' official code ([github.com/giparisi/GDM](https://github.com/giparisi/GDM)) on the same features and protocol, with its threshold tuned on validation session 11. STAM (Smith et al., IJCAI 2021) was also run from the official code on raw 64×64 pixels, but the full protocol did not finish within 12 hours. Its measured runtime is in [`results/stam_timing.csv`](results/stam_timing.csv).

**Reference points.** Offline k-means given the true number of objects reaches ARI 0.557 (ten_categories) and 0.277 (six_objects). A supervised 1-NN classifier reaches 0.911 and 0.630.

### Streams without recording boundaries

| Protocol | Metric | RKEM-Net, no boundaries | **RKEM-Net (task-free)** | RKEM-Net, true boundaries |
|---|---|---|---|---|
| ten_categories | AUC_matched | 0.265 | **0.558** | 0.642 |
| ten_categories | ARI | 0.006 | **0.861** | 0.891 |
| ten_categories | ACC | 0.129 | **0.781** | 0.819 |
| six_objects | AUC_matched | 0.304 | **0.437** | 0.490 |

The task-free detector improves results in 5 of 5 runs. It is meant only for streams without boundaries: used inside known recordings, it lowers AUC_matched by 0.07 to 0.08.

### Ablations

- **Short-term memory** is the most important component. Removing it reduces AUC_matched by 0.185 to 0.268, depending on the protocol.
- **Episode keys** add 0.145 to 0.185 AUC_matched.
- **Recording boundaries** matter: running without them costs 0.377 (ten_categories) and 0.186 (six_objects).
- The margin gate and the budget at B = 64 were not active at this scale (change of 0.000), and frame order had little effect.

---

## Limitations

- **Look-alike instances remain hard.** On six_objects (five similar plug adapters), absolute scores are modest, and RKEM-Net's ARI and forgetting (RET_F 0.050) are worse than one of its own ablation variants (RET_F 0.007).
- **Whitening favours coarse distinctions.** It removes hubness but can weaken the fine-grained cues that separate near-identical objects.
- **The task-free detector over-fragments.** It produces 33 concepts for 10 objects. Accuracy stays high, but the concept count is inflated.
- **Small scale.** The protocols use 6 and 10 objects with 5 runs each. Scaling to many more objects and longer streams is future work.

---

## Repository contents

```
rkem-net-continual-learning/
├── README.md
├── rkem-net-ms-thesis.ipynb    # full notebook with all cell outputs (one clean "Save & Run All" on Kaggle, T4 GPU)
├── RKEM_v3_evidence.zip        # every result file: per-stage metrics, retention matrices, ablations, baselines
└── results/                    # key CSVs from the zip, readable in the browser
    ├── ten_categories_final_metrics_per_run.csv
    ├── ten_categories_summary.csv
    ├── six_objects_final_metrics_per_run.csv
    ├── six_objects_summary.csv
    ├── comparison_table.csv    # RKEM-Net vs GDM
    └── stam_timing.csv         # STAM runtime
```

The last section of the notebook, *"Where each thesis result comes from"*, maps every thesis table and figure to its file.

**Naming in the code:** `v1` = Baseline · `v2` variants = ablation rows · `v3` = RKEM-Net · `v3.1` = RKEM-Net (task-free).

---

## Reproducing the results

1. Open `rkem-net-ms-thesis.ipynb` on [Kaggle](https://www.kaggle.com/) with a T4 GPU and the CORe50 128×128 dataset attached.
2. Run **Save & Run All**. Outputs are written to `/kaggle/working/rkem_v3/` and packed into `RKEM_v3_evidence.zip`.

**Built with:** Python, NumPy, SciPy, scikit-learn, pandas, PyTorch / torchvision.

---

## Citation

```bibtex
@mastersthesis{naseer2023rkemnet,
  author = {Sana Akhtar Naseer},
  title  = {{RKEM-Net}: Label-Free Continual Object Learning from Video},
  school = {University of Central Punjab},
  address = {Lahore, Pakistan},
  year   = {2023},
  type   = {{MS} thesis}
}
```

## References

- V. Lomonaco and D. Maltoni, "CORe50: a new dataset and benchmark for continuous object recognition," *CoRL*, 2017.
- G. I. Parisi, J. Tani, C. Weber and S. Wermter, "Lifelong learning of spatiotemporal representations with dual-memory recurrent self-organization," *Frontiers in Neurorobotics*, 2018.
- J. Smith, C. Taylor, S. Baer and C. Dovrolis, "Unsupervised progressive learning and the STAM architecture," *IJCAI*, 2021.

## Contact

Sana Akhtar Naseer · [LinkedIn](https://www.linkedin.com/in/sana-naseer-68a41114a) · [GitHub](https://github.com/SanaAkhtarNaseer)
