# Supply Chain Risk — Distributed ML Pipeline

End-to-end PySpark pipeline on Databricks that scales the public **DataCo Smart Supply Chain** dataset to ~100 GB in Delta Lake, enriches it with graph-derived risk signals, and discovers the most predictive features for late-delivery risk using a genetic algorithm. The full pipeline — ingestion, storage tuning, graph analytics, clustering, feature discovery, and reporting — runs on a single Databricks cluster backed by Azure Data Lake Storage (ADLS) Gen2.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Datasets](#datasets)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
6. [Tech Stack](#tech-stack)
7. [Prerequisites](#prerequisites)
8. [Setup](#setup)
9. [How to Run](#how-to-run)
10. [Outputs](#outputs)
11. [Project Artefacts](#project-artefacts)
12. [Design Notes](#design-notes)
13. [Contributors](#contributors)

---

## Motivation

Late deliveries are the single largest driver of customer churn and margin erosion in e-commerce supply chains. This project asks two questions:

1. **Does data scale matter?** Do distributed Spark/Delta techniques actually beat single-node baselines as the dataset grows from 1 GB to 100 GB?
2. **Do network-topology features add predictive value?** When we model hubs (Market / Region / Country) as graph vertices connected by shared customer flows, do centrality signals like **PageRank** improve clustering and classification beyond classical transactional features?

The pipeline is engineered to answer both empirically, with MLflow experiments capturing every run.

---

## Architecture

The pipeline follows a **medallion architecture** on ADLS Gen2:

```
Bronze  →  Silver (Delta, Z-ordered)  →  Gold (analytics + ML outputs)
   |               |                               |
   | raw CSVs      | 100GB synthetic table         | PageRank, clusters,
   | (Kaggle)      | partitioned Market/Region     | GA catalog, figures
```

```
            +--------------------------------------+
  Bronze -> | NB01  Synthetic data + DQ gate       | -> Silver (Delta)
            +--------------------------------------+
                                |
                                v
            +--------------------------------------+
            | NB02  OPTIMIZE + Z-ORDER + VACUUM    |
            +--------------------------------------+
                                |
                                v
            +--------------------------------------+
            | NB03  GraphFrames: PageRank + motifs | -> Gold (pagerank,
            +--------------------------------------+      motifs)
                                |
                                v
            +--------------------------------------+
            | NB04  K-Means: transactional vs      | -> Gold (clusters
            |       graph-augmented                |      x 2 models)
            +--------------------------------------+
                                |
                                v
            +--------------------------------------+
            | NB05  Genetic Algorithm feature      | -> Gold (feature
            |       discovery + stability check    |      catalog,
            +--------------------------------------+      importances,
                                |                        stability)
                                v
            +--------------------------------------+
            | NB06  Report figures + ADLS upload   | -> Gold (report
            +--------------------------------------+      figures)
```

All runs are tracked under the MLflow experiment `/supply-chain-pipeline`.

---

## Repository Layout

```
supply-chain-risk-ml-project/
├── Databricks Notebook/
│   ├── Notebook 01 — Synthetic Data Generation + Quality Gate.ipynb
│   ├── Notebook 02 — Storage Optimization (Z-Order).ipynb
│   ├── Notebook 03 — Graph Analytics.ipynb
│   ├── Notebook 04 — Risk Clustering.ipynb
│   ├── Notebook 05 — Genetic Algorithm Feature Discovery.ipynb
│   └── Notebook 06 — Report Outputs.ipynb
├── Dataset/
│   ├── DataCoSupplyChainDataset.csv      (~91 MB, 180,519 rows)
│   └── tokenized_access_logs.csv         (~91 MB)
└── .gitignore.txt
```

---

## Datasets

**Source:** [DataCo Smart Supply Chain for Big Data Analysis (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

| File | Rows | Purpose |
|---|---|---|
| `DataCoSupplyChainDataset.csv` | 180,519 | Orders, customers, shipping, profit — the feature-rich base table |
| `tokenized_access_logs.csv` | ~1M | Anonymised clickstream / session logs (kept as a companion table) |

The base table is replicated **600×** in Notebook 01 with Gaussian noise, timestamp shifts, and FK-preserving ID mutation to produce a realistic ~108M-row / ~100 GB Delta table without synthesising out-of-distribution values.

---

## Pipeline Walkthrough

### Notebook 01 — Synthetic Data Generation + Quality Gate

- Reads both raw CSVs from Bronze.
- Applies a per-batch variance function: σ=3 % Gaussian noise on 15 numeric columns, cyclic 0–180-day timestamp shift, FK-preserving offset on IDs, 4 % label flip on `Late_delivery_risk`, 2 % order-status randomisation.
- Unions 600 batches, repartitions to 200, writes to the Silver Delta table partitioned by **Market** and **Order_Region**.
- Runs a **Data Quality Gate** with 12 rules (negative shipping days, null IDs, real-vs-scheduled shipping sanity, etc.); failing rows are routed to a Dead-Letter Queue.
- Sanitises column names (Delta rejects spaces / parentheses) and registers `supply_chain_db.supply_chain_100gb`.

### Notebook 02 — Storage Optimization (Z-Order)

- Runs `OPTIMIZE … ZORDER BY (Late_delivery_risk, Order_Country)` to co-locate rows the downstream models filter on.
- `VACUUM` with 168-hour retention (keeps Time Travel).
- Reports `numFiles`, `sizeInBytes`, and partition distribution by Market / Order_Region.

### Notebook 03 — Graph Analytics

- Builds **hub vertices** keyed as `Market|Region|Country` with aggregated transactional metrics (avg delay risk, total sales, unique customers, avg ship-days, delay gap).
- Builds **real edges** from order flows: two hubs are connected if customers placed orders at both. Edge weight = shared-customer count; edge risk = source-hub delay risk.
- Constructs a `GraphFrame`, runs **PageRank** (`resetProbability=0.15`, `maxIter=10`).
- Runs a motif query `(a)->(b)->(c); !(a)->(c)` to find **triangular transshipment chains** of dependent hubs, excluding self-loops.
- Writes `gold/pagerank_results` and `gold/high_risk_paths`.

### Notebook 04 — Risk Clustering

Trains **two K-Means models side by side** to prove graph features add value:

| Model | Features |
|---|---|
| A — Transactional only | 9 features: avg_delay_risk, avg_discount_rate, avg_profit, avg_sales, order_volume, avg_real_ship_days, avg_delay_gap, avg_profit_ratio, avg_benefit |
| B — Graph-augmented | Model A + 4 graph features: avg_hub_pagerank, max_hub_pagerank, avg_hub_network_risk, **centrality_weighted_risk** |

- Elbow search over `k ∈ [2, 8]`, selects best `k` by **silhouette** score.
- Logs both models to MLflow (nested runs per `k`) and writes the predictions for each to Gold.
- Final cell produces human-readable cluster profiles: `LOW / MEDIUM / HIGH RISK` × `bottleneck / peripheral hub`.

### Notebook 05 — Genetic Algorithm Feature Discovery

Three phases, engineered to directly address the "why use 100 GB just to pick 5 features?" criticism:

- **Phase A — Discovery (1 GB sample, ~0.01 fraction).** GA with population 30, 10 generations, 30 % elite, 25 % mutation. Chromosomes are feature subsets (2–12 features) evaluated by 5-fold AUC of a RandomForestClassifier predicting `Late_delivery_risk`.
- **Phase B — Independent validation (Pearson).** Recomputes `|corr|` of each candidate against the label on a fresh sample. Measures overlap between top-Pearson and GA-selected feature sets.
- **Phase C — Stability (1 GB → 100 GB).** Trains the chosen feature set at 0.01, 0.05, 0.10, 0.25, 0.50, and 1.00 fractions, logs AUC at each scale to confirm features **generalise**.

Outputs: `gold/feature_catalog`, `gold/feature_importances`, `gold/feature_stability`.

### Notebook 06 — Report Outputs

Generates eight report figures locally (`matplotlib` / `networkx`) and uploads them to `gold/report_figures/` in ADLS via `azure-storage-blob`:

1. `fig1_spark_scalability.png` — Spark scalability benchmark
2. `fig2_baseline_comparison.png` — Distributed vs single-node crossover
3. `fig3_risk_map.png` — Geographic supply-chain risk map
4. `fig4_clusters.png` — Risk cluster scatter plots
5. `fig5_ga.png` — GA convergence + feature importances
6. `fig6_stability.png` — AUC stability 1 GB → 100 GB
7. `fig7_pearson_vs_ga.png` — Pearson vs GA validation
8. `fig8_worker_scaling.png` — Worker scaling efficiency

---

## Tech Stack

- **Compute:** Databricks Runtime 13.x / 14.x LTS — cluster node type `Standard_D4s_v3`
- **Storage:** Azure Data Lake Storage Gen2 (`abfss://supplychain-data@adlssupplychain.dfs.core.windows.net`)
- **Data format:** Delta Lake (bronze / silver / gold), Hive metastore (`supply_chain_db`)
- **ML / analytics:** PySpark ML (`KMeans`, `RandomForestClassifier`, `VectorAssembler`, `StandardScaler`), GraphFrames, scikit-learn, NetworkX
- **Experiment tracking:** MLflow (experiment: `/supply-chain-pipeline`)
- **Secrets:** Databricks secret scope `adls-scope`, key `adls-account-key`
- **Reporting:** matplotlib, azure-storage-blob

---

## Prerequisites

- Databricks workspace on **Azure** (the notebooks use the Azure-native ADLS Gen2 driver and `dbutils.secrets`).
- ADLS Gen2 storage account named `adlssupplychain` with container `supplychain-data` (or update the constants at the top of every notebook).
- A Databricks secret scope:
  ```bash
  databricks secrets create-scope --scope adls-scope
  databricks secrets put --scope adls-scope --key adls-account-key
  ```
- A Databricks cluster with:
  - **GraphFrames** installed (Maven coordinate `graphframes:graphframes:0.8.3-spark3.4-s_2.12` or matching your DBR).
  - Libraries: `mlflow`, `azure-storage-blob`, `networkx`, `scikit-learn`, `matplotlib`.
- An MLflow experiment at `/supply-chain-pipeline` (created automatically on first run).

---

## Setup

1. **Clone the repo** into your Databricks workspace (Repos → Add Repo) or import notebooks manually.
2. **Upload raw CSVs** from `Dataset/` into the Bronze layer:
   ```
   abfss://supplychain-data@adlssupplychain.dfs.core.windows.net/bronze/DataCoSupplyChainDataset.csv
   abfss://supplychain-data@adlssupplychain.dfs.core.windows.net/bronze/tokenized_access_logs.csv
   ```
3. **Configure secrets** (see Prerequisites).
4. **Attach the cluster** with GraphFrames + required libraries to every notebook.

If you are deploying to a different storage account or container, update these constants at the top of each notebook:

```python
STORAGE_ACCOUNT = "adlssupplychain"
CONTAINER       = "supplychain-data"
```

---

## How to Run

Run the notebooks **in order**. Each one depends on Delta tables produced by the previous one.

| # | Notebook | Approx. runtime on `D4s_v3` | Output tables |
|---|---|---|---|
| 01 | Synthetic Data Generation + Quality Gate | 30–60 min | `supply_chain_100gb`, DLQ |
| 02 | Storage Optimization (Z-Order) | 10–20 min | (rewrites Silver in place) |
| 03 | Graph Analytics | 15–30 min | `pagerank_results`, `high_risk_paths` |
| 04 | Risk Clustering | 20–40 min | `risk_clusters_transactional`, `risk_clusters_graph_augmented` |
| 05 | Genetic Algorithm Feature Discovery | 20–40 min | `feature_catalog`, `feature_importances`, `feature_stability` |
| 06 | Report Outputs | 10–20 min | 8 PNGs in `gold/report_figures/` |

You can also chain them together as a Databricks **Workflow / Job** with sequential tasks — the table names and ADLS paths are stable across notebooks.

---

## Outputs

### Delta tables (`supply_chain_db`)

| Table | Produced by | Description |
|---|---|---|
| `supply_chain_100gb` | NB01 → NB02 | Cleaned, Z-ordered, ~108M-row silver table |
| `pagerank_results` | NB03 | Hub centrality scores + hub aggregates |
| `high_risk_paths` | NB03 | Triangular transshipment motifs |
| `risk_clusters_transactional` | NB04 | Cluster assignments — baseline model |
| `risk_clusters_graph_augmented` | NB04 | Cluster assignments — graph-augmented model |
| `feature_catalog` | NB05 | All GA chromosomes ranked by AUC |
| `feature_importances` | NB05 | RF importances on the winning feature set |
| `feature_stability` | NB05 | AUC across 1 GB → 100 GB scales |

### MLflow runs

Under experiment `/supply-chain-pipeline`, tagged by team member, compute type, and model type. Key metrics logged:

- `pagerank_time_sec`, `motif_count`
- `silhouette_score` (per `k`, per model)
- `best_auc`, `avg_auc`, `diversity_pct` per GA generation
- `auc_*` per stability fraction

---

## Project Artefacts

- **Report figures** — PNG files under `gold/report_figures/` in ADLS. Download via Azure Portal or `azcopy`.
- **Cluster profiles** — printed at the end of Notebook 04; also persisted in the cluster Delta tables for SQL analysis.
- **Feature catalog** — the top chromosomes ranked by AUC in `feature_catalog`. The winning feature set is used downstream for any production risk model.

---

## Design Notes

- **Why synthetic replication instead of fresh data?** The 180 K base dataset is feature-rich but too small to exercise distributed-Spark bottlenecks. Replicating with per-batch noise preserves realistic marginal distributions while generating a dataset large enough that `OPTIMIZE`, Z-Order, and shuffle behaviour genuinely matter.
- **Why edges built from shared customers?** Statistical-similarity edges (e.g. region-region cosine similarity) collapse to trivial clusters. Shared-customer edges capture **real supply dependency** — a disruption in one hub actually affects a shared downstream customer base.
- **Why two K-Means models?** A single model cannot prove graph features are worth their engineering cost. Running the transactional baseline and the graph-augmented model side-by-side, with identical preprocessing and identical `k` search, isolates the marginal contribution of centrality features.
- **Why a GA on 1 GB and not 100 GB?** Feature discovery is an expensive combinatorial search; running it on the full dataset would be wasteful and non-generalising. Phase C (stability at 1 → 100 GB) is what proves the discovered feature set is stable at production scale.
- **Z-Order choice:** `Late_delivery_risk` is the label and the most-filtered column downstream; `Order_Country` is the most-filtered categorical. Z-ordering by these two gives the downstream models highly skip-friendly file layout.

---

## Contributors

| Member | Notebooks | Area |
|---|---|---|
| Chinmaye | 01, 04 | Synthetic data generation + Quality Gate; risk clustering |
| Sudheesh U | 02, 03 | Storage optimization (Z-Order); graph analytics |
| Mohamed Shiyas | 05, 06 | Genetic algorithm feature discovery; report outputs |

---

## License

Educational / coursework project. The underlying DataCo dataset is distributed under its original Kaggle terms.
