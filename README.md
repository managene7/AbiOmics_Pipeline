# AbiOmics: An end-to-end pipeline to train AI models for the classification of plant abiotic stress using transcriptomic profiling

## Abstract
Abiotic stress is a primary constraint on global crop productivity, reducing potential yields by up to 70%. While advanced sensing and imaging technologies can detect physiological decline, they often fail to discriminate among specific stressor types, a critical requirement for precise agricultural management. Transcriptomic profiling offers a high-dimensional solution to this diagnostic bottleneck, given the rapid and sensitive shifts in gene expression that occur upon stress onset. In this study, we developed a robust machine learning framework that can identify and distinguish among four major abiotic stressors: salt, cold, heat, and drought. Using a curated metadata collection of 1,243 Arabidopsis leaf transcriptomes from public databases, we identified 320 stress-specific marker genes by differential expression and confirmed their functions using Gene Ontology (GO) enrichment analyses. A single-layer perceptron was trained on these features using five-fold cross-validation. The model achieved a macro-average F1-score of 0.90 and an overall accuracy of 91% during cross-validation, with 93% accuracy on an independent test set. Furthermore, the model demonstrated the capacity to generalize to multi-stress conditions, successfully identifying both heat and salt signatures in combinatorial treatment samples. Our results provide proof of concept for the first transcriptomic-based diagnostic tool capable of discriminating among abiotic stressors. This approach offers a foundation for high-confidence stress labeling in AI-driven crop management and provides a quantitative molecular framework for precision breeding under global climate change.

## Installation

```
pip install torch pydeseq2 shap scikit-learn pandas numpy matplotlib seaborn pyyaml tqdm

```

---

## Data Format

Place exactly **two files** in `Data/`:

```
TPM_all_samples.csv    # one TPM matrix for ALL samples
                       # rows = genes, columns = sample IDs
                       # first column = Gene_ID (configurable)

sample_labels.csv      # sample ID → label mapping
                       # two comma-separated columns, NO header
```

### TPM file (`tpm_file` in config)

Standard CSV with genes as rows and sample IDs as column headers:

```
Gene_ID,    SampleA, SampleB, SampleC, ...
AT1G01010,  12.3,    0.0,     45.6,    ...
AT1G01020,  8.7,     23.1,    0.4,     ...
```

### ID / label file (`id_file` in config)

No header. Two columns: `sample_ID, label`.  
The label must be either the `control_label` (default: `"control"`) or one of the `stress_types` declared in `config.yaml`:

```
SampleA, control
SampleB, control
SampleC, Salt
SampleD, Salt
SampleE, Cold
SampleF, Cold
SampleG, Heat
SampleH, Drought
```

Labels are **case-sensitive** and must exactly match `stress_types` in `config.yaml`.

---

## Running the Pipeline

```bash
# Full pipeline (DESeq2 → train → evaluate → SHAP)
python run_pipeline.py

# Skip DESeq2 (reuse marker_genes.txt from a previous run — much faster)
python run_pipeline.py --skip-deseq

# Skip training too (evaluate + SHAP on existing checkpoints)
python run_pipeline.py --skip-deseq --skip-training

# SHAP only
python run_pipeline.py --shap-only

# Load data + DESeq2 only, no training
python run_pipeline.py --dry-run

# Custom config (e.g., for a different organism)
python run_pipeline.py --config experiments/rice/config.yaml

# Write a log file
python run_pipeline.py --log-file logs/run_001.log
```

---

## Adapting to a New Dataset

1. **Edit `config/config.yaml`** — change stress names, file paths, sample counts,
   DEG thresholds, and model hyperparameters.
2. **Put your data files in `Data/`** following the naming convention above.
3. **Run `python run_pipeline.py`** — everything else is automatic.

### Example: adding a new stress type

```yaml
data:
  stress_types: ["Salt", "Cold", "Heat", "Drought", "Waterlogging"]
```

Then add the corresponding files to `Data/` and re-run.

---

## Output Files

All outputs are written to `Export/` (configurable):

 `marker_genes.txt`  Final marker gene list used as model input 
 `{stress}_DEG_{up/down}.txt`  All DEGs per stress 
 `{stress}_DEG_{up/down}_unique.txt`  Stress-exclusive DEGs 
 `{stress}_DEG_{up/down}_sub.txt`  Subsampled marker DEGs 
 `{stress}_DEG.csv`  Full DESeq2 results table 
 `confusion_matrix_avg.pdf`  Averaged confusion matrix across all CV folds 
 `SHAP_summary_{class}.pdf`  Beeswarm plot — single checkpoint 
 `SHAP_top_genes_all_classes.pdf`  Mean |SHAP| bar chart — single checkpoint 
 `SHAP_values_{class}.csv`  Per-sample SHAP values — single checkpoint 
 `SHAP_avg_summary_{class}.pdf`  Beeswarm — CV-averaged SHAP 
 `SHAP_avg_top_genes_all_classes.pdf`  Bar chart — CV-averaged SHAP 
 `SHAP_avg_values_{class}.csv`  Per-sample averaged SHAP values 
 `SHAP_gene_ranking_all_classes.csv`  Gene importance ranking, all classes 

Loss curves are saved to `Loss_graph/` (one per CV fold).

