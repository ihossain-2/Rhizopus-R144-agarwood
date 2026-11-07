# Rhizopus isolate R144 – growth, enzyme and discoloration analysis

Analysis code for the manuscript:

> *Rhizopus* isolate R144 as an inducer of resinous discoloration in *Aquilaria malaccensis*: model-based niche, enzyme profile and anisotropic expansion kinetics *in planta*.

## Contents

| Path | Description |
|---|---|
| `notebooks/R144_sectioned_Colab.ipynb` | Step-by-step Google Colab notebook (statistics → growth models → kinetics → figures) |
| `scripts/R144_analysis.py` | Full analysis script that writes all tables and `results.json` |
| `requirements.txt` | Python packages |

## Data

The raw data are not included in this repository. They are available from the corresponding author on reasonable request. To run the code, place the data file `Tur_compiled_analysis_ready.xlsx` in a folder named `data/` (local run) or upload it when Cell 1 asks for it (Colab).

## Notebook cells

1. Setup and data loading
2. Statistics: assumption checks (Shapiro–Wilk, Brown–Forsythe), ANOVA/Welch ANOVA, Tukey/Games–Howell, letters, Figure S3
3. Growth models: candidate models ranked by AICc; only the best model is reported (beta for temperature, Gaussian for pH)
4. Discolored-zone kinetics: expansion rates and best area model (exponential)
5. Figure 3, 6. Figure 5, 7. Figure 6
8. Download all figures (PNG, TIFF, PDF at 600 dpi)

## How to run

**Google Colab:** open `notebooks/R144_sectioned_Colab.ipynb` in Colab, run Cell 1, upload the data file when asked, then run the remaining cells in order.

**Local:**
```bash
pip install -r requirements.txt
python scripts/R144_analysis.py      # run from the repository root; needs data/Tur_compiled_analysis_ready.xlsx
```

## Citation

Please cite the article once published. GenBank accession of the ITS sequence: PZ236463.
