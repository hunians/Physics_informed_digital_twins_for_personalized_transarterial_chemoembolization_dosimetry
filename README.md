# Physics-informed digital twins for personalized transarterial chemoembolization dosimetry

ECPC-PINN couples hepatic arterial hemodynamics, doxorubicin transport, tumor response, differentiable embolization, hepatic arterial buffer response, inverse patient calibration, and physics-aware conformal prediction. The package supports retrospective validation from paired scans and prospective protocol simulation from pre-treatment imaging.

## Installation

Python 3.11 and an NVIDIA A100 40 GB are the reference environment.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Conda users can run `conda env create -f environment.yml`. The container can be built with `docker build -t ecpc-pinn .`.

## Data

Dataset access endpoints are stored in `dataset_links.txt`. HCC-TACE-Seg contains 105 subjects, occupies about 28.57 GB, and is distributed by TCIA under CC BY 4.0. WAW-TACE contains 233 subjects and is distributed as Zenodo record 12741586 under CC BY 4.0. Dataset files are not included. After download, construct patient manifests with subject identifiers, arterial-phase image locations, segmentation locations, response labels, and split assignments. Keep clinical identifiers outside the project tree.

The paper split is stratified by mRECIST category. HCC-TACE-Seg uses 74 training, 10 validation, and 21 test subjects. WAW-TACE uses 163 training, 23 validation, and 47 test subjects. Arterial segments below 1 mm are excluded, and cases with fewer than four visible arterial branches receive population-mean physics features.

## Training

The default network uses four hidden layers of 256 units. The Methods text also reports 64 units; that specification is retained in `configs/experiment/methods_64.yaml`. Main-result training uses 1,000 uniformly sampled collocation points per patient, Adam at 1e-3, and the staged schedule recorded in `configs/experiment/main.yaml`.

```bash
ecpc-train --config configs/experiment/main.yaml --output outputs/main.pt
```

The reported curriculum converges in about 772 epochs and 4.2 hours on one NVIDIA A100 40 GB. Stage transitions target velocity L2 relative error below 5% and drug concentration R2 of at least 0.80. The expected WAW-TACE response AUC is 0.91 with a reported standard deviation of 0.02 across five-fold validation. End-to-end inference is reported as 38 ± 5 seconds per patient.

## Evaluation

Prediction CSV files contain `target,score` columns.

```bash
ecpc-evaluate outputs/predictions.csv
```

The primary endpoint is 90-day mRECIST responder classification. Secondary measures include sensitivity, specificity, F1, velocity and pressure relative L2 error, wall-shear-stress MAE, concentration R2, RMSE, MAE, and conformal coverage. Patient-level bootstrap intervals use 1,000 resamples. Paired AUC comparisons use DeLong tests with a Bonferroni threshold of 0.0033 for fifteen comparisons.

## Inference

```bash
ecpc-infer outputs/main.pt
```

Retrospective calibration constrains diffusion to 100–800 µm²/s and tumor proliferation to 0.01–0.10 day⁻¹. Prospective simulation initializes transport from population priors and pre-treatment perfusion surrogates. Predictions are research outputs and are not clinical treatment recommendations.

## Verification

```bash
pytest -q
ruff check .
mypy --strict code/ecpc_pinn
```

The training test uses a small CPU tensor problem and verifies parameter updates. Physics tests cover the embolization transition and HABR bounds. Conformal tests verify residual-dependent interval expansion.

