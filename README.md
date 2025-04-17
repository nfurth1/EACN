# Comparison of Machine Learning Approaches for Prediction of the Equivalent Alkane Carbon Number for Microemulsions

This repository accompanies the research article:

**Nicholas R. Furth, Adam E. Imel, and Thomas A. Zawodzinski**  
*Published in* The Journal of Physical Chemistry A, 2024.  
[DOI: 10.1021/acs.jpca.4c00936](https://doi.org/10.1021/acs.jpca.4c00936)

## Overview

This project explores machine learning (ML) methods for predicting the Equivalent Alkane Carbon Number (EACN) of oils based on molecular structure.  
Experimental determination of EACN is traditionally time-consuming and purity-sensitive. We developed models to predict EACN quickly and reliably using:

- Graph Neural Networks (GNNs)
- XGBoost Decision Tree Model

**Key highlights:**
- Three GNN architectures: MPNN, CGCNN, and GCN
- Geometry optimization studies (none, MMFF94, GAFF)
- Feature-based XGBoost using RDKit descriptors
- Best model achieved R² ≈ 0.9 with MAE ≈ 1.15 EACN units

## Models Implemented

| Model  | Input Type | Special Features |
|:------:|:----------:|:----------------:|
| MPNN   | SMILES (with/without geometry optimization) | Deep message passing |
| CGCNN  | SMILES (with/without geometry optimization) | Crystal graph convolution |
| GCN    | SMILES (with/without geometry optimization) | Spectral convolution |
| XGBoost | Molecular descriptors from SMILES | Recursive feature elimination |

## Dataset

- **Size:** 183 organic molecules
- **Data Source:** Published experimental EACN values from literature references.
- **SMILES Input:** Obtained via PubChem or cheminfo.org.

## Results

- Geometry optimization improves model performance significantly.
- CGCNN with MMFF94 optimization provided the best GNN results.
- XGBoost with RDKit descriptors performed extremely well with minimal computational resources.

| Model | R² (Test Set) | MAE | RMSE |
|:-----:|:-------------:|:---:|:----:|
| CGCNN (MMFF94) | 0.90 | 1.15 | ~1.6 |
| XGBoost | 0.89 | 1.17 | ~1.68 |

## Repository Structure

```
/data/            # SMILES, descriptors, and EACN values
/scripts/         # Data preprocessing, model training, evaluation
/models/          # Trained model checkpoints
/results/         # Regression plots, error distributions
README.md
requirements.txt  # Python dependencies
```

## Installation

```bash
git clone https://github.com/nfurth1/EACN.git
cd EACN
pip install -r requirements.txt
```

## Usage

Train a model (example for XGBoost):

```bash
python scripts/train_xgboost.py --input data/descriptors.csv --output models/xgboost_model.pkl
```

Evaluate model performance:

```bash
python scripts/evaluate_model.py --model models/xgboost_model.pkl --test data/test_set.csv
```

## Citation

If you use this work, please cite:

```
@article{furth2024eacn,
  title={Comparison of Machine Learning Approaches for Prediction of the Equivalent Alkane Carbon Number for Microemulsions Based on Molecular Properties},
  author={Nicholas R. Furth, Adam E. Imel, and Thomas A. Zawodzinski},
  journal={The Journal of Physical Chemistry A},
  volume={128},
  number={32},
  pages={6763-6773},
  year={2024},
  publisher={American Chemical Society},
  doi={10.1021/acs.jpca.4c00936}
}
```

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
