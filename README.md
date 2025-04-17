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
- Geometry optimization studies (None, MMFF94, GAFF)
- Feature-based XGBoost using RDKit descriptors

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

## Repository Structure

```
/MatDeepLearn/           #Our Implementation of MatDeepLearn for GNNs
/Model Results           #Meta Data for our Model Results
Descriptors.ipynb        #XGBoost Implementation
Error Plots.ipynb        #Plot Generation for Results
SSBSSW.ipynb             #F-Statistic Generation for Results (Functional Groups)
Descriptors.py           #RDKit Descriptor Generation called by the Notebook files
README.md            
requirements.txt  # Python dependencies
```

## Installation

```bash
git clone https://github.com/nfurth1/EACN.git
cd EACN
pip install -r requirements.txt
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
