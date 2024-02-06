import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import os
from math import sqrt
import time
from scipy.integrate import cumtrapz

from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import make_scorer

from keras.models import Sequential
from keras.layers import Input, Conv1D, Flatten, Dense, MaxPooling1D, Reshape
from tensorflow.keras import layers
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def count_carbons(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return sum([atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()])
    else:
        return None

def process_data(rfecv, restrict, minim, maxim, grid, carbon, charged, folder):
    df = pd.read_csv(folder)
    df.reset_index(drop=True, inplace=True)
    df
    
    l = df.Molecule
    for i in range(len(l)):
        l[i] = str(l[i])

    count = 0 
    flag = 0

    for i in range(len(l)):
        if i != 0:
            tmp = l[i-1][:-2]

        if i == 0:
            continue
        elif l[i] == l[i-1] and tmp != '_':
            l[i - 1] += ('_' + str(count))
            count += 1
            l[i] += ('_' + str(count))
            count += 1
        elif l[i] == tmp and l[i-1][-2] == '_':
            l[i] += ('_' + str(count))
            count += 1
        elif l[i] != l[i-1]:
            count = 0

    df.Molecule = l

    df['Target'] = df['Target'].astype('float64')
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    df['Molecule'] = df['Molecule'].astype(str)
    count = 0

    for i in range(1, len(df)):
        prev = df['Molecule'].iloc[i - 1]
        current = df['Molecule'].iloc[i]

        if prev == current:
            if '_' not in prev:
                df.at[i - 1, 'Molecule'] += f'_{count}'
                df.at[i, 'Molecule'] += f'_{count}'
                count += 1
            elif prev.endswith('_'):
                df.at[i, 'Molecule'] += f'_{count}'
                count += 1
        else:
            count = 0

    df['Target'] = df['Target'].astype(float)
    df = df.dropna().reset_index(drop=True)

    with_plus, without_plus = filter_smiles_with_plus(df)
    if charged == 0:
        df = without_plus
    elif charged == 1:
        df = with_plus
        
    if restrict == 1:
        df = df[df['Target'] > minim]
        df = df[df['Target'] < maxim]
        df = df.reset_index()
        df = df[['Molecule', 'Target', 'SMILES']]
        
    descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
    descriptor_names = descriptor_names + ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']

    mol_descriptor_calculator = MolecularDescriptorCalculator(descriptor_names)
    mols, desc = [None] * len(df.Molecule), [None] * len(df.Molecule)

    feature_names = mol_descriptor_calculator.GetDescriptorNames()

    molecule_names = [None] * len(df.Molecule)
    for i in range(len(df.SMILES)):
        mols[i] = Chem.MolFromSmiles(df['SMILES'][i])
        desc[i] = list_of_descriptor_vals = list(mol_descriptor_calculator.CalcDescriptors(mols[i]))

        molecule_names[i] = df.Molecule[i]
        
    if carbon == 1:
        for i in range(len(desc)):
            tot = count_carbons(df.SMILES[i])
            desc[i].append(tot)    
            
    x = np.array(desc)

    y = np.array(df['Target'])
    x = np.array(x, dtype=float)
    x = np.nan_to_num(x, nan=0, posinf=np.finfo(x.dtype).max, neginf=np.finfo(x.dtype).min)

    scaler = StandardScaler()
    scal = scaler.fit(x)
    x = scal.transform(x)
    
    if rfecv == 1:
        rfecv = RFECV(estimator=XGBRegressor())

        # Define early stopping parameters
        best_score = -999
        best_n_features = 0
        threshold = 0.001
        non_improvement_counter = 0
        max_non_improvement_iterations = 5  # Maximum number of consecutive non-improvements

        # Fit the RFECV model
        sel = rfecv.fit(x, y)

        early_stop = 0
        if early_stop == 1:
            for n_features in range(1, len(feature_names) + 1):
                # Set the number of features to consider
                rfecv.n_features_ = n_features

                # Check if the mean test score has improved
                if sel.cv_results_['mean_test_score'][n_features - 1] > best_score + threshold:
                    best_score = sel.cv_results_['mean_test_score'][n_features - 1]
                    best_n_features = n_features
                    non_improvement_counter = 0  # Reset the counter
                else:
                    non_improvement_counter += 1

                # Check if the stopping criterion is met
                if non_improvement_counter >= max_non_improvement_iterations:
                    # Stop the RFE process
                    rfecv.n_features_ = best_n_features
                    break

        # Optimal number of features based on RFECV
        best_n_features = rfecv.n_features_

        # Get the feature rankings and selected feature indices
        feature_rankings = rfecv.ranking_
        selected_feature_indices = feature_rankings[:best_n_features]

        # Get the selected feature names
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        
        # Print the optimal number of features
        print(f"Optimal number of features with Early Stopping: {best_n_features}" if early_stop == 1 else
              f"Optimal number of features without Early Stopping: {best_n_features}")
        
        print("Optimal feature names:", selected_feature_names)

        x_selected = x[:, selected_feature_indices]

        # Plot the results
        n_scores = len(sel.cv_results_["mean_test_score"])
        plt.figure()
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Mean Test Score")
        plt.errorbar(range(1, n_scores + 1), sel.cv_results_["mean_test_score"], 0)

        if early_stop == 1:
            plt.axvline(x=best_n_features, linestyle='--', color='black', label='Optimal number of features (Early Stopping)')
        else:
            plt.axvline(x=rfecv.n_features_, linestyle='--', color='black', label='Optimal number of features')

        plt.title("Recursive Feature Elimination \nwith correlated features")
        plt.show()
        
    if rfecv == 0:
        x_selected = x

    molecule_column = df['Molecule'].values.reshape((-1, 1))
    x_selected = np.concatenate((x_selected, molecule_column), axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.3, random_state=4)

    train_names = x_train[:,-1]
    test_names = x_test[:,-1]

    x_train = np.delete(x_train, -1, axis=1)
    x_test = np.delete(x_test, -1, axis=1)

    x_selected = np.delete(x_selected, -1, axis=1)
    
    if grid == 1:
        param_grid = {'max_depth': list(range(3, 8)),  
                      'gamma': [0, 0.001, 0.005, 0.01],
                      'n_estimators': list(range(25, 50)),
                       'reg_lambda': list(range(2, 9)) } 

        grid = GridSearchCV(XGBRegressor(), param_grid, n_jobs= -1) 

        grid.fit(x_selected, y) 
        print(grid.best_params_) 
    return df, x_train, y_train, x_test, y_test, train_names, test_names, scal

def error_plots(y_train, y_pred, y_test, y_pred_test):
    train_errors = abs(y_train-y_pred)
    train_errors.sort()
    test_errors = abs(y_test-y_pred_test)
    test_errors.sort()

    train_percentiles = np.percentile(train_errors, [25, 50, 75])
    test_percentiles = np.percentile(test_errors, [25, 50, 75])

    plt.hist(train_errors, bins=100, edgecolor='black')
    plt.xlabel('Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title("Train Errors", fontsize=14)

    for percentile in train_percentiles:
        plt.axvline(percentile, color='red', linestyle='dashed')

    text_y_position = plt.ylim()[1] * 0.75 
    for i, percentile in enumerate(train_percentiles):
        plt.text(plt.xlim()[1] * 0.95, text_y_position - i * 0.04 * plt.ylim()[1],
                 f"{int((i+1)*25)}th  Percentile: {percentile:.2f}{i + 1} Degrees", color='black', ha='right')

    plt.show()

    plt.hist(test_errors, bins=100, edgecolor='black')
    plt.xlabel('Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title("Test Errors", fontsize=14)

    for percentile in test_percentiles:
        plt.axvline(percentile, color='red', linestyle='dashed')

    text_y_position = plt.ylim()[1] * 0.75  
    for i, percentile in enumerate(test_percentiles):
        plt.text(plt.xlim()[1] * 0.95, text_y_position - i * 0.04 * plt.ylim()[1],
                 f"{int((i+1)*25)}th Percentile: {percentile:.2f}{i + 1} Degrees", color='black', ha='right')

    plt.show()

def charged_frac(df):
    charged_df = df[df['SMILES'].str.contains('\+')]

    total_molecules = len(df)
    charged_count = len(charged_df)
    percentage_charged = (charged_count / total_molecules) * 100

    print(f"Percentage of charged molecules: {percentage_charged:.2f}%")
    
def NN(x_train, x_test, y_train, y_test, filters, kernel, pool, dense):
    num_features = len(x_train[0])

    input_shape = (num_features, 1) 

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool))
    model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool))

    model.add(Flatten())

    model.add(Dense(dense, activation='relu'))
    model.add(Dense(dense, activation='relu'))
    model.add(Dense(dense, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

def filter_smiles_with_plus(df):
    has_plus = df['SMILES'].str.contains('\+')
    smiles_with_plus = df[has_plus].reset_index(drop=True)
    smiles_without_plus = df[~has_plus].reset_index(drop=True)

    return smiles_with_plus, smiles_without_plus
