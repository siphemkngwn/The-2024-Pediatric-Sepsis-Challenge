#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import pandas as pd
import mne
from sklearn.impute import SimpleImputer
sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, data, label, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Generate dummies and store the column names for consistency
    data = pd.get_dummies(data)
    columns = data.columns

    # Save the column names for later use during inference
    with open(os.path.join(model_folder, 'columns.txt'), 'w') as f:
        f.write("\n".join(columns))
        
    # Define parameters for random forest classifier and regressor.
    n_estimators   = 100  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 42  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(data)

    # Train the models.
    data_imputed = imputer.transform(data)
    prediction_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(data_imputed, label.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, prediction_model)

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')

    # Load the saved column names
    with open(os.path.join(model_folder, 'columns.txt'), 'r') as f:
        columns = f.read().splitlines()

    model = joblib.load(os.path.join(model_folder, 'model.sav'))
    model['columns'] = columns
    return model

def run_challenge_model(model, data_folder, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']
    columns = model['height_cm_adm','muac_mm_adm','glucose_mmolpl_adm','lengthadm','rr_brpm_app_adm','weight_kg_adm','diasbp_mmhg_adm','sqi1_perc_oxi_adm','bcsverbal_adm','temp_c_adm','lactate_mmolpl_adm','hematocrit_gpdl_adm','bcsmotor_adm','sysbp_mmhg_adm','inhospital_mortality','agecat']

    # Load data.
    patient_ids, data, label, features = load_challenge_data(data_folder)
    
    data = pd.get_dummies(data)

    # Align test data with training columns, filling missing columns with 0
    data = data.reindex(columns=columns, fill_value=0)
    
    # Impute missing data.
    data_imputed = imputer.transform(data)

    # Apply model to data.
    prediction_binary = prediction_model.predict(data_imputed)[:]
    prediction_probability = prediction_model.predict_proba(data_imputed)[:, 1]

    return patient_ids, prediction_binary, prediction_probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, prediction_model):
    d = {'imputer': imputer, 'prediction_model': prediction_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
