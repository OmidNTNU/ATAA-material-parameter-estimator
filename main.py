import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import json
import numpy as np

# Define the columns for features and parameters
FEATURE_COLS = ['Thickness', 'Age', 'Diameter', 'Sex', 'Valve_phenotype', 'Region']
PARAM_COLS = ['mu', 'k1', 'k2', 'k_op', 'k_ip1', 'k_ip2', 'alpha1', 'alpha2']
NUMERIC_FEATURES = ['Thickness', 'Age', 'Diameter']
CATEGORICAL_FEATURES = ['Sex', 'Valve_phenotype', 'Region']

# This will hold our full dataset
model_data = {}

def initialize_model(csv_data):
    """
    Loads the full dataset. This function runs once when the page loads.
    """
    global model_data
    # Load the dataset from the CSV string and store it
    model_data['full_dataframe'] = pd.read_csv(pd.io.common.StringIO(csv_data))
    print("Model initialized successfully with full dataset!")

def find_closest_patient(new_patient_json, patient_group):
    """
    Takes new patient data and the selected group ('Aneurysmal' or 'Healthy'),
    filters the data, trains a temporary model with feature weighting, 
    finds the closest match, and returns the parameters.
    """
    full_df = model_data.get('full_dataframe')

    if full_df is None:
        return {"error": "Model is not initialized."}

    # 1. Filter the DataFrame based on the selected patient group
    if patient_group == 'Healthy':
        search_df = full_df[full_df['Valve_phenotype'] == 'TAV Healthy'].copy()
    else: # Aneurysmal
        search_df = full_df[full_df['Valve_phenotype'] != 'TAV Healthy'].copy()
    
    if search_df.empty:
        return {"error": f"No data available for the '{patient_group}' group."}

    # 2. Set up the preprocessor
    X = search_df[FEATURE_COLS]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor and transform the data
    X_processed = preprocessor.fit_transform(X)

    # --- NEW: APPLY WEIGHTING ---
    # Define the weight for the Valve Phenotype feature
    valve_phenotype_weight = 4.0
    
    # Get the names of the columns created by the OneHotEncoder
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
    
    # Find the indices of the columns that belong to 'Valve_phenotype'
    # These column names will look like 'Valve_phenotype_BAV', 'Valve_phenotype_TAV', etc.
    valve_indices = [i for i, col_name in enumerate(categorical_feature_names) if col_name.startswith('Valve_phenotype_')]
    
    # We also need to offset these indices by the number of numeric features
    numeric_features_count = len(NUMERIC_FEATURES)
    valve_indices = [i + numeric_features_count for i in valve_indices]

    # Apply the weight by multiplying the selected columns
    if valve_indices:
        X_processed[:, valve_indices] *= valve_phenotype_weight
    # --- END OF NEW CODE ---

    # 3. Fit the k-NN model on the processed and weighted data
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(X_processed)

    # 4. Process the new patient input
    new_patient_data = json.loads(new_patient_json)
    new_patient_df = pd.DataFrame([new_patient_data])[FEATURE_COLS]
    new_patient_processed = preprocessor.transform(new_patient_df)
    
    # --- NEW: APPLY WEIGHTING TO THE NEW INPUT ---
    if valve_indices:
        new_patient_processed[:, valve_indices] *= valve_phenotype_weight
    # --- END OF NEW CODE ---
    
    # 5. Find the closest match
    distances, indices = knn.kneighbors(new_patient_processed)
    closest_index_in_subset = indices[0][0]
    
    # 6. Retrieve the original data and return it
    closest_patient_info = search_df.iloc[closest_index_in_subset][FEATURE_COLS].to_dict()
    closest_patient_params = search_df.iloc[closest_index_in_subset][PARAM_COLS].to_dict()

    return {
        "closest_match_info": closest_patient_info,
        "parameters": closest_patient_params
    }