import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import json

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
    filters the data, trains a temporary model, finds the closest match,
    and returns the corresponding parameters.
    """
    full_df = model_data.get('full_dataframe')

    if full_df is None:
        return {"error": "Model is not initialized."}

    # 1. Filter the DataFrame based on the selected patient group
    if patient_group == 'Healthy':
        # Search only within the 'TAV Healthy' data
        search_df = full_df[full_df['Valve_phenotype'] == 'TAV Healthy'].copy()
    else: # Aneurysmal
        # Search within all other data
        search_df = full_df[full_df['Valve_phenotype'] != 'TAV Healthy'].copy()
    
    if search_df.empty:
        return {"error": f"No data available for the '{patient_group}' group."}

    # 2. Set up and train the k-NN model on the filtered data
    X = search_df[FEATURE_COLS]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(X_processed)

    # 3. Process the new patient input and find the match
    new_patient_data = json.loads(new_patient_json)
    new_patient_df = pd.DataFrame([new_patient_data])[FEATURE_COLS]
    new_patient_processed = preprocessor.transform(new_patient_df)
    
    distances, indices = knn.kneighbors(new_patient_processed)
    closest_index_in_subset = indices[0][0]
    
    # 4. Retrieve the original data and return it
    closest_patient_info = search_df.iloc[closest_index_in_subset][FEATURE_COLS].to_dict()
    closest_patient_params = search_df.iloc[closest_index_in_subset][PARAM_COLS].to_dict()

    return {
        "closest_match_info": closest_patient_info,
        "parameters": closest_patient_params
    }