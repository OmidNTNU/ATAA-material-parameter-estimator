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

# This will hold our trained model and data
model_data = {}

def initialize_model(csv_data):
    """
    Loads data, preprocesses it, and trains the k-NN model.
    This function runs once when the page loads.
    """
    global model_data

    # Load the dataset from the CSV string
    df = pd.read_csv(pd.io.common.StringIO(csv_data))

    # Define the features (X) and store the full dataframe for later
    X = df[FEATURE_COLS]
    model_data['dataframe'] = df

    # Create a preprocessor for our data
    # Numerical features will be scaled (so 'Age' doesn't dominate 'Thickness')
    # Categorical features will be one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ])

    # Fit the preprocessor and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Store the fitted preprocessor
    model_data['preprocessor'] = preprocessor
    
    # Define and train the k-Nearest Neighbors model
    # We set n_neighbors=1 to find the single closest match
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(X_processed)

    # Store the trained model
    model_data['knn_model'] = knn
    
    print("Model initialized successfully!")


def find_closest_patient(new_patient_json):
    """
    Takes new patient data, finds the closest match in the dataset,
    and returns the corresponding parameters.
    """
    # Load the trained model, preprocessor, and original data
    knn = model_data.get('knn_model')
    preprocessor = model_data.get('preprocessor')
    df = model_data.get('dataframe')

    if not all([knn, preprocessor, df is not None]):
        return {"error": "Model is not initialized."}

    # Parse the incoming JSON data
    new_patient_data = json.loads(new_patient_json)
    
    # Convert the new data into a DataFrame with the correct column order
    new_patient_df = pd.DataFrame([new_patient_data])[FEATURE_COLS]

    # Preprocess the new data using the FITTED preprocessor
    new_patient_processed = preprocessor.transform(new_patient_df)

    # Find the nearest neighbor
    distances, indices = knn.kneighbors(new_patient_processed)
    
    # Get the index of the closest patient
    closest_index = indices[0][0]

    # Retrieve the clinical info and parameters of the closest match
    closest_patient_info = df.loc[closest_index, FEATURE_COLS].to_dict()
    closest_patient_params = df.loc[closest_index, PARAM_COLS].to_dict()

    # Return both as a dictionary
    return {
        "closest_match_info": closest_patient_info,
        "parameters": closest_patient_params
    }