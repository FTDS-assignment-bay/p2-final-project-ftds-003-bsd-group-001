import pickle
import pandas as pd
# In your model training script and your Streamlit app script (app.py)
from transformers import UnitPriceTransformer, KMeansAndLabelTransformer, DynamicOneHotEncoder

# Load the pipeline and model
# Load the pipeline object from the file
with open('full_pipeline_with_unit_price.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Load the preprocessor object from the file
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Load the model object from the file
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

def make_prediction(input_features):
    # Assuming input_features is a DataFrame with the correct structure
    processed_features = pipeline.transform(input_features)
    prediction = model.predict(processed_features)
    return prediction[0]
