import os
# Disable TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#from keras.models import load_model
import lightgbm as lgb
#from xgboost import XGBClassifier

import warnings
#import tensorflow as tf
#import keras.backend as K

# Suppress TensorFlow deprecation warnings
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress only specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress all warnings
warnings.filterwarnings('ignore')

working_dir = "C:/Users/david/OneDrive/Documents/LolaRa"  # Update this as necessary
threshold_0 = 0.4171 
threshold_1 = 0.3862 
#threshold_2 = 0.3312


# Custom F1-score metric
#def f1_m(y_true, y_pred):
#    def recall_m(y_true, y_pred):
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#        recall = true_positives / (possible_positives + K.epsilon())
#        return recall
#
#    def precision_m(y_true, y_pred):
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#        precision = true_positives / (predicted_positives + K.epsilon())
#        return precision
#
#    precision = precision_m(y_true, y_pred)
#    recall = recall_m(y_true, y_pred)
#    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load the trained models
with open(os.path.join(working_dir, "models/LolaRa_lg_model_pure.sav"), "rb") as file:
    model1 = pickle.load(file)

#with open(os.path.join(working_dir, "models/LolaRa_xg_model_pure.sav"), "rb") as file:
#    model2 = pickle.load(file)
#
#model3 = load_model('C:/Users/david/OneDrive/Documents/LolaRa/models/best_model_last_w.hdf5', custom_objects={'f1_m': f1_m})
#
#with open(os.path.join(working_dir, "models/best_meta_model_pure.sav"), "rb") as file:
#    best_meta_model = pickle.load(file)

# Load the conformal threshold
#with open(os.path.join(working_dir, "models/conformal_threshold_pure.sav"), "rb") as file:
 #   threshold = pickle.load(file)
#print(threshold)
    
# Read and preprocess the new dataset
data = pd.read_csv(os.path.join(working_dir, "data/ExportedMinutePeakData - 5 pips.csv"),delimiter=';', header=0) # Update this as necessary 
new_data = data.tail(500).copy() # Get the last row of the dataset - comment out if you want to use the whole dataset
#new_data.to_csv(os.path.join(working_dir, "data/new_data.csv"), index=False)
new_data.drop(columns=['Id','Size','Type','Classification'], inplace=True)
new_data['datetime'] = pd.to_datetime(new_data['DateTime'])
new_data.drop(columns=['DateTime'], inplace=True)
new_data.dropna(inplace=True)
print(new_data.shape)

def engineer_features(data):
    # Ensure the data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    # Check if the necessary columns are present
    required_columns = ['Tick1Mvnt', 'Tick2Mvnt', 'Tick3Mvnt', 'datetime']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    features = pd.DataFrame()
    
    # Absolute movements
    features['AbsTick1Mvnt'] = data['Tick1Mvnt'].abs()
    features['AbsTick2Mvnt'] = data['Tick2Mvnt'].abs()
    features['AbsTick3Mvnt'] = data['Tick3Mvnt'].abs()
    
    # Interaction terms
    features['Tick1_2_Interaction'] = data['Tick1Mvnt'] * data['Tick2Mvnt']
    features['Tick2_3_Interaction'] = data['Tick2Mvnt'] * data['Tick3Mvnt']
    features['Tick1_3_Interaction'] = data['Tick1Mvnt'] * data['Tick3Mvnt']
    
    # Differences and ratios
    features['Tick1_Tick2_Diff'] = data['Tick1Mvnt'] - data['Tick2Mvnt']
    features['Tick2_Tick3_Diff'] = data['Tick2Mvnt'] - data['Tick3Mvnt']
    features['Tick1_Tick3_Diff'] = data['Tick1Mvnt'] - data['Tick3Mvnt']
    
    features['Tick1_Tick2_Ratio'] = data['Tick1Mvnt'] / (data['Tick2Mvnt'] + 1e-6)  # adding a small value to avoid division by zero
    features['Tick2_Tick3_Ratio'] = data['Tick2Mvnt'] / (data['Tick3Mvnt'] + 1e-6)
    features['Tick1_Tick3_Ratio'] = data['Tick1Mvnt'] / (data['Tick3Mvnt'] + 1e-6)
    
    # Logarithmic transformations
    features['LogAbsTick1Mvnt'] = np.log(features['AbsTick1Mvnt'] + 1e-6)  # log transformation
    features['LogAbsTick2Mvnt'] = np.log(features['AbsTick2Mvnt'] + 1e-6)
    features['LogAbsTick3Mvnt'] = np.log(features['AbsTick3Mvnt'] + 1e-6)
    
    # Square and Square root transformations
    features['SqrTick1Mvnt'] = data['Tick1Mvnt'] ** 2
    features['SqrTick2Mvnt'] = data['Tick2Mvnt'] ** 2
    features['SqrTick3Mvnt'] = data['Tick3Mvnt'] ** 2
    
    features['SqrtAbsTick1Mvnt'] = np.sqrt(features['AbsTick1Mvnt'])
    features['SqrtAbsTick2Mvnt'] = np.sqrt(features['AbsTick2Mvnt'])
    features['SqrtAbsTick3Mvnt'] = np.sqrt(features['AbsTick3Mvnt'])
    
    # Indicator variables for movements
    features['IsTick1Positive'] = (data['Tick1Mvnt'] > 0).astype(int)
    features['IsTick2Positive'] = (data['Tick2Mvnt'] > 0).astype(int)
    features['IsTick3Positive'] = (data['Tick3Mvnt'] > 0).astype(int)
    
    features['IsTick1Negative'] = (data['Tick1Mvnt'] < 0).astype(int)
    features['IsTick2Negative'] = (data['Tick2Mvnt'] < 0).astype(int)
    features['IsTick3Negative'] = (data['Tick3Mvnt'] < 0).astype(int)
    
    # Composite features
    features['Tick1_Tick2_LogRatio'] = features['LogAbsTick1Mvnt'] / (features['LogAbsTick2Mvnt'] + 1e-6)
    features['Tick2_Tick3_LogRatio'] = features['LogAbsTick2Mvnt'] / (features['LogAbsTick3Mvnt'] + 1e-6)
    features['Tick1_Tick3_LogRatio'] = features['LogAbsTick1Mvnt'] / (features['LogAbsTick3Mvnt'] + 1e-6)
    
    # Time-related features
    data['datetime'] = pd.to_datetime(data['datetime'])
    features['Hour'] = data['datetime'].dt.hour
    features['DayOfWeek'] = data['datetime'].dt.dayofweek
    features['IsWeekend'] = data['datetime'].dt.dayofweek.isin([5, 6]).astype(int)  # 5 for Saturday, 6 for Sunday
    features['DayOfMonth'] = data['datetime'].dt.day
    features['IsEndOfMonth'] = (data['datetime'] + pd.offsets.MonthEnd(0)).dt.day == data['datetime'].dt.day
    features['Quarter'] = data['datetime'].dt.quarter
    features['IsEndOfQuarter'] = (data['datetime'] + pd.offsets.QuarterEnd(0)).dt.day == data['datetime'].dt.day
    features['Month'] = data['datetime'].dt.month
    
    # Return the engineered features
    return features

features_df = engineer_features(new_data)

# Columns used for modeling, make sure this matches your training columns
columns_to_model = [
'Hour',
'DayOfWeek',
'Quarter',
'Month',
'DayOfMonth',
'SqrtAbsTick3Mvnt',
'Tick1_Tick3_LogRatio'
]
new_data_features = features_df[columns_to_model]
print(new_data_features.shape)

# Predict probabilities for each base model
lg_preds = model1.predict_proba(new_data_features)
#xg_preds = model2.predict_proba(new_data_features)
#lstm_preds = model3.predict(new_data_features)

# Combine these probabilities into a single input for the meta-model
#combined_preds = np.hstack([lg_preds, xg_preds, lstm_preds])

# Predict class probabilities with the meta-model
#final_proba = best_meta_model.predict_proba(combined_preds)
final_proba = lg_preds

# Apply Conformal Prediction
def apply_conformal_prediction(probs, threshold):
    conforms = np.max(probs, axis=1) >= threshold
    return conforms

conformal_intervals = apply_conformal_prediction(final_proba, threshold_2)

print("Final Probabilities:")
print(final_proba)
print("Conformal Intervals:")
print(conformal_intervals)

print(str(final_proba[0, 0]) + "|" + str(final_proba[0, 1]) + "|" + str(final_proba[0, 2]) + "|" + str(conformal_intervals[0]))