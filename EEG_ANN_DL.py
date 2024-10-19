# -*- coding: utf-8 -*-
'''
# Autoencoder + LSTM Model for time series data

This Python code implements an **Autoencoder** using **LSTM** for time series data, where each feature has a different time window interval. Here's an overview of its functionality:

1. **Data Preprocessing**:
   - The code reads multiple CSV files and merges them into a single DataFrame.
   - The user can select relevant features and map specific window sizes to each feature, which applies sliding windows to form time-series sequences.
   - It normalizes the data using **MinMaxScaler** to ensure the values are within a uniform range.

2. **Autoencoder-LSTM Model**:
   - An **Autoencoder** with an LSTM layer is used to compress (encode) and reconstruct (decode) the time series data.
   - The encoder uses **LSTM** units to capture temporal patterns, and the decoder reconstructs the sequence using **RepeatVector** and **TimeDistributed** layers.
   
3. **Hyperparameter Tuning**:
   - **KerasTuner** is used for tuning hyperparameters like the number of LSTM units, latent space dimensions, and learning rate. The best model is selected based on validation loss.
   
4. **Model Training and Evaluation**:
   - The model is trained on time-series data, and the best model is saved after hyperparameter optimization.
   - The model is then used to make predictions on the test data, and the results are evaluated and saved for further analysis.

This approach is useful for anomaly detection, trend discovery, or feature compression in time series data with varying window sizes per feature.

=========
By ali.bavarchee@gmail.com

'''

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
import zipfile
import csv
import os
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import os

from tqdm import tqdm
import time

from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import optuna

from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras_tuner import RandomSearch

from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam




seed = 777
np.random.seed(seed)

#folder_path = os.getcwd()
#zip_file = "index_one_year_new.zip"
'''
# Extract the CSV file assuming there's only one file in the zip
with zipfile.ZipFile(f"{folder_path}/{zip_file}", 'r') as zip_ref:
    zip_ref.extractall(folder_path)
    csv_file = zip_ref.namelist()[0]

# Specify the desired columns to read from the CSV
desired_columns = ['index', 'Time of Day', 'Percentage_Change',
                   'Sequence_Change', 'Label_0.3_local_3', 'Label_0.4_local_3',
                   'p_1', 'hl_1', 'p_2', 'hl_2', 'p_3', 'hl_3',
                   'p_5', 'hl_5', 'p_10', 'hl_10', 'p_20', 'hl_20',
                   'p_30', 'hl_30', 'p_40', 'hl_40', 'p_60', 'hl_60',
                   'p_90', 'hl_90', 'p_120', 'hl_120', 'p_180',
                   'hl_180', 'p_240', 'hl_240', 'p_360', 'hl_360',
                   'p_480', 'hl_480', 'p_720', 'hl_720', 'p_1440',
                   'hl_1440', 'local_0.4', 'local_0.3']

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(f"{folder_path}/{csv_file}", usecols=desired_columns)
'''
def create_dataframe_from_csv(folder_path, column_names=None):
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Loop through each CSV file and read it into a dataframe
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        
        # Read the CSV into a dataframe
        df = pd.read_csv(file_path)
        
        # If column_names are provided, select only those columns
        if column_names:
            df = df[column_names]
        
        # Append the dataframe to the list
        dataframes.append(df)
    
    # Concatenate all dataframes into one
    final_df = pd.concat(dataframes, ignore_index=True)
    
    return final_df

# How it works==>
# folder_path = '/path/to/your/csv/files'
# columns_to_keep = ['column1', 'column2', 'column3']
# df = create_dataframe_from_csv(folder_path, columns_to_keep)
# print(df)

'''
def select_features_with_window(df, window_size_map):
    """
    Select relevant features based on the window size map and create a new DataFrame 
    with time windowed features.
    
    Parameters:
    - df: Original time series DataFrame
    - window_size_map: A dictionary where keys are column names and values are window sizes
    
    Returns:
    - A new DataFrame with the windowed features
    """
    
    # Initialize an empty DataFrame to store windowed features
    windowed_df = pd.DataFrame()

    for feature, window_size in window_size_map.items():
        if feature not in df.columns:
            print(f"Warning: {feature} not found in DataFrame.")
            continue
        
        # Apply the rolling window to each feature
        # .shift() will ensure that we are getting the past values
        for window in range(window_size):
            windowed_df[f"{feature}_t-{window}"] = df[feature].shift(window)
    
    # Drop any rows that contain NaN values as a result of shifting
    windowed_df.dropna(inplace=True)
    
    return windowed_df
'''

folder_path =input("insert the path file")

# Inserting column names (comma-separated input)
user_input = input("Enter column names, separated by commas: ")

# Convert the input string to a list of column names
columns = [col.strip() for col in user_input.split(',')]

columns_to_keep = columns


#folder_path ="/content/"
#columns_to_keep = ['index',	'Time of Day',	'Percentage_Change',
 #                  'Sequence_Change',	'Label_0.3_local_3',	'Label_0.4_local_3',
  #                 'p_1',	'hl_1',	'p_2',	'hl_2',	'p_3',	'hl_3',
   #                'p_5',	'hl_5',	'p_10',	'hl_10',	'p_20',	'hl_20',
    #               'p_30',	'hl_30',	'p_40',	'hl_40',	'p_60',	'hl_60',
     #              'p_90',	'hl_90',	'p_120',	'hl_120',	'p_180',
      #             'hl_180',	'p_240',	'hl_240',	'p_360',	'hl_360',
       #            'p_480',	'hl_480',	'p_720',	'hl_720',	'p_1440',
        #           'hl_1440',	'local_0.4',	'local_0.3']
#df = create_dataframe_from_csv(folder_path, columns_to_keep)

df = create_dataframe_from_csv(folder_path, columns_to_keep)

print(df.head())

df = df.dropna()

#df = df.astype(int)

#df = df.round(1)

def select_features(dataframe):
    print("Available columns in the DataFrame:")
    print(dataframe.columns.tolist())
    
    selected_columns = []
    
    while True:
        print("\nType the name of the columns you want to select for training (one at a time).")
        print("Type 'done' when finished or 'all' to select all columns.")
        
        user_input = input("Select feature: ")
        
        if user_input == 'done':
            break
        elif user_input == 'all':
            selected_columns = dataframe.columns.tolist()
            break
        elif user_input in dataframe.columns:
            if user_input not in selected_columns:
                selected_columns.append(user_input)
                print(f"'{user_input}' has been added to the selection.")
            else:
                print(f"'{user_input}' is already selected.")
        else:
            print(f"'{user_input}' is not a valid column name. Please try again.")
    
    if not selected_columns:
        print("No features selected. Exiting...")
        return None
    
    selected_df = dataframe[selected_columns]
    return selected_df

do = select_features(df)

features = selected_features_df.columns
features = list(features)

def map_features_to_window_size(features_list):
    # Initialize an empty dictionary to store feature-window size mapping
    window_size_map = {}

    # Loop through each feature and ask the user for a window size
    for feature in features_list:
        while True:
            try:
                # Prompt the user to input a window size for the current feature
                user_input = input(f"Enter the window size for {feature}: ")
                
                # Convert input to integer
                window_size = int(user_input)
                
                # Store the window size in the dictionary
                window_size_map[feature] = window_size
                
                break  # Exit the loop if valid input is provided
            except ValueError:
                print("Invalid input. Please enter an integer.")
    
    return window_size_map

# Example usage:
# features_list = ['feature_1', 'feature_2', 'feature_3']  # List of feature names
# win_size = map_features_to_window_size(features_list)

window_size_map = map_features_to_window_size(features)

# print(win_size)

target = input("insert the target")


# Load your DataFrame df
X = df[features].values
y = df[target].values

# Step 2: Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Prepare the time-series data (Same as the previous code)
max_window_size = max(window_size_map.values())
X_time_series = []

# Loop through each feature, apply window size, and reshape accordingly
for i, feature in enumerate(features):
    feature_series = X_scaled[:, i]  # Extract the feature series
    window_size = window_size_map[feature]
    
    # Create sliding windows for this feature
    feature_time_series = np.array([feature_series[j:j + window_size] for j in range(len(feature_series) - window_size + 1)])
    
    # Pad to make all series have the same length (matching max_window_size)
    if feature_time_series.shape[1] < max_window_size:
        pad_length = max_window_size - feature_time_series.shape[1]
        feature_time_series = np.pad(feature_time_series, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
    
    X_time_series.append(feature_time_series)

# Step 4: Find the minimum number of time steps across all features
min_time_steps = min([ts.shape[0] for ts in X_time_series])

# Truncate all feature time series to the minimum time steps to make them the same length
X_time_series = [ts[:min_time_steps] for ts in X_time_series]

# Stack the features along the last axis
X_time_series = np.stack(X_time_series, axis=-1)

# Define the model builder function for KerasTuner
def build_autoencoder_model(hp):
    latent_dim = hp.Int('latent_dim', min_value=10, max_value=35, step=5)
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=1024, step=64)
    
    inputs = Input(shape=(X_time_series.shape[1], X_time_series.shape[2]))
    
    # Encoder
    encoded = LSTM(lstm_units, activation='relu')(inputs)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = RepeatVector(X_time_series.shape[1])(encoded)
    decoded = LSTM(lstm_units, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(X_time_series.shape[2]))(decoded)
    
    # Compile the model
    autoencoder = Model(inputs, decoded)
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    return autoencoder

# Hyperparameter tuning using RandomSearch
tuner = RandomSearch(
    build_autoencoder_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='autoencoder_tuning',
    project_name='autoencoder')

# Run the hyperparameter search
tuner.search(X_time_series, X_time_series, epochs=50, batch_size=64, validation_split=0.4)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss = best_model.evaluate(X_time_series, X_time_series)

# Save the best model and results
best_model.save('best_autoencoder_model.h5')
np.save('scaler.npy', scaler)

# Report the results
print("Best hyperparameters: ", tuner.get_best_hyperparameters()[0].values)
print("Best model evaluation loss: ", loss)


scaler = np.load('scaler.npy', allow_pickle=True).item()
#best_model = load_model('best_autoencoder_model.h5')

X = df[features].values
X_scaled = scaler.transform(X)

# Prepare the time-series data (similar to previous code)
max_window_size = max(window_size_map.values())
X_time_series = []

for i, feature in enumerate(features):
    feature_series = X_scaled[:, i]  # Extract the feature series
    window_size = window_size_map[feature]
    
    # Create sliding windows for this feature
    feature_time_series = np.array([feature_series[j:j + window_size] for j in range(len(feature_series) - window_size + 1)])
    
    # Pad to make all series have the same length (matching max_window_size)
    if feature_time_series.shape[1] < max_window_size:
        pad_length = max_window_size - feature_time_series.shape[1]
        feature_time_series = np.pad(feature_time_series, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
    
    X_time_series.append(feature_time_series)

# Truncate all feature time series to the minimum time steps
min_time_steps = min([ts.shape[0] for ts in X_time_series])
X_time_series = [ts[:min_time_steps] for ts in X_time_series]

# Stack the features along the last axis
X_time_series = np.stack(X_time_series, axis=-1)

# Make predictions using the best model
X_pred = best_model.predict(X_time_series)

# Reshape predictions and true values to match for visualization
X_pred_reshaped = X_pred.reshape(-1, X_pred.shape[-1])
X_time_series_reshaped = X_time_series.reshape(-1, X_time_series.shape[-1])

# Save predictions vs true values plots for each feature as PDF
for feature_index, feature_name in enumerate(features):
    plt.figure(figsize=(12, 6))
    plt.plot(X_time_series_reshaped[:, feature_index], label='True Values', alpha=0.6)
    plt.plot(X_pred_reshaped[:, feature_index], label='Predicted Values', alpha=0.6)
    plt.title(f'Feature: {feature_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    
    # Save the plot as a PDF file
    plt.savefig(f'plot_{feature_name}.pdf', format='pdf')
    plt.close()  # Close the figure after saving to avoid memory issues

# Optional: Calculate and print evaluation metrics for the overall model
mse = mean_squared_error(X_time_series_reshaped, X_pred_reshaped)
mae = mean_absolute_error(X_time_series_reshaped, X_pred_reshaped)

print("Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")



# best_model = tf.keras.models.load_model('best_autoencoder_model.h5')
scaler = np.load('scaler.npy', allow_pickle=True).item()

# Make predictions
X_pred = best_model.predict(X_time_series)

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(X_time_series - X_pred), axis=(1, 2))

# Calculate MSE and R² score for evaluation
mse = mean_squared_error(X_time_series.reshape(-1), X_pred.reshape(-1))
r2 = r2_score(X_time_series.reshape(-1), X_pred.reshape(-1))


plt.figure(figsize=(12, 6))
plt.plot(reconstruction_error, label='Reconstruction Error', color='orange')
plt.title('Reconstruction Error of the Autoencoder Model')
plt.xlabel('Sample Index')
plt.ylabel('Mean Squared Error')
plt.axhline(y=np.mean(reconstruction_error), color='r', linestyle='--', label='Mean Error')
plt.legend()
plt.grid()
plt.savefig('reconstruction_error.pdf', format='pdf')
plt.close()


n_samples = 30  # Number of samples to visualize
plt.figure(figsize=(15, 10))

for i in range(n_samples):
    plt.subplot(n_samples, 2, 2 * i + 1)
    plt.plot(X_time_series[i, :, 0], label='Original')
    plt.title(f'Original Data Sample {i + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(n_samples, 2, 2 * i + 2)
    plt.plot(X_pred[i, :, 0], label='Reconstructed', color='orange')
    plt.title(f'Reconstructed Data Sample {i + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.savefig('original_vs_reconstructed_samples.pdf', format='pdf')
plt.close()


print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R² Score: {r2:.4f}')

