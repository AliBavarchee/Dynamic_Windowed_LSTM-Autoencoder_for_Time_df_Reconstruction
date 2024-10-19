# Multi-Window LSTM Autoencoder for Time Series Data Compression and Prediction

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



\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}

\begin{tikzpicture}[
  every node/.style={draw, rounded corners, align=center, minimum height=2em},
  layer/.style={minimum width=3em},
  line/.style={->, thick}
]

% Input
\node[layer] (input) {Input \\ Time-Series Data};

% Encoder
\node[layer, right=3em of input] (encoder_lstm) {LSTM \\ Layer};
\node[layer, right=3em of encoder_lstm] (encoder_dense) {Dense \\ Layer};

% Bottleneck
\node[layer, right=3em of encoder_dense] (bottleneck) {Latent \\ Space};

% Decoder
\node[layer, right=3em of bottleneck] (decoder_repeat) {Repeat \\ Vector};
\node[layer, right=3em of decoder_repeat] (decoder_lstm) {LSTM \\ Layer};
\node[layer, right=3em of decoder_lstm] (decoder_dense) {Dense \\ Layer};

% Output
\node[layer, right=3em of decoder_dense] (output) {Reconstructed \\ Time-Series Data};

% Connections
\draw[line] (input) -- (encoder_lstm);
\draw[line] (encoder_lstm) -- (encoder_dense);
\draw[line] (encoder_dense) -- (bottleneck);
\draw[line] (bottleneck) -- (decoder_repeat);
\draw[line] (decoder_repeat) -- (decoder_lstm);
\draw[line] (decoder_lstm) -- (decoder_dense);
\draw[line] (decoder_dense) -- (output);

\end{tikzpicture}

\end{document}

