# predict_bitcoin_price.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime

def load_data(filename='bitcoin_prices.csv'):
    """
    Loads the Bitcoin price data from a CSV file.
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def visualize_data(df):
    """
    Visualizes the Bitcoin price data.
    """
    plt.figure(figsize=(14,7))
    plt.plot(df['Date'], df['Price'], label='Bitcoin Price')
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def preprocess_data(df, feature='Price', sequence_length=60):
    """
    Preprocesses the data for modeling.

    Parameters:
        df (DataFrame): The Bitcoin price data.
        feature (str): The feature to use for prediction.
        sequence_length (int): The number of past days to use for prediction.

    Returns:
        X, y: Features and target arrays.
        scaler: The fitted scaler object.
    """
    data = df[[feature]].values  # Use only the 'Price' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    # Reshape for RNN [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def split_data(X, y, train_size=0.8):
    """
    Splits the data into training and testing sets.

    Parameters:
        X, y: Features and target arrays.
        train_size (float): Proportion of data to use for training.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    split_index = int(len(X) * train_size)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

def build_rnn_model(input_shape):
    """
    Builds a simple RNN (LSTM) model.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        model: Compiled RNN model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next price

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn_model(input_shape):
    """
    Builds a simple CNN model for time series prediction.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        model: Compiled CNN model.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(df, train_size, scaler, y_train, y_test, y_pred, model_type='RNN'):
    """
    Plots the training data, actual prices, and predicted prices.

    Parameters:
        df (DataFrame): The original Bitcoin price data.
        train_size (float): Proportion of data used for training.
        scaler: The fitted scaler object.
        y_train, y_test: Actual scaled target values.
        y_pred: Predicted scaled target values.
        model_type (str): Type of the model ('RNN' or 'CNN').
    """
    # Inverse transform the scaled data
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Create a dataframe to hold actual and predicted values
    train_len = int(len(df) * train_size)
    df_plot = df.copy()
    df_plot['Predicted_Price'] = np.nan
    df_plot['Predicted_Price'].iloc[train_len + 60:] = y_pred_inv.flatten()

    plt.figure(figsize=(14,7))
    plt.plot(df_plot['Date'], df_plot['Price'], label='Actual Price')
    plt.plot(df_plot['Date'], df_plot['Predicted_Price'], label=f'Predicted Price ({model_type})')
    plt.title(f'Bitcoin Price Prediction using {model_type}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def main():
    # Load data
    df = load_data()
    print("Data Loaded Successfully.")
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    sequence_length = 60  # Number of past days to use for prediction
    X, y, scaler = preprocess_data(df, sequence_length=sequence_length)
    print(f"Data preprocessed: {X.shape}, {y.shape}")
    
    # Split data
    train_size = 0.8
    X_train, X_test, y_train, y_test = split_data(X, y, train_size=train_size)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # Build and train RNN model
    rnn_model = build_rnn_model((X_train.shape[1], 1))
    print("RNN Model Summary:")
    rnn_model.summary()
    
    history_rnn = rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Predict with RNN model
    y_pred_rnn = rnn_model.predict(X_test)
    
    # Evaluate RNN model
    rmse_rnn = np.sqrt(mean_squared_error(y_test, y_pred_rnn))
    print(f"RNN Model RMSE: {rmse_rnn}")
    
    # Plot RNN predictions
    plot_predictions(df, train_size, scaler, y_train, y_test, y_pred_rnn, model_type='RNN')
    
    # Optional: Build and train CNN model
    cnn_model = build_cnn_model((X_train.shape[1], 1))
    print("CNN Model Summary:")
    cnn_model.summary()
    
    history_cnn = cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Predict with CNN model
    y_pred_cnn = cnn_model.predict(X_test)
    
    # Evaluate CNN model
    rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
    print(f"CNN Model RMSE: {rmse_cnn}")
    
    # Plot CNN predictions
    plot_predictions(df, train_size, scaler, y_train, y_test, y_pred_cnn, model_type='CNN')
    
    # Save the models (optional)
    rnn_model.save('bitcoin_rnn_model.h5')
    cnn_model.save('bitcoin_cnn_model.h5')
    print("Models saved to disk.")

if __name__ == "__main__":
    main()
