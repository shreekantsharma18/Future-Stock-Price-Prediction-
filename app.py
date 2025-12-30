import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

@app.route('/')
def index():
    return "Flask server is running. Send a POST request to /upload with your CSV file."

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received a request to /upload endpoint.")

    if 'csvFile' not in request.files:
        print("Error: No file part in the request.")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['csvFile']

    if file.filename == '':
        print("Error: No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    if file.filename != 'your_stock_data.csv':
        print(f"Error: Invalid file name. Expected 'your_stock_data.csv', got '{file.filename}'.")
        return jsonify({'error': 'Invalid file name. Please upload "your_stock_data.csv".'}), 400

    if file:
        try:
            print(f"Processing file: {file.filename}")
            df = pd.read_csv(io.BytesIO(file.read()))

            required_columns = ['Date', 'Actual Price']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Missing required columns: {', '.join(required_columns)}.")
                return jsonify({'error': f'CSV must contain columns: {", ".join(required_columns)}'}), 400

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            actual_prices = df['Actual Price'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_actual_prices = scaler.fit_transform(actual_prices)

            look_back = 10
            X, y = create_dataset(scaled_actual_prices, look_back)

            if len(X) == 0:
                print(f"Error: Not enough data for LSTM with look_back={look_back}. Need at least {look_back + 1} data points.")
                return jsonify({'error': f'Not enough data for LSTM with look_back={look_back}. Need at least {look_back + 1} data points.'}), 400

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            print("Training LSTM model...")
            model.fit(X, y, epochs=20, batch_size=32, verbose=0)
            print("LSTM model training complete.")

            train_predict_scaled = model.predict(X)
            train_predict = scaler.inverse_transform(train_predict_scaled)

            df['LSTM Predicted Price'] = np.nan
            df.loc[look_back:len(train_predict) + look_back - 1, 'LSTM Predicted Price'] = train_predict[:, 0]

            num_future_steps = 7
            future_predictions_scaled = []
            last_sequence = scaled_actual_prices[-look_back:].reshape(1, look_back, 1)

            for _ in range(num_future_steps):
                next_prediction_scaled = model.predict(last_sequence)
                future_predictions_scaled.append(next_prediction_scaled[0, 0])
                last_sequence = np.append(last_sequence[:, 1:, :], next_prediction_scaled.reshape(1, 1, 1), axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(num_future_steps)]

            future_df = pd.DataFrame({
                'Date': future_dates,
                'Actual Price': np.full(num_future_steps, np.nan),
                'LSTM Predicted Price': future_predictions[:,0]
            })

            plot_df = pd.concat([df, future_df], ignore_index=True)

            print("Generating plot...")
            plt.figure(figsize=(14, 7))
            ax = plt.gca() # Get current axes for plotting segments

            # --- Plotting Actual Price with Up/Down Colors ---
            # Calculate daily change for actual prices
            actual_price_diff = plot_df['Actual Price'].diff()
            for i in range(1, len(plot_df)):
                if pd.notna(plot_df['Actual Price'].iloc[i]) and pd.notna(plot_df['Actual Price'].iloc[i-1]):
                    color = '#28a745' if actual_price_diff.iloc[i] >= 0 else '#dc3545' # Green for up, Red for down
                    ax.plot(plot_df['Date'].iloc[i-1:i+1], plot_df['Actual Price'].iloc[i-1:i+1],
                            color=color, linewidth=2)
            # Add a single label for the actual price line in the legend
            ax.plot([], [], label='Actual Price (Up/Down)', color='#28a745', linewidth=2)


            # --- Plotting LSTM Predicted Price with Up/Down Colors ---
            # Calculate daily change for predicted prices
            predicted_price_diff = plot_df['LSTM Predicted Price'].diff()
            for i in range(1, len(plot_df)):
                if pd.notna(plot_df['LSTM Predicted Price'].iloc[i]) and pd.notna(plot_df['LSTM Predicted Price'].iloc[i-1]):
                    color = '#17a2b8' if predicted_price_diff.iloc[i] >= 0 else '#ffc107' # Cyan for predicted up, Yellow for predicted down
                    ax.plot(plot_df['Date'].iloc[i-1:i+1], plot_df['LSTM Predicted Price'].iloc[i-1:i+1],
                            color=color, linestyle='--', linewidth=2)
            # Add a single label for the predicted price line in the legend
            ax.plot([], [], label=f'LSTM Predicted Price (Up/Down, Lookback={look_back})', color='#17a2b8', linestyle='--', linewidth=2)


            # Highlight the future prediction area (using the original predicted line for fill)
            if num_future_steps > 0:
                plt.axvline(x=last_date, color='#cbd5e0', linestyle=':', linewidth=1.5, label='Last Actual Date')
                # Use the original predicted series for fill_between to avoid complex segment logic
                plt.fill_between(plot_df['Date'], 0, plot_df['LSTM Predicted Price'],
                                 where=(plot_df['Date'] > last_date),
                                 color='#f6ad55', alpha=0.1, transform=plt.gca().get_xaxis_transform(),
                                 label='Future Prediction Area')


            plt.title('Actual vs. LSTM Predicted Stock Prices (Up/Down Format)', fontsize=18, color='#e2e8f0')
            plt.xlabel('Date', fontsize=14, color='#e2e8f0')
            plt.ylabel('Price', fontsize=14, color='#e2e8f0')
            plt.legend(fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7, color='#a0aec0')

            plt.tick_params(axis='x', colors='#e2e8f0', labelsize=10)
            plt.tick_params(axis='y', colors='#e2e8f0', labelsize=10)

            # Set y-axis limits to 0-300
            plt.ylim(0, 300)

            plt.gca().set_facecolor('#2d3748')
            plt.gcf().set_facecolor('#2d3748')
            plt.box(on=True)
            plt.gca().spines['top'].set_color('#e2e8f0')
            plt.gca().spines['bottom'].set_color('#e2e8f0')
            plt.gca().spines['left'].set_color('#e2e8f0')
            plt.gca().spines['right'].set_color('#e2e8f0')

            plt.tight_layout()

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            img_buffer.seek(0)
            plt.close()
            print("Plot generated and sent.")
            return send_file(img_buffer, mimetype='image/png')

        except Exception as e:
            print(f"Unhandled exception during file processing or LSTM: {e}")
            return jsonify({'error': f'Error processing CSV file or training LSTM: {str(e)}. Please ensure it has "Date" and "Actual Price" columns and correct data format.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
