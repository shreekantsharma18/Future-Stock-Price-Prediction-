import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import os

app = Flask(__name__)
# Enable CORS for all routes, allowing requests from our frontend
CORS(app)

@app.route('/')
def index():
    return "Flask server is running. Send a POST request to /upload with your CSV file."

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['csvFile']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the uploaded file is named 'your_stock_data.csv'
    if file.filename != 'your_stock_data.csv':
        return jsonify({'error': 'Invalid file name. Please upload "your_stock_data.csv".'}), 400

    if file:
        try:
            # Read the CSV file into a pandas DataFrame
            # We use io.BytesIO to read the file directly from memory
            df = pd.read_csv(io.BytesIO(file.read()))

            # Validate required columns
            required_columns = ['Date', 'Actual Price', 'Predicted Price']
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'CSV must contain columns: {", ".join(required_columns)}'}), 400
            # Convert 'Date' column to datetime objects
            df['Date'] = pd.to_datetime(df['Date'])

            # Sort by date to ensure correct plotting order
            df = df.sort_values(by='Date')

            # Create the plot
            plt.figure(figsize=(12, 6)) # Set figure size for better readability
            plt.plot(df['Date'], df['Actual Price'], label='Actual Price', color='#4299e1', linewidth=2) # Blue line
            plt.plot(df['Date'], df['Predicted Price'], label='Predicted Price', color='#f6ad55', linestyle='--', linewidth=2) # Orange dashed line

            plt.title('Actual vs. Predicted Stock Prices', fontsize=16, color='#e2e8f0')
            plt.xlabel('Date', fontsize=12, color='#e2e8f0')
            plt.ylabel('Price', fontsize=12, color='#e2e8f0')
            plt.legend(fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7) # Add a subtle grid

            # Customize tick parameters for better dark mode visibility
            plt.tick_params(axis='x', colors='#e2e8f0')
            plt.tick_params(axis='y', colors='#e2e8f0')

            # Set plot background and border color
            plt.gca().set_facecolor('#2d3748') # Plot area background
            plt.gcf().set_facecolor('#2d3748') # Figure background
            plt.box(on=True) # Ensure box is drawn
            plt.gca().spines['top'].set_color('#e2e8f0')
            plt.gca().spines['bottom'].set_color('#e2e8f0')
            plt.gca().spines['left'].set_color('#e2e8f0')
            plt.gca().spines['right'].set_color('#e2e8f0')


            # Save the plot to a BytesIO object
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100) # Save as PNG
            img_buffer.seek(0) # Rewind the buffer to the beginning
            plt.close() # Close the plot to free up memory

            # Send the image back as a response
            return send_file(img_buffer, mimetype='image/png')

        except Exception as e:
            # Log the error for debugging
            print(f"Error processing file: {e}")
            return jsonify({'error': f'Error processing CSV file: {str(e)}. Please ensure it has "Date", "Actual Price", and "Predicted Price" columns and correct data format.'}), 500

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, you would use a WSGI server like Gunicorn or uWSGI
    app.run(debug=True) # debug=True allows for automatic reloading on code changes
