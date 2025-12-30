// JavaScript for handling file upload and displaying the graph
document.addEventListener('DOMContentLoaded', () => {
    const csvFileInput = document.getElementById('csvFile');
    const uploadButton = document.getElementById('uploadButton');
    const graphImage = document.getElementById('graphImage');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');

    // Initial state setup for debugging: ensure image and spinner are correctly hidden/shown
    graphImage.classList.add('hidden');
    loadingSpinner.style.display = 'none';
    errorMessage.classList.add('hidden');

    uploadButton.addEventListener('click', async () => {
        console.log('Upload button clicked.'); // Debugging: Button click detected
        const file = csvFileInput.files[0];

        // Hide previous graph and error message, show spinner
        graphImage.classList.add('hidden');
        errorMessage.classList.add('hidden');
        errorMessage.textContent = '';
        loadingSpinner.style.display = 'block'; // Show spinner

        if (!file) {
            errorMessage.textContent = 'Please select a CSV file first.';
            errorMessage.classList.remove('hidden');
            loadingSpinner.style.display = 'none'; // Hide spinner
            console.log('No file selected.'); // Debugging: No file
            return;
        }

        if (file.name !== 'your_stock_data.csv') {
            errorMessage.textContent = 'Please upload a file named "your_stock_data.csv".';
            errorMessage.classList.remove('hidden');
            loadingSpinner.style.display = 'none'; // Hide spinner
            console.log('Incorrect file name selected:', file.name); // Debugging: Wrong file name
            return;
        }

        const formData = new FormData();
        formData.append('csvFile', file);
        console.log('Sending fetch request to backend...'); // Debugging: Sending request

        try {
            const response = await fetch('http://127.0.0.1:5000/upload', {
                method: 'POST',
                body: formData,
            });

            console.log('Fetch response received. Status:', response.status); // Debugging: Response status

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }

            const blob = await response.blob();
            console.log('Response converted to Blob. Blob type:', blob.type, 'size:', blob.size); // Debugging: Blob info

            if (blob.type.startsWith('image/')) {
                const imageUrl = URL.createObjectURL(blob);
                graphImage.src = imageUrl;
                graphImage.classList.remove('hidden'); // Show the image
                console.log('Image URL created and assigned:', imageUrl); // Debugging: Image displayed
            } else {
                // If the response is not an image, it might be a JSON error from Flask
                const textResponse = await blob.text();
                let parsedError = textResponse;
                try {
                    const jsonError = JSON.parse(textResponse);
                    if (jsonError.error) {
                        parsedError = jsonError.error;
                    }
                } catch (e) {
                    // Not a JSON error, use raw text
                }
                errorMessage.textContent = `Received non-image response from server: ${parsedError}`;
                errorMessage.classList.remove('hidden');
                console.error('Received non-image response:', textResponse); // Debugging: Non-image response
            }

        } catch (error) {
            console.error('Error during fetch or image processing:', error); // Debugging: General error
            errorMessage.textContent = `Failed to generate graph: ${error.message}`;
            errorMessage.classList.remove('hidden');
        } finally {
            loadingSpinner.style.display = 'none'; // Hide spinner
            console.log('Process finished. Spinner hidden.'); // Debugging: Process end
        }
    });
});
