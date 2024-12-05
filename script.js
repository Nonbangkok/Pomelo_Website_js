// script.js

// Assuming you have a TensorFlow.js model saved at 'test_model/model.json'

let model;
const fileInput = document.getElementById('file-input');
const selectedImage = document.getElementById('selected-image');
const predictButton = document.getElementById('predict-button');
const resultDiv = document.getElementById('result');

// Load the model asynchronously
(async function() {
    try {
        model = await tf.loadLayersModel('teach/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading the model:', error);
        alert('Failed to load the model. Please check the console for more details.');
    }
})();

// Handle image selection
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedImage.src = e.target.result;
            selectedImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// Handle prediction
predictButton.addEventListener('click', async () => {
    if (!model) {
        alert('The model is not loaded yet. Please wait and try again.');
        return;
    }
    if (!fileInput.files[0]) {
        alert('Please select an image to predict.');
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = 'Predicting...';
    resultDiv.textContent = '';

    // Preprocess the image
    const imageTensor = preprocessImage(selectedImage);

    // Make prediction
    try {
        const prediction = await model.predict(imageTensor).data();
        displayResult(prediction);
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('An error occurred during prediction. Please check the console for more details.');
    }

    predictButton.disabled = false;
    predictButton.textContent = 'Predict';
});

// Preprocess the image to match the model's expected input
function preprocessImage(image) {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([256, 256]) // Change to the input size your model expects
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
    return tensor;
}

// Display the prediction result
function displayResult(prediction) {
    const classes = ['Diseased', 'Good']; // Replace with your actual class names
    const confidence = prediction[0];
    const predictedClass = confidence > 0.5 ? classes[0] : classes[1];

    resultDiv.textContent = `Prediction: ${predictedClass}`;
}