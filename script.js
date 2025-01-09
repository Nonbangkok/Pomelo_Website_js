let model;
const fileInput = document.getElementById('file-input');
const selectedImage = document.getElementById('selected-image');
const predictButton = document.getElementById('predict-button');
const resultDiv = document.getElementById('result');
let imageName = '';

(async function() {
    try {
        model = await tf.loadLayersModel('model_30_no_augmented_json/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading the model:', error);
        alert('Failed to load the model. Please check the console for more details.');
    }
})();

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        imageName = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedImage.src = e.target.result;
            selectedImage.style.display = 'block';
            selectedImage.offsetHeight;
            selectedImage.classList.add('show');
        };
        reader.readAsDataURL(file);
    }
});

predictButton.addEventListener('click', async () => {
    if (!model) {
        alert('The model is not loaded yet. Please wait and try again.');
        return;
    }
    if (!fileInput.files[0]) {
        alert('Please select an image to predict.');
        return;
    }

    const loadingSpinner = document.getElementById('loading-spinner');
    loadingSpinner.style.display = 'block';

    predictButton.disabled = true;
    predictButton.textContent = 'Predicting...';
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    resultDiv.className = 'alert alert-secondary';

    const imageTensor = preprocessImage(selectedImage);

    try {
        const prediction = await model.predict(imageTensor).data();
        displayResult(prediction);
    } catch (error) {
        console.error('Error during prediction:', error);
        resultDiv.style.display = 'block';
        resultDiv.className = 'alert alert-danger';
        resultDiv.textContent = 'An error occurred during prediction. Check console for details.';
    } finally {
        loadingSpinner.style.display = 'none';
        predictButton.disabled = false;
        predictButton.textContent = 'Predict';
    }
});

function preprocessImage(image) {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([256, 256])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
    return tensor;
}

function displayResult(prediction) {
    const classes = ['Diseased', 'Good'];
    const confidence = prediction[0];
    let predictedClass, confidencePercentage;

    if (confidence > 0.5) {
        predictedClass = classes[1];
        confidencePercentage = (confidence * 100).toFixed(2);
    } else {
        predictedClass = classes[0];
        confidencePercentage = ((1 - confidence) * 100).toFixed(2);
    }

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    
    if (predictedClass === 'Good') {
        resultDiv.className = 'alert alert-success';
    } else {
        resultDiv.className = 'alert alert-danger';
    }

    resultDiv.innerHTML = `<strong>Image Name:</strong> ${imageName}<br><strong>Prediction:</strong> ${predictedClass}<br><strong>Confidence:</strong> ${confidencePercentage}%`;
}

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-input");
    const predictButton = document.getElementById("predict-button");
    const selectedImage = document.getElementById("selected-image");

    predictButton.style.display = "none";

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                selectedImage.src = e.target.result;
                selectedImage.style.opacity = "1";
                selectedImage.classList.add("show");
            };
            reader.readAsDataURL(file);
            predictButton.style.display = "block";
        } else {
            predictButton.style.display = "none";
        }
    });
});