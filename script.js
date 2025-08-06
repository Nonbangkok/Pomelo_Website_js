let model;
const fileInput = document.getElementById('file-input');
const selectedImage = document.getElementById('selected-image');
const predictButton = document.getElementById('predict-button');
const resultDiv = document.getElementById('result');
let imageName = '';
let currentImageSource = ''; // 'file' or 'sample'

// Sample images array - add all images from /img folder
const sampleImages = [
    { src: './img/1.jpg', name: 'Sample 1' },
    { src: './img/2.jpg', name: 'Sample 2' },
    { src: './img/3.jpg', name: 'Sample 3' },
    { src: './img/4.jpg', name: 'Sample 4' },
    { src: './img/5.jpg', name: 'Sample 5' },
    { src: './img/6.jpg', name: 'Sample 6' }
];

(async function() {
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log('Model loaded successfully');
        loadSampleImages(); // Load sample images after model loads
    } catch (error) {
        console.error('Error loading the model:', error);
        alert('Failed to load the model. Please check the console for more details.');
    }
})();

function loadSampleImages() {
    const sampleGrid = document.getElementById('sample-images-grid');
    
    sampleImages.forEach((image, index) => {
        const imageItem = document.createElement('div');
        imageItem.className = 'sample-image-item';
        imageItem.innerHTML = `
            <img src="${image.src}" alt="${image.name}" loading="lazy">
            <div class="image-label">${image.name}</div>
        `;
        
        imageItem.addEventListener('click', () => {
            // Remove selected class from all items
            document.querySelectorAll('.sample-image-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // Add selected class to clicked item
            imageItem.classList.add('selected');
            
            // Set the selected image
            selectedImage.src = image.src;
            selectedImage.style.display = 'block';
            selectedImage.classList.add('show');
            
            // Show predict button
            predictButton.style.display = 'block';
            
            // Set current image source and name
            currentImageSource = 'sample';
            imageName = image.name;
            
            // Clear file input
            fileInput.value = '';
        });
        
        sampleGrid.appendChild(imageItem);
    });
}

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        // Remove selected class from sample images
        document.querySelectorAll('.sample-image-item').forEach(item => {
            item.classList.remove('selected');
        });
        
        imageName = file.name;
        currentImageSource = 'file';
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedImage.src = e.target.result;
            selectedImage.style.display = 'block';
            selectedImage.offsetHeight;
            selectedImage.classList.add('show');
        };
        reader.readAsDataURL(file);
        predictButton.style.display = 'block';
    }
});

predictButton.addEventListener('click', async () => {
    if (!model) {
        alert('The model is not loaded yet. Please wait and try again.');
        return;
    }
    
    if (!selectedImage.src || selectedImage.src === '#') {
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

    const sourceText = currentImageSource === 'sample' ? 'Sample Image' : 'Uploaded File';
    resultDiv.innerHTML = `
        <strong>Image Name:</strong> ${imageName}<br>
        <strong>Type:</strong> ${sourceText}<br>
        <strong>Prediction:</strong> ${predictedClass}<br>
        <strong>Confidence:</strong> ${confidencePercentage}%
    `;
}

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-input");
    const predictButton = document.getElementById("predict-button");
    const selectedImage = document.getElementById("selected-image");

    predictButton.style.display = "none";

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];

        if (file) {
            // Remove selected class from sample images
            document.querySelectorAll('.sample-image-item').forEach(item => {
                item.classList.remove('selected');
            });
            
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