// URL to the TensorFlow.js model (hosted on GitHub Pages or locally)
const modelURL = './tfjs_model/model.json';

let model;

// Load the TensorFlow.js model
async function loadModel() {
    model = await tf.loadGraphModel(modelURL);
    console.log("Model loaded successfully");
}

// Preprocess image for prediction
function preprocessImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 256; // Resize to match model input
            canvas.height = 256;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const input = tf.browser.fromPixels(imageData).toFloat().div(255).expandDims();
            resolve(input);
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

// Predict and display the result
async function predict(file) {
    const inputTensor = await preprocessImage(file);
    const predictions = model.predict(inputTensor);
    predictions.print();
    document.getElementById('result').innerText = `Prediction: ${predictions.dataSync()}`;
}

// Event listeners
document.getElementById('file-input').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        await predict(file);
    }
});

document.addEventListener('DOMContentLoaded', loadModel);