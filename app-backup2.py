# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import logging

# ----------------------------
# Configuration
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.secret_key = 'your-secret-key-here'  # Add secret key for flash messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Models with error handling
UNET_MODEL_PATH = 'model/unet_cdr_model.h5'  # Fixed path
RESNET_MODEL_PATH = 'model/glaucoma-DenseNet121.h5'  # Fixed path

try:
    logger.info("Loading U-Net model...")
    def dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return 1.0 - dice
    unet_model = load_model(UNET_MODEL_PATH, custom_objects={'dice_loss': dice_loss}, compile=False)
    logger.info("U-Net model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load U-Net model: {e}")
    unet_model = None

try:
    logger.info("Loading Densenet-121 model...")
    resnet_model = load_model(RESNET_MODEL_PATH)
    logger.info("Densenet-121 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Densenet-121 model: {e}")
    resnet_model = None

# ----------------------------
# Utility Functions
# ----------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def compute_cdr(disc_mask, cup_mask):
    """Compute Cup-to-Disc Ratio from segmentation masks"""
    def get_diameter(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return max(w, h)
    
    disc_diam = get_diameter(disc_mask)
    cup_diam = get_diameter(cup_mask)
    cdr = cup_diam / (disc_diam + 1e-6)
    return cdr

def get_last_conv_layer_name(model):
    """Find the last convolutional layer before the dense head."""
    conv_layers = []
    for layer in model.layers:
        if 'conv' in layer.name and 'block' in layer.name:
            conv_layers.append(layer.name)
    return conv_layers[-1] if conv_layers else None

def make_gradcam_heatmap(img_array, model):
    """Generate GradCAM heatmap for model attention visualization"""
    try:
        # Get the last convolutional layer (finds the last layer before dense head)
        conv_layers = []
        for layer in model.layers:
            if 'conv' in layer.name and 'block' in layer.name and 'concat' in layer.name:
                conv_layers.append(layer)

        if not conv_layers:
            raise ValueError("No suitable convolutional layer found for Grad-CAM")

        last_conv_layer = conv_layers[-1]
        print(f"✅ Using layer '{last_conv_layer.name}' for Grad-CAM")

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 1]  # Class 1 = Glaucoma

        gradients = tape.gradient(loss, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Error generating GradCAM: {e}")
        return None

def save_gradcam(img_path, heatmap, cam_path, alpha=0.6):
    """Save GradCAM visualization"""
    try:
        # Load original image and resize to 256x256 for consistent display
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # Target display size

        # Resize heatmap from 224x224 to 256x256
        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap = np.uint8(255 * heatmap)
        jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose on original
        superimposed = cv2.addWeighted(jet_heatmap, alpha, img, 1 - alpha, 0)
        plt.figure(figsize=(6, 6))
        plt.imshow(superimposed)
        plt.title("Grad-CAM (Glaucoma Attention)")
        plt.axis('off')
        plt.savefig(cam_path, bbox_inches='tight', dpi=150)
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Error saving GradCAM: {e}")
        return False

# Post-processing identical to training script

def postprocess_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def plot_results(image, pred_disc, pred_cup, cdr, filename):
    """Create segmentation visualization"""
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(pred_disc, cmap='gray')
        plt.title("Predicted Disc")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(pred_cup, cmap='gray')
        plt.title("Predicted Cup")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        overlay = (image * 255).astype(np.uint8).copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        disc_overlay = np.dstack([pred_disc*255, np.zeros_like(pred_disc), np.zeros_like(pred_disc)]).astype(np.uint8)
        cup_overlay = np.dstack([np.zeros_like(pred_cup), np.zeros_like(pred_cup), pred_cup*255]).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.7, disc_overlay, 0.7, 0)
        overlay = cv2.addWeighted(overlay, 1.0, cup_overlay, 0.7, 0)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted (CDR: {cdr:.3f})")
        plt.axis('off')

        plt.suptitle(f"CDR: {cdr:.3f}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['RESULTS_FOLDER'], f"{filename}_segmentation.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        return False

# ----------------------------
# Routes
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Load original image
                img = cv2.imread(filepath)
                if img is None:
                    flash('Invalid image file')
                    return redirect(request.url)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # U-Net: 256x256
                img_unet = cv2.resize(img, (256, 256))
                img_unet_normalized = img_unet.astype(np.float32) / 255.0

                # ResNet50: 224x224
                img_resnet = cv2.resize(img, (224, 224))
                img_resnet_normalized = img_resnet.astype(np.float32) / 255.0

                # Predict with U-Net
                if unet_model is None:
                    flash('U-Net model not available')
                    return redirect(request.url)
                
                pred_disc_out, pred_cup_out = unet_model.predict(img_unet_normalized[None, ...])
                pred_disc = (pred_disc_out[0, :, :, 0] > 0.5).astype(np.uint8)
                pred_cup = (pred_cup_out[0, :, :, 0] > 0.5).astype(np.uint8)
                # Apply the same post-processing as in training evaluation
                pred_disc = postprocess_mask(pred_disc)
                pred_cup = postprocess_mask(pred_cup)
                cdr = compute_cdr(pred_disc, pred_cup)

                # Predict with Densenet-121
                if resnet_model is None:
                    flash('Densenet-121 model not available')
                    return redirect(request.url)
                
                resnet_pred = resnet_model.predict(img_resnet_normalized[None, ...])
                prediction_prob = resnet_pred[0][0]
                classification = "Glaucoma" if prediction_prob > 0.5 else "Normal"
                confidence = prediction_prob if classification == "Glaucoma" else (1 - prediction_prob)

                # Generate Grad-CAM
                heatmap = make_gradcam_heatmap(img_resnet_normalized[None, ...], resnet_model)
                cam_path = os.path.join(app.config['RESULTS_FOLDER'], f"{filename}_gradcam.png")
                gradcam_success = False
                if heatmap is not None:
                    gradcam_success = save_gradcam(filepath, heatmap, cam_path)

                # Save segmentation result
                segmentation_success = plot_results(img_unet, pred_disc, pred_cup, cdr, filename)

                return render_template('result.html',
                                       filename=filename,
                                       cdr=round(cdr, 3),
                                       classification=classification,
                                       confidence=round(confidence, 3),
                                       result_image=f"{filename}_segmentation.png",
                                       gradcam_image=f"{filename}_gradcam.png",
                                       gradcam_available=gradcam_success,
                                       segmentation_available=segmentation_success)

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload JPG, JPEG, or PNG files only.')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_image(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Check if models are loaded
    if unet_model is None or resnet_model is None:
        logger.warning("Some models failed to load. The application may not work properly.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
