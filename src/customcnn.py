# Preprocessing & prediction utilities — matches your training notebook exactly
# Use these in your Flask app so deployed inference uses the SAME pipeline as training.

import os
import numpy as np
from PIL import Image, ImageChops
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io

print("🔄 Loading model... (only once)")
model = load_model("./models/model_casia_run2.h5")  # replace with your path
print("✅ Model Loaded Successfully!")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # match training input size
    img = img_to_array(img)
    img = img / 255.0            # normalize (must match training)
    img = np.expand_dims(img, axis=0)
    return img  

# --------------------- ELA (exact same as training) ---------------------
def ELA(img_path, quality=90, output_size=(224, 224), scale=10):
    """
    Error Level Analysis exactly as used in training.
    - img_path: path to image file (string)
    - quality: JPEG quality used for recompression (training used 90)
    - output_size: final ELA image size (training used (224,224))
    - scale: multiply factor for difference amplification (training used 10)
    Returns: PIL Image (RGB) sized output_size
    """
    TEMP = "ela_temp.jpg"

    # Ensure original is loaded in RGB (same as training)
    original = Image.open(img_path).convert("RGB")

    # Save a recompressed JPEG copy then reopen
    original.save(TEMP, quality=quality)
    temporary = Image.open(TEMP)

    # Pixel-wise difference
    diff = ImageChops.difference(original, temporary)

    # Convert to numpy, amplify and clip like training code
    diff_np = np.array(diff).astype(np.float32)
    diff_np = np.clip(diff_np * scale, 0, 255).astype(np.uint8)

    # Convert back to PIL and resize with Lanczos (same resampling)
    ela_rgb = Image.fromarray(diff_np)
    ela_resized = ela_rgb.resize(output_size, Image.Resampling.LANCZOS)

    # clean up temp file (best-effort)
    try:
        os.remove(TEMP)
    except Exception:
        pass

    return ela_resized


# --------------------- Preprocess (match training) ---------------------
def effnet_preprocess(image_path):
    """
    Replicates the pipeline used during training:
      1. Run ELA on input image
      2. Resize to (224,224) (ELA already returns this, but keep for safety)
      3. Convert to numpy array (dtype float32) WITHOUT scaling to [0,1]
         -> training used raw 0-255 pixel values in X (no division by 255)
      4. Return array shaped (224,224,3)

    Note: Your Flask code can save the uploaded file to disk (e.g. "uploaded.jpg")
    and pass that path into this function.
    """
    # Ensure path exists and run ELA exactly like training
    ela_img = ELA(image_path, quality=90, output_size=(224, 224), scale=10)

    # Safety: confirm RGB & size
    if ela_img.mode != "RGB":
        ela_img = ela_img.convert("RGB")
    if ela_img.size != (224, 224):
        ela_img = ela_img.resize((224, 224), Image.Resampling.LANCZOS)

    arr = np.array(ela_img)             # dtype will be uint8
    arr = arr.astype("float32")         # training used numeric arrays (float32 okay)
    return arr                           # shape (224,224,3), values ~0-255


# --------------------- Predict helper (use same logic as training notebook) ---------------------
def effnet_predict(model, image_path, class_labels=("Real", "Fake")):
    """
    model: loaded Keras model (same .h5 you saved)
    image_path: path to image file (string)
    class_labels: tuple/list mapping argmax -> label (training used ["Real","Fake"])
    Returns: (predicted_label, confidence_float, raw_prediction_array)
    """
    x = effnet_preprocess(image_path)         # (224,224,3)
    x = np.expand_dims(x, axis=0)                      # (1,224,224,3)
    # Note: do NOT apply extra normalization here (we're matching training)

    preds = model.predict(x)[0]                        # shape (2,) as in training
    predicted_idx = int(np.argmax(preds))              # training used argmax on output
    confidence = float(preds[predicted_idx])          # use raw sigmoid output for that class

    return class_labels[predicted_idx], confidence, preds


# --------------------- Example usage ---------------------
# from tensorflow.keras.models import load_model
# model = load_model("model_casia_run2.h5")
# label, conf, raw = predict_from_path(model, "uploaded.jpg")
# print(label, conf, raw)
