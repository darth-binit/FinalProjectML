import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
from Model.ResNet_Attn import ResNetAttention, BasicBlock, MutliheadAttention

# OPTIONAL: If you have a label mapping from class indices to class names
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]


def load_model(model_path, device):
    """
    Load your trained PyTorch model from a .pth checkpoint.
    """
    # Example: If you have your ResNetAttention in a separate file
    # from your_code import ResNetAttention, BasicBlock

    # 1. Instantiate model with same architecture
      # ADAPT to your actual import
    model = ResNetAttention(
        block=BasicBlock,
        layers=[2, 2, 2, 1],
        num_classes=7,
        use_cbam=True,
        use_multihead=True
    ).to(device)

    # 2. Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, image, device):
    """
    Given a PIL image, apply the necessary transforms, run inference,
    and return the predicted class index (and optionally probabilities).
    """
    # Define your test transforms - match training transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Convert PIL image to PyTorch tensor
    input_tensor = test_transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        # If multi-class classification, you might do argmax
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        # Optionally get softmax probabilities
        probs = torch.softmax(outputs, dim=1)[0]
        prob = probs[pred_idx].item()

    return pred_idx, prob


def main():
    st.set_page_config(page_title="Image Classification Demo", layout="centered")
    st.title("Image Classification with Streamlit")

    st.sidebar.title("Model & Settings")
    model_path = st.sidebar.text_input("Model Checkpoint Path", value="best_model_epoch_6.pth")
    device_option = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
    device = torch.device(device_option if torch.cuda.is_available() else "cpu")

    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            global model
            model = load_model(model_path, device)
        st.sidebar.success("Model loaded successfully!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Display the uploaded image using use_container_width
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify"):
            if 'model' not in globals():
                st.warning("Please load a model first from the sidebar.")
            else:
                with st.spinner("Classifying..."):
                    time.sleep(1)  # Optional: simulate processing delay
                    pred_idx, confidence = predict_image(model, image, device)
                    class_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
                st.success(f"Prediction: {class_name} (Confidence: {confidence * 100:.2f}%)")

    st.write("---")
    st.write("Created by [Your Name].")

if __name__ == "__main__":
    main()
