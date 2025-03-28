import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from Project_File.config.configuration import GradCAM, compute_cam_and_overlay
from Project_File.Model.ResNet_Attn import ResNetAttention, BasicBlock, ChBAM
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/binit/PycharmProjects/FinalProject/Project_File/data/HAM10000_metadata.csv')


st.set_page_config(page_title="Skin Image Lesion Classification", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: left; font-size: 2.0em;'>Skin Cancer Image Lesion Classification</h1>", unsafe_allow_html=True)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Device selection (example for CPU vs. cuda)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load lesion details from JSON file
@st.cache_data
def load_lesion_details(json_path):
    with open(json_path, "r") as f:
        details = json.load(f)
    return details

# Adjust the path to your JSON file
lesion_details = load_lesion_details("/Users/binit/PycharmProjects/FinalProject/Project_File/Front_End/explain.json")

@st.cache_resource
def load_model(model_path, device):
    # Instantiate your model with the same architecture as during training.
    model = ResNetAttention(
        block=BasicBlock,
        layers=[2, 2, 2, 1],  # Must match your training configuration.
        num_classes=7,
        use_cbam=True,
        use_multihead=True
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image, device):
    """
    Given a PIL image, apply the test transform, run inference,
    and return the predicted class index and probability.
    """
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        # For debugging: print raw outputs
        print("Raw model outputs:", outputs)
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=1)[0]
        # Get the predicted class and its probability
        pred_idx = torch.argmax(probs).item()
        prob = probs[pred_idx].item()

    return pred_idx, prob


# Display GradCAM for multiple layers in a grid
def display_gradcam_flow(model, input_tensor, pil_img):
    """
    Creates one figure per layer (heatmap & overlay side by side).
    After each layer's figure, show a down arrow in the Streamlit UI.
    """
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    layers = [model.layer1, model.layer2, model.layer3, model.cbam, model.mha, model.layer4]
    names = ['layer1_conv', 'layer2_conv', 'layer3_conv', 'cbam', 'Attention', 'layer4_conv']

    for i, (layer, layer_name) in enumerate(zip(layers, names)):
        gradcam_current = GradCAM(model, target_layer=layer)
        cam_np = gradcam_current.generate_cam(input_tensor, class_idx=None)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_resized = cv2.resize(heatmap, (pil_img.width, pil_img.height))

        overlayed = cv2.addWeighted(np.array(pil_img), 0.5, heatmap_resized, 0.5, 0)

        # Create a figure with 1 row, 2 columns for this single layer
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

        axes[0].imshow(heatmap_resized)
        axes[0].set_title(f"{layer_name} Heatmap", fontsize=8)
        axes[0].axis('off')

        axes[1].imshow(overlayed)
        axes[1].set_title(f"{layer_name} Overlay", fontsize=8)
        axes[1].axis('off')

        fig.tight_layout()
        st.pyplot(fig)

        # If not the last layer, insert a down arrow in the page
        if i < len(layers) - 1:
            st.markdown("<center style='font-size:2em;'>⬇️</center>", unsafe_allow_html=True)


def show_image_hist_and_button(image):
    """
    Creates a two-column layout for the image & "Classify" button on the left
    and the RGB histogram on the right.
    """
    col_left, col_right = st.columns([3, 2])

    # LEFT COLUMN: Show the original image, then place the "Classify" button below it.
    with col_left:
        st.image(image, caption="Uploaded Your Image", use_container_width=True)
        classify_clicked = st.button("Classify", use_container_width=True)  # The button is in the left column

    # RIGHT COLUMN: Plot the RGB histogram
    with col_right:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 6))
        np_img = np.array(image)
        r_vals = np_img[:, :, 0].ravel()
        g_vals = np_img[:, :, 1].ravel()
        b_vals = np_img[:, :, 2].ravel()

        axes[0].hist(r_vals, bins=256, color='red', alpha=0.7)
        axes[0].set_title('Red Channel')
        axes[0].set_xlim([0, 256])

        axes[1].hist(g_vals, bins=256, color='green', alpha=0.7)
        axes[1].set_title('Green Channel')
        axes[1].set_xlim([0, 256])

        axes[2].hist(b_vals, bins=256, color='blue', alpha=0.7)
        axes[2].set_title('Blue Channel')
        axes[2].set_xlim([0, 256])

        fig.tight_layout()
        st.pyplot(fig)

    # Return whether the "Classify" button was clicked
    return classify_clicked


def main():

    css_path = "/Users/binit/PycharmProjects/FinalProject/Project_File/Front_End/style.css"
    with open(css_path) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.sidebar.title("Model & Settings")
    model_path = "/Users/binit/PycharmProjects/FinalProject/Project_File/model_checkpoint/cnn_attn_chk_pt/best_model_epoch_13.pth"

    selected_age = st.sidebar.slider("Age", min_value=df['age'].min().astype('int'),
                                     max_value=df['age'].max().astype('int'), value=df['age'].min().astype('int'),  step=1)
    selected_localization = st.sidebar.selectbox("Localization", df['localization'].unique().tolist())
    selected_sex = st.sidebar.selectbox("Gender", df['sex'].unique().tolist())
    selected_dx_type = st.sidebar.selectbox("DX Type", df['dx_type'].unique().tolist())
    selected_dataset = st.sidebar.selectbox("Dataset", df['dataset'].unique().tolist())

    with st.spinner("Loading model..."):
        model = load_model(model_path, device)
    st.sidebar.success("Model loaded successfully!")

    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Show columns: image + classify button on left, histogram on right
        classify_clicked = show_image_hist_and_button(image)

        if classify_clicked:
            with st.spinner("Classifying..."):
                pred_idx, confidence = predict_image(model, image, device)
                class_name = f"Class {pred_idx}"

                # Grad-CAM
                input_tensor = test_transform(image).unsqueeze(0).to(device)
                gradcam = GradCAM(model, target_layer=model.layer4)
                display_gradcam_flow(model, input_tensor, image)

                # Lesion details
                lesion_text = lesion_details.get(str(pred_idx), "No details available.")
                parts = lesion_text.split('\n', 1)
                lesion_name = parts[0].strip()
                lesion_description = parts[1].strip() if len(parts) > 1 else ""
                lesion_html = f"""<div class="diagnosis" style="text-align:center;">{lesion_name}</div>"""
                st.sidebar.markdown(lesion_html, unsafe_allow_html=True)
                st.sidebar.markdown("<br>", unsafe_allow_html=True)
                st.sidebar.info(f"{lesion_description}")

            st.success(f"Prediction: {class_name} (Confidence: {confidence * 100:.2f}%)")

    st.write("---")

if __name__ == "__main__":
    main()
