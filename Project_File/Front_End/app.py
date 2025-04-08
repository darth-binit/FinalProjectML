import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os
# Add the root directory of your project to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

import matplotlib.pyplot as plt
import pandas as pd


import seaborn as sns
import plotly.figure_factory as ff

from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import base64

from Project_File.config.configuration import GradCAM
from Project_File.Model.ResNet_Attn import ResNetAttention, BasicBlock
import cv2


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

data_path = os.path.join(project_dir, "data", "HAM10000_metadata.csv")
df = pd.read_csv(data_path)

st.set_page_config(page_title="Skin Image Lesion Classification", layout="wide", initial_sidebar_state="expanded")


def get_base64_image(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


current_dir = os.path.dirname(os.path.abspath(__file__))

# Build dynamic paths
img_path_main = os.path.join(current_dir, "back-1.jpg")
image_path_side = os.path.join(current_dir, "61802.jpg")
img_base64_main = get_base64_image(img_path_main)
img_base64_side = get_base64_image(image_path_side)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img_base64_main}");
    background-size: cover;
    background-position: center;
}}
[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}
[data-testid="stSidebar"]{{
    background-image: url("data:image/jpg;base64,{img_base64_side}");
    background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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
lesion_path = os.path.join(current_dir, "explain.json")
lesion_details = load_lesion_details(lesion_path)

@st.cache_resource
def load_model(model_path, device, selected_model):
    if selected_model == 'CNN_With_Attention':
     # Instantiate your model with the same architecture as during training.
        model = ResNetAttention(
            block=BasicBlock,
            layers=[3, 3, 2, 2],  # Must match your training configuration.
            num_classes=7,
            use_cbam=True,
            use_multihead=True
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        model = ResNetAttention(
            block=BasicBlock,
            layers=[2, 2, 2, 1],  # Must match your training configuration.
            num_classes=7,
            use_cbam=False,
            use_multihead=False
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
def get_gradcam_layers(model):
    """
    Dynamically get the layers for GradCAM visualization.
    Works for both CNN with and without attention blocks.
    """
    layers = []
    if hasattr(model, 'layer1'):
        layers.append((model.layer1, 'layer1_conv'))
    if hasattr(model, 'layer2'):
        layers.append((model.layer2, 'layer2_conv'))
    if hasattr(model, 'layer3'):
        layers.append((model.layer3, 'layer3_conv'))
    if hasattr(model, 'cbam'):
        layers.append((model.cbam, 'cbam'))
    if hasattr(model, 'mha'):
        layers.append((model.mha, 'Attention'))
    if hasattr(model, 'layer4'):
        layers.append((model.layer4, 'layer4_conv'))
    return layers

def display_gradcam_flow(model, input_tensor, pil_img):
    """
    Creates one figure per layer (heatmap & overlay side by side) with a transparent background.
    After each layer's figure, an alternate down arrow (⏬) is displayed in the Streamlit UI.
    """
    layers = get_gradcam_layers(model)
    # layers = [model.layer1, model.layer2, model.layer3, model.cbam, model.mha, model.layer4]
    # names = ['layer1_conv', 'layer2_conv', 'layer3_conv', 'cbam', 'Attention', 'layer4_conv']

    for i, (layer, layer_name) in enumerate(layers):
        gradcam_current = GradCAM(model, target_layer=layer)
        cam_np = gradcam_current.generate_cam(input_tensor, class_idx=None)
        # Generate heatmap using cv2
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_resized = cv2.resize(heatmap, (pil_img.width, pil_img.height))

        overlayed = cv2.addWeighted(np.array(pil_img), 0.5, heatmap_resized, 0.5, 0)

        # Create a figure with 1 row and 2 columns for this layer
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

        #Set figure and axes background transparent
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.set_facecolor("none")

        # Display the heatmap with a colorbar
        im = axes[0].imshow(heatmap_resized, cmap='jet')
        axes[0].set_title(f"{layer_name} Heatmap", fontsize=8)
        axes[0].axis('off')
        fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.04, shrink=0.6)

        # Display the overlay image
        axes[1].imshow(overlayed)
        axes[1].set_title(f"{layer_name} Overlay", fontsize=8.5)
        axes[1].axis('off')

        fig.tight_layout()
        st.pyplot(fig)

        # If not the last layer, display a down arrow between figures
        if i < len(layers) - 1:
            st.markdown("<center style='font-size:2em;'>⏬</center>", unsafe_allow_html=True)


def show_image_hist_and_button(image):
    """
    Creates a two-column layout for the image & "Classify" button on the left
    and the RGB histogram on the right. The histogram figure is styled for visibility on any background.
    """
    col_left, col_right = st.columns([3, 2])

    # LEFT COLUMN
    with col_left:
        st.image(image, caption="Uploaded Your Image", use_container_width=True)
        classify_clicked = st.button("Classify", use_container_width=True)

    # RIGHT COLUMN: Transparent background histograms
    with col_right:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 6))

        # Set figure background transparent
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)

        np_img = np.array(image)
        r_vals = np_img[:, :, 0].ravel()
        g_vals = np_img[:, :, 1].ravel()
        b_vals = np_img[:, :, 2].ravel()

        channel_data = [
            ('Red Channel', r_vals, (1.0, 0.0, 0.0)),
            ('Green Channel', g_vals, (0.0, 0.8, 0.0)),
            ('Blue Channel', b_vals, (0.0, 0.4, 1.0))
        ]

        for ax, (title, data, color) in zip(axes, channel_data):
            ax.hist(data,bins=256,color=color,edgecolor='black',linewidth=0.4,alpha=0.90)
            ax.set_title(title, fontsize=10)
            ax.set_xlim([0, 256])
            ax.set_facecolor("none")  # Transparent axes
            ax.tick_params(colors='black')
            ax.set_ylabel("Count", fontsize=8)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)

    return classify_clicked


def highlight_top_n_regions(cam: np.ndarray, base_image: Image.Image, N: int = 10, mode: str = "box") -> np.ndarray:

    cam_resized = cv2.resize(cam, base_image.size)
    cam_norm = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8)

    # Determine threshold using top-N values
    flattened = cam_norm.flatten()

    # Instead of using top-N
    threshold = np.percentile(flattened, 85)  # top 5% pixels
    mask = cam_norm >= threshold

    # Create binary mask for top regions
    mask = cam_norm >= threshold
    mask_uint8 = np.uint8(mask * 255)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert PIL image to NumPy RGB
    base_np = np.array(base_image.convert("RGB"))

    drawn = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:  # Ignore tiny specks
            continue

        drawn = True
        if mode == "box":
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            cv2.rectangle(base_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif mode == "contour":
            cv2.drawContours(base_np, [cnt], -1, (255, 0, 0), 2)

    # Fallback: if nothing drawn, display a notice and return original image
    if not drawn:
        st.warning("No high-activation regions found. Try lowering N or using contour mode.")

    return base_np


def streamlit_menu():

    st.markdown("""<style>.option-menu-container {margin-top: 0px !important;}</style>""", unsafe_allow_html=True)

    # Use a container with the custom class to wrap the option menu
    with st.container():
        st.markdown('<div class="option-menu-container">', unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,  # required
            options=["Home | Descriptive", "Predictive Analytics"],
            icons=["bi bi-activity", "bi bi-clipboard-data", "bi bi-pie-chart"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#e6f7ff"},##C6DDB6
                "icon": {"color": "#072810", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#cceeff",
                    "padding": "10px"
                },
                "nav-link-selected": {"background-color": "#009688"},
            },
        )
        st.markdown('</div>', unsafe_allow_html=True)
    return selected

select  = streamlit_menu()


def main():
    css_path = os.path.join(current_dir, "style.css")

    with open(css_path) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.sidebar.markdown("""
        <div style="margin-top: 40px;">
          <h1 style="font-size: 1.5em; color: black;">Skin Legion Classification</h1>
        </div>""",unsafe_allow_html=True)

    preview_image_path = os.path.join(current_dir, "paper_preview.png")
    # Sidebar expander for research paper
    with st.sidebar.expander("📄 Attention-Based CNN Paper ", expanded=False):
        st.image(preview_image_path, caption="Click below to read full paper", use_container_width=True)

        paper_url = "https://www.researchgate.net/publication/370388454_An_Attention-Based_Convolutional_Neural_Network_for_Intrusion_Detection_Model"
        st.markdown(f"[🔗 Read Full Paper]({paper_url})", unsafe_allow_html=True)


    if select == "Home | Descriptive":
        csv_mod_path = os.path.join(project_dir, "data", "df.csv")
        df_mod = pd.read_csv(csv_mod_path)

        selected_dataset = st.sidebar.selectbox("Select Feature", ['Age', 'dx_type', 'cell_type', 'localization'])

        if selected_dataset == 'cell_type':
            #Figure 1
            sns.set_palette("crest")
            # Create figure and axes with desired size
            fig, axes = plt.subplots(figsize=(12, 8))
            # Set figure and axes background to transparent
            fig.patch.set_facecolor("none")
            axes.set_facecolor("none")

            # Plot the countplot on the axes
            ax = sns.countplot(x='cell_type', data=df_mod,order=df_mod['cell_type'].value_counts().index, palette='crest', hue='cell_type',ax=axes)
            for container in ax.containers:
                ax.bar_label(container)
            plt.title('Cell Types Skin Cancer Affected patients')
            plt.xticks(rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Instead of plt.show(), use st.pyplot
            st.pyplot(fig)

            #Figure 2
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor("none")  # Make figure background transparent
            ax.set_facecolor("none")  # Make axes background transparent

            ax = sns.countplot(x='cell_type', hue='sex', data=df_mod,
                               order=df_mod['cell_type'].value_counts().index,
                               palette='crest', ax=ax)
            for container in ax.containers:
                ax.bar_label(container)

            ax.set_title('Cell Types Frequencies')
            plt.xticks(rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Make legend background transparent
            legend = ax.get_legend()
            legend.get_frame().set_facecolor('none')  # fully transparent
            legend.get_frame().set_alpha(0.0)  # ensure alpha is 0
            legend.get_frame().set_edgecolor('none')  # optional: remove legend border

            st.pyplot(fig)

            #figure 3
            maskcsv_mod_path = os.path.join(project_dir, "data", "mask_statistics.csv")
            mask_stats_df = pd.read_csv(maskcsv_mod_path)

            def display_mask_stats_table(df):
                fig = go.Figure(data=[go.Table(
                    columnwidth=[110] + [70] * (len(df.columns) - 1),
                    header=dict(
                        values=[f"<b>{col}</b>" for col in df.columns],
                        fill_color="#009fb7",  # Darker blue to match chart
                        font=dict(color='white', size=12),
                        align="center",
                        height=30,
                        line_color='white'
                    ),
                    cells=dict(
                        values=[df[col] for col in df.columns],
                        fill_color='rgba(255,255,255,0)',
                        font=dict(color='black', size=11),
                        align="center",
                        height=26,
                        line_color='gainsboro'
                    )
                )])

                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=10),
                    height=480,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )

                # Centered and smaller heading
                st.markdown("<h5 style='text-align: center;'>Lesion Mask Statistics Overview</h5>",
                            unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

            display_mask_stats_table(mask_stats_df)


        elif selected_dataset == "localization":
            #Figure 1
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor("none")  # Make figure background transparent
            ax.set_facecolor("none")  # Make axes background transparent

            ax = sns.countplot(x='localization',data=df_mod,order=df_mod['localization'].value_counts().index,palette='crest',hue='localization',
                ax=ax)
            for container in ax.containers:
                ax.bar_label(container)

            ax.set_title('Localization Area Frequencies')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            #Figure 2
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor("none")  # Make figure background transparent
            ax.set_facecolor("none")
            ax = sns.countplot(x='localization', hue='sex', data=df_mod, order=df['localization'].value_counts().index,
                               palette='crest')
            for container in ax.containers:
                ax.bar_label(container)
            plt.title('Localization Area Frequencies')
            plt.xticks(rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Make legend background transparent
            legend = ax.get_legend()
            legend.get_frame().set_facecolor('none')  # fully transparent
            legend.get_frame().set_alpha(0.0)  # ensure alpha is 0
            legend.get_frame().set_edgecolor('none')  # optional: remove legend border

            st.pyplot(fig)

        elif selected_dataset == "Age":
            #Figure 1
            clean_age = df_mod['age'].replace([np.inf, -np.inf], np.nan).dropna()
            data = [clean_age]
            group_labels = ['Age']

            fig = ff.create_distplot(data, group_labels, show_hist=True, show_rug=False)

            # Update the layout to use "Age Distribution" as the title, centered.
            fig.update_layout(
                title={
                    'text': "Age Distribution",'y': 0.95,  'x': 0.5,  'xanchor': 'center','yanchor': 'top'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black'))

            st.plotly_chart(fig, use_container_width=True)

            #Figure 2
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.set_style('whitegrid')

            # Transparent figure and axis background
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            # Create the histogram
            ax = sns.histplot(data=df_mod, x='age', hue='cell_type', multiple='stack', kde=True, ax=ax)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Title
            plt.title('Age Histogram Cell Type Wise')

            # Make legend background transparent
            legend = ax.get_legend()
            legend.get_frame().set_facecolor('none')  # fully transparent
            legend.get_frame().set_alpha(0.0)  # ensure alpha is 0
            legend.get_frame().set_edgecolor('none')  # optional: remove legend border

            # Display in Streamlit
            st.pyplot(fig)

            #Figure 3
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.set_style('whitegrid')
            # Make the figure and axes transparent
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")
            ax = sns.histplot(data=df_mod, x='age', hue='localization', multiple='stack')
            plt.title('Age Histogram Localization Area Wise')

            # Make legend background transparent
            legend = ax.get_legend()
            legend.get_frame().set_facecolor('none')  # fully transparent
            legend.get_frame().set_alpha(0.0)  # ensure alpha is 0
            legend.get_frame().set_edgecolor('none')  # optional: remove legend border

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

        elif selected_dataset == "dx_type":
            fig, ax = plt.subplots(figsize=(12, 8))
            ax = sns.countplot(x='dx_type', data=df, order=df['dx_type'].value_counts().index, palette='crest',
                               hue='dx_type')
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")
            for container in ax.containers:
                ax.bar_label(container)
            plt.title('Cell Types Frequencies')
            plt.xticks(rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)


    if select == "Predictive Analytics":
        uploaded_file = st.file_uploader("Upload Image for Classification", type=["jpg", "jpeg", "png"])

        select_model = st.sidebar.selectbox("Select Model", ['CNN_With_Attention', 'Vanilla CNN'])

        if select_model == 'CNN_With_Attention':
            model_path = os.path.join(project_dir, "model_checkpoint", "cnn_attn_chk_pt", "fusion_best_model_epoch_20.pth")
        else: #Focal_best_model_epoch_30.pth
            model_path = os.path.join(project_dir, "model_checkpoint", "cnn_chk_pt", "No_Attn_best_model_epoch_13.pth")

        with st.spinner("Loading model..."):
            model = load_model(model_path, device, select_model)
        st.sidebar.success("Model loaded successfully!")

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

                    cam_np = gradcam.generate_cam(input_tensor, class_idx=pred_idx)
                    highlighted_img = highlight_top_n_regions(cam_np, image, N=25, mode="box")
                    st.image(highlighted_img, caption="Top Attention Zones", use_container_width=True)

                    # Lesion details
                    lesion_text = lesion_details.get(str(pred_idx), "No details available.")
                    parts = lesion_text.split('\n', 1)
                    lesion_name = parts[0].strip()
                    lesion_description = parts[1].strip() if len(parts) > 1 else ""

                    # Decide CSS class based on pred_idx
                    if str(pred_idx) in ['1', '5']:
                        css_class = "diagnosis-cancer"
                    elif str(pred_idx) == '0':
                        css_class = "diagnosis-precancerous"
                    else:
                        css_class = "diagnosis-non-cancer"

                    # Display styled lesion header
                    lesion_html = f"""<div class="{css_class}" style="text-align:center;">{lesion_name}</div>"""
                    st.sidebar.markdown(lesion_html, unsafe_allow_html=True)
                    st.sidebar.markdown("<br>", unsafe_allow_html=True)
                    st.sidebar.info(lesion_description)

                st.success(f"Prediction: {class_name} (Confidence: {confidence * 100:.2f}%)")

        st.write("---")

if __name__ == "__main__":
    main()
