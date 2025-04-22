## Skin Lesion Classification – Final Project (ML 2025)

This repository contains the complete implementation of our Machine Learning Final Project, focused on classifying skin lesions using a multimodal deep learning approach combining CNNs with Attention Mechanisms.

🧪 **Project Overview**

Our initial idea was to build a multimodal model that accepts both:
	•	Image data of skin lesions, and
	•	Structured metadata such as age, sex, localization, and acquisition method

However, after conducting several experiments, we discovered that the metadata alone had low predictive power and offered minimal independent information. Due to this, and with limited hardware resources, we decided to pivot and focus exclusively on image-based classification.

🧩 **Architecture**

We built a ResNet-18 inspired CNN from scratch and experimented with different placements of attention mechanisms:
	•	We tested applying attention (CBAM + Multi-Head Self-Attention) at various layers.
	•	After multiple ablation studies, we found the best trade-off by applying attention at Layer 3 and Layer 4.
	•	Applying attention at earlier stages like Layer 1 was computationally expensive and not feasible on our hardware.

✅ **Final Architecture**
	•	CNN backbone (custom ResNet)
	•	CBAM (Convolutional Block Attention Module)
	•	Multi-Head Attention for capturing long-range dependencies
	•	Applied in Layer 3 & 4

🌐 **Web Application (Streamlit)**

Our solution is deployed as an interactive web application using Streamlit. It features:
	•	Descriptive Analytics tab with interactive data visualization (using Plotly, Seaborn)
	•	Predictive Analytics tab where users can:
	•	Upload a skin lesion image
	•	Receive real-time predictions from our attention-enhanced CNN
	•	View Grad-CAM visualizations showing where the model focused during prediction

🔗 Try the app here: https://ait-ml.streamlit.app

<pre>
Final Project Directory
📦 FinalProject/
│
├── Project_File/
│   ├── Model/                # CNN and Attention model definitions
│   ├── config/               # GradCAM, utilities, etc.
│   └── model_checkpoint/     # Trained model checkpoints
│
├── data/                     # CSV files and metadata
├── Front_End/app.py          # Main Streamlit application
├── requirements.txt
├── Dockerfile (optional)
└── README.md
</pre>


💡 **Features**
	•	Custom attention-based CNN model
	•	Grad-CAM and heatmap support
	•	Interactive charts and data exploration
	•	Clean UI with light/dark backgrounds
	•	Clear modular codebase
