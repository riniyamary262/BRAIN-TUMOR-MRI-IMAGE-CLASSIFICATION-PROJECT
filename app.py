# ============================================================
# STREAMLIT APPLICATION FOR BRAIN TUMOR CLASSIFICATION
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import cv2

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #27ae60;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and class indices
@st.cache_resource
def load_model():
    """Load the trained model"""
    model = tf.keras.models.load_model('best_brain_tumor_model.h5')
    return model

@st.cache_resource
def load_class_indices():
    """Load class indices"""
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Invert dictionary
    return {v: k for k, v in class_indices.items()}

# Tumor type descriptions
TUMOR_INFO = {
    'glioma': {
        'description': 'Gliomas are tumors that arise from glial cells in the brain or spine.',
        'risk_level': 'High',
        'recommendation': 'Immediate consultation with neuro-oncologist recommended.'
    },
    'meningioma': {
        'description': 'Meningiomas are tumors that arise from the meninges, the protective layers around the brain.',
        'risk_level': 'Medium',
        'recommendation': 'Regular monitoring and specialist consultation advised.'
    },
    'pituitary': {
        'description': 'Pituitary tumors develop in the pituitary gland at the base of the brain.',
        'risk_level': 'Medium',
        'recommendation': 'Endocrinology consultation and hormone level assessment recommended.'
    },
    'notumor': {
        'description': 'No tumor detected in the MRI scan.',
        'risk_level': 'Low',
        'recommendation': 'Continue with regular check-ups as advised by your physician.'
    }
}

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return 'confidence-high'
    elif confidence >= 0.5:
        return 'confidence-medium'
    else:
        return 'confidence-low'

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Brain Tumor Detection and Classification</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses deep learning to classify brain tumors from MRI images. "
        "Upload an MRI scan to get instant classification results."
    )
    
    st.sidebar.header("Classes")
    st.sidebar.markdown("""
    - **Glioma**: Tumor from glial cells
    - **Meningioma**: Tumor from meninges
    - **Pituitary**: Tumor in pituitary gland
    - **No Tumor**: Healthy brain scan
    """)
    
    st.sidebar.header("Disclaimer")
    st.sidebar.warning(
        "This tool is for educational and research purposes only. "
        "Always consult qualified medical professionals for diagnosis."
    )
    
    # Load model
    try:
        model = load_model()
        class_names = load_class_indices()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file 'best_brain_tumor_model.h5' exists.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear brain MRI image for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    with col2:
        st.header("🔬 Classification Results")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess
                processed_image = preprocess_image(image)
                
                # Predict
                predictions = model.predict(processed_image, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx]
                
                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction: {predicted_class.upper()}</h2>
                    <p>Confidence: <span class="{get_confidence_class(confidence)}">
                    {confidence*100:.2f}%</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tumor information
                if predicted_class.lower() in TUMOR_INFO:
                    info = TUMOR_INFO[predicted_class.lower()]
                    st.subheader("📋 Details")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Risk Level:** {info['risk_level']}")
                    st.write(f"**Recommendation:** {info['recommendation']}")
                
                # Probability distribution
                st.subheader("📊 Probability Distribution")
                
                prob_df = {
                    'Tumor Type': list(class_names.values()),
                    'Probability': [f"{p*100:.2f}%" for p in predictions[0]]
                }
                
                st.dataframe(prob_df, use_container_width=True)
                
                # Bar chart
                import plotly.express as px
                fig = px.bar(
                    x=list(class_names.values()),
                    y=predictions[0],
                    labels={'x': 'Tumor Type', 'y': 'Probability'},
                    title='Classification Probabilities',
                    color=predictions[0],
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Warning for low confidence
                if confidence < 0.5:
                    st.warning(
                        "⚠️ Low confidence prediction. "
                        "Please consult a medical professional for accurate diagnosis."
                    )
        else:
            st.info("👆 Upload an MRI image to get classification results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #7f8c8d;'>"
        "Brain Tumor MRI Classifier | Deep Learning Project"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
