import streamlit as st
from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import cv2


model = load_model('3d_image_classification.h5')

def final_pre(folder):
    cut_cubes_L = []
    for i in range(1):
        img = []
        for j in range(8):
            im=np.array(Image.open(folder).convert('L'))
            img.append(im)
        img = np.array(img)
        cut_cubes_L.append(img)
    cut_cubes_L = np.array(cut_cubes_L)
    pe = cut_cubes_L.transpose(0,2,3,1)
    p = tf.data.Dataset.from_tensor_slices(pe)
    def train_preprocessing(volume):
        """Process training data by only adding a channel and not rotating."""
   
        volume = tf.expand_dims(volume, axis=3)
        return volume
    batch_size = 8
    train_dataset = (
        p.shuffle(len(pe))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
        )
    model = load_model(r'C:\Users\Owner\Desktop\final_project\3d_image_classification.h5')
    preds = model.predict(pe, batch_size=1)
    pred_labels = preds.argmax(axis=-1)
    return pred_labels
model = load_model(r'C:\Users\Owner\Desktop\final_project\3d_image_classification.h5')
def generate_grad_cam(model, img_array):
    last_conv_layer_name = 'conv3d_3'  # Change this to the name of your last convolutional layer

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_output[0]
    grads = tape.gradient(loss, conv_output)[0]

    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    return heatmap

# Load and preprocess your CT scan image
def preprocess_image(img):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize image to match model input size
    img_resized = cv2.resize(img_gray, (128, 128))
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_normalized, axis=0)
    # Add channel dimension
    img_array = np.expand_dims(img_array, axis=3)
    return img_array


import streamlit as st

def colorize_word(word):
    colored_word = ""
    colors = ["#008080", "#9F6BA0"]  # Gold for "dete" and Green for "CT"
    color_index = 0
    for char in word:
        
        if char.islower():  # Check if the character belongs to "dete"
            color_index = 0
        else:
            color_index = 1
        colored_word += f'<span style="color:{colors[color_index]}">{char}</span>'
    return colored_word

word = "deteCT"
colored_word = colorize_word(word)

# Define file variable outside of the tab selection block
file = None


# Create selectbox for tab selection
selected_tab = st.sidebar.selectbox("Select Tab", ("Home","Our Work","Detect"))

# Home section
if selected_tab == "Home":
    st.markdown(
    f'<h1 style="text-align: center;">{colored_word}</h1>',
    unsafe_allow_html=True
)
    st.markdown("<h2 style='text-align: center; color: #E82D05;'>Understanding the Importance and Functionality</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <p>Medical imaging plays a crucial role in modern healthcare, aiding clinicians in the diagnosis and treatment of various conditions. Among these imaging modalities, computed tomography (CT) scans are widely used due to their ability to provide detailed cross-sectional images of the body.</p>
        <p>However, the rise of digital manipulation techniques has led to concerns about the authenticity of CT scan images. Detecting fake CT scans is of paramount importance to ensure patient safety and the integrity of medical diagnoses.</p>
        <p>Here are some key aspects of detecting fake CT scans:</p>
        <ul>
            <li><strong>Image Analysis:</strong> Sophisticated algorithms are used to analyze CT scan images for signs of manipulation or inconsistencies.</li>
            <li><strong>Pattern Recognition:</strong> Machine learning models can identify patterns indicative of tampering or alteration in CT scan images.</li>
            <li><strong>Quality Assurance:</strong> Automated tools assist radiologists and medical professionals in verifying the authenticity and quality of CT scan images.</li>
        </ul>
        <p>By leveraging advanced image processing techniques and artificial intelligence, detecting fake CT scans helps maintain the trust and reliability of medical imaging, ultimately benefiting patient care and healthcare outcomes.</p>
    """, unsafe_allow_html=True)

# Image Upload section
elif selected_tab == "Detect":
    st.markdown(
    f'<h1 style="text-align: center;">{colored_word}</h1>',
    unsafe_allow_html=True
)
    st.markdown("<h2 style='text-align: center; color: #E82D05;'>🔍 Medical DeepFake Image Classifier</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #E82D05;'>Is it Real or Fake?</h2>", unsafe_allow_html=True)
    st.header('Image Upload and Classification')

    file = st.file_uploader('Please upload an image', type=['jpeg', 'jpg', 'png'])

    # Display uploaded image and perform classification
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, width=300, caption='Uploaded Image', use_column_width=True, output_format='JPEG', clamp=True, channels='RGB')
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        
        # Perform classification
        output = final_pre(file)
        classification_result = None
        if output == 0 or output == 1:
            st.error("🛑 Ouch! That looks like a fake. Don't believe everything you see. Even salt looks like sugar.")
        elif output == 2 or output == 3:
            st.success("✅ Congratulations! It seems authentic. Authenticity speaks louder than imitation.")
#     # Process uploaded image for Grad-CAM heatmap generation
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     img_array = np.expand_dims(img, axis=0)

# # Generate Grad-CAM heatmap
#     # Generate Grad-CAM heatmap
#     heatmap = generate_grad_cam(model, img_array)


# # Resize heatmap to match original image size
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

# # Overlay heatmap on the original image
#     overlay_img = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0)
#     st.image(overlay_img, width=300, caption='Grad-CAM Heatmap', use_column_width=True, output_format='JPEG', clamp=True, channels='RGB')

elif selected_tab == "Our Work":
    st.markdown(
    f'<h1 style="text-align: center;">{colored_word}</h1>',
    unsafe_allow_html=True
)
    st.markdown("<h2 style='text-align: center; color: #E82D05;'>Our Work</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #900C3F;'>Approach and Model</h2>", unsafe_allow_html=True)

    st.markdown("""
        <p>At our lab, we have developed a state-of-the-art approach for detecting fake CT scans using a 3D convolutional neural network (CNN) model. This approach allows us to accurately predict whether an image is fake or not by analyzing its three-dimensional structure.</p>
        <p>We have trained our 3D CNN model on a dataset consisting of 6624 images, meticulously curated to cover a wide range of scenarios and variations. Through rigorous training and validation, we have achieved an impressive accuracy of around 97%.</p>
        <p>The use of a 3D CNN model offers several advantages over traditional 2D CNNs or other machine learning models. By considering the spatial and temporal information present in the 3D CT scan volumes, our model can better capture the complex patterns and nuances indicative of tampering or manipulation.</p>
        <p>Furthermore, the 3D CNN architecture allows for more efficient feature extraction and learning, leading to improved performance and robustness in detecting fake CT scans. This makes our approach highly reliable and effective in real-world medical imaging scenarios.</p>
    """, unsafe_allow_html=True)
    st.title('Results Summary')
    results_data = {
        'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_score', 'Mean Sensitivity', 'Mean Specificity'],
        'Value': [0.9927536231884058, 0.9928301823119539, 0.9927536231884058, 0.992753550277708, 0.9927269200497285, 0.9975927750693677]
    }
    df_results = pd.DataFrame(results_data)
    st.table(df_results)
    st.title('Confusion Matrix')
    st.write('Confusion matrix for demonstration purposes:')
    
    confusion_data = {
        '': ['False Benign', 'False Malignant ','True Benign','True Malignant '],
        'False Benign': [215, 0, 1, 0],
        'False Malignant': [0, 205, 0, 0],
        'True Benign': [1, 0, 201, 0],
        'True Malignant': [0, 1, 3, 201]
    }
    df_confusion = pd.DataFrame(confusion_data)
    st.table(df_confusion)

