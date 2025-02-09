import streamlit as st
from streamlit_option_menu import option_menu
import os
from datetime import datetime
import base64
import json 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, GlobalAveragePooling1D
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.saving import register_keras_serializable
from paddleocr import PaddleOCR
import easyocr
import time
import logging

logging.basicConfig(level=logging.ERROR)
#-----------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Sarcasm Detection on Vietnamese Social Media Texts",
    page_icon=":material/group:"
)
st.markdown(
    f"""
    <style>
    /* Remove default header and manage app button */
    .stApp [data-testid="stHeader"]{{
        display:none;
    }}
    .stApp [data-testid="manage-app-button"]{{
        display:none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="column"]:first-child {
        position: fixed;
        top: 0;
        left: 0;
        width: 33%;
        height: 100%;
        overflow: hidden;
    }
    [data-testid="column"]:last-child {
        margin-left: 33%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for styling
st.markdown(""" 
    <style>
    .css-1y4p8pa {
        padding-top: 0;
        padding-bottom: 0;
    }
    .banner-container {
        position: relative;
        width: 100%;
    }
    .banner-image {
        width: 100%;
        filter: brightness(0.7);  /* Make background darker */
    }
    .group-name {
        position: absolute;
        bottom: 20px;
        left: 20px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        z-index: 2;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);  /* Add shadow for better readability */
    }
    /* Custom menu styling */
    .nav-link {
        background-color: #ffffff;
        color: #333333 !important;
    }
    .nav-link:hover {
        background-color: #e6e6e6 !important;
    }
    .nav-link.active {
        background-color: #AAAAAA !important;
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Custom title with banner image and group name
st.markdown(""" 
    <div class="banner-container">
        <img class="banner-image" src="https://th.bing.com/th/id/OIP.H7M0FNk53OGh1LAVDCCwcQHaCz?rs=1&pid=ImgDetMain">
        <div class="group-name">Nhóm 5 - Tư duy Trí tuệ nhân tạo - AI002.P11</div>
    </div>
""", unsafe_allow_html=True)

# Menu options with custom styling
page = option_menu(
    menu_title="",
    options=["Main Posts", "Review Posts"],
    icons=["clipboard", "check-circle"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#e6e6e6",
        },
        "nav-link-selected": {
            "background-color": "#AAAAAA",
        },
    }
)
#-----------------------------------------------------------------------------------------------------
class CombinedSarcasmClassifier:
    def __init__(self):
        self.model = None
        self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.jina_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        self.jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", 
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=True)
        self.reader = easyocr.Reader(['en', 'vi'])
        # Define label mapping
         # Define label mapping
        self.label_mapping = {
            'not-sarcasm': 0,
            'image-sarcasm': 1,
            'text-sarcasm': 2,
            'multi-sarcasm': 3
        }
        
        # Ensure models are in float32
        self.vit_model.to(self.device).to(torch.float32)
        self.jina_model.to(self.device).to(torch.float32)

    def encode_labels(self, labels):
        """Convert text labels to one-hot encoded format"""
        numerical_labels = [self.label_mapping[label] for label in labels]
        return tf.keras.utils.to_categorical(numerical_labels, num_classes=4)

    def decode_labels(self, one_hot_labels):
        numerical_labels = np.argmax(one_hot_labels, axis=1)
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return [reverse_mapping[idx] for idx in numerical_labels]

    def build(self, image_dim=2024, text_dim=768):
        image_input = Input(shape=(image_dim,), name='image_input')
        text_input = Input(shape=(text_dim,), name='text_input')
    
        # Image processing branch
        image_dense1 = Dense(1024, activation='relu')(image_input)
        # image_dropout1 = Dropout(0.1)(image_dense1)
        image_dense2 = Dense(512, activation='relu')(image_dense1)
        # image_dropout2 = Dropout(0.1)(image_dense2)
    
        # Text processing branch
        text_dense1 = Dense(512, activation='relu')(text_input)
        # text_dropout1 = Dropout(0.1)(text_dense1)
        text_dense2 = Dense(256, activation='relu')(text_dense1)
        # text_dropout2 = Dropout(0.1)(text_dense2)
    
        # Combine both branches
        combined = concatenate([image_dense2, text_dense2])
        dense_combined1 = Dense(768, activation='relu')(combined)
        # dropout_combined1 = Dropout(0.1)(dense_combined1)
        dense_combined2 = Dense(384, activation='relu')(dense_combined1)
        # dropout_combined2 = Dropout(0.1)(dense_combined2)
    
        # Output layer
        output = Dense(4, activation='softmax', name='output')(dense_combined2)
    
        # Create the model
        self.model = Model(inputs=[image_input, text_input], outputs=output, name='multimodal_classifier')


    def preprocess_data(self, images, texts, is_test=0):

        combined_features = []
        
        print("\nProcessing images and texts:")
        if is_test == 1:
            temp = cv2.imread(images)
            inputs = self.vit_processor(images=temp, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
            image_features = outputs.logits.cpu().numpy().squeeze()
            text_feats = None
            try:
                ocr_results = self.paddle_ocr.ocr(images, cls=True)
                recognized_texts = []
                boxes = []
        
                image = Image.open(images)
        
                for line in ocr_results[0]:
                    if len(line) == 2:
                        bbox, text = line
                        confidence = None
                    elif len(line) == 3:
                        bbox, text, confidence = line
                    else:
                        continue
        
                padding_ratio_y = 0.25
                padding_ratio_x = 0.015
                width = bbox[2][0] - bbox[0][0]
                height = bbox[2][1] - bbox[0][1]
                
                padding_width = int(width * padding_ratio_x)
                padding_height = int(height * padding_ratio_y)
                
                x_min = max(0, bbox[0][0] - padding_width)
                y_min = max(0, bbox[0][1] - padding_height)
                x_max = bbox[2][0] + padding_width
                y_max = bbox[2][1] + padding_height
    
                boxes.append([x_min, y_min, x_max, y_max])
        
                merged_boxes = self.merge_boxes(boxes)
        
                recognized_texts = [] 
        
                for idx, box in enumerate(merged_boxes):
                    x_min, y_min, x_max, y_max = map(int, box)
        
                    cropped_region = image.crop((x_min, y_min, x_max, y_max))
                    cropped_region_np = np.array(cropped_region)

                    recognized_text = self.reader.readtext(cropped_region_np)
                    recognized_texts.append(' '.join([text for (_, text, _) in recognized_text]))
        
                combined_text = "\n".join(recognized_texts)
                text_inputs = self.jina_tokenizer(
                    combined_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    text_outputs = self.jina_model(**text_inputs)
                text_feats = text_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            except Exception as e:
                text_feats = np.zeros(1024)
            # Concatenate image and text features
            combined_feature = np.concatenate([image_features, text_feats])
            combined_features.append(combined_feature)
        text_features = []
        print("\nProcessing texts:")
        if is_test == 1:
            inputs = self.jina_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                        outputs = self.jina_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            text_features.append(features)

        return np.array(combined_features), np.array(text_features)
    @staticmethod
    @register_keras_serializable(package="Custom", name="f1_macro")
    def f1_macro(y_true, y_pred):
        """Custom F1 macro metric for Keras"""
        y_true_class = tf.argmax(y_true, axis=1)
        y_pred_class = tf.argmax(y_pred, axis=1)
        
        f1_scores = []
        for i in range(4):  # 4 classes
            true_positives = tf.reduce_sum(tf.cast(
                tf.logical_and(tf.equal(y_true_class, i), tf.equal(y_pred_class, i)),
                tf.float32
            ))
            false_positives = tf.reduce_sum(tf.cast(
                tf.logical_and(tf.not_equal(y_true_class, i), tf.equal(y_pred_class, i)),
                tf.float32
            ))
            false_negatives = tf.reduce_sum(tf.cast(
                tf.logical_and(tf.equal(y_true_class, i), tf.not_equal(y_pred_class, i)),
                tf.float32
            ))
            
            precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
            recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
            f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        
        return tf.reduce_mean(f1_scores)


    def learning_rate_schedule(self, epoch, lr):
        """Learning rate scheduler function."""
        if 5 <= epoch :
            return lr * 0.5
        elif 20 <= epoch < 21:
            return lr * 0.01
        elif 21 <= epoch :
            return lr * 0.005
        return lr

    def train(self, x_train_images, x_train_texts, y_train):
        print("Starting preprocessing...")
        image_features, text_features = self.preprocess_data(x_train_images, x_train_texts)

        print(f"Image feature shape: {image_features.shape}")
        print(f"Text feature shape: {text_features.shape}")
        
        # Convert labels to numerical format for stratification
        numerical_labels = [self.label_mapping[label] for label in y_train]
                
        # Calculate class weights for the resampled data
        # class_weights = compute_class_weight(
        #     'balanced',
        #     classes=np.unique(numerical_labels),
        #     y=numerical_labels
        # )
        # class_weight_dict = dict(enumerate(class_weights))
        # print("Class weights:", class_weight_dict)
        
        # Perform stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, val_idx = next(sss.split(image_features, numerical_labels))
        
        # Split the data
        train_image_features = image_features[train_idx]
        train_text_features = text_features[train_idx]
        val_image_features = image_features[val_idx]
        val_text_features = text_features[val_idx]
        
        # Encode labels after splitting
        y_train_encoded = self.encode_labels([y_train[i] for i in train_idx])
        y_val_encoded = self.encode_labels([y_train[i] for i in val_idx])

        initial_lr = 7.5e-5

        print("\nCompiling model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(), CombinedSarcasmClassifier.f1_macro]
        )
        
        class BatchProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                print(f"\nEpoch {epoch + 1} starting...")
            
            def on_batch_begin(self, batch, logs=None):
                print(f"Training batch {batch + 1}", end='\r')

        lr_scheduler = LearningRateScheduler(self.learning_rate_schedule)

        print("\nStarting training...")
        history = self.model.fit(
            [train_image_features, train_text_features],
            y_train_encoded,
            epochs=25,
            batch_size=256,
            validation_data=([val_image_features, val_text_features], y_val_encoded),
            callbacks=[BatchProgressCallback(), lr_scheduler]
            # class_weight=class_weight_dict
        )
        
        print("\nTraining completed!")
        return history

    def predict(self, x_test_images, x_test_texts):
        print("Preprocessing test data...")
        image_features, text_features = self.preprocess_data(x_test_images, x_test_texts, 1)
        print("Making predictions...")
        predictions = self.model.predict([image_features, text_features])
        return self.decode_labels(predictions)

    def load(self, model_file):
        self.model = load_model(model_file, custom_objects={'f1_macro': self.f1_macro})

    def save(self, model_file):
        self.model.save(model_file)

    def summary(self):
        self.model.summary()
#-----------------------------------------------------------------------------------------------------
@st.cache_resource
def load_combined_sarcasm_classifier():
    classifier = CombinedSarcasmClassifier()
    classifier.build()
    classifier.load('model.keras')
    return classifier

classifier = load_combined_sarcasm_classifier()

def load_predictions():
    try:
        with open('predictions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_prediction(image_path, text, prediction):
    predictions = load_predictions()
    predictions[f"{image_path}_{text}"] = prediction
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False)

def get_cached_prediction(image_path, text):
    predictions = load_predictions()
    return predictions.get(f"{image_path}_{text}")
def load_posts(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            return json.loads(content) if content else []
    except (FileNotFoundError, json.JSONDecodeError):
        # Create the file if it doesn't exist
        save_posts([], filename)
        return []

# Initialize session state variables if not already present


# Initialize session state variables
if 'pending_posts' not in st.session_state:
    st.session_state.pending_posts = []
if 'approved_posts' not in st.session_state:
    st.session_state.approved_posts = []

# Load existing posts
st.session_state.pending_posts = load_posts('pending_posts.json')
st.session_state.approved_posts = load_posts('approved_posts.json')

def save_posts(posts, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False)

def add_post(post):
    st.session_state.pending_posts.append(post)
    save_posts(st.session_state.pending_posts, 'pending_posts.json')

# Modify approve_post function:
def approve_post(index):
    post = st.session_state.pending_posts.pop(index)
    st.session_state.approved_posts.append(post)
    save_posts(st.session_state.pending_posts, 'pending_posts.json')
    save_posts(st.session_state.approved_posts, 'approved_posts.json')

# Modify decline_post function:
def decline_post(index):
    st.session_state.pending_posts.pop(index)
    save_posts(st.session_state.pending_posts, 'pending_posts.json')


def format_timestamp(timestamp):
    # Định dạng timestamp từ datetime string sang "Giờ:Phút, Ngày/Tháng/Năm"
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')  # Parse string to datetime
    return dt.strftime('%H:%M, %d/%m/%Y')  # Format as Hour:Minute, Day/Month/Year


def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"File not found: {image_path}")
        return None


def show_post(post, index=None, prediction=None):
    # Handle image source
    if post['image'].startswith('http'):  # Online URL
        img_src = post['image']
    else:  # Local file path
        encoded_image = encode_image(post['image'])
        img_src = f"data:image/png;base64,{encoded_image}"
    print(prediction)
    # Xác định màu và nhãn cho dự đoán
    if prediction == ['not-sarcasm']:
        prediction_label = '<span style="color: green; font-weight: bold;">Not Sarcasm</span>'
    elif prediction == ['image-sarcasm']:
        prediction_label = '<span style="color: red; font-weight: bold;">Image Sarcasm</span>'
    elif prediction == ['text-sarcasm']:
        prediction_label = '<span style="color: red; font-weight: bold;">Text Sarcasm</span>'
    elif prediction == ['multi-sarcasm']:
        prediction_label = '<span style="color: red; font-weight: bold;">Multi Sarcasm</span>'
    else:
        prediction_label = ''  # Không hiển thị nếu prediction là None
    post['text'] = post['text'].replace('\n', '<br>')
    # Container for the post layout
    with st.container():
        # Styled HTML post
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff; 
                border: 1px solid #d3d3d3; 
                border-radius: 15px; 
                padding: 20px; 
                margin-bottom: 20px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">

            <!-- Timestamp và Prediction trên cùng một hàng -->
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 15px; color: gray;">Posted at {format_timestamp(post['timestamp'])}</span>
                {prediction_label}
            </div>
            
            <!-- Caption -->
            <div style="margin-bottom: 15px;">
                <p style="font-size: 16px; margin: 0;">{post['text']}</p>
            </div>
            
            <!-- Image -->
            <div style="text-align: center;">
                <img src="{img_src}" style="max-width: 100%; border-radius: 10px;">
            </div>
            
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Buttons in container
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✔", key=f"approve_{index}", help="Approve post"):
                approve_post(index)
                st.rerun()
        with col2:
            if st.button("✖", key=f"decline_{index}", help="Decline post"):
                decline_post(index)
                st.rerun()
def count_words(input_string):
    cleaned_string = input_string.replace('\n', ' ')
    words = cleaned_string.split()
    word_count = len(words)
    return word_count

def display_post(post):
    # Handle image source
    if post['image'].startswith('http'):  # Online URL
        img_src = post['image']
    else:  # Local file path
        encoded_image = encode_image(post['image'])
        img_src = f"data:image/png;base64,{encoded_image}"

    # Container for the post layout
    with st.container():
        # Styled HTML post
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff; 
                border: 1px solid #d3d3d3; 
                border-radius: 15px; 
                padding: 20px; 
                margin-bottom: 20px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">

            <!-- Timestamp -->
            <div style="display: flex; justify-content: flex-end; margin-bottom: 5px;">
                <span style="font-size: 15px; color: gray;">Posted at {format_timestamp(post['timestamp'])}</span>
            </div>
            
            <!-- Caption -->
            <div style="margin-bottom: 15px;">
                <p style="font-size: 16px; margin: 0;">{post['text']}</p>
            </div>
            
            <!-- Image -->
            <div style="text-align: center;">
                <img src="{img_src}" style="max-width: 100%; border-radius: 10px;">
            </div>
            
            </div>
            """, 
            unsafe_allow_html=True
        )     
if page == 'Main Posts':
    text = st.text_area(label = "Post text", placeholder="Write something here...", label_visibility="hidden")
    if text and count_words(text) <= 256:
        image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if image:
            img = Image.open(image)
            height, width = img.size
            print(img.size)
            if height < 224 or width < 224:
                st.error("Please upload an image with dimensions less than or equal to 224x224.")
            else:
                if st.button("Post"):
                    if image and text:
                        # Save the uploaded image
                        image_path = os.path.join(os.getcwd() + '/uploads', image.name)
                        os.makedirs('uploads', exist_ok=True)
                        with open(image_path, "wb") as f:
                            f.write(image.getbuffer())

                        # Create post
                        post = {
                            "image": 'uploads/' + image.name,
                            "text": text,
                            "timestamp": str(datetime.now())
                        }
                        add_post(post)
                        st.success("Your post has been submitted for review!")
                    else:
                        st.error("Please upload an image and write text.")
    elif count_words(text) > 256:
        st.error("The text must be less than or equal to 256 words.")
    if (len(st.session_state.approved_posts) > 0):
        for post in st.session_state.approved_posts:
            display_post(post)       
 
elif page == 'Review Posts':
    if len(st.session_state.pending_posts) == 0:
        st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
            <h1>No pending posts.</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    else:
        # Display pending posts with approve buttons
        for i, post in enumerate(st.session_state.pending_posts):
            prediction = get_cached_prediction(post['image'], post['text'])
            if prediction is None:
                prediction = classifier.predict(post['image'], post['text'])
                save_prediction(post['image'], post['text'], prediction)
            show_post(post, index=i, prediction=prediction)
            st.markdown("---")
            

