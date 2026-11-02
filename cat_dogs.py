import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# -------------------------------
# 1️⃣ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2️⃣ Load your trained model
# -------------------------------
model = models.mobilenet_v3_small(weights=None)  # Use your fine-tuned architecture
num_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_features, 2)  # 2 classes: cat, dog

# Path to saved model
model_path = r"your model path"

# Load weights
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# 3️⃣ Prediction function with confidence
# -------------------------------
def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)  # Convert logits to probabilities
        conf, predicted = torch.max(probs, 1)
    classes = ['cat', 'dog']  # ImageFolder ordering
    return classes[predicted.item()], conf.item()  # Return class and confidence

# -------------------------------
# 4️⃣ Streamlit UI
# -------------------------------
st.title("Cat vs Dog Classifier 🐱🐶")
st.write("Upload a photo and the model will predict if it's a cat or a dog, along with confidence.")
st.write("If the model is not confident enough, it will indicate that it does not know.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # 70%

if uploaded_file is not None:
    # Open uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the image
    st.image(image, caption='Uploaded Image',  width="content")
    
    # Make prediction
    prediction, confidence = predict_image(image)
    
    # Display result with confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown("### The model is not confident enough to classify this image.")
    else:
        st.markdown(f"### The model is **{confidence*100:.2f}% confident** that this image is a **{prediction}**.")
        st.markdown(f"### Prediction: **{prediction.upper()}**")
