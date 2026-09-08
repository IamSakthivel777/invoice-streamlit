"""
Streamlit app: YOLO invoice field detection + visualization.
Deploy for free on Streamlit Community Cloud (share.streamlit.io).
"""

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

MODEL_PATH = "best.pt"  # place your trained weights in the repo root


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def draw_boxes(img: Image.Image, detections: list) -> Image.Image:
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        label = f'{det["class_name"]} ({det["confidence"]:.2f})'
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill="red")
    return annotated


def predict(model, image: Image.Image):
    results = model(image)
    result = results[0]

    detections = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]

            detections.append({
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": [round(v, 2) for v in [x1, y1, x2, y2]],
            })

    return draw_boxes(image, detections)


st.set_page_config(page_title="Invoice Field Detection", layout="centered")
st.title("Invoice Field Detection")
st.write("Upload an invoice image to detect fields with a custom-trained YOLO model.")

model = load_model()

uploaded_file = st.file_uploader("Upload invoice image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running detection..."):
        annotated = predict(model, image)

    st.image(annotated, caption="Detections", use_container_width=True)
