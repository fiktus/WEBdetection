import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import back

@st.cache_resource
def load_models_cached():
    back.load_models()
load_models_cached()

def draw_bbox(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(20)
    for det in detections:
        box = det["box"]
        cls = det["class"]
        conf = det["conf"]
        draw.rectangle(box, outline="red", width=2)
        label = f'class:{cls}: {conf}'
        text_pos = (box[0], max(0, box[1]-25))
        draw.text(text_pos, label, fill="white", font = font)
    return image

with st.sidebar:
    st.title("Настройки")
    model_choice = st.radio(
        "Модель",
        ("fast", "accurate"),
        format_func=lambda x: "Быстрая (YOLOv26s)" if x == "fast" else "Точная (YOLOv26m)")
    conf = st.slider("Порог уверенности", 0.0, 1.0, 0.5, 0.05)
    max_bbox = st.slider("Максимум объектов", 1, 200, 100)

uploaded_file = st.file_uploader("Выберите изображение")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Исходное изображение", use_container_width=True)
    if st.button("Детектировать"):
        with st.spinner("Обработка"):
            img_bytes = uploaded_file.getvalue()
            detections = back.predict(img_bytes, model_choice, conf, max_bbox)

            st.success(f"Модель: {model_choice} | Найдено объектов: {len(detections)}")

            if detections:
                result_image = draw_bbox(image.copy(), detections)
                st.image(result_image, caption="Результат детекции", use_container_width=True)
            else:
                st.info("Объекты не найдены.")