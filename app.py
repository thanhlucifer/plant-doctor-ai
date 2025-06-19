
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from typing import List, Tuple

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Bác sĩ Cây trồng AI",
    page_icon="🌱",
    layout="centered"
)

# --- Tải Mô hình và Tên Lớp (Cache để tối ưu) ---
@st.cache_resource
def load_model_and_classes():
    """Tải mô hình và danh sách tên các lớp, chỉ chạy một lần."""
    try:
        model = tf.keras.models.load_model('artifacts/plant_doctor_model.h5')
        
        with open('artifacts/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        # Sắp xếp lại tên lớp theo đúng chỉ số (0, 1, 2, ...)
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
        
        return model, class_names
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tải mô hình hoặc file class: {e}")
        return None, None

model, class_names = load_model_and_classes()

def predict(image_data: Image.Image) -> Tuple[str, float]:
    img_resized = image_data.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_scaled = img_array / 255.0
    st.info("Đã chuẩn hóa ảnh đầu vào thành công.") 
    predictions = model.predict(img_array_scaled)[0]
    confidence = float(np.max(predictions)) * 100
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    
    return predicted_class, confidence


# --- Giao diện ứng dụng ---
st.title("Bác sĩ Cây trồng AI 🧑‍⚕️🌱")
st.write("Chào mừng đến với phiên bản cuối cùng! Hãy tải ảnh lên để xem kết quả chính xác.")

uploaded_file = st.file_uploader("Chọn một hình ảnh lá cây...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở ảnh bằng PIL
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Ảnh bạn đã tải lên.", use_column_width=True)
    
    if st.button("Chẩn đoán"):
        if model is not None and class_names is not None:
            with st.spinner('Bắt đầu chẩn đoán...'):
                predicted_class, confidence = predict(image)

                st.success("Đã có kết quả!")
                try:
                    plant_name, disease_name = predicted_class.split('___')
                    st.markdown(f"### **Cây trồng:** `{plant_name.replace('_', ' ')}`")
                    st.markdown(f"### **Chẩn đoán:** `{disease_name.replace('_', ' ')}`")
                    st.markdown(f"### **Độ tin cậy:** `{confidence:.2f}%`")
                    
                    if "healthy" in disease_name.lower():
                        st.balloons()
                except Exception:
                    st.error(f"Không thể phân tích kết quả: {predicted_class}")
        else:
            st.error("Mô hình chưa được tải, vui lòng kiểm tra lại lỗi ở terminal.")