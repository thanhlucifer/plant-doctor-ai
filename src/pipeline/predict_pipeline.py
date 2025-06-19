
import tensorflow as tf
import numpy as np
from PIL import Image
import os

class PredictionPipeline:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.class_names = self.load_class_names()

    def load_model(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return None

    def load_class_names(self):
        try:
            data_dir = os.path.join('data', 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)', 'train')
            class_names = sorted(os.listdir(data_dir))
            return class_names
        except FileNotFoundError:
            print("Không tìm thấy thư mục dữ liệu.")
            return []

    def predict(self, image_data):
        if not self.model or not self.class_names:
            return "Lỗi: Mô hình hoặc lớp bệnh chưa được tải.", 0.0

        # Tiền xử lý ảnh
        img = Image.open(image_data).resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Dự đoán
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class_name = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        return predicted_class_name, confidence