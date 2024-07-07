import streamlit as st
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from ultralytics import YOLOv10

# Đường dẫn tới mô hình YOLOv10
MODEL_PATH = 'D:\\VS_code\\Yolov10_Helmet_detect\\yolov10n.pt'


def load_model(model_path):
    """Tải mô hình YOLOv10"""
    return YOLOv10(model_path)


def read_image(uploaded_file):
    """Đọc file ảnh và chuyển đổi thành ảnh RGB"""
    image = np.array(Image.open(uploaded_file))
    return image


def detect_helmets(model, image):
    """Thực hiện phát hiện mũ bảo hiểm trên ảnh"""
    # Lưu ảnh tạm thời để phát hiện
    temp_img_path = 'temp.jpg'
    cv2.imwrite(temp_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Thực hiện phát hiện
    result = model(source=temp_img_path)[0]
    return result


def plot_annotated_image(result):
    """Chú thích và hiển thị ảnh đã phát hiện"""
    annotated_image = result.plot()

    # Chuyển đổi ảnh chú thích về RGB
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image_rgb


def extract_detections(result):
    """Trích xuất thông tin về các đối tượng được phát hiện"""
    detections = {}
    for box in result.boxes:
        cls = box.cls.item()
        class_name = result.names[int(cls)]
        detections[class_name] = detections.get(class_name, 0) + 1
    return detections


def main():
    # Tải mô hình
    model = load_model(MODEL_PATH)

    # Tiêu đề ứng dụng Streamlit
    st.title("Object Detection using YOLOv10")
    st.write("Upload an image to detect helmets")

    # Tải lên file ảnh
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc ảnh
        image = read_image(uploaded_file)

        # Thực hiện phát hiện mũ bảo hiểm
        result = detect_helmets(model, image)

        # Chú thích ảnh
        annotated_image = plot_annotated_image(result)

        # Trích xuất thông tin về các đối tượng được phát hiện
        detections = extract_detections(result)
        detection_text = ', '.join(
            [f"{count} {cls}" for cls, count in detections.items()])

        # Hiển thị ảnh đã chú thích
        st.image(annotated_image, caption='Detected Image',
                 use_column_width=True)

        # Hiển thị thông tin các đối tượng được phát hiện
        st.write(f"Objects detected: {detection_text}")


if __name__ == "__main__":
    main()
