import tensorflow as tf
import os

# Đường dẫn tương đối đến file model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_model', 'omr_bubble_model.h5')

def load_bubble_model():
    """
    Tải mô hình nhận diện bong bóng đã được huấn luyện.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"--- Đã tải model thành công từ: {MODEL_PATH} ---")
        return model
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể tải model tại: {MODEL_PATH}")
        print(f"Lỗi: {e}")
        print("Hãy chắc chắn rằng bạn đã huấn luyện và lưu model thành công.")
        return None

# Tải model 1 lần duy nhất khi script bắt đầu
bubble_model = load_bubble_model()