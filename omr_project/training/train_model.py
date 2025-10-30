import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.models import Model
import os

# --- 1. CẤU HÌNH ---
# Lấy đường dẫn của thư mục 'training/' hiện tại
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Đường dẫn đến dữ liệu đã xử lý 
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, 'processed_data')
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VALID_DIR = os.path.join(PROCESSED_DATA_DIR, 'valid')

# Đường dẫn để lưu model sau khi huấn luyện
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'saved_model', 'omr_bubble_model.h5')

# Thông số hình ảnh 
IMG_HEIGHT = 20
IMG_WIDTH = 20
IMG_CHANNELS = 1 # 1 cho ảnh xám (grayscale)

# Thông số huấn luyện
BATCH_SIZE = 64
EPOCHS = 15 

# --- 2. HÀM TẢI DỮ LIỆU ---
def load_data_from_folders(train_dir, valid_dir):
    """
    Sử dụng tiện ích của Keras để tải dữ liệu từ các thư mục
    đã được sắp xếp (processed_data/train/0, processed_data/train/1, ...)
    """
    print(f"Đang tải dữ liệu huấn luyện từ: {train_dir}")
    print(f"Đang tải dữ liệu kiểm thử từ: {valid_dir}")
    
    # Keras tự động gán nhãn '0' cho thư mục '0' và '1' cho thư mục '1'
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale', 
        class_names=['0', '1'] 
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_names=['0', '1']
    )
    
    # Tối ưu hóa hiệu suất tải dữ liệu
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset

# --- 3. HÀM XÂY DỰNG MODEL (CNN) ---
def build_model(input_shape):
    """Xây dựng một mô hình CNN đơn giản."""
    
    inputs = Input(shape=input_shape)
    
    # Chuẩn hóa giá trị pixel từ [0, 255] về [0, 1]
    x = Rescaling(1./255)(inputs)
    
    # Lớp Conv 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Lớp Conv 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Lớp Conv 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Làm phẳng (Flatten)
    x = Flatten()(x)
    
    # Lớp Fully Connected
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout để chống overfitting 
    
    # Lớp Output
    # Dùng 'sigmoid' vì đây là bài toán phân loại nhị phân (0 hoặc 1)
    outputs = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Biên dịch model
    # Dùng 'binary_crossentropy' vì đây là bài toán phân loại nhị phân
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']) # Theo dõi độ chính xác
    
    return model

# --- 4. HÀM CHẠY CHÍNH ---
def main():
    # Bước 1: Tải dữ liệu
    try:
        train_dataset, val_dataset = load_data_from_folders(TRAIN_DIR, VALID_DIR)
    except Exception as e:
        print(f"\n--- LỖI TẢI DỮ LIỆU ---")
        print(f"Lỗi: {e}")
        print("\nVui lòng kiểm tra lại các đường dẫn:")
        print(f"TRAIN_DIR: {TRAIN_DIR}")
        print(f"VALID_DIR: {VALID_DIR}")
        print("Hãy chắc chắn rằng bên trong 2 thư mục này có 2 thư mục con '0' và '1'.")
        return

    print("--- Đã tải dữ liệu thành công ---")
    
    # Bước 2: Xây dựng model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = build_model(input_shape)
    
    print("\n--- Cấu trúc Model ---")
    model.summary() # In ra cấu trúc của model
    
    # Bước 3: Huấn luyện model
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset
    )
    
    # Bước 4: Lưu model
    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")
    
    # Tạo thư mục 'data/saved_model' nếu chưa có
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model đã được lưu tại: {MODEL_SAVE_PATH}")
    print("Bạn đã sẵn sàng cho bước tiếp theo: xây dựng file 'grader.py'!")

# --- Chạy script ---
if __name__ == "__main__":
    main()