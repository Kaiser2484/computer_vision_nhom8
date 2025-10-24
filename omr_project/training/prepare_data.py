import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Giả sử cấu trúc là: training/Dataset/train/train_001.jpg
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, 'processed_data')
LABEL_FILES = {
    'train': os.path.join(SCRIPT_DIR, 'train_labels.csv'),
    'valid': os.path.join(SCRIPT_DIR, 'valid_labels.csv')
}
# -----------------------------------------------------

# --- 2. CẤU HÌNH LAYOUT (CỦA BẠN) ---
WARPED_IMAGE_WIDTH = 793 
WARPED_IMAGE_HEIGHT = 1122 
START_X_COL_1 = 138
START_X_COL_2 = 290
START_X_COL_3 = 444
START_X_COL_4 = 598
START_Y_ROWS = 466
OPTION_X_SPACING = 20
QUESTION_Y_SPACING = 38
BUBBLE_W = 20 
BUBBLE_H = 20

NUM_QUESTIONS = 60
NUM_OPTIONS = 4 
QUESTIONS_PER_COLUMN = 15 
OPTIONS_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
# -----------------------------------------------------


def get_bubble_coordinates(q_num, opt_idx):
    """Tính tọa độ (x, y) của 1 ô bong bóng"""
    col_idx = q_num // QUESTIONS_PER_COLUMN
    row_idx = q_num % QUESTIONS_PER_COLUMN
    
    if col_idx == 0: start_x = START_X_COL_1
    elif col_idx == 1: start_x = START_X_COL_2
    elif col_idx == 2: start_x = START_X_COL_3
    elif col_idx == 3: start_x = START_X_COL_4
    else:
        print(f"Lỗi: Chỉ số cột không hợp lệ: {col_idx} cho câu {q_num}")
        return 0, 0 
        
    x = start_x + (opt_idx * OPTION_X_SPACING)
    y = START_Y_ROWS + (row_idx * QUESTION_Y_SPACING)
    
    return int(x), int(y)

def process_data(data_type, label_file_path):
    """
    Hàm chính: Đọc CSV, resize ảnh (KHÔNG NẮN), cắt và lưu.
    """
    print(f"\n--- Bắt đầu xử lý bộ: {data_type} ---")
    
    output_dir_0 = os.path.join(PROCESSED_DATA_DIR, data_type, '0')
    output_dir_1 = os.path.join(PROCESSED_DATA_DIR, data_type, '1')
    os.makedirs(output_dir_0, exist_ok=True)
    os.makedirs(output_dir_1, exist_ok=True)
    
    try:
        with open(label_file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {label_file_path}. Bỏ qua...")
        return
    
    data_rows = []
    line_number = 1
    for line in lines[1:]: 
        line_number += 1
        line = line.strip()
        if not line:
            continue
        try:
            filename, answers_str = line.split(';', 1) 
            data_rows.append({'filename': filename.strip(), 'answers_string': answers_str.strip(), 'line': line_number})
        except ValueError:
            print(f"Cảnh báo (Dòng {line_number}): Dòng lỗi định dạng (không có dấu chấm phẩy?). Bỏ qua. Dòng: {line}")
            continue
            
    for index, row in enumerate(tqdm(data_rows, desc=f"Xử lý {data_type}")):
        filename = row['filename']
        answers_str = row['answers_string']
        line_num = row['line']
        
        img_path = os.path.join(RAW_DATA_DIR, data_type, filename)
        if not os.path.exists(img_path):
            print(f"\nCảnh báo (Dòng {line_num}): Không tìm thấy file {img_path}. Bỏ qua.")
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            print(f"\nCảnh báo (Dòng {line_num}): Không thể đọc file {img_path}. Bỏ qua.")
            continue
            
        # Resize ảnh về đúng kích thước bạn đã đo tọa độ
        try:
            resized_img = cv2.resize(image, (WARPED_IMAGE_WIDTH, WARPED_IMAGE_HEIGHT))
            warped_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"\nLỗi khi resize ảnh {filename}: {e}. Bỏ qua.")
            continue
            
        # 5. Lấy chuỗi đáp án
        try:
            student_answers = answers_str.split(',')
            if len(student_answers) != NUM_QUESTIONS:
                 print(f"\nCảnh báo (Dòng {line_num}): Ảnh {filename} có {len(student_answers)} đáp án, mong đợi {NUM_QUESTIONS}. Bỏ qua.")
                 continue
        except Exception as e:
            print(f"Lỗi xử lý đáp án cho {filename}: {e}. Bỏ qua.")
            continue

        # 6. Lặp qua từng câu hỏi
        for q_num in range(NUM_QUESTIONS):
            correct_answers_str = student_answers[q_num]
            correct_answers_list = correct_answers_str.split('|') 
            
            for opt_idx in range(NUM_OPTIONS):
                x, y = get_bubble_coordinates(q_num, opt_idx)
                
                # Cắt ảnh bong bóng từ ảnh đã resize
                bubble_roi = warped_img[y:y+BUBBLE_H, x:x+BUBBLE_W]
                
                if bubble_roi.shape[0] != BUBBLE_H or bubble_roi.shape[1] != BUBBLE_W:
                    continue
                
                # Resize về kích thước chuẩn (28x28) cho model
                bubble_img = cv2.resize(bubble_roi, (28, 28))
                
                option_char = OPTIONS_MAP[opt_idx] 
                
                if option_char in correct_answers_list:
                    save_dir = output_dir_1
                else:
                    save_dir = output_dir_0
                    
                out_filename = f"{os.path.splitext(filename)[0]}_q{q_num+1}_opt{option_char}.png"
                save_path = os.path.join(save_dir, out_filename)
                cv2.imwrite(save_path, bubble_img)

    print(f"Hoàn tất xử lý bộ: {data_type}!")
    
if __name__ == "__main__":
    print("Bắt đầu script chuẩn bị dữ liệu...")
    print(f"Script đang chạy từ: {SCRIPT_DIR}")
    print(f"Sẽ đọc dữ liệu thô từ: {RAW_DATA_DIR}")
    print(f"Sẽ lưu dữ liệu đã xử lý vào: {PROCESSED_DATA_DIR}")
    
    for data_type, label_file_path in LABEL_FILES.items():
        process_data(data_type, label_file_path)
        
    print("\n--- HOÀN TẤT TẤT CẢ --- 🚀")
    print("Bạn có thể kiểm tra các thư mục sau:")
    print(f"- {PROCESSED_DATA_DIR}/train/0")
    print(f"- {PROCESSED_DATA_DIR}/train/1")
    print(f"- {PROCESSED_DATA_DIR}/valid/0")
    print(f"- {PROCESSED_DATA_DIR}/valid/1")