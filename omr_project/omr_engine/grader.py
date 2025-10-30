import cv2
import numpy as np
from . import template_config as config # Import cấu hình layout
from .model_loader import bubble_model # Import "bộ não" AI

# --- 1. HÀM NẮN ẢNH ---
def find_and_warp(image):
    """
    (PHIÊN BẢN ĐƠN GIẢN)
    Chỉ resize ảnh về kích thước chuẩn và chuyển sang ảnh xám.
    Giống hệt logic trong 'prepare_data.py' (phiên bản đơn giản).
    """
    try:
        resized_img = cv2.resize(image, (config.WARPED_IMAGE_WIDTH, config.WARPED_IMAGE_HEIGHT))
        warped_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # Trả về ảnh màu (để vẽ) và ảnh xám (để đọc)
        return resized_img, warped_gray
    except Exception as e:
        print(f"Lỗi khi resize ảnh: {e}")
        return None, None

# --- 2. HÀM DỰ ĐOÁN BONG BÓNG (ĐÃ SỬA) ---
def predict_bubble(bubble_roi, is_id_bubble=False):
    """
    Nhận 1 ảnh bong bóng, dùng model AI để dự đoán 0 (trống) hay 1 (tô).
    is_id_bubble: Nếu là True, sẽ dùng ngưỡng thấp hơn (linh hoạt hơn).
    """
    if bubble_model is None:
        raise ValueError("Model chưa được tải! Hãy kiểm tra file model_loader.py")
        
    # Luôn resize về 28x28 (kích thước model mong đợi)
    bubble_img_resized = cv2.resize(bubble_roi, config.MODEL_INPUT_IMG_SIZE)
    
    img_array = np.expand_dims(bubble_img_resized, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = bubble_model.predict(img_array, verbose=0)[0][0] # Lấy giá trị float (0.0 -> 1.0)
    
    # Đặt ngưỡng dự đoán
    if is_id_bubble:
        threshold = 0.3 # Ngưỡng thấp hơn cho SBD/Mã đề (thử nghiệm với 0.3 hoặc 0.4)
    else:
        threshold = 0.5 # Ngưỡng chuẩn cho ô đáp án
        
    if prediction > threshold:
        return 1, prediction # Trả về cả dự đoán (1) và độ tự tin (prediction)
    else:
        return 0, prediction # Trả về cả dự đoán (0) và độ tự tin (prediction)

# --- 3. HÀM ĐỌC CÁC KHU VỰC ---

def get_bubble_coordinates(q_num, opt_idx):
    """Lấy tọa độ từ file config (CHO CÂU HỎI)"""
    col_idx = q_num // config.QUESTIONS_PER_COLUMN
    row_idx = q_num % config.QUESTIONS_PER_COLUMN
    if col_idx == 0: start_x = config.START_X_COL_1
    elif col_idx == 1: start_x = config.START_X_COL_2
    elif col_idx == 2: start_x = config.START_X_COL_3
    elif col_idx == 3: start_x = config.START_X_COL_4
    else: return 0, 0 
    x = start_x + (opt_idx * config.OPTION_X_SPACING)
    y = config.START_Y_ROWS + (row_idx * config.QUESTION_Y_SPACING)
    return int(x), int(y)

def get_test_id_bubble_coordinates(digit_idx, option_num):
    """Lấy tọa độ từ file config (CHO MÃ ĐỀ)"""
    x = config.TEST_ID_START_X + (digit_idx * config.TEST_ID_X_SPACING)
    y = config.TEST_ID_START_Y + (option_num * config.TEST_ID_Y_SPACING)
    return int(x), int(y)

def get_sbd_bubble_coordinates(digit_idx, option_num):
    """Lấy tọa độ từ file config (CHO SỐ BÁO DANH)"""
    x = config.SBD_START_X + (digit_idx * config.SBD_X_SPACING)
    y = config.SBD_START_Y + (option_num * config.SBD_Y_SPACING)
    return int(x), int(y)

def read_id_grid(warped_gray, num_digits, num_options, coord_func):
    """
    Hàm chung để đọc SBD hoặc Mã Đề.
    (Đã cập nhật để dùng ngưỡng thấp hơn và logic chọn độ tự tin cao nhất)
    """
    id_str = ""
    for digit_idx in range(num_digits):
        best_option = -1 # Lưu số có độ tự tin cao nhất
        highest_confidence = -1.0 # Lưu độ tự tin cao nhất

        # Lặp qua từng ô (0-9) trong cột đó
        for option_num in range(num_options):
            x, y = coord_func(digit_idx, option_num)
            bubble_roi = warped_gray[y:y+config.BUBBLE_H, x:x+config.BUBBLE_W]
            
            if bubble_roi.shape[0] != config.BUBBLE_H or bubble_roi.shape[1] != config.BUBBLE_W:
                continue
            
            # Gọi predict_bubble với is_id_bubble=True
            prediction, confidence = predict_bubble(bubble_roi, is_id_bubble=True)
            
            # LOGIC MỚI: Chọn ô có độ tự tin cao nhất (nếu có tô)
            if prediction == 1: # Chỉ xem xét các ô được dự đoán là đã tô
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    # Lấy giá trị thực tế từ bản đồ
                    actual_value = config.ID_BUBBLE_MAP[option_num]
                    best_option = actual_value 
            
        # Nối kết quả
        if best_option == -1: # Nếu không có ô nào được tô (hoặc độ tự tin quá thấp)
            id_str += "X" 
        else:
            id_str += str(best_option) # Nối số có độ tự tin cao nhất
            
    return id_str

# --- 4. HÀM CHẤM ĐIỂM CHÍNH ---

def grade_paper(image_path, answer_key=None):
    """
    Hàm chính để xử lý một bài làm.
    """
    
    # 1. Tải ảnh và "nắn thẳng" (resize)
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Không thể đọc file ảnh."}
    
    warped_color, warped_gray = find_and_warp(image)
    if warped_color is None:
        return {"error": "Lỗi khi resize ảnh."}

    # 2. Đọc Mã đề và SBD
    try:
        test_id = read_id_grid(warped_gray, 
                               config.NUM_TEST_ID_DIGITS, 
                               config.NUM_TEST_ID_OPTIONS, 
                               get_test_id_bubble_coordinates)
    except Exception as e:
        print(f"Lỗi khi đọc mã đề: {e}")
        test_id = "ERROR"
        
    try:
        sbd = read_id_grid(warped_gray,
                           config.NUM_SBD_DIGITS,
                           config.NUM_SBD_OPTIONS,
                           get_sbd_bubble_coordinates)
    except Exception as e:
        print(f"Lỗi khi đọc SBD: {e}")
        sbd = "ERROR"
    
    # 3. Đọc đáp án câu hỏi
    student_answers = {} 
    
    for q_num in range(config.NUM_QUESTIONS):
        choices = [] 
        for opt_idx in range(config.NUM_OPTIONS):
            x, y = get_bubble_coordinates(q_num, opt_idx)
            bubble_roi = warped_gray[y:y+config.BUBBLE_H, x:x+config.BUBBLE_W]
            
            if bubble_roi.shape[0] != config.BUBBLE_H or bubble_roi.shape[1] != config.BUBBLE_W:
                continue
            
            # Gọi predict_bubble với ngưỡng chuẩn (is_id_bubble=False)
            prediction, _ = predict_bubble(bubble_roi, is_id_bubble=False)
            
            if prediction == 1:
                option_char = config.OPTIONS_MAP[opt_idx]
                choices.append(option_char)
                
        if len(choices) == 0:
            student_answers[q_num] = "X"
        else:
            student_answers[q_num] = "|".join(choices)

    # 4. Chuẩn bị kết quả trả về
    result = {
        "status": "success",
        "sbd": sbd,
        "test_id": test_id,
        "student_answers": student_answers
    }

    # 5. So sánh với đáp án (NẾU CÓ)
    if answer_key is not None:
        if len(answer_key) != config.NUM_QUESTIONS:
            return {"error": f"Lỗi đáp án: Cần 60 câu, nhưng file đáp án có {len(answer_key)} câu."}
            
        total_correct = 0
        for q_num in range(config.NUM_QUESTIONS):
            correct_answer = answer_key[q_num]
            student_choice = student_answers[q_num]
            
            if correct_answer == student_choice:
                total_correct += 1
        
        # Thêm thông tin điểm vào kết quả
        score = (total_correct / config.NUM_QUESTIONS) * 10.0
        result["total_questions"] = config.NUM_QUESTIONS
        result["total_correct"] = total_correct
        result["score_10"] = round(score, 2)

    # 6. Trả về kết quả
    return result