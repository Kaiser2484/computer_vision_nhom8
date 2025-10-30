from flask import Flask, request, jsonify, render_template
import os
import json
import uuid 

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from omr_engine.grader import grade_paper

# --- CẤU HÌNH ---
app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_temp_file(file):
    """Hàm tiện ích: Lưu file tạm và trả về đường dẫn"""
    ext = os.path.splitext(file.filename)[1]
    temp_filename = str(uuid.uuid4()) + ext
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)
    return temp_path

# --- TRANG CHỦ ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- API CHẤM ĐIỂM (ĐÃ SỬA) ---
@app.route('/grade', methods=['POST'])
def grade_exam():
    """API nhận 2 ảnh, chấm điểm, và kiểm tra Mã đề."""
    
    if 'answer_key_image' not in request.files:
        return jsonify({"error": "Không có file 'Ảnh Đáp Án'."}), 400
    answer_key_file = request.files['answer_key_image']
    if answer_key_file.filename == '':
        return jsonify({"error": "Chưa chọn file 'Ảnh Đáp Án'."}), 400
        
    if 'student_image' not in request.files:
        return jsonify({"error": "Không có file 'Ảnh Bài Làm'."}), 400
    student_file = request.files['student_image']
    if student_file.filename == '':
        return jsonify({"error": "Chưa chọn file 'Ảnh Bài Làm'."}), 400

    temp_key_path = None
    temp_student_path = None
    
    try:
        temp_key_path = save_temp_file(answer_key_file)
        temp_student_path = save_temp_file(student_file)

        # 1. Đọc ảnh đáp án
        print(f"--- Đang đọc đáp án từ: {answer_key_file.filename} ---")
        key_result = grade_paper(temp_key_path, answer_key=None) 
        
        if key_result.get("status") != "success":
            os.remove(temp_key_path)
            os.remove(temp_student_path)
            return jsonify({"error": f"Không thể đọc file đáp án: {key_result.get('error')}"}), 500

        key_test_id = key_result.get("test_id", "ERROR_KEY")
        extracted_answer_key_list = list(key_result['student_answers'].values())

        # 2. Đọc ảnh bài làm (chưa chấm)
        print(f"--- Đang đọc bài làm: {student_file.filename} ---")
        student_read_result = grade_paper(temp_student_path, answer_key=None)

        if student_read_result.get("status") != "success":
             os.remove(temp_key_path)
             os.remove(temp_student_path)
             return jsonify({"error": f"Không thể đọc file bài làm: {student_read_result.get('error')}"}), 500

        student_test_id = student_read_result.get("test_id", "ERROR_STUDENT")
        student_sbd = student_read_result.get("sbd", "ERROR_SBD")
        student_answers_dict = student_read_result.get("student_answers", {})

        # 3. --- LOGIC KIỂM TRA MÃ ĐỀ ---
        test_id_mismatch = False
        # Chỉ kiểm tra nếu cả 2 mã đề đọc được và không chứa lỗi/bỏ trống ('X')
        if ("ERROR" not in [key_test_id, student_test_id] and 
            "X" not in key_test_id and "X" not in student_test_id and
            key_test_id != student_test_id):
             test_id_mismatch = True
        # --- KẾT THÚC KIỂM TRA ---

        # 4. TÍNH ĐIỂM
        final_result = {
             "status": "success",
             "sbd": student_sbd,
             "test_id": student_test_id,
             "student_answers": student_answers_dict,
             "answer_key_info": { # Thông tin từ phiếu đáp án
                  "read_from_image": answer_key_file.filename,
                  "sbd": key_result.get("sbd"),
                  "test_id": key_test_id,
                  "student_answers": key_result.get("student_answers")
             },
             "test_id_mismatch": test_id_mismatch # Thêm cờ báo lỗi
        }

        if test_id_mismatch:
             # GHI ĐÈ ĐIỂM = 0 NẾU SAI MÃ ĐỀ
             final_result["total_questions"] = len(extracted_answer_key_list)
             final_result["total_correct"] = 0
             final_result["score_10"] = 0.0
        else:
             # Nếu mã đề khớp (hoặc không thể so sánh), tiến hành chấm điểm
             total_correct = 0
             num_questions = len(extracted_answer_key_list)
             
             for i in range(num_questions):
                 # So sánh đáp án của học sinh (từ dict) và đáp án key (từ list)
                 if student_answers_dict.get(i) == extracted_answer_key_list[i]:
                      total_correct += 1
             
             score = (total_correct / num_questions) * 10.0 if num_questions > 0 else 0.0
             final_result["total_questions"] = num_questions
             final_result["total_correct"] = total_correct
             final_result["score_10"] = round(score, 2)

        # 5. Xóa file tạm và trả kết quả
        os.remove(temp_key_path)
        os.remove(temp_student_path)
        return jsonify(final_result)
        
    except Exception as e:
        if temp_key_path and os.path.exists(temp_key_path):
            os.remove(temp_key_path)
        if temp_student_path and os.path.exists(temp_student_path):
            os.remove(temp_student_path)
        print(f"LỖI NGHIÊM TRỌNG KHI CHẤM: {e}")
        # Thêm str(e) để hiển thị lỗi rõ hơn trên web
        return jsonify({"error": f"Lỗi server nghiêm trọng: {str(e)}"}), 500

# --- Chạy server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)