# **Báo cáo Bài tập lớn: Hệ thống Chấm Trắc nghiệm Tự động (OMR)**

Dự án này là một ứng dụng web "full-stack" được xây dựng bằng Python, sử dụng Thị giác Máy tính (OpenCV) và Học sâu (TensorFlow/Keras) để tự động hóa hoàn toàn quy trình chấm điểm phiếu trả lời trắc nghiệm.

Hệ thống có khả năng đọc ảnh scan hoặc ảnh chụp (đã căn thẳng) của phiếu trả lời, trích xuất Số Báo Danh (SBD), Mã Đề, và 60 câu trả lời. Giao diện web cho phép giáo viên tải lên "Phiếu Đáp Án" (đã tô) và "Phiếu Bài Làm" (của học sinh) để thực hiện so sánh, chấm điểm, và tự động phát hiện sai mã đề.

*(Bạn hãy chèn link ảnh sơ đồ kiến trúc của bạn vào đây)*

## **1\. Link Github**

Bạn có thể tìm thấy toàn bộ mã nguồn của dự án tại đây:

https://github.com/Kaiser2484/computer_vision_nhom8.git

## **2\. Tính năng nổi bật**

* **Nhận dạng Bong bóng bằng AI:** Sử dụng mô hình CNN (Mạng nơ-ron tích chập) đã huấn luyện (omr\_bubble\_model.h5) để phân loại ô "đã tô" (lớp 1\) và "không tô" (lớp 0\) với độ chính xác cao.  
* **Đọc SBD & Mã Đề:** Tự động đọc các khối SBD và Mã đề, bao gồm cả việc xử lý logic thứ tự bong bóng 1-2-3-4-5-6-7-8-9-0.  
* **Gỡ lỗi Đọc sai (64X \-\> 640):** Áp dụng hệ thống **Hai Ngưỡng (Dual-Threshold)**: một ngưỡng thấp (0.3) linh hoạt cho SBD/Mã đề (để bắt ô tô mờ) và một ngưỡng chuẩn (0.5) cho đáp án.  
* **Giao diện Web 2 Cột:** Cung cấp giao diện index.html (dùng Flask) cho phép giáo viên upload 2 ảnh (Key và Bài làm) và nhận về kết quả so sánh 2 cột trực quan.  
* **Tự động phát hiện Sai Mã Đề:** Server (app/main.py) tự động so sánh Mã đề của giáo viên và học sinh. Nếu không khớp, hệ thống tự động trả về **0 điểm** và hiển thị cảnh báo **"(SAI MÃ ĐỀ\!)"** màu đỏ trên giao diện.  
* **Đồng bộ hóa Logic:** Sử dụng cùng một phương pháp tiền xử lý (cv2.resize) trong cả lúc huấn luyện (prepare\_data.py) và lúc chấm điểm (grader.py) để giải quyết triệt để lỗi logic "60/60" (chấm đúng sai).

## **3\. Công nghệ sử dụng**

* **Backend:** Python 3.10+  
* **Web Server:** Flask  
* **Học sâu (Deep Learning):** TensorFlow / Keras  
* **Thị giác Máy tính (Computer Vision):** OpenCV-Python  
* **Xử lý dữ liệu:** Pandas, Numpy  
* **Giao diện (Frontend):** HTML5, CSS3, JavaScript (Fetch API)

## **4\. Cấu trúc Thư mục**

omr\_project/  
├── 📁 app/                    \# Chứa ứng dụng Web (Flask)  
│   ├── 📁 templates/  
│   │   └── 📄 index.html     \# Giao diện người dùng  
│   ├── 📄 \_\_init\_\_.py  
│   └── 📄 main.py           \# Server Flask, API /grade, logic Sai Mã Đề  
│  
├── 📁 data/  
│   ├── 📁 uploads/            \# Nơi lưu tạm ảnh (sẽ bị xóa sau khi chấm)  
│   └── 📁 saved\_model/  
│       └── 📄 omr\_bubble\_model.h5 \# "Bộ não" AI đã huấn luyện  
│  
├── 📁 omr\_engine/             \# Lõi chấm điểm (Python)  
│   ├── 📄 \_\_init\_\_.py  
│   ├── 📄 grader.py           \# File logic chính: grade\_paper(), read\_id\_grid()...  
│   ├── 📄 model\_loader.py     \# Tải file .h5  
│   └── 📄 template\_config.py  \# CHỨA TẤT CẢ TỌA ĐỘ (layout)  
│  
├── 📁 training/               \# Thư mục huấn luyện (chỉ dùng 1 lần)  
│   ├── 📁 Dataset/            \# (Đầu vào) Chứa ảnh phiếu thô (train/, valid/)  
│   ├── 📁 processed\_data/     \# (Đầu ra) Chứa ảnh bong bóng đã cắt (0/, 1/)  
│   ├── 📄 prepare\_data.py     \# Script cắt ảnh (Resize \-\> Crop \-\> Save)  
│   ├── 📄 train\_labels.csv    \# File nhãn thủ công  
│   ├── 📄 train\_model.py      \# Script huấn luyện CNN  
│   └── 📄 valid\_labels.csv    \# File nhãn thủ công  
│  
└── 📄 README.md               \# File hướng dẫn này

## **5\. Cài đặt**

1. Clone repository này về máy.  
2. Tạo một môi trường ảo (virtual environment) cho Python:

python \-m venv .venv  
source .venv/bin/activate  \# Trên Mac/Linux  
.\\.venv\\Scripts\\activate   \# Trên Windows

3. Cài đặt tất cả các thư viện cần thiết:  
   (Bạn nên tạo 1 file requirements.txt với nội dung bên dưới và chạy pip install \-r requirements.txt)

pip install tensorflow  
pip install opencv-python  
pip install flask  
pip install pandas  
pip install numpy  
pip install tqdm

## **6\. Hướng dẫn sử dụng**

Dự án có 2 giai đoạn: Huấn luyện (chỉ làm 1 lần hoặc khi đổi phiếu) và Chạy Ứng dụng (sử dụng chính).

### **Giai đoạn 1: Huấn luyện Model (Tùy chọn)**

*(Bạn chỉ làm bước này nếu muốn huấn luyện lại model cho một phiếu mẫu mới)*

1. **Chuẩn bị Dữ liệu thô:** Đặt các ảnh scan/chụp thẳng vào training/Dataset/train/ và training/Dataset/valid/.  
2. **Gán nhãn:** Mở training/train\_labels.csv và valid\_labels.csv. Gán nhãn thủ công cho từng file ảnh (dùng X cho ô trống, A|B cho tô 2 đáp án, 1-9-0 cho SBD/Mã đề).  
3. **Đo tọa độ (Quan trọng):**  
   * Chạy python training/prepare\_data.py (lần đầu có thể bị lỗi).  
   * Mở file training/DEBUG\_WARPED\_IMAGE.png (được tạo ra tự động) bằng MS Paint.  
   * Đo chính xác 100% tọa độ của SBD, Mã đề, 4 cột đáp án.  
   * Cập nhật các tọa độ này vào phần CẤU HÌNH LAYOUT (Bước 2\) trong file training/prepare\_data.py.  
4. **Tạo dữ liệu:** Chạy python training/prepare\_data.py một lần nữa. Lần này, nó sẽ tạo ra hàng ngàn ảnh trong training/processed\_data/.  
5. **Huấn luyện:** Chạy python training/train\_model.py. File omr\_bubble\_model.h5 mới sẽ được tạo và lưu vào data/saved\_model/.

### **Giai đoạn 2: Chạy Ứng dụng Web (Sử dụng chính)**

1. **Cập nhật Tọa độ Lõi:** Mở omr\_engine/template\_config.py. **Copy-paste** toàn bộ các tọa độ layout bạn đã đo ở Giai đoạn 1 vào file này. (Bước này đảm bảo grader.py và prepare\_data.py dùng chung 1 "bản đồ").  
2. **Chạy Server:** Mở terminal, cd vào thư mục gốc (omr\_project/) và chạy:

python app/main.py

3. Truy cập Giao diện: Mở trình duyệt (Chrome, Firefox...) và truy cập địa chỉ:  
   http://127.0.0.1:5000  
4. **Chấm điểm:**  
   * Tải "Ảnh Đáp Án" (phiếu của giáo viên đã tô).  
   * Tải "Ảnh Bài Làm" (phiếu của học sinh).  
   * Nhấn "Chấm điểm" và xem kết quả 2 cột.

## **7\. Tài liệu tham khảo và Tài nguyên**

\[1\] OpenCV, “OpenCV (Open Source Computer Vision Library).” \[Trực tuyến\]. Địa chỉ: https://opencv.org/ \[Truy cập ngày 04/11/2025\].

\[2\] M. Abadi, A. Agarwal, P. Barham, và cs., “TensorFlow: Large-scale machine learning on heterogeneous systems,” trong *Kỷ yếu Hội nghị USENIX Symposium on Operating Systems Design and Implementation (OSDI '16) lần thứ 12*, Savannah, GA, USA, 2016, tr. 1-16.**

\[3\] Pallets Projects, “Flask: A web framework for Python.” \[Trực tuyến\]. Địa chỉ: https://flask.palletsprojects.com/ \[Truy cập ngày 04/11/2025\].

\[4\] A. R. Setiawan, A. Harjoko, và A. M. Bachtiar, “A robust Optical Mark Recognition using Convolutional Neural Network,” *Tạp chí International Journal of Advanced Computer Science and Applications (IJACSA)*, tập 9, số 10, tr. 1-6, 2018\.

\[5\] C. R. Harris, K. J. Millman, S. J. van der Walt, và cs., “Array programming with NumPy,” *Tạp chí Nature*, tập 585, tr. 357–362, 2020\.

\[6\] N. V. An, “*Nghiên cứu và xây dựng hệ thống chấm trắc nghiệm tự động sử dụng xử lý ảnh và học máy*,” Luận văn Thạc sĩ, Trường Đại học Bách khoa Hà Nội, Hà Nội, 2020\.

\[7\] Pandas Development Team, “pandas: powerful Python data analysis toolkit.” \[Trực tuyến\]. Địa chỉ: https://pandas.pydata.org/ \[Truy cập ngày 04/11/2025\].

\[8\] A. Rosebrock, “Bubble Sheet Multiple Choice Scanner and Test Grader using OMR, Python and OpenCV,” *PyImageSearch*. \[Trực tuyến\]. Địa chỉ: https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/ \[Truy cập ngày 04/11/2025\].