> **Dự án:** PROJECT 2 - Content-Based Company Similarity Recommendation & 'Recommend or Not' Classification for IT Candidates  

---

## 1. Giới thiệu dự án
Dự án ứng dụng Khoa học Dữ liệu (Data Science) và Học máy (Machine Learning) nhằm hỗ trợ ứng viên IT tìm hiểu, so sánh doanh nghiệp và tham khảo đánh giá từ nhân viên thông qua hai bài toán chính:

1. **Bài toán 1 (Yêu cầu 1): Hệ thống Gợi ý Doanh nghiệp tương đồng (IT Company Recommender System)**  
   - Gợi ý doanh nghiệp tương đồng dựa trên mô tả, kỹ năng, quy mô và quốc gia bằng TF-IDF và Cosine Similarity.
2. **Bài toán 2 (Yêu cầu 2): Mô hình phân loại "Recommend or Not"**  
   - Phân loại đánh giá của nhân viên thành hai nhóm *Recommend* (Khuyên làm việc) và *Not Recommend* (Không khuyên làm việc) bằng thuật toán Logistic Regression kết hợp xử lý NLP tiếng Việt.

---

## 2. Công nghệ sử dụng
- **Ngôn ngữ lập trình:** Python
- **Xử lý dữ liệu:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Xử lý ngôn ngữ tự nhiên (NLP):** TF-IDF, PyVi, xử lý teencode và phủ định
- **Đánh giá mô hình:** StratifiedGroupKFold, tối ưu ngưỡng xác suất (Threshold Optimization)
- **Triển khai (Deployment):** Streamlit Community Cloud

---

## 3. Kết quả nổi bật

### **Hệ thống gợi ý doanh nghiệp (Bài toán 1)**
- Hỗ trợ gợi ý dựa trên doanh nghiệp có sẵn hoặc dựa trên mô tả kỹ năng, JD tuyển dụng thực tế.
- Thuật toán TF-IDF của Scikit-learn chạy nhanh hơn khoảng 9 lần so với phương pháp Gensim trong thử nghiệm trong khi độ tương quan giữa điểm similarity giữa 2 phương pháp đạt r = 0.9234.

### **Mô hình phân loại đánh giá (Bài toán 2)**
- Sử dụng mô hình Logistic Regression kết hợp cân bằng trọng số lớp (Class Weight).
- Áp dụng kỹ thuật `StratifiedGroupKFold` nhằm hạn chế tối đa rò rỉ dữ liệu (data leakage) giữa các công ty.
- Ngưỡng xác suất tối ưu: **0.80**.
- Hiệu suất trên nhãn 0 (Not Recommend): Precision = 0.7938, Recall = 0.7440, F1-score = 0.7681.

---

## 4. Cấu trúc thư mục dự án (Project Structure)

```text
Project 2/
│
├── Project2_Bài1.ipynb
├── Project2_Bài2.ipynb  
│
├── Data:
│   ├── Overview_Companies.xlsx           # Dữ liệu gốc 478 công ty x 13 thuộc tính từ ITviec
│   ├── Overview_Companies_Cleaned.csv    # Dữ liệu công ty đã tiền xử lý hoàn chỉnh cho Bài 1 
│   ├── Reviews.xlsx                      # Dữ liệu 8,417 đánh giá chi tiết của nhân viên phục vụ Bài 2
│   ├── Reviews_Cleaned.csv               # Dữ liệu 8,417 review đã tiền xử lý NLP tiếng Việt & ghép thuộc tính
│   ├── Overview_Reviews.xlsx             # Dữ liệu gốc bảng tổng hợp đánh giá cấp độ công ty
│   ├── english-vnmese.txt                # Từ điển dịch tiếng Anh sang tiếng Việt
│
├── ITviec_models1
│   ├── company_recommender_bundle.joblib # Gói mô hình gợi ý tích hợp Bài 1 (TF-IDF + Matrix + Data)
│   ├── best_logistic_model.pkl           # Gói mô hình phân loại Bài 2 (Pipeline + Threshold 0.80 + Keywords)
│   ├── tfidf_vectorizer.joblib           # Vectorizer từ điển (Bài 1)
│   ├── tfidf_matrix.joblib               # Ma trận thưa TF-IDF công ty (Bài 1)
│   ├── cosine_similarity_matrix.joblib   # Ma trận tương đồng Cosine 478 x 478 (Bài 1)
│   └── companies_df.joblib               # Bảng metadata 478 công ty
│
├── ITViec_app.py                         # Web App Streamlit, giao diện và điều hướng
├── requirements.txt                      # Danh sách thư viện Python (Scikit-Learn, PyVi, Imblearn, Streamlit,...)
├── Procfile
├── setup.sh
└── README.md                             # Tài liệu tổng quan và hướng dẫn vận hành chi tiết

```
---
## 5. Hướng dẫn Chạy & Triển khai ứng dụng (Deployment Guide)
- Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

- Khởi chạy ứng dụng cục bộ (Local):

```bash
streamlit run ITViec_app.py
```

- Live Demo: ITViec AI System (`https://itviec-jvva3t6zx5cyqqnq6qbxhc.streamlit.app/`)

## 6. Thông tin tác giả & Bản quyền
- **Đồ án tốt nghiệp / Project 2:** Khoa học Dữ liệu & Học máy ứng dụng (Applied Data Science & Machine Learning) 
- Tài liệu phục vụ mục đích học tập, nghiên cứu và triển khai ứng dụng thực tế.
