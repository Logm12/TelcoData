# 📡 Telco Customer Churn Prediction (Dự Đoán Khách Hàng Rời Mạng)

Chào mừng bạn đến với repo **Telco Customer Churn Prediction**. Đây không chỉ là một bài tập huấn luyện mô hình Machine Learning đơn thuần, mà là một giải pháp **End-to-End Data Engineering** hoàn chỉnh, được thiết kế để giải quyết bài toán thực tế của doanh nghiệp viễn thông: **Giữ chân khách hàng**.

Dự án này được xây dựng với tư duy của một **Data Engineer/Scientist**, tập trung vào sự bền vững của hệ thống, chất lượng code (Clean Code), và quan trọng nhất là khả năng giải thích được kết quả (Explainability) để mang lại giá trị kinh doanh thực sự.

---

## 🎯 Mục Tiêu Dự Án
Xây dựng một hệ thống có khả năng:
1.  **Xử lý dữ liệu tự động**: Từ khâu làm sạch, mã hóa (encoding) đến chuẩn hóa (scaling) data một cách bài bản.
2.  **Dự báo chính xác**: Sử dụng các thuật toán mạnh mẽ (Random Forest, XGBoost) kết hợp kỹ thuật xử lý mất cân bằng dữ liệu (SMOTE).
3.  **Thấu hiểu khách hàng (Business Insight)**: Tích hợp **SHAP Values** để trả lời câu hỏi *"Tại sao khách hàng này lại muốn rời đi?"* (do giá cước cao, gói mạng kém, hay dịch vụ hỗ trợ tệ?).
4.  **Triển khai thực tế**: Đóng gói ứng dụng thành Web App (Streamlit) và Container (Docker) để dễ dàng demo và deploy.

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

*   **Ngôn ngữ**: Python 3.9
*   **Data Processing**: Pandas, NumPy (Ưu tiên Vectorization thay vì Loop để tối ưu hiệu năng).
*   **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn (SMOTE).
*   **Explainability**: SHAP (SHapley Additive exPlanations).
*   **App & UI**: Streamlit.
*   **DevOps**: Docker.
*   **Quản lý Code**: Tuân thủ PEP8, Type Hinting, Modular Design (tách file `utils`, `preprocessing`, `train_model` riêng biệt).

---

## 📂 Cấu Trúc Dự Án
Dự án được tổ chức tách bạch, rõ ràng để dễ bảo trì và mở rộng:

```
telco-churn-prediction/
├── app/
│   └── main.py              # Giao diện Web App (Streamlit)
├── data/
│   ├── raw/                 # Dữ liệu thô (được sinh tự động hoặc file csv gốc)
│   └── processed/           # Dữ liệu sau khi xử lý (nếu cần lưu)
├── models/
│   ├── best_model.joblib    # Model tốt nhất sau khi training
│   └── preprocessor.joblib  # Pipeline xử lý dữ liệu (để đảm bảo tính nhất quán khi predict)
├── src/
│   ├── preprocessing.py     # Class DataPreprocessor (Clean, Split, Transform)
│   ├── train_model.py       # Pipeline huấn luyện, đánh giá và lưu model
│   └── utils.py             # Các hàm tiện ích (Logging, Config...)
├── Dockerfile               # Cấu hình đóng gói Docker
├── generate_data.py         # Script tạo dữ liệu giả lập (để test pipeline ngay lập tức)
├── requirements.txt         # Danh sách thư viện
└── README.md                # Tài liệu hướng dẫn (Bạn đang đọc nó)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### Cách 1: Chạy trực tiếp trên máy (Window/Linux/Mac)

1.  **Cài đặt thư viện**:
    Khuyên dùng môi trường ảo (Virtual Env) hoặc Conda.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chuẩn bị dữ liệu**:
    Nếu chưa có dữ liệu, chạy script sau để tạo 5000 dòng dữ liệu giả lập chất lượng cao:
    ```bash
    python generate_data.py
    ```

3.  **Huấn luyện mô hình (Training Pipeline)**:
    Bước này sẽ chạy toàn bộ quy trình: Load -> Clean -> Split -> SMOTE -> Train (RF & XGB) -> Evaluate -> Save Artifacts.
    ```bash
    python src/train_model.py
    ```
    *Check log để xem độ chính xác (Accuracy) và AUC score.*

4.  **Khởi chạy Ứng dụng**:
    ```bash
    streamlit run app/main.py
    ```
    Truy cập vào đường dẫn `http://localhost:8501` để trải nghiệm.

### Cách 2: Chạy bằng Docker 🐳

Đóng gói và chạy môi trường độc lập, không lo xung đột thư viện.

1.  **Build Image**:
    ```bash
    docker build -t telco-churn .
    ```

2.  **Run Container**:
    ```bash
    docker run -p 8501:8501 telco-churn
    ```

---

## 💡 Điểm Nổi Bật (Highlights)

*   **Tính Mô Đun (Modularity)**: Code xử lý dữ liệu được đóng gói vào class `DataPreprocessor`, dễ dàng tái sử dụng cho cả lúc Train và lúc Predict trên App. Không có chuyện xử lý thủ công rời rạc.
*   **Xử Lý Mất Cân Bằng (Imbalance Handling)**: Dữ liệu rời mạng thường ít hơn dữ liệu ở lại. Việc áp dụng SMOTE giúp model học tốt hơn nhóm khách hàng rời mạng (Churn = Yes).
*   **Giải Thích Minh Bạch**: Ứng dụng tích hợp biểu đồ SHAP Force Plot, giúp nhân viên CSKH nhìn vào và biết ngay cần tác động vào yếu tố nào (ví dụ: Giảm giá cước tháng này) để giữ chân khách.

---
*Dự án được xây dựng với sự chỉn chu và tâm huyết. Hi vọng nó sẽ hữu ích cho portfolio của bạn!* 🚀
