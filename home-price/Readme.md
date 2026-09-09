# Bài tập 1: Dự báo chỉ số giá nhà (Linear Regression)

Note lại cách làm bài dự báo chỉ số giá nhà `National-US` từ tập dữ liệu Case-Shiller.

### Các bước thực hiện:
1. **Lấy dữ liệu:** Tải file csv từ Kaggle về và đọc bằng `pandas`.
2. **Tiền xử lý:**
   - Chuyển cột `Date` thành năm (`Year`) và tháng (`Month`).
   - Chọn các thành phố đại diện làm đặc trưng đầu vào: Los Angeles, San Diego, San Francisco, Washington, Miami, Chicago.
   - Bỏ các dòng bị trống (`dropna()`).
3. **Chia tập dữ liệu:** 80% train, 20% test (đặt `shuffle=False` để giữ đúng thứ tự thời gian).
4. **Chạy mô hình:** Dùng `LinearRegression` của `scikit-learn` để huấn luyện và dự đoán.

### Kết quả chạy được:
- R2 Score: ~0.3074
- MAE: ~35.21
- RMSE: ~39.83

### Cách chạy lại code:
```bash
pip install pandas numpy scikit-learn
python home_price.py