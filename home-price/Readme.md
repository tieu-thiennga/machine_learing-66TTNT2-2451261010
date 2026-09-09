# Dự báo giá nhà

## 1. Bài toán

Bài này sử dụng phương pháp hồi quy tuyến tính (Linear Regression) để dự báo giá nhà.

Dữ liệu gồm các thông tin về nhà như:
- Diện tích
- Số tầng
- Số phòng ngủ
- Giá/m2

Mục tiêu là dựa vào các thông tin của căn nhà để dự đoán `Giá/m2`.

## 2. Các biến sử dụng

### X (Features)

Các biến đầu vào:
- Diện tích
- Số tầng
- Số phòng ngủ

### y (Target)

Biến cần dự đoán:
- Giá/m2

Có thể hiểu đơn giản:

X → Mô hình → y

Diện tích + Số tầng + Số phòng ngủ → Linear Regression → Giá/m2

## 3. Hồi quy tuyến tính

Hồi quy tuyến tính tìm ra một phương trình để dự đoán giá trị.

Dạng tổng quát:

y = b + w1*x1 + w2*x2 + ... + wn*xn

Trong đó:
- y là giá trị cần dự đoán
- x là các biến đầu vào
- w là hệ số của các biến
- b là hệ số chặn

Sau khi học, mô hình sẽ tìm ra các hệ số này.

## 4. Các bước làm

### Bước 1: Đọc dữ liệu

Đọc file csv bằng pandas.

### Bước 2: Xử lý dữ liệu

Chuyển các dữ liệu như:
- `46 m²` → `46`
- `5 phòng` → `5`
- `86,96 triệu/m²` → số

Các ô không có dữ liệu sẽ được xử lý bằng `NaN` và loại bỏ khi tạo dữ liệu cho mô hình.

### Bước 3: Chọn X và y

```python
X = data[["Diện tích", "Số tầng", "Số phòng ngủ"]]
y = data["Giá/m2"]
```

### Bước 4: Chia dữ liệu

Chia dữ liệu thành:
- 80% để train
- 20% để test

Dùng `train_test_split()`.

### Bước 5: Tạo mô hình

```python
model = LinearRegression()
```

### Bước 6: Train

```python
model.fit(X_train, y_train)
```

Đây là bước mô hình học từ dữ liệu và tìm ra các hệ số.

### Bước 7: Dự đoán

```python
y_pred = model.predict(X_test)
```

Mô hình sử dụng những gì đã học để dự đoán giá.

### Bước 8: Đánh giá

Sử dụng:
- MAE
- RMSE
- R2

để xem mô hình dự đoán tốt hay không.

## 5. Kết quả

Mô hình hiện tại sử dụng 3 đặc trưng:
- Diện tích
- Số tầng
- Số phòng ngủ

Sau khi chạy mô hình sẽ thu được:
- MAE
- RMSE
- R2
- Các Weight
- Intercept

Từ Weight và Intercept có thể viết được phương trình mà mô hình đã học.

## 6. Kiến thức học được

Qua bài này tìm hiểu được:
- Dataset
- Feature
- Target
- X và y
- Train/Test
- Linear Regression
- Weight
- Intercept
- `fit()`
- `predict()`
- MAE
- RMSE
- R2
