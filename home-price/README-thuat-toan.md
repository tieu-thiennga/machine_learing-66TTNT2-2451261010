# Thuật toán dự báo giá nhà

## 1. Bài toán

Bài này dùng thuật toán **Linear Regression (hồi quy tuyến tính)** để dự đoán `Giá/m2` của nhà.

Các thông tin được dùng để dự đoán gồm:
- Diện tích
- Số tầng
- Số phòng ngủ

Có thể hiểu đơn giản:

```text
Diện tích
Số tầng
Số phòng ngủ
        ↓
Linear Regression
        ↓
     Giá/m2
```

## 2. Linear Regression là gì?

Linear Regression là thuật toán dùng để tìm ra mối quan hệ giữa các biến đầu vào và giá trị cần dự đoán.

Dạng phương trình:

```text
y = b + w1*x1 + w2*x2 + w3*x3
```

Trong bài này:

```text
Giá/m2 = b
        + w1 * Diện tích
        + w2 * Số tầng
        + w3 * Số phòng ngủ
```

Trong đó:
- `y`: giá trị cần dự đoán
- `x`: các biến đầu vào
- `w`: trọng số (Weight)
- `b`: Intercept

## 3. Thuật toán hoạt động như thế nào?

Đầu tiên đưa dữ liệu nhà cho mô hình.

Ví dụ:

```text
Diện tích = 50
Số tầng = 3
Số phòng ngủ = 4
Giá/m2 = 80
```

Mô hình sẽ lấy rất nhiều dữ liệu tương tự để tìm ra các `Weight` và `Intercept`.

Sau khi học xong, mô hình có một phương trình.

Ví dụ:

```text
Giá/m2 = 45.88
        + 0.0002 * Diện tích
        + 4.53 * Số tầng
        + 9.91 * Số phòng ngủ
```

Khi có một căn nhà mới, mô hình đưa các thông tin của căn nhà vào phương trình để tính ra giá dự đoán.

## 4. Các bước của chương trình

### Bước 1: Đọc dữ liệu

Dùng `pandas` để đọc file CSV:

```python
data = pd.read_csv("data.csv")
```

Sau đó lấy các cột cần dùng.

### Bước 2: Xử lý dữ liệu

Một số dữ liệu trong file có dạng chữ, ví dụ:

```text
46 m²
5 phòng
86,96 triệu/m²
```

Cần chuyển chúng về dạng số để thuật toán có thể sử dụng.

Các ô bị thiếu dữ liệu sẽ trở thành `NaN`.

Sau đó loại bỏ những dòng không đủ dữ liệu:

```python
data = data.dropna()
```

### Bước 3: Chọn Feature và Target

Feature là dữ liệu đầu vào:

```python
X = data[["Diện tích", "Số tầng", "Số phòng ngủ"]]
```

Target là thứ cần dự đoán:

```python
y = data["Giá/m2"]
```

Có thể hiểu:

```text
X = thông tin đầu vào
y = kết quả cần dự đoán
```

### Bước 4: Chia dữ liệu Train và Test

Dữ liệu được chia thành 2 phần:

```text
80% → Train
20% → Test
```

Dùng:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Phần Train dùng để cho mô hình học.

Phần Test dùng để kiểm tra mô hình sau khi học.

### Bước 5: Tạo mô hình

```python
model = LinearRegression()
```

Lúc này mới chỉ tạo mô hình, chưa học dữ liệu.

### Bước 6: Train mô hình

```python
model.fit(X_train, y_train)
```

Đây là bước quan trọng nhất.

Mô hình sẽ dựa vào dữ liệu Train để tìm ra:

```text
Weight
Intercept
```

Mục tiêu là tìm ra phương trình dự đoán phù hợp với dữ liệu.

### Bước 7: Dự đoán

Sau khi train xong:

```python
y_pred = model.predict(X_test)
```

Mô hình lấy dữ liệu `X_test` và tạo ra giá dự đoán `y_pred`.

Có thể so sánh:

```text
y_test → Giá thật
y_pred → Giá mô hình dự đoán
```

### Bước 8: Đánh giá mô hình

Bài sử dụng 3 chỉ số:

- MAE
- RMSE
- R2

#### MAE

MAE cho biết trung bình mô hình dự đoán sai bao nhiêu.

```text
MAE càng nhỏ → càng tốt
```

#### RMSE

RMSE cũng đo độ sai lệch nhưng phạt những lỗi lớn mạnh hơn MAE.

```text
RMSE càng nhỏ → càng tốt
```

#### R2

R2 cho biết mô hình giải thích được bao nhiêu sự thay đổi của dữ liệu.

```text
R2 càng gần 1 → mô hình càng tốt
R2 gần 0 → mô hình giải thích được rất ít
```

## 5. Kết quả của mô hình

Kết quả đã chạy:

```text
MAE  = 34.98
RMSE = 63.03
R2   = 0.0584
```

Các hệ số:

```text
Intercept       = 45.88
Diện tích       = 0.0002049
Số tầng         = 4.5344
Số phòng ngủ    = 9.9110
```

Phương trình mô hình:

```text
Giá/m2 = 45.88
       + 0.0002049 * Diện tích
       + 4.5344 * Số tầng
       + 9.9110 * Số phòng ngủ
```

R2 hiện tại khoảng `0.0584`, tức là mô hình với 3 biến này mới giải thích được khoảng `5.84%` sự thay đổi của `Giá/m2`.

Vì vậy mô hình hiện tại chưa dự đoán tốt. Có thể cải thiện bằng cách xử lý dữ liệu tốt hơn và thêm các đặc trưng có liên quan đến giá nhà.

## 6. Một lưu ý về dữ liệu

Cột `Giá/m2` trong dữ liệu có thể xuất hiện nhiều đơn vị khác nhau, ví dụ:

```text
86,96 triệu/m²
247.787 đ/m²
```

Không nên coi các giá trị này là cùng một đơn vị.

Trước khi đánh giá hoặc cải thiện mô hình, cần đưa tất cả `Giá/m2` về cùng một đơn vị.

## 7. Kiến thức cần nhớ

```text
Dataset
   ↓
Xử lý dữ liệu
   ↓
Chọn X và y
   ↓
Train / Test
   ↓
Linear Regression
   ↓
fit()
   ↓
predict()
   ↓
MAE / RMSE / R2
```

Các kiến thức chính:
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
