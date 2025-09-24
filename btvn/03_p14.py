"""
Phương pháp Pointwise
Phương pháp Pairwise
Phương pháp Listwise

--> Trình bày lý thuyết + Mỗi phương pháp cho 1 ví dụ
"""



"""
1. Phương pháp Pointwise
Lý thuyết
- Ý tưởng: Xem bài toán ranking như hồi quy hoặc phân loại trên từng tài liệu.
- Mỗi tài liệu (document) được gán nhãn relevance 
- Model dự đoán score cho từng document → sắp xếp theo score giảm dần.
Ưu điểm: đơn giản.
Nhược điểm: Không trực tiếp tối ưu thứ hạng, chỉ tối ưu trên từng item.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Dữ liệu: Tìm kiếm "máy pha cà phê"
# Features: [giá, đánh giá, số bán, từ khóa khớp]
X_train = np.array([
    [500, 4.2, 100, 0.8],   # Cà phê máy A
    [800, 4.8, 250, 0.9],   # Cà phê máy B (tốt nhất)
    [300, 3.5, 50, 0.6],    # Cà phê máy C
    [600, 4.0, 120, 0.7]    # Cà phê máy D
])

# Nhãn thực tế (relevance score 0-5)
y_train = np.array([3, 5, 2, 4])

# Huấn luyện mô hình Pointwise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_scaled, y_train)

# Dự đoán cho dữ liệu mới
X_new = np.array([
    [700, 4.5, 180, 0.85],  # Sản phẩm mới 1
    [400, 3.8, 80, 0.7]     # Sản phẩm mới 2
])
X_new_scaled = scaler.transform(X_new)
scores = model.predict(X_new_scaled)

# Xếp hạng theo điểm số
ranking = sorted(zip(scores, ["Sản phẩm mới 1", "Sản phẩm mới 2"]), reverse=True)
print("Pointwise Ranking:")
for score, name in ranking:
    print(f"{name}: {score:.2f} điểm")


"""
2. Phương pháp Pairwise
Lý thuyết:
- Ý tưởng: Biến bài toán xếp hạng thành bài toán so sánh cặp
- Thay vì dự đoán điểm tuyệt đối, mô hình học xác suất/khả năng một tài liệu “quan trọng hơn” tài liệu khác đối với một truy vấn.
- Ưu điểm: Tập trung vào thứ tự tương đối
Nhược điểm: Tốn nhiều phép tính khi có nhiều cặp
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Dữ liệu giống Pointwise
X_train = np.array([
    [500, 4.2, 100, 0.8],   # Cà phê máy A
    [800, 4.8, 250, 0.9],   # Cà phê máy B
    [300, 3.5, 50, 0.6],    # Cà phê máy C
    [600, 4.0, 120, 0.7]    # Cà phê máy D
])

# Tạo các cặp để so sánh
# Format: [features_A - features_B, label (1 nếu A > B, 0 nếu ngược lại)]
pairs_X = []
pairs_y = []

# So sánh tất cả các cặp
y_labels = [3, 5, 2, 4]  # relevance scores
for i in range(len(X_train)):
    for j in range(len(X_train)):
        if i != j:
            # Tính difference features
            diff_features = X_train[i] - X_train[j]
            # Label: 1 nếu item i tốt hơn j
            label = 1 if y_labels[i] > y_labels[j] else 0
            pairs_X.append(diff_features)
            pairs_y.append(label)

pairs_X = np.array(pairs_X)
pairs_y = np.array(pairs_y)

# Huấn luyện mô hình Pairwise
scaler = StandardScaler()
pairs_X_scaled = scaler.fit_transform(pairs_X)

model_pairwise = GradientBoostingClassifier(n_estimators=50, random_state=42)
model_pairwise.fit(pairs_X_scaled, pairs_y)

# Dự đoán thứ tự cho sản phẩm mới
def pairwise_ranking(model, scaler, X_items, item_names):
    scores = []
    for i, item in enumerate(X_items):
        # So sánh với một item reference (có thể là average)
        ref_score = 0
        for j, other in enumerate(X_items):
            if i != j:
                diff = item - other
                diff_scaled = scaler.transform([diff])
                pred = model.predict_proba(diff_scaled)[0][1]  # prob A > B
                ref_score += pred
        scores.append(ref_score / (len(X_items) - 1))

    return sorted(zip(scores, item_names), reverse=True)

# Test với sản phẩm mới
X_test = np.array([
    [700, 4.5, 180, 0.85],  # Sản phẩm mới 1
    [400, 3.8, 80, 0.7]     # Sản phẩm mới 2
])


ranking = pairwise_ranking(model_pairwise, scaler, X_test, ["Sản phẩm mới 1", "Sản phẩm mới 2"])
print("\nPairwise Ranking:")
for score, name in ranking:
    print(f"{name}: {score:.3f} (xác suất thắng các cặp)")


"""
3. Phương pháp Listwise
Lý thuyết:
- Ý tưởng: Xem xét toàn bộ danh sách tài liệu cùng lúc thay vì từng tài liệu (Pointwise) hoặc từng cặp (Pairwise).
- Mục tiêu: Tối ưu trực tiếp các chỉ số xếp hạng (như NDCG, MAP, MRR) thay vì chỉ dự đoán điểm hoặc quan hệ cặp.
- Thường dùng trong các thuật toán như ListNet, ListMLE.
Ưu điểm: Tối ưu trực tiếp metric thực tế
Nhược điểm: Phức tạp, khó implement
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Dữ liệu: 3 truy vấn với danh sách sản phẩm
# Mỗi truy vấn có 3 sản phẩm với features và relevance labels

class SimpleListwiseNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Dự đoán score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Dữ liệu huấn luyện
X_data = np.array([
    # Query 1: 3 sản phẩm
    [[500, 4.2, 100, 0.8], [800, 4.8, 250, 0.9], [300, 3.5, 50, 0.6]],
    # Query 2: 3 sản phẩm khác
    [[600, 4.0, 120, 0.7], [450, 3.9, 90, 0.75], [900, 4.7, 300, 0.95]],
    # Query 3
    [[350, 3.2, 60, 0.65], [750, 4.6, 200, 0.88], [550, 4.1, 110, 0.82]]
])

y_data = np.array([
    # Query 1 relevance: [3, 5, 2]
    [3, 5, 2],
    # Query 2: [4, 3, 5]
    [4, 3, 5],
    # Query 3: [2, 5, 4]
    [2, 5, 4]
])

# Chuyển sang tensor
X_tensor = torch.FloatTensor(X_data.reshape(-1, X_data.shape[-1]))
y_tensor = torch.FloatTensor(y_data.reshape(-1))

# Model và training
model_listwise = SimpleListwiseNet(input_size=4)
criterion = nn.MSELoss()  # Simplified loss
optimizer = optim.Adam(model_listwise.parameters(), lr=0.01)

# Training loop đơn giản
for epoch in range(100):
    optimizer.zero_grad()
    scores = model_listwise(X_tensor)
    loss = criterion(scores.squeeze(), y_tensor)
    loss.backward()
    optimizer.step()

# Test với query mới
def listwise_ranking(model, X_query, query_name="Query mới"):
    with torch.no_grad():
        X_t = torch.FloatTensor(X_query)
        scores = model(X_t).squeeze().numpy()

    # Tạo ranking
    ranking = sorted(zip(scores, [f"Sản phẩm {i + 1}" for i in range(len(scores))]), reverse=True)

    print(f"\n{query_name} - Listwise Ranking:")
    for score, name in ranking:
        print(f"{name}: {score:.2f} điểm")
    return ranking

# Test query mới
X_new_query = np.array([
    [700, 4.5, 180, 0.85],  # Sản phẩm 1
    [400, 3.8, 80, 0.7],  # Sản phẩm 2
    [650, 4.3, 150, 0.82]  # Sản phẩm 3
])

listwise_ranking(model_listwise, X_new_query, "Query: Tìm máy pha cà phê")
