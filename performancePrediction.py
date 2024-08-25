import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dữ liệu vào DataFrame
df = pd.read_csv('./archive/top5-players.csv')

# Lựa chọn các cột liên quan
features = ['Age', 'MP', 'Starts', 'Min', '90s', 'xG', 'npxG', 'xAG', 
            'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls_90', 'Ast_90', 
            'G+A_90', 'G-PK_90', 'G+A-PK_90', 'xG_90', 'xAG_90', 'xG+xAG_90', 
            'npxG_90', 'npxG+xAG_90']

target = 'Gls'

# One-hot encoding cho các biến phân loại (chỉ áp dụng cho cột Pos nếu cần)
df = pd.get_dummies(df, columns=['Pos'], drop_first=True)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
string_columns = df.select_dtypes(include=['object']).columns
# Điền NaN cho các cột số bằng giá trị trung bình
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Điền NaN cho các cột chuỗi bằng một giá trị mặc định
df[string_columns] = df[string_columns].fillna('Unknown')

# Chia tách dữ liệu
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')
