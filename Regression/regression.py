import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('matches.csv')
#EDA (Exploratory data analysis) 
print("sample:")
print(df.head())
print("\n data info:")
print(df.info())
print("\n data describtion:")
print(df.describe())

print(f"\n Missing values in target (xg) before cleaning: {df['xg'].isna().sum()}")
df = df.dropna(subset=['xg'])
print(f"Dataset shape after dropping NaN in xg: {df.shape}") # الداتا بعد الclaening فيها كام صف وكام عمود

#  (xg التارجت )Histogram 
plt.figure(figsize=(8, 4))
sns.histplot(df['xg'], kde=True, color='purple')
plt.title('Distribution of Expected Goals (xg)')
plt.xlabel('xg')
plt.ylabel('Frequency')
plt.show()
print("\n-----------------------------")
# بنحدد كل ال columns الغير عددية
cat_cardinality = df.select_dtypes(include=['object', 'category']).nunique()

print(cat_cardinality)
print("\n-----------------------------")

high_card_cols = []
for col, val in cat_cardinality.items():
    if val > 20:                           # one hot encoding هيعمل اعمدة لكل قيمة مختلفة او غير مكررة فبنعملها علشان نتجنب الniose في(KNN)
        high_card_cols.append(col)        

print(f"Dropping high-cardinality categorical columns: {high_card_cols}")

drop_cols = ['Unnamed: 0', 'xg', 'match report', 'notes', 'date', 'time', 'gf'] + high_card_cols
X = df.drop(columns=[col for col in drop_cols if col in df.columns])  # هنتنبأ ال xg  بالداتا دي
y = df['xg']  #target

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\n Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")
print("\n----------------------------")
#  visualization للقيم العددية 
df[numerical_cols].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()
corr_cols = numerical_cols + ['xg']
print("-----------------------------------")
print("\n the heatmap:")
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
print("\n-----------------------------")
#Train-Validation-Test Split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1764, random_state=42)

print(f"\n Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# the pipline 
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), #هيستبدل ال قيم المفقودة بال وسيط الحسابي
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), #هيستبدل ال قيم المفقودة ب mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ]
)

# Linear Regression  Pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

y_val_pred_lr = lr_pipeline.predict(X_val)
print("\nLinear Regression on Validation :")
print(f"MAE: {mean_absolute_error(y_val, y_val_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred_lr)):.4f}")
print(f"R2 Score: {r2_score(y_val, y_val_pred_lr):.4f}")

# KNN pipline
best_k = None
best_val_r2 = -float('inf')
k_values = list(range(1, 25))
val_r2_scores = []

for k in k_values:
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(n_neighbors=k))
    ])

    knn_pipeline.fit(X_train, y_train)
    val_pred = knn_pipeline.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    val_r2_scores.append(val_r2)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_k = k

print(f"\n The best k is: {best_k}")

plt.figure(figsize=(8, 4))
plt.plot(k_values, val_r2_scores, marker='o', color='teal', linestyle='-')
plt.title('Validation R2 Score vs. Number of Neighbors (k)')
plt.xlabel('Value of k')
plt.ylabel('Validation R2 Score')
plt.grid(True)
plt.show()

best_knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor(n_neighbors=best_k))
])

best_knn_pipeline.fit(X_train, y_train)

print("\n-----------------------------")

#هنعمل loop تطلع نتائج ال squered(r),MSE,MAE لل linear reg , KNN

for name, model in [("Linear Regression", lr_pipeline), ("KNN Regressor", best_knn_pipeline)]:
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    print(f"\n{name}:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R2 Score: {r2:.4f}")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values (xg)')
    plt.ylabel('Predicted Values (xg)')
    plt.title(f'{name}: Predicted vs Actual')
    plt.grid(True)
    plt.show()
    print("\n-----------------------------------")
print("linear regression is better then KNN model in this case ")
