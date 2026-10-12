import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv("clear_data.csv")

X = df.drop(columns='Risk_Derecesi')
y = df['Risk_Derecesi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000, multi_class='multinomial')
model.fit(X_train, y_train)

# 5. Başarı Oranını (Accuracy) Ölç
basari = model.score(X_test, y_test)
print(f"Modelin 5 sınıflı tahmin başarısı (Accuracy): %{basari * 100:.2f}")

# 6. Modeli API'de kullanmak üzere kaydet
joblib.dump(model, 'hassas_risk_modeli.joblib')
