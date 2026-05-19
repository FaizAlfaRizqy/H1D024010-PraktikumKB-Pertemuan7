import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset iris dari file CSV/data
dataset = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, sep=',')
# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values # 4 kolom pertama sebagai fitur
y = dataset.iloc[:, -1].values # Kolom terakhir sebagai label

# Mengonversi label dari string menjadi numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) # Mengubah label jadi 0, 1, 2

# Memisahkan dataset menjadi data latih dan data validasi dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat dan melatih model neural network
model = MLPClassifier(
    hidden_layer_sizes=(1000, 500, 300),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42,
    batch_size=32
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluasi model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

predictions = model.predict(X_test)
# Mengambil hasil prediksi
predicted_classes = predictions
print("Prediksi:", predicted_classes)
print("Label Asli:", y_test)

# Buat confusion matrix
cm = confusion_matrix(y_test, predicted_classes)
# Visualisasikan confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Fungsi untuk memprediksi data input baru
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))
    # Membuat data array baru
    new_data = np.array([[sepal_length, sepal_width, petal_length,
    petal_width]])
    # Normalisasi data baru
    new_data = scaler.transform(new_data)
    # Melakukan prediksi
    prediction = model.predict(new_data)
    # Mengonversi hasil prediksi numerik menjadi label asli
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()
