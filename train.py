import pandas as pd
import joblib
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Kiểm tra text là NaN hoặc không phải là chuỗi
    if not isinstance(text, str):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def create_and_train_models():
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('Dataset/WELFake_Dataset.csv')  # Đường dẫn thực tế đến file CSV

    # Tiền xử lý dữ liệu
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Tạo TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Logistic Regression
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    logreg_pred = logreg_model.predict(X_test)
    logreg_acc = accuracy_score(y_test, logreg_pred)
    print(f"Logistic Regression Accuracy: {logreg_acc}")

    # Huấn luyện mô hình Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_acc}")

    # Huấn luyện mô hình Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc}")

    # Kiểm tra và tạo thư mục 'model' nếu chưa có
    if not os.path.exists('model'):
        os.makedirs('model')



    # Lưu mô hình và vectorizer
    joblib.dump(logreg_model, 'model/logistic_model.pkl')
    joblib.dump(nb_model, 'model/nb_model.pkl')
    joblib.dump(rf_model, 'model/rf_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

if __name__ == "__main__":
    create_and_train_models()
