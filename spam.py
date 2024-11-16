import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Đọc dữ liệu từ file Excel đã lọc
df = pd.read_excel('/Users/phuongquynh/Downloads/SMSSpamCollection_filtered.xlsx', names=["Label", "Message"])

# Kiểm tra giá trị NaN
print("Số lượng NaN trước khi loại bỏ:")
print(df.isna().sum())

# Xóa các hàng có NaN
df = df.dropna()

# Đảm bảo rằng cột 'Label' chỉ có các giá trị hợp lệ: ham, spam
df['Label'] = df['Label'].str.lower()

# Đảm bảo các nhãn hợp lệ: ham, spam
valid_labels = ['ham', 'spam']
df = df[df['Label'].isin(valid_labels)]

# Gán nhãn 2 cho "spam" và 3 cho "ham"
label_mapping = {'spam': 2, 'ham': 3}
df['Label'] = df['Label'].map(label_mapping)

# Kiểm tra lại dữ liệu sau khi gán nhãn
print("\nDữ liệu sau khi gán nhãn:")
print(df.head())

# Tải stopwords từ nltk nếu chưa tải
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))  # Loại bỏ dấu câu
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Loại bỏ ký tự đặc biệt
    return text

# Tiền xử lý văn bản trong cột 'Message'
df['Message'] = df['Message'].apply(preprocess_text)

# Loại bỏ stopwords
df['Message'] = df['Message'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Kiểm tra lại dữ liệu sau khi tiền xử lý
print("\nDữ liệu sau khi tiền xử lý:")
print(df.head())

# Vector hóa văn bản bằng TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = tfidf.fit_transform(df['Message'])  # Chuyển văn bản thành dạng ma trận

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['Label'], test_size=0.2, random_state=42)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Lưu mô hình và vectorizer đã huấn luyện
joblib.dump(model, 'model/spam_classifier.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')

print("\nĐã lưu mô hình và vectorizer vào thư mục 'model/'")

