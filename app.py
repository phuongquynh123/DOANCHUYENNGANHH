from flask import Flask, render_template, request, redirect, url_for, flash
import os
import re
import string
import joblib
from pymongo import MongoClient
from spam import tfidf



app = Flask(__name__)
app.secret_key = "supersecretkey"  # Khóa bí mật để sử dụng flash messages

# Đường dẫn tới thư mục static
UPLOAD_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://phuongquynh:Chihchi1973%40@cluster0.50xns.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['your_database_name']  # Thay 'your_database_name' bằng tên cơ sở dữ liệu
feedback_collection = db['feedback']  # Tạo hoặc sử dụng collection 'feedback'

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Tải các mô hình và vectorizer
MODEL_PATH = "model"
logreg_model = joblib.load(os.path.join(MODEL_PATH, "logistic_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))
nb_model = joblib.load(os.path.join(MODEL_PATH, "nb_model.pkl"))
rf_model = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))
spam_classifier = joblib.load(os.path.join(MODEL_PATH, "spam_classifier.pkl"))
tfidf.vectorizer = joblib.load(os.path.join(MODEL_PATH,"tfidf_vectorizer.pkl"))



# Trang chính hiển thị giao diện
@app.route('/')
def home():
    return render_template('index.html', final_result=None)

# Xử lý phân tích văn bản
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')
    if not input_text:
        flash("Bạn chưa nhập văn bản nào để phân tích!", "error")
        return redirect(url_for('home'))

    # Tiền xử lý văn bản
    processed_text = preprocess_text(input_text)
    vectorized_text = vectorizer.transform([processed_text])

    # Dự đoán với Logistic Regression
    prediction = logreg_model.predict(vectorized_text)[0]
    final_result = "Thông tin giả " if prediction == 1 else "Thông tin thật "

    return render_template('index.html', final_result=final_result)

# Xử lý phản hồi từ người dùng và lưu vào MongoDB
@app.route('/feedback', methods=['POST'])
def feedback():
    name = request.form.get('name')
    feedback_text = request.form.get('feedback')

    if not name or not feedback_text:
        flash("Vui lòng nhập đầy đủ thông tin phản hồi!", "error")
        return redirect(url_for('home'))

    # Lưu phản hồi vào MongoDB
    feedback_data = {"name": name, "feedback": feedback_text}
    feedback_collection.insert_one(feedback_data)

    flash("Cảm ơn bạn đã gửi phản hồi!", "success")
    return redirect(url_for('home'))

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
