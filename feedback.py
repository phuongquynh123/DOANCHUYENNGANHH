import pandas as pd

# Tạo một DataFrame với tiêu đề
data = {
    'Name': [],
    'Feedback': []
}
df = pd.DataFrame(data)

# Lưu DataFrame vào tệp CSV lần đầu tiên (nếu cần thiết)
df.to_csv('feedback.csv', index=False)
