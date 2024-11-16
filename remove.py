import pandas as pd

# Đọc dữ liệu từ file Excel
df = pd.read_excel('/Users/phuongquynh/Downloads/SMSSpamCollection.xlsx', names=["Label", "Message"])

# Hiển thị số lượng các nhãn trước khi lọc
print("Số lượng nhãn trước khi lọc:")
print(df['Label'].value_counts())

# Lọc các dòng không phải là 'am'
df_filtered = df[df['Label'].str.lower() != 'am']

# Hiển thị số lượng các nhãn sau khi lọc
print("\nSố lượng nhãn sau khi lọc:")
print(df_filtered['Label'].value_counts())

# Lưu lại dữ liệu đã lọc vào tệp Excel mới
df_filtered.to_excel('/Users/phuongquynh/Downloads/SMSSpamCollection_filtered.xlsx', index=False)

print("\nĐã lưu dữ liệu đã lọc vào 'SMSSpamCollection_filtered.xlsx'")
