import pandas as pd
from sklearn.model_selection import train_test_split

# 加载修改后的数据
train = pd.read_csv(r'Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\dataset\modified_train.csv')

# 划分训练集和验证集
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# 保存划分后的数据集
train_df.to_csv(r'Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\dataset\train_split.csv', index=False)  # 保存训练集
val_df.to_csv(r'Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\dataset\val_split.csv', index=False)      # 保存验证集

# 查看划分后的数据集的基本信息
print(f"Train set size: {train_df.shape}")
print(f"Validation set size: {val_df.shape}")
