
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml
import re

# 1. 加载数据

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from transformers import AutoTokenizer
import numpy as np

# 加载数据
train = pd.read_csv(r'Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\dataset\train.csv')


# 填充 Misconception 列中的缺失值
train.Misconception = train.Misconception.fillna('NA')

# 创建 target 列
# 合并 Category 和 Misconception 列，生成标签
train['target'] = train['Category'] + ":" + train['Misconception']

# 标签编码：将 target 列转换为数字标签
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])

# 打印数据集的基本信息
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
print(train.head())

# 生成格式化文本
def format_input(row):
    # 判断是否正确回答
    correct_answer = "Yes" if "True" in row['Category'] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {correct_answer}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# 应用格式化函数到每一行
train['text'] = train.apply(format_input, axis=1)

# 查看格式化后的文本示例
print("Example prompt for our LLM:")
print(train.text.values[500])
train.to_csv('Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\dataset\modified_train.csv', index=False)

# Tokenizer 准备（你可以选择合适的预训练模型）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 计算每个文本的 token 数量
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]

# 绘制 token 长度的直方图
plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('token_length_distribution.png')

# 检查哪些样本的 token 长度超过最大限制
MAX_LEN = 256
L = (np.array(lengths) > MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")

# 排序 token 长度
np.sort(lengths)

