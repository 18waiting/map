import pandas as pd
import re

# 清洗 latex 公式为可读文本
def latex_to_plain(latex_string):
    text = re.sub(r"\\\(|\\\)", '', latex_string)  # 移除 \( 和 \)
    # 处理分数 \frac{a}{b} → a/b
    text = re.sub(r"\\frac{([^{}]+)}{([^{}]+)}", r"\1/\2", text)
    text = text.replace("\\times", "*")  # 处理乘法符号
    text = text.replace("\\div", "÷")    # 处理除法符号
    text = re.sub(r"\s+", " ", text).strip()  # 去除多余空格
    return text

# 处理文本，去除不需要的符号和格式化
def clean_text(text):
    # 移除 emoji
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # 将智能引号转为标准引号
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")
    # 处理分数格式，如：3\n_\n9 → 3/9
    text = re.sub(r"\n?\s*(\d+)\s*\n[_‐-]+\n\s*(\d+)", r"\1/\2", text)
    # 处理类似 5|120 的格式 → "120 ÷ 5"
    text = re.sub(r"(\d+)\s*\|\s*(\d+)", r"\2 ÷ \1", text)
    # 替换 \ 或 \\ 为 /
    text = re.sub(r"(?<=\d)\\+(?=\d)", "/", text)
    text = text.replace("\\", "")  # 移除多余的反斜杠
    return text

# 加载数据
df = pd.read_csv(r'Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\dataset\train.csv')

# 处理数据中的 NaN 值
df["Misconception"] = df["Misconception"].fillna("NA")

# 清理 QuestionText 和 MC_Answer 列中的 LaTeX 表达式
df["QuestionText_clean"] = df["QuestionText"].apply(latex_to_plain)
df["MC_Answer_clean"] = df["MC_Answer"].apply(latex_to_plain)

# 生成模型输入 'input' 列
df["input"] = (
    "Question: " + df["QuestionText_clean"] + "\n" +
    "Answer: " + df["MC_Answer_clean"] + "\n" +
    "Explanation: " + df["StudentExplanation"]
)

# 创建 output 列 (Category + Misconception)
df["output"] = df["Category"] + ":" + df["Misconception"]

# 生成模型指令列
category_list = df["Category"].unique()
miscon_list = df["Misconception"].unique()

cat_string = ', '.join(category_list)
misc_string = ', '.join(miscon_list)

df["instruction"] = f"Given the math question, student answer, and their explanation, determine if there is a misconception and classify it.\nValid categories: {cat_string}\nValid misconceptions (only when applicable): {misc_string}\nFormat your answer as: Category[:Misconception]"

# 清理 'input' 和 'output' 中的文本
df["input"] = df["input"].apply(clean_text)
df["output"] = df["output"].apply(clean_text)

# 查看数据预处理后的部分
print(df[["QuestionText_clean", "MC_Answer_clean", "input", "output", "instruction"]].head())

# 保存处理后的数据
df.to_csv('processed_train.csv', index=False)
