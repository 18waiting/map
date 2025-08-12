import os
from openai import OpenAI

# 初始化API客户端
client = OpenAI(api_key="sk-2961e797a4f244e797e6f0b54ea3d369", base_url="https://api.deepseek.com")


# 定义遍历文件夹并处理txt文件的函数
def process_txt_files(directory):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # 读取文件内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 如果行数少于20行
            if len(lines) < 20:
                print(f"Processing {filename} with less than 20 lines.")

                # 构建更详细的内容生成提示
                original_text = ''.join(lines)
                prompt = (
                    "You are a helpful assistant specialized in explaining mathematical concepts in a simple and understandable way. "
                    "The following text is an explanation of a mathematical concept with errors or simplifications. "
                    "Your task is to generate additional explanations of similar type, but with varied wording or approach. "
                    "Please ensure that the generated explanations are consistent with the original style and subject matter. "
                    f"Here is the original text: {original_text}"
                )

                # 调用DeepSeek API进行内容扩充
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant specialized in explaining mathematical concepts."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )

                # 获取生成的内容并扩充
                generated_text = response.choices[0].message.content
                augmented_text = original_text + "\n" + generated_text

                # 保存扩充后的内容
                with open(file_path, 'w') as file:
                    file.write(augmented_text)
                print(f"Augmented {filename}.")


# 设定你的文件夹路径
directory = r"Y:\ChormDownload\llm-misconception-classifier-main\llm-misconception-classifier-main\miss"
process_txt_files(directory)
