# 学生数学误解检测任务数据增强详细方案

本方案基于《高分方案设计文档》中的数据增强策略，结合 Kaggle **MAP: Charting Student Math Misunderstandings** 竞赛任务特性，提供一份可落地、分阶段执行的数据增强方案，以提升模型的泛化能力与 MAP@3 分数。

---

## 1. 数据增强目标

1. **提升模型鲁棒性**：增加模型对不同表述方式的理解能力，降低因语言多样性导致的误判。
2. **缓解数据稀疏**：尤其是低频误解类型样本数量有限，通过扩充训练数据提升模型学习效果。
3. **支持多任务训练**：为分类正确性、误解检测、误解类型识别提供足够多样的训练样本。

---

## 2. 数据清洗与预处理

### 2.1 文本规范化
- 将数学公式标准化（`\frac{3}{4}` → `3/4` 或 `0.75`）
- 去除 HTML 标签、特殊字符、冗余空格
- 对公式进行可读性增强：
  - 例如 `1/2 ÷ 6` → `0.5 divided by 6`

### 2.2 拼写与语法纠正
- 使用工具如 `language-tool` 或 GPT API 自动修正拼写错误
- 保留数学符号和公式的正确表达

### 2.3 解释文本标准化
- 将“bcuz”“cuz”统一为“because”
- 将“×”统一为“*”，“÷”统一为“/”

---

## 3. 数据增强策略

### 3.1 释义增强（Paraphrasing）
**目标**：增加语言多样性，提高模型对同义表达的理解

- 方法：
  1. 使用大语言模型（如 GPT-4、Gemma）生成 2-3 个不同风格的解释版本
  2. 保证数学逻辑一致，不改变正确性或误解标签
- 示例：
  - 原文："I divided 1/2 by 6 because 2 goes into 6 three times"
  - 增强：
    1. "Half divided by six gives one third since 6 contains three halves"
    2. "Because two fits into six three times, dividing a half by six equals one third"

### 3.2 反事实增强（Counterfactual Augmentation）
**目标**：构造正负样本，帮助模型理解边界条件

- 方法：
  1. 将正确解释改为含误解的解释，或反之
  2. 确保 `Category` 与 `Misconception` 标签相应修改
- 示例：
  - 正确 → 误解："1/2 ÷ 6 = 1/12 because we divide numerator and denominator separately"
  - 误解 → 正确："1/2 ÷ 6 = 1/12 because dividing half by six gives twelfths"

### 3.3 负样本扩充（Hard Negative Mining）
**目标**：增加模型区分相似解释的能力

- 方法：
  1. 从不同误解类别中取解释并随机替换或拼接
  2. 构建“不相关”或“中立”解释，提高 True_Neither/False_Neither 分类效果

### 3.4 伪标签（Pseudo-labeling）
**目标**：利用模型对未标注数据的高置信度预测提升训练规模

- 方法：
  1. 用当前模型预测测试集
  2. 选取置信度 > 0.9 的样本加入训练集
  3. 标明来源为伪标签数据，训练时降低权重

### 3.5 外部数据迁移
**目标**：引入更多教育场景数据，提高泛化能力

- 可选数据集：
  - **Eedi 2023 Mining Misconceptions**
  - **ASSISTments 2017**
  - **EdNet 1.0**
- 处理方式：
  1. 统一成与 MAP 相同的格式（Question、Answer、Explanation、Category、Misconception）
  2. 对低频误解类别进行重点采样

---

## 4. 增强样本管理与版本控制

1. 使用 `train_v1`（原始）→ `train_v2`（清洗）→ `train_aug_v3`（增强）逐步迭代
2. 每次增强保留生成日志，包括：
   - 原始文本
   - 增强方法
   - 新文本
   - 标签变动

---

## 5. 验证与评估策略

1. 增强数据后，使用 K-fold 交叉验证监控效果
2. 关注以下指标：
   - MAP@3（核心）
   - Misconception 分类的 F1-score
   - 对低频类别的召回率
3. 定期在原始验证集上评估，防止增强引入噪声导致过拟合

---

## 6. 实施时间线

- **阶段 1**：数据清洗 + 拼写修正（1 周）
- **阶段 2**：释义增强 + 反事实增强（1 周）
- **阶段 3**：负样本扩充 + 伪标签（1 周）
- **阶段 4**：外部数据迁移与微调训练（2 周）

---

📌 **总结**：本数据增强方案通过多种方法（释义、反事实、伪标签、外部数据）系统性提升训练数据的覆盖度与多样性，从而帮助模型更好理解学生解释，增强对低频误解类别的识别能力，为后续高分冲榜和泛化奠定基础。

