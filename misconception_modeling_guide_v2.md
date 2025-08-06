# 学生数学误解类型分析与建模指南 v2

> 适用于 Kaggle “MAP: Charting Student Math Misunderstandings” 或其他教育 NLP 场景  
> 共覆盖 **35 类** `Misconception`

---

## 1 任务回顾
- **输入**：`QuestionText` + `MC_Answer` + `StudentExplanation`
- **输出**  
  1. `Category` （True / False × Correct / Misconception / Neither）  
  2. 如为 _Misconception_，再预测具体 `Misconception`（35 类）
- **挑战**：文本噪声、类别长尾、数学‑语言混合表达

---

## 2 误解类型总览

| 误解类型 | 频次 | 认知特征 / 场景 | 典型关键词或模式 | 可编程规则提示 | 数据增强建议 |
|----------|-----:|-----------------|-----------------|----------------|--------------|
| **Incomplete** | 1454 | 步骤/论证缺失 | 解释字数 < 8；“so …” | `len(text)<N` | 合成完整 vs 缺失版 |
| **Additive** | 929 | 把加法用于乘/除 | “add”, “plus”, “total” | 关键词 + 运算符检测 | 反事实：乘↔加 |
| **Duplication** | 704 | 重复项/倍数 | “twice”, “double” | 连续相同数字/词 | 插入或删除重复 |
| **Subtraction** | 620 | 误用减法 | “minus”, “take away” | 错误运算符 | 加↔减对照 |
| **Positive** | 566 | 忽略负号 | 无 “‑”；“positive” | 题含负数但解释全正 | 添加含负号样本 |
| **Wrong_term** | 558 | 代数项用错 | “term”, 错变量名 | 变量数不匹配 | 变量替换增强 |
| **Irrelevant** | 497 | 完全离题 | “I think”, “maybe” | 缺题干关键词 | 随机干扰句 |
| **Wrong_fraction** | 418 | 分数拆分错误 | “numerator”, “denominator” | 分子分母单独运算 | 交换/错误分母合成 |
| **Inversion** | 414 | 颠倒次序/倒数误用 | “flip”, “reciprocal” | 关键词 + 除法 | 正/翻转对照 |
| **Mult** | 353 | 错用乘法 | “times”, “multiply” | 运算符与题干不符 | 乘↔加/除对照 |
| **Denominator-only_change** | 336 | 只改分母 | “common denominator” | 分母变化 分子不变 | 合成样本 |
| **Whole_numbers_larger** | 329 | 整数>分数偏见 | “whole number bigger” | 比较符 “>” | 比较题增强 |
| **Adding_across** | 307 | 分数跨加 | “add across” | “fraction” + “add” | 正/误对照 |
| **WNB** | 299 | Whole‑Number Bias | “convert to whole” | 整数化描述 | 小数/分数互换 |
| **Tacking** | 290 | 贴附无关数字 | “attach”, “tag on” | 解释含额外数字 | 随机数字插入 |
| **Unknowable** | 282 | 无法判定 | “don’t know”, “guess” | 不确定词 | 生成模糊句 |
| **Wrong_Fraction** | 273 | 同 Wrong_fraction （大小写差异） | 同上 | 建议与 Wrong_fraction 合并 | — |
| **SwapDividend** | 206 | 除数/被除数互换 | “dividend”, “divisor” | 关键词顺序错误 | 互换合成 |
| **Scale** | 179 | 比例缩放误用 | “scale”, “ratio” | 缩放系数错误 | 倍数变换 |
| **Not_variable** | 154 | 把变量当常数 | “treat x as” | 变量出现在常数侧 | 变量替换 |
| **Firstterm** | 107 | 只取首项 | “first term” | 仅引用第 1 项 | 多项→首项对照 |
| **Adding_terms** | 97 | 单纯相加代数项 | “add the terms” | 丢失次项/系数 | 系数替换 |
| **Multiplying_by_4** | 96 | 固定乘 4 | “× 4”, “times four” | 固定数字 4 | 插入 ×4 错误 |
| **FlipChange** | 78 | 翻转并改变符号 | “flip and change” | 关键词连用 | 翻转+符号合成 |
| **Division** | 63 | 误用除法 | “divide by” | 除号在不当场景 | 除↔乘对照 |
| **Definition** | 54 | 概念定义错误 | “means”, “definition” | 定义句 + 错误 | 正/误定义对照 |
| **Interior** | 50 | 几何内角误解 | “interior angle” | 错角度公式 | 多边形角度题 |
| **Longer_is_bigger** | 24 | 线段长→值大 | “longer”, “bigger” | 长度词 + 比较 | 线段/刻度题 |
| **Base_rate** | 23 | 忽视基准率 | “base rate” | 百分比误判 | 比率题增强 |
| **Ignores_zeroes** | 23 | 忽略 0 影响 | “ignore zeros” | 去零描述 | 插入 0 对照 |
| **Shorter_is_bigger** | 23 | 线段短→值大 | “shorter”, “bigger” | 同上反向 | 线段题 |
| **Inverse_operation** | 21 | 逆运算误用 | “inverse operation” | 求逆但错误 | 正/误逆运算 |
| **Certainty** | 18 | 过度自信 | “definitely”, “for sure” | 高置信词 | 反事实自信度 |
| **Incorrect_equivalent_fraction_addition** | 9 | 等价分数相加错误 | “equivalent fraction” | 错等价+相加 | 合成等价分数 |
| **Wrong_Operation** | 6 | 运算类型全错 | “wrong operation” | 关键词直接出现 | 混合运算干扰 |

> *可根据真实语料补充/修订关键词与规则阈值。*

---

## 3 建模关键策略

### 3.1 多任务 & 条件触发  
- **Encoder**：Gemma‑7B / LLaMA‑3‑8B + LoRA  
- **Task‑1**：预测 4 类 `Category`  
- **Task‑2**：仅对 _Misconception_ 样本触发 35 类细分类

### 3.2 类别不平衡处理  
- 计算类别权重 `w_i = log(N / n_i)`  
- 二级任务使用 **Focal Loss (γ = 2)**  
- 对频次 < 100 的类别采用 oversample + 合成数据

### 3.3 规则特征拼接  
- 为固定模式类（如 `Multiplying_by_4`, `FlipChange`, `Ignores_zeroes`）抽取 **二进制关键词向量**，与 LLM [`CLS`] 表征拼接

### 3.4 数据增强落地  
1. **Paraphrase** (GPT‑4 / Mistral‑Instruct)：多样化表达  
2. **Counterfactual**：运算符 / 变量互换  
3. **Hard Negatives**：跨类别解释错配  
4. **Pseudo‑label**：高置信样本加权 0.3  
5. **外部迁移**：Eedi、ASSISTments 等

---

## 4 实验路线（6 周示例）

| 周 | 目标 | 里程碑 |
|----|------|--------|
| 1 | 数据清洗 + EDA | 频次分桶、生成 `train_clean.csv` |
| 2 | 增强 V1 | Paraphrase + Counterfactual |
| 3 | LoRA‑模型 V1 | Gemma‑7B 多任务头 |
| 4 | 增强 V2 | Hard Neg + 伪标签 |
| 5 | LoRA‑模型 V2 | 加规则特征 + Focal |
| 6 | 集成 & 提交 | LoRA × n + TF‑IDF + SBERT |

---

### 📌 使用方式
- 将本表 **关键词/规则列** 配置到预处理脚本，生成特征向量  
- 按 **数据增强建议** 批量扩充低频类别  
- 依据 **类别权重** 自动调整损失函数  
- 在训练日志中 **单独监控低频类别召回率**

---

若需 Python 示例脚本（关键词抽取、权重计算等）或数据增强模板，请告知！