# ============ 0) Install/upgrade deps (run if needed) ============
# 此部分保持不变，依赖安装无需修改

# ============ 1) Imports, device, seeds ============
import os, random, math, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 设置随机种子
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# ============ 2) Model choice & constants ============
# 模型路径已更新为本地路径
MODEL_PATH = "/kaggle/input/ganmy-map-model/gemma3-4b-it-transformers"  # 使用离线的本地模型
MAX_SEQ_LEN = 512
GRAD_ACCUM = 4
TRAIN_BS = 2
EVAL_BS = 2
LR = 2e-4
WARMUP_STEPS = 50
MAX_STEPS = 800
OUTPUT_DIR = "gemma3_4b_it_qlora"

# 4-bit quantization (bitsandbytes) 配置
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
bnb_config

# ============ 3) Load tokenizer & model (from offline) ============
# 从本地路径加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
# 禁用缓存以便梯度检查点可用
model.config.use_cache = False
model.eval()

# 打印模型的近似大小
def print_model_size_in_mb_gb(m):
    param_bytes = sum(p.numel()*p.element_size() for p in m.parameters())
    buf_bytes   = sum(b.numel()*b.element_size() for b in m.buffers())
    total_mb = (param_bytes + buf_bytes) / (1024**2)
    print(f"Approx (dtype-based) size: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")

print_model_size_in_mb_gb(model)

# ============ 4) Prepare QLoRA (PEFT) ============
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 设置 LoRA 配置
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 假设目标模块
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ============ 5) Load JSONL dataset (instruction, input, output) ============
JSONL_PATH = "/kaggle/input/train-fine/train_finetune.jsonl"
raw = load_dataset("json", data_files={"train": JSONL_PATH})
split = raw["train"].train_test_split(test_size=0.2, seed=42)
ds = DatasetDict(train=split["train"], validation=split["test"])
ds

# ============ 6) Build supervised samples (mask prompt labels) ============
IGNORE_INDEX = -100
ANSWER_PREFIX = "\n### Answer:\n"

def build_sample(example):
    prompt = f"{example['instruction']}\n{example['input']}{ANSWER_PREFIX}"
    answer = str(example['output']).strip()

    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    ids = prompt_ids + answer_ids
    if len(ids) > tokenizer.model_max_length:
        excess = len(ids) - tokenizer.model_max_length
        keep_prompt_len = max(0, len(prompt_ids) - excess)
        prompt_ids = prompt_ids[-keep_prompt_len:]
        ids = prompt_ids + answer_ids

    prompt_len = len(prompt_ids)
    labels = [IGNORE_INDEX] * prompt_len + ids[prompt_len:]
    attn = [1] * len(ids)

    return {"input_ids": ids, "attention_mask": attn, "labels": labels}

proc = ds.map(build_sample, remove_columns=ds["train"].column_names, desc="Tokenizing + building labels")
proc

# ============ 7) Data collator with dynamic padding ============
class CausalLMDynamicPadCollator:
    def __init__(self, tokenizer, ignore_index=-100, pad_to_multiple_of=None, max_len=None):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.ign = ignore_index
        self.mult = pad_to_multiple_of
        self.max_len = max_len or tokenizer.model_max_length

    def _pad_len(self, cur_max):
        L = min(self.max_len, cur_max)
        if self.mult:
            L = int(math.ceil(L / self.mult) * self.mult)
            L = min(L, self.max_len)
        return L

    def __call__(self, features):
        cur_max = max(len(f["input_ids"]) for f in features)
        tgt_len = self._pad_len(cur_max)

        def pad_to(x, pad_val):
            if len(x) > tgt_len:
                x = x[:tgt_len]
            if len(x) < tgt_len:
                x = x + [pad_val] * (tgt_len - len(x))
            return x

        batch = {}
        batch["input_ids"] = torch.tensor([pad_to(f["input_ids"], self.pad_id) for f in features], dtype=torch.long)
        batch["attention_mask"] = torch.tensor([pad_to(f["attention_mask"], 0) for f in features], dtype=torch.long)
        batch["labels"] = torch.tensor([pad_to(f["labels"], self.ign) for f in features], dtype=torch.long)
        return batch

data_collator = CausalLMDynamicPadCollator(
    tokenizer=tokenizer,
    ignore_index=IGNORE_INDEX,
    pad_to_multiple_of=8,
    max_len=tokenizer.model_max_length
)

# ============ 8) TrainingArguments (safe across versions) ============
from transformers import TrainingArguments
from transformers import Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    save_strategy="steps",
    bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc["train"],
    eval_dataset=proc["validation"],
    data_collator=data_collator,
)

trainer.train()

# ============ 9) Save LoRA adapter + tokenizer ============
ADAPTER_DIR = "gemma3_4b_it_lora_adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# ============ 10) TrainingArguments (safe across versions) ============
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,               # or num_train_epochs=...
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    save_strategy="steps",
    bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",                  # Kaggle-friendly; set "wandb" if needed
)
training_args

# ============ 11) Standard Trainer (processing_class & dynamic pad collator) ============
from transformers import Trainer, TrainingArguments
DEMO_STEPS = 10

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    max_steps=DEMO_STEPS,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    save_strategy="steps",
    bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc["train"],
    eval_dataset=proc["validation"],
    processing_class=tokenizer,   # ← 用这个消除 deprecation 警告
    data_collator=data_collator,  # ← 用我们自定义的动态 padding collator
)

trainer.train();


# ============ 12) Manual evaluate (avoid eval_strategy param issues) ============
metrics = trainer.evaluate()
metrics


# ============ 13) Save LoRA adapter + tokenizer ============
ADAPTER_DIR = "gemma3_4b_it_lora_adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# =========================
# 14) Inference helper  —— 仅替换生成这几行
# =========================
model.eval()
model.config.use_cache = True  # 推理开启 cache 更稳也更快

def generate_once(question, answer, explanation, max_new_tokens=32, do_sample=False):
    instruction = (
        "Given the math question, student answer, and their explanation, determine if there is a "
        "misconception and classify it.\n"
        "Valid categories: True_Correct, True_Neither, True_Misconception, "
        "False_Neither, False_Misconception, False_Correct\n"
        "Valid misconceptions (only when applicable): NA, Incomplete, WNB, SwapDividend, Mult, FlipChange, "
        "Irrelevant, Wrong_Fraction, Additive, Not_variable, Adding_terms, Inverse_operation, Inversion, "
        "Duplication, Wrong_Operation, Whole_numbers_larger, Longer_is_bigger, Ignores_zeroes, Shorter_is_bigger, "
        "Wrong_fraction, Adding_across, Denominator-only_change, Incorrect_equivalent_fraction_addition, "
        "Division, Subtraction, Unknowable, Definition, Interior, Positive, Tacking, Wrong_term, Firstterm, "
        "Base_rate, Multiplying_by_4, Certainty, Scale\n"
        "Format your answer as: Category[:Misconception]"
    )
    prompt = f"{instruction}\nQuestion: {question}\nAnswer: {answer}\nExplanation: {explanation}\n### Answer:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=tokenizer.model_max_length).to(model.device)

    # ⭐ 关键修复：用 autocast 统一到 bfloat16，避免 SDPA dtype 不匹配
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.7 if do_sample else 1.0,
                top_p=0.9 if do_sample else 1.0,
                eos_token_id=tokenizer.eos_token_id,
            )
    else:
        # CPU 路径（没有 autocast）
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# 15-A) 定义标签空间 & 解析/规范化工具
import re

VALID_CATEGORIES = [
    "True_Correct","True_Neither","True_Misconception",
    "False_Neither","False_Misconception","False_Correct"
]

VALID_MISCONCEPTIONS = [
    "NA","Incomplete","WNB","SwapDividend","Mult","FlipChange","Irrelevant",
    "Wrong_Fraction","Additive","Not_variable","Adding_terms","Inverse_operation",
    "Inversion","Duplication","Wrong_Operation","Whole_numbers_larger","Longer_is_bigger",
    "Ignores_zeroes","Shorter_is_bigger","Wrong_fraction","Adding_across",
    "Denominator-only_change","Incorrect_equivalent_fraction_addition","Division",
    "Subtraction","Unknowable","Definition","Interior","Positive","Tacking","Wrong_term",
    "Firstterm","Base_rate","Multiplying_by_4","Certainty","Scale"
]

# 统一一些大小写/变体（如果模型偶尔输出错别字）
CANON_MAP = {
    "wrong_fraction": "Wrong_fraction",
    "wrong_fraction_alt": "Wrong_Fraction",
    "adding_terms": "Adding_terms",
    "inverse_operation": "Inverse_operation",
    "wrong_operation": "Wrong_Operation",
    "not_variable": "Not_variable",
    "firstterm": "Firstterm",
    "base_rate": "Base_rate",
}

def canon_category(x: str) -> str:
    return x if x in VALID_CATEGORIES else None

def canon_miscon(x: str) -> str:
    if x in VALID_MISCONCEPTIONS:
        return x
    # 降噪：统一小写再尝试映射
    xl = x.lower().strip()
    if xl in CANON_MAP:
        return CANON_MAP[xl]
    # 另一种大小写变形
    xus = x.replace("-", "_")
    if xus in VALID_MISCONCEPTIONS:
        return xus
    return None

# 提取 "Category:Misconception" 形式；允许一些空格与换行
PAT = re.compile(r"(True|False)_(Correct|Neither|Misconception)\s*:\s*([A-Za-z_]+)")

def extract_topk_labels(text: str, k: int = 3):
    seen = []
    for m in PAT.finditer(text):
        cat = f"{m.group(1)}_{m.group(2)}"
        mis = m.group(3)
        cat = canon_category(cat)
        mis = canon_miscon(mis)
        if cat is None or mis is None:
            continue
        lbl = f"{cat}:{mis}"
        if lbl not in seen:
            seen.append(lbl)
        if len(seen) >= k:
            break
    return seen

# 兜底：如果解析不到足够标签
FALLBACKS = ["True_Correct:NA", "False_Neither:NA", "False_Misconception:Incomplete"]

def pad_to_k(labels, k=3):
    out = list(labels)
    for fb in FALLBACKS:
        if len(out) >= k: break
        if fb not in out:
            out.append(fb)
    return out[:k]


# 15-B) 加载测试集并构造提示
import pandas as pd

test_path = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
test_df = pd.read_csv(test_path)

def build_instruction():
    return (
        "Given the math question, student answer, and their explanation, determine if there is a "
        "misconception and classify it.\n"
        "Valid categories: True_Correct, True_Neither, True_Misconception, "
        "False_Neither, False_Misconception, False_Correct\n"
        "Valid misconceptions (only when applicable): NA, Incomplete, WNB, SwapDividend, Mult, FlipChange, "
        "Irrelevant, Wrong_Fraction, Additive, Not_variable, Adding_terms, Inverse_operation, Inversion, "
        "Duplication, Wrong_Operation, Whole_numbers_larger, Longer_is_bigger, Ignores_zeroes, Shorter_is_bigger, "
        "Wrong_fraction, Adding_across, Denominator-only_change, Incorrect_equivalent_fraction_addition, "
        "Division, Subtraction, Unknowable, Definition, Interior, Positive, Tacking, Wrong_term, Firstterm, "
        "Base_rate, Multiplying_by_4, Certainty, Scale\n"
        "Format your answer as: Category[:Misconception]"
    )

ANSWER_PREFIX = "\n### Answer:\n"

def row_to_prompt(row):
    instr = build_instruction()
    # 某些测试样本可能 Explanation 为空，做个稳妥填充
    exp = row.get("StudentExplanation", "")
    if pd.isna(exp):
        exp = ""
    return f"{instr}\nQuestion: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nExplanation: {exp}{ANSWER_PREFIX}"

prompts = [row_to_prompt(r) for _, r in test_df.iterrows()]
len(prompts), test_df.head(2)# 15-C) 批量生成（beam search 取前3个序列），含 dtype 修复（autocast bf16）
import math, torch
from tqdm.auto import tqdm

model.eval()
model.config.use_cache = True

def batched(iterable, bs):
    for i in range(0, len(iterable), bs):
        yield iterable[i:i+bs]

# 生成参数：用 beam search 拿 Top-3，避免采样随机性
GEN_K = 3
GEN_PARAMS = dict(
    max_new_tokens=48,
    do_sample=False,
    num_beams=max(4, GEN_K),
    num_return_sequences=GEN_K,
    eos_token_id=tokenizer.eos_token_id,
)

batch_size = 4  # 按显存调
all_pred_top3 = []

for batch_prompts in tqdm(batched(prompts, batch_size), total=math.ceil(len(prompts)/batch_size)):
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(**inputs, **GEN_PARAMS)
    else:
        out = model.generate(**inputs, **GEN_PARAMS)

    # 因为 num_return_sequences=GEN_K，输出数量 = bs * GEN_K
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    # 将每个样本的 GEN_K 个输出分组解析
    for i in range(len(batch_prompts)):
        cand_texts = decoded[i*GEN_K:(i+1)*GEN_K]
        # 从每个候选里抓取 Category[:Misconception]
        labels = []
        for t in cand_texts:
            ext = extract_topk_labels(t, k=1)  # 每个候选抓1个
            if ext:
                for e in ext:
                    if e not in labels:
                        labels.append(e)
        labels = pad_to_k(labels, k=GEN_K)   # 不足3个则补足
        all_pred_top3.append(labels)

len(all_pred_top3), all(len(x)==GEN_K for x in all_pred_top3)



# 15-D) 生成 submission.csv （严格列名/格式）
sub = pd.DataFrame({
    "row_id": test_df["row_id"].values,
    "Category:Misconception": [" ".join(lbls) for lbls in all_pred_top3],
})
sub.head(10)

# 15-E) 保存并展示路径
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path)
print(sub.head(3).to_string(index=False))

