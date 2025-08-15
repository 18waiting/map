# ============ 0) Install/upgrade deps (run if needed) ============

# ============ 1) Imports, login, device, seeds ============
import os, random, math, torch, numpy as np
from huggingface_hub import login

# HF token (set in Kaggle Secrets → "HF_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=True)
else:
    login(add_to_git_credential=True)  # will prompt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# ============ 2) Model choice & constants ============
MODEL_ID = "google/gemma-3-4b-it"   # instruction-tuned
MAX_SEQ_LEN = 512                   # adjust per VRAM
GRAD_ACCUM = 4
TRAIN_BS = 2
EVAL_BS  = 2
LR = 2e-4
WARMUP_STEPS = 50
MAX_STEPS = 800                     # or use num_train_epochs
OUTPUT_DIR = "gemma3_4b_it_qlora"

# ============ 3) 4-bit quantization (bitsandbytes) ============
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
bnb_config

# =========================
# 4) Load tokenizer & model (pure Transformers) — FIXED (no use_cache kw)
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    use_fast=True,
    trust_remote_code=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# 以我们设的上限和模型/分词器自身的限制二者取 min
try:
    max_len_from_tok = getattr(tokenizer, "model_max_length", None)
    if max_len_from_tok is None or max_len_from_tok > 10**6:
        max_len_from_tok = MAX_SEQ_LEN
    tokenizer.model_max_length = min(MAX_SEQ_LEN, max_len_from_tok)
except Exception:
    tokenizer.model_max_length = MAX_SEQ_LEN

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
# ⚠️ 在加载后再关 use_cache，以便梯度检查点可用
model.config.use_cache = False
model.eval();

# ============ 5) Print approximate size ============
def print_model_size_in_mb_gb(m):
    param_bytes = sum(p.numel()*p.element_size() for p in m.parameters())
    buf_bytes   = sum(b.numel()*b.element_size() for b in m.buffers())
    total_mb = (param_bytes + buf_bytes) / (1024**2)
    print(f"Approx (dtype-based) size: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
print_model_size_in_mb_gb(model)


# ============ 6) Prepare QLoRA (PEFT) ============
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Robustly guess LoRA target modules across versions
def guess_lora_targets(m):
    # Common endings in Gemma-type models
    candidates = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    found = set()
    names = [n for n,_ in m.named_modules()]
    for n in names:
        for c in candidates:
            if n.endswith(c):
                found.add(c)
    if not found:
        # Fallback to wq/wk/wv/wo patterns
        alts = ["wq","wk","wv","wo"]
        for n in names:
            for c in alts:
                if n.endswith(c):
                    found.add(c)
        if found:
            return list(found)
    return list(found) if found else candidates  # if nothing matched, try defaults

target_modules = guess_lora_targets(model)
print("LoRA target modules:", target_modules)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters();

# ============ 7) Load JSONL dataset (instruction, input, output) ============
from datasets import load_dataset, DatasetDict

JSONL_PATH = "/kaggle/input/train-fine/train_finetune.jsonl"

# 把 "all" 改成 "train"
raw = load_dataset("json", data_files={"train": JSONL_PATH})
split = raw["train"].train_test_split(test_size=0.2, seed=42)
ds = DatasetDict(train=split["train"], validation=split["test"])
ds

# ============ 8) Build supervised samples (mask prompt labels) ============
IGNORE_INDEX = -100
ANSWER_PREFIX = "\n### Answer:\n"

def build_sample(example):
    # 1) 拼提示与答案
    prompt = f"{example['instruction']}\n{example['input']}{ANSWER_PREFIX}"
    answer = str(example['output']).strip()

    # 2) 分开编码（不加 special tokens）
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    # 3) 拼接后若超长：优先从“左侧的 prompt 部分”裁掉，尽量保留答案
    ids = prompt_ids + answer_ids
    if len(ids) > tokenizer.model_max_length:
        excess = len(ids) - tokenizer.model_max_length  # 需要裁掉的长度
        # 仅从 prompt 裁（如 excess 超过 prompt_len，就把 prompt 清空）
        keep_prompt_len = max(0, len(prompt_ids) - excess)
        prompt_ids = prompt_ids[-keep_prompt_len:]
        ids = prompt_ids + answer_ids

    # 4) 重新计算 prompt_len，并构造 labels（prompt 区域为 -100）
    prompt_len = len(prompt_ids)
    labels = [IGNORE_INDEX] * prompt_len + ids[prompt_len:]
    attn = [1] * len(ids)

    return {"input_ids": ids, "attention_mask": attn, "labels": labels}

proc = ds.map(build_sample, remove_columns=ds["train"].column_names, desc="Tokenizing + building labels")
proc

# ============ 9) Data collator with dynamic padding ============
import torch

class CausalLMDynamicPadCollator:
    def __init__(self, tokenizer, ignore_index=-100, pad_to_multiple_of=None, max_len=None):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.ign = ignore_index
        self.mult = pad_to_multiple_of  # e.g., 8 for tensor cores; 可为 None
        self.max_len = max_len or tokenizer.model_max_length

    def _pad_len(self, cur_max):
        L = min(self.max_len, cur_max)
        if self.mult:
            # 向上对齐到 mult 的倍数（不超过 max_len）
            L = int(math.ceil(L / self.mult) * self.mult)
            L = min(L, self.max_len)
        return L

    def __call__(self, features):
        # 1) 本 batch 的最大长度
        cur_max = max(len(f["input_ids"]) for f in features)
        tgt_len = self._pad_len(cur_max)

        # 2) 截断到 tgt_len，并做右侧 padding
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
    pad_to_multiple_of=8,              # 显存足够时建议 8；要绝对保守改为 None
    max_len=tokenizer.model_max_length # 与上面保持一致
)

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
    processing_class=tokenizer,   # ← 用这个消除 deprecation 警告
    data_collator=data_collator,  # ← 用我们自定义的动态 padding collator
)

trainer.train();

