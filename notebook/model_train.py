# =========================
# 0) Install dependencies
# =========================

# =========================
# 1) Imports & HF login
# =========================
import os, torch
from huggingface_hub import login

# Optional: read token from environment or Colab/Kaggle secret
hf_token = os.environ.get("HF_TOKEN", None)
if hf_token:
    login(token=hf_token, add_to_git_credential=True)
else:
    login(add_to_git_credential=True)  # will prompt for token interactively

os.environ["WANDB_DISABLED"] = "true"  # set to "false" if you want W&B logging
device = "cuda" if torch.cuda.is_available() else "cpu"
device
# =========================
# 2) Choose Gemma 3 model
# =========================
# Variants typically include: 1b-it, 13b-it, 27b-it (instruction-tuned text model)
VARIANT = "1b-it"
MODEL_ID = f"google/gemma-3-{VARIANT}"
MODEL_ID
# =========================
# 3) Quantization config (QLoRA-ready)
# =========================
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
bnb_config
# =========================
# 4) Load tokenizer & model (Transformers, no third-party backends)
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM

token = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=token,
    use_fast=True,
    trust_remote_code=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=token,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    use_cache=False,  # enable gradient checkpointing compatibility
    trust_remote_code=False,
)
model.eval();

model.config

# =========================
# 5) Utility: print (approx) model size
# =========================
def print_model_size_in_mb_gb(model):
    param_bytes = 0
    for p in model.parameters():
        try:
            param_bytes += p.numel() * p.element_size()
        except Exception:
            pass
    buf_bytes = 0
    for b in model.buffers():
        try:
            buf_bytes += b.numel() * b.element_size()
        except Exception:
            pass
    size_mb = (param_bytes + buf_bytes) / (1024**2)
    size_gb = size_mb / 1024
    print(f"Approx param+buffer size (unquantized dtype-based): {size_mb:.2f} MB ({size_gb:.2f} GB)")

print_model_size_in_mb_gb(model)

# =========================
# 6) Prepare QLoRA (PEFT) for training
# =========================
from peft import LoraConfig, prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
peft_config

# =========================
# 7) Load your JSONL dataset
#     Expect lines with keys: instruction, input, output
# =========================
from datasets import load_dataset, DatasetDict

jsonl_path = "train_finetune.jsonl"  # update path if needed
raw_ds = load_dataset("json", data_files={"all": jsonl_path})
split = raw_ds["all"].train_test_split(test_size=0.2, seed=42)
ds = DatasetDict(train=split["train"], validation=split["test"])
ds

# =========================
# 8) Formatting function for SFT
# =========================
def formatting_func(example):
    # Single-sample format
    return f"{example['instruction']}\n{example['input']}\n### Answer:\n{example['output']}"

# (Optional) quick sanity check
print(formatting_func(ds["train"][0]))

# =========================
# 9) Define Trainer (TRL SFTTrainer + QLoRA)
# =========================
import transformers
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

max_seq_len = 512
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = transformers.TrainingArguments(
    output_dir="gemma3_qlora_outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    max_steps=800,            # or set num_train_epochs
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",         # set to "wandb" if WANDB_DISABLED=false
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    peft_config=peft_config,
    formatting_func=formatting_func,
    max_seq_length=max_seq_len,
    data_collator=data_collator,
    packing=False,
)
trainer

# =========================
# 10) Train
# =========================
train_result = trainer.train()
trainer.save_state()
train_result

# =========================
# 11) Save LoRA adapter & tokenizer
# =========================
save_dir = "gemma3_lora_adapter"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# =========================
# 12) Inference (generate)
# =========================
model.eval()
tokenizer.pad_token = tokenizer.eos_token

prompt = (
    "instruction : Given the math question, student answer, and their explanation, "
    "determine if there is a misconception and classify it.\n"
    "Valid categories: True_Correct, True_Neither, True_Misconception, "
    "False_Neither, False_Misconception, False_Correct\n"
    "Valid misconceptions (only when applicable): NA, Incomplete, WNB, SwapDividend, Mult, FlipChange, "
    "Irrelevant, Wrong_Fraction, Additive, Not_variable, Adding_terms, Inverse_operation, Inversion, "
    "Duplication, Wrong_Operation, Whole_numbers_larger, Longer_is_bigger, Ignores_zeroes, Shorter_is_bigger, "
    "Wrong_fraction, Adding_across, Denominator-only_change, Incorrect_equivalent_fraction_addition, "
    "Division, Subtraction, Unknowable, Definition, Interior, Positive, Tacking, Wrong_term, Firstterm, "
    "Base_rate, Multiplying_by_4, Certainty, Scale\n"
    "Format your answer as: Category[:Misconception], "
    "input: Question: Calculate 1/2 ÷ 6\n"
    "Answer: 1/3\n"
    "Explanation: if you divide 1/2 by 6 you get 6/3 because 2 goes into 6 3 times and 1 goes into 6 six times, "
    "then simplify to 1/3."
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# =========================
# 13) (Optional) Merge LoRA into base weights for export
# =========================
from peft import PeftModel

merged = PeftModel.from_pretrained(model, save_dir).merge_and_unload()
merged.save_pretrained("gemma3_merged_fp16")
tokenizer.save_pretrained("gemma3_merged_fp16")
