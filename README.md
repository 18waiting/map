# Math Misconception Classifier using LLM + LoRA

This repository contains a fine-tuned LLM model using LoRA adapters based on Google’s [`gemma-2b`](https://huggingface.co/google/gemma-2b) to classify student misconceptions in math problems. The model takes a math question, student’s answer, and explanation, then predicts whether there’s a misconception and classifies it into predefined categories.

## 🔍 Problem Statement

Understanding how students think is essential for educators. This project aims to automatically analyze math problem responses and:
- Detect if the answer is correct or incorrect.
- Identify the reasoning behind the student’s explanation.
- Classify specific **misconceptions** if present.

---

## 📚 Dataset Information

This project is based on the dataset from the [**MAP: Charting Student Math Misunderstandings**](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) Kaggle competition. The dataset contains:

* **Student-written responses** to open-ended math questions.
* **Multiple annotations** per response indicating:

  * Whether the response is correct/incorrect.
  * If incorrect, what kind of **misconception** was present (e.g., `SwapDividend`, `Inverse_operation`, `Wrong_Fraction`, etc.).
  * Categorization such as `True_Correct`, `False_Misconception`, `True_Neither`, etc.

## Dataset Files:

* `train.csv`: Annotated training data with explanations and labels
* `test.csv`: Evaluation set with responses to classify
* `sample_submission.csv`: Submission format for competition entries

> For more details, visit the [Kaggle competition page](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/overview).

---

## 🧠 Model Overview

- **Base model**: `google/gemma-2b`
- **Fine-tuning strategy**: Parameter-efficient fine-tuning (PEFT) using **LoRA**
- **Quantization**: 4-bit using `bitsandbytes` for memory efficiency
- **Trainer**: `trl.SFTTrainer` from Hugging Face

### 🏷️ Output Format
```

Category\[:Misconception]

```

Example:
```

False_Misconception\:SwapDividend

```

## 📁 Project Structure

```

math_compe/
│
├── artifacts/
│   └── checkpoint-900/         # LoRA fine-tuned weights
│
├── dataset/
│   ├── train.csv               # Raw training data
│   ├── test.csv                # Evaluation data
│   ├── sample_submission.csv   # Submission format
│   └── train_finetune.jsonl    # Instruction-formatted dataset
│
├── notebook/
│   ├── exploration.ipynb       # Data cleaning, analysis, preprocessing
│   └── load_model.ipynb        # Inference with fine-tuned model

```

## 🏗️ Pipeline Summary

### 1. Data Preparation (`exploration.ipynb`)
- Clean LaTeX equations and student explanations
- Normalize math expressions (e.g. `\frac{3}{4}` → `3/4`)
- Construct input-output pairs in instruction format

### 2. Fine-tuning (`SFTTrainer`)
- Tokenized instruction+input+output using `AutoTokenizer`
- Configured LoRA on key transformer layers
- Logged training with Weights & Biases (`wandb`)

### 3. Evaluation
- Visualized training and validation loss
- Tracked token-level accuracy on `wandb`
- Performed manual inference tests post-finetuning

## 🧪 Example Inference

```

Input:
Question: Calculate 1/2 ÷ 6
Answer: 1/3
Explanation: dividing 1/2 by 6 gives 6/3 because 2 goes into 6 three times...

Output:
False_Misconception\:SwapDividend

```

## 🚀 How to Use

1. Clone this repository
2. Install dependencies (see `requirements.txt`)
3. Load and run the inference file `load_model.ipynb` using the fine-tuned weights in `artifacts/checkpoint-900/`

## 🔧 Dependencies

* `transformers`
* `peft`
* `trl`
* `bitsandbytes`
* `datasets`
* `wandb`

## 📊 Tracking

Training progress and metrics are available at:

**WandB Run**: [`finetune-gemma-math-add-patter-answer`](https://wandb.ai/your-profile/finetune-gemma-math-add-patter-answer)

## ✍️ Author

**Muhammad Fikry Rizal** – AI Curriculum Developer, passionate about LLMs and education technology.

---

📬 Feel free to contribute or raise an issue!