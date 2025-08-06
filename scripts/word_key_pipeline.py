#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
keyword_pipeline.py
----------------------------------
Auto-extract category-specific keywords,
export YAML rules, and give a quick hit-rate report.
"""

import re, os, yaml, argparse
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ---------- 1. 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--csv",   default="../dataset/train.csv", help="train.csv path")
parser.add_argument("--topk",  type=int, default=20,       help="TF-IDF top-k terms")
parser.add_argument("--out",   default="../output/keyword_auto_v1.5.yml", help="YAML output")
parser.add_argument("--min_df",type=int, default=3,        help="min doc freq in TF-IDF")
args = parser.parse_args() if "__main__" in __name__ else parser.parse_args([])

# ---------- 2. 读取数据 ----------
df = pd.read_csv(args.csv).fillna("")
df["StudentExplanation"] = df["StudentExplanation"].str.lower()

# ---------- 3. 自动挖关键词 ----------
def top_k_terms(texts, k=20, min_df=3):
    vec = TfidfVectorizer(stop_words="english",
                          ngram_range=(1, 2),
                          min_df=min_df)
    X = vec.fit_transform(texts)
    idx = X.sum(axis=0).A1.argsort()[-k:][::-1]
    return [vec.get_feature_names_out()[i] for i in idx]

yml_dict = {}

print("▶  挖掘关键词 …")
for mis, g in tqdm(df.groupby("Misconception")):
    terms = top_k_terms(g["StudentExplanation"], args.topk, args.min_df)
    # 过滤纯数字和极短词
    terms = [t for t in terms if re.match(r"[a-z]", t) and len(t) > 2]
    regexes = [fr"\b{re.escape(t)}s?\b" for t in terms[:6]]  # 每类保留 6 条
    yml_dict[mis] = {"regex": regexes}

# 手工逻辑规则可在此处追加
yml_dict["Incomplete"] = {"rule": "len(text.split()) < 18"}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    yaml.dump(yml_dict, f, allow_unicode=True)

print(f"✅  已生成词典：{args.out}")

# ---------- 4. 规则引擎 ----------
def has_any(text, patterns):
    return any(re.search(p, text) for p in patterns)

def rule_positive(exp, ques):
    return "-" in ques and "-" not in exp

def feature_hits(row, cfg):
    text = row["StudentExplanation"]
    question = row["QuestionText"].lower()
    if "regex" in cfg:
        return int(has_any(text, cfg["regex"]))
    elif "rule" in cfg and cfg["rule"] == "len(text.split()) < 18":
        return int(len(text.split()) < 18)
    elif row["Misconception"] == "Positive":
        return int(rule_positive(text, question))
    return 0

# ---------- 5. 快速评估 ----------
print("\n▶  计算规则命中率 …")
stat = []
for mis, cfg in yml_dict.items():
    sub = df[df["Misconception"] == mis]
    if sub.empty:
        continue
    hits = sub.apply(lambda r: feature_hits(r, cfg), axis=1).sum()
    stat.append((mis, len(sub), hits, hits / len(sub)))
stat_df = pd.DataFrame(stat, columns=["Misconception", "Samples", "Hits", "HitRate"])
stat_df = stat_df.sort_values("HitRate", ascending=False)

print(stat_df.to_string(index=False, formatters={"HitRate": "{:.2%}".format}))
stat_df.to_csv("../output/hitrate_v1.5.csv", index=False)
print("✅  已保存命中率表 → hitrate_v1.csv")
# ---------- 6. 输出示例 ----------
"""
Misconception   Samples  Hits HitRate
     Positive       566   412 72.79%
     Additive       929   591 63.61%
   Incomplete      1454   895 61.56%
         ...       ...   ...   ...
"""
