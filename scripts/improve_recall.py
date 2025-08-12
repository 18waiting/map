import re
import yaml
import argparse
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import chi2
from tqdm import tqdm
import csv


def load_data(csv_path, yaml_path):
    df = pd.read_csv(csv_path).fillna("")
    try:
        with open(yaml_path, encoding='utf-8') as f:
            lex = yaml.safe_load(f) or {}
    except FileNotFoundError:
        lex = {}
    df["exp"] = df["StudentExplanation"].str.lower()
    df["ques"] = df["QuestionText"].str.lower()
    return df, lex


def has_any(text, patterns):
    return any(re.search(p, text) for p in patterns)


def rule_hit(row, mis, lex):
    cfg = lex.get(mis, {})
    text = row.exp
    if "regex" in cfg:
        return int(has_any(text, cfg["regex"]))
    if cfg.get("rule") == "len(text.split()) < 8":
        return int(len(text.split()) < 8)
    if mis == "Positive":
        return int('-' in row.ques and '-' not in text)
    return 0


def mine_keywords(neg, pos, topk=30):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(neg + pos)
    y = [1] * len(neg) + [0] * len(pos)
    chi = chi2(X, y)[0]
    topk_idx = chi.argsort()[-topk:][::-1]
    cands = [vec.get_feature_names_out()[i] for i in topk_idx]
    # 清洗
    cands = [w for w in cands if w not in ENGLISH_STOP_WORDS and not re.match(r"^\d", w) and len(w) > 2]
    return cands


def update_yaml(lex, mis, new_words):
    regex_new = [fr"\b{re.escape(w)}s?\b" for w in new_words]
    if "regex" in lex.get(mis, {}):
        lex[mis]["regex"].extend(regex_new)
    else:
        lex[mis] = {"regex": regex_new}
    return lex


def save_miss_samples(df, mis, output_dir="../miss"):
    remain = df[(df["Misconception"] == mis) & (df["hit_new"] == 0)]["exp"]
    output_path = Path(output_dir) / f"{mis}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in remain:
            line = "- " + t[:120]
            print(line)
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../dataset/train.csv")
    parser.add_argument("--yaml", default="../output/keyword_auto_v1.5.yml")
    parser.add_argument("--new_yaml", default="../output/keyword_auto_v2.5.yml")
    parser.add_argument("--hitrate_csv", default="../output/hitrate_v1.5.csv")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    # 读取待改进误解类型
    with open(args.hitrate_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        wrongnames = [row[0] for row in reader if row]

    df, lex = load_data(args.csv, args.yaml)

    for mis in tqdm(wrongnames, desc="Processing"):
        print(f"\n======= 分析误解类型: {mis} =======")

        df["hit"] = df.apply(lambda r: rule_hit(r, mis, lex), axis=1)
        sub = df[df["Misconception"] == mis]

        if sub.empty:
            print(f"❗ 没有找到误解类型 {mis} 的数据，跳过。")
            continue

        hit_rate = sub["hit"].mean()
        print(f"▶ 原始命中率 {mis}: {hit_rate:.2%} ({sub['hit'].sum()}/{len(sub)})")

        neg = sub[sub["hit"] == 0]["exp"].tolist()
        pos = sub[sub["hit"] == 1]["exp"].tolist()

        if not neg:
            print("✅ 全部已命中，无需改进。")
            continue

        # 提取关键词
        cands = mine_keywords(neg, pos, args.topk)
        print(f"▶ 候选关键词 (前 {len(cands)}) → {cands[:15]}")

        # 更新规则
        lex = update_yaml(lex, mis, cands[:8])

        # 再评估
        df["hit_new"] = df.apply(lambda r: rule_hit(r, mis, lex), axis=1)
        new_rate = df[df["Misconception"] == mis]["hit_new"].mean()
        print(f"▶ 新命中率 {mis}: {new_rate:.2%}")

        # 保存未命中样本
        print("▶ 仍未命中的示例:")
        save_miss_samples(df, mis)

    # 保存最终 YAML
    Path(args.new_yaml).write_text(yaml.dump(lex, allow_unicode=True))
    print(f"\n✅ 所有规则已写入 {args.new_yaml}")


if __name__ == "__main__":
    main()
