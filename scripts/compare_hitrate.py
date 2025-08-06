#!/usr/bin/env python
# compare_hitrate.py
"""
比较两个命中率 CSV，输出：
1) 每个类别的 HitRate 提升/下降
2) 加权平均提升 (按 Samples)
"""
import pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv1", default="../output/hitrate_v1.0.csv", help="8版 CSV")
parser.add_argument("--csv2", default="../output/hitrate_v1.1.csv", help="12版 CSV")
args = parser.parse_args() if "__main__" in __name__ else parser.parse_args([])

# 读取
df1 = pd.read_csv(args.csv1)
df2 = pd.read_csv(args.csv2)

# 合并比较
cmp = df1.merge(df2, on="Misconception", suffixes=("_old", "_new"))

# 计算差值
cmp["ΔHitRate"]   = cmp["HitRate_new"]   - cmp["HitRate_old"]
cmp["ΔHits"]      = cmp["Hits_new"]      - cmp["Hits_old"]

# 按提升幅度排序
cmp_sorted = cmp.sort_values("ΔHitRate", ascending=False)

print("\n=== 每类提升情况（Top 15）===")
print(cmp_sorted.head(15).to_string(
        index=False,
        formatters={"HitRate_old": "{:.2%}".format,
                    "HitRate_new": "{:.2%}".format,
                    "ΔHitRate":    "{:+.2%}".format}))

# 计算加权平均提升（整体水平）
total_old = (cmp["HitRate_old"] * cmp["Samples_old"]).sum() / cmp["Samples_old"].sum()
total_new = (cmp["HitRate_new"] * cmp["Samples_new"]).sum() / cmp["Samples_new"].sum()
print(f"\n整体加权平均 HitRate: 旧版 {total_old:.2%} → 新版 {total_new:.2%} "
      f"({total_new-total_old:+.2%})")

# 保存对比文件
cmp_sorted.to_csv("hitrate_compare.csv", index=False)
print("✅  对比结果已保存 → hitrate_compare.csv")
