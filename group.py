import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def stratified_split(df, test_size=0.2, random_state=42):
    """
    分层划分：TJ -> train/inter，非TJ -> outer
    必要列: name, hospital, grade, event, time
    """
    tj_df    = df[df['hospital'] == 'TJ'].copy().reset_index(drop=True)
    outer_df = df[df['hospital'] != 'TJ'].copy().reset_index(drop=True)

    print(f"TJ: {len(tj_df)}, 外部: {len(outer_df)}")

    # 联合分层标签
    tj_df['strata'] = tj_df['grade'].astype(str) + '_' + tj_df['event'].astype(str)

    # 分层划分 train / inter
    train_df, inter_df = train_test_split(
        tj_df,
        test_size=test_size,
        stratify=tj_df['strata'],
        random_state=random_state
    )

    cols = ['name', 'hospital', 'grade', 'event', 'time']

    train_df = train_df[cols].copy().reset_index(drop=True)
    inter_df = inter_df[cols].copy().reset_index(drop=True)
    outer_df = outer_df[cols].copy().reset_index(drop=True)

    # ── 分布验证 ────────────────────────────────────────────
    print(f"\n{'='*50}")
    for name, subset in [('Train', train_df), ('Inter', inter_df), ('Outer', outer_df)]:
        print(f"\n{name} (n={len(subset)})")
        print(f"  Grade : {dict(subset['grade'].value_counts().sort_index())}")
        print(f"  Event : {dict(subset['event'].value_counts().sort_index())}")
        print(f"  Time  : mean={subset['time'].mean():.2f}, std={subset['time'].std():.2f}")
    print(f"{'='*50}")

    return train_df, inter_df, outer_df


# ── 主程序 ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_excel("./data/modeling_data.xlsx")

    train_df, inter_df, outer_df = stratified_split(df, test_size=0.3, random_state=42)

    os.makedirs('./data', exist_ok=True)

    # inter.csv 包含 train/inter 标记
    train_df['split'] = 'train'
    inter_df['split'] = 'inter'
    pd.concat([train_df, inter_df]).reset_index(drop=True).to_csv('./data/inter.csv', index=False)
    outer_df.to_csv('./data/outer.csv', index=False)

    print(f"\n✓ inter.csv : train={len(train_df)}, inter={len(inter_df)}, 共{len(train_df)+len(inter_df)}行")
    print(f"✓ outer.csv : {len(outer_df)} 行")