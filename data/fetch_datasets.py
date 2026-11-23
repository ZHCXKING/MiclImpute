# %%
"""
UCI 基准数据集一键下载器
下载 10 个标准表格缺失值填补基准数据集，自动清洗并保存为 parquet + JSON 格式

输出结构：data/<数据集名称>/
    ├── AllData.parquet     (特征 X + 目标 y 合并，保留原始缺失值)
    ├── DropNaData.parquet  (特征 X + 目标 y 合并，去除含缺失行的完整样本)
    └── Metadata.json       (数据集描述、特征类型、目标信息)

使用方式：
    # 加载数据（X 和 y 已合并）
    df = pd.read_parquet("data/Adult/AllData.parquet")
    X = df.drop(columns=[meta["target"]["name"]])
    y = df[meta["target"]["name"]]
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# %%
# 10 个标准 UCI 表格基准数据集（6 个混合类型 + 4 个纯数值）
# 格式：(UCI ID, 任务类型, 领域描述)
# 任务类型: binary (二分类) / multiclass (多分类) / regression (回归)
UCI_DATASETS = {
    "Adult": (2, "binary", "人口普查收入预测"),
    "Credit": (350, "binary", "信用卡违约预测"),
    "Bank": (222, "binary", "银行营销认购预测"),
    "Heart": (45, "binary", "心脏病诊断"),
    "Australian": (143, "binary", "澳大利亚信贷审批"),
    "Abalone": (1, "regression", "鲍鱼年龄（环数）回归"),
    "Letter": (59, "multiclass", "字母图像特征识别（26 类）"),
    "Breast": (17, "binary", "乳腺癌诊断"),
    "Wine": (186, "regression", "葡萄酒质量评分回归"),
    "Spam": (94, "binary", "垃圾邮件识别"),
}

# 整数编码类别变量的显式覆盖（如信用卡数据中的性别、教育程度等整数类别）
CATEGORICAL_OVERRIDES = {
    "Credit": ["X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"],
    "Heart": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    "Australian": ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"],
}

# %%
def download_dataset(name: str, uci_id: int, task: str, description: str, data_root: Path):
    """下载并处理单个数据集"""
    print(f"[{name}] 正在获取 UCI ID={uci_id} ...")
    dataset = fetch_ucirepo(id=uci_id)
    X = dataset.data.features.copy()
    
    # 读取目标变量
    y = None
    target_name = None
    target_is_categorical = (task in ("binary", "multiclass"))
    target_n_classes = None
    
    if hasattr(dataset.data, "targets") and dataset.data.targets is not None:
        y = dataset.data.targets.copy()
        if isinstance(y, pd.DataFrame):
            target_name = y.columns[0]
            y = y.iloc[:, 0]
        else:
            target_name = "target"
            y = pd.Series(y, name=target_name)
        
        if target_is_categorical:
            # 先尝试转为数值类型
            y_numeric = pd.to_numeric(y, errors="coerce")
            # 如果转换成功且没有丢失太多值，直接使用数值
            if y_numeric.notna().sum() > len(y) * 0.5:
                y = y_numeric
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, name=target_name)
            else:
                # 转换失败，说明是字符串标签，需要 factorize
                # 先清理字符串：去除首尾空格和末尾的点号
                y_str = y.astype(str).str.strip().str.rstrip('.')
                codes, uniques = pd.factorize(y_str, sort=True)
                y = pd.Series(codes, name=target_name)
            target_n_classes = int(y.nunique())
        else:
            y = pd.to_numeric(y, errors="coerce").astype("float64")
            y = pd.Series(y, name=target_name)
    
    # 获取变量类型元数据并排除 ID 列
    variables_df = getattr(dataset, "variables", None)
    if variables_df is not None and {"name", "role"}.issubset(variables_df.columns):
        id_cols = variables_df.loc[variables_df["role"].str.lower() == "id", "name"].tolist()
        X = X.drop(columns=id_cols, errors="ignore")
        var_types = dict(zip(variables_df["name"], variables_df["type"]))
    else:
        var_types = {}
    
    # 清洗特征数据
    override_cats = set(CATEGORICAL_OVERRIDES.get(name, []))
    categorical_cols, numerical_cols = [], []
    df_clean = pd.DataFrame(index=X.index)
    
    for col in X.columns:
        s = X[col].replace(["?", "? ", "None", "null", "nan", "NaN", ""], np.nan)
        uci_type = str(var_types.get(col, "")).lower()
        is_cat = (
            col in override_cats or 
            uci_type in ("categorical", "binary") or 
            s.dtype == object or 
            isinstance(s.dtype, pd.CategoricalDtype) or 
            str(s.dtype) in ("string", "boolean", "bool")
        )
        
        if is_cat:
            categorical_cols.append(col)
            s_str = s.dropna().astype(str).str.strip()
            codes = pd.Series(np.nan, index=s.index, dtype="float64")
            if len(s_str) > 0:
                fact_codes, _ = pd.factorize(s_str, sort=True)
                codes.loc[s_str.index] = fact_codes.astype(float)
            df_clean[col] = codes
        else:
            numerical_cols.append(col)
            df_clean[col] = pd.to_numeric(s, errors="coerce").astype("float64")
    
    # 合并特征和目标变量
    df_all = df_clean.reset_index(drop=True)
    if y is not None:
        y = y.reset_index(drop=True)
        df_all[target_name] = y
    
    # 去除含缺失值的行
    if y is not None:
        dropna_mask = df_clean.notna().all(axis=1) & y.notna()
    else:
        dropna_mask = df_clean.notna().all(axis=1)
    df_dropna = df_all[dropna_mask].reset_index(drop=True)
    
    # 保存数据
    out_dir = data_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_dir / "AllData.parquet", index=False)
    df_dropna.to_parquet(out_dir / "DropNaData.parquet", index=False)
    
    # 计算统计信息
    n_samples = len(df_all)
    n_features = len(df_clean.columns)  # 不包括目标变量
    n_categorical = len(categorical_cols)
    n_numerical = len(numerical_cols)
    
    n_missing = df_clean.isna().sum().sum()  # 只统计特征列的缺失
    missing_rate = float(n_missing / (n_samples * n_features)) if n_samples * n_features > 0 else 0.0
    missing_cols = int((df_clean.isna().sum() > 0).sum())
    
    cardinalities = {col: int(df_all[col].nunique()) for col in categorical_cols}
    
    size_category = "small" if n_samples < 1000 else ("medium" if n_samples < 10000 else "large")
    
    # 构建目标变量信息
    target_info = None
    if y is not None:
        target_info = {
            "name": target_name,
            "type": task,
            "is_categorical": target_is_categorical,
        }
        if target_is_categorical and target_n_classes is not None:
            target_info["n_classes"] = target_n_classes
            target_info["class_distribution"] = {int(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()}
        else:
            target_info.update({
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            })
    
    # 生成 Metadata
    columns_schema = {}
    for col in df_all.columns:
        if col == target_name:
            columns_schema[col] = {
                "sdtype": "categorical" if target_is_categorical else "numerical",
                "role": "target"
            }
        elif col in categorical_cols:
            columns_schema[col] = {"sdtype": "categorical"}
        else:
            columns_schema[col] = {"sdtype": "numerical", "computer_representation": "Float"}
    
    metadata = {
        "name": name,
        "uci_id": uci_id,
        "uci_url": f"https://archive.ics.uci.edu/dataset/{uci_id}",
        "task": task,
        "description": description,
        "size_category": size_category,
        "n_samples": n_samples,
        "n_samples_complete": len(df_dropna),
        "n_features": n_features,
        "n_categorical": n_categorical,
        "n_numerical": n_numerical,
        "n_missing": int(n_missing),
        "missing_rate": missing_rate,
        "missing_cols": missing_cols,
        "categorical": categorical_cols,
        "categorical_cardinalities": cardinalities,
        "numerical": numerical_cols,
        "target": target_info,
        "tables": {"table": {"columns": columns_schema}},
        "relationships": [],
        "METADATA_SPEC_VERSION": "V1",
    }
    
    with open(out_dir / "Metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    # 打印摘要
    target_summary = ""
    if target_info:
        if target_is_categorical:
            target_summary = f" | 目标={target_name}({target_n_classes}类)"
        else:
            target_summary = f" | 目标={target_name}(回归)"
    
    print(
        f"[{name}] 已保存至 {out_dir} | "
        f"样本={n_samples}, 特征={n_features} ({n_categorical}类+{n_numerical}数) | "
        f"任务={task} | 缺失率={missing_rate:.2%}{target_summary}"
    )

# %%
def main():
    """主函数：遍历所有数据集并下载"""
    data_root = Path(__file__).resolve().parent
    for name, (uci_id, task, description) in UCI_DATASETS.items():
        try:
            download_dataset(name, uci_id, task, description, data_root)
        except Exception as exc:
            print(f"[{name}] 下载失败 (UCI ID={uci_id}): {exc}")

# %%
if __name__ == "__main__":
    main()
