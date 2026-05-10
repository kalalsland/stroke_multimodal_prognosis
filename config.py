"""stroke_multimodal_prognosis 项目的中心配置。

数据路径通过 __file__ 动态计算，指向本包内的 data/encoded/ 目录，
无论从哪个工作目录运行都可正确定位，便于他人复现实验。
"""

from pathlib import Path

# 本文件所在目录：stroke_multimodal_prognosis/
_HERE = Path(__file__).parent
_ENCODED = _HERE / "data" / "encoded"

CONFIG = {
    # ------------------------------------------------------------------ 数据
    "clinic_path":     str(_ENCODED / "clinic_data_1.xlsx"),
    "image_feat_path": str(_ENCODED / "image_feature.xlsx"),
    "text_feat_path":  str(_ENCODED / "medical_text_features_by_qianwen_best.xlsx"),

    # ---------------------------------------------------------------- 训练
    "batch_size":         64,
    "epochs":             100,
    "learning_rate":      1e-4,
    "min_lr":             1e-7,
    "lambda_cl":          0.02,
    "lambda_align":       0.20,
    "patience":           30,
    "weight_decay":       0,
    "use_lr_scheduler":   True,
    # "cosine_warmup"（按批次）| "cosine"（按 epoch）| "reduce_on_plateau"
    "scheduler_type":     "cosine_warmup",
    "warmup_epochs":      15,
    "grad_clip":          1.0,

    # ---------------------------------------------------- 数据增强 / 类别平衡
    "use_smote":          True,
    # "none" | "weighted" | "tabular_copy" | "image_copy"
    "smote_method":       "tabular_copy",
    "use_mixup":          True,
    "mixup_alpha":        0.4,
    "use_gaussian_noise": True,

    # ------------------------------------------------------------------ 模型
    "hidden_dim":      512,
    "projection_dim":  256,
    "num_heads":       16,
    "dropout_rate":    0.5,

    # ----------------------------------------------------------------- 输出
    "output_dir":       str(_HERE / "experiments"),
    "save_best_model":  True,

    # ----------------------------------------------------------------- Optuna
    "optuna": {
        "n_trials":  1,
        "timeout":   14400,
        "pruning":   True,
        "direction": "maximize",
    },

    # ----------------------------------------------- 交叉验证 / 数据集划分
    "use_cross_validation": False,
    "n_splits":             5,
    "test_size":            0.2,
    "val_size":             0.2,

    # -------------------------------------------------- 多随机种子评估
    "num_random_seeds": 4,
    "initial_seed":     39,
}
