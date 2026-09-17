# LAPT-ONLY

这是共享 HyDRO 底座上的 LAPT 实验步骤。它只提供当前正式流程：

1. HyDRO 训练并选择凝聚图；
2. LAPT 在验证集上选择 lambda 与排序校准强度；
3. 对 HyDRO 和 LAPT 使用相同固定链路预测协议；
4. 输出逐凝聚种子与汇总结果。

历史 Step、RCTC、SCLTC、候选融合和其他实验运行脚本均未包含。
`src/graphslim` 中保留的 models、dataset、coarsening 和 sparsification 是
HyDRO 初始化与训练所需的底层框架依赖，不是额外实验入口。

## 目录

```text
LAPT -ONLY/
├─ data/                           所有步骤共享的数据
├─ outputs/                        所有步骤共享的实验输出
├─ src/                            共享的 HyDRO 基线与 LAPT 核心实现
│  ├─ run/
│  │  ├─ train_all.py              仅运行 HyDRO
│  │  ├─ select_adaptive_rank_lapt.py
│  │  └─ evaluate_fixed_seed_outputs.py
│  └─ graphslim/
│     ├─ condensation/
│     │  ├─ hydro.py
│     │  ├─ paper_lapt.py
│     │  ├─ link_aware_transport.py
│     │  ├─ link_task_data.py
│     │  └─ graph_io.py
│     └─ ...                       必要的模型、数据与优化依赖
└─ LAPT/                           当前可独立更新的实验步骤
   ├─ run_pipeline.py              完整 HyDRO → LAPT → 评测 → 汇总
   ├─ run_photo_r005.py            当前 Photo R=0.05 实验
   ├─ show_progress.py             查看运行进度
   ├─ verify_environment.py        检查环境和共享目录
   └─ verify_bundle.py             检查文件完整性
```

## PyCharm 运行当前 Photo R=0.05

直接运行：

```text
E:\CODE\LAPT -ONLY\LAPT\run_photo_r005.py
```

Script parameters 可以留空，默认参数为：

```text
--dataset photo --rate 0.05 --seeds 1,2,3,4 --stage all --gpu 0
--split-seed 1 --eval-seed 1 --validation-runs 3 --eval-runs 10
```

也可运行通用入口：

```powershell
python "E:\CODE\LAPT -ONLY\LAPT\run_pipeline.py" --dataset photo --rate 0.05 --seeds 1,2,3,4 --stage all --gpu 0 --split-seed 1 --eval-seed 1 --validation-runs 3 --eval-runs 10
```

`--stage all` 依次执行 `hydro`、`lapt`、`evaluate`、`summarize`。
若只运行某一阶段，可分别使用：

```text
--stage hydro
--stage lapt
--stage evaluate
--stage summarize
```

结果保存到：

```text
E:\CODE\LAPT -ONLY\outputs\photo\r_0.05
```

## 迁移到另一台电脑

整体复制 `LAPT -ONLY` 目录，不需要修改源码路径。选择安装了
PyTorch、PyG、Geoopt 和 CUDA 依赖的 Python 解释器，将 Working directory
设为 `LAPT -ONLY\LAPT`。先运行：

```powershell
python verify_environment.py
python verify_bundle.py
```

完整流水线会跳过已经存在的完整凝聚图和评测报告。改变种子集合后，
建议使用新的 `--output-root`，避免旧评测报告与新种子列表混合。

## 统计口径

每个凝聚种子独立产生一个 HyDRO 图和一个验证选择的 LAPT 图。
最终评测固定原图划分与 LP 初始化种子，对每个凝聚图运行 10 次，
使用验证集选择凝聚邻接阈值，报告 Macro-F1。测试边不用于 LAPT 候选选择。
