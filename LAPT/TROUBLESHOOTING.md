# 常见问题

## CUDA 不可用

先运行 `python verify_environment.py`。若 `CUDA available: False`，检查 NVIDIA 驱动、PyTorch CUDA wheel 和当前 PyCharm interpreter 是否属于同一个环境。

## 缺少 torch_sparse 或 torch_scatter

必须安装与 Python 3.10、PyTorch 2.2 和 CUDA 12.1匹配的 wheel：

```powershell
python -m pip install torch-scatter==2.1.2+pt22cu121 torch-sparse==0.6.18+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

## no file found, please run the reduction method first

表示还没有对应数据集、凝聚率和种子的完整 HyDRO 图。先运行：

```powershell
python run_pipeline.py --dataset computers --rate 0.005 --seeds 1 --stage hydro
```

然后运行 `--stage lapt`。必须确保两阶段的 dataset、rate 和 seed 完全一致。

## 显存不足

先关闭其他 GPU 程序。较大的凝聚率会产生更大的稠密邻接矩阵，例如 Photo `R=0.05` 明显比 `R=0.01` 占用更多显存。不要通过改变模型参数来伪装成同一实验；应记录硬件限制，并分种子运行。

## Windows 中文进度条乱码

日志采用 UTF-8 保存，乱码只影响终端中的进度条字符，不影响计算结果。可在 PyCharm 中启用 UTF-8，或直接查看 `..\outputs\...\logs`。

## 中断后恢复

再次执行相同命令。程序只跳过三个图文件 `adj/feat/label` 均存在且非空的种子。不要把不同参数的结果写入同一输出目录。
