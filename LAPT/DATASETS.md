# 数据集与默认实验设置

公共项目根目录包含 Computers、Photo 和 Cora 的原始文件（`raw`）及当前环境生成的缓存文件（`processed`）。LAPT 脚本通过 `..\data` 使用共享数据，不依赖原电脑上的其他数据路径。

| 数据集 | 节点数 | 本项目常用凝聚率 | 单种子示例 |
|---|---:|---:|---|
| Computers | 13,752 | 0.005 | `python run_pipeline.py --dataset computers --rate 0.005 --seeds 1 --stage all` |
| Photo | 7,650 | 0.01 | `python run_pipeline.py --dataset photo --rate 0.01 --seeds 1 --stage all` |
| Cora | 2,708 | 0.5 | `python run_pipeline.py --dataset cora --rate 0.5 --seeds 1 --stage all` |

Computers 和 Photo 是 Amazon 商品共购图，Cora 是引文网络。三者均作为图凝聚后的链路预测评测数据使用。本文实验中的正负节点对、训练/验证/测试划分以及 LAPT 参数选择由代码按固定协议生成，并非数据目录中预先保存的测试结果。

## 可选凝聚率

可以通过 `--rate` 改变凝聚率，例如：

```powershell
python run_pipeline.py --dataset computers --rate 0.01 --seeds 1 --stage all
python run_pipeline.py --dataset computers --rate 0.02 --seeds 1 --stage all
python run_pipeline.py --dataset photo --rate 0.02 --seeds 1 --stage all
python run_pipeline.py --dataset photo --rate 0.05 --seeds 1 --stage all
python run_pipeline.py --dataset cora --rate 0.25 --seeds 1 --stage all
python run_pipeline.py --dataset cora --rate 1.0 --seeds 1 --stage all
```

不同凝聚率是独立实验，输出目录自动按数据集和凝聚率隔离。不得用测试集选择凝聚率；应预先声明凝聚率，或只用验证集完成选择。

## 跨电脑注意事项

- 推荐使用 `requirements-windows.txt` 中固定的 PyTorch/PyG 版本；
- 首次运行前执行 `python verify_environment.py`；
- 若缓存文件因依赖版本不同无法读取，先备份对应数据集的 `processed` 目录，再让程序根据包内 `raw` 文件重新处理；
- 不要把新电脑运行生成的 `outputs` 当作随包提供的参考结果，参考结果单独位于 `reference_results`。
