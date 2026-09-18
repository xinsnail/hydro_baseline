# Step58 CA-LGM

本步骤只执行冻结协议中的阶段A：Gradient Compatibility Audit。

- 不修改公共 `src`；
- 不改变 HyDRO 优化目标；
- 不读取链路验证边或测试边作为训练监督；
- 分别审计 PGE 邻接生成阶段和合成特征阶段；
- 输出 CSV 和 TXT，不生成 JSON。

公共目录通过相对路径访问：

```text
..\src
..\data
..\outputs\step58_CA_LGM
```

正式 Computers 审计：

```powershell
python run_pipeline.py --dataset computers --rate 0.005 --seed 1 --gpu 0 --epochs 120 --adj-points 0,50,100 --feat-points 10,60,110
```

V1配对120轮实验（HyDRO与邻接阶段LGM，固定权重0.05）：

```powershell
python run_pipeline.py --stage pilot --dataset computers --rate 0.005 --seed 1 --gpu 0 --epochs 120 --eval-interval 60 --eval-runs 10 --lp-weight 0.05
```
