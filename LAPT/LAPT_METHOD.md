# LAPT 方法与完整数据流

HyDRO 首先将原图 `G=(A,X,Y)` 凝聚为 `G_H=(A_H,X_H,Y_H)`。LAPT 不重新训练 HyDRO，而是以该凝聚图为输入，构造面向链路预测的目标拓扑并通过验证集选择混合强度。

对于原图训练节点对 `(u,v)`，链路教师产生关系得分：

```text
s_uv = sigmoid(z_u^T z_v)
```

原节点通过类别约束的软分配矩阵 `S` 映射到凝聚节点，得到原型关系目标：

```text
T_raw = sigmoid(W_s * S^T S + W_c * C)
```

其中 `C` 表示类别一致性先验。为保持 HyDRO 已有边权分布，同时校正重要节点对的相对顺序，LAPT 使用排序校准：

```text
T_rank = P_dist(T_raw; A_H)
```

然后在验证集上选择 `lambda` 和 `rho`：

```text
T(rho) = (1-rho) * T_raw + rho * T_rank
A_LAPT = (1-lambda) * A_H + lambda * T(rho)
```

选择过程只使用验证集 Macro-F1：

```text
(lambda*, rho*) = argmax ValidationMacroF1(A_LAPT(lambda, rho))
```

最终冻结 `A_LAPT(lambda*,rho*)`，再在独立测试节点对上报告 Macro-F1。`run_pipeline.py` 对每个凝聚种子独立执行上述过程，并与同种子的 HyDRO 做配对比较：

```text
Delta_s = F1_LAPT,s - F1_HyDRO,s
```

多种子报告 `mean(Delta_s)`、配对95%置信区间和胜负数。LAPT 的核心价值是把链路任务信号用于凝聚拓扑后处理，同时保留 HyDRO 的节点特征、标签和主体凝聚过程。
