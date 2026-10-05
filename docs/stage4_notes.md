# Stage 4 — BERT 内部计算与推理机制

## 本阶段目标与当前进度

本阶段使用 Stage 3 已训练好的 BERT，理解一次 forward 内部实际执行的计算，并为后续 profiling 和推理优化建立基础。当前已完成单条与 batch inference 对照，并使用真实输入和训练后的权重，手动重建第一个 Encoder layer，包括 Embedding、QKV projection、Multi-head attention、输出投影、两次残差与 LayerNorm，以及 FFN。手动结果与模型原生第一层在设置的浮点容差内一致。目前尚未完成全部 12 层到分类输出的手动对照，也未开始算子性能分析。

## 模型整体结构

当前任务的输入是“目标公司 + 标题与正文”，输出是该目标公司的二分类结果：non-negative=0，negative=1。这里的 negative 是二分类中的 positive class，因此 FP 表示真实为 non-negative、预测为 negative，FN 表示真实为 negative、预测为 non-negative。

当前模型包含 12 个 Encoder layers，每层 hidden size 为 768，attention heads 为 12，每个 head 的维度为 64，FFN intermediate size 为 3072。12 层与 12 个 heads 是两个独立的结构配置：层依次处理表示，heads 在同一层中计算不同的注意力分布。不同层拥有各自的参数，head 数量不是模型在推理时自行选择的。

![BERT 模型结构与单层计算流程](../figures/stage4_bert/bert_architecture.png)

图：当前 BERT 分类模型的整体流程、单个 Encoder layer，以及 Multi-head attention 的内部计算。图中省略推理模式下不改变数值的 Dropout。

## 从输入到 Q、K、V

Tokenizer 将文本转换为离散 token IDs，ID 本身不是语义向量。模型根据 ID 查找 word embedding，并加入 position embedding 和 token type embedding，再经过 LayerNorm，得到进入第一层的表示。当前样本的 input IDs shape 为 `(1, 512)`，Embedding 输出为 `(1, 512, 768)`，每一行对应一个 token 的 768 维表示。配对输入中的 512 包括目标公司、新闻和特殊 tokens；`truncation="only_second"` 优先保留公司 segment，截断新闻 segment。

同一个输入表示 X 分别经过三个可训练的 Linear，生成 Q、K、V。需要区分投影层参数与投影结果：投影权重和 bias 是训练学习的参数，在本次推理中保持固定；Q、K、V 是使用这些参数处理当前输入得到的中间结果。每层使用自己的投影参数，不能将 QKV 理解为推理时不断自行更新的三套权重。

Q、K、V 首先分别具有 `(1, 512, 768)` 的 shape，然后重排为 `(1, 12, 512, 64)`。拆分 heads 只重新组织已有数字，不改变数值。每个 head 的 64 维特征来自对完整 768 维输入的投影，并不是只能看到原始 Embedding 的某一段；这 64 个数字也是计算结果，而不是该 head 只有 64 个可训练参数。

## Multi-head attention 的计算

对于某一个 head，当前 token 的 Q 与所有位置的 K 计算点积，除以 `sqrt(64)=8`，加入 mask，再经过 softmax，得到分配给各个来源位置的权重。这些权重用于对对应的 V 做加权求和，得到当前 token 在该 head 中的 64 维输出。Q 和 K 决定信息汇总的比例，V 提供被汇总的内容。

一次性处理所有 token 后，attention weights 的 shape 为 `(1, 12, 512, 512)`。两个 512 分别代表 query 位置和信息来源位置，每一行的权重之和约为 1。多头允许同一个 token 同时使用多套学习得到的信息汇总方式，但没有预先规定各个 head 的语义职责，也不能仅凭某些高权重位置就认定一个 head 专门识别公司或负面事件。

本次观察第一层、第 0 个 head、位置 1 的“深”，权重最高的位置包括“妆”“纪”“有”“世”“化”，其中“妆”的权重约为 0.040683。这只是当前输入中某个 head 的局部计算结果，不等同于最终分类原因。计算输出时全部有效位置都参与加权求和，并非只使用打印出的前 5 个位置。

所有 heads 的输出为 `(1, 12, 512, 64)`，拼接后得到 `(1, 512, 768)`。拼接本身不修改数值，后面的 `768 → 768` 输出投影才对同一个 token 的不同 head 特征进行混合。跨 token 的信息汇总发生在 attention 中，输出投影处理的是汇总后每个 token 自身的特征。

## Residual connection 与 LayerNorm

Attention 输出投影后，模型将结果与这一层的输入 X 相加，再进行 LayerNorm。残差连接允许模型在原有表示上学习补充或修正信息，并为训练提供直接的梯度传播路径。后续层加回的是各自的层输入，而不是始终加回最初的 Embedding。

LayerNorm 分别根据每个 token 的 768 维特征计算均值和方差，先标准化，再乘以可学习的 gamma、加上 beta。不同 token 各自计算统计量，但共用当前 LayerNorm 模块的 gamma、beta；不同 LayerNorm 模块拥有各自的参数。归一化不等同于稀疏化，它不会以大量置零为目标，也不减少向量维度。

本次“深”的表示在标准化后均值约为 0、方差约为 1，经过 gamma 和 beta 后均值为 -0.007669、方差为 0.726860。因此，完整 LayerNorm 输出不要求继续保持零均值、单位方差。当前模块的 gamma、beta shape 均为 `(768,)`，epsilon 为 `1e-12`。

## FFN 与 GELU

Attention 后的表示 H 进入 FFN，依次经过 `768 → 3072` 的 Linear、GELU 和 `3072 → 768` 的 Linear。整个过程不改变 token 数，各个 token 独立使用同一套 FFN 参数。扩维为每个 token 提供更多中间特征来加工已有信息，再压回 768 维以维持层间接口并进行残差相加；四倍扩维是当前模型的架构选择，不是数学上的必然要求。

GELU 是逐元素的非线性激活函数，不改变 shape。若两个 Linear 之间没有非线性，它们可以合并成一个 Linear，中间扩维本身无法带来同样的非线性表达能力。GELU 的作用不是简单地把负数清零，本次观察到 `-1.501522 → -0.100017`、`-2.373180 → -0.020926` 等变化。打印出的前 8 个中间特征均为负值，并不表示全部 3072 个特征都是负值。

FFN 输出经过第二次残差和 LayerNorm 后，才形成完整 Encoder layer 的输出。第二次残差加回 FFN 的输入 H，而不是最开始的 X。第一层前后 shape 都是 `(1, 512, 768)`，但每个 token 的表示已经经过跨位置的信息汇总与非线性特征加工。

## 实验与验证结果

单条与 batch inference 使用已训练的模型，在 GPU、float32、eval 模式下运行。同一条样本单独推理与放入 4 条样本的 batch 后，预测类别一致，最大概率差约为 `5.59e-9`。batch 中各条样本分别计算，不会互相进行 attention。

初步 forward 计时在输入准备和 CPU 到 GPU 传输完成后进行，使用 CUDA Events、10 次预热和 100 次重复。该测量不包含 tokenizer、数据传输及后处理，也不代表完整请求延迟。

| 输入 shape | 平均每 batch forward 耗时 | Forward throughput |
|---|---:|---:|
| `(1, 512)` | 8.351 ms | 119.75 samples/s |
| `(4, 512)` | 27.955 ms | 143.08 samples/s |

这组结果只说明当前配置下 batch=4 的吞吐量高于 batch=1。batch 耗时除以样本数是摊销计算时间，不能当成单条请求的实际延迟，也不能仅凭此判断具体性能瓶颈。

内部计算验证使用 CPU、eval 模式和真实验证样本，目标公司为“深圳市华彩世纪化妆品有限公司”。正式脚本为 `scripts/stage4_verify_bert_layer.py`，对照使用相同输入和 attention mask。

| 验证内容 | 最大绝对差异 | 结果 |
|---|---:|---|
| 单 token attention 与整体矩阵计算 | 1.49e-7 | 数值吻合 |
| 手动第一次 LayerNorm 与模块输出 | 3.81e-6 | 容差内一致 |
| FFN 及第二次残差、LayerNorm，同一 H | 9.54e-7 | 容差内一致 |
| 完整第一层，同一 X | 1.91e-6 | 容差内一致 |

完整第一层的平均绝对差异约为 `2.14e-7`，对照使用 `atol=1e-5、rtol=1e-5`。这验证了当前输入下手动计算与模型原生计算的数值一致性，不代表已穷尽所有输入或完成性能优化。

原生模型使用 `BertSdpaSelfAttention`，手动路径显式构建 attention 矩阵，目的是理解和验证数学过程。不能直接用这段教学实现推断原生实现的耗时、显存占用，或认定 SDPA 一定使用了某个特定融合内核。
