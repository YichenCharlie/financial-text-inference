# Stage 3 ------ BERT 微调与错误分析

## 1. Stage 目标

Stage 1 使用 TF-IDF + Logistic Regression 建立
baseline，发现关键词统计方法可以完成基础分类，但无法很好判断负面事件是否真正属于目标公司。

例如，一篇新闻可能同时出现多个公司：

-   公司 A：受到处罚
-   公司 B：作为投资方被提及

TF-IDF 可能因为看到"处罚""监管"等关键词，将公司 B 也判断为负面。

因此 Stage 3 引入 pretrained BERT，研究 contextual
representation（上下文表示）是否能够提升公司级金融文本分类效果。

------------------------------------------------------------------------

## 2. Label 定义

本项目任务：

> 判断新闻对于指定目标公司是否属于负面事件。

数据转换：

  原始 sentiment_level   Label   含义
  ---------------------- ------- --------------
  -1 / -2 / -3           1       Negative
  0 / 正向类别           0       Non-negative

注意：

Non-negative 不等于 Positive。

它包括： - 中性新闻； - 背景信息； - 与目标公司无负面关系的信息。

因此任务不是：

Positive vs Negative

而是：

Negative event vs Non-negative event

------------------------------------------------------------------------

## 3. FP / FN 定义

本项目将 Negative 作为 positive class。

  真实标签       Label   预测           类型   含义
  -------------- ------- -------------- ------ ----------------------
  Negative       1       Negative       TP     正确识别负面事件
  Negative       1       Non-negative   FN     漏掉负面事件
  Non-negative   0       Negative       FP     误把非负面判断为负面
  Non-negative   0       Non-negative   TN     正确识别非负面

------------------------------------------------------------------------

## 4. Token 长度分析

BERT 最大输入长度：

    512 tokens

训练集统计：

  统计量     Token 数
  -------- ----------
  Mean           1377
  Median         1223
  P95            2809
  Max            3417

超过 512 token 的比例：

91.1%

因此采用：

    max_length = 512

并使用：

    truncation="only_second"

保证目标公司不会被截断。

输入形式：

    [CLS]
    目标公司
    [SEP]
    标题 + 正文
    [SEP]

------------------------------------------------------------------------

## 5. BERT Fine-tuning

模型：

    bert-base-chinese

训练配置：

  参数            值
  --------------- -------
  Batch size      16
  Epoch           3
  Learning rate   2e-5
  Optimizer       AdamW
  Device          CUDA

Pretrained BERT 的 encoder 参数继续训练，同时新增 classification head
进行二分类。

------------------------------------------------------------------------

## 6. 训练结果

训练 loss：

  Epoch       Loss
  ------- --------
  1         0.5027
  2         0.4105
  3         0.3312

Validation Macro-F1：

  Epoch     Macro-F1
  ------- ----------
  1           0.7375
  2           0.7485
  3           0.7548

图片：

![Training Curve](../figures/stage3_bert/training_curve.png)

------------------------------------------------------------------------

## 7. 模型对比

  模型                  Accuracy   Macro-F1
  ------------------- ---------- ----------
  TF-IDF + Logistic       0.7240     0.6211
  BERT Fine-tuning        0.7873     0.7548

![Model Comparison](../figures/stage3_bert/model_comparison.png)

BERT Macro-F1：

    0.6211 → 0.7548

说明上下文表示相比关键词统计可以更好处理公司级金融文本分类。

------------------------------------------------------------------------

## 8. Confusion Matrix 分析

BERT：

    [[411,212],
     [201,1118]]

                        预测 Non-negative   预测 Negative
  ------------------- ------------------- ---------------
  真实 Non-negative                   411             212
  真实 Negative                       201            1118

对比 TF-IDF：

  错误类型     TF-IDF   BERT
  ---------- -------- ------
  FP              426    212
  FN              110    201

BERT 明显减少 False Positive，说明模型减少了关键词导致的误杀。

![Confusion
Matrix](../figures/stage3_bert/confusion_matrix_comparison.png)

------------------------------------------------------------------------

## 9. High Confidence Error Analysis

高置信错误：

  类型                   数量
  -------------------- ------
  High confidence FP       71
  High confidence FN       48

![High Confidence
Error](../figures/stage3_bert/high_confidence_errors.png)

### FP 主要问题

模型能够识别负面词汇，但是无法完全判断：

> 负面事件到底属于哪个实体。

例如：

-   审计机构；
-   投资公司；
-   关联企业；

可能只是新闻背景，但被预测为负面。

------------------------------------------------------------------------

### FN 主要问题

主要包括：

-   公司关系复杂；
-   负面信息位于文章较后位置；
-   512 token 截断导致信息丢失。

------------------------------------------------------------------------

## 10. Stage 3 总结

BERT 相比 TF-IDF：

优势：

-   使用上下文信息；
-   减少关键词导致的误判；
-   提升 Macro-F1。

限制：

1.  Entity-event attribution（实体-事件关系）仍然困难。
2.  长文本超过 512 tokens 时存在信息损失。
3.  公司关系和标签定义可能存在一定复杂性。

------------------------------------------------------------------------

## 11. Future Directions

### 1. Long Document Modeling

可以尝试：

-   Longformer
-   BigBird
-   Hierarchical Transformer

解决长文本限制。

### 2. Explicit Entity Modeling

加入目标实体标记：

    [TARGET]
    公司名称
    [/TARGET]

帮助模型关注目标公司。

### 3. Data Quality Improvement

进一步研究：

-   公司别名映射；
-   Entity linking；
-   标签一致性检查。
