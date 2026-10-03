# Experiment 3：只读算法联动可视化

本层用于探索已有冻结模型的单篇判别过程，不是新实验 Stage，不训练、不选参、不重新计算正式评价，也不更新任何 Run、latest 或 artifact。

## 共享选择

`classification-state.js` 中的 `selectedEvidence` 只保存在前端内存。C1/C2 保存词项、矩阵列与局部 occurrence；C3 保存 kernel、filter、channel、position 和判别目标。数据与表示、分类预测使用同一个选择。更换 Run、模型、文档或文档来源时重置；切换 Tab 或 Analysis Mode 时保留。

## C1 / C2 决策分解

只读接口 `GET /api/v1/experiments/3/visualization/decision` 接受 `run_id`、`document_id`、`model_id=C1|C2`。它读取并校验冻结 manifest 中的 CSR、vocabulary、IDF 和 LinearSVC 系数，对单行进行代数分解，并验证五类分数与保存记录一致（`atol=rtol=2e-8`）。不调用 fit、transform、predict 或评价指标。

显示的 margin 为：

`intercept_pred - intercept_runner + Σ TF-IDF_j × (w_pred,j - w_runner,j)`。

词项选择联动原文 occurrence、局部 token/bigram、TF-IDF magnitude、相对 margin 贡献和 Inspector。C2 同时读取完整冻结 χ² score 与 DF，区分选中、raw vocabulary 内未选中、词表外三种状态。χ² 显示排名按保存 score 降序、raw column 升序排列；这是展示排名，不是重新选择特征。候选字符按 NFKC/lowercase 后的连续内容片段生成，不跨标点或空白。

完整 C1 HMM token 序列与原始 TF 未保存，因此不反推 TF。若存在 C3 保存的 HMM 编码，则仅作为明确标注的保留 token 预览；word bigram 的两个词分别高亮，不把独立 occurrence 伪装成连续原文短语。

## C3 冻结 diagnostic trace

`GET /api/v1/experiments/3/visualization/c3-trace` 接受 `run_id`、`document_id`、`kernel=3|4|5`、`filter_id=0..127`。只支持冻结 Stage3 Test 编码中的单篇文档，不重新 tokenization。

已有 artifact 保存了每通道 global max、argmax、classifier weight、logits 和预测，但没有完整逐位置 activation。因此接口使用现有 `hanlp-venv` PyTorch CPU runtime，在 `eval()` 和 `inference_mode()` 下执行一次单篇 forward，hook 真实 Conv 输出并取 ReLU，只返回合法窗口。模型 NPZ 使用 `allow_pickle=False` 加载。

返回前必须核对：

- 五类 raw logits 与预测类别匹配冻结记录；
- 384 个 pooled activation 匹配，argmax 完全一致；
- bias 加全部通道贡献还原 logits。

浮点核对容差为 `atol=rtol=2e-5`，不匹配时拒绝展示。校验结果和实际环境随响应返回。完整 trace 仅缓存于服务进程内存（最多三篇），并串行保护缓存创建；切换 filter 重用同次 forward。API 不把任何 trace 或展示派生值写回冻结资产。

Token lane 分开显示原词、原 token 序号、模型 Token ID 和实际 `<UNK>`。head-tail 拼接窗口明确标注，原文只高亮对应的分离片段。Embedding 仅显示实际 shape，不用彩色矩阵代替未保存的向量值。

当前位置可由 track、token、步进或扫描控制选择。非最大位置的 activation 不替换 pooled 值：分类分解始终使用 `max_activation × classifier_weight`。池化通道点击自动定位该通道 argmax；预测类、第二名和两者之差均由真实权重与池化值计算。减少动态效果偏好下禁用播放，手动操作仍可用。

这类片段称为“卷积最大响应片段”，不是 attention 或因果解释。既有正式指标、Stage4/5/6 与错误分析的实验语义保持不变。

## 验证记录

本轮相关 toy/只读回归测试共 14 项通过；toy TextCNN 仅随机初始化，不训练。真实单篇 trace 首次核对的 logits 最大绝对误差为 `4.76837158203125e-7`，pooled、argmax 和 prediction 均匹配。浏览器覆盖两篇冻结文档，切换 filter 重用单篇缓存。

1440×1000 与 390×844 Chromium 验证包括：词项/bigram 联动、直接点击 activation track、pool → argmax、三种目标分解、非最大位置保持池化贡献、跨 Tab 选择保持、换文档重置、touch 步进、播放/暂停和动态 reduced-motion 切换。最终 QA 的 POST、console/page/HTTP errors 均为 0，没有页面级横向溢出。

冻结审计中旧基线的 1,307 个文件 SHA 全部匹配，0 changed、0 missing；Exp3 共 686 个文件，没有新增 Stage7 或更改 latest。另有 6 个旧基线未收录的 Exp2 可视化文件，来源未确认，单独记录且未改动；不把这项库存差异算作本轮模型/指标修改。
