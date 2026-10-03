# Experiment 3：单文档 Decision / Shortcut Diff

本次为 Web 只读交互增强，不创建新的实验 Stage，不训练模型或计算正式指标。

## Error Diff

错误分析保留三栏。正文与 A/B 证据 lane、Decision Inspector 共用 `selectedEvidence`。选中词项或卷积通道后联动原文 occurrence、贡献条及检查器；进入分类预测时保留模型、文档及可用的 feature/channel 选择。切换文档、模型、比较对象或清理条件时重置选择。

C1/C2 直接读取冻结稀疏向量和系数的既有只读决策分解；C0 和 C3 读取保存的解释。线性证据展示 predicted−runner-up margin 贡献，C3 展示预测类 raw-logit 贡献，各 lane 独立缩放。类别输出保留 NB posterior / LinearSVC decision score / TextCNN Softmax 语义，类型不同时提示不可直接比较数值。

共同/独有证据仅指当前 Top 证据在原文可定位区域的重合，不代表完整词表的交集或语义等价。没有保存的完整 HMM token 序列不补造；词级特征仅定位其组成词的原文 occurrence。

## Stage5 单文档消融

新增 GET `/api/v1/experiments/3/visualization/shortcut-document`，只投影冻结 Stage5 flip 文档的五条件预测、解释和删除 spans。它不读取 Test 原文、不执行 cleaning rules，不加载模型或运行推理。所有 offsets 均校验原文子串；基于保存的删除区间重建保留 segments，使用换行显示边界，防止视觉上拼接出原文不存在的片段。

- OO：Original model + Original input。
- OS / OT / OA：同一 Stage4 Train420 original model + 清理输入。
- AA：Stage5 独立重新训练的 clean model + All-clean input。

正文使用保存的 NFKC/lowercase 文本与 Unicode codepoint offsets，在前端转换为 UTF-16 显示坐标。Original 显示冻结 shortcut 区域虚线，清理条件显示实际删除线；没有重新设计清理规则。

Stage5 保存了五条件 Top 特征、C3 Top spans 及全部 384 channel 的最大响应和位置，因此无需新增 cleaned-input diagnostic trace。同一 original model 的通道可以比较前后最大响应；AA filter 编号不与原模型作同一通道对齐。原词与模型实际 `<UNK>` token 分开显示。

Stage5 深跳转继续显示对应条件的保存预测，不使用 Stage3 正式模型推断 Validation 输入。既有 Test C3 完整过程深跳转仍可使用已批准且核验 logits / prediction / pooled / argmax 的单篇 diagnostic trace；不写回 frozen artifact。

## 验证

新增 toy 测试覆盖冻结 spans、Unicode、五条件边界、原模型/AA 身份、Test 文档拒绝、完整 C3 通道与只读 API。浏览器 QA 对所有 POST 作拦截，验证 A/B 联动、条件切换、deep link、文档重置及 desktop/mobile 布局。哈希审计和截图存于工作区外的 `classification-error-diff` 验证目录。
