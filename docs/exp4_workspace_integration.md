# 09 Workspace Final Integration

日期：2026-10-07（Asia/Taipei）。范围只有分类证据、独立应用分类索引、关系到文档的入口；不改变四个工作空间、共享状态、已有算法或正式实验协议。

## 分类证据与原文

直接复用冻结 C2 的 `C2LinearClassifier.explain` / `top_relative_features`，没有新增解释算法。文档画像中显示预测相对第二名的正向 char-bigram 证据；选中后高亮原文 occurrence，并联动现有 token/POS/NER/依存与 Inspector。χ² relevance、TF-IDF、系数差和相对贡献分别显示，不把 χ² 当本文贡献或统计显著性结论。直接贡献/预测权重仅在已有直接解释包含该特征时显示。

原文映射采用 Unicode code-point 索引；逐字符 NFKC/lowercase 映射必须与整文归一化一致，且映射回的原文切片归一化后必须等于特征。组合字符等无法稳定映射时返回空 occurrence，不猜测位置。点击证据时原文只高亮真实字符范围，其他任务的相交词项仍可联动。选中 token/entity 的 incoming/outgoing 依存弧突出，其余弧线弱化；仍只显示实际执行的句段。

独立只读核验：索引中29,763个 occurrence 全部原文切片一致；所有 relative contribution 均等于 TF-IDF × coefficient difference，误差检查无失败。LinearSVC score/margin 不是概率，0中线标明判别超平面参考而非概率阈值。完整算法仍链接06。

## Classification Application Index

路径：`outputs/applications/lab4/classification-index.json`。这是09的 frozen-model inference cache，不是 Exp3 Run、指标或新 Stage，也没有 latest 指针。

显式 `POST /api/v1/comprehensive/classification-index/build` 从既有 Train/Validation 白名单和5000字符上限构造输入；批次仅传 id/text，不传 gold label。独立 worker 在既有 exp2-venv 中一次安全加载冻结 C2/representation，调用原 transform/explain，不 fit、不选参、不计算 Accuracy/F1、不运行 NER。只对实际读取的模型/表示文件按冻结 manifest 校验，没有读取 Test 正文或 Test/metrics artifact 内容。

持久缓存记录正文 hash、模型 Run、源文件 SHA、Train/Validation 文件 SHA、证据、分数、协议标记和缓存 records SHA。首次读取验证全部来源、正文身份与白名单，失败拒绝使用，失败状态不能变成空缓存绕过检查。重复建立索引复用已有条目，本次复验 new_inferences=0。GET课程/快照/文档只是读取已存在结果；不自动运行批量分类或NER。补全索引后，已打开的 partial profile 同步更新真实分类，保留已有词法结果。

实际覆盖：507/507 eligible（600历史聚类文档中93篇不进入本应用分类：Test90 + 超长3篇）。首次应用推理507篇，约9.32秒。

| Cluster | Classification known / eligible | NER known / eligible（本次验收快照） |
| --- | --- | --- |
| 0 | 81 / 81 | 0 / 81 |
| 1 | 94 / 94 | 0 / 94 |
| 2 | 197 / 197 | 0 / 197 |
| 3 | 38 / 38 | 1 / 38 |
| 4 | 97 / 97 | 0 / 97 |

全语料 Sankey 汇总507条真实应用 prediction；五类数量为财经101、教育101、科技102、体育101、娱乐102。这里只是应用预测分布，未与 gold 比较，也不是新独立泛化估计。Cluster 不等于类别；过滤后流宽只代表当前集合中的已知预测数。93篇未知点继续中性着色。

NER继续按需/簇均衡样本运行。本次仅对 `THU-e73f28462c7115d85fbd9a4b` 实际运行基础词法分析，得到2个LOC「中关村」提及；冻结C2复用应用索引，不另推理。NER coverage为1/507，不冒充分类coverage。Entity profile 默认每已NER文档平均提及数，可切换 mention count；无NER覆盖留空，已NER无某类实体则观察零。

## 关系到文档

Sankey下方新增紧凑的当前筛选文档区域，使用同一个 `matchingDocuments` 交集。显示标题、Cluster、真实prediction和已知entity summary；未知NER明确标为未知。点击复用原文档打开流程，自动进入文档工作台并保留Cluster、刷选、类别和实体条件。Reset仅清筛选，保留已知应用结果。

## 验证记录

- Scoped Python suite首轮44 passed/1旧语义回归失败；恢复 `analyzed_count` 的原基础记录计数后，单独补跑该项通过。另新增并通过partial profile升级回归，共46个目标用例均有通过结果（未重复整套）。
- 五个工作台 JS node --check通过；controller最后新增当前profile刷新后再次局部syntax check通过；Python compile通过。无toy/formal training。
- 独立最终功能审查5/5，无Critical；检查了507个应用索引ID、实际贡献、原文映射、缓存来源与跨视图选择。
- 修改前后1547个 protected 路径原始字节SHA-256完全匹配，missing/mismatch为0；允许09修改和新文件另列。没有重定义保护基线。
- 本次单文档真实 smoke：分词/POS/NER成功；全文685字符，2个LOC。显式原文 `[20,112)` 依存输入92字符，50个词的全篇offset正确。后续浏览器复用该真实缓存，不为截图创建模型输出。

实时浏览器29项检查全部通过：真实矩形刷选14个全局ID，其中12篇与Cluster3相交；科技flow为11篇，LOC「中关村」进一步得到1篇匹配文档。点击名单进入同一文档，选中C2「用户」证据联动4处原文位置、词项与Inspector；显式依存请求精确复用已准备的真实结果（cached=true），50条弧渲染，选择实体对应token突出相关弧。返回关联探索仍保留原Cluster/刷选/类别/实体条件。

1440×1000与390×844页面宽度分别为1440与390，无整体横向溢出；console/page/非预期HTTP/request errors均为0。额外截图展示未筛选Sankey总数507、NER未知簇留空与已NER的PER/ORG观察零，并展示移动端1篇当前匹配文档。最终视觉检查4/5、无Critical或必须修复项；保留现有布局，未因信息密度等非阻断意见继续扩大polish。

截图与 `browser-qa-report.json` 放在 `C:/Users/29513/.codex/visualizations/2026/10/07/lab4-integration/browser/`；真实服务结果在同目录 `real/`，冻结审计在 `baseline/`。浏览器最终报告为验收证据，不写入正式实验。

执行边界：Test正文访问=false，gold label使用=false，模型训练/选参=false，Accuracy/F1计算=false，Exp3正式Run/latest写入=false，Stage7=false。01～08、Exp1/2/3正式模型、数据协议、Run/latest、正式预测/指标均未修改。
