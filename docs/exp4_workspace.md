# 09 连续探索工作台：实施与验收

完成日期：2026-10-07（Asia/Taipei）。协议见 `exp4_workspace_plan.md`；原应用与模型复用记录见 `exp4_application.md`。本次只重构09交互、显示和所需的轻量查询，不新增 NLP 算法、模型或分类 Stage。

## 信息架构与共享对象

四个工作空间为文档工作台、语料空间、关联探索、应用评测；底层 Tab key 保持 single / documents / linkage / testing。

`comprehensive-workspace.js` 维护唯一 Workspace：corpus、documentRef、profiles、summaries、brushIds、filters、span、focus、layers、deepResults/deepTexts、selectionTerms、view/history。所有图、列表和 Inspector 使用同一选择器。Cluster、预测类别、精确实体条件及刷选集合取交集；实体条件可以累积且逐个删除。`brushIds=null` 表示没有区域条件，`[]` 表示确实选中零篇。未分析、模块未知和已观察零值分别处理。

Context Bar 显示语料、Cluster、预测类别、实体、刷选和文档，可以单独移除条件、Reset 或返回前一上下文。切 Tab 不丢失当前文档与筛选；换文档重置局部 span；异步结果按文档身份、原文和 generation 校验，不能覆盖已切换对象。手动编辑原文后不会沿用旧依存、IE 或摘要。

## 主画布与联动

- 文档工作台：canonical 原文多层 annotation canvas，token / POS / NER 均按 Unicode code-point 半开区间对齐。实体点击联动原文、词项、NER、Inspector 和全局实体筛选。POS / 实体组成可反向选择；C2 五类新闻 decision score 点击聚焦整篇，不伪造局部分类证据。
- 依存、IE、摘要作为当前对象的按需上下文动作。选中对象默认扩展至所在句，运行前显示并允许修改实际原文起止位置；无选区使用全文，超出旧模块限额时明确提示。返回原文的依存/IE跨度仅平移偏移，head、relation 与真实模型数值不修改。
- 语料空间：保存的二维投影散点为主，支持 Cluster/真实预测着色、矩形/多边形刷选、文档复选、多条件交集和 Reset。未预测点中性显示。刷选完成后退出绘制并移除阻挡点选的图形覆盖层，共享文档集合仍保留；显式清除刷选才清除条件。
- 当前选择的分类组成、实体提及和 TF-IDF 均值关键词同步更新；未知结果不替换为0。每簇已分析/可分析数量注明全语料范围，当前交集计数另外展示。整簇历史关键词降至折叠区，不冒充当前选择结果。
- 关联探索：Cluster → 五类 prediction Sankey 的流宽为实际已分析文档数；点击流、类别、实体类型或精确实体继续形成交集。实体 profile 统计真实提及；Cluster 不等于类别，跨任务关系不表述为因果。
- 应用评测：保留原功能/边界测试，新增可重放输入案例进入对应工作台；没有预写模型输出。纯标点在前端提示，不触发模型。

`comprehensive-document.js` 与 `comprehensive-corpus.js` 分别渲染文档、语料和关系视图；`comprehensive.js` 编排请求和上下文动作。旧模型服务和算法未复制。

## 最小后端支撑

新增 `/api/v1/comprehensive` 查询：

| 接口 | 作用 |
| --- | --- |
| GET `/collections/{id}/workspace` | 一致的语料与已分析结果快照 |
| POST `/collections/{id}/sampling-plan` | 仅生成确定性抽样计划，不执行模型 |
| POST `/collections/{id}/terms` | 汇总当前选择的已有数值表示，不读取正文或重新 fit |
| GET `/cases` | 返回重放输入与边界预期，不返回预写模型结果 |

既有 `/optional` 增加可选 start/end。没有范围参数的旧请求仍按全文处理，原模块限制保留。

抽样只在 eligible Train/Validation 白名单内，从未完成成员选择；固定 seed 文本为 `lab4-cluster-balanced-v1`，簇内按 `SHA256(seed:cluster_id:document_id)` 排序，簇间优先 completed+allocated 最少者。不用 gold classification label。前端每批5篇，API允许1–10篇。600篇历史聚类坐标/摘要只读展示；在5000字上限内可新分析507篇，分类 Test 正文不能进入新应用推理。

课程 selection terms 只读取冻结 R2 的 vocabulary、ID 和 word_vectors 数值行求均值；这是原聚类表示的描述性统计，不是分类贡献或新训练。自定义集合复用临时聚类已有实际矩阵，不更新正式 Run/latest。

## 真实模型与浏览器主链

本轮真实应用捕获两篇 Cluster 3 白名单文档：

- `THU-e73f28462c7115d85fbd9a4b`：685字符，2个 LOC 提及；
- `THU-dbc63e8ba45cd216b40095bb`：236字符，5个实体提及。

两篇分词、POS、NER和冻结C2分类均成功，预测均为科技，共同实体为 LOC「中关村」。全部 annotation 与原文切片一致。第一次对首篇原文 `[20,112)` 的92字符片段真实运行依存分析，50个词的偏移均映射回全文；没有训练。

实时 Chromium 验收路径：THUCNews600 → Cluster3 → 真实矩形刷选12篇（2篇分类/NER已知、10篇未知）→ 科技 prediction → LOC中关村 → 2篇交集 → 直接点击真实散点自动进入同一文档 → 点击 `[23,26)` 中关村联动原文/POS/NER/Inspector → 预览并显式执行 `[20,112)` 依存动作 → 50条依存弧 → 返回语料空间。

返回时12个刷选ID、Cluster3、科技、LOC中关村均保留。每个 Context chip 也实际单独移除和恢复。浏览器依存请求复用先前同片段的真实结果（cached=true），没有再次模型 forward。测试脚本不伪造结果，也没有执行新的基础分析、临时聚类或系统测试批次。

本轮新真实调用：2次基础应用 pipeline、1次依存 forward；浏览器另有1次精确缓存依存请求。正式模型 fit、超参数搜索、旧正式 Test inference/metrics、Stage7 均为0。

## 验证与冻结审计

- Python：原22项 + 新12项，共34 passed；既有 smoke 1 passed。新测试覆盖确定性抽样/无gold、scope、全原文offset、原分数不变、数值关键词、未知值及旧optional兼容。
- JS：5个工作台文件 node --check 通过；外部纯状态检查覆盖交集、零刷选、unknown、手动文档、返回、原文版本隔离及 Unicode/context 规则。
- 浏览器：1440×1000、390×844；完整主链通过。14个记录HTTP均200，console/page/非预期HTTP errors=0；页面无整体横向溢出，实体token在移动端换行，长依存图在自身容器查看。
- 独立功能审查4.4/5，最终独立视觉审查4.3/5，均无Critical。稀疏单路径Sankey的视觉权重与移动首屏列表长度仍可进一步优化，本轮在主链通过后停止，不继续扩展功能。
- 修改前后对基线1562条路径逐个原始字节SHA-256；1547条 protected 全部匹配，差异/缺失0。15条允许09路径中9变化、6未变。没有重新定义保护基线；本轮新文件另列为09实现，不属于历史冻结资产。

01～08页面与正式算法协议、Exp1/Exp2/Exp3 Stage1～6模型/数据/Run/latest/指标/预测、运行环境均未修改。没有读取分类 Test 分割正文、重算正式指标或创建 Stage7。

证据目录：`C:/Users/29513/.codex/visualizations/2026/10/06/lab4-workspace/`。`real/` 是真实调用资产，`browser/qa.json` 是实时主链，`baseline/audit-after.json` 和 `protected-after.json` 是冻结审计；桌面/移动最终PNG位于 `browser/`。
