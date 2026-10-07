# 09 连续探索工作台重构协议

范围仅09交互与可视化，保留ComprehensiveAnalysisService、原文Unicode code-point半开跨度和既有模型协议。01～08、旧数据/模型/Run/latest/正式指标/环境冻结。没有新NLP算法或Stage7。

## 统一对象与状态

Corpus → Cluster/刷选/交集 → Document → Entity/Span/Token → 当前任务结果。

共享Workspace包含corpus、documentRef、profiles、summaries、brushIds（null为无刷选，空集合为零选中）、filters（cluster、prediction、entities精确交集）、span、documentFocus、deepResults、view/history。所有视图使用同一选择器，不再分别保存当前文档。换文档只重置局部span，换corpus清理其所属筛选与请求；切Tab、返回保留上下文。未分析、模块未知与真实零分别定义。

## 四个工作空间

1. 文档工作台：原文多层Annotation Canvas为主；token/POS/NER按原文对齐，已执行句级dependency为可选句法层。真实C2 scores、POS/实体类型计数可选中并反向定位；整篇classification score不得伪造局部证据。对象上下文动作显式运行IE/依存/摘要。
2. 语料空间：真实2Dscatter为主，brush/多选、Cluster与prediction着色、文档列表、selection统计联动。未分类点中性。二维只是投影，簇由原高维输入决定。
3. 关联探索：当前已分析selection的Cluster→五类prediction Sankey，流宽为文档数；实体profile与精确实体交集。Context Bar逐个删除条件/重置，Unknown不画假流。
4. 应用评测：原系统功能/边界记录保留；可重放案例填入/执行真实原应用。内部测试/hash仍在开发记录。

## 最小后端扩展

- 已分析summary增加实际entities（type/text/原文span）、模块known标记与POS counts，以支持前端selection统计，不调用新模型。
- GET `/collections/{id}/workspace`：只读集合与已有统计snapshot。
- POST `/collections/{id}/sampling-plan`：可选scope_ids，limit≤10；只从eligible Train/Validation/custom与未完成成员抽样；簇内SHA256固定规则，簇间按已处理数轮询，完全不看gold label。
- GET `/cases`：学生replay输入/预期边界，不预写模型结果。
- `/optional`可传原文start/end，在实际contiguous原文选区执行旧模型，再将原生offset加回全篇；保持dependency300字/60词、summary500字和IE原限额。结果带source_span/完整document_id，不能冒充全文分析。

## 验收

主链必须用真实THUCNews冻结corpus及Train/Validation白名单实际NLP结果走通：Cluster→brush→分布→class/entity交集→文档→token/entity联动→显式句级依存→返回selection仍在。先真实模型capture，再浏览器回放与至少一次实时浏览器交互链，二者分别计数；禁止旧Test正文/新benchmark。纯selector/抽样/offset tests、API/smoke/JS、1440×1000与390×844、console/page/非预期HTTP=0、protected SHA前后全部匹配。独立功能/视觉≥4.3且无Critical后停止，不追加模型或功能。
