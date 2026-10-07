# 实验四：09 综合分析应用

完成日期：2026-10-06（Asia/Taipei）。实施计划见 `exp4_application_plan.md`。09 是原文中心的应用编排，不是新的分类 Stage 或算法实验；01～08 页面、模型、数据、Run、latest、指标和输入协议保持冻结。

## 架构与复用

`ComprehensiveAnalysisService` 编排现有能力；`comprehensive/contracts.py` 负责原文跨度校验和描述性计数；`comprehensive/worker.py` 只是现有 runtime 下的调用桥接。

| 能力 | 复用实现 | 当前应用协议 |
| --- | --- | --- |
| 分词 | NLPService / HMMSegmenter | 显式绑定 HMM-1 Smooth `20260926_130612_32a5e168_smooth`；不使用易变 latest |
| POS | 原 HanLPPosTagger | 原 HMM 词输入；在已有 token 边界分批，保留原词和空白，偏移累加回原文 |
| NER | 原 HanLPRecognizer | 原字符输入和已有 WordPiece-aware 分块；不强迫与分词共享 token |
| 五类新闻分类 | ClassificationStage2Service（继承原 ClassificationService） | 冻结 C2 `20261001_133829_68ee50d7`；原 char bigram / χ² / TF-IDF / LinearSVC 协议 |
| 课程聚类 | ClusteringService.result / projection | 固定 R2 `20260930_102533_9c28e666`、K=5；读取保存的二维坐标，不 fit |
| 临时聚类 | execute_unsupervised / TextTFIDF / KMeansClusterer | 自定义正文独立 fit；固定 word uni+bi TF-IDF，min_df=1、max_df=1.0；只开放 K；原执行器固定 seeds42/43，仅使用42分配，不选参 |
| 更多分析 | InformationExtractionService、CourseNeuralService | 抽取 / 依存 / 本地 Qwen 摘要各自显式触发，不在基础分析中自动运行 |

没有复制模型算法或引入新 provider。POS/NER 复用 `hanlp-venv`，C2/临时聚类复用 `exp2-venv`，没有安装或升降级依赖。原适配器在主 Web runtime 不直接加载 PyTorch，由新调用桥接进入已有隔离环境；worker 禁止 INET 连接。

## 原文与数据范围

Document 保留 `original_text`。Python Unicode code-point 的半开区间 `[start,end)` 是 canonical reference；分词原样拼回原文，POS/NER 每个 annotation 必须满足 `original_text[start:end] == annotation.text/word`。前端用 `Array.from` 处理 emoji 的 code-point 索引，不把 JavaScript UTF-16 长度混入模型偏移。自定义正文保留首尾空格和空行，只消耗显式 `---` 分隔行。

分类限定财经、教育、科技、体育、娱乐；score 与 relative margin 不是校准概率，不作为通用分类器。无实体、未完成 NER、没有分类结果分别呈现；没有定义或生成综合评分。

课程模式展示历史冻结的600篇分配、摘要和坐标。这些 R2 表示原本在600篇上拟合，不能称为 Test 未参与。新增应用预测和实体分析只读取 Train420/Validation90 白名单全文（本次507篇在5000字应用上限内），不读取分类 Test 分割正文、不生成新的 Test 预测或评价，不用 gold label 替代分类 prediction。原 C2 推理保留已有 artifact SHA 校验，它会散列历史派生产物字节；没有改变旧校验或 Test 协议。

临时集合和应用缓存只在09内存中存放，失效后提示重新加载；不创建正式 Run，不保存临时模型到 exp2/exp3，不更新旧 latest。旧 `/api/v1/analyze` 保留原语义。新增 API 位于 `/api/v1/comprehensive`，详见计划。

## 四个 Tab

1. 综合分析：一次显式运行分词 / POS / NER / C2；原文、词项、实体和片段 Inspector 联动；模型来源与完整原始数据折叠；更深分类解释链接06。
2. 批量聚类：课程冻结 / 自定义临时两种来源；文档列表、簇筛选、二维散点、代表文档、Top terms 和 Inspector；二维位置仅显示投影，实际分配依据高维表示。
3. 结果联动：真实 Cluster × prediction 和实体提及 × Cluster 矩阵；每个未观察簇行显示未知而非零；展示处理范围；点击矩阵、文档和 span 钻取；批次完成自动只读刷新当前档案。簇不等于类别，计数是应用预测分布而非独立泛化估计。
4. 系统测试：面向实验报告，呈现基础能力、边界、真实状态与耗时；没有把 pytest、SHA、worker、LRU 或内部 Run 作为学生主体。

部分模块失败只降级该模块，保留其他有效输出，并允许重试。旧网络响应不能覆盖新原文/文档；跨度选择绑定文档身份。联动聚合在一致快照上执行。09重型操作采用重入调度门闩，系统测试独占且单飞，测试内部可复用服务，外部请求收到忙碌提示，不把并发抢锁误报为模型失败。

## 真实模型与应用 smoke

真实结果保存在外部 `C:/Users/29513/.codex/visualizations/2026/10/06/lab4-final/real/`，与 toy/mock tests 分开。

- 单文本含 emoji/空白，得到25个原始 token（含空白）、24个POS项、4个实体；实体为微软公司ORG、北京LOC、张明PER、北京大学ORG。C2 prediction 为 science，score_type 为 decision_score。全部 annotation 原文切片匹配。
- 课程冻结600篇 / K5、簇大小94/112/232/46/116，二维位置完整，没有正式 fit。
- 6篇短新闻应用示例临时 K3，每簇2篇，真实坐标、top terms、代表文档齐备；不是正式模型 benchmark。
- 完整六文档分类矩阵（finance/education/science/sports/entertainment）：cluster0 `[0,0,0,2,0]`，cluster1 `[2,0,0,0,0]`，cluster2 `[0,0,2,0,0]`。
- 实体提及矩阵（PER/LOC/ORG）：`[0,1,0]`、`[0,0,0]`、`[0,1,2]`。各簇实际完成NER文档数均为2，所以零行是观察零，不是缺失值。初次只处理2篇时，未观察的簇行均为 null。
- 依存、摘要、IE各显式运行一次，基础流程没有自动调用；5类边界请求均返回422。
- 学生系统测试实际运行15/15通过，包含受控缺资源路径（没有删除或变更模型）；耗时约82.7秒，不当作模型准确率。

按服务调用链：13次综合分析分派中9次新执行、4次复用已有应用结果；9次POS/NER worker与9次C2应用推理；IE另有一次NER，依存和摘要各一次。自定义聚类2次应用执行（用户场景和系统测试各一次），没有正式模型重训或评测。冷启动/本地NER可能需要数十秒，页面显示加载状态；缓存显示原分析耗时与复用说明，不把它冒充本次计算。

## 验收

`tests/test_comprehensive.py` 22 passed，既有 smoke 首轮1 passed；1条既有 Starlette/AnyIO 弃用警告。覆盖原文/emoji、冻结绑定、无自动optional、partial与重试、未知和零计数、隔离custom、Train/Val访问白名单、空联动只读、集合ID预检、快照、锁释放与系统测试独占。独立功能评分4.5/5，无Critical。

Chromium 148在1440×1000 / 390×844完成四Tab、散点/列表/矩阵/原文联动、当前档案刷新、错误/重试与lazy optional验证。浏览器回放已捕获的真实响应，不增加模型执行；故障注入、预期422和正文捕获测试单列。console/page/非预期HTTP与未映射模型POST均为0。移动批量页稳定帧几何为x16～374、页面宽390；先前侧栏过渡时点截图已替换，未修改全站样式。最终视觉4.6/5，无Critical。

保护审计：1539条当前基线记录全部散列，其中1535条保护记录全部匹配、缺失0、保护差异0。许可入口中仅app/main.py与analysis.html改变；旧analysis API和integrated-page.js未变。原数据/模型/Run/latest、01～08页面与环境保护通过。开发证据为外部 `baseline/protected-before.json`、`baseline/protected-after.json`、`audit-after.json`、`browser/qa.json` 与截图。

完成核心后停止：无Stage7、无远程provider、无RAG/聊天机器人/翻译等额外任务，无旧正式训练、预测或指标改写。
