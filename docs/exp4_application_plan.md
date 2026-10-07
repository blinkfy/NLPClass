# 实验四：09 综合分析应用计划

本次在现有 NLPLab 中完成综合应用，依据用户给定实验四要求及 README 应用路线。未发现独立原始实验四讲义。01～08 模型、实验、页面、协议和环境保护；没有新的分类模型或 Stage7。

## 最小交付

- 薄的 ComprehensiveAnalysisService：调用冻结 HMM-1 Smooth、原 POS/NER 适配器、冻结 C2 ClassificationStage2Service；不复制算法。
- 原文 Unicode code-point 半开区间 `[start,end)` 是唯一对齐基准。分词原样累计，POS 原生词协议、NER 原生字符/WordPiece 分块、C2 原生字符 bigram 各自保留。POS 长文只在已有 token 边界分批调用原适配器，再加原文偏移；不重分词，不截断。
- 默认流程没有依存、IE 或 Qwen。三项更多分析仅显式调用旧服务。
- 课程聚类读取冻结 R2/K=5 与已有二维投影，不拟合、不改已有结果。600 篇历史分配与摘要可展示；新预测/实体联动只对 Train/Validation 白名单全文或自定义文本运行，不读取分类 Test 文件，不产生 Test benchmark。
- 自定义文档调用既有 execute_unsupervised，固定应用配置、仅开放 K；全过程在独立临时内存会话，没有保存到 exp2/exp3，不更新任何 latest。
- 联动矩阵使用实际 C2 prediction 和实际 NER；明确已处理/可处理/总文档数，未处理不能当零实体或预测。簇不是类别，应用预测不是独立泛化估计。
- 系统测试页呈现功能、边界、真实状态和耗时；开发测试和 SHA 记录放开发文档/外部证据。

## 页面与状态

四个一级 Tab：综合分析、批量聚类、结果联动、系统测试。单文本和集合共享当前文档、原文 span、簇选择。初始不自动推理；加载、空输入、纯标点、超长、缺资源、部分成功、空实体、无集合和失效临时会话均显示真实状态。

## API / 统一契约

新命名空间 `/api/v1/comprehensive`，旧 `/api/v1/analyze` 不变。

- POST `/analyze` `{text}` → Document：`id, original_text, lexical:{tokens,pos,entities}, classification, modules, provenance, elapsed_ms, status`。
- POST `/optional` `{text,kind:information_extraction|dependency|summary}` → 原服务结果与来源；保留各自长度限制。
- GET `/course` / POST `/custom` `{documents:[{title?,text}],k}` → Collection：`id,mode,documents,clusters,projection,representation,provenance`。
- GET `/collections/{id}/documents/{document_id}`（`analyze=false|true`）→ 原文/聚类信息及按需 Document 分析。
- POST `/collections/{id}/link` `{document_ids:[...]}`，每请求最多10篇 → 累计真实分类矩阵、实体矩阵与处理范围。
- GET `/status` / POST `/system-tests` → 应用资源状态 / 功能与边界测试结果。

分类只能是冻结 C2 五类新闻域，展示 prediction、runner-up、decision scores/relative margin，不提供模型比较或校准概率。深入解释链接到 06。

## 验收与停止

模型 smoke 与 toy/mock 测试分别记录；验证原文 offset（含空白/emoji）、真实单文本四能力、自定义隔离、冻结聚类读取、两个统计矩阵、drill-down、lazy optional、边界和学生系统测试。1440×1000 / 390×844、console/page/非预期 HTTP errors=0。保护集修改前后 SHA-256 校验。独立功能/视觉评分至少4.3且无 critical；完成核心后停止。
