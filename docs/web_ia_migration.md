# 全课程 Web 迁移记录

## 修改前审计

| 当前 route | 当前 template | 当前 JS / API | 新模块职责 |
| --- | --- | --- | --- |
| `/` | `index.html` | 无业务 JS | 课程能力总览 |
| `/sequence` | `sequence.html` | `sequence.js`、`charts.js`、`algorithm-view.js`；analyze / trace / parameters / metrics / errors | CWS → `/segmentation`；HMM → `/sequence-labeling`；NER → `/tagging`；BiLSTM 模型 → `/deep-learning` |
| `/clustering` | `future.html` | public dataset status | 数据状态与四个聚类功能 Tab；模型未开发 |
| `/classification` | `future.html` | public dataset status | 数据状态与四个分类功能 Tab；模型未开发 |
| `/analysis` | `analysis.html` | 当前是能力说明与分词/NER入口 | `/integrated-analysis`；现有 service 的单文本能力与真实状态 |
| `/annotation/sequence` | `annotation_sequence.html` | `annotation.js` / annotation API | 保留原路由与可选人工 Gold 工作流，不进主导航 |

已有 FastAPI API router 为 `analysis`、`experiment`、`annotation`。原主导航只有五项；移动端是横向导航。重构扩展原 `base.html`，改为单一十项导航和移动 drawer，不增加实验编号树。

## 迁移边界

旧 `/sequence` 重定向至 `/segmentation`；旧 `/analysis` 重定向至 `/integrated-analysis`。现有 clustering/classification/annotation URL 保留。
七模型完整 MSR 表只在中文分词模块展示；NER exact-span 指标只在词性与命名实体模块展示。
模型页面复用相同参数/trace API、chart 与算法过程 renderer，不创建第二套模型。
未实现的 POS、感知机、独立 CRF、IE、聚类/分类模型、依存、Transformer、LLM 显示真实 shell。

## 冻结与证据

开始前已核验原有 264 + extension 90 个冻结文件，并额外记录 359 个模型/Run/数据/协议文件的当前 SHA256。
本次 QA 目录由 `artifacts/cache/course_ia_current.json` 指向；`before.json` 保存修改前模型与 Web source 哈希。
training、official-test inference、scorer、重新评价及数据重选禁止；短用户式 demo 允许。

## 已落地组件

- `app/main.py` 注册十个 canonical route；旧入口307重定向，无重复页面实现。
- `base.html` 保留原配色/字体，新增单层 sidebar、当前模块 topbar、移动 drawer。`layout.js` 统一 Tab/hash、键盘切换、报告模式、drawer 焦点与 Esc。
- `partials/` 复用文本输入、模型选择、分词/NER结果、算法过程、参数、错误与模块 Tab。
- `course-ui.js` 复用 API client、资源选择、code-point 结果/实体展示、trace、只读 errors；四个页面控制器初始化各自职责，没有复制整份 `sequence.js`。
- `algorithm-view.js` 沿用五类 trace renderer，抽取 CRF transition 展示与任务页参数摘要；模型详细参数指向模型模块。图表销毁限定于所属容器，避免跨面板清空。
- `charts.js` 复用π/A/B与训练曲线；BiLSTM 页不重复训练曲线/CRF，emission 留在短文本 trace。完整七模型表只由中文分词调用。
- 历史 `sequence.html/sequence.js` 不再注册或加载，保留历史报告来源引用；可选 annotation 页面/代码/API保持可用。

## API

原 analyze、trace、parameters、metrics、comparison、NER/errors、annotation/public-data APIs 及语义保持。
新增 `GET /api/v1/views/cws/errors`：只读聚合对应模型的 frozen MSR errors，复用 `RunStore`、`extension_status`、`read_errors`；验证模型/Run、分页与类别，拒绝把 MSRA 无 CWS Gold 的 Run 当成分词错误集。不增加模型、评价或写入端点。

## 回归与证据

证据：`outputs/browser/course_ia_20260927_181741/`，包含 before/after SHA256、分阶段浏览器记录、qa_summary、桌面/390px截图和PNG导出。

- 核心环境 full `pytest --verify-only`：125 passed / 35 skipped；神经环境：127 passed / 35 skipped。跳过发生在训练、正式 benchmark/scorer 调用之前；不为测试训练新模型。
- Ruff check、format check、两环境 pip check 通过；保留一条既有 Starlette/AnyIO 弃用提醒。
- Chromium 检查十个 canonical route；重点五模块每个 Tab、独立NER与模型trace、默认HMM1 Smooth、不可用状态、报告模式、PNG导出、旧路由、drawer键盘/焦点。1440px/390px无整页overflow，application console errors=0。
- 初次 HanLP demo 加载用时约43秒，超过原QA等待窗口；请求完成后验收通过，不属于 benchmark。错误tuple显示和drawer焦点问题均已修复复验。
- 原264 + extension90冻结文件 unchanged；额外359模型/Run/预测/指标/scorer/个人子集/数据清单文件 unchanged。
- 旧 final-report asset_manifest 的 Web/README hash 是历史来源快照。保留旧manifest及全部报告资产；测试仅对明确授权变更的展示文件允许历史hash，模型/数据/benchmark仍严格校验。

`training_executed=false`、`official_test_inference_executed=false`、`scorer_executed=false`。
未开发 Lab2/Lab3 或其它新算法。
