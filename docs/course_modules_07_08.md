# 07 / 08 课程工作台实现与验收

本次交付属于课程模块，不是文本分类的新 Stage。设计、目标和验收范围见 [实施协议](course_modules_07_08_plan.md)。Exp1、Exp2、Exp3 Stage1–6 的模型、预测、指标、Run、latest、固定数据及已有运行环境保持只读。

## 用户可操作的结果

| 模块 / Tab | 当前实现 | 证据与适用边界 |
| --- | --- | --- |
| 07 交互分析 | 冻结 HMM-1 Smooth 分词，再由 HanLP CTB9 ELECTRA-small biaffine parser 推断 head / relation；词、原文、树、检查器联动 | 固定分词来源 `20260926_130612_32a5e168_smooth`；原文 code-point offsets，前端转换 UTF-16；不使用 demo/latest |
| 07 依存树 | ROOT、带方向和标签的依存弧；多句切换、原文高亮、CoNLL 十列导出 | 单根、无自环、无环、head 范围、根可达与正文覆盖均校验；移动端树图在自身容器横向滚动 |
| 07 算法过程 | 实际 biaffine arc logits、候选 head 与树约束解码的对照 | forward hook 获取真实分数；保留上游 tree/projective 设置；局部候选分数不是概率，局部 argmax 不替代全句 decoder |
| 07 实验评测 | 六个人工固定 Gold-token 教学例句，逐句正确 / 错误弧及双标点协议 UAS / LAS | 小型教学诊断；不是 CTB benchmark，也不是分类 Test |
| 08 BiLSTM-CRF | 冻结字符分词模型的 emission、BMES、转移、Viterbi / Forward 数值、路径和真实 train-loss 曲线 | 固定 Run `20260926_185552_0ffff5d1`；重建路径与第三方 CRF decode 一致，配分函数误差检查；没有续训或补写 Validation 曲线 |
| 08 Transformer | BERT WordPiece / ID / offsets、MASK 候选、层 / head / query / key 联动、真实 attention 和 QK 缩放数值 | 双向 MASK 模型，不是聊天模型；attention 不是因果归因；每层重建 softmax 与真实 attention 核验 |
| 08 大语言模型 | 本地 Qwen 的实际 messages / chat template / IDs、逐 token 输出、候选分布、greedy 或固定 seed 采样、EOS / 预算 | 没有远程 provider；token 分布不是回答正确率；达到 token 预算可能使回答不完整 |
| 08 扩展应用 | 中文摘要、结构化抽取、提示词边界任务，以及模型能力 / 任务协议对照 | 保留原始生成；JSON 语法与字段协议分别检查，不修补输出、不把格式有效等同于事实正确；边界实验不承诺抵抗正文指令 |

## 工程与资源

算法、数值契约位于 `src/nlplab/course_neural/`，服务位于 `src/nlplab/services/course_neural_service.py`；FastAPI 路由位于 `app/api/course_modules.py`。Jinja2 / Vanilla JS 页面继续使用原有四个 Tab，新增文件均限定在课程模块。Web 进程不直接导入 PyTorch；一个独立 worker 只处理一次诊断请求。

复用 `artifacts/runtime/hanlp-venv/Scripts/python.exe`：Python 3.11.16、PyTorch 2.14.0+cpu、Transformers 4.44.2、HanLP 2.1.5。实际 device 为 CPU，Torch threads=4、interop=1；未安装、升级或降级旧运行环境依赖。

新资源由显式准备脚本 `scripts/prepare_course_models.py` 下载到 `artifacts/course_models/`，来源、revision、许可证记录和各文件 SHA-256 保存在 `manifest.json`。该脚本不作为 API 的自动恢复手段。

- HanLP 官方 CTB9 ELECTRA-small archive SHA-256：`dc421a93ba025e494b3b02368614c4aac27194b3b1b6663fa6daac0a7772f256`。资源没有捆绑许可证文本，来源记录保留这一限制和上游链接。
- Qwen2.5-0.5B-Instruct revision：`c89bee90d9f811437d9735454613c35b4a3c4dc8`；safetensors SHA-256：`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`；已保存官方 LICENSE / Apache-2.0 元数据。
- BERT 复用本机现有 safetensors，不下载或复制：revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`，权重 SHA-256 `3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3`。`bert-reference.json` 是当前本地资源的记录，不冒充上游 checksum。加载按固定 revision 和五个文件的内容校验；记录时的绝对路径仅保留为来源。
- BiLSTM 使用已有 metadata / state_dict 指纹检查和 `torch.load(weights_only=True)`，课程 worker 显式绑定冻结 final Run。

API：`GET /api/v1/course/modules/resources`、`POST /api/v1/course/dependency/parse`、`GET /api/v1/course/dependency/evaluation`、`POST /api/v1/course/deep/{bilstm|bert|llm}`。资源 GET 检查来源记录和文件存在；新准备资源还检查 manifest 中的文件大小，不在 GET 中加载神经模型。未缓存模型调用在加载前检查 SHA-256。交互 worker 禁止 INET 连接，同时设置 Hugging Face / Transformers offline 和 local-only；缺失资源返回错误，不自动下载。

依存输入上限为 300 字、6 句、每句 60 词；BiLSTM 为 30 字；BERT 为 120 字且不超过 64 个 WordPiece；Qwen 为 500 字且聊天模板后不超过 512 token，输出预算 1–64 token。超限拒绝而非自动截断。LRU 缓存最多八条；单 worker 串行调用，120 秒服务端超时。编辑输入会清除旧视图并取消浏览器请求，但不宣称取消正在运行的 CPU forward；后续提交等待已有 bounded job 结束，不复用旧输出。

## 真实诊断与产物

- `outputs/course_dependency/20261003_171921_teaching/`：六条预先固定教学 Gold、结果、来源和 manifest。排除标点：22 弧，UAS=100%、LAS=95.4545%；纳入标点：28 弧，UAS=100%、LAS=96.4286%。这些很小的教学结果不能证明模型总体准确率。
- `outputs/course_deep/20261004_013629_diagnostics/`：真实 BiLSTM / BERT / Qwen / 摘要响应及 manifest。这次发布复制此前捕获的真实响应，未增加模型 forward，也未更新旧 latest。
- BiLSTM 样例“中国石油大学”：保存的 Viterbi 路径与模型一致，配分函数重建误差约 `2.20e-6`；BERT 各层 softmax 重建最大误差为 0，行和误差小于 `4e-7`。
- 摘要真实输出为“教师可以使用依存句法来解释句子的结构。”；另一 16-token 回答达到预算时仍未说完，页面保留原状。抽取样例产生 `机构` 而非要求的 `事件`：JSON 可解析但字段协议失败，保留这一失败现象。
- CPU 请求包含启动、导入、资源校验、模型加载及 forward。已捕获的依存 / BiLSTM / BERT 约 3–5.6 秒，Qwen 短输出约 15–22 秒；不是纯 forward 性能 benchmark。早期响应的 `forward_ms` 实际包含 worker 导入 / 校验 / 加载，当前字段更名为 `worker_elapsed_ms`；旧记录不改写。
- 最后验收额外执行五个当前代码下的短篇请求（两次依存、各一次 BiLSTM / BERT / Qwen）：依存 head / relation / offsets 与保存样例完全一致；BiLSTM 原始 emissions 最大差为 0，固定 Run 与数值路径检查通过；BERT 固定 revision / hash、softmax 误差 0、最大行和误差约 `2.67e-7`。Qwen 的提示词边界输入实际生成“已经完成。”并以 EOS 结束，表明该例中它跟从了正文指令；保留原始输出，不宣称抵抗攻击，也不修改提示词后重试替换它。

每个保存结果的 manifest 记录该次执行或发布时的来源。后续服务修复不会回写历史诊断的 source hashes；最终代码与 QA 的证据另行记录。没有模型训练，也没有重算旧实验正式指标或访问旧分类 Test 正文。

## 验收证据

最终验收已完成。独立功能审查评分 **4.6/5**；视觉审查 07 为 **4.6/5**、08 为 **4.4/5**，均无关键未解决问题。审查依据当前源码、实际模型响应、状态验证和桌面 / 移动截图。

测试范围是 `tests/test_course_dependency.py`、`tests/test_course_neural_service.py`、`tests/test_course_output_contracts.py` 及既有 `tests/test_smoke.py`。本轮实际结果为 **46 passed、0 failed、1 warning**（既有 Starlette 弃用警告）；两个 course JS 的 `node --check` 通过。覆盖树与原文契约、双标点指标、CRF exhaustive fixture、资源/hash/固定来源、缓存 / 错误 / 超时 / schema、抽取 JSON / 字段协议。测试使用临时数据或 mock，不把 toy fixture 当真实模型诊断。

浏览器使用既有 `127.0.0.1:8000` 服务，Browser 插件未提供，因此使用现有 Playwright Chromium，不安装依赖或创建重复 server。桌面 1440×1000、移动端 390×844；实际模型响应与 mobile replay 分开记录，replay 只用于验证同一真实响应的渲染与联动。

外部证据位于 `C:/Users/29513/.codex/visualizations/2026/10/01/01a0f750-fe2e-7b83-a6c6-d991ff6ccbab/course07-08/`。其中 `qa_20261004_012457` 保留前几轮真实模型及页面证据；`final-validation` 保存最后定向测试、状态验证和移动端覆盖。最终冻结审计为 `final-validation/protected-hashes-final.json`：1,313 / 1,313 匹配、0 差异；只读取文件字节计算哈希，没有解析旧分类 Test。当前新增模块修改没有写入保护集。


### 计划逐项完成核对

| 计划要求 | 当前状态与验收证据 |
| --- | --- |
| 07 四 Tab 与词 / 弧联动 | 真实解析、依存树、候选 head、六句教学评测均已渲染；`browser-replay.json` 与桌面截图 |
| 多句独立树与全局 offsets | `two-sentence-api.json`：两句各单 ROOT、无环、可达，所有词跨度吻合；第二句 start=9；`two-sentence-browser.json`：切句与弧点击同步原文 / Inspector |
| CoNLL 与双标点评价协议 | 展开后六行、每行十列，内容与 API 一致；独立教学产物与契约测试验证 UAS/LAS |
| BiLSTM 真实过程与历史 | 当前请求固定 final Run、emission 与旧真实诊断差为 0、Viterbi / Forward 数值核验通过；桌面响应与真实训练历史 |
| BERT 内部张量与来源 | 固定 revision / 文件 SHA 加载，实际 QK / attention 数值核验；真实 MASK 输出及 query/key 联动 |
| Qwen 本地生成与解码 | 实际消息、模板、token / 候选分布、EOS 与预算均保留；greedy 和 seed=42 采样参数在 worker / UI 中固定且明确 |
| 四项扩展教学任务 | 摘要、抽取、正文指令边界的真实结果及能力 / 协议对照；语法有效和字段失败分别呈现，不修补失败 |
| 缺资源、空输入、超限、加载与错误 | 首次资源状态、空输入不发请求、超限拒绝、忙碌等待、资源 503 后清除旧输出；故障注入与真实响应分开记录 |
| offline、并发、超时、缓存与 API | worker INET 禁止连接及 local-only；服务串行 / 120 秒 / LRU 8；46 项测试包括超时、并发、路径和内容校验、schema 与旧页面 smoke |
| 桌面 / 移动 | 1440×1000 桌面与 390×844 移动；树图局部滚动，BERT / Qwen 完整页面宽度 390，选择后 Inspector 可见 |
| 输出产物和当前源码 | 两个独立输出目录及 manifest；`api-evidence.json` 当前源码哈希；`completion-data-audit.json` 再核对源码 / 产物 / 指针 |
| 冻结与独立审查 | 1,313 项匹配；功能 4.6、视觉 07=4.6 / 08=4.4，超过 4.3 且无 critical |

最终浏览器记录有 26 个通过的定向检查；console / page / 非预期 HTTP / request failure 均为 0。预期 422 和模拟资源 503 另行记录，不把故障注入当成功推理，也不把真实模型的任务失败当 HTTP 故障。五次当前代码真实诊断与浏览器回放分开计数；回放没有额外模型 forward。此前几轮真实 smoke 保留在原证据目录，不与这五次混计。

移动端第四项“扩展应用”位于可横向滚动的 Tab rail；首屏可能只露部分标题。该可选发现不阻断选择或造成页面横向溢出。本轮结束，不追加装饰或新的实验阶段。


## 2026-10-04：前端工作台改进

本轮继续完善 07 / 08 的展示层，保留四个 Tab、API 和全部模型协议。07 的桌面输入区横向排布，说明区缩高；原文 / 树 / 检查器统一字号与选中层级，候选 head 用真实原始 logit 的正负条表示，中线为 0，条长不是概率。08 统一页面 shell、输入 / 过程 / 对象三区层级，新增处理路径和运行前操作引导；BiLSTM 分词以实际词块展示，完整 Viterbi δ 保留在折叠区；attention cell 使用固定尺寸，矩阵和树继续在自身容器滚动。

新增 `course-view-focus.js` 只维护 UI：重绘后按稳定对象 key 恢复焦点，并保留已展开的数值 / 来源详情。实际 Enter 回放验证依存词项 / 弧 / head、教学样本 / 词项，以及 BiLSTM / BERT 的 token / cell。手机四个 Tab 均完整可点击，页面整体无横向溢出。

本轮 smoke：`tests/test_smoke.py` 为 1 passed（既有 Starlette warning），三个 course JS 的 `node --check` 通过。Chromium 在 1440×1000 / 390×844 下 59 项检查通过；14 次 QA 模型请求均拦截并回放已有真实响应，独立功能审查也使用保存响应，没有调用新模型 worker。console / page / HTTP / request failure 均为 0。

证据目录为 `course07-08/frontend-refinement-20261004/`，主报告 `qa-report.json`，键盘补验 `focus-keyboard-qa.json`，截图补充 `screenshot-scroll-supplement.json`。`presentation-scope-audit.json` 确认后端 / 算法源码与上一轮验收哈希一致，两个课程输出目录的 manifest 产物哈希仍匹配。上方旧验收的源码哈希记录属于当时版本；后续 UI 改动不回写历史 manifest。

独立复核：功能 4.6/5、视觉 4.4/5，无 critical，达到验收线。本轮停止；移动树图和 attention 矩阵允许容器内横向滚动。空状态没有专门截图，其功能已回放验证，视觉证据覆盖范围另行注明。

## 2026-10-06：大语言模型实时流式输出

大语言模型页通过新增 `POST /api/v1/course/deep/llm/stream` 接收 SSE。worker 在 `generate()` 的真实 token 回调中推送累计解码文本，跳过 prompt；中文多字节 token 尚未构成完整字符时不展示末尾临时替换字符。没有分割最终文本来模拟打字。`done` 携带原有完整生成结果、token 候选分布、停止原因、模型来源与运行环境，完成后继续使用既有逐 token Inspector。原 JSON 接口和扩展应用协议保持兼容。

事件为 `status`、`token`、`heartbeat`、`done`、`error`。前端增量 UTF-8 解码并跨网络 chunk 解析 SSE；只有 `done` 才视为完整结果。停止、超时或断连会关闭子进程并释放课程推理锁；界面保留已收到文本，明确标为未完成输出。修改输入会清除旧结果，并防止旧事件更新当前视图。服务器限时 120 秒、生成预算 1–64 token、seed=42 和原解码参数保持不变。

流式请求每次执行真实单篇生成，不用缓存内容伪造流。仅完整成功结果可进入既有内存缓存；不写入模型、正式 Run、latest 或输出 artifact。缓存后的旧 JSON 请求可以读取同一完整诊断结果。

验证：38 项 mock 服务 / API / 输出契约测试通过；SSE parser 的 1-byte UTF-8 / CRLF 分片、中文与 emoji、提前 EOF 与异常处理通过。一次真实 16-token 交互生成确认首 token 先于完成事件，累计解码、完整 response 与页面终态一致；旧 JSON 缓存结果的全部 token 诊断字段一致，无第二次 forward。8 个既有实验 latest 指针哈希前后匹配。桌面 / 手机定向交互复用已有数据和模拟流，不产生新的正式评价。

## 2026-10-06：最终教学性增强

先审计计划、当前页面与实际安装的 HanLP / BERT 源码。项目课程相关目录未发现原始 PDF / PPT / DOCX 讲义，因此本轮依据项目课程方案、已有真实诊断和当前模型实现，不声称核对过原始课堂讲义。结构、四 Tab、模型配置、原提示词与既有交互不重做。

07 的 forward hook 同时复制真实 arc / relation logits；按解码后的最终 head 提取关系分数，使用 parser 的原生 46-label vocabulary。每个词的关系 argmax 都必须与 parser 最终 deprel 相同，未通过即拒绝展示；页面只列 Top-5，明确这些 logits 条件为实际树 head，不能解释成正在点击的其他候选 head 的关系。历史响应缺字段时只展示最终关系，并说明内部 score 未暴露。流程说明补完整 dependent → head 候选 → 全句树约束 → 最终 head → relation → 有标签树。教学 Gold 句和保存 UAS/LAS 没有重新推理或更新，页面明确其不是 CTB 正式 benchmark，也不代表总体准确率。

BERT 三条公式与当前 layer/head/query/key 的真实缩放分数和 attention 绑定。新增真实 Value / head Context 的数值切片和 norm；AV 在完整 head 维度上与 self-attention forward hook 的实际输出核验，未通过不返回 trace。Context 指单个 head 在输出投影之前的加权和，不能用 Value norm 的加权和代替向量计算。保留 attention 非因果归因、MASK softmax 非答案正确率和 BERT 非聊天模型的说明。

BiLSTM 只补充分数术语，没有新增模型功能或训练。Qwen 明确为小型本地 next-token 教学模型，不代表大语言模型能力上限；候选分布以聊天模板及此前生成前缀为条件，显示真实 score，不作为回答正确率或事实可信度。

扩展页复用现有 summary（基础提示）和 boundary（明确边界提示）两个任务端点。同一文本 / 模型 / seed / 解码配置各请求一次，手动点击 A/B 结果切换对应的 token 过程与 Inspector；没有新 API、自动重试或输出修补。输入变更取消旧视图；B 请求失败时保留真实 A 并说明 B 未获得输出。此次实际 A 为“已经完成”，B 为“已经完成。”：两者都跟从了待处理文本中的命令，标点差异没有消除这一现象，不据此宣称防御 prompt injection。

四次单篇真实诊断（依存 1、BERT 1、Qwen A/B 各 1）全部成功；依存 5 个词的最终 head 与 relation Top-1 全一致。BERT 11 个 WordPiece、12 层、12 head、每 head 64D；最大 softmax 和 Context 核验误差均为 0。模型诊断只保存在外部 `course07-08/teaching-final-20261006/real-diagnostics.json`，不写正式 Run / latest。64 项 tests（含 smoke）通过，三个修改 JS 的 node --check 全部退出 0；1 条既有 Starlette/AnyIO 弃用警告。

修改前原保护集 1,313 / 1,313 匹配；加入既有页面、09 和环境记录后共 1,391 项。修改后 1,391 / 1,391 匹配、0 缺失、0 差异；只计算文件字节 SHA-256，不解析旧分类 Test 正文。保护证据为外部 protected-before.json / protected-after.json。没有新增 Stage7、benchmark、远程 provider、环境升降级或训练。

最终 Chromium 验收为 34 项通过：1440×1000 / 390×844，页面整体无横向溢出，树图 / attention 矩阵在自身容器滚动；选择与 Inspector、A/B Enter→Tab→Enter 焦点链稳定，输入变更会清除旧输出。console / page / 非预期 HTTP errors 均为 0。浏览器 POST 全为真实响应回放，没有增加模型 forward。证据 qa.json、真实诊断响应和 5 张最终截图已与 protected-before / after 汇总到本线程的外部 teaching-final-20261006 目录。两次补验只修正测试等待器对瞬时状态和隐藏表单的错误假设，不重跑模型。完成本轮定点增强后停止。
