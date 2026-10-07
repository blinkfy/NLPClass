# 07 / 08：目标、实现与验收协议

这是连续课程平台的新增模块交付计划，不属于 Experiment 3 的新 Stage。旧 Exp1/Exp2/Exp3 模型、Run、指标、预测、latest 与数据保持冻结。采用 `src/nlplab` 算法 / 服务、FastAPI API、Jinja2 / Vanilla JS 前端；不引入框架或更改旧实验协议。

## 预期成果

### 07 依存句法分析

保留交互分析 / 依存树 / 算法过程 / 实验评测四个 Tab。用户输入中文短句，使用冻结 HMM 分词及新增预训练 HanLP CTB9 ELECTRA-small biaffine parser 推断每词 head/deprel。选择词或弧时，原文、词列表、依存树和 Inspector 联动；有明确 ROOT、方向、关系释义和完整 CoNLL 导出。

解析结果必须验证词跨度、词覆盖、head 范围、自环、单根、无环和可达性。多句分别建树，不能把整篇文章强行解释为一个句子；限制每句词数及总输入长度。模型缺失、超限和无有效内容返回明确状态，不用规则伪装模型输出。

算法过程展示依存表示、biaffine 打分与树约束的区别。课程过程演示必须明确区分真实模型输出与教学 decoder；未获取的神经内部张量不能编造。优先可验证的真实 head/label 输出，随后接入可核验的内部打分/树解码。

评测使用独立、人工编写并预先固定的教学 Gold 句组，明确是小型教学诊断、gold-token parser evaluation，不宣称 CTB 正式 benchmark。保存 UAS/LAS、标点纳入/排除协议、每句正确弧/错误弧及新模块 provenance；不读取或改写已有分类 Test。

### 08 深度学习与大模型

保留 BiLSTM-CRF / Transformer / 大语言模型 / 扩展应用四个 Tab。BiLSTM-CRF 读取现有冻结资源与真实曲线；仅短句诊断推理，不续训、不重评。使用隔离 worker 复用已有 PyTorch 环境，避免 Web 进程缺依赖造成旧实验资源不可用。

Transformer 使用现存 `bert-base-chinese` safetensors：WordPiece IDs / offsets、MASK 的真实候选 logits/softmax、层/head/query 选择、真实 attention 权重与 QK/缩放/softmax 校验。注明 BERT 是双向掩码模型，不冒充生成式聊天模型；attention 不是因果归因。

生成式部分采用官方 `Qwen2.5-0.5B-Instruct` safetensors，固定 revision 和 SHA-256，本地 CPU 推理。显示实际 messages/chat template、BPE IDs、生成 token、真实 token 分布和停止原因。默认 greedy，采样实验使用固定 seed 与受限参数；所有输出是模型生成文本，不是事实保证或校准正确率。没有远程 provider、密钥或自动外发文本。

扩展应用提供中文摘要、结构化抽取、提示词边界和模型能力比较，所有示例与实际推理绑定；不预写假生成结果。实验任务可以检查输出格式，但不把教学示例升级成独立泛化 benchmark。

## 实现方式与边界

- 新增资源位于 `artifacts/course_models/`，准备记录含官方来源、revision、文件 hash 与 license；下载为显式准备步骤，交互 API 一律 offline/local-only。
- 复用已有 `artifacts/runtime/hanlp-venv` Python 3.11 / PyTorch CPU / Transformers 4.44.2，不升级或降级环境，不更改已有依赖。
- 新 worker 与 service 负责缓存、请求长度限制、串行模型调用、超时、CPU 线程上限及可审计版本；机器仅 CPU，不承诺 GPU 性能。
- 新 API 使用 `/api/v1/course/dependency`、`/api/v1/course/deep` 命名空间。不会扩充分类器、写 Stage7 或更改 frozen API 语义。
- 新运行与教学评测资产独立保存到 `outputs/course_dependency/` / `outputs/course_deep/`，沿用 SHA-256 / JSON 数值策略，绝不更新旧 latest。

## 状态矩阵

两个模块均覆盖：资源未准备、初始示例、加载、成功、空输入、过长输入、模型错误、桌面、移动端。模型调用只在明确用户交互或标明的初始教学示例发生；输入可编辑、请求可防抖/防竞态，错误时不保留上一条输出冒充当前结果。

## 验收方式

1. 算法与结构：依存树约束、原文 offsets、句边界、CoNLL、UAS/LAS 计算；BERT attention 行和/掩码/真实 forward 一致；LLM token 输出、采样 seed、EOS 与预算；BiLSTM emission / path 来自冻结模型。
2. 服务与协议：本地资源/hash 校验、非法参数、超限、offline 行为、无模型训练调用、缓存/并发/超时、API schema 和旧 API 回归。
3. 真实模型 smoke：中文依存句、BERT MASK、Qwen 短回答、冻结 BiLSTM 短句。记录 runtime 和实际 CPU 延迟；不能用 toy tests 替代真实 smoke。
4. 浏览器：07 四 Tab 的真实结果与选中联动；08 四 Tab 的模型响应与过程选择；1440×1000 / 390×844、可读图形、无页面横向溢出、console/page/HTTP errors。
5. 独立功能 / 视觉 review 与修复；已冻结 1,313 文件的 SHA-256 审计。目标功能和视觉评分均至少 4.3/5，无关键未解决问题。

## 推进与迭代

Round 1：资源准备、07 解析与树 schema、08 模型 worker / BERT trace、核心测试和真实 smoke。

Round 2：07 完整交互与教学评测；08 四 Tab 工作台、LLM 解码与扩展任务；浏览器证据。

Round 3：独立审查、定点修复、完整验收与冻结审计。只有上述预期成果全部被当前证据证明时，才完成整个目标。

参考：[HanLP CTB9 parser](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/dep.html)、[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。官方模型卡中的 benchmark 仅为上游报告，不能当成本项目评价结果。
