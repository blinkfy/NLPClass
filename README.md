# NLPLab

大学《自然语言处理》连续实验工程。当前实现实验一：最长匹配、二元语法、一/二阶 BMES HMM、BiLSTM-CRF、标准评价、透明模型存储、独立 Run 和轻量网页。HanLP MSRA BERT 用于独立 NER 评价，不替代 HMM。

## 四个实验的发展路线

| 实验 | 数据与算法 | 当前状态 |
| --- | --- | --- |
| 一：序列标注 | 七方法 MSR CWS + MSRA NER test personal-200 | Completed / Frozen：标准评价、错误分析、算法过程与 Web 均完成 |
| 二：文本聚类 | THUCNews 五类平衡全文 600 → K-means | 数据已准备；尚未训练模型，输入不含 primary_label |
| 三：文本分类 | 同一 600 条 → NB / SVM | 固定分层 420/90/90 已冻结；尚未训练模型 |
| 四：应用程序 | Web → NLPService → 分词 / NER / 分类 / 聚类 | 网页骨架与实验一系统测试，后续逐步接入 |

## 当前状态：实验一 Completed / Frozen

MSR standard 最终 benchmark 已完成并冻结，结果保存在 outputs/exp1/benchmarks/20260926_130612_32a5e168。
原有一阶 HMM/Viterbi/MSR importer、最终配置、模型及已有 Run 保持不变。默认课程数据使用公开语料个人子集：
MSRA NER test 200 条、官方 THUCNews 全文五类各 120 条。自采集与人工 Gold 工具保留为可选方案。
正式 NER Run：`20260926_152118_34a953fb`，Overall P/R/F1 = 91.88% / 92.74% / 92.31%。
toy 仅验证功能，不能计入课程样本和实验指标。实验二/三仍未实现。

## 实验一课程方法拓展（MSR 冻结评测）

在冻结的一阶 HMM 之外，`src/nlplab/extensions` 提供最长匹配、二元语法、二阶 HMM
和 BiLSTM-CRF。它们都使用同一份 MSR standard training/raw test，Run 写入
`outputs/exp1/extensions/`，模型写入 `artifacts/models/msr-extension-*`，不更新旧的
`latest.json`。完整算法定义、HanLP 二阶实现 revision、pytorch-crf 来源和课程参考值见
[docs/segmentation_extensions.md](docs/segmentation_extensions.md)。

```powershell
# 历史训练/评测命令，仅供了解已完成协议；冻结后不得执行或据 test 调整
.\.venv\python.exe -m nlplab.extensions longest
.\.venv\python.exe -m nlplab.extensions bigram

# 历史二阶 HMM 协议：development → 固定配置 final（已完成冻结）
.\.venv\python.exe -m nlplab.extensions hmm2-mle --phase development
.\.venv\python.exe -m nlplab.extensions hmm2-smooth --phase development
.\.venv\python.exe -m nlplab.extensions hmm2-mle --phase final
.\.venv\python.exe -m nlplab.extensions hmm2-smooth --phase final

# 历史 BiLSTM-CRF 协议：development best epoch=10 → final 10轮（已完成冻结）
$env:PYTHONPATH="src;."
artifacts/runtime/hanlp-venv/Scripts/python.exe -m nlplab.extensions bilstm-crf --phase development
artifacts/runtime/hanlp-venv/Scripts/python.exe -m nlplab.extensions bilstm-crf --phase final
```

BiLSTM-CRF 的固定初始配置在 `configs/bilstm_crf.yaml`：Embedding 128、双向 LSTM
128/方向、AdamW、lr=1e-3、batch=64、最多 10 epoch、patience=2、seed=42。
CRF 使用 MIT 许可的 `pytorch-crf==0.7.2`，项目只负责 adapter 和合法 BMES mask，
不把库实现称为自行重写。模型保存为 JSON metadata + vocabulary + PyTorch state_dict，
不 pickle-only；当前环境没有可用资源时 Web 将该模型标记为 unavailable，默认 HMM 仍可用。
NLP4J 未接入：当前可核验组件需要额外 JVM/语言模型，不能低成本匹配本项目中文 MSR BMES 任务。

### 本次拓展最终验收

七方法正式结果与课件差异见 [最终结果与验收](docs/segmentation_extensions.md#e7最终结果与冻结验收2026-09-27)。
Longest F1=93.65%、Bigram=94.32%、HMM2-MLE=78.93%、HMM2-Smooth=79.53%、
BiLSTM-CRF=94.41%（OOV-R=67.48%，coverage=100%）。一阶 HMM 和 MSRA NER 旧结果保持冻结。
BiLSTM-CRF 由 training-derived dev 冻结 best epoch=10，再从头 full training 训练 10 轮；
所有新方法各执行一次 official raw test 推理，未使用 test 调参。
模型、Run、预测、错误、两个 scorer 输出与 SHA256 清单均已保存；原有 264 文件指纹未变。
拓展阶段历史验收：129 项 pytest 通过，Ruff/format/pip check 与 Chromium 交互验收通过。

```powershell
# 可选神经依赖只装在独立环境，不改变核心冻结环境
artifacts/runtime/hanlp-venv/Scripts/python.exe -m pip install -e ".[sequence-neural]"
# 仅汇总已保存 Run，不重新训练或推理
.\.venv\python.exe -m nlplab.extensions compare
```

## 实验一最终报告资产（只读收尾）

已完成 Longest Matching、Bigram、HMM-1 MLE/Smooth、HMM-2 MLE/Smooth、BiLSTM-CRF、MSRA NER、标准评价、错误分析、算法过程可视化及 Web。
资产目录：`outputs/exp1/final_report_assets/20260927_010652`，含 summary、七模型 CSV/JSON、NER metrics、真实代码摘录、参数/训练预算、错误/代表案例、截图、来源 SHA256 和执行声明。
Longest/Bigram Tag Accuracy 为保存预测的只读派生值，不修改旧 Run。MSRA-200 不报告不存在的 CWS Gold F1。
BiLSTM development 达到 max_epochs=10，best_epoch=10，patience=2 未提前停止；不增加轮数、不重新评价。
当前数据/模型/权重与正式 benchmark 已准备并冻结。本收尾提取保存资产，并验收短 demo 的模型特异化 Web 展示；训练、official-test 推理、scorer 执行均为 false。

- [数据与评价口径](docs/exp1_data_methodology.md)
- [实验要求—证据矩阵](docs/exp1_requirement_traceability.md)
- [最终冻结声明](docs/exp1_freeze.md)

实验二 THUCNews 600 全文数据已准备，模型未实现；实验三固定 420/90/90 split 已准备，模型未实现。
custom corpus / 620-slot / manual annotation 保留为 Optional Workflow，不作为当前主线条件。

## 环境安装

目标 Python 3.11，本机验证版本 3.11.16。以下命令从工程根目录执行：

```powershell
conda create --prefix .venv python=3.11 pip -y
.\.venv\python.exe -m pip install -e ".[dev]"
.\.venv\python.exe -m pytest --verify-only -q
```

依赖声明在 pyproject.toml。requirements.lock 记录验证过的核心与开发依赖，可先安装该文件，再用 `pip install --no-deps -e .`。基础依赖为 NumPy、PyYAML、FastAPI、Jinja2、Uvicorn。pandas/scikit-learn 在后续需要时安装 `.[experiments]`；HanLP/PyTorch 为 `.[ner]` 可选依赖。网页使用原生 HTML/CSS/JavaScript，本地 ECharts 5.6.0 保留 Apache 许可，无 React/Vue/Node 构建工程。

## 工程与数据目录

src/nlplab 使用标准 Python src 布局，内部保持 core、data、sequence_labeling、segmentation、ner、features、clustering、classification、evaluation、services 分层。app 仅处理 API、模板和静态资源，算法不依赖 FastAPI。

| 目录 | 用途 |
| --- | --- |
| data/raw | 可选自建课程 Document JSONL |
| data/annotations | 可选自建语料人工 Gold 与草稿 |
| data/course | 冻结 MSRA 200 / THUCNews 600 个人子集 |
| data/manifests | 个人 selection/split 原始键，仅本地保留 |
| data/external | MSR 等外部原文件，导入不改写 |
| data/splits | train/dev/test 与来源指纹清单 |
| data/demo | 明确标记的合成 toy 训练/留出集，仅测试 |
| artifacts/models | JSON + NPZ 模型与同名 Run 模型快照 |
| artifacts/cache | 可选 HanLP 模型缓存 |
| outputs/exp1 | 不覆盖的 Run 与 latest 指针 |

真实数据、权重、cache 和 outputs 默认忽略，不提交 Git；仅示例与测试 fixture 可以提交。

## 默认公共数据方案

执行配置见 [public_datasets.yaml](configs/public_datasets.yaml)。三个指标来源分开：

- **CWS benchmark**：SIGHAN Bakeoff 2005 MSR，沿用冻结正式结果，不重训、不重新评分。
- **课程 200 样本 / NER benchmark**：MSRA NER **test** personal subset。MSRA 只有 NER BIO Gold，**没有 CWS Gold**；不得将分词预测当 Gold 或计算 200 条 CWS F1。
- **实验二/三数据**：官方 THUCNews 全文，finance/education/science/sports/entertainment 各 120；聚类用全部 600，分类共用固定 train 420 / val 90 / test 90。

### MSRA 固定来源、严格 BIO 与个人 200

使用 [levow/msra_ner 1.0.0 固定脚本](https://huggingface.co/datasets/levow/msra_ner/blob/250b3cc647ff1cef9d592301a5ba501679e27e91/msra_ner.py)
指向的 [OYE93 MSRA BIO](https://github.com/OYE93/Chinese-NLP-Corpus/tree/934e50bdbbd27f5ffa47eb4eff04651c6eef0a32/NER/MSRA)。
原文件 SHA256 与 URL 固定在配置中，导入再核验。实际 split 大小为 train 45,000 / test 3,442；不要与其他 MSRA 版本混称。
此版本 train 第 365 号记录含空标签，严格导入拒绝整个 train split，并记录 `bio_import_error.json`，不补 O、不跳过坏样本。
test 全部通过 BIO2 校验后转换为 Python code-point exact-span PER/LOC/ORG。train 不用于 200 条选择或 NER 训练。

```powershell
.\.venv\python.exe -m nlplab.cli prepare-msra-ner
# 也可 --source data/external/msra-ner-bio-934e50b 使用本地原文件
.\.venv\python.exe -m nlplab.cli select-exp1-samples
```

首次产生 `secrets.token_hex(16)`，持久保存于 `data/manifests/personal_selection.json`。
按 `SHA256(dataset_version + selection_key + stable_record_id)` 排序选择 test 200，选择不读取模型结果。
只有某类别 Gold support 恰好为零时，才允许推理前重新生成一次键；拒绝记录写入 manifest。
现有子集重跑只验证，不覆盖；来源/版本/哈希或键绑定变化时拒绝。不要删除键或子集以追求更高分。
个人键用于可复现地选择不同的个人集合，不检查未提供的其他学生清单。

输出 `data/course/exp1_msra_200.jsonl` 与 `_manifest.json`；保存 source record ID、原文哈希、source hashes、score 和键 fingerprint。
原始键不进入 Run，完整 selected IDs 和来源哈希进入独立 Run。

### THUCNews 官方全文、600 与固定 split

本机从 THUNLP 使用的 OSS 存储域公开地址下载
[THUCNews.zip](https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip)，UTF-8 全文，14 类，版本标记 `official-2005-2011`。
[清华 THUCTC 官方入口](http://thuctc.thunlp.org/)当前链接到个人信息表单，本次未填写或提交该表单。
存储域归属依据清华团队 [OpenCLaP 官方仓库](https://github.com/thunlp/OpenCLaP)使用的同一 `thunlp.oss-cn-qingdao.aliyuncs.com`；
归档内容另行核验分类、UTF-8 全文、长度及指纹，不使用未说明来源的标题派生集。
归档 SHA256：`8d6aef71c4431ba27c217a00adb2be7a78f8aa44f3c0bcb99d57a9026bcf163c`。
没有使用 cnews fallback 或短标题数据。数据许可未见明确声明，记录 `unknown`；仅课程实验本地使用。

```powershell
# 把官方 ZIP 放在 data/external/THUCNews.zip，或解压为 data/external/THUCNews/<中文类别>/*.txt
# provenance JSON 示例需填写真实下载时间，不要将占位文字用于正式导入
.\.venv\python.exe -m nlplab.cli import-thucnews --source data/external/THUCNews.zip --provenance artifacts/cache/thucnews_download.json
.\.venv\python.exe -m nlplab.cli select-thucnews-600
.\.venv\python.exe -m nlplab.cli split-thucnews-600
.\.venv\python.exe -m nlplab.cli audit-public-datasets
```

来源 JSON 格式见 [provenance 示例](docs/examples/thucnews_provenance.example.json)。归档或目录均可导入；ZIP 中文文件名按实际编码还原，排除 Mac 资源文件。
导入统计全部 14 类，只输出五个目标类别的全文候选，保留原文与相对路径。选择去空、NFC/空白规范化 exact duplicate、至少 80 code point；不分词/去停用词后再选样本。
使用与 MSRA 不同的持久 selection key，每类内部按 stable hash 取 120；不足时报错，不重复、不放宽。

冻结产物：`data/course/thucnews_600.jsonl`、`_manifest.json`；split 在 `data/splits/thucnews_600/`。
独立 split key 每类稳定哈希排序为 84/18/18，三个 split 的 ID 互斥，清单和文件哈希重跑校验。
`clustering_all.jsonl` 只有 id/text，**不含 primary_label**；标签只能另行用于聚类评价。所有实验三模型共用此 split。

可选 fallback 只接受 `dataset_variant=cnews-fulltext-10` 的全文版，必须提供 URL、哈希、版本及 `derivation_evidence`。
[原 cnews 派生脚本](https://github.com/gaussic/text-classification-cnn-rnn/blob/master/helper/cnews_group.py)保留整篇文本并删除换行/制表符，不是标题提取。
如果没有可信全文来源，停在 **waiting for dataset**；将官方分类全文目录或 ZIP 放入 `data/external/`，不填写个人资料表、不自动改用标题版。

### 实验一课程样本 Run

HanLP 指定模型需要 TensorFlow/Keras；本机使用独立 `artifacts/runtime/hanlp-venv`，不改核心环境或冻结模型。
可复现运行依赖：HanLP 2.1.5、TensorFlow 2.15.1、tf-keras 2.15.1、NumPy 1.26.4、Transformers 4.44.2。
安装核心 Web 依赖后，在该环境安装上述版本。可以运行 `pip install -e .`（不重新训练任何 HMM）。

```powershell
$env:PYTHONPATH="src;."
# 公共 HF 资源下载时关闭隐式登录 token，避免旧凭据导致 401
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN="1"
artifacts/runtime/hanlp-venv/Scripts/python.exe -m nlplab.cli setup-ner
artifacts/runtime/hanlp-venv/Scripts/python.exe -m nlplab.cli run-exp1-course-samples
```

运行前必须验证冻结 Smooth 权重及 HMM/Viterbi/MSR/config 指纹。没有 CWS Gold 的 MSRA 不使用旧 `SequenceSample`，也不传入分词 evaluator。
每条保存分词 tokens、BMES、NER Gold 与预测。实体评价复用项目 exact-span evaluator，输出 PER/LOC/ORG/Overall P/R/F1、代表性正确/错误案例。
NER 使用已核验 NR→PER、NS→LOC、NT→ORG；按模型实际 WordPiece 预算分块，避免 128-token 模型被旧 200 字分块静默截断。
不改模型 max_seq_length、权重、平滑或 UNK，不据结果调参。

独立输出 `outputs/exp1/course_samples/<run_id>/`，保存配置、manifest、指纹、预测、错误、样例、指标及 report。
不覆盖 development/最终 MSR Run 或旧 latest.json。中文分词页绑定冻结 MSR 模型；词性与命名实体页独立选择课程 200 NER Run，不显示不存在的 CWS F1，正式 Run 只读。
中文分词交互默认 HMM-1 Smooth；进入分析页、修改输入或切换模型后自动分析，输入修改使用短延迟合并。中文分词页默认在结果下方展开算法过程，按钮仍可手动刷新。HMM 参数见序列标注，BiLSTM 结构与训练见深度学习。聚类/分类页面仅显示数据状态。

审计输出 JSON + Markdown 到不覆盖的 `outputs/data_quality/<id>/`。
[子集 manifest schema](docs/public_subset_manifest.schema.json)描述提交格式；真实全文、原始语料、键、权重、Run 均忽略 Git。

## Optional: custom corpus / manual annotation workflow

如果教师明确要求“自行抓取/人工标注”，可启用以下旧流程。当前默认课程方案使用公开标准数据集的个人子集；
`collection_slots_620_v2` 不再是主线验收条件，也不要求先收集 620 条网页才能继续实验。
原 annotation 页面与代码/测试完整保留，可用于自建数据检查、修订和未来人工 Gold。
公开 MSRA 记录没有 CWS Gold，不可直接伪装为人工 CWS 标注记录。

### 准备统一自建课程语料

Document 的 source_type 表示 news/comment/blog/fiction 等体裁，primary_label 表示主题，两者独立。保留原文、来源标题、实际 URL、类别与原文 SHA256。data/demo/documents.jsonl 是合成测试结构示例，不是真实课程语料。

```powershell
.\.venv\python.exe -m nlplab.cli import-corpus --input path/to/collected.jsonl --output data/raw/documents.jsonl
```

每行需有 id、text、source_type、primary_label、source_title、source_url，可选 secondary_labels。sha256 省略时自动计算，提供时必须与原文相符。真实数据 URL 必须为 http/https，synthetic 数据必须标记 is_demo=true。

导入严格校验每行、拒绝重复 ID，输出唯一文档及 `.duplicates.json` 报告。去重键使用 NFC 与空白标准化，原文及原文指纹不修改；错误显示文件和行号，目标已存在时拒绝覆盖。数据导入脚本位于 scripts，模型算法在 src/nlplab/sequence_labeling。

### 200 条序列与 NER Gold

下面是向后兼容的旧结构示例，不能直接作为本阶段正式 Gold；正式流程见下节：

```json
{"sample_id":"SEQ00001","doc_id":"DOC000001","text":"张三来到北京。","gold_tokens":["张三","来到","北京","。"],"entities":[{"text":"张三","type":"PER","start":0,"end":2},{"text":"北京","type":"LOC","start":4,"end":6}]}
```

doc_id 必须存在于主语料，text 必须来自该文档原文。必须满足 `''.join(gold_tokens)==text`；BMES 只自动生成，不独立存储。跨度为 Python Unicode 字符索引 `[start,end)`，不是字节或 JS UTF-16 单元。entities=null/缺失表示未标注，[] 表示人工确认没有实体。

```powershell
.\.venv\python.exe -m nlplab.cli validate-course --corpus data/raw/documents.jsonl --gold data/annotations/sequence_gold.jsonl
```

验收 >=500 篇去重真实文档、>=200 条关联且不重复的 Gold，拒绝 demo。学生间去重还需和同学核对来源及指纹，本地工具无法检查未提供的他人语料。自建 Gold 当前作为独立泛化评价集，不混入 MSR 训练或调参。

### 自建语料与人工 Gold 工作流

正式约束来自 [课程计划](configs/course_corpus_plan.yaml)、[人工标注规范](docs/annotation_guideline.md)
和 [Gold JSON schema](docs/sequence_gold.schema.json)。执行配置是 [course_corpus.yaml](configs/course_corpus.yaml)，
[决策日志](docs/annotation_decisions.md)记录已定规则。notes 不自动修改规范。

先人工收集 600 条真实原文：六个主题各 100；体裁 news/blog/comment/fiction 为 340/145/95/20，
逐类配额见配置。每条保留真实 URL、标题、主题、体裁与原文，40–400 code point；同 URL 最多 3 条。
导入工具计算原文 SHA256，NFC/空白规范化只用于重复检测，不改原文。raw/annotations/outputs 均被 Git 忽略。

```powershell
python -m nlplab.cli import-corpus --input path/to/collected.jsonl --output data/raw/documents.jsonl
python -m nlplab.cli audit-course-corpus --corpus data/raw/documents.jsonl --config configs/course_corpus.yaml
python -m nlplab.cli sample-sequence-gold --corpus data/raw/documents.jsonl --config configs/course_corpus.yaml --output data/annotations/candidates.jsonl
```

审计只读输出 outputs/data_quality/唯一目录下的 statistics.json + report.md：总数、主题/体裁/交叉配额、
code-point 长度、域名、URL 贡献数、重复、缺失 metadata 和指纹问题。根据报告补齐真实来源后，用新输出路径导入。
已有文件拒绝覆盖，不自动生成缺失文档。

抽样固定 seed=42，每 doc 最多一条，15–80 code point；保留引号与全部原文位置，不截短长句。
使用 metadata 和数字/ASCII/全角/引号/中英混合/日期百分比表面特征，不调用模型或利用模型错误。
主题精确 34/34/33/33/33/33，体裁优先精确 110/45/30/15，必要时仅使用配置允许的 ±3，
实际偏差显式记录。联合配额不足时返回非零退出码，保存缺口 manifest/report，不写部分候选。
同输入和 seed 的候选及 manifest 内容完全确定，无随机时间戳。--allow-demo 仅用于测试，正式导出始终拒绝 demo。

候选保存 candidates.jsonl + candidates.manifest.json，初始 gold_tokens/entities 均为 null。
manifest 保存 seed、来源/候选指纹、各层分布、排除原因、来源 offsets 和文档指纹；不得在抽样后修改 raw 或候选。
来源满足 `document.text[source_start:source_end] == sample.text`，所有索引为 Python code point。

启动原有 FastAPI 服务，访问 **http://127.0.0.1:8000/annotation/sequence**（不新增主导航）：

1. 按主题/体裁/状态筛选，查看四种状态进度；上一条/下一条会先保存当前编辑。
2. 点击字符间边界切词，实时派生 tokens/BMES。初始不生成 Gold；无切词边界也须人工点击“确认当前分词”。
3. 点击两个端点、拖选或填写连续 span，标 PER/LOC/ORG；可以修改/删除，禁止重叠。
   JS 使用 Array.from(text)，emoji 算一个 code point；`entities=[]` 须人工确认无实体。
4. 编辑自动原子保存到 data/annotations/work/每个 sample_id.json，仅改当前草稿；版本冲突返回 409。
   保存失败保留页面编辑并提供重试；切换样本前必须保存成功。notes 不改变 guideline。
5. 完成分词与 NER 后标记 done，第二遍复核后勾选确认并标记 reviewed。
   revision_count/review_history 仅记录显式复核，自动保存不增加次数；修改 Gold 后要求重新 done/复核。

“参考建议，非 Gold”默认隐藏，只读，不提供自动应用 Gold。HanLP unavailable 时照常标注。
若需 frozen Smooth 建议，在 course_corpus.yaml 的 suggestions 中填写冻结模型目录与 hmm_freeze.json 路径，
例如 artifacts/models/msr-final-20260926_130612_32a5e168_smooth 和
outputs/exp1/benchmarks/20260926_130612_32a5e168/hmm_freeze.json；加载前验证模型文件指纹。
不读取 latest toy 模型，不训练或改动模型，不创建性能 Run。

```powershell
python -m nlplab.cli validate-sequence-annotations --corpus data/raw/documents.jsonl --config configs/course_corpus.yaml
python -m nlplab.cli export-sequence-gold --corpus data/raw/documents.jsonl --config configs/course_corpus.yaml
```

两者默认读取 candidates.jsonl 和 work；可显式指定 --candidates/--work-dir/--reports。
正式导出检查 exactly 200 reviewed、sample/doc 唯一、来源还原、tokens roundtrip、合法 BMES、实体 text/span、
PER/LOC/ORG only、无重叠、主题/体裁配额、>=100 含实体样本、三类实体各 >=40、困难样本 >=40、
至少 40 条明确第二遍复核记录，并拒绝 demo。校验失败保存报告并返回非零，不能生成正式 Gold。

成功时输出 data/annotations/sequence_gold.jsonl、sequence_gold_manifest.json，
以及 outputs/data_quality/唯一目录下的报告和统计 JSON。manifest 包含配置、数据、草稿和 Gold 指纹。
Gold entity 只含 text/type/start/end，不写模型 score；BMES 不独立保存。已有正式输出拒绝覆盖。
旧 SequenceSample/旧 validate-course 入口继续兼容，但旧入口不能替代本阶段正式验收。
本阶段没有计算自建集 HMM/NER 性能，也没有实现聚类/分类。

真实浏览器验收使用独立 toy 目录和临时服务，不影响已有开发服务器：

```powershell
.venv\python.exe -m pip install -e ".[browser]"
.venv\python.exe -m playwright install chromium
.venv\python.exe scripts/check_annotation_browser.py
```

该脚本只生成 3 条明确标记 toy 的候选并执行真实 Chromium 交互，结果保存于 outputs/data_quality/browser_*。
不会生成正式 Gold、600 条真实语料或性能结果；完成后关闭临时服务。应用使用 Python/FastAPI/Jinja2/原生 JS，
无 React/Vue/Node 工程或数据库/任务队列依赖。

## MSR 数据导入说明（历史流程；当前 benchmark 已冻结）

从 [SIGHAN Bakeoff 2005 官方数据页](http://sighan.cs.uchicago.edu/bakeoff2005/) 获取研究使用的 icwb2-data，或使用课程提供的同一归档，遵循原始数据说明。[Bakeoff 原始论文](https://aclanthology.org/I05-3017/)

解压到 data/external/icwb2-data，至少需要 training/msr_training.utf8 和 gold/msr_test_gold.utf8。使用 UTF-8 版本，不把 CP936 文件伪装为 UTF-8。

```powershell
.\.venv\python.exe -m nlplab.cli import-msr --source data/external/icwb2-data --output data/splits/msr
```

源文件按空白分词，导入只去除词间分隔符，不修改源文件。正式 benchmark 默认 `--protocol standard`，可显式选择 `--protocol clean`：

| Protocol | 官方训练样本 | 官方测试样本 | dev 与用途 |
| --- | --- | --- | --- |
| standard（默认） | 全部保留原顺序、Gold 和重复次数；overlap 检测不删除样本 | 全部保留，包括重复记录 | dev.jsonl 为空，实际 dev_ratio=0；用于完整官方 training/test benchmark |
| clean | 先排除与 test 同文的训练记录，再去重；同文不同 Gold 报错 | 全部保留 | seed=42、dev_ratio=0.1，从清理后的训练集划分 train/dev；用于清理与调参对照 |

standard 不调用清理/随机分割函数；配置中的 dev_ratio 仅对 clean 生效。潜在重复检测使用 NFC 与空白标准化，只记录训练重复数、测试重复数、训练样本与测试文本的重叠数；所有 protocol 都保存源文件 SHA256、输出文件/完整样本记录指纹、原始及实际数量。standard 的两个删除数量必须为 0。clean 的结果必须单独标记，不能当作保持官方训练样本的标准 benchmark。

```powershell
.\.venv\python.exe -m nlplab.cli import-msr --source data/external/icwb2-data --output data/splits/msr-clean --protocol clean
```

数据缺失时明确报错，不自动生成替代语料；已有划分不会覆盖。旧版未记录 protocol 的 MSR 划分不能声明为 standard，应重新导入至新目录。选择模型配置只使用训练侧数据或 clean 的 dev，参数确定后重新在完整 standard train 上训练，test 不用于调参。

## HMM 训练与评测说明（历史流程，当前不重复执行）

```powershell
# 课程 baseline：CLI train 默认配置也是 hmm_mle.yaml
.\.venv\python.exe -m nlplab.cli train --train data/splits/msr/train.jsonl --model artifacts/models/hmm-msr-mle --config configs/hmm_mle.yaml
.\.venv\python.exe -m nlplab.cli evaluate --model artifacts/models/hmm-msr-mle --test data/splits/msr/test.jsonl --dataset msr-standard-mle
# 增强模型，使用相同的完整 standard training/test 划分
.\.venv\python.exe -m nlplab.cli train --train data/splits/msr/train.jsonl --model artifacts/models/hmm-msr-smooth --config configs/hmm_smooth.yaml
.\.venv\python.exe -m nlplab.cli evaluate --model artifacts/models/hmm-msr-smooth --test data/splits/msr/test.jsonl --dataset msr-standard-smooth
.\.venv\python.exe -m nlplab.cli evaluate --model artifacts/models/hmm-msr-smooth --test data/annotations/sequence_gold.jsonl --corpus data/raw/documents.jsonl --dataset course-gold
```

| 配置 | 定位 | 统计与未知字符 |
| --- | --- | --- |
| configs/hmm_mle.yaml | 课程 baseline；train 默认 | smoothing=none，纯 MLE；alpha 不参与统计，UNK=0 |
| configs/hmm_smooth.yaml | 增强模型 | Add-alpha=0.1，UNK 随词表共同归一化 |
| configs/hmm.yaml | 保留旧命令及 toy 演示兼容 | 与增强模型相同的参数；不作为课程 baseline 默认 |

两者均保持合法 BMES 起始、转移及结束 mask。若 MLE 遇到未见字或其他零概率路径，会明确报错并停止该 Run；不能悄悄使用增强模型，也不生成伪造 baseline 指标。

模型训练元数据保留 MSR protocol 与清单指纹。evaluate 校验输入文件和包含 Gold、顺序、重复次数的记录指纹，要求 MSR 训练/评价来自同一清单的正确划分。仅完整、匹配且校验通过的 standard test 允许官方划分原有的文本 overlap，并记录实际评价重叠条数；不删除测试样本。普通评价与 clean 仍拒绝训练/评价文本重叠。dataset 名称本身不能绕过检查。protocol 写入 run.json 与 config.yaml，供报告核查。

clean 调参示例：在 data/splits/msr-clean/train.jsonl 上训练，在其 dev.jsonl 上 evaluate；正式 benchmark 再使用完整 standard train/test，保持 protocol 独立。调整配置时使用新的模型目录。NER Gold 与真实模型可用后在 evaluate 后附加 `--ner`；没有标注不生成 NER 指标，NER 失败不会保存成功 Run。

### HMM 与评价说明

自行统计初始、转移和发射频次，显式形成 π、A、B。START→B/S，B/M→M/E，E/S→B/S，END←E/S；非法结构始终为概率 0、log -inf。

增强模型（及 Python HMMConfig 默认）为 smoothing=add_alpha、alpha=0.1；课程 baseline/CLI train 默认使用上述 MLE 配置。合法转移分母为合法计数总和 **加** alpha×合法后继数；初始仅平滑 B/S。B 的词表包含训练字符与显式 UNK 列，分母为状态发射总数加 alpha×(字符数+1)。未见字符映射到 UNK，概率来自归一化统计，不用任意常数。

smoothing=none 为纯 MLE：UNK 概率为 0，无观测的状态/转移行拒绝训练；无完整有效路径时明确报错。Viterbi 保留 score 与 backpointer，分为初始化、递推、终止、回溯，短输入通过穷举最优路径测试。codec 对非法结构保留字符；正常 HMM 有合法 mask。交互空白字符独立标 S，词列表拼接保持原文。

评价包括 BMES Tag Accuracy、按字符跨度逐句累计的 word P/R/F1、基于训练词表的 OOV/IV Recall 和混淆矩阵。重复词按位置计数；无对应 OOV/IV Gold 时 Recall 为 null，网页显示无样本。NER 只在 start/end/type 全匹配时计 TP，输出 PER/LOC/ORG/Overall P/R/F1。

### 模型与 Run

模型为 metadata.json + probabilities.npz，含 states、vocabulary、π/A/B、配置、训练元数据、训练词表和指纹。加载检查形状、概率、合法 mask 和参数 SHA256，再重建 log 值，不使用 pickle-only 存储。

每次评价生成 UTC 时间戳＋随机后缀的 Run，保存 config.yaml、metrics.json、predictions.jsonl、errors.jsonl、run.json、parameters/start.json、transition.json、emission_topk.json、parameters.json、figures/。artifacts/models 下保存同名模型快照；全部写出后原子更新 latest.json。

网页所有数字与错误来自选定 Run。自动错误类型依据启发式标记 heuristic；OOV、Proper noun（利用已有实体 Gold）、Number/date、English mixed 和 Other 可自动判断，Ambiguity 等可人工指定。人工类别写入独立 sidecar，不改写原始预测/指标；下载原始错误仍保留首次运行结果。PNG 文件名含 Run 标识，浏览器下载后可归档至该 Run 的 figures/。

## HanLP adapter 与可选自建 NER

指定 MSRA BERT 是 TensorFlow 组件；`.[ner]` 的基础 HanLP 依赖不足以独立运行该组件。
使用上面已验证的独立 NER 环境和 `setup-ner`，Web 也使用该环境，核心 `.venv` 保留冻结 benchmark 环境。

setup-ner 默认读取 `configs/hanlp_msra_bert.yaml`，显式下载并校验 MSRA_NER_BERT_BASE_ZH，默认 CPU。可通过 `setup-ner --config path/to/model.yaml` 提供该模型专属的 model、name、expected_entity_tags、label_map、tagset_source；不能将不同 tagset 的 HanLP 模型当作同一个模型直接替换。

本次检查了[官方模型定义](https://github.com/hankcs/HanLP/blob/master/hanlp/pretrained/ner.py)所指向的[模型归档](https://ftp.hankcs.com/hanlp/ner/ner_bert_base_msra_20211227_114712.zip)内的小型 vocabs.json，实体类型确为 NR、NS、NT（词表中的标签带 B/M/E/S 前缀）。因此此模型配置明确写为 NR→PER、NS→LOC、NT→ORG。这不代表所有 HanLP NER 都使用这些标签；官方 [MSRA 标注说明](https://hanlp.hankcs.com/docs/annotations/ner/msra.html)还列出 PERSON/LOCATION/ORGANIZATION 等命名规范。该归档的 meta.json 为 TensorFlow 组件，未安装相应运行依赖时仍会明确失败，不能以 mock 宣称真实模型可用。

setup-ner 从实际加载组件的 TF transform.tag_vocab 或 PyTorch vocabs['tag'] 提取实体 tagset，并要求配置映射完整匹配；不匹配、不能读取词表或不支持的序列标签均明确报错，成功状态不会写出或覆盖。成功后将模型路径、名称、映射、实际 tagset 与模型配置指纹一起保存至 artifacts/cache/ner_setup.json。adapter 懒加载时再次校验实际 tagset，并只按已记录映射转换实体，未知输出标签报错；旧版缺少映射的 setup 文件必须重新 setup-ner。

默认缓存 artifacts/cache/hanlp，已有 HANLP_HOME 时尊重其配置；初次准备还可能需要 Hugging Face 资源。下载失败按照官方缓存/离线说明处理。[安装说明](https://hanlp.hankcs.com/docs/install.html) · [缓存配置](https://hanlp.hankcs.com/docs/configure.html)

adapter 检查原文跨度，按句和最多 200 字分块，跨块实体可能遗漏；输入是字符，不使用 HMM 分词替代。模型未准备时网页展示 unavailable 原因，不生成伪造实体；分词结果可正常返回。当前指定模型权重已准备；课程 200 正式结果保存在独立 Run，不能用 unavailable 代替正式评价。

若 Web 已启动，setup-ner 完成后重启一次以读取新的模型配置。推理使用已准备的 transformer 缓存；缓存被清理时需重新准备，不能把 unavailable 当成“没有实体”。

## 启动 Web

```powershell
# 仅分词/展示可用核心环境；完整 HanLP 交互使用上述独立 NER 环境
$env:PYTHONPATH="src;."
artifacts/runtime/hanlp-venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app --reload-dir src
```

访问 [首页](http://127.0.0.1:8000/) 或 [中文分词](http://127.0.0.1:8000/segmentation)。桌面只有一套课程 sidebar，移动端使用导航 drawer；模块内部使用水平 Tab。支持图表 PNG 导出。

| 一级模块 | Canonical route | 当前职责 |
| --- | --- | --- |
| 首页 | `/` | 课程能力总览；实验一 Completed / Frozen |
| 中文分词 | `/segmentation` | 五模型交互、算法过程、唯一七模型 MSR 表、CWS 错误 |
| 序列标注 | `/sequence-labeling` | BMES、HMM1/HMM2、平均结构化感知机与独立线性链 CRF 的真实交互与机制对比 |
| 词性与命名实体 | `/tagging` | HanLP CTB9 POS 交互；独立 NER 交互、MSRA-200 指标与错误 |
| 信息抽取 | `/information-extraction` | 课程补充实验：复用现有 HanLP NER，以实体类型和显式触发词抽取带原文证据的 SPO；提供实体关系图与 JSON。不是四个正式实验之一，没有 Relation Gold 或正式关系 benchmark。 |
| 文本聚类 | `/clustering` | THUCNews600 数据状态，模型未实现 |
| 文本分类 | `/classification` | 固定420/90/90 split 状态，模型未实现 |
| 依存句法分析 | `/dependency-parsing` | 课程框架，待接入 |
| 深度学习与大模型 | `/deep-learning` | BiLSTM-CRF 结构、CRF、冻结训练曲线；Transformer/LLM 待接入 |
| 综合分析 | `/integrated-analysis` | 同一 service 的分词+NER；多文档能力待接入 |

旧 `/sequence` → `/segmentation`、`/analysis` → `/integrated-analysis` 使用307兼容重定向。`/annotation/sequence` 保留为可选人工标注，不进入主导航。模型/数据/API 复用关系及迁移证据见 [Web IA 迁移记录](docs/web_ia_migration.md)。

无数据时默认真实空态。需要演示可显式执行：

```powershell
.\.venv\python.exe -m nlplab.cli demo
```

toy 的训练/测试不同，结果来自实际训练、预测和计分，始终标记为演示，不能用于正式实验性能。NLPLAB_ROOT 可指定模型与 Run 工程根目录。当前面向本机使用，无账号、Redis、数据库或后台任务队列。

## API 与未来扩展

| API | 用途 |
| --- | --- |
| POST /api/v1/analyze | text、tasks=[segmentation,ner]、可选 run_id；classification=null |
| GET /api/v1/models/hmm/parameters | run_id、topk；π/A/Top-K B |
| GET /api/v1/experiments/1/runs | 完整 Run 列表与 latest |
| GET /api/v1/experiments/1/metrics | 选定 Run 标准指标 |
| GET /api/v1/experiments/1/errors | run_id、category、page、page_size（NER 页复用） |
| GET /api/v1/views/cws/errors | 只读聚合 CWS frozen errors；复用 RunStore / read_errors |
| GET /api/v1/course/resources | 课程扩展模型和 POS 资源状态 |
| POST /api/v1/course/sequence | 感知机或独立线性链 CRF 的 BMES、词序列与解释数据 |
| POST /api/v1/course/pos | 现有 HMM 分词 + HanLP POS 原始标签及 Unicode 跨度 |
| PATCH /api/v1/experiments/1/errors/{sample_id} | 设置或清除人工类别 |
| GET /api/v1/experiments/1/export/{name} | metrics/predictions/errors/config |

输入错误及无有效路径为 422，缺失 Run/模型为 503，均有清晰消息。分类/聚类 API 未注册。SequenceTagger、Segmenter、NERRecognizer 为结构化接口；NLPService 可以注入分词器加载器与 NER。TextClassifier、TextClusterer 仅保留接口，features 保留扩展位置，后续按课件实现真实模型。

### 课程序列标注与 POS 补充模块

`/sequence-labeling#perceptron` 是项目自行实现的平均结构化感知机薄层：字符窗口、相邻字符、字符类型和 BOS/EOS 特征，加上前一标签到当前标签的转移权重；沿用已有合法 BMES Viterbi。页面展示当前字符各标签累计得分、特征贡献和训练中真实保存的一次 Gold/预测差分更新。`/sequence-labeling#crf` 使用固定 `python-crfsuite==0.9.12` 的线性链 CRF，页面展示 BMES、位置边缘概率、状态特征与学习到的转移权重；训练优化与解码由该库提供。二者在同一份 MSR **official training 内部**用 seed 42 的固定哈希子集训练 8,000 句、开发 800 句，配置见 `configs/course_sequence.yaml`，模型和记录分别存于 `artifacts/models/course_sequence/` 与 `artifacts/cache/course_sequence_subset.json`。这是独立课程演示，不进入实验一七模型 official test 结果表，未运行 official test 或 scorer。

`/tagging#pos` 使用已有 HMM-1 Smooth 对交互文本分词，再由 HanLP `CTB9_POS_ELECTRA_SMALL` 预训练模型标注词性。页面保留 HanLP 原始 CTB 标签、Unicode code point `[start,end)` 与辅助粗类配色；没有 POS Gold 和正式 POS benchmark。POS 模型来源为 [HanLP 官方 POS 模型目录](https://github.com/hankcs/hanlp/blob/master/docs/api/hanlp/pretrained/pos.md)，本地模型信息保存在 `artifacts/cache/pos_setup.json`。运行 POS 页面需使用已安装 HanLP 的 `artifacts/runtime/hanlp-venv`，缺失模型时 API 会返回明确的 unavailable 状态。

课程第 5 章对应结构化感知机序列标注实验，第 6 章对应线性链 CRF 序列标注实验，第 7 章对应流水线式 POS 实验。三个模块依据课程核心算法思想组织实验说明和真实输出；工程实现沿用当前 Python 生态，不复刻课件示例代码或工具。感知机和 CRF 的开发集 Tag Accuracy 分别为 91.08% 和 93.33%（MSR official training 内部固定子集，train 8,000 / dev 800 / seed 42）；它们不是 official test 指标。`/sequence-labeling#comparison` 只比较机制与同一句的 BMES、分词及交互推理时间，不加入七模型正式排行榜。

## 测试与已知限制

```powershell
.\.venv\python.exe -m pytest --verify-only -q
.\.venv\python.exe -m ruff check src app scripts tests
.\.venv\python.exe -m ruff format --check src app scripts tests
.\.venv\python.exe -m pip check
```

覆盖 Gold/BMES、π/A/B、Viterbi 约束与最优性、未知字符、Unicode/空白往返、重复词跨度、实体精确匹配、保存/加载、Run 不覆盖与泄漏拦截、MSR standard/clean 的样本保留、去重、重叠、清单指纹与 CLI 端到端流程，MLE/增强配置的合法 mask、NER 显式映射与加载词表不匹配拦截，以及 API、十个 canonical 模块、旧路由重定向和独立任务/模型页面。冻结后的回归使用 `--verify-only`，在训练及正式 benchmark/scorer 调用前跳过相关测试。单测不下载大语料；NER mock 仅验证 adapter 类型与索引，不能证明真实 NER 性能。

公共个人子集已准备；可选自建语料仍需真实采集与人工标注，不能用 toy 结果宣布自建实验达标。当前 Starlette/AnyIO 有一条第三方弃用提醒，测试通过。分词标注规范不同的模型不放在同一 MSR 排行榜；错误类别只是辅助启发式。大规模/GPU、其他浏览器和公网多用户场景尚未验收。

### 模型特异化交互过程（已验收）

公共分词结果下分别展示最长匹配扫描、Bigram 词网/边代价/前驱链、一阶 Viterbi state-score、二阶 state-pair/backpointer，以及 BiLSTM-CRF emission/解码路径。参数区分别显示词典统计、固定插值、π/A/B、A2 切片或神经网络架构/CRF heatmap/development 曲线。≤30 Unicode code point 自动展示；长文本仍完整分词。NER 继续独立，默认仍为 HMM1 Smooth。

冻结收尾验证使用 `pytest --verify-only`，在训练/正式 benchmark/scorer 调用前跳过相关历史测试；无需重训 toy 模型。

最终收尾验收：全项目 `pytest --verify-only` 109 passed / 35 skipped，训练与正式 benchmark/scorer 在调用前阻止；Ruff check、format（87 文件）、两个环境 pip check 和 Chromium 1440/390 验收通过。原有 264 与拓展 90 冻结文件保持 unchanged。
