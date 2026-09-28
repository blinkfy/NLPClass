# 实验一分词方法拓展

## 数据与冻结边界

七种 CWS 方法共享 MSR standard 的 86,918 条官方训练样本及 3,985 个 raw test
物理 CRLF 行。一阶 HMM 的模型、配置和 Run，以及 MSRA NER personal-200 Run
只读引用，不重训、不重评。MSRA NER 不进入 CWS 比较。

`python -m nlplab.extensions` 是拓展的独立阶段入口。新模型位于
`artifacts/models/msr-extension-<run_id>`，新 Run 位于
`outputs/exp1/extensions/<run_id>`；不会更新旧的 `latest.json`。

预测时只读取 official raw test；预测文件落盘后才读取评价 Gold。
沿用已核验的 raw/Gold 16 处行边界差异协议：官方 scorer 使用原始 Gold，
项目 evaluator 使用既有的 raw 坐标对齐 Gold。两者输出分别保存。
若有无有效路径，采用既定 abstention（空预测，不伪造单字预测），并保留
完整 Gold 分母与 successful-only 指标。官方脚本的除零保护版本沿用旧 Run，
不改原评分规则。

## E1：最长匹配

词典来自完整官方 training 的 Gold 词集合；使用 Python set 索引。
从当前 code-point offset 向右扫描至训练词典的最大词长，选择匹配的最长词。
未登录字符退化为单字，输入空白作为独立字符保留。不是另写 Trie/DAT。
短文本 trace 保存 offset、候选词和选择结果。

## E2：二元语法

训练统计一元词频、相邻词二元频次及 BOS/EOS。
固定采用 `0.9 * P_MLE(w|previous) + 0.1 * P_add-0.1(w)`。
无观测的上下文完整回退至平滑一元分布；未知单字共享 UNK_WORD。
这确保每个上下文的条件分布归一化。

词网包含每个位置的训练词典匹配与单字回退。
DP 的状态是 `(end_offset, last_word)`，边权为 `-log P(word|previous)`。
逐个 end 保存最小累计代价和前驱；加入 EOS 代价后终止并回溯。
trace 展示词网、候选边、局部/累计代价、前驱、EOS 和最优路径。
所有配置在 official test 推理前确定，不为课件数值调参。

## E3：二阶 HMM 的定义与来源

逻辑在项目内实现；复用冻结 HMM1 的词表、π、发射估计、codec、持久化，
独立估计首对转移与二阶张量，不调用外部黑盒预测。

编码前核验的课程配套源码：
[HanLP SecondOrderHiddenMarkovModel](https://github.com/hankcs/HanLP/blob/dadd5c76a53eb05cc84720628b89e4a334d5d753/src/main/java/com/hankcs/hanlp/model/hmm/SecondOrderHiddenMarkovModel.java)，
revision `dadd5c76a53eb05cc84720628b89e4a334d5d753`。
参考其算法定义，自行编写 Python/NumPy 逻辑，没有复制 Java 源码。

- 首字符：`first[s] = log π[s] + log B[s,x0]`。
- 长度 1：在同时合法的起止状态中选择（BMES 仅 S）。
- 第二字符：`D[1,r,s] = first[r] + log A_initial[r,s] + log B[s,x1]`。
- `A_initial` 只统计每句的首对，与 companion 一致；不以所有相邻转移替代。
- 从第三字符开始：`D[t,s,u] = max_r(D[t-1,r,s]+log A2[r,s,u])+log B[u,xt]`。
- `A2[r,s,u]` 统计三元组，按最后一维归一化；合法三元组需两对都合法。
- MLE 无观测行保持零；Smooth 只在合法起始/转移/三元组上加 0.1。
- 终止限定 E/S；回指针 `back[t,s,u]` 存 r，向左恢复整条 BMES。
- companion 不专门约束 BMES，本项目显式加入合法起止和转移 mask。
- companion 的 normalize 直接除行和；本项目显式将无观测 MLE 行保持为零，
  避免 0/0 NaN。该约定在推理前确定，没有用 test 选择回退策略。

开发评测只复用现有 `msr_development_..._attempt2`；不重新划分。
toy 长度 1–5 穷举全部合法 BMES 路径，核对最优分数与路径。

## E4：BiLSTM-CRF 应用

采用 PyTorch Embedding、双向 LSTM、Linear 与第三方
[pytorch-crf 0.7.2](https://pypi.org/project/pytorch-crf/0.7.2/)（MIT）。
CRF forward/partition/Viterbi 来自库，不能宣称是本人从零实现。
该版本发布较早，已在本机 Python 3.11 / PyTorch 2.14.0+cpu 验证
前向、反向、bool padding mask 与变长解码；项目 adapter 隔离其格式。
wheel SHA256 `1b2d7d5eea3255f6e0cac09ab8b645472e76ff70d9333bc88762cf7317a4992d`。

固定初始配置：embedding 128，hidden 128/方向，1 层，dropout 0.2，
AdamW lr 0.001，batch 64，最多 10 epoch，patience 2，clip 5，seed 42。
dev 只决定 best epoch；final 从头用完整训练集训练该固定 epoch。
不截断、不丢弃长训练句。保存 JSON 配置/词表/历史/metadata 与 state_dict。
固定 CPU 并行度为 8；seed 在网络权重初始化之前设置，并启用严格 deterministic
algorithms。训练、dev loss 和 dev 解码均按 batch 处理。每轮打印进度，并在独立
`artifacts/cache/bilstm-training-*` 保存检查点；patience=2 表示连续两轮 dev F1
未改善即停止。toy 测试验证相同 seed 的初始化与训练历史完全一致。

开发验收发现并修复两处 adapter bug：全 dev 一次解码造成不必要的内存峰值；
seed 原先晚于权重初始化。未完成的诊断训练均在 official-test 推理之前停止，
没有作为正式模型/Run 保存。

## 课件对照与选做边界

用户提供的第三章表 3-3、第四章 4.7.3 数值保存在
`configs/extension_course_references.yaml`，只作结果 sanity comparison。
不预设二阶更好：课件一阶 F1 79.42、二阶 79.16，OOV-R 二阶稍高。

NLP4J 未选做：核验的 [Emory NLP4J](https://github.com/emorynlp/nlp4j)
需要 JVM 工具链，未找到能低成本直接匹配当前中文 MSR BMES 任务的模型。
同名 [org.nlp4j](https://github.com/oyahiroki/nlp4j) 主要列出英文/日文组件。
BiLSTM-CRF 已覆盖指导书的一个现代选做项，当前不再增加模型。

## E7：最终结果与冻结验收（2026-09-27）

| 方法 | P | R | F1 | OOV-R | IV-R | Tag Accuracy | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Longest Matching | 91.72% | 95.67% | 93.65% | 2.47% | 98.21% | — | 100.00% |
| Bigram | 92.07% | 96.68% | 94.32% | 2.47% | 99.25% | — | 100.00% |
| HMM-1 MLE | 78.69% | 79.38% | 79.03% | 35.42% | 80.57% | 81.00% | 98.92% |
| HMM-1 Smooth | 78.79% | 80.45% | 79.61% | 38.92% | 81.58% | 82.17% | 100.00% |
| HMM-2 MLE | 78.79% | 79.08% | 78.93% | 36.23% | 80.24% | 80.95% | 98.92% |
| HMM-2 Smooth | 78.92% | 80.16% | 79.53% | 39.91% | 81.25% | 82.14% | 100.00% |
| BiLSTM-CRF | 94.60% | 94.22% | 94.41% | 67.48% | 94.95% | 95.39% | 100.00% |

课件参考值来自 `configs/extension_course_references.yaml`，只作 sanity comparison，不用于 test 调参。

| 方法 | 课件 P/R/F1 | 课件 OOV-R/IV-R | 本项目 F1 差（百分点） |
| --- | --- | --- | ---: |
| Longest Matching | 89.41 / 94.64 / 91.95 | 2.58 / 97.14 | +1.70 |
| Bigram | 92.38 / 96.70 / 94.49 | 2.58 / 99.26 | -0.17 |
| HMM-1 MLE | 78.49 / 80.38 / 79.42 | 41.11 / 81.44 | -0.39 |
| HMM-1 Smooth | 78.49 / 80.38 / 79.42 | 41.11 / 81.44 | +0.19 |
| HMM-2 MLE | 78.34 / 80.01 / 79.16 | 42.06 / 81.04 | -0.23 |
| HMM-2 Smooth | 78.34 / 80.01 / 79.16 | 42.06 / 81.04 | +0.37 |

最长匹配与 Bigram 使用训练词典、单字回退，OOV 召回有限；Bigram 的统计消歧改善 IV 召回。
二阶 HMM 的状态组合更稀疏；阶数增加不保证 F1 提升，平滑改善 coverage 与 OOV 召回。
词典构造、平滑/UNK、初始化、BMES 约束与课件 HanLP 版本可能解释合理差异。
所有 final 模型的配置在 official raw test 推理之前确定，未按 test 分数调整。

### 正式 Run 与运行时间

| 新增方法 | final Run | 训练秒数 | 3985 句推理秒数 |
| --- | --- | ---: | ---: |
| longest | `20260926_160759_676a087e` | 2.16 | 5.01 |
| bigram | `20260926_161208_5bfe382f` | 1.51 | 8.75 |
| hmm2-mle | `20260926_161715_d8132887` | 6.56 | 2.12 |
| hmm2-smooth | `20260926_161715_5a7e4a3c` | 6.61 | 2.12 |
| bilstm-crf | `20260926_185552_0ffff5d1` | 1994.55 | 19.26 |

时间为本机 CPU 实测；推理不包含官方 Perl scorer、图表和文件导出。

### BiLSTM-CRF 训练协议结果

development Run：`20260926_182152_d9bcee57`。使用现有 78,220/8,698 的 deterministic train/dev；
best epoch=10，dev F1=95.0573%。随后用 86,918 条完整 training 从头训练固定 10 轮，
未在 final 阶段做 early stopping，也未查看 test 选 epoch。final 参数数 926,876，训练字符词表 5,167；
CPU 8 threads，Python 3.11.16、PyTorch 2.14.0+cpu、pytorch-crf 0.7.2、seed=42。
final 训练 1,994.55 秒，official raw test 推理 19.26 秒；coverage=100%，无 NoValidViterbiPath。
训练历史包含完整 10 轮 loss 以及选 epoch 的 dev 历史；权重 SHA256 在模型 metadata 与冻结清单中。

### 评分协议交叉验证

沿用原始 official raw test 3985 行，仅移除 CRLF，保留输入中的 ASCII 空格；不以 Gold 构造模型输入。
原始数据的 8 个左引号位于 raw/Gold 的 CRLF 两侧不同，影响 16 行；没有 Unicode 行分隔或空行。
官方评分使用原始 Gold 与逐行 GNU diff；项目 evaluator 使用冻结的 raw 坐标 Gold 词跨度。
官方输出的三位小数与从其 diff 计数恢复的未舍入比值均保留，统一比较表采用后者。
MLE 继续按 abstention 使用既有除零保护副本；未改原 scorer 的匹配规则。

| 方法 | official − project 正确词数 | 16 个边界差异行贡献 | 其余 diff/span 对齐贡献 |
| --- | ---: | ---: | ---: |
| longest | -20 | -8 | -12 |
| bigram | -21 | -8 | -13 |
| hmm2-mle | -7 | -8 | +1 |
| hmm2-smooth | -8 | -8 | +0 |
| bilstm-crf | -7 | -8 | +1 |

BiLSTM-CRF 的项目 F1=94.4140%，官方 F1=94.4074%，差 0.0066 个百分点：边界贡献 −8，
另一个相同行边界案例中官方 diff 比 exact-span 多计 1 个正确词，净差 −7。
逐行差异证据保存在验收目录的 `*_scorer_difference_cases.json`；该诊断只读取已保存输出，
没有重新推理、重训、重新执行 scorer 或调整算法。

### 结果解释

Bigram 比最长匹配 F1 高 0.6657 个百分点，IV-R 高 1.0390；两者 OOV-R 都为 2.4744%，
训练词典加单字回退限制了多字 OOV 的召回。最长匹配比课件 F1 高 1.70 个百分点，
Bigram 比课件低 0.17；词典构建和课件 HanLP 版本不同可能解释偏差，未为参考值调 test。
HMM2 Smooth 相比 HMM1 Smooth：F1 −0.0777、OOV-R +0.9897、IV-R −0.3316 个百分点。
这符合二阶组合更稀疏、OOV 改善而总体 F1 未提高的课件趋势。
HMM2 MLE 3942 句成功、43 句无路径，其中 37 句含 unseen character；Smooth 恢复到 3985 句。
完整分母、successful-only 指标及 BMES/ABSTAIN 混淆矩阵均保存于独立 metrics.json。
BiLSTM-CRF 比 Bigram F1 高 0.0878 个百分点、OOV-R 高 65.0053，但 IV-R 低 4.3001；
上下文神经特征改善 OOV 召回，不能据此宣称所有维度都超过词典语言模型。

### Web 与浏览器验收

保持原有页面设计，增加分词模型选择、七方法 CWS 表和 <=30 code point 的算法过程。
参数页按模型展示词典/二元统计、HMM2 tensor slice、BiLSTM 架构/CRF 转移/训练曲线；
资源不存在时显示 unavailable，默认仍是 HMM1 Smooth。NER 与 MSRA 200 指标独立展示。
顶部原有 Run 控制 NER/原有评价；新增模型使用自己的 final MSR Run，页脚分别标明二者，
参数图导出文件名使用对应分词模型 Run。
Chromium 已完成所有五个交互模型的中文/emoji 验收，以及神经模型空格往返、过程输出、
参数曲线、PNG 导出身份和七行比较表；控制台 0 errors / 0 warnings。
截图：`outputs/browser/extensions-hmm2-trace.png`、`extensions-bilstm-parameters.png`、
`extensions-seven-model-comparison.png`。

### 文件与测试

新增 `src/nlplab/extensions/` 九个模块：共享词典/Bigram、HMM2、神经 adapter、独立 CLI、
benchmark、comparison、trace、integrity 和包入口。新增四个 configs：HMM2 MLE/Smooth、
BiLSTM-CRF、课件参考值，以及本文档和六个 `tests/test_extensions_*.py`。
修改 README、pyproject.toml（可选 sequence-neural）、.gitignore、NLPService、analysis/experiment API、
sequence Jinja2、sequence.js/charts.js 和 API 测试；没有改一阶 HMM、Viterbi 或 MSR importer。
完整 neural 环境 pytest：129 passed，只有原有 Starlette/AnyIO 弃用提醒；Ruff check 通过，
Ruff format --check：82 files already formatted；核心/神经环境 pip check 均无依赖冲突。
测试覆盖词网/平滑归一化、HMM2 长度 1–5 穷举、合法 triple、UNK、持久化、
CRF smoke/变长 mask/确定性初始化与训练、abstention、参考指标单位、API Unicode 与旧模块回归。

### 冻结清单与停止条件

验收输出：`outputs/exp1/extensions/acceptance_20260926_190001/`。
`frozen_integrity.json` 确认既有 264 个 HMM1/MSR/MSRA 冻结文件指纹全部 unchanged；
`extension_freeze.json` 固定新增模型代码、配置、完整模型和 final Run 文件 SHA256。
`artifact_validation.json` 验证五个新 final Run 各 3985 条 official raw 输入、每个成功预测 roundtrip、
测试推理次数为 1、官方 scorer exit=0，并核对保存的 diff 计数。
`verification.json` 保存测试和浏览器验收摘要；`verify_artifacts.py` 仅检查已保存内容。
所有模型、预测、错误、概率/参数、训练历史、配置、输入指纹、官方 stdout/stderr/invocation
与项目 evaluator 输出保存在上述独立 Run；旧 development Run、latest 和冻结 benchmark 没有覆盖。
拓展 final 结果现在冻结。NLP4J 未选做，原因见上文；未开始实验二/三或增加其他模型。
