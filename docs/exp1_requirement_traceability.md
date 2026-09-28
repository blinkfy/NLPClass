# 实验指导书要求—证据矩阵

Completed / Frozen；CWS 与 NER 数据及指标分开。

| 实验要求 | 实现 | 数据 | 指标/输出 | 代码证据 | Web/报告证据 |
| --- | --- | --- | --- | --- | --- |
| HMM 原理 | HMMTagger | MSR training | π/A/B 概率与说明 | [HMMTagger](../src/nlplab/sequence_labeling/hmm.py#L44) | hmm1_parameters.json; HMM π/A/B 截图 |
| HMM 序列预测 | log-space Viterbi | MSR raw test | BMES / 保存路径 | [viterbi](../src/nlplab/sequence_labeling/viterbi.py#L19) | report_code_snippets.md; HMM1 路径截图 |
| >=200 中文样本 | test-only personal hash selection | MSRA test personal-200 | 200 unique IDs / selection fingerprint | [select_exp1](../src/nlplab/data/msra_ner.py#L191) | data/course/exp1_msra_200_manifest.json |
| π 构建 | 起始计数与合法 B/S 平滑 | MSR training | 4 维 π | [HMMTagger.fit](../src/nlplab/sequence_labeling/hmm.py#L63) | hmm1_parameters.json |
| A 构建 | 相邻状态计数与行归一化 | MSR training | 4×4 A | [HMMTagger.fit](../src/nlplab/sequence_labeling/hmm.py#L63) | hmm1_parameters.json |
| B 构建 | 状态字符频次、UNK、行归一化 | MSR training | 各状态 Top-K emission | [HMMTagger.fit](../src/nlplab/sequence_labeling/hmm.py#L63) | hmm1_parameters.json |
| BMES 标签 | 词边界 Gold codec | MSR Gold tokens | B/M/E/S | [encode](../src/nlplab/segmentation/bmes.py#L7) | report_code_snippets.md |
| 中文分词结果 | codec roundtrip | MSR raw + MSRA200 input | 分词与 BMES，MSRA 不计 CWS F1 | [decode](../src/nlplab/segmentation/bmes.py#L17) | interactive segmentation 截图; representative_examples.jsonl |
| PER | HanLP NR→PER 显式映射 | MSRA200 | P/R/F1; F1≈96.89% | [HanLPRecognizer.recognize](../src/nlplab/ner/hanlp_adapter.py#L126) | metrics_ner.json; NER metrics 截图 |
| LOC | HanLP NS→LOC 显式映射 | MSRA200 | P/R/F1; F1≈93.25% | [HanLPRecognizer.recognize](../src/nlplab/ner/hanlp_adapter.py#L126) | metrics_ner.json |
| ORG | HanLP NT→ORG 显式映射 | MSRA200 | P/R/F1; F1≈85.33% | [HanLPRecognizer.recognize](../src/nlplab/ner/hanlp_adapter.py#L126) | metrics_ner.json; ner_error_examples.jsonl |
| 准确率/P/R/F1 | 词跨度/实体跨度评价 | MSR 与 MSRA 分开 | Tag Accuracy、word PRF、NER exact-span PRF | [evaluate_segmentation](../src/nlplab/evaluation/segmentation.py#L36) | cws_seven_model_comparison.csv; metrics_ner.json |
| 误差分析 | 保存差异与透明 heuristic | frozen predictions/errors | 25 CWS / >=8 NER 错例 | [error_record](../src/nlplab/evaluation/errors.py#L13) | cws_error_examples.jsonl; ner_error_examples.jsonl |
| 可视化 | 原生 JS + ECharts | saved parameters/trace/metrics | 概率、DP、曲线、comparison | [loadParameters](../app/static/js/charts.js#L114) | screenshots/; browser_replay.json |
| BiLSTM-CRF 选做 | PyTorch + pytorch-crf 0.7.2 adapter | MSR | F1≈94.41%; architecture/history | [BiLSTMCRFSegmenter](../src/nlplab/extensions/bilstm_crf.py#L166) | bilstm_architecture.json; bilstm_training_summary.json |
| 二阶 HMM 课件拓展 | 状态三元组 tensor + pair DP | MSR | MLE/Smooth metrics; A2 slices | [viterbi2](../src/nlplab/extensions/hmm2.py#L25) | hmm2_parameters.json; second-order trace 截图 |
| Longest Matching | 训练词典前向最长匹配 | MSR | F1≈93.65%; dictionary/max-length | [DictionarySegmenter._longest](../src/nlplab/extensions/dictionary.py#L54) | longest_statistics.json; longest trace 截图 |
| Bigram | 词网、固定插值、负对数代价 DP | MSR | F1≈94.32%; predecessor/path | [BigramSegmenter._decode](../src/nlplab/extensions/dictionary.py#L143) | bigram_statistics.json; lattice 截图 |

## 当前 Web 证据

`app/static/js/algorithm-view.js`：scan（最长候选）、lattice（词网/边代价/前驱）、hmm1（state DP）、hmm2（pair DP/backpointer）、neural（emission/CRF path）、renderModelInfo（A2 slice/架构/CRF heatmap/训练曲线）。
API 使用 `app/api/trace_display.py::display_trace` 展示适配，调用原有 Segmenter；没有新增算法。
≤30 code point 展示完整过程，长文本只展示完整分词与模型参数；NER 独立。
截图的交互内容仅来自短 demo，不含 official-test 推理；正式指标读取 frozen Run。
报告根目录：`outputs/exp1/final_report_assets/20260927_010652`；截图索引：screenshots/index.md。
CRF forward/partition/Viterbi 来自 pytorch-crf 0.7.2，未宣称自行实现 CRF。NLP4J 未选做。
