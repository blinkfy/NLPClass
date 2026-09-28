# 实验一数据与评价口径

状态：Completed / Frozen。CWS 和 NER 分开报告；toy 不作为正式样本或指标。

## CWS

SIGHAN Bakeoff 2005 MSR，standard protocol。七种方法共享完整 official training（86918 句、4050469 字符、2368391 词）及 official raw test（3985 physical lines）。
Longest Matching、Bigram、HMM1 MLE/Smooth、HMM2 MLE/Smooth、BiLSTM-CRF 的正式 word P/R/F1、OOV-R、IV-R 来自已保存官方 scorer。
BMES Tag Accuracy 来自 raw-coordinate 项目 evaluator；Longest/Bigram 新报告值只读派生于保存 prediction tokens 与冻结 Gold alignment，旧 Run 不改。
Coverage 是成功解码句数/3985；MLE 无路径采用既定 abstention，完整 Gold 分母与 successful-only 均保留。
原始 CRLF/ASCII space/16 行边界差异、diff/span 对齐口径见报告资产 scoring_protocol.md；不因差异调整模型。

## NER

MSRA NER test personal subset 200；BIO2 PER/LOC/ORG 转为 Python Unicode code-point [start,end) 实体，并验证 text[start:end]==entity.text。
固定版本 `levow-msra_ner-1.0.0-oye93-934e50bdbbd27f5ffa47eb4eff04651c6eef0a32`；个人 key fingerprint 和 selected IDs 保存于原 selection manifest。
抽样在模型推理前冻结，不根据预测换样本。正式实体评价要求 sample/start/end/type 全匹配，报告 PER/LOC/ORG/Overall P/R/F1。
MSRA-200 没有与当前 MSR CWS 标注规范一致的分词 Gold：不报告 MSRA-200 CWS P/R/F1；HMM 只输出 BMES/segmentation。
CWS 正式指标来自 MSR，NER 正式指标来自 MSRA-200，两者不能混成同一个 benchmark。

## 后续实验数据

THUCNews 官方全文个人平衡子集 600：财经/finance、教育/education、科技/science、体育/sports、娱乐/entertainment，各120。
固定分层 train/val/test=420/90/90，各类84/18/18，ID互斥。实验二使用全部600的 id/text 视图，标签仅供事后评价。
实验二、三模型未实现；本收尾不启动它们。custom corpus / 620-slot / manual annotation 保留为 Optional Workflow。

## BiLSTM 训练预算

在预先固定的 10 epoch 训练预算下，development 最优结果出现在第 10 轮，说明模型仍可能具有进一步训练空间。但 official test 已在配置冻结后完成。为避免 test 后继续调整训练预算，本实验保留原 10 epoch 设置，不再增加训练轮数或重新评价。

max_epochs=10，development best_epoch=10、F1=95.0573%；达到预算而非 patience=2 提前停止。final 固定10轮。
CRF forward/partition/Viterbi 来自 pytorch-crf 0.7.2，不宣称自行实现 CRF。
Bigram：依据课件的二元语法与低阶平滑思想，本项目采用固定插值平滑实现。

## 本次收尾

资产目录：`outputs/exp1/final_report_assets/20260927_010652`。本阶段 training_executed=false、official_test_inference_executed=false、scorer_executed=false。
本次 Web 补丁已验收：Longest 扫描；Bigram 词网、代价与前驱链；HMM1 state-score；HMM2 state-pair/backpointer/A2 slice；BiLSTM 架构、emission、CRF transition 与 development 曲线。截图使用短 demo 的正常交互推理及已保存正式指标；未执行 official-test inference、训练或 scorer。browser_demo_responses.json 保存本次 demo 响应，browser_replay.json 是插入补丁前的只读回放包，未用于本次截图。
