# 实验一正式冻结

## Dataset

MSR：SIGHAN Bakeoff 2005 MSR / standard。URL `http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip`；archive SHA256 `296c54adf4abc3702eca0cbce70b55fe22a344ea02d33175a4d1183e074072b4`。
MSRA：`levow-msra_ner-1.0.0-oye93-934e50bdbbd27f5ffa47eb4eff04651c6eef0a32`；test URL `https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/934e50bdbbd27f5ffa47eb4eff04651c6eef0a32/NER/MSRA/msra_test_bio.txt`；test source SHA256 `2be8e580bcb22e9086e4403d82a7ec0f6e18019ebef35b0fd52941724bdf89a3`。
MSRA-200 selection key fingerprint：`715a974b4f8b38482a567b78ed12e90c3475a706a7e92b0e415ea34ad37cb696`；manifest SHA256 `d810542880417d1d4b7994a5808c700ae4d79c2166a934893c7c6100fbded73a`。
THUCNews：thucnews-official-14 / official-2005-2011，600 manifest SHA256 `23f5a4aa04f65eabe96f3fdd62f32e0aa29e87fe0526ed03cd81a97182992928`。
THUCNews split manifest SHA256 `4dd233114f2192d155ff3eb76267a59d524992f42af189ae34fb8427ea77d200`；split key fingerprint `256e15f547527718f2a7fa29d22b5b7eef67bf3e42571c27ae5efaf4d896bb38`；420/90/90。
原始公开数据、selection 原始键及权重不提交 Git；许可未明确者沿用 provenance 的 unknown。

## Final Runs

| 方法 | Final Run ID |
| --- | --- |
| Longest Matching | `20260926_160759_676a087e` |
| Bigram | `20260926_161208_5bfe382f` |
| HMM-1 MLE | `20260926_130612_32a5e168_mle` |
| HMM-1 Smooth | `20260926_130612_32a5e168_smooth` |
| HMM-2 MLE | `20260926_161715_d8132887` |
| HMM-2 Smooth | `20260926_161715_5a7e4a3c` |
| BiLSTM-CRF | `20260926_185552_0ffff5d1` |
| MSRA NER personal-200 | `20260926_152118_34a953fb` |

## Dependency versions

Python `3.11.16`，NumPy `1.26.4`，PyTorch `2.14.0+cpu`（CPU），pytorch-crf `0.7.2`，HanLP `2.1.5`。
scorer 环境来自已保存 runtime_provenance：Perl 5.38.2，GNU diffutils 3.12；本阶段未执行 scorer。
Perl executable SHA256 `ba92b05b95b2a8c70534b7517d23623332e15b3395641c30ecc1deffc1ed5f27`；diff executable SHA256 `d64621f766d8910a92cb3d99bd7f23c0b1c736eece0e19dccbccead93aa81854`。

## Freeze statement

实验一正式冻结。后续实验二、实验三开发不得根据实验一 official-test 结果重新调整实验一模型、超参数、数据子集或评价协议。

原有264文件与拓展90文件在收尾开始前均 unchanged；本次 begin/end snapshot 保护全部模型、Run、预测、metrics、scorer输出和 selected200。结束报告见 closeout_hash_validation.json。
只读报告资产：`outputs/exp1/final_report_assets/20260927_010652`；资产清单：asset_manifest.json。
training_executed=false；official_test_inference_executed=false；scorer_executed=false。

在预先固定的 10 epoch 训练预算下，development 最优结果出现在第 10 轮，说明模型仍可能具有进一步训练空间。但 official test 已在配置冻结后完成。为避免 test 后继续调整训练预算，本实验保留原 10 epoch 设置，不再增加训练轮数或重新评价。
