# 实验三 Stage 3：word-level TextCNN

本阶段研究 420 篇小样本训练下，随机、可训练词向量与局部卷积组合相对稀疏线性分类的表现；不以超过 C2 为目标。C1/C3 共享冻结 HMM-1 Smooth 与相同 NFKC/lowercase/token 清洗，但表示、优化过程、截断、分类器均不同，不是单变量实验。

Stage1 `20261001_123337_9c1e7341` 与 Stage2 `20261001_133829_68ee50d7` 的资产、预测、指标及两个 latest 指针保持冻结。该 Test 已被历史实验评价，C3 是后续扩展实验；本阶段没有根据历史 Test 错误设计超参数，也不把这个已使用 Test 描述成从未见过的研究基准。

## 协议与运行时

沿用 THUCNews 600 固定 420/90/90 split，每类 84/18/18。阶段 A 词表只拟合 Train：PAD=0、UNK=1、min_freq=2，无 max_vocab，按频率降序/同频字典序编号。预处理完全复用 C1：NFKC、lowercase、冻结 HMM 分词及至少 2 字中文/英文 token 清洗；无停用词、外部语料、主题词典或预训练向量。

max_len 只从 Train 长度的线性插值 P95 得到：`max(128,min(768,ceil(P95/32)*32))`。短序列右 PAD，长序列取前 ceil(max_len/2) 与后 floor(max_len/2)。阶段 B 重新 fit 510 篇词表，仍使用阶段 A 的 max_len，不根据 Validation 改长度。编码保存原 token 位置；头尾拼接处跨越原文删除区域的卷积片段明确标记 `crosses_truncation_join`。

两套已有环境保持不变：exp2-venv Python 3.11.16 / sklearn 1.9.1 负责原有分词、数据与统一评价；hanlp-venv Python 3.11.16 / torch 2.14.0+cpu 负责神经数值训练/推理。后者没有 sklearn/scipy，前者没有 torch，故以 JSON/NPZ 数值 job 通信，未安装、升级或降级依赖。CPU、2 threads、固定 seeds 与 deterministic algorithms；无 CUDA。

算法新增 `features/word_sequence.py`、`classification/textcnn.py`、`textcnn_training.py`、`stage3_neural.py`、`stage3_config.py`、`stage3_experiment.py`、`stage3.py`；复用原 RunStore、冻结数据加载器、HMM/预处理、完整分类指标和 artifact hash 校验。没有另建平行平台或在 Web 中实现训练。

## 网络、训练与选择

固定 trainable random Embedding 128D → Conv1d 三路 kernel=3/4/5、各128 channels → ReLU → global max → 384D → Dropout(0.5) → Linear(384,5)。输出 raw logits。仅完整非 PAD 窗口参与最大池化；某序列短于对应 kernel 时该路 pooled activation=0、argmax=-1，避免 bias 或 PAD 混合窗口产生伪响应。

CrossEntropyLoss，无 class weight/label smoothing；AdamW、weight_decay=1e-4、batch=32、max_epochs=30、gradient_clip_norm=5。无 scheduler、augmentation 或结构搜索。唯一候选 lr=0.001/0.0005，各 seed42/43/44。每 epoch 真实 Validation 评价；Macro-F1 严格增加才替换 best，并列保留最早 epoch，连续5次未增加停止。

神经进程仅用固定5类 confusion counts 计算 early stopping Macro-F1；协调器对每 epoch 的预测重新使用现有 `classification_metrics` 生成完整指标，并核验该标量一致。这是每 epoch 的一次 Validation 预测记录，不是重复 Test。

lr 按三个 seed 最佳 Validation Macro-F1 的 mean 降序、population std(ddof=0) 升序、lr 升序选择。final_epochs 为选中 lr 三个 best epoch 中位数，final seed固定42，不选最佳 seed。阶段 B 新初始化、510 篇词表与序列，训练恰好 final_epochs，无 early stopping。模型与 metadata 保存后才打开 Test，正式评价一次。

## 保存与解释

每个模型 state_dict 保存为 JSON metadata + NPZ float32 数值数组，load 使用 allow_pickle=False，严格核验键、shape、dtype、架构与参数数目；未使用 torch.save/pickle。保存预注册文件及其 SHA256、代码/数据/split/tokenizer hashes、六组 epoch histories 和最佳 checkpoint、length/vocab/max_len freeze、selection freeze、final fit metadata、逐篇 logits/softmax/通道响应、错误与两组配对。独立 `stage3_latest.json` 和 exclusive 执行 guard，原指针不更新。

解释为每通道最大响应位置对应的 token span：`logit_c = bias_c + Σ max_activation_j × classifier_weight_cj`。保存所有384通道、最大位置、kernel/filter编号、原/模型 token（UNK明确显示）、原文位置、weight 和 contribution。Top spans 按正直接贡献排序，完整通道和加 bias 还原 raw logit；不是 attention、因果解释或模型“真正关注”。Softmax 输出未校准，不表示经验正确率。

## 正式记录

Run：`20261001_153153_c99a8d6f`。preregistration 的 opened_splits 只有 Train，先落盘词表/长度/代码与环境哈希，再开始 formal。Train 长度：min=17、mean=324.4429、median=258.5、P90=625.2、P95=843.45、P99=1319.88、max=2329。P95 向32取整为864，经768上限得到 frozen max_len=768。

| 阶段 | vocabulary（含 PAD/UNK） | 全文 UNK / tokens | UNK rate | 截断篇数 | 截断率 | retained-token ratio |
|---|---:|---|---:|---:|---:|---:|
| Train 420 | 12077 | 16955 / 136266 | 12.4426% | 29 | 6.9048% | 93.4496% |
| Validation 90，Train vocab transform | 12077 | 6752 / 26079 | 25.8906% | 4 | 4.4444% | 97.9332% |
| Final Train+Validation 510 | 13721 | 19169 / 162345 | 11.8076% | 33 | 6.4706% | 94.1698% |
| Test 90，final vocab transform | 13721 | 6595 / 29961 | 22.0119% | 7 | 7.7778% | 92.3434% |

UNK rate 分母是未截断全文的 token 次数，不含 PAD；另保存保留段 UNK rate：Train12.7250%、Val25.9397%、FinalTrainVal12.0434%、Test22.5106%。各篇长度/UNK/截断均可追溯。Stage A trainable parameter count=1,744,773，final=1,955,205；变化仅由允许的词表重新 fit 产生。

| learning rate | seed | best epoch | best Validation Macro-F1 | 实际 epochs | 训练 runtime 秒 |
|---|---:|---:|---:|---:|---:|
| .001 | 42 | 10 | 90.1686% | 15 | 55.12 |
| .001 | 43 | 12 | 93.3486% | 17 | 64.33 |
| .001 | 44 | 12 | 92.2838% | 17 | 61.60 |
| .0005 | 42 | 12 | 89.1459% | 17 | 67.15 |
| .0005 | 43 | 10 | 93.3145% | 15 | 56.53 |
| .0005 | 44 | 11 | 91.2364% | 16 | 58.94 |

lr=.001：mean=91.9337%、population std=1.3216个百分点；lr=.0005：mean=91.2323%、std=1.7018个百分点。选择 .001。对应 best epochs=[10,12,12]，中位数12，因此新的 final 模型 seed42 恰好训练12 epochs，无 Validation/early stopping。最佳 seed43 没有作为 final seed。

## Test 评价与冻结基线

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| C0 frozen NB | 92.2222% | 92.1667% | 92.2222% | 92.0846% |
| C1 frozen word SVC | 94.4444% | 94.8947% | 94.4444% | 94.5252% |
| C2 frozen char χ² SVC | 96.6667% | 96.7836% | 96.6667% | 96.6313% |
| C3 word TextCNN | 91.1111% | 92.4358% | 91.1111% | 91.0310% |

C3 正确82/90，Micro F1=91.1111%、Weighted F1=91.0310%。C0/C1/C2 全部指标直接复制冻结 Stage2 comparison_metrics，未训练、未重评。C3 Test 正式评价 count=1；后续展示、配对和诊断只读预测资产。

| C3 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 财经 | 94.4444% | 94.4444% | 94.4444% | 18 |
| 教育 | 100% | 72.2222% | 83.8710% | 18 |
| 科技 | 89.4737% | 94.4444% | 91.8919% | 18 |
| 体育 | 100% | 94.4444% | 97.1429% | 18 |
| 娱乐 | 78.2609% | 100% | 87.8049% | 18 |

固定顺序财经/教育/科技/体育/娱乐，行 gold、列 prediction：

```text
17  0  0  0  1
 1 13  2  0  2
 0  0 17  0  1
 0  0  0 17  1
 0  0  0  0 18
```

教育5篇错误：1→财经、2→科技、2→娱乐。娱乐接收5篇其他类样本，形成23个娱乐预测，虽然Recall100%，Precision仅78.26%。没有据此改 class weight、网络、词表或 epoch。

| 配对 | 两者都对 | 仅旧模型对 | 仅C3对 | 两者都错 | disagreement |
|---|---:|---:|---:|---:|---:|
| C1/C3 | 81 | 4 | 1 | 4 | 7 |
| C2/C3 | 82 | 5 | 0 | 3 | 7 |

## 样本、来源线索与局部组合

C3 唯一修正的 C1 错误：`THU-0fd1c56dc512bcdf68c6c908`，娱乐类电影人眼球摄像头，C1→科技、C3→娱乐。“拍摄 画面 传输 电脑”(kernel4，direct contribution .183)、“电池 一个 可以 拍摄 画面”(kernel5，.169)提供局部组合响应。C2 已正确，故 C2 corrected 组为空。

C2 原本正确、C3 新错5篇（完整 IDs/原文/两模型输出见 paired_c2_c3.json）：

- `THU-2a909c0b5de227fed5ab8214`，央行纪念币：财经→娱乐；“相关 报道 新版” .187、“定于 发行 中国 改革 开放” .181 等响应。不能因为单个Top片段具有语义就认定整体分类合理。
- `THU-d18289d57f3c13904bd0efec`，留法学生租房受骗：教育→娱乐；“近日 一留 学生” .190、“此类 诈骗 案件” .152。共享 HMM 的切词仍把留学生拆开，没有恢复 C2 的“留学”字符证据。
- `THU-2be6ee560350da95111b650f`，加拿大移民就业：教育→财经；“经济 衰退 劳动力 市场” .350，跨主题就业/经济内容主导。
- `THU-a550e3b1fc9b33eb5e7c05e3`，教育部商务部共建大学：教育→科技；“支持 商务部 人才 培养” .194 等被学习为支持科技的通道响应。
- `THU-21586cca637d6558f70f9fa9`，王治郅/李楠恢复训练：体育→娱乐；“李楠 何时 回归 成为 迷们” .157，人物伤情与娱乐报道语言相似。

C2/C3 共同错误仍为大学亿万富翁（教育，C2科技/C3娱乐）、薪资排行（教育，C2财经/C3科技）、佳能DV电影幕后（科技，均娱乐）。C3 没有解决 C2 留下的三个主题边界错误。C1/C3共同错误另外含纪念币与留学生租房，而薪资排行在C1中正确。

正确样本的局部响应有合理主题线索：财经 `THU-893d00a4ca690efe5a9e0853`“证券 投资 基金 股票 投资”贡献.399；体育 `THU-c45ef026310c95107422d5c8`“全国 体育 大会 担当”贡献.567。这展示了可学习3–5词局部组合和最大响应，与 C1 的词/二元组加法权重不同，但不是因果或语义理解证据。

来源捷径实际存在。90篇中31篇的正贡献Top12 spans含“新浪”。科技 `THU-c5ec39d5007922bbd6438379` 的“厦门 新浪 科技 上午”贡献.296、“新浪 科技 上午” .291。多篇教育新闻重复尾部免责声明“不断 调整 变化 新浪 网所”(kernel5，.4121)、“新浪 网所 提供 所有 考试”(.3382)。head-tail 保留了这些来源/模板文字；此为配置的真实局限，未在 Test 后去除。

90篇中76篇Top12 spans包含至少一个模型 `<UNK>`；界面同时列出原 tokens 与模型实际 tokens，防止把不同未知词共享的同一嵌入解释为已学会具体名称。例如娱乐产子新闻“妇产 科剖 腹产 下女 宝宝”的前四词实际都为UNK。因此Top spans可审计响应位置与分数，却不应过度作词义解释。Top12 中没有跨head-tail拼接的片段；全通道仍保留原位置和拼接标记。

## 曲线、耗时与结论

六组 curves 全部来自 Stage A 真实记录，可在 Web 切换。选中lr seed42：Train loss从1.9622降到best epoch10的.1175，最后epoch15为.0583；best时Train在线accuracy97.86%，Validation Macro-F1 90.17%，停止时Validation F1回落87.76%。另外两个seeds的终点train loss .0504/.0345，Validation loss .3573/.3489。存在明显训练/泛化差距与F1峰后波动，符合过拟合倾向；Validation loss总体继续下降，没有捏造持续loss上升。

六次 Validation 数值训练合计363.68s，final训练48.83s，整个formal阶段451.15s（含子进程加载/评价/保存）。90篇数值推理与解释合计.5833s；Test推理进程5.3432s含PyTorch启动和数据加载。preregistration/tokenization与浏览器QA不计入formal时间；这些是本机CPU单次测量，不作为通用速度排名。

在本数据与固定协议下，随机词向量TextCNN没有优势：比C1净少3篇正确，比C2少5篇。它确实产生不同的局部组合响应，但小样本、高UNK率、主题混合与来源模板线索限制了泛化，不能据此声称TextCNN普遍不适用。当前没有仅为提高同一已使用Test分数而继续新增模型的必要；应保留冻结结果用于课程原理对比。新stage若将来开展，需有预先明确的新问题与评价边界，本轮没有自动新增模型或Lab4。


## 最终验证与冻结审计

exp2 分类回归26 passed（18个既有测试、8个Stage3测试）；hanlp神经toy测试10 passed；新增测试Ruff通过。覆盖词表/长度/split边界、PAD/UNK、head-tail、有效卷积窗口、argmax映射、贡献和还原logit、softmax、确定性、early stopping、lr/epoch选择、NPZ保存加载/tamper、配对、历史hash和API schema。toy训练与正式6/1/1计数分开。

Chromium 153.0.8010.12 在 http://127.0.0.1:8000/classification 验证1440×1000桌面、390×844手机，沿用现有服务和Python Playwright。Browser插件/skill与package.json不可用，使用已有Playwright，未安装依赖。四个Tab、C3新文本推理、五类Softmax/raw logits/卷积片段、六组真实曲线切换、四模型指标、配对和错误展开通过。筛选计数8/7/7/0/5；Stage1禁用C2/C3、Stage2禁用C3，历史指标和保存预测正确。页面无横向溢出，console/page/HTTP errors为0。只验收相关关键路径，未扩展到其他浏览器或任意视口。

最终审计10/10通过：1003基线文件无缺失，999不变；获准修改为共享API、classification模板、classification-page.js，另有自动生成API Python缓存（非冻结资产，单独列明）。Stage1/Stage2/Exp1/Exp2模型、Run、metrics、predictions、latest和固定数据均未变。新Run manifest84/84、preregistration source12/12、HMM2/2哈希通过；guard完成且禁止再次formal执行。验收记录位于outputs/exp3/quality/20261001_153153_c99a8d6f.json，不改动冻结Run。

外部截图、详细检查与before/after hashes：C:/Users/29513/.codex/visualizations/2026/10/01/01a0f750-fe2e-7b83-a6c6-d991ff6ccbab/stage3/final_qa_receipt.json。初次脚本的异步读取/selector断言错误已用窄复核纠正；最终receipt区分被替代的脚本断言和实际应用结果。

```text
formal_exp3_stage3_validation_training_runs=6
formal_exp3_stage3_final_fit_executed=true
formal_exp3_stage3_test_evaluation_count=1
stage1_modified=false
stage2_modified=false
exp1_modified=false
exp2_modified=false
lab4_started=false
```
