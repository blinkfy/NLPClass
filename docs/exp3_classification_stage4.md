# 实验三 Stage4：冻结方法的开发集学习曲线与样本效率

完成日期：2026-10-02（Asia/Taipei）。独立 analysis Run：`20261001_164602_48e6c6b6`，目录 `outputs/exp3/stage4/20261001_164602_48e6c6b6/`。

这是 post-hoc development-set learning-curve analysis，不是新模型选择或新 Test benchmark。Validation 90 已在历史阶段参与超参数选择，因此这里仅解释冻结方法在不同训练规模下的开发集性能趋势，不能作为新的独立泛化估计。未访问 Test90 正文、标签、历史 Test预测/指标；formal_test_evaluation_count=0。

## 引用冻结配置与工程复用

引用 Stage1 `20261001_123337_9c1e7341`、Stage2 `20261001_133829_68ee50d7`、Stage3 `20261001_153153_c99a8d6f` 的 config.json 与 selection_freeze.json 路径及 SHA256，保存于 frozen_config_references.json。运行时直接读取引用，不复制修改旧配置；旧模型、Run、predictions、metrics、latest 均不写入。

C0 alpha=.5；C1 word uni+bi TF-IDF（min_df2/max_df.3/max_features20000/sublinear/smooth/L2），LinearSVC C=2；C2 原 char analyzer、Count min_df2/max_df1、χ² target10000、原TF-IDF、SVC C=2；C3原HMM token清洗、min_freq2、max_len768、128D embedding、kernels3/4/5各128、Dropout.5、AdamW lr=.001/wd1e-4/batch32、12epochs、seed42、CPU2threads/deterministic。

每个 subset 从头拟合表示：C0/C1共享同一word matrix对象，C2重新拟合Count/χ²/IDF，C3重新fit词表并随机初始化。C3始终12epochs，不早停、不选最佳epoch、不重新计算max_len；训练完成后才进行一次Validation forward pass。Train指标用最终模型eval模式计算，区别于训练中dropout下的在线accuracy。

复用已有TextTFIDF、CharacterCounts/Chi2TFIDF、线性分类器、WordSequenceVocabulary、TextCNN/train_textcnn、classification_metrics、RunStore基础目录与安全数值保存。新增development-only loader、stage4_analysis/stage4/stage4_neural/stage4_finalize、只读service/API和独立UI模块。算法层没有FastAPI依赖，Web没有训练路由。

旧FrozenClassificationData构造会hash Test，因此未调用。新DevelopmentClassificationData只核验/解码Train、Val；I/O audit guard禁止Test split、600全文文件与旧Test资产。共享split manifest只提取Train/Val元数据，不打开Test文件。CLI和数值worker不接收Test输入。

## 三个nested ladders

subset seeds=101/202/303，各类按 SHA256(f"{seed}:{document_id}")排序，取16/32/48/64/84前缀。拟合顺序统一为冻结Train manifest顺序；避免全量420虽集合相同却因排序不同改变SGD。15组真实document IDs、每类prefix、ordered IDs SHA256完整保存在subset_ladders.json，没有重新划分。

| 每类数量 | 总Train数 | ladder数 | 固定Validation |
| --- | --- | --- | --- |
| 16 | 80 | 3 | 90（每类18） |
| 32 | 160 | 3 | 90（每类18） |
| 48 | 240 | 3 | 90（每类18） |
| 64 | 320 | 3 | 90（每类18） |
| 84 | 420 | 3 | 90（每类18） |

每个ladder严格D80⊂D160⊂D240⊂D320⊂D420，完全平衡；三个420 ID列表和顺序相同，等于原Train420。sanity_420.json验证四模型的Train/Val预测、指标与数值参数签名全部一致。420 std=0来自同一输入与固定seed，不代表泛化不确定性为0。

## Validation Macro-F1

单位为百分比；±为三个ladder的population std（ddof=0），不是置信区间，无显著性检验。

| Train size | C0 | C1 | C2 | C3 |
| --- | --- | --- | --- | --- |
| 80 | 85.9214% ± 3.5796 pp | 89.9770% ± 2.1872 pp | 90.0505% ± 2.8790 pp | 69.6624% ± 4.0959 pp |
| 160 | 91.2710% ± 2.7241 pp | 93.9745% ± 1.0710 pp | 91.3368% ± 1.8030 pp | 80.9890% ± 3.1653 pp |
| 240 | 91.6014% ± 2.2395 pp | 96.6624% ± 1.8161 pp | 96.2819% ± 2.2839 pp | 85.9927% ± 4.5182 pp |
| 320 | 93.5480% ± 1.0738 pp | 97.3928% ± 1.3859 pp | 96.2823% ± 1.4029 pp | 90.1588% ± 3.1724 pp |
| 420 | 94.2828% ± 0.0000 pp | 97.7424% ± 0.0000 pp | 98.8880% ± 0.0000 pp | 86.6184% ± 0.0000 pp |

## Train–Validation gap

下表为mean Train Macro-F1 − mean Validation Macro-F1，单位pp。完整Train Accuracy/Macro-F1、Validation Accuracy/Macro P/R/F1、每类P/R/F1及confusion matrix保存于60个point result和raw_results.json。

| Train size | C0 gap | C1 gap | C2 gap | C3 gap |
| --- | --- | --- | --- | --- |
| 80 | 14.0786 | 10.0230 | 9.9495 | 30.3376 |
| 160 | 7.8956 | 6.0255 | 8.6632 | 19.0110 |
| 240 | 7.7041 | 3.3376 | 3.7181 | 14.0073 |
| 320 | 5.7228 | 2.6072 | 3.7177 | 9.8412 |
| 420 | 5.0029 | 2.2576 | 1.1120 | 13.3816 |

C1/C2/C3各点最终Train Macro-F1均100%；C0各规模均值为80篇:100.0000%, 160篇:99.1666%, 240篇:99.3055%, 320篇:99.2708%, 420篇:99.2857%。C3 gap前段收窄，但420重新扩大。

## 表示诊断

C3词表包含PAD/UNK。UNK rate定义为Validation原始token sequence中未知token出现次数/全部token次数，不含PAD。

| Train size | vocab mean ± std | Val UNK rate mean ± std | parameter count mean ± std |
| --- | --- | --- | --- |
| 80 | 3,203.0 ± 198.1 | 45.6689% ± 0.8749 pp | 608,901.0 ± 25,353.9 |
| 160 | 5,837.7 ± 122.5 | 36.1312% ± 0.2931 pp | 946,138.3 ± 15,683.8 |
| 240 | 7,874.3 ± 81.7 | 31.8046% ± 0.1980 pp | 1,206,831.7 ± 10,455.5 |
| 320 | 9,911.7 ± 110.3 | 28.5734% ± 0.1680 pp | 1,467,610.3 ± 14,117.3 |
| 420 | 12,077.0 ± 0.0 | 25.8906% ± 0.0000 pp | 1,744,773.0 ± 0.0 |

Validation tokens固定，因此所有C3点截断4/90篇（4.4444%），retained-token ratio=97.9332%。Train的截断和保留率、全文/保留段UNK counts、逐文长度均保存。词表增长改变可训练embedding行数和参数量；这是完整frozen pipeline曲线，而非一个固定维度网络的纯样本数实验。

C1覆盖率为Validation抽取word unigram/bigram候选出现次数中命中当前subset fitted vocabulary的比例。

| Train size | C1 dimension mean ± std | Val occurrence coverage mean ± std |
| --- | --- | --- |
| 80 | 2,465.7 ± 127.4 | 24.9046% ± 0.2452 pp |
| 160 | 5,207.7 ± 125.5 | 30.6887% ± 0.1921 pp |
| 240 | 7,851.0 ± 179.2 | 33.6906% ± 0.0847 pp |
| 320 | 10,620.7 ± 65.3 | 35.7520% ± 0.1239 pp |
| 420 | 14,000.0 ± 0.0 | 37.8179% ± 0.0000 pp |

C2的raw coverage对Count词表，selected coverage对χ²后词表；word/char候选单位不同，不能比较比例推断性能。

| Train size | raw dimension mean ± std | effective/selected dimension mean ± std | selected coverage | raw coverage |
| --- | --- | --- | --- | --- |
| 80 | 5,586.7 ± 371.8 | 5,586.7 ± 371.8 | 41.6048% ± 0.8376 pp | 41.6048% ± 0.8376 pp |
| 160 | 11,972.3 ± 378.0 | 10,000.0 ± 0.0 | 48.4327% ± 0.0706 pp | 52.5790% ± 0.4594 pp |
| 240 | 17,662.3 ± 499.3 | 10,000.0 ± 0.0 | 45.5706% ± 0.7730 pp | 58.2388% ± 0.2498 pp |
| 320 | 23,411.7 ± 303.4 | 10,000.0 ± 0.0 | 44.4795% ± 0.1114 pp | 62.1890% ± 0.1578 pp |
| 420 | 29,927.0 ± 0.0 | 10,000.0 ± 0.0 | 43.7323% ± 0.0000 pp | 65.8420% ± 0.0000 pp |

80篇三个raw/effective k实际值：seed101=5650, seed202=5103, seed303=6007，均低于10000，已明确采用raw dimension，不声称10000。160及以上effective k均10000。所有稀疏点Train/Val零向量均0；shape、nnz、sparsity和coverage counts/types均保存。selected coverage后段下降而F1仍提高，说明覆盖率本身不能充当性能指标。

## 每类趋势

以下为Validation F1的三ladder均值；每类std另保存在per_class_curves.json，并可在Web展开。

| 模型 | 类别 | 80 | 160 | 240 | 320 | 420 |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 财经 | 89.65% | 90.59% | 90.04% | 91.54% | 94.74% |
| C0 | 教育 | 90.92% | 93.83% | 92.08% | 94.54% | 97.30% |
| C0 | 科技 | 68.74% | 81.40% | 82.41% | 86.22% | 87.50% |
| C0 | 体育 | 94.71% | 98.04% | 99.05% | 99.05% | 97.14% |
| C0 | 娱乐 | 85.59% | 92.49% | 94.44% | 96.39% | 94.74% |
| C1 | 财经 | 89.65% | 93.01% | 95.44% | 99.10% | 100.00% |
| C1 | 教育 | 94.68% | 97.25% | 96.30% | 97.25% | 97.30% |
| C1 | 科技 | 76.53% | 85.94% | 94.28% | 95.13% | 94.12% |
| C1 | 体育 | 97.14% | 98.04% | 99.05% | 99.05% | 100.00% |
| C1 | 娱乐 | 91.89% | 95.63% | 98.25% | 96.44% | 97.30% |
| C2 | 财经 | 91.36% | 90.46% | 97.25% | 98.20% | 100.00% |
| C2 | 教育 | 94.83% | 95.44% | 95.40% | 95.40% | 100.00% |
| C2 | 科技 | 77.79% | 81.71% | 93.38% | 94.34% | 100.00% |
| C2 | 体育 | 95.23% | 98.04% | 98.04% | 97.09% | 97.14% |
| C2 | 娱乐 | 91.04% | 91.04% | 97.34% | 96.39% | 97.30% |
| C3 | 财经 | 78.04% | 87.47% | 92.85% | 97.25% | 92.31% |
| C3 | 教育 | 71.35% | 89.77% | 92.31% | 93.16% | 94.12% |
| C3 | 科技 | 51.81% | 65.17% | 73.16% | 82.97% | 76.19% |
| C3 | 体育 | 75.27% | 87.75% | 92.15% | 89.18% | 97.14% |
| C3 | 娱乐 | 71.85% | 74.78% | 79.49% | 88.25% | 73.33% |

科技是增长最明显的类别之一：C0 68.74→87.50%、C1 76.53→94.12%、C2 77.79→100%。C3科技51.81→320时82.97%，420回落76.19%；娱乐320为88.25%、420降至73.33%，是最后一步Macro-F1回落的重要组成。C3教育71.35→94.12%和体育75.27→97.14%总体改善，但各点并非全部单调。

## 描述性观察

C1较早趋缓：240/320/420为96.66/97.39/97.74%；只能称后段增益变小，5个点不足以确定永久平台。C0主要增益来自80→160，但后段仍上升。C2在240→320近乎持平，420又提高到98.89%，不宜称已稳定平台。

低样本80下C2 90.05%、C1 89.98%，仅差0.07pp、远小于ladder波动；没有稳定的char优势。160–320的均值C1高于C2，420则C2更高。NB在80下85.92%，低于两种SVC，但80→160改善5.35pp、分类器拟合开销小；本数据不支持NB低样本性能优于SVC的结论。

C3 80→320从69.66%提高到90.16%，同期UNK下降、词表增长，表现出描述性关联；420仍UNK下降，却F1回落3.54pp、gap由9.84扩大到13.38pp。故C3到420并没有明显向上，也不是随数据量持续改善。数据仅支持“更多数据可能帮助”的待验证假设，不能宣称因果或必然效果。子集内容、表示词表、参数数目与初始化对应都随规模变化，而Validation早已用于选择冻结超参数。本轮没有根据曲线重选模型/epochs或开展新stage。

## 中断恢复与运行时间

首次controller import遭遇WinError1450，在execution guard/任何fit前失败。随后正常执行保留59个成功点；最后C3 worker非零退出，仅留下空输出目录、无checkpoint，不能断言训练从未开始，确切退出码未持久化。一次防覆盖调用在mkdir处被拒绝、未训练。恢复在C3_recovery新目录使用同一subset词表/编码、seed42和12epochs，新初始化模型；15个成功C3模型才进入曲线。保留中断目录与两个job，recovery_record.json明确1个中断尝试和1个训练前拒绝调用，没有静默增加replicate或替换成功结果。

I/O guard原先因latest包含test子串误挡Stage4-only metadata。只增加stage4_元数据例外，真实Test/full600/旧Test路径仍禁止；source_amendment.json记录唯一源码修复的before/after hash，original_development_guard.py可还原预注册SHA。stage4_finalize只读取已有结果、做sanity/聚合/发布，不训练。训练和表示配置始终未变。

成功C0/C1/C2分类器拟合合计分别0.0391/0.1328/0.1655s；C3成功15次训练合计406.74s。共享word表示15次fit合计3.28s。整体wall=1171.45s（execution guard开始至恢复/聚合完成，含中断和恢复等待）。每点表示/模型fit与Train/Val inference时间在runtimes.json；恢复点原表示fit计时未持久化，保留null而未伪造0。失败尝试时间不计入成功模型训练总和，wall包含它们。本机资源压力与进程加载影响wall，不作通用速度排名。

## Web、测试与冻结完整性

模型对比Tab新增样本效率区域，默认四模型Validation mean±population std；单模型显示Train/Validation，附C3词表/UNK/参数及C1/C2表示维度/coverage。原分类Run selector不加入analysis，不新增C4或Tab，不更新Stage1/2/3 latest；仅新增stage4_latest.json。

本轮浏览器使用/classification?analysis=stage4独立开发集模式，跳过旧overview/predictions初始化，旧Test面板不显示且旧classifier/Run控件只读；普通分类视图保持原初始化语义。网络拦截任何非/stage4/的分类API并记录请求，实际没有触发拦截或旧Test请求。

已有exp2/hanlp环境未安装、升级或降级依赖。controller/API/recovery toy tests13 passed，neural toy tests8 passed，Ruff通过；仅使用toy fixtures。覆盖平衡/嵌套/canonical420、split禁用与真实audit-hook拒绝、subset-only表示和C0/C1矩阵identity、effective_k、固定C3配置与postfit Val、ddof0聚合、420 mismatch拒绝、只读API白名单/C4拒绝、metadata例外和恢复协议。

Chromium153.0.8010.12桌面1440×1000、手机390×844：16/16检查通过；真实曲线/误差条、模型切换、60replicates/15subsets、诊断和导出通过，无横向页面溢出、console/page/HTTP errors。Browser插件不可用，复用已安装Python Playwright；现有服务刷新后复用。未重复验证无关页面或其他浏览器。

冻结审计：480个允许读取的基线文件，477相同，仅共享API/template/classification-page.js三个获准扩展；无缺失、无意外变化。124个受保护Test/结果文件仅size/mtime比对且一致，不读取字节；Test90本轮没有hash或解码。新Run manifest412/412通过；预注册17个源文件中16个当前一致，唯一开发集guard修复由原始源码重构SHA+amendment当前SHA验证，finalizer源码hash一致。旧Exp1/Exp2/Stage1/2/3、数据split、latest没有修改。

外部QA截图、网络日志、哈希/元数据差异记录：C:/Users/29513/.codex/visualizations/2026/10/01/01a0f750-fe2e-7b83-a6c6-d991ff6ccbab/stage4/。质量收据位于outputs/exp3/quality/20261001_164602_48e6c6b6.json，写于冻结Run之外。

```text
formal_exp3_stage4_completed_curve_points=60
formal_exp3_stage4_completed_c3_training_runs=15
formal_exp3_stage4_interrupted_c3_attempts=1
formal_test_evaluation_count=0
test_accessed=false
stage1_modified=false
stage2_modified=false
stage3_modified=false
exp1_modified=false
exp2_modified=false
lab4_started=false
```
