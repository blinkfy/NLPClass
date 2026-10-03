# 实验三 Stage 2：character bigram + χ² + LinearSVC

本阶段只新增 C2。C0/C1 的模型、预测与指标直接读取冻结 Stage 1 Run `20261001_123337_9c1e7341`，没有重训。C1/C2 的比较是同一 LinearSVC 分类器族下两种经 Validation 选择的表示方案，不是最终模型仅改变一个参数的受控实验。

## 表示与数据边界

字符 analyzer 先执行 Unicode NFKC 与 lowercase，再在连续内容片段内提取重叠的长度 2 bigram。内容字符为 Unicode Han ideograph、ASCII a-z、Unicode 十进制数字 Nd；其他字符均为边界。中文、英文、数字可以在同一连续片段中相邻；不跨空白、标点、符号生成特征。没有 HMM 调用、停用词或人工词典。比如 `ＡB基金，招生 １２3` → `ab, b基, 基金, 招生, 12, 23`。

Count 固定 `min_df=2, max_df=1.0, max_features=None`。使用非负 Count 计算 χ² 排序，然后选择 Top-K，再拟合 TF-IDF：`sublinear_tf=True, smooth_idf=True, use_idf=True, norm='l2'`。χ² 分数只用于关联排序，不解释为统计显著性结论。并列 χ² 按 feature 名字典序选择，矩阵列保持原始词表顺序。顺序符合 [scikit-learn χ² 文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) 的非负 Count 输入约束。

冻结的 600 篇及 420/90/90 split 保持不变。阶段 A 的 Count、χ²、IDF 仅拟合 Train 420，Validation 90 仅 transform。网格为 `k=[2000,5000,10000,all] × C=[0.1,0.5,1,2]`，按 Macro-F1、Accuracy 降序，再按实际维度、C 升序选择。所有候选保存完整 per-class、confusion matrix、预测和耗时。

阶段 B 先保存 selection freeze，再用 Train+Validation 510 从头拟合 Count、χ²、IDF 与固定参数 SVC。完整表示和模型保存后才打开 Test 90，正式评价一次；后续错误、coverage、配对分析没有反馈到配置选择。

LinearSVC 配置与 C1 相同：OvR、L2、squared_hinge、dual=True、random_state=42、tol=1e-4、max_iter=10000。liblinear 不提供支持向量列表；保存 n_iter、系数、截距和耗时，不虚构支持向量数量。

## 解释与覆盖率定义

单篇解释遵循 `decision_score_c = intercept_c + Σ x_j w_cj`。直接贡献为 `x_j w_cj`；相对第二名的贡献为 `x_j (w_pred,j - w_runner,j)`，加上截距差还原 score gap。Web 展示原始 decision score，不转成概率。全局表分别显示 χ² relevance 与 SVC coefficient，单篇显示 TF-IDF × coefficient，三者含义不同。

Coverage 定义为：从全文抽取的候选特征出现次数中，存在于指定 final fitted vocabulary 的次数比例；另保存不同候选类型的命中比例。空分母返回 null。分别计算 C1 final word vocabulary、C2 final raw Count vocabulary、C2 selected vocabulary。word 与 char 候选单位不同，不能据比例直接推断性能优势，也不称为 OOV Recall。C1 候选来自冻结 tokens.jsonl，不重新调用 HMM。

## 工程与产物

复用固定数据加载器、RunStore、既有 LinearSVC 参数导出、指标函数与 Stage1 artifact hash 校验。新增 `features/char_chi2.py`、`classification/c2.py`、`classification/stage2_experiment.py`、`classification/stage2.py` 与扩展服务 `classification_stage2_service.py`。算法不依赖 FastAPI，Web 只读取资产或调用只读推理 CLI。

Stage2 使用独立 `stage2_latest.json` 和一次执行 guard，不更新 Stage1 `latest.json`。保存 selection/final vocabulary、DF、χ²、selected indices、IDF、稀疏矩阵、模型系数/截距、全部候选、selection freeze、Test predictions/errors、paired document IDs、coverage、环境、源码哈希与完整 artifact manifest。无 pickle。

四个 Web Tab 保持不变，新增 C2 数据链、模型选项、候选表、paired outcomes 与错误筛选。查看 Stage1 Run 时 C2 禁用；C0/C1 资产仍来自冻结 Run。

## 正式结果

Stage2 Run：`20261001_133829_68ee50d7`。选定 `k=10000, C=2.0`。16 个候选均合法且完整执行；Train raw vocabulary 为 29,927，故 all 是 29,927 维，没有静默截断。下表为 Validation Macro-F1（%），与正式 Test 分开：

| χ² k | C=0.1 | C=0.5 | C=1.0 | C=2.0 |
|---|---:|---:|---:|---:|
| 2000 | 94.3940 | 97.7761 | 97.7761 | 97.7761 |
| 5000 | 93.2805 | 95.5478 | 96.6589 | 97.7424 |
| 10000 | 92.0910 | 95.5314 | 97.7761 | **98.8880** |
| all | 90.8855 | 94.3916 | 96.6589 | 96.6589 |

winner Accuracy=98.8889%。完整 Accuracy、Macro P/R/F1、每类 P/R/F1、混淆矩阵、逐篇预测、实际维度与时间保存在 `validation_selection.json`。selection freeze 的 opened_splits 为 train/val，记录配置与选参文件 SHA256。

最终 510 篇中抽取 108,253 种不同 char bigrams，去除 73,338 个 DF=1 类型后 raw Count vocabulary 为 34,915，χ² 后维度为 10,000。DF 数组、过滤前 DF histogram 与 observed_df.json 已保存。

| 矩阵 | shape | nnz | 稀疏率 | 零向量 |
|---|---|---:|---:|---:|
| Train Count | 420×29927 | 166582 | 98.6747% | 0 |
| Train selected / TF-IDF | 420×10000 | 91643 | 97.8180% | 0 |
| Final Count | 510×34915 | 204705 | 98.8504% | 0 |
| Final selected / TF-IDF | 510×10000 | 106725 | 97.9074% | 0 |
| Test Count（transform） | 90×34915 | 30775 | 99.0206% | 0 |
| Test selected / TF-IDF（transform） | 90×10000 | 17511 | 98.0543% | 0 |

Test 候选覆盖诊断：

| fitted vocabulary | 候选出现次数 | 命中次数 | 次数覆盖率 | 不同类型命中 / 总数 | 类型覆盖率 |
|---|---:|---:|---:|---|---:|
| C1 final word | 59832 | 24647 | 41.1937% | 5460 / 34015 | 16.0517% |
| C2 raw Count | 75015 | 52457 | 69.9287% | 13098 / 29637 | 44.1948% |
| C2 selected | 75015 | 34754 | 46.3294% | 5420 / 29637 | 18.2880% |

正式 Test 指标（%）：

| 模型 | Accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|
| C0 · frozen NB | 92.2222 | 92.1667 | 92.2222 | 92.0846 |
| C1 · frozen word SVC | 94.4444 | 94.8947 | 94.4444 | 94.5252 |
| C2 · char χ² SVC | **96.6667** | **96.7836** | **96.6667** | **96.6313** |

C0/C1 完整原始指标未经修改复制到 `comparison_metrics.json`。C2 Micro/Weighted F1 分别为 96.6667% / 96.6313%。

| C2 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 财经 | 94.7368% | 100% | 97.2973% | 18 |
| 教育 | 100% | 88.8889% | 94.1176% | 18 |
| 科技 | 94.4444% | 94.4444% | 94.4444% | 18 |
| 体育 | 100% | 100% | 100% | 18 |
| 娱乐 | 94.7368% | 100% | 97.2973% | 18 |

混淆矩阵行是真实、列是预测，固定财经/教育/科技/体育/娱乐顺序：

| 真 / 预测 | 财经 | 教育 | 科技 | 体育 | 娱乐 |
|---|---:|---:|---:|---:|---:|
| 财经 | 18 | 0 | 0 | 0 | 0 |
| 教育 | 1 | 16 | 1 | 0 | 0 |
| 科技 | 0 | 0 | 17 | 0 | 1 |
| 体育 | 0 | 0 | 0 | 18 | 0 |
| 娱乐 | 0 | 0 | 0 | 0 | 18 |

## C1 ↔ C2 的逐篇事实

两者都对 84、仅 C1 对 1、仅 C2 对 3、两者都错 2。prediction disagreement 为 4；共同错误的两个样本预测类别也相同。完整四个 bucket 的 document IDs 及以下各组记录保存在 `paired_c1_c2.json`。

C2 修正的 3 篇：

| document ID | 原类别与标题 | C1 → C2 | C2 预测类主要直接贡献 |
|---|---|---|---|
| THU-2a909c0b5de227fed5ab8214 | 财经：央行将发行中国改革开放30周年金银纪念币一套 | 娱乐 → 财经 | 央行 .0592、银行 .0583、发行 .0582、收藏 .0578 |
| THU-d18289d57f3c13904bd0efec | 教育：留法学生租房频受骗，使馆提醒先办手续再付钱 | 科技 → 教育 | 学生 .1778、留学 .0976、提醒 .0400 |
| THU-0fd1c56dc512bcdf68c6c908 | 娱乐：电影人欲装眼球摄像头“亲眼”拍摄纪录片 | 科技 → 娱乐 | 拍摄 .0991、电影 .0545、节目 .0429 |

新增错误：`THU-7dca826c9f67fa0ea69ffb44`（教育，世界工薪族年薪超三万 TOP8 国家排行）。C1 预测教育，主要贡献大学 .1015、教育 .0882；C2 预测财经，主要贡献消费 .0505、保险 .0329、收入 .0247，score gap .2990。薪资与货币线索压过教育内容，是主题边界问题。

共同错误：

- `THU-60df0088609e4fcd82da01ae`（教育，培养亿万富翁最多的美国大学）：均预测科技。C2 软件 .0549、北大 .0481、公司 .0424 等贡献较高；跨词二字串“北大”在此来自“西北大学”，说明字符匹配也会丢失语义边界。
- `THU-d913ec3bc121d093ab3ec383`（科技，佳能 DV 演绎《让子弹飞》精彩幕后）：均预测娱乐。C2 宣传 .0924、演员 .0802、拍摄 .0748、观众 .0720 主导，科技产品新闻的娱乐内容仍混淆。

直接贡献解释模型的加法分数，不能单独证明因果。相对 runner-up 贡献及完整五类分数也已保存，可用于检验两类分数差。

## 技术观察与后续判断

χ² Top 特征包含基金、考生、考试、市场、比赛、招生、学生，主题关联合理。每类 SVC Top positive 示例：财经基金/银行/投资；教育考试/考生/大学；科技网络/用户/手机；体育比赛/体育/球员；娱乐娱乐/香港/宣传。

同时出现来源片段：科技“浪科、技讯”、体育“浪体、育讯”、娱乐“浪娱、乐讯”，对应新浪频道署名的局部字符。C2 使用了主题和来源线索；不能将提升完全归因于分词改善，也不能声称跨来源泛化已得到验证。

留学租房样本提供了具体的分词缓解证据：冻结 C1 tokens 将多处“留学生”拆为“一留/学生、租留/学生、我留/学生”等；C1 final vocabulary 虽含留学/留学生，文档没有命中它们。C2 的连续字符表示保留“留学”，贡献 .0976，配合“学生”修正预测。央行纪念币案例显示对长词与切词变体的部分鲁棒性，但单篇不构成专名/型号的一般结论。佳能 DV 案例仍共同出错，不能说型号问题已经解决。

C2 净多正确 2 篇，Macro-F1 比 C1 高 2.1061 个百分点，但 Test 仅 90、每类 18，不能据此给出全面或理论最优结论。体育在本 Test 完全区分，教育与科技边界仍困难；娱乐主题与消费电子宣传也存在混合。

当前没有必须进入 TextCNN 的证据。现有残余错误更明显涉及类别边界、来源线索和全文主题混合；600 篇规模也不足以预设神经模型能提升。保留 C0/C1/C2 冻结结果，后续若要研究 C3，应先明确它要检验的上下文问题与训练协议，本阶段没有实现 C3 或 Lab4。

最终模型拟合 .0304 秒、最终表示拟合 1.5498 秒、90 篇矩阵推理 .0022 秒、Test transform .0935 秒。总执行 523.2885 秒包含解释资产导出中的重复词表名称读取；已缓存该只读名称数组，排序与数值不变，未重训或重复 Test。

## 验证与执行标记

- 新增 8 个 toy-only 测试；合并既有分类测试共 18 passed。覆盖 analyzer、HMM 禁用、非负 Count、fit/transform split、16 网格与 tie-break、列对应、保存加载、贡献还原、配对、Stage1 哈希和 API schema。缓存排序调整后 Stage2 8 项再次通过；仅一个既有 Starlette deprecation warning。新增代码与 API Ruff 通过。
- 开发前 C0→C1→C0→C1 连续 API/浏览器推理均 200，重复输出一致，无 503。QA 初次发现服务热重载失效，旧 API schema 与新 UI 不匹配；刷新原有单一 localhost:8000 服务后通过，没有重训。
- 通过现有 Playwright Chromium 验证（Browser 插件不可用，采用工程已有浏览器环境）：1440×1000 / 390×844 四 Tab 无页面横向溢出，console/page errors=0；C2 五类 decision score、direct/relative contribution、三模型指标、paired outcomes 正常。
- 错误筛选实际数量：errors=3、all=90、C2 errors=3、disagreements=4、corrected C1=3、new errors=1。Stage1 视图 C2 禁用，C0/C1 原指标及模型切换正常。
- 基线 SHA256 审计：Exp1 404、Exp2 97、原有 Exp3 37、模型 110、两组数据各 5、原有 src 80 个文件均无变化/缺失。Stage1 latest、原分类服务和原 CSS 未变化。UI/API 为明确允许的新增接入；Stage2 资产独立保存。

截图与完整 QA / integrity 收据位于 `C:/Users/29513/.codex/visualizations/2026/10/01/01a0f750-fe2e-7b83-a6c6-d991ff6ccbab/stage2/`。

```text
formal_exp3_stage2_validation_fits_executed=true
formal_exp3_stage2_final_fit_executed=true
formal_exp3_stage2_test_evaluation_executed=true
formal_validation_model_fit_count=16
formal_final_model_fit_count=1
formal_test_evaluation_count=1
toy_test_fit_executed=true
exp3_stage1_modified=false
exp1_modified=false
exp2_modified=false
c3_started=false
lab4_started=false
```
