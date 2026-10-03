# 实验二：TF-IDF 与 K-means 文本聚类

本文保留第一阶段 R0 baseline 的协议与结果。实验二最终主表示已确定为 R2、K=5 为5主题参考配置；完整最终实验结构见 [exp2_final_summary.md](exp2_final_summary.md)，表示对照见 [exp2_representation_comparison.md](exp2_representation_comparison.md)。

## 固定协议

使用已冻结的 THUCNews 五类各 120 篇全文，600 篇全部参与无监督拟合。唯一模型输入为
`data/splits/thucnews_600/clustering_all.jsonl`，严格只有 `id`、`text`；带标签或多余字段的
输入会被拒绝。输入哈希必须匹配现有 split manifest，不重新选择样本或生成分类 split。

`execute_unsupervised` 只接受 `ClusteringDocument(id, text)`。它完成预处理、TF-IDF、全部
K/seed 拟合、内部评价、代表文档和投影后，service 才打开冻结子集，按 ID 和原文指纹关联
`primary_label`。标签不参与词表、停用词、向量、拟合、稳定性或 K 的无监督分析。

## 文本表示与参数

复用现有冻结 HMM-1 Smooth 分词模型，仅执行新输入的分词推理，不训练、修改或重评该模型。
最长匹配也可复用，但其现有模型加载会同时读取约 53 MB 的词典/二元计数；HMM 模型约 8 MB，
有确定性的 BMES 解码和未登录字符支持，因此选择 HMM。没有新增分词依赖。

处理顺序：NFKC → 英文小写 → 提取连续中文/英文数字混合词段 → 中文 HMM 分词 →
过滤长度小于 2 的 token → word unigram TF-IDF → L2 文档归一化。
标点、空白、纯数字不作为特征；英文混合词保留。默认停用词列表为空，不按主题手工挑词。
`min_df=2`、`max_df=0.95`、`max_features=20000`；sublinear TF 和 smooth IDF 均开启，
float64。词表只由这 600 篇无标签文本产生。出现零向量时明确失败，不静默删除文档。
配置位于 `configs/clustering.yaml`，每个 Run 保存完整配置、词表、IDF、稀疏向量及切词器指纹。

K-means 使用 sklearn 的 Lloyd 算法，`init=k-means++`、`n_init=10`、`max_iter=300`、
`tol=1e-4`。主结果 seed=42；每个 K 另执行 seed=43、44，合计 27 次 K-means fit。
稳定性是同一 K 不同种子分配的两两 ARI（平均值、最小值），不涉及真实标签。
数值线程限制为 1；相同数据、配置、依赖环境下复现。跨版本或平台的浮点行为可能不同，
Run 保存实际环境，不声称任意环境下位级一致。

## K sweep 与已知主题参考

K=2…10 完整保存，不自动产生唯一最佳 K。内部指标、SSE 趋势、种子稳定性、簇规模及语义
解释共同用于后续分析。不能用外部指标挑选 K 或反复调整参数直到类别一致程度更高。

K=5 另保存为 `reference/k5.json`，标记 `known-topic-reference`，原因是抽样设计包含五个
新闻主题。它复用 sweep 中 seed=42 的同一拟合，另存参考结果和声明，不重复执行完全相同
的 fit。它不是无监督选出的最佳 K；`run.json` 的 `best_k` 为 null。

## 评价与簇解释

内部指标：inertia/SSE、Euclidean silhouette、Calinski–Harabasz、Davies–Bouldin。
全部在原 L2 TF-IDF 空间计算；CH/DB 对本规模矩阵作一次稠密转换，不在二维空间代算。
silhouette/CH 越大通常越好，DB 越小通常越好；inertia 随 K 增加通常下降，不能单独据此选 K。

外部指标：ARI、NMI、homogeneity、completeness、V-measure。它们只描述聚类与原始新闻
主题的一致程度。cluster ID 是任意编号，不直接当作类别编号计算 accuracy，本阶段不报告
accuracy 或 purity。簇 dominant category 只是事后解释字段。

每簇输出数量、中心权重最高的 12 个词、按原空间 Euclidean 距离排序的 3 篇近中心代表
和 3 篇远中心边界文本，保留 ID、原文 snippet、距离及事后类别组成。不伪造标题；远中心
文本仅表示与本簇中心的距离较大，不把它们自动判为异常新闻。

所有文档共享 seed=42、n_iter=7 的二维 TruncatedSVD。它只用于可视化，不参与 K-means、
内部指标或选 K。图中默认按 cluster 着色；gold 着色只出现在事后评价视图。页面报告投影
解释方差，二维图中看似重叠不等于原空间没有区分。

## 执行与产物

Web/API 不训练模型，读取 `RunStore(settings, "exp2")`。新结果写入
`outputs/exp2/<run_id>/`，模型写入 `artifacts/models/exp2/<run_id>/`。
目录拒绝覆盖；结果和模型完整写出后才写 `run.json` 并原子更新 exp2 的 `latest.json`。
不修改 exp1 指针、旧 Run、MSRA-200、课程拓展模型或 THUCNews 选择/分类 split。

产物包含 config JSON/YAML、输入 IDs、dataset fingerprint、token JSONL、词表/IDF、CSR
文档矩阵、各 K 中心/参数、sweep、稳定性种子分配、逐 K assignments 与 summaries、
K=5 reference、二维坐标/SVD components、gold provenance、依赖环境、源码和产物 SHA256。
透明 JSON/NumPy/CSR 保存，不使用 pickle-only 模型。`ClusteringService.verify_artifacts`
支持只读指纹检查；TF-IDF 和 K-means 支持加载后 transform/predict。

本轮正式运行命令（会执行实验二 fit；后续重跑会产生新的独立 Run）：

```powershell
.\.venv\python.exe -m venv artifacts/runtime/exp2-venv --system-site-packages
artifacts/runtime/exp2-venv/Scripts/python.exe -m pip install -r requirements-exp2.lock
$env:PYTHONPATH = "src;."
artifacts/runtime/exp2-venv/Scripts/python.exe -m nlplab.clustering
```

核心环境不安装新依赖；实验二环境复用已有基础依赖，版本记录见 Run 的 `environment.json`。
冻结后的回归测试使用 `--verify-only` 以跳过相关实验一训练入口，但该选项仍允许本次实验二
及部分课程扩展的 toy fit。必须分别报告 `toy_test_fit_executed` 与
`formal_exp2_fit_executed`；不能把该选项描述成绝对不训练。

四个 Web Tab 沿用原信息架构：数据与表示、聚类分析、可视化、实验评测。Run/K 选择、簇
筛选、真实指标、代表文本、SVD/规模/sweep 和组成矩阵均来自已有产物；无完成 Run 时展示
真实空态。实验三没有启动。

## 实际运行结果

首轮正式 Run：`20260930_094537_befa08d2`。未进行参数优化或标签驱动调参，保留此 baseline。
600 篇、11,254 维、91,513 个非零特征、稀疏度 98.6447%、192,306 个 token、零向量为 0。
Python 3.11，sklearn 1.9.1、SciPy 1.17.1、NumPy 2.4.6；完整版本/平台见 environment.json。

| K | SSE | silhouette | CH | DB | 稳定性平均 ARI | 簇规模（按 cluster ID） | 迭代 |
|---|---:|---:|---:|---:|---:|---|---:|
| 2 | 573.9499 | 0.009278 | 7.7741 | 6.9251 | 0.9010 | 119,481 | 5 |
| 3 | 567.4175 | 0.012422 | 7.3617 | 6.8278 | 0.8672 | 382,122,96 | 7 |
| 4 | 561.8930 | 0.016353 | 6.9010 | 6.4672 | 0.8732 | 330,87,50,133 | 6 |
| 5 | 557.3392 | 0.018636 | 6.4247 | 6.9070 | 0.8339 | 91,48,121,107,233 | 9 |
| 6 | 553.3110 | 0.018140 | 6.0334 | 6.5368 | 0.6502 | 54,229,63,50,92,112 | 11 |
| 7 | 551.1720 | 0.017723 | 5.4224 | 6.8173 | 0.5741 | 50,106,81,92,39,183,49 | 10 |
| 8 | 549.3915 | 0.017110 | 4.9290 | 7.0549 | 0.4935 | 46,98,69,131,99,55,47,55 | 8 |
| 9 | 547.5812 | 0.015855 | 4.5641 | 6.4585 | 0.5928 | 143,65,65,16,109,49,39,58,56 | 12 |
| 10 | 546.3489 | 0.012672 | 4.2071 | 6.8823 | 0.4843 | 93,49,44,31,58,21,51,72,93,88 | 17 |

silhouette 的本轮最高值在 K=5，但各 K 均接近 0，原 TF-IDF 空间的几何分离有限。
CH 更偏向较小 K；DB 本轮最低在 K=9，但该配置最小簇只有 16 篇，最大/最小比为 8.94。
K≥6 的跨种子稳定性整体低于 K=2…5。SSE 降低与规模/语义/稳定性的取舍尚不足以
宣布唯一最佳 K，完整曲线与种子结果保留供后续决策。

### K=5 已知主题参考

9 次主拟合迭代，SSE=557.3391815、silhouette=0.0186355、CH=6.4247029、DB=6.9070008。
稳定性平均 ARI=0.8338588，最小 ARI=0.7904444；这里比较不同种子分组，不是与 gold 比较。

事后主题一致性：ARI=0.5560774、NMI=0.6848986、homogeneity=0.6585903、
completeness=0.7133963、V-measure=0.6848986。未计算 cluster-ID/category-ID 的普通 accuracy。

| 簇 | 数量 | Top terms（前 5） | 财经 | 教育 | 科技 | 体育 | 娱乐 |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 91 | 考生、考试、招生、高考、录取 | 0 | 90 | 1 | 0 | 0 |
| 1 | 48 | 采用、相机、机身、mm、功能 | 0 | 0 | 48 | 0 | 0 |
| 2 | 121 | 基金、投资、市场、上涨、美元 | 111 | 0 | 10 | 0 | 0 |
| 3 | 107 | 比赛、球队、球员、联赛、体育 | 0 | 0 | 0 | 107 | 0 |
| 4 | 233 | 娱乐、自己、一个、媒体、表示 | 9 | 30 | 61 | 13 | 120 |

教育、财经、体育的核心用词形成了相对清晰的组；科技并非完整对应一个簇，而是相机/数码
产品子主题形成 48 篇的簇。簇 4 最大，娱乐占 120/233，其余来自多个主题；它的通用报道
词与人物叙事不能解释成“纯娱乐簇”。真实五类平衡不意味着无监督五簇也应各为 120 篇。

各簇最近中心的第一篇代表（以下文字均是原文片段，不新增 title 字段）：

| 簇 | 文档 ID | 距离 | snippet 开头 |
|---|---|---:|---|
| 0 | THU-844d3cd1ec2ab8074c4156dc | 0.879344 | 云南大学2012年硕士研究生招生简章… |
| 1 | THU-fbdf9c094d22eb36d87351d8 | 0.859553 | 又是一年春 首选便携广角索尼WX5特价… |
| 2 | THU-8417eecdb6aae9d1f72396d1 | 0.891665 | 新华期货12月21日观点 美国股市方面… |
| 3 | THU-bd4e67fd9d3bb7d5870c1c1e | 0.923570 | 多特蒙德造德甲史上最成熟球队… |
| 4 | THU-8deaeabf0abdae03067dab26 | 0.950270 | Facebook扎克伯格：财富勇者的美丽新世界… |

每簇完整 12 个词、3 篇近中心及 3 篇远中心文本、距离、组成均在 results/reference 文件。
所有九个 K 都保存同样的簇分析，不仅保存 K=5。

二维 SVD 的解释方差为 0.4700% 和 1.4754%，合计仅 1.9455%。因此图形主要提供浏览和
对照，不能据此断言二维视觉重叠代表模型失败，或凭图形选择 K。当前不引入其他表示、
降维聚类或其他聚类算法；后续优化需保存前后独立配置/Run。

### 验证记录

新增聚类测试 10 passed；定向 API、信息架构、公共数据、信息抽取及冻结/收尾回归
79 passed / 2 skipped / 1 第三方 warning。两项跳过发生在实验一 toy HMM fit 调用前。
本次不是最新全量 pytest 结果；toy 聚类拟合已执行，正式实验二拟合已执行，实验三未启动。
Ruff check/format 通过。正式产物 43 项 SHA256 一致，九个 K 的模型加载后预测与保存分配
完全一致；TF-IDF 加载后重建矩阵最大绝对差 4.44e-16。所有簇数量与事后组成合计为 600。

实验一与课程扩展保护：432 个本轮基线文件及四份既有快照的 699 项声明全部匹配，合并
检查 475 个唯一路径，missing=0、changed=0、conflicts=0，原快照清单本身也未改变。
范围包含七模型/MSRA/exp1 Run 和报告、课程感知机/CRF、POS、信息抽取规则，以及
THUCNews 选择和分类 split。详细证据为 `outputs/exp2/protection/final_integrity.json`。

Chromium QA：复用 `http://127.0.0.1:8000/clustering`，Python Playwright（本会话未提供
Browser plugin）。1440/390 两种宽度均检查四 Tab；默认 K5→K2→K5 的指标和规模与 API
一致；scatter 有 600 点，筛选簇 0 后为 91 点，gold 散点和组成矩阵也覆盖 600 篇。
图表隐藏/切换/resize 后尺寸正常，console/page/request errors 均为 0，页面无水平溢出。
empty Run 和模拟 result 503 状态验证通过，不触碰真实数据或生成新 Run。
独立视觉评审 4.8/5，无 critical；移动端长簇卡片采用自然纵向滚动，宽表在卡片内滚动。
QA JSON 和 1440/390 四 Tab 截图保存在当前 chat 的 workspace 外 `exp2-qa` 目录。

最终执行声明：`formal_exp2_fit_executed=true`、`toy_test_fit_executed=true`、
`lab3_started=false`。正式 fit 为一次完整 27 配置主实验，不包含实验一重训/重评。
