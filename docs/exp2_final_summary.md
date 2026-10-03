# 实验二最终实验材料

## 1. 数据集

THUCNews 固定个人平衡全文子集 600 篇，财经、教育、科技、体育、娱乐各 120 篇。聚类使用全部 clustering_all.jsonl 的 id/text，文档及顺序在三组实验中一致。分类 train/val/test=420/90/90 不进入聚类；本轮未开始实验三。

## 2. 数据预处理

沿用冻结 HMM-1 Smooth：NFKC → 英文小写 → 连续中文/英文混合词段 → 中文 HMM 分词 → 最短词长 2 过滤。R1/R2 复用经过 hash/ID 验证的 R0 token 缓存，不重新选择数据、不使用主题词表。gold primary_label 只在全部无监督拟合完成后按 ID 和原文 hash 关联。

## 3. TF-IDF baseline

R0：word unigram，min_df=2、max_df=0.95、max_features=20000、sublinear_tf=true、smooth_idf=true、L2 normalize。词表/聚类维度 11254，零元素比例 98.6447%。历史 Run 20260930_094537_befa08d2 完全保留。

## 4. K-means

同一 Lloyd KMeans；k-means++、n_init=10、max_iter=300、tol=1e-4。主 seed=42，稳定性重复 seeds43/44。每种表示 27 次正式 fit 均已在前两阶段完成，本最终阶段没有聚类拟合。距离与代表文本选择始终在各自实际聚类输入空间。

## 5. K sweep

K=2…10 的 SSE、silhouette、CH、DB、簇规模以及跨 seed ARI 完整保留。R2 K=6/7/8 silhouette 约 0.06717/0.07156/0.07159，略高于 K=5 的 0.06384；但 K=5 stability 0.96181 高于这三个 K 的约 0.78691/0.85613/0.80198。因此不把最高单项指标当作自动选择规则。

## 6. Baseline 结果

R0 K=5 silhouette 0.018636，ARI 0.556077，NMI 0.684899，stability 0.833859。教育、财经、体育主题明显；科技分裂出消费电子子主题，233 篇最大簇混合了娱乐等内容，关键词包含自己、一个、表示等泛化词。详细历史协议和结果见 exp2_clustering.md。

## 7. 表示优化

R1：unigram+bigram，max_df=0.30，其余 TF-IDF 参数不变。阈值依据无标签 corpus DF 一次制定，过滤 10 个高 DF unigram，无人工 stopwords；词表 19940（11244 unigram+8696 bigram）。R2：同一 R1 矩阵 → TruncatedSVD 100 维（randomized、n_iter=7、seed42）→ L2。累计解释方差 29.888604%。未搜索参数或维度。

## 8. R0 / R1 / R2 对照

| K=5 | R0 | R1 | R2 |
|---|---:|---:|---:|
| 维度 | 11254 | 19940 | 100 |
| silhouette | 0.018636 | 0.013424 | 0.063837 |
| CH | 6.424703 | 5.020252 | 18.900652 |
| DB | 6.907001 | 7.700220 | 3.934688 |
| stability ARI | 0.833859 | 0.647988 | 0.961805 |
| post-hoc ARI | 0.556077 | 0.569738 | 0.593404 |
| NMI | 0.684899 | 0.687963 | 0.720673 |
| homogeneity | 0.658590 | 0.670240 | 0.692977 |
| completeness | 0.713396 | 0.706648 | 0.750676 |
| V-measure | 0.684899 | 0.687963 | 0.720673 |

R1 泛化词减少且科技主题簇扩大，但内部质量、稳定性下降，优化不是单调改善。R2 几何分离、种子稳定性与事后一致程度同时提高，但平衡性没有同步改善。SSE 属于表示空间，不放进跨表示主对照表；LSA 空间与原 TF-IDF 空间的几何指标也不是绝对排名。完整 sweep、组成、关键词、代表/边界文本和环境见 exp2_representation_comparison.md。

## 9. 最终主方案 R2

**综合内部结构、跨随机种子稳定性、语义可解释性和事后类别一致性，本实验采用 R2 作为最终主表示方案。**这是用户在观察三组冻结结果后作出的实验方案决定；没有再次训练、调参或构造人工综合分数。不能表述为 R2 全面优于其他表示或理论最优模型。

实际输入从 19940 词维压缩到 100 个潜在维度，再 L2 归一化，聚类空间更加紧凑。关键词由成员在原 R1 TF-IDF 空间的均值解释，不把潜在维度当词项。

## 10. K=5 reference 分析

默认 R2 + K=5，标记「5主题参考配置 / known-topic reference」。五个已知主题的平衡抽样给它明确分析价值，且此配置稳定性高、语义结构较清楚。本实验没有依据 gold 标签自动选择最佳 K，也没有宣称 K=5 是无监督意义下唯一最优解。

| cluster ID | size | 财经 | 教育 | 科技 | 体育 | 娱乐 | 词空间 Top-3 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 94 | 0 | 94 | 0 | 0 | 0 | 考生 / 考试 / 招生 |
| 1 | 112 | 0 | 0 | 0 | 112 | 0 | 比赛 / 球队 / 球员 |
| 2 | 232 | 9 | 26 | 69 | 8 | 120 | 娱乐 / 媒体 / 新浪 娱乐 |
| 3 | 46 | 0 | 0 | 46 | 0 | 0 | 相机 / 采用 / 机身 |
| 4 | 116 | 111 | 0 | 5 | 0 | 0 | 基金 / 投资 / 市场 |

以上是事后解释，cluster ID 不是类别编号。教育簇94篇全部教育，体育簇112篇全部体育；财经主题簇116篇含111篇财经；消费电子簇46篇全部科技；232篇混合簇含娱乐120、科技69、教育26、财经9、体育8。

## 11. 聚类可视化与 Web

现有 /clustering 四个 Tab 保持：数据与表示展示紧凑演进/维度/方差与折叠配置；聚类分析默认 Top-8、一篇中心代表、事后组成，其余完整内容折叠；可视化保留 sweep/规模/cluster filter 与 R0/R1/R2 切换；评测展示内部/外部指标分区及同 K 的三表示对照。主视图默认 R2/K5，用户仍可切换所有历史 Run 与 K。

100D SVD 实际参与 R2 聚类；2D SVD 仅为已有可视化。新增 3D SVD 同样只为显示，从保存的实际输入矩阵派生，固定 seed42/n_iter7、不做维度比较或聚类拟合。3D 缓存独立保存在 outputs/exp2/visualizations/<run>/，记录来源矩阵/ID hash、配置、方差和自身校验 hash；旧 projection.json 与所有 Run 文件完全不变。显示支持拖动旋转/滚轮缩放/重置，键盘方向键/+/-，默认按 cluster 着色；gold 着色仍仅在事后评测页。低维投影不能直接代表高维聚类距离。

二维/三维点浮窗均限制宽度、按可用空间选择左右侧并换行，保持在图表边界内。

## 12. 错误与边界样本

科技人工栏目内部有显著相机/消费电子子主题；人物、媒体、互联网与娱乐内容共享叙事词汇，栏目标签和自然词汇结构并不完全一致。大混合簇反映了题材交叉，不能仅按类别不纯将它判为失败。混合簇中心代表包括扎克伯格人物报道，消费电子簇代表包括柯达Z981促销。短促销、查询/预告、网页说明等常位于边界；没有删除样本或把远距直接当异常。

## 13. 结论、局限与冻结记录

主表示 R2、参考 K=5。局限：科技仍分裂，娱乐与科技/叙事内容仍混合，最大/最小簇比例约5.043；100D仅解释约29.89%方差；几何评价具有空间依赖，gold只用于事后评价，不能替代无监督分析。单一增强配置无法分离DF与bigram各自贡献，100维也未声称最优。未来可以研究其他表示或降维维度，但本轮没有实施。

轻量推荐记录 outputs/exp2/final_selection.json 指向三组历史 Run 和已校验的 comparison.json，记录决定来源/理由/局限/reference角色。它与 baseline latest.json 语义分离；后者仍指向 R0，不改变。

针对性测试：`pytest --verify-only tests/test_clustering_projection.py tests/test_clustering_final_summary.py tests/test_web_course_ia.py tests/test_information_extraction.py -q`，55 passed，1条Starlette deprecation warning。toy 投影测试执行了小规模显示SVD，没有KMeans fit。Ruff check/format 与两个JavaScript模块的语法检查通过。

保护审计：`outputs/exp2/protection/final_stage_integrity.json`，621/621受保护文件未变，覆盖原实验一/课程扩展/IE相关保护资产、THUCNews与分类split、R0/R1/R2全部Run/model和固定比较档案；baseline latest仍是R0。

Chromium 1440/390最终交互QA通过：默认R2/K5、全局Run与表示下拉切换、K切换、Top8+一代表+折叠完整内容均与API一致；K5对照表和指标卡精确对应冻结结果。三种3D API均为600点；实际3D画布拖动/滚轮均产生像素变化，R2簇0筛选94点，返回2D通过。二维左右真实点悬浮可换行且位于图内；1440悬浮后同页缩到390时旧框隐藏/定位重置，重新hover受当前图宽/图高约束。390目标Tab无页面级横向滚动，console/pageerror/requestfail均0。复用8000开发服务；Browser plugin未提供，使用项目既有Python Playwright/Chromium，不新增依赖。合并29项检查全通过，独立视觉评分4.74/5，无Critical。QA证据：`C:/Users/29513/.codex/visualizations/2026/09/30/01a0f196-35c0-7ee2-9bea-06c74a6f1f42/scatter3d/scatter3d_stage3_qa.json`，同目录保留1440桌面与390缩放后的两张悬浮截图。

```text
formal_exp2_fit_executed=false
historical_runs_modified=false
exp2_primary_representation=R2
exp2_primary_reference_k=5
lab3_started=false
```
