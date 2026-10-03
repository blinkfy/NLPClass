# 实验三第一阶段：C0 / C1 文本分类

正式 Run：`20261001_123337_9c1e7341`。本轮完成 MultinomialNB 与 LinearSVC，同一表示、固定单标签五分类；未开始 C2、C3 或 Lab4。

## 工程与数据边界

复用 `ChinesePreprocessor`、`TextTFIDF`（含 vocabulary / IDF 保存加载）、冻结 HMM-1 Smooth tokenizer、`RunStore(Settings, "exp3")`、现有模板与四个 Tab、现有 `artifacts/runtime/exp2-venv`。未安装依赖，也未复用实验二拟合的词表、IDF、LSA 或模型。

新增：

- `src/nlplab/data/classification_input.py`：只读冻结 split loader，核验来源、文件 SHA256、ID 顺序、类别数量与正文 SHA256。
- `src/nlplab/classification/linear.py`：两分类器、无 pickle 数值参数保存、posterior / decision score、全局与单篇解释。
- `src/nlplab/classification/experiment.py`：分开的 Validation 选择、final fit、Test 评价；选参函数无 Test 参数。
- `src/nlplab/classification/__main__.py`：正式执行与只读推理入口；正式开始标记阻止意外重复执行。
- `src/nlplab/evaluation/classification.py`、`services/classification_service.py`、`app/api/classification.py`。
- `configs/classification.yaml`、`app/templates/classification.html`、分类专用 CSS / JS、`tests/test_classification.py`。

现有 `app/main.py` 仅增加分类服务与 API 注册、将 `/classification` 指向新模板；首页仅更新分类卡片的状态与入口 Tab，README 仅更新实验三状态及报告链接。全站导航、实验二页面和既有算法保持原样。

THUCNews 固定全文 600 篇，每类 120；Train / Validation / Test = 420 / 90 / 90，每类 84 / 18 / 18。完整来源指纹和三组 IDs 在正式 Run 的 `dataset.json` 中；全部文件与原有冻结 manifest 一致。

阶段 A 仅解码 Train 与 Validation，Train 拟合表示，再分别训练 3 个 NB 和 4 个 SVC 候选。以 Validation Macro-F1 选择，预先声明并列时取候选顺序第一项。`selection_freeze.json` 落盘时 `opened_splits=[train,val]`，Test 文本尚未打开。阶段 B 使用 Train+Validation 510 篇重新拟合最终表示、训练固定参数的两个最终模型，模型保存后才打开 Test，正式评价每模型一次。完整性检查仅读取 Test 文件字节以核验哈希，不构建特征或读取类别进行选择。

## 固定表示与统计

全文 → NFKC → 英文 lowercase → HMM-1 Smooth 分词 → 保留至少 2 字的中文 / 英文 token → word unigram + bigram TF-IDF。未使用主题词典或 gold label 选特征。没有停用词资源。

配置：`min_df=2, max_df=0.30, max_features=20000, sublinear_tf=true, smooth_idf=true, norm=l2, ngram_range=[1,2]`，float64 / CSR。该固定配置借鉴已验证的 R1 无标签 DF 处理经验，未进行 TF-IDF 参数搜索。

Train 420 实际统计：136,266 tokens，DF 过滤前 29,030 个 unigram；最大 unigram DF=174。高于 30% 阈值的十个泛化词为：一个、可以、新浪、时间、没有、目前、自己、表示、记者、进行。诊断不使用类别，不据此在 Test 后调整表示。

| 阶段 / split | shape | 稀疏率 | 零向量 |
|---|---:|---:|---:|
| 选参 Train | 420 × 14,000 | 98.7148% | 0 |
| 选参 Validation | 90 × 14,000 | 99.0439% | 0 |
| 正式 Train+Validation | 510 × 16,721 | 98.9063% | 0 |
| 正式 Test（仅 transform） | 90 × 16,721 | 99.1007% | 0 |

每阶段 C0/C1 使用完全相同的 matrix 对象；两阶段词表分别拟合，维度不同是正常现象。对应稀疏矩阵、vocabulary、IDF 均已保存。

## Validation 选择（与 Test 分开）

| 模型 | 参数 | Accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|
| C0 | alpha=0.1 | 93.33% | 94.09% | 93.33% | 93.19% |
| C0 | **alpha=0.5** | **94.44%** | **94.95%** | **94.44%** | **94.28%** |
| C0 | alpha=1.0 | 94.44% | 95.04% | 94.44% | 94.15% |
| C1 | C=0.1 | 93.33% | 94.03% | 93.33% | 93.23% |
| C1 | C=0.5 | 96.67% | 96.95% | 96.67% | 96.66% |
| C1 | C=1.0 | 96.67% | 96.95% | 96.67% | 96.66% |
| C1 | **C=2.0** | **97.78%** | **97.89%** | **97.78%** | **97.74%** |

每个候选的完整每类 P/R/F1、confusion matrix、fit / inference runtime 另存 `validation_selection.json`。最终冻结 C0 alpha=0.5，C1 C=2.0；LinearSVC 使用 liblinear、OvR、L2 penalty、squared_hinge、dual=true、random_state=42、tol=1e-4、max_iter=10000，最终 33 次迭代，未出现收敛警告。LinearSVC 不暴露支持向量或支持向量数量，未虚构此字段。

## 正式 Test

| 模型 | 正确 / 90 | Accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|
| C0 · MultinomialNB | 83 | 92.22% | 92.17% | 92.22% | 92.08% |
| C1 · LinearSVC | 85 | 94.44% | 94.89% | 94.44% | 94.53% |

| 类别（Support 均为 18） | C0 Precision | C0 Recall | C0 F1 | C1 Precision | C1 Recall | C1 F1 |
|---|---:|---:|---:|---:|---:|---:|
| 财经 | 90.00% | 100.00% | 94.74% | 100.00% | 94.44% | 97.14% |
| 教育 | 88.89% | 88.89% | 88.89% | 100.00% | 88.89% | 94.12% |
| 科技 | 87.50% | 77.78% | 82.35% | 85.00% | 94.44% | 89.47% |
| 体育 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 娱乐 | 94.44% | 94.44% | 94.44% | 89.47% | 94.44% | 91.89% |

下列矩阵行是真实类别、列是预测类别；顺序严格为财经、教育、科技、体育、娱乐。

```text
C0                    C1
18  0  0  0  0        17  0  0  0  1
 1 16  1  0  0         0 16  2  0  0
 1  2 14  0  1         0  0 17  0  1
 0  0  0 18  0         0  0  0 18  0
 0  0  1  0 17         0  0  1  0 17
```

C0 科技 4 篇错误：2 篇→教育、1 篇→财经、1 篇→娱乐；C1 科技只剩 1 篇→娱乐，但教育→科技增为 2 篇。两个模型都发生科技与娱乐相互混淆。体育在本 Test 上最易区分。不能据 18 篇 / 类宣称普遍可完美分类，一个样本影响该类 Recall 约 5.56 个百分点。

配对样本上 C1 修正 C0 的 4 个错误，同时新增 2 个错误；3 个错误共通，净增加 2 篇正确。Accuracy +2.22 个百分点、Macro-F1 +2.44 个百分点，未进行显著性检验，不能称为全面或显著优于。

## 样本与解释

所有 90 篇 / 模型都保存 document ID、gold、prediction、correct、模型分数、五类分数、runner-up、Top 贡献、原文片段、截距与贡献和。以下观察来自真实保存资产，未据这些案例重新选参。

正确案例（两个模型均正确，词贡献示例取 C1）：

- 财经 `THU-9345726fb97192bb361cbe52`：基金投资新闻；基金、投资、行业、市场提高财经分数。
- 教育 `THU-844d3cd1ec2ab8074c4156dc`：云南大学硕士招生简章；考生、考试、招生、大学提高教育分数。
- 科技 `THU-301b2e63d4ffec8c4ee449b7`：佳能 550D 产品行情；用户、佳能、产品、功能提高科技分数。
- 体育 `THU-59c2fe69b769b285a326c806`：火箭、姚明季后赛；比赛、火箭、姚明、湖人提高体育分数。
- 娱乐 `THU-ad173286247a25b5d5804f3e`：谭咏麟工作与伤情；娱乐、新浪 娱乐、香港、演出提高娱乐分数。来源栏目词也是可利用线索，提示存在来源捷径；这不是 Test 泄漏，但可能限制跨来源泛化。

典型错误：

- `THU-d913ec3bc121d093ab3ec383`：gold 科技，《佳能DV演绎〈让子弹飞〉精彩幕后》；两模型均预测娱乐，第二名科技。演员、影片、观众、拍摄等电影主题词支持娱乐，科技栏目来源与正文主体存在边界。
- `THU-0fd1c56dc512bcdf68c6c908`：gold 娱乐，电影人安装眼球摄像头；两模型均预测科技，第二名娱乐。电脑、手机、技术、摄像头等词贡献支持科技。
- `THU-60df0088609e4fcd82da01ae`：gold 教育，美国大学亿万富翁毕业生；两模型均预测科技。软件、公司、CEO、创始等产业词成为判别证据。
- `THU-48b8b97a7a31f1fdf8c2ba69`：gold 科技，阿里巴巴股票评级；NB 预测财经，预期、收益、股票、交易贡献支持财经，SVC 正确恢复科技。
- `THU-2a909c0b5de227fed5ab8214`：gold 财经，央行纪念币；NB 正确，SVC 预测娱乐，第二名财经。SVC 娱乐直接贡献较大的词包括周年、发行、一套、造型；所有 OvR 分数均可为负，仍由最大分数决定预测。
- `THU-d18289d57f3c13904bd0efec`：gold 教育，留法学生租房被骗；NB 正确，SVC 预测科技，网站、手机、房间、租房等正贡献支持该预测。直接正贡献只表示增加科技分数，并不能单独证明它超过教育；相对第二名的贡献也保存在资产中。

NB：`joint log-score_c = log prior_c + Σ x_j log θ_cj`，使用 logsumexp 归一化得到模型 posterior。五类训练样本均衡，final class prior 均为 0.2。原始 log θ 为负，不能直接把最大的任意内部量称为“置信度”。Top 判别特征采用 `x_j(log θ_pred,j − log θ_runner-up,j)`，同时保存原始直接项、先验差和贡献和；posterior 未校准，不代表经验正确率。

SVC：`decision score_c = b_c + Σ x_j w_cj`，Top 直接贡献为 `x_j w_pred,j`。不使用 softmax 或概率百分比；不把分数误称几何归一化距离。数值保存与 sklearn 预测一致，并列时保留 sklearn 字典序规则。Toy 测试证明截距和全部直接贡献精确还原模型分数；相对第二名的贡献和还原分数差。Top 截断后的列表本身不必加和为完整分数。

## Run、环境与复现

正式 Run 目录：`outputs/exp3/20261001_123337_9c1e7341/`。32 个文件由 `artifact_manifest.json` 覆盖，包含数据与 split IDs、config、tokenizer hashes、selection / freeze、两个阶段 TF-IDF、稀疏矩阵、最终模型参数、class mapping、逐篇预测、错误、全局解释、per-class metrics / confusion matrix、runtime、source hashes 与环境版本。latest 仅在完整写出并校验后原子更新，exp1 / exp2 latest 完全未动。

环境：现有 Python 3.11.16、scikit-learn 1.9.1；其余确切版本见 `environment.json`。最终 NB fit 约 0.0024s，SVC fit 约 0.0170s；90 篇批量纯分类分别约 0.00020s / 0.00017s。完整正式管线约 7.17s。上述是本机单次测量，不含 Web 进程启动开销，不能作为通用速度排名。

正式命令为 `artifacts/runtime/exp2-venv/Scripts/python.exe -m nlplab.classification --formal --root F:\projects\NLPClass`。已执行后不得重复运行；`stage1_formal_execution.json` 保留完成标记。查看结果与 API/Web 推理只加载模型，不拟合，不重评 Test。独立数值模型可用同一冻结环境重新加载。

## 技术观察与下一阶段

在相同表示上，NB 聚合类别词分布的 log-score，SVC 学习类间线性判别边界。本次 SVC 改善科技 Recall，同时科技 Precision 略降，新增教育与科技、财经与娱乐错误；不能将差异全归因于一个特征或断言 SVC 全面更好。

后续 C2（char bigram + χ²）优先检验分词边界、专有名词、技术型号与未登录词对表示覆盖的影响；χ² 必须只在允许的训练数据上拟合。它未必能解决正文跨主题与来源类别边界（例如电影拍摄设备报道），也可能放大来源线索。上述是下一阶段假设，本轮没有实现或选参。

目前 510 篇最终训练数据、90 篇 Test，线性基线已达到较高表现，没有充分证据必须继续 TextCNN。先完成 C2 的训练协议与误差定位，再决定是否为课程原理展示增加 C3；若开展新阶段，不应以重复查看这份 Test 来迭代调参，需保留本阶段结果冻结并事先明确新增比较协议。

## 验证与执行记录

新增 10 项 toy 分类测试通过，覆盖 frozen split 指纹/互斥、fit 数据范围、共享 matrix、确定性及 sklearn 并列预测一致性、指标/混淆矩阵顺序、保存加载、特征贡献还原、artifact 篡改、API schema。Toy 测试没有在 THUCNews 上拟合。另有 37 项既有 API / 页面 / 实验一 closeout / 实验二 summary / 扩展完整性定向回归测试通过；仅出现一条既有 Starlette/AnyIO 弃用警告。新增 Python 代码和共享注册文件 Ruff 检查通过。

冻结审计：原基线 782 个文件中，780 个字节不变；两项获准变更为 `app/main.py` 的分类注册和首页分类卡片状态。5 个固定 split 文件全部哈希不变，实验一、实验二模型/Run/metrics/final selection、课程扩展资产未修改；未安装包或重启开发服务。

Chromium / Playwright 复用 `http://127.0.0.1:8000`，检查 1440×1000 与 390×844。最新一轮 C0→C1 同一中文输入推理均返回 200、五个分数与特征贡献显示正确；NB 后验与 joint log-score、SVC 纯 decision score 区分正确。四 Tab 手机视口无页面横向溢出，错误/全部筛选分别为 5/90，桌面展示两个模型指标和五行混淆矩阵，无 console/page/HTTP 错误；已保存截图和准确 UTF-8 请求/响应。此前也验证 overview、90 条 predictions、errors_only、五类全局特征、validation export 与非法请求处理。

验收历史局限：早期 C1 浏览器 POST 两次出现 503，但相同 UTF-8 payload 的独立 CLI 推理成功。服务已细分超时、进程启动和非零退出诊断；随后独立 C1 路径以及完整 C0→C1 复验均通过。早期 503 根因仍未确认，不宣称已彻底消除；没有为此重训或重评 Test。历史与最新成功证据同时保留。

Chromium QA 与旧资产哈希审计的实际结果见完成记录 `outputs/exp3/quality/20261001_123337_9c1e7341.json`（与正式 Run 分开，避免修改冻结产物）。

```text
formal_exp3_validation_fits_executed=true
formal_exp3_final_fits_executed=true
formal_exp3_test_evaluation_executed=true
toy_test_fit_executed=true
exp1_modified=false
exp2_modified=false
lab4_started=false
```
