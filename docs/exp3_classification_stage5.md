# 实验三 Stage5：来源元数据与固定模板消融

独立 analysis Run：`20261002_023920_52c0b596`。完成日期：2026-10-02。

本阶段是探索性的 development-set diagnostic，只用固定 Train420 / Validation90，不是新的独立泛化 benchmark。Test90正文、历史Test预测与指标未读取；formal_test_evaluation_count=0。没有增加模型、重新选参或启动下一Stage。

## 协议与冻结

Cleaning rules只根据Train420制定，匹配阶段不使用label。冻结时间 `2026-10-02T02:39:20.352757+00:00`；首次Validation打开记录 `2026-10-02T02:39:32.042952+00:00`。cleaning_freeze.json保存规则、Train审计、协议、父配置和四个Stage5实现文件的SHA256；验证在Validation读取前、worker启动前和运行结束执行，规则/实现未变。标签仅用于冻结规则后的分组统计、原训练任务和评价。

C0 alpha=.5；C1冻结word TF-IDF、SVC C=2；C2冻结Count min_df2、χ² k10000、TF-IDF、SVC C=2；C3冻结word tokenizer/min_freq2/max_len768/Embedding128/Conv3,4,5各128/Dropout.5/AdamW lr.001、wd1e-4、batch32、12epochs、seed42/clip5/CPU2threads deterministic。无早停、无超参数搜索。

OO/OS/OT/OA复用Stage4 `20261001_164602_48e6c6b6` 的ladder101_n420模型。C0/C1/C2只load一次并复用同一对象，C3四种输入使用同一已加载模型。16个原表示/模型相关文件与Stage4manifest核验一致。四模型OO predictions和完整metrics均与Stage4 420点完全一致，通过后才开始AA。

OO = Original Train模型→Original Validation；OS = 原模型→Source-clean Validation；OT = 原模型→Template-clean Validation；OA = 原模型→All-clean Validation；AA = All-clean Train从头拟合→All-clean Validation。

## Train-only审计与规则

Offsets统一指NFKC/lowercase文档。仅删除明确元数据span，保留正文实体和后续段落。

| 规则 | Train涉及篇数 | Match次数 |
| --- | ---: | ---: |
| sina_paragraph_prefix | 124 | 128 |
| standalone_author | 13 | 13 |
| parenthetical_reporter | 4 | 4 |
| exam_disclaimer | 20 | 21 |
| exam_channel_navigation | 20 | 21 |
| comment_cta | 7 | 8 |
| mydaily_copyright | 1 | 1 |

具体规则：

1. 行/段首八个真实署名：新浪体育讯、新浪娱乐讯、新浪科技讯、新浪教育讯、新浪财经讯、新浪环球地理讯、新浪考试讯、新浪音乐讯；只删署名token及相邻水平空白。
2. 独立“作者:”行（≤80字符），以及“(记者:短中文姓名)”这种明确括号署名。
3. 完整考试免责声明、七条Train观察到的完整考试频道导航、独立评论CTA、完整“版权所有mydaily禁止转载”标记。

规则完整regex、七条导航字符串与理由在cleaning_rules.json，匹配id/text/offset在train_audit.json。没有全局删除“新浪”“体育”“娱乐”“考试”等词。22篇中发现URL；它们包括投票与反钓鱼报道内容，保留。采访中的“记者：”、正文来源描述、相关报道/编辑观点标题和边界不确定的通讯社电头保留。因此本消融是保守、有限模式的诊断，不是彻底清除一切来源痕迹。

删除边界：C0/C1对每个保留segment分别生成原有word uni+bi，Count/TF-IDF参数不变；C2用newline边界分隔；C3词序列保留segment IDs，仅同segment完整窗口参与原Conv/ReLU/max pooling。没有新增层或参数。原head-tail截断语义保留；清理span形成的连接不能产生新窗口。Toy验证无删除时与原TextCNN数值一致。

## 分类别出现率

清除字符比例=各类别被规则删除的NFKC字符总数/该类别原NFKC字符总数；不是平均每篇比例。source/template篇数可重叠。

| Split | 类别 | 篇数 | Source出现率 | Template出现率 | 清除字符比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 财经 | 84 | 5.9524% | 8.3333% | 0.1408% |
| train | 教育 | 84 | 15.4762% | 23.8095% | 2.0531% |
| train | 科技 | 84 | 30.9524% | 0.0000% | 0.3040% |
| train | 体育 | 84 | 64.2857% | 0.0000% | 0.4826% |
| train | 娱乐 | 84 | 51.1905% | 1.1905% | 0.5532% |
| validation | 财经 | 18 | 0.0000% | 11.1111% | 0.1565% |
| validation | 教育 | 18 | 11.1111% | 22.2222% | 1.7107% |
| validation | 科技 | 18 | 22.2222% | 0.0000% | 0.1947% |
| validation | 体育 | 18 | 55.5556% | 0.0000% | 0.4402% |
| validation | 娱乐 | 18 | 50.0000% | 0.0000% | 0.4950% |

来源明显按频道分布：Train体育64.29%、娱乐51.19%，Validation体育55.56%、娱乐50%；财经Validation未匹配source。教育template集中，Train23.81%、Val22.22%。这里的出现率只针对冻结匹配规则。

## 全部条件指标

| 模型 | 条件 | Accuracy | Macro P | Macro R | Macro F1 | F1 Δ vs OO (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C0 | OO | 94.4444% | 94.9474% | 94.4444% | 94.2828% | +0.0000 |
| C0 | OS | 92.2222% | 92.6140% | 92.2222% | 92.0948% | -2.1880 |
| C0 | OT | 94.4444% | 94.9474% | 94.4444% | 94.2828% | +0.0000 |
| C0 | OA | 92.2222% | 92.6140% | 92.2222% | 92.0948% | -2.1880 |
| C0 | AA | 92.2222% | 92.6140% | 92.2222% | 92.0948% | -2.1880 |
| C1 | OO | 97.7778% | 97.8947% | 97.7778% | 97.7424% | +0.0000 |
| C1 | OS | 95.5556% | 95.6656% | 95.5556% | 95.5521% | -2.1903 |
| C1 | OT | 97.7778% | 97.8947% | 97.7778% | 97.7424% | +0.0000 |
| C1 | OA | 95.5556% | 95.6656% | 95.5556% | 95.5521% | -2.1903 |
| C1 | AA | 94.4444% | 94.7709% | 94.4444% | 94.4634% | -3.2790 |
| C2 | OO | 98.8889% | 98.9474% | 98.8889% | 98.8880% | +0.0000 |
| C2 | OS | 97.7778% | 97.8363% | 97.7778% | 97.7769% | -1.1111 |
| C2 | OT | 98.8889% | 98.9474% | 98.8889% | 98.8880% | +0.0000 |
| C2 | OA | 97.7778% | 97.8363% | 97.7778% | 97.7769% | -1.1111 |
| C2 | AA | 94.4444% | 94.9405% | 94.4444% | 94.5115% | -4.3765 |
| C3 | OO | 86.6667% | 88.8095% | 86.6667% | 86.6184% | +0.0000 |
| C3 | OS | 80.0000% | 83.4411% | 80.0000% | 78.6476% | -7.9708 |
| C3 | OT | 86.6667% | 88.8095% | 86.6667% | 86.6184% | +0.0000 |
| C3 | OA | 80.0000% | 83.4411% | 80.0000% | 78.6476% | -7.9708 |
| C3 | AA | 91.1111% | 91.6784% | 91.1111% | 91.2519% | +4.6335 |

每个条件完整per-class P/R/F1/support、5×5confusion matrix、90篇predictions/score/explanations已保存；固定顺序财经/教育/科技/体育/娱乐。

| 模型/条件 | 财经 F1 | 教育 F1 | 科技 F1 | 体育 F1 | 娱乐 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0/OO | 94.74% | 97.30% | 87.50% | 97.14% | 94.74% |
| C0/OS | 94.74% | 97.30% | 84.85% | 94.12% | 89.47% |
| C0/OT | 94.74% | 97.30% | 87.50% | 97.14% | 94.74% |
| C0/OA | 94.74% | 97.30% | 84.85% | 94.12% | 89.47% |
| C0/AA | 94.74% | 97.30% | 84.85% | 94.12% | 89.47% |
| C1/OO | 100.00% | 97.30% | 94.12% | 100.00% | 97.30% |
| C1/OS | 100.00% | 97.30% | 91.43% | 97.14% | 91.89% |
| C1/OT | 100.00% | 97.30% | 94.12% | 100.00% | 97.30% |
| C1/OA | 100.00% | 97.30% | 91.43% | 97.14% | 91.89% |
| C1/AA | 100.00% | 97.30% | 91.43% | 94.12% | 89.47% |
| C2/OO | 100.00% | 100.00% | 100.00% | 97.14% | 97.30% |
| C2/OS | 100.00% | 100.00% | 97.30% | 97.14% | 94.44% |
| C2/OT | 100.00% | 100.00% | 100.00% | 97.14% | 97.30% |
| C2/OA | 100.00% | 100.00% | 97.30% | 97.14% | 94.44% |
| C2/AA | 100.00% | 100.00% | 88.24% | 97.14% | 87.18% |
| C3/OO | 92.31% | 94.12% | 76.19% | 97.14% | 73.33% |
| C3/OS | 90.00% | 91.43% | 66.67% | 97.14% | 48.00% |
| C3/OT | 92.31% | 94.12% | 76.19% | 97.14% | 73.33% |
| C3/OA | 90.00% | 91.43% | 66.67% | 97.14% | 48.00% |
| C3/AA | 97.14% | 94.44% | 81.08% | 94.12% | 89.47% |

C3对source最敏感：OS/OA Macro-F1下降7.9708pp、6篇correct→wrong；娱乐F1从73.33%降至48%，科技从76.19%降至66.67%。C0/C1来源删除各新增2个错误，C2新增1个。小样本差异只作描述，不能根据1个样本宣称稳定模型优势。

OT四模型全部90篇prediction保持不变，但scores/响应可以改变，且原解释确实有template命中。因此不能推断完全不使用模板；本次删除未越过预测边界。

AA相对OA：C0 F1不变；C1再下降1.0887pp；C2再下降3.2654pp；C3提高12.6043pp，并高于OO 4.6335pp。AA使用重新拟合词表/IDF/χ²或embedding，不能将恢复或下降归因于单一内部机制；这是清理pipeline的重新学习表现。C3 AA词表12071（original420=12077），同seed下词表编号/embedding对应改变，不能把它解读为纯“去掉几个词”的严格因果实验。

## Prediction flips

| 模型 | 条件 | correct→wrong | wrong→correct | wrong→wrong |
| --- | --- | ---: | ---: | ---: |
| C0 | OS | 2 | 0 | 0 |
| C0 | OT | 0 | 0 | 0 |
| C0 | OA | 2 | 0 | 0 |
| C0 | AA | 2 | 0 | 0 |
| C1 | OS | 2 | 0 | 0 |
| C1 | OT | 0 | 0 | 0 |
| C1 | OA | 2 | 0 | 0 |
| C1 | AA | 3 | 0 | 0 |
| C2 | OS | 1 | 0 | 0 |
| C2 | OT | 0 | 0 | 0 |
| C2 | OA | 1 | 0 | 0 |
| C2 | AA | 4 | 0 | 0 |
| C3 | OS | 6 | 0 | 0 |
| C3 | OT | 0 | 0 | 0 |
| C3 | OA | 6 | 0 | 0 |
| C3 | AA | 4 | 8 | 2 |

共45个model-condition flip记录（同一document在多个模型/条件可重复，并非45篇不同文档）。典型真实案例：

- C1 OA，THU-602f75176bc9f096a6553511：“帕克伊娃第3者曝光 前队友爱妻与帕克偷情1年(图)”，sports→entertainment。原高贡献含“新浪 体育”(0.0511)；正文确有体育与娱乐交叉，说明频道署名能帮助贴合数据标签。
- C2 OA，THU-b48229817f4aa7be3ff0eed2：“大牌明星时兴全裸拍广告 拉抬时尚买气(图)”，entertainment→science。原前三高贡献为“娱乐”(0.0793)、“乐讯”(0.0778)、“浪娱”(0.0772)，均与被删来源区间重合。
- C3 OA，THU-e23f5b0a5d00f03696be9566：“Gartner：第一季度服务器市场跌幅创纪录”，science→finance。最大响应片段包括“新浪 科技 北京 时间”，直接贡献0.2276；市场/营收内容与频道归类形成边界。
- C3 AA，THU-fc624d8c3c51cf36b9817509：“联合国里的中国美眉：实习生也可以有薪水(图)”，finance→education（修正）。原响应有“基金 提供 资助”；clean模型Top响应含“多国 学生 面对”“基金 一些 学生 免费”。不能将这次修正直接归因于删除记者姓名，因为AA还改变了训练表示与权重。
- C3 AA，THU-dcc1522f7bae66a44f30316c：“信用卡短信收费：兴业广发先发 中行交行跟进”，finance→science（新增错误）；该文未匹配任何清理span，说明clean retraining也会改变未被清理文档的决策。

## 原模型shortcut解释

| 模型 | Top解释与source重合篇数 | 与template重合篇数 |
| --- | ---: | ---: |
| C0 | 18 /90 | 5 /90 |
| C1 | 23 /90 | 6 /90 |
| C2 | 21 /90 | 6 /90 |
| C3 | 22 /90 | 6 /90 |

C1高贡献出现“新浪 体育”“新浪 娱乐”“新浪 科技”，C2出现“浪体”“育讯”“浪娱”“乐讯”“浪科”“技讯”。标注依据实际feature occurrence的字符offset与被删区域相交；词也可能在正文重复，TF-IDF不是区域可加分解，不能将整项contribution当成来源区域的精确贡献。

C3在16692个预测类别正贡献channel中，630个最大响应片段与source相交、334个与template相交；相交channel的正贡献分别占全体正贡献之和的8.7399%、4.2857%。这只是加法分解中的位置重合统计，片段可同时包含正文，不是attention或来源因果贡献比例。全部384channel保留；bias+channel contributions还原raw logit。C3展示Softmax输出与raw logits，未称已校准置信度。

结合类别相关出现率、真实解释重合和OS翻转，存在明显可利用的来源/频道捷径，C3的敏感性最明显。但正文与原标签之间也有真实主题边界：署名可能帮助复现原频道归类，删除后不能仅凭错误数量判断语义理解优劣。本结论局限于冻结规则和90篇开发集。

## 工程、运行时间与资产

复用冻结HMM、TextTFIDF、CharacterChi2TFIDF、WordSequenceVocabulary、TextCNN与统一classification_metrics；新增lazy Stage5Data/freeze I/O guard、span cleaner/segment adapter、Stage5 controller/neural worker、只读service/API和独立UI模块。无FastAPI算法依赖、无Web训练入口，模型保持C0/C1/C2/C3。

既有exp2-venv Python3.11.16/sklearn1.9.1与hanlp-venv Python3.11.16/torch2.14.0+cpu分别处理稀疏与神经，CPU2threads deterministic；未安装或变更依赖。AA保存JSON/NPY/NPZ无pickle模型。

总wall=122.59s；AA稀疏表示与三分类器合计4.11s；AA C3训练63.30s，12epochs。无训练重试、无参数选择。

Run内包含protocol、cleaning_rules/freeze、Train审计、父配置/模型hash引用、Validation打开记录、prevalence、cleaning offsets/segments、20组metrics与predictions、shortcut summary、45flip记录、AA表示/模型/词表、12epoch history、runtime/environment和artifact_manifest。仅新增outputs/exp3/stage5_latest.json，未更新旧latest。

## 测试、Chromium与冻结审计

11个toy tests通过（exp2数据/API6，hanlp神经5），Ruff通过。测试用synthetic fixtures，不混入formal结果；覆盖Test拒绝、freeze前Val拒绝、规则hash修改拒绝、OO表示等价、word/char/CNN删除边界、无新参数、贡献还原、固定12epochs和API模型白名单。

Chromium使用已有Python Playwright，Browser插件不可用。仅访问/classification?analysis=stage5；此视图跳过历史Test结果初始化与Stage4请求，保留四Tab和历史selector语义，comparison/errors两个Tab可操作。普通视图以source检查确认接入，未在本轮打开以免请求旧Test结果。桌面1440×1000、手机390×844，检查表见外部qa-evidence.json；旧API请求必须为0。正常显示20行结果、prevalence，C3/OA flip筛选与展开、rules导出和非法导出拒绝。无页面横向溢出、console/page errors。

冻结审计精确区分hash与metadata-only。Stage1~4模型/允许读取的资产hash核验；旧Test/result-bearing和Val预冻结快照仅stat比对，不读取Test字节。110个artifacts/models文件及小runtime配置独立baseline。Stage5 manifest76/76、cleaning freeze source hashes、OO sanity、冻结时间先于Val打开均核验。只有三个获准共享app API/template/page.js扩展和Stage5新增文件；旧数据、Run/latest没有修改。

外部证据目录：C:/Users/29513/.codex/visualizations/2026/10/01/01a0f750-fe2e-7b83-a6c6-d991ff6ccbab/stage5/。QA脚本经历nested summary locator歧义、异步flip渲染时序、compact table替代JSON后的旧断言三项修正；手机截图等待既有侧栏过渡完成，并验证收起后内容可读。训练、规则和Run资产均未因此修改。

```text
formal_exp3_stage5_original_models_reused=4
formal_exp3_stage5_AA_fits_executed=4
formal_exp3_stage5_condition_evaluations=20
formal_test_evaluation_count=0
test_accessed=false
stage1_modified=false
stage2_modified=false
stage3_modified=false
stage4_modified=false
exp1_modified=false
exp2_modified=false
lab4_started=false
```

