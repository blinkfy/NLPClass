# 阶段交付与验证记录

本记录区分工程功能完成和外部数据/模型准备。P0 已完成后再进入 P1；P2 保留为后续扩展。

| Phase | 完成内容 / 关键模块 | 验证 |
| --- | --- | --- |
| 0 | 空目录检查，Git 初始化，确认解释器 | Python 3.11.16 独立环境 |
| 1 | pyproject、配置、异常、FastAPI 健康检查 | editable 安装、test_smoke |
| 2 | schema、JSONL、来源/去重、BMES codec、toy | test_data_bmes |
| 3 | HMM 自行统计 π/A/B，合法平滑与 UNK | test_hmm：归一化、MLE、非法 Gold |
| 4 | score/backpointer 四步 Viterbi | test_viterbi：与穷举一致、首尾/长度、未知字符 |
| 5 | HMMSegmenter、原文/空白保留 | test_segmenter：中文、混合 Unicode、空输入往返 |
| 6 | span P/R/F1、Tag Accuracy、OOV/IV、NER exact span | test_evaluation：重复词、句间隔离、类型/跨度 |
| 7 | JSON+NPZ 模型、独立 Run、原子 latest | test_persistence：加载一致、损坏模型、留出拦截 |
| 8 | MSR standard/clean、完整官方划分或清理划分、来源与 protocol 清单；CLI | test_msr_cli、test_msr_protocols：样本保留、清理、重叠及指纹校验 |
| 9 | 五个网页、版本化 API、服务注入 | test_api：页面 200、API、缺失模型、未来端点未注册 |
| 10 | 参数/指标/混淆图、错误分析、报表与导出 | Playwright：状态/Top-K、PNG、与 Run 指标一致、人工类别 |
| 11 | 现代 HanLP adapter、本地模型准备命令 | test_ner_adapter：标签映射、字符偏移、缺失资源；未做真实模型性能测试 |
| 12 | README、依赖锁、格式、安装包、验收核对 | 41 项 pytest；Ruff；pip check；wheel 与网页实测 |

浏览器采用 Playwright CLI（当前会话未列出 Browser 技能），桌面 1440×1000、窄屏 390×844。已验证交互分析、参数切换、指标和存储一致、导出 PNG、错误筛选/人工类别、报告模式、窄屏不横向溢出和无应用控制台错误。概念与实际截图使用 view_image 对照导航、布局、文字、颜色、间距和图表；必要运行态差异见 design/README.md。

当前外部准备仍缺 MSR 原始语料、统一真实课程数据与人工 Gold、HanLP 权重。按正式需求提供导入/验证、setup、错误状态和说明；没有用合成数据或 mock 结果代替真实课程验收。下一步是导入本人语料并准备模型，执行两条正式评价线，再随课程资料实现实验二、三、四。


## 正式实验前规范修订

- 正式 benchmark 默认 standard，完整保留官方 train/test；仅 clean 执行去重、去污染和 dev 划分。protocol 随训练元数据与 Run 保存，允许官方 overlap 的例外须经同一清单及完整记录指纹校验。
- configs/hmm_mle.yaml 为课程 baseline/CLI train 默认，configs/hmm_smooth.yaml 为 alpha=0.1 增强模型；两者保留合法 BMES mask。
- HanLP 映射移至 configs/hanlp_msra_bert.yaml，准备与运行加载均校验实际模型词表。仅通过小型官方归档元数据核对了当前模型的 NR/NS/NT，未下载权重或生成正式数据。
- 本次完整验收：67 项 pytest 通过（保留一条既有第三方弃用警告），Ruff check 与 format --check 通过，pip check 无依赖冲突。上表 Phase 12 的 41 项为初始交付记录。
- 当前页面设计和实验二、三保持原有状态。
