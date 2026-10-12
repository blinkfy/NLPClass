# 静态部署改动审核

原系统使用 FastAPI 渲染 Jinja 模板，前端通过根路径 API 读取本地冻结 Run 或启动 Python 模型。GitHub Pages 提供文件服务，无法执行这些 Python 接口，因此仅复制 `app/static` 不能运行系统。

| 依赖与风险 | 已实施改动 | 验证依据 |
| --- | --- | --- |
| 服务端 Jinja 页面 | 导出器调用页面函数，保存 UTF-8 HTML | 每个课程页面生成独立 `index.html` |
| 根路径 `/static`、导航、JS 动态跳转 | 相对资源、目录路由、页面级 `<base>` 和静态 URL 适配 | 在 `/github-pages/` 仓库子路径下浏览器验证 |
| `/sequence`、`/analysis` 旧路由 | 静态重定向页面保留查询与 hash | 导出文件检查 |
| 依赖 `outputs` 的 GET API | 只读导出冻结响应，浏览器按清单加载 JSON | 清单保存响应路径、状态、缺失项 |
| 文档检视重复附带大型词表/特征统计 | 相同模型元数据按内容 SHA256 共享，浏览器恢复完整响应 | 只改变存储结构，不改变原始数值 |
| HMM 分词及 Viterbi Trace | 导出冻结概率，在浏览器按合法 BMES 路径解码 | 与现有 Python 实现逐项比对，覆盖空白/emoji/英文 |
| Longest 分词与候选过程 | 导出冻结训练词典，浏览器最长匹配 | 与 Python 词典算法及 Trace 对照 |
| 错误案例页码/筛选，HMM Top-K | 导出全量错误记录及 Top-100；浏览器筛选和截取 | 浏览器交互与实际结果检查 |
| 聚类/分类运行参数和图表 | 保存冻结矩阵摘要、预测、坐标与文档检视 | 不拟合表示，不重算正式指标 |
| Python C0/C1/C2 分类 | 冻结特征、IDF、NB/SVC 权重导出，浏览器端到端计算 | 与 Python 预处理、概率/分数及特征贡献对照 |
| TextCNN / BiLSTM-CRF | 原 float32 参数转二进制，Web Worker执行卷积/LSTM/CRF | 合成输入与 PyTorch 数值及路径对照 |
| Bigram / HMM2 / 感知机 / CRF | 导出冻结计数与权重，迁移特征模板及动态规划 | 20组合成输入与原模型数值/路径对照 |
| NER/POS、依存、BERT | 实际冻结权重转ONNX分片、原tokenizer/后处理，Worker执行WASM | 原模型数值、实体/树、浏览器实际推理检查 |
| 信息抽取/综合单篇分析 | 前端编排已接入模型、原类型关系规则与原文选区跨度 | 合成规则对照，实际API/页面检查 |
| 服务器文件下载 | 静态快照 Blob 下载 | 真实下载事件验证 |
| 人工标注、模型索引构建 | 静态版不保存服务器状态 | 使用本地完整版完成这些操作 |
| 大型权重/运行环境/个人数据 | 仅导出必要资源；全文字段短预览或省略，清理本机路径/个人密钥 | 默认数据策略和清单 |
| Jekyll 与分支发布 | `docs/.nojekyll`、可直接提交的 `docs/` 目录 | 增量发布保留原课程文档；Pages选择分支的 `/docs` |

实现位于 `scripts/export_github_pages.py`、`scripts/publish_github_pages.py`、三个模型导出脚本、`scripts/static_export_client.py`、`scripts/compact_static_snapshots.py` 和新增 `static-*.js`。静态适配脚本仅注入导出 HTML，不改变本地 FastAPI 页面行为，也不修改冻结实验文件。

导出直接调用同步只读接口，避免 Windows 沙箱中 TestClient 的 socketpair 限制，不启动后端服务。输出再次导出时清理已识别的旧导出目录，防止先前包含全文的静态文件残留。未知目录不会被清理。

目前文档检视的导出范围为：实验二主选 R2 及 latest Run 的 K=5，实验三保存的 Test 文档及部分模型解释，综合分析课程结构和保存分类索引。其他参数组合或未导出的模型过程会提示不可用；缺失项目记录在 `static-manifest.json`，不会用另一个 Run 代替。

部署步骤见 [GitHub Pages 静态版](github_pages.md)。Qwen、临时聚类、批量索引及共享标注仍有明确边界；详见 [推理迁移审核](static_inference.md)。资源状态来自实际导出的模型，不依据本机Python资源是否存在。
