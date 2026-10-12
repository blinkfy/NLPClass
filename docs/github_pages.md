# GitHub Pages 静态版

仓库的 `docs/` 目录包含已经渲染的 HTML、本地 CSS/JavaScript/ECharts、冻结实验响应和浏览器模型。访问网站不需要 Python、Node、数据库、远程 API 或 CDN。原有课程 Markdown 文档保留。

## 部署

1. 将本次修改及 `docs/` 中生成的静态资源提交并同步到 GitHub（例如 `git add docs app/static/js app/static/vendor/onnxruntime scripts tests README.md` 后正常提交、推送）。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **Deploy from a branch**。
3. 选择包含这些文件的分支（例如 `main`），Folder 选择 **`/docs`**，点击 **Save**。
4. 等待 GitHub 发布完成，Pages 设置会显示公开网址。以后推送该目录的修改会自动更新。

无需 ZIP、解压或自定义 Actions 工作流。GitHub 分支发布只支持仓库根目录和 `/docs`，不是任意子目录，见 [GitHub 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。本项目采用 `/docs`；保留 `docs/.nojekyll`，直接提供已经生成的文件。

所有页面使用相对资源和目录路由，同时兼容 `https://用户名.github.io/仓库名/`、用户主页仓库和自定义域名。请使用导航生成的、以 `/` 结尾的页面 URL。旧 `/sequence/` 和 `/analysis/` 链接保留重定向。

## 功能边界

| 功能 | 静态版行为 |
| --- | --- |
| 首页、课程说明、页面导航 | 静态 HTML，保留全部模块入口 |
| 冻结指标、对比、误差案例、参数、训练记录 | 从导出的 JSON 加载；不重训、不重评 |
| 聚类与分类图表 | 浏览器绘图；读取保存的冻结结果和坐标 |
| 错误案例筛选与翻页 | 浏览器筛选完整导出记录 |
| HMM-1 / 最长匹配 / Bigram / 二阶HMM | 浏览器读取冻结模型并计算；支持 Unicode 与过程回放 |
| 感知机 / 线性链 CRF | 浏览器计算原特征权重、最优路径、贡献及边缘概率 |
| C0 / C1 / C2 新文本分类 | 浏览器复用冻结 HMM / 字符特征、IDF 和线性权重；输出预测与贡献解释 |
| C3 TextCNN | 浏览器 Web Worker 运行真实卷积、有效窗口池化、分类；支持新文本与已保存编码的过程检视 |
| BiLSTM-CRF | 浏览器 Web Worker 运行双向 LSTM、发射层和受约束 CRF，支持分词与短文本诊断 |
| POS、NER、依存句法、BERT | 浏览器Worker / 单线程WASM；按需下载权重，NER采用INT8，其余保持float32 |
| 信息抽取、综合单篇/选区分析 | 前端复用HMM、POS、NER、C2、依存与原关系规则 |
| Qwen生成 / 新文档临时聚类 | 尚未移植；保留实验记录和具体原因 |
| 人工标注与索引构建 | 静态网站不保存服务器状态，请使用本地完整版 |
| 下载 | 从导出的静态文件下载；未导出的组合会提示 |

`static-manifest.json` 记录实际导出的响应、模型和缺失项；`export-integrity.json` 保存导出文件大小与 SHA256。可选资源缺失不会伪装为可用。原始正式实验的数值和来源标识不修改。

前端推理的具体可行性与当前实现见 [推理迁移审核](static_inference.md)。

## 数据与重新导出

默认省略课程语料的完整原文字段，保留节选、实验统计和必要展示字段；个人选择密钥与本机路径清理。当前转换的 TextCNN / BiLSTM 权重约为 7.8 / 3.7 MB，按需加载。确认拥有语料再分发权后，可在导出命令中添加 `--include-corpus`。

本轮加入大模型后完整目录约870 MB。BERT首次运行约需下载410 MB，NER约152 MB、POS约49 MB、依存约53 MB；无需打开页面时预下载。模型分片小于90 MB，继续使用普通Git文件与`/docs`发布。

在已有模型和冻结结果的原工程根目录执行：

```powershell
.\.venv\python.exe scripts/export_github_pages.py
```

命令先在 `build/github-pages/` 生成资源，再按完整性清单增量发布到 `docs/`。不生成压缩包；不会删除或覆盖原有课程文档。只提交 `docs/` 生成的资源及本次源码修改，不要提交 `.venv`、`artifacts/runtime`、原始 `data` 或整个 `outputs`。

重新转换ONNX需要原HanLP/PyTorch运行环境及转换依赖。本次转换依赖隔离安装在`build/onnx-converter`，未升级原环境：`onnx==1.17.0`、`onnxruntime==1.31.0`。安装到此目录后导出器自动为转换子进程设置依赖路径。现有`docs/`已经转换完成，GitHub发布和访问无需这些环境。

本地预览必须通过 HTTP 静态服务，不能双击 `file://` HTML（浏览器会限制 JSON fetch 和模块）：

```powershell
.\.venv\python.exe scripts/preview_github_pages.py --port 8765 --directory docs
```

打开 `http://127.0.0.1:8765/`。该命令只提供文件服务，没有 FastAPI 后端，并为`.mjs`设置JavaScript MIME；某些Windows环境的默认HTTP服务会错误返回text/plain，导致模型Worker无法加载。
