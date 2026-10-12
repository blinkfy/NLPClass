# 浏览器推理可行性与落地范围

GitHub Pages 可以托管模型参数文件。网页加载这些静态参数后，计算发生在用户浏览器中，仍然属于纯静态部署；不要求 Python API。

| 模型 / 任务 | 可行性 | 当前实现 |
| --- | --- | --- |
| HMM-1、Longest | 高；词典 / 概率表 + 动态规划 | 已实现分词和 Trace，与 Python 对照 |
| C0 MultinomialNB | 高；HMM 分词、词级 TF-IDF、log 概率加权与 Softmax | 已实现新文本预测、后验和特征贡献 |
| C1 LinearSVC | 高；与 C0 共用 HMM / TF-IDF，线性加权 | 已实现新文本预测、decision score 和特征贡献 |
| C2 char + χ² + LinearSVC | 高；10,000 个冻结特征、5 类，字符处理无需分词 | 已实现字符 bigram、TF-IDF/L2、线性得分及解释 |
| 感知机、线性链 CRF | 原理可移植；需要另迁特征模板、稀疏权重及 CRF 动态规划 | 当前版没有完整移植 |
| C3 TextCNN | 原始 float32 二进制权重；Embedding、Conv1d、ReLU、有效窗口 max pool、Linear | 已在 Web Worker 实现新文本分类和卷积过程 |
| BiLSTM-CRF | 原始 float32 权重；双向 LSTM、Linear emission、BMES mask、CRF动态规划 | 已在 Web Worker 实现真实分词与短文本诊断 |
| BERT / HanLP NER、POS、依存模型 | 技术可行，需要转换模型、迁 tokenizer/后处理并评估体积 | 当前版未移植 |
| Qwen 0.5B | 技术可行，但需要量化、分片、生成循环、内存与设备适配评估 | 当前版未移植；不能仅把 Python API URL 改掉 |

神经模型可使用 [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 的 WASM / WebGPU 执行；GPU 后端支持的算子有范围限制。多线程 WASM 还受 [crossOriginIsolated](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html) 条件约束，不能把普通 GitHub Pages 的响应头配置当作已经具备。因此迁移必须核对实际算子、单线程回退、首载体积和峰值内存。

本轮 TextCNN / BiLSTM 直接转换为 little-endian float32 参数文件和 JS 运算内核；采用 Web Worker CPU，不需要额外运行库、CDN、COOP/COEP 响应头或 GPU。模型在首次实际推理时加载，长度与 SHA256 检查后复用；保留最近三个输入的计算缓存。TextCNN 权重 7,820,820 bytes，BiLSTM 权重 3,707,504 bytes，未量化、不重训。C3 保留 PAD/UNK、768词 head-tail、短序列零响应及通道贡献；BiLSTM 保留每个空白字符、逐 code point 编码、LSTM门顺序和合法BMES约束。

对照只使用合成输入。Node 内核与原 PyTorch 的 CNN logits/softmax/pooled 满足 `atol=rtol=2e-5`，argmax/类别一致；BiLSTM emission 满足 `atol=rtol=1e-5`，标签和 CRF 路径一致。CRF 同时执行 float32 Viterbi 解码与独立 log-sum-exp 数值重建，检查路径一致；不把它替换成规则分词。此数值容差不表示每种浏览器/设备的性能都相同。

传统分类器的静态导出包含冻结词表、IDF、模型权重、截距、标签顺序和 Run 映射。C0/C1 使用原 HMM tokenizer 和 unigram/bigram；C2 保留原字符规则、χ²选定列、sublinear TF 和 L2。没有在浏览器重新拟合词表、IDF、χ²或权重，也不会把新输入计入正式 Test 结果。

C2 的 Unicode 有效字符范围按照原 Python Unicode 版本导出；浏览器按 code point 遍历，未在该版本分配的字符保留原样，避免直接套用较新的 `Script=Han` 范围。分类结果不是规则关键词猜测；NB 后验未校准，SVC decision score 不是概率，卷积片段也不是因果解释。
