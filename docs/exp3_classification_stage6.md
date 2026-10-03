# Experiment 3 Stage6：Test 统计不确定性与配对比较

Stage6 analysis Run：`20261002_141803_299c95d8`。
本阶段只分析已冻结的正式 Test 标签和预测，不训练、不重新推理、不调参。
它不是新的独立 Test benchmark，也没有创建新分类器。

## 输入边界和对齐

引用 Stage1 `20261001_123337_9c1e7341` 的 C0/C1、Stage2
`20261001_133829_68ee50d7` 的 C2、Stage3 `20261001_153153_c99a8d6f` 的 C3。
没有找到覆盖完整 Test 的纯标签预测数组，因此使用最窄的已有
`C*_test_predictions.jsonl` 容器，只投影根字段 `document_id/gold/prediction`。
其余 JSON 值通过结构扫描跳过，不解码、提取或保留正文、分数、特征或解释。
原容器字节在同一次投影读取中计算 SHA256，与原 artifact manifest 核验。
这仍需读取容器字节，不能理解为物理上没有读到正文所在字节。
未打开原始数据 split、模型参数或特征资产。

四模型均有同一组 90 个唯一 ID，按显式 ID join、C0 保存顺序对齐；
gold 完全相同，财经/教育/科技/体育/娱乐各 18 篇。
重新计算的完整 point metrics 与冻结正式 metrics 字典完全相等。
不一致时控制器会在 bootstrap 前停止，而不会修正历史结果。

## 预注册方法

先写 protocol、输入引用/hash、bootstrap config、source hashes 和 preregistration，
再分析预测。固定 primary pairs 为 C0↔C1、C1↔C2、C2↔C3，不扩展假设族。
所有差异方向为 B−A。

- Accuracy、每类 Recall：95% Wilson score interval，无连续性校正。
- Macro-F1：class-stratified percentile bootstrap，B=20,000、seed=606、PCG64。
  每个 replicate 在每一类 18 篇中有放回抽 18 篇，仍保持五类各 18 篇。
  四模型和三个比较共用完全相同的重采样 indices；分位数为 2.5%/97.5%，linear。
- Accuracy 配对检验：two-sided exact McNemar，条件于 discordant counts 的二项检验。
  仅三个 primary McNemar p-values 做 Holm correction。
- ΔMacro-F1：共享 indices 下直接计算 B−A 的 percentile interval。
  这些是 marginal intervals，未进行多重比较校正，不能视为 Holm 检验的替代。

## 单模型结果

以下值均为百分比，方括号内为 95% interval。

| 模型 | 正确/n | Accuracy / Wilson CI | Macro-F1 / bootstrap CI |
|---|---:|---:|---:|
| C0 | 83/90 | 92.22 [84.81, 96.18] | 92.08 [86.13, 96.67] |
| C1 | 85/90 | 94.44 [87.65, 97.60] | 94.53 [89.21, 98.89] |
| C2 | 87/90 | 96.67 [90.65, 98.86] | 96.63 [92.10, 100.00] |
| C3 | 82/90 | 91.11 [83.43, 95.43] | 91.03 [84.90, 96.59] |

每类 Recall 及 Wilson 95% interval（每格 support=18；都是 marginal intervals）：

| 类别 | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| 财经 | 100.00 [82.41, 100] | 94.44 [74.24, 99.01] | 100.00 [82.41, 100] | 94.44 [74.24, 99.01] |
| 教育 | 88.89 [67.20, 96.90] | 88.89 [67.20, 96.90] | 88.89 [67.20, 96.90] | 72.22 [49.13, 87.50] |
| 科技 | 77.78 [54.79, 91.00] | 94.44 [74.24, 99.01] | 94.44 [74.24, 99.01] | 94.44 [74.24, 99.01] |
| 体育 | 100.00 [82.41, 100] | 100.00 [82.41, 100] | 100.00 [82.41, 100] | 94.44 [74.24, 99.01] |
| 娱乐 | 94.44 [74.24, 99.01] | 94.44 [74.24, 99.01] | 100.00 [82.41, 100] | 100.00 [82.41, 100] |

## Primary paired comparisons

差异单位为百分点；A/B 是左/右模型。

| A↔B | 都对 | 仅 A 对 | 仅 B 对 | 都错 | ΔAccuracy | exact p | Holm p | ΔMacro-F1 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0↔C1 | 81 | 2 | 4 | 3 | +2.22 | 0.6875 | 1.0000 | +2.44 [−2.29, +8.02] |
| C1↔C2 | 84 | 1 | 3 | 2 | +2.22 | 0.6250 | 1.0000 | +2.11 [−2.23, +6.54] |
| C2↔C3 | 82 | 5 | 0 | 3 | −5.56 | 0.0625 | 0.1875 | −5.60 [−10.81, −1.14] |

每个 outcome 的具体 document IDs 保存于 `pairwise_outcomes.json`，没有读取正文做样本解释。

C0→C1 与 C1→C2 看起来分别改善约 2 个百分点，但配对 F1 区间包含 0，
当前 Test 下差异具有较大不确定性。这不表示两模型相同。
C2↔C3 有相对更强的配对方向性：C2 独自正确 5 篇，C3 独自正确 0 篇，
且 C3−C2 的 marginal F1 interval 完全为负。
但是 exact McNemar raw p=0.0625、Holm p=0.1875，未达到 0.05；
当前样本不足以提供强证据表明正确率存在差异。
Macro-F1 与 Accuracy 是不同统计量，bootstrap percentile 与离散 exact 检验也不同，
不能将上述 F1 区间解读为正确率检验显著或模型普遍优越。

Test 仅 90 篇，1 篇对应总体正确率 1.11 个百分点；每类 18 篇，
1 篇对应 Recall 5.56 个百分点。即使 18/18，也只有 [82.41%,100%] 的 Wilson 区间。
区间主要反映冻结模型、固定类别数量下 Test 样本的采样不确定性；
不包含训练随机性、换数据集、分布漂移，也没有消除历史上已使用此 Test 的限制。

## 工程、产物和环境

新增 `classification/stage6.py` 控制器、`data/frozen_prediction_projection.py` 字段投影、
`evaluation/classification_uncertainty.py` 统计，以及只读 Stage6 service/API。
复用现有统一 classification metrics 和安全 JSON/NPZ 数值保存方式。
Web 在现有“模型对比”Tab 增加统计区间、三组配对表和零参考线图；
`/classification?analysis=stage6` 只请求 Stage6 API，不请求旧正文/模型推理接口。
普通历史 Run 和 classifier selector 保持原语义，不增加 C4 或 Tab。

目录：`outputs/exp3/stage6/20261002_141803_299c95d8/`。
保存 protocol、输入引用/hash、alignment、窄字段 aligned predictions、point metrics、
Wilson/Recall intervals、bootstrap config/summaries/数值 replicate NPZ、paired outcomes、
exact/Holm p-values、环境、execution、source hashes 和 artifact manifest。
仅新增 `stage6_preregistration.json` 和 `stage6_latest.json` 指针。

环境：Python 3.11.16、NumPy 2.4.6、SciPy 1.17.1、scikit-learn 1.9.1，
复用 exp2-venv，CPU statistics only；正式分析耗时 2.13 秒，不含实现/QA。
没有安装或变更冻结环境依赖。

初次 preregistration 发布因 I/O guard 拒绝根目录通用临时文件而停止，
未开始统计；未完成目录 `20261002_141222_d2196ea6` 留存 failure receipt。
修复为自己 Stage6 目录内 staging 后重新预注册，以上完成 Run 仅执行统计一次。

## 验证记录

Stage6 toy/regression tests：19 项通过（18 项主测试，加 1 项 manifest 导出回归）；覆盖对齐和正式指标复现、Wilson、
固定分层/shared indices、确定性、bootstrap 数值、exact McNemar/Holm、
根字段投影/hash、无 fit/inference、只读 guard/原子 pointer、API schema。
Toy 输入没有混入正式统计。
Chromium desktop 1440×1000 / mobile 390×844 均通过：四模型、三个 primary pairs、
B−A 方向、零参考线、20 行 Recall、90 行六字段导出均正常，无意外 console/page errors。
故意请求未知导出的 422 浏览器资源消息单列为预期结果。
移动端图表轴文字重叠及裁切已局部修复并聚焦复验；未请求非 Stage6 分类 API。
QA、quality receipt、截图和最终 frozen integrity 的证据保存在外部 Stage6 目录，
不回写已完成 Run 的 manifest。
完整性检查 15/15 通过：1,163 个旧 outputs、110 个模型文件、47 个数据文件的
size/mtime 与基线完全一致；45 个已钉住的 manifest/pointer SHA256 与基线一致。
七个旧 latest 指针初始基线没有 SHA pin，因此核验其 size/mtime 不变并记录当前 SHA，
不能将这七个指针称为已与初始 SHA 比较。Stage6 manifest 的 15 个条目全部 SHA256 通过。
四个正式预测容器及 formal metrics 则由本阶段投影流程按历史 manifest 验证 SHA256。
不能把 metadata-only 验证称为所有旧模型二进制均重新计算了 hash。

执行标记：`model_fit_count=0`、`new_test_inference_count=0`；
Stage1~5/Exp1/Exp2 frozen assets 未修改，Lab4 未开始。
