# NLPLab 自建语料与 Sequence Gold 标注规范（v1）

## 1. 目标
统一构建约 600 条真实中文 Document，供实验二/三复用；从中抽取 200 个代表句/短文本，作为实验一独立泛化评价集。自建 Gold 不参与 HMM 训练或调参，已冻结的 MSR benchmark 不修改。

## 2. 主语料类别
固定 6 个主题，每类 100：
- technology 科技
- finance 财经
- sports 体育
- education 教育
- society 社会民生
- culture_entertainment 文化娱乐

`primary_label` 表示主题，`source_type` 表示体裁，二者严格分开。

体裁目标：
- news 340
- blog 145
- comment 95
- fiction 20

前五类各为 60 news + 25 blog + 15 comment；文化娱乐为 40 news + 20 blog + 20 comment + 20 fiction。

单条 Document：
- 推荐 40–400 Unicode code point；
- 保留原文，不在 raw 层删除标点、空白或做繁简转换；
- 保存真实 URL、来源标题、主题、体裁与指纹；
- 同一 URL 最多贡献 3 条 Document；
- 自动生成内容、toy、模型改写文本不得作为真实课程语料。

## 3. 200 条 Sequence Gold
每个 Document 最多抽 1 条，样本长度建议 15–80 code point，并保存 `source_start/source_end`，必须满足：
`document.text[source_start:source_end] == sample.text`

主题目标：34/34/33/33/33/33。
体裁目标（允许 ±3）：news 110、blog 45、comment 30、fiction 15。

抽样只能使用主题、体裁、长度、数字/ASCII/引号/复杂标点等表面属性；禁止依据 HMM 是否分错来挑样本。

## 4. 中文分词 Gold：msr_like_v1
这是“MSR-like、本地固定”的规范，不声称逐条复刻全部 MSR 细则。目标是尽量减少与课程采用的 MSR 训练语料的粒度偏移。

- Gold 只存 `gold_tokens`；BMES 必须自动生成。
- 普通标点独立为 token；外围引号/括号一般不进入实体。
- 人名、明确地名、完整机构全称尽量整体成词。
- 机构后缀（大学、公司、委员会、医院、研究所等）属于机构名的一部分。
- 固定词、成语、明显词典化表达优先整体标注。
- 连续数字/小数整体；百分数整体；明确日期如 `2026年9月26日` 整体。
- 日期与时间可分，例如 `9月26日 | 下午3点`。
- 连续英文/缩写作为一个 token；连字符属于名称内部时保留。

遇到重复出现的歧义边界，记录到 `docs/annotation_decisions.md`，之后统一执行。

## 5. NER Gold
只标 PER / LOC / ORG。

采用非嵌套最长实体原则：
- `北京市电信局` 整体 ORG，不再嵌套标 LOC；
- `清华大学计算机系人工智能实验室` 整体 ORG；
- `张三教授` 一般只标 `张三` 为 PER；
- 泛指“公司/学校/医院/政府部门”不标，确指简称才标。

Span：
- Python Unicode code point；
- `[start,end)`；
- 必须满足 `text[start:end] == entity.text`；
- 不允许实体重叠；
- `entities=null` 表示未标注，`[]` 表示人工确认无实体。

前端 JS 必须用 `Array.from(text)` 或等价 code-point 映射，不能把 UTF-16 index 直接当 Python span。

## 6. 标注流程
状态：`pending -> in_progress -> done -> reviewed`

1. 人工确定分词边界；
2. 标 PER/LOC/ORG；
3. 自动校验；
4. done 后第二遍审查；
5. reviewed 后才能进入最终 JSONL。

至少复核 40 条，记录 `revision_count`。

## 7. 标注工具
分词：把文本按 Unicode 字符渲染，在字符之间点击切换边界，内部维护 boundary positions，实时派生 tokens/BMES。
NER：点击或选择连续字符 span，再选 PER/LOC/ORG；可修改/删除；禁止重叠。
建议结果：冻结 Smooth HMM 和 HanLP NER 只能作为默认隐藏的“参考建议”，不能自动写入 Gold。
Draft：每条独立原子保存到 `data/annotations/work/<sample_id>.json`；最终 CLI 校验并合并成 `sequence_gold.jsonl`。

## 8. 最终验收
600 条：
- 真实去重 Document >=600；
- 6 类各 100；
- source_type 基本满足目标；
- URL、标题、指纹齐全，无 demo。

200 条：
- reviewed == 200；
- doc_id 各不相同；
- source offsets 可还原文本；
- `''.join(gold_tokens) == text`；
- entity text/span 一致且不重叠；
- 至少 100 条含实体；
- PER/LOC/ORG 各至少 40 个 Gold 实体；
- 至少 40 条具有数字、ASCII、全角、引号、混合脚本、日期/百分比等困难特征。

200 条冻结后才运行最终自建集评价。
