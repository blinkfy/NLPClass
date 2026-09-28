# 标注决策日志

## v1 已定规则

依据 `annotation_guideline.md`、`sequence_gold.schema.json` 与
`configs/course_corpus_plan.yaml` 初始化。当前使用 `configs/course_corpus.yaml` 执行。

- 主题与体裁分开。600 文档目标：六类各 100；news/blog/comment/fiction 为
  340/145/95/20，逐类交叉配额按计划文件执行。同 URL 最多 3 文档。
- 原文不删标点、空白，不繁简转换；NFC/空白归一化仅用于重复检测。
- 200 候选固定 seed=42，每文档最多一条，15–80 Python code point。
  主题配额 34/34/33/33/33/33；体裁 110/45/30/15，允许 ±3。
  优先精确配额；仅使用计划允许的容差，并记录实际偏差。禁止模型错误驱动抽样。
- 句切分保留原字符和 source offsets；引号中的内部句末不截断成失配引号。
  无合格句时报告缺口，不截短、不重复文档、不放宽长度。
- `document.text[source_start:source_end] == sample.text`；实体跨度同样使用
  Python Unicode code point `[start,end)`。浏览器使用 `Array.from(text)`，不用 UTF-16 下标。
- 分词遵循固定 `msr_like_v1`：标点独立；人名、明确地名、完整机构名尽量整体。
  机构后缀在名称内；固定词/成语整体；连续数字/小数、百分数、明确日期整体；
  日期与时间可分；连续英文/缩写整体，名称内部连字符保留。
- NER 仅 PER/LOC/ORG，非嵌套最长实体；机构名内不再嵌套 LOC。
  人名外的职称一般不进入 PER；外围引号/括号一般不进入实体；泛指机构不标。
- `entities=null` 是未标注，`[]` 是人工确认无实体。所有实体须 text/span 一致、不重叠。
- 候选 `gold_tokens=null`，不得自动造 Gold。人工点击边界或明确确认分词后才写草稿。
  BMES 仅从 tokens 派生，始终验证 tokens roundtrip，不单独保存或手改 BMES。
- HMM/HanLP 参考建议默认隐藏、只读，标为“参考建议，非 Gold”；不自动应用。
  未准备模型不阻止标注。HMM 只可使用与冻结指纹一致的 Smooth 模型。
- 状态 pending/in_progress/done/reviewed。done 要求两项标注均已人工确认；
  reviewed 需要对 done 显式进行第二遍复核。修改已完成 Gold 后重新进入 in_progress。
- `revision_count` 定义为显式第二遍复核次数，不是保存次数；草稿保存 `review_history`。
  样本 notes 不自动修改全局规范。版本冲突拒绝覆盖；每条草稿原子写入独立 JSON。
- 正式导出 exactly 200 reviewed；sample_id/doc_id 唯一、来源/分词/BMES/实体全部合法，
  主题/体裁配额满足；含实体样本 >=100，PER/LOC/ORG 各 >=40 个实体，
  困难样本 >=40，至少 40 条含明确第二遍复核记录。拒绝 demo，拒绝覆盖已导出结果。
- 旧 SequenceSample 必须有完成的 tokens；未完成候选使用独立草稿结构。
  可选来源/工作流字段不改变旧 demo/MSR 的 JSON 形状及指纹。
  正式 entity JSON 仅含 text/type/start/end，禁止将模型 score 写成 Gold。
- 自建 Gold 不参与 HMM 训练/调参；冻结前不计算自建集最终性能。

## 待人工补充的歧义决策

目前无新增歧义决策。人工发现重复歧义时，记录日期、问题、候选方案、最终规则、
影响样本及复核人，再统一执行；不得让某一条 notes 自动成为全局规则。
