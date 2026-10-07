"""Student-facing application checks; not a model-accuracy benchmark."""

from time import perf_counter

from nlplab.comprehensive.contracts import validate_annotations, validate_text
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError

SAMPLE = "微软公司在北京举办技术交流活动。张明介绍了人工智能软件的开发进展，并计划与北京大学合作研究。"
CUSTOM_EXAMPLES = [
    {"title": "软件开发", "text": "微软公司在北京发布人工智能软件，研究团队介绍软件开发和计算机技术。"},
    {"title": "计算机研究", "text": "北京大学研究人工智能算法，计算机团队开发新的软件技术。"},
    {"title": "篮球比赛", "text": "篮球球队在北京参加体育比赛，球员训练后准备篮球联赛。"},
    {"title": "足球训练", "text": "足球球队加强训练，球员准备体育比赛和足球联赛。"},
    {"title": "基金市场", "text": "基金公司介绍股票市场走势，投资者关注基金投资和金融市场。"},
    {"title": "股票投资", "text": "金融公司发布股票市场报告，投资者研究基金和股票投资。"},
]


def run_system_tests(service):
    started, records = perf_counter(), []
    def record(name, category, passed, expected, actual, elapsed=0):
        records.append({"name": name, "category": category, "passed": bool(passed),
                        "expected": expected, "actual": actual, "elapsed_ms": round(elapsed, 2)})
    tick = perf_counter()
    try:
        document = service.analyze(SAMPLE)
        record("单文本综合分析", "功能", document["status"] == "ok", "四项基础能力有真实结果", document["status"], (perf_counter()-tick)*1000)
        for key, label in (("segmentation", "中文分词"), ("pos", "词性标注"), ("ner", "命名实体识别"), ("classification", "五类新闻分类")):
            status = document["modules"][key]
            record(label, "功能", status["status"] == "ok", "成功或明确资源提示", status.get("error") or "真实模型已运行", status.get("elapsed_ms", 0))
        for field, key in (("form", "tokens"), ("word", "pos"), ("text", "entities")):
            validate_annotations(SAMPLE, document["lexical"][key], field)
        record("原文跨度对齐", "功能", True, "所有词/实体能映射回同一原文", "字符跨度逐项吻合；不同任务保留原输入协议")
    except (DataValidationError, ResourceUnavailableError) as error:
        record("单文本综合分析", "功能", False, "四项基础能力有真实结果", str(error))
    tick = perf_counter()
    collection_id = None
    try:
        collection = service.custom_collection(CUSTOM_EXAMPLES, 3)
        collection_id = collection["id"]
        record("多文档临时聚类", "功能", len(collection["clusters"]) == 3, "6篇文档分为3个有效簇", f"{len(collection['documents'])}篇 / {len(collection['clusters'])}簇", (perf_counter()-tick)*1000)
        profile = service.profile(collection_id, collection["documents"][0]["id"], analyze=True)
        linked = service.link(collection_id, [profile["id"]])
        record("文档与集合结果联动", "功能", linked["analyzed_count"] == 1 and profile["clustering"]["cluster_id"] == collection["documents"][0]["cluster_id"],
               "同一文档、真实簇分配与分析结果对应", "已完成文档可从二维位置钻取到原文和各任务结果；图形点击另由页面检验")
    except (DataValidationError, ResourceUnavailableError) as error:
        record("多文档与联动", "功能", False, "临时聚类可运行且不覆盖正式结果", str(error))
    for name, action in (
        ("空文本", lambda: validate_text("  ")),
        ("纯标点", lambda: validate_text("，。！？")),
        ("超长输入", lambda: validate_text("文" * 5001)),
        ("文档数量不足", lambda: service.custom_collection(CUSTOM_EXAMPLES[:2], 2)),
        ("非法K", lambda: service.custom_collection(CUSTOM_EXAMPLES[:3], 3)),
        ("模型资源缺失（受控场景）", lambda: service._run_worker("lexical", {}, runtime=service.settings.root / "__lab4_missing_resource_probe__/python.exe")),
    ):
        tick = perf_counter()
        try:
            action()
            record(name, "边界", False, "拒绝无效请求并提示原因", "未拒绝请求")
        except (DataValidationError, ResourceUnavailableError) as error:
            record(name, "边界", True, "拒绝/资源提示；不伪造结果", str(error), (perf_counter()-tick)*1000)
    tick = perf_counter()
    try:
        plain = service.analyze("生活越来越美好。")
        ner_ok = plain["modules"]["ner"]["status"] == "ok"
        count = len(plain["lexical"]["entities"])
        record("无实体文本", "边界", ner_ok and count == 0, "真实NER为空时显示空集合", f"真实输出 {count} 个实体；不手工删除模型误识别", (perf_counter()-tick)*1000)
    except (DataValidationError, ResourceUnavailableError) as error:
        record("无实体文本", "边界", False, "空实体集合可显示", str(error))
    return {"scope": "实验四应用系统功能/边界测试", "records": records,
            "elapsed_ms": round((perf_counter()-started)*1000, 2), "collection_id": collection_id,
            "notice": "测试验证功能契约与边界处理，不是模型总体准确率。资源缺失使用受控路径，不删除现有模型；图形交互由当前页面选择动作检验。"}
