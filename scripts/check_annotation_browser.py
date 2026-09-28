"""Real Chromium acceptance using three explicitly toy candidates; no real Gold.

Uses an isolated app root because the existing development server serves the real
workspace. The temporary server is stopped on completion; Node is not required.
"""

import json
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from uuid import uuid4

import uvicorn
from playwright.sync_api import expect, sync_playwright

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.schema import Document
from nlplab.data.corpus import write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import AnnotationStore


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    out = Path.cwd() / "outputs/data_quality" / ("browser_" + stamp)
    root = out / "toy_workspace"
    texts = [
        "🙂张三来到中国石油大学（华东），随后前往青岛参加自然语言处理课程实验并讨论人工标注规范。",
        "第二条测试文本用于确认无实体的标注流程以及保存之后能够安全返回上一条并检验备注持久化。",
        "第三条测试文本带有AI和２０２６年日期仅用于浏览器筛选与候选队列进度的真实交互验收。",
    ]
    documents, candidates = [], []
    for index, text in enumerate(texts):
        label = "technology" if index == 2 else "education"
        kind = "blog" if index == 1 else "news"
        document = Document(
            f"TOY{index}",
            "题🙂：" + text + "后记",
            kind,
            label,
            "合成浏览器测试，非真实语料",
            f"https://example.invalid/toy/{index}",
            is_demo=True,
        )
        documents.append(document.to_dict())
        candidates.append(
            {
                "sample_id": f"SEQ{index + 1:05d}",
                "doc_id": document.id,
                "text": text,
                "source_start": 3,
                "source_end": 3 + len(text),
                "primary_label": label,
                "source_type": kind,
                "is_demo": True,
                "status": "pending",
                "gold_tokens": None,
                "entities": None,
                "revision_count": 0,
                "notes": "",
                "selection_reasons": ["explicit-toy-qa"],
            }
        )
    write_jsonl(root / "data/raw/documents.jsonl", documents)
    write_jsonl(root / "data/annotations/candidates.jsonl", candidates)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(Settings(root)), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        sleep(0.05)
    assert server.started
    errors, requests, checks = [], [], []
    endpoint = f"http://127.0.0.1:{port}/api/v1/annotation/sequence"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1440, "height": 1100})
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: requests.append(request.url))
            page.goto(f"http://127.0.0.1:{port}/annotation/sequence")
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00001")
            expect(page.locator("#ner-characters button")).to_have_count(len(texts[0]))
            expect(page.locator("#cws-characters .boundary")).to_have_count(len(texts[0]) - 1)
            assert not any("suggestions" in url for url in requests)
            assert not (root / "data/annotations/work/SEQ00001.json").exists()
            checks.append("Null initial Gold; code-point rendering; suggestions hidden")
            page.locator('[data-boundary="1"]').click()
            page.locator('[data-boundary="3"]').click()
            expect(page.locator("#save-state")).to_have_text("已原子保存")
            value = context.request.get(endpoint + "/SEQ00001").json()
            assert value["gold_tokens"][:2] == ["🙂", "张三"]
            page.locator('#ner-characters [data-index="1"]').click()
            page.locator('#ner-characters [data-index="2"]').click()
            assert page.locator("#entity-start").input_value() == "1"
            assert page.locator("#entity-end").input_value() == "3"
            page.locator("#save-entity").click()
            expect(page.locator("#save-state")).to_have_text("已原子保存")
            assert context.request.get(endpoint + "/SEQ00001").json()["entities"] == [
                {"text": "张三", "type": "PER", "start": 1, "end": 3}
            ]
            checks.append("CWS boundary → tokens/BMES; emoji spans align Python and JS")
            page.locator("#entity-start").fill("2")
            page.locator("#entity-end").fill("4")
            page.locator("#save-entity").click()
            expect(page.locator("#annotation-error")).to_contain_text("重叠")
            page.locator("#cancel-entity").click()
            page.locator(".entity-row").get_by_role("button", name="修改").click()
            page.locator("#entity-type").select_option("LOC")
            page.locator("#save-entity").click()
            expect(page.locator("#save-state")).to_have_text("已原子保存")
            assert (
                context.request.get(endpoint + "/SEQ00001").json()["entities"][0]["type"] == "LOC"
            )
            page.locator(".entity-row").get_by_role("button", name="删除").click()
            expect(page.locator("#entity-list .entity-row")).to_have_count(0)
            page.locator("#entity-start").fill("1")
            page.locator("#entity-end").fill("3")
            page.locator("#entity-type").select_option("PER")
            page.locator("#save-entity").click()
            expect(page.locator("#save-state")).to_have_text("已原子保存")
            checks.append("Entity modification/deletion and overlapping span rejection")
            draft = root / "data/annotations/work/SEQ00001.json"
            before = file_sha256(draft)
            page.locator("#reference-suggestions summary").click()
            expect(page.locator("#suggestion-content")).to_contain_text('"unavailable"')
            assert file_sha256(draft) == before
            checks.append("Unavailable HMM/HanLP suggestions do not modify Gold")
            page.locator("#annotation-notes").fill("浏览器人工测试备注🙂")
            page.locator("#next-sample").click()
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00002")
            page.locator("#previous-sample").click()
            expect(page.locator("#annotation-notes")).to_have_value("浏览器人工测试备注🙂")
            page.locator("#mark-done").click()
            expect(page.locator("#sample-heading")).to_contain_text("done")
            page.locator("#review-confirm").check()
            page.locator("#mark-reviewed").click()
            expect(page.locator("#sample-heading")).to_contain_text("reviewed")
            expect(page.locator("#review-count")).to_contain_text("复核次数 1")
            page.locator('[data-boundary="4"]').click()
            expect(page.locator("#sample-heading")).to_contain_text("in_progress")
            expect(page.locator("#mark-reviewed")).to_be_disabled()
            page.locator("#mark-done").click()
            expect(page.locator("#sample-heading")).to_contain_text("done")
            page.locator("#review-confirm").check()
            page.locator("#mark-reviewed").click()
            expect(page.locator("#review-count")).to_contain_text("复核次数 2")
            checks.append(
                "Navigation flush/autosave, done→explicit review, edits invalidate reviewed"
            )
            page.locator("#next-sample").click()
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00002")
            page.locator("#confirm-cws").click()
            page.locator("#no-entities").click()
            page.locator("#mark-done").click()
            expect(page.locator("#sample-heading")).to_contain_text("done")
            page.locator("#review-confirm").check()
            page.locator("#mark-reviewed").click()
            expect(page.locator("#sample-heading")).to_contain_text("reviewed")
            assert context.request.get(endpoint + "/SEQ00002").json()["entities"] == []
            page.locator("#filter-label").select_option("technology")
            expect(page.locator("#annotation-queue .queue-item")).to_have_count(1)
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00003")
            page.locator("#filter-label").select_option("")
            page.locator("#filter-source").select_option("blog")
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00002")
            expect(page.locator("#annotation-queue .queue-item")).to_have_count(1)
            page.locator("#filter-source").select_option("")
            page.locator("#filter-status").select_option("pending")
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00003")
            page.locator("#filter-status").select_option("")
            expect(page.locator("#sample-heading")).to_contain_text("SEQ00001")
            expect(page.locator('#annotation-queue [aria-current="true"]')).to_have_text(
                "SEQ00001 · reviewed"
            )
            expect(page.locator("#annotation-progress")).to_contain_text("reviewed 2")
            checks.append(
                "Queue label/source/status filters, progress, explicit empty NER confirmation"
            )
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 1100})
                page.screenshot(path=str(out / f"annotation-{width}.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            checks.append("Desktop/mobile rendering without horizontal overflow")
            assert not errors, errors
            assert not (root / "data/annotations/sequence_gold.jsonl").exists()
            _, validation = AnnotationStore(root).validation()
            assert not validation["valid"] and validation["reviewed_count"] == 2
            checks.append("Toy candidates cannot pass formal export; no final Gold generated")
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    result = {
        "passed": True,
        "browser": "real Chromium via Python Playwright",
        "toy_only": True,
        "checks": checks,
        "page_errors": errors,
        "server_stopped": not thread.is_alive(),
        "final_gold_generated": False,
    }
    (out / "statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "report.md").write_text(
        "# 人工标注浏览器验收（TOY）\n\n"
        + "\n".join(f"- {check}" for check in checks)
        + "\n\n无正式语料或 Gold，无模型性能评价。临时服务已停止。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"passed": True, "checks": len(checks), "report": str(out / "report.md")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
