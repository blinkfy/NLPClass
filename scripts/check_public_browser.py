"""真实 Chromium 检查公开数据 Run 接入；只对 toy 输入做额外交互推理。"""

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from urllib.request import urlopen
from uuid import uuid4

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    root = Path.cwd()
    run = sorted((root / "outputs/exp1/course_samples").glob("*/run.json"))[-1].parent.name
    frozen = "20260926_130612_32a5e168"
    out = (
        root
        / "outputs/data_quality"
        / (datetime.now(timezone.utc).strftime("public_browser_%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    )
    out.mkdir(parents=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    python = root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
    checks, errors = [], []
    with (out / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(python),
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = monotonic() + 45
            while True:
                try:
                    with urlopen(base + "/api/v1/health", timeout=2):
                        break
                except OSError:
                    if monotonic() > deadline:
                        raise RuntimeError("NER runtime Web 未启动")
                    sleep(0.25)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1050})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + "/sequence")
                expect(page.locator("#run-select")).to_have_value(run)
                page.locator("#tab-metrics").click()
                expect(page.locator("#ner-metrics tr")).to_have_count(5)
                expect(page.locator("#evaluation-info")).to_contain_text(
                    "MSRA NER test personal subset"
                )
                expect(page.locator("#metrics-values")).to_contain_text("MSRA 无 CWS Gold")
                assert page.locator("#metrics-values .metric").count() == 0
                checks.append("课程 Run 四类 exact-span 指标与来源；不显示 CWS Gold F1")
                page.screenshot(path=str(out / "course_ner_metrics.png"), full_page=True)
                for variant, f1 in (("smooth", "79.61%"), ("mle", "79.03%")):
                    page.locator("#run-select").select_option(frozen + "_" + variant)
                    expect(page.locator("#evaluation-info")).to_contain_text("SIGHAN")
                    expect(page.locator("#metrics-values")).to_contain_text(f1)
                checks.append("冻结 MSR MLE/Smooth 原评分显示与 4×5 confusion matrix")
                page.screenshot(path=str(out / "frozen_msr_metrics.png"), full_page=True)
                page.locator("#run-select").select_option(run)
                page.locator("#tab-errors").click()
                expect(page.locator("#error-list")).to_contain_text("NER exact-span")
                assert "Gold 分词" not in page.locator("#error-list").inner_text()
                checks.append("NER 错误 Gold/预测跨度对照，无伪造 CWS Gold")
                page.locator("#tab-interactive").click()
                text = "🙂张三来到中国石油大学（华东），随后去青岛。"
                page.locator("#input-text").fill(text)
                with page.expect_response("**/api/v1/analyze", timeout=90000) as response:
                    page.locator("#analyze-button").click()
                result = response.value.json()
                assert result["ner"]["status"] == "ok", result
                assert "".join(result["segmentation"]["tokens"]) == text
                for entity in result["ner"]["entities"]:
                    assert text[entity["start"] : entity["end"]] == entity["text"]
                expect(page.locator("#model-info")).to_contain_text("hanlp-msra-bert")
                checks.append(
                    "真实 HanLP + frozen Smooth 对 toy 文本推理，emoji code-point 与原文往返"
                )
                page.locator("#tab-parameters").click()
                expect(page.locator("#model-metadata")).to_contain_text("α=0.1")
                checks.append("课程 Run 参数来自冻结 Smooth")
                for route in ("clustering", "classification"):
                    page.goto(base + "/" + route)
                    expect(page.locator("#public-data-status")).to_contain_text("600")
                    expect(page.locator("#public-data-status")).to_contain_text("train 420")
                    expect(page.locator("#public-data-status")).to_contain_text("尚未训练")
                checks.append("实验二/三仅显示 THUCNews 数据与固定 split 状态")
                assert not errors, errors
                checks.append("无 JavaScript pageerror")
                browser.close()
        finally:
            process.terminate()
            process.wait(timeout=15)
    (out / "checks.json").write_text(
        json.dumps({"passed": checks, "page_errors": errors}, ensure_ascii=False, indent=2), "utf-8"
    )
    (out / "report.md").write_text(
        "# 公共数据 Web 浏览器验收\n\n"
        + "\n".join("- " + item for item in checks)
        + "\n\n临时 NER 环境服务已关闭；原 8000 服务未重启。\n",
        "utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
