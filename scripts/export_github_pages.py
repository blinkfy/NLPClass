"""Export the existing workbench as a self-contained, read-only Pages site.

This reads frozen runs; it never trains models or executes an official evaluation.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def export_site(destination, *, include_corpus=False, data=True):
    from static_export_client import StaticExportClient

    from app.main import create_app

    destination = Path(destination).resolve()
    build_root = (ROOT / "build").resolve()
    if not destination.is_relative_to(build_root) or destination == build_root:
        raise ValueError(
            "Export destination must be a subdirectory of the project's build directory"
        )
    # A fresh export must not retain old full-text snapshots after a policy change.
    # Never clear an unrelated directory: require our own generated site marker.
    if destination.exists() and any(destination.iterdir()):
        index = destination / "index.html"
        if not (
            (destination / ".nojekyll").is_file()
            and index.is_file()
            and "window.NLPLAB_PAGE=" in index.read_text(encoding="utf-8")
        ):
            raise ValueError(
                "Output directory contains files not recognized as an NLPLab static export"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    app = create_app()
    client = StaticExportClient(app)
    shutil.copytree(ROOT / "app/static", destination / "static", dirs_exist_ok=True)
    # Existing modules distinguish pages by pathname. Directory routes on Pages have
    # a repository prefix and a trailing slash, so use the rendered canonical route.
    for script in (destination / "static/js").glob("*.js"):
        if script.name.startswith("static-"):
            continue
        text = script.read_text(encoding="utf-8")
        text = text.replace("location.pathname", "(window.NLPLAB_PAGE || location.pathname)")
        text = text.replace(
            "window.location.assign(paths[module]||'/tagging')",
            "window.location.assign(window.NLPLAB_STATIC.url(paths[module]||'/tagging'))",
        )
        if script.name == "classification-algorithm-explorer.js":
            text = text.replace(
                "单篇 trace 已核对冻结 logits / pooled / argmax / prediction。",
                "浏览器 trace 核对保存 logits / pooled / prediction 与已保存通道 argmax。",
            )
        script.write_text(text, encoding="utf-8", newline="")

    pages = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if (
            getattr(route, "include_in_schema", True)
            or not getattr(route, "methods", None)
            or "GET" not in route.methods
            or path in ("/sequence", "/analysis", "/docs", "/redoc", "/docs/oauth2-redirect")
        ):
            continue
        response = client.get(path)
        if "text/html" not in response.headers.get("content-type", ""):
            continue
        directory = destination / path.lstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        depth = len(Path(path.strip("/")).parts) if path != "/" else 0
        base = "../" * depth or "./"
        html = response.text
        html = re.sub(r"https?://testserver/static/", "static/", html)
        html = html.replace('"/static/', '"static/')
        # Inline module imports must be explicit URLs, rather than bare
        # specifiers such as "static/js/...", on every repository subpath.
        html = re.sub(
            r"import\(([\"'])(static/[^\"']+)\1\)",
            lambda match: f"import(new URL({json.dumps(match[2])}, document.baseURI).href)",
            html,
        )
        html = html.replace(
            "<head>",
            "<head>\n"
            + f'<base href="{base}">\n'
            + f"<script>window.NLPLAB_PAGE={json.dumps(path)};</script>\n"
            + '<script src="static/js/static-runtime.js"></script>',
        )

        # Relative directory URLs work on user sites, project sites and custom domains.
        def link(match):
            url = match[1]
            if url.startswith("/api/"):
                return match[0]
            parts = url.split("#", 1)
            target = parts[0].strip("/")
            fragment = "#" + parts[1] if len(parts) > 1 else ""
            return 'href="' + (target + "/" if target else "./") + fragment + '"'

        html = re.sub(r'href="(/[^"\s]*)"', link, html)
        banner = (
            '<aside class="static-edition-note" role="note" style="margin:0 0 16px;'
            'padding:12px 16px;border:1px solid #bfd9d9;border-radius:10px;background:#f2fafa">'
            "<strong>GitHub Pages 静态版</strong> · 冻结实验数据与图表可直接浏览；"
            "分词、C0–C3 分类及 BiLSTM-CRF 在浏览器运算。其余模型保留说明与实验记录。"
            '<a href="DEPLOYMENT.md">部署与功能说明</a></aside>'
        )
        html = html.replace('<main class="main">', '<main class="main">' + banner)
        (directory / "index.html").write_text(html, encoding="utf-8", newline="")
        pages.append(path)
    for old, new in {"sequence": "segmentation", "analysis": "integrated-analysis"}.items():
        directory = destination / old
        directory.mkdir(exist_ok=True)
        (directory / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><script>location.replace("../'
            + new
            + '/"+location.search+location.hash)</script><a href="../'
            + new
            + '/">打开页面</a>',
            encoding="utf-8",
        )
    (destination / ".nojekyll").touch()
    (destination / "404.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>页面不存在</title>'
        "<h1>页面不存在</h1><p>请使用导航中的静态页面地址（以 / 结尾）。</p></html>",
        encoding="utf-8",
    )
    manifest = {"format": 1, "pages": pages, "responses": {}, "issues": [], "models": {}}
    if data:
        from export_static_data import export_data

        manifest.update(export_data(app, destination, include_corpus=include_corpus))
        from export_browser_models import export_models

        manifest["models"] = export_models(app, destination)
        from export_browser_classifiers import export_classifiers

        manifest["classifiers"] = export_classifiers(app, destination)
        from export_browser_neural import export_neural

        manifest["neural"] = export_neural(app, destination)
        if manifest["neural"].get("textcnn"):
            manifest["issues"] = [
                item
                for item in manifest["issues"]
                if item["key"] != "/api/v1/experiments/3/visualization/c3-trace"
            ]
        from compact_static_snapshots import compact_snapshots

        manifest["compression"] = compact_snapshots(destination, manifest)
    write_json(destination / "static-manifest.json", manifest)
    shutil.copyfile(ROOT / "docs/github_pages.md", destination / "DEPLOYMENT.md")
    shutil.copyfile(ROOT / "docs/static_export_audit.md", destination / "AUDIT.md")
    shutil.copyfile(ROOT / "docs/static_inference.md", destination / "static_inference.md")
    inventory = package_site(destination)
    print(
        json.dumps(
            {
                "directory": str(destination),
                "pages": len(pages),
                "responses": len(manifest["responses"]),
                "issues": len(manifest["issues"]),
                "files": len(inventory),
            },
            ensure_ascii=False,
        )
    )
    return manifest


def package_site(destination):
    """Refresh checksums for the publishable directory; no archive is created."""
    destination = Path(destination).resolve()
    inventory = []
    for file in sorted(destination.rglob("*")):
        if file.is_file() and file.name != "export-integrity.json":
            inventory.append(
                {
                    "path": file.relative_to(destination).as_posix(),
                    "bytes": file.stat().st_size,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            )
    write_json(destination / "export-integrity.json", {"files": inventory})
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs",
        help="Publish to docs (default), or export a staging directory under build",
    )
    parser.add_argument(
        "--include-corpus",
        action="store_true",
        help="Include original course text only when redistribution is permitted",
    )
    args = parser.parse_args()
    if args.output == "docs":
        from publish_github_pages import publish_site

        staging = ROOT / "build/github-pages"
        export_site(staging, include_corpus=args.include_corpus)
        print(json.dumps(publish_site(staging, ROOT / "docs"), ensure_ascii=False))
    else:
        export_site(ROOT / args.output, include_corpus=args.include_corpus)
