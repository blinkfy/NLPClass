"""Compare exported browser WordPiece tokenization with HanLP's frozen fast tokenizer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HANLP_HOME"] = str(ROOT / "artifacts/cache/hanlp")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CASES = [
    "中文测试",
    "中华人民共和国",
    "abc",
    "ABC",
    "AI2026",
    "12345",
    "AI-Model_2.0",
    "商品🙂和服务",
    "🙂🚀👩‍💻",
    "Café déjà vu",
    "Cafe\u0301",
    "İstanbul",
    "Ångström",
    "\t制表符\n换行\r回车",
    "前\x00中\x01后",
    "零宽\u200b空格",
    "不换行\u00a0空格",
    "全角ＡＢＣ１２３",
    "（中文）【标点】，。！？",
    "English, punctuation!",
    "foo/bar\\baz",
    "[UNK]",
    "[SEP]",
    "[PAD]",
    "[CLS]",
    "[MASK]",
    "[unused1]",
    "[CLS]中文[SEP]",
    "中🙂文",
    "𠮷野家",
    "𠀀𪛖",
    "한글 테스트",
    "العَرَبِيَّة",
    "中文	English\n123",
    "a" * 99,
    "a" * 100,
    "a" * 101,
    "中" * 100,
    "中" * 101,
    "©®™℃№",
    "①②③",
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
    "\u2028行分隔\u2029段分隔",
    "\ufeffBOM\u2060word joiner",
    "\\u0000 literal slash text",
    "重复重复重复重复",
    "a\u0308\u0323 test",
    "<script>alert(1)</script>",
]

NODE_RUNNER = r"""
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const [root, modulePath] = process.argv.slice(1);
const {wordpieces} = await import(pathToFileURL(modulePath).href);
let input=''; for await (const part of process.stdin) input += part;
const {cases,vocabulary}=JSON.parse(input);
process.stdout.write(JSON.stringify(cases.map(text=>wordpieces(text,vocabulary))));
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import hanlp

    model_path = ROOT / "artifacts/cache/hanlp/pos/pos_ctb_electra_small_20220215_111944"
    component = hanlp.load(str(model_path), devices=-1)
    tokenizer = component.transformer_tokenizer
    descriptor = json.loads(
        (ROOT / "build/github-pages/models/transformers/pos.json").read_text(encoding="utf-8")
    )
    vocabulary = descriptor["vocabulary"]

    expected = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in CASES]
    payload = json.dumps({"cases": CASES, "vocabulary": vocabulary}, ensure_ascii=False)
    process = subprocess.run(
        [
            "node",
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            NODE_RUNNER,
            str(ROOT),
            str(ROOT / "app/static/js/static-wordpiece.js"),
        ],
        cwd=ROOT,
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if process.returncode:
        raise RuntimeError(f"Node WordPiece call failed:\n{process.stderr}")
    actual = json.loads(process.stdout)

    differences = []
    for text, reference, browser in zip(CASES, expected, actual):
        if reference != browser:
            differences.append(
                {
                    "text": text,
                    "text_repr": repr(text),
                    "hanlp_ids": reference,
                    "browser_ids": browser,
                }
            )
    print(
        json.dumps(
            {
                "status": "passed" if not differences else "mismatch",
                "cases": len(CASES),
                "mismatch_count": len(differences),
                "mismatches": differences,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
