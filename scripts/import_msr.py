"""独立 MSR 数据准备入口，原始文件保持不变。"""

import sys

from nlplab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["import-msr", *sys.argv[1:]]))
