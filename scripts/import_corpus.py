"""独立数据导入入口，不含模型训练算法。"""

import sys

from nlplab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["import-corpus", *sys.argv[1:]]))
