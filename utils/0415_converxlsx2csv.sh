#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法: $0 [目录]" >&2
  echo "  递归转换目录及子目录中所有 .xlsx 为同名 .csv，输出与源文件同一路径。" >&2
  echo "  未指定目录时默认使用脚本所在项目下的 data/rawdata。" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
  RAW_DIR="${SCRIPT_DIR}/../data/rawdata"
else
  RAW_DIR="$1"
fi

if [[ ! -d "$RAW_DIR" ]]; then
  echo "目录不存在: $RAW_DIR" >&2
  exit 1
fi
RAW_DIR="$(cd "$RAW_DIR" && pwd)"

# 跳过 Excel 临时锁文件（~$ 开头）
while IFS= read -r -d '' xlsx; do
  out="${xlsx%.*}.csv"
  xlsx2csv "$xlsx" "$out"
done < <(find "$RAW_DIR" -type f -iname '*.xlsx' ! -name '~$*' -print0)
