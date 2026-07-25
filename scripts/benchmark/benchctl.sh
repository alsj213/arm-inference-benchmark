#!/bin/bash
# benchctl shell wrapper — 确保 Python 环境和 ADB 就绪
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
exec python3 -m benchctl "$@"
