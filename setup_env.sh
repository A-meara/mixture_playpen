#!/usr/bin/env bash
# Creates .venv in the repo directory using uv.
# Works on macOS and Linux.
# Usage: bash setup_env.sh
#
# Requires:
#   - uv  (https://github.com/astral-sh/uv)
#   - graphviz system binary for plate diagrams:
#       macOS:  brew install graphviz
#       Linux:  sudo apt install graphviz  (or equivalent)

set -euo pipefail
export UV_NO_CONFIG=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating .venv ..."
uv venv .venv --python 3.11

echo "Installing dependencies ..."
uv pip install --python .venv/bin/python -r requirements.txt

echo "Registering Jupyter kernel ..."
.venv/bin/python -m ipykernel install --user --name mixture_playpen --display-name "mixture_playpen"

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then run:             jupytext --to notebook mixture_playpen.py"
echo "                      jupyter notebook mixture_playpen.ipynb"
echo "Select kernel:        Kernel → Change kernel → mixture_playpen"
