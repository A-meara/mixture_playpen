# Mixture Playpen

Interactive notebook for exploring Bayesian microbial source tracking with mock communities.

Set mixture weights and community structure in the CONFIG cell, run all, and inspect how well the model recovers the true weights.

## What it does

- Generates mock source communities with controllable overlap and sparsity
- Simulates a mixed sink sample at specified true weights
- Fits a Bayesian model (ZeroSumNormal source profiles + Dirichlet mixing weights)
- Visualises posterior recovery, HDI intervals, and plate diagrams

## Setup

Requires [uv](https://github.com/astral-sh/uv) and the `graphviz` system binary:

```bash
# macOS
brew install graphviz

# Linux
sudo apt install graphviz
```

Then:

```bash
bash setup_env.sh
source .venv/bin/activate
jupytext --to notebook mixture_playpen.py
jupyter notebook mixture_playpen.ipynb
```
