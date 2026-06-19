# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mixture Source-Tracking — Playpen
#
# Shows what mock source communities look like, and what the sink looks like
# after mixing them at a chosen weight. Tweak the CONFIG cell and re-run all.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytensor
pytensor.config.cxx = ""
import pymc as pm
import arviz as az
import graphviz
# %matplotlib inline

from mock_communities import make_communities, plot_community_heatmap

try:
    import nutpie
    HAS_NUTPIE = True
except ImportError:
    HAS_NUTPIE = False

COMM_COLORS = list(plt.cm.Set2.colors)

# %% [markdown]
# ## Config

# %%
# True mixture weights per source — must sum to 1 (renormalised automatically).
# Length determines number of sources.
WEIGHTS      = [0.71, 0.21, 0.08]
SEED         = 42

# MCMC settings
TUNE   = 1000
DRAWS  = 1000
CHAINS = 4

# Community structure
TAXA_PER_COMMUNITY = 30
N_SOURCE_SAMPLES   = 3    # pooled source replicates (what the model sees)
LIBRARY_SIZE       = 10_000
ALPHA              = 0.3  # Dirichlet concentration: <1 sparse/uneven, >1 even

# Overlap between source communities
# overlap_mode: 'chain' (adjacent share taxa), 'core' (all share), 'both'
OVERLAP_MODE = 'core'
OVERLAP      = 0.0
CORE_OVERLAP = 3   # taxa shared by ALL communities (set 0 for fully disjoint)

# derived
true_weights = np.array(WEIGHTS, dtype=float)
true_weights /= true_weights.sum()
n_sources = len(true_weights)

print(f'n_sources = {n_sources}')
for i, w in enumerate(true_weights):
    print(f'  w_source_{i} = {w:.3f}')
print(f'TAXA_PER_COMMUNITY={TAXA_PER_COMMUNITY}  ALPHA={ALPHA}  SEED={SEED}')

# %% [markdown]
# ## Core functions

# %%
def simulate_mixture(true_weights, n_sources, taxa_per_comm, alpha, overlap,
                     n_source_samples=3, library_size=10_000, seed=42):
    rng = np.random.default_rng(seed)
    true_weights = np.array(true_weights)

    comm = make_communities(
        n_communities=n_sources,
        taxa_per_community=taxa_per_comm,
        core_overlap=overlap,
        overlap_mode='core',
        alpha=alpha,
        n_samples=1,
        library_size=library_size,
        seed=seed,
    )
    means    = comm["community_means"].values  # (n_sources, n_taxa)
    all_taxa = list(comm["counts"].columns)

    p_sink = (true_weights[:, None] * means).sum(axis=0)
    p_sink /= p_sink.sum()
    sink_counts = rng.multinomial(library_size, p_sink)

    source_pooled = np.zeros((n_sources, len(all_taxa)), dtype=int)
    for i in range(n_sources):
        p_src = means[i] / means[i].sum()
        for _ in range(n_source_samples):
            source_pooled[i] += rng.multinomial(library_size, p_src)

    src_labels = [f"source_{i}" for i in range(n_sources)]
    gt = {f"w_source_{i}": float(true_weights[i]) for i in range(n_sources)}
    gt["w_unknown"] = 0.0

    return {
        "sink_counts":   pd.Series(sink_counts, index=all_taxa, name="sink"),
        "source_pooled": pd.DataFrame(source_pooled, index=src_labels, columns=all_taxa),
        "feature_names": all_taxa,
        "ground_truth":  gt,
    }


def build_bayesian_model(source_pooled, sink_counts, feature_names, n_sources):
    src_labels     = [f"src_{i}" for i in range(n_sources)]
    mixture_labels = src_labels + ["unknown"]
    with pm.Model(coords={
        "features": feature_names,
        "sources":  src_labels,
        "mixture":  mixture_labels,
    }) as model:
        source_logits = pm.ZeroSumNormal("source_logits", sigma=3, dims=("sources", "features"))
        p_sources     = pm.Deterministic(
            "p_sources", pm.math.softmax(source_logits, axis=1), dims=("sources", "features")
        )
        pm.Multinomial(
            "fit_sources",
            n=source_pooled.sum(axis=1).values,
            p=p_sources,
            observed=source_pooled.values,
            dims=("sources", "features"),
        )
        unknown_logits = pm.ZeroSumNormal("unknown_logits", sigma=3, dims=("features",))
        p_unknown      = pm.math.softmax(unknown_logits)
        W = pm.Dirichlet("W", a=np.ones(n_sources + 1), dims=("mixture",))
        p_sink = pm.Deterministic(
            "p_sink",
            (W[:n_sources, None] * p_sources).sum(axis=0) + W[n_sources] * p_unknown,
            dims=("features",),
        )
        pm.Multinomial(
            "fit_sink",
            n=int(sink_counts.sum()),
            p=p_sink,
            observed=sink_counts.values,
            dims=("features",),
        )
    return model


def fit_bayesian(model, tune=1000, draws=1000, chains=4):
    if HAS_NUTPIE:
        compiled = nutpie.compile_pymc_model(model)
        return nutpie.sample(
            compiled, tune=tune, draws=draws, chains=chains,
            target_accept=0.93, maxdepth=10, blocking=True, save_warmup=False,
        )
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=0.93, progressbar=True,
        )


def check_recovery_bayes(idata, ground_truth, n_sources, hdi_prob=0.94):
    W_flat  = idata.posterior["W"].values.reshape(-1, n_sources + 1)
    results = {}
    for i in range(n_sources):
        key     = f"w_source_{i}"
        samples = W_flat[:, i]
        tv      = ground_truth.get(key, 0.0)
        hdi_v   = az.hdi(samples, hdi_prob=hdi_prob)
        lo, hi  = float(hdi_v[0]), float(hdi_v[1])
        results[key] = dict(
            true=tv, mean=float(samples.mean()),
            hdi_low=lo, hdi_high=hi,
            in_hdi=(lo <= tv <= hi),
            error=abs(float(samples.mean()) - tv),
            width=hi - lo,
        )
    unk    = W_flat[:, n_sources]
    unk_tv = ground_truth.get("w_unknown", 0.0)
    hdi_v  = az.hdi(unk, hdi_prob=hdi_prob)
    lo, hi = float(hdi_v[0]), float(hdi_v[1])
    results["w_unknown"] = dict(
        true=unk_tv, mean=float(unk.mean()),
        hdi_low=lo, hdi_high=hi,
        in_hdi=(lo <= unk_tv <= hi),
        error=abs(float(unk.mean()) - unk_tv),
        width=hi - lo,
    )
    return results

# %% [markdown]
# ## Source community structure preview
#
# Each row is a sample from one source community. Colours = community membership.

# %%
_preview = make_communities(
    n_communities=n_sources,
    taxa_per_community=TAXA_PER_COMMUNITY,
    overlap=OVERLAP, overlap_mode=OVERLAP_MODE, core_overlap=CORE_OVERLAP,
    alpha=ALPHA, n_samples=N_SOURCE_SAMPLES,
    library_size=LIBRARY_SIZE, seed=SEED,
)

fig, ax = plt.subplots(figsize=(12, 3))
plot_community_heatmap(_preview, log_scale=True, show_boundaries=True, ax=ax)
ax.set_title(
    f'{n_sources} source communities, {TAXA_PER_COMMUNITY} taxa each  '
    f'(overlap_mode={OVERLAP_MODE!r}, core_overlap={CORE_OVERLAP}, seed={SEED})',
    fontsize=11,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Simulate mixture
#
# `simulate_mixture` draws pooled source counts and a sink sample whose true
# composition is the weighted average of source means.

# %%
sim = simulate_mixture(
    true_weights=true_weights,
    n_sources=n_sources,
    taxa_per_comm=TAXA_PER_COMMUNITY,
    alpha=ALPHA,
    overlap=CORE_OVERLAP,         # simulate_mixture uses 'overlap' for core_overlap
    n_source_samples=N_SOURCE_SAMPLES,
    library_size=LIBRARY_SIZE,
    seed=SEED,
)
source_pooled = sim['source_pooled']   # (n_sources, n_taxa)
sink_counts   = sim['sink_counts']     # (n_taxa,)
feature_names = sim['feature_names']
gt            = sim['ground_truth']

print('Source pooled shape:', source_pooled.shape)
print('Sink counts shape:  ', sink_counts.shape)
print('Ground truth:', {k: f'{v:.3f}' for k, v in gt.items()})

# %% [markdown]
# ## Side-by-side: sources vs sink
#
# Each column is a taxon. Top rows = source profiles; bottom row = mixed sink.
# The weight label on the left shows how much each source contributes.

# %%
src_frac  = source_pooled.div(source_pooled.sum(axis=1), axis=0)   # (n_sources, n_taxa)
sink_frac = (sink_counts / sink_counts.sum()).values.reshape(1, -1) # (1, n_taxa)

stacked = np.vstack([src_frac.values, sink_frac])
log_stacked = np.log10(stacked + 1e-6)

cmap = 'viridis'

fig, ax = plt.subplots(figsize=(13, 3 + n_sources * 0.6))
im = ax.imshow(log_stacked, aspect='auto', cmap=cmap, interpolation='nearest')

ytick_pos    = list(range(n_sources)) + [n_sources]
ytick_labels = [f'Source {i}  (w={gt[f"w_source_{i}"]:.2f})' for i in range(n_sources)] \
             + ['Sink (mixed)']
ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_labels, fontsize=10)
for i, label in enumerate(ax.get_yticklabels()):
    color = COMM_COLORS[i % len(COMM_COLORS)] if i < n_sources else 'black'
    label.set_color(color)

n_taxa = len(feature_names)
if n_taxa <= 40:
    ax.set_xticks(range(n_taxa))
    ax.set_xticklabels(feature_names, fontsize=7, rotation=90)
else:
    ax.set_xticks([])
    ax.set_xlabel(f'Taxa  (n={n_taxa})', fontsize=10)

ax.set_title(
    f'Source profiles vs mixed sink  '
    f'(weights: {", ".join(f"w{i}={v:.2f}" for i, v in enumerate(true_weights))})',
    fontsize=12,
)
plt.colorbar(im, ax=ax, label='log₁₀(relative abundance + 1e-6)', pad=0.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Per-taxon decomposition
#
# For each taxon: bar showing how much each source contributes to the sink
# (weight × source abundance), stacked to give the predicted sink abundance.
# The dot shows the actual observed sink relative abundance.

# %%
src_means = src_frac.values   # (n_sources, n_taxa)
sink_obs  = sink_frac.flatten()

contributions = true_weights[:, None] * src_means   # (n_sources, n_taxa)
order = np.argsort(contributions.sum(axis=0))[::-1]

top_n = min(40, n_taxa)
order_top     = order[:top_n]
contrib_top   = contributions[:, order_top]
sink_obs_top  = sink_obs[order_top]
taxa_top      = [feature_names[i] for i in order_top]

fig, ax = plt.subplots(figsize=(max(10, top_n * 0.4), 4))
x = np.arange(top_n)
bottom = np.zeros(top_n)
for i in range(n_sources):
    ax.bar(x, contrib_top[i], bottom=bottom,
           color=COMM_COLORS[i % len(COMM_COLORS)], label=f'Source {i}  (w={true_weights[i]:.2f})',
           width=0.7, edgecolor='none')
    bottom += contrib_top[i]

ax.scatter(x, sink_obs_top, color='black', s=25, zorder=5,
           label='Observed sink', marker='o')

ax.set_xticks(x)
ax.set_xticklabels(taxa_top, rotation=90, fontsize=7)
ax.set_ylabel('Relative abundance')
ax.set_xlabel(f'Top {top_n} taxa (by predicted abundance)')
ax.set_title('Stacked source contributions vs observed sink  (dots = observed)')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Mixing weights summary

# %%
fig, ax = plt.subplots(figsize=(4, 3))
labels = [f'Source {i}' for i in range(n_sources)] + ['Unknown']
sizes  = [gt[f'w_source_{i}'] for i in range(n_sources)] + [gt['w_unknown']]
colors_pie = [COMM_COLORS[i % len(COMM_COLORS)] for i in range(n_sources)] + ['lightgray']
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors_pie,
    autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
    startangle=90, pctdistance=0.7,
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title('True mixing weights', fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bayesian inference
#
# Fit the model: each source profile is learned via ZeroSumNormal logits, the
# unknown source gets its own logit vector, and W is a Dirichlet-distributed
# mixing vector over sources + unknown.

# %%
model = build_bayesian_model(source_pooled, sink_counts, feature_names, n_sources)

# %% [markdown]
# ### Plate diagram — actual model

# %%
pm.model_to_graphviz(model)

# %% [markdown]
# ### Plate diagram — symbolic (for paper)
#
# Same model structure, but coord names are single letters so the plates are
# labelled **F** (features), **K** (sources), **K+1** (sources + unknown).

# %%
def make_plate_diagram_symbolic(save_path="mixture_model_plate_symbolic"):
    """
    Hand-built plate diagram with symbolic labels (F, K, K+1).
    K+1/W plate is forced to the left via rank=same + invisible ordering edge.
    All plate labels bottom-right (labeljust=r labelloc=b).
    """
    dot = """\
digraph {
\tcompound=true
\tsubgraph "clusterK+1" {
\t\tlabel="K+1" labeljust=r labelloc=b style=rounded
\t\tW [label="W\\n~\\nDirichlet" shape=ellipse]
\t}
\tsubgraph "clusterKxF" {
\t\tlabel="K \\u00D7 F" labeljust=r labelloc=b style=rounded
\t\tsource_logits [label="source_logits\\n~\\nZeroSumNormal" shape=ellipse]
\t\tp_sources [label="p_sources\\n~\\nDeterministic" shape=box]
\t\tfit_sources [label="fit_sources\\n~\\nMultinomial" shape=ellipse style=filled]
\t}
\tsubgraph "clusterF" {
\t\tlabel="F" labeljust=r labelloc=b style=rounded
\t\tunknown_logits [label="unknown_logits\\n~\\nZeroSumNormal" shape=ellipse]
\t\tp_sink [label="p_sink\\n~\\nDeterministic" shape=box]
\t\tfit_sink [label="fit_sink\\n~\\nMultinomial" shape=ellipse style=filled]
\t}
\tW -> p_sink
\tsource_logits -> p_sources
\tp_sources -> fit_sources
\tp_sources -> p_sink
\tunknown_logits -> p_sink
\tp_sink -> fit_sink
\t{ rank=same; W; source_logits }
\tW -> source_logits [style=invis]
}
"""
    gv = graphviz.Source(dot)
    gv.render(save_path, format="pdf", cleanup=True)
    return gv

gv_sym = make_plate_diagram_symbolic()
gv_sym

# %%
idata = fit_bayesian(model, tune=TUNE, draws=DRAWS, chains=CHAINS)

# %% [markdown]
# ### Posterior diagnostics

# %%
az.plot_energy(idata)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Recovered mixing weights vs truth
#
# Violin = posterior over W; dot = posterior mean; horizontal line = true weight.

# %%
recovery = check_recovery_bayes(idata, gt, n_sources)

W_samples = idata.posterior["W"].values.reshape(-1, n_sources + 1)  # (samples, n_sources+1)
weight_labels = [f'Source {i}' for i in range(n_sources)] + ['Unknown']
colors_all    = [COMM_COLORS[i % len(COMM_COLORS)] for i in range(n_sources)] + ['lightgray']
true_vals     = [gt.get(f'w_source_{i}', 0.0) for i in range(n_sources)] + [gt.get('w_unknown', 0.0)]

fig, ax = plt.subplots(figsize=(max(5, (n_sources + 1) * 1.4), 4))
parts = ax.violinplot(W_samples, positions=range(n_sources + 1),
                      showmeans=False, showmedians=False, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_all[i])
    pc.set_alpha(0.6)

for i in range(n_sources + 1):
    key = f'w_source_{i}' if i < n_sources else 'w_unknown'
    r   = recovery[key]
    ax.scatter(i, r['mean'], color=colors_all[i], s=60, zorder=5, edgecolors='black', linewidths=0.8)
    ax.hlines(true_vals[i], i - 0.35, i + 0.35,
              colors='black', linewidths=1.5, linestyles='--', zorder=6)
    ax.text(i, true_vals[i] + 0.02, f'true={true_vals[i]:.2f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(range(n_sources + 1))
ax.set_xticklabels(weight_labels, fontsize=10)
ax.set_ylabel('Weight')
ax.set_ylim(-0.05, 1.05)
ax.set_title('Posterior mixing weights  (violin=posterior, dot=mean, dashed=truth)', fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### HDI summary table

# %%
rows = []
for i in range(n_sources):
    key = f'w_source_{i}'
    r   = recovery[key]
    rows.append(dict(component=f'Source {i}', true=r['true'], mean=r['mean'],
                     hdi_low=r['hdi_low'], hdi_high=r['hdi_high'],
                     in_hdi=r['in_hdi'], error=r['error'], width=r['width']))
r_unk = recovery['w_unknown']
rows.append(dict(component='Unknown', true=r_unk['true'], mean=r_unk['mean'],
                 hdi_low=r_unk['hdi_low'], hdi_high=r_unk['hdi_high'],
                 in_hdi=r_unk['in_hdi'], error=r_unk['error'], width=r_unk['width']))

summary_df = pd.DataFrame(rows).set_index('component')
summary_df = summary_df.round(4)
print(summary_df.to_string())
summary_df
