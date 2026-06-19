"""
Mock community simulator and bias pipeline.

Part 1: Generate structured mock microbial communities with controllable
        overlap, sparsity, and size.
Part 2: Apply realistic measurement biases (multiplicative, detection
        threshold, contamination) via a composable pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Part 1 — Community Simulator
# ---------------------------------------------------------------------------

def make_communities(
    n_communities: int = 3,
    taxa_per_community: int | list[int] = 10,
    overlap: float = 0.0,
    core_overlap: float = 0.0,
    overlap_mode: str = 'chain',
    groups: list[list[int]] | None = None,
    group_core_overlap: float | int = 0.0,
    alpha: float = 0.5,
    core_signature_strength: float = 1.0,
    core_weight: float | None = None,
    n_samples: int | list[int] = 1,
    library_size: int = 10_000,
    shuffle_taxa: bool = False,
    seed: int | None = None,
) -> dict:
    """Generate mock microbial communities with controlled structure.

    Parameters
    ----------
    n_communities : int
        Number of distinct communities.
    taxa_per_community : int or list[int]
        Number of taxa per community. Scalar for uniform size, list for
        per-community sizes (e.g. [5, 20, 10]).
    overlap : float
        Fraction (0–1) of taxa shared between adjacent communities (chain).
        Used when overlap_mode is 'chain' or 'both'.
    core_overlap : float or int
        Taxa shared by ALL communities. If float (0–1), treated as a fraction
        of the smallest community. If int, used as an exact count. Used when
        overlap_mode is 'core' or 'both'.
    overlap_mode : str
        'chain' — adjacent communities share taxa (default, original behavior).
        'core'  — all communities share a common taxa block.
        'both'  — core shared by all, plus chain overlap among unique portions.
    groups : list[list[int]] or None
        Optional grouping of communities. Each inner list names the community
        indices that share a group-level core taxa block; communities in
        different groups have entirely independent taxa. Must partition
        0..n_communities-1. Singleton groups are allowed (no group core is
        created for them). Orthogonal to overlap_mode: the existing chain/core
        overlap structure still applies within each group's unique portion.
        Example: [[0, 1], [2, 3]] — communities 0 & 1 share one base set,
        communities 2 & 3 share a different base set.
    group_core_overlap : float or int
        Taxa shared within each non-singleton group. Same semantics as
        core_overlap but applied per group. Ignored for singleton groups.
    alpha : float
        Dirichlet concentration parameter.
        <1 → sparse/uneven, =1 → uniform, >1 → even.
    core_signature_strength : float
        Controls how distinctly each community dominates a bloc of core taxa.
        1.0 (default) → uniform concentration, no signature. Values >1 give
        each community an elevated concentration on roughly 1/n_communities of
        the core taxa (assigned by splitting core indices into equal chunks).
        E.g. with n_core=10, n_communities=3 and strength=10: community 0
        emphasises core taxa 0-2, community 1 emphasises 3-5, community 2
        emphasises 6-9.
    core_weight : float or None
        Fraction of each community's total abundance allocated to core taxa
        when core_signature_strength != 1.0. None (default) uses
        n_core / taxa_per_community, preserving the natural weighting.
    n_samples : int or list[int]
        Samples per community. Scalar for uniform, list for per-community.
    library_size : int
        Total counts per sample (multinomial draw depth).
    shuffle_taxa : bool
        Randomly permute taxon column order.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        counts        — DataFrame (samples × taxa), integer counts
        proportions   — DataFrame (samples × taxa), true proportions
        metadata      — DataFrame with columns sample_id, community
        taxa_table    — DataFrame with columns taxon, communities
        params        — dict echoing all generation parameters
    """
    if overlap_mode not in ('chain', 'core', 'both'):
        raise ValueError(f"overlap_mode must be 'chain', 'core', or 'both', got {overlap_mode!r}")

    rng = np.random.default_rng(seed)

    if isinstance(n_samples, int):
        samples_per = [n_samples] * n_communities
    else:
        if len(n_samples) != n_communities:
            raise ValueError(
                f"n_samples list length ({len(n_samples)}) != "
                f"n_communities ({n_communities})"
            )
        samples_per = list(n_samples)

    if isinstance(taxa_per_community, int):
        taxa_per = [taxa_per_community] * n_communities
    else:
        if len(taxa_per_community) != n_communities:
            raise ValueError(
                f"taxa_per_community list length ({len(taxa_per_community)}) != "
                f"n_communities ({n_communities})"
            )
        taxa_per = list(taxa_per_community)

    # --- taxa layout ---
    if overlap_mode not in ('core', 'both'):
        n_core = 0
    elif isinstance(core_overlap, int):
        n_core = core_overlap
    else:
        n_core = round(core_overlap * min(taxa_per))

    if groups is None:
        if overlap_mode == 'chain':
            overlap_counts = [
                round(overlap * min(taxa_per[i], taxa_per[i + 1]))
                for i in range(n_communities - 1)
            ]
            for i, ov in enumerate(overlap_counts):
                if taxa_per[i] - ov < 1:
                    raise ValueError(
                        f"overlap={overlap} between communities {i} and {i+1} leaves stride<1"
                    )
            starts = [0]
            for i in range(n_communities - 1):
                starts.append(starts[-1] + taxa_per[i] - overlap_counts[i])
            total_taxa = starts[-1] + taxa_per[-1]
            all_taxa = [f"taxon_{i + 1:03d}" for i in range(total_taxa)]
            taxon_membership: list[list[int]] = [[] for _ in range(total_taxa)]
            community_indices = []
            for c in range(n_communities):
                idx = list(range(starts[c], starts[c] + taxa_per[c]))
                community_indices.append(np.array(idx))
                for t in idx:
                    taxon_membership[t].append(c)
            boundary_lines = starts[1:]  # vertical lines between community blocks

        else:  # 'core' or 'both'
            for c, tp in enumerate(taxa_per):
                if n_core > tp:
                    raise ValueError(
                        f"core_overlap={core_overlap} gives n_core={n_core} which exceeds "
                        f"community {c} (taxa_per={tp})"
                    )
            unique_per = [tp - n_core for tp in taxa_per]

            if overlap_mode == 'core':
                overlap_counts = []
                unique_starts = [n_core]
                for c in range(n_communities - 1):
                    unique_starts.append(unique_starts[-1] + unique_per[c])
            else:  # 'both': chain overlap among unique portions
                overlap_counts = [
                    round(overlap * min(unique_per[i], unique_per[i + 1]))
                    for i in range(n_communities - 1)
                ]
                for i, ov in enumerate(overlap_counts):
                    if unique_per[i] - ov < 1:
                        raise ValueError(
                            f"overlap={overlap} on unique portion of communities {i} and {i+1} "
                            f"leaves stride<1; reduce overlap or increase taxa_per_community"
                        )
                unique_starts = [n_core]
                for i in range(n_communities - 1):
                    unique_starts.append(unique_starts[-1] + unique_per[i] - overlap_counts[i])

            total_taxa = unique_starts[-1] + unique_per[-1]
            all_taxa = [f"taxon_{i + 1:03d}" for i in range(total_taxa)]
            taxon_membership = [[] for _ in range(total_taxa)]

            # core taxa belong to all communities
            core_idx = list(range(n_core))
            for t in core_idx:
                taxon_membership[t] = list(range(n_communities))

            community_indices = []
            for c in range(n_communities):
                unique_idx = list(range(unique_starts[c], unique_starts[c] + unique_per[c]))
                community_indices.append(np.array(core_idx + unique_idx))
                for t in unique_idx:
                    if c not in taxon_membership[t]:
                        taxon_membership[t].append(c)

            # vertical lines: after core block, then between unique blocks
            boundary_lines = unique_starts  # unique_starts[0]=n_core, rest are unique block starts

        n_group_core_list: list[int] = []

    else:
        # ---- groups layout ----
        # validate: groups must partition 0..n_communities-1 exactly once
        all_in_groups = sorted(c for g in groups for c in g)
        if all_in_groups != list(range(n_communities)):
            raise ValueError(
                "groups must partition all community indices 0..n_communities-1 exactly once"
            )

        community_to_group = {c: g_idx for g_idx, g in enumerate(groups) for c in g}

        # group core sizes (0 for singletons — nothing to share)
        n_group_core_list = []
        for g in groups:
            if len(g) <= 1:
                n_group_core_list.append(0)
            elif isinstance(group_core_overlap, int):
                n_group_core_list.append(group_core_overlap)
            else:
                n_group_core_list.append(
                    round(group_core_overlap * min(taxa_per[c] for c in g))
                )

        # validate feasibility: every community must have at least 1 unique taxon
        for c in range(n_communities):
            g_idx = community_to_group[c]
            n_unique_c = taxa_per[c] - n_core - n_group_core_list[g_idx]
            if n_unique_c < 1:
                raise ValueError(
                    f"community {c}: taxa_per={taxa_per[c]} - n_core={n_core} "
                    f"- n_group_core={n_group_core_list[g_idx]} = {n_unique_c} < 1"
                )

        # build taxa block by block:
        # [global_core] [group_0_core] [group_0_unique...] [group_1_core] [group_1_unique...] ...
        taxa_offset = n_core
        boundary_lines_list: list[int] = []
        if n_core > 0:
            boundary_lines_list.append(n_core)

        group_core_idx_map: dict[int, list[int]] = {}
        community_unique_idx_map: dict[int, list[int]] = {}
        overlap_counts: list[int] = []

        for g_idx, g in enumerate(groups):
            gc = n_group_core_list[g_idx]
            group_core_idx_map[g_idx] = list(range(taxa_offset, taxa_offset + gc))
            taxa_offset += gc
            if gc > 0:
                boundary_lines_list.append(taxa_offset)  # after group core, before unique

            unique_sizes = [taxa_per[c] - n_core - gc for c in g]

            if overlap_mode in ('chain', 'both') and len(g) > 1:
                ov_g = [
                    round(overlap * min(unique_sizes[i], unique_sizes[i + 1]))
                    for i in range(len(g) - 1)
                ]
                for i, ov in enumerate(ov_g):
                    if unique_sizes[i] - ov < 1:
                        raise ValueError(
                            f"overlap={overlap} between communities {g[i]} and {g[i+1]} "
                            f"in group {g_idx} leaves stride<1"
                        )
                overlap_counts.extend(ov_g)
                u_starts = [taxa_offset]
                for i in range(len(g) - 1):
                    u_starts.append(u_starts[-1] + unique_sizes[i] - ov_g[i])
                taxa_offset = u_starts[-1] + unique_sizes[-1]
                boundary_lines_list.extend(u_starts[1:])
            else:
                u_starts = [taxa_offset + sum(unique_sizes[:i]) for i in range(len(g))]
                taxa_offset += sum(unique_sizes)
                if len(g) > 1:
                    boundary_lines_list.extend(u_starts[1:])

            for i, c in enumerate(g):
                community_unique_idx_map[c] = list(
                    range(u_starts[i], u_starts[i] + unique_sizes[i])
                )

            if g_idx < len(groups) - 1:
                boundary_lines_list.append(taxa_offset)  # between groups

        total_taxa = taxa_offset
        all_taxa = [f"taxon_{i + 1:03d}" for i in range(total_taxa)]
        taxon_membership: list[list[int]] = [[] for _ in range(total_taxa)]
        community_indices: list = [None] * n_communities

        for t in range(n_core):
            taxon_membership[t] = list(range(n_communities))

        for g_idx, g in enumerate(groups):
            gc_idx = group_core_idx_map[g_idx]
            for t in gc_idx:
                taxon_membership[t] = list(g)
            for c in g:
                u_idx = community_unique_idx_map[c]
                community_indices[c] = np.array(list(range(n_core)) + gc_idx + u_idx)
                for t in u_idx:
                    taxon_membership[t].append(c)

        boundary_lines = sorted(set(boundary_lines_list))

    # --- core signature matrix ---
    use_signatures = core_signature_strength != 1.0 and n_core > 0
    if use_signatures:
        sig = np.ones((n_communities, n_core))
        chunks = np.array_split(range(n_core), n_communities)
        for c, chunk in enumerate(chunks):
            sig[c, chunk] = core_signature_strength

    # --- generate mean proportions per community ---
    community_means = np.zeros((n_communities, total_taxa))
    for c in range(n_communities):
        idx = community_indices[c]
        if use_signatures:
            core_pos = idx[:n_core]
            other_pos = idx[n_core:]
            w = core_weight if core_weight is not None else n_core / len(idx)
            community_means[c, core_pos] = rng.dirichlet(sig[c]) * w
            if len(other_pos) > 0:
                community_means[c, other_pos] = rng.dirichlet(np.full(len(other_pos), alpha)) * (1 - w)
        else:
            community_means[c, idx] = rng.dirichlet(np.full(len(idx), alpha))

    # --- draw sample proportions & counts ---
    all_proportions = []
    all_counts = []
    meta_rows = []
    sample_idx = 0

    for c in range(n_communities):
        mean_p = community_means[c]
        for _ in range(samples_per[c]):
            concentration = alpha * taxa_per[c] * mean_p
            nonzero = mean_p > 0
            p = np.zeros(total_taxa)
            p[nonzero] = rng.dirichlet(concentration[nonzero])

            counts = rng.multinomial(library_size, p)

            sid = f"sample_{sample_idx:03d}"
            all_proportions.append(p)
            all_counts.append(counts)
            meta_rows.append({"sample_id": sid, "community": c})
            sample_idx += 1

    proportions_arr = np.array(all_proportions)
    counts_arr = np.array(all_counts)
    sample_ids = [r["sample_id"] for r in meta_rows]

    # --- optional shuffle ---
    col_order = np.arange(total_taxa)
    if shuffle_taxa:
        col_order = rng.permutation(total_taxa)
    ordered_taxa = [all_taxa[i] for i in col_order]

    counts_df = pd.DataFrame(
        counts_arr[:, col_order], index=sample_ids, columns=ordered_taxa
    )
    proportions_df = pd.DataFrame(
        proportions_arr[:, col_order], index=sample_ids, columns=ordered_taxa
    )
    metadata = pd.DataFrame(meta_rows).set_index("sample_id")

    taxa_table = pd.DataFrame({
        "taxon": [all_taxa[i] for i in col_order],
        "communities": [taxon_membership[i] for i in col_order],
    })

    return {
        "counts": counts_df,
        "proportions": proportions_df,
        "metadata": metadata,
        "taxa_table": taxa_table,
        "community_means": pd.DataFrame(
            community_means[:, col_order],
            index=[f"community_{c}" for c in range(n_communities)],
            columns=ordered_taxa,
        ),
        "params": {
            "n_communities": n_communities,
            "taxa_per_community": taxa_per,
            "overlap": overlap,
            "core_overlap": core_overlap,
            "overlap_mode": overlap_mode,
            "n_core": n_core,
            "groups": groups,
            "group_core_overlap": group_core_overlap,
            "n_group_core": n_group_core_list,
            "overlap_counts": overlap_counts,
            "boundary_lines": boundary_lines,
            "alpha": alpha,
            "core_signature_strength": core_signature_strength,
            "core_weight": core_weight,
            "n_samples": samples_per,
            "library_size": library_size,
            "shuffle_taxa": shuffle_taxa,
            "seed": seed,
            "total_taxa": total_taxa,
        },
    }


# ---------------------------------------------------------------------------
# Part 1 — Visualization
# ---------------------------------------------------------------------------

def plot_community_heatmap(
    result: dict,
    log_scale: bool = True,
    show_boundaries: bool = True,
    figsize: tuple | None = None,
    ax=None,
):
    """Heatmap of community proportions with community annotations.

    Parameters
    ----------
    result : dict
        Output of make_communities().
    log_scale : bool
        If True, plot log10(proportion + 1e-6).
    show_boundaries : bool
        Draw lines between community blocks (only meaningful if not shuffled).
    figsize : tuple or None
        Figure size; auto-scaled if None.
    ax : matplotlib Axes or None
    """
    props = result["proportions"]
    metadata = result["metadata"]
    params = result["params"]
    n_samples_total, n_taxa = props.shape

    if figsize is None:
        w = min(max(6, n_taxa * 0.25), 20)
        h = min(max(4, n_samples_total * 0.35), 20)
        figsize = (w, h)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = props.values.copy()
    if log_scale:
        data = np.log10(data + 1e-6)
        label = "log₁₀(proportion + 1e-6)"
    else:
        label = "proportion"

    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label)

    # y-axis: sample labels, colored by community
    comm_colors_list = plt.cm.Set2.colors
    communities = metadata["community"].values
    if n_samples_total <= 50:
        ax.set_yticks(range(n_samples_total))
        ax.set_yticklabels(props.index, fontsize=7)
        for i, c in enumerate(communities):
            ax.get_yticklabels()[i].set_color(
                comm_colors_list[c % len(comm_colors_list)]
            )
    else:
        cum = 0
        positions, labels = [], []
        for c in range(params["n_communities"]):
            n = params["n_samples"][c]
            positions.append(cum + (n - 1) / 2.0)
            labels.append(f"Community {c}  (n={n})")
            cum += n
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        for i, c in enumerate(range(params["n_communities"])):
            ax.get_yticklabels()[i].set_color(
                comm_colors_list[c % len(comm_colors_list)]
            )

    # x-axis: suppress labels when too many taxa to read
    if n_taxa <= 50:
        ax.set_xticks(range(n_taxa))
        ax.set_xticklabels(props.columns, fontsize=6, rotation=90)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Taxa (n={n_taxa})")

    # community block boundaries (suppress at large scale — too cluttered)
    if show_boundaries and not params["shuffle_taxa"] and n_taxa <= 200 and n_samples_total <= 100:
        for x in params["boundary_lines"]:
            ax.axvline(x - 0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
        cum = 0
        for c in range(params["n_communities"] - 1):
            cum += params["n_samples"][c]
            ax.axhline(cum - 0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)

    # legend
    handles = [
        Patch(facecolor=comm_colors_list[c % len(comm_colors_list)],
              label=f"Community {c}")
        for c in range(params["n_communities"])
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.15, 1.0),
              fontsize=7, frameon=False)

    if n_taxa <= 50:
        ax.set_xlabel("Taxa")
    ax.set_ylabel("Samples")
    ax.set_title("Mock Community Heatmap")
    return fig, ax


# ---------------------------------------------------------------------------
# Part 2 — Bias / Perturbation Functions
# ---------------------------------------------------------------------------

def apply_multiplicative_bias(
    proportions: pd.DataFrame,
    log_scale: float = 0.5,
    bias_factors: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply per-taxon log-normal multiplicative bias, then renormalize.

    Simulates systematic PCR/extraction efficiency differences across taxa.

    Parameters
    ----------
    proportions : DataFrame
        Rows = samples, columns = taxa, values on the simplex.
    log_scale : float
        Std dev of the log-normal distribution for bias factors.
        0 = no bias, larger = more distortion.
    bias_factors : ndarray or None
        Pre-computed per-taxon factors (length = n_taxa). If None, drawn
        from LogNormal(0, log_scale).
    seed : int or None

    Returns
    -------
    (biased_proportions, bias_factors)
        biased_proportions — DataFrame same shape, renormalized rows
        bias_factors — ndarray of per-taxon multiplicative factors used
    """
    rng = np.random.default_rng(seed)
    n_taxa = proportions.shape[1]

    if bias_factors is None:
        bias_factors = rng.lognormal(mean=0.0, sigma=log_scale, size=n_taxa)

    biased = proportions.values * bias_factors[np.newaxis, :]
    row_sums = biased.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # guard against all-zero rows
    biased = biased / row_sums

    return (
        pd.DataFrame(biased, index=proportions.index, columns=proportions.columns),
        bias_factors,
    )


def apply_detection_threshold(
    proportions: pd.DataFrame,
    min_abundance: float = 1e-4,
    stochastic: bool = False,
    steepness: float = 2.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Zero out taxa below a detection threshold, then renormalize.

    Parameters
    ----------
    proportions : DataFrame
    min_abundance : float
        Threshold below which taxa are dropped.
    stochastic : bool
        If True, detection probability is a sigmoid around the threshold
        rather than a hard cutoff.
    steepness : float
        Controls sigmoid steepness when stochastic=True. Higher = sharper.
    seed : int or None

    Returns
    -------
    DataFrame with sub-threshold taxa zeroed and rows renormalized.
    """
    rng = np.random.default_rng(seed)
    vals = proportions.values.copy()

    if stochastic:
        # sigmoid: P(detect) = 1 / (1 + exp(-steepness * (log(p) - log(threshold))))
        log_p = np.log(vals + 1e-30)
        log_thresh = np.log(min_abundance)
        detect_prob = 1.0 / (1.0 + np.exp(-steepness * (log_p - log_thresh)))
        mask = rng.random(vals.shape) < detect_prob
        vals = vals * mask
    else:
        vals[vals < min_abundance] = 0.0

    row_sums = vals.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    vals = vals / row_sums

    return pd.DataFrame(vals, index=proportions.index, columns=proportions.columns)


def apply_contamination(
    proportions: pd.DataFrame,
    contaminant: np.ndarray | pd.Series | None = None,
    fraction: float = 0.01,
    per_sample: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Mix in a contaminant community at a given fraction.

    Parameters
    ----------
    proportions : DataFrame
    contaminant : array-like or None
        Contaminant proportion profile (length = n_taxa). If None, a uniform
        profile over all taxa is used.
    fraction : float
        Mixing fraction. Result = (1-f)*original + f*contaminant.
    per_sample : bool
        If True, fraction varies per sample drawn from Beta distribution.
    seed : int or None

    Returns
    -------
    DataFrame with contamination mixed in (already on simplex).
    """
    rng = np.random.default_rng(seed)
    n_samples, n_taxa = proportions.shape

    if contaminant is None:
        contaminant = np.ones(n_taxa) / n_taxa
    else:
        contaminant = np.asarray(contaminant, dtype=float)
        if contaminant.shape[0] != n_taxa:
            raise ValueError(
                f"contaminant length ({contaminant.shape[0]}) != n_taxa ({n_taxa})"
            )
        s = contaminant.sum()
        if s > 0:
            contaminant = contaminant / s

    if per_sample:
        # Beta parameterized so mean ≈ fraction
        a = 2.0
        b = a * (1.0 / max(fraction, 1e-8) - 1.0)
        fractions = rng.beta(a, b, size=n_samples)[:, np.newaxis]
    else:
        fractions = fraction

    mixed = (1.0 - fractions) * proportions.values + fractions * contaminant[np.newaxis, :]

    return pd.DataFrame(mixed, index=proportions.index, columns=proportions.columns)


# ---------------------------------------------------------------------------
# Part 2 — Pipeline
# ---------------------------------------------------------------------------

_BIAS_FUNCTIONS = {
    "multiplicative": apply_multiplicative_bias,
    "threshold": apply_detection_threshold,
    "contamination": apply_contamination,
}


def apply_bias_pipeline(
    proportions: pd.DataFrame,
    steps: list[dict],
    seed: int | None = None,
) -> dict:
    """Apply a sequence of bias steps to community proportions.

    Parameters
    ----------
    proportions : DataFrame
        Clean proportions (rows on simplex).
    steps : list of dict
        Each dict must have a 'type' key ('multiplicative', 'threshold',
        'contamination') plus any keyword arguments for that function.
    seed : int or None
        Base seed; each step gets seed+i for reproducibility.

    Returns
    -------
    dict with:
        proportions — final biased proportions (DataFrame)
        details     — list of per-step metadata dicts
    """
    current = proportions.copy()
    details = []
    rng_base = seed if seed is not None else 0

    for i, step in enumerate(steps):
        step = dict(step)  # copy so we don't mutate caller's dicts
        step_type = step.pop("type")
        if step_type not in _BIAS_FUNCTIONS:
            raise ValueError(
                f"Unknown bias type '{step_type}'. "
                f"Available: {list(_BIAS_FUNCTIONS.keys())}"
            )

        fn = _BIAS_FUNCTIONS[step_type]
        step_seed = rng_base + i + 1 if seed is not None else None
        step["seed"] = step_seed

        if step_type == "multiplicative":
            current, factors = fn(current, **step)
            details.append({"type": step_type, "bias_factors": factors, **step})
        else:
            current = fn(current, **step)
            details.append({"type": step_type, **step})

    return {"proportions": current, "details": details}


# ---------------------------------------------------------------------------
# Part 2 — Bias Visualization
# ---------------------------------------------------------------------------

def plot_bias_effect(
    before: pd.DataFrame,
    after: pd.DataFrame,
    title: str | None = None,
    figsize: tuple = (12, 5),
    log_scale: bool = True,
):
    """Side-by-side heatmaps showing bias effect on community proportions.

    Parameters
    ----------
    before : DataFrame
        Clean proportions.
    after : DataFrame
        Biased proportions.
    title : str or None
        Figure-level suptitle.
    figsize : tuple
    log_scale : bool

    Returns
    -------
    (fig, axes)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                              gridspec_kw={"width_ratios": [1, 1, 0.8]})

    if title:
        fig.suptitle(title)

    for ax, data, panel_title in zip(
        axes[:2], [before, after], ["Before bias", "After bias"]
    ):
        vals = data.values.copy()
        if log_scale:
            vals = np.log10(vals + 1e-6)
        im = ax.imshow(vals, aspect="auto", cmap="viridis", interpolation="none")
        ax.set_title(panel_title)
        ax.set_xlabel("Taxa")
        ax.set_ylabel("Samples")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("log₁₀(proportion + 1e-6)" if log_scale else "proportion")

    # scatter: per-taxon mean before vs after
    ax_sc = axes[2]
    mean_before = np.maximum(before.mean(axis=0).values, 1e-6)
    mean_after = np.maximum(after.mean(axis=0).values, 1e-6)
    ax_sc.scatter(mean_before, mean_after, s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [1e-6, max(mean_before.max(), mean_after.max()) * 1.2]
    ax_sc.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.set_xlabel("Mean proportion (before)")
    ax_sc.set_ylabel("Mean proportion (after)")
    ax_sc.set_title("Per-taxon shift")

    rect = [0, 0, 1, 0.88] if title else [0, 0, 1, 1]
    fig.tight_layout(rect=rect)

    return fig, axes


# ---------------------------------------------------------------------------
# Part 3 — Convenience wrapper
# ---------------------------------------------------------------------------

def generate_mock_data(
    community_kwargs: dict | None = None,
    bias_steps: list[dict] | None = None,
    seed: int | None = None,
) -> dict:
    """End-to-end: generate communities, optionally apply bias, resample counts.

    Parameters
    ----------
    community_kwargs : dict
        Keyword arguments for make_communities(). Seed is overridden by
        the top-level seed if provided.
    bias_steps : list of dict or None
        Steps for apply_bias_pipeline(). If None, no bias applied.
    seed : int or None

    Returns
    -------
    dict with:
        clean       — make_communities() output (unbiased)
        counts      — DataFrame of final counts (biased if steps provided)
        proportions — DataFrame of final proportions
        metadata    — sample metadata
        bias        — apply_bias_pipeline() output (or None)
    """
    if community_kwargs is None:
        community_kwargs = {}

    comm_seed = seed
    community_kwargs = dict(community_kwargs)
    if seed is not None:
        community_kwargs["seed"] = seed

    clean = make_communities(**community_kwargs)

    if bias_steps:
        bias_seed = seed + 1000 if seed is not None else None
        bias_result = apply_bias_pipeline(
            clean["proportions"], bias_steps, seed=bias_seed
        )
        final_props = bias_result["proportions"]
    else:
        bias_result = None
        final_props = clean["proportions"]

    # resample counts from (potentially biased) proportions
    rng = np.random.default_rng(seed + 2000 if seed is not None else None)
    lib_size = clean["params"]["library_size"]
    final_counts = pd.DataFrame(
        np.array([rng.multinomial(lib_size, row) for row in final_props.values]),
        index=final_props.index,
        columns=final_props.columns,
    )

    return {
        "clean": clean,
        "counts": final_counts,
        "proportions": final_props,
        "metadata": clean["metadata"],
        "bias": bias_result,
    }
