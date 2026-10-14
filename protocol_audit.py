#!/usr/bin/env python3
"""
protocol_audit — a PoseBusters-style *protocol-audit* toolkit for binding-prediction
leaderboards.

WHY
---
A leaderboard ("which in-silico score / generator / model is best?") is not a property
of the data.  It is a property of the **evaluation protocol**: which label definition
counts as positive, whether items are pooled or averaged per target, which cutoff the
enrichment is measured at, and which metric family is reported.  Change any of those and
the ranking can change — even when the labels contain no noise at all.

The literature is full of checklists for the *inputs* of a prediction (PoseBusters:
does the predicted pose satisfy basic chemistry?).  There is no equivalent checklist for
the *evaluation*.  This module is that checklist, made executable.

WHAT IT DOES
------------
Given a table of items (designs / poses / compounds), one or more candidate score
columns, one or more candidate label definitions, and a grouping column (target,
provenance cluster, ...), `protocol_audit` enumerates a protocol grid

    aggregation  x  label definition  x  metric

and reports, per protocol: the full leaderboard, the winner, how many methods are
statistically inseparable from that winner (two-level paired bootstrap), and how much
the leaderboard as a whole moves between protocols.  The output is a single **audit
card** plus machine-readable tables.

It is deliberately opinionated about three things that are almost always skipped:

1. **Comparability.**  A protocol where only 3 of 40 methods have a value cannot be
   ranked.  Coverage is computed and reported; incomparable protocols are excluded
   rather than silently averaged.
2. **Direction.**  A score whose sign is unknown is not a score.  Each column is
   oriented against the primary label (AUC >= 0.5) and the raw un-oriented AUC is
   recorded in `orientation.csv` so the reader can see which columns had to be flipped.
3. **Separability.**  "Method X is #1" is meaningless without a paired test against
   the champion.  Marginal confidence intervals are wrong here because the methods are
   evaluated on the same items; a paired two-level bootstrap is used instead.

USAGE
-----
    from protocol_audit import ProtocolAudit

    audit = ProtocolAudit(df,
                          score_cols=[...], label_cols=[...], group_col="Target",
                          n_boot=200, seed=0)
    card = audit.run()          # dict of DataFrames + a 'meta' dict
    audit.write("out_dir")      # writes CSVs, JSON, Markdown audit card

    # already-computed leaderboard matrix?
    from protocol_audit import audit_from_matrix
    card = audit_from_matrix(matrix_df, boot=boot_tensor)

CLI
---
    python3 protocol_audit.py --table designs.parquet \
        --scores plddt iptm pae_interaction --labels bind medium strong \
        --group Target --out reports/ --n-boot 200

Author: Yaowei Lin.  MIT licence, no warranty.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

__version__ = "0.1.0"

# ----------------------------------------------------------------------------- config
METRIC_NAMES = ("auc", "ap", "ef1", "ef5", "ef10", "ef20", "ndcg10")
EF_LEVELS = ((0.01, "ef1"), (0.05, "ef5"), (0.10, "ef10"), (0.20, "ef20"))
DEFAULT_AGGS = ("pooled", "macro", "median")
MIN_GROUP_N = 10        # a group needs this many labelled items to enter a macro mean
MIN_GROUP_POS = 2       # ... and this many positives
MIN_LABEL_COVERAGE = 0.5   # a label column below this non-null rate is dropped
MIN_METHOD_COVERAGE = 0.5  # ... same for a score column
MIN_PROTOCOL_COVERAGE = 0.8  # a protocol must cover this fraction of methods to be ranked
TIE_ALPHA = 0.05


# ------------------------------------------------------------------------- core maths
def _metrics(y, s, k: int = 10):
    """All headline metrics for one (label, score) pair on one collection of items.

    y : bool array (positive = binder / success)
    s : float array (higher = predicted better); NaN scores are dropped
    Returns {'auc','ap','ef1','ef5','ef10','ef20','ndcg10'}; any undefined metric -> NaN.
    """
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    out = {m: np.nan for m in METRIC_NAMES}
    n = len(y)
    if n == 0:
        return out
    n1 = int(y.sum())
    if n1 == 0 or n1 == n:
        return out                      # AUC / AP / EF undefined with a single class
    n0 = n - n1
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    r = rankdata(s, method="average")
    out["auc"] = (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    c = np.cumsum(ys)
    idx = np.arange(1, n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(ys, c / idx, 0.0)
    out["ap"] = prec.sum() / n1
    base = n1 / n
    for f, nm in EF_LEVELS:
        kk = max(1, int(round(f * n)))
        out[nm] = ys[:kk].mean() / base
    kk = min(k, n)
    rel = ys[:kk]
    d = 1.0 / np.log2(np.arange(2, kk + 2))
    ideal = np.sort(y)[::-1][:kk]
    idcg = float((ideal * d).sum())
    out["ndcg10"] = float((rel * d).sum() / idcg) if idcg > 0 else np.nan
    return out


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else np.nan


def _nanmedian(a):
    a = np.asarray(a, dtype=float)
    return float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else np.nan


# --------------------------------------------------------------------------- the class
class ProtocolAudit:
    """Enumerate a protocol grid over (aggregation x label x metric) and audit it."""

    def __init__(self, df: pd.DataFrame, score_cols, label_cols, group_col,
                 aggs=DEFAULT_AGGS, min_group_n=MIN_GROUP_N,
                 min_group_pos=MIN_GROUP_POS, orient="primary",
                 n_boot=200, seed=0, min_label_coverage=MIN_LABEL_COVERAGE,
                 min_method_coverage=MIN_METHOD_COVERAGE,
                 subsample_per_group=None):
        self.raw = df
        self.score_cols = list(score_cols)
        self.label_cols = list(label_cols)
        self.group_col = group_col
        self.aggs = list(aggs)
        self.min_group_n = min_group_n
        self.min_group_pos = min_group_pos
        self.orient = orient          # 'primary' | 'none'
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.min_label_coverage = min_label_coverage
        self.min_method_coverage = min_method_coverage
        self.subsample_per_group = subsample_per_group
        self.notes = []

    # ---------------------------------------------------------------- preparation
    def _prepare(self):
        df = self.raw
        rng = np.random.default_rng(self.seed)
        g = df[self.group_col].astype(str).fillna("(missing)").values
        g[g == "nan"] = "(missing)"

        if self.subsample_per_group:
            keep = []
            for gg in np.unique(g):
                ix = np.where(g == gg)[0]
                if len(ix) > self.subsample_per_group:
                    ix = rng.choice(ix, self.subsample_per_group, replace=False)
                keep.append(ix)
            keep = np.sort(np.concatenate(keep))
            df, g = df.iloc[keep].reset_index(drop=True), g[keep]
            self.notes.append(f"subsampled to <= {self.subsample_per_group} items per group "
                              f"(seeded), n = {len(df)}")

        Y, dropped_y = {}, []
        for c in self.label_cols:
            if c not in df.columns:
                dropped_y.append((c, "absent")); continue
            v = df[c]
            if v.dtype == bool:
                y = v.values.astype(bool)
            else:
                num = pd.to_numeric(v, errors="coerce")
                if num.notna().mean() > 0.5:
                    y = num.fillna(0).values > 0.5
                else:
                    y = (v.astype(str).str.strip().str.lower()
                         .isin(["true", "1", "yes", "y", "positive", "success"]).values)
            cov = float(pd.to_numeric(v, errors="coerce").notna().mean()
                        if v.dtype != bool else 1.0)
            if cov < self.min_label_coverage:
                dropped_y.append((c, f"coverage {cov:.1%}")); continue
            if 0 < y.sum() < len(y):
                Y[c] = y
            else:
                dropped_y.append((c, f"degenerate ({int(y.sum())} pos / {len(y)})"))
        if not Y:
            raise ValueError("no usable label column after filtering")

        X, raw_auc_rows, dropped_s = {}, [], []
        primary = Y[list(Y)[0]]
        for c in self.score_cols:
            if c not in df.columns:
                dropped_s.append((c, "absent")); continue
            v = pd.to_numeric(df[c], errors="coerce").values.astype(float)
            cov = float(np.isfinite(v).mean())
            if cov < self.min_method_coverage:
                dropped_s.append((c, f"coverage {cov:.1%}")); continue
            m = _metrics(primary, v)
            auc_raw = m["auc"]
            flip = bool(np.isfinite(auc_raw) and auc_raw < 0.5)
            X[c] = -v if (flip and self.orient == "primary") else v
            raw_auc_rows.append(dict(score=c, coverage=cov, auc_raw_primary_label=auc_raw,
                                     flipped=flip))
        if not X:
            raise ValueError("no usable score column after filtering")
        self.orientation = pd.DataFrame(raw_auc_rows)
        self.dropped = pd.DataFrame(
            [dict(kind="label", name=n, reason=r) for n, r in dropped_y] +
            [dict(kind="score", name=n, reason=r) for n, r in dropped_s])
        self.Y, self.X, self.g = Y, X, g
        self.groups = np.unique(g)
        self.gidx = {gg: np.where(g == gg)[0] for gg in self.groups}
        self.gcode = pd.Categorical(g, categories=list(self.groups)).codes.astype(int)
        self.methods = list(X)
        self.labels = list(Y)
        self.protocols = [f"{a}_{m}[{L}]" for L in self.labels for a in self.aggs
                          for m in METRIC_NAMES]
        return self

    # ---------------------------------------------------------------- evaluation
    def _block(self, scores, ix):
        """All protocols for one method on one draw (positional index array)."""
        y = {L: self.Y[L][ix] for L in self.labels}
        s = scores[ix]
        gc = self.gcode[ix]
        ok = np.isfinite(s)
        out = {}
        for L in self.labels:
            base = _metrics(y[L][ok], s[ok])
            for m, v in base.items():
                out[f"pooled_{m}[{L}]"] = v
            per = {m: [] for m in METRIC_NAMES}
            for gj in range(len(self.groups)):
                sel = np.where(gc == gj)[0]
                if len(sel) < self.min_group_n:
                    continue
                ss = s[sel]
                okk = np.isfinite(ss)
                yy, ss = y[L][sel][okk], ss[okk]
                if yy.sum() < self.min_group_pos or yy.sum() == len(yy):
                    continue
                b = _metrics(yy, ss)
                for m in METRIC_NAMES:
                    per[m].append(b[m])
            for m in METRIC_NAMES:
                if "macro" in self.aggs:
                    out[f"macro_{m}[{L}]"] = _nanmean(per[m])
                if "median" in self.aggs:
                    out[f"median_{m}[{L}]"] = _nanmedian(per[m])
        return out

    def run(self):
        self._prepare()
        t0 = time.time()
        n_all = len(self.raw) if self.subsample_per_group is None else None
        rows = []
        for mi, mth in enumerate(self.methods):
            r = {"method": mth}
            r.update(self._block(self.X[mth], np.arange(len(self.g))))
            rows.append(r)
        self.matrix = pd.DataFrame(rows).reindex(columns=["method"] + self.protocols)
        self.stage_times = {"point_estimates": time.time() - t0}

        # ---- two-level bootstrap: resample groups, then items within group --------
        if self.n_boot > 0:
            t0 = time.time()
            rng = np.random.default_rng(self.seed + 1)
            boot = {p: np.full((len(self.methods), self.n_boot), np.nan)
                    for p in self.protocols}
            for b in range(self.n_boot):
                pick = rng.integers(0, len(self.groups), len(self.groups))
                draw = np.concatenate([
                    self.gidx[self.groups[j]][
                        rng.integers(0, len(self.gidx[self.groups[j]]),
                                     len(self.gidx[self.groups[j]]))]
                    for j in pick])
                for mi, mth in enumerate(self.methods):
                    v = self._block(self.X[mth], draw)
                    for p in self.protocols:
                        boot[p][mi, b] = v[p]
                if (b + 1) % max(1, self.n_boot // 4) == 0:
                    print(f"    boot {b+1}/{self.n_boot}  {time.time()-t0:.0f}s",
                          file=sys.stderr)
            self.boot = boot
            self.stage_times["bootstrap"] = time.time() - t0
        else:
            self.boot = None

        self._summarise()
        self.meta.update({"elapsed_s": round(sum(self.stage_times.values()), 1)})
        return self.card()

    # ---------------------------------------------------------------- summaries
    def _summarise(self):
        M, PROTOS = self.matrix, self.protocols
        methods = M["method"].values
        cov = M[PROTOS].notna().sum()
        comparable = [p for p in PROTOS if cov[p] >= max(2, MIN_PROTOCOL_COVERAGE * len(methods))]
        self.coverage = pd.DataFrame({"protocol": PROTOS,
                                      "n_methods_with_value": cov[PROTOS].values,
                                      "comparable": [p in comparable for p in PROTOS]})
        sub = M.set_index("method")[comparable] if comparable else None
        if sub is not None and len(comparable) >= 2:
            AG = sub.rank(ascending=False).corr(method="spearman")
            win = sub.idxmax()
        else:
            AG = pd.DataFrame(); win = pd.Series(dtype=object)
        self.agreement = AG
        self.top1 = pd.DataFrame({"protocol": comparable,
                                  "winner": win.values if len(win) else []})
        pairs = [AG.loc[a, b] for a, b in itertools.combinations(comparable, 2)
                 if np.isfinite(AG.loc[a, b])]
        rk = M[PROTOS].rank(ascending=False)
        self.instability = pd.DataFrame({
            "method": methods,
            "mean_rank": rk.mean(axis=1).values,
            "rank_sd": rk.std(axis=1).values,
            "rank_range": (rk.max(axis=1) - rk.min(axis=1)).values,
            "rank_sd_comparable": (sub.rank(ascending=False).std(axis=1).values
                                   if sub is not None else np.nan),
        })

        # separability: paired bootstrap of (method - winner) per protocol
        sep = []
        if self.boot is not None:
            for p in comparable:
                bm = self.boot[p]
                est = M[p].values
                o = np.argsort(-est)
                b1 = bm[o[0]]
                tied = 1
                for k in range(1, len(o)):
                    d = bm[o[k]] - b1
                    d = d[np.isfinite(d)]
                    if len(d) < 30:
                        break
                    lo, hi = np.percentile(d, [100 * TIE_ALPHA / 2,
                                               100 * (1 - TIE_ALPHA / 2)])
                    if lo <= 0 <= hi:
                        tied += 1
                    else:
                        break
                sep.append(dict(protocol=p, winner=M.loc[o[0], "method"],
                                n_tied_with_winner=tied, n_methods=len(methods),
                                runner_up=M.loc[o[1], "method"] if len(o) > 1 else None,
                                sep_margin=float(est[o[0]] - est[o[1]]) if len(o) > 1 else np.nan))
        self.separability = pd.DataFrame(sep)

        self.meta = {
            "toolkit_version": __version__,
            "n_items": int(len(self.g)),
            "n_groups": int(len(self.groups)),
            "group_sizes": {str(k): int(v) for k, v in
                            zip(*np.unique(self.g, return_counts=True))},
            "n_methods": int(len(methods)),
            "n_labels": int(len(self.labels)),
            "n_protocols": int(len(PROTOS)),
            "n_comparable_protocols": int(len(comparable)),
            "n_distinct_winners": int(len(set(win.values))) if len(win) else 0,
            "winners": sorted(set(map(str, win.values))) if len(win) else [],
            "winner_counts": ({str(k): int(v) for k, v in
                               pd.Series(win.values).value_counts().items()}
                              if len(win) else {}),
            "pairwise_spearman_min": (round(float(min(pairs)), 4) if pairs else None),
            "pairwise_spearman_median": (round(float(np.median(pairs)), 4) if pairs else None),
            "pairwise_spearman_frac_negative": (round(float(np.mean(np.array(pairs) < 0)), 4)
                                                if pairs else None),
            "n_boot": self.n_boot,
            "stage_times": {k: round(v, 1) for k, v in self.stage_times.items()},
            "notes": list(self.notes),
            "verdict": None,
        }
        # per-numeric verdict
        frac = self.meta["pairwise_spearman_frac_negative"]
        ndw = self.meta["n_distinct_winners"]
        nm = self.meta["n_methods"]
        if ndw <= 1:
            v = "STABLE"
        elif ndw <= max(2, 0.20 * nm):
            v = "MODERATE"
        else:
            v = "FRAGILE"
        if frac is not None and frac > 0.02:
            v += "+SIGN-DISAGREEMENT"
        self.meta["verdict"] = v
        if len(self.separability):
            self.meta["median_tied_with_winner"] = int(self.separability.n_tied_with_winner.median())
            self.meta["max_tied_with_winner"] = int(self.separability.n_tied_with_winner.max())
        self.comparable = comparable
        self.method_names = list(methods)

    # ---------------------------------------------------------------- report
    def card(self):
        m = self.meta
        return {"meta": m, "matrix": self.matrix, "coverage": self.coverage,
                "agreement": self.agreement, "top1": self.top1,
                "instability": self.instability, "separability": self.separability,
                "orientation": self.orientation, "dropped": self.dropped}

    def markdown(self, title="Protocol audit card"):
        m, L = self.meta, []
        A = L.append
        A(f"# {title}\n")
        A(f"`protocol_audit` v{__version__} — {m['n_items']:,} items · "
          f"{m['n_groups']} groups · {m['n_methods']} methods · "
          f"{m['n_labels']} label definitions · {m['n_protocols']} protocols.\n")
        A(f"**Verdict: `{m['verdict']}`** — {m['n_distinct_winners']} distinct winners "
          f"over {m['n_comparable_protocols']} comparable protocols; "
          f"pairwise leaderboard Spearman min "
          f"{m['pairwise_spearman_min']} / median {m['pairwise_spearman_median']}.\n")
        if m.get("notes"):
            A("> " + "; ".join(m["notes"]) + "\n")
        A("## 1. Input audit\n")
        A("| | |")
        A("|---|---|")
        A(f"| items | {m['n_items']:,} |")
        A(f"| groups | {m['n_groups']} |")
        A(f"| methods (score columns) | {m['n_methods']} |")
        A(f"| label definitions | {m['n_labels']} |")
        A(f"| protocols enumerated | {m['n_protocols']} |")
        A(f"| protocols actually comparable | {m['n_comparable_protocols']} |")
        A("")
        if len(self.dropped):
            A("**Dropped columns** (silently averaging over these is the most common "
              "audit failure):\n")
            A("| kind | name | reason |")
            A("|---|---|---|")
            for _, r in self.dropped.iterrows():
                A(f"| {r.kind} | `{r['name']}` | {r.reason} |")
            A("")
        A("**Orientation** (each score flipped so that higher = better vs the primary "
          "label; the raw un-oriented AUC is preserved):\n")
        A("| score | coverage | raw AUC | flipped |")
        A("|---|---|---|---|")
        for _, r in self.orientation.iterrows():
            A(f"| `{r.score}` | {r.coverage:.1%} | "
              f"{'n/a' if not np.isfinite(r.auc_raw_primary_label) else f'{r.auc_raw_primary_label:.3f}'} | "
              f"{'yes' if r.flipped else 'no'} |")
        A("")
        A("## 2. Winner diversity\n")
        if m["winner_counts"]:
            A("| winner | protocols won |")
            A("|---|---|")
            for k, v in sorted(m["winner_counts"].items(), key=lambda kv: -kv[1]):
                A(f"| `{k}` | {v} |")
            A("")
            A(f"> {m['n_distinct_winners']} distinct winners over "
              f"{m['n_comparable_protocols']} comparable protocols. "
              f"Any claim of the form “method X is best” is therefore a claim about "
              f"the protocol, not about the method.\n")
        if len(self.separability):
            A("## 3. Separability (two-level paired bootstrap)\n")
            s = self.separability.sort_values("n_tied_with_winner", ascending=False)
            A("| protocol | winner | # tied with winner | sample |")
            A("|---|---|---|---|")
            for _, r in s.head(15).iterrows():
                A(f"| `{r.protocol}` | `{r.winner}` | **{r.n_tied_with_winner} / "
                  f"{r.n_methods}** | {r.n_tied_with_winner / r.n_methods:.0%} |")
            A("")
            A(f"> Median protocol has **{m.get('median_tied_with_winner')} of "
              f"{m['n_methods']}** methods statistically tied with its winner. "
              f"If that number is close to the method count, the leaderboard carries "
              f"no ranking information at all at this sample size.\n")
        A("## 4. Stability of individual methods\n")
        i_s = self.instability.sort_values("rank_sd", ascending=False)
        A("| method | mean rank | rank SD | worst-case rank range |")
        A("|---|---|---|---|")
        for _, r in pd.concat([i_s.head(5), i_s.tail(3)]).iterrows():
            A(f"| `{r.method}` | {r.mean_rank:.1f} | {r.rank_sd:.1f} | "
              f"{r.rank_range:.0f} |")
        A("")
        A("## 5. What to report\n")
        A("- The **number of comparable protocols** and the **winner set**, not a "
          "single ranking.\n")
        A("- The **paired-bootstrap tied set** for the top method, not a bare `#1`.\n")
        A("- Any protocol in which the leaderboard Spearman is < 0 against the "
          "headline protocol — those pairs are the ones a reviewer will find.\n")
        return "\n".join(L)

    def write(self, outdir, title="Protocol audit card"):
        os.makedirs(outdir, exist_ok=True)
        self.matrix.to_csv(f"{outdir}/matrix.csv", index=False)
        self.coverage.to_csv(f"{outdir}/coverage.csv", index=False)
        self.agreement.to_csv(f"{outdir}/agreement.csv")
        self.top1.to_csv(f"{outdir}/top1.csv", index=False)
        self.instability.to_csv(f"{outdir}/instability.csv", index=False)
        if len(self.separability):
            self.separability.to_csv(f"{outdir}/separability.csv", index=False)
        self.orientation.to_csv(f"{outdir}/orientation.csv", index=False)
        if len(self.dropped):
            self.dropped.to_csv(f"{outdir}/dropped.csv", index=False)
        with open(f"{outdir}/audit.json", "w") as fh:
            json.dump(self.meta, fh, indent=2, ensure_ascii=False)
        with open(f"{outdir}/audit_card.md", "w") as fh:
            fh.write(self.markdown(title))
        if self.boot is not None:
            np.savez_compressed(f"{outdir}/boot.npz",
                                protocols=np.array(self.protocols, dtype=object),
                                methods=np.array(self.methods, dtype=object),
                                **self.boot)
        return outdir


# ------------------------------------------------- audit an existing leaderboard matrix
def audit_from_matrix(matrix: pd.DataFrame, protocol_cols=None, boot=None,
                      n_methods_min=2, title="Protocol audit card (from matrix)"):
    """Audit a pre-computed (method x protocol) leaderboard matrix.

    Use this when the protocols were produced elsewhere (e.g. by an older script) and
    only the agreement / winner / separability stage is needed.  `boot` is an optional
    {protocol: (n_methods x n_boot) array} for the paired separability test.
    """
    methods = matrix["method"].values if "method" in matrix.columns else matrix.index.values
    P = protocol_cols or [c for c in matrix.columns if c != "method"]
    cov = matrix[P].notna().sum()
    comparable = [p for p in P if cov[p] >= max(n_methods_min, MIN_PROTOCOL_COVERAGE * len(methods))]
    sub = matrix.set_index(matrix.columns[0])[comparable]
    win = sub.idxmax()
    AG = sub.rank(ascending=False).corr(method="spearman")
    pairs = [AG.loc[a, b] for a, b in itertools.combinations(comparable, 2)
             if np.isfinite(AG.loc[a, b])]
    sep = []
    if boot is not None:
        for p in comparable:
            if p not in boot:
                continue
            bm = boot[p]; est = matrix[p].values; o = np.argsort(-est); b1 = bm[o[0]]
            tied = 1
            for k in range(1, len(o)):
                d = bm[o[k]] - b1
                d = d[np.isfinite(d)]
                if len(d) < 30:
                    break
                lo, hi = np.percentile(d, [100 * TIE_ALPHA / 2, 100 * (1 - TIE_ALPHA / 2)])
                if lo <= 0 <= hi:
                    tied += 1
                else:
                    break
            sep.append(dict(protocol=p, winner=matrix.iloc[o[0]][matrix.columns[0]],
                            n_tied_with_winner=tied, n_methods=len(methods)))
    meta = {
        "n_methods": int(len(methods)), "n_protocols": int(len(P)),
        "n_comparable_protocols": int(len(comparable)),
        "n_distinct_winners": int(len(set(win.values))),
        "winners": sorted(set(map(str, win.values))),
        "winner_counts": {str(k): int(v) for k, v in pd.Series(win.values).value_counts().items()},
        "pairwise_spearman_min": round(float(min(pairs)), 4) if pairs else None,
        "pairwise_spearman_median": round(float(np.median(pairs)), 4) if pairs else None,
        "pairwise_spearman_frac_negative": round(float(np.mean(np.array(pairs) < 0)), 4) if pairs else None,
        "median_tied_with_winner": int(np.median([s["n_tied_with_winner"] for s in sep])) if sep else None,
        "max_tied_with_winner": int(max([s["n_tied_with_winner"] for s in sep])) if sep else None,
        "verdict": ("STABLE" if len(set(win.values)) <= 1 else
                    "MODERATE" if len(set(win.values)) <= max(2, .2 * len(methods)) else "FRAGILE"),
    }
    if meta["pairwise_spearman_frac_negative"] and meta["pairwise_spearman_frac_negative"] > 0.02:
        meta["verdict"] += "+SIGN-DISAGREEMENT"
    return {"meta": meta, "top1": pd.DataFrame({"protocol": comparable, "winner": win.values}),
            "agreement": AG, "separability": pd.DataFrame(sep)}


# --------------------------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser(description="protocol_audit — audit an evaluation protocol grid")
    ap.add_argument("--table", required=True, help="CSV/parquet of items")
    ap.add_argument("--scores", nargs="+", required=True, help="candidate score columns")
    ap.add_argument("--labels", nargs="+", required=True, help="candidate label columns")
    ap.add_argument("--group", required=True, help="grouping column (target / provenance)")
    ap.add_argument("--aggs", nargs="+", default=list(DEFAULT_AGGS))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample-per-group", type=int, default=None)
    ap.add_argument("--orient", choices=["primary", "none"], default="primary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Protocol audit card")
    a = ap.parse_args()
    df = pd.read_parquet(a.table) if a.table.endswith(".parquet") else pd.read_csv(a.table)
    au = ProtocolAudit(df, a.scores, a.labels, a.group, aggs=a.aggs, n_boot=a.n_boot,
                       seed=a.seed, orient=a.orient,
                       subsample_per_group=a.subsample_per_group)
    au.run()
    au.write(a.out, a.title)
    print(au.markdown(a.title))


if __name__ == "__main__":
    _cli()
