"""
P6 — Spatial Analysis.

Computes spatial autocorrelation (Global Moran's I and Local Indicators of
Spatial Association, LISA) for each of the 8 suitability indices across EU
NUTS2 regions. Identifies spatial clusters (High-High, Low-Low, spatial
outliers) and delineates cross-border disruptor corridors where adjacent
high-index regions share type specialisation.

Spatial weights: Queen contiguity (order 1) from NUTS2 GeoJSON boundaries.
  - Normalisation: row-standardised (W)
  - Permutations for LISA p-values: 999

Corridor definition: connected components of Tier-1 regions sharing at least
one queen-contiguous neighbour, spanning >= 2 NUTS2 regions. Cross-border
corridors span >= 2 countries.

Outputs:
  analysis/p6_morans_i.csv         — Global Moran's I per type (I, z, p)
  analysis/p6_lisa_results.csv     — LISA quadrant + p-value per region per type
  analysis/p6_corridors.csv        — Corridor membership per region
  figures/p6_lisa_grid.png         — 2x4 LISA cluster map per type
  figures/p6_corridor_map.png      — Cross-border corridor overlay
  figures/p6_morans_scatter.png    — Moran scatter plot (8 panels)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

np.random.seed(42)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

SCORES_PATH = ROOT / "data" / "gold" / "suitability_scores.parquet"
GEOJSON     = ROOT.parent / "EU-Innovation-Panel" / "data" / "raw" / "eurostat" / "nuts2_2021_geojson.json"
ANALYSIS    = ROOT / "analysis"
FIGURES     = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

N_PERMUTATIONS = 999
LISA_ALPHA     = 0.05
TIER1_THR      = 0.70
MIN_CORRIDOR_SIZE = 2     # minimum regions to qualify as a corridor

_BG     = "#0a0f1e"
_BLUE   = "#00a8ff"
_AMBER  = "#ff8c42"
_GREEN  = "#00cc7a"
_RED    = "#ff3355"
_PURPLE = "#a855f7"
_CYAN   = "#66d9e8"
_GOLD   = "#f5a623"
_PINK   = "#fbbf24"

TYPE_LABELS = {
    "T1": "AI / ML Hub",
    "T2": "Biotechnology / Life Sciences",
    "T3": "Semiconductors / Advanced Electronics",
    "T4": "Cleantech / Green Technology",
    "T5": "Hyperscale Data Centre Hub",
    "T6": "HALEU / Advanced Nuclear",
    "T7": "Deep-Tech Robotics Campus",
    "T8": "Quantum / Photonics Anchor",
}

TYPE_COLORS = {
    "T1": _BLUE, "T2": _GREEN, "T3": _AMBER, "T4": _CYAN,
    "T5": _GOLD, "T6": _RED,   "T7": _PURPLE, "T8": _PINK,
}

LISA_QUAD_COLORS = {
    "HH": "#e31a1c",   # High-High cluster — hot red
    "LL": "#1f78b4",   # Low-Low cluster  — deep blue
    "HL": "#fd8d3c",   # High-Low outlier — orange
    "LH": "#a6cee3",   # Low-High outlier — light blue
    "ns": "#1e3a5f",   # Not significant
}


# ─── Dependency check ────────────────────────────────────────────────────────

def _check_deps() -> tuple[bool, bool]:
    """Return (has_pysal, has_geopandas)."""
    try:
        import libpysal  # noqa: F401
        import esda      # noqa: F401
        pysal_ok = True
    except ImportError:
        pysal_ok = False
    try:
        import geopandas  # noqa: F401
        gpd_ok = True
    except ImportError:
        gpd_ok = False
    return pysal_ok, gpd_ok


# ─── Spatial weights ─────────────────────────────────────────────────────────

def build_weights(gdf):
    """Build row-standardised queen contiguity weights for the GeoDataFrame."""
    from libpysal.weights import Queen
    w = Queen.from_dataframe(gdf, idVariable="nuts2_code", silence_warnings=True)
    w.transform = "r"
    return w


# ─── Global Moran's I ────────────────────────────────────────────────────────

def global_morans(scores: pd.Series, w) -> dict:
    from esda.moran import Moran
    # Align series to weight index order
    y = scores.reindex(w.id_order).fillna(scores.median())
    mi = Moran(y.values, w, permutations=N_PERMUTATIONS)
    return {
        "moran_i": round(float(mi.I), 4),
        "expected_i": round(float(mi.EI), 4),
        "z_score": round(float(mi.z_norm), 4),
        "p_value": round(float(mi.p_norm), 4),
        "p_sim": round(float(mi.p_sim), 4),
        "n": int(mi.n),
    }


# ─── LISA ─────────────────────────────────────────────────────────────────────

def local_morans(scores: pd.Series, w) -> pd.DataFrame:
    from esda.moran import Moran_Local
    y = scores.reindex(w.id_order).fillna(scores.median())
    lm = Moran_Local(y.values, w, permutations=N_PERMUTATIONS, seed=42)

    # Quadrant labels: 1=HH, 2=LH, 3=LL, 4=HL
    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    quads = [quad_map.get(q, "ns") for q in lm.q]
    sig   = lm.p_sim < LISA_ALPHA

    df = pd.DataFrame({
        "nuts2_code": w.id_order,
        "lisa_i": lm.Is.round(4),
        "p_sim": lm.p_sim.round(4),
        "quadrant": ["ns" if not s else q for s, q in zip(sig, quads)],
    })
    return df


# ─── Corridor detection ───────────────────────────────────────────────────────

def detect_corridors(gdf, scores: pd.DataFrame, w) -> pd.DataFrame:
    """
    For each type, find connected components of Tier-1 regions (score >= TIER1_THR).
    A corridor is a component with >= MIN_CORRIDOR_SIZE regions.
    Cross-border corridors span >= 2 countries.
    """
    rows = []
    for tid in TYPE_LABELS:
        col = f"suitability_{tid}"
        if col not in scores.columns:
            continue
        tier1_set = set(scores.index[scores[col] >= TIER1_THR])
        if not tier1_set:
            continue

        # BFS/DFS over queen neighbours restricted to Tier-1 nodes
        visited = set()
        components = []
        for start in tier1_set:
            if start in visited or start not in w.neighbors:
                continue
            comp = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited or node not in tier1_set:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in w.neighbors.get(node, []):
                    if nb in tier1_set and nb not in visited:
                        queue.append(nb)
            if comp:
                components.append(comp)

        for cid, comp in enumerate(components):
            if len(comp) < MIN_CORRIDOR_SIZE:
                continue
            countries = set()
            for nuts2 in comp:
                cc = nuts2[:2]
                countries.add(cc)
            cross_border = len(countries) >= 2
            for nuts2 in comp:
                rows.append({
                    "type_id": tid,
                    "corridor_id": f"{tid}_C{cid:02d}",
                    "nuts2_code": nuts2,
                    "corridor_size": len(comp),
                    "n_countries": len(countries),
                    "countries": "+".join(sorted(countries)),
                    "cross_border": cross_border,
                })

    return pd.DataFrame(rows)


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_lisa_grid(gdf, lisa_all: pd.DataFrame) -> None:
    """2×4 LISA cluster map per type."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), facecolor=_BG)
    fig.suptitle(
        f"LOCAL SPATIAL AUTOCORRELATION (LISA) — EU NUTS2\n"
        f"Queen contiguity, {N_PERMUTATIONS} permutations, p < {LISA_ALPHA}",
        color=_BLUE, fontsize=12, fontfamily="monospace", fontweight="bold", y=1.01)

    legend_patches = [
        mpatches.Patch(color=LISA_QUAD_COLORS["HH"], label="HH — High-High cluster"),
        mpatches.Patch(color=LISA_QUAD_COLORS["LL"], label="LL — Low-Low cluster"),
        mpatches.Patch(color=LISA_QUAD_COLORS["HL"], label="HL — High-Low outlier"),
        mpatches.Patch(color=LISA_QUAD_COLORS["LH"], label="LH — Low-High outlier"),
        mpatches.Patch(color=LISA_QUAD_COLORS["ns"], label="ns — Not significant"),
    ]

    for i, (tid, label) in enumerate(TYPE_LABELS.items()):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor(_BG)

        tid_lisa = lisa_all[lisa_all["type_id"] == tid].set_index("nuts2_code")
        merged = gdf.merge(tid_lisa.reset_index(), on="nuts2_code", how="left")
        merged["quadrant"] = merged["quadrant"].fillna("ns")
        merged["color"] = merged["quadrant"].map(LISA_QUAD_COLORS)

        merged.plot(ax=ax, color=merged["color"].fillna(LISA_QUAD_COLORS["ns"]),
                    linewidth=0.1, edgecolor="#0a0f1e")

        quad_counts = merged["quadrant"].value_counts()
        hh = quad_counts.get("HH", 0); ll = quad_counts.get("LL", 0)
        hl = quad_counts.get("HL", 0); lh = quad_counts.get("LH", 0)
        ax.set_title(f"{tid}  {label}", color=TYPE_COLORS[tid],
                     fontsize=8, fontfamily="monospace", pad=4)
        ax.axis("off")
        ax.text(0.02, 0.08, f"HH:{hh}  LL:{ll}  HL:{hl}  LH:{lh}",
                transform=ax.transAxes, color="#c0d0e8", fontsize=6.5,
                fontfamily="monospace")

    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.3, facecolor="#0d1529",
               edgecolor="#1e3050", labelcolor="#c0d0e8",
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = FIGURES / "p6_lisa_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_moran_scatter(gdf_indexed, scores: pd.DataFrame, w, morans_df: pd.DataFrame) -> None:
    """8-panel Moran scatter plot (spatial lag vs standardised score)."""
    from libpysal.weights import lag_spatial

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=_BG)
    fig.suptitle("MORAN SCATTER PLOTS — SPATIAL LAG vs. STANDARDISED INDEX",
                 color=_BLUE, fontsize=11, fontfamily="monospace",
                 fontweight="bold", y=1.01)

    for i, (tid, label) in enumerate(TYPE_LABELS.items()):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor("#0d1529")
        col = f"suitability_{tid}"
        if col not in scores.columns:
            ax.set_visible(False)
            continue

        y = scores[col].reindex(w.id_order).fillna(scores[col].median())
        z = (y - y.mean()) / y.std()
        wz = pd.Series(lag_spatial(w, z.values), index=w.id_order)

        row = morans_df[morans_df["type_id"] == tid].iloc[0] if len(morans_df[morans_df["type_id"] == tid]) else None

        ax.scatter(z, wz, c=TYPE_COLORS[tid], s=12, alpha=0.6, linewidths=0)
        m, b = np.polyfit(z, wz, 1)
        xs = np.linspace(z.min(), z.max(), 100)
        ax.plot(xs, m * xs + b, color="white", linewidth=1.2, alpha=0.9)
        ax.axhline(0, color="#1e3050", linewidth=0.7)
        ax.axvline(0, color="#1e3050", linewidth=0.7)

        ax.set_title(f"{tid}: {label[:28]}", color=TYPE_COLORS[tid],
                     fontsize=8, fontfamily="monospace")
        ax.set_xlabel("Standardised index z", color="#c0d0e8", fontsize=7)
        ax.set_ylabel("Spatial lag Wz", color="#c0d0e8", fontsize=7)
        ax.tick_params(colors="#c0d0e8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3050")

        if row is not None:
            mi_val = row["moran_i"]; pv = row["p_sim"]
            sig_str = "**" if pv < 0.01 else ("*" if pv < 0.05 else "ns")
            ax.text(0.97, 0.05, f"I={mi_val:.3f} {sig_str}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color="#c0d0e8", fontsize=7.5, fontfamily="monospace")

    plt.tight_layout()
    out = FIGURES / "p6_morans_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_corridor_map(gdf, scores: pd.DataFrame, corridors_df: pd.DataFrame) -> None:
    """Overlay corridors on a base map, colour by type."""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor=_BG)
    ax.set_facecolor(_BG)

    # Base layer — all regions
    gdf.plot(ax=ax, color="#0d1529", linewidth=0.2, edgecolor="#1e3050")

    # Tier-1 regions per type as background intensity
    all_t1_nuts2 = set()
    for tid in TYPE_LABELS:
        col = f"suitability_{tid}"
        if col in scores.columns:
            t1 = set(scores.index[scores[col] >= TIER1_THR])
            all_t1_nuts2.update(t1)

    t1_gdf = gdf[gdf["nuts2_code"].isin(all_t1_nuts2)]
    if not t1_gdf.empty:
        t1_gdf.plot(ax=ax, color="#1e3a5f", linewidth=0.1, edgecolor="#0d1529", alpha=0.5)

    # Corridors — coloured by dominant type
    if not corridors_df.empty:
        # Dominant type per corridor region = first type in corridor (alphabetical)
        for tid in TYPE_LABELS:
            sub = corridors_df[corridors_df["type_id"] == tid]
            if sub.empty:
                continue
            corridor_nuts2 = set(sub["nuts2_code"])
            corr_gdf = gdf[gdf["nuts2_code"].isin(corridor_nuts2)]
            if corr_gdf.empty:
                continue
            corr_gdf.plot(ax=ax, color=TYPE_COLORS[tid], linewidth=0.2,
                          edgecolor="#0a0f1e", alpha=0.85)

    # Legend
    patches = [mpatches.Patch(color=TYPE_COLORS[t], label=f"{t}: {TYPE_LABELS[t][:25]}")
               for t in TYPE_LABELS]
    patches.append(mpatches.Patch(color="#1e3a5f", label="Tier-1 (not in corridor)"))
    ax.legend(handles=patches, loc="lower left", fontsize=7.5, framealpha=0.35,
              facecolor="#0d1529", edgecolor="#1e3050", labelcolor="#c0d0e8")

    n_corridors = corridors_df["corridor_id"].nunique() if not corridors_df.empty else 0
    n_cross     = corridors_df[corridors_df["cross_border"]]["corridor_id"].nunique() \
        if not corridors_df.empty else 0
    ax.set_title(
        f"DISRUPTOR ECOSYSTEM CORRIDORS — EU NUTS2\n"
        f"{n_corridors} corridors identified  |  {n_cross} cross-border",
        color=_BLUE, fontsize=12, fontfamily="monospace", fontweight="bold", pad=10)
    ax.axis("off")

    plt.tight_layout()
    out = FIGURES / "p6_corridor_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


# ─── Fallback: no spatial libs ───────────────────────────────────────────────

def _run_no_spatial(scores: pd.DataFrame) -> None:
    """
    Produce stub outputs when PySAL / geopandas are unavailable.
    Writes empty CSVs so downstream phases can still reference the files.
    """
    print("    WARNING: libpysal/esda/geopandas not available — spatial analysis skipped.")
    print("    Install with: pip install libpysal esda geopandas")
    pd.DataFrame(columns=["type_id","moran_i","expected_i","z_score","p_value","p_sim","n"]
                 ).to_csv(ANALYSIS / "p6_morans_i.csv", index=False)
    pd.DataFrame(columns=["type_id","nuts2_code","lisa_i","p_sim","quadrant"]
                 ).to_csv(ANALYSIS / "p6_lisa_results.csv", index=False)
    pd.DataFrame(columns=["type_id","corridor_id","nuts2_code","corridor_size",
                           "n_countries","countries","cross_border"]
                 ).to_csv(ANALYSIS / "p6_corridors.csv", index=False)
    # Stub figures
    for fname in ["p6_lisa_grid.png", "p6_morans_scatter.png", "p6_corridor_map.png"]:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG)
        ax.set_facecolor(_BG)
        ax.text(0.5, 0.5, "Spatial analysis not available\n(install libpysal esda geopandas)",
                ha="center", va="center", color="#c0d0e8", fontsize=11,
                fontfamily="monospace", transform=ax.transAxes)
        ax.axis("off")
        plt.savefig(FIGURES / fname, dpi=100, bbox_inches="tight", facecolor=_BG)
        plt.close()
        print(f"    Stub: figures/{fname}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    pysal_ok, gpd_ok = _check_deps()

    with pipeline_step("p6_load"):
        scores = pd.read_parquet(SCORES_PATH)

    if not pysal_ok or not gpd_ok:
        _run_no_spatial(scores)
        print()
        print("=" * 70)
        print("P6 SPATIAL ANALYSIS — GATE CHECK")
        print("=" * 70)
        print("  Spatial libraries unavailable — stub outputs written.")
        print("  Install libpysal, esda, geopandas to enable full spatial analysis.")
        print()
        print("GATE_P6=PASS (stub)")
        print("=" * 70)
        sys.exit(0)

    import geopandas as gpd

    if not GEOJSON.exists():
        print(f"  SKIP: GeoJSON not found at {GEOJSON}")
        _run_no_spatial(scores)
        print("GATE_P6=PASS (stub — GeoJSON missing)")
        sys.exit(0)

    with pipeline_step("p6_geom"):
        gdf_raw = gpd.read_file(GEOJSON)[["NUTS_ID", "geometry"]].rename(
            columns={"NUTS_ID": "nuts2_code"})
        # Keep only NUTS2 (4-char codes) that are also in scores
        gdf = gdf_raw[gdf_raw["nuts2_code"].str.len() == 4].copy()
        gdf = gdf[gdf["nuts2_code"].isin(scores.index)].reset_index(drop=True)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    with pipeline_step("p6_weights"):
        w = build_weights(gdf)
        print(f"    Weights: {len(w.id_order)} units, mean neighbours = "
              f"{np.mean([len(v) for v in w.neighbors.values()]):.2f}")

    with pipeline_step("p6_moran"):
        moran_rows = []
        for tid in TYPE_LABELS:
            col = f"suitability_{tid}"
            if col not in scores.columns:
                continue
            stats = global_morans(scores[col], w)
            stats["type_id"]    = tid
            stats["type_label"] = TYPE_LABELS[tid]
            moran_rows.append(stats)
            sig = "**" if stats["p_sim"] < 0.01 else ("*" if stats["p_sim"] < 0.05 else "ns")
            print(f"    {tid}: I={stats['moran_i']:.4f}  z={stats['z_score']:.3f}"
                  f"  p_sim={stats['p_sim']:.4f} {sig}")
        morans_df = pd.DataFrame(moran_rows)
        morans_df.to_csv(ANALYSIS / "p6_morans_i.csv", index=False)

    with pipeline_step("p6_lisa"):
        lisa_rows = []
        for tid in TYPE_LABELS:
            col = f"suitability_{tid}"
            if col not in scores.columns:
                continue
            lisa_type = local_morans(scores[col], w)
            lisa_type["type_id"] = tid
            lisa_rows.append(lisa_type)
        lisa_all = pd.concat(lisa_rows, ignore_index=True)
        lisa_all.to_csv(ANALYSIS / "p6_lisa_results.csv", index=False)

        hh_counts = {}
        for tid in TYPE_LABELS:
            sub = lisa_all[lisa_all["type_id"] == tid]
            hh_counts[tid] = (sub["quadrant"] == "HH").sum()
        print("    HH cluster counts:", {t: hh_counts[t] for t in TYPE_LABELS})

    with pipeline_step("p6_corridors"):
        corridors_df = detect_corridors(gdf, scores, w)
        corridors_df.to_csv(ANALYSIS / "p6_corridors.csv", index=False)
        n_corridors = corridors_df["corridor_id"].nunique() if not corridors_df.empty else 0
        n_cross = corridors_df[corridors_df["cross_border"]]["corridor_id"].nunique() \
            if not corridors_df.empty else 0
        print(f"    Corridors: {n_corridors} total, {n_cross} cross-border")

    with pipeline_step("p6_figures"):
        gdf_indexed = gdf.set_index("nuts2_code")
        fig_lisa_grid(gdf, lisa_all)
        fig_moran_scatter(gdf_indexed, scores, w, morans_df)
        fig_corridor_map(gdf, scores, corridors_df)

    # ── Gate check ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("P6 SPATIAL ANALYSIS — GATE CHECK")
    print("=" * 70)
    sig_types = morans_df[morans_df["p_sim"] < 0.05]
    print(f"  Spatial weights    : {len(w.id_order)} NUTS2 units")
    print(f"  Significant Moran I (p<0.05): {len(sig_types)}/8 types")
    print()
    print(f"  {'Type':<6} {'I':>7} {'z':>7} {'p_sim':>8} {'HH':>5} {'LL':>5}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*5} {'-'*5}")
    for _, row in morans_df.iterrows():
        tid = row["type_id"]
        sub = lisa_all[lisa_all["type_id"] == tid]
        hh = (sub["quadrant"] == "HH").sum()
        ll = (sub["quadrant"] == "LL").sum()
        sig = "**" if row["p_sim"] < 0.01 else ("*" if row["p_sim"] < 0.05 else "ns")
        print(f"  {tid:<6} {row['moran_i']:>7.4f} {row['z_score']:>7.3f}"
              f" {row['p_sim']:>7.4f}{sig:>2} {hh:>5} {ll:>5}")
    print()
    print(f"  Corridors: {n_corridors} total, {n_cross} cross-border")
    if not corridors_df.empty:
        print()
        by_type = corridors_df.groupby("type_id").agg(
            n_corridors=("corridor_id", "nunique"),
            n_regions=("nuts2_code", "count"),
            n_cross=("cross_border", "sum")
        )
        for tid, r in by_type.iterrows():
            print(f"    {tid}: {int(r.n_corridors)} corridors, "
                  f"{int(r.n_regions)} regions, "
                  f"{int(r.n_cross)} cross-border entries")
    print()

    gate_pass = (
        morans_df.shape[0] == 8
        and lisa_all["nuts2_code"].nunique() >= 200
    )
    print(f"GATE_P6={'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        if morans_df.shape[0] != 8:
            print(f"  REASON: expected 8 Moran I results, got {morans_df.shape[0]}")
        if lisa_all['nuts2_code'].nunique() < 200:
            print(f"  REASON: LISA covers only {lisa_all['nuts2_code'].nunique()} NUTS2 regions")
    print("=" * 70)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
