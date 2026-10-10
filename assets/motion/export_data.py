"""
export_data.py — every number the explainer video shows, read from the
project's own outputs (analysis/p5_*, p6_*) and the NUTS-2 outlines that
p4/p6 already use (sibling EU-Innovation-Panel checkout, as in those scripts).

    python assets/motion/export_data.py [--geojson PATH]   → assets/motion/data.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AN = os.path.join(ROOT, "analysis")
GEOJSON = os.path.join(os.path.dirname(ROOT), "EU-Innovation-Panel", "data", "raw", "eurostat", "nuts2_2021_geojson.json")
W, H = 640, 560
LON0, LAT0 = math.radians(10), math.radians(52)


def laea(lon, lat):
    l, p = math.radians(lon) - LON0, math.radians(lat)
    k = math.sqrt(2 / (1 + math.sin(LAT0) * math.sin(p) + math.cos(LAT0) * math.cos(p) * math.cos(l)))
    return (k * math.cos(p) * math.sin(l),
            -k * (math.cos(LAT0) * math.sin(p) - math.sin(LAT0) * math.cos(p) * math.cos(l)))


def rows(name):
    with open(os.path.join(AN, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def paths(geojson: str, codes: set[str]) -> dict[str, dict]:
    with open(geojson, encoding="utf-8") as fh:
        geo = json.load(fh)
    feats = {}
    for f in geo["features"]:
        c = f["properties"]["NUTS_ID"]
        if c not in codes:
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        # mainland Europe only: drop overseas parts (Canaries, Azores, Madeira, French DOMs)
        rings = [[laea(x, y) for x, y in ring] for p in polys for ring in p
                 if -12 < ring[0][0] < 35 and 34 < ring[0][1] < 72]
        feats[c] = (f["properties"]["NUTS_NAME"], rings)
    xs = [x for _, rs in feats.values() for r in rs for x, _ in r]
    ys = [y for _, rs in feats.values() for r in rs for _, y in r]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    s = min(W / (x1 - x0), H / (y1 - y0))
    ox, oy = (W - s * (x1 - x0)) / 2, (H - s * (y1 - y0)) / 2
    return {c: {"name": n, "d": "".join("M" + "L".join(f"{ox + s * (x - x0):.1f},{oy + s * (y - y0):.1f}" for x, y in r) + "Z"
                                       for r in rs)}
            for c, (n, rs) in feats.items() if rs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=GEOJSON)
    a = ap.parse_args()
    tiers = rows("p5_shortlist_tiered.csv")
    with open(os.path.join(AN, "p5_summary_report.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    corr = rows("p6_corridors.csv")
    moran = {r["type_id"]: float(r["moran_i"]) for r in rows("p6_morans_i.csv")}
    geo = paths(a.geojson, {r["nuts2_code"] for r in tiers})

    in_corr = {}
    for r in corr:
        cur = in_corr.get(r["nuts2_code"], 0)
        in_corr[r["nuts2_code"]] = max(cur, 2 if r["cross_border"] == "True" else 1)
    ids = {r["corridor_id"]: r["cross_border"] == "True" for r in corr}
    regions = [{"code": r["nuts2_code"], "name": geo[r["nuts2_code"]]["name"], "d": geo[r["nuts2_code"]]["d"],
                "s": {t: round(float(r[f"suitability_{t}"]), 3) for t in ("T1", "T3", "T8")},
                "n_types": int(r["n_types_above_threshold"]), "corridor": in_corr.get(r["nuts2_code"], 0)}
               for r in tiers if r["nuts2_code"] in geo]
    data = {
        "map": {"w": W, "h": H}, "regions": regions, "n_regions": summary["n_regions"],
        "threshold": summary["tier2_threshold"], "tier1": summary["tier1_threshold"],
        "types": [{"id": k, "label": v["label"], "n": v["n_shortlisted"], "tier1": v["n_tier1"], "moran": moran[k]}
                  for k, v in summary["types"].items()],
        "corridors": {"n": len(ids), "cross_border": sum(ids.values())},
        "n_none": sum(int(r["n_types_above_threshold"]) == 0 for r in tiers),   # all regions, drawn or not
    }
    out = os.path.join(HERE, "data.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out) // 1024} kB), {len(regions)} regions drawn")
    print(data["corridors"], [(t["id"], t["n"], t["tier1"], t["moran"]) for t in data["types"]])
    print(sorted(((r["n_types"], r["name"]) for r in regions), reverse=True)[:6])


if __name__ == "__main__":
    main()
