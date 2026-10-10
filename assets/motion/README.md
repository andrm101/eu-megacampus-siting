# assets/motion — code-drawn explainer video

`explainer.html` draws every frame as a pure function of time, `render(t)`, from
`data.json`. `render.mjs` captures the page frame by frame with Playwright and
encodes it with ffmpeg. Opened in a browser, the page simply plays.

```bash
python assets/motion/export_data.py [--geojson PATH]       # numbers from analysis/p5_*, p6_*
npm i playwright                                           # once (any folder on NODE_PATH)
node assets/motion/render.mjs --name megacampus-explainer  # needs ffmpeg on PATH (or $FFMPEG)
```

The NUTS-2 outlines default to the sibling `EU-Innovation-Panel` checkout, the
same file `p4_suitability_scores.py` and `p6_spatial_analysis.py` use. Outermost
regions (Canaries, Azores, Madeira, French overseas) are not drawn. Outputs go to
`docs/media/`: the MP4 (1280×720, H.264), a 640 px GIF preview of the three
suitability maps and a poster frame.

| Time (s) | Scene | Source |
|---|---|---|
| 0–6 | Title | `p5_summary_report.json` |
| 6–24 | Suitability maps: T1, T3, T8 | `p5_shortlist_tiered.csv`, `p5_summary_report.json` |
| 24–34 | Shortlist and top-tier counts per type | `p5_summary_report.json` |
| 34–46 | Types shortlisted per region | `p5_shortlist_tiered.csv` (`n_types_above_threshold`) |
| 46–56 | Spatial corridors, Moran's I | `p6_corridors.csv`, `p6_morans_i.csv` |
| 56–62 | Outro | — |

Wording follows the language contract in `CLAUDE.md` (suitability index,
shortlisted, associations rather than causes). `motion-kit.js` and `motion.css`
are shared with the other explainers in this portfolio.
