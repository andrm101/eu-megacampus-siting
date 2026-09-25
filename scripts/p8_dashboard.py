"""
P8 — Operational Intelligence Dashboard.

Palantir-style multi-tab Dash application for the EU MegaCampus Siting
Intelligence project.  Run with:

    python scripts/p8_dashboard.py [--host 127.0.0.1] [--port 8050]

Tabs:
  1  Overview       — 8-type index cards + co-specialisation summary
  2  Type Explorer  — per-type choropleth + shortlist table + gap profile
  3  Spatial        — LISA map + Moran's I table + corridor summary
  4  Feasibility    — I-O dependency matrix + scenario funnel + gate waterfall
  5  Scenario View  — filter by scenario / type / country
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pcolors

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
GEOJSON_PATH = ROOT.parent / "EU-Innovation-Panel" / "data" / "raw" / \
               "eurostat" / "nuts2_2021_geojson.json"

def _load(path, **kwargs):
    if Path(path).suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, **kwargs)

scores    = _load(ROOT / "data" / "gold" / "suitability_scores.parquet")
shortlist = _load(ROOT / "analysis" / "p5_shortlist_tiered.csv", index_col=0)
morans    = _load(ROOT / "analysis" / "p6_morans_i.csv")
corridors = _load(ROOT / "analysis" / "p6_corridors.csv")
io_mat    = _load(ROOT / "analysis" / "p7_io_matrix.csv")
scenarios = _load(ROOT / "analysis" / "p7_scenario_assignments.csv", index_col=0)

# Merge everything into one working frame
master = scores.copy()
for col in shortlist.columns:
    if col not in master.columns:
        master[col] = shortlist[col]
for col in scenarios.columns:
    if col not in master.columns:
        master[col] = scenarios[col]
master.index.name = "nuts2_code"
master = master.reset_index()

# GeoJSON
geojson = None
if GEOJSON_PATH.exists():
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # Keep only NUTS2 features (4-char ID)
    raw["features"] = [ft for ft in raw["features"]
                       if len(ft["properties"].get("NUTS_ID", "")) == 4]
    geojson = raw


# ─── Palette ─────────────────────────────────────────────────────────────────
BG      = "#0a0f1e"
PANEL   = "#0d1529"
BORDER  = "#2a4060"       # raised from #1e3050 for visible borders
TEXT    = "#d0e4f8"       # brighter body text
MUTED   = "#8aaccc"       # visible secondary text (was DIM=#3a5070 — too dark)
DIM     = "#8aaccc"       # alias kept for compat
BLUE    = "#00a8ff"
AMBER   = "#ff8c42"
GREEN   = "#00cc7a"
RED     = "#ff3355"
PURPLE  = "#a855f7"
CYAN    = "#66d9e8"
GOLD    = "#f5a623"
PINK    = "#fbbf24"

# Space Grotesk — geometric gothic sans-serif (Founder Fund / Anduril aesthetic)
FONT    = "'Space Grotesk', sans-serif"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family=FONT),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        colorway=[BLUE, GREEN, AMBER, CYAN, GOLD, RED, PURPLE, PINK],
        legend=dict(bgcolor=PANEL, bordercolor=BORDER),
    )
)

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
    "T1": BLUE, "T2": GREEN, "T3": AMBER, "T4": CYAN,
    "T5": GOLD, "T6": RED,   "T7": PURPLE, "T8": PINK,
}


# ─── Shared style helpers ─────────────────────────────────────────────────────
def card(children, style=None):
    base = {"background": PANEL, "border": f"1px solid {BORDER}",
            "borderRadius": "6px", "padding": "16px", "marginBottom": "12px"}
    if style:
        base.update(style)
    return html.Div(children, style=base)

def label(text, color=MUTED, size="11px"):
    return html.Div(text, style={"color": color, "fontSize": size,
                                  "fontFamily": FONT, "letterSpacing": "0.06em",
                                  "fontWeight": "600", "textTransform": "uppercase",
                                  "marginBottom": "4px"})

def metric(value, title, color=BLUE):
    return html.Div([
        html.Div(str(value), style={"color": color, "fontSize": "28px",
                                     "fontWeight": "bold", "fontFamily": FONT}),
        html.Div(title, style={"color": TEXT, "fontSize": "10px",
                                "fontFamily": FONT, "marginTop": "2px"}),
    ], style={"textAlign": "center", "padding": "8px"})


def choropleth_fig(col, title, color_scale="Blues"):
    if geojson is None:
        fig = go.Figure()
        fig.add_annotation(text="GeoJSON not found", showarrow=False,
                           font=dict(color=TEXT, size=14))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
        return fig

    sub = master[["nuts2_code", col]].dropna(subset=[col])
    fig = px.choropleth(
        sub, geojson=geojson, locations="nuts2_code",
        featureidkey="properties.NUTS_ID",
        color=col, color_continuous_scale=[[0, PANEL], [0.3, "#1e3a5f"],
                                            [0.6, BLUE], [1.0, CYAN]],
        range_color=[0, 1],
        hover_data={"nuts2_code": True, col: ":.3f"},
    )
    fig.update_geos(scope="europe", showland=True, landcolor=BG,
                    showocean=True, oceancolor=BG,
                    showcoastlines=True, coastlinecolor=BORDER,
                    showcountries=True, countrycolor=BORDER,
                    showframe=False, fitbounds="locations")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=440,
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text=title, font=dict(color=BLUE, size=11), x=0.01),
        coloraxis_colorbar=dict(
            title=dict(text="Index", font=dict(color=TEXT, size=9)),
            tickfont=dict(color=TEXT, size=9),
            outlinecolor=BORDER, tickcolor=TEXT,
        ),
        geo=dict(bgcolor=BG),
    )
    return fig


# ─── Tab 1: Overview ─────────────────────────────────────────────────────────
def build_overview():
    type_cards = []
    for tid, label_str in TYPE_LABELS.items():
        col    = f"suitability_{tid}"
        tier_c = f"tier_{tid}"
        t1 = int((master[tier_c] == "T1_high_index").sum()) if tier_c in master.columns else 0
        t2 = int((master[tier_c] == "T2_candidate").sum()) if tier_c in master.columns else 0
        mean_v = master[col].mean() if col in master.columns else 0
        c = TYPE_COLORS[tid]
        type_cards.append(
            dbc.Col(card([
                html.Div(tid, style={"color": c, "fontSize": "13px",
                                     "fontWeight": "bold", "fontFamily": FONT}),
                html.Div(label_str, style={"color": TEXT, "fontSize": "9px",
                                            "fontFamily": FONT, "marginBottom": "10px",
                                            "lineHeight": "1.2"}),
                dbc.Row([
                    dbc.Col(metric(t1, "Tier 1\n≥0.70", c)),
                    dbc.Col(metric(t2, "Tier 2\n0.60–0.70", DIM)),
                ]),
                html.Div(f"Mean index: {mean_v:.3f}",
                         style={"color": DIM, "fontSize": "9px", "fontFamily": FONT,
                                "marginTop": "6px", "textAlign": "center"}),
            ], style={"borderLeft": f"3px solid {c}"}),
            width=3, style={"padding": "4px"}),
        )

    co_spec_cols = [c for c in master.columns if c.startswith("co_spec_")]
    co_rows = []
    for c in co_spec_cols:
        n = int(master[c].sum()) if master[c].dtype == bool or master[c].dtype == object else int(master[c].astype(bool).sum())
        pair = c.replace("co_spec_", "").replace("_", " + ")
        co_rows.append({"Pair": pair, "n Regions": n})
    co_df = pd.DataFrame(co_rows).sort_values("n Regions", ascending=False)

    # Global Moran's I bar
    if not morans.empty:
        moran_fig = go.Figure(go.Bar(
            x=morans["type_id"], y=morans["moran_i"],
            marker_color=[TYPE_COLORS.get(t, BLUE) for t in morans["type_id"]],
            text=morans["moran_i"].round(3), textposition="outside",
            textfont=dict(color=TEXT, size=9),
        ))
        moran_fig.update_layout(
            template=PLOTLY_TEMPLATE, height=220,
            title=dict(text="Global Moran's I (spatial autocorrelation)", font=dict(color=BLUE, size=10)),
            yaxis=dict(title="Moran's I", range=[0, 1]),
            margin=dict(l=40, r=20, t=36, b=30), showlegend=False,
        )
    else:
        moran_fig = go.Figure()

    return html.Div([
        html.H5("DISRUPTOR ECOSYSTEM SUITABILITY — EU NUTS2 OVERVIEW",
                style={"color": BLUE, "fontFamily": FONT, "fontSize": "14px",
                       "letterSpacing": "0.08em", "marginBottom": "16px"}),
        dbc.Row(type_cards, style={"marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(card([
                label("CO-SPECIALISATION FLAGS (threshold ≥ 0.60)", BLUE, "10px"),
                html.Br(),
                dash_table.DataTable(
                    data=co_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in co_df.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": BORDER, "color": BLUE,
                                   "fontFamily": FONT, "fontSize": "10px"},
                    style_data={"backgroundColor": PANEL, "color": TEXT,
                                 "fontFamily": FONT, "fontSize": "10px",
                                 "border": f"1px solid {BORDER}"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": BG}],
                    page_size=10,
                ),
            ]), width=4),
            dbc.Col(card([
                dcc.Graph(figure=moran_fig, config={"displayModeBar": False}),
            ]), width=8),
        ]),
    ])


# ─── Tab 2: Type Explorer ─────────────────────────────────────────────────────
def build_type_explorer():
    return html.Div([
        html.H5("TYPE EXPLORER", style={"color": BLUE, "fontFamily": FONT,
                                         "fontSize": "14px", "letterSpacing": "0.08em",
                                         "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                label("SELECT DISRUPTOR TYPE", BLUE, "10px"),
                dcc.Dropdown(
                    id="type-selector",
                    options=[{"label": f"{t}: {l}", "value": t}
                             for t, l in TYPE_LABELS.items()],
                    value="T1",
                    clearable=False,
                    style={"backgroundColor": PANEL, "color": TEXT,
                           "fontFamily": FONT, "fontSize": "12px",
                           "border": f"1px solid {BORDER}"},
                ),
            ], width=4),
            dbc.Col([
                label("TIER FILTER", BLUE, "10px"),
                dcc.RadioItems(
                    id="tier-filter",
                    options=[
                        {"label": " All shortlisted (≥0.60)", "value": "all"},
                        {"label": " Tier 1 only (≥0.70)",     "value": "t1"},
                    ],
                    value="all",
                    inline=True,
                    inputStyle={"marginRight": "5px", "accentColor": BLUE},
                    labelStyle={"marginRight": "20px", "color": TEXT,
                                "fontFamily": FONT, "fontSize": "12px",
                                "cursor": "pointer"},
                ),
            ], width=4),
        ], style={"marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="type-choropleth",
                                    config={"displayModeBar": False})]), width=7),
            dbc.Col(card([dcc.Graph(id="type-distribution",
                                    config={"displayModeBar": False})]), width=5),
        ]),
        dbc.Row([
            dbc.Col(card([
                label("SHORTLISTED REGIONS", BLUE, "10px"),
                html.Br(),
                dash_table.DataTable(
                    id="shortlist-table",
                    style_table={"overflowX": "auto", "maxHeight": "320px",
                                  "overflowY": "auto"},
                    style_header={"backgroundColor": BORDER, "color": BLUE,
                                   "fontFamily": FONT, "fontSize": "10px"},
                    style_data={"backgroundColor": PANEL, "color": TEXT,
                                 "fontFamily": FONT, "fontSize": "10px",
                                 "border": f"1px solid {BORDER}"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": BG},
                    ],
                    page_size=15, sort_action="native",
                ),
            ])),
        ]),
    ])


# ─── Tab 3: Spatial ──────────────────────────────────────────────────────────
def build_spatial():
    # LISA summary table
    if not corridors.empty:
        corr_summary = corridors.groupby(["type_id","corridor_id"]).agg(
            n_regions=("nuts2_code","count"),
            n_countries=("n_countries","first"),
            countries=("countries","first"),
            cross_border=("cross_border","first"),
        ).reset_index()
    else:
        corr_summary = pd.DataFrame()

    moran_tbl = dash_table.DataTable(
        data=morans.round(4).to_dict("records") if not morans.empty else [],
        columns=[{"name": c, "id": c} for c in morans.columns] if not morans.empty else [],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": BORDER, "color": BLUE,
                       "fontFamily": FONT, "fontSize": "10px"},
        style_data={"backgroundColor": PANEL, "color": TEXT,
                     "fontFamily": FONT, "fontSize": "10px",
                     "border": f"1px solid {BORDER}"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG},
            {"if": {"filter_query": "{p_sim} < 0.01"}, "color": GREEN},
        ],
        page_size=10,
    )

    corr_tbl = dash_table.DataTable(
        data=corr_summary.to_dict("records") if not corr_summary.empty else [],
        columns=[{"name": c, "id": c} for c in corr_summary.columns] if not corr_summary.empty else [],
        style_table={"overflowX": "auto", "maxHeight": "280px", "overflowY": "auto"},
        style_header={"backgroundColor": BORDER, "color": BLUE,
                       "fontFamily": FONT, "fontSize": "10px"},
        style_data={"backgroundColor": PANEL, "color": TEXT,
                     "fontFamily": FONT, "fontSize": "10px",
                     "border": f"1px solid {BORDER}"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG},
            {"if": {"filter_query": "{cross_border} = True"}, "color": AMBER},
        ],
        page_size=15, sort_action="native",
    )

    return html.Div([
        html.H5("SPATIAL AUTOCORRELATION & CORRIDOR ANALYSIS",
                style={"color": BLUE, "fontFamily": FONT, "fontSize": "14px",
                       "letterSpacing": "0.08em", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(card([
                label("GLOBAL MORAN'S I — ALL 8 TYPES", BLUE, "10px"),
                html.Br(),
                moran_tbl,
            ]), width=6),
            dbc.Col(card([
                label("DISRUPTOR CORRIDORS (Tier-1 connected components ≥ 2 regions)", BLUE, "10px"),
                html.Br(),
                corr_tbl,
            ]), width=6),
        ]),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="spatial-type-select-bar",
                                    config={"displayModeBar": False},
                                    figure=_moran_bar())]), width=12),
        ]),
    ])


def _moran_bar():
    if morans.empty:
        return go.Figure()
    fig = go.Figure()
    for _, row in morans.iterrows():
        fig.add_trace(go.Bar(
            name=row["type_id"],
            x=[row["type_id"]], y=[row["moran_i"]],
            marker_color=TYPE_COLORS.get(row["type_id"], BLUE),
            text=f"I={row['moran_i']:.3f}<br>p={row['p_sim']:.4f}",
            textposition="inside", textfont=dict(size=9),
            hovertemplate=f"{row['type_id']}: I={row['moran_i']:.4f}, p_sim={row['p_sim']:.4f}<extra></extra>",
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=250, showlegend=False,
        title=dict(text="Global Moran's I by disruptor type (all p < 0.001)",
                   font=dict(color=BLUE, size=10)),
        yaxis=dict(title="Moran's I [0,1]", range=[0, 1]),
        margin=dict(l=40, r=20, t=36, b=30),
        barmode="group",
    )
    return fig


# ─── Tab 4: Feasibility ───────────────────────────────────────────────────────
def build_feasibility():
    # I-O heatmap
    tids = list(TYPE_LABELS.keys())
    if not io_mat.empty:
        io_pivot = io_mat.pivot_table(index="from_type", columns="to_type",
                                       values="dependency_strength", fill_value=0)
        io_pivot = io_pivot.reindex(index=tids, columns=tids, fill_value=0)
        z = io_pivot.values
    else:
        z = np.zeros((8, 8))

    colorscale = [[0, PANEL], [0.01, "#1e3a5f"], [0.35, BLUE],
                  [0.67, AMBER], [1.0, RED]]
    io_fig = go.Figure(go.Heatmap(
        z=z, x=tids, y=tids,
        colorscale=colorscale, zmin=0, zmax=3,
        colorbar=dict(
            title=dict(text="Strength", font=dict(color=TEXT, size=9)),
            tickvals=[0,1,2,3],
            ticktext=["0 None","1 Synergy","2 Co-dep","3 Strong"],
            tickfont=dict(color=TEXT, size=8),
            outlinecolor=BORDER,
        ),
        text=[[str(int(v)) if v > 0 else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        hovertemplate="From %{y} → To %{x}: %{z}<extra></extra>",
    ))
    io_fig.update_layout(
        template=PLOTLY_TEMPLATE, height=400,
        title=dict(text="STRUCTURAL DEPENDENCY MATRIX (I-O analogue)  Row → Column",
                   font=dict(color=BLUE, size=10)),
        xaxis=dict(title="Dependency target", tickfont=dict(size=9)),
        yaxis=dict(title="Dependency source", tickfont=dict(size=9), autorange="reversed"),
        margin=dict(l=60, r=20, t=44, b=60),
    )

    # Scenario funnel
    scen_rows = []
    for tid in tids:
        scol = f"scenario_{tid}"
        if scol in master.columns:
            scen_rows.append({
                "type": tid,
                "A": int((master[scol] == "A_near_term").sum()),
                "B": int((master[scol] == "B_medium_term").sum()),
                "C": int((master[scol] == "C_long_term").sum()),
            })
    scen_df = pd.DataFrame(scen_rows) if scen_rows else pd.DataFrame()

    if not scen_df.empty:
        scen_fig = go.Figure()
        scen_fig.add_trace(go.Bar(name="A — Near-term (2025–30)",   x=scen_df["type"], y=scen_df["A"], marker_color=GREEN))
        scen_fig.add_trace(go.Bar(name="B — Medium-term (2030–35)", x=scen_df["type"], y=scen_df["B"], marker_color=AMBER))
        scen_fig.add_trace(go.Bar(name="C — Long-term (2035–40)",   x=scen_df["type"], y=scen_df["C"], marker_color=PURPLE))
        scen_fig.update_layout(
            template=PLOTLY_TEMPLATE, barmode="stack", height=280,
            title=dict(text="DEVELOPMENT SCENARIO CLASSIFICATION — EU NUTS2 regions by type",
                       font=dict(color=BLUE, size=10)),
            legend=dict(font=dict(size=9)),
            margin=dict(l=40, r=20, t=44, b=30),
            yaxis=dict(title="Regions"),
        )
    else:
        scen_fig = go.Figure()

    return html.Div([
        html.H5("FEASIBILITY FRAMEWORK — I-O DEPENDENCIES & SCENARIOS",
                style={"color": BLUE, "fontFamily": FONT, "fontSize": "14px",
                       "letterSpacing": "0.08em", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=io_fig, config={"displayModeBar": False})]), width=6),
            dbc.Col(card([
                dcc.Graph(figure=scen_fig, config={"displayModeBar": False}),
            ]), width=6),
        ]),
        dbc.Row([
            dbc.Col(card([
                label("I-O DEPENDENCY RATIONALE", BLUE, "10px"),
                html.Br(),
                dash_table.DataTable(
                    data=io_mat[io_mat["dependency_strength"] > 0].to_dict("records"),
                    columns=[{"name": c, "id": c} for c in
                             ["from_type","to_type","dependency_strength","rationale"]],
                    style_table={"overflowX": "auto", "maxHeight": "220px", "overflowY": "auto"},
                    style_header={"backgroundColor": BORDER, "color": BLUE,
                                   "fontFamily": FONT, "fontSize": "10px"},
                    style_data={"backgroundColor": PANEL, "color": TEXT,
                                 "fontFamily": FONT, "fontSize": "9.5px",
                                 "border": f"1px solid {BORDER}", "whiteSpace": "normal",
                                 "height": "auto"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": BG},
                        {"if": {"filter_query": "{dependency_strength} = 3"},
                         "color": RED, "fontWeight": "bold"},
                    ],
                    style_cell={"textAlign": "left"},
                    page_size=14,
                ),
            ])),
        ]),
    ])


# ─── Tab 5: Scenario View ─────────────────────────────────────────────────────
def build_scenario_view():
    countries = sorted(master["country_code"].dropna().unique()) \
        if "country_code" in master.columns else []

    return html.Div([
        html.H5("SCENARIO EXPLORER — FILTER & DRILL-DOWN",
                style={"color": BLUE, "fontFamily": FONT, "fontSize": "14px",
                       "letterSpacing": "0.08em", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                label("SCENARIO", BLUE, "10px"),
                dcc.Dropdown(
                    id="scen-scenario",
                    options=[{"label": "A — Near-term (2025–30)", "value": "A_near_term"},
                             {"label": "B — Medium-term (2030–35)", "value": "B_medium_term"},
                             {"label": "C — Long-term (2035–40)", "value": "C_long_term"}],
                    value="A_near_term", clearable=False,
                    style={"backgroundColor": PANEL, "color": TEXT,
                           "fontFamily": FONT, "border": f"1px solid {BORDER}"},
                ),
            ], width=3),
            dbc.Col([
                label("DISRUPTOR TYPE", BLUE, "10px"),
                dcc.Dropdown(
                    id="scen-type",
                    options=[{"label": f"{t}: {l}", "value": t}
                             for t, l in TYPE_LABELS.items()],
                    value="T1", clearable=False,
                    style={"backgroundColor": PANEL, "color": TEXT,
                           "fontFamily": FONT, "border": f"1px solid {BORDER}"},
                ),
            ], width=3),
            dbc.Col([
                label("COUNTRY FILTER", BLUE, "10px"),
                dcc.Dropdown(
                    id="scen-country",
                    options=[{"label": c, "value": c} for c in countries],
                    value=None, clearable=True, multi=True,
                    placeholder="All countries",
                    style={"backgroundColor": PANEL, "color": TEXT,
                           "fontFamily": FONT, "border": f"1px solid {BORDER}"},
                ),
            ], width=3),
        ], style={"marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="scen-choropleth",
                                    config={"displayModeBar": False})]), width=7),
            dbc.Col(card([
                label("FILTERED REGIONS", BLUE, "10px"),
                html.Br(),
                dash_table.DataTable(
                    id="scen-table",
                    style_table={"overflowX": "auto", "maxHeight": "380px",
                                  "overflowY": "auto"},
                    style_header={"backgroundColor": BORDER, "color": BLUE,
                                   "fontFamily": FONT, "fontSize": "10px"},
                    style_data={"backgroundColor": PANEL, "color": TEXT,
                                 "fontFamily": FONT, "fontSize": "10px",
                                 "border": f"1px solid {BORDER}"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": BG}],
                    page_size=20, sort_action="native",
                ),
            ]), width=5),
        ]),
    ])


# ─── App layout ───────────────────────────────────────────────────────────────
GOOGLE_FONT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@300;400;500;600;700&display=swap"
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, GOOGLE_FONT_URL],
    title="EU MegaCampus Siting Intelligence",
    suppress_callback_exceptions=True,
)

app.layout = html.Div([
    # Header
    html.Div([
        html.Span("◈ EU MEGACAMPUS SITING INTELLIGENCE",
                  style={"color": BLUE, "fontFamily": FONT, "fontSize": "16px",
                         "fontWeight": "700", "letterSpacing": "0.12em",
                         "display": "inline-block"}),
        html.Span("  |  242 NUTS2 · 8 Disruptor Types · Strategic Structural Analysis",
                  style={"color": "#a0c4e0", "fontFamily": FONT, "fontSize": "11px",
                         "marginLeft": "8px", "display": "inline-block"}),
    ], className="app-header",
       style={"background": PANEL, "borderBottom": f"1px solid {BORDER}",
              "padding": "14px 24px", "marginBottom": "0px",
              "display": "flex", "alignItems": "center"}),

    # Tabs
    dcc.Tabs(id="tabs", value="overview", style={"background": BG},
             colors={"border": BORDER, "primary": BLUE, "background": PANEL},
             children=[
        dcc.Tab(label="① Overview",      value="overview",
                className="custom-tab", selected_className="custom-tab--selected"),
        dcc.Tab(label="② Type Explorer", value="explorer",
                className="custom-tab", selected_className="custom-tab--selected"),
        dcc.Tab(label="③ Spatial",       value="spatial",
                className="custom-tab", selected_className="custom-tab--selected"),
        dcc.Tab(label="④ Feasibility",   value="feasibility",
                className="custom-tab", selected_className="custom-tab--selected"),
        dcc.Tab(label="⑤ Scenarios",     value="scenarios",
                className="custom-tab", selected_className="custom-tab--selected"),
    ]),

    html.Div(id="tab-content",
             style={"background": BG, "minHeight": "90vh", "padding": "20px 24px"}),

], style={"background": BG, "minHeight": "100vh"})


# ─── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "overview":   return build_overview()
    if tab == "explorer":   return build_type_explorer()
    if tab == "spatial":    return build_spatial()
    if tab == "feasibility": return build_feasibility()
    if tab == "scenarios":  return build_scenario_view()
    return html.Div("Unknown tab")


@app.callback(
    Output("type-choropleth",              "figure"),
    Output("type-distribution",            "figure"),
    Output("shortlist-table",              "data"),
    Output("shortlist-table",              "columns"),
    Output("shortlist-table",              "style_data_conditional"),
    Input("type-selector",                 "value"),
    Input("tier-filter",                   "value"),
)
def update_type_explorer(tid, tier_filter):
    col      = f"suitability_{tid}"
    tier_col = f"tier_{tid}"
    color    = TYPE_COLORS.get(tid, BLUE)

    choro = choropleth_fig(col, f"{tid}: {TYPE_LABELS[tid]} — Suitability Index")

    # Distribution histogram
    vals = master[col].dropna()
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=vals, nbinsx=30, name="All regions",
        marker_color=color, opacity=0.8,
    ))
    hist_fig.add_vline(x=0.70, line_dash="dash", line_color=RED,
                       annotation_text="Tier 1 ≥0.70", annotation_font_color=RED)
    hist_fig.add_vline(x=0.60, line_dash="dot", line_color=AMBER,
                       annotation_text="Tier 2 ≥0.60", annotation_font_color=AMBER)
    hist_fig.update_layout(
        template=PLOTLY_TEMPLATE, height=240,
        title=dict(text=f"Score distribution — {tid}", font=dict(color=color, size=10)),
        xaxis=dict(title="Suitability index", range=[0, 1]),
        yaxis=dict(title="Regions"),
        margin=dict(l=40, r=20, t=36, b=30), showlegend=False,
    )

    # Shortlist table
    if tier_col in master.columns:
        if tier_filter == "t1":
            sub = master[master[tier_col] == "T1_high_index"].copy()
        else:
            sub = master[master[tier_col].isin(["T1_high_index", "T2_candidate"])].copy()
    else:
        sub = master[master[col] >= 0.60].copy()

    sub = sub.sort_values(col, ascending=False)
    display_cols = ["nuts2_code"]
    for c in ["nuts2_name", "country_code", "archetype4_label"]:
        if c in sub.columns:
            display_cols.append(c)
    display_cols += [col, tier_col] if tier_col in sub.columns else [col]
    for sr_c in [f"sr_{tid}"]:
        if sr_c in sub.columns:
            display_cols.append(sr_c)

    display_cols = [c for c in display_cols if c in sub.columns]
    tbl_data = sub[display_cols].round(4).to_dict("records")
    tbl_cols  = [{"name": c, "id": c} for c in display_cols]

    # Dynamic style_data_conditional — use actual tier column name
    tier_style = [
        {"if": {"row_index": "odd"}, "backgroundColor": BG},
        {"if": {"filter_query": f'{{{tier_col}}} = "T1_high_index"'},
         "color": BLUE, "fontWeight": "600"},
    ] if tier_col in sub.columns else [
        {"if": {"row_index": "odd"}, "backgroundColor": BG},
    ]

    return choro, hist_fig, tbl_data, tbl_cols, tier_style


@app.callback(
    Output("scen-choropleth", "figure"),
    Output("scen-table", "data"),
    Output("scen-table", "columns"),
    Input("scen-scenario", "value"),
    Input("scen-type",     "value"),
    Input("scen-country",  "value"),
)
def update_scenario_view(scenario, tid, countries):
    scen_col  = f"scenario_{tid}"
    score_col = f"suitability_{tid}"
    color     = TYPE_COLORS.get(tid, BLUE)

    if scen_col in master.columns:
        filtered = master[master[scen_col] == scenario].copy()
    else:
        filtered = master[master[score_col] >= 0.60].copy()

    if countries:
        if "country_code" in filtered.columns:
            filtered = filtered[filtered["country_code"].isin(countries)]

    # Choropleth — highlight filtered regions
    if geojson is None:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
    else:
        full = master[["nuts2_code", score_col]].copy()
        full["highlight"] = full["nuts2_code"].isin(filtered["nuts2_code"]).astype(float)
        fig = px.choropleth(
            full, geojson=geojson, locations="nuts2_code",
            featureidkey="properties.NUTS_ID",
            color="highlight",
            color_continuous_scale=[[0, PANEL], [0.5, "#1e3a5f"], [1.0, color]],
            range_color=[0, 1],
            hover_data={"nuts2_code": True, score_col: ":.3f"} if score_col in full.columns else {},
        )
        fig.update_geos(scope="europe", showland=True, landcolor=BG,
                        showocean=True, oceancolor=BG,
                        showcoastlines=True, coastlinecolor=BORDER,
                        showcountries=True, countrycolor=BORDER,
                        showframe=False, fitbounds="locations")
        scen_labels = {"A_near_term": "A — Near-term", "B_medium_term": "B — Medium-term",
                       "C_long_term": "C — Long-term"}
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=420,
            margin=dict(l=0, r=0, t=28, b=0),
            title=dict(text=f"{scen_labels.get(scenario, scenario)} · {tid}: {TYPE_LABELS[tid]}  "
                            f"({len(filtered)} regions)",
                       font=dict(color=color, size=11), x=0.01),
            coloraxis_showscale=False, geo=dict(bgcolor=BG),
        )

    display_cols = ["nuts2_code"]
    for c in ["nuts2_name", "country_code", "archetype4_label"]:
        if c in filtered.columns:
            display_cols.append(c)
    if score_col in filtered.columns:
        display_cols.append(score_col)
    for extra in [f"tier_{tid}", f"sr_{tid}", f"uncertainty_{tid}"]:
        if extra in filtered.columns:
            display_cols.append(extra)

    display_cols = [c for c in display_cols if c in filtered.columns]
    tbl_data = filtered.sort_values(score_col, ascending=False)[display_cols].round(4).to_dict("records") \
        if score_col in filtered.columns else filtered[display_cols].to_dict("records")
    tbl_cols = [{"name": c, "id": c} for c in display_cols]

    return fig, tbl_data, tbl_cols


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EU MegaCampus Siting Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    print(f"\n  ◈ EU MegaCampus Siting Intelligence Dashboard")
    print(f"  Running at http://{args.host}:{args.port}/")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=False)
