"""Shared styling and chart helpers for both app pages."""
import plotly.graph_objects as go
import streamlit as st

# Colourblind-safe pair (validated): blue = healthy / lowers risk, orange = disease / raises risk
GOOD = "#2a78d6"
BAD = "#eb6834"
# Status colours for risk bands (always shown with a text label)
STATUS = {"low": "#1baf7a", "medium": "#eda100", "high": "#e34948"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.5rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #e11d48 0%, #9f1239 60%, #4c0519 100%);
    border-radius: 20px; padding: 2.2rem 2.5rem; color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(225, 29, 72, .25);
}
.hero.lifestyle {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 55%, #1e1b4b 100%);
    box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
}
.hero h1 { color: white; font-size: 2.4rem; font-weight: 800; margin: 0; }
.hero p { color: rgba(255,255,255,.85); font-size: 1.05rem; margin: .4rem 0 0 0; }
.hero .badges span {
    display: inline-block; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; padding: .25rem .8rem; margin: .9rem .4rem 0 0; font-size: .85rem;
}

.section-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .2rem; }
.section-sub { opacity: .65; font-size: .85rem; margin-bottom: .8rem; }

.result { border-radius: 16px; padding: 1.4rem; text-align: center; color: white; }
.result h2 { color: white; margin: 0; font-size: 1.8rem; font-weight: 800; }
.result p { margin: .3rem 0 0 0; opacity: .92; }
.low    { background: linear-gradient(135deg, #10b981, #047857); }
.medium { background: linear-gradient(135deg, #f59e0b, #b45309); }
.high   { background: linear-gradient(135deg, #ef4444, #991b1b); }

.tip { border-left: 4px solid; padding: .6rem .9rem; border-radius: 8px; margin-bottom: .5rem;
       background: var(--secondary-background-color); }
.tip.bad  { border-color: #eb6834; }
.tip.good { border-color: #2a78d6; }

.stat { border-radius: 14px; padding: 1rem 1.2rem; background: var(--secondary-background-color); }
.stat .label { font-size: .85rem; opacity: .7; }
.stat .value { font-size: 1.9rem; font-weight: 800; }

div.stButton > button {
    background: linear-gradient(135deg, #e11d48, #9f1239); color: white; border: 0;
    font-weight: 700; padding: .75rem; border-radius: 12px; font-size: 1.05rem;
}
div.stButton > button:hover { filter: brightness(1.1); color: white; }
.footer { text-align: center; opacity: .6; font-size: .85rem; margin-top: 2rem; }
</style>
"""


def hero(title, subtitle, badges, variant=""):
    spans = "".join(f"<span>{b}</span>" for b in badges)
    st.markdown(f'<div class="hero {variant}"><h1>{title}</h1><p>{subtitle}</p>'
                f'<div class="badges">{spans}</div></div>', unsafe_allow_html=True)


def card_header(title, subtitle):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def result_box(level, css, msg):
    st.markdown(f'<div class="result {css}"><h2>{level}</h2><p>{msg}</p></div>', unsafe_allow_html=True)


def tips(items, kind):
    for text in items:
        st.markdown(f'<div class="tip {kind}">{text}</div>', unsafe_allow_html=True)


def style(fig, height=380, **layout):
    """Recessive grid/axes, transparent background, consistent sizing."""
    layout = {"margin": dict(t=30, b=40, l=10, r=10), **layout}
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_size=13), legend=dict(orientation="h", y=1.08, x=0), **layout)
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def gauge(value_pct, axis_max, bands, title="Estimated probability"):
    """bands: list of (start, end, status_key)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=min(value_pct, axis_max),
        number={"suffix": "%", "font": {"size": 46}, "valueformat": ".1f"},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, axis_max]},
               "bar": {"color": "#1f2937", "thickness": 0.25},
               "steps": [{"range": [a, b], "color": STATUS[k]} for a, b, k in bands]}))
    if value_pct > axis_max:
        fig.data[0].number.prefix = ">"
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def impact_bar(series, labels, height=420, xaxis_title="← lowers risk     raises risk →", suffix=""):
    """Horizontal diverging bar: blue lowers risk, orange raises it."""
    fig = go.Figure(go.Bar(
        x=series.values, y=[labels[i] for i in series.index], orientation="h",
        marker=dict(color=[BAD if v > 0 else GOOD for v in series.values], cornerradius=4),
        hovertemplate="%{y}: %{x:+.2f}" + suffix + "<extra></extra>"))
    return style(fig, height=height, xaxis_title=xaxis_title, bargap=0.25)


def rate_bar(x, y_pct, height=360, ytitle="% with heart disease", highlight=None):
    colors = [BAD if highlight is not None and str(k) == str(highlight) else GOOD for k in x]
    fig = go.Figure(go.Bar(
        x=[str(k) for k in x], y=y_pct, marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.1f}%" for v in y_pct], textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    return style(fig, height=height, yaxis_title=ytitle, yaxis_range=[0, max(y_pct) * 1.2], bargap=0.3)
