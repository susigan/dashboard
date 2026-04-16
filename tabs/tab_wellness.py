from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

MC = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}

_CM = {
    'hrv':           '#2ecc71',
    'rhr':           '#e74c3c',
    'sleep_quality': '#9b59b6',
    'fatiga':        '#f39c12',
    'stress':        '#c0392b',
    'humor':         '#1d8348',
    'soreness':      '#3498db',
}

_LABELS = {
    'hrv':           'HRV (ms)',
    'rhr':           'RHR (bpm)',
    'sleep_quality': 'Sono Qualidade',
    'fatiga':        'Energia/Vontade',
    'stress':        'Stress',
    'humor':         'Humor',
    'soreness':      'Cansaço Muscular',
}


def tab_wellness(dw):
    st.header("🧘 Wellness")

    if len(dw) == 0:
        st.warning("Sem dados de wellness.")
        return

    mets = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness']
            if m in dw.columns and dw[m].notna().any()]
    if not mets:
        st.warning("Sem métricas wellness disponíveis.")
        return

    # ── Controlos ────────────────────────────────────────────────────────────
    sel = st.multiselect("Métricas a visualizar", mets,
                          default=mets[:5], key="well_sel")
    if not sel:
        st.info("Selecciona pelo menos uma métrica.")
        return

    n_dias = st.slider("Dias a mostrar", 14, min(len(dw), 365),
                        min(90, len(dw)), key="well_dias")

    dw_plot = dw.copy().sort_values('Data')
    dw_plot['Data'] = pd.to_datetime(dw_plot['Data'])
    dw_plot = dw_plot.tail(n_dias)

    # ── Gráfico: subplots empilhados, 1 por métrica ───────────────────────────
    n = len(sel)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[_LABELS.get(m, m.replace('_', ' ').title()) for m in sel],
        vertical_spacing=0.04,
    )

    for row_i, met in enumerate(sel, 1):
        v = pd.to_numeric(dw_plot[met], errors='coerce')
        cor = _CM.get(met, '#3498db')
        dates = dw_plot['Data'].tolist()

        # Raw line + scatter
        fig.add_trace(go.Scatter(
            x=dates, y=v.tolist(),
            mode='lines+markers',
            name=_LABELS.get(met, met),
            line=dict(color=cor, width=2),
            marker=dict(size=4, color=cor),
            showlegend=(row_i == 1),
            hovertemplate='%{x|%d/%m/%Y}: <b>%{y:.1f}</b><extra></extra>',
        ), row=row_i, col=1)

        # Rolling 7d mean
        v_roll = v.rolling(7, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=dates, y=v_roll.tolist(),
            mode='lines',
            name='Média 7d',
            line=dict(color='#2c3e50', width=1.5, dash='dash'),
            opacity=0.55,
            showlegend=(row_i == 1),
            hovertemplate='7d: <b>%{y:.1f}</b><extra></extra>',
        ), row=row_i, col=1)

        # Y-axis label per subplot
        fig.update_yaxes(
            title_text=_LABELS.get(met, met),
            title_font=dict(size=9, color='#555'),
            tickfont=dict(color='#111', size=9),
            showgrid=True, gridcolor='#eee',
            row=row_i, col=1,
        )

    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='#111', size=10),
        height=max(280, n * 160),
        margin=dict(t=40, b=80, l=65, r=20),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.12,
                    font=dict(color='#111', size=10)),
        title=dict(text='Métricas Wellness', font=dict(size=13, color='#111')),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#eee',
        tickfont=dict(color='#111', size=9),
        tickangle=-45,
        row=n, col=1,
    )

    st.plotly_chart(fig, use_container_width=True, config=MC, key="well_chart")

    # ── Resumo 7 dias ─────────────────────────────────────────────────────────
    st.subheader("📋 Resumo — últimos 7 dias")
    if len(dw) >= 7:
        u7 = dw.sort_values('Data').tail(7)
        rows_tbl = []
        for m in mets:
            col_v = pd.to_numeric(u7[m], errors='coerce')
            if col_v.notna().sum() == 0:
                continue
            rows_tbl.append({
                'Métrica': _LABELS.get(m, m.replace('_', ' ').title()),
                'Média':   f"{col_v.mean():.1f}",
                'Mín':     f"{col_v.min():.0f}",
                'Máx':     f"{col_v.max():.0f}",
                'Registos': f"{col_v.notna().sum()}/7",
            })
        if rows_tbl:
            st.dataframe(pd.DataFrame(rows_tbl),
                         use_container_width=True, hide_index=True)
