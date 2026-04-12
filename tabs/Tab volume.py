from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re as _re
import warnings
warnings.filterwarnings('ignore')

def tab_volume(da, dw):
    st.header("📦 Volume & Carga")
    if len(da) == 0: st.warning("Sem dados de atividades."); return
    df = filtrar_principais(da).copy()
    df = add_tempo(df); df['Data'] = pd.to_datetime(df['Data'])
    df['horas'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 3600).fillna(0)
    ciclicos = ['Bike', 'Run', 'Row', 'Ski']
    CORES_MOD = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo'], 'WeightTraining': CORES['laranja']}

    st.subheader("🚴 Volume Mensal — Atividades Cíclicas (horas)")
    df_cic = df[df['type'].isin(ciclicos)].copy()
    if len(df_cic) > 0:
        pivot = df_cic.pivot_table(index='mes', columns='type', values='horas', aggfunc='sum', fill_value=0).sort_index()
        _fig_sb = go.Figure()
        if 'pivot' in dir() and len(pivot) > 0:
            for _tc in [c for c in pivot.columns if c in CORES_MOD]:
                _fig_sb.add_trace(go.Bar(x=[str(x) for x in pivot.index],
                    y=pivot[_tc].tolist(), name=_tc,
                    marker_color=CORES_MOD.get(_tc,'gray'),
                    marker_line_width=0, opacity=0.85))
        _fig_sb.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), barmode='stack', height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_sb, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
        c1, c2 = st.columns(2)
        c1.metric("Total horas cíclicos", f"{pivot.values.sum():.1f}h")
        c2.metric("Média mensal", f"{media:.1f}h")

    st.subheader("🏋️ Volume Mensal — WeightTraining (horas)")
    df_wt = da[da['type'] == 'WeightTraining'].copy()
    if len(df_wt) > 0:
        df_wt = add_tempo(df_wt); df_wt['horas'] = (pd.to_numeric(df_wt['moving_time'], errors='coerce') / 3600).fillna(0)
        mensal = df_wt.groupby('mes').agg(horas=('horas', 'sum'), sessoes=('Data', 'count')).reset_index().sort_values('mes')
        _fig_gen = go.Figure()
        _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
        # TODO: chart content (converted from matplotlib)
    else:
        st.info("Sem sessões de WeightTraining no período.")

    st.subheader("💥 Strain Score (XSS)")
    xss_col = next((c for c in ['xss', 'SS', 'XSS'] if c in df.columns and df[c].notna().any()), None)
    if xss_col:
        df_xss = df[df['type'].isin(ciclicos)].dropna(subset=[xss_col]).copy()
        if len(df_xss) > 3:
            df_xss = df_xss.sort_values('Data'); df_xss['xss_s'] = pd.to_numeric(df_xss[xss_col], errors='coerce').rolling(7, min_periods=1).mean()
            _fig_sb = go.Figure()
            if 'pivot' in dir() and len(pivot) > 0:
                for _tc in [c for c in pivot.columns if c in CORES_MOD]:
                    _fig_sb.add_trace(go.Bar(x=[str(x) for x in pivot.index],
                        y=pivot[_tc].tolist(), name=_tc,
                        marker_color=CORES_MOD.get(_tc,'gray'),
                        marker_line_width=0, opacity=0.85))
            _fig_sb.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), barmode='stack', height=340,
                legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'), showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
            st.plotly_chart(_fig_sb, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    st.subheader("📊 Volume de Horas por Intensidade (Trimestral)")
    if 'rpe' in df.columns and 'moving_time' in df.columns:
        df_rpe = df[df['type'].isin(ciclicos)].copy()
        df_rpe['rpe_cat'] = df_rpe['rpe'].apply(classificar_rpe); df_rpe = df_rpe.dropna(subset=['rpe_cat'])
        if len(df_rpe) > 0:
            piv = df_rpe.pivot_table(index='trimestre', columns='rpe_cat', values='horas', aggfunc='sum', fill_value=0).sort_index()
            CORES_RPE = {'Leve': CORES['verde'], 'Moderado': CORES['laranja'], 'Pesado': CORES['vermelho']}
            _fig_gen = go.Figure()
            _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
                legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
                xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
            st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
            # TODO: chart content (converted from matplotlib)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — eFTP
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_eftp.py
# ════════════════════════════════════════════════════════════════════════════
