from utils.config import *
from utils.helpers import *
from utils.data import *

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
import sys, os as _os
from scipy import stats

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
warnings.filterwarnings('ignore')


def tab_recovery(dw):

    st.header("🔋 Recovery Score & HRV Analysis")

    if len(dw) == 0 or 'hrv' not in dw.columns:
        st.warning("Sem dados de HRV.")
        return

    rec = calcular_recovery(dw)
    if len(rec) == 0:
        return

    u = rec.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{u['recovery_score']:.0f}")
    c2.metric("HRV", f"{u['hrv']:.0f}" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline", f"{u['hrv_baseline']:.0f}" if pd.notna(u['hrv_baseline']) else "—")

    _cv7 = u.get('hrv_cv_7d', None)
    c4.metric("CV%", f"{_cv7:.1f}%" if _cv7 is not None else "—")

    st.markdown("---")

    col1, col2 = st.columns(2)
    n_dias = col1.slider("Dias", 14, min(len(dw), 365), 90)
    janela_cv = col2.slider("Janela CV", 3, 14, 7)

    modo_modelo = st.radio(
        "Modelo",
        ["Mode 1 — Altini", "Mode 2 — Plews"],
        horizontal=True
    )

    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.tail(n_dias)

    df['LnrMSSD'] = np.where(df['hrv'] > 0, np.log(df['hrv']), np.nan)
    df = df.dropna(subset=['LnrMSSD'])

    if len(df) < 10:
        st.warning("Poucos dados.")
        return

    # ✅ CORREÇÃO AQUI
    baseline = 7 if "Mode 1" in modo_modelo else 60

    df['baseline'] = df['LnrMSSD'].rolling(baseline, min_periods=5).mean()
    df['std'] = df['LnrMSSD'].rolling(baseline, min_periods=5).std()

    df['cv'] = (
        df['LnrMSSD'].rolling(janela_cv, min_periods=3).std() /
        df['LnrMSSD'].rolling(janela_cv, min_periods=3).mean()
    ) * 100

    df['SWC'] = 0.5 * (df['std'] / df['baseline'] * 100)

    df['upper'] = df['baseline'] * (1 + df['SWC']/100)
    df['lower'] = df['baseline'] * (1 - df['SWC']/100)

    cv_hist = df['cv'].dropna()

    if len(cv_hist) > 10:
        cv_mean = cv_hist.mean()
        cv_std = cv_hist.std()
        cv_low = cv_mean - 0.5 * cv_std
        cv_high = cv_mean + 0.5 * cv_std
    else:
        cv_low, cv_high = 0.5, 1.5

    def slope(x):
        if len(x.dropna()) < 5:
            return np.nan
        return stats.linregress(range(len(x.dropna())), x.dropna())[0]

    df['slope'] = df['LnrMSSD'].rolling(7, min_periods=5).apply(slope)

    def altini(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']):
            return 'Sem dados', '#808080'

        if r['LnrMSSD'] < r['baseline'] and r['cv'] < cv_low:
            return 'Accumulated Fatigue', '#e74c3c'
        if r['LnrMSSD'] < r['baseline'] and r['cv'] > cv_high:
            return 'Maladaptation', '#f1c40f'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] < cv_low:
            return 'Good Adaptation', '#27ae60'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] > cv_high:
            return 'High Variability', '#2c3e50'

        return 'Normal', '#95a5a6'

    def plews(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']):
            return 'Sem dados', '#808080'

        declinio = r['slope'] < -0.01 if pd.notna(r['slope']) else False

        if r['cv'] < cv_low and declinio:
            return 'NFOR', '#8b0000'
        if r['LnrMSSD'] < r['lower']:
            return 'Overreaching', '#e67e22'
        if r['cv'] > cv_high:
            return 'High Variability', '#2c3e50'

        return 'Normal', '#27ae60'

    if "Mode 1" in modo_modelo:
        df[['zona', 'cor']] = df.apply(lambda r: pd.Series(altini(r)), axis=1)
    else:
        df[['zona', 'cor']] = df.apply(lambda r: pd.Series(plews(r)), axis=1)

    df_plot = df.dropna(subset=['baseline', 'cv'])

    if len(df_plot) == 0:
        st.warning("Sem dados suficientes após processamento.")
        return

    fig = go.Figure()

    zonas = df_plot[['zona', 'cor']].drop_duplicates()

    for _, z in zonas.iterrows():
        d = df_plot[df_plot['zona'] == z['zona']]

        fig.add_trace(go.Scatter(
            x=d['Data'],
            y=d['LnrMSSD'],
            mode='markers',
            name=z['zona'],
            marker=dict(color=z['cor'], size=10),
            customdata=d['cv'],
            text=d['slope'],
            hovertemplate='%{x}<br>LnRMSSD: %{y:.3f}<br>CV: %{customdata:.2f}%<br>Slope: %{text:.4f}'
        ))

    fig.add_trace(go.Scatter(
        x=df_plot['Data'],
        y=df_plot['baseline'],
        name='Baseline',
        line=dict(dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df_plot['Data'],
        y=df_plot['upper'],
        line=dict(width=1),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_plot['Data'],
        y=df_plot['lower'],
        fill='tonexty',
        name='SWC'
    ))

    fig.add_trace(go.Scatter(
        x=df_plot['Data'],
        y=df_plot['cv'],
        name='CV%',
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=[df_plot['Data'].iloc[0], df_plot['Data'].iloc[-1]],
        y=[cv_low, cv_low],
        name='CV baixo',
        yaxis='y2',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        height=500,
        yaxis=dict(title='LnRMSSD'),
        yaxis2=dict(overlaying='y', side='right')
    )

    st.plotly_chart(fig, use_container_width=True)

    ultimo = df_plot.iloc[-1]

    st.markdown("### Status Atual")

    c1, c2, c3 = st.columns(3)
    c1.metric("Zona", ultimo['zona'])
    c2.metric("CV%", f"{ultimo['cv']:.2f}")
    c3.metric("Slope", f"{ultimo['slope']:.4f}")
