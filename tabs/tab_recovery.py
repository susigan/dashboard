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
        st.warning("Sem dados.")
        return

    rec = calcular_recovery(dw)
    u = rec.iloc[-1]

    st.metric("Recovery Score", f"{u['recovery_score']:.0f}")

    modo_modelo = st.radio(
        "Modelo",
        ["Mode 1 — Altini", "Mode 2 — Plews"],
        horizontal=True
    )

    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df['LnrMSSD'] = np.log(df['hrv'])

    janela_cv = 7
    baseline = 7 if "Mode 1" in modo_modelo else 60

    df['baseline'] = df['LnrMSSD'].rolling(baseline).mean()
    df['std'] = df['LnrMSSD'].rolling(baseline).std()

    df['cv'] = (df['LnrMSSD'].rolling(janela_cv).std() /
                df['LnrMSSD'].rolling(janela_cv).mean()) * 100

    df['SWC'] = 0.5 * (df['std'] / df['baseline'] * 100)

    df['upper'] = df['baseline'] * (1 + df['SWC']/100)
    df['lower'] = df['baseline'] * (1 - df['SWC']/100)

    cv_mean = df['cv'].mean()
    cv_std = df['cv'].std()

    cv_low = cv_mean - 0.5 * cv_std
    cv_high = cv_mean + 0.5 * cv_std

    def slope(x):
        if len(x) < 5:
            return np.nan
        return stats.linregress(range(len(x)), x)[0]

    df['slope'] = df['LnrMSSD'].rolling(7).apply(slope)

    def altini(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']):
            return 'Sem dados', 'gray'

        if r['LnrMSSD'] < r['baseline'] and r['cv'] < cv_low:
            return 'Fatigue', 'red'
        if r['LnrMSSD'] < r['baseline'] and r['cv'] > cv_high:
            return 'Maladaptation', 'yellow'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] < cv_low:
            return 'Adaptation', 'green'
        return 'Normal', 'blue'

    def plews(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']):
            return 'Sem dados', 'gray'

        if r['cv'] < cv_low and r['slope'] < -0.01:
            return 'NFOR', 'darkred'
        if r['LnrMSSD'] < r['lower']:
            return 'Overreaching', 'orange'
        return 'Normal', 'green'

    if "Mode 1" in modo_modelo:
        df[['zona', 'cor']] = df.apply(lambda r: pd.Series(altini(r)), axis=1)
    else:
        df[['zona', 'cor']] = df.apply(lambda r: pd.Series(plews(r)), axis=1)

    df = df.dropna()

    fig = go.Figure()

    for z in df['zona'].unique():
        d = df[df['zona'] == z]
        fig.add_trace(go.Scatter(
            x=d['Data'],
            y=d['LnrMSSD'],
            mode='markers',
            name=z,
            marker=dict(color=d['cor'])
        ))

    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df['baseline'],
        name='baseline'
    ))

    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df['upper'],
        name='upper'
    ))

    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df['lower'],
        name='lower',
        fill='tonexty'
    ))

    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df['cv'],
        name='CV',
        yaxis='y2'
    ))

    fig.update_layout(
        yaxis=dict(title='LnRMSSD'),
        yaxis2=dict(overlaying='y', side='right')
    )

    st.plotly_chart(fig)

    st.markdown("### Status Atual")
    last = df.iloc[-1]

    st.metric("Zona", last['zona'])
    st.metric("CV%", f"{last['cv']:.2f}")
    st.metric("Slope", f"{last['slope']:.4f}")
