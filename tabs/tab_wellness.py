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
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def tab_wellness(dw):
    st.header("🧘 Wellness")
    if len(dw) == 0: st.warning("Sem dados de wellness."); return
    mets = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness'] if m in dw.columns and dw[m].notna().any()]
    if not mets: st.warning("Sem métricas wellness."); return
    sel = st.multiselect("Métricas", mets, default=mets[:5])
    if not sel: return
    _fig_gen = go.Figure()
    _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
        legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
    st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="wellness_chart")
    # TODO: chart content (converted from matplotlib)
    st.subheader("📋 Resumo (últimos 7 dias)")
    if len(dw) >= 7:
        u7 = dw.tail(7); rows = []
        for m in mets:
            col = pd.to_numeric(u7[m], errors='coerce')
            rows.append({'Métrica': m.replace('_', ' ').title(), 'Média': f"{col.mean():.1f}", 'Mín': f"{col.min():.0f}", 'Máx': f"{col.max():.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
