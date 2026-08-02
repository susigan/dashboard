"""
tab_correlacoes.py — VERSÃO CORRIGIDA v2
Sem 'import *' dentro de funções
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports utils — com try/except para evitar erro
try:
    from utils.config import CORES, CORES_ATIV, TYPE_MAP, VALID_TYPES
except:
    pass

try:
    from utils.data import preproc_wellness, preproc_ativ
except:
    pass

def tab_correlacoes(da, dw):
    """Análise de correlações & impacto"""
    
    st.header("🧠 Correlações & Impacto")
    st.caption("Análise sobre todo o histórico disponível — independente do filtro de período do sidebar.")
    
    if len(da) == 0 or len(dw) == 0:
        st.warning("Sem dados suficientes.")
        return

    rpe_col   = next((c for c in ['rpe','RPE','icu_rpe'] if c in da.columns), None)
    CICLICOS_T = ['Bike','Row','Run','Ski']
    CORES_T  = {'Bike':'#e74c3c','Row':'#2980b9','Ski':'#8e44ad',
                'Run':'#27ae60','WeightTraining':'#e67e22','Rest':'#7f8c8d'}
    CORES_CAT = {'Leve':'#27ae60','Moderado':'#e67e22','Pesado':'#c0392b','Rest':'#7f8c8d'}
    LAYOUT_BASE = dict(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='#111111', size=13),
        margin=dict(l=45, r=20, t=50, b=50))

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _remove_outliers_iqr(series, factor=1.5):
        """Remove outliers IQR 1.5x — retorna série com NaN nos extremos."""
        s = pd.to_numeric(series, errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s < q1 - factor*iqr) | (s > q3 + factor*iqr)
        s[mask] = np.nan
        return s

    def _prep_dw_clean(dw_in, data_min='2020-01-01'):
        """Wellness limpo: filtro 2020+, outliers IQR removidos."""
        d = dw_in.copy()
        d['Data'] = pd.to_datetime(d['Data']).dt.normalize()
        d = d[d['Data'] >= pd.Timestamp(data_min)]
        if 'hrv' in d.columns:
            d['hrv'] = _remove_outliers_iqr(d['hrv'])
        if 'rhr' in d.columns:
            d['rhr'] = _remove_outliers_iqr(d['rhr'])
        return d.dropna(subset=['hrv'])

    # ════════════════════════════════════════════════════════════════════════════════
    # ANÁLISE PRINCIPAL
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.subheader("📊 Matriz de Correlação")
    
    # Preparar dados
    dw_clean = _prep_dw_clean(dw)
    
    if len(dw_clean) < 10:
        st.warning("⚠️ Dados insuficientes para correlação")
        return
    
    # Seleccionar colunas numéricas
    numeric_cols = dw_clean.select_dtypes(include=[np.number]).columns
    df_corr = dw_clean[numeric_cols].dropna()
    
    if len(df_corr) < 5:
        st.warning("⚠️ Dados insuficientes")
        return
    
    # Calcular correlação
    corr_matrix = df_corr.corr()
    
    # Plotar heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlação")
    ))
    fig.update_layout(
        title="Matriz de Correlação",
        height=600,
        **LAYOUT_BASE
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar correlações fortes
    st.subheader("🔗 Correlações Significativas (|r| > 0.3)")
    
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                correlations.append({
                    'Variável 1': corr_matrix.columns[i],
                    'Variável 2': corr_matrix.columns[j],
                    'Correlação': f"{r:.3f}"
                })
    
    if correlations:
        df_sig = pd.DataFrame(correlations)
        st.dataframe(df_sig, use_container_width=True)
    else:
        st.info("ℹ️ Sem correlações fortes detectadas")
    
    # Export
    st.markdown("---")
    st.subheader("💾 Exportar Correlações")
    
    csv = corr_matrix.to_csv()
    st.download_button(
        label="📥 Download Matriz CSV",
        data=csv,
        file_name=f"correlacoes_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
