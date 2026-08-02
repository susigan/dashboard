"""
tab_correlacoes.py — CORRIGIDO
Removes duplicate variables + Fix complex() error
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from utils.config import *
    from utils.helpers import *
    from utils.data import *
except:
    pass

def tab_correlacoes(da, dw):
    """Análise de correlações — com deduplicação de variáveis"""
    
    st.header("🧠 Correlações & Impacto")
    st.caption("Análise sobre todo o histórico disponível")
    
    if len(da) == 0 or len(dw) == 0:
        st.warning("Sem dados suficientes.")
        return

    # ════════════════════════════════════════════════════════════════════════════════
    # DEDUPLICAR VARIÁVEIS (remover duplicatas como HRV/hrv, RHR/rhr, etc)
    # ════════════════════════════════════════════════════════════════════════════════
    
    dw_copy = dw.copy()
    dw_copy['Data'] = pd.to_datetime(dw_copy['Data'], errors='coerce')
    
    # Mapa de variáveis duplicadas → manter a CANÓNICA
    dedup_map = {
        'hrv': 'HRV',           # Manter HRV (maiúscula)
        'rhr': 'RHR',           # Manter RHR
        'fat': 'FAT',           # Manter FAT
        'peso': 'Peso',         # Manter Peso
        'Horas de Sono': 'Sono Qualidade',  # Mapear
        'sleep_quality': 'Sono Qualidade',
        'Stress Do dia': 'Stress',
        'stress': 'Stress',
        'fatiga': 'Cansaço Muscular Geral',
        'humor': 'Humor',
        'hf_power': 'HF Power',
        'soreness': 'Cansaço Muscular Geral'
    }
    
    # Aplicar mapa
    for col_old, col_new in dedup_map.items():
        if col_old in dw_copy.columns and col_old != col_new:
            # Se a coluna nova não existe, renomear
            if col_new not in dw_copy.columns:
                dw_copy = dw_copy.rename(columns={col_old: col_new})
            # Se ambas existem, remover a antiga (mantemos a nova)
            else:
                dw_copy = dw_copy.drop(columns=[col_old])
    
    # Colunas numéricas APÓS deduplicação
    numeric_cols = dw_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    st.info(f"📊 Registos: {len(dw_copy)} | Variáveis únicas: {len(numeric_cols)}")
    st.write(f"Variáveis: {numeric_cols}")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # MATRIZ DE CORRELAÇÃO (SEM duplicatas)
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.subheader("📊 Matriz de Correlação")
    
    try:
        df_corr = dw_copy[numeric_cols].copy()
        
        # Garantir que TUDO é numeric (fix para error complex())
        for col in df_corr.columns:
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
        
        # Remove linhas completamente vazias
        df_corr = df_corr.dropna(how='all')
        
        st.success(f"✅ Dados para correlação: {df_corr.shape[0]} registos")
        
        # Calcular correlação
        corr_matrix = df_corr.corr(method='pearson')
        
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Correlação")
        ))
        fig.update_layout(
            title="Matriz de Correlação (sem duplicatas)",
            height=800,
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Erro matriz: {e}")
        st.write(f"Debug: {df_corr.dtypes}")
        return
    
    # ════════════════════════════════════════════════════════════════════════════════
    # CORRELAÇÕES SIGNIFICATIVAS (SEM autocorrelação)
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.subheader("🔗 Correlações Significativas (|r| > 0.3, excluindo diagonal)")
    
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            
            # Garantir que r é um número (não Series)
            if isinstance(r, (float, int, np.number)):
                if abs(r) > 0.3:
                    correlations.append({
                        'Variável 1': corr_matrix.columns[i],
                        'Variável 2': corr_matrix.columns[j],
                        'Correlação': f"{r:.3f}",
                        'Tipo': 'Positiva ✅' if r > 0 else 'Negativa ⚠️'
                    })
    
    if correlations:
        df_sig = pd.DataFrame(correlations)
        st.dataframe(df_sig, use_container_width=True)
        st.success(f"✅ {len(correlations)} correlações significativas encontradas")
    else:
        st.info("ℹ️ Sem correlações fortes")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # ESTATÍSTICAS
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.subheader("📈 Estatísticas Básicas")
    
    stats_df = pd.DataFrame({
        'Variável': df_corr.columns,
        'Média': df_corr.mean().values,
        'Std': df_corr.std().values,
        'Min': df_corr.min().values,
        'Max': df_corr.max().values,
        'Missing': df_corr.isnull().sum().values
    })
    
    st.dataframe(stats_df, use_container_width=True)
    
    # ════════════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("💾 Exportar")
    
    csv_corr = corr_matrix.to_csv()
    st.download_button(
        label="📥 Download Matriz de Correlação (CSV)",
        data=csv_corr,
        file_name=f"correlacoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    csv_stats = stats_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Estatísticas (CSV)",
        data=csv_stats,
        file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
