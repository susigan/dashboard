"""
tab_correlacoes.py — VERSÃO v3
Filtragem mais permissiva para usar TODOS os dados disponíveis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
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

    # ════════════════════════════════════════════════════════════════════════════════
    # PREPARAR DADOS — SEM FILTROS RIGOROSOS
    # ════════════════════════════════════════════════════════════════════════════════
    
    dw_copy = dw.copy()
    
    # Converter Data para datetime
    if 'Data' in dw_copy.columns:
        dw_copy['Data'] = pd.to_datetime(dw_copy['Data'], errors='coerce')
    
    st.info(f"📊 Registos disponíveis: {len(dw_copy)}")
    
    # Seleccionar colunas numéricas (TODOS os dados)
    numeric_cols = dw_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("❌ Nenhuma coluna numérica encontrada")
        return
    
    st.info(f"📊 Colunas numéricas: {numeric_cols}")
    
    # Criar dataframe com dados numéricos
    df_numeric = dw_copy[numeric_cols].copy()
    
    # Remover linhas com NaN (mas manter TODOS os dados disponíveis)
    df_clean = df_numeric.dropna(how='all')  # Remove linhas COMPLETAMENTE vazias
    
    st.info(f"📊 Registos após limpeza: {len(df_clean)}")
    
    if len(df_clean) < 3:
        st.error(f"❌ Dados insuficientes: apenas {len(df_clean)} registos")
        st.write("Debug: Dataframe vazio?")
        st.write(df_numeric.head())
        return
    
    # ════════════════════════════════════════════════════════════════════════════════
    # MATRIZ DE CORRELAÇÃO
    # ════════════════════════════════════════════════════════════════════════════════
    
    try:
        st.subheader("📊 Matriz de Correlação")
        
        # Calcular correlação (Pearson)
        corr_matrix = df_clean.corr(method='pearson')
        
        st.success(f"✅ Correlação calculada: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")
        
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlação"),
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        fig.update_layout(
            title="Matriz de Correlação (Pearson)",
            height=700,
            width=900,
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Erro calcular correlação: {e}")
        st.write(f"Dados: {df_clean.shape}")
        return
    
    # ════════════════════════════════════════════════════════════════════════════════
    # CORRELAÇÕES SIGNIFICATIVAS
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.subheader("🔗 Correlações Significativas (|r| > 0.3)")
    
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                correlations.append({
                    'Variável 1': corr_matrix.columns[i],
                    'Variável 2': corr_matrix.columns[j],
                    'Correlação': f"{r:.3f}",
                    'Tipo': 'Positiva' if r > 0 else 'Negativa'
                })
    
    if correlations:
        df_sig = pd.DataFrame(correlations)
        st.dataframe(df_sig, use_container_width=True)
        st.success(f"✅ {len(correlations)} correlações significativas encontradas")
    else:
        st.info("ℹ️ Sem correlações fortes detectadas (|r| > 0.3)")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # ESTATÍSTICAS BÁSICAS
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.subheader("📈 Estatísticas Básicas")
    
    stats_df = pd.DataFrame({
        'Variável': df_clean.columns,
        'Média': df_clean.mean().values,
        'Std': df_clean.std().values,
        'Min': df_clean.min().values,
        'Max': df_clean.max().values,
        'Missing': df_clean.isnull().sum().values
    })
    
    st.dataframe(stats_df, use_container_width=True)
    
    # ════════════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("💾 Exportar")
    
    # Download correlação
    csv_corr = corr_matrix.to_csv()
    st.download_button(
        label="📥 Download Matriz de Correlação (CSV)",
        data=csv_corr,
        file_name=f"correlacoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Download estatísticas
    csv_stats = stats_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Estatísticas (CSV)",
        data=csv_stats,
        file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
