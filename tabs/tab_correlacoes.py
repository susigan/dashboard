"""
tab_correlacoes_avancadas.py — Análise de Correlações + Clustering + Relevância
§08 — Advanced Correlation Analysis with Feature Importance

Análisa:
  - 20+ variáveis (treino, bem-estar, composição corporal, biomarcadores)
  - Matriz de correlação completa
  - Clustering (K-means + Hierarchical)
  - Feature importance (Random Forest)
  - Correlação parcial + VIF
  - PCA (componentes principais)
  - Interpretação automática dos padrões
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def _carregar_dados_completos(da, dw):
    """Carrega e cruza todas as variáveis para análise."""
    # TODO: Implementar carregamento de:
    #   - Duração treino (horas)
    #   - kJ (energia)
    #   - CTLy, ATL, TSB
    #   - Monotonia, Strain
    #   - RPE
    #   - Sono (horas)
    #   - Qualidade sono
    #   - HRV, RHR
    #   - Wellness
    #   - Peso, BF%, Lean Mass
    #   - FMT (se existir)
    pass

def _limpeza_outliers(df, factor=1.5):
    """Remove outliers IQR 1.5x."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].mask(
            (df_clean[col] < Q1 - factor*IQR) | (df_clean[col] > Q3 + factor*IQR)
        )
    return df_clean

def _matriz_correlacao(df):
    """Calcula matriz de correlação + p-values."""
    corr_matrix = df.corr(method='pearson')
    
    # P-values
    n = len(df)
    pvalues = pd.DataFrame(
        np.zeros_like(corr_matrix),
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )
    
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i != j:
                r = corr_matrix.iloc[i, j]
                t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2 + 1e-10)
                pvalues.iloc[i, j] = 2 * (1 - stats.t.cdf(abs(t), n - 2))
    
    return corr_matrix, pvalues

def _clustering_kmeans(df_scaled, max_k=10):
    """K-means clustering com silhouette score."""
    from sklearn.metrics import silhouette_score
    
    inertias = []
    silhouettes = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            silhouettes.append(silhouette_score(df_scaled, labels))
        else:
            silhouettes.append(0)
    
    return inertias, silhouettes

def _feature_importance_rf(df, target_col='hrv'):
    """Importância das variáveis (Random Forest)."""
    if target_col not in df.columns:
        st.warning(f"Coluna '{target_col}' não encontrada")
        return None
    
    X = df.drop(columns=[target_col]).fillna(df.mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        'variavel': X.columns,
        'importancia': rf.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return importance

def _pca_analysis(df_scaled):
    """Análise PCA — componentes principais."""
    pca = PCA()
    pca.fit(df_scaled)
    
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Componentes dos primeiros 2 PCs
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=df_scaled.columns
    )
    
    return pca, explained_var, loadings

# ════════════════════════════════════════════════════════════════════════════════
# MAIN TAB FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def tab_correlacoes_avancadas(da, dw):
    """Tab de Correlações Avançadas + Clustering + Relevância."""
    
    st.header("🧬 Correlações Avançadas + Clustering")
    st.caption("Análise profunda: 20+ variáveis, clustering automático, relevância das variáveis")
    
    if len(da) == 0 or len(dw) == 0:
        st.warning("Sem dados suficientes")
        return
    
    # ── 1. Carregamento ──────────────────────────────────────────────────────────
    st.subheader("1️⃣ Carregamento de Dados")
    
    with st.spinner("Carregando e limpando dados..."):
        df_completo = _carregar_dados_completos(da, dw)
        
        if df_completo is None:
            st.error("❌ Erro ao carregar dados")
            return
        
        df_clean = _limpeza_outliers(df_completo)
        st.success(f"✅ Dados: {len(df_clean)} registos, {len(df_clean.columns)} variáveis")
    
    # ── 2. Matriz de Correlação ──────────────────────────────────────────────────
    st.subheader("2️⃣ Matriz de Correlação")
    
    corr_matrix, pvalues = _matriz_correlacao(df_clean)
    
    # Heatmap interactivo
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlação")
    ))
    fig_heatmap.update_layout(title="Matriz de Correlação (Pearson)", height=800)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ── 3. Clustering ────────────────────────────────────────────────────────────
    st.subheader("3️⃣ Clustering Automático")
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # Elbow curve
    inertias, silhouettes = _clustering_kmeans(df_scaled)
    
    fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
    fig_elbow.add_trace(
        go.Scatter(y=inertias, name="Inércia", mode='lines+markers'),
        secondary_y=False
    )
    fig_elbow.add_trace(
        go.Scatter(y=silhouettes, name="Silhouette", mode='lines+markers'),
        secondary_y=True
    )
    fig_elbow.update_layout(title="Elbow Curve + Silhouette Score")
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    # K-means com melhor K
    best_k = np.argmax(silhouettes) + 2 if len(silhouettes) > 1 else 2
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    
    # PCA para visualização 2D
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    
    fig_clusters = go.Figure(data=go.Scatter(
        x=df_pca[:, 0], y=df_pca[:, 1],
        mode='markers',
        marker=dict(size=8, color=labels, colorscale='Viridis', showscale=True),
        text=[f"Cluster {l}" for l in labels],
        hoverinfo='text'
    ))
    fig_clusters.update_layout(
        title=f"Clusters em PCA (K={best_k})",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    # ── 4. Relevância das Variáveis ──────────────────────────────────────────────
    st.subheader("4️⃣ Relevância das Variáveis (Feature Importance)")
    
    importance = _feature_importance_rf(df_clean, target_col='hrv')
    
    if importance is not None:
        fig_importance = go.Figure(
            data=go.Bar(x=importance['importancia'], y=importance['variavel'], orientation='h')
        )
        fig_importance.update_layout(title="Feature Importance (Random Forest → HRV)")
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # ── 5. Interpretação Automática ──────────────────────────────────────────────
    st.subheader("5️⃣ Padrões Detectados")
    
    for c in range(best_k):
        st.markdown(f"**Cluster {c+1}:**")
        mask = labels == c
        
        # Média por cluster
        cluster_mean = df_clean[mask].mean()
        st.write(cluster_mean)
        st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Teste local
    st.title("Teste: tab_correlacoes_avancadas")
    st.info("Este é um esboço — falta implementar carregamento de dados")
