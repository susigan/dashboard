"""
tab_correlacoes_avancada.py — VERSÃO COMPLETA
§08 Advanced Correlation Analysis with Feature Importance + Clustering + Lag Analysis

Análisa:
  - Feature Importance (Random Forest) → Qual variável impacta HRV/Wellness?
  - Clustering automático (K-means) → Padrões naturais dos dados
  - Lag Analysis → O que ontem afecta hoje? (D-1, D-2, D-3, D-7)
  - Interpretação automática dos padrões
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except:
    st.error("❌ Sklearn não instalado. Instala: pip install scikit-learn")

def tab_correlacoes(da, dw):
    """Análise avançada: Feature Importance + Clustering + Lag Analysis"""
    
    st.header("🧠 Correlações & Impacto Avançado")
    st.caption("Feature Importance + Clustering + Lag Analysis (D-1, D-2, D-3, D-7)")
    
    if len(da) == 0 or len(dw) == 0:
        st.warning("Sem dados suficientes.")
        return

    # ════════════════════════════════════════════════════════════════════════════════
    # PREPARAR DADOS
    # ════════════════════════════════════════════════════════════════════════════════
    
    dw_copy = dw.copy()
    dw_copy['Data'] = pd.to_datetime(dw_copy['Data'], errors='coerce')
    dw_copy = dw_copy.sort_values('Data')
    
    # Variáveis de Wellness (alvo para análise)
    target_vars = ['HRV', 'hrv', 'Wellness', 'wellness', 'Horas de Sono', 'sleep_quality']
    target_col = next((col for col in target_vars if col in dw_copy.columns), None)
    
    if target_col is None:
        st.error("❌ Coluna HRV/Wellness não encontrada")
        return
    
    # Colunas numéricas (features)
    numeric_cols = dw_copy.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    st.info(f"📊 Alvo: {target_col} | Features: {len(numeric_cols)}")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # TAB 1: FEATURE IMPORTANCE
    # ════════════════════════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Feature Importance",
        "🧬 Clustering",
        "📊 Lag Analysis",
        "📈 Matriz de Correlação"
    ])
    
    with tab1:
        st.subheader(f"🎯 Feature Importance → {target_col}")
        st.caption("Qual variável TEM IMPACTO no teu HRV/Wellness?")
        
        try:
            # Preparar dados para Random Forest
            df_ml = dw_copy[numeric_cols + [target_col]].dropna()
            
            if len(df_ml) < 10:
                st.error(f"❌ Dados insuficientes ({len(df_ml)})")
            else:
                X = df_ml[numeric_cols].fillna(df_ml[numeric_cols].mean())
                y = df_ml[target_col]
                
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf.fit(X, y)
                
                # Feature Importance
                importance = pd.DataFrame({
                    'Variável': numeric_cols,
                    'Importância': rf.feature_importances_
                }).sort_values('Importância', ascending=False)
                
                st.success(f"✅ R² = {rf.score(X, y):.3f}")
                
                # Plotar
                fig = go.Figure(data=go.Bar(
                    y=importance['Variável'][:15],
                    x=importance['Importância'][:15],
                    orientation='h',
                    marker_color='#2980b9'
                ))
                fig.update_layout(title=f"Top 15 Variáveis que Impactam {target_col}", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela
                st.dataframe(importance.head(20), use_container_width=True)
                
                # Interpretação
                st.markdown("---")
                st.subheader("💡 Interpretação")
                top3 = importance.head(3)['Variável'].tolist()
                st.success(f"✅ As 3 variáveis com MAIOR impacto em {target_col}:")
                for i, var in enumerate(top3, 1):
                    imp = importance[importance['Variável'] == var]['Importância'].values[0]
                    st.write(f"{i}. **{var}** (importância: {imp:.3f})")
                
        except Exception as e:
            st.error(f"❌ Erro: {e}")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # TAB 2: CLUSTERING
    # ════════════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.subheader("🧬 Clustering Automático")
        st.caption("Que padrões naturais existem nos teus dados?")
        
        try:
            df_cluster = dw_copy[numeric_cols].dropna()
            
            if len(df_cluster) < 5:
                st.error("❌ Dados insuficientes")
            else:
                # Normalizar
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_cluster)
                
                # Elbow curve
                inertias = []
                silhouettes = []
                for k in range(2, min(10, len(df_cluster))):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(X_scaled)
                    inertias.append(km.inertia_)
                    silhouettes.append(silhouette_score(X_scaled, labels))
                
                # Plot Elbow
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=inertias, mode='lines+markers', name='Inércia', yaxis='y1'))
                fig.add_trace(go.Scatter(y=silhouettes, mode='lines+markers', name='Silhouette', yaxis='y2'))
                fig.update_layout(
                    title="Elbow Curve + Silhouette",
                    yaxis=dict(title='Inércia'),
                    yaxis2=dict(title='Silhouette', overlaying='y', side='right'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Best K
                best_k = np.argmax(silhouettes) + 2
                st.success(f"✅ Melhor K: {best_k} clusters")
                
                # K-means com melhor K
                km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                
                # PCA para visualização 2D
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig = go.Figure(data=go.Scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    mode='markers',
                    marker=dict(size=8, color=labels, colorscale='Viridis', showscale=True),
                    text=[f"Cluster {l}" for l in labels],
                    hoverinfo='text'
                ))
                fig.update_layout(
                    title=f"Clusters em PCA (K={best_k})",
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas por cluster
                st.subheader("📊 Estatísticas por Cluster")
                dw_copy['Cluster'] = labels
                
                for c in range(best_k):
                    cluster_data = dw_copy[dw_copy['Cluster'] == c]
                    st.markdown(f"**Cluster {c}** ({len(cluster_data)} registos)")
                    
                    stats = cluster_data[numeric_cols].describe().loc[['mean', 'std']].T
                    st.dataframe(stats, use_container_width=True)
                    st.markdown("---")
                
        except Exception as e:
            st.error(f"❌ Erro: {e}")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # TAB 3: LAG ANALYSIS
    # ════════════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.subheader("📊 Lag Analysis — O que ONTEM afecta HOJE?")
        st.caption("Correlação com lag: D-1, D-2, D-3, D-7")
        
        try:
            df_lag = dw_copy[[target_col] + numeric_cols].copy()
            df_lag = df_lag.set_index('Data') if 'Data' in df_lag.columns else df_lag
            
            # Lags
            lags = [1, 2, 3, 7]
            lag_results = []
            
            for feature in numeric_cols[:10]:  # Top 10 features
                if feature not in df_lag.columns:
                    continue
                
                for lag in lags:
                    df_lag_temp = df_lag[[target_col, feature]].copy()
                    df_lag_temp[f'{feature}_lag{lag}'] = df_lag_temp[feature].shift(lag)
                    df_lag_temp = df_lag_temp.dropna()
                    
                    if len(df_lag_temp) > 5:
                        corr = df_lag_temp[target_col].corr(df_lag_temp[f'{feature}_lag{lag}'])
                        lag_results.append({
                            'Variável': feature,
                            'Lag': f'D-{lag}',
                            'Correlação': corr
                        })
            
            if lag_results:
                df_lags = pd.DataFrame(lag_results)
                
                # Plotar
                fig = go.Figure()
                for feature in df_lags['Variável'].unique()[:5]:
                    df_feat = df_lags[df_lags['Variável'] == feature]
                    fig.add_trace(go.Scatter(
                        x=df_feat['Lag'],
                        y=df_feat['Correlação'],
                        mode='lines+markers',
                        name=feature
                    ))
                
                fig.update_layout(
                    title=f"Lag Correlations → {target_col}",
                    xaxis_title="Lag (dias antes)",
                    yaxis_title="Correlação",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_lags.sort_values('Correlação', ascending=False), use_container_width=True)
            else:
                st.warning("⚠️ Sem dados para lag analysis")
        
        except Exception as e:
            st.error(f"❌ Erro: {e}")
    
    # ════════════════════════════════════════════════════════════════════════════════
    # TAB 4: MATRIZ DE CORRELAÇÃO
    # ════════════════════════════════════════════════════════════════════════════════
    
    with tab4:
        st.subheader("📊 Matriz de Correlação Completa")
        
        try:
            df_corr = dw_copy[numeric_cols + [target_col]].dropna()
            corr_matrix = df_corr.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                zmin=-1, zmax=1
            ))
            fig.update_layout(title="Matriz de Correlação", height=800)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlações fortes
            st.subheader("🔗 Correlações Fortes (|r| > 0.3)")
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    r = corr_matrix.iloc[i, j]
                    if abs(r) > 0.3:
                        correlations.append({
                            'Var1': corr_matrix.columns[i],
                            'Var2': corr_matrix.columns[j],
                            'Correlação': f"{r:.3f}"
                        })
            
            if correlations:
                st.dataframe(pd.DataFrame(correlations), use_container_width=True)
            else:
                st.info("ℹ️ Sem correlações fortes")
        
        except Exception as e:
            st.error(f"❌ Erro: {e}")
