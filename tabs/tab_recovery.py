from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def tab_recovery(dw):
    st.header("🔋 Recovery Score & HRV Analysis")
    if len(dw) == 0 or 'hrv' not in dw.columns: st.warning("Sem dados de wellness/HRV."); return
    rec = calcular_recovery(dw)
    if len(rec) == 0: return
    u = rec.iloc[-1]; score = u['recovery_score']
    cat = ('🟢 Excelente' if score >= 80 else '🟡 Bom' if score >= 60 else '🟠 Moderado' if score >= 40 else '🔴 Baixo')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{score:.0f}/100", delta=cat)
    c2.metric("HRV atual", f"{u['hrv']:.0f} ms" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline HRV", f"{u['hrv_baseline']:.0f} ms" if pd.notna(u['hrv_baseline']) else "—")
    _cv7 = u.get('hrv_cv_7d', u.get('hrv_cv7', u.get('hrv_cv', None)))
    c4.metric("CV% 7d", f"{_cv7:.1f}%" if _cv7 is not None and pd.notna(_cv7) else "—")
    st.markdown("---")
    
    # Configurações de janela
    col1, col2 = st.columns(2)
    with col1:
        n_dias = st.slider("Dias a mostrar", 14, min(len(dw), 365), min(90, len(dw)), key="dias_hrv")
    with col2:
        janela_cv = st.slider("Janela CV% (dias)", 3, 14, 7, key="janela_cv")
    
    # Preparar dados
    df_hrv = dw.copy().sort_values('Data')
    df_hrv['Data'] = pd.to_datetime(df_hrv['Data'])
    df_hrv = df_hrv[df_hrv['Data'] >= (df_hrv['Data'].max() - pd.Timedelta(days=n_dias))]
    
    if 'hrv' not in df_hrv.columns or df_hrv['hrv'].notna().sum() < janela_cv + 3:
        st.warning("Dados de HRV insuficientes para análise de tendências.")
        return
    
    # Calcular LnRMSSD
    df_hrv['LnrMSSD'] = np.where(df_hrv['hrv'] > 0, np.log(df_hrv['hrv']), np.nan)
    df_hrv = df_hrv.dropna(subset=['LnrMSSD'])
    
    if len(df_hrv) < janela_cv + 3:
        st.warning("Dados insuficientes após filtrar HRV válido.")
        return
    
    # Baseline de 60 dias (média móvel)
    baseline_dias = 60
    df_hrv['ln_baseline'] = df_hrv['LnrMSSD'].rolling(baseline_dias, min_periods=14).mean()
    df_hrv['ln_std'] = df_hrv['LnrMSSD'].rolling(baseline_dias, min_periods=14).std()
    
    # CV% do LnRMSSD (janela móvel)
    df_hrv['ln_mean_short'] = df_hrv['LnrMSSD'].rolling(janela_cv, min_periods=3).mean()
    df_hrv['ln_std_short'] = df_hrv['LnrMSSD'].rolling(janela_cv, min_periods=3).std()
    df_hrv['cv_lnrmssd'] = (df_hrv['ln_std_short'] / df_hrv['ln_mean_short']) * 100
    
    # Classificação das 4 zonas (modelo Altini)
    # Threshold CV% = 1.0 (baseado na literatura de HRV)
    CV_THRESHOLD = 1.0
    
    def classificar_zona(row):
        cv = row['cv_lnrmssd']
        ln = row['LnrMSSD']
        base = row['ln_baseline']
        
        if pd.isna(cv) or pd.isna(base):
            return 'Sem dados', '#808080'
        
        cv_estavel = cv < CV_THRESHOLD
        ln_normal = ln >= base * 0.98  # 2% de tolerância abaixo do baseline
        
        if cv_estavel and ln_normal:
            return 'Coping Well', '#27ae60'      # Verde
        elif cv_estavel and not ln_normal:
            return 'Fatigue', '#f39c12'          # Amarelo/Laranja
        elif not cv_estavel and ln_normal:
            return 'Stable', '#3498db'           # Azul
        else:
            return 'Maladaptation', '#e74c3c'    # Vermelho
    
    df_hrv[['zona', 'cor_zona']] = df_hrv.apply(
        lambda row: pd.Series(classificar_zona(row)), axis=1
    )
    
    df_plot = df_hrv.dropna(subset=['cv_lnrmssd', 'ln_baseline'])
    
    if len(df_plot) == 0:
        st.warning("Dados insuficientes após cálculos.")
        return
    
    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'CV% LnRMSSD ({janela_cv}d) com Baseline ({baseline_dias}d) — Threshold: {CV_THRESHOLD}%',
            'LnRMSSD com Classificação de Status (Altini et al.)'
        ),
        row_heights=[0.4, 0.6]
    )
    
    # === GRÁFICO 1: CV% LnRMSSD ===
    # Linha do CV%
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['cv_lnrmssd'],
            mode='lines',
            name=f'CV% ({janela_cv}d)',
            line=dict(color='#2c3e50', width=2),
            hovertemplate='Data: %{x}<br>CV%: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Threshold 1.0%
    fig.add_hline(
        y=CV_THRESHOLD, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text=f"Threshold CV% = {CV_THRESHOLD}%",
        annotation_position="right",
        row=1, col=1
    )
    
    # Zona ideal (abaixo de 1.0%)
    fig.add_hrect(
        y0=0, 
        y1=CV_THRESHOLD,
        fillcolor="green", 
        opacity=0.1,
        line_width=0,
        row=1, col=1
    )
    
    # === GRÁFICO 2: LnRMSSD com pontos coloridos ===
    # Linha de baseline
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['ln_baseline'],
            mode='lines',
            name=f'Baseline ({baseline_dias}d)',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='Data: %{x}<br>Baseline: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Banda de ±1 SD do baseline
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['ln_baseline'] + df_plot['ln_std'],
            mode='lines',
            name='+1 SD',
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['ln_baseline'] - df_plot['ln_std'],
            mode='lines',
            name='-1 SD',
            line=dict(color='rgba(0,0,0,0)', width=0),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.15)',
            showlegend=True,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Pontos LnRMSSD coloridos por zona
    for zona, cor in [
        ('Coping Well', '#27ae60'),
        ('Fatigue', '#f39c12'),
        ('Stable', '#3498db'),
        ('Maladaptation', '#e74c3c')
    ]:
        df_zona = df_plot[df_plot['zona'] == zona]
        if len(df_zona) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_zona['Data'],
                    y=df_zona['LnrMSSD'],
                    mode='markers',
                    name=zona,
                    marker=dict(
                        color=cor,
                        size=10,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{zona}</b><br>Data: %{{x}}<br>LnRMSSD: %{{y:.3f}}<br>CV%: %{{customdata:.2f}}%<extra></extra>',
                    customdata=df_zona['cv_lnrmssd']
                ),
                row=2, col=1
            )
    
    # Layout geral
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#111', size=12),
        margin=dict(t=80, b=70, l=60, r=40),
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.15,
            x=0.5,
            xanchor='center',
            font=dict(size=11)
        ),
        hovermode='x unified'
    )
    
    # Eixos Y
    fig.update_yaxes(
        title_text=f'CV% LnRMSSD ({janela_cv}d)',
        showgrid=True, 
        gridcolor='#eee',
        tickfont=dict(color='#111'),
        range=[0, max(2.5, df_plot['cv_lnrmssd'].max() * 1.2)],
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='LnRMSSD',
        showgrid=True, 
        gridcolor='#eee',
        tickfont=dict(color='#111'),
        row=2, col=1
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#eee',
        tickfont=dict(color='#111'),
        row=2, col=1
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_altini_chart")
    
    # Legenda explicativa
    st.markdown("""
    ### 📊 Interpretação das Zonas (Modelo Altini et al.)
    
    | Zona | Cor | Condição | Interpretação |
    |------|-----|----------|---------------|
    | **🟢 Coping Well** | Verde | CV% < 1.0% + LnRMSSD ≥ baseline | **Recuperando bem** — pronto para carga de treino |
    | **🟡 Fatigue** | Amarelo | CV% < 1.0% + LnRMSSD < baseline | **Fadiga funcional** — necessita descanso ativo |
    | **🔵 Stable** | Azul | CV% ≥ 1.0% + LnRMSSD ≥ baseline | **Variável mas estável** — monitorar, carga moderada |
    | **🔴 Maladaptation** | Vermelho | CV% ≥ 1.0% + LnRMSSD < baseline | **Estresse crônico** — risco de overtraining, descanso obrigatório |
    
    **Threshold CV% = 1.0%** — baseado em literatura de HRV (Flatt & Esco, 2016; Altini, 2020)
    """)
    
    # Status atual
    ultimo = df_plot.iloc[-1]
    col_status, col_cv, col_ln = st.columns(3)
    
    with col_status:
        st.metric(
            "Status Atual",
            ultimo['zona'],
            delta="Bom" if ultimo['zona'] in ['Coping Well', 'Stable'] else "Atenção"
        )
    
    with col_cv:
        st.metric(
            f"CV% ({janela_cv}d)",
            f"{ultimo['cv_lnrmssd']:.2f}%",
            delta="Estável" if ultimo['cv_lnrmssd'] < CV_THRESHOLD else "Variável"
        )
    
    with col_ln:
        ln_pct = (ultimo['LnrMSSD'] / ultimo['ln_baseline'] - 1) * 100
        st.metric(
            "LnRMSSD vs Baseline",
            f"{ultimo['LnrMSSD']:.3f}",
            delta=f"{ln_pct:+.1f}%"
        )

    # Continuação: BPE e HRV-Guided Training (mantidos do código anterior)
    st.markdown("---")
    st.subheader("📊 BPE — Z-Score Semanal (Método SWC)")
    mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] if m in dw.columns and dw[m].notna().any()]
    n_semanas_disp = max(1, len(dw) // 7)
    _skip_bpe = n_semanas_disp < 4
    if _skip_bpe:
        st.info(f"Dados insuficientes para BPE (min 4 semanas, disponivel: {n_semanas_disp}).")
        n_sem = n_semanas_disp
    else:
        _slider_max = min(52, n_semanas_disp)
        _slider_val = min(16, _slider_max)
        if _slider_max > 4:
            n_sem = st.slider("Semanas (BPE)", 4, _slider_max, _slider_val, key="bpe_sem")
        else:
            n_sem = _slider_max
            st.caption(f"BPE: {n_sem} semanas disponíveis")
    dados_bpe = {}
    if not _skip_bpe:
        for met in mets_bpe:
            s = calcular_bpe(dw, met, 60)
            if len(s) > 0: dados_bpe[met] = s.tail(n_sem)
    if dados_bpe:
        semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
        nm = len(dados_bpe); mat = np.zeros((nm, len(semanas)))
        nomes_bpe = {'hrv': 'HRV', 'rhr': 'RHR (inv)', 'sleep_quality': 'Sono',
                     'fatiga': 'Energia', 'stress': 'Relaxamento'}
        for i, met in enumerate(dados_bpe.keys()):
            z = dados_bpe[met]['zscore'].values; mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
        import numpy as _npIM
        _zIM = [[float(mat[r][c]) if not _npIM.isnan(mat[r][c]) else None
                 for c in range(mat.shape[1])] for r in range(mat.shape[0])]
        _yIM = list(nomes_bpe.values())
        _fIM = go.Figure(go.Heatmap(z=_zIM,
            x=[str(s) for s in semanas],
            y=_yIM, colorscale='RdYlGn', zmid=0, zmin=-2, zmax=2,
            colorbar=dict(title='Z', tickfont=dict(color='#111'))))
        _fIM.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=max(280, len(_yIM)*35),
            title=dict(text='BPE — Z-Score com SWC', font=dict(size=13,color='#111')),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9,color='#111')),
            yaxis=dict(tickfont=dict(size=9,color='#111')))
        st.plotly_chart(_fIM, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_bpe_chart")

    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")
    if len(dw) >= 14 and 'hrv' in dw.columns and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data'); df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'] > 0, np.log(df_hg['hrv']), np.nan); df_hg = df_hg.dropna(subset=['LnrMSSD'])
        dias_fam = st.slider("Dias baseline rolling", 7, 28, 14, key="hg_baseline")
        df_hg['bm'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg['bs'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg['linf'] = df_hg['bm'] - 0.5 * df_hg['bs']; df_hg['lsup'] = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs']
        df_hg['intens'] = df_hg.apply(lambda r: 'HIIT' if pd.notna(r['bm']) and r['linf'] <= r['LnrMSSD'] <= r['lsup'] else ('Recuperação' if pd.notna(r['bm']) else 'Sem dados'), axis=1)
        n_hg = st.slider("Dias HRV-Guided", 14, min(len(df_hg), 180), min(60, len(df_hg)), key="hg_dias")
        df_p = df_hg.tail(n_hg).copy()
        
        # Gráfico HRV-Guided
        _fig_hg = go.Figure()
        
        if len(df_p) > 0:
            # LnrMSSD com cores por intensidade
            cores_hg = ['green' if i == 'HIIT' else 'orange' if i == 'Recuperação' else 'gray' for i in df_p['intens']]
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'],
                y=df_p['LnrMSSD'],
                mode='markers',
                name='LnrMSSD',
                marker=dict(color=cores_hg, size=10, line=dict(width=1, color='white'))
            ))
            
            # Bandas
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'], y=df_p['lsup'], mode='lines',
                name='Limite Superior', line=dict(color='green', width=1, dash='dash')
            ))
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'], y=df_p['linf'], mode='lines',
                name='Limite Inferior', line=dict(color='green', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(46, 204, 113, 0.1)'
            ))
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'], y=df_p['bm'], mode='lines',
                name='Baseline', line=dict(color='gray', width=2)
            ))
        
        _fig_hg.update_layout(
            title='HRV-Guided Training',
            paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'),
            margin=dict(t=50,b=70,l=55,r=20), height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            yaxis=dict(showgrid=True, gridcolor='#eee', title='LnRMSSD')
        )
        st.plotly_chart(_fig_hg, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_hg_chart")
        
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum(); rec_n = (df_val['intens'] == 'Recuperação').sum(); total_n = len(df_val)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            c2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            c3.metric("Prescrição HOJE", '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT' else '🟠 Recuperação')
