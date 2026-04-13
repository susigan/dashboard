from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    
    n_dias = st.slider("Dias a mostrar", 14, min(len(rec), 365), min(90, len(rec)))
    df_tl = rec.tail(n_dias).copy()
    
    # GRÁFICO 1: Recovery Score ao longo do tempo
    _fig_gen = go.Figure()
    # ADICIONADO: Trace com dados reais
    _fig_gen.add_trace(go.Scatter(
        x=df_tl['Data'], 
        y=df_tl['recovery_score'],
        mode='lines+markers',
        name='Recovery Score',
        line=dict(color=CORES.get('azul', '#1f77b4'), width=2),
        marker=dict(size=6)
    ))
    # Linha de referência em 80
    _fig_gen.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excelente")
    _fig_gen.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Bom")
    _fig_gen.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Moderado")
    
    _fig_gen.update_layout(
        title='Recovery Score ao longo do tempo',
        paper_bgcolor='white', plot_bgcolor='white', 
        font=dict(color='#111'), 
        margin=dict(t=50,b=70,l=55,r=20), 
        height=340,
        legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), 
        hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), 
        yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'), range=[0, 100])
    )
    st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_chart_1")

    st.subheader("📊 HRV com Normal Range")
    
    # GRÁFICO 2: HRV com bandas de normalidade
    _fig_gen2 = go.Figure()
    df_hrv = df_tl[['Data', 'hrv', 'hrv_baseline']].dropna()
    
    if len(df_hrv) > 0:
        # Calcular bandas ±1 SD
        hrv_mean = df_hrv['hrv'].mean()
        hrv_std = df_hrv['hrv'].std()
        
        # HRV real
        _fig_gen2.add_trace(go.Scatter(
            x=df_hrv['Data'],
            y=df_hrv['hrv'],
            mode='lines+markers',
            name='HRV',
            line=dict(color=CORES.get('verde', '#2ecc71'), width=2)
        ))
        
        # Baseline
        _fig_gen2.add_trace(go.Scatter(
            x=df_hrv['Data'],
            y=df_hrv['hrv_baseline'],
            mode='lines',
            name='Baseline',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Bandas de normalidade
        _fig_gen2.add_trace(go.Scatter(
            x=df_hrv['Data'],
            y=[hrv_mean + hrv_std] * len(df_hrv),
            mode='lines',
            name='+1 SD',
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False
        ))
        _fig_gen2.add_trace(go.Scatter(
            x=df_hrv['Data'],
            y=[hrv_mean - hrv_std] * len(df_hrv),
            mode='lines',
            name='-1 SD',
            line=dict(color='rgba(0,0,0,0)', width=0),
            fill='tonexty',
            fillcolor='rgba(46, 204, 113, 0.2)',
            showlegend=False
        ))
    
    _fig_gen2.update_layout(
        title='HRV com Faixa Normal (±1 SD)',
        paper_bgcolor='white', plot_bgcolor='white', 
        font=dict(color='#111'), 
        margin=dict(t=50,b=70,l=55,r=20), 
        height=340,
        legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), 
        hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), 
        yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'))
    )
    st.plotly_chart(_fig_gen2, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_chart_2")

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
            n_sem = st.slider("Semanas (BPE)", 4, _slider_max, _slider_val)
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
        st.plotly_chart(_fIM, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_chart_3")

    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")
    if len(dw) >= 14 and 'hrv' in dw.columns and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data'); df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'] > 0, np.log(df_hg['hrv']), np.nan); df_hg = df_hg.dropna(subset=['LnrMSSD'])
        dias_fam = st.slider("Dias baseline rolling", 7, 28, 14)
        df_hg['bm'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg['bs'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg['linf'] = df_hg['bm'] - 0.5 * df_hg['bs']; df_hg['lsup'] = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs']
        df_hg['intens'] = df_hg.apply(lambda r: 'HIIT' if pd.notna(r['bm']) and r['linf'] <= r['LnrMSSD'] <= r['lsup'] else ('Recuperação' if pd.notna(r['bm']) else 'Sem dados'), axis=1)
        n_hg = st.slider("Dias HRV-Guided", 14, min(len(df_hg), 180), min(60, len(df_hg)))
        df_p = df_hg.tail(n_hg).copy()
        
        # GRÁFICO 4: HRV-Guided Training
        _fig_gen3 = go.Figure()
        
        if len(df_p) > 0:
            # LnrMSSD
            _fig_gen3.add_trace(go.Scatter(
                x=df_p['Data'],
                y=df_p['LnrMSSD'],
                mode='lines+markers',
                name='LnrMSSD',
                line=dict(color='blue', width=2),
                marker=dict(
                    color=['green' if i == 'HIIT' else 'orange' if i == 'Recuperação' else 'gray' for i in df_p['intens']],
                    size=8
                )
            ))
            
            # Bandas
            _fig_gen3.add_trace(go.Scatter(
                x=df_p['Data'],
                y=df_p['lsup'],
                mode='lines',
                name='Limite Superior',
                line=dict(color='green', width=1, dash='dash'),
                showlegend=True
            ))
            _fig_gen3.add_trace(go.Scatter(
                x=df_p['Data'],
                y=df_p['linf'],
                mode='lines',
                name='Limite Inferior',
                line=dict(color='green', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(46, 204, 113, 0.1)',
                showlegend=True
            ))
            
            # Baseline
            _fig_gen3.add_trace(go.Scatter(
                x=df_p['Data'],
                y=df_p['bm'],
                mode='lines',
                name='Baseline',
                line=dict(color='gray', width=2)
            ))
        
        _fig_gen3.update_layout(
            title='HRV-Guided Training (LnrMSSD)',
            paper_bgcolor='white', plot_bgcolor='white', 
            font=dict(color='#111'), 
            margin=dict(t=50,b=70,l=55,r=20), 
            height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), 
            hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), 
            yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'))
        )
        st.plotly_chart(_fig_gen3, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="recovery_chart_4")
        
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum(); rec_n = (df_val['intens'] == 'Recuperação').sum(); total_n = len(df_val)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            c2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            c3.metric("Prescrição HOJE", '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT' else '🟠 Recuperação')
