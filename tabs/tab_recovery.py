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
    if len(dw) == 0 or 'hrv' not in dw.columns: 
        st.warning("Sem dados de wellness/HRV.")
        return
    
    rec = calcular_recovery(dw)
    if len(rec) == 0: 
        return
    
    u = rec.iloc[-1]
    score = u['recovery_score']
    cat = ('🟢 Excelente' if score >= 80 else '🟡 Bom' if score >= 60 else '🟠 Moderado' if score >= 40 else '🔴 Baixo')
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{score:.0f}/100", delta=cat)
    c2.metric("HRV atual", f"{u['hrv']:.0f} ms" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline HRV", f"{u['hrv_baseline']:.0f} ms" if pd.notna(u['hrv_baseline']) else "—")
    _cv7 = u.get('hrv_cv_7d', u.get('hrv_cv7', u.get('hrv_cv', None)))
    c4.metric("CV% 7d", f"{_cv7:.1f}%" if _cv7 is not None and pd.notna(_cv7) else "—")
    
    st.markdown("---")
    
    # Configurações
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
    
    # Baseline de 60 dias
    baseline_dias = 60
    df_hrv['ln_baseline'] = df_hrv['LnrMSSD'].rolling(baseline_dias, min_periods=14).mean()
    df_hrv['ln_std'] = df_hrv['LnrMSSD'].rolling(baseline_dias, min_periods=14).std()
    
    # CV% do LnRMSSD (janela móvel)
    df_hrv['ln_mean_short'] = df_hrv['LnrMSSD'].rolling(janela_cv, min_periods=3).mean()
    df_hrv['ln_std_short'] = df_hrv['LnrMSSD'].rolling(janela_cv, min_periods=3).std()
    df_hrv['cv_lnrmssd'] = (df_hrv['ln_std_short'] / df_hrv['ln_mean_short']) * 100
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # THRESHOLD ADAPTATIVO CORRIGIDO (sem clamp de 2%)
    # ═══════════════════════════════════════════════════════════════════════════════
    dw_full = dw.copy()
    dw_full['LnrMSSD_full'] = np.where(dw_full['hrv'] > 0, np.log(dw_full['hrv']), np.nan)
    dw_full = dw_full.dropna(subset=['LnrMSSD_full'])
    
    if len(dw_full) >= janela_cv + 10:
        dw_full['ln_mean_temp'] = dw_full['LnrMSSD_full'].rolling(janela_cv, min_periods=3).mean()
        dw_full['ln_std_temp'] = dw_full['LnrMSSD_full'].rolling(janela_cv, min_periods=3).std()
        dw_full['cv_temp'] = (dw_full['ln_std_temp'] / dw_full['ln_mean_temp']) * 100
        cv_historico = dw_full['cv_temp'].dropna()
        
        if len(cv_historico) > 20:
            # Usa média + 0.5*SD do histórico (mais representativo que P75)
            CV_THRESHOLD = cv_historico.mean() + (0.5 * cv_historico.std())
            # Garante mínimo de 0.5% para evitar threshold muito baixo
            CV_THRESHOLD = max(0.5, CV_THRESHOLD)
        else:
            CV_THRESHOLD = 1.0
    else:
        CV_THRESHOLD = 1.0
    
    # Classificação das 4 zonas (modelo Altini) - EXATAMENTE COMO ESTAVA
    def classificar_zona(row):
        cv = row['cv_lnrmssd']
        ln = row['LnrMSSD']
        base_ln = row['ln_baseline']
        
        rhr = row.get('rhr', np.nan)
        base_rhr = row.get('rhr_baseline', np.nan)
        
        if pd.isna(cv) or pd.isna(base_ln):
            return 'Sem dados', '#808080'
        
        cv_alto = cv > CV_THRESHOLD
        ln_baixo = ln < base_ln * 0.97
        rhr_alto = False
        if not pd.isna(rhr) and not pd.isna(base_rhr) and base_rhr > 0:
            rhr_alto = rhr > base_rhr * 1.03
        
        if not cv_alto and not ln_baixo:
            return 'Coping Well', '#27ae60'
        elif cv_alto and (ln_baixo or rhr_alto):
            return 'Maladaptation', '#f1c40f'
        elif not cv_alto and ln_baixo:
            return 'Accumulated Fatigue', '#e74c3c'
        else:
            return 'Stable', '#2c3e50'
    
    if 'rhr' in df_hrv.columns:
        df_hrv['rhr_baseline'] = df_hrv['rhr'].rolling(baseline_dias, min_periods=14).mean()
    
    df_hrv[['zona', 'cor_zona']] = df_hrv.apply(
        lambda row: pd.Series(classificar_zona(row)), axis=1
    )
    
    df_plot = df_hrv.dropna(subset=['cv_lnrmssd', 'ln_baseline'])
    
    if len(df_plot) == 0:
        st.warning("Dados insuficientes após cálculos.")
        return
    
    # GRÁFICO COMBINADO - EXATAMENTE COMO ESTAVA ORIGINALMENTE
    fig = go.Figure()
    
    # Eixo Y esquerdo: LnRMSSD em pontos coloridos
    for zona, cor in [
        ('Coping Well', '#27ae60'),
        ('Stable', '#2c3e50'),
        ('Maladaptation', '#f1c40f'),
        ('Accumulated Fatigue', '#e74c3c'),
        ('Sem dados', '#808080')
    ]:
        df_zona = df_plot[df_plot['zona'] == zona]
        if len(df_zona) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_zona['Data'],
                    y=df_zona['LnrMSSD'],
                    mode='markers',
                    name=zona,
                    marker=dict(color=cor, size=12, line=dict(width=2, color='white')),
                    yaxis='y1',
                    hovertemplate=f'<b>{zona}</b><br>Data: %{{x}}<br>LnRMSSD: %{{y:.3f}}<br>CV%: %{{customdata:.2f}}%<extra></extra>',
                    customdata=df_zona['cv_lnrmssd']
                )
            )
    
    # Linha de baseline LnRMSSD
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['ln_baseline'],
            mode='lines',
            name=f'Baseline LnRMSSD ({baseline_dias}d)',
            line=dict(color='#34495e', width=2, dash='dash'),
            yaxis='y1'
        )
    )
    
    # Banda de ±1 SD
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['ln_baseline'] + df_plot['ln_std'],
            mode='lines',
            name='+1 SD',
            line=dict(color='rgba(52, 73, 94, 0.3)', width=1),
            yaxis='y1',
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['ln_baseline'] - df_plot['ln_std'],
            mode='lines',
            line=dict(color='rgba(52, 73, 94, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(52, 73, 94, 0.1)',
            yaxis='y1',
            name='Faixa normal (±1 SD)',
            hoverinfo='skip'
        )
    )
    
    # Eixo Y direito: CV% em linha
    fig.add_trace(
        go.Scatter(
            x=df_plot['Data'],
            y=df_plot['cv_lnrmssd'],
            mode='lines+markers',
            name=f'CV% LnRMSSD ({janela_cv}d)',
            line=dict(color='#e67e22', width=2),
            marker=dict(size=6, symbol='diamond'),
            yaxis='y2'
        )
    )
    
    # LINHA DE THRESHOLD ADAPTATIVO
    fig.add_trace(
        go.Scatter(
            x=[df_plot['Data'].iloc[0], df_plot['Data'].iloc[-1]],
            y=[CV_THRESHOLD, CV_THRESHOLD],
            mode='lines',
            name=f'Threshold CV% = {CV_THRESHOLD:.2f}%',
            line=dict(color='#c0392b', width=2, dash='dash'),
            yaxis='y2',
            hoverinfo='skip'
        )
    )
    
    # Layout EXATAMENTE COMO ESTAVA
    fig.update_layout(
        title=dict(
            text=f'LnRMSSD (pontos) + CV% (linha) — Threshold adaptativo: {CV_THRESHOLD:.2f}%',
            font=dict(color='#000000', size=14)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#000000', size=12),
        margin=dict(t=80, b=100, l=80, r=80),
        height=500,
        showlegend=True,
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(color='#000000', size=10)),
        hovermode='x unified',
        yaxis=dict(
            title=dict(text='LnRMSSD', font=dict(color='#000000')),
            tickfont=dict(color='#000000'),
            showgrid=True, gridcolor='#e0e0e0', side='left'
        ),
        yaxis2=dict(
            title=dict(text=f'CV% ({janela_cv}d)', font=dict(color='#000000')),
            tickfont=dict(color='#000000'),
            overlaying='y', side='right', showgrid=False,
            range=[0, max(2.5, df_plot['cv_lnrmssd'].max() * 1.3, CV_THRESHOLD * 1.5)]
        ),
        xaxis=dict(
            title=dict(text='Data', font=dict(color='#000000')),
            tickfont=dict(color='#000000'),
            showgrid=True, gridcolor='#e0e0e0'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, 
                   config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, 
                   key="recovery_altini_chart")
    
    # Status atual
    ultimo = df_plot.iloc[-1]
    
    st.markdown("### 📊 Status Atual")
    cols = st.columns([2, 1, 1, 1])
    
    with cols[0]:
        cor_status = ultimo['cor_zona']
        zona_status = ultimo['zona']
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; background-color: {cor_status}20; border-left: 5px solid {cor_status};">
            <h4 style="margin: 0; color: {cor_status};">{zona_status}</h4>
            <p style="margin: 5px 0 0 0; color: #000000; font-size: 12px;">
                CV%: {ultimo['cv_lnrmssd']:.2f}% | 
                LnRMSSD: {ultimo['LnrMSSD']:.3f} | 
                Baseline: {ultimo['ln_baseline']:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.metric("CV% Atual", f"{ultimo['cv_lnrmssd']:.2f}%", 
                 delta=f"Threshold: {CV_THRESHOLD:.2f}%")
    
    with cols[2]:
        ln_pct = (ultimo['LnrMSSD'] / ultimo['ln_baseline'] - 1) * 100
        st.metric("LnRMSSD vs Base", f"{ln_pct:+.1f}%")
    
    with cols[3]:
        st.metric("Threshold", f"{CV_THRESHOLD:.2f}%", 
                 delta="Adaptativo (Média+0.5SD)")
    
    # Legenda explicativa
    st.markdown("""
    ### 📖 Guia de Interpretação (Modelo Altini)
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
    
    <div style="padding: 10px; background-color: #27ae6020; border-left: 4px solid #27ae60; border-radius: 5px;">
        <b style="color: #27ae60;">🟢 Coping Well</b><br>
        <span style="color: #000000; font-size: 12px;">
        CV% ≤ threshold + LnRMSSD normal/alto<br>
        <b>Ação:</b> Continue o plano de treino
        </span>
    </div>
    
    <div style="padding: 10px; background-color: #2c3e5020; border-left: 4px solid #2c3e50; border-radius: 5px;">
        <b style="color: #2c3e50;">⚫ Stable</b><br>
        <span style="color: #000000; font-size: 12px;">
        CV% > threshold + LnRMSSD normal<br>
        <b>Ação:</b> Monitore, variação alta mas estável
        </span>
    </div>
    
    <div style="padding: 10px; background-color: #f1c40f20; border-left: 4px solid #f1c40f; border-radius: 5px;">
        <b style="color: #d4ac0d;">🟡 Maladaptation</b><br>
        <span style="color: #000000; font-size: 12px;">
        CV% > threshold + (LnRMSSD baixo ou RHR alto)<br>
        <b>Ação:</b> Reduza intensidade, estresse agudo
        </span>
    </div>
    
    <div style="padding: 10px; background-color: #e74c3c20; border-left: 4px solid #e74c3c; border-radius: 5px;">
        <b style="color: #e74c3c;">🔴 Accumulated Fatigue</b><br>
        <span style="color: #000000; font-size: 12px;">
        CV% ≤ threshold + LnRMSSD suprimido<br>
        <b>Ação:</b> Descanso prioritário, fadiga crônica
        </span>
    </div>
    
    </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HRV-GUIDED TRAINING - EXATAMENTE COMO ESTAVA NO CÓDIGO ORIGINAL
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")
    
    if len(dw) >= 14 and 'hrv' in dw.columns and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data')
        df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'] > 0, np.log(df_hg['hrv']), np.nan)
        df_hg = df_hg.dropna(subset=['LnrMSSD'])
        
        dias_fam = st.slider("Dias baseline rolling", 7, 28, 14, key="hg_baseline")
        df_hg['bm'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg['bs'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg['linf'] = df_hg['bm'] - 0.5 * df_hg['bs']
        df_hg['lsup'] = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs']
        df_hg['intens'] = df_hg.apply(
            lambda r: 'HIIT' if pd.notna(r['bm']) and r['linf'] <= r['LnrMSSD'] <= r['lsup'] 
            else ('Recuperação' if pd.notna(r['bm']) else 'Sem dados'), 
            axis=1
        )
        
        n_hg = st.slider("Dias HRV-Guided", 14, min(len(df_hg), 180), min(60, len(df_hg)), key="hg_dias")
        df_p = df_hg.tail(n_hg).copy()
        
        # Gráfico HRV-Guided - EXATAMENTE COMO ESTAVA
        _fig_hg = go.Figure()
        
        if len(df_p) > 0:
            for intensidade, cor in [('HIIT', '#27ae60'), ('Recuperação', '#f39c12'), ('Sem dados', '#95a5a6')]:
                df_i = df_p[df_p['intens'] == intensidade]
                if len(df_i) > 0:
                    _fig_hg.add_trace(go.Scatter(
                        x=df_i['Data'],
                        y=df_i['LnrMSSD'],
                        mode='markers',
                        name=intensidade,
                        marker=dict(color=cor, size=10, line=dict(width=1, color='white')),
                        hovertemplate=f'<b>{intensidade}</b><br>Data: %{{x}}<br>LnRMSSD: %{{y:.3f}}<extra></extra>'
                    ))
            
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'], y=df_p['lsup'], mode='lines',
                name='Limite Superior', line=dict(color='#27ae60', width=1, dash='dash'),
                hoverinfo='skip'
            ))
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'], y=df_p['linf'], mode='lines',
                name='Limite Inferior', line=dict(color='#27ae60', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(39, 174, 96, 0.1)',
                hoverinfo='skip'
            ))
            _fig_hg.add_trace(go.Scatter(
                x=df_p['Data'], y=df_p['bm'], mode='lines',
                name='Baseline', line=dict(color='#34495e', width=2)
            ))
        
        # CORREÇÃO APLICADA AQUI: titlefont → title=dict()
        _fig_hg.update_layout(
            title=dict(text='HRV-Guided Training', font=dict(color='#000000')),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#000000'),
            margin=dict(t=50, b=70, l=60, r=40),
            height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#000000')),
            xaxis=dict(
                showgrid=True, 
                gridcolor='#e0e0e0', 
                tickfont=dict(color='#000000'),
                title=dict(font=dict(color='#000000'))  # CORRIGIDO
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#e0e0e0', 
                tickfont=dict(color='#000000'), 
                title=dict(text='LnRMSSD', font=dict(color='#000000'))  # CORRIGIDO
            )
        )
        
        st.plotly_chart(_fig_hg, use_container_width=True, 
                       config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, 
                       key="recovery_hg_chart")
        
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum()
            rec_n = (df_val['intens'] == 'Recuperação').sum()
            total_n = len(df_val)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            c2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            c3.metric("Prescrição HOJE", 
                     '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT' else '🟠 Recuperação')
    
    # BPE - EXATAMENTE COMO ESTAVA
    st.markdown("---")
    st.subheader("📊 BPE — Z-Score Semanal (Método SWC)")
    
    mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] 
                if m in dw.columns and dw[m].notna().any()]
    n_semanas_disp = max(1, len(dw) // 7)
    _skip_bpe = n_semanas_disp < 4
    
    if _skip_bpe:
        st.info(f"Dados insuficientes para BPE (min 4 semanas, disponivel: {n_semanas_disp}).")
    else:
        _slider_max = min(52, n_semanas_disp)
        _slider_val = min(16, _slider_max)
        if _slider_max > 4:
            n_sem = st.slider("Semanas (BPE)", 4, _slider_max, _slider_val, key="bpe_sem")
        else:
            n_sem = _slider_max
        
        dados_bpe = {}
        for met in mets_bpe:
            s = calcular_bpe(dw, met, 60)
            if len(s) > 0: 
                dados_bpe[met] = s.tail(n_sem)
        
        if dados_bpe:
            semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
            nm = len(dados_bpe)
            mat = np.zeros((nm, len(semanas)))
            nomes_bpe = {
                'hrv': 'HRV', 
                'rhr': 'RHR (inv)', 
                'sleep_quality': 'Sono',
                'fatiga': 'Energia', 
                'stress': 'Relaxamento'
            }
            
            for i, met in enumerate(dados_bpe.keys()):
                z = dados_bpe[met]['zscore'].values
                mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
            
            import numpy as _npIM
            _zIM = [[float(mat[r][c]) if not _npIM.isnan(mat[r][c]) else None
                     for c in range(mat.shape[1])] for r in range(mat.shape[0])]
            _yIM = [nomes_bpe.get(m, m) for m in dados_bpe.keys()]
            
            _fIM = go.Figure(go.Heatmap(
                z=_zIM,
                x=[str(s) for s in semanas],
                y=_yIM, 
                colorscale='RdYlGn', 
                zmid=0, 
                zmin=-2, 
                zmax=2,
                colorbar=dict(title='Z', tickfont=dict(color='#000000'))
            ))
            
            _fIM.update_layout(
                paper_bgcolor='white', 
                plot_bgcolor='white', 
                font=dict(color='#000000'),
                margin=dict(t=50, b=70, l=55, r=20), 
                height=max(280, len(_yIM)*35),
                title=dict(text='BPE — Z-Score com SWC', font=dict(size=13, color='#000000')),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='#000000')),
                yaxis=dict(tickfont=dict(size=9, color='#000000'))
            )
            
            st.plotly_chart(_fIM, use_container_width=True, 
                           config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, 
                           key="recovery_bpe_chart")
