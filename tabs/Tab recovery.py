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
    _fig_gen = go.Figure()
    _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
        legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
    st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
    # TODO: chart content (converted from matplotlib)

    st.subheader("📊 HRV com Normal Range")
    _fig_gen = go.Figure()
    _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
        legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
    st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
    # TODO: chart content (converted from matplotlib)

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
        for i, met in enumerate(dados_bpe.keys()):
            z = dados_bpe[met]['zscore'].values; mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
        cmap = LinearSegmentedColormap.from_list('bpe', [CORES['vermelho'], CORES['amarelo'], CORES['verde']], N=100)
        import numpy as _npIM
        _zIM = [[float(mat[r][c]) if not _npIM.isnan(mat[r][c]) else None
                 for c in range(mat.shape[1])] for r in range(mat.shape[0])]
        _yIM = list(nomes.values()) if 'nomes' in dir() else [str(i) for i in range(mat.shape[0])]
        _fIM = go.Figure(go.Heatmap(z=_zIM,
            x=[str(s) for s in semanas.index] if 'semanas' in dir() else [],
            y=_yIM, colorscale='RdYlGn', zmid=0, zmin=-2, zmax=2,
            colorbar=dict(title='Z', tickfont=dict(color='#111'))))
        _fIM.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=max(280, len(_yIM)*35),
            title=dict(text='BPE — Z-Score com SWC', font=dict(size=13,color='#111')),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9,color='#111')),
            yaxis=dict(tickfont=dict(size=9,color='#111')))
        st.plotly_chart(_fIM, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

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
        df_p = df_hg.tail(n_hg).copy(); xh = range(len(df_p)); dh = df_p['Data'].dt.strftime('%d/%m'); sh = max(1, len(xh) // 15)
        ch = [CORES['verde'] if i == 'HIIT' else CORES['laranja'] if i == 'Recuperação' else CORES['cinza'] for i in df_p['intens']]
        _fig_gen = go.Figure()
        _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
        # TODO: chart content (converted from matplotlib)
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum(); rec_n = (df_val['intens'] == 'Recuperação').sum(); total_n = len(df_val)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            c2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            c3.metric("Prescrição HOJE", '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT' else '🟠 Recuperação')

# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — WELLNESS
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_wellness.py
# ════════════════════════════════════════════════════════════════════════════
