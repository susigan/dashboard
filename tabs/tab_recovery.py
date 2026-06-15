from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import sys, os as _os
from scipy import stats
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

def tab_recovery(dw, da=None, wc_full=None, da_full=None):
    st.header("🔋 Recovery Score & HRV Analysis")
    if da is None:
        da = pd.DataFrame()
    if len(dw) == 0 or 'hrv' not in dw.columns:
        st.warning("Sem dados de HRV.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # MODELO β
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🧠 Modelo β — Estado Autonómico Integrado")
    st.caption(
        "Baseado em Della Mattia (ednacore AI, 2025): *VFC y Sistema Nervioso Autónomo*. "
        "O SNA responde a factores invisíveis (glucogénio, osmolaridade, microinflamação) "
        "antes da percepção consciente. A HRV sozinha num único dia não é accionável — "
        "a tendência de 3 e 7 dias é."
    )

    def _calcular_modelo_beta(wc_src):
        import scipy.stats as _sst
        src = wc_src.copy() if wc_src is not None and len(wc_src) > 0 else dw.copy()
        src['Data'] = pd.to_datetime(src['Data'])
        src = src.sort_values('Data').set_index('Data')
        date_range = pd.date_range(src.index.min(), src.index.max(), freq='D')
        src = src.reindex(date_range)
        if 'hrv' not in src.columns:
            return None
        src['LnrMSSD'] = np.where(src['hrv'].notna() & (src['hrv'] > 0), np.log(src['hrv']), np.nan)
        src['bm28'] = src['LnrMSSD'].rolling(28, min_periods=7).mean()
        src['bs28'] = src['LnrMSSD'].rolling(28, min_periods=7).std()
        src['z28'] = (src['LnrMSSD'] - src['bm28']) / src['bs28'].replace(0, np.nan)
        src['beta'] = src['z28'].apply(lambda z: round(float(_sst.norm.cdf(z) * 100), 1) if pd.notna(z) else np.nan)
        m3 = src['LnrMSSD'].rolling(3, min_periods=2).mean()
        m7 = src['LnrMSSD'].rolling(7, min_periods=4).mean()
        src['beta_agudo'] = np.where(m7.notna() & m3.notna() & (m7 != 0), ((m3 - m7) / m7.abs()) * 100, np.nan)
        src['beta_cronico'] = np.where(src['bm28'].notna() & m7.notna() & (src['bm28'] != 0), ((m7 - src['bm28']) / src['bm28'].abs()) * 100, np.nan)
        return src[['LnrMSSD', 'bm28', 'bs28', 'beta', 'beta_agudo', 'beta_cronico']].tail(90)

    def _regra_convergencia(beta, b_agudo, b_cronico, hrv_hoje_notna):
        sinais = []
        if pd.isna(beta): sinais.append(('β actual', 0, 'NaN — sem medição hoje', '#888'))
        elif beta >= 60: sinais.append(('β actual', +1, f'{beta:.0f} ≥ 60 ✅', '#27ae60'))
        elif beta <= 40: sinais.append(('β actual', -1, f'{beta:.0f} ≤ 40 ⚠️', '#e74c3c'))
        else: sinais.append(('β actual', 0, f'{beta:.0f} zona neutra (40-60)', '#f39c12'))
        if pd.isna(b_agudo): sinais.append(('βAgudo 3d', 0, 'NaN — dados insuficientes', '#888'))
        elif b_agudo >= 1.0: sinais.append(('βAgudo 3d', +1, f'{b_agudo:+.1f}% ≥ +1% ✅', '#27ae60'))
        elif b_agudo <= -1.0: sinais.append(('βAgudo 3d', -1, f'{b_agudo:+.1f}% ≤ -1% ⚠️', '#e74c3c'))
        else: sinais.append(('βAgudo 3d', 0, f'{b_agudo:+.1f}% zona neutra', '#f39c12'))
        if pd.isna(b_cronico): sinais.append(('βCrónico 7d', 0, 'NaN — dados insuficientes', '#888'))
        elif b_cronico >= 1.0: sinais.append(('βCrónico 7d', +1, f'{b_cronico:+.1f}% ≥ +1% ✅', '#27ae60'))
        elif b_cronico <= -1.0: sinais.append(('βCrónico 7d', -1, f'{b_cronico:+.1f}% ≤ -1% ⚠️', '#e74c3c'))
        else: sinais.append(('βCrónico 7d', 0, f'{b_cronico:+.1f}% zona neutra', '#f39c12'))
        n_pos = sum(1 for _, s, _, _ in sinais if s == +1)
        n_neg = sum(1 for _, s, _, _ in sinais if s == -1)
        n_inc = sum(1 for _, s, _, _ in sinais if s == 0)
        if not hrv_hoje_notna:
            return "⚠️ SEM MEDIÇÃO HOJE — Não prescrever HIIT", "#e67e22", n_pos, n_neg, n_inc, sinais
        if n_pos >= 2: return "✅ HIIT / Alta intensidade — ≥2 sinais positivos", "#27ae60", n_pos, n_neg, n_inc, sinais
        elif n_neg >= 2: return "🔴 Recuperação activa — ≥2 sinais negativos", "#e74c3c", n_pos, n_neg, n_inc, sinais
        elif n_neg >= 1 and n_inc >= 1: return "🟠 Sessão moderada Z1/Z2 — 1 sinal negativo + incerteza", "#e67e22", n_pos, n_neg, n_inc, sinais
        elif n_pos == 1 and n_inc >= 2: return "🟡 Sessão moderada Z1/Z2 — sinais insuficientes para HIIT", "#f39c12", n_pos, n_neg, n_inc, sinais
        else: return "🟡 Zona neutra — manter intensidade planeada", "#f39c12", n_pos, n_neg, n_inc, sinais

    beta_df = _calcular_modelo_beta(wc_full)
    if beta_df is None or beta_df.empty or beta_df['beta'].isna().all():
        st.info("Dados insuficientes para calcular Modelo β (mínimo 14 dias de HRV).")
    else:
        ult = beta_df.iloc[-1]
        beta_hoje = ult['beta']; b_agudo_hoje = ult['beta_agudo']; b_cron_hoje = ult['beta_cronico']
        data_ultimo_idx = beta_df.index[-1]; hrv_hoje_notna = pd.notna(ult['LnrMSSD'])
        ultima_med = beta_df['LnrMSSD'].dropna(); dias_sem_medicao = 0
        if not ultima_med.empty:
            ultima_data_med = ultima_med.index[-1]
            dias_sem_medicao = (data_ultimo_idx - ultima_data_med).days
        if dias_sem_medicao > 0:
            st.error(f"⚠️ **ATENÇÃO — {dias_sem_medicao} dia(s) sem medição de HRV.** Última medição: {ultima_data_med.strftime('%d/%m/%Y')}. Sem sinal autonómico actual, o sistema não pode confirmar estado de readiness. **Não prescrever HIIT por precaução.**")
        prescricao, cor_pres, n_pos, n_neg, n_inc, sinais_detalhe = _regra_convergencia(beta_hoje, b_agudo_hoje, b_cron_hoje, hrv_hoje_notna)
        cb1, cb2, cb3, cb4 = st.columns(4)
        beta_label = f"{beta_hoje:.0f}/100" if pd.notna(beta_hoje) else "— (sem dados)"
        beta_delta = ("Alta frescura ✅" if pd.notna(beta_hoje) and beta_hoje >= 65 else ("Zona funcional" if pd.notna(beta_hoje) and beta_hoje >= 50 else ("Possível fadiga ⚠️" if pd.notna(beta_hoje) else "Sem medição hoje")))
        cb1.metric("β Frescura actual", beta_label, delta=beta_delta, delta_color="normal" if pd.notna(beta_hoje) and beta_hoje >= 50 else "inverse")
        ba_label = f"{b_agudo_hoje:+.1f}%" if pd.notna(b_agudo_hoje) else "— (NaN)"
        ba_delta = ("Tendência +3d ↗" if pd.notna(b_agudo_hoje) and b_agudo_hoje >= 1 else ("Estável" if pd.notna(b_agudo_hoje) and b_agudo_hoje >= -1 else ("Queda aguda ↘ ⚠️" if pd.notna(b_agudo_hoje) else "Incerto")))
        cb2.metric("βAgudo (3d)", ba_label, delta=ba_delta, delta_color="normal" if pd.notna(b_agudo_hoje) and b_agudo_hoje >= 0 else "inverse")
        bc_label = f"{b_cron_hoje:+.1f}%" if pd.notna(b_cron_hoje) else "— (NaN)"
        bc_delta = ("Adaptação positiva ↗" if pd.notna(b_cron_hoje) and b_cron_hoje >= 1 else ("Estável" if pd.notna(b_cron_hoje) and b_cron_hoje >= -1 else ("Declínio crónico ↘ ⚠️" if pd.notna(b_cron_hoje) else "Incerto")))
        cb3.metric("βCrónico (7d)", bc_label, delta=bc_delta, delta_color="normal" if pd.notna(b_cron_hoje) and b_cron_hoje >= 0 else "inverse")
        cb4.metric("Sinais convergentes", f"{max(n_pos, n_neg)}/3", delta=f"+{n_pos} pos | -{n_neg} neg | ~{n_inc} inc", delta_color="normal" if n_pos >= 2 else ("inverse" if n_neg >= 2 else "off"))
        h_r, h_g, h_b = int(cor_pres[1:3], 16), int(cor_pres[3:5], 16), int(cor_pres[5:7], 16)
        st.markdown(f'<div style="padding:16px 20px;border-radius:10px;margin:12px 0;background:rgba({h_r},{h_g},{h_b},0.10);border-left:6px solid {cor_pres};"><span style="font-size:1.15em;font-weight:700;color:{cor_pres};">Prescrição Modelo β: {prescricao}</span></div>', unsafe_allow_html=True)
        with st.expander("🔍 Detalhe dos 3 indicadores β", expanded=False):
            st.markdown("**Regra de decisão:** actuar apenas quando ≥2 dos 3 indicadores convergem na mesma direcção.")
            for nome, sinal, desc, cor_s in sinais_detalhe:
                hs_r, hs_g, hs_b = int(cor_s[1:3], 16), int(cor_s[3:5], 16), int(cor_s[5:7], 16)
                icone = "✅" if sinal == +1 else ("⚠️" if sinal == -1 else "⬜")
                st.markdown(f'<div style="padding:8px 14px;margin:4px 0;border-radius:6px;background:rgba({hs_r},{hs_g},{hs_b},0.10);border-left:4px solid {cor_s};"><b>{icone} {nome}:</b> {desc}</div>', unsafe_allow_html=True)
        st.markdown("#### Evolução β — últimos 90 dias")
        beta_plot = beta_df.dropna(subset=['bm28']).copy(); beta_plot.index.name = 'Data'; beta_plot = beta_plot.reset_index()
        fig_b = go.Figure()
        sem_med = beta_df[beta_df['LnrMSSD'].isna()].copy().reset_index()
        for _, row_sm in sem_med.iterrows():
            fig_b.add_vrect(x0=row_sm['Data'] - pd.Timedelta(hours=12), x1=row_sm['Data'] + pd.Timedelta(hours=12), fillcolor="rgba(150,150,150,0.15)", line_width=0)
        fig_b.add_hrect(y0=65, y1=100, fillcolor="rgba(39,174,96,0.07)", line_width=0, annotation_text="Alta frescura (>65)", annotation_position="left", annotation_font_size=10, annotation_font_color="#27ae60")
        fig_b.add_hrect(y0=40, y1=65, fillcolor="rgba(243,156,18,0.05)", line_width=0, annotation_text="Zona funcional", annotation_position="left", annotation_font_size=10, annotation_font_color="#f39c12")
        fig_b.add_hrect(y0=0, y1=40, fillcolor="rgba(231,76,60,0.07)", line_width=0, annotation_text="Fadiga possível (<40)", annotation_position="left", annotation_font_size=10, annotation_font_color="#e74c3c")
        fig_b.add_trace(go.Scatter(x=beta_plot['Data'], y=beta_plot['beta'], mode='lines+markers', name='β (frescura)', line=dict(color='#2471A3', width=2.5), marker=dict(size=6), hovertemplate='%{x|%d/%m/%Y}<br>β: <b>%{y:.0f}</b><extra></extra>'))
        fig_b.add_trace(go.Scatter(x=beta_plot['Data'], y=beta_plot['beta_agudo'], mode='lines', name='βAgudo 3d (%)', line=dict(color='#E74C3C', width=1.5, dash='dot'), yaxis='y2', hovertemplate='%{x|%d/%m/%Y}<br>βAgudo: <b>%{y:+.1f}%</b><extra></extra>'))
        fig_b.add_trace(go.Scatter(x=beta_plot['Data'], y=beta_plot['beta_cronico'], mode='lines', name='βCrónico 7d (%)', line=dict(color='#9B59B6', width=1.5, dash='dash'), yaxis='y2', hovertemplate='%{x|%d/%m/%Y}<br>βCrónico: <b>%{y:+.1f}%</b><extra></extra>'))
        fig_b.add_hline(y=0, line_dash='solid', line_color='rgba(150,150,150,0.4)', line_width=1, yref='y2')
        fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=12), height=420, hovermode='x unified', margin=dict(t=40, b=70, l=60, r=80), legend=dict(orientation='h', y=-0.18, font=dict(size=10), bgcolor='rgba(0,0,0,0)'), yaxis=dict(title='β (0–100)', range=[0, 100], showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#2471A3'), title_font=dict(color='#2471A3')), yaxis2=dict(title='βAgudo / βCrónico (%)', overlaying='y', side='right', showgrid=False, zeroline=True, zerolinecolor='rgba(150,150,150,0.4)', tickfont=dict(color='#888'), title_font=dict(color='#888')), xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict()))
        st.plotly_chart(fig_b, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="rec_beta_chart")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # INTEGRAR RPE
    # ════════════════════════════════════════════════════════════════════════
    def _remove_outliers_iqr(series, factor=1.5):
        s = pd.to_numeric(series, errors='coerce'); q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
        mask = (s < q1 - factor*iqr) | (s > q3 + factor*iqr); s[mask] = np.nan; return s

    def classificar_rpe(val):
        if pd.isna(val): return None
        if val < 5: return 'Leve'
        elif val < 7: return 'Moderado'
        else: return 'Pesado'

    rpe_col = next((c for c in ['icu_rpe', 'rpe', 'RPE'] if c in da.columns), None) if not da.empty else None
    if rpe_col:
        da_proc = da.copy(); da_proc['Data'] = pd.to_datetime(da_proc['Data']).dt.normalize(); dw['Data'] = pd.to_datetime(dw['Data']).dt.normalize()
        da_proc[rpe_col] = _remove_outliers_iqr(da_proc[rpe_col]); da_proc = da_proc.dropna(subset=[rpe_col])
        rpe_diario = da_proc.groupby('Data')[rpe_col].agg([('icu_rpe', 'mean'), ('icu_rpe_max', 'max'), ('treinos_count', 'count')]).reset_index()
        rpe_diario['rpe_cat'] = rpe_diario['icu_rpe'].apply(classificar_rpe)
        dw = dw.merge(rpe_diario, on='Data', how='left')
        dw['treino_pesado'] = (dw['icu_rpe'] >= 7).astype(int); dw['treino_moderado'] = ((dw['icu_rpe'] >= 5) & (dw['icu_rpe'] < 7)).astype(int)
        dw['treino_leve'] = (dw['icu_rpe'] < 5).astype(int); dw['descanso'] = dw['icu_rpe'].isna().astype(int)
        st.success(f"✅ Dados de carga integrados: {dw['icu_rpe'].notna().sum()} dias com RPE")
    else:
        dw['icu_rpe'] = np.nan; dw['icu_rpe_max'] = np.nan; dw['treinos_count'] = 0
        dw['rpe_cat'] = 'Sem dados'; dw['treino_pesado'] = 0; dw['treino_moderado'] = 0; dw['treino_leve'] = 0; dw['descanso'] = 1
        if not da.empty: st.info("ℹ️ Atividades disponíveis mas sem coluna de RPE (icu_rpe/rpe)")

    rec = calcular_recovery(dw)
    if len(rec) == 0: return
    u = rec.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{u['recovery_score']:.0f}")
    c2.metric("HRV", f"{u['hrv']:.0f}" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline", f"{u['hrv_baseline']:.0f}" if pd.notna(u['hrv_baseline']) else "—")
    _cv7 = u.get('hrv_cv_7d', None)
    c4.metric("CV%", f"{_cv7:.1f}%" if _cv7 is not None else "—")
    st.markdown("---")

    col1, col2 = st.columns(2)
    n_dias = col1.slider("Dias", 14, min(len(dw), 365), 90, key="rec_dias")
    janela_cv = col2.slider("Janela CV", 3, 14, 7, key="rec_jcv")
    modo_modelo = st.radio("Modelo", ["Mode 1 — Altini", "Mode 2 — Plews"], horizontal=True, key="rec_modo")

    df = dw.copy().sort_values('Data'); df['Data'] = pd.to_datetime(df['Data']); df = df.tail(n_dias)
    df['LnrMSSD'] = np.where(df['hrv'] > 0, np.log(df['hrv']), np.nan); df = df.dropna(subset=['LnrMSSD'])
    if len(df) < 10: st.warning("Poucos dados."); return

    baseline_w = 7 if "Mode 1" in modo_modelo else 60
    df['baseline'] = df['LnrMSSD'].rolling(baseline_w, min_periods=5).mean()
    df['std'] = df['LnrMSSD'].rolling(baseline_w, min_periods=5).std()
    df['cv'] = (df['LnrMSSD'].rolling(janela_cv, min_periods=3).std() / df['LnrMSSD'].rolling(janela_cv, min_periods=3).mean()) * 100
    df['SWC'] = 0.5 * (df['std'] / df['baseline'] * 100)
    df['upper'] = df['baseline'] * (1 + df['SWC'] / 100); df['lower'] = df['baseline'] * (1 - df['SWC'] / 100)
    cv_hist = df['cv'].dropna()
    if len(cv_hist) > 10:
        cv_mean = cv_hist.mean(); cv_std = cv_hist.std()
        cv_low = max(0.1, cv_mean - 0.5 * cv_std); cv_high = cv_mean + 0.5 * cv_std
    else: cv_low, cv_high = 0.5, 1.5

    def slope_fn(x):
        xd = x.dropna(); return stats.linregress(range(len(xd)), xd)[0] if len(xd) >= 5 else np.nan

    df['slope'] = df['LnrMSSD'].rolling(7, min_periods=5).apply(slope_fn)

    def altini(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']): return 'Sem dados', '#808080'
        if r['LnrMSSD'] < r['baseline'] and r['cv'] < cv_low: return 'Accumulated Fatigue', '#e74c3c'
        if r['LnrMSSD'] < r['baseline'] and r['cv'] > cv_high: return 'Maladaptation', '#f1c40f'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] < cv_low: return 'Good Adaptation', '#27ae60'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] > cv_high: return 'High Variability', '#2c3e50'
        return 'Normal', '#95a5a6'

    def plews(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']): return 'Sem dados', '#808080'
        declinio = r['slope'] < -0.01 if pd.notna(r['slope']) else False
        if r['cv'] < cv_low and declinio: return 'NFOR', '#8b0000'
        if r['LnrMSSD'] < r['lower']: return 'Overreaching', '#e67e22'
        if r['cv'] > cv_high: return 'High Variability', '#2c3e50'
        return 'Normal', '#27ae60'

    if "Mode 1" in modo_modelo: df[['zona', 'cor']] = df.apply(lambda r: pd.Series(altini(r)), axis=1)
    else: df[['zona', 'cor']] = df.apply(lambda r: pd.Series(plews(r)), axis=1)
    df_plot = df.dropna(subset=['baseline', 'cv'])
    if len(df_plot) == 0: st.warning("Sem dados suficientes após processamento."); return

    # ════════════════════════════════════════════════════════════════════════
    # GRÁFICO PRINCIPAL
    # ════════════════════════════════════════════════════════════════════════
    fig = go.Figure()
    zonas_ordem = (['Good Adaptation', 'Normal', 'High Variability', 'Maladaptation', 'Accumulated Fatigue', 'Sem dados'] if "Mode 1" in modo_modelo else ['Normal', 'High Variability', 'Overreaching', 'NFOR', 'Sem dados'])
    zonas_vistas = df_plot[['zona', 'cor']].drop_duplicates().set_index('zona')['cor'].to_dict()
    for zona in zonas_ordem:
        if zona not in zonas_vistas: continue
        cor = zonas_vistas[zona]; d = df_plot[df_plot['zona'] == zona]
        r_h, g_h, b_h = int(cor[1:3],16), int(cor[3:5],16), int(cor[5:7],16)
        fig.add_trace(go.Bar(x=d['Data'], y=d['LnrMSSD'], name=zona, marker=dict(color=f'rgba({r_h},{g_h},{b_h},0.55)', line=dict(color=f'rgba({r_h},{g_h},{b_h},0.85)', width=1)), customdata=np.stack([d['cv'], d['slope'].fillna(0)], axis=1), hovertemplate='<b>' + zona + '</b><br>Data: %{x|%d/%m/%Y}<br>LnRMSSD: %{y:.3f}<br>CV%: %{customdata[0]:.2f}%<br>Slope 7d: %{customdata[1]:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['baseline'], name=f'Baseline ({baseline_w}d)', line=dict(color='#2c3e50', width=4, dash='dash'), hovertemplate='Baseline: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['upper'], line=dict(color='rgba(44,62,80,0.40)', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['lower'], fill='tonexty', fillcolor='rgba(44,62,80,0.20)', line=dict(color='rgba(44,62,80,0.40)', width=1), name='SWC band', hovertemplate='SWC lower: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['cv'], name='CV% (eixo direito)', line=dict(color='#e67e22', width=2), marker=dict(size=4), yaxis='y2', hovertemplate='CV%: %{y:.2f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=[df_plot['Data'].iloc[0], df_plot['Data'].iloc[-1]], y=[cv_low, cv_low], name=f'CV low ({cv_low:.2f}%)', yaxis='y2', line=dict(color='#e67e22', width=3, dash='dot'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[df_plot['Data'].iloc[0], df_plot['Data'].iloc[-1]], y=[cv_high, cv_high], name=f'CV high ({cv_high:.2f}%)', yaxis='y2', line=dict(color='#c0392b', width=3, dash='dot'), hoverinfo='skip'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=12), height=500, barmode='relative', hovermode='x unified', margin=dict(t=60, b=80, l=60, r=80), title=dict(text=f'{"Mode 1 — Altini" if "Mode 1" in modo_modelo else "Mode 2 — Plews"} | Baseline {baseline_w}d | CV thresholds: low={cv_low:.2f}% / high={cv_high:.2f}%', font=dict(size=13)), legend=dict(orientation='h', y=-0.22, font=dict(size=10), bgcolor='rgba(0,0,0,0)'), yaxis=dict(title='LnRMSSD', showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(), range=[0, 8], dtick=1), yaxis2=dict(title=f'CV% ({janela_cv}d)', overlaying='y', side='right', showgrid=False, tickfont=dict(color='#e67e22'), title_font=dict(color='#e67e22'), range=[0, max(3.0, df_plot['cv'].max() * 1.3)]), xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict()))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="rec_main_chart")

    ultimo = df_plot.iloc[-1]
    st.markdown("### 📊 Status Atual")
    cs1, cs2, cs3, cs4 = st.columns(4)
    cor_s = ultimo['cor']; r_h, g_h, b_h = int(cor_s[1:3],16), int(cor_s[3:5],16), int(cor_s[5:7],16)
    cs1.markdown(f'<div style="padding:12px;border-radius:8px;background:rgba({r_h},{g_h},{b_h},0.15);border-left:5px solid {cor_s};"><b style="color:{cor_s};font-size:14px;">{ultimo["zona"]}</b></div>', unsafe_allow_html=True)
    cs2.metric("CV%", f"{ultimo['cv']:.2f}%", delta=f"low:{cv_low:.2f}% | high:{cv_high:.2f}%")
    cs3.metric("LnRMSSD", f"{ultimo['LnrMSSD']:.3f}", delta=f"baseline: {ultimo['baseline']:.3f}")
    cs4.metric("Slope 7d", f"{ultimo['slope']:.4f}" if pd.notna(ultimo['slope']) else "—")

    with st.expander("📖 Como foi calculado este resultado?", expanded=True):
        if "Mode 1" in modo_modelo:
            st.markdown("""**Mode 1 — Altini (Baseline Curto)**

| Condição | Significado |
|----------|-------------|
| **Accumulated Fatigue** | LnRMSSD < Baseline + CV < low |
| **Maladaptation** | LnRMSSD < Baseline + CV > high |
| **Good Adaptation** | LnRMSSD > Baseline + CV < low |
| **High Variability** | LnRMSSD > Baseline + CV > high |
| **Normal** | Valores intermediários |

- **Baseline**: Média móvel {baseline_w}d | **CV%**: DP/média × 100 ({janela_cv}d) | **Thresholds CV**: Média histórica ± 0.5 DP
- **Status atual**: {zona} (CV={cv:.2f}%, baseline={base:.3f})""".format(baseline_w=baseline_w, janela_cv=janela_cv, zona=ultimo['zona'], cv=ultimo['cv'], base=ultimo['baseline']))
        else:
            st.markdown("""**Mode 2 — Plews (Baseline Longo)**

| Condição | Significado |
|----------|-------------|
| **NFOR** | CV < low + Slope negativo |
| **Overreaching** | LnRMSSD < Lower SWC |
| **High Variability** | CV > high |
| **Normal** | Dentro dos parâmetros normais |

- **Baseline**: Média móvel {baseline_w}d | **SWC**: 0.5 × (DP/baseline) × 100 | **Slope 7d**: regressão linear
- **Status atual**: {zona} (Slope={slope:.4f}, SWC lower={lower:.3f})""".format(baseline_w=baseline_w, zona=ultimo['zona'], slope=ultimo['slope'] if pd.notna(ultimo['slope']) else 0, lower=ultimo['lower']))

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # HRV-GUIDED TRAINING
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")
    if len(dw) >= 14 and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data'); df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg = df_hg.set_index('Data')
        date_range_hg = pd.date_range(df_hg.index.min(), df_hg.index.max(), freq='D')
        df_hg = df_hg.reindex(date_range_hg); df_hg.index.name = 'Data'; df_hg = df_hg.reset_index()
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'].notna() & (df_hg['hrv'] > 0), np.log(df_hg['hrv']), np.nan)
        df_hg['sem_medicao'] = df_hg['LnrMSSD'].isna()

        hg_c1, hg_c2 = st.columns(2)
        dias_fam = hg_c1.slider("Dias baseline rolling", 7, 28, 14, key="hg_baseline")
        n_hg = hg_c2.slider("Dias a mostrar", 14, min(len(df_hg), 180), min(60, len(df_hg)), key="hg_dias")

        _mp = max(5, dias_fam // 2)
        df_hg['bm'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=_mp).mean()
        df_hg['bs'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=_mp).std()
        df_hg['linf'] = df_hg['bm'] - 0.5 * df_hg['bs']; df_hg['lsup'] = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio_dp'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs'].replace(0, np.nan)

        def _classif_hg(r):
            if r['sem_medicao']: return 'Sem medição ⭐'
            if pd.isna(r['bm']): return 'Sem dados'
            if r['linf'] <= r['LnrMSSD'] <= r['lsup']: return 'HIIT'
            return 'Recuperação'

        df_hg['intens'] = df_hg.apply(_classif_hg, axis=1)
        df_analysis = df_hg.copy()

        df_hg_raw = dw.copy().sort_values('Data'); df_hg_raw['Data'] = pd.to_datetime(df_hg_raw['Data'])
        df_hg_raw['RMSSD'] = df_hg_raw['hrv'].where(df_hg_raw['hrv'] > 0)
        df_hg_raw = df_hg_raw.set_index('Data')
        _dr_raw = pd.date_range(df_hg_raw.index.min(), df_hg_raw.index.max(), freq='D')
        df_hg_raw = df_hg_raw.reindex(_dr_raw); df_hg_raw.index.name = 'Data'; df_hg_raw = df_hg_raw.reset_index()
        _mp_raw = max(5, dias_fam // 2)
        df_hg_raw['bm_raw'] = df_hg_raw['RMSSD'].rolling(dias_fam, min_periods=_mp_raw).mean()
        df_hg_raw['bs_raw'] = df_hg_raw['RMSSD'].rolling(dias_fam, min_periods=_mp_raw).std()
        df_hg_raw['desvio_dp_raw'] = (df_hg_raw['RMSSD'] - df_hg_raw['bm_raw']) / df_hg_raw['bs_raw'].replace(0, np.nan)
        df_hg_raw['intens_raw'] = df_hg_raw.apply(lambda r: ('Sem medição ⭐' if pd.isna(r['RMSSD']) else 'HIIT' if pd.notna(r['bm_raw']) and (r['bm_raw'] - 0.5*r['bs_raw']) <= r['RMSSD'] <= (r['bm_raw'] + 0.5*r['bs_raw']) else ('Recuperação' if pd.notna(r['bm_raw']) else 'Sem dados')), axis=1)

        df_p = df_hg.tail(n_hg).copy()
        COR_MAP = {'HIIT': '#27ae60', 'Recuperação': '#f39c12', 'Sem dados': '#95a5a6', 'Sem medição ⭐': '#cccccc'}
        _fig_hg = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06, subplot_titles=[f'LnrMSSD — Baseline {dias_fam}d ± 0.5 DP', 'Desvio do Baseline (unidades de DP) — Range ±5'])
        for intensidade, cor in COR_MAP.items():
            df_i = df_p[df_p['intens'] == intensidade]
            if len(df_i) == 0: continue
            if intensidade == 'Sem medição ⭐':
                y_vals = df_i['bm'].fillna(df_i['LnrMSSD'].mean())
                _fig_hg.add_trace(go.Scatter(x=df_i['Data'], y=y_vals, mode='markers', name='Sem medição ⭐', marker=dict(color='#aaaaaa', size=14, symbol='star', line=dict(width=1.5, color='white')), hovertemplate='<b>⭐ Sem medição HRV</b><br>%{x|%d/%m/%Y}<br>Prescrição: Recuperação (por precaução)<extra></extra>'), row=1, col=1)
            else:
                _fig_hg.add_trace(go.Scatter(x=df_i['Data'], y=df_i['LnrMSSD'], mode='markers', name=intensidade, marker=dict(color=cor, size=10, line=dict(width=1.5, color='white')), hovertemplate=f'<b>{intensidade}</b><br>%{{x|%d/%m}}: %{{y:.3f}}<extra></extra>'), row=1, col=1)
        _fig_hg.add_trace(go.Scatter(x=df_p['Data'], y=df_p['lsup'], line=dict(color='rgba(39,174,96,0.4)', width=1), showlegend=False, hoverinfo='skip'), row=1, col=1)
        _fig_hg.add_trace(go.Scatter(x=df_p['Data'], y=df_p['linf'], fill='tonexty', fillcolor='rgba(39,174,96,0.12)', line=dict(color='rgba(39,174,96,0.4)', width=1), name='Zona HIIT (±0.5 DP)', hoverinfo='skip'), row=1, col=1)
        _fig_hg.add_trace(go.Scatter(x=df_p['Data'], y=df_p['bm'], name=f'Baseline {dias_fam}d', line=dict(color='#2c3e50', width=2, dash='dash'), hovertemplate='Baseline: %{y:.3f}<extra></extra>'), row=1, col=1)
        _fig_hg.add_hrect(y0=-0.5, y1=0.5, fillcolor='rgba(39,174,96,0.20)', line_width=0, row=2, col=1, annotation_text="Zona HIIT", annotation_position="left")
        _fig_hg.add_hrect(y0=-5, y1=-0.5, fillcolor='rgba(243,156,18,0.10)', line_width=0, row=2, col=1)
        _fig_hg.add_hrect(y0=0.5, y1=5, fillcolor='rgba(243,156,18,0.10)', line_width=0, row=2, col=1)
        _fig_hg.add_hline(y=0, line_dash='solid', line_color='#27ae60', line_width=1.5, row=2, col=1)
        _fig_hg.add_hline(y=0.5, line_dash='dot', line_color='#f39c12', line_width=1, row=2, col=1)
        _fig_hg.add_hline(y=-0.5, line_dash='dot', line_color='#f39c12', line_width=1, row=2, col=1)
        for intensidade, cor in COR_MAP.items():
            df_i = df_p[df_p['intens'] == intensidade]
            if len(df_i) == 0: continue
            _fig_hg.add_trace(go.Scatter(x=df_i['Data'], y=df_i['desvio_dp'], mode='markers', name=intensidade, showlegend=False, marker=dict(color=cor, size=7, opacity=0.8, line=dict(width=1, color='white')), hovertemplate=f'%{{x|%d/%m}}: %{{y:.2f}} DP<extra></extra>'), row=2, col=1)
        _fig_hg.add_trace(go.Scatter(x=df_p['Data'], y=df_p['desvio_dp'], mode='lines', line=dict(color='#7f8c8d', width=1, dash='dot'), showlegend=False, hoverinfo='skip'), row=2, col=1)
        _fig_hg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=11), height=520, hovermode='x unified', margin=dict(t=60, b=70, l=60, r=40), legend=dict(orientation='h', y=-0.18, font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
        _fig_hg.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict())
        _fig_hg.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(), row=1, col=1, title_text='LnRMSSD')
        _fig_hg.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(), row=2, col=1, title_text='Desvio (DP)', range=[-5, 5], zeroline=True, zerolinecolor='#27ae60', zerolinewidth=1.5)
        st.plotly_chart(_fig_hg, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="rec_hg_chart")

        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum(); rec_n = (df_val['intens'] == 'Recuperação').sum(); total_n = len(df_val)
            m1, m2, m3 = st.columns(3)
            m1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            m2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            _ultimo_hg = df_val.iloc[-1]; _data_ultimo = _ultimo_hg['Data']
            _data_hoje = pd.Timestamp('today').normalize(); _dias_sem = (_data_hoje - _data_ultimo).days if pd.notna(_data_ultimo) else 999
            if _ultimo_hg['intens'] == 'Sem medição ⭐' or _dias_sem > 0: _pres_hoje = f'⭐ Sem medição ({_dias_sem}d) — Recuperação'
            elif _ultimo_hg['intens'] == 'HIIT': _pres_hoje = '✅ HIIT'
            else: _pres_hoje = '🟠 Recuperação'
            m3.metric("Prescrição HOJE", _pres_hoje)

        # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 1: HIIT LnRMSSD vs Recovery Modes
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Análise 1: HIIT HRV-Guided vs Recovery Modes (14d)")
        df_corr = df_analysis.copy()
        df_corr['baseline_7'] = df_corr['LnrMSSD'].rolling(7, min_periods=5).mean()
        df_corr['baseline_60'] = df_corr['LnrMSSD'].rolling(60, min_periods=30).mean()
        df_corr['std_7'] = df_corr['LnrMSSD'].rolling(7, min_periods=5).std()
        df_corr['std_60'] = df_corr['LnrMSSD'].rolling(60, min_periods=30).std()
        df_corr['cv_7'] = (df_corr['LnrMSSD'].rolling(7, min_periods=3).std() / df_corr['LnrMSSD'].rolling(7, min_periods=3).mean()) * 100
        cv_hist_full = df_corr['cv_7'].dropna()
        if len(cv_hist_full) > 10:
            cv_m = cv_hist_full.mean(); cv_s = cv_hist_full.std()
            cv_l = max(0.1, cv_m - 0.5 * cv_s); cv_h = cv_m + 0.5 * cv_s
        else: cv_l, cv_h = 0.5, 1.5
        df_corr['slope_7'] = df_corr['LnrMSSD'].rolling(7, min_periods=5).apply(slope_fn)
        df_corr['SWC_60'] = 0.5 * (df_corr['std_60'] / df_corr['baseline_60'] * 100)
        df_corr['lower_60'] = df_corr['baseline_60'] * (1 - df_corr['SWC_60'] / 100)

        def altini_full(r):
            if pd.isna(r['cv_7']) or pd.isna(r['baseline_7']): return 'Sem_dados'
            if r['LnrMSSD'] < r['baseline_7'] and r['cv_7'] < cv_l: return 'Accumulated_Fatigue'
            if r['LnrMSSD'] < r['baseline_7'] and r['cv_7'] > cv_h: return 'Maladaptation'
            if r['LnrMSSD'] > r['baseline_7'] and r['cv_7'] < cv_l: return 'Good_Adaptation'
            if r['LnrMSSD'] > r['baseline_7'] and r['cv_7'] > cv_h: return 'High_Variability'
            return 'Normal'

        def plews_full(r):
            if pd.isna(r['cv_7']) or pd.isna(r['baseline_60']): return 'Sem_dados'
            declinio = r['slope_7'] < -0.01 if pd.notna(r['slope_7']) else False
            if r['cv_7'] < cv_l and declinio: return 'NFOR'
            if r['LnrMSSD'] < r['lower_60']: return 'Overreaching'
            if r['cv_7'] > cv_h: return 'High_Variability'
            return 'Normal'

        df_corr['mode1_zone'] = df_corr.apply(altini_full, axis=1)
        df_corr['mode2_zone'] = df_corr.apply(plews_full, axis=1)
        df_corr['hiit_ln'] = (df_corr['intens'] == 'HIIT').astype(int)
        mode1_zones = ['Accumulated_Fatigue', 'Maladaptation', 'Good_Adaptation', 'High_Variability', 'Normal']
        mode2_zones = ['NFOR', 'Overreaching', 'High_Variability', 'Normal']
        for zone in mode1_zones: df_corr[f'm1_{zone}'] = (df_corr['mode1_zone'] == zone).astype(int)
        for zone in mode2_zones: df_corr[f'm2_{zone}'] = (df_corr['mode2_zone'] == zone).astype(int)
        rolling_cols = ['hiit_ln'] + [f'm1_{z}' for z in mode1_zones] + [f'm2_{z}' for z in mode2_zones]
        for col in rolling_cols: df_corr[f'{col}_r14'] = df_corr[col].rolling(14, min_periods=7).mean()
        corr_results = []
        for mode, zones in [('m1', mode1_zones), ('m2', mode2_zones)]:
            mode_name = "Mode 1 (Altini)" if mode == 'm1' else "Mode 2 (Plews)"
            for zone in zones:
                valid_data = df_corr[['hiit_ln_r14', f'{mode}_{zone}_r14']].dropna()
                if len(valid_data) > 10:
                    corr, p_val = stats.pearsonr(valid_data['hiit_ln_r14'], valid_data[f'{mode}_{zone}_r14'])
                    strength = "Forte" if abs(corr) >= 0.7 else ("Moderada" if abs(corr) >= 0.4 else ("Fraca" if abs(corr) >= 0.2 else "Desprezível"))
                    corr_results.append({'Modo': mode_name, 'Evento': zone.replace('_', ' '), 'Correlação': corr, 'Direção': "Positiva" if corr > 0 else "Negativa", 'Força': strength, 'P-valor': p_val, 'Significativo': "Sim" if p_val < 0.05 else "Não", 'N': len(valid_data)})
        if corr_results:
            df_corr_display = pd.DataFrame(corr_results)
            for mode_name in ["Mode 1 (Altini)", "Mode 2 (Plews)"]:
                st.markdown(f"**{mode_name}**")
                df_mode = df_corr_display[df_corr_display['Modo'] == mode_name].copy()
                def color_forca(val):
                    if val == "Forte": return "background-color: rgba(231, 76, 60, 0.3); color: #c0392b; font-weight: bold"
                    elif val == "Moderada": return "background-color: rgba(241, 196, 15, 0.3)"
                    return ""
                st.dataframe(df_mode[['Evento', 'Correlação', 'Direção', 'Força', 'P-valor', 'Significativo']].style.map(color_forca, subset=['Força']).format({'Correlação': '{:.3f}', 'P-valor': '{:.4f}'}), use_container_width=True, hide_index=True)
        else: st.info("Dados insuficientes para calcular correlações.")

        # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 1.7: Slope 7d Individualizado
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📉 Slope 7d Individualizado + Persistência do Estado")
        slope_series = df_corr['slope_7'].dropna()
        if len(slope_series) > 20:
            slope_mean = slope_series.mean(); slope_std = slope_series.std(); swc = 0.5 * slope_std
            thresh_recuperacao = slope_mean + swc; thresh_estavel_sup = slope_mean + (0.5 * swc)
            thresh_estavel_inf = slope_mean - (0.5 * swc); thresh_declinio = slope_mean - swc; thresh_nfor = slope_mean - (2 * slope_std)
            def classificar_zona_slope(val):
                if pd.isna(val): return 'Sem dados'
                elif val >= thresh_recuperacao: return 'Supercompensação'
                elif val >= thresh_estavel_sup: return 'Recuperação'
                elif val >= thresh_estavel_inf: return 'Estável'
                elif val >= thresh_declinio: return 'Declínio Leve'
                elif val >= thresh_nfor: return 'Fadiga'
                else: return 'NFOR Crítico'
            df_corr['zona_slope'] = df_corr['slope_7'].apply(classificar_zona_slope)
            zona_atual = df_corr['zona_slope'].iloc[-1]; dias_na_zona = 0
            for i in range(len(df_corr) - 1, -1, -1):
                if df_corr['zona_slope'].iloc[i] == zona_atual: dias_na_zona += 1
                else: break
            todas_sequencias = []; seq_atual = 1; zona_anterior = df_corr['zona_slope'].iloc[0] if len(df_corr) > 0 else None
            for i in range(1, len(df_corr)):
                if df_corr['zona_slope'].iloc[i] == zona_anterior: seq_atual += 1
                else:
                    if zona_anterior != 'Sem dados': todas_sequencias.append({'zona': zona_anterior, 'dias': seq_atual})
                    seq_atual = 1; zona_anterior = df_corr['zona_slope'].iloc[i]
            if zona_anterior and zona_anterior != 'Sem dados': todas_sequencias.append({'zona': zona_anterior, 'dias': seq_atual})
            df_seq = pd.DataFrame(todas_sequencias); stats_seq = {}
            if not df_seq.empty:
                for zona in df_seq['zona'].unique():
                    dados_zona = df_seq[df_seq['zona'] == zona]['dias']
                    stats_seq[zona] = {'media': dados_zona.mean(), 'max': dados_zona.max(), 'atual': dias_na_zona if zona == zona_atual else 0}
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1: st.metric("Zona Atual", zona_atual)
            with col_p2: st.metric("Dias nesta Zona", f"{dias_na_zona} dias", delta=f"Seu recorde: {stats_seq.get(zona_atual, {}).get('max', 'N/A')} dias" if zona_atual in stats_seq else None)
            with col_p3:
                _slope_now = df_corr['slope_7'].dropna().iloc[-1] if df_corr['slope_7'].notna().any() else 0.0
                st.metric("Slope Atual", f"{_slope_now:.4f}")
            with col_p4:
                media_historica = stats_seq.get(zona_atual, {}).get('media', 0)
                if media_historica > 0:
                    razao = dias_na_zona / media_historica
                    st.metric("vs Sua Média", f"{razao:.1f}x", help=f"Você está nesta zona há {razao:.1f} vezes a sua média histórica ({media_historica:.1f} dias)")
            st.markdown("---")
            st.markdown("**🎯 Recomendação Baseada na Persistência do Estado**")
            if zona_atual == 'NFOR Crítico':
                if dias_na_zona >= 3: st.error(f"🚨 **AÇÃO IMEDIATA**: NFOR há **{dias_na_zona} dias consecutivos**. Descanso COMPLETO por pelo menos {dias_na_zona + 2} dias.")
                else: st.warning(f"⚠️ **NFOR Detectado ({dias_na_zona} dias)**. Reduzir carga em 50% imediatamente.")
            elif zona_atual == 'Fadiga':
                if dias_na_zona >= 5: st.error(f"🚨 **Fadiga Persistente ({dias_na_zona} dias)**. Descanso ativo por 3-5 dias.")
                elif dias_na_zona >= 3: st.warning(f"⚠️ **Declínio Prolongado ({dias_na_zona} dias)**. Reduzir intensidade em 30%.")
                else: st.info(f"ℹ️ **Fadiga Aguda ({dias_na_zona} dia{'s' if dias_na_zona > 1 else ''})**. Normal após treino intenso. Monitorar 48h.")
            elif zona_atual == 'Declínio Leve':
                if dias_na_zona >= 7: st.warning(f"⚠️ **Declínio Crónico Leve ({dias_na_zona} dias)**. 2-3 dias de recuperação antes de nova carga.")
                else: st.success(f"✅ **Estável ({dias_na_zona} dias)**. Dentro da variação normal.")
            elif zona_atual in ['Recuperação', 'Supercompensação']:
                if dias_na_zona >= 2: st.success(f"🚀 **Pronto para Carga! ({dias_na_zona} dias)**. Período ótimo para alta intensidade.")
                else: st.info("ℹ️ **Início de Recuperação**. Aguarde mais 1 dia antes de carga alta.")
            _df_slope_plot = df_corr[df_corr['slope_7'].notna()].copy()
            _slope_start = _df_slope_plot['Data'].min() if len(_df_slope_plot) > 0 else pd.Timestamp.now().normalize() - pd.Timedelta(days=86)
            _df_slope_30 = df_corr[df_corr['Data'] >= _slope_start].copy()
            fig_slope_ind = go.Figure()
            fig_slope_ind.add_trace(go.Scatter(x=_df_slope_30['Data'], y=_df_slope_30['slope_7'], name='Slope 7d', line=dict(color='#2c3e50', width=2), fill='tozeroy', fillcolor='rgba(44, 62, 80, 0.1)', hovertemplate='Data: %{x}<br>Slope: %{y:.4f}<extra></extra>'))
            fig_slope_ind.add_hrect(y0=thresh_recuperacao, y1=slope_mean + (3*slope_std), fillcolor="rgba(39, 174, 96, 0.15)", line_width=0, annotation_text="Supercompensação", annotation_position="top right", annotation_font=dict(color='black', size=11))
            fig_slope_ind.add_hrect(y0=thresh_estavel_sup, y1=thresh_recuperacao, fillcolor="rgba(46, 204, 113, 0.15)", line_width=0, annotation_text="Recuperação", annotation_position="right", annotation_font=dict(color='black', size=11))
            fig_slope_ind.add_hrect(y0=thresh_estavel_inf, y1=thresh_estavel_sup, fillcolor="rgba(149, 165, 166, 0.15)", line_width=0, annotation_text="Estável", annotation_position="right", annotation_font=dict(color='black', size=11))
            fig_slope_ind.add_hrect(y0=thresh_declinio, y1=thresh_estavel_inf, fillcolor="rgba(241, 196, 15, 0.15)", line_width=0, annotation_text="Declínio Leve", annotation_position="right", annotation_font=dict(color='black', size=11))
            fig_slope_ind.add_hrect(y0=thresh_nfor, y1=thresh_declinio, fillcolor="rgba(231, 76, 60, 0.15)", line_width=0, annotation_text="Fadiga", annotation_position="right", annotation_font=dict(color='black', size=11))
            fig_slope_ind.add_hrect(y0=slope_mean - (4*slope_std), y1=thresh_nfor, fillcolor="rgba(192, 57, 43, 0.2)", line_width=0, annotation_text="NFOR", annotation_position="bottom right", annotation_font=dict(color='black', size=11))
            fig_slope_ind.add_hline(y=slope_mean, line_dash="solid", line_color="blue", line_width=2, annotation_text=f"Sua Média ({slope_mean:.3f})", annotation_position="right", annotation_font=dict(color='black', size=10))
            fig_slope_ind.add_hline(y=thresh_nfor, line_dash="dash", line_color="red", line_width=3, annotation_text=f"Limite NFOR ({thresh_nfor:.3f})", annotation_position="bottom right", annotation_font=dict(color='black', size=10))
            fig_slope_ind.update_layout(title=dict(text=f"Slope 7d Individualizado - Baseado em {len(slope_series)} dias", font=dict(color='black', size=14)), xaxis=dict(title=dict(text="Data", font=dict(color='black')), tickfont=dict(color='black'), showgrid=True, gridcolor='rgba(128,128,128,0.2)'), yaxis=dict(title=dict(text="Slope 7d do LnRMSSD", font=dict(color='black')), tickfont=dict(color='black'), showgrid=True, gridcolor='rgba(128,128,128,0.2)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, showlegend=True, legend=dict(font=dict(color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='black', borderwidth=1), font=dict(color='black'))
            st.plotly_chart(fig_slope_ind, use_container_width=True, config={'displayModeBar': False})
            with st.expander("📋 Ver Histórico de Sequências (Streaks)"):
                df_hist_seq = pd.DataFrame(todas_sequencias[-10:])
                if not df_hist_seq.empty:
                    df_hist_seq['Data Fim'] = [df_corr['Data'].iloc[-(sum([s['dias'] for s in todas_sequencias[i:]]) if i < len(todas_sequencias)-1 else 0) - 1].strftime('%d/%m/%Y') for i in range(max(0, len(todas_sequencias)-10), len(todas_sequencias))]
                    st.dataframe(df_hist_seq[['Data Fim', 'zona', 'dias']].rename(columns={'zona': 'Zona', 'dias': 'Duração (dias)'}), use_container_width=True, hide_index=True)
        else: st.info(f"Dados insuficientes ({len(slope_series)} dias, mínimo 20) para análise individualizada.")

        # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 2: Javaloyes et al. — SEM SLIDERS AQUI
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Análise 2: HRV-Guided Protocol — Javaloyes et al.")
        st.caption("Protocolo publicado: Javaloyes A et al. (2018/2020). LnRMSSD rolling 7 dias vs SWC (mean ± 0.5×SD do baseline). Máx 2 sessões consecutivas de alta intensidade. Máx 2 dias REST consecutivos.")
        with st.expander("📖 Metodologia — como funciona", expanded=False):
            st.markdown("""**Protocolo Javaloyes et al. 2018/2020**
1. **Métrica**: LnRMSSD rolling 7 dias
2. **Banda SWC**: mean ± 0.5 × SD (primeiros 28 dias reais)
3. **Decisão**: Acima SWC sup → HIGH | Dentro → LOW | Abaixo SWC inf → REST
4. **Restrições**: Máx 2 HIGH consecutivos → força LOW | Máx 2 REST consecutivos → força LOW""")

        # Usar wc_full directamente — só datas reais de medição (sem reindex fictício)
        _wc_j = (wc_full.copy() if (wc_full is not None and len(wc_full) > 0) else dw.copy())
        _wc_j['Data'] = pd.to_datetime(_wc_j['Data']).dt.normalize()
        _wc_j = _wc_j.sort_values('Data').reset_index(drop=True)
        _wc_j['LnrMSSD'] = np.where(_wc_j['hrv'].notna() & (_wc_j['hrv'] > 0), np.log(_wc_j['hrv']), np.nan)
        _wc_j['sem_medicao'] = _wc_j['LnrMSSD'].isna()

        if _wc_j['LnrMSSD'].notna().sum() < 14:
            st.info("Dados insuficientes para o protocolo Javaloyes (mínimo 14 dias).")
        else:
            _wc_j['ln7'] = _wc_j['LnrMSSD'].rolling(7, min_periods=4).mean()

            # Aviso apenas se hoje não tem medição
            _hoje_ts = pd.Timestamp.now().normalize()
            _ultima_med_jav = _wc_j[_wc_j['LnrMSSD'].notna()]['Data'].max()
            _dias_sem_hoje = int((_hoje_ts - _ultima_med_jav).days) if pd.notna(_ultima_med_jav) else 999
            if _dias_sem_hoje >= 1:
                st.warning(f"⚠️ Última medição HRV há {_dias_sem_hoje} dia(s). LnRMSSD₇ de hoje é estimado.")

            # SWC: primeiros 28 dias COM medição real
            _ln_real = _wc_j[_wc_j['LnrMSSD'].notna()]['LnrMSSD']
            _ln_base = _ln_real.head(28) if len(_ln_real) >= 7 else _ln_real
            _swc_mean = float(_ln_base.mean()); _swc_sd = float(_ln_base.std())
            _swc_sup = _swc_mean + 0.5 * _swc_sd; _swc_inf = _swc_mean - 0.5 * _swc_sd

            # Máquina de estados
            _prescricoes = []; _consec_high = 0; _consec_rest = 0; _razoes = []
            for _, row in _wc_j.iterrows():
                ln7 = row['ln7']
                if pd.isna(ln7):
                    _pres = 'LOW'; _razao = 'sem dados'; _consec_high = 0; _consec_rest = 0
                elif ln7 > _swc_sup:
                    if _consec_high >= 2:
                        _pres = 'LOW'; _razao = 'HIGH forçado LOW (máx 2 consec.)'; _consec_high = 0; _consec_rest = 0
                    else:
                        _pres = 'HIGH'; _razao = 'Acima SWC sup'; _consec_high += 1; _consec_rest = 0
                elif ln7 >= _swc_inf:
                    _pres = 'LOW'; _razao = 'Dentro da banda'; _consec_high = 0; _consec_rest = 0
                else:
                    _consec_high = 0
                    if _consec_rest >= 2:
                        _pres = 'LOW'; _razao = 'REST forçado LOW (máx 2 consec.)'; _consec_rest = 0
                    else:
                        _pres = 'REST'; _razao = 'Abaixo SWC inf'; _consec_rest += 1
                _prescricoes.append(_pres); _razoes.append(_razao)
            _wc_j['prescricao'] = _prescricoes; _wc_j['razao'] = _razoes

            _COR_MAP = {'HIGH': '#27ae60', 'LOW': '#3498db', 'REST': '#e74c3c'}
            _LABEL_MAP = {'HIGH': '🟢 HIGH — treino intenso', 'LOW': '🔵 LOW — treino leve', 'REST': '🔴 REST — descanso activo'}

            _wc_j_val = _wc_j[_wc_j['ln7'].notna()]
            if len(_wc_j_val) > 0:
                _ult_j = _wc_j_val.iloc[-1]; _pres_hoje = _ult_j['prescricao']
                _ln7_hoje = _ult_j['ln7']; _sem_hoje = bool(_ult_j.get('sem_medicao', False))
                _cj1, _cj2, _cj3, _cj4 = st.columns(4)
                _cj1.metric("LnRMSSD₇ avg", f"{_ln7_hoje:.3f}" + (" ⚠️" if _sem_hoje else ""), help="Rolling 7 dias. ⚠️ = estimado.")
                _cj2.metric("SWC superior", f"{_swc_sup:.3f}", help=f"mean {_swc_mean:.3f} + 0.5×SD → HIGH acima daqui")
                _cj3.metric("SWC inferior", f"{_swc_inf:.3f}", help=f"mean {_swc_mean:.3f} - 0.5×SD → REST abaixo daqui")
                _cj4.metric("Prescrição HOJE", _LABEL_MAP.get(_pres_hoje, _pres_hoje))
                _cor_j = _COR_MAP.get(_pres_hoje, '#888')
                _rj, _gj, _bj = int(_cor_j[1:3],16), int(_cor_j[3:5],16), int(_cor_j[5:7],16)
                st.markdown(f'<div style="padding:14px 20px;border-radius:8px;margin:8px 0;background:rgba({_rj},{_gj},{_bj},0.10);border-left:5px solid {_cor_j};"><b style="font-size:1.1em;color:{_cor_j};">{_LABEL_MAP.get(_pres_hoje, _pres_hoje)}</b><span style="color:#888;margin-left:12px;font-size:0.9em;">LnRMSSD₇={_ln7_hoje:.3f} | SWC [{_swc_inf:.3f}, {_swc_sup:.3f}]</span></div>', unsafe_allow_html=True)

            # Tabela últimos 5 dias
            _ult5_j = _wc_j_val.tail(5).copy(); _rows5_j = []
            for _, r in _ult5_j.iterrows():
                _sem = bool(r.get('sem_medicao', False))
                _rows5_j.append({'Data': r['Data'].strftime('%d/%m') + (' ⚠️' if _sem else ''), 'LnRMSSD₇': f"{r['ln7']:.3f}" + (' (est.)' if _sem else ''), 'Zona': r.get('razao') or '—', 'Prescrição': _LABEL_MAP.get(r['prescricao'], r['prescricao'])})
            st.markdown("**Últimos 5 dias — protocolo Javaloyes:**")
            st.dataframe(pd.DataFrame(_rows5_j), hide_index=True, use_container_width=True)
            st.caption(f"SWC baseline (primeiros 28 dias reais): mean={_swc_mean:.3f} ± 0.5×SD={0.5*_swc_sd:.3f} → banda [{_swc_inf:.3f}, {_swc_sup:.3f}]. Máx 2 HIGH consecutivos → força LOW. Máx 2 REST consecutivos → força LOW.")

            # Gráfico — usa n_hg do slider do HRV-Guided (mesmo período, sem slider novo)
            _df_plot = _wc_j_val.tail(n_hg).copy()
            import plotly.graph_objects as _go_j
            _fig_j = _go_j.Figure()
            _fig_j.add_hrect(y0=_swc_inf, y1=_swc_sup, fillcolor='rgba(52,152,219,0.08)', line_width=0, annotation_text="LOW (dentro banda)", annotation_position="right", annotation_font_size=9, annotation_font_color='#3498db')
            _fig_j.add_hline(y=_swc_sup, line_dash='dash', line_color='rgba(39,174,96,0.6)', line_width=1.2, annotation_text="SWC sup", annotation_position="right", annotation_font_color='#27ae60', annotation_font_size=9)
            _fig_j.add_hline(y=_swc_inf, line_dash='dash', line_color='rgba(231,76,60,0.6)', line_width=1.2, annotation_text="SWC inf", annotation_position="right", annotation_font_color='#e74c3c', annotation_font_size=9)
            _fig_j.add_trace(_go_j.Scatter(x=_df_plot['Data'], y=_df_plot['ln7'], mode='lines', line=dict(color='rgba(44,62,80,0.35)', width=1.5), showlegend=False, hoverinfo='skip'))
            for _pg, _cg in _COR_MAP.items():
                _sg = _df_plot[_df_plot['prescricao'] == _pg]
                if len(_sg) > 0:
                    _fig_j.add_trace(_go_j.Scatter(x=_sg['Data'], y=_sg['ln7'], mode='markers', name=_LABEL_MAP[_pg], marker=dict(color=_cg, size=7, line=dict(width=1.5, color='white')), hovertemplate='%{x|%d/%m}<br>LnRMSSD₇: %{y:.3f}<extra></extra>'))
            if 'sem_medicao' in _df_plot.columns:
                _sem_plot = _df_plot[_df_plot['sem_medicao'].astype(bool)]
                if len(_sem_plot) > 0:
                    _fig_j.add_trace(_go_j.Scatter(x=_sem_plot['Data'], y=_sem_plot['ln7'], mode='markers', name='Sem medição', marker=dict(symbol='star', color='gray', size=9, line=dict(width=1, color='white')), hovertemplate='%{x|%d/%m}<br>Estimado: %{y:.3f}<extra></extra>'))
            _fig_j.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=11), hovermode='x unified', margin=dict(t=30, b=60, l=60, r=130), legend=dict(orientation='h', y=-0.22, font=dict(size=10)), title=dict(text='LnRMSSD rolling 7 dias — protocolo Javaloyes/Kiviniemi', font=dict(size=13)), xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'), yaxis=dict(title='LnRMSSD₇', showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
            st.plotly_chart(_fig_j, use_container_width=True, config={'displayModeBar': False}, key='javaloyes_hrv_chart')

            # Correlação Javaloyes vs HRV-Guided
            st.markdown("**Correlação: protocolo Javaloyes vs HRV-Guided (Altini/Plews):**")
            try:
                _df_hg_corr = df_hg[['Data', 'intens']].copy(); _df_hg_corr['Data'] = pd.to_datetime(_df_hg_corr['Data']).dt.normalize()
                _wc_j_corr = _wc_j[_wc_j['ln7'].notna()][['Data', 'prescricao']].copy(); _wc_j_corr['Data'] = pd.to_datetime(_wc_j_corr['Data']).dt.normalize()
                _merged_j = _wc_j_corr.merge(_df_hg_corr, on='Data', how='inner')
                _merged_j = _merged_j[_merged_j['intens'].isin(['HIIT', 'Recuperação'])].copy()
                _merged_j['high_hrv'] = (_merged_j['intens'] == 'HIIT').astype(int)
                _merged_j['high_jav'] = (_merged_j['prescricao'] == 'HIGH').astype(int)
                if len(_merged_j) >= 10:
                    from scipy import stats as _sst_j
                    _n_j = len(_merged_j); _concord_j = int((_merged_j['high_hrv'] == _merged_j['high_jav']).sum()); _pct_j = _concord_j / _n_j * 100
                    _r_j, _p_j = _sst_j.pearsonr(_merged_j['high_hrv'], _merged_j['high_jav'])
                    _ct_j = pd.crosstab(_merged_j['intens'].map({'HIIT': 'HRV-Guided: HIIT', 'Recuperação': 'HRV-Guided: Recuperação'}), _merged_j['prescricao'].map({'HIGH': 'Javaloyes: HIGH', 'LOW': 'Javaloyes: LOW', 'REST': 'Javaloyes: REST'}))
                    st.dataframe(_ct_j, use_container_width=True)
                    _cc1, _cc2, _cc3 = st.columns(3)
                    _cc1.metric("Concordância", f"{_pct_j:.0f}%", f"{_concord_j}/{_n_j} dias")
                    _cc2.metric("Correlação r", f"{_r_j:.3f}", "p<0.05" if _p_j < 0.05 else f"p={_p_j:.3f}")
                    _cc3.metric("Dias analisados", _n_j)
                    _hh = int(((_merged_j['high_hrv']==1)&(_merged_j['high_jav']==1)).sum()); _hr = int(((_merged_j['high_hrv']==1)&(_merged_j['high_jav']==0)).sum())
                    _rh = int(((_merged_j['high_hrv']==0)&(_merged_j['high_jav']==1)).sum()); _rr = int(((_merged_j['high_hrv']==0)&(_merged_j['high_jav']==0)).sum())
                    st.caption(f"Ambos HIGH/HIIT: {_hh}d | Ambos Rec/LOW/REST: {_rr}d | HRV-G HIIT mas Jav LOW/REST: {_hr}d | HRV-G Rec mas Jav HIGH: {_rh}d")
                else: st.info(f"Dados insuficientes ({len(_merged_j)} dias comuns).")
            except Exception as _j_err: st.info(f"Correlação indisponível: {_j_err}")

    # ════════════════════════════════════════════════════════════════════════
    # EXPORT CSV
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📥 Export Recovery CSV")
    st.caption("Histórico completo — independente do filtro sidebar. Inclui todas as análises: Recovery Score, HRV-Guided, Javaloyes, Slope 7d, Modelo β, HF_Power, RPE.")
    try:
        _dw_csv = (wc_full if (wc_full is not None and len(wc_full) > 0) else dw).copy()
        _da_csv = (da_full if (da_full is not None and len(da_full) > 0) else (da if da is not None else pd.DataFrame()))
        _dw_csv['Data'] = pd.to_datetime(_dw_csv['Data']).dt.normalize()
        _rec_dl = calcular_recovery(_dw_csv)
        if len(_rec_dl) == 0: st.info("Sem dados para exportar.")
        else:
            _base_cols = [c for c in ['Data','hrv','rhr','recovery_score','hrv_baseline','hrv_cv_7d','hrv_cv_30d','normal_range_inf','normal_range_sup','hrv_component','rhr_component','sleep_component','fatiga_component','stress_component'] if c in _rec_dl.columns]
            _exp = _rec_dl[_base_cols].copy()
            if 'rhr' not in _exp.columns and 'rhr' in _dw_csv.columns:
                _exp = _exp.merge(_dw_csv[['Data','rhr']].drop_duplicates('Data'), on='Data', how='left')
            _exp['Data'] = pd.to_datetime(_exp['Data']).dt.normalize()
            for _wc in ['sleep_quality','fatiga','stress','humor','soreness','peso','fat','hf_power']:
                if _wc in _dw_csv.columns: _exp = _exp.merge(_dw_csv[['Data',_wc]].drop_duplicates('Data'), on='Data', how='left')
            if not _da_csv.empty:
                _rpe_col = next((c for c in ['icu_rpe','rpe'] if c in _da_csv.columns), None)
                if _rpe_col:
                    _rpe_d = _da_csv.copy(); _rpe_d['Data'] = pd.to_datetime(_rpe_d['Data']).dt.normalize()
                    _rpe_d = _rpe_d.groupby('Data')[_rpe_col].mean().reset_index(); _rpe_d.columns = ['Data','rpe_diario']
                    _exp = _exp.merge(_rpe_d, on='Data', how='left')
            try:
                _hg_exp = df_hg[['Data','LnrMSSD','bm','bs','linf','lsup','desvio_dp','intens']].copy()
                _hg_exp.columns = ['Data','LnrMSSD','HRVg_baseline','HRVg_sd','HRVg_linf','HRVg_lsup','HRVg_desvio','HRVg_prescricao']
                _hg_exp['Data'] = pd.to_datetime(_hg_exp['Data']).dt.normalize()
                _exp = _exp.merge(_hg_exp, on='Data', how='left')
            except Exception: pass
            try:
                _sl_exp = df_corr[['Data','LnrMSSD','slope_7','baseline_7','std_7']].copy()
                _sl_exp.columns = ['Data','LnrMSSD_raw','slope_7d','slope_baseline7','slope_std7']
                _sl_exp['Data'] = pd.to_datetime(_sl_exp['Data']).dt.normalize()
                _exp = _exp.merge(_sl_exp, on='Data', how='left')
            except Exception: pass
            try:
                _jav_exp = _wc_j[['Data','LnrMSSD','ln7','prescricao']].copy()
                _jav_exp.columns = ['Data','LnrMSSD_jav','LnRMSSD7_jav','Javaloyes_prescricao']
                _jav_exp['Data'] = pd.to_datetime(_jav_exp['Data']).dt.normalize()
                _jav_exp['Javaloyes_SWC_sup'] = round(_swc_sup, 4); _jav_exp['Javaloyes_SWC_inf'] = round(_swc_inf, 4)
                _exp = _exp.merge(_jav_exp, on='Data', how='left')
            except Exception: pass
            try:
                _beta_exp = beta_df[['LnrMSSD','beta','bm28','bs28']].copy(); _beta_exp.index.name = 'Data'; _beta_exp = _beta_exp.reset_index()
                _beta_exp['Data'] = pd.to_datetime(_beta_exp['Data']).dt.normalize()
                _beta_exp.columns = ['Data','LnrMSSD_beta','beta_score','beta_baseline28','beta_sd28']
                _exp = _exp.merge(_beta_exp, on='Data', how='left')
            except Exception: pass
            if 'hrv_sem_medicao' in _dw_csv.columns:
                _exp = _exp.merge(_dw_csv[['Data','hrv_sem_medicao']].drop_duplicates('Data'), on='Data', how='left')
            _exp['Data'] = _exp['Data'].astype(str); _exp = _exp.drop_duplicates('Data').sort_values('Data').round(4)
            _c_dl1, _c_dl2 = st.columns([2, 1])
            with _c_dl1:
                st.dataframe(_exp.tail(14), use_container_width=True, hide_index=True)
                st.caption(f"Últimos 14 de {len(_exp)} dias | {len(_exp.columns)} colunas")
            with _c_dl2:
                st.metric("Dias no CSV", len(_exp)); st.metric("Colunas", len(_exp.columns)); st.metric("Sidebar (gráficos)", len(dw))
                st.download_button(label="📥 Download Recovery CSV", data=_exp.to_csv(index=False, sep=';', decimal=',').encode('utf-8'), file_name="atheltica_recovery.csv", mime="text/csv", key="rec_dl_csv")
    except Exception as _rec_err: st.info(f"Export não disponível: {_rec_err}")
