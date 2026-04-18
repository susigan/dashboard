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
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def tab_pmc(da, wc=None):
    """
    PMC — icu_training_load para CTL/ATL/FTLM.
    Barras de Load = TRIMP (session_rpe).
    Tabelas mensais: eFTP por modalidade + KM/KJ por modalidade (como Intervals.icu).
    Gráfico KM semanal stacked com linha de média por modalidade.
    """
    st.header("📈 PMC — Performance Management Chart")
    if len(da) == 0: st.warning("Sem dados de atividades."); return

    da_full = st.session_state.get('da_full', da)
    _mods = st.session_state.get('mods_sel', None)
    if _mods and 'type' in da_full.columns:
        da_full = da_full[da_full['type'].isin(_mods + ['WeightTraining'])]
    df = filtrar_principais(da_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # ── CTL/ATL: icu_training_load primeiro, session_rpe fallback ──
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['icu_tl'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        _metrica_ctl = "icu_training_load (Intervals.icu)"
    elif 'moving_time' in df.columns and 'rpe' in df.columns and df['rpe'].notna().sum() > 10:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['icu_tl'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        _metrica_ctl = "session_rpe (fallback)"
    else:
        st.warning("Sem icu_training_load nem RPE para calcular CTL/ATL."); return

    # ── Load bars: TRIMP = session_rpe ──
    _rpe2 = pd.to_numeric(df['rpe'], errors='coerce') if 'rpe' in df.columns else pd.Series(dtype=float)
    _mt   = pd.to_numeric(df['moving_time'], errors='coerce') if 'moving_time' in df.columns else pd.Series(dtype=float)
    if _rpe2.notna().sum() > 5:
        df['trimp_val'] = (_mt / 60) * _rpe2.fillna(_rpe2.median())
    else:
        df['trimp_val'] = df['icu_tl']

    # ── Série diária CTL/ATL ──
    ld = df.groupby('Data')['icu_tl'].sum().reset_index().sort_values('Data')
    idx_full = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx_full, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']
    ld['CTL']  = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL']  = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB']  = ld['CTL'] - ld['ATL']

    # ── FTLM fraccionário — Della Mattia (2025) Dual-Gamma ──────────────────
    # Usa calcular_series_carga para fitting de γ_perf (icu_pm_cp) e γ_rec (HRV)
    # O bloco CTL/ATL/TSB acima é mantido para compatibilidade e velocidade.
    # O FTLM fraccionário é computado separadamente e adicionado a ld.
    with st.spinner("A calcular FTLM fraccionário (γ dual)..."):
        _ld_frac, _info = calcular_series_carga(da_full, df_wellness=wc, ate_hoje=True)

    if len(_ld_frac) > 0 and 'FTLM' in _ld_frac.columns:
        # Align fractional ld with our ld (may have different date range)
        _frac_idx = _ld_frac.set_index('Data')
        for _col in ['FTLM', 'CTLg_perf', 'CTLg_rec', 'HRV_trend']:
            if _col in _frac_idx.columns:
                ld[_col] = ld['Data'].map(_frac_idx[_col])
        best_g = _info.get('gamma_best', 0.35)
    else:
        # Fallback: classic EWM
        best_g, best_r = 0.30, -1.0
        for g in np.arange(0.25, 0.36, 0.01):
            ema = ld['load_val'].ewm(alpha=g, adjust=False).mean()
            if ema.std() > 0:
                r = abs(np.corrcoef(ld['load_val'].values, ema.values)[0, 1])
                if r > best_r: best_r, best_g = r, g
        ld['FTLM'] = ld['load_val'].ewm(alpha=best_g, adjust=False).mean()
        _info = {'gamma_perf': best_g, 'gamma_rec': best_g, 'fonte': 'fallback',
                 'r2_perf': 0.0, 'r2_rec': 0.0, 'gamma_source': 'classic'}

    u = ld.iloc[-1]

    # Caption with dual-gamma info
    _gp   = _info.get('gamma_perf', best_g)
    _gr   = _info.get('gamma_rec',  best_g)
    _r2p  = _info.get('r2_perf',  0.0)
    _r2r  = _info.get('r2_rec',   0.0)
    _gsrc = _info.get('gamma_source', 'classic')
    _n_mmp= _info.get('n_mmp', 0)
    st.caption(
        f"FTLM fraccionário | Fonte carga: **{_metrica_ctl}** | "
        f"γ_perf={_gp:.3f} (R²={_r2p:.2f}, icu_pm_cp) | "
        f"γ_rec={_gr:.3f} (R²={_r2r:.2f}, HRV trend) | "
        f"γ activo: **{_gsrc}** | MMP PR points: {_n_mmp} | "
        f"Histórico: {len(ld)} dias"
    )

    # ── Controlos ──
    col1, col2, col3 = st.columns(3)
    dias_opts = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
                 "180 dias": 180, "1 ano": 365, "Todo histórico": len(ld)}
    dias_exib = dias_opts[col1.selectbox("Período exibido", list(dias_opts.keys()), index=2)]
    ld_plot   = ld.tail(dias_exib).copy()
    smooth    = col2.checkbox("Suavizar CTL/ATL (3d)", value=False)
    show_ftlm = col3.checkbox("Mostrar FTLM", value=True)
    if smooth:
        ld_plot['CTL'] = ld_plot['CTL'].rolling(3, min_periods=1).mean()
        ld_plot['ATL'] = ld_plot['ATL'].rolling(3, min_periods=1).mean()

    # ── GRÁFICO 1: PMC + Load (TRIMP) ──
    _fig_pmc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.70, 0.30], vertical_spacing=0.04)
    _dates = ld_plot['Data'].tolist()

    _fig_pmc.add_trace(go.Scatter(x=_dates, y=ld_plot['CTL'].tolist(),
        name='CTL (Fitness)', line=dict(color=CORES['azul'], width=2.5),
        hovertemplate='CTL: %{y:.1f}<extra></extra>'), row=1, col=1)
    _fig_pmc.add_trace(go.Scatter(x=_dates, y=ld_plot['ATL'].tolist(),
        name='ATL (Fadiga)', line=dict(color=CORES['vermelho'], width=2.5),
        hovertemplate='ATL: %{y:.1f}<extra></extra>'), row=1, col=1)

    _fig_pmc.add_trace(go.Scatter(x=_dates, y=ld_plot['TSB'].tolist(),
        fill='tozeroy', fillcolor='rgba(39,174,96,0.15)',
        line=dict(color='rgba(39,174,96,0.5)', width=1),
        name='TSB (Forma/Fadiga)',
        hovertemplate='TSB: %{y:.1f}<extra></extra>'), row=1, col=1)
    _fig_pmc.add_hline(y=0, line_dash='dash', line_color='#999', line_width=1, row=1, col=1)

    if show_ftlm:
        _ctl_max  = max(ld_plot['CTL'].max(), ld_plot['ATL'].max(), 1)
        # FTLM fraccionário — γ activo (melhor R²)
        if 'FTLM' in ld_plot.columns and ld_plot['FTLM'].notna().any():
            _ftlm_max = max(ld_plot['FTLM'].max(), 1)
            _ftlm_n   = ld_plot['FTLM'] / _ftlm_max * _ctl_max * 0.85
            _fig_pmc.add_trace(go.Scatter(x=_dates, y=_ftlm_n.tolist(),
                name=f'CTLγ ({_gsrc}, γ={best_g:.3f})',
                line=dict(color=CORES['laranja'], width=2.5, dash='dash'), opacity=0.85,
                hovertemplate='CTLγ: %{y:.1f}<extra></extra>'), row=1, col=1)
        # CTLg_perf (γ_performance, normalised)
        if 'CTLg_perf' in ld_plot.columns and ld_plot['CTLg_perf'].notna().any():
            _gp_max = max(ld_plot['CTLg_perf'].max(), 1)
            _gp_n   = ld_plot['CTLg_perf'] / _gp_max * _ctl_max * 0.75
            _fig_pmc.add_trace(go.Scatter(x=_dates, y=_gp_n.tolist(),
                name=f'CTLγ perf (γ={_gp:.3f}, R²={_r2p:.2f})',
                line=dict(color='#2980b9', width=1.5, dash='dot'), opacity=0.7,
                hovertemplate='CTLγ_perf: %{y:.1f}<extra></extra>'), row=1, col=1)
        # CTLg_rec (γ_recovery, normalised)
        if 'CTLg_rec' in ld_plot.columns and ld_plot['CTLg_rec'].notna().any():
            _gr_max = max(ld_plot['CTLg_rec'].max(), 1)
            _gr_n   = ld_plot['CTLg_rec'] / _gr_max * _ctl_max * 0.75
            _fig_pmc.add_trace(go.Scatter(x=_dates, y=_gr_n.tolist(),
                name=f'CTLγ rec (γ={_gr:.3f}, R²={_r2r:.2f})',
                line=dict(color='#8e44ad', width=1.5, dash='dot'), opacity=0.7,
                hovertemplate='CTLγ_rec: %{y:.1f}<extra></extra>'), row=1, col=1)

    _u_ctl = float(u['CTL']); _u_atl = float(u['ATL']); _u_tsb = float(u['TSB'])
    _fig_pmc.add_annotation(
        x=_dates[-1], y=_u_ctl, xref='x', yref='y',
        text=f"CTL {_u_ctl:.1f} | ATL {_u_atl:.1f} | TSB {_u_tsb:+.1f}",
        showarrow=False, bgcolor='rgba(255,235,200,0.9)',
        bordercolor='#aaa', borderwidth=1,
        font=dict(size=10, color='#111'), xanchor='right', yanchor='top')

    trimp_d = df.groupby(['Data', 'type'])['trimp_val'].sum().reset_index()
    trimp_d['Data'] = pd.to_datetime(trimp_d['Data'])
    tipos_ord = [t for t in ['Bike','Row','Ski','Run','WeightTraining']
                 if t in trimp_d['type'].unique()]
    tipos_ord += [t for t in trimp_d['type'].unique() if t not in tipos_ord]
    for tipo in tipos_ord:
        _dt = trimp_d[trimp_d['type']==tipo][['Data','trimp_val']]
        _merged = ld_plot[['Data']].merge(_dt, on='Data', how='left').fillna(0)
        _fig_pmc.add_trace(go.Bar(
            x=_dates, y=_merged['trimp_val'].tolist(),
            name=tipo, marker_color=get_cor(tipo),
            marker_line_width=0, opacity=0.85,
            hovertemplate=tipo+': %{y:.0f}<extra></extra>'), row=2, col=1)

    _fig_pmc.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='#111', size=11), height=460,
        barmode='stack', hovermode='closest',
        legend=dict(orientation='h', y=-0.15, font=dict(color='#111', size=10),
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', borderwidth=1),
        margin=dict(t=50, b=60, l=55, r=40),
        title=dict(text='PMC — CTL / ATL / TSB' + (' / FTLM' if show_ftlm else ''),
                   font=dict(size=14, color='#111')))
    _fig_pmc.update_xaxes(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'))
    _fig_pmc.update_yaxes(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'))
    _fig_pmc.update_yaxes(title_text='CTL/ATL/TSB', row=1, col=1)
    _fig_pmc.update_yaxes(title_text='Load (TRIMP)', row=2, col=1)
    st.plotly_chart(_fig_pmc, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    # ── RESUMO PMC ──
    st.subheader("📊 Resumo PMC")
    tsb_v = u['TSB']
    if   tsb_v >  25: tsb_i = "🟢 Forma — pronto para competir/esforço máximo"
    elif tsb_v >   5: tsb_i = "🟡 Fresco — bom equilíbrio treino/recuperação"
    elif tsb_v > -10: tsb_i = "🟠 Neutro — zona de manutenção"
    elif tsb_v > -25: tsb_i = "🔴 Fatigado — carga elevada, monitorar recuperação"
    else:              tsb_i = "⛔ Sobrecarregado — reduzir carga imediatamente"
    resumo = pd.DataFrame([
        {'Métrica': 'CTL (Fitness atual)',  'Valor': f"{u['CTL']:.1f}",
         'Interpretação': 'Capacidade aeróbica crónica (42d). Maior = melhor condição.'},
        {'Métrica': 'ATL (Fadiga atual)',   'Valor': f"{u['ATL']:.1f}",
         'Interpretação': 'Fadiga aguda (7d). Maior = mais fatigado recentemente.'},
        {'Métrica': 'TSB (Forma atual)',    'Valor': f"{u['TSB']:+.1f}",
         'Interpretação': tsb_i},
        {'Métrica': 'CTL max histórico',    'Valor': f"{ld['CTL'].max():.1f}",
         'Interpretação': 'Pico de fitness no período carregado.'},
        {'Métrica': 'ATL max histórico',    'Valor': f"{ld['ATL'].max():.1f}",
         'Interpretação': 'Pico de fadiga no período carregado.'},
    ])
    st.dataframe(resumo, width="stretch", hide_index=True)

    # ── FTLM — explicação + resultado atual ──
    st.subheader("🔁 FTLM — Fast Training Load Monitor")
    ftlm_v = u['FTLM']
    ctl_v  = u['CTL']
    pct    = (ftlm_v / ctl_v * 100) if ctl_v > 0 else 0
    if   pct > 110: ftlm_i = "⚠️ Carga muito acima do crónico — risco de overreaching"
    elif pct > 100: ftlm_i = "🔴 Carga acima do CTL — fase de acumulação/sobrecarga"
    elif pct >  90: ftlm_i = "🟡 Ligeiramente abaixo do CTL — manutenção/tapering leve"
    elif pct >  75: ftlm_i = "🟢 Tapering activo — carga a baixar, forma a subir"
    else:            ftlm_i = "⬇️ Destreino — carga muito abaixo do nível crónico"

    with st.expander("📖 O que é o FTLM e como interpretar", expanded=True):
        st.markdown(f"""
**FTLM (Fast Training Load Monitor)** é uma média exponencial da carga diária com
um factor gamma (γ) **optimizado automaticamente** por correlação com os teus dados.

| Parâmetro | Valor actual |
|---|---|
| Gamma (γ) | `{best_g:.3f}` |
| FTLM actual | `{ftlm_v:.1f}` |
| CTL actual | `{ctl_v:.1f}` |
| FTLM / CTL | `{pct:.0f}%` |
| Interpretação | {ftlm_i} |

**Como interpretar:**
- **FTLM > CTL (>100%)** — Carga recente **acima** da capacidade crónica. Bloco de carga intenso. Monitorar recuperação.
- **FTLM ≈ CTL (90–110%)** — Carga estável. Manutenção do nível actual.
- **FTLM < CTL (<90%)** — Carga recente **abaixo** do crónico. Tapering intencional ou destreino.

**Diferença para o ATL:**
O ATL usa sempre span=7 (fixo). O FTLM usa γ=`{best_g:.3f}`
(equivalente a span≈{int(round(2/best_g - 1))}), **optimizado para os teus dados**,
tornando-o mais sensível ao teu padrão específico de treino.
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOLUME & CARGA
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_volume.py
# ════════════════════════════════════════════════════════════════════════════
