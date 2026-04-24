"""
tab_fmt_tensor.py — ATHELTICA
Tab dedicada ao FMT Tensor κ — visualização completa com gráficos,
regimes fisiológicos, e explicação baseada no paper Della Mattia 2019.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from utils.config import *


# ── Regime classification ────────────────────────────────────────────────────
REGIME_CFG = {
    'silent':    {'label': 'Fadiga silenciosa', 'color': '#E8593C',
                  'desc': 'TSB positivo mas κ alto — o TSB "mente". HRV pode confirmar.'},
    'intense':   {'label': 'Acumulação intensa', 'color': '#BA7517',
                  'desc': 'Carga alta, TSB negativo — atleta em bloco de carga.'},
    'super':     {'label': 'Supercompensação',   'color': '#1D9E75',
                  'desc': 'κ baixo + TSB positivo — janela de forma potencial.'},
    'recovery':  {'label': 'Recovery',           'color': '#378ADD',
                  'desc': 'κ baixo + carga a cair — sistema a restaurar.'},
    'normal':    {'label': 'Normal',             'color': '#888780',
                  'desc': 'κ médio — nenhum padrão extremo detectado.'},
    'overreach': {'label': 'Overreach',          'color': '#9B3EA3',
                  'desc': 'κ alto + HRV baixo + TSB negativo — risco clínico.'},
    'none':      {'label': 'Sem dados',          'color': '#b4b2a9', 'desc': ''},
}


def _rolling_slope(series, w=14):
    arr = np.array(series, dtype=np.float64)
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        seg = arr[i - w + 1:i + 1]
        mask = np.isfinite(seg)
        if mask.sum() >= w // 2:
            sl, *_ = linregress(np.arange(w)[mask], seg[mask])
            out[i] = sl
    return out


def _classify_regime(kappa, lam1, tsb, hrv, q25, q75):
    """Classify each day into a physiological regime."""
    n = len(kappa)
    regimes = ['none'] * n
    for i in range(n):
        k = kappa[i]; l = lam1[i]; t = tsb[i]; h = hrv[i]
        if k is None or np.isnan(k): continue
        if np.isnan(t): t = 0
        if k > q75 and h is not None and not np.isnan(h) and h < -1 and t < -10:
            regimes[i] = 'overreach'
        elif k > q75 and t < 0:
            regimes[i] = 'intense'
        elif k > q75 and t > 0:
            regimes[i] = 'silent'
        elif k < q25 and t > 5:
            regimes[i] = 'super'
        elif k < q25:
            regimes[i] = 'recovery'
        else:
            regimes[i] = 'normal'
    return regimes


def tab_fmt_tensor(da, wc=None):
    """
    Tab FMT Tensor κ — visualização completa.
    da   : ac_full DataFrame (actividades completo)
    wc   : wc_full DataFrame (wellness completo)
    """
    from utils.helpers import calcular_series_carga

    st.title("📐 FMT Tensor κ — Curvatura do Estado Fisiológico")
    st.caption("Della Mattia (2019) · ednaLabs · Functional Multidimensional Tensor")

    @st.cache_data(show_spinner="A calcular tensor FMT...", ttl=3600)
    def _cached_fmt(_da_key, _wc_key, da_arg, wc_arg):
        return calcular_series_carga(da_arg, df_wellness=wc_arg, ate_hoje=True)

    _da_key = (str(da['Data'].max()) if 'Data' in da.columns else '', len(da))
    _wc_key = (str(wc['Data'].max()) if wc is not None and 'Data' in wc.columns else '', len(wc) if wc is not None else 0)

    try:
        ld, info = _cached_fmt(_da_key, _wc_key, da, wc)
    except Exception as _e:
        st.error(f"Erro ao calcular tensor: {_e}")
        return

    if ld is None or len(ld) == 0:
        st.info("Dados não disponíveis. Recarrega os dados primeiro.")
        return

    if 'FMT_kappa' not in ld.columns or ld['FMT_kappa'].notna().sum() < 20:
        st.warning("FMT_kappa não disponível — são necessários pelo menos 20 dias com tensor calculado.")
        return

    # ── Prepare signals ───────────────────────────────────────────────────────
    kappa_s  = pd.to_numeric(ld['FMT_kappa'], errors='coerce')
    lam1_s   = pd.to_numeric(ld.get('FMT_lambda1_frac', pd.Series()), errors='coerce') \
               if 'FMT_lambda1_frac' in ld.columns else pd.Series(np.nan, index=ld.index)
    tsb_s    = pd.to_numeric(ld['TSB'],       errors='coerce')
    hrv_s    = pd.to_numeric(ld.get('HRV_trend', pd.Series()), errors='coerce') \
               if 'HRV_trend' in ld.columns else pd.Series(np.nan, index=ld.index)
    weed_s   = pd.to_numeric(ld.get('WEED_z',    pd.Series()), errors='coerce') \
               if 'WEED_z'    in ld.columns else pd.Series(np.nan, index=ld.index)
    sleep_s  = pd.to_numeric(ld.get('sleep_z',   pd.Series()), errors='coerce') \
               if 'sleep_z'   in ld.columns else pd.Series(np.nan, index=ld.index)
    ws_s     = pd.to_numeric(ld.get('w_stress',  pd.Series()), errors='coerce') \
               if 'w_stress'  in ld.columns else pd.Series(np.nan, index=ld.index)
    hq_s     = pd.to_numeric(ld.get('hq_drift_z',pd.Series()), errors='coerce') \
               if 'hq_drift_z' in ld.columns else pd.Series(np.nan, index=ld.index)
    ctl_s    = pd.to_numeric(ld['CTL'], errors='coerce')
    dates    = pd.to_datetime(ld['Data'])

    # Percentile thresholds from full history
    q25 = float(kappa_s.quantile(0.25))
    q50 = float(kappa_s.quantile(0.50))
    q75 = float(kappa_s.quantile(0.75))

    # Last values
    kappa_now = float(kappa_s.dropna().iloc[-1])
    lam1_now  = float(lam1_s.dropna().iloc[-1]) if lam1_s.notna().any() else None
    tsb_now   = float(tsb_s.iloc[-1])

    # Rolling slope of kappa (14d)
    kappa_slope = _rolling_slope(kappa_s.values, 14)

    # Regime classification
    kappa_arr = kappa_s.values.tolist()
    lam1_arr  = lam1_s.values.tolist()
    tsb_arr   = tsb_s.values.tolist()
    hrv_arr   = hrv_s.values.tolist()
    regimes   = _classify_regime(
        [x if (x is not None and not (isinstance(x,float) and np.isnan(x))) else None for x in kappa_arr],
        [x if (x is not None and not (isinstance(x,float) and np.isnan(x))) else None for x in lam1_arr],
        tsb_arr, hrv_arr, q25, q75
    )

    # ── Period selector ──────────────────────────────────────────────────────
    st.markdown("---")
    _period = st.selectbox("Período de análise", ['90 dias','180 dias','365 dias','Tudo'],
                           index=1, key='fmt_period')
    _ndays = {'90 dias':90,'180 dias':180,'365 dias':365,'Tudo':len(ld)}[_period]
    _sl = slice(max(0, len(ld)-_ndays), len(ld))

    dates_plot   = dates.iloc[_sl]
    kappa_plot   = kappa_s.iloc[_sl]
    lam1_plot    = lam1_s.iloc[_sl]
    tsb_plot     = tsb_s.iloc[_sl]
    ctl_plot     = ctl_s.iloc[_sl]
    hrv_plot     = hrv_s.iloc[_sl]
    regimes_plot = regimes[_sl.start:_sl.stop]
    slope_plot   = kappa_slope[_sl.start:_sl.stop]

    today_regime = regimes[-1]

    # ── SECTION 1: Summary cards ─────────────────────────────────────────────
    st.subheader("📊 Estado actual")
    _c1,_c2,_c3,_c4 = st.columns(4)

    with _c1:
        _kp_lbl = "alto ⚠️" if kappa_now > q75 else ("baixo ✅" if kappa_now < q25 else "médio")
        st.metric("κ actual", f"{kappa_now:.3f}",
                  delta=f"p{int(100*kappa_s.rank(pct=True).iloc[-1]):.0f} — {_kp_lbl}",
                  help="Curvatura escalar — trace(cov(Δx)). Alto=dimensões a mudar em simultâneo")

    with _c2:
        if lam1_now is not None:
            _lam_pct = lam1_now * 100
            _lam_lbl = "Focal" if lam1_now > 0.65 else ("Misto" if lam1_now > 0.45 else "Multissistémico")
            st.metric("λ₁/Σλ", f"{_lam_pct:.0f}%", delta=_lam_lbl,
                      help="Fracção do stress na dimensão dominante. Focal=1 dim domina")
        else:
            st.metric("λ₁/Σλ", "N/D")

    with _c3:
        _rc = REGIME_CFG[today_regime]
        st.markdown(
            f"<div style='background:{_rc['color']}22;border-left:3px solid {_rc['color']};"
            f"padding:8px 10px;border-radius:5px'>"
            f"<small style='color:var(--text-color)'>Regime hoje</small><br>"
            f"<b style='color:{_rc['color']}'>{_rc['label']}</b></div>",
            unsafe_allow_html=True)

    with _c4:
        _tensor_d = info.get('tensor_dim', 0)
        _tensor_n = info.get('tensor_dim_names','CTLγ')
        st.metric("Tensor", f"{_tensor_d}×{_tensor_d}",
                  delta=_tensor_n.replace('·','/'),
                  help=f"Dimensões activas: {_tensor_n}")

    # ── SECTION 2: κ timeline with regime background ──────────────────────────
    st.markdown("---")
    st.subheader("📈 Evolução de κ e regimes")

    _fig_kappa = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.65, 0.35], vertical_spacing=0.04)

    # Regime background bands on row=1
    _prev_r = None; _band_s = None; _prev_col = '#888780'
    for _di, (_dt, _rg) in enumerate(zip(dates_plot, regimes_plot)):
        if _rg != _prev_r:
            if _band_s is not None:
                _hx = _prev_col.lstrip('#')
                _r2,_g2,_b2 = int(_hx[0:2],16),int(_hx[2:4],16),int(_hx[4:6],16)
                _fig_kappa.add_shape(type='rect', x0=_band_s, x1=_dt,
                    y0=0, y1=1, xref='x', yref='paper',
                    fillcolor=f'rgba({_r2},{_g2},{_b2},0.12)',
                    line_width=0, layer='below')
            _band_s = _dt; _prev_r = _rg; _prev_col = REGIME_CFG.get(_rg,{}).get('color','#888780')
    if _band_s is not None:
        _hx = _prev_col.lstrip('#')
        _r2,_g2,_b2 = int(_hx[0:2],16),int(_hx[2:4],16),int(_hx[4:6],16)
        _fig_kappa.add_shape(type='rect', x0=_band_s, x1=dates_plot.iloc[-1],
            y0=0, y1=1, xref='x', yref='paper',
            fillcolor=f'rgba({_r2},{_g2},{_b2},0.12)',
            line_width=0, layer='below')

    # κ line
    _fig_kappa.add_trace(go.Scatter(
        x=dates_plot, y=kappa_plot,
        name='κ', line=dict(color=CORES['azul'], width=2),
        hovertemplate='κ: %{y:.3f}<extra></extra>'), row=1, col=1)

    # Percentile reference lines
    for _qv, _qlbl, _qcol in [(q75,'p75',CORES['vermelho']),(q50,'p50',CORES['cinza']),(q25,'p25',CORES['verde'])]:
        _fig_kappa.add_hline(y=_qv, line_dash='dot', line_color=_qcol,
                             line_width=1, annotation_text=f'  {_qlbl}={_qv:.2f}',
                             annotation_font_size=9, row=1, col=1)

    # κ slope (14d) on row=2
    _slope_s = pd.Series(slope_plot, index=dates_plot.values)
    _colors_slope = [CORES['vermelho'] if s > 0 else CORES['verde']
                     for s in _slope_s.fillna(0)]
    _fig_kappa.add_trace(go.Bar(
        x=dates_plot, y=_slope_s,
        name='Δκ/14d', marker_color=_colors_slope,
        hovertemplate='Δκ/d: %{y:.4f}<extra></extra>'), row=2, col=1)
    _fig_kappa.add_hline(y=0, line_color='gray', line_width=0.5, row=2, col=1)

    _fig_kappa.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=380,
        margin=dict(t=10, b=50, l=50, r=20), hovermode='x unified',
        legend=dict(orientation='h', y=-0.18, font=dict(color='#111', size=9)),
        font=dict(color='#111', size=10))
    _fig_kappa.update_xaxes(tickfont=dict(size=9, color='#111'),
                             linecolor='#333', tickcolor='#333', tickangle=-30)
    _fig_kappa.update_yaxes(tickfont=dict(size=9, color='#111'),
                             linecolor='#333', tickcolor='#333')
    _fig_kappa.update_yaxes(title_text='κ', row=1, col=1,
                             title_font=dict(size=9, color='#111'))
    _fig_kappa.update_yaxes(title_text='Δκ/d', row=2, col=1,
                             title_font=dict(size=9, color='#111'))
    st.plotly_chart(_fig_kappa, use_container_width=True,
                    config={'displayModeBar':False}, key='fmt_kappa_chart')

    # ── SECTION 3: κ vs TSB scatter (regime map) ──────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Mapa de regimes — κ vs TSB")
    st.caption("Cada ponto = 1 dia. Cor = regime fisiológico detectado.")

    _fig_scatter = go.Figure()
    for _rk, _rcfg in REGIME_CFG.items():
        if _rk == 'none': continue
        _mask = [r == _rk for r in regimes_plot]
        _kdats = [kappa_plot.iloc[i] for i, m in enumerate(_mask) if m]
        _tdats = [tsb_plot.iloc[i]   for i, m in enumerate(_mask) if m]
        _ddats = [str(dates_plot.iloc[i])[:10] for i, m in enumerate(_mask) if m]
        if not _kdats: continue
        _fig_scatter.add_trace(go.Scatter(
            x=_tdats, y=_kdats, mode='markers',
            name=_rcfg['label'],
            marker=dict(color=_rcfg['color'], size=5, opacity=0.75),
            text=_ddats,
            hovertemplate='%{text}<br>TSB: %{x:.1f} | κ: %{y:.3f}<extra></extra>'))

    # Today marker
    _fig_scatter.add_trace(go.Scatter(
        x=[tsb_now], y=[kappa_now], mode='markers',
        name='Hoje',
        marker=dict(color='black', size=12, symbol='star',
                    line=dict(color='white', width=1.5)),
        hovertemplate=f'Hoje: TSB={tsb_now:.1f} | κ={kappa_now:.3f}<extra></extra>'))

    # Quadrant lines
    _fig_scatter.add_vline(x=0, line_color='gray', line_width=0.5, line_dash='dot')
    _fig_scatter.add_hline(y=q75, line_color=CORES['vermelho'], line_width=0.5, line_dash='dot')
    _fig_scatter.add_hline(y=q25, line_color=CORES['verde'],    line_width=0.5, line_dash='dot')

    # Annotations for quadrants
    _tsb_min = float(tsb_plot.min())
    _tsb_max = float(tsb_plot.max())
    for _xann, _yann, _txt, _col in [
        (_tsb_max, q75+0.1, 'Fadiga silenciosa', CORES['vermelho']),
        (_tsb_min, q75+0.1, 'Acumulação intensa', CORES['laranja']),
        (_tsb_max, q25-0.15,'Supercompensação',   CORES['verde']),
    ]:
        _fig_scatter.add_annotation(x=_xann, y=_yann, text=_txt,
                                    font=dict(size=9, color=_col),
                                    showarrow=False, xanchor='right' if _xann>0 else 'left')

    _fig_scatter.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=320,
        margin=dict(t=10, b=40, l=50, r=20),
        xaxis_title='TSB (Forma)', yaxis_title='κ (Curvatura)',
        xaxis=dict(tickfont=dict(size=9,color='#111'), linecolor='#333'),
        yaxis=dict(tickfont=dict(size=9,color='#111'), linecolor='#333'),
        legend=dict(orientation='h', y=-0.22, font=dict(color='#111', size=9)),
        font=dict(color='#111', size=10))
    st.plotly_chart(_fig_scatter, use_container_width=True,
                    config={'displayModeBar':False}, key='fmt_scatter_chart')

    # ── SECTION 4: λ₁ + dimension signals ────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Concentração de stress (λ₁/Σλ) e sinais do tensor")

    _fig_lam = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.5, 0.5], vertical_spacing=0.04,
                             subplot_titles=['λ₁/Σλ — concentração do stress','Dimensões do tensor'])

    if lam1_plot.notna().any():
        _fig_lam.add_trace(go.Scatter(
            x=dates_plot, y=lam1_plot * 100,
            name='λ₁/Σλ (%)', line=dict(color=CORES['laranja'], width=2),
            fill='tozeroy',
            fillcolor='rgba(243,156,18,0.08)',
            hovertemplate='λ₁/Σλ: %{y:.1f}%<extra></extra>'), row=1, col=1)
        _fig_lam.add_hline(y=65, line_color=CORES['vermelho'], line_width=1,
                           line_dash='dot', annotation_text='  Focal',
                           annotation_font_size=9, row=1, col=1)
        _fig_lam.add_hline(y=45, line_color=CORES['verde'], line_width=1,
                           line_dash='dot', annotation_text='  Multissist.',
                           annotation_font_size=9, row=1, col=1)

    # Dimension signals on row=2
    _dim_sigs = [
        (hrv_plot,  'HRV trend',  CORES['verde']),
        (weed_s.iloc[_sl], 'WEED_z',   CORES['azul']),
        (sleep_s.iloc[_sl],'Sleep_z',  CORES['roxo']),
        (ws_s.iloc[_sl],   "W' stress",CORES['laranja']),
        (hq_s.iloc[_sl],   'HR drift', CORES['vermelho']),
    ]
    for _dv, _dn, _dc in _dim_sigs:
        if _dv.notna().any():
            _fig_lam.add_trace(go.Scatter(
                x=dates_plot, y=_dv,
                name=_dn, line=dict(color=_dc, width=1.2),
                opacity=0.8,
                hovertemplate=f'{_dn}: %{{y:.3f}}σ<extra></extra>'), row=2, col=1)
    _fig_lam.add_hline(y=0, line_color='gray', line_width=0.5, row=2, col=1)

    _fig_lam.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=380,
        margin=dict(t=30, b=50, l=50, r=20), hovermode='x unified',
        legend=dict(orientation='h', y=-0.20, font=dict(color='#111', size=9)),
        font=dict(color='#111', size=10))
    _fig_lam.update_xaxes(tickfont=dict(size=9,color='#111'),
                           linecolor='#333', tickangle=-30)
    _fig_lam.update_yaxes(tickfont=dict(size=9,color='#111'), linecolor='#333')
    _fig_lam.update_yaxes(title_text='%', row=1, col=1,
                           title_font=dict(size=9,color='#111'))
    _fig_lam.update_yaxes(title_text='σ', row=2, col=1,
                           title_font=dict(size=9,color='#111'))
    st.plotly_chart(_fig_lam, use_container_width=True,
                    config={'displayModeBar':False}, key='fmt_lam_chart')

    # ── SECTION 5: Regime distribution ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Distribuição de regimes")
    _col_dist, _col_guide = st.columns([1, 1])

    with _col_dist:
        _rc_counts = pd.Series(regimes_plot).value_counts()
        _rc_counts = _rc_counts[_rc_counts.index != 'none']
        _fig_dist = go.Figure(go.Bar(
            x=[REGIME_CFG[k]['label'] for k in _rc_counts.index],
            y=_rc_counts.values,
            marker_color=[REGIME_CFG[k]['color'] for k in _rc_counts.index],
            text=_rc_counts.values,
            textposition='outside',
        ))
        _fig_dist.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=220,
            margin=dict(t=10, b=80, l=30, r=10),
            font=dict(color='#111', size=10),
            xaxis=dict(tickfont=dict(size=9,color='#111'), tickangle=-20),
            yaxis=dict(tickfont=dict(size=9,color='#111'), title='dias'))
        st.plotly_chart(_fig_dist, use_container_width=True,
                        config={'displayModeBar':False}, key='fmt_dist_chart')

    with _col_guide:
        st.markdown("**Guia de regimes**")
        for _rk, _rcfg in REGIME_CFG.items():
            if _rk == 'none' or not _rcfg['desc']: continue
            _hx = _rcfg['color'].lstrip('#')
            _r2,_g2,_b2 = int(_hx[0:2],16),int(_hx[2:4],16),int(_hx[4:6],16)
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:6px'>"
                f"<span style='width:10px;height:10px;border-radius:2px;"
                f"background:{_rcfg['color']};flex-shrink:0;margin-top:3px'></span>"
                f"<div><b>{_rcfg['label']}</b><br>"
                f"<small style='color:var(--text-color);opacity:0.7'>{_rcfg['desc']}</small></div></div>",
                unsafe_allow_html=True)

    # ── SECTION 6: Explanation expander ──────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 O que é o FMT Tensor κ — teoria e interpretação", expanded=False):
        st.markdown("""
**Functional Multidimensional Tensor (FMT)** — Della Mattia, 2019

O TSS colapsa uma sessão complexa num único número. Duas sessões com o mesmo TSS podem deixar o atleta em estados fisiológicos completamente diferentes — o TSS descarta exactamente a informação que determina a adaptação do dia seguinte.

**A ideia central:**
Stress de treino e frescura fisiológica não são variáveis independentes. São dimensões covariantes da mesma entidade fisiológica. Apenas um tensor representa esta covariação completamente.

```
x(t) = [CTLγ, HRV_trend, WEED_z, Sleep_z, W'_stress, HR_drift]
F(t) = cov(Δx) sobre janela 28 dias
κ(t) = trace(F(t))   ← curvatura escalar
```

**Interpretação de κ:**
- **κ crescente** → múltiplas dimensões a mudar simultaneamente → stress acumulado não compensado
- **κ decrescente** → adaptação a estabilizar, sinais a convergir
- **κ alto + TSB positivo** → *fadiga silenciosa* — o TSB mente. O FMT detecta este padrão antes do colapso de performance (AUC=0.553 vs 0.185 para TSB clássico, paper §12)

**Interpretação de λ₁/Σλ:**
- **>65%** → stress **focal**: uma dimensão domina (ex: só CTLγ cresceu)
  → identificar e tratar a dimensão dominante
- **45-65%** → stress **misto**: duas dimensões perturbadas
- **<45%** → stress **multissistémico**: todas as dimensões em simultâneo
  → sinal clássico de overreaching não-funcional

**Comparação FMT vs TSB clássico (cohort 30 atletas × 365 dias):**

| Métrica | FMT-Transformer | TSB clássico |
|---------|----------------|--------------|
| AUC-ROC | 0.553 | 0.185 |
| Lead time detecção | 3 dias | 0.2 dias |
| False positive rate | 28.2% | 62.4% |
""")

    # ── SECTION 7: Download ───────────────────────────────────────────────────
    st.markdown("---")
    _dl_cols = ['Data','CTL','ATL','TSB','CTLg_perf','HRV_trend','WEED_z',
                'sleep_z','w_stress','hq_drift_z','wp_prime',
                'FMT_kappa','FMT_lambda1_frac','FMT_kappa_4d',
                'FMT_kappa_Bike','FMT_kappa_Row','FMT_kappa_Ski','FMT_kappa_Run']
    _avail = [c for c in _dl_cols if c in ld.columns]
    _dl_df = ld[_avail].copy()
    _dl_df['_regime'] = regimes
    _dl_df['Data'] = _dl_df['Data'].astype(str)
    _dl_df = _dl_df.round(4)
    st.download_button(
        label="📥 Download FMT completo (.csv)",
        data=_dl_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
        file_name="atheltica_fmt_tensor.csv",
        mime="text/csv", key="fmt_dl_full")
    st.caption(f"Tensor {info.get('tensor_dim',0)}×{info.get('tensor_dim',0)}: "
               f"{info.get('tensor_dim_names','')} | "
               f"Thresholds: p25={q25:.3f} p50={q50:.3f} p75={q75:.3f}")
