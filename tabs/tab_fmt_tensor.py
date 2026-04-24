"""
tab_fmt_tensor.py — ATHELTICA
Tab dedicada ao FMT Tensor κ — visualização completa com gráficos,
regimes fisiológicos, e explicação baseada no paper Della Mattia 2019.

ALTERAÇÕES v2:
  - REST days: _prepare_dims() — w_stress decay 0.5/dia; hq_drift_z → 0
  - Legibilidade: todos os update_layout com font 12+, axis labels contrastantes
  - SECTION 6 nova: Análise de Stress Focal — dropdown janela + slider limiar
                    top-3 dimensões + cards + sugestões de acção + calendário
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


# ── Metadados das dimensões — para a secção focal ────────────────────────────
DIM_META = {
    'HRV_trend': {
        'label': 'HRV Trend',
        'color': '#4C9BE8',
        'icon': '💓',
        'descricao': 'Tendência do HRV matinal nos últimos 14d.',
        'negativo_sig': 'HRV em queda — sistema nervoso autónomo sobrecarregado.',
        'positivo_sig': 'HRV a subir — boa adaptação e recuperação.',
        'acao_alta': [
            'Reduzir volume total em 20–30% por 5–7 dias.',
            'Priorizar sessões Z1/Z2 — sem intervalos de alta intensidade.',
            'Verificar sono: mínimo 8h, evitar luz azul após 22h.',
            'Considerar sessão de avaliação HRV ortostático.',
        ],
    },
    'WEED_z': {
        'label': 'WEED (Wellness)',
        'color': '#F5A623',
        'icon': '🧠',
        'descricao': 'Z-score composto de fadiga, humor, dor muscular e stress subjetivo.',
        'negativo_sig': 'Wellness degradado — fadiga/stress acumulados acima do normal.',
        'positivo_sig': 'Wellness elevado — condições subjetivas favoráveis.',
        'acao_alta': [
            'Rever carga de treino vs. carga de vida (trabalho, viagens).',
            'Adicionar dia completo de descanso ativo (caminhada ≤45min).',
            'Técnicas de recovery não-físico: meditação, respiração, massagem.',
            'Monitorar por 3 dias — se persistir, reduzir CTL alvo em 10%.',
        ],
    },
    'sleep_z': {
        'label': 'Qualidade do Sono',
        'color': '#7B68EE',
        'icon': '🌙',
        'descricao': 'Z-score da qualidade e duração do sono vs. baseline pessoal.',
        'negativo_sig': 'Sono abaixo do baseline — recuperação comprometida.',
        'positivo_sig': 'Sono acima do baseline — recovery otimizado.',
        'acao_alta': [
            'Priorizar higiene do sono: horário fixo, quarto frio e escuro.',
            'Eliminar treinos vespertinos tardios (>20h) por 1 semana.',
            'Reduzir cafeína após 14h.',
            'Evitar sessões de alta intensidade quando sleep_z < -1.5.',
        ],
    },
    'w_stress': {
        'label': "W' Stress (Anaeróbico)",
        'color': '#E8524C',
        'icon': '⚡',
        'descricao': "Utilização de W' (capacidade anaeróbica) relativa ao baseline.",
        'negativo_sig': "W' pouco utilizado — sem estímulo anaeróbico recente.",
        'positivo_sig': "W' muito utilizado — alto gasto anaeróbico acumulado.",
        'acao_alta': [
            'Reduzir sessões acima de CP por 5–7 dias.',
            'Substituir intervalos curtos por tempo estendido em Z3/SS.',
            'Verificar se há múltiplos dias consecutivos de trabalho supra-CP.',
            "W' precisa de 24–48h para reconstituição completa.",
        ],
    },
    'hq_drift_z': {
        'label': 'HR Drift (Cardíaco)',
        'color': '#50C878',
        'icon': '📈',
        'descricao': 'Deriva cardíaca intra-sessão (HR Q4/Q1). Alto = fadiga cardiovascular.',
        'negativo_sig': 'Sem deriva — FC estável. Dias REST ou sessões curtas.',
        'positivo_sig': 'Deriva elevada — FC desacoplada da potência. Fadiga cardiovascular.',
        'acao_alta': [
            'Sessões seguintes: manter potência alvo mas aceitar FC mais alta.',
            'Evitar treino de limiar por 3–5 dias.',
            'Hidratar bem — desacoplamento agravado por desidratação.',
            'Se deriva >15% por 3 dias consecutivos: reduzir carga geral.',
        ],
    },
}


def _prepare_dims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata NaN nas dimensões do tensor para dias REST ou dados ausentes.

    Estratégia fisiológica por dimensão:
      w_stress   → REST day sem atividade. Forward-fill com decay 0.5/dia
                   (stress anaeróbico persiste mas dissipa com meia-vida ~1 dia)
      hq_drift_z → Sem sessão = sem deriva cardíaca. Preenche com 0 (baseline neutro).
      WEED_z     → Forward-fill simples (wellness persiste dia a dia)
      sleep_z    → Forward-fill simples (sono do dia anterior é o melhor proxy)
      HRV_trend  → Raramente NaN — sem tratamento
    """
    df = df.copy()

    # hq_drift_z: REST day → 0 (ausência de deriva = neutro)
    if 'hq_drift_z' in df.columns:
        df['hq_drift_z'] = df['hq_drift_z'].fillna(0.0)

    # w_stress: forward-fill com decay 50% por dia de REST
    if 'w_stress' in df.columns:
        ws = df['w_stress'].tolist()
        for i in range(1, len(ws)):
            if pd.isna(ws[i]) and not pd.isna(ws[i - 1]):
                ws[i] = ws[i - 1] * 0.5
        df['w_stress'] = pd.array(ws, dtype='float64')
        df['w_stress'] = df['w_stress'].fillna(0.0)

    # WEED_z e sleep_z: forward-fill simples
    for col in ['WEED_z', 'sleep_z']:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    return df


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
                             annotation_font_size=11, row=1, col=1)

    # κ slope (14d) on row=2
    _slope_s = pd.Series(slope_plot, index=dates_plot.values)
    _colors_slope = [CORES['vermelho'] if s > 0 else CORES['verde']
                     for s in _slope_s.fillna(0)]
    _fig_kappa.add_trace(go.Bar(
        x=dates_plot, y=_slope_s,
        name='Δκ/14d', marker_color=_colors_slope,
        hovertemplate='Δκ/d: %{y:.4f}<extra></extra>'), row=2, col=1)
    _fig_kappa.add_hline(y=0, line_color='#666', line_width=0.8, row=2, col=1)

    _fig_kappa.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=400,
        margin=dict(t=20, b=55, l=60, r=20), hovermode='x unified',
        legend=dict(orientation='h', y=-0.16, font=dict(color='#222', size=11),
                    bgcolor='rgba(255,255,255,0.85)', borderwidth=0),
        font=dict(color='#222', size=12))
    _fig_kappa.update_xaxes(tickfont=dict(size=11, color='#333'),
                             linecolor='#555', tickcolor='#555', tickangle=-30)
    _fig_kappa.update_yaxes(tickfont=dict(size=11, color='#333'),
                             linecolor='#555', tickcolor='#555', gridcolor='rgba(0,0,0,0.06)')
    _fig_kappa.update_yaxes(title_text='κ — curvatura', row=1, col=1,
                             title_font=dict(size=12, color='#333'))
    _fig_kappa.update_yaxes(title_text='Δκ/14d', row=2, col=1,
                             title_font=dict(size=12, color='#333'))
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
            marker=dict(color=_rcfg['color'], size=6, opacity=0.75),
            text=_ddats,
            hovertemplate='%{text}<br>TSB: %{x:.1f} | κ: %{y:.3f}<extra></extra>'))

    # Today marker
    _fig_scatter.add_trace(go.Scatter(
        x=[tsb_now], y=[kappa_now], mode='markers',
        name='Hoje',
        marker=dict(color='black', size=13, symbol='star',
                    line=dict(color='white', width=1.5)),
        hovertemplate=f'Hoje: TSB={tsb_now:.1f} | κ={kappa_now:.3f}<extra></extra>'))

    # Quadrant lines
    _fig_scatter.add_vline(x=0, line_color='#999', line_width=0.8, line_dash='dot')
    _fig_scatter.add_hline(y=q75, line_color=CORES['vermelho'], line_width=0.8, line_dash='dot')
    _fig_scatter.add_hline(y=q25, line_color=CORES['verde'],    line_width=0.8, line_dash='dot')

    # Annotations for quadrants
    _tsb_min = float(tsb_plot.min())
    _tsb_max = float(tsb_plot.max())
    for _xann, _yann, _txt, _col in [
        (_tsb_max, q75+0.1, 'Fadiga silenciosa', CORES['vermelho']),
        (_tsb_min, q75+0.1, 'Acumulação intensa', CORES['laranja']),
        (_tsb_max, q25-0.15,'Supercompensação',   CORES['verde']),
    ]:
        _fig_scatter.add_annotation(x=_xann, y=_yann, text=_txt,
                                    font=dict(size=11, color=_col),
                                    showarrow=False, xanchor='right' if _xann>0 else 'left')

    _fig_scatter.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=340,
        margin=dict(t=15, b=55, l=65, r=20),
        xaxis=dict(
            title=dict(text='TSB (Forma)', font=dict(size=13, color='#333')),
            tickfont=dict(size=11, color='#333'),
            linecolor='#555', gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(
            title=dict(text='κ (Curvatura do estado)', font=dict(size=13, color='#333')),
            tickfont=dict(size=11, color='#333'),
            linecolor='#555', gridcolor='rgba(0,0,0,0.06)'),
        legend=dict(orientation='h', y=-0.20, font=dict(color='#222', size=11),
                    bgcolor='rgba(255,255,255,0.85)', borderwidth=0),
        font=dict(color='#222', size=12))
    st.plotly_chart(_fig_scatter, use_container_width=True,
                    config={'displayModeBar':False}, key='fmt_scatter_chart')

    # ── SECTION 4: λ₁ + dimension signals ────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Concentração de stress (λ₁/Σλ) e sinais do tensor")

    _fig_lam = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.5, 0.5], vertical_spacing=0.06,
                             subplot_titles=['λ₁/Σλ — concentração do stress','Dimensões do tensor'])

    if lam1_plot.notna().any():
        _fig_lam.add_trace(go.Scatter(
            x=dates_plot, y=lam1_plot * 100,
            name='λ₁/Σλ (%)', line=dict(color=CORES['laranja'], width=2),
            fill='tozeroy',
            fillcolor='rgba(243,156,18,0.10)',
            hovertemplate='λ₁/Σλ: %{y:.1f}%<extra></extra>'), row=1, col=1)
        _fig_lam.add_hline(y=65, line_color=CORES['vermelho'], line_width=1.2,
                           line_dash='dot', annotation_text='  Focal',
                           annotation_font_size=11,
                           annotation_font_color=CORES['vermelho'], row=1, col=1)
        _fig_lam.add_hline(y=45, line_color=CORES['verde'], line_width=1.2,
                           line_dash='dot', annotation_text='  Multissist.',
                           annotation_font_size=11,
                           annotation_font_color=CORES['verde'], row=1, col=1)

    # Dimension signals on row=2 — usando _prepare_dims para REST days
    _ld_dims = ld.iloc[_sl].copy()
    _ld_dims_prep = _prepare_dims(_ld_dims)

    _dim_sigs = [
        (hrv_plot,                             'HRV trend',  CORES['verde']),
        (pd.to_numeric(_ld_dims_prep.get('WEED_z',   pd.Series(np.nan, index=_ld_dims_prep.index)), errors='coerce'), 'WEED_z',   CORES['azul']),
        (pd.to_numeric(_ld_dims_prep.get('sleep_z',  pd.Series(np.nan, index=_ld_dims_prep.index)), errors='coerce'), 'Sleep_z',  CORES['roxo']),
        (pd.to_numeric(_ld_dims_prep.get('w_stress', pd.Series(np.nan, index=_ld_dims_prep.index)), errors='coerce'), "W' stress",CORES['laranja']),
        (pd.to_numeric(_ld_dims_prep.get('hq_drift_z',pd.Series(np.nan, index=_ld_dims_prep.index)), errors='coerce'),'HR drift', CORES['vermelho']),
    ]
    for _dv, _dn, _dc in _dim_sigs:
        if isinstance(_dv, pd.Series) and _dv.notna().any():
            _fig_lam.add_trace(go.Scatter(
                x=dates_plot, y=_dv.values if hasattr(_dv, 'values') else _dv,
                name=_dn, line=dict(color=_dc, width=1.5),
                opacity=0.85,
                hovertemplate=f'{_dn}: %{{y:.3f}}σ<extra></extra>'), row=2, col=1)
    _fig_lam.add_hline(y=0, line_color='#999', line_width=0.8, row=2, col=1)

    _fig_lam.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=420,
        margin=dict(t=40, b=60, l=65, r=20), hovermode='x unified',
        legend=dict(orientation='h', y=-0.18, font=dict(color='#222', size=11),
                    bgcolor='rgba(255,255,255,0.85)', borderwidth=0),
        font=dict(color='#222', size=12))
    _fig_lam.update_xaxes(tickfont=dict(size=11, color='#333'),
                           linecolor='#555', tickangle=-30,
                           gridcolor='rgba(0,0,0,0.06)')
    _fig_lam.update_yaxes(tickfont=dict(size=11, color='#333'),
                           linecolor='#555', gridcolor='rgba(0,0,0,0.06)')
    _fig_lam.update_yaxes(title_text='λ₁/Σλ (%)', row=1, col=1,
                           title_font=dict(size=12, color='#333'))
    _fig_lam.update_yaxes(title_text='z-score (σ)', row=2, col=1,
                           title_font=dict(size=12, color='#333'))
    # Subplot titles font
    for ann in _fig_lam.layout.annotations:
        ann.font.size = 13
        ann.font.color = '#333'
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
            textfont=dict(size=12, color='#222'),
        ))
        _fig_dist.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=240,
            margin=dict(t=15, b=85, l=50, r=15),
            font=dict(color='#222', size=12),
            xaxis=dict(
                title=dict(text='Regime', font=dict(size=13, color='#333')),
                tickfont=dict(size=11, color='#333'),
                tickangle=-20, linecolor='#555'),
            yaxis=dict(
                title=dict(text='Nº de dias', font=dict(size=13, color='#333')),
                tickfont=dict(size=11, color='#333'),
                linecolor='#555', gridcolor='rgba(0,0,0,0.06)'))
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

    # ── SECTION 6: Focal Stress Analysis ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Análise de Stress Focal — qual dimensão domina?")
    st.caption(
        "Identifica as dimensões com maior stress absoluto e quantos dias estiveram "
        "acima do limiar. λ₁/Σλ alto indica concentração — aqui vês em qual dimensão."
    )

    _dims_available = [d for d in DIM_META if d in ld.columns]

    _col_ctrl1, _col_ctrl2 = st.columns([1, 1])
    with _col_ctrl1:
        _janela_focal = st.selectbox(
            "Janela de análise",
            options=[14, 28, 60, 90],
            index=1,
            format_func=lambda x: f"Últimos {x} dias",
            key="fmt_focal_janela",
        )
    with _col_ctrl2:
        _threshold_focal = st.slider(
            "Limiar |z-score| para considerar stress",
            min_value=0.5, max_value=2.5, value=1.0, step=0.25,
            key="fmt_focal_threshold",
        )

    # Preparar dados com NaN tratados (REST days)
    _df_focal_raw = ld.tail(_janela_focal).copy()
    _df_focal = _prepare_dims(_df_focal_raw)
    _n_dias = len(_df_focal)

    # Calcular score por dimensão
    _dim_scores = {}
    for _d in _dims_available:
        _vals = pd.to_numeric(_df_focal[_d], errors='coerce').dropna()
        if len(_vals) == 0:
            continue
        _dim_scores[_d] = {
            'mean_abs':  float(_vals.abs().mean()),
            'max_abs':   float(_vals.abs().max()),
            'n_acima':   int((_vals.abs() >= _threshold_focal).sum()),
            'pct_acima': float((_vals.abs() >= _threshold_focal).sum() / _n_dias * 100),
            'tendencia': float(_vals.iloc[-1] - _vals.iloc[0]) if len(_vals) > 1 else 0.0,
            'ultimo':    float(_vals.iloc[-1]) if len(_vals) > 0 else 0.0,
        }

    if not _dim_scores:
        st.info("Dados insuficientes para a janela selecionada.")
    else:
        # Ordenar por mean_abs — top 3 em destaque
        _sorted_dims = sorted(_dim_scores.items(), key=lambda x: x[1]['mean_abs'], reverse=True)
        _top3 = _sorted_dims[:3]

        # Gráfico de barras — stress médio por dimensão
        _bar_labels = [DIM_META[d]['icon'] + ' ' + DIM_META[d]['label'] for d, _ in _sorted_dims]
        _bar_values = [v['mean_abs'] for _, v in _sorted_dims]
        _bar_colors = [
            DIM_META[d]['color'] if i < 3 else 'rgba(180,180,180,0.45)'
            for i, (d, _) in enumerate(_sorted_dims)
        ]

        _fig_focal = go.Figure(go.Bar(
            x=_bar_labels,
            y=_bar_values,
            marker_color=_bar_colors,
            text=[f"{v:.2f}" for v in _bar_values],
            textposition='outside',
            textfont=dict(size=12, color='#222'),
            hovertemplate='<b>%{x}</b><br>Stress médio |z|: %{y:.3f}<extra></extra>',
        ))
        _fig_focal.add_hline(
            y=_threshold_focal,
            line_dash="dash",
            line_color='rgba(200,80,0,0.7)',
            line_width=1.5,
            annotation_text=f"Limiar {_threshold_focal}σ",
            annotation_font=dict(size=11, color='#CC5200'),
        )
        _fig_focal.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=300,
            margin=dict(t=20, b=90, l=60, r=20),
            title=dict(text=f"Stress por dimensão — últimos {_janela_focal} dias",
                       font=dict(size=14, color='#222')),
            showlegend=False,
            font=dict(color='#222', size=12),
            xaxis=dict(
                title=dict(text='Dimensão do tensor', font=dict(size=13, color='#333')),
                tickfont=dict(size=11, color='#333'), tickangle=-15, linecolor='#555'),
            yaxis=dict(
                title=dict(text='Stress médio |z-score|', font=dict(size=13, color='#333')),
                tickfont=dict(size=11, color='#333'), linecolor='#555',
                gridcolor='rgba(0,0,0,0.06)'),
        )
        st.plotly_chart(_fig_focal, use_container_width=True,
                        config={'displayModeBar': False}, key='fmt_focal_bar')

        # Cards top-3
        st.markdown(f"#### Top {min(3, len(_top3))} dimensões mais stressadas")
        _cols_top = st.columns(min(3, len(_top3)))

        for _i, (_d, _v) in enumerate(_top3):
            _meta = DIM_META[_d]
            with _cols_top[_i]:
                _cor = _meta['color']
                _trend_icon = '↑' if _v['tendencia'] > 0.1 else ('↓' if _v['tendencia'] < -0.1 else '→')
                _rank = ['🥇', '🥈', '🥉'][_i]
                st.markdown(
                    f"<div style='border:1px solid {_cor}55;border-radius:10px;padding:14px;"
                    f"background:rgba(245,245,248,0.9);margin-bottom:8px'>"
                    f"<div style='font-size:15px;font-weight:700;color:{_cor}'>"
                    f"{_rank} {_meta['icon']} {_meta['label']}</div>"
                    f"<div style='font-size:26px;font-weight:700;color:#111;margin:6px 0'>"
                    f"{_v['mean_abs']:.2f}"
                    f"<span style='font-size:12px;color:#666;margin-left:4px'>|z| médio</span></div>"
                    f"<div style='font-size:12px;color:#555'>"
                    f"{_trend_icon} tendência &nbsp;|&nbsp; último: <b>{_v['ultimo']:+.2f}</b></div>"
                    f"<div style='font-size:12px;color:#C0392B;font-weight:600;margin-top:4px'>"
                    f"⚠️ {_v['n_acima']} / {_n_dias} dias acima do limiar ({_v['pct_acima']:.0f}%)</div>"
                    f"<div style='font-size:11px;color:#777;margin-top:6px'>{_meta['descricao']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Expander com sugestões — top-1 obrigatório + secundários com dias acima
        _d1, _v1 = _top3[0]
        _meta1 = DIM_META[_d1]
        with st.expander(
            f"📋 Sugestões de ação — {_meta1['icon']} {_meta1['label']} "
            f"({_v1['n_acima']} dias acima do limiar nos últimos {_janela_focal}d)",
            expanded=True,
        ):
            _sig = _meta1['positivo_sig'] if _v1['ultimo'] > 0 else _meta1['negativo_sig']
            st.markdown(f"**Situação atual:** {_sig}")
            st.markdown("**Ações recomendadas:**")
            for _acao in _meta1['acao_alta']:
                st.markdown(f"- {_acao}")

            _secundarios = [(_dx, _vx) for _dx, _vx in _top3[1:] if _vx['n_acima'] > 0]
            if _secundarios:
                st.markdown("---")
                st.markdown("**Ações adicionais (dimensões secundárias também elevadas):**")
                for _dx, _vx in _secundarios:
                    _mx = DIM_META[_dx]
                    st.markdown(
                        f"**{_mx['icon']} {_mx['label']}** "
                        f"({_vx['n_acima']} dias) — {_mx['acao_alta'][0]}"
                    )

        # Calendário de stress — barras sobrepostas por dimensão
        st.markdown(f"##### Calendário de stress focal — últimos {_janela_focal} dias")
        _date_vals = pd.to_datetime(_df_focal['Data']) if 'Data' in _df_focal.columns \
                     else pd.RangeIndex(len(_df_focal))

        _fig_heat_data = []
        for _d in _dims_available:
            _vals_abs = pd.to_numeric(_df_focal[_d], errors='coerce').abs()
            _above_mask = (_vals_abs >= _threshold_focal).astype(float)
            _y_vals = _above_mask * _vals_abs  # altura = |z| apenas nos dias acima; 0 abaixo
            _fig_heat_data.append(
                go.Bar(
                    name=DIM_META[_d]['icon'] + ' ' + DIM_META[_d]['label'],
                    x=_date_vals,
                    y=_y_vals,
                    marker_color=DIM_META[_d]['color'],
                    opacity=0.65,
                    hovertemplate=(
                        f"<b>{DIM_META[_d]['label']}</b><br>"
                        "%{x|%d/%m/%Y}<br>|z|: %{customdata:.2f}<extra></extra>"
                    ),
                    customdata=_vals_abs.values,
                )
            )
        _fig_heat = go.Figure(data=_fig_heat_data)
        _fig_heat.add_hline(
            y=_threshold_focal, line_dash="dot",
            line_color='rgba(200,80,0,0.5)', line_width=1.2,
        )
        _fig_heat.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=290,
            barmode='overlay',
            margin=dict(t=20, b=55, l=65, r=20),
            title=dict(text=f"Quais dias cada dimensão esteve acima de {_threshold_focal}σ",
                       font=dict(size=13, color='#222')),
            font=dict(color='#222', size=12),
            xaxis=dict(
                title=dict(text='Data', font=dict(size=13, color='#333')),
                tickfont=dict(size=11, color='#333'), linecolor='#555',
                tickangle=-30, gridcolor='rgba(0,0,0,0.04)'),
            yaxis=dict(
                title=dict(text='|z-score| (dias acima do limiar)', font=dict(size=13, color='#333')),
                tickfont=dict(size=11, color='#333'), linecolor='#555',
                gridcolor='rgba(0,0,0,0.06)'),
            legend=dict(orientation='h', y=-0.22, font=dict(size=11, color='#222'),
                        bgcolor='rgba(255,255,255,0.85)', borderwidth=0),
        )
        st.plotly_chart(_fig_heat, use_container_width=True,
                        config={'displayModeBar': False}, key='fmt_heat_chart')

        # Tabela resumo compacta
        _rows = []
        for _d, _v in _sorted_dims:
            _rows.append({
                'Dimensão': DIM_META[_d]['icon'] + ' ' + DIM_META[_d]['label'],
                '|z| médio': f"{_v['mean_abs']:.3f}",
                '|z| pico':  f"{_v['max_abs']:.3f}",
                f'Dias ≥{_threshold_focal}σ': f"{_v['n_acima']} ({_v['pct_acima']:.0f}%)",
                'Último':    f"{_v['ultimo']:+.3f}",
                'Tend.':     '↑' if _v['tendencia'] > 0.1 else ('↓' if _v['tendencia'] < -0.1 else '→'),
            })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    # ── SECTION 7: Explanation expander ──────────────────────────────────────
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
  → usar a secção "Análise de Stress Focal" acima para identificar qual
- **45-65%** → stress **misto**: duas dimensões perturbadas
- **<45%** → stress **multissistémico**: todas as dimensões em simultâneo
  → sinal clássico de overreaching não-funcional

**Dias REST — tratamento das dimensões:**
- `w_stress` em REST → decai 50%/dia (W' reconstitui-se mas não zera de imediato)
- `hq_drift_z` em REST → 0 (sem sessão = sem deriva cardíaca = posição neutra)
- `WEED_z`, `sleep_z` → forward-fill (wellness do dia anterior é o melhor proxy)

**Comparação FMT vs TSB clássico (cohort 30 atletas × 365 dias):**

| Métrica | FMT-Transformer | TSB clássico |
|---------|----------------|--------------|
| AUC-ROC | 0.553 | 0.185 |
| Lead time detecção | 3 dias | 0.2 dias |
| False positive rate | 28.2% | 62.4% |
""")

    # ── SECTION 8: Download ───────────────────────────────────────────────────
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
