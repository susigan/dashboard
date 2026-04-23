from utils.config import *
from utils.phase_detector import detect_all_phases, phase_summary, PHASE_LABELS
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

def _interp_gamma(gv, rv, mod):
    """Interpreta γ e R² para uma modalidade específica."""
    # Memory depth label
    if gv < 0.20:
        mem_lbl  = "Muito longa (meses)"
        mem_icon = "🧬"
        mem_text = (f"γ={gv:.3f} indica que adaptações de **{mod}** acumulam "
                    f"ao longo de meses. Treinos de {int(min(365, 1/gv*30))}+ dias "
                    f"atrás ainda influenciam o CTLγ hoje. Típico de desportos com "
                    f"alta dependência de base aeróbica crónica (capilarização, "
                    f"mitocôndrias, economia de movimento).")
    elif gv < 0.35:
        mem_lbl  = "Longa (semanas-meses)"
        mem_icon = "🏔️"
        mem_text = (f"γ={gv:.3f} — intervalo óptimo para resistência de longa duração "
                    f"(Imbach et al. 2021: γ≈0.35 melhorou predição em 18%). "
                    f"O historial de semanas contribui mais do que a forma recente. "
                    f"Ideal para modelar base aeróbica de {mod}.")
    elif gv < 0.55:
        mem_lbl  = "Moderada (dias-semanas)"
        mem_icon = "⚖️"
        mem_text = (f"γ={gv:.3f} — equilíbrio entre carga recente e historial. "
                    f"Adaptações de {mod} respondem em escala de dias a semanas. "
                    f"Modelo similar ao CTL clássico mas sem saturação exponencial.")
    elif gv < 0.75:
        mem_lbl  = "Curta (dias)"
        mem_icon = "⚡"
        mem_text = (f"γ={gv:.3f} — comportamento reactivo, próximo do ATL. "
                    f"As adaptações de {mod} são dominadas pela carga recente. "
                    f"Pode indicar poucos dados ou alta variabilidade inter-sessão.")
    else:
        mem_lbl  = "Muito curta (EWM-like)"
        mem_icon = "🔄"
        mem_text = (f"γ={gv:.3f} ≈ 1.0 — comporta-se como CTL clássico. "
                    f"O fitting não encontrou vantagem no kernel fraccionário "
                    f"para esta modalidade. Verificar qualidade dos dados de {mod}.")

    # R² quality
    if rv >= 0.70:
        r2_text = f"R²={rv:.2f} — ajuste forte. O CTLγ prediz bem a performance de {mod}."
    elif rv >= 0.45:
        r2_text = f"R²={rv:.2f} — ajuste moderado. Dados de performance limitados ou variabilidade alta."
    elif rv > 0.0:
        r2_text = f"R²={rv:.2f} — ajuste fraco. Poucos pontos de MMP20/CP para calibração fiável."
    else:
        r2_text = f"R²=0 — sem dados suficientes. γ={gv:.3f} é o default (não calibrado)."

    return mem_icon, mem_lbl, mem_text, r2_text


def tab_pmc(da, wc=None):
    """
    PMC — icu_training_load para CTL/ATL/FTLM.
    Barras de Load = TRIMP (session_rpe).
    Tabelas mensais: eFTP por modalidade + KM/KJ por modalidade (como Intervals.icu).
    Gráfico KM semanal stacked com linha de média por modalidade.
    """
    st.header("📈 PMC — Performance Management Chart")
    if len(da) == 0: st.warning("Sem dados de atividades."); return

    def _add_phase_bg(fig, phase_df, plot_dates):
        """
        Add phase colour bands to a Plotly figure.
        - Historical bands: opacity 0.08 (very subtle)
        - Current phase band: opacity 0.18 (slightly more visible)
        - Current phase annotation in top-right corner
        """
        if phase_df is None or len(phase_df) == 0:
            return
        _ph_s  = phase_df[['Data','phase_smooth','phase_color','phase_label']].copy()
        _ph_s['_ds'] = _ph_s['Data'].astype(str).str[:10]
        _ph_map = _ph_s.set_index('_ds')
        _dates_pd = pd.to_datetime(plot_dates)
        _last_date = _dates_pd.max()

        _prev_ph, _band_st, _prev_col = None, None, '#95a5a6'
        _prev_label = ''
        _bands = []  # collect all bands first

        for _dt in _dates_pd:
            _ds = str(_dt)[:10]
            if _ds in _ph_map.index:
                _row = _ph_map.loc[_ds]
                _cur_ph    = _row['phase_smooth'] if isinstance(_row, pd.Series) else _row['phase_smooth'].iloc[0]
                _cur_col   = _row['phase_color']  if isinstance(_row, pd.Series) else _row['phase_color'].iloc[0]
                _cur_label = _row['phase_label']  if isinstance(_row, pd.Series) else _row['phase_label'].iloc[0]
            else:
                _cur_ph, _cur_col, _cur_label = 'TRANSITION', '#95a5a6', '⬜ Transition'

            if _cur_ph != _prev_ph:
                if _band_st is not None:
                    _bands.append((_band_st, _dt, _prev_col, _prev_ph, _prev_label))
                _band_st, _prev_ph, _prev_col, _prev_label = _dt, _cur_ph, _cur_col, _cur_label

        # Close last band
        if _band_st is not None:
            _bands.append((_band_st, _last_date, _prev_col, _prev_ph, _prev_label))

        # Get current (last) phase
        _current_ph  = _bands[-1][3] if _bands else 'TRANSITION'
        _current_col = _bands[-1][2] if _bands else '#95a5a6'
        _current_lbl = _bands[-1][4] if _bands else '⬜ Transition'

        # Draw bands
        for _b0, _b1, _bcol, _bph, _blbl in _bands:
            _hx = _bcol.lstrip('#')
            _r2,_g2,_b2c = int(_hx[0:2],16),int(_hx[2:4],16),int(_hx[4:6],16)
            # Current phase slightly more opaque
            _alpha = 0.18 if _bph == _current_ph else 0.08
            fig.add_shape(type='rect', x0=_b0, x1=_b1,
                y0=0, y1=1, xref='x', yref='paper',
                fillcolor=f'rgba({_r2},{_g2},{_b2c},{_alpha})',
                line_width=0, layer='below')

        # Annotation: current phase label in top-right
        _hx2 = _current_col.lstrip('#')
        _r3,_g3,_b3 = int(_hx2[0:2],16),int(_hx2[2:4],16),int(_hx2[4:6],16)
        fig.add_annotation(
            x=1.0, y=1.02,
            xref='paper', yref='paper',
            text=f'Fase actual: <b>{_current_lbl}</b>',
            showarrow=False,
            font=dict(size=11, color=f'rgb({_r3},{_g3},{_b3})'),
            bgcolor=f'rgba({_r3},{_g3},{_b3},0.12)',
            bordercolor=f'rgb({_r3},{_g3},{_b3})',
            borderwidth=1, borderpad=4,
            xanchor='right', yanchor='bottom'
        )



    # da = ac_full (full unfiltered history from app.py)
    da_full = da
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
    @st.cache_data(show_spinner="A calcular FTLM fraccionário (γ dual)...", ttl=3600)
    def _cached_csc(_da_hash, _wc_hash, da, wc_arg):
        # _da_hash and _wc_hash are used only as cache keys (not computed again)
        return calcular_series_carga(da, df_wellness=wc_arg, ate_hoje=True)

    # Cache key: tuple of (last date, n_rows) — recomputes only when data changes
    _da_key  = (str(da_full['Data'].max()) if 'Data' in da_full.columns else '',
                len(da_full))
    _wc_key  = (str(wc['Data'].max())      if wc is not None and 'Data' in wc.columns else '',
                len(wc) if wc is not None else 0)
    _ld_frac, _info = _cached_csc(_da_key, _wc_key, da_full, wc)

    if len(_ld_frac) > 0 and 'FTLM' in _ld_frac.columns:
        # Align fractional ld with our ld (may have different date range)
        _frac_idx = _ld_frac.set_index('Data')
        # Copy all fractional columns (FTLM, CTLg_perf, CTLg_rec, CTLg_Bike, etc.)
        _frac_cols = ['FTLM', 'CTLg_perf', 'CTLg_rec', 'HRV_trend',
                      'CTLg_Bike', 'CTLg_Row', 'CTLg_Ski', 'CTLg_Run',
                      'CTL_Bike',  'CTL_Row',  'CTL_Ski',  'CTL_Run',
                      'ATL_Bike',  'ATL_Row',  'ATL_Ski',  'ATL_Run']
        for _col in _frac_cols:
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
                 'r2_perf': 0.0, 'r2_rec': 0.0, 'gamma_source': 'classic',
                 'mods': {}, 'best_mod': 'N/A'}

    u = ld.iloc[-1]

    # Caption with dual-gamma info
    _gp   = _info.get('gamma_perf', best_g)
    _gr   = _info.get('gamma_rec',  best_g)
    _r2p  = _info.get('r2_perf',  0.0)
    _r2r  = _info.get('r2_rec',   0.0)
    _gsrc = _info.get('gamma_source', 'classic')
    _n_mmp= _info.get('n_mmp', 0)
    _perf_col_lbl = _info.get('perf_col', 'icu_eftp')
    _best_lag_lbl = _info.get('best_lag_rec', 1)
    st.caption(
        f"FTLM fraccionário | Carga: **{_metrica_ctl}** | Perf proxy: **{_perf_col_lbl}** (smooth 3d) | "
        f"γ_perf={_gp:.3f} (R²={_r2p:.2f}, test set) | "
        f"γ_rec={_gr:.3f} (R²={_r2r:.2f}, lag={_best_lag_lbl}d, LnRMSSD intercept+slope) | "
        f"γ activo: **{_gsrc}** | Histórico: {len(ld)} dias"
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

    @st.cache_data(show_spinner="A detectar fases de treino...", ttl=3600)
    def _cached_phases(_ld_hash, ld_arg):
        return detect_all_phases(ld_arg)

    try:
        _ld_hash2 = (str(_ld_frac['Data'].max()) if len(_ld_frac) > 0 else '',
                     len(_ld_frac))
        _phase_results = _cached_phases(_ld_hash2, _ld_frac) if len(_ld_frac) > 30 else {}
    except Exception as _pe:
        _phase_results = {}
        st.caption(f"⚠️ Detector de fases indisponível: {_pe}")

    # ── RESUMO PMC ──
    # ── Fase actual card (below PMC chart) ──────────────────────────────────
    try:
        if _phase_results and 'overall' in _phase_results:
            _ph_now   = _phase_results['overall'].iloc[-1]
            _ph_lbl   = _ph_now['phase_label']
            _ph_col   = _ph_now['phase_color']
            _ph_desc  = _ph_now['phase_desc']
            _ph_dias  = int(_ph_now['dias_na_fase']) + 1
            _ph_dctl  = float(_ph_now['dCTLg_14d']) if 'dCTLg_14d' in _ph_now and pd.notna(_ph_now['dCTLg_14d']) else 0.0
            _ph_hrv   = float(_ph_now['HRV_z'])    if 'HRV_z'    in _ph_now and pd.notna(_ph_now['HRV_z'])    else None
            _hx_ph = _ph_col.lstrip('#')
            _rp,_gp,_bp = int(_hx_ph[0:2],16),int(_hx_ph[2:4],16),int(_hx_ph[4:6],16)
            _hrv_txt = f" | HRV {_ph_hrv:+.2f}σ" if _ph_hrv is not None else ""
            st.markdown(
                f"<div style='background:rgba({_rp},{_gp},{_bp},0.10);"
                f"border-left:4px solid {_ph_col};"
                f"padding:8px 14px;border-radius:5px;margin-bottom:8px'>"
                f"<b>Fase actual:</b> {_ph_lbl} — {_ph_desc}<br>"
                f"<small>📅 {_ph_dias}d nesta fase | "
                f"ΔCTLγ {'↑' if _ph_dctl>0 else '↓'}{abs(_ph_dctl):.4f}/d"
                f"{_hrv_txt}</small></div>",
                unsafe_allow_html=True)
    except Exception:
        pass

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
    # ── Phase background on PMC (added after phase detection, placeholder for now)
    # Actual shapes added via _add_phase_bg() called after phase results available
    st.plotly_chart(_fig_pmc, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="pmc_main_chart")

    # ── RESUMO PMC ──

    if _phase_results:
        # ── Global + per-modality cards ───────────────────────────────────────
        # Weighted global phase card (CTLγ-weighted modal consensus)
        _gw_phase = _phase_results.get('_global_phase_today', None)
        _gw_weights = _phase_results.get('_modal_weights', {})
        _phase_mods = ['overall'] + [m for m in ['Bike','Row','Ski','Run']
                                      if m in _phase_results]
        _phase_cols = st.columns(len(_phase_mods))

        for _pi, _pm in enumerate(_phase_mods):
            _pdf  = _phase_results[_pm]
            _psum = phase_summary(_pdf, last_n=30)
            _pmod_label = '🌐 CTLγ Combined' if _pm == 'overall' else _pm

            with _phase_cols[_pi]:
                st.markdown(
                    f"<div style='background:{_psum['current_color']}22;"
                    f"border-left:4px solid {_psum['current_color']};"
                    f"padding:10px;border-radius:6px;margin-bottom:6px'>"
                    f"<b>{_pmod_label}</b><br>"
                    f"<span style='font-size:1.3em'>{_psum['current_label']}</span><br>"
                    f"<small>{_psum['current_desc']}</small><br><br>"
                    f"<small>📅 {_psum['streak_days']}d nesta fase"
                    f"{'  ✅ estável' if _psum['stable'] else '  🔄 recente'}</small><br>"
                    f"<small>CTLγ {_psum['current_ctlg']:.1f} | "
                    f"ΔCTL {'↑' if _psum['current_dctlg']>0 else '↓'}"
                    f"{abs(_psum['current_dctlg']):.4f}/d</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Weighted global phase card ────────────────────────────────────────
        if _gw_phase and _gw_weights:
            _gw_info  = PHASE_LABELS.get(_gw_phase, PHASE_LABELS['TRANSITION'])
            _gw_label = _gw_info['label']
            _gw_color = _gw_info['color']
            _gw_desc  = _gw_info['desc']
            _weights_str = ' | '.join(
                f"{m}: {int(w*100)}%" for m, w in sorted(
                    _gw_weights.items(), key=lambda x: -x[1]))
            st.info(
                f"**🏋️ Fase Global Ponderada (por CTLγ): {_gw_label}**  \n"
                f"{_gw_desc}  \n"
                f"Pesos: {_weights_str}  \n"
                f"*Diferente do 'CTLγ Combined' quando a modalidade dominante "
                f"está em fase distinta do total combinado.*"
            )


    # ── Phase timeline + download ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Fase de Treino Actual")
    with st.expander("📊 Timeline de Fases (últimos 180d)", expanded=False):
        _pov = _phase_results.get('overall') if _phase_results else None
        if _pov is not None:
            _pov180    = _pov.tail(180).copy()
            _dates_p   = _pov180['Data'].tolist()
            _fig_ph    = go.Figure()
            _prev_phase  = None
            _band_start  = None
            _prev_color  = '#95a5a6'
            for _ti, (_row_date, _row_phase, _row_color) in enumerate(
                    zip(_dates_p, _pov180['phase_smooth'].tolist(),
                        _pov180['phase_color'].tolist())):
                if _row_phase != _prev_phase:
                    if _band_start is not None:
                        _hx = _prev_color.lstrip('#')
                        _r, _g, _b = int(_hx[0:2],16), int(_hx[2:4],16), int(_hx[4:6],16)
                        _fig_ph.add_shape(type='rect',
                            x0=_band_start, x1=_row_date,
                            y0=0, y1=1, xref='x', yref='paper',
                            fillcolor=f'rgba({_r},{_g},{_b},0.15)',
                            line_width=0, layer='below')
                    _band_start = _row_date
                    _prev_phase = _row_phase
                    _prev_color = _row_color
            if _band_start is not None:
                _hx = _prev_color.lstrip('#')
                _r, _g, _b = int(_hx[0:2],16), int(_hx[2:4],16), int(_hx[4:6],16)
                _fig_ph.add_shape(type='rect',
                    x0=_band_start, x1=_dates_p[-1],
                    y0=0, y1=1, xref='x', yref='paper',
                    fillcolor=f'rgba({_r},{_g},{_b},0.15)',
                    line_width=0, layer='below')
            _fig_ph.add_trace(go.Scatter(
                x=_dates_p, y=_pov180['CTLg'].tolist(),
                name='CTLγ global',
                line=dict(color=CORES['azul'], width=2),
                hovertemplate='CTLγ: %{y:.1f}<extra></extra>'))
            if _pov180['HRV_z'].notna().any():
                _hrv_sc = _pov180['HRV_z'] * (_pov180['CTLg'].max() / 4)
                _fig_ph.add_trace(go.Scatter(
                    x=_dates_p, y=_hrv_sc.tolist(),
                    name='HRV trend (escala)',
                    line=dict(color=CORES['verde'], width=1.5, dash='dot'),
                    opacity=0.7,
                    hovertemplate='HRV_z: %{customdata:.2f}<extra></extra>',
                    customdata=_pov180['HRV_z'].tolist()))
            # Modal divergence markers
            _mod_colors_ph = {'Bike': CORES['vermelho'], 'Row': CORES['azul'],
                              'Ski': CORES['roxo'], 'Run': CORES['verde']}
            _overall_ph_map = dict(zip(
                [str(d)[:10] for d in _pov180['Data'].tolist()],
                _pov180['phase_smooth'].tolist()))
            for _mod_t in ['Bike','Row','Ski','Run']:
                if _mod_t not in _phase_results: continue
                _pdf_mod180 = _phase_results[_mod_t][
                    _phase_results[_mod_t]['Data'].astype(str).str[:10].isin(
                        [str(d)[:10] for d in _dates_p])].tail(180)
                if len(_pdf_mod180) == 0: continue
                _diff_rows = _pdf_mod180[_pdf_mod180.apply(
                    lambda r: r['phase_smooth'] != _overall_ph_map.get(str(r['Data'])[:10],''),
                    axis=1)]
                if len(_diff_rows) == 0: continue
                _fig_ph.add_trace(go.Scatter(
                    x=_diff_rows['Data'].tolist(), y=[0]*len(_diff_rows),
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=8,
                                color=_mod_colors_ph.get(_mod_t,'#888'),
                                line=dict(width=1, color='white')),
                    name=f'{_mod_t} ≠ overall',
                    text=_diff_rows['phase_smooth'].tolist(),
                    hovertemplate=f'<b>{_mod_t}</b>: %{{text}}<extra></extra>',
                    showlegend=True))
            _fig_ph.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                height=300,
                margin=dict(t=40, b=70, l=55, r=20),
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.30,
                            font=dict(color='#111', size=9),
                            bgcolor='rgba(255,255,255,0.9)'),
                font=dict(color='#111', size=10),
                title=dict(text='Timeline de Fases — CTLγ + HRV',
                           font=dict(size=12, color='#111')))
            _fig_ph.update_xaxes(showgrid=True, gridcolor='#ddd',
                                  tickfont=dict(size=9, color='#111'),
                                  linecolor='#333', tickcolor='#333', tickangle=-45)
            _fig_ph.update_yaxes(showgrid=True, gridcolor='#ddd',
                                  tickfont=dict(size=9, color='#111'),
                                  linecolor='#333', tickcolor='#333',
                                  title_text='CTLγ', title_font=dict(color='#111'))
            st.plotly_chart(_fig_ph, use_container_width=True,
                            config={'displayModeBar': False}, key="pmc_phase_timeline")
            _cols_leg = st.columns(3)
            for _li, (_pk, _pv) in enumerate(PHASE_LABELS.items()):
                _cols_leg[_li % 3].markdown(
                    f"<span style='color:{_pv['color']};font-weight:bold'>"
                    f"{_pv['label']}</span> — {_pv['desc']}",
                    unsafe_allow_html=True)
    # Download phases CSV
    _dl_phase_frames = []
    _dl_phase_cols = ['Data','phase','phase_smooth','phase_label',
                      'CTLg','dCTLg_14d','HRV_rel','WEED_z',
                      'CTLg_z','dCTLg_z','dias_na_fase']
    for _pm, _pdf in _phase_results.items():
        if not isinstance(_pdf, pd.DataFrame): continue
        _avail = [c for c in _dl_phase_cols if c in _pdf.columns]
        _pdfx  = _pdf[_avail].copy()
        _pdfx.insert(1, 'Modalidade', _pm)
        _dl_phase_frames.append(_pdfx)
    if _dl_phase_frames:
        _dl_phase_df = pd.concat(_dl_phase_frames, ignore_index=True)
        _dl_phase_df['Data'] = _dl_phase_df['Data'].astype(str)
        _dl_phase_df = _dl_phase_df.round(4)
        st.download_button(
            label="📥 Download Fases (.csv)",
            data=_dl_phase_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
            file_name="atheltica_fases_treino.csv",
            mime="text/csv", key="pmc_dl_phases")
        _exportable = [k for k,v in _phase_results.items() if isinstance(v, pd.DataFrame)]
        st.caption(f"Exporta fases para: {', '.join(_exportable)} | "
                   f"Thresholds: percentil rolante 60d | ΔCTLγ slope 14d")


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
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # GRÁFICO 2 — CTLγ por modalidade (FTLM fraccionário)
    # ════════════════════════════════════════════════════════════════════════

    st.subheader("📊 CTLγ por Modalidade — FTLM Fraccionário")
    st.caption(
        "Cada modalidade tem o seu próprio γ ajustado independentemente. "
        "A carga de Bike não prediz adaptações de Row — "
        "os sistemas energéticos e padrões de recuperação são diferentes."
    )

    _CORES_MOD_PMC = {
        'Bike': CORES['vermelho'], 'Row': CORES['azul'],
        'Ski':  CORES['roxo'],    'Run': CORES['verde'],
    }
    # Also check _ld_frac directly in case merge missed cols
    if len(_ld_frac) > 0:
        for _mc in ['CTLg_Bike','CTLg_Row','CTLg_Ski','CTLg_Run',
                    'CTL_Bike','CTL_Row','CTL_Ski','CTL_Run',
                    'ATL_Bike','ATL_Row','ATL_Ski','ATL_Run']:
            if _mc in _ld_frac.columns and _mc not in ld_plot.columns:
                _fi = _ld_frac.set_index('Data')
                ld_plot[_mc] = ld_plot['Data'].map(_fi[_mc]) if 'Data' in ld_plot.columns else None

    _mods_with_data = [m for m in ['Bike','Row','Ski','Run']
                       if f'CTLg_{m}' in ld_plot.columns
                       and ld_plot[f'CTLg_{m}'].notna().any()
                       and ld_plot[f'CTLg_{m}'].max() > 0]

    if _mods_with_data:
        # ── Subplots: 1 coluna por modalidade ────────────────────────────────────
        # Eixo Y esquerdo (y1): CTLγ fraccionário (escala própria por modalidade)
        # Eixo Y direito (y2): CTL + ATL clássicos (pontilhado) — escala menor
        # Assim o CTLγ domina visualmente e CTL/ATL servem de referência contextual
        _n_mods = len(_mods_with_data)

        # make_subplots with secondary_y per column
        from plotly.subplots import make_subplots as _msp
        fig_mod = _msp(
            rows=1, cols=_n_mods,
            subplot_titles=_mods_with_data,
            shared_yaxes=False,
            specs=[[{'secondary_y': True}] * _n_mods],
        )

        for _ci, _mod in enumerate(_mods_with_data, 1):
            _gi   = _info.get('mods', {}).get(_mod, {})
            _gv   = _gi.get('gamma_perf', 0.35)
            _rv   = _gi.get('r2_perf', 0.0)
            _nm   = _gi.get('n_mmp', 0)
            _src  = ('MMP5' if _gi.get('mmp_col','') == 'mmp5_pr_w'
                     else 'MMP20' if _nm >= 5 else 'CP')
            _cor  = _CORES_MOD_PMC.get(_mod, '#888')

            # ── Y1 (left): CTLγ fraccionário — linha sólida, escala própria ────
            _ctlg_col = f'CTLg_{_mod}'
            if _ctlg_col in ld_plot.columns and ld_plot[_ctlg_col].notna().any():
                fig_mod.add_trace(go.Scatter(
                    x=_dates,
                    y=ld_plot[_ctlg_col].tolist(),
                    mode='lines',
                    name=f'{_mod} CTLγ γ={_gv:.3f} R²={_rv:.2f} [{_src}]',
                    line=dict(color=_cor, width=2.5),
                    legendgroup=f'ctlg_{_mod}',
                    showlegend=True,
                    hovertemplate=f'<b>{_mod} CTLγ</b>: %{{y:.1f}}<extra></extra>',
                ), row=1, col=_ci, secondary_y=False)

            # ── Y2 (right): CTL + ATL clássicos — pontilhado/tracejado, opaco ─
            _ctl_col = f'CTL_{_mod}'
            _atl_col = f'ATL_{_mod}'

            if _ctl_col in ld_plot.columns and ld_plot[_ctl_col].notna().any():
                fig_mod.add_trace(go.Scatter(
                    x=_dates,
                    y=ld_plot[_ctl_col].tolist(),
                    mode='lines',
                    name=f'{_mod} CTL (42d)',
                    line=dict(color=_cor, width=1.5, dash='dot'),
                    opacity=0.50,
                    legendgroup=f'ctl_{_mod}',
                    showlegend=True,
                    hovertemplate=f'<b>{_mod} CTL</b>: %{{y:.1f}}<extra></extra>',
                ), row=1, col=_ci, secondary_y=True)

            if _atl_col in ld_plot.columns and ld_plot[_atl_col].notna().any():
                fig_mod.add_trace(go.Scatter(
                    x=_dates,
                    y=ld_plot[_atl_col].tolist(),
                    mode='lines',
                    name=f'{_mod} ATL (7d)',
                    line=dict(color=_cor, width=1.5, dash='dash'),
                    opacity=0.35,
                    legendgroup=f'atl_{_mod}',
                    showlegend=True,
                    hovertemplate=f'<b>{_mod} ATL</b>: %{{y:.1f}}<extra></extra>',
                ), row=1, col=_ci, secondary_y=True)

            # Y-axis labels
            fig_mod.update_yaxes(
                title_text='CTLγ', title_font=dict(size=8, color=_cor),
                showgrid=True, gridcolor='#eee',
                tickfont=dict(color='#111', size=8),
                row=1, col=_ci, secondary_y=False)
            fig_mod.update_yaxes(
                title_text='CTL/ATL', title_font=dict(size=8, color='#888'),
                showgrid=False,
                tickfont=dict(color='#888', size=8),
                row=1, col=_ci, secondary_y=True)

        fig_mod.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10),
            height=400,
            margin=dict(t=65, b=90, l=50, r=50),
            hovermode='x unified',
            barmode='overlay',
            legend=dict(orientation='h', y=-0.32,
                        font=dict(color='#111', size=9),
                        bgcolor='rgba(255,255,255,0.9)'),
            title=dict(
                text='CTLγ (Y esq, sólido) | CTL/ATL clássicos (Y dir, pontilhado/tracejado)',
                font=dict(size=12, color='#111')),
        )
        fig_mod.update_xaxes(showgrid=True, gridcolor='#eee',
                              tickangle=-45, tickfont=dict(color='#111', size=9))
        st.plotly_chart(fig_mod, use_container_width=True,
                        config={'displayModeBar': False, 'responsive': True},
                        key="pmc_ctlg_mod")

        # ── Interpretação γ por modalidade ──────────────────────────────────
        st.markdown("**🔍 Interpretação de γ por Modalidade**")
        _cols_mod = st.columns(len(_mods_with_data))
        for _ci2, _mod in enumerate(_mods_with_data):
            _gi2  = _info.get('mods', {}).get(_mod, {})
            _gv2  = _gi2.get('gamma_perf', 0.35)
            _rv2  = _gi2.get('r2_perf', 0.0)
            _nm2  = _gi2.get('n_mmp', 0)
            _nc2  = _gi2.get('n_cp', 0)
            _ns2  = _gi2.get('n_sessions', 0)
            _mmp_col_used = _gi2.get('mmp_col', 'mmp20_pr_w').replace('_pr_w','').upper()
            _perf_col_m   = _gi2.get('perf_col', 'icu_pm_cp')
            if _nm2 >= 5:
                _src2 = f'{_mmp_col_used} ({_nm2} PR)'
            elif _nc2 >= 5:
                _src2 = f'{_perf_col_m} ({_nc2} sessões)'
            else:
                _src2 = 'sem dados suficientes (γ=default 0.35)'
            _icon, _lbl, _mtxt, _r2txt = _interp_gamma(_gv2, _rv2, _mod)
            with _cols_mod[_ci2]:
                st.markdown(f"**{_icon} {_mod}**")
                st.metric("γ", f"{_gv2:.3f}", help=_lbl)
                st.metric("R²", f"{_rv2:.2f}", help=_src2)
                st.caption(_mtxt)
                st.caption(_r2txt)
    else:
        st.info("CTLγ por modalidade não disponível — sem dados suficientes.")

    # ── FMT Tensor — explicação + resultados actuais ──────────────────────────
    with st.expander("🧮 FMT Tensor κ — Curvatura do Estado Fisiológico", expanded=False):
        try:
            _kappa_last = (float(_ld_frac['FMT_kappa'].dropna().iloc[-1])
                if (len(_ld_frac) > 0 and 'FMT_kappa' in _ld_frac.columns
                    and _ld_frac['FMT_kappa'].notna().any()) else None)
            _lambda1_last = (float(_ld_frac['FMT_lambda1_frac'].dropna().iloc[-1])
                if (len(_ld_frac) > 0 and 'FMT_lambda1_frac' in _ld_frac.columns
                    and _ld_frac['FMT_lambda1_frac'].notna().any()) else None)
            _tensor_dim_v = int(_info.get('tensor_dim', 0))
            _tensor_names = _info.get('tensor_dim_names', 'CTLγ')
            _has_ws = _info.get('has_w_stress', False)
            _has_hq = _info.get('has_hq_drift', False)
            _has_sl = _info.get('has_sleep', False)

            st.markdown("**O que é o FMT κ (Functional Multidimensional Tensor)?**")
            st.markdown(
                "Proposto por Della Mattia (2019), o FMT substitui o TSS como métrica de referência. "
                "O TSS colapsa uma sessão complexa num único número — perde informação sobre como cada "
                "dimensão fisiológica mudou. O FMT constrói uma **matriz de covariância** das variações "
                "diárias de múltiplos sinais: `F(t) = cov(Δx)` sobre janela 28 dias, onde "
                "`x(t) = [CTLγ, HRV, WEED, Sleep, W', HR_drift]`. "
                "**κ = trace(F)** — curvatura escalar do estado fisiológico."
            )
            st.markdown(
                "**κ crescente** → múltiplas dimensões a mudar simultaneamente → stress não compensado  \n"
                "**κ decrescente** → adaptação a estabilizar, sinais a convergir"
            )

            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                if _kappa_last is not None:
                    st.metric("κ actual", f"{_kappa_last:.3f}",
                              help="Curvatura do tensor — mudanças simultâneas em todas as dimensões")
                else:
                    st.metric("κ actual", "N/D", help="Disponível após 28 dias de dados")
            with _c2:
                if _lambda1_last is not None:
                    _lam_pct = _lambda1_last * 100
                    _stress_lbl = ("🎯 Focal" if _lambda1_last > 0.65
                                   else ("⚖️ Misto" if _lambda1_last > 0.45
                                         else "🌐 Multissistémico"))
                    st.metric("λ₁/Σλ", f"{_lam_pct:.0f}%", delta=_stress_lbl,
                              help="Concentração do stress. >65%=focal, <45%=multissistémico")
                else:
                    st.metric("λ₁/Σλ", "N/D", help="Disponível após 28 dias de dados")
            with _c3:
                st.metric("Tensor", f"{_tensor_dim_v}×{_tensor_dim_v}",
                          help=f"Dimensões activas: {_tensor_names}")

            st.markdown(f"**Dimensões activas:** `{_tensor_names}`")

            _sig_rows = [
                ("CTLγ perf",  "✅ sempre",
                 "Carga fraccionária acumulada — dimensão Load"),
                ("HRV trend",
                 "✅" if ('HRV_trend' in _ld_frac.columns
                          and _ld_frac['HRV_trend'].notna().sum() >= 20) else "⚠️ < 20 dias",
                 "Estado autonómico — LnRMSSD intercept+slope 7d"),
                ("WEED_z",
                 "✅" if ('WEED_z' in _ld_frac.columns
                          and _ld_frac['WEED_z'].notna().sum() >= 20) else "⚠️ < 20 dias",
                 "Readiness subjectivo — z-score 28d stress+soreness+fatiga"),
                ("Sleep_z",    "✅" if _has_sl else "⚠️ sem dados",
                 "Qualidade sono — z-score 28d (separado do WEED, per paper)"),
                ("W' stress",  "✅" if _has_ws else "⚠️ sem AllWorkFTP/eW",
                 "AllWorkFTP_kJ / (eW_kJ) — fracção W' consumida acima de FTP"),
                ("HR drift",   "✅" if _has_hq else "⚠️ sem Hq1/Hq4",
                 "hq_4/hq_1 z-score 14d — intensidade intra-sessão (HR quartiles)"),
            ]
            _sig_df = pd.DataFrame(_sig_rows, columns=["Dimensão", "Estado", "Significado"])
            st.dataframe(_sig_df, hide_index=True, use_container_width=True)

            st.markdown("**Interpretação λ₁/Σλ:**")
            st.markdown(
                "- **>65%** → stress **focal**: uma dimensão domina (ex: só carga cresceu)  \n"
                "- **45-65%** → stress **misto**: duas dimensões perturbadas  \n"
                "- **<45%** → stress **multissistémico**: todas as dimensões em simultâneo — "
                "sinal de overreaching. Paper §12: FMT AUC=0.553 vs TSB clássico AUC=0.185."
            )
        except Exception as _fmt_err:
            st.caption(f"ℹ️ FMT tensor a calcular no próximo carregamento. ({type(_fmt_err).__name__})")


    # ── FTLM download CSV ─────────────────────────────────────────────────────
    if len(_ld_frac) > 0:
        _dl_cols_ftlm = [
            'Data', 'load_val', 'CTL', 'ATL', 'TSB', 'FTLM',
            'CTLg_perf', 'CTLg_rec', 'HRV_trend',
            'CTLg_Bike', 'CTLg_Row', 'CTLg_Ski', 'CTLg_Run',
            'WEED_z', 'sleep_z', 'w_stress', 'hq_drift_z',
            'wp_prime',
            'FMT_kappa', 'FMT_lambda1_frac', 'FMT_kappa_4d',
            'FMT_kappa_Bike', 'FMT_kappa_Row', 'FMT_kappa_Ski', 'FMT_kappa_Run',
        ]
        _avail_cols = [c for c in _dl_cols_ftlm if c in _ld_frac.columns]
        _dl_df = _ld_frac[_avail_cols].copy()
        _dl_df['Data'] = _dl_df['Data'].astype(str)
        _dl_df = _dl_df.round(4)
        st.download_button(
            label="📥 Download PMC/FTLM completo (.csv)",
            data=_dl_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
            file_name="atheltica_pmc_ftlm_completo.csv",
            mime="text/csv",
            key="pmc_dl_ftlm",
        )
        _new_cols = [c for c in ['sleep_z','w_stress','hq_drift_z','FMT_lambda1_frac'] if c in _ld_frac.columns and _ld_frac[c].notna().sum() > 0]
        if _new_cols:
            st.caption(f"Novos campos no CSV: {', '.join(_new_cols)} | Tensor {_info.get('tensor_dim',0)}×{_info.get('tensor_dim',0)}: {_info.get('tensor_dim_names','')}")
