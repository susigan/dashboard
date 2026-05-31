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

    # Guardar no session_state para que tab_visao_geral possa ler κ sem recalcular
    if len(_ld_frac) > 0:
        st.session_state['ld_frac_cache'] = _ld_frac

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


    # ══════════════════════════════════════════════════════════════════════════
    # MODELO HOMEOSTÁTICO & ÍNDICE ALOSTÁTICO
    # K1/K2/T1/T2 Hierarchical Bayesian NLSS — Della Mattia 2026
    # p̂(t) = p0 + K1·EWM(T1) − K2·EWM(T2)
    # Índice Alostático: delta normalizado de 6 dimensões entre dois períodos
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏠 Modelo Homeostático — Reserva de Performance")
    with st.expander("📖 O que é este modelo?", expanded=False):
        st.markdown("""
**Modelo Homeostático (K₁K₂T₁T₂ — Hierarchical Bayesian NLSS)**

Ao contrário do PMC clássico (CTL/ATL com τ fixos 42/7 dias), este modelo estima
os parâmetros **individualizados** para este atleta via regressão Bayesiana nos
testes de potência reais (Bike MMP20 por ano):
- **K₁** — ganho de fitness (magnitude da resposta adaptativa)
- **K₂** — ganho de fadiga (custo acumulado por sessão)
- **T₁** — τ fitness em dias (velocidade de absorção da adaptação)
- **T₂** — τ fadiga em dias (velocidade de dissipação da fadiga)

A **reserva de performance** p̂(t) é a diferença escalada entre fitness e fadiga:
```
p̂(t) = p₀ + K₁ · EWM(carga, T₁) − K₂ · EWM(carga, T₂)
```
O **Índice Alostático** mede se o atleta está a adaptar-se (homeostasia positiva)
ou a acumular sobrecarga não compensada (allostatic overload), comparando
6 dimensões fisiológicas entre dois períodos configuráveis.
        """)

    @st.cache_data(show_spinner="A estimar K₁K₂T₁T₂ (Bayesian NLSS)...", ttl=7200)
    def _cached_nlss(_key_da, _key_wc, _mod, da_arg, wc_arg):
        from utils.data import calcular_nlss
        return calcular_nlss(da_arg, df_wellness=wc_arg)

    _nlss_key_da = (str(da_full['Data'].max()) if 'Data' in da_full.columns else '', len(da_full))
    _nlss_key_wc = (str(wc['Data'].max()) if wc is not None and 'Data' in wc.columns else '',
                    len(wc) if wc is not None else 0)
    try:
        # Global NLSS usa Bike MMP20 (modalidade base para p̂ global)
        _nlss = _cached_nlss(_nlss_key_da, _nlss_key_wc, 'Bike', da_full, wc)
    except Exception as _ne:
        st.warning(f"NLSS indisponível: {_ne}")
        _nlss = None

    if _nlss and _nlss.get('error') is None and 'p_hat_series' in _nlss:
        _K1  = _nlss['K1'];  _K2 = _nlss['K2']
        _T1  = _nlss['T1'];  _T2 = _nlss['T2']
        _ph_s       = _nlss['p_hat_series'].copy()
        _ctl_n      = _nlss['CTL_nlss'].copy()
        _atl_n      = _nlss['ATL_nlss'].copy()
        _tsb_n      = _nlss['TSB_nlss'].copy()
        _test_dates = _nlss.get('test_dates', [])
        _test_watts = _nlss.get('test_watts', [])
        _n_tests    = _nlss.get('n_tests', 0)
        _fonte_t    = _nlss.get('fonte_testes', '—')
        _p0         = _nlss.get('p0', 200.0)

        # ── Cards K1 K2 T1 T2 ────────────────────────────────────────────────
        _pop_mu = [2.87, 4.09, 40.64, 6.584]
        _kc1, _kc2, _kc3, _kc4 = st.columns(4)
        def _dpop(val, mu):
            d = (val - mu) / max(abs(mu), 1e-9) * 100
            return f"{d:+.0f}% vs pop."
        _kc1.metric("K₁ (ganho fitness)",  f"{_K1:.3f}", delta=_dpop(_K1, _pop_mu[0]),
                    help=f"Prior pop.: {_pop_mu[0]}. Alto=resposta rápida ao treino.")
        _kc2.metric("K₂ (ganho fadiga)",   f"{_K2:.3f}", delta=_dpop(_K2, _pop_mu[1]),
                    help=f"Prior pop.: {_pop_mu[1]}. Alto=fadiga acumula mais rápido.")
        _kc3.metric("T₁ (τ fitness, dias)", f"{_T1:.1f}d", delta=_dpop(_T1, _pop_mu[2]),
                    help=f"Prior pop.: {_pop_mu[2]}d. Meia-vida do fitness.")
        _kc4.metric("T₂ (τ fadiga, dias)",  f"{_T2:.1f}d", delta=_dpop(_T2, _pop_mu[3]),
                    help=f"Prior pop.: {_pop_mu[3]}d. Meia-vida da fadiga.")
        st.caption(
            f"Estimado via Hierarchical Bayesian NLSS | "
            f"Testes: {_n_tests} ({_fonte_t}) | p₀={_p0:.0f}W"
        )

        # ── Selectores de período ────────────────────────────────────────────
        st.markdown("---")
        _ph_index = pd.DatetimeIndex(_ph_s.index)
        _ph_min   = _ph_index.min().date()
        _ph_max   = _ph_index.max().date()
        # Converter para datetime para aritmética, depois voltar a date
        _ph_max_dt = pd.Timestamp(_ph_max)
        _ph_min_dt = pd.Timestamp(_ph_min)

        _col_p1, _col_p2 = st.columns(2)
        with _col_p1:
            st.markdown("**Período Anterior**")
            _pa_start = st.date_input("Início", key="nlss_pa_start",
                value=max(_ph_min, (_ph_max_dt - pd.Timedelta(days=120)).date()),
                min_value=_ph_min, max_value=_ph_max)
            _pa_end = st.date_input("Fim", key="nlss_pa_end",
                value=max(_ph_min, (_ph_max_dt - pd.Timedelta(days=61)).date()),
                min_value=_ph_min, max_value=_ph_max)
        with _col_p2:
            st.markdown("**Período Recente**")
            _pr_start = st.date_input("Início", key="nlss_pr_start",
                value=max(_ph_min, (_ph_max_dt - pd.Timedelta(days=60)).date()),
                min_value=_ph_min, max_value=_ph_max)
            _pr_end = st.date_input("Fim", key="nlss_pr_end",
                value=_ph_max, min_value=_ph_min, max_value=_ph_max)

        if _pa_start > _pa_end:   _pa_start, _pa_end = _pa_end, _pa_start
        if _pr_start > _pr_end:   _pr_start, _pr_end = _pr_end, _pr_start

        _pa_mask = (_ph_index.date >= _pa_start) & (_ph_index.date <= _pa_end)
        _pr_mask = (_ph_index.date >= _pr_start) & (_ph_index.date <= _pr_end)
        _ph_ant  = _ph_s[_pa_mask]
        _ph_rec  = _ph_s[_pr_mask]

        if len(_ph_ant) < 5 or len(_ph_rec) < 5:
            st.warning("Período(s) com menos de 5 dias — alarga o intervalo.")
        else:
            # ── GRÁFICO: Reserva de Performance p̂(t) ─────────────────────────
            # ── GRÁFICO: Reserva de Performance — igual à foto ────────────────
            # Título + subtitle no topo; curvas sobrepostas; legenda integrada no gráfico
            # Botões Todo / Fit+banda / Só fits + modalidade

            _col_btns1, _col_btns2 = st.columns([2, 3])
            with _col_btns1:
                _mod_sel = st.radio(
                    "Modalidade",
                    ["Todas"] + [m for m in ["Bike","Row","Ski","Run"]
                                  if da_full is not None and "type" in da_full.columns
                                  and m in da_full["type"].unique()],
                    horizontal=True, key="nlss_mod_radio",
                    help="Todas = curva global. Por modalidade = NLSS recalculado.")
            with _col_btns2:
                _view_mode = st.radio(
                    "Visualização",
                    ["Todo", "Fit + banda", "Só fits"],
                    horizontal=True, key="nlss_view_mode",
                    help="Todo = dados brutos + fit + banda | Fit+banda = sem pontos brutos | Só fits = linhas limpas")

            # Suavização Savitzky-Golay
            def _smooth_sg(s, window=21, poly=3):
                try:
                    from scipy.signal import savgol_filter
                    arr = s.fillna(method="ffill").fillna(method="bfill").values
                    w   = min(window, len(arr) if len(arr) % 2 == 1 else len(arr)-1)
                    w   = max(w, poly+2) if w % 2 == 0 else max(w, poly+1)
                    return pd.Series(savgol_filter(arr, window_length=w, polyorder=poly), index=s.index)
                except Exception:
                    return s.rolling(14, min_periods=3, center=True).mean()

            def _band_sg(s, window=21):
                mu = _smooth_sg(s, window)
                sd = s.rolling(14, min_periods=3, center=True).std().fillna(0)
                return mu, mu + sd, mu - sd

            # Função para obter p̂ por modalidade
            def _get_phat_for_mod(mod):
                if mod == "Todas":
                    return _ph_s[_pa_mask], _ph_s[_pr_mask]
                if da_full is None or "type" not in da_full.columns: return _ph_s[_pa_mask], _ph_s[_pr_mask]
                _da_mod = da_full[da_full["type"] == mod].copy()
                if len(_da_mod) < 30:
                    st.caption(f"Poucos dados para {mod} ({len(_da_mod)} sessões) — usando global.")
                    return _ph_s[_pa_mask], _ph_s[_pr_mask]
                try:
                    from utils.data import calcular_nlss
                    _nlss_m = calcular_nlss(_da_mod, df_wellness=wc)
                    if _nlss_m and not _nlss_m.get("error") and "p_hat_series" in _nlss_m:
                        _ph_m = _nlss_m["p_hat_series"]
                        _ph_i = pd.DatetimeIndex(_ph_m.index)
                        return _ph_m[(_ph_i.date >= _pa_start) & (_ph_i.date <= _pa_end)],                                _ph_m[(_ph_i.date >= _pr_start) & (_ph_i.date <= _pr_end)]
                except Exception as _em:
                    st.caption(f"NLSS {mod}: {_em}")
                return _ph_s[_pa_mask], _ph_s[_pr_mask]

            # Cores: anterior = verde, recente = laranja/vermelho (igual à foto)
            _COR_ANT = "#27ae60"   # verde — Anterior
            _COR_REC = "#e67e22"   # laranja — Recente (como na foto)
            _MOD_CORES = {"Bike":"#e74c3c","Row":"#3498db","Ski":"#9b59b6","Run":"#27ae60"}

            # Construir figura
            _fig_phat = go.Figure()

            # Título e subtitle no topo (como na foto)
            _pa_lbl = f"Anterior · {_pa_start.strftime('%d/%m/%y')}→{_pa_end.strftime('%d/%m/%y')}"
            _pr_lbl = f"Recente · {_pr_start.strftime('%d/%m/%y')}→{_pr_end.strftime('%d/%m/%y')}"

            if _mod_sel == "Todas":
                # ── Modo Todas: todas as modalidades sobrepostas ─────────────
                # Global como referência cinza
                _ant_mu_g, _ant_hi_g, _ant_lo_g = _band_sg(_ph_s[_pa_mask])
                _rec_mu_g, _rec_hi_g, _rec_lo_g = _band_sg(_ph_s[_pr_mask])
                _ant_pico = float(_ant_mu_g.max()) if _ant_mu_g.notna().any() else float(_ph_s[_pa_mask].max())
                _rec_pico = float(_rec_mu_g.max()) if _rec_mu_g.notna().any() else float(_ph_s[_pr_mask].max())

                if _view_mode != "Só fits":
                    # Banda global cinza
                    for _s_g, _hi_g, _lo_g, _fill_g in [
                        (_ph_s[_pa_mask], _ant_hi_g, _ant_lo_g, "rgba(100,100,100,0.08)"),
                        (_ph_s[_pr_mask], _rec_hi_g, _rec_lo_g, "rgba(50,50,50,0.10)"),
                    ]:
                        _fig_phat.add_trace(go.Scatter(
                            x=list(_s_g.index)+list(_s_g.index[::-1]),
                            y=list(_hi_g.values)+list(_lo_g.values[::-1]),
                            fill="toself", fillcolor=_fill_g,
                            line=dict(width=0), showlegend=False, hoverinfo="skip"))

                # Linhas global
                _fig_phat.add_trace(go.Scatter(
                    x=list(_ph_s[_pa_mask].index), y=list(_ant_mu_g.values),
                    name=f"Global {_pa_lbl} n={len(_ph_s[_pa_mask])}",
                    line=dict(color="#666", width=1.5, dash="dash"), opacity=0.7,
                    hovertemplate="Global ant: %{y:.0f}<extra></extra>"))
                _fig_phat.add_trace(go.Scatter(
                    x=list(_ph_s[_pr_mask].index), y=list(_rec_mu_g.values),
                    name=f"Global {_pr_lbl} n={len(_ph_s[_pr_mask])}",
                    line=dict(color="#333", width=1.5, dash="dash"), opacity=0.7,
                    hovertemplate="Global rec: %{y:.0f}<extra></extra>"))

                # Modalidades individuais sobrepostas
                for _mod_i, _col_i in _MOD_CORES.items():
                    try:
                        if da_full is None or "type" not in da_full.columns: continue
                        _da_i = da_full[da_full["type"] == _mod_i]
                        if len(_da_i) < 20: continue
                        from utils.data import calcular_nlss
                        _nlss_i = calcular_nlss(_da_i, df_wellness=wc)
                        if not _nlss_i or _nlss_i.get("error") or "p_hat_series" not in _nlss_i: continue
                        _ph_i   = _nlss_i["p_hat_series"]
                        _ph_ix  = pd.DatetimeIndex(_ph_i.index)
                        _ma_i   = (_ph_ix.date >= _pa_start) & (_ph_ix.date <= _pa_end)
                        _mr_i   = (_ph_ix.date >= _pr_start) & (_ph_ix.date <= _pr_end)
                        for _mask_i, _dash_i, _sfx, _w in [
                            (_ma_i, "dot",   "ant", 1.8),
                            (_mr_i, "solid", "rec", 2.2),
                        ]:
                            _s_i = _ph_i[_mask_i]
                            if len(_s_i) < 5: continue
                            _mu_i = _smooth_sg(_s_i)
                            _fig_phat.add_trace(go.Scatter(
                                x=list(_s_i.index), y=list(_mu_i.values),
                                name=f"{_mod_i} {_sfx}",
                                line=dict(color=_col_i, width=_w, dash=_dash_i),
                                hovertemplate=f"{_mod_i}: %{{y:.0f}}<extra></extra>"))
                    except Exception: continue

            else:
                # ── Modo individual — fiel à foto ────────────────────────────
                _ph_ant_m, _ph_rec_m = _get_phat_for_mod(_mod_sel)
                if len(_ph_ant_m) < 5 or len(_ph_rec_m) < 5:
                    st.warning(f"Dados insuficientes para {_mod_sel}.")
                    _ph_ant_m, _ph_rec_m = _ph_s[_pa_mask], _ph_s[_pr_mask]

                _cor_a = _MOD_CORES.get(_mod_sel, _COR_ANT) if _mod_sel != "Todas" else _COR_ANT
                # Anterior sempre verde, recente sempre laranja (como na foto)
                _cor_a = _COR_ANT
                _cor_r = _COR_REC

                _ant_mu, _ant_hi, _ant_lo = _band_sg(_ph_ant_m)
                _rec_mu, _rec_hi, _rec_lo = _band_sg(_ph_rec_m)
                _ant_pico    = float(_ant_mu.max()) if _ant_mu.notna().any() else float(_ph_ant_m.max())
                _rec_pico    = float(_rec_mu.max()) if _rec_mu.notna().any() else float(_ph_rec_m.max())
                _ant_pico_dt = _ant_mu.idxmax() if _ant_mu.notna().any() else _ph_ant_m.idxmax()
                _rec_pico_dt = _rec_mu.idxmax() if _rec_mu.notna().any() else _ph_rec_m.idxmax()

                _ra, _ga, _ba = int(_cor_a[1:3],16), int(_cor_a[3:5],16), int(_cor_a[5:7],16)
                _rr, _gr, _br = int(_cor_r[1:3],16), int(_cor_r[3:5],16), int(_cor_r[5:7],16)

                if _view_mode != "Só fits":
                    # Banda ±1SD Anterior
                    _fig_phat.add_trace(go.Scatter(
                        x=list(_ph_ant_m.index)+list(_ph_ant_m.index[::-1]),
                        y=list(_ant_hi.values)+list(_ant_lo.values[::-1]),
                        fill="toself", fillcolor=f"rgba({_ra},{_ga},{_ba},0.15)",
                        line=dict(width=0), showlegend=True, name="±1 SD",
                        legendgroup="ant", hoverinfo="skip"))
                    # Banda ±1SD Recente
                    _fig_phat.add_trace(go.Scatter(
                        x=list(_ph_rec_m.index)+list(_ph_rec_m.index[::-1]),
                        y=list(_rec_hi.values)+list(_rec_lo.values[::-1]),
                        fill="toself", fillcolor=f"rgba({_rr},{_gr},{_br},0.15)",
                        line=dict(width=0), showlegend=True, name="±1 SD ",
                        legendgroup="rec", hoverinfo="skip"))

                # Linha Anterior fit
                _fig_phat.add_trace(go.Scatter(
                    x=list(_ph_ant_m.index), y=list(_ant_mu.values),
                    name=f"Anterior fit  n={len(_ph_ant_m)}",
                    legendgroup="ant",
                    line=dict(color=_cor_a, width=3),
                    hovertemplate="Anterior: %{y:.0f}<extra></extra>"))
                # Linha Recente fit
                _fig_phat.add_trace(go.Scatter(
                    x=list(_ph_rec_m.index), y=list(_rec_mu.values),
                    name=f"Recente fit  n={len(_ph_rec_m)}",
                    legendgroup="rec",
                    line=dict(color=_cor_r, width=3),
                    hovertemplate="Recente: %{y:.0f}<extra></extra>"))

                # Testes reais (só em "Todo")
                if _view_mode == "Todo" and _test_dates and _test_watts:
                    _td_dt = pd.to_datetime(_test_dates)
                    for _t_m, _t_c, _t_n, _t_grp in [
                        ((_td_dt.normalize().dt.date >= _pa_start) & (_td_dt.normalize().dt.date <= _pa_end),
                         _cor_a, "Testes ant", "ant"),
                        ((_td_dt.normalize().dt.date >= _pr_start) & (_td_dt.normalize().dt.date <= _pr_end),
                         _cor_r, "Testes rec", "rec"),
                    ]:
                        _td_s = [(d,w) for d,w,m in zip(_td_dt,_test_watts,_t_m) if m]
                        if _td_s:
                            _fig_phat.add_trace(go.Scatter(
                                x=[d for d,_ in _td_s], y=[w for _,w in _td_s],
                                mode="markers", name=_t_n, legendgroup=_t_grp,
                                marker=dict(color=_t_c, size=10, symbol="circle",
                                            line=dict(color="white", width=2)),
                                hovertemplate="Teste: %{y:.0f}W<extra></extra>"))

                # Círculo aberto no pico (como na foto)
                _fig_phat.add_trace(go.Scatter(
                    x=[_ant_pico_dt], y=[_ant_pico], mode="markers",
                    showlegend=False, legendgroup="ant",
                    marker=dict(color="white", size=13, symbol="circle",
                                line=dict(color=_cor_a, width=2.5)),
                    hoverinfo="skip"))
                _fig_phat.add_trace(go.Scatter(
                    x=[_rec_pico_dt], y=[_rec_pico], mode="markers",
                    showlegend=False, legendgroup="rec",
                    marker=dict(color="white", size=13, symbol="circle",
                                line=dict(color=_cor_r, width=2.5)),
                    hoverinfo="skip"))

                # Anotações de pico — texto inline sem seta (como na foto)
                _fig_phat.add_annotation(
                    x=_ant_pico_dt, y=_ant_pico,
                    text=f"<b>Anterior · pico {_ant_pico:.0f}</b>",
                    xshift=8, yshift=16, showarrow=False,
                    font=dict(size=12, color=_cor_a),
                    bgcolor="rgba(255,255,255,0.0)")
                _fig_phat.add_annotation(
                    x=_rec_pico_dt, y=_rec_pico,
                    text=f"<b>Recente · pico {_rec_pico:.0f}</b>",
                    xshift=8, yshift=16, showarrow=False,
                    font=dict(size=12, color=_cor_r),
                    bgcolor="rgba(255,255,255,0.0)")

                # Insight
                _delta_pico = _rec_pico - _ant_pico
                _delta_pct  = (_delta_pico / max(abs(_ant_pico), 1)) * 100

            # Legenda dos badges de período (como na foto — em cima do gráfico)
            _fig_phat.add_annotation(
                x=0, y=1.08, xref="paper", yref="paper",
                text=(f"<span style='color:{_COR_ANT}'>● {_pa_lbl}  n={len(_ph_s[_pa_mask])}</span>"
                      f"   <span style='color:{_COR_REC}'>● {_pr_lbl}  n={len(_ph_s[_pr_mask])}</span>"),
                showarrow=False, font=dict(size=11, color="#555"),
                xanchor="left", yanchor="bottom", align="left")

            # Subtitle (como na foto)
            _fig_phat.add_annotation(
                x=0, y=1.15, xref="paper", yref="paper",
                text="<b>MODELADO HOMEOSTÁTICO</b>",
                showarrow=False, font=dict(size=13, color="#222"),
                xanchor="left", yanchor="bottom")
            _fig_phat.add_annotation(
                x=0, y=1.095, xref="paper", yref="paper",
                text="Ciclo diário da reserva · ajuste Savitzky-Golay · banda ±1 SD",
                showarrow=False, font=dict(size=10, color="#888"),
                xanchor="left", yanchor="top")

            # Layout — fundo branco, legenda horizontal integrada em baixo
            _fig_phat.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                height=400,
                margin=dict(t=80, b=80, l=60, r=20),
                hovermode="x unified",
                legend=dict(
                    orientation="h", y=-0.22,
                    font=dict(color="#333", size=11),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#ddd", borderwidth=1,
                    itemsizing="constant",
                    traceorder="normal"),
                font=dict(color="#222", size=12),
                xaxis=dict(
                    tickfont=dict(size=11, color="#444"),
                    linecolor="#ccc", linewidth=1,
                    tickangle=-25,
                    gridcolor="rgba(0,0,0,0.04)",
                    showgrid=True),
                yaxis=dict(
                    title=dict(text="p̂(t) — Reserva (u.a.)",
                               font=dict(size=11, color="#666")),
                    tickfont=dict(size=11, color="#444"),
                    linecolor="#ccc", linewidth=1,
                    gridcolor="rgba(0,0,0,0.05)",
                    showgrid=True,
                    zeroline=True, zerolinecolor="#ccc", zerolinewidth=1),
            )
            st.plotly_chart(_fig_phat, use_container_width=True,
                            config={"displayModeBar": False}, key="nlss_phat_chart")

            # Insight — só no modo individual
            if _mod_sel != "Todas":
                _insight = (
                    f"O pico de reserva {'subiu' if _delta_pico >= 0 else 'desceu'} "
                    f"{abs(_delta_pico):.0f} unidades entre fases "
                    f"({_ant_pico:.0f} → {_rec_pico:.0f}, {_delta_pct:+.0f}%). "
                    + ("Melhor adaptação na fase recente." if _delta_pico >= 0
                       else "Fadiga acumulada a comprimir a reserva.")
                )
                st.info(f"💡 {_insight}")


            # ── ÍNDICE ALOSTÁTICO ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📊 Índice Alostático")
            st.caption(
                "6 dimensões · adaptação vs allostatic overload. "
                "Score normalizado ±1: positivo = adaptação / negativo = sobrecarga."
            )

            # Wellness helper
            _wc2 = None
            if wc is not None and len(wc) > 0:
                _wc2 = wc.copy()
                _wc2['Data'] = pd.to_datetime(_wc2['Data'])

            def _wc_period_mean(col_candidates, pa_s, pa_e, pr_s, pr_e):
                if _wc2 is None: return np.nan, np.nan
                _col = next((c for c in col_candidates if c in _wc2.columns), None)
                if _col is None: return np.nan, np.nan
                _a = _wc2[(_wc2['Data'].dt.date >= pa_s) &
                           (_wc2['Data'].dt.date <= pa_e)][_col].dropna()
                _r = _wc2[(_wc2['Data'].dt.date >= pr_s) &
                           (_wc2['Data'].dt.date <= pr_e)][_col].dropna()
                return (float(_a.mean()) if len(_a) > 2 else np.nan,
                        float(_r.mean()) if len(_r) > 2 else np.nan)

            # 6 dimensões — SEMPRE dados globais (independente da modalidade seleccionada)
            # Reserva pico: media da p̂(t) global no período (não por modalidade)
            # CTL/TSB: NLSS global (K1/K2/T1/T2 Bike MMP20)
            _alo_dims_raw = [
                {'dim': 'Reserva pico',
                 'ant': float(_ph_s[_pa_mask].mean()) if _pa_mask.any() else np.nan,
                 'rec': float(_ph_s[_pr_mask].mean()) if _pr_mask.any() else np.nan,
                 'uni': 'u.a.', 'bom_positivo': True},
                {'dim': 'CTL fitness',   'ant': float(_ctl_n[_pa_mask].mean()) if _pa_mask.any() else np.nan,
                 'rec': float(_ctl_n[_pr_mask].mean()) if _pr_mask.any() else np.nan,
                 'uni': 'au', 'bom_positivo': True},
                {'dim': 'Recovery TSB',  'ant': float(_tsb_n[_pa_mask].mean()) if _pa_mask.any() else np.nan,
                 'rec': float(_tsb_n[_pr_mask].mean()) if _pr_mask.any() else np.nan,
                 'uni': 'au', 'bom_positivo': True},
                {'dim': 'HRV matinal',
                 'ant': _wc_period_mean(['hrv','HRV'], _pa_start, _pa_end, _pr_start, _pr_end)[0],
                 'rec': _wc_period_mean(['hrv','HRV'], _pa_start, _pa_end, _pr_start, _pr_end)[1],
                 'uni': 'ms', 'bom_positivo': True},
                {'dim': 'HR repouso',
                 'ant': _wc_period_mean(['rhr','RHR'], _pa_start, _pa_end, _pr_start, _pr_end)[0],
                 'rec': _wc_period_mean(['rhr','RHR'], _pa_start, _pa_end, _pr_start, _pr_end)[1],
                 'uni': 'bpm', 'bom_positivo': False},
                {'dim': 'Sono',
                 'ant': _wc_period_mean(['sleep_quality','sleep_hours'], _pa_start, _pa_end, _pr_start, _pr_end)[0],
                 'rec': _wc_period_mean(['sleep_quality','sleep_hours'], _pa_start, _pa_end, _pr_start, _pr_end)[1],
                 'uni': '/5', 'bom_positivo': True},
            ]

            # Calcular scores
            _alo_scores = []
            _alo_rows   = []
            _n_dims_ok  = 0
            for _d in _alo_dims_raw:
                _a = _d['ant']; _r = _d['rec']
                if np.isnan(_a) or np.isnan(_r) or abs(_a) < 0.001:
                    _alo_rows.append({'Dimensão': _d['dim'], 'ant_v': np.nan,
                                      'rec_v': np.nan, 'delta_pct': np.nan,
                                      'uni': _d['uni'], 'score': np.nan})
                    continue
                _dp  = (_r - _a) / abs(_a) * 100
                _sgn = 1 if _d['bom_positivo'] else -1
                _sc  = _sgn * float(np.clip(_dp / 50.0, -1.0, 1.0))
                _alo_scores.append(_sc)
                _n_dims_ok += 1
                _alo_rows.append({'Dimensão': _d['dim'], 'ant_v': _a,
                                   'rec_v': _r, 'delta_pct': _dp,
                                   'uni': _d['uni'], 'score': _sc})

            _alo_total = float(np.clip(np.nanmean(_alo_scores) if _alo_scores else 0.0, -1.0, 1.0))

            # Classificação
            if _alo_total > 0.20:
                _alo_lbl = "GOOD ADAPTATION"; _alo_emoji = "🚀"; _alo_cor = '#27ae60'
                _alo_desc = "Adaptação alostática clara — o corpo responde positivamente à carga"
            elif _alo_total > -0.10:
                _alo_lbl = "ESTÁVEL";          _alo_emoji = "⚖️"; _alo_cor = '#f39c12'
                _alo_desc = "Sistema em equilíbrio — sem adaptação clara nem sobrecarga"
            else:
                _alo_lbl = "OVERLOAD";         _alo_emoji = "⚠️"; _alo_cor = '#e74c3c'
                _alo_desc = "Allostatic overload — o corpo não está a compensar a carga"

            _slider_pct = int((_alo_total + 1.0) / 2.0 * 100)

            # Card visual
            st.markdown(
                f"<div style='background:rgba(0,0,0,0.03);border:0.5px solid #ddd;"
                f"border-radius:10px;padding:16px 20px;margin-bottom:12px'>"
                f"<div style='font-size:13px;font-weight:500;color:#555;margin-bottom:6px'>"
                f"ÍNDICE ALOSTÁTICO · {_n_dims_ok}/6 dimensões · adaptação vs allostatic overload</div>"
                f"<div style='font-size:32px;font-weight:500;color:{_alo_cor};margin-bottom:4px'>"
                f"{_alo_emoji} {_alo_total:+.2f} "
                f"<span style='font-size:14px;color:#888'>/±1.00</span>"
                f"&nbsp;&nbsp;<span style='font-size:14px'>{_n_dims_ok}/6 dimensões</span></div>"
                f"<div style='font-size:13px;font-weight:500;color:{_alo_cor};margin-bottom:14px'>"
                f"{_alo_lbl}</div>"
                f"<div style='display:flex;justify-content:space-between;font-size:11px;"
                f"color:#888;margin-bottom:4px'>"
                f"<span>OVERLOAD</span><span>ESTÁVEL</span><span>ADAPTAÇÃO</span></div>"
                f"<div style='position:relative;height:10px;border-radius:5px;"
                f"background:linear-gradient(to right,#e74c3c,#f39c12 45%,#27ae60);"
                f"margin-bottom:8px'>"
                f"<div style='position:absolute;left:{_slider_pct}%;top:50%;"
                f"transform:translate(-50%,-50%);width:16px;height:16px;"
                f"border-radius:50%;background:{_alo_cor};"
                f"border:2px solid white;box-shadow:0 0 0 2px {_alo_cor}'></div>"
                f"</div>"
                f"<div style='font-size:12px;color:#555;margin-top:6px'>{_alo_desc}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Badges de período
            st.markdown(
                f"<div style='display:flex;gap:10px;margin-bottom:10px'>"
                f"<span style='font-size:12px;padding:3px 10px;border-radius:12px;"
                f"background:#27ae6022;color:#27ae60;border:1px solid #27ae6055'>"
                f"● Anterior · {_pa_start.strftime('%d/%m')}→{_pa_end.strftime('%d/%m')}</span>"
                f"<span style='font-size:12px;padding:3px 10px;border-radius:12px;"
                f"background:#e74c3c22;color:#e74c3c;border:1px solid #e74c3c55'>"
                f"● Recente · {_pr_start.strftime('%d/%m')}→{_pr_end.strftime('%d/%m')}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Gráfico de barras horizontal
            _alo_df_plot = pd.DataFrame([r for r in _alo_rows if not np.isnan(r['delta_pct'])])
            if len(_alo_df_plot) > 0:
                _alo_df_plot['cor'] = _alo_df_plot['score'].apply(
                    lambda s: '#27ae60' if s > 0.05 else ('#e74c3c' if s < -0.05 else '#95a5a6'))
                _fig_alo = go.Figure()
                _fig_alo.add_trace(go.Bar(
                    y=_alo_df_plot['Dimensão'].tolist(),
                    x=_alo_df_plot['delta_pct'].tolist(),
                    orientation='h',
                    marker_color=_alo_df_plot['cor'].tolist(),
                    text=[f"{v:+.0f}%" for v in _alo_df_plot['delta_pct'].tolist()],
                    textposition='outside',
                    hovertemplate='%{y}: %{x:+.1f}%<extra></extra>'))
                _fig_alo.add_vline(x=0, line_color='#888', line_width=1)
                _fig_alo.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=max(220, len(_alo_df_plot) * 44),
                    margin=dict(t=10, b=40, l=140, r=80),
                    font=dict(color='#222', size=12),
                    xaxis=dict(
                        title=dict(text='Variação % (Recente vs Anterior)',
                                   font=dict(size=12, color='#333')),
                        tickfont=dict(size=11, color='#333'),
                        linecolor='#555', gridcolor='rgba(0,0,0,0.05)',
                        ticksuffix='%'),
                    yaxis=dict(tickfont=dict(size=12, color='#222')),
                    showlegend=False)
                st.plotly_chart(_fig_alo, use_container_width=True,
                                config={'displayModeBar': False}, key='nlss_alo_bar')

                # Tabela resumo
                _tbl = []
                for r in _alo_rows:
                    if np.isnan(r['ant_v']):
                        _tbl.append({'Dimensão': r['Dimensão'], 'Anterior → Recente': 'sem dados', 'Δ': '—'})
                    else:
                        _a_f = f"{r['ant_v']:.0f}" if abs(r['ant_v']) >= 10 else f"{r['ant_v']:.1f}"
                        _r_f = f"{r['rec_v']:.0f}" if abs(r['rec_v']) >= 10 else f"{r['rec_v']:.1f}"
                        _tbl.append({'Dimensão': r['Dimensão'],
                                     'Anterior → Recente': f"{_a_f} → {_r_f} {r['uni']}",
                                     'Δ': f"{r['delta_pct']:+.0f}%"})
                st.dataframe(pd.DataFrame(_tbl), use_container_width=True, hide_index=True)

                if _alo_total < -0.10:
                    st.caption("◄ peor en fase reciente")
                elif _alo_total > 0.10:
                    st.caption("mejor en fase reciente ►")

    else:
        _err = _nlss.get('error', 'erro desconhecido') if _nlss else 'não executado'
        st.info(
            f"ℹ️ Modelo Homeostático indisponível: {_err}. "
            "São necessários pelo menos 2 testes anuais de Bike MMP20."
        )



    st.markdown("---")
    with st.expander("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)", expanded=False):
        st.caption(
            "Ajuste polinomial grau 2 e 3 sobre CTL/ATL. "
            "Movido de Análises para PMC — mesma métrica de carga."
        )
        # ── Secção 3: Polynomial CTL/ATL ────────────────────────────────────────
        st.subheader("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)")
        with st.spinner("Calculando polynomial fits..."):
            poli = calcular_polinomios_carga(da_full)

        if poli is None:
            st.warning("Sem dados suficientes para polynomial analysis.")
        else:
            _ld = poli['_ld']
            _MC_ANA = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}
            _POLY_COLORS = {'CTL': CORES['azul'], 'ATL': CORES['vermelho']}
            _POLY_DASH   = {2: 'dash', 3: 'dot'}

            def _poly_fig(res_met, ld_df, title):
                dates = pd.to_datetime(ld_df['Data']).tolist()
                fp = go.Figure()
                for met, cor in _POLY_COLORS.items():
                    y_raw = ld_df[met].tolist() if met in ld_df.columns else []
                    if y_raw:
                        fp.add_trace(go.Scatter(
                            x=dates, y=y_raw, mode='lines', name=met,
                            line=dict(color=cor, width=2), opacity=0.45,
                            hovertemplate=f'{met}: %{{y:.1f}}<extra></extra>'))
                    for grau_k, gd in res_met.get(met, {}).items():
                        grau_n = int(grau_k[-1])
                        r2 = gd['r2']; xarr = gd['x']; poly = gd['poly']
                        fp.add_trace(go.Scatter(
                            x=dates[:len(xarr)], y=poly(xarr).tolist(),
                            mode='lines',
                            name=f'{met} G{grau_n} R²={r2:.3f}',
                            line=dict(color=cor, width=2.5,
                                      dash=_POLY_DASH.get(grau_n, 'solid')),
                            hovertemplate=f'{met} poly: %{{y:.1f}}<extra></extra>'))
                fp.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=11), height=380,
                    margin=dict(t=55, b=80, l=55, r=20), hovermode='x unified',
                    title=dict(text=title, font=dict(size=13, color='#111')),
                    legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=10)),
                    xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')),
                    yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'),
                               title='CTL / ATL'))
                return fp

            # Overall
            _fonte_lbl = poli['_ld'].get('_fonte', poli['_ld']['_fonte'].iloc[0] if '_fonte' in poli['_ld'].columns else 'session_rpe')
            st.markdown(f"**Overall CTL vs ATL** — fonte: `{_fonte_lbl}`")
            st.caption("Mesma métrica de carga do PMC — valores comparáveis.")
            st.plotly_chart(
                _poly_fig(poli['overall'], _ld, 'CTL/ATL Overall — Polynomial Fit'),
                use_container_width=True, config=_MC_ANA, key="pmc_poly_overall")

            # ── Download Overall ──────────────────────────────────────────────────
            _dl_ov = _ld[['Data', 'CTL', 'ATL']].copy()
            _dl_ov['Data'] = _dl_ov['Data'].astype(str)
            for _m_ov in ['CTL', 'ATL']:
                for _gk_ov, _gd_ov in poli['overall'].get(_m_ov, {}).items():
                    _col_ov = f'{_m_ov}_poly_G{_gk_ov[-1]}'
                    _yp_ov  = _gd_ov['poly'](_gd_ov['x'])
                    _dl_ov[_col_ov] = np.nan
                    _dl_ov.iloc[:len(_yp_ov), _dl_ov.columns.get_loc(_col_ov)] = _yp_ov
            st.download_button(
                label="📥 Download Overall CTL/ATL + Polynomial (.csv)",
                data=_dl_ov.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                file_name="atheltica_poly_overall.csv",
                mime="text/csv",
                key="pmc_poly_dl_overall",
            )

            # Per modality — 2-column grid
            tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
            if tipos_poli:
                st.markdown("**Por Modalidade**")
                _mods_ord = [m for m in ['Bike', 'Row', 'Ski', 'Run'] if m in tipos_poli]
                for _mi in range(0, len(_mods_ord), 2):
                    _cols = st.columns(2)
                    for _ci, mod in enumerate(_mods_ord[_mi:_mi+2]):
                        res_mod = poli[tipos_poli[mod]]
                        # Get x range from first available metric
                        _xarr = None
                        for _m in ['CTL', 'ATL']:
                            for _gd in res_mod.get(_m, {}).values():
                                _xarr = _gd['x']; break
                            if _xarr is not None: break
                        if _xarr is None: continue
                        # Build ld_mod with raw CTL/ATL values from stored 'y'
                        _ld_mod = _ld[['Data']].iloc[:len(_xarr)].copy().reset_index(drop=True)
                        for _m in ['CTL', 'ATL']:
                            _best = (res_mod.get(_m, {}).get('grau2') or
                                     res_mod.get(_m, {}).get('grau3'))
                            _ld_mod[_m] = pd.Series(_best['y'][:len(_xarr)] if _best else
                                                      np.zeros(len(_xarr)))
                        with _cols[_ci]:
                            st.plotly_chart(
                                _poly_fig(res_mod, _ld_mod,
                                          f'{mod} — CTL/ATL Polynomial Fit'),
                                use_container_width=True,
                                config=_MC_ANA, key=f"pmc_poly_{mod}")
                            # Download per modality
                            _dl_mod = _ld_mod[['Data', 'CTL', 'ATL']].copy()
                            _dl_mod['Data'] = _dl_mod['Data'].astype(str)
                            for _m2 in ['CTL', 'ATL']:
                                for _gk2, _gd2 in res_mod.get(_m2, {}).items():
                                    _col2 = f'{_m2}_poly_{_gk2[-1]}'
                                    _yp2 = _gd2['poly'](_gd2['x'])
                                    _dl_mod[_col2] = np.nan
                                    _dl_mod.iloc[:len(_yp2), _dl_mod.columns.get_loc(_col2)] = _yp2
                            st.download_button(
                                label=f"📥 {mod} (.csv)",
                                data=_dl_mod.round(3).to_csv(
                                    index=False, sep=';', decimal=',').encode('utf-8'),
                                file_name=f"atheltica_poly_{mod.lower()}.csv",
                                mime="text/csv",
                                key=f"pmc_poly_dl_{mod}",
                            )

    st.subheader("📊 Resumo PMC")
    tsb_v = u['TSB']
    atl_v = u['ATL']
    ctl_v = u['CTL']
    # Zonas calibradas para este atleta (Auto-Runner 1ano, estado óptimo CV%<Q33+HRV>média)
    # Estado óptimo: ATL=40-61, CTL=46-53, TSB=-12 a +3  |  Estado mau: ATL=37, CTL=37, TSB=-0.2
    if   tsb_v >   3: tsb_i = "🟢 Forma — TSB acima do range óptimo (+3). Janela de performance."
    elif tsb_v >= -12: tsb_i = "🟢 Óptimo — TSB no range calibrado (−12 a +3). Estado de adaptação."
    elif tsb_v >= -20: tsb_i = "🟡 Atenção — TSB abaixo do range óptimo. Monitorar HRV."
    elif tsb_v >= -30: tsb_i = "🔴 Fatigado — carga elevada. Reduzir intensidade."
    else:              tsb_i = "⛔ Sobrecarregado — reduzir carga imediatamente."

    # ATL calibrado: óptimo 40-61
    if   atl_v > 63:  atl_i = "⚠️ Acima do range óptimo (>63) — risco HRV instável em ~27d"
    elif atl_v >= 40: atl_i = "✅ Range óptimo (40–61) — zone de adaptação e HRV estável"
    elif atl_v >= 25: atl_i = "🟡 Abaixo do range óptimo (<40) — carga insuficiente para adaptação"
    else:             atl_i = "⬇️ ATL muito baixo — fase de deload ou inactividade"

    # CTL calibrado: óptimo 46-53
    if   ctl_v > 55:  ctl_i = "🟡 CTL alto (>55) — monitorar CV% HRV"
    elif ctl_v >= 46: ctl_i = "✅ Range óptimo (46–53) — fitness na zona calibrada"
    elif ctl_v >= 35: ctl_i = "🟡 CTL em crescimento — abaixo do óptimo calibrado"
    else:             ctl_i = "⬇️ CTL baixo — condição base em desenvolvimento"

    resumo = pd.DataFrame([
        {'Métrica': 'CTL (Fitness atual)',  'Valor': f"{ctl_v:.1f}",
         'Interpretação': ctl_i},
        {'Métrica': 'ATL (Fadiga atual)',   'Valor': f"{atl_v:.1f}",
         'Interpretação': atl_i},
        {'Métrica': 'TSB (Forma atual)',    'Valor': f"{tsb_v:+.1f}",
         'Interpretação': tsb_i},
        {'Métrica': 'CTL max histórico',    'Valor': f"{ld['CTL'].max():.1f}",
         'Interpretação': 'Pico de fitness no período carregado.'},
        {'Métrica': 'ATL max histórico',    'Valor': f"{ld['ATL'].max():.1f}",
         'Interpretação': 'Pico de fadiga no período carregado.'},
    ])
    st.dataframe(resumo, width="stretch", hide_index=True)
    st.caption("⚙️ Zonas calibradas para este atleta — Auto-Runner 1ano (CV%<Q33 + HRV>média, N=167d). "
               "ATL óptimo 40–61 | CTL óptimo 46–53 | TSB óptimo −12 a +3.")

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

            # Status detalhado — verifica directamente _ld_frac para diagnóstico
            def _col_status(col, min_n=20):
                if col not in _ld_frac.columns:
                    return f"❌ coluna ausente"
                n = int(_ld_frac[col].notna().sum())
                if n == 0:   return f"❌ 0 valores"
                if n < min_n: return f"⚠️ {n} valores (mín {min_n})"
                return f"✅ {n} valores"

            _sig_rows = [
                ("CTLγ perf",  _col_status('CTLg_perf'),
                 "Carga fraccionária acumulada — dimensão Load"),
                ("HRV trend",  _col_status('HRV_trend'),
                 "Estado autonómico — LnRMSSD intercept+slope 7d"),
                ("WEED_z",     _col_status('WEED_z'),
                 "Readiness — z-score 28d de stress+soreness+fatiga"),
                ("sleep_z",    _col_status('sleep_z'),
                 "Qualidade sono — z-score 28d (escala 1=mau, 5=óptimo → sem inversão)"),
                ("w_stress",   _col_status('w_stress'),
                 "AllWorkFTP_kJ / (eW_kJ) — fracção W' consumida acima de FTP"),
                ("hq_drift_z", _col_status('hq_drift_z'),
                 "Hq4/Hq1 z-score 14d — intensidade intra-sessão (HR quartiles)"),
                ("FMT_kappa",  _col_status('FMT_kappa'),
                 "κ = trace(F) — curvatura escalar"),
                ("λ₁/Σλ",      _col_status('FMT_lambda1_frac'),
                 "Concentração de stress — focal vs multissistémico"),
            ]
            _sig_df = pd.DataFrame(_sig_rows, columns=["Sinal", "Estado", "Significado"])
            st.dataframe(_sig_df, hide_index=True, use_container_width=True)

            with st.expander("🔍 Diagnóstico de colunas", expanded=False):
                st.markdown("**Colunas presentes em `_ld_frac`:**")
                _diag = []
                for _dc in ['HRV_trend','WEED_z','sleep_z','w_stress','hq_drift_z',
                             'FMT_kappa','FMT_lambda1_frac','wp_prime']:
                    _inn = _dc in _ld_frac.columns if len(_ld_frac) > 0 else False
                    _nv  = int(_ld_frac[_dc].notna().sum()) if _inn else 0
                    _diag.append({'Coluna': _dc, 'Existe': '✅' if _inn else '❌', 'Non-NaN': _nv})
                st.dataframe(pd.DataFrame(_diag), hide_index=True)
                st.markdown("**Wellness — colunas com sleep/sono:**")
                if wc is not None and len(wc) > 0:
                    _slp_cols = [c for c in wc.columns if 'sleep' in c.lower() or 'sono' in c.lower()]
                    st.text(f"sleep_quality mapeada: {'sleep_quality' in wc.columns}")
                    st.text(f"Colunas encontradas: {_slp_cols}")
                    if 'sleep_quality' in wc.columns:
                        _sq = wc['sleep_quality'].dropna()
                        st.text(f"Registos: {len(_sq)} | min={_sq.min():.0f} max={_sq.max():.0f} mean={_sq.mean():.2f}")
                else:
                    st.text("wc: vazio ou None")

            # Raw columns diagnostic — shows ALL columns in _ld_frac
            with st.expander("🔬 Diagnóstico: colunas disponíveis em _ld_frac", expanded=False):
                _avail = sorted(_ld_frac.columns.tolist())
                st.caption(f"Total: {len(_avail)} colunas")
                st.code(', '.join(_avail))
                # Show key columns non-null counts
                _key_cols = ['sleep_z','w_stress','hq_drift_z','FMT_kappa',
                             'FMT_lambda1_frac','WEED_z','HRV_trend','wp_prime']
                _diag = {c: int(_ld_frac[c].notna().sum()) if c in _ld_frac.columns else 0
                         for c in _key_cols}
                st.json(_diag)

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
