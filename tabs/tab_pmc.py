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
    with st.spinner("A calcular FTLM fraccionário (γ dual)..."):
        _ld_frac, _info = calcular_series_carga(da_full, df_wellness=wc, ate_hoje=True)

    if len(_ld_frac) > 0 and 'FTLM' in _ld_frac.columns:
        # Align fractional ld with our ld (may have different date range)
        _frac_idx = _ld_frac.set_index('Data')
        # Copy all fractional columns (FTLM, CTLg_perf, CTLg_rec, CTLg_Bike, etc.)
        _frac_cols = ['FTLM', 'CTLg_perf', 'CTLg_rec', 'HRV_trend',
                      'CTLg_Bike', 'CTLg_Row', 'CTLg_Ski', 'CTLg_Run',
                      'CTL_Bike', 'CTL_Row', 'CTL_Ski', 'CTL_Run']
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
                    'CTL_Bike','CTL_Row','CTL_Ski','CTL_Run']:
            if _mc in _ld_frac.columns and _mc not in ld_plot.columns:
                _fi = _ld_frac.set_index('Data')
                ld_plot[_mc] = ld_plot['Data'].map(_fi[_mc]) if 'Data' in ld_plot.columns else None

    _mods_with_data = [m for m in ['Bike','Row','Ski','Run']
                       if f'CTLg_{m}' in ld_plot.columns
                       and ld_plot[f'CTLg_{m}'].notna().any()
                       and ld_plot[f'CTLg_{m}'].max() > 0]

    if _mods_with_data:
        # Subplots: 1 coluna por modalidade, cada uma com CTL (clássico) + CTLγ
        _n_mods = len(_mods_with_data)
        fig_mod = make_subplots(
            rows=1, cols=_n_mods,
            subplot_titles=_mods_with_data,
            shared_yaxes=False,
        )
        for _ci, _mod in enumerate(_mods_with_data, 1):
            _gi   = _info.get('mods', {}).get(_mod, {})
            _gv   = _gi.get('gamma_perf', 0.35)
            _rv   = _gi.get('r2_perf', 0.0)
            _nm   = _gi.get('n_mmp', 0)
            _src  = 'MMP20' if _nm >= 5 else 'CP'
            _cor  = _CORES_MOD_PMC.get(_mod, '#888')

            # CTL clássico desta modalidade
            _ctl_col = f'CTL_{_mod}'
            if _ctl_col in ld_plot.columns and ld_plot[_ctl_col].notna().any():
                fig_mod.add_trace(go.Scatter(
                    x=_dates, y=ld_plot[_ctl_col].tolist(),
                    mode='lines', name=f'{_mod} CTL',
                    line=dict(color=_cor, width=2, dash='dot'),
                    opacity=0.55, legendgroup=_mod,
                    showlegend=(_ci == 1),
                    hovertemplate=f'<b>{_mod} CTL</b>: %{{y:.1f}}<extra></extra>',
                ), row=1, col=_ci)

            # CTLγ fraccionário desta modalidade
            _ctlg_col = f'CTLg_{_mod}'
            fig_mod.add_trace(go.Scatter(
                x=_dates, y=ld_plot[_ctlg_col].tolist(),
                mode='lines', name=f'{_mod} CTLγ γ={_gv:.3f} [{_src}]',
                line=dict(color=_cor, width=2.5),
                legendgroup=_mod,
                showlegend=True,
                hovertemplate=f'<b>{_mod} CTLγ</b>: %{{y:.1f}}<extra></extra>',
            ), row=1, col=_ci)

        fig_mod.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10),
            height=340,
            margin=dict(t=60, b=80, l=45, r=20),
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.30, font=dict(color='#111', size=9)),
            title=dict(text='CTL (pontilhado) + CTLγ (sólido) por Modalidade',
                       font=dict(size=13, color='#111')),
        )
        fig_mod.update_xaxes(showgrid=True, gridcolor='#eee',
                              tickangle=-45, tickfont=dict(color='#111', size=9))
        fig_mod.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111', size=9))
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
            _src2 = f'{_mmp_col_used} ({_nm2} PR)' if _nm2 >= 5 else f'CP ({_nc2} sess)'
            _icon, _lbl, _mtxt, _r2txt = _interp_gamma(_gv2, _rv2, _mod)
            with _cols_mod[_ci2]:
                st.markdown(f"**{_icon} {_mod}**")
                st.metric("γ", f"{_gv2:.3f}", help=_lbl)
                st.metric("R²", f"{_rv2:.2f}", help=_src2)
                st.caption(_mtxt)
                st.caption(_r2txt)
    else:
        st.info("CTLγ por modalidade não disponível — sem dados suficientes.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # FTLM — Resultado actual + explicação científica completa
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔁 FTLM — Fractional Training Load Memory")

    _ftlm_v  = float(u.get('FTLM', 0) or 0)
    _ctl_v   = float(u['CTL'])
    _pct     = (_ftlm_v / _ctl_v * 100) if _ctl_v > 0 else 0
    _mods_gi = _info.get('mods', {})
    _best_m  = _info.get('best_mod', '—')

    if   _pct > 110: _ftlm_i = "⚠️ Carga muito acima do crónico — risco de overreaching"
    elif _pct > 100: _ftlm_i = "🔴 Carga acima do CTL — fase de acumulação/sobrecarga"
    elif _pct >  90: _ftlm_i = "🟡 Carga estável — manutenção/tapering leve"
    elif _pct >  75: _ftlm_i = "🟢 Tapering activo — carga baixa, forma a subir"
    else:             _ftlm_i = "⬇️ Destreino — carga muito abaixo do crónico"

    # ── Tabela resumo por modalidade ──────────────────────────────────────
    _rows_mod = []
    for _mod in ['Bike','Row','Ski','Run']:
        _gi = _mods_gi.get(_mod, {})
        if _gi.get('n_sessions', 0) == 0:
            continue
        _gv   = _gi.get('gamma_perf', 0.35)
        _rv   = _gi.get('r2_perf',   0.0)
        _nm   = _gi.get('n_mmp',     0)
        _nc   = _gi.get('n_cp',      0)
        _ns   = _gi.get('n_sessions', 0)
        _mem  = ('Muito longa (>6m)' if _gv < 0.20
                 else 'Longa (3-6m)'  if _gv < 0.35
                 else 'Moderada (6-12s)' if _gv < 0.60
                 else 'Curta (<6s)')
        _src  = f'MMP ({_nm} PR)' if _nm >= 5 else f'CP ({_nc} sess)'
        _rows_mod.append({
            'Modalidade': _mod, 'γ': f'{_gv:.3f}', 'R²': f'{_rv:.2f}',
            'Memória': _mem, 'Fonte γ': _src, 'Sessões': _ns,
        })
    if _rows_mod:
        st.dataframe(pd.DataFrame(_rows_mod),
                     use_container_width=True, hide_index=True)

    _max_lag_display = min(365, len(ld))  # max history used by kernel
    with st.expander("📖 Como funciona o FTLM Fraccionário — Guia de Interpretação", expanded=False):
        st.markdown("## O problema do CTL clássico")
        st.markdown(
            "O PMC usa EWM com τ=42 dias fixo:"
            " `CTL(t) = e^(-1/42)·CTL(t-1) + (1-e^(-1/42))·Load(t)`  \n"
            "**Problema 1 — Saturação:** o CTL não cresce sem limite (converge para τ=42).  \n"
            "**Problema 2 — Amnésia exponencial:** treino de 3 semanas atrás pesa <50% do de hoje, "
            "independentemente do atleta ou do desporto."
        )
        st.markdown("---")
        st.markdown("## O kernel fraccionário (Riemann-Liouville)")
        st.code(
            "CTLγ(t) = (1/Γ(γ)) · Σ Load(t-k) · k^(γ-1)\n"
            "         k=1..t\n"
            "\n"
            "∫ τ^(γ-1) dτ = t^γ/γ  →  cresce sem limite (sem saturação)",
            language="text"
        )
        st.markdown(
            "O decaimento é **hiperbólico** (lei de potência), não exponencial.  \n"
            "Treinos de 6+ meses atrás continuam a influenciar CTLγ — "
            "reflecte a base aeróbica construída ao longo de anos."
        )
        st.markdown("---")
        st.markdown("## O parâmetro γ — tabela de interpretação")
        st.markdown(
            "| γ | Memória | Significado fisiológico |\n"
            "|---|---|---|\n"
            "| 0.10–0.20 | Muito longa (meses) | Capilarização, mitocôndrias, economia de movimento |\n"
            "| 0.25–0.40 | Longa (semanas-meses) | Base aeróbica crónica — óptimo para resistência |\n"
            "| 0.40–0.60 | Moderada | Equilíbrio carga recente/histórica |\n"
            "| 0.70–0.90 | Curta (dias) | Reactivo, similar ao ATL |\n"
            "| → 1.0 | Exponencial | Comporta-se como CTL clássico |"
        )
        st.caption("Imbach et al. (2021): γ≈0.35 melhorou predição de potência em 18% vs Banister.")
        st.markdown("---")
        st.markdown("## Como o fitting é feito por modalidade")
        st.markdown(
            "Para cada modalidade (Bike, Row, Ski, Run) independentemente:  \n"
            "**Passo 1 — Carga modal:** só sessões dessa modalidade constroem `Load_mod(t)`.  \n"
            "**Passo 2 — Proxy de performance:** `icu_pm_cp` (cada sessão) "
            "+ MMP PR confirmados (`is_pr=True`, data exacta).  \n"
            "**Passo 3 — Grid search γ ∈ [0.10, 0.90]:** maximiza R²(CTLγ_mod ↔ performance_mod)."
        )
        if _rows_mod:
            _mod_lines = "\n".join(
                f"- **{r['Modalidade']}**: γ={r['γ']} (R²={r['R²']}, {r['Fonte γ']}, {r['Memória']})"
                for r in _rows_mod
            )
            st.markdown("**Resultado actual por modalidade:**")
            st.markdown(_mod_lines)
        st.markdown("---")
        st.markdown("## CTLγ_perf vs CTLγ_recovery")
        st.markdown(
            f"**CTLγ_perf (γ={_gp:.3f}, R²={_r2p:.2f})** — modelado sobre **{_best_m}** (maior R²):  \n"
            "- *Quanto fitness acumulado prediz a performance actual?*  \n"
            "- γ baixo → historial de anos pesa mais do que forma recente  \n"
            "- Útil para planear blocos de base, tapering e peaks"
        )
        st.markdown(
            f"**CTLγ_rec (γ={_gr:.3f}, R²={_r2r:.2f})** — fitado sobre LnRMSSD trend (HRV janela 7d):  \n"
            "- *Quanto CTLγ(t-1) prediz a recuperação autonómica de hoje?*  \n"
            "- Usa LnRMSSD (não RMSSD bruto) — distribuição normal, sensibilidade linear  \n"
            "- Lag=1 dia: ontem prediz hoje"
        )
        st.markdown(
            "**Quando divergem:**  \n"
            "- CTLγ_perf ↑ + CTLγ_rec ↓ → fitness alto, recuperação autonómica fraca "
            "→ risco de overreaching silencioso  \n"
            "- CTLγ_perf ↓ + CTLγ_rec ↑ → boa recuperação, fitness a decair "
            "→ indicação para retomar carga"
        )
        st.markdown("---")
        st.markdown("## Resultado actual")
        st.markdown(
            f"| Métrica | Valor |\n"
            f"|---|---|\n"
            f"| FTLM actual (CTLγ activo) | {_ftlm_v:.1f} |\n"
            f"| CTL clássico | {_ctl_v:.1f} |\n"
            f"| FTLM / CTL | {_pct:.0f}% |\n"
            f"| γ activo | {best_g:.3f} ({_gsrc}) |\n"
            f"| Interpretação | {_ftlm_i} |"
        )
        st.markdown(
            f"**Diferença crítica para o ATL:** O ATL esquece treinos de >21 dias. "
            f"O FTLM com γ={best_g:.3f} atribui peso a treinos de até {_max_lag_display} dias atrás."
        )
    # ── Resultado global combinado ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("\U0001f310 Resultado Global — Todas as Modalidades Combinadas")
    _mods_gi2 = _info.get('mods', {})
    _tot_sess = sum(_mods_gi2.get(m, {}).get('n_sessions', 0) for m in ['Bike','Row','Ski','Run'])
    if _tot_sess > 0:
        _gamma_w = sum(_mods_gi2.get(m,{}).get('gamma_perf',0.35) * _mods_gi2.get(m,{}).get('n_sessions',0)
                       for m in ['Bike','Row','Ski','Run']) / _tot_sess
        _r2_w    = sum(_mods_gi2.get(m,{}).get('r2_perf',0.0) * _mods_gi2.get(m,{}).get('n_sessions',0)
                       for m in ['Bike','Row','Ski','Run']) / _tot_sess
    else:
        _gamma_w, _r2_w = best_g, 0.0
    _icon_g, _lbl_g, _mtxt_g, _r2txt_g = _interp_gamma(_gamma_w, _r2_w, 'multi-sport')
    _best_m2 = _info.get('best_mod', 'N/A')
    _gc1, _gc2, _gc3 = st.columns(3)
    _gc1.metric("gamma ponderado (multi-sport)", f"{_gamma_w:.3f}",
                help="Media de gamma por modalidade ponderada pelo num de sessoes")
    _gc2.metric("R2 medio ponderado", f"{_r2_w:.2f}")
    _gc3.metric("Modalidade dominante", _best_m2,
                help="Modalidade cujo gamma tem R2 mais alto")
    st.info("\n\n".join([
        f"{_icon_g} Memoria global: {_lbl_g}",
        _mtxt_g, _r2txt_g,
        f"FTLM actual: {_ftlm_v:.1f} | CTL: {_ctl_v:.1f} | FTLM/CTL: {_pct:.0f}% | {_ftlm_i}",
    ]))

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Dados PMC")
    _dl_cols = [c for c in ['Data','load_val','CTL','ATL','TSB','FTLM',
                             'CTLg_perf','CTLg_rec'] + [f'CTLg_{m}' for m in ['Bike','Row','Ski','Run']]
                if c in ld.columns]
    _dl_df = ld[_dl_cols].copy()
    _dl_df['Data'] = _dl_df['Data'].astype(str)
    _dl_df = _dl_df.round(3)
    st.download_button(
        label="📥 Download CTL/ATL/TSB/FTLM/CTLγ por modalidade (.csv)",
        data=_dl_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
        file_name="atheltica_pmc_ftlm_completo.csv",
        mime="text/csv",
        key="pmc_dl_csv",
    )
    st.caption(f"Exporta {len(_dl_df)} dias | CTL, ATL, TSB clássicos + CTLγ_perf, CTLγ_rec, CTLγ_Bike/Row/Ski/Run")


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_volume.py
# ════════════════════════════════════════════════════════════════════════════
