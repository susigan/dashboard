from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

MC = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}

_CORES_MOD = {
    'Bike': CORES['vermelho'], 'Run': CORES['verde'],
    'Row':  CORES['azul'],    'Ski': CORES['roxo'],
    'WeightTraining': CORES['laranja'],
}
_CICLICOS = ['Bike', 'Run', 'Row', 'Ski']
_CORES_RPE = {
    'Leve':     CORES['verde'],
    'Moderado': CORES['laranja'],
    'Pesado':   CORES['vermelho'],
}
_CORES_SYS = {
    'oxidative':  '#2ecc71',
    'glycolytic': '#e67e22',
    'sprint':     '#e74c3c',
}

_LAY_BASE = dict(
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(color='#111', size=11),
    margin=dict(t=55, b=85, l=55, r=20),
    legend=dict(orientation='h', y=-0.30, font=dict(color='#111', size=10),
                bgcolor='rgba(255,255,255,0.9)'),
    xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(color='#111')),
    yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'),
               autorange=True),   # ← autorange so Plotly rescales when traces toggled
)


def _pct_text(v_abs, v_total):
    """Return '12%' if segment >= 5% of total, else '' to avoid clutter."""
    if v_total == 0 or v_abs / v_total < 0.05:
        return ''
    return f"{v_abs / v_total * 100:.0f}%"


def _stacked_pct_fig(pivot_abs, name_order, color_map, title, y_title, key):
    """
    Build a stacked bar figure with % text inside each segment.
    autorange=True means Plotly rescales Y when the user hides/shows traces via legend.
    """
    fig = go.Figure()
    totals = pivot_abs.sum(axis=1)
    for cat in name_order:
        if cat not in pivot_abs.columns:
            continue
        y = pivot_abs[cat]
        fig.add_trace(go.Bar(
            x=[str(x) for x in pivot_abs.index],
            y=y.tolist(),
            name=cat,
            marker_color=color_map.get(cat, '#888'),
            marker_line_width=0,
            opacity=0.88,
            text=[_pct_text(v, t) for v, t in zip(y, totals)],
            textposition='inside',
            textfont=dict(color='white', size=9),
            hovertemplate=f'<b>{cat}</b><br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>',
        ))
    fig.update_layout(
        **_LAY_BASE,
        barmode='stack',
        height=360,
        yaxis_title=y_title,
        title=dict(text=title, font=dict(size=13, color='#111')),
    )
    return fig


def _delta_icon(cur, prev, thr=3.0):
    """Return (arrow, colour) comparing cur vs prev with threshold %."""
    if prev is None or prev == 0:
        return '—', '#888888'
    pct = (cur - prev) / abs(prev) * 100
    if pct > thr:
        return f'↗ +{pct:.0f}%', '#27ae60'
    if pct < -thr:
        return f'↘ {pct:.0f}%', '#e74c3c'
    return f'→ {pct:+.0f}%', '#7f8c8d'


def tab_volume(da, dw):
    st.header("📦 Volume & Carga")
    if len(da) == 0:
        st.warning("Sem dados de atividades.")
        return

    df = filtrar_principais(da).copy()
    df = add_tempo(df)
    df['Data']  = pd.to_datetime(df['Data'])
    df['horas'] = pd.to_numeric(df['moving_time'], errors='coerce').div(3600).fillna(0)
    df['km']    = pd.to_numeric(df.get('distance', pd.Series(dtype=float)),
                                 errors='coerce').div(1000).fillna(0)
    df['type']  = df['type'].apply(norm_tipo)

    # ── Período selector ──────────────────────────────────────────────────────
    _periodo_opts = {'Semanal': 'semana', 'Mensal': 'mes', 'Trimestral': 'trimestre'}

    # Ensure 'semana' column exists
    if 'semana' not in df.columns:
        df['semana'] = pd.to_datetime(df['Data']).dt.to_period('W').astype(str)

    periodo_lbl = st.radio("Período", list(_periodo_opts.keys()),
                            horizontal=True, key="vol_periodo", index=1)
    periodo_col = _periodo_opts[periodo_lbl]

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 1. HORAS — Cíclicos com % por modalidade
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🕐 Volume — Horas por Modalidade (cíclicos)")
    df_cic = df[df['type'].isin(_CICLICOS)].copy()
    if len(df_cic) > 0:
        pivot_h = (df_cic.pivot_table(index=periodo_col, columns='type',
                                       values='horas', aggfunc='sum', fill_value=0)
                         .sort_index())
        fig_h = _stacked_pct_fig(pivot_h, _CICLICOS, _CORES_MOD,
                                  f'Horas — {periodo_lbl}', 'Horas', 'vol_h_cic')
        st.plotly_chart(fig_h, use_container_width=True, config=MC, key="vol_h_cic")
        _c1, _c2 = st.columns(2)
        _c1.metric("Total horas cíclicos", f"{pivot_h.values.sum():.1f}h")
        _c2.metric(f"Média {periodo_lbl.lower()}",
                   f"{pivot_h.values.sum() / max(len(pivot_h), 1):.1f}h")
    else:
        st.info("Sem actividades cíclicas no período.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 2. KM — Cíclicos com % por modalidade
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📏 Volume — KM por Modalidade (cíclicos)")
    df_km = df[df['type'].isin(_CICLICOS) & (df['km'] > 0)].copy()
    if len(df_km) > 0:
        pivot_km = (df_km.pivot_table(index=periodo_col, columns='type',
                                       values='km', aggfunc='sum', fill_value=0)
                        .sort_index())
        fig_km = _stacked_pct_fig(pivot_km, _CICLICOS, _CORES_MOD,
                                   f'Distância (km) — {periodo_lbl}', 'km', 'vol_km_cic')
        st.plotly_chart(fig_km, use_container_width=True, config=MC, key="vol_km_cic")
    else:
        st.info("Sem dados de distância para o período.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 3. HORAS por INTENSIDADE (RPE) — com % dentro de cada barra
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("💪 Horas por Intensidade (RPE) — cíclicos")
    df_rpe = df[df['type'].isin(_CICLICOS)].copy()
    if 'rpe' in df_rpe.columns and df_rpe['rpe'].notna().any():
        df_rpe['rpe_cat'] = df_rpe['rpe'].apply(classificar_rpe)
        df_rpe = df_rpe.dropna(subset=['rpe_cat'])
        if len(df_rpe) > 0:
            pivot_rpe = (df_rpe.pivot_table(index=periodo_col, columns='rpe_cat',
                                             values='horas', aggfunc='sum', fill_value=0)
                               .sort_index())
            fig_rpe = _stacked_pct_fig(pivot_rpe, ['Leve', 'Moderado', 'Pesado'],
                                        _CORES_RPE, f'Horas por Intensidade — {periodo_lbl}',
                                        'Horas', 'vol_rpe')
            st.plotly_chart(fig_rpe, use_container_width=True, config=MC, key="vol_rpe")
        else:
            st.info("Sem dados de RPE válidos.")
    else:
        st.info("Coluna RPE não disponível.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 4. WeightTraining — horas
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏋️ WeightTraining — Horas")
    df_wt = da[da['type'].apply(norm_tipo) == 'WeightTraining'].copy()
    if len(df_wt) > 0:
        df_wt = add_tempo(df_wt)
        if 'semana' not in df_wt.columns:
            df_wt['semana'] = pd.to_datetime(df_wt['Data']).dt.to_period('W').astype(str)
        df_wt['horas'] = pd.to_numeric(df_wt['moving_time'], errors='coerce').div(3600).fillna(0)
        pivot_wt = df_wt.groupby(periodo_col)['horas'].sum().sort_index()
        fig_wt = go.Figure(go.Bar(
            x=[str(x) for x in pivot_wt.index],
            y=pivot_wt.tolist(),
            marker_color=_CORES_MOD['WeightTraining'],
            marker_line_width=0, opacity=0.88,
            text=[f"{v:.1f}h" for v in pivot_wt],
            textposition='outside',
            hovertemplate='%{x}: <b>%{y:.1f}h</b><extra></extra>',
        ))
        fig_wt.update_layout(
            **_LAY_BASE, height=320, yaxis_title='Horas',
            title=dict(text=f'WeightTraining — {periodo_lbl}',
                       font=dict(size=13, color='#111')))
        st.plotly_chart(fig_wt, use_container_width=True, config=MC, key="vol_wt")
        _wc1, _wc2 = st.columns(2)
        _wc1.metric("Total horas WT", f"{pivot_wt.sum():.1f}h")
        _wc2.metric(f"Média {periodo_lbl.lower()}",
                    f"{pivot_wt.sum() / max(len(pivot_wt), 1):.1f}h")
    else:
        st.info("Sem sessões de WeightTraining no período.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 5. STRAIN SCORE — XSS + Decomposição por sistema energético
    #    oxidative (aerobic/cp_usage) | glycolytic | sprint (pmax)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("⚡ Strain Score — Sistema Energético por Modalidade")
    st.caption(
        "**oxidative** = aerobic (base aeróbica)  |  "
        "**glycolytic** = glicolítico  |  "
        "**sprint** = neuromuscular (p_max)  ·  "
        "% dentro de cada barra = proporção do sistema. "
        "Ocultar traços na legenda reajusta o eixo Y automaticamente."
    )

    _xss_col = next((c for c in ['xss', 'SS', 'XSS']
                     if c in df.columns and df[c].notna().any()), None)
    _aer_col = next((c for c in ['aerobic', 'cp_usage', 'AllWorkFTP']
                     if c in df.columns and df[c].notna().any()), None)
    _gly_col = next((c for c in ['glycolytic', 'glycolytic_usage']
                     if c in df.columns and df[c].notna().any()), None)
    _spr_col = next((c for c in ['pmax', 'p_max_usage', 'icu_pm_p_max']
                     if c in df.columns and df[c].notna().any()), None)

    df_str = df[df['type'].isin(_CICLICOS)].copy()

    # ── 5a. XSS total por modalidade ──────────────────────────────────────
    if _xss_col and len(df_str) > 0:
        df_str['_xss'] = pd.to_numeric(df_str[_xss_col], errors='coerce').fillna(0)
        pivot_xss = (df_str.pivot_table(index=periodo_col, columns='type',
                                         values='_xss', aggfunc='sum', fill_value=0)
                           .sort_index())
        fig_xss = _stacked_pct_fig(pivot_xss, _CICLICOS, _CORES_MOD,
                                    f'XSS Total por Modalidade — {periodo_lbl}',
                                    'XSS (Strain)', 'vol_xss')
        st.plotly_chart(fig_xss, use_container_width=True, config=MC, key="vol_xss")
    else:
        st.info("Coluna XSS/Strain Score não disponível.")

    # ── 5b. Decomposição por sistema energético — subplots por modalidade ─
    _sys_map = {}
    if _aer_col: _sys_map[_aer_col] = 'oxidative'
    if _gly_col: _sys_map[_gly_col] = 'glycolytic'
    if _spr_col: _sys_map[_spr_col] = 'sprint'

    _mods_avail = [m for m in _CICLICOS if m in df_str['type'].unique()]

    if _sys_map and _mods_avail:
        st.markdown("**Decomposição por Sistema Energético**")
        n_mods = len(_mods_avail)

        fig_sys = make_subplots(
            rows=1, cols=n_mods,
            subplot_titles=_mods_avail,
            shared_yaxes=True,
        )

        for col_i, mod in enumerate(_mods_avail, 1):
            df_mod = df_str[df_str['type'] == mod].copy()
            sys_series = {}

            for src_col, sys_name in _sys_map.items():
                if src_col in df_mod.columns:
                    df_mod[f'_{sys_name}'] = pd.to_numeric(
                        df_mod[src_col], errors='coerce').fillna(0)
                    sys_series[sys_name] = (df_mod.groupby(periodo_col)[f'_{sys_name}']
                                             .sum().sort_index())

            if not sys_series:
                continue

            all_idx = sorted(set().union(*[s.index for s in sys_series.values()]))

            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                if sys_name not in sys_series:
                    continue
                s = sys_series[sys_name].reindex(all_idx, fill_value=0)
                totals_mod = sum(
                    sys_series[sn].reindex(all_idx, fill_value=0)
                    for sn in sys_series)
                text_v = [_pct_text(v, t) for v, t in zip(s, totals_mod)]
                fig_sys.add_trace(go.Bar(
                    x=[str(x) for x in all_idx],
                    y=s.tolist(),
                    name=sys_name,
                    marker_color=_CORES_SYS[sys_name],
                    marker_line_width=0, opacity=0.88,
                    text=text_v,
                    textposition='inside',
                    textfont=dict(color='white', size=8),
                    legendgroup=sys_name,
                    showlegend=(col_i == 1),
                    hovertemplate=(
                        f'<b>{sys_name}</b><br>'
                        f'{mod} %{{x}}: <b>%{{y:.1f}}</b><extra></extra>'
                    ),
                ), row=1, col=col_i)

        fig_sys.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10),
            margin=dict(t=65, b=85, l=55, r=20),
            barmode='stack',
            height=400,
            legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=10)),
            title=dict(
                text=f'Sistema Energético por Modalidade — {periodo_lbl}',
                font=dict(size=13, color='#111')),
        )
        # autorange on all y-axes so toggling traces rescales Y
        fig_sys.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'), autorange=True)
        fig_sys.update_xaxes(tickangle=-45, tickfont=dict(color='#111'),
                              showgrid=False)
        st.plotly_chart(fig_sys, use_container_width=True, config=MC, key="vol_sys")

        # ── Tabela: % por sistema por modalidade + Δ vs período anterior ──
        st.markdown("**📋 % por Sistema Energético por Modalidade — vs período anterior**")

        # Build per-period aggregation for all mods
        # Need ordered periods to compute "previous"
        _all_periods = sorted(df_str[periodo_col].dropna().unique())

        # For each mod × system, build a dict: period → value
        _sys_data = {mod: {sn: {} for sn in ['oxidative', 'glycolytic', 'sprint']}
                     for mod in _mods_avail}

        for mod in _mods_avail:
            df_mod = df_str[df_str['type'] == mod].copy()
            for src_col, sys_name in _sys_map.items():
                if src_col not in df_mod.columns:
                    continue
                df_mod[f'_{sys_name}'] = pd.to_numeric(
                    df_mod[src_col], errors='coerce').fillna(0)
                agg = df_mod.groupby(periodo_col)[f'_{sys_name}'].sum()
                for p in _all_periods:
                    _sys_data[mod][sys_name][p] = float(agg.get(p, 0))

        # Use last 2 available periods for comparison
        if len(_all_periods) >= 2:
            _p_cur  = _all_periods[-1]
            _p_prev = _all_periods[-2]
        elif len(_all_periods) == 1:
            _p_cur  = _all_periods[0]
            _p_prev = None
        else:
            _p_cur = _p_prev = None

        tab_rows = []
        for mod in _mods_avail:
            grand_cur  = sum(_sys_data[mod][sn].get(_p_cur, 0)
                             for sn in ['oxidative', 'glycolytic', 'sprint'])
            grand_prev = sum(_sys_data[mod][sn].get(_p_prev, 0)
                             for sn in ['oxidative', 'glycolytic', 'sprint']) if _p_prev else None

            if grand_cur == 0:
                continue

            row = {'Modalidade': mod, 'Período': str(_p_cur)}
            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                v_cur  = _sys_data[mod][sys_name].get(_p_cur,  0)
                v_prev = _sys_data[mod][sys_name].get(_p_prev, 0) if _p_prev else None

                pct_cur  = v_cur / grand_cur * 100 if grand_cur > 0 else 0
                pct_prev = (v_prev / grand_prev * 100
                            if (v_prev is not None and grand_prev and grand_prev > 0)
                            else None)

                arrow, color = _delta_icon(pct_cur, pct_prev)

                label = sys_name.capitalize()
                row[label] = f"{pct_cur:.1f}%"
                row[f"Δ {label}"] = arrow  # coloured in styling

            tab_rows.append(row)

        if tab_rows:
            import pandas as _pd2
            _tdf = _pd2.DataFrame(tab_rows)

            # Colour the Δ columns using st.dataframe styler
            delta_cols = [c for c in _tdf.columns if c.startswith('Δ')]

            def _colour_delta(v):
                v = str(v)
                if '↗' in v: return 'color: #27ae60; font-weight: bold'
                if '↘' in v: return 'color: #e74c3c; font-weight: bold'
                if '→' in v: return 'color: #7f8c8d'
                return ''

            styled = _tdf.style.map(_colour_delta, subset=delta_cols)
            st.caption(f"Comparação: **{_p_cur}** vs **{_p_prev}**" if _p_prev
                       else f"Apenas um período disponível: **{_p_cur}**")
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("Dados de sistema energético insuficientes para tabela.")

    else:
        st.info("Colunas aerobic/glycolytic/pmax não disponíveis nos dados.")
