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
LAY = dict(paper_bgcolor='white', plot_bgcolor='white',
           font=dict(color='#111', size=11),
           margin=dict(t=50, b=80, l=55, r=20),
           legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=10)),
           xaxis=dict(tickangle=-45, showgrid=False, gridcolor='#eee', tickfont=dict(color='#111')),
           yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))

_CORES_MOD = {
    'Bike': CORES['vermelho'], 'Run': CORES['verde'],
    'Row':  CORES['azul'],    'Ski': CORES['roxo'],
    'WeightTraining': CORES['laranja'],
}
_CICLICOS = ['Bike', 'Run', 'Row', 'Ski']
_CORES_RPE = {'Leve': CORES['verde'], 'Moderado': CORES['laranja'], 'Pesado': CORES['vermelho']}

# Energia por sistema energético (Xert)
#   aerobic   = oxidative (cp_usage)  → Z1 / base aeróbica
#   glycolytic = glycolytic_usage      → Z2/Z3 anaeróbio láctico
#   pmax       = p_max_usage / sprint  → sprint / neuromuscular
_CORES_SYS = {
    'oxidative':   '#2ecc71',   # verde
    'glycolytic':  '#e67e22',   # laranja
    'sprint':      '#e74c3c',   # vermelho
}


def _pct_label(y_abs, y_total):
    """Retorna texto '12%' se >= 5%, senão '' (para não poluir barra pequena)."""
    if y_total == 0 or y_abs / y_total < 0.05:
        return ''
    return f"{y_abs / y_total * 100:.0f}%"


def _bar_pct_traces(pivot_abs, name_order, color_map, fig, row=None, col=None):
    """
    Adiciona traces de barras empilhadas com % dentro de cada segmento.
    pivot_abs: DataFrame (index=períodos, columns=categorias, valores=absolutos)
    """
    totals = pivot_abs.sum(axis=1)
    kwargs = dict(row=row, col=col) if row else {}
    for cat in name_order:
        if cat not in pivot_abs.columns:
            continue
        y_abs = pivot_abs[cat]
        text_vals = [_pct_label(abs_v, tot) for abs_v, tot in zip(y_abs, totals)]
        trace = go.Bar(
            x=[str(x) for x in pivot_abs.index],
            y=y_abs.tolist(),
            name=cat,
            marker_color=color_map.get(cat, '#888'),
            marker_line_width=0,
            opacity=0.88,
            text=text_vals,
            textposition='inside',
            textfont=dict(color='white', size=9),
            hovertemplate=f'<b>{cat}</b><br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>',
        )
        if row:
            fig.add_trace(trace, **kwargs)
        else:
            fig.add_trace(trace)


def tab_volume(da, dw):
    st.header("📦 Volume & Carga")
    if len(da) == 0:
        st.warning("Sem dados de atividades.")
        return

    df = filtrar_principais(da).copy()
    df = add_tempo(df)
    df['Data']  = pd.to_datetime(df['Data'])
    df['horas'] = pd.to_numeric(df['moving_time'], errors='coerce').div(3600).fillna(0)
    df['km']    = pd.to_numeric(df.get('distance', pd.Series(dtype=float)), errors='coerce').div(1000).fillna(0)
    df['type']  = df['type'].apply(norm_tipo)

    # ── período selector ─────────────────────────────────────────────────────
    periodo_opts = {'Mensal': 'mes', 'Trimestral': 'trimestre'}
    periodo_lbl  = st.radio("Período", list(periodo_opts.keys()),
                             horizontal=True, key="vol_periodo")
    periodo_col  = periodo_opts[periodo_lbl]

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 1. HORAS — Cíclicos (com % por modalidade)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🕐 Volume — Horas por Modalidade (cíclicos)")
    df_cic = df[df['type'].isin(_CICLICOS)].copy()
    if len(df_cic) > 0:
        pivot_h = (df_cic.pivot_table(index=periodo_col, columns='type',
                                       values='horas', aggfunc='sum', fill_value=0)
                         .sort_index())
        fig_h = go.Figure()
        _bar_pct_traces(pivot_h, _CICLICOS, _CORES_MOD, fig_h)
        total_h = pivot_h.values.sum()
        media_h = total_h / max(len(pivot_h), 1)
        fig_h.update_layout(**LAY, barmode='stack', height=360,
                             yaxis_title='Horas',
                             title=dict(text=f'Horas — {periodo_lbl}',
                                        font=dict(size=13, color='#111')))
        st.plotly_chart(fig_h, use_container_width=True, config=MC, key="vol_h_cic")
        _c1, _c2 = st.columns(2)
        _c1.metric("Total horas cíclicos", f"{total_h:.1f}h")
        _c2.metric(f"Média {periodo_lbl.lower()}", f"{media_h:.1f}h")
    else:
        st.info("Sem actividades cíclicas no período.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 2. KM — Cíclicos (com % por modalidade)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📏 Volume — KM por Modalidade (cíclicos)")
    df_km = df[df['type'].isin(_CICLICOS)].copy()
    df_km = df_km[df_km['km'] > 0]
    if len(df_km) > 0:
        pivot_km = (df_km.pivot_table(index=periodo_col, columns='type',
                                       values='km', aggfunc='sum', fill_value=0)
                        .sort_index())
        fig_km = go.Figure()
        _bar_pct_traces(pivot_km, _CICLICOS, _CORES_MOD, fig_km)
        fig_km.update_layout(**LAY, barmode='stack', height=360,
                              yaxis_title='KM',
                              title=dict(text=f'Distância (km) — {periodo_lbl}',
                                         font=dict(size=13, color='#111')))
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
            fig_rpe = go.Figure()
            _bar_pct_traces(pivot_rpe, ['Leve', 'Moderado', 'Pesado'], _CORES_RPE, fig_rpe)
            fig_rpe.update_layout(**LAY, barmode='stack', height=360,
                                   yaxis_title='Horas',
                                   title=dict(text=f'Horas por Intensidade — {periodo_lbl}',
                                              font=dict(size=13, color='#111')))
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
    # Use da directly (filtrar_principais removes WT when coexists with other sports)
    df_wt = da[da['type'].apply(norm_tipo) == 'WeightTraining'].copy()
    if len(df_wt) > 0:
        df_wt = add_tempo(df_wt)
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
        fig_wt.update_layout(**LAY, height=320, yaxis_title='Horas',
                              title=dict(text=f'WeightTraining — {periodo_lbl}',
                                         font=dict(size=13, color='#111')))
        st.plotly_chart(fig_wt, use_container_width=True, config=MC, key="vol_wt")
        _wt_total = pivot_wt.sum()
        _wt_med   = _wt_total / max(len(pivot_wt), 1)
        _wc1, _wc2 = st.columns(2)
        _wc1.metric("Total horas WT", f"{_wt_total:.1f}h")
        _wc2.metric(f"Média {periodo_lbl.lower()}", f"{_wt_med:.1f}h")
    else:
        st.info("Sem sessões de WeightTraining no período.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 5. STRAIN SCORE — XSS total + decomposição por sistema energético
    #    oxidative (aerobic/cp_usage) | glycolytic | sprint (pmax/p_max_usage)
    #    Separado por modalidade + tabela de % por sistema
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("⚡ Strain Score — Sistema Energético por Modalidade")
    st.caption("oxidative = aerobic (cp_usage) | glycolytic = glicólise láctica | sprint = p_max_usage. "
               "% dentro de cada barra = proporção do sistema no período.")

    # Find available columns
    _xss_col  = next((c for c in ['xss', 'SS', 'XSS'] if c in df.columns and df[c].notna().any()), None)
    _aer_col  = next((c for c in ['aerobic', 'cp_usage', 'AllWorkFTP'] if c in df.columns and df[c].notna().any()), None)
    _gly_col  = next((c for c in ['glycolytic', 'glycolytic_usage'] if c in df.columns and df[c].notna().any()), None)
    _spr_col  = next((c for c in ['pmax', 'p_max_usage', 'icu_pm_p_max'] if c in df.columns and df[c].notna().any()), None)

    df_str = df[df['type'].isin(_CICLICOS)].copy()

    # ── 5a. XSS total por modalidade ──────────────────────────────────────
    if _xss_col and len(df_str) > 0:
        df_str['_xss'] = pd.to_numeric(df_str[_xss_col], errors='coerce').fillna(0)
        pivot_xss = (df_str.pivot_table(index=periodo_col, columns='type',
                                         values='_xss', aggfunc='sum', fill_value=0)
                           .sort_index())
        fig_xss = go.Figure()
        _bar_pct_traces(pivot_xss, _CICLICOS, _CORES_MOD, fig_xss)
        fig_xss.update_layout(**LAY, barmode='stack', height=360,
                               yaxis_title='XSS (Strain)',
                               title=dict(text=f'XSS Total por Modalidade — {periodo_lbl}',
                                          font=dict(size=13, color='#111')))
        st.plotly_chart(fig_xss, use_container_width=True, config=MC, key="vol_xss")
    else:
        st.info("Coluna XSS/Strain Score não disponível.")

    # ── 5b. Decomposição por sistema energético ────────────────────────────
    _sys_available = [c for c in [_aer_col, _gly_col, _spr_col] if c is not None]

    if _sys_available:
        st.markdown("**Decomposição por Sistema Energético**")

        # Map internal col names → display names
        _sys_map = {
            _aer_col: 'oxidative',
            _gly_col: 'glycolytic',
            _spr_col: 'sprint',
        }
        _mods_avail = [m for m in _CICLICOS if m in df_str['type'].unique()]

        # Build one subplot per modality (columns) or show per-modality tabs
        n_mods = len(_mods_avail)
        if n_mods == 0:
            st.info("Sem dados de sistema energético.")
        else:
            # ── Per-modality stacked bars (oxidative/glycolytic/sprint) ──
            fig_sys = make_subplots(
                rows=1, cols=n_mods,
                subplot_titles=_mods_avail,
                shared_yaxes=True,
            )
            for col_i, mod in enumerate(_mods_avail, 1):
                df_mod = df_str[df_str['type'] == mod].copy()
                sys_data = {}
                for src_col, sys_name in _sys_map.items():
                    if src_col and src_col in df_mod.columns:
                        df_mod[f'_{sys_name}'] = pd.to_numeric(df_mod[src_col], errors='coerce').fillna(0)
                        agg = df_mod.groupby(periodo_col)[f'_{sys_name}'].sum().sort_index()
                        sys_data[sys_name] = agg

                if not sys_data:
                    continue

                # Align indexes
                all_idx = sorted(set().union(*[s.index for s in sys_data.values()]))
                for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                    if sys_name not in sys_data:
                        continue
                    s = sys_data[sys_name].reindex(all_idx, fill_value=0)
                    totals_mod = sum(sys_data[sn].reindex(all_idx, fill_value=0)
                                     for sn in sys_data)
                    text_vals = [_pct_label(v, t) for v, t in zip(s, totals_mod)]
                    fig_sys.add_trace(go.Bar(
                        x=[str(x) for x in all_idx],
                        y=s.tolist(),
                        name=sys_name,
                        marker_color=_CORES_SYS[sys_name],
                        marker_line_width=0,
                        opacity=0.88,
                        text=text_vals,
                        textposition='inside',
                        textfont=dict(color='white', size=8),
                        legendgroup=sys_name,
                        showlegend=(col_i == 1),
                        hovertemplate=f'<b>{sys_name}</b><br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>',
                    ), row=1, col=col_i)

            fig_sys.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=10),
                margin=dict(t=60, b=80, l=55, r=20),
                barmode='stack', height=380,
                legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=10)),
                title=dict(text=f'Sistema Energético por Modalidade — {periodo_lbl}',
                           font=dict(size=13, color='#111')),
            )
            fig_sys.update_xaxes(tickangle=-45, tickfont=dict(color='#111'),
                                  showgrid=False)
            fig_sys.update_yaxes(showgrid=True, gridcolor='#eee',
                                  tickfont=dict(color='#111'))
            st.plotly_chart(fig_sys, use_container_width=True, config=MC, key="vol_sys")

        # ── Tabela resumo % por sistema por modalidade ─────────────────────
        st.markdown("**📋 Tabela — % por Sistema Energético por Modalidade**")
        st.caption(f"Período: {periodo_lbl} | agregação: todo o histórico filtrado")

        tab_rows = []
        for mod in _mods_avail:
            df_mod = df_str[df_str['type'] == mod].copy()
            totals_sys = {}
            for src_col, sys_name in _sys_map.items():
                if src_col and src_col in df_mod.columns:
                    totals_sys[sys_name] = pd.to_numeric(df_mod[src_col], errors='coerce').sum()
                else:
                    totals_sys[sys_name] = 0.0
            grand_total = sum(totals_sys.values())
            if grand_total == 0:
                continue
            row = {'Modalidade': mod}
            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                v = totals_sys.get(sys_name, 0)
                row[sys_name.capitalize()] = f"{v / grand_total * 100:.1f}%"
                row[f"{sys_name.capitalize()} (abs)"] = f"{v:.0f}"
            tab_rows.append(row)

        if tab_rows:
            _tdf = pd.DataFrame(tab_rows)
            # Also add XSS totals column
            if _xss_col:
                _xss_by_mod = (df_str.groupby('type')[_xss_col]
                               .apply(lambda x: pd.to_numeric(x, errors='coerce').sum())
                               .reset_index()
                               .rename(columns={_xss_col: 'XSS Total'}))
                _tdf = _tdf.merge(_xss_by_mod.rename(columns={'type': 'Modalidade'}),
                                  on='Modalidade', how='left')
            st.dataframe(_tdf, use_container_width=True, hide_index=True)
        else:
            st.info("Dados de sistema energético insuficientes para tabela.")
    else:
        st.info("Colunas aerobic/glycolytic/pmax não disponíveis nos dados.")
