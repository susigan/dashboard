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

_PERIODO_OPTS = {'Semanal': 'semana', 'Mensal': 'mes', 'Trimestral': 'trimestre'}
_LAY = dict(
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(color='#111', size=11),
    margin=dict(t=55, b=85, l=55, r=20),
    legend=dict(orientation='h', y=-0.30, font=dict(color='#111', size=10),
                bgcolor='rgba(255,255,255,0.9)'),
    xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(color='#111')),
    yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'),
               autorange=True),
)


def _pct_text(v, total):
    return f"{v/total*100:.0f}%" if total > 0 and v/total >= 0.05 else ''


def _stacked_pct(pivot, order, cmap, title, ytitle, key):
    fig = go.Figure()
    totals = pivot.sum(axis=1)
    for cat in order:
        if cat not in pivot.columns:
            continue
        y = pivot[cat]
        fig.add_trace(go.Bar(
            x=[str(x) for x in pivot.index], y=y.tolist(),
            name=cat, marker_color=cmap.get(cat, '#888'),
            marker_line_width=0, opacity=0.88,
            text=[_pct_text(v, t) for v, t in zip(y, totals)],
            textposition='inside', textfont=dict(color='white', size=9),
            hovertemplate=f'<b>{cat}</b><br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>',
        ))
    fig.update_layout(**_LAY, barmode='stack', height=360,
                      yaxis_title=ytitle,
                      title=dict(text=title, font=dict(size=13, color='#111')))
    return fig


def _delta_icon(cur, prev, thr=3.0):
    if prev is None or prev == 0:
        return '—', '#888888'
    pct = (cur - prev) / abs(prev) * 100
    if pct > thr:   return f'↗ +{pct:.0f}%', '#27ae60'
    if pct < -thr:  return f'↘ {pct:.0f}%',  '#e74c3c'
    return f'→ {pct:+.0f}%', '#7f8c8d'


def _summary_table(pivot, unit, key_suffix):
    """
    Show % + total per category, with group-by dropdown (Mes/Ano).
    pivot: already-aggregated DataFrame (index=períodos, cols=categories).
    """
    gb_opts = {'Mês': 'M', 'Ano': 'Y'}
    gb_lbl  = st.selectbox("Agrupar tabela por", list(gb_opts.keys()),
                            key=f"vol_tb_{key_suffix}", index=0)

    # We need the raw df to re-aggregate — pass it through the function
    # Instead: re-aggregate the pivot by collapsing its index
    # pivot.index should be period strings; we rebuild by parsing
    # Simpler: just show the existing pivot with % column
    totals = pivot.sum(axis=1)
    rows = []
    for p_idx, row in pivot.iterrows():
        t = totals[p_idx]
        r = {'Período': str(p_idx)}
        for cat in pivot.columns:
            v = row[cat]
            r[cat] = f"{v:.1f} {unit}"
            r[f"% {cat}"] = f"{v/t*100:.1f}%" if t > 0 else '—'
        r['Total'] = f"{t:.1f} {unit}"
        rows.append(r)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _summary_table_raw(df_raw, periodo_col_tbl, value_col, groupby_col,
                        name_col, cmap_keys, unit, key_suffix):
    """
    Summary table with its own group-by dropdown (Mes/Ano).
    df_raw: source DataFrame with groupby_col, name_col, value_col already computed.
    """
    gb_opts = {'Mês': 'mes', 'Ano': 'ano'}
    gb_lbl  = st.selectbox("Agrupar tabela por", list(gb_opts.keys()),
                            key=f"vol_tb_{key_suffix}", index=0)
    gb_col  = gb_opts[gb_lbl]

    if gb_col not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw[gb_col] = pd.to_datetime(df_raw['Data']).dt.strftime(
            '%Y-%m' if gb_col == 'mes' else '%Y')

    pivot_tbl = (df_raw.pivot_table(
                    index=gb_col, columns=name_col,
                    values=value_col, aggfunc='sum', fill_value=0)
                 .sort_index())

    totals = pivot_tbl.sum(axis=1)
    rows = []
    for p_idx, row in pivot_tbl.iterrows():
        t = totals[p_idx]
        r = {'Período': str(p_idx)}
        for cat in cmap_keys:
            if cat not in pivot_tbl.columns: continue
            v = row[cat]
            r[cat] = f"{v:.1f} {unit}"
            r[f"% {cat}"] = f"{v/t*100:.1f}%" if t > 0 else '—'
        r['Total'] = f"{t:.1f} {unit}"
        rows.append(r)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Sem dados para a tabela.")


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
    if 'semana' not in df.columns:
        df['semana'] = pd.to_datetime(df['Data']).dt.to_period('W').astype(str)

    # ════════════════════════════════════════════════════════════════════════
    # 1. HORAS — Cíclicos
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🕐 Horas por Modalidade — cíclicos")
    df_cic = df[df['type'].isin(_CICLICOS)].copy()
    if len(df_cic) > 0:
        _p1 = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                            key="vol_p1", index=1)
        _pc1 = _PERIODO_OPTS[_p1]
        pivot_h = (df_cic.pivot_table(index=_pc1, columns='type',
                                       values='horas', aggfunc='sum', fill_value=0)
                         .sort_index())
        st.plotly_chart(
            _stacked_pct(pivot_h, _CICLICOS, _CORES_MOD,
                         f'Horas — {_p1}', 'Horas', 'vol_h_cic'),
            use_container_width=True, config=MC, key="vol_h_cic")

        with st.expander("📋 Tabela: % e total horas por modalidade", expanded=False):
            _summary_table_raw(df_cic, _pc1, 'horas', _pc1, 'type',
                                _CICLICOS, 'h', 'h_cic')
    else:
        st.info("Sem actividades cíclicas.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 2. KM — Cíclicos
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📏 KM por Modalidade — cíclicos")
    df_km = df[df['type'].isin(_CICLICOS) & (df['km'] > 0)].copy()
    if len(df_km) > 0:
        _p2 = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                            key="vol_p2", index=1)
        _pc2 = _PERIODO_OPTS[_p2]
        pivot_km = (df_km.pivot_table(index=_pc2, columns='type',
                                       values='km', aggfunc='sum', fill_value=0)
                        .sort_index())
        st.plotly_chart(
            _stacked_pct(pivot_km, _CICLICOS, _CORES_MOD,
                         f'Distância (km) — {_p2}', 'km', 'vol_km_cic'),
            use_container_width=True, config=MC, key="vol_km_cic")

        with st.expander("📋 Tabela: % e total km por modalidade", expanded=False):
            _summary_table_raw(df_km, _pc2, 'km', _pc2, 'type',
                                _CICLICOS, 'km', 'km_cic')
    else:
        st.info("Sem dados de distância.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 3. HORAS por INTENSIDADE (RPE)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("💪 Horas por Intensidade (RPE)")
    df_rpe = df[df['type'].isin(_CICLICOS)].copy()
    if 'rpe' in df_rpe.columns and df_rpe['rpe'].notna().any():
        df_rpe['rpe_cat'] = df_rpe['rpe'].apply(classificar_rpe)
        df_rpe = df_rpe.dropna(subset=['rpe_cat'])
        if len(df_rpe) > 0:
            _p3 = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                                key="vol_p3", index=2)
            _pc3 = _PERIODO_OPTS[_p3]
            pivot_rpe = (df_rpe.pivot_table(index=_pc3, columns='rpe_cat',
                                             values='horas', aggfunc='sum', fill_value=0)
                               .sort_index())
            st.plotly_chart(
                _stacked_pct(pivot_rpe, ['Leve', 'Moderado', 'Pesado'],
                             _CORES_RPE, f'Horas por Intensidade — {_p3}',
                             'Horas', 'vol_rpe'),
                use_container_width=True, config=MC, key="vol_rpe")

            with st.expander("📋 Tabela: % por RPE por modalidade", expanded=False):
                # Per-modality RPE breakdown
                gb_opts_rpe = {'Mês': 'mes', 'Ano': 'ano'}
                gb_rpe = st.selectbox("Agrupar por", list(gb_opts_rpe.keys()),
                                       key="vol_tb_rpe", index=0)
                gb_col_rpe = gb_opts_rpe[gb_rpe]
                # Cross-tab: modal + rpe_cat
                _rpe_rows = []
                for mod in _CICLICOS:
                    _df_m = df_rpe[df_rpe['type'] == mod]
                    if len(_df_m) == 0: continue
                    _piv_m = (_df_m.pivot_table(
                                  index=gb_col_rpe, columns='rpe_cat',
                                  values='horas', aggfunc='sum', fill_value=0)
                              .sort_index())
                    _tot_m = _piv_m.sum(axis=1)
                    for p_idx, row in _piv_m.iterrows():
                        t = _tot_m[p_idx]
                        r = {'Período': str(p_idx), 'Modalidade': mod}
                        for cat in ['Leve', 'Moderado', 'Pesado']:
                            v = row.get(cat, 0)
                            r[f"% {cat}"] = f"{v/t*100:.1f}%" if t > 0 else '—'
                        r['Total h'] = f"{t:.1f}h"
                        _rpe_rows.append(r)
                if _rpe_rows:
                    st.dataframe(pd.DataFrame(_rpe_rows),
                                 use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados de RPE válidos.")
    else:
        st.info("Coluna RPE não disponível.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 4. WeightTraining
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏋️ WeightTraining — Horas")
    df_wt = da[da['type'].apply(norm_tipo) == 'WeightTraining'].copy()
    if len(df_wt) > 0:
        df_wt = add_tempo(df_wt)
        if 'semana' not in df_wt.columns:
            df_wt['semana'] = pd.to_datetime(df_wt['Data']).dt.to_period('W').astype(str)
        df_wt['horas'] = pd.to_numeric(df_wt['moving_time'], errors='coerce').div(3600).fillna(0)
        _p4 = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                            key="vol_p4", index=1)
        _pc4 = _PERIODO_OPTS[_p4]
        pivot_wt = df_wt.groupby(_pc4)['horas'].sum().sort_index()
        fig_wt = go.Figure(go.Bar(
            x=[str(x) for x in pivot_wt.index], y=pivot_wt.tolist(),
            marker_color=_CORES_MOD['WeightTraining'],
            marker_line_width=0, opacity=0.88,
            text=[f"{v:.1f}h" for v in pivot_wt], textposition='outside',
            hovertemplate='%{x}: <b>%{y:.1f}h</b><extra></extra>',
        ))
        fig_wt.update_layout(**_LAY, height=320, yaxis_title='Horas',
                              title=dict(text=f'WeightTraining — {_p4}',
                                         font=dict(size=13, color='#111')))
        st.plotly_chart(fig_wt, use_container_width=True, config=MC, key="vol_wt")
        _wc1, _wc2 = st.columns(2)
        _wc1.metric("Total horas WT", f"{pivot_wt.sum():.1f}h")
        _wc2.metric(f"Média {_p4.lower()}",
                    f"{pivot_wt.sum()/max(len(pivot_wt),1):.1f}h")
    else:
        st.info("Sem sessões de WeightTraining no período.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 5. STRAIN SCORE — XSS + Sistema Energético
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("⚡ Strain Score — Sistema Energético")
    st.caption(
        "**oxidative** = aerobic (base aeróbica)  |  "
        "**glycolytic** = glicolítico  |  "
        "**sprint** = neuromuscular.  "
        "Ocultar traços na legenda reajusta o eixo Y.")

    _xss_col = next((c for c in ['xss','SS','XSS'] if c in df.columns and df[c].notna().any()), None)
    _aer_col = next((c for c in ['aerobic','cp_usage','AllWorkFTP'] if c in df.columns and df[c].notna().any()), None)
    _gly_col = next((c for c in ['glycolytic','glycolytic_usage'] if c in df.columns and df[c].notna().any()), None)
    _spr_col = next((c for c in ['pmax','p_max_usage','icu_pm_p_max'] if c in df.columns and df[c].notna().any()), None)

    df_str = df[df['type'].isin(_CICLICOS)].copy()
    _p5 = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                        key="vol_p5", index=1)
    _pc5 = _PERIODO_OPTS[_p5]

    # ── 5a. XSS total ──────────────────────────────────────────────────────
    if _xss_col and len(df_str) > 0:
        df_str['_xss'] = pd.to_numeric(df_str[_xss_col], errors='coerce').fillna(0)
        pivot_xss = (df_str.pivot_table(index=_pc5, columns='type',
                                         values='_xss', aggfunc='sum', fill_value=0)
                           .sort_index())
        st.plotly_chart(
            _stacked_pct(pivot_xss, _CICLICOS, _CORES_MOD,
                         f'XSS Total — {_p5}', 'XSS', 'vol_xss'),
            use_container_width=True, config=MC, key="vol_xss")
    else:
        st.info("Coluna XSS não disponível.")

    # ── 5b. Decomposição por sistema ───────────────────────────────────────
    _sys_map = {}
    if _aer_col: _sys_map[_aer_col] = 'oxidative'
    if _gly_col: _sys_map[_gly_col] = 'glycolytic'
    if _spr_col: _sys_map[_spr_col] = 'sprint'

    _mods_avail = [m for m in _CICLICOS if m in df_str['type'].unique()]

    if _sys_map and _mods_avail:
        st.markdown("**Decomposição por Sistema Energético**")
        n_mods = len(_mods_avail)
        fig_sys = make_subplots(rows=1, cols=n_mods,
                                subplot_titles=_mods_avail,
                                shared_yaxes=True)

        for col_i, mod in enumerate(_mods_avail, 1):
            df_mod = df_str[df_str['type'] == mod].copy()
            sys_series = {}
            for src_col, sys_name in _sys_map.items():
                if src_col in df_mod.columns:
                    df_mod[f'_{sys_name}'] = pd.to_numeric(df_mod[src_col], errors='coerce').fillna(0)
                    sys_series[sys_name] = df_mod.groupby(_pc5)[f'_{sys_name}'].sum().sort_index()
            if not sys_series: continue
            all_idx = sorted(set().union(*[s.index for s in sys_series.values()]))
            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                if sys_name not in sys_series: continue
                s = sys_series[sys_name].reindex(all_idx, fill_value=0)
                totals_mod = sum(sys_series[sn].reindex(all_idx, fill_value=0) for sn in sys_series)
                fig_sys.add_trace(go.Bar(
                    x=[str(x) for x in all_idx], y=s.tolist(),
                    name=sys_name, marker_color=_CORES_SYS[sys_name],
                    marker_line_width=0, opacity=0.88,
                    text=[_pct_text(v, t) for v, t in zip(s, totals_mod)],
                    textposition='inside', textfont=dict(color='white', size=8),
                    legendgroup=sys_name, showlegend=(col_i == 1),
                    hovertemplate=f'<b>{sys_name}</b><br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>',
                ), row=1, col=col_i)

        fig_sys.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10),
            margin=dict(t=65, b=85, l=55, r=20),
            barmode='stack', height=400,
            legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=10)),
            title=dict(text=f'Sistema Energético — {_p5}',
                       font=dict(size=13, color='#111')),
        )
        fig_sys.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'), autorange=True)
        fig_sys.update_xaxes(tickangle=-45, tickfont=dict(color='#111'), showgrid=False)
        st.plotly_chart(fig_sys, use_container_width=True, config=MC, key="vol_sys")

        # ── Tabela: % por sistema + Δ vs período anterior ──────────────────
        st.markdown("**📋 % Sistema Energético — vs período anterior**")
        _all_periods = sorted(df_str[_pc5].dropna().unique())
        _p_cur  = _all_periods[-1] if _all_periods else None
        _p_prev = _all_periods[-2] if len(_all_periods) >= 2 else None

        _sys_data = {mod: {sn: {} for sn in ['oxidative', 'glycolytic', 'sprint']}
                     for mod in _mods_avail}
        for mod in _mods_avail:
            df_mod = df_str[df_str['type'] == mod].copy()
            for src_col, sys_name in _sys_map.items():
                if src_col not in df_mod.columns: continue
                df_mod[f'_{sys_name}'] = pd.to_numeric(df_mod[src_col], errors='coerce').fillna(0)
                agg = df_mod.groupby(_pc5)[f'_{sys_name}'].sum()
                for p in _all_periods:
                    _sys_data[mod][sys_name][p] = float(agg.get(p, 0))

        tab_rows = []
        for mod in _mods_avail:
            gc = sum(_sys_data[mod][sn].get(_p_cur, 0)  for sn in ['oxidative','glycolytic','sprint'])
            gp = sum(_sys_data[mod][sn].get(_p_prev, 0) for sn in ['oxidative','glycolytic','sprint']) if _p_prev else None
            if gc == 0: continue
            row = {'Modalidade': mod, 'Período': str(_p_cur)}
            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                vc = _sys_data[mod][sys_name].get(_p_cur, 0)
                vp = _sys_data[mod][sys_name].get(_p_prev, 0) if _p_prev else None
                pct_c = vc / gc * 100 if gc > 0 else 0
                pct_p = (vp / gp * 100 if (vp is not None and gp and gp > 0) else None)
                arrow, _ = _delta_icon(pct_c, pct_p)
                row[sys_name.capitalize()] = f"{pct_c:.1f}%"
                row[f"Δ {sys_name.capitalize()}"] = arrow
            tab_rows.append(row)

        if tab_rows:
            _tdf = pd.DataFrame(tab_rows)
            dcols = [c for c in _tdf.columns if c.startswith('Δ')]
            def _cc(v):
                v = str(v)
                if '↗' in v: return 'color:#27ae60;font-weight:bold'
                if '↘' in v: return 'color:#e74c3c;font-weight:bold'
                if '→' in v: return 'color:#7f8c8d'
                return ''
            st.caption(f"Comparação: **{_p_cur}** vs **{_p_prev}**" if _p_prev
                       else f"Período único: **{_p_cur}**")
            st.dataframe(_tdf.style.map(_cc, subset=dcols),
                         use_container_width=True, hide_index=True)
    else:
        st.info("Colunas aerobic/glycolytic/pmax não disponíveis.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 6. ÍNDICE DE MONOTONIA DE FRY — Análise completa
    # Fry RW, Morton AR & Keast D (1992). Can J Sport Sci.
    # Seiler S & Tønnessen E (2009). Sportscience 13, 32-53.
    # IM = carga_média_7d / std_carga_7d  |  Training Strain = IM × carga_média
    # IM <1.5 ✅ | 1.5-2.0 🟡 | >2.0 🔴
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔄 Índice de Monotonia de Fry")
    st.caption(
        "**IM = carga_média_diária / std_carga_diária** (janela 7 dias). "
        "Dias sem treino contam como **0 kJ** — o descanso activo reduz a monotonia. "
        "**Training Strain = IM × carga_média** combina quantidade e uniformidade. "
        "Referências: Fry et al. (1992); Seiler & Tønnessen (2009)."
    )

    # ── Detectar coluna kJ ─────────────────────────────────────────────────
    _kj_fry_col = next(
        (c for c in ['icu_joules', 'joules', 'kj_total']
         if c in df.columns and df[c].notna().any()), None
    )

    if _kj_fry_col is None:
        st.info("Coluna kJ (icu_joules / kj_total) não disponível.")
    else:
        # ── Preparar série diária — GLOBAL e POR MODALIDADE ───────────────
        df_fry = df.copy()
        df_fry['Data'] = pd.to_datetime(df_fry['Data']).dt.normalize()
        df_fry['_kj'] = pd.to_numeric(df_fry[_kj_fry_col], errors='coerce').fillna(0)
        if _kj_fry_col == 'icu_joules':
            df_fry['_kj'] = df_fry['_kj'] / 1000  # J → kJ

        # Calendário completo (dias sem treino = 0)
        _fry_date_min = df_fry['Data'].min()
        _fry_date_max = df_fry['Data'].max()
        _fry_cal = pd.date_range(_fry_date_min, _fry_date_max, freq='D')

        # GLOBAL — todos os cíclicos juntos
        _gl_daily = (df_fry[df_fry['type'].isin(_CICLICOS)]
                     .groupby('Data')['_kj'].sum()
                     .reindex(_fry_cal, fill_value=0))

        # POR MODALIDADE
        _mod_daily = {}
        for _mod in _CICLICOS:
            _d = (df_fry[df_fry['type'] == _mod]
                  .groupby('Data')['_kj'].sum()
                  .reindex(_fry_cal, fill_value=0))
            if _d.sum() > 0:
                _mod_daily[_mod] = _d

        def _calc_fry(series, window=7, min_p=4):
            """Calcula IM e Training Strain para uma série diária."""
            m = series.rolling(window, min_periods=min_p).mean()
            s = series.rolling(window, min_periods=min_p).std()
            im = (m / s.replace(0, np.nan)).round(3)
            strain = (im * m).round(1)
            return m, s, im, strain

        _gl_m, _gl_s, _gl_im, _gl_strain = _calc_fry(_gl_daily)

        # ── Cards actuais — GLOBAL ─────────────────────────────────────────
        _im_now     = float(_gl_im.iloc[-1])    if _gl_im.notna().any()     else None
        _m7_now     = float(_gl_m.iloc[-1])     if _gl_m.notna().any()      else None
        _strain_now = float(_gl_strain.iloc[-1]) if _gl_strain.notna().any() else None
        _im_prev7   = float(_gl_im.iloc[-8])    if len(_gl_im) > 8 and pd.notna(_gl_im.iloc[-8]) else None

        if _im_now is None:
            _im_cor = "#888"; _im_lbl = "Sem dados"
        elif _im_now > 2.0:
            _im_cor = "#e74c3c"; _im_lbl = "🔴 Supressão imune provável"
        elif _im_now > 1.5:
            _im_cor = "#f39c12"; _im_lbl = "🟡 Monitorizar"
        else:
            _im_cor = "#27ae60"; _im_lbl = "✅ Polarizado"

        _fc1, _fc2, _fc3, _fc4 = st.columns(4)
        _fc1.metric("IM Global (7d)",
                    f"{_im_now:.2f}" if _im_now else "—",
                    delta=_im_lbl, delta_color="off",
                    help="IM = média kJ 7d / std kJ 7d (dias sem treino = 0)")
        _fc2.metric("Carga média 7d (kJ/dia)",
                    f"{_m7_now:.0f}" if _m7_now else "—")
        _fc3.metric("Training Strain",
                    f"{_strain_now:.0f}" if _strain_now else "—",
                    help="IM × carga_média. Alto+monotonia alta = maior risco")
        _fc4.metric("Δ IM vs semana ant.",
                    f"{_im_now - _im_prev7:+.2f}" if (_im_now and _im_prev7) else "—",
                    delta_color="inverse" if (_im_now and _im_now > 1.5) else "off")

        # Alerta visual
        if _im_now and _im_now > 2.0:
            _hr2, _hg2, _hb2 = int(_im_cor[1:3],16), int(_im_cor[3:5],16), int(_im_cor[5:7],16)
            st.markdown(
                f'<div style="padding:12px 16px; border-radius:8px; margin:8px 0; '
                f'background:rgba({_hr2},{_hg2},{_hb2},0.10); '
                f'border-left:5px solid {_im_cor};">'
                f'<b style="color:{_im_cor};">⚠️ IM={_im_now:.2f} — Monotonia elevada</b><br>'
                f'Treino demasiado uniforme. Prescrição: introduzir 1-2 dias de carga muito baixa '
                f'(Z1 exclusivo, &lt;50% kJ médio) entre as próximas sessões intensas.'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Gráfico série temporal IM — global + por modalidade ────────────
        st.markdown("#### Evolução IM e Training Strain — últimos 90 dias")

        _fry_plot_gl = pd.DataFrame({
            'Data': _fry_cal,
            'IM_global': _gl_im.values,
            'Strain_global': _gl_strain.values,
            'kJ_global': _gl_daily.values,
        }).tail(90)

        fig_fry = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4], vertical_spacing=0.06,
            subplot_titles=['Índice de Monotonia (IM) — Global e por Modalidade',
                            'Carga diária (kJ) — Global']
        )

        # Zonas IM
        for _y0, _y1, _fc, _flbl in [
            (0, 1.5,  'rgba(39,174,96,0.08)',  ''),
            (1.5, 2.0,'rgba(243,156,18,0.08)', ''),
            (2.0, 6.0,'rgba(231,76,60,0.08)',  ''),
        ]:
            fig_fry.add_hrect(y0=_y0, y1=_y1, fillcolor=_fc,
                              line_width=0, row=1, col=1)

        # Linhas threshold
        for _yv, _ydash, _yc in [(1.5,'dot','#f39c12'), (2.0,'dash','#e74c3c')]:
            fig_fry.add_hline(y=_yv, line_dash=_ydash, line_color=_yc,
                              line_width=1.5, row=1, col=1,
                              annotation_text=f"IM={_yv}",
                              annotation_position="right",
                              annotation_font_color=_yc, annotation_font_size=10)

        # IM global
        fig_fry.add_trace(go.Scatter(
            x=_fry_plot_gl['Data'], y=_fry_plot_gl['IM_global'],
            mode='lines', name='IM Global',
            line=dict(color='#2c3e50', width=3),
            hovertemplate='%{x|%d/%m/%Y}<br>IM Global: <b>%{y:.2f}</b><extra></extra>'
        ), row=1, col=1)

        # IM por modalidade
        _fry_cores_mod = {'Bike':'#E74C3C','Row':'#3498DB','Ski':'#9B59B6','Run':'#2ECC71'}
        for _mod, _d_mod in _mod_daily.items():
            _, _, _im_mod, _ = _calc_fry(_d_mod)
            _plot_mod = pd.DataFrame({'Data': _fry_cal, 'IM': _im_mod.values}).tail(90)
            fig_fry.add_trace(go.Scatter(
                x=_plot_mod['Data'], y=_plot_mod['IM'],
                mode='lines', name=f'IM {_mod}',
                line=dict(color=_fry_cores_mod.get(_mod,'#888'), width=1.5, dash='dot'),
                hovertemplate=f'%{{x|%d/%m/%Y}}<br>IM {_mod}: <b>%{{y:.2f}}</b><extra></extra>'
            ), row=1, col=1)

        # kJ diário global (painel inferior)
        fig_fry.add_trace(go.Bar(
            x=_fry_plot_gl['Data'], y=_fry_plot_gl['kJ_global'],
            name='kJ global',
            marker_color='rgba(44,62,80,0.5)',
            hovertemplate='%{x|%d/%m/%Y}<br>kJ: <b>%{y:.0f}</b><extra></extra>'
        ), row=2, col=1)

        fig_fry.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=11),
            height=500, hovermode='x unified',
            margin=dict(t=60, b=60, l=60, r=60),
            legend=dict(orientation='h', y=-0.12,
                        font=dict(color='#111', size=10),
                        bgcolor='rgba(255,255,255,0.9)'),
        )
        fig_fry.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'), row=1, col=1,
                              title_text='IM', range=[0, min(5, float(_gl_im.max()*1.2 if _gl_im.notna().any() else 4))])
        fig_fry.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'), row=2, col=1,
                              title_text='kJ/dia')
        fig_fry.update_xaxes(showgrid=False, tickfont=dict(color='#111'))

        st.plotly_chart(fig_fry, use_container_width=True, config=MC, key="vol_fry")

        # ── Tabela resumo por modalidade ────────────────────────────────────
        st.markdown("#### Resumo actual por modalidade (últimos 7 dias)")
        _fry_rows = []
        # Global
        _fry_rows.append({
            'Modalidade': '🌐 Global',
            'kJ/dia médio': f"{_m7_now:.0f}" if _m7_now else "—",
            'std kJ/dia':   f"{float(_gl_s.iloc[-1]):.0f}" if _gl_s.notna().any() else "—",
            'IM (7d)':      f"{_im_now:.2f}" if _im_now else "—",
            'Training Strain': f"{_strain_now:.0f}" if _strain_now else "—",
            'Status':       _im_lbl,
        })
        # Por modalidade
        for _mod, _d_mod in _mod_daily.items():
            _mm, _ms, _mim, _mst = _calc_fry(_d_mod)
            _im_m  = float(_mim.iloc[-1])  if _mim.notna().any()  else None
            _m7_m  = float(_mm.iloc[-1])   if _mm.notna().any()   else None
            _s7_m  = float(_ms.iloc[-1])   if _ms.notna().any()   else None
            _st_m  = float(_mst.iloc[-1])  if _mst.notna().any()  else None
            if _im_m is None: _st_mod = "—"
            elif _im_m > 2.0: _st_mod = "🔴 Alta"
            elif _im_m > 1.5: _st_mod = "🟡 Média"
            else:             _st_mod = "✅ Baixa"
            _fry_rows.append({
                'Modalidade':     _mod,
                'kJ/dia médio':   f"{_m7_m:.0f}" if _m7_m else "—",
                'std kJ/dia':     f"{_s7_m:.0f}"  if _s7_m else "—",
                'IM (7d)':        f"{_im_m:.2f}"  if _im_m else "—",
                'Training Strain':f"{_st_m:.0f}"  if _st_m else "—",
                'Status':         _st_mod,
            })

        st.dataframe(pd.DataFrame(_fry_rows),
                     use_container_width=True, hide_index=True)

        # ── Nota metodológica ───────────────────────────────────────────────
        with st.expander("ℹ️ Metodologia — Índice de Fry"):
            st.markdown("""
**Índice de Monotonia (IM)**

`IM = carga_média_diária_7d / desvio_padrão_carga_diária_7d`

Dias sem treino contam como **0 kJ** — não são ignorados. Um dia de repouso
aumenta o desvio padrão e reduz o IM. Isto é o comportamento correcto: o descanso
polariza o treino.

**Training Strain**

`Training Strain = IM × carga_média_7d`

Combina quantidade e uniformidade. Um atleta pode ter Strain alto com IM baixo
(muito volume mas bem polarizado) — menos risco que IM alto com Strain alto
(volume moderado, todos os dias iguais).

**Thresholds (Fry et al., 1992; Seiler & Tønnessen, 2009)**

| IM | Interpretação |
|---|---|
| < 1.5 | ✅ Treino polarizado — padrão de atletas de elite |
| 1.5 – 2.0 | 🟡 Zona de atenção — monitorizar resposta |
| > 2.0 | 🔴 Supressão imune provável — risco de plateau |

**Prescrição quando IM > 2.0:**
Introduzir 1-2 dias de carga muito reduzida (Z1 exclusivo, <50% da carga média)
entre sessões intensas. O objectivo não é reduzir o volume total, mas aumentar
o desvio padrão da distribuição diária.

**Referências**

Fry RW, Morton AR & Keast D (1992). Periodisation and the Prevention of Overtraining.
*Can J Sport Sci.* | Seiler S & Tønnessen E (2009). Intervals, Thresholds, and Long Slow
Distance. *Sportscience 13*, 32-53.
            """)
