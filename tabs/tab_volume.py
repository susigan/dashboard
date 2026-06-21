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


def _val_text(v, total, unit='', decimals=1):
    """Texto de valor absoluto dentro da coluna (em vez de %)."""
    if total <= 0 or v / total < 0.05:
        return ''
    return f"{v:.{decimals}f}{unit}"


def _fmt_h(horas):
    """Formata horas decimais como 'Hh MMm' / 'MMmin'. Ex: 2.7 → '2h42', 0.8 → '48min'."""
    if horas is None or horas <= 0:
        return ''
    total_min = int(round(horas * 60))
    h, m = divmod(total_min, 60)
    if h == 0:
        return f"{m}min"
    if m == 0:
        return f"{h}h"
    return f"{h}h{m:02d}"


def _stacked_pct(pivot, order, cmap, title, ytitle, key, show_mean_line=False,
                 unit='', decimals=1, fmt_func=None):
    fig = go.Figure()
    totals = pivot.sum(axis=1)
    for cat in order:
        if cat not in pivot.columns:
            continue
        y = pivot[cat]
        if fmt_func is not None:
            # Usa formatador dedicado (ex: horas → 'Hh MMm'), respeitando threshold 5%
            txt = [fmt_func(v) if (t > 0 and v / t >= 0.05) else ''
                   for v, t in zip(y, totals)]
        else:
            txt = [_val_text(v, t, unit, decimals) for v, t in zip(y, totals)]
        fig.add_trace(go.Bar(
            x=[str(x) for x in pivot.index], y=y.tolist(),
            name=cat, marker_color=cmap.get(cat, '#888'),
            marker_line_width=0, opacity=0.88,
            text=txt,
            textposition='inside', textfont=dict(color='white', size=12),
            hovertemplate=f'<b>{cat}</b><br>%{{x}}: <b>%{{y:.1f}}{unit}</b><extra></extra>',
        ))
    if show_mean_line and len(totals) > 0:
        mean_val = float(totals.mean())
        _mean_lbl = fmt_func(mean_val) if fmt_func is not None else f"{mean_val:.1f}{unit}"
        fig.add_hline(
            y=mean_val, line_dash='dash', line_color='#2c3e50', line_width=1.5,
            annotation_text=f"Média: {_mean_lbl}",
            annotation_position="top right",
            annotation_font=dict(color='#2c3e50', size=10))
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
                         f'Horas — {_p1}', 'Horas', 'vol_h_cic',
                         show_mean_line=True, fmt_func=_fmt_h),
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
                         f'Distância (km) — {_p2}', 'km', 'vol_km_cic',
                         show_mean_line=True, unit='km', decimals=1),
            use_container_width=True, config=MC, key="vol_km_cic")

        with st.expander("📋 Tabela: % e total km por modalidade", expanded=False):
            _summary_table_raw(df_km, _pc2, 'km', _pc2, 'type',
                                _CICLICOS, 'km', 'km_cic')
    else:
        st.info("Sem dados de distância.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 2b. kJ — Cíclicos (z1_kj + z2_kj + z3_kj)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔥 kJ por Modalidade — cíclicos")
    _kj_cols = [c for c in ['z1_kj', 'z2_kj', 'z3_kj'] if c in df.columns]
    if _kj_cols:
        df_kj = df[df['type'].isin(_CICLICOS)].copy()
        df_kj['kj'] = sum(
            pd.to_numeric(df_kj[c], errors='coerce').fillna(0) for c in _kj_cols)
        df_kj = df_kj[df_kj['kj'] > 0]
        if len(df_kj) > 0:
            _pkj = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                                key="vol_pkj", index=1)
            _pckj = _PERIODO_OPTS[_pkj]
            pivot_kj = (df_kj.pivot_table(index=_pckj, columns='type',
                                          values='kj', aggfunc='sum', fill_value=0)
                            .sort_index())
            st.plotly_chart(
                _stacked_pct(pivot_kj, _CICLICOS, _CORES_MOD,
                             f'Trabalho (kJ) — {_pkj}', 'kJ', 'vol_kj_cic',
                             show_mean_line=True, unit='', decimals=0),
                use_container_width=True, config=MC, key="vol_kj_cic")

            with st.expander("📋 Tabela: % e total kJ por modalidade", expanded=False):
                _summary_table_raw(df_kj, _pckj, 'kj', _pckj, 'type',
                                    _CICLICOS, 'kJ', 'kj_cic')
        else:
            st.info("Sem dados de kJ por zona.")
    else:
        st.info("Colunas z1_kj/z2_kj/z3_kj não disponíveis.")

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
                             'Horas', 'vol_rpe', fmt_func=_fmt_h),
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
        # Linha de média
        if len(pivot_wt) > 0:
            _wt_mean = float(pivot_wt.mean())
            fig_wt.add_hline(
                y=_wt_mean, line_dash='dash', line_color='#2c3e50', line_width=1.5,
                annotation_text=f"Média: {_wt_mean:.1f}h",
                annotation_position="top right",
                annotation_font=dict(color='#2c3e50', size=10))
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
        "Outliers removidos por modalidade (IQR×3 independente por tipo).")

    _xss_col = next((c for c in ['xss','SS','XSS'] if c in df.columns and df[c].notna().any()), None)
    _aer_col = next((c for c in ['aerobic','cp_usage','AllWorkFTP'] if c in df.columns and df[c].notna().any()), None)
    _gly_col = next((c for c in ['glycolytic','glycolytic_usage'] if c in df.columns and df[c].notna().any()), None)
    _spr_col = next((c for c in ['pmax','p_max_usage','icu_pm_p_max'] if c in df.columns and df[c].notna().any()), None)

    df_str = df[df['type'].isin(_CICLICOS)].copy()

    # ── Limpeza de outliers por modalidade (IQR×3) ─────────────────────────
    # Run tem XSS intrinsecamente mais alto — tratar cada modalidade separada
    def _clean_outliers_by_modal(df_in, cols_to_clean, factor=3.0):
        """Remove outliers por modalidade independentemente usando IQR×factor."""
        df_out = df_in.copy()
        for mod in df_out['type'].unique():
            mask_mod = df_out['type'] == mod
            for col in cols_to_clean:
                if col not in df_out.columns: continue
                s = pd.to_numeric(df_out.loc[mask_mod, col], errors='coerce')
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0: continue
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                # Marcar outliers como NaN (mantém a linha mas exclui o valor)
                outlier_mask = mask_mod & ((df_out[col] < lower) | (df_out[col] > upper))
                n_out = outlier_mask.sum()
                if n_out > 0:
                    df_out.loc[outlier_mask, col] = np.nan
        return df_out

    _cols_clean = [c for c in [_xss_col, _aer_col, _gly_col, _spr_col] if c]
    df_str_clean = _clean_outliers_by_modal(df_str, _cols_clean, factor=3.0)

    # Contar outliers removidos por modalidade
    _n_outliers_total = 0
    for col in _cols_clean:
        if col in df_str.columns:
            _n_outliers_total += (df_str[col].notna() & df_str_clean[col].isna()).sum()
    if _n_outliers_total > 0:
        st.caption(f"⚠️ {_n_outliers_total} outliers removidos (IQR×3 por modalidade).")

    _p5 = st.selectbox("Período", list(_PERIODO_OPTS.keys()),
                        key="vol_p5", index=1)
    _pc5 = _PERIODO_OPTS[_p5]

    # ── 5a. XSS total ──────────────────────────────────────────────────────
    if _xss_col and len(df_str_clean) > 0:
        df_str_clean['_xss'] = pd.to_numeric(df_str_clean[_xss_col], errors='coerce').fillna(0)
        pivot_xss = (df_str_clean.pivot_table(index=_pc5, columns='type',
                                              values='_xss', aggfunc='sum', fill_value=0)
                                 .sort_index())
        st.plotly_chart(
            _stacked_pct(pivot_xss, _CICLICOS, _CORES_MOD,
                         f'XSS Total — {_p5}', 'XSS', 'vol_xss',
                         show_mean_line=True),
            use_container_width=True, config=MC, key="vol_xss")
    else:
        st.info("Coluna XSS não disponível.")

    # ── 5b. Decomposição por sistema ───────────────────────────────────────
    _sys_map = {}
    if _aer_col: _sys_map[_aer_col] = 'oxidative'
    if _gly_col: _sys_map[_gly_col] = 'glycolytic'
    if _spr_col: _sys_map[_spr_col] = 'sprint'

    # Usar TODAS as modalidades presentes no histórico, não só no período filtrado
    _mods_avail = [m for m in _CICLICOS if m in df_str_clean['type'].unique()]

    if _sys_map and _mods_avail:
        st.markdown("**Decomposição por Sistema Energético**")
        n_mods = len(_mods_avail)
        fig_sys = make_subplots(rows=1, cols=n_mods,
                                subplot_titles=_mods_avail,
                                shared_yaxes=True)

        for col_i, mod in enumerate(_mods_avail, 1):
            df_mod = df_str_clean[df_str_clean['type'] == mod].copy()
            sys_series = {}
            for src_col, sys_name in _sys_map.items():
                if src_col in df_mod.columns:
                    df_mod[f'_{sys_name}'] = pd.to_numeric(df_mod[src_col], errors='coerce').fillna(0)
                    grp = df_mod.groupby(_pc5)[f'_{sys_name}'].sum().sort_index()
                    if grp.sum() > 0:
                        sys_series[sys_name] = grp
            if not sys_series: continue
            all_idx = sorted(set().union(*[s.index for s in sys_series.values()]))
            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                if sys_name not in sys_series: continue
                s = sys_series[sys_name].reindex(all_idx, fill_value=0)
                totals_mod = sum(sys_series[sn].reindex(all_idx, fill_value=0)
                                 for sn in sys_series)
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
        _all_periods = sorted(df_str_clean[_pc5].dropna().unique())
        _p_cur  = _all_periods[-1] if _all_periods else None
        _p_prev = _all_periods[-2] if len(_all_periods) >= 2 else None

        # Usar TODAS as modalidades que têm dados nos sistemas energéticos
        # (não só as que têm dados no período actual — fix para Ski aparecer)
        _sys_data = {}
        for mod in _mods_avail:
            df_mod = df_str_clean[df_str_clean['type'] == mod].copy()
            has_data = False
            _sys_data[mod] = {sn: {} for sn in ['oxidative', 'glycolytic', 'sprint']}
            for src_col, sys_name in _sys_map.items():
                if src_col not in df_mod.columns: continue
                df_mod[f'_{sys_name}'] = pd.to_numeric(df_mod[src_col], errors='coerce').fillna(0)
                agg = df_mod.groupby(_pc5)[f'_{sys_name}'].sum()
                for p in _all_periods:
                    v = float(agg.get(p, 0))
                    _sys_data[mod][sys_name][p] = v
                    if v > 0: has_data = True

        tab_rows = []
        for mod in _mods_avail:
            gc = sum(_sys_data[mod][sn].get(_p_cur, 0) for sn in ['oxidative','glycolytic','sprint'])
            gp = (sum(_sys_data[mod][sn].get(_p_prev, 0) for sn in ['oxidative','glycolytic','sprint'])
                  if _p_prev else None)
            # Incluir modalidade mesmo se gc==0 no período actual
            # (pode ter dados no período anterior → mostra o Δ)
            if gc == 0 and (gp is None or gp == 0):
                continue  # sem dados em nenhum período → ignorar
            row = {'Modalidade': mod, 'Período': str(_p_cur)}
            for sys_name in ['oxidative', 'glycolytic', 'sprint']:
                vc  = _sys_data[mod][sys_name].get(_p_cur, 0)
                vp  = _sys_data[mod][sys_name].get(_p_prev, 0) if _p_prev else None
                pct_c = vc / gc * 100 if gc > 0 else 0
                pct_p = (vp / gp * 100 if (vp is not None and gp and gp > 0) else None)
                arrow, _ = _delta_icon(pct_c, pct_p)
                row[sys_name.capitalize()] = f"{pct_c:.1f}%" if gc > 0 else '—'
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

    # ── Training Load Mensal por Modalidade (movido de tab_analises) ──────────
    # ── Secção 2: Training Load Mensal Stacked ──────────────────────────────
    st.subheader("📊 Training Load Mensal por Modalidade (TRIMP = min × RPE)")
    df_tl = filtrar_principais(da).copy()  # da = variável correcta em tab_volume
    df_tl = add_tempo(df_tl)
    if 'moving_time' in df_tl.columns and 'rpe' in df_tl.columns:
        df_tl['rpe_fill']    = df_tl['rpe'].fillna(df_tl['rpe'].median())
        df_tl['session_rpe'] = (pd.to_numeric(df_tl['moving_time'], errors='coerce') / 60) * df_tl['rpe_fill']
        df_tl = df_tl[df_tl['type'].isin(['Bike', 'Run', 'Row', 'Ski', 'WeightTraining'])]
        pivot_tl = df_tl.pivot_table(index='mes', columns='type', values='session_rpe', aggfunc='sum', fill_value=0).sort_index()
        CORES_MOD = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo'], 'WeightTraining': CORES['laranja']}
        _fig_sb = go.Figure()
        if len(pivot_tl) > 0:
            for _tc in [c for c in pivot_tl.columns if c in CORES_MOD]:
                _fig_sb.add_trace(go.Bar(x=[str(x) for x in pivot_tl.index],
                    y=pivot_tl[_tc].tolist(), name=_tc,
                    marker_color=CORES_MOD.get(_tc,'gray'),
                    marker_line_width=0, opacity=0.85))
        _fig_sb.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), barmode='stack', height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(tickangle=-45, showgrid=False, gridcolor='#eee', tickfont=dict(color='#111')),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_sb, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="volume_training_load_mensal")
