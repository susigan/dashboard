from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from datetime import datetime, timedelta
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

MC = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}

# ── Métricas subjectivas 1-5 que partilham o mesmo eixo ──────────────────────
# Escala original do formulário: 1=mau → 5=óptimo
# EXCEPÇÃO: stress — aqui 1=relaxado(bom) → 5=muito stressado(mau)
# → invertemos stress para que 5 = sempre "melhor" em todos
_SUBJ_METS = ['sleep_quality', 'fatiga', 'humor', 'soreness', 'stress']
_SUBJ_INVERT = {'stress'}   # estas métricas são invertidas (5-x+1)
_SUBJ_LABELS = {
    'sleep_quality': 'Sono',
    'fatiga':        'Energia',
    'humor':         'Humor',
    'soreness':      'Sem Dor',   # soreness invertida = sem dor
    'stress':        'Relaxamento',
}
_SUBJ_COLORS = {
    'sleep_quality': '#9b59b6',
    'fatiga':        '#f39c12',
    'humor':         '#2ecc71',
    'soreness':      '#3498db',
    'stress':        '#1abc9c',
}

# Fisiológicas — eixo próprio
_PHYSIO = {
    'hrv': {'label': 'HRV (ms)',  'color': '#27ae60'},
    'rhr': {'label': 'RHR (bpm)', 'color': '#e74c3c'},
}

# Colorscale semáforo: 0→vermelho, 3→amarelo, 5→verde
_GREEN_RED = [
    [0.0, '#e74c3c'],
    [0.3, '#f39c12'],
    [0.6, '#f1c40f'],
    [1.0, '#2ecc71'],
]


def _invert(series, scale=5):
    """Invert so that high = always better (stress: 1→5, 5→1)."""
    return scale + 1 - series


def _score_colour(v, vmin=1, vmax=5):
    """Hex colour for a value on a 1-5 scale (red→green)."""
    if pd.isna(v): return '#cccccc'
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    if t < 0.5:
        r = 255; g = int(t * 2 * 255)
    else:
        r = int((1 - t) * 2 * 255); g = 255
    return f'#{r:02x}{g:02x}00'


def tab_wellness(dw):
    st.header("🧘 Wellness")

    if len(dw) == 0:
        st.warning("Sem dados de wellness.")
        return

    dw = dw.copy().sort_values('Data')
    dw['Data'] = pd.to_datetime(dw['Data'])

    # Available metrics
    subj_avail  = [m for m in _SUBJ_METS  if m in dw.columns and dw[m].notna().any()]
    physio_avail= [m for m in _PHYSIO     if m in dw.columns and dw[m].notna().any()]

    if not subj_avail and not physio_avail:
        st.warning("Sem métricas wellness disponíveis.")
        return

    # ── Controlos globais ─────────────────────────────────────────────────────
    _c1, _c2 = st.columns(2)
    n_dias = _c1.slider("Dias a mostrar", 14, min(len(dw), 365),
                         min(90, len(dw)), key="well_dias")
    roll_w = _c2.slider("Rolling médio (dias)", 3, 21, 7, key="well_roll")

    dw_p = dw.tail(n_dias).copy()

    # ════════════════════════════════════════════════════════════════════════
    # GRÁFICO 1 — Métricas subjectivas no MESMO eixo 1–5
    # stress, sono, energia, humor, soreness (todos escalados 1=mau→5=bom)
    # ════════════════════════════════════════════════════════════════════════
    if subj_avail:
        st.subheader("📊 Métricas Subjectivas — escala 1 (mau) → 5 (bom)")
        st.caption(
            "Stress e Cansaço Muscular estão invertidos — 5 = melhor. "
            "Linha colorida = raw diário | tracejada preta = rolling médio."
        )

        sel_subj = st.multiselect(
            "Métricas", subj_avail,
            default=subj_avail,
            format_func=lambda m: _SUBJ_LABELS.get(m, m),
            key="well_subj_sel"
        )

        if sel_subj:
            fig_subj = go.Figure()
            dates = dw_p['Data'].tolist()

            # Score combinado (média das métricas seleccionadas, após inversão)
            _scores = pd.DataFrame(index=dw_p.index)
            for met in sel_subj:
                raw = pd.to_numeric(dw_p[met], errors='coerce')
                v = _invert(raw) if met in _SUBJ_INVERT else raw
                v = v.clip(1, 5)
                _scores[met] = v
                lbl = _SUBJ_LABELS.get(met, met)
                cor = _SUBJ_COLORS.get(met, '#888')

                # Raw markers
                fig_subj.add_trace(go.Scatter(
                    x=dates, y=v.tolist(),
                    mode='markers',
                    name=lbl,
                    marker=dict(color=cor, size=6, opacity=0.7),
                    showlegend=True,
                    hovertemplate=f'<b>{lbl}</b><br>%{{x|%d/%m/%Y}}: <b>%{{y:.1f}}</b><extra></extra>',
                ))
                # Rolling mean
                v_roll = v.rolling(roll_w, min_periods=2).mean()
                fig_subj.add_trace(go.Scatter(
                    x=dates, y=v_roll.tolist(),
                    mode='lines',
                    name=f'{lbl} {roll_w}d',
                    line=dict(color=cor, width=2.5),
                    showlegend=False,
                    hovertemplate=f'{lbl} roll: <b>%{{y:.2f}}</b><extra></extra>',
                ))

            # Reference lines
            fig_subj.add_hline(y=3, line_dash='dot', line_color='#f1c40f',
                                line_width=1.5, annotation_text='neutro (3)',
                                annotation_font_color='#888', annotation_font_size=9)
            fig_subj.add_hline(y=4, line_dash='dot', line_color='#2ecc71',
                                line_width=1, annotation_text='bom (4)',
                                annotation_font_color='#888', annotation_font_size=9)
            fig_subj.add_hline(y=2, line_dash='dot', line_color='#e74c3c',
                                line_width=1, annotation_text='mau (2)',
                                annotation_font_color='#888', annotation_font_size=9)

            fig_subj.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=11),
                height=420,
                margin=dict(t=50, b=85, l=45, r=20),
                hovermode='x unified',
                yaxis=dict(range=[0.8, 5.2], dtick=1,
                           tickvals=[1, 2, 3, 4, 5],
                           ticktext=['1 ●', '2', '3', '4', '5 ★'],
                           showgrid=True, gridcolor='#eee',
                           tickfont=dict(color='#111')),
                xaxis=dict(showgrid=True, gridcolor='#eee',
                           tickangle=-45, tickfont=dict(color='#111')),
                legend=dict(orientation='h', y=-0.25,
                            font=dict(color='#111', size=10)),
                title=dict(text='Métricas Subjectivas (1→5)',
                           font=dict(size=13, color='#111')),
            )
            st.plotly_chart(fig_subj, use_container_width=True, config=MC,
                            key="well_subj_chart")

        st.markdown("---")

        # ════════════════════════════════════════════════════════════════════
        # GRÁFICO 2 — Score composto diário (média das subjectivas)
        # Colorido por semáforo e com baseline estatístico
        # ════════════════════════════════════════════════════════════════════
        st.subheader("🎯 Score Composto de Wellness — Baseline Estatístico")
        st.caption(
            "Média das métricas subjectivas (stress e cansaço invertidos). "
            "Baseline = média rolante 28d ± 0.5 SD. "
            "Pontos coloridos por semáforo: verde ≥ baseline+0.5SD, "
            "amarelo = zona normal, vermelho ≤ baseline−0.5SD."
        )

        if subj_avail:
            # Compute composite score over full history (for stable baseline)
            _all_scores = pd.DataFrame(index=dw.index)
            for met in subj_avail:
                raw = pd.to_numeric(dw[met], errors='coerce')
                _all_scores[met] = (_invert(raw) if met in _SUBJ_INVERT else raw).clip(1, 5)

            dw['_composite'] = _all_scores.mean(axis=1)

            # Statistical baseline: 28d rolling mean ± 0.5 SD
            dw['_base_mean'] = dw['_composite'].rolling(28, min_periods=7).mean()
            dw['_base_std']  = dw['_composite'].rolling(28, min_periods=7).std()
            dw['_band_hi']   = dw['_base_mean'] + 0.5 * dw['_base_std']
            dw['_band_lo']   = dw['_base_mean'] - 0.5 * dw['_base_std']

            # Zone classification
            def _zone(row):
                if pd.isna(row['_composite']) or pd.isna(row['_base_mean']):
                    return 'sem dados', '#cccccc'
                if row['_composite'] >= row['_band_hi']:
                    return 'acima baseline', '#27ae60'
                if row['_composite'] <= row['_band_lo']:
                    return 'abaixo baseline', '#e74c3c'
                return 'normal', '#f1c40f'

            dw[['_zone', '_zcol']] = dw.apply(
                lambda r: pd.Series(_zone(r)), axis=1)

            dw_comp = dw.tail(n_dias).copy()

            fig_comp = go.Figure()

            # Band fill
            fig_comp.add_trace(go.Scatter(
                x=dw_comp['Data'].tolist(),
                y=dw_comp['_band_hi'].tolist(),
                line=dict(color='rgba(39,174,96,0.3)', width=1),
                showlegend=False, hoverinfo='skip'))
            fig_comp.add_trace(go.Scatter(
                x=dw_comp['Data'].tolist(),
                y=dw_comp['_band_lo'].tolist(),
                fill='tonexty', fillcolor='rgba(39,174,96,0.08)',
                line=dict(color='rgba(39,174,96,0.3)', width=1),
                name='Zona normal (baseline ±0.5SD)',
                hoverinfo='skip'))

            # Baseline mean
            fig_comp.add_trace(go.Scatter(
                x=dw_comp['Data'].tolist(),
                y=dw_comp['_base_mean'].tolist(),
                mode='lines',
                name='Baseline (28d)',
                line=dict(color='#2c3e50', width=2, dash='dash'),
                hovertemplate='Baseline: <b>%{y:.2f}</b><extra></extra>'))

            # Composite score — coloured by zone
            for zone, zcol, zname in [
                ('acima baseline', '#27ae60', '↑ Acima baseline'),
                ('normal',         '#f1c40f', '→ Normal'),
                ('abaixo baseline', '#e74c3c', '↓ Abaixo baseline'),
                ('sem dados',      '#cccccc', 'Sem dados'),
            ]:
                mask = dw_comp['_zone'] == zone
                if mask.any():
                    fig_comp.add_trace(go.Scatter(
                        x=dw_comp.loc[mask, 'Data'].tolist(),
                        y=dw_comp.loc[mask, '_composite'].tolist(),
                        mode='markers',
                        name=zname,
                        marker=dict(color=zcol, size=8,
                                    line=dict(color='white', width=1)),
                        hovertemplate=(
                            f'<b>{zname}</b><br>'
                            '%{x|%d/%m/%Y}: <b>%{y:.2f}</b><extra></extra>'
                        )))

            # Rolling composite
            _comp_roll = dw_comp['_composite'].rolling(roll_w, min_periods=2).mean()
            fig_comp.add_trace(go.Scatter(
                x=dw_comp['Data'].tolist(),
                y=_comp_roll.tolist(),
                mode='lines',
                name=f'Score {roll_w}d roll',
                line=dict(color='#2c3e50', width=2.5),
                hovertemplate='Score roll: <b>%{y:.2f}</b><extra></extra>'))

            fig_comp.add_hline(y=3, line_dash='dot', line_color='#888',
                                line_width=1)

            fig_comp.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=11),
                height=400, hovermode='x unified',
                margin=dict(t=55, b=85, l=45, r=20),
                yaxis=dict(range=[0.8, 5.2],
                           showgrid=True, gridcolor='#eee',
                           tickfont=dict(color='#111'),
                           title='Score composto'),
                xaxis=dict(showgrid=True, gridcolor='#eee',
                           tickangle=-45, tickfont=dict(color='#111')),
                legend=dict(orientation='h', y=-0.28,
                            font=dict(color='#111', size=10)),
                title=dict(
                    text='Score Composto Wellness + Baseline (28d ±0.5SD)',
                    font=dict(size=13, color='#111')),
            )
            st.plotly_chart(fig_comp, use_container_width=True, config=MC,
                            key="well_comp_chart")

            # ── Análise estatística do baseline ──────────────────────────────
            st.markdown("### 📐 Análise Estatística — Baseline Composto")

            _comp_full = dw['_composite'].dropna()
            if len(_comp_full) >= 14:
                _mean  = _comp_full.mean()
                _med   = _comp_full.median()
                _sd    = _comp_full.std()
                _cv    = _sd / _mean * 100
                _last  = float(dw_comp['_composite'].dropna().iloc[-1]) if len(dw_comp) > 0 else None
                _base_last = float(dw_comp['_base_mean'].dropna().iloc[-1]) if len(dw_comp) > 0 else None
                _band_hi_l = float(dw_comp['_band_hi'].dropna().iloc[-1]) if len(dw_comp) > 0 else None
                _band_lo_l = float(dw_comp['_band_lo'].dropna().iloc[-1]) if len(dw_comp) > 0 else None

                _sc1, _sc2, _sc3, _sc4 = st.columns(4)
                _sc1.metric("Média histórica",    f"{_mean:.2f} / 5")
                _sc2.metric("Mediana",             f"{_med:.2f}")
                _sc3.metric("SD",                  f"{_sd:.2f}")
                _sc4.metric("CV%",                 f"{_cv:.1f}%",
                             help="Coeficiente de variação — variabilidade relativa do wellness")

                if _last and _base_last:
                    _sc5, _sc6, _sc7, _sc8 = st.columns(4)
                    _sc5.metric("Score HOJE",       f"{_last:.2f}")
                    _sc6.metric("Baseline 28d",     f"{_base_last:.2f}")
                    _sc7.metric("Zona normal",
                                f"{_band_lo_l:.2f} – {_band_hi_l:.2f}" if _band_lo_l else "—")
                    _dev = (_last - _base_last) / _sd if _sd > 0 else 0
                    _sc8.metric("Desvio vs baseline",
                                f"{_dev:+.2f} SD",
                                delta=("🟢 Acima" if _dev > 0.5
                                       else "🔴 Abaixo" if _dev < -0.5
                                       else "🟡 Normal"))

                # Trend: slope over last 14d
                _comp_14 = dw['_composite'].dropna().tail(14)
                if len(_comp_14) >= 7:
                    _x14 = np.arange(len(_comp_14))
                    _sl, _int, _r, _p, _ = scipy_stats.linregress(_x14, _comp_14.values)
                    _trend_lbl = (
                        f"↗ Melhorando ({_sl:+.3f}/dia, p={_p:.3f})" if _sl > 0.02 and _p < 0.10
                        else f"↘ Deteriorando ({_sl:+.3f}/dia, p={_p:.3f})" if _sl < -0.02 and _p < 0.10
                        else f"→ Estável ({_sl:+.3f}/dia, p={_p:.3f})"
                    )
                    st.info(f"**Tendência 14d:** {_trend_lbl}")

                # Per-metric breakdown
                st.markdown("**📊 Breakdown por Métrica (todo o histórico)**")
                _rows_stat = []
                for met in subj_avail:
                    raw = pd.to_numeric(dw[met], errors='coerce')
                    v = (_invert(raw) if met in _SUBJ_INVERT else raw).clip(1, 5).dropna()
                    if len(v) < 5: continue
                    _m = v.mean(); _s = v.std()
                    _p25, _p75 = v.quantile(0.25), v.quantile(0.75)
                    _rows_stat.append({
                        'Métrica':    _SUBJ_LABELS.get(met, met),
                        'Média':      f"{_m:.2f}",
                        'SD':         f"{_s:.2f}",
                        'CV%':        f"{_s/_m*100:.1f}%",
                        'P25–P75':    f"{_p25:.1f} – {_p75:.1f}",
                        'Mín':        f"{v.min():.0f}",
                        'Máx':        f"{v.max():.0f}",
                        'N':          len(v),
                        'Cor média':  _score_colour(_m),
                    })
                if _rows_stat:
                    _df_stat = pd.DataFrame(_rows_stat)
                    # Colour the Média column
                    def _style_mean(v):
                        try:
                            fv = float(v)
                            c = _score_colour(fv)
                            return f'background-color:{c}22;color:#111'
                        except: return ''
                    st.dataframe(
                        _df_stat.drop(columns=['Cor média'])
                                .style.map(_style_mean, subset=['Média']),
                        use_container_width=True, hide_index=True)

        st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # GRÁFICO 3 — HRV e RHR (eixo fisiológico separado)
    # ════════════════════════════════════════════════════════════════════════
    if physio_avail:
        st.subheader("❤️ HRV & RHR")
        n_phys = len(physio_avail)
        fig_ph = make_subplots(rows=n_phys, cols=1, shared_xaxes=True,
                                subplot_titles=[_PHYSIO[m]['label'] for m in physio_avail],
                                vertical_spacing=0.06)
        for ri, met in enumerate(physio_avail, 1):
            v = pd.to_numeric(dw_p[met], errors='coerce')
            cor = _PHYSIO[met]['color']
            dates = dw_p['Data'].tolist()
            fig_ph.add_trace(go.Scatter(
                x=dates, y=v.tolist(), mode='lines+markers',
                name=_PHYSIO[met]['label'],
                line=dict(color=cor, width=2),
                marker=dict(size=4, color=cor),
                hovertemplate='%{x|%d/%m/%Y}: <b>%{y:.0f}</b><extra></extra>',
            ), row=ri, col=1)
            v_roll = v.rolling(roll_w, min_periods=2).mean()
            fig_ph.add_trace(go.Scatter(
                x=dates, y=v_roll.tolist(), mode='lines',
                name=f'{_PHYSIO[met]["label"]} {roll_w}d',
                line=dict(color='#2c3e50', width=1.5, dash='dash'),
                showlegend=False,
                hovertemplate='roll: <b>%{y:.0f}</b><extra></extra>',
            ), row=ri, col=1)
            fig_ph.update_yaxes(
                title_text=_PHYSIO[met]['label'],
                title_font=dict(size=9, color='#555'),
                tickfont=dict(color='#111', size=9),
                showgrid=True, gridcolor='#eee', row=ri, col=1)

        fig_ph.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10),
            height=max(260, n_phys * 180),
            margin=dict(t=40, b=80, l=65, r=20),
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.18, font=dict(color='#111', size=10)),
        )
        fig_ph.update_xaxes(showgrid=True, gridcolor='#eee',
                             tickfont=dict(color='#111', size=9),
                             tickangle=-45, row=n_phys, col=1)
        st.plotly_chart(fig_ph, use_container_width=True, config=MC,
                        key="well_physio_chart")

        st.markdown("---")

    # ── Resumo 7 dias ─────────────────────────────────────────────────────────
    st.subheader("📋 Resumo — últimos 7 dias")
    u7 = dw.sort_values('Data').tail(7)
    all_mets = subj_avail + physio_avail
    rows_tbl = []
    for m in all_mets:
        v7 = pd.to_numeric(u7[m], errors='coerce')
        if v7.notna().sum() == 0: continue
        if m in subj_avail:
            v7_sc = (_invert(v7) if m in _SUBJ_INVERT else v7).clip(1, 5)
            mean_v = v7_sc.mean()
        else:
            mean_v = v7.mean()
        rows_tbl.append({
            'Métrica':  _SUBJ_LABELS.get(m, _PHYSIO.get(m, {}).get('label', m)),
            'Média 7d': f"{mean_v:.1f}",
            'Mín':      f"{v7.min():.0f}",
            'Máx':      f"{v7.max():.0f}",
            'N':        f"{v7.notna().sum()}/7",
        })
    if rows_tbl:
        st.dataframe(pd.DataFrame(rows_tbl),
                     use_container_width=True, hide_index=True)
