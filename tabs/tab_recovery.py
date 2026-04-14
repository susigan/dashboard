from utils.config import *
from utils.helpers import *
from utils.data import *

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import sys, os as _os
from scipy import stats

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
warnings.filterwarnings('ignore')


def tab_recovery(dw):

    st.header("🔋 Recovery Score & HRV Analysis")

    if len(dw) == 0 or 'hrv' not in dw.columns:
        st.warning("Sem dados de HRV.")
        return

    rec = calcular_recovery(dw)
    if len(rec) == 0:
        return

    u = rec.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{u['recovery_score']:.0f}")
    c2.metric("HRV", f"{u['hrv']:.0f}" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline", f"{u['hrv_baseline']:.0f}" if pd.notna(u['hrv_baseline']) else "—")
    _cv7 = u.get('hrv_cv_7d', None)
    c4.metric("CV%", f"{_cv7:.1f}%" if _cv7 is not None else "—")

    st.markdown("---")

    col1, col2 = st.columns(2)
    n_dias    = col1.slider("Dias", 14, min(len(dw), 365), 90, key="rec_dias")
    janela_cv = col2.slider("Janela CV", 3, 14, 7, key="rec_jcv")

    modo_modelo = st.radio(
        "Modelo",
        ["Mode 1 — Altini", "Mode 2 — Plews"],
        horizontal=True,
        key="rec_modo"
    )

    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.tail(n_dias)

    df['LnrMSSD'] = np.where(df['hrv'] > 0, np.log(df['hrv']), np.nan)
    df = df.dropna(subset=['LnrMSSD'])

    if len(df) < 10:
        st.warning("Poucos dados.")
        return

    # ── Baseline: Mode 1 = 7d (Altini, janela curta p/ CV), Mode 2 = 60d (Plews) ──
    baseline_w = 7 if "Mode 1" in modo_modelo else 60

    df['baseline'] = df['LnrMSSD'].rolling(baseline_w, min_periods=5).mean()
    df['std']      = df['LnrMSSD'].rolling(baseline_w, min_periods=5).std()

    df['cv'] = (
        df['LnrMSSD'].rolling(janela_cv, min_periods=3).std() /
        df['LnrMSSD'].rolling(janela_cv, min_periods=3).mean()
    ) * 100

    # ── SWC = Smallest Worthwhile Change = 0.5 * CV do baseline ──────────────
    df['SWC']   = 0.5 * (df['std'] / df['baseline'] * 100)
    df['upper'] = df['baseline'] * (1 + df['SWC'] / 100)
    df['lower'] = df['baseline'] * (1 - df['SWC'] / 100)

    # ── Thresholds de CV: média ± 0.5 SD do histórico de CV ──────────────────
    cv_hist = df['cv'].dropna()
    if len(cv_hist) > 10:
        cv_mean = cv_hist.mean()
        cv_std  = cv_hist.std()
        cv_low  = max(0.1, cv_mean - 0.5 * cv_std)
        cv_high = cv_mean + 0.5 * cv_std
    else:
        cv_low, cv_high = 0.5, 1.5

    # ── Slope 7d (regressão linear sobre LnrMSSD) ────────────────────────────
    def slope_fn(x):
        xd = x.dropna()
        return stats.linregress(range(len(xd)), xd)[0] if len(xd) >= 5 else np.nan

    df['slope'] = df['LnrMSSD'].rolling(7, min_periods=5).apply(slope_fn)

    # ── Classificação Mode 1 — Altini ────────────────────────────────────────
    def altini(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']):
            return 'Sem dados', '#808080'
        if r['LnrMSSD'] < r['baseline'] and r['cv'] < cv_low:
            return 'Accumulated Fatigue', '#e74c3c'
        if r['LnrMSSD'] < r['baseline'] and r['cv'] > cv_high:
            return 'Maladaptation', '#f1c40f'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] < cv_low:
            return 'Good Adaptation', '#27ae60'
        if r['LnrMSSD'] > r['baseline'] and r['cv'] > cv_high:
            return 'High Variability', '#2c3e50'
        return 'Normal', '#95a5a6'

    # ── Classificação Mode 2 — Plews ─────────────────────────────────────────
    def plews(r):
        if pd.isna(r['cv']) or pd.isna(r['baseline']):
            return 'Sem dados', '#808080'
        declinio = r['slope'] < -0.01 if pd.notna(r['slope']) else False
        if r['cv'] < cv_low and declinio:
            return 'NFOR', '#8b0000'
        if r['LnrMSSD'] < r['lower']:
            return 'Overreaching', '#e67e22'
        if r['cv'] > cv_high:
            return 'High Variability', '#2c3e50'
        return 'Normal', '#27ae60'

    if "Mode 1" in modo_modelo:
        df[['zona', 'cor']] = df.apply(lambda r: pd.Series(altini(r)), axis=1)
    else:
        df[['zona', 'cor']] = df.apply(lambda r: pd.Series(plews(r)), axis=1)

    df_plot = df.dropna(subset=['baseline', 'cv'])
    if len(df_plot) == 0:
        st.warning("Sem dados suficientes após processamento.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # GRÁFICO PRINCIPAL — Barras coloridas + Baseline + SWC + CV% (y2)
    # ════════════════════════════════════════════════════════════════════════
    fig = go.Figure()

    # ── Barras coloridas por zona (transparentes) ─────────────────────────
    zonas_ordem = (
        ['Good Adaptation', 'Normal', 'High Variability', 'Maladaptation', 'Accumulated Fatigue', 'Sem dados']
        if "Mode 1" in modo_modelo else
        ['Normal', 'High Variability', 'Overreaching', 'NFOR', 'Sem dados']
    )
    zonas_vistas = df_plot[['zona', 'cor']].drop_duplicates().set_index('zona')['cor'].to_dict()
    for zona in zonas_ordem:
        if zona not in zonas_vistas:
            continue
        cor = zonas_vistas[zona]
        d   = df_plot[df_plot['zona'] == zona]
        r_h, g_h, b_h = int(cor[1:3],16), int(cor[3:5],16), int(cor[5:7],16)
        cor_fill = f'rgba({r_h},{g_h},{b_h},0.55)'
        cor_line = f'rgba({r_h},{g_h},{b_h},0.85)'
        fig.add_trace(go.Bar(
            x=d['Data'],
            y=d['LnrMSSD'],
            name=zona,
            marker=dict(color=cor_fill, line=dict(color=cor_line, width=1)),
            customdata=np.stack([d['cv'], d['slope'].fillna(0)], axis=1),
            hovertemplate=(
                '<b>' + zona + '</b><br>'
                'Data: %{x|%d/%m/%Y}<br>'
                'LnRMSSD: %{y:.3f}<br>'
                'CV%: %{customdata[0]:.2f}%<br>'
                'Slope 7d: %{customdata[1]:.4f}'
                '<extra></extra>'
            )
        ))

    # ── Baseline (linha tracejada ESCURA E GROSSA) ─────────────────────────
    fig.add_trace(go.Scatter(
        x=df_plot['Data'], y=df_plot['baseline'],
        name=f'Baseline ({baseline_w}d)',
        line=dict(color='#2c3e50', width=4, dash='dash'),
        hovertemplate='Baseline: %{y:.3f}<extra></extra>'
    ))

    # ── SWC band: maior opacidade ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_plot['Data'], y=df_plot['upper'],
        line=dict(color='rgba(44,62,80,0.40)', width=1),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['Data'], y=df_plot['lower'],
        fill='tonexty', fillcolor='rgba(44,62,80,0.20)',
        line=dict(color='rgba(44,62,80,0.40)', width=1),
        name='SWC band',
        hovertemplate='SWC lower: %{y:.3f}<extra></extra>'
    ))
    # ── CV% no eixo Y2 ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_plot['Data'], y=df_plot['cv'],
        name='CV% (eixo direito)',
        line=dict(color='#e67e22', width=2),
        marker=dict(size=4),
        yaxis='y2',
        hovertemplate='CV%: %{y:.2f}%<extra></extra>'
    ))
    # Threshold cv_low (LINHA GROSSA)
    fig.add_trace(go.Scatter(
        x=[df_plot['Data'].iloc[0], df_plot['Data'].iloc[-1]],
        y=[cv_low, cv_low],
        name=f'CV low ({cv_low:.2f}%)',
        yaxis='y2',
        line=dict(color='#e67e22', width=3, dash='dot'),
        hoverinfo='skip'
    ))
    # Threshold cv_high (LINHA GROSSA)
    fig.add_trace(go.Scatter(
        x=[df_plot['Data'].iloc[0], df_plot['Data'].iloc[-1]],
        y=[cv_high, cv_high],
        name=f'CV high ({cv_high:.2f}%)',
        yaxis='y2',
        line=dict(color='#c0392b', width=3, dash='dot'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='#111', size=12),
        height=500, barmode='relative',
        hovermode='x unified',
        margin=dict(t=60, b=80, l=60, r=80),
        title=dict(
            text=f'{"Mode 1 — Altini" if "Mode 1" in modo_modelo else "Mode 2 — Plews"}'
                 f' | Baseline {baseline_w}d | CV thresholds: low={cv_low:.2f}% / high={cv_high:.2f}%',
            font=dict(size=13, color='#111')),
        legend=dict(orientation='h', y=-0.22, font=dict(color='#111', size=10),
                    bgcolor='rgba(255,255,255,0.9)'),
        yaxis=dict(title='LnRMSSD',
                   showgrid=True, gridcolor='#eee',
                   tickfont=dict(color='#111'),
                   range=[0, 8],
                   dtick=1),
        yaxis2=dict(title=f'CV% ({janela_cv}d)',
                    overlaying='y', side='right',
                    showgrid=False,
                    tickfont=dict(color='#e67e22'),
                    title_font=dict(color='#e67e22'),
                    range=[0, max(3.0, df_plot['cv'].max() * 1.3)]),
        xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'))
    )

    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': False, 'responsive': True,
                            'scrollZoom': False},
                    key="rec_main_chart")

    # ── Status actual ────────────────────────────────────────────────────
    ultimo = df_plot.iloc[-1]
    st.markdown("### 📊 Status Atual")
    cs1, cs2, cs3, cs4 = st.columns(4)
    cor_s = ultimo['cor']
    r_h, g_h, b_h = int(cor_s[1:3],16), int(cor_s[3:5],16), int(cor_s[5:7],16)
    cs1.markdown(
        f'<div style="padding:12px;border-radius:8px;background:rgba({r_h},{g_h},{b_h},0.15);'
        f'border-left:5px solid {cor_s};">'
        f'<b style="color:{cor_s};font-size:14px;">{ultimo["zona"]}</b></div>',
        unsafe_allow_html=True)
    cs2.metric("CV%",     f"{ultimo['cv']:.2f}%",
               delta=f"low:{cv_low:.2f}% | high:{cv_high:.2f}%")
    cs3.metric("LnRMSSD", f"{ultimo['LnrMSSD']:.3f}",
               delta=f"baseline: {ultimo['baseline']:.3f}")
    cs4.metric("Slope 7d", f"{ultimo['slope']:.4f}" if pd.notna(ultimo['slope']) else "—")

    # ════════════════════════════════════════════════════════════════════════
    # EXPLICAÇÃO DO STATUS E CÁLCULO
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("📖 Como foi calculado este resultado?", expanded=True):
        if "Mode 1" in modo_modelo:
            st.markdown("""
            **Mode 1 — Altini (Baseline Curto)**
            
            **Lógica:** Matriz 2×2 baseada na posição do LnRMSSD vs Baseline e estabilidade do CV (Coeficiente de Variação).
            
            | Condição | Significado | Interpretação |
            |----------|-------------|---------------|
            | **Accumulated Fatigue** | LnRMSSD < Baseline + CV < low | Fadiga crônica acumulada. HRV está consistentemente suprimido abaixo do baseline com baixa variação. Indica necessidade de descanso. |
            | **Maladaptation** | LnRMSSD < Baseline + CV > high | Resposta inconsistente ao treino. HRV baixo mas com alta variabilidade, indicando instabilidade do sistema nervoso autônomo. |
            | **Good Adaptation** | LnRMSSD > Baseline + CV < low | Estado ideal! HRV elevado e estável. Sistema bem recuperado e adaptado. |
            | **High Variability** | LnRMSSD > Baseline + CV > high | Atenção: HRV está elevado mas instável. Pode indicar sobrecompensação ou estresse agudo não resolvido. |
            | **Normal** | Valores intermediários | Estado neutro, sem sinais claros de fadiga ou supercompensação. |
            
            **Cálculos:**
            - **Baseline**: Média móvel de 7 dias do LnRMSSD
            - **CV%**: Desvio padrão / média × 100 (janela de {janela_cv} dias)
            - **Thresholds CV**: Média histórica ± 0.5 DP do CV
            - **Status atual**: {zona_atual} (CV={cv_atual:.2f}%, vs baseline={baseline_atual:.3f})
            """.format(janela_cv=janela_cv, zona_atual=ultimo['zona'], 
                      cv_atual=ultimo['cv'], baseline_atual=ultimo['baseline']))
        else:
            st.markdown("""
            **Mode 2 — Plews (Baseline Longo)**
            
            **Lógica:** Baseada na tendência (slope 7d) + posição relativa à banda SWC (Smallest Worthwhile Change).
            
            | Condição | Significado | Interpretação |
            |----------|-------------|---------------|
            | **NFOR** (Non-Functional Overreaching) | CV < low + Slope negativo | Fadiga severa funcional. HRV estável mas em declínio contínuo. Risco de overtraining. |
            | **Overreaching** | LnRMSSD < Lower SWC | HRV abaixo da banda de variação mínima importante. Indica sobrecrecheamento agudo. |
            | **High Variability** | CV > high | Instabilidade autonômica. Resposta ao treino inconsistente, possível estresse não funcional. |
            | **Normal** | Dentro dos parâmetros normais | Recuperação adequada, pronto para carga de treino. |
            
            **Cálculos:**
            - **Baseline**: Média móvel de 60 dias do LnRMSSD (mais estável, menos sensível a flutuações agudas)
            - **SWC (Smallest Worthwhile Change)**: 0.5 × (DP do baseline / baseline) × 100
            - **Bandas**: Baseline ± SWC%
            - **Slope 7d**: Coeficiente angular da regressão linear dos últimos 7 dias
            - **Status atual**: {zona_atual} (Slope={slope_atual:.4f}, vs SWC lower={lower_atual:.3f})
            """.format(zona_atual=ultimo['zona'], slope_atual=ultimo['slope'] if pd.notna(ultimo['slope']) else 0,
                      lower_atual=ultimo['lower']))

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # HRV-GUIDED TRAINING — 2 painéis: LnrMSSD colorido + Desvio em DP (±5)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")

    if len(dw) >= 14 and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data')
        df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'] > 0, np.log(df_hg['hrv']), np.nan)
        df_hg = df_hg.dropna(subset=['LnrMSSD'])

        hg_c1, hg_c2 = st.columns(2)
        dias_fam = hg_c1.slider("Dias baseline rolling", 7, 28, 14, key="hg_baseline")
        n_hg     = hg_c2.slider("Dias a mostrar", 14, min(len(df_hg), 180),
                                  min(60, len(df_hg)), key="hg_dias")

        df_hg['bm']    = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg['bs']    = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg['linf']  = df_hg['bm'] - 0.5 * df_hg['bs']
        df_hg['lsup']  = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio_dp'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs'].replace(0, np.nan)
        df_hg['intens'] = df_hg.apply(
            lambda r: 'HIIT'       if pd.notna(r['bm']) and r['linf'] <= r['LnrMSSD'] <= r['lsup']
            else ('Recuperação'    if pd.notna(r['bm']) else 'Sem dados'), axis=1)

        # Guardar para análise de correlação posterior
        df_analysis = df_hg.copy()
        df_hg_raw = dw.copy().sort_values('Data')
        df_hg_raw['Data'] = pd.to_datetime(df_hg_raw['Data'])
        df_hg_raw['RMSSD'] = df_hg_raw['hrv'].where(df_hg_raw['hrv'] > 0)
        df_hg_raw = df_hg_raw.dropna(subset=['RMSSD'])
        df_hg_raw['bm_raw'] = df_hg_raw['RMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg_raw['bs_raw'] = df_hg_raw['RMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg_raw['desvio_dp_raw'] = (df_hg_raw['RMSSD'] - df_hg_raw['bm_raw']) / df_hg_raw['bs_raw'].replace(0, np.nan)
        df_hg_raw['intens_raw'] = df_hg_raw.apply(
            lambda r: 'HIIT' if pd.notna(r['bm_raw']) and (r['bm_raw'] - 0.5*r['bs_raw']) <= r['RMSSD'] <= (r['bm_raw'] + 0.5*r['bs_raw'])
            else ('Recuperação' if pd.notna(r['bm_raw']) else 'Sem dados'), axis=1)

        df_p = df_hg.tail(n_hg).copy()

        COR_MAP = {'HIIT': '#27ae60', 'Recuperação': '#f39c12', 'Sem dados': '#95a5a6'}

        _fig_hg = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.06,
            subplot_titles=[
                f'LnrMSSD — Baseline {dias_fam}d ± 0.5 DP',
                'Desvio do Baseline (unidades de DP) — Range ±5'
            ]
        )

        for intensidade, cor in COR_MAP.items():
            df_i = df_p[df_p['intens'] == intensidade]
            if len(df_i) == 0:
                continue
            r_h2, g_h2, b_h2 = int(cor[1:3],16), int(cor[3:5],16), int(cor[5:7],16)
            _fig_hg.add_trace(go.Scatter(
                x=df_i['Data'], y=df_i['LnrMSSD'],
                mode='markers', name=intensidade,
                marker=dict(color=cor, size=10, line=dict(width=1.5, color='white')),
                hovertemplate=f'<b>{intensidade}</b><br>%{{x|%d/%m}}: %{{y:.3f}}<extra></extra>'
            ), row=1, col=1)

        _fig_hg.add_trace(go.Scatter(
            x=df_p['Data'], y=df_p['lsup'],
            line=dict(color='rgba(39,174,96,0.4)', width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
        _fig_hg.add_trace(go.Scatter(
            x=df_p['Data'], y=df_p['linf'],
            fill='tonexty', fillcolor='rgba(39,174,96,0.12)',
            line=dict(color='rgba(39,174,96,0.4)', width=1),
            name='Zona HIIT (±0.5 DP)', hoverinfo='skip'
        ), row=1, col=1)

        _fig_hg.add_trace(go.Scatter(
            x=df_p['Data'], y=df_p['bm'],
            name=f'Baseline {dias_fam}d',
            line=dict(color='#2c3e50', width=2, dash='dash'),
            hovertemplate='Baseline: %{y:.3f}<extra></extra>'
        ), row=1, col=1)

        # Painel inferior: Desvio com range expandido para ±5 DP
        _fig_hg.add_hrect(y0=-0.5, y1=0.5, fillcolor='rgba(39,174,96,0.20)',
                          line_width=0, row=2, col=1, annotation_text="Zona HIIT", 
                          annotation_position="left")
        _fig_hg.add_hrect(y0=-5, y1=-0.5, fillcolor='rgba(243,156,18,0.10)',
                          line_width=0, row=2, col=1)
        _fig_hg.add_hrect(y0=0.5, y1=5, fillcolor='rgba(243,156,18,0.10)',
                          line_width=0, row=2, col=1)
        _fig_hg.add_hline(y=0, line_dash='solid', line_color='#27ae60',
                          line_width=1.5, row=2, col=1)
        _fig_hg.add_hline(y=0.5, line_dash='dot', line_color='#f39c12',
                          line_width=1, row=2, col=1)
        _fig_hg.add_hline(y=-0.5, line_dash='dot', line_color='#f39c12',
                          line_width=1, row=2, col=1)

        for intensidade, cor in COR_MAP.items():
            df_i = df_p[df_p['intens'] == intensidade]
            if len(df_i) == 0:
                continue
            _fig_hg.add_trace(go.Scatter(
                x=df_i['Data'], y=df_i['desvio_dp'],
                mode='markers', name=intensidade,
                showlegend=False,
                marker=dict(color=cor, size=7, opacity=0.8,
                            line=dict(width=1, color='white')),
                hovertemplate=f'%{{x|%d/%m}}: %{{y:.2f}} DP<extra></extra>'
            ), row=2, col=1)

        _fig_hg.add_trace(go.Scatter(
            x=df_p['Data'], y=df_p['desvio_dp'],
            mode='lines', line=dict(color='#7f8c8d', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ), row=2, col=1)

        _fig_hg.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=11),
            height=520,
            hovermode='x unified',
            margin=dict(t=60, b=70, l=60, r=40),
            legend=dict(orientation='h', y=-0.18, font=dict(color='#111', size=10),
                        bgcolor='rgba(255,255,255,0.9)')
        )
        _fig_hg.update_xaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'))
        _fig_hg.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'), row=1, col=1,
                              title_text='LnRMSSD')
        _fig_hg.update_yaxes(showgrid=True, gridcolor='#eee',
                              tickfont=dict(color='#111'), row=2, col=1,
                              title_text='Desvio (DP)',
                              range=[-5, 5],  # ALTERADO: Range expandido para ±5
                              zeroline=True,
                              zerolinecolor='#27ae60', zerolinewidth=1.5)

        st.plotly_chart(_fig_hg, use_container_width=True,
                        config={'displayModeBar': False, 'responsive': True,
                                'scrollZoom': False},
                        key="rec_hg_chart")

        # ── Métricas resumo ───────────────────────────────────────────────
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n  = (df_val['intens'] == 'HIIT').sum()
            rec_n   = (df_val['intens'] == 'Recuperação').sum()
            total_n = len(df_val)
            m1, m2, m3 = st.columns(3)
            m1.metric("Dias HIIT",       f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            m2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            m3.metric("Prescrição HOJE",
                      '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT'
                      else '🟠 Recuperação')

        # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE DE CORRELAÇÃO — HRV Guided vs Mode 1/Mode 2
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Análise de Correlação: HRV-Guided vs Recovery Modes")
        
        # Preparar dados para análise
        # Juntar classificações do Mode 1 e Mode 2 no dataframe de análise
        df_corr = df_analysis.copy()
        
        # Recalcular zonas Mode 1 e Mode 2 para todo o histórico
        df_corr['baseline_7'] = df_corr['LnrMSSD'].rolling(7, min_periods=5).mean()
        df_corr['baseline_60'] = df_corr['LnrMSSD'].rolling(60, min_periods=30).mean()
        df_corr['std_7'] = df_corr['LnrMSSD'].rolling(7, min_periods=5).std()
        df_corr['std_60'] = df_corr['LnrMSSD'].rolling(60, min_periods=30).std()
        
        # CV móvel de 7 dias
        df_corr['cv_7'] = (df_corr['LnrMSSD'].rolling(7, min_periods=3).std() / 
                          df_corr['LnrMSSD'].rolling(7, min_periods=3).mean()) * 100
        
        # Thresholds CV
        cv_hist_full = df_corr['cv_7'].dropna()
        if len(cv_hist_full) > 10:
            cv_m = cv_hist_full.mean()
            cv_s = cv_hist_full.std()
            cv_l = max(0.1, cv_m - 0.5 * cv_s)
            cv_h = cv_m + 0.5 * cv_s
        else:
            cv_l, cv_h = 0.5, 1.5
        
        # Slope 7d
        df_corr['slope_7'] = df_corr['LnrMSSD'].rolling(7, min_periods=5).apply(slope_fn)
        
        # SWC para Mode 2
        df_corr['SWC_60'] = 0.5 * (df_corr['std_60'] / df_corr['baseline_60'] * 100)
        df_corr['lower_60'] = df_corr['baseline_60'] * (1 - df_corr['SWC_60'] / 100)
        
        # Classificação Mode 1 (Altini)
        def altini_full(r):
            if pd.isna(r['cv_7']) or pd.isna(r['baseline_7']):
                return 'Sem_dados'
            if r['LnrMSSD'] < r['baseline_7'] and r['cv_7'] < cv_l:
                return 'Accumulated_Fatigue'
            if r['LnrMSSD'] < r['baseline_7'] and r['cv_7'] > cv_h:
                return 'Maladaptation'
            if r['LnrMSSD'] > r['baseline_7'] and r['cv_7'] < cv_l:
                return 'Good_Adaptation'
            if r['LnrMSSD'] > r['baseline_7'] and r['cv_7'] > cv_h:
                return 'High_Variability'
            return 'Normal'
        
        # Classificação Mode 2 (Plews)
        def plews_full(r):
            if pd.isna(r['cv_7']) or pd.isna(r['baseline_60']):
                return 'Sem_dados'
            declinio = r['slope_7'] < -0.01 if pd.notna(r['slope_7']) else False
            if r['cv_7'] < cv_l and declinio:
                return 'NFOR'
            if r['LnrMSSD'] < r['lower_60']:
                return 'Overreaching'
            if r['cv_7'] > cv_h:
                return 'High_Variability'
            return 'Normal'
        
        df_corr['mode1_zone'] = df_corr.apply(altini_full, axis=1)
        df_corr['mode2_zone'] = df_corr.apply(plews_full, axis=1)
        
        # Criar variáveis binárias para HIIT (1) vs Recuperação (0)
        df_corr['hiit_ln'] = (df_corr['intens'] == 'HIIT').astype(int)
        
        # Juntar com RMSSD bruto
        df_corr = df_corr.merge(
            df_hg_raw[['Data', 'intens_raw']], 
            on='Data', 
            how='left'
        )
        df_corr['hiit_raw'] = (df_corr['intens_raw'] == 'HIIT').astype(int)
        
        # Criar variáveis binárias para cada zona Mode 1 e Mode 2
        mode1_zones = ['Accumulated_Fatigue', 'Maladaptation', 'Good_Adaptation', 'High_Variability', 'Normal']
        mode2_zones = ['NFOR', 'Overreaching', 'High_Variability', 'Normal']
        
        for zone in mode1_zones:
            df_corr[f'm1_{zone}'] = (df_corr['mode1_zone'] == zone).astype(int)
        
        for zone in mode2_zones:
            df_corr[f'm2_{zone}'] = (df_corr['mode2_zone'] == zone).astype(int)
        
        # Calcular rolling de 14 dias para todas as variáveis
        rolling_cols = ['hiit_ln', 'hiit_raw'] + [f'm1_{z}' for z in mode1_zones] + [f'm2_{z}' for z in mode2_zones]
        for col in rolling_cols:
            df_corr[f'{col}_r14'] = df_corr[col].rolling(14, min_periods=7).mean()
        
        # Calcular correlações
        corr_results = []
        
        # Correlações: HIIT LnRMSSD rolling vs Modos rolling
        for mode in ['m1', 'm2']:
            zones = mode1_zones if mode == 'm1' else mode2_zones
            mode_name = "Mode 1 (Altini)" if mode == 'm1' else "Mode 2 (Plews)"
            
            for zone in zones:
                col_hiit = 'hiit_ln_r14'
                col_zone = f'{mode}_{zone}_r14'
                
                valid_data = df_corr[[col_hiit, col_zone]].dropna()
                if len(valid_data) > 10:
                    corr, p_val = stats.pearsonr(valid_data[col_hiit], valid_data[col_zone])
                    
                    # Interpretação da correlação
                    if abs(corr) >= 0.7:
                        strength = "Forte"
                    elif abs(corr) >= 0.4:
                        strength = "Moderada"
                    elif abs(corr) >= 0.2:
                        strength = "Fraca"
                    else:
                        strength = "Desprezível"
                    
                    direction = "Positiva" if corr > 0 else "Negativa"
                    
                    corr_results.append({
                        'Métrica HIIT': 'HIIT LnRMSSD (14d)',
                        'Modo': mode_name,
                        'Evento': zone.replace('_', ' '),
                        'Correlação': f"{corr:.3f}",
                        'Direção': direction,
                        'Força': strength,
                        'P-valor': f"{p_val:.4f}",
                        'Significativo': "Sim" if p_val < 0.05 else "Não"
                    })
        
        # Correlações: HIIT RMSSD bruto rolling vs Modos rolling  
        for mode in ['m1', 'm2']:
            zones = mode1_zones if mode == 'm1' else mode2_zones
            mode_name = "Mode 1 (Altini)" if mode == 'm1' else "Mode 2 (Plews)"
            
            for zone in zones:
                col_hiit = 'hiit_raw_r14'
                col_zone = f'{mode}_{zone}_r14'
                
                valid_data = df_corr[[col_hiit, col_zone]].dropna()
                if len(valid_data) > 10:
                    corr, p_val = stats.pearsonr(valid_data[col_hiit], valid_data[col_zone])
                    
                    if abs(corr) >= 0.7:
                        strength = "Forte"
                    elif abs(corr) >= 0.4:
                        strength = "Moderada"
                    elif abs(corr) >= 0.2:
                        strength = "Fraca"
                    else:
                        strength = "Desprezível"
                    
                    direction = "Positiva" if corr > 0 else "Negativa"
                    
                    corr_results.append({
                        'Métrica HIIT': 'HIIT RMSSD Bruto (14d)',
                        'Modo': mode_name,
                        'Evento': zone.replace('_', ' '),
                        'Correlação': f"{corr:.3f}",
                        'Direção': direction,
                        'Força': strength,
                        'P-valor': f"{p_val:.4f}",
                        'Significativo': "Sim" if p_val < 0.05 else "Não"
                    })
        
        # Exibir resultados em tabela
        if corr_results:
            df_corr_display = pd.DataFrame(corr_results)
            
            # Filtrar apenas correlações Fortes ou Moderadas para destaque
            df_signif = df_corr_display[df_corr_display['Força'].isin(['Forte', 'Moderada'])]
            
            if len(df_signif) > 0:
                st.markdown("**Correlações Fortes e Moderadas detectadas:**")
                st.dataframe(
                    df_signif.style.apply(
                        lambda x: ['background-color: rgba(39, 174, 96, 0.2)' if x['Força'] == 'Forte' 
                                  else 'background-color: rgba(241, 196, 15, 0.2)' for _ in x],
                        axis=1
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            
            with st.expander("Ver todas as correlações calculadas"):
                st.dataframe(df_corr_display, use_container_width=True, hide_index=True)
            
            # Explicação dos resultados
            st.markdown("""
            **Como interpretar:**
            - **Correlação Positiva**: Quando a frequência de HIIT aumenta, o evento do Mode também tende a aumentar (aparecer mais)
            - **Correlação Negativa**: Quando a frequência de HIIT aumenta, o evento do Mode tende a diminuir (aparecer menos)
            - **Força Forte (≥0.7)**: Relação muito consistente, útil para previsão
            - **Força Moderada (0.4-0.7)**: Relação consistente, padrão claro observado
            - **P-valor < 0.05**: Correlação estatisticamente significativa (não é ao acaso)
            """)
            
            # Análise comparativa LnRMSSD vs RMSSD Bruto
            st.markdown("**Comparação: LnRMSSD vs RMSSD Bruto**")
            
            comp_data = []
            for mode in ['Mode 1 (Altini)', 'Mode 2 (Plews)']:
                ln_data = df_corr_display[(df_corr_display['Métrica HIIT'] == 'HIIT LnRMSSD (14d)') & 
                                         (df_corr_display['Modo'] == mode)]
                raw_data = df_corr_display[(df_corr_display['Métrica HIIT'] == 'HIIT RMSSD Bruto (14d)') & 
                                          (df_corr_display['Modo'] == mode)]
                
                # Calcular média das correlações absolutas por modo
                ln_mean = ln_data['Correlação'].astype(float).abs().mean()
                raw_mean = raw_data['Correlação'].astype(float).abs().mean()
                
                melhor = "LnRMSSD" if ln_mean > raw_mean else "RMSSD Bruto"
                
                comp_data.append({
                    'Modo': mode,
                    'Média |Correlação| LnRMSSD': f"{ln_mean:.3f}",
                    'Média |Correlação| RMSSD': f"{raw_mean:.3f}",
                    'Melhor preditor': melhor
                })
            
            st.table(pd.DataFrame(comp_data))
            
        else:
            st.info("Dados insuficientes para calcular correlações (mínimo 10 observações válidas).")

    else:
        st.info("Mínimo 14 dias de HRV para HRV-Guided Training.")

    # ════════════════════════════════════════════════════════════════════════
    # BPE — Z-Score Semanal
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 BPE — Z-Score Semanal (Método SWC)")

    mets_bpe = [m for m in ['hrv','rhr','sleep_quality','fatiga','stress']
                if m in dw.columns and dw[m].notna().any()]
    n_semanas_disp = max(1, len(dw) // 7)
    if n_semanas_disp < 4:
        st.info(f"Dados insuficientes para BPE (min 4 semanas, disponível: {n_semanas_disp}).")
    else:
        _slider_max = min(52, n_semanas_disp)
        _slider_val = min(16, _slider_max)
        n_sem = (st.slider("Semanas (BPE)", 4, _slider_max, _slider_val, key="bpe_sem")
                 if _slider_max > 4 else _slider_max)
        dados_bpe = {}
        for met in mets_bpe:
            s = calcular_bpe(dw, met, 60)
            if len(s) > 0:
                dados_bpe[met] = s.tail(n_sem)
        if dados_bpe:
            semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
            nm  = len(dados_bpe)
            mat = np.zeros((nm, len(semanas)))
            nomes_bpe = {'hrv':'HRV','rhr':'RHR (inv)','sleep_quality':'Sono',
                         'fatiga':'Energia','stress':'Relaxamento'}
            for i, met in enumerate(dados_bpe.keys()):
                z = dados_bpe[met]['zscore'].values
                mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
            import numpy as _npIM
            _zIM = [[float(mat[r][c]) if not _npIM.isnan(mat[r][c]) else None
                     for c in range(mat.shape[1])] for r in range(mat.shape[0])]
            _yIM = [nomes_bpe.get(m, m) for m in dados_bpe.keys()]
            _fIM = go.Figure(go.Heatmap(
                z=_zIM, x=[str(s) for s in semanas], y=_yIM,
                colorscale='RdYlGn', zmid=0, zmin=-2, zmax=2,
                colorbar=dict(title='Z', tickfont=dict(color='#111'))))
            _fIM.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111'), margin=dict(t=50, b=70, l=55, r=20),
                height=max(280, len(_yIM)*35),
                title=dict(text='BPE — Z-Score com SWC', font=dict(size=13, color='#111')),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='#111')),
                yaxis=dict(tickfont=dict(size=9, color='#111')))
            st.plotly_chart(_fIM, use_container_width=True,
                            config={'displayModeBar': False, 'responsive': True,
                                    'scrollZoom': False},
                            key="rec_bpe_chart")
