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
        # ANÁLISE 1: HIIT LnRMSSD vs Recovery Modes (Rolling 14d)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Análise 1: HIIT HRV-Guided vs Recovery Modes (14d)")
        
        # Preparar dados
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
        
        # Criar variável binária para HIIT do LnRMSSD (1 = HIIT, 0 = Recuperação)
        df_corr['hiit_ln'] = (df_corr['intens'] == 'HIIT').astype(int)
        
        # Criar variáveis binárias para cada zona Mode 1 e Mode 2
        mode1_zones = ['Accumulated_Fatigue', 'Maladaptation', 'Good_Adaptation', 'High_Variability', 'Normal']
        mode2_zones = ['NFOR', 'Overreaching', 'High_Variability', 'Normal']
        
        for zone in mode1_zones:
            df_corr[f'm1_{zone}'] = (df_corr['mode1_zone'] == zone).astype(int)
        
        for zone in mode2_zones:
            df_corr[f'm2_{zone}'] = (df_corr['mode2_zone'] == zone).astype(int)
        
        # Calcular rolling de 14 dias para HIIT e para os eventos
        rolling_cols = ['hiit_ln'] + [f'm1_{z}' for z in mode1_zones] + [f'm2_{z}' for z in mode2_zones]
        for col in rolling_cols:
            df_corr[f'{col}_r14'] = df_corr[col].rolling(14, min_periods=7).mean()
        
        # Calcular correlações: HIIT LnRMSSD rolling vs Modos rolling
        corr_results = []
        
        for mode, zones in [('m1', mode1_zones), ('m2', mode2_zones)]:
            mode_name = "Mode 1 (Altini)" if mode == 'm1' else "Mode 2 (Plews)"
            
            for zone in zones:
                col_hiit = 'hiit_ln_r14'
                col_zone = f'{mode}_{zone}_r14'
                
                valid_data = df_corr[[col_hiit, col_zone]].dropna()
                if len(valid_data) > 10:
                    corr, p_val = stats.pearsonr(valid_data[col_hiit], valid_data[col_zone])
                    
                    # Interpretação
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
                        'Modo': mode_name,
                        'Evento': zone.replace('_', ' '),
                        'Correlação': corr,
                        'Direção': direction,
                        'Força': strength,
                        'P-valor': p_val,
                        'Significativo': "Sim" if p_val < 0.05 else "Não",
                        'N': len(valid_data)
                    })
        
        if corr_results:
            df_corr_display = pd.DataFrame(corr_results)
            
            # Separar por modo
            for mode_name in ["Mode 1 (Altini)", "Mode 2 (Plews)"]:
                st.markdown(f"**{mode_name}**")
                df_mode = df_corr_display[df_corr_display['Modo'] == mode_name].copy()
                
                # Destacar fortes e moderadas
                def color_forca(val):
                    if val == "Forte":
                        return "background-color: rgba(231, 76, 60, 0.3); color: #c0392b; font-weight: bold"
                    elif val == "Moderada":
                        return "background-color: rgba(241, 196, 15, 0.3)"
                    return ""
                
                # Aplicar apenas na coluna Força
                def aplicar_cores(df):
                    cols = [''] * len(df.columns)
                    força_idx = df.columns.get_loc('Força')
                    cols[força_idx] = 'background-color: rgba(241, 196, 15, 0.3)'  # default
                    return cols
                
                st.dataframe(
                    df_mode[['Evento', 'Correlação', 'Direção', 'Força', 'P-valor', 'Significativo']]
                    .style.map(color_forca, subset=['Força'])
                    .format({'Correlação': '{:.3f}', 'P-valor': '{:.4f}'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            with st.expander("ℹ️ Como interpretar estas correlações"):
                st.markdown("""
                Estas correlações mostram a relação entre **frequência de dias HIIT** (rolling 14d) e **frequência de eventos de recuperação** (rolling 14d).
                
                - **Positiva**: Mais dias HIIT ↔ Mais eventos deste tipo
                - **Negativa**: Mais dias HIIT ↔ Menos eventos deste tipo  
                - **Forte (≥0.7)**: Relação muito consistente
                - **Moderada (0.4-0.7)**: Relação clara e útil
                - **Significativo (p<0.05)**: A correlação não é ao acaso
                
                **Exemplo prático**: Se "Accumulated Fatigue" tem correlação negativa forte com HIIT, significa que quando você faz mais HIIT, tende a ter menos fadiga acumulada (bom sinal!), ou vice-versa.
                """)
        else:
            st.info("Dados insuficientes para calcular correlações.")

                # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 1.5: Lag Analysis - Treino Pesado (RPE≥7) → Fadiga Futura
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("⏱️ Análise Temporal: Treino Pesado (ICU_RPE ≥7) → Fadiga Futura")
        
        if 'icu_rpe' in df_corr.columns and df_corr['icu_rpe'].notna().any():
            
            # Criar indicador de treino pesado (RPE >= 7)
            df_corr['treino_pesado'] = (df_corr['icu_rpe'] >= 7).astype(int)
            
            # Definir lags para testar
            lags_rpe = [1, 2, 3, 5, 7, 10, 14]
            resultados_lag_rpe = []
            
            # Eventos de fadiga para monitorar
            eventos_fadiga = [
                ('m1', 'Accumulated_Fatigue', 'Mode 1'),
                ('m1', 'Maladaptation', 'Mode 1'),
                ('m2', 'NFOR', 'Mode 2'),
                ('m2', 'Overreaching', 'Mode 2')
            ]
            
            for lag in lags_rpe:
                for mode_prefix, evento, mode_nome in eventos_fadiga:
                    col_evento = f'{mode_prefix}_{evento}'
                    col_futuro = f'{evento}_future_rpe'
                    
                    # Shift no evento de fadiga (para o futuro)
                    df_lag = df_corr[['Data', 'treino_pesado', col_evento]].copy()
                    df_lag[col_futuro] = df_lag[col_evento].shift(-lag)
                    
                    # Calcular correlação entre treino pesado hoje e fadiga no futuro
                    valid = df_lag[['treino_pesado', col_futuro]].dropna()
                    
                    if len(valid) > 5:  # Mínimo de observações
                        # Se poucas variações (raramente tem fadiga), correlação é instável
                        if valid[col_futuro].nunique() > 1:
                            corr, p_val = stats.pearsonr(valid['treino_pesado'], valid[col_futuro])
                            
                            resultados_lag_rpe.append({
                                'Lag (dias)': lag,
                                'Modo': mode_nome,
                                'Evento Futuro': evento.replace('_', ' '),
                                'Correlação': corr,
                                'P-valor': p_val,
                                'N': len(valid),
                                'Significativo': "Sim" if p_val < 0.05 else "Não",
                                'Risco': 'Alto' if corr > 0.3 and p_val < 0.05 else 'Moderado' if corr > 0.1 else 'Baixo'
                            })
            
            if resultados_lag_rpe:
                df_lag_rpe = pd.DataFrame(resultados_lag_rpe)
                
                # Mostrar resultados por lag
                for lag in lags_rpe:
                    df_lag_day = df_lag_rpe[df_lag_rpe['Lag (dias)'] == lag]
                    if not df_lag_day.empty:
                        with st.expander(f"Após {lag} dias do treino pesado"):
                            st.dataframe(
                                df_lag_day[['Modo', 'Evento Futuro', 'Correlação', 'Risco', 'P-valor', 'Significativo']]
                                .style.format({'Correlação': '{:.3f}', 'P-valor': '{:.4f}'})
                                .apply(lambda x: ['background-color: rgba(231, 76, 60, 0.3)' if x['Risco'] == 'Alto' 
                                                  else 'background-color: rgba(241, 196, 15, 0.3)' if x['Risco'] == 'Moderado' 
                                                  else '' for _ in x], axis=1),
                                use_container_width=True,
                                hide_index=True
                            )
                
                # Identificar o lag mais crítico
                df_alto_risco = df_lag_rpe[(df_lag_rpe['Risco'] == 'Alto') & (df_lag_rpe['Significativo'] == 'Sim')]
                
                if not df_alto_risco.empty:
                    # Pegar o de maior correlação
                    maior_risco = df_alto_risco.loc[df_alto_risco['Correlação'].idxmax()]
                    
                    st.error(f"""
                    🚨 **Padrão de Overreaching Detectado!**
                    
                    Treinos com **ICU_RPE ≥ 7** têm correlação de **{maior_risco['Correlação']:.3f}** com **{maior_risco['Evento Futuro']}** em **{maior_risco['Lag (dias)']} dias**.
                    
                    **Interpretação:**
                    - Quando você treina pesado (RPE≥7), existe {maior_risco['Correlação']*100:.0f}% de chance de estar em estado de {maior_risco['Evento Futuro'].lower()} {maior_risco['Lag (dias)']} dias depois.
                    - Este é um padrão de **acúmulo de fadiga não recuperável**.
                    
                    **Recomendações:**
                    1. Após treinos RPE≥7, garantir **{maior_risco['Lag (dias)']+1} dias** de recuperação antes do próximo treino pesado
                    2. Ou reduzir intensidade do próximo treino para RPE < 7 se ainda dentro do período de {maior_risco['Lag (dias)']} dias
                    3. Considerar periodização: treino pesado → {maior_risco['Lag (dias)']} dias leves → próximo pesado
                    """)
                    
                    # Gráfico de heatmap dos lags
                    st.markdown("**Mapa de Calor - Correlação por Tempo**")
                    
                    # Pivot para heatmap
                    df_pivot = df_lag_rpe.pivot_table(
                        index=['Modo', 'Evento Futuro'], 
                        columns='Lag (dias)', 
                        values='Correlação',
                        aggfunc='mean'
                    ).fillna(0)
                    
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=df_pivot.values,
                        x=df_pivot.columns,
                        y=[f"{idx[0]} - {idx[1]}" for idx in df_pivot.index],
                        colorscale='RdYlGn',
                        zmid=0,
                        zmin=-0.5,
                        zmax=0.5,
                        colorbar=dict(title='Correlação')
                    ))
                    
                    fig_heat.update_layout(
                        title="Correlação: Treino Pesado Hoje → Fadiga no Futuro",
                        xaxis_title="Dias após treino",
                        yaxis_title="Evento de Fadiga",
                        height=300
                    )
                    
                    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
                    
                else:
                    st.success("""
                    ✅ **Boa recuperação entre treinos pesados!**
                    
                    Não foi detectada correlação significativa entre treinos RPE≥7 e estados de fadiga futuros.
                    Seu sistema está recuperando adequadamente entre sessões de alta carga.
                    """)
            else:
                st.info("Dados insuficientes para análise temporal (mínimo 6 observações por período).")
        else:
            st.info("Coluna 'icu_rpe' não encontrada nos dados.")

                # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 1.7: Slope 7d Individualizado com Análise de Persistência
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📉 Slope 7d Individualizado + Persistência do Estado")
        
        # Calcular estatísticas do slope usando TODO o histórico disponível
        slope_series = df_corr['slope_7'].dropna()
        
        if len(slope_series) > 20:
            slope_mean = slope_series.mean()
            slope_std = slope_series.std()
            swc = 0.5 * slope_std
            
            # Thresholds individualizados
            thresh_recuperacao = slope_mean + swc
            thresh_estavel_sup = slope_mean + (0.5 * swc)
            thresh_estavel_inf = slope_mean - (0.5 * swc)
            thresh_declinio = slope_mean - swc
            thresh_nfor = slope_mean - (2 * slope_std)
            
            # Classificar cada dia em uma zona
            def classificar_zona_slope(val):
                if pd.isna(val):
                    return 'Sem dados'
                elif val >= thresh_recuperacao:
                    return 'Supercompensação'
                elif val >= thresh_estavel_sup:
                    return 'Recuperação'
                elif val >= thresh_estavel_inf:
                    return 'Estável'
                elif val >= thresh_declinio:
                    return 'Declínio Leve'
                elif val >= thresh_nfor:
                    return 'Fadiga'
                else:
                    return 'NFOR Crítico'
            
            df_corr['zona_slope'] = df_corr['slope_7'].apply(classificar_zona_slope)
            
            # Calcular dias consecutivos (streak) na zona atual
            zona_atual = df_corr['zona_slope'].iloc[-1]
            dias_na_zona = 0
            
            # Contar quantos dias seguidos estamos na mesma zona (de trás pra frente)
            for i in range(len(df_corr) - 1, -1, -1):
                if df_corr['zona_slope'].iloc[i] == zona_atual:
                    dias_na_zona += 1
                else:
                    break
            
            # Calcular histórico de streaks (para contextualizar)
            todas_sequencias = []
            seq_atual = 1
            zona_anterior = df_corr['zona_slope'].iloc[0] if len(df_corr) > 0 else None
            
            for i in range(1, len(df_corr)):
                if df_corr['zona_slope'].iloc[i] == zona_anterior:
                    seq_atual += 1
                else:
                    if zona_anterior != 'Sem dados':
                        todas_sequencias.append({'zona': zona_anterior, 'dias': seq_atual})
                    seq_atual = 1
                    zona_anterior = df_corr['zona_slope'].iloc[i]
            
            # Adicionar a última sequência
            if zona_anterior and zona_anterior != 'Sem dados':
                todas_sequencias.append({'zona': zona_anterior, 'dias': seq_atual})
            
            # Estatísticas de streaks por zona
            df_seq = pd.DataFrame(todas_sequencias)
            stats_seq = {}
            if not df_seq.empty:
                for zona in df_seq['zona'].unique():
                    dados_zona = df_seq[df_seq['zona'] == zona]['dias']
                    stats_seq[zona] = {
                        'media': dados_zona.mean(),
                        'max': dados_zona.max(),
                        'atual': dias_na_zona if zona == zona_atual else 0
                    }
            
            # Display métricas
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            with col_p1:
                st.metric("Zona Atual", zona_atual)
            with col_p2:
                st.metric("Dias nesta Zona", f"{dias_na_zona} dias",
                         delta=f"Seu recorde: {stats_seq.get(zona_atual, {}).get('max', 'N/A')} dias" if zona_atual in stats_seq else None)
            with col_p3:
                st.metric("Slope Atual", f"{df_corr['slope_7'].iloc[-1]:.4f}")
            with col_p4:
                media_historica = stats_seq.get(zona_atual, {}).get('media', 0)
                if media_historica > 0:
                    razao = dias_na_zona / media_historica
                    st.metric("vs Sua Média", f"{razao:.1f}x",
                             help=f"Você está nesta zona há {razao:.1f} vezes a sua média histórica ({media_historica:.1f} dias)")
            
            # Recomendação baseada em PERSISTÊNCIA (não apenas valor atual)
            st.markdown("---")
            st.markdown("**🎯 Recomendação Baseada na Persistência do Estado**")
            
            if zona_atual == 'NFOR Crítico':
                if dias_na_zona >= 3:
                    st.error(f"""
                    🚨 **AÇÃO IMEDIATA NECESSÁRIA**
                    
                    Você está em NFOR há **{dias_na_zona} dias consecutivos**.
                    
                    **Histórico:** Seu recorde foi {stats_seq[zona_atual]['max']} dias nesta zona.
                    **Risco:** Extremo - overtraining estabelecido.
                    
                    **Protocolo obrigatório:**
                    - Descanso COMPLETO (sem treino) por pelo menos {dias_na_zona + 2} dias
                    - Só retornar quando slope > {thresh_declinio:.4f} por 2 dias seguidos
                    - Considerar consulta médica se sintomas físicos presentes
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **NFOR Detectado ({dias_na_zona} dias)**
                    
                    Primeiros sinais de fadiga severa. 
                    **Ação:** Reduzir carga em 50% imediatamente. Se persistir por mais 2 dias, descanso total.
                    """)
            
            elif zona_atual == 'Fadiga':
                if dias_na_zona >= 5:
                    st.error(f"""
                    🚨 **Fadiga Persistente ({dias_na_zona} dias)**
                    
                    Você excedeu sua média histórica de permanência nesta zona ({stats_seq[zona_atual]['media']:.1f} dias).
                    
                    **Risco:** Transição para NFOR em {7 - dias_na_zona} dias se mantiver carga atual.
                    
                    **Ação:** Descanso ativo (caminhada, yoga) por 3-5 dias até retornar à zona "Estável".
                    """)
                elif dias_na_zona >= 3:
                    st.warning(f"""
                    ⚠️ **Declínio Prolongado ({dias_na_zona} dias)**
                    
                    Carga de treino está superando capacidade de recuperação.
                    **Ação:** Reduzir intensidade em 30% até normalização (1-2 dias).
                    """)
                else:
                    st.info(f"""
                    ℹ️ **Fadiga Aguda ({dias_na_zona} dia{'s' if dias_na_zona > 1 else ''})**
                    
                    Normal após treino intenso. Monitorar próximas 48h.
                    """)
            
            elif zona_atual == 'Declínio Leve':
                if dias_na_zona >= 7:
                    st.warning(f"""
                    ⚠️ **Declínio Crônico Leve ({dias_na_zona} dias)**
                    
                    Padrão de overreaching funcional. Não é crítico, mas performance está comprometida.
                    **Ação:** 2-3 dias de recuperação ativa antes de novo ciclo de carga.
                    """)
                else:
                    st.success(f"""
                    ✅ **Estável ({dias_na_zona} dias)**
                    
                    Dentro da variação normal. Pode manter carga atual.
                    """)
            
            elif zona_atual in ['Recuperação', 'Supercompensação']:
                if dias_na_zona >= 2:
                    st.success(f"""
                    🚀 **Pronto para Carga! ({dias_na_zona} dias)**
                    
                    Período ótimo para treinos de alta intensidade ou competição.
                    Aproveite esta janela - seucorpo está supercompensando.
                    """)
                else:
                    st.info(f"""
                    ℹ️ **Início de Recuperação**
                    
                    Aguarde mais 1 dia na zona para garantir estabilidade antes de carga alta.
                    """)
            
            # Gráfico com legendas em PRETO e anotações de dias consecutivos
            fig_slope_ind = go.Figure()
            
            # Linha do slope
            fig_slope_ind.add_trace(go.Scatter(
                x=df_corr['Data'],
                y=df_corr['slope_7'],
                name='Slope 7d',
                line=dict(color='#2c3e50', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 62, 80, 0.1)',
                hovertemplate='Data: %{x}<br>Slope: %{y:.4f}<extra></extra>'
            ))
            
            # Adicionar anotações de streaks (sequências) visíveis
            if len(todas_sequencias) > 0:
                # Mostrar apenas as últimas 3 sequências para não poluir
                for seq in todas_sequencias[-3:]:
                    # Encontrar posição no tempo (aproximada)
                    idx_fim = len(df_corr) - 1 - (0 if seq == todas_sequencias[-1] else sum([s['dias'] for s in todas_sequencias[todas_sequencias.index(seq)+1:]]))
                    idx_inicio = max(0, idx_fim - seq['dias'] + 1)
                    
                    if idx_fim < len(df_corr) and idx_inicio < len(df_corr):
                        data_meio = df_corr['Data'].iloc[(idx_inicio + idx_fim) // 2]
                        valor_y = df_corr['slope_7'].iloc[idx_inicio:idx_fim+1].mean()
                        
                        cor_anot = 'red' if seq['zona'] in ['NFOR Crítico', 'Fadiga'] else \
                                  'orange' if seq['zona'] == 'Declínio Leve' else \
                                  'green' if seq['zona'] in ['Recuperação', 'Supercompensação'] else 'gray'
                        
                        fig_slope_ind.add_annotation(
                            x=data_meio,
                            y=valor_y,
                            text=f"{seq['dias']}d",
                            showarrow=False,
                            font=dict(size=10, color=cor_anot, family="Arial Black"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor=cor_anot,
                            borderwidth=1,
                            borderpad=2
                        )
            
            # Bandas de referência individualizadas
            fig_slope_ind.add_hrect(y0=thresh_recuperacao, y1=slope_mean + (3*slope_std), 
                                   fillcolor="rgba(39, 174, 96, 0.15)", 
                                   line_width=0, 
                                   annotation_text="Supercompensação",
                                   annotation_position="top right",
                                   annotation_font=dict(color='black', size=11))
            
            fig_slope_ind.add_hrect(y0=thresh_estavel_sup, y1=thresh_recuperacao,
                                   fillcolor="rgba(46, 204, 113, 0.15)",
                                   line_width=0,
                                   annotation_text="Recuperação",
                                   annotation_position="right",
                                   annotation_font=dict(color='black', size=11))
            
            fig_slope_ind.add_hrect(y0=thresh_estavel_inf, y1=thresh_estavel_sup,
                                   fillcolor="rgba(149, 165, 166, 0.15)",
                                   line_width=0,
                                   annotation_text="Estável",
                                   annotation_position="right",
                                   annotation_font=dict(color='black', size=11))
            
            fig_slope_ind.add_hrect(y0=thresh_declinio, y1=thresh_estavel_inf,
                                   fillcolor="rgba(241, 196, 15, 0.15)",
                                   line_width=0,
                                   annotation_text="Declínio Leve",
                                   annotation_position="right",
                                   annotation_font=dict(color='black', size=11))
            
            fig_slope_ind.add_hrect(y0=thresh_nfor, y1=thresh_declinio,
                                   fillcolor="rgba(231, 76, 60, 0.15)",
                                   line_width=0,
                                   annotation_text="Fadiga",
                                   annotation_position="right",
                                   annotation_font=dict(color='black', size=11))
            
            fig_slope_ind.add_hrect(y0=slope_mean - (4*slope_std), y1=thresh_nfor,
                                   fillcolor="rgba(192, 57, 43, 0.2)",
                                   line_width=0,
                                   annotation_text="NFOR",
                                   annotation_position="bottom right",
                                   annotation_font=dict(color='black', size=11))
            
            # Linhas de threshold com labels em preto
            fig_slope_ind.add_hline(y=slope_mean, line_dash="solid", line_color="blue", line_width=2,
                                   annotation_text=f"Sua Média ({slope_mean:.3f})",
                                   annotation_position="right",
                                   annotation_font=dict(color='black', size=10, family='Arial'))
            
            fig_slope_ind.add_hline(y=thresh_nfor, line_dash="dash", line_color="red", line_width=3,
                                   annotation_text=f"Limite NFOR ({thresh_nfor:.3f})",
                                   annotation_position="bottom right",
                                   annotation_font=dict(color='black', size=10, family='Arial'))
            
            # Layout com legendas e textos em PRETO
            fig_slope_ind.update_layout(
                title=dict(
                    text=f"Slope 7d Individualizado - Baseado em {len(slope_series)} dias",
                    font=dict(color='black', size=14)
                ),
                xaxis=dict(
                    title=dict(text="Data", font=dict(color='black')),
                    tickfont=dict(color='black'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    title=dict(text="Slope 7d do LnRMSSD", font=dict(color='black')),
                    tickfont=dict(color='black'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=500,
                showlegend=True,
                legend=dict(
                    font=dict(color='black'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1
                ),
                font=dict(color='black')  # Fonte geral preta
            )
            
            st.plotly_chart(fig_slope_ind, use_container_width=True, config={'displayModeBar': False})
            
            # Tabela de histórico de sequências
            with st.expander("📋 Ver Histórico de Sequências (Streaks)"):
                df_hist_seq = pd.DataFrame(todas_sequencias[-10:])  # Últimas 10 sequências
                df_hist_seq['Data Fim'] = [df_corr['Data'].iloc[-(sum([s['dias'] for s in todas_sequencias[i:]]) if i < len(todas_sequencias)-1 else 0) - 1].strftime('%d/%m/%Y') 
                                          for i in range(len(todas_sequencias)-10, len(todas_sequencias))]
                st.dataframe(df_hist_seq[['Data Fim', 'zona', 'dias']].rename(columns={'zona': 'Zona', 'dias': 'Duração (dias)'}),
                            use_container_width=True, hide_index=True)
                
        else:
            st.info(f"Dados insuficientes ({len(slope_series)} dias, mínimo 20) para análise individualizada.")
        # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 1.6: Correlação Temporal com Lag (HIIT prediz fadiga futura?)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("⏱️ Análise Temporal: HIIT hoje → Fadiga futura? (Lag Analysis)")
        
        st.markdown("""
        Esta análise verifica se dias de **HIIT sugerido pelo HRV-Guided** predizem estados de fadiga (NFOR, Overreaching, Accumulated Fatigue) após X dias.
        """)
        
        # Definir lags para testar
        lags = [1, 3, 5, 7, 14]
        resultados_lag = []
        
        for lag in lags:
            # Criar versões deslocadas dos eventos de fadiga
            for mode, zones, mode_name in [
                ('m1', ['Accumulated_Fatigue', 'Maladaptation'], 'Mode 1 (Fadiga)'),
                ('m2', ['NFOR', 'Overreaching'], 'Mode 2 (Fadiga Severa)')
            ]:
                for zone in zones:
                    # Shift nos dados do evento (futuro) vs HIIT atual
                    df_lag = df_corr[['Data', 'hiit_ln', f'{mode}_{zone}']].copy()
                    df_lag[f'{zone}_future'] = df_lag[f'{mode}_{zone}'].shift(-lag)  # Evento no futuro
                    
                    # Calcular correlação
                    valid = df_lag[['hiit_ln', f'{zone}_future']].dropna()
                    if len(valid) > 10:
                        corr, p_val = stats.pearsonr(valid['hiit_ln'], valid[f'{zone}_future'])
                        
                        resultados_lag.append({
                            'Lag (dias)': lag,
                            'Modo': mode_name,
                            'Evento Futuro': zone.replace('_', ' '),
                            'Correlação': corr,
                            'P-valor': p_val,
                            'Significativo': "Sim" if p_val < 0.05 else "Não",
                            'Interpretação': "Positiva" if corr > 0 else "Negativa"
                        })
        
        if resultados_lag:
            df_lag_results = pd.DataFrame(resultados_lag)
            
            # Mostrar resultados em tabela
            st.markdown("**Correlações por período de latência:**")
            
            for lag in lags:
                with st.expander(f"Lag de {lag} dias"):
                    df_lag_day = df_lag_results[df_lag_results['Lag (dias)'] == lag]
                    if not df_lag_day.empty:
                        st.dataframe(
                            df_lag_day[['Modo', 'Evento Futuro', 'Correlação', 'Interpretação', 'P-valor', 'Significativo']]
                            .style.format({'Correlação': '{:.3f}', 'P-valor': '{:.4f}'})
                            .apply(lambda x: ['background-color: rgba(231, 76, 60, 0.3)' if x['Correlação'] > 0.3 and x['Significativo'] == 'Sim' else '' for _ in x], axis=1),
                            use_container_width=True,
                            hide_index=True
                        )
            
            # Identificar o lag mais preditivo
            df_signif = df_lag_results[df_lag_results['Significativo'] == 'Sim']
            if not df_signif.empty:
                maior_corr = df_signif.loc[df_signif['Correlação'].idxmax()]
                
                st.markdown("---")
                st.markdown("**🔍 Insight Principal:**")
                
                if maior_corr['Correlação'] > 0.3:
                    st.warning(f"""
                    **Preditor encontrado!**
                    
                    Fazer **HIIT hoje** tem correlação de **{maior_corr['Correlação']:.3f}** com aparecer **{maior_corr['Evento Futuro']}** em **{maior_corr['Lag (dias)']} dias**.
                    
                    **Interpretação:** O HRV-Guided está sugerindo HIIT em dias que, {maior_corr['Lag (dias)']} dias depois, resultam em estado de fadiga ({maior_corr['Evento Futuro']}).
                    
                    **Possíveis causas:**
                    1. O baseline de {dias_fam}d do HRV-Guided está muito curto para captar a tendência de {maior_corr['Lag (dias)']}d
                    2. Você pode estar fazendo HIIT em dias que parecem "bons" no curto prazo, mas que na verdade estavam em início de acúmulo de fadiga
                    3. O HRV leva {maior_corr['Lag (dias)']} dias para refletir o estresse do HIIT neste caso específico
                    
                    **Recomendação:** Ajuste o HRV-Guided para usar baseline de pelo menos {maior_corr['Lag (dias)']}d quando este padrão aparecer.
                    """)
                else:
                    st.info("""
                    **Nenhum preditor forte encontrado.**
                    
                    As correlações temporais são fracas, sugerindo que o HRV-Guided está bem calibrado:
                    - HIIT sugerido não está sistematicamente predizendo fadiga futura
                    - A resposta ao treino está dentro do esperado
                    """)
        else:
            st.info("Dados insuficientes para análise temporal (mínimo 10 observações por lag).")

        # ════════════════════════════════════════════════════════════════════════
        # ANÁLISE 2: Comparação LnRMSSD vs RMSSD Bruto (HRV-Guided)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Análise 2: LnRMSSD vs RMSSD Bruto como HRV-Guided")
        
        # Preparar dados de RMSSD bruto
        df_raw = df_hg_raw.copy()
        df_raw['hiit_raw'] = (df_raw['intens_raw'] == 'HIIT').astype(int)
        
        # Juntar com dados LnRMSSD
        df_comp = df_corr[['Data', 'hiit_ln', 'intens']].merge(
            df_raw[['Data', 'hiit_raw', 'intens_raw', 'RMSSD', 'bm_raw', 'bs_raw']], 
            on='Data', 
            how='inner'
        )
        
        # Métricas de concordância
        total_dias = len(df_comp)
        ambos_hiit = ((df_comp['hiit_ln'] == 1) & (df_comp['hiit_raw'] == 1)).sum()
        ambos_rec = ((df_comp['hiit_ln'] == 0) & (df_comp['hiit_raw'] == 0)).sum()
        ln_hiit_raw_rec = ((df_comp['hiit_ln'] == 1) & (df_comp['hiit_raw'] == 0)).sum()
        ln_rec_raw_hiit = ((df_comp['hiit_ln'] == 0) & (df_comp['hiit_raw'] == 1)).sum()
        
        # Concordância geral (acurácia)
        concordancia = (ambos_hiit + ambos_rec) / total_dias * 100 if total_dias > 0 else 0
        
        # Concordância específica em HIIT (sensibilidade do LnRMSSD vs RMSSD)
        total_hiit_ln = df_comp['hiit_ln'].sum()
        total_hiit_raw = df_comp['hiit_raw'].sum()
        
        # Cohen's Kappa para concordância ajustada ao acaso
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(df_comp['hiit_ln'], df_comp['hiit_raw'])
        
        # Interpretação Kappa
        if kappa >= 0.8:
            kappa_interp = "Quase perfeita"
        elif kappa >= 0.6:
            kappa_interp = "Substancial"
        elif kappa >= 0.4:
            kappa_interp = "Moderada"
        elif kappa >= 0.2:
            kappa_interp = "Fraca"
        else:
            kappa_interp = "Mínima"
        
        # Display métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Concordância Geral", f"{concordancia:.1f}%", 
                 help="Percentual de dias em que ambos os métodos deram a mesma prescrição (HIIT ou Recuperação)")
        c2.metric("Cohen's Kappa", f"{kappa:.3f}", 
                 delta=kappa_interp,
                 help="Concordância ajustada ao acaso. >0.6 é considerada boa.")
        c3.metric("Dias HIIT LnRMSSD", f"{total_hiit_ln}", 
                 help=f"Dias prescritos como HIIT pelo método LnRMSSD ({total_hiit_ln/total_dias*100:.0f}%)")
        c4.metric("Dias HIIT RMSSD", f"{total_hiit_raw}", 
                 help=f"Dias prescritos como HIIT pelo método RMSSD Bruto ({total_hiit_raw/total_dias*100:.0f}%)")
        
        # Tabela de contingência
        st.markdown("**Tabela de Contingência**")
        cont_table = pd.DataFrame({
            'LnRMSSD \\ RMSSD Bruto': ['HIIT', 'Recuperação'],
            'HIIT': [ambos_hiit, ln_rec_raw_hiit],
            'Recuperação': [ln_hiit_raw_rec, ambos_rec]
        })
        st.table(cont_table.set_index('LnRMSSD \\ RMSSD Bruto'))
        
        # Análise de diferenças
        st.markdown("**Análise das Divergências**")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.metric("LnRMSSD diz HIIT mas RMSSD diz Rec", 
                     f"{ln_hiit_raw_rec} dias",
                     delta=f"{ln_hiit_raw_rec/total_dias*100:.1f}%",
                     delta_color="off")
            st.caption("LnRMSSD é mais 'agressivo' - permite HIIT quando RMSSD sugere descanso")
        
        with col_d2:
            st.metric("LnRMSSD diz Rec mas RMSSD diz HIIT", 
                     f"{ln_rec_raw_hiit} dias",
                     delta=f"{ln_rec_raw_hiit/total_dias*100:.1f}%",
                     delta_color="off")
            st.caption("RMSSD Bruto é mais 'agressivo' - permite HIIT quando LnRMSSD sugere descanso")
        
        # Recomendação
        st.markdown("---")
        st.markdown("**💡 Recomendação de Uso**")
        
        if kappa >= 0.6:
            st.success(f"""
            **Alta concordância ({kappa_interp})**: Ambos os métodos são equivalentes para este atleta.
            - LnRMSSD é preferível por normalizar a distribuição e reduzir efeito de outliers
            - RMSSD bruto pode ser mais intuitivo (ms reais)
            """)
        elif kappa >= 0.4:
            st.warning(f"""
            **Concordância moderada**: Os métodos diferem com frequência.
            - **LnRMSSD**: Mais conservador, tende a sugerir mais dias de recuperação ({ln_hiit_raw_rec} dias a mais que RMSSD)
            - **RMSSD Bruto**: Mais permissivo, pode subestimar necessidade de recuperação em alguns casos
            - Recomendação: Usar LnRMSSD para decisões importantes de treino
            """)
        else:
            st.error(f"""
            **Baixa concordância**: Os métodos divergem significativamente!
            - Isso pode indicar grande variabilidade nos dados de HRV ou presença de outliers
            - Verificar qualidade dos dados de medição
            - Considerar outros marcadores (sensação subjetiva, sono, etc.)
            """)
        
        
