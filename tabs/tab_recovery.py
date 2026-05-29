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


def tab_recovery(dw, da=None, wc_full=None, da_full=None):
    
    st.header("🔋 Recovery Score & HRV Analysis")
    
    # Garantir que da existe (mesmo que vazio)
    if da is None:
        da = pd.DataFrame()

    if len(dw) == 0 or 'hrv' not in dw.columns:
        st.warning("Sem dados de HRV.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # MODELO β — Paper: "VFC y Sistema Nervioso Autónomo" (Della Mattia, 2025)
    # Implementação: β (frescura actual), βAgudo (3d), βCrónico (7d)
    # Regra: só prescrever alta intensidade quando ≥2 de 3 indicadores convergem
    # Tratamento de NAs: dado ausente = INCERTEZA = prescrição conservadora
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🧠 Modelo β — Estado Autonómico Integrado")
    st.caption(
        "Baseado em Della Mattia (ednacore AI, 2025): *VFC y Sistema Nervioso Autónomo*. "
        "O SNA responde a factores invisíveis (glucogénio, osmolaridade, microinflamação) "
        "antes da percepção consciente. A HRV sozinha num único dia não é accionável — "
        "a tendência de 3 e 7 dias é."
    )

    def _calcular_modelo_beta(wc_src):
        """
        Calcula β, βAgudo e βCrónico a partir da série de HRV.

        Tratamento de NAs — regra do paper:
        - NaN em hrv hoje → β = NaN (não temos sinal, não podemos confirmar)
        - NaN em janela 3d → βAgudo = NaN (incerteza aguda)
        - NaN em janela 7d com <3 valores → βCrónico = NaN
        - NaN em qualquer componente → prescrição conservadora por defeito

        β é calculado como score 0-100 baseado em:
            - Posição do LnrMSSD vs baseline 28d (z-score normalizado)
            - Não usa o baseline da tab principal (que elimina NAs)
            - Usa reindex para manter dias calendário reais
        """
        import scipy.stats as _sst

        # Usar wc_full se disponível (histórico completo sem filtro sidebar)
        src = wc_src.copy() if wc_src is not None and len(wc_src) > 0 else dw.copy()
        src['Data'] = pd.to_datetime(src['Data'])
        src = src.sort_values('Data').set_index('Data')

        # CRÍTICO: reindex para datas calendário contínuas — preserva NAs reais
        # Em vez de dropna(), mantemos os NaN onde não houve medição
        date_range = pd.date_range(src.index.min(), src.index.max(), freq='D')
        src = src.reindex(date_range)

        if 'hrv' not in src.columns:
            return None

        # LnrMSSD — preserva NaN onde hrv é NaN ou 0
        src['LnrMSSD'] = np.where(
            src['hrv'].notna() & (src['hrv'] > 0),
            np.log(src['hrv']),
            np.nan
        )

        # Baseline 28d — rolling calendário, min_periods=7
        # NaNs são ignorados pelo rolling mas os dias calendário são preservados
        src['bm28'] = src['LnrMSSD'].rolling(28, min_periods=7).mean()
        src['bs28'] = src['LnrMSSD'].rolling(28, min_periods=7).std()

        # β — score 0-100 baseado em z-score vs baseline 28d
        # z = (hoje - baseline) / std_baseline → normalizado para 0-100
        # z=0 → β=50 | z=+2 → β≈95 | z=-2 → β≈5
        src['z28'] = (src['LnrMSSD'] - src['bm28']) / src['bs28'].replace(0, np.nan)
        src['beta'] = src['z28'].apply(
            lambda z: round(float(_sst.norm.cdf(z) * 100), 1) if pd.notna(z) else np.nan
        )

        # βAgudo — % mudança média 3d vs média 7d (janela calendário)
        # min_periods=2 para 3d, min_periods=4 para 7d
        m3  = src['LnrMSSD'].rolling(3,  min_periods=2).mean()
        m7  = src['LnrMSSD'].rolling(7,  min_periods=4).mean()
        src['beta_agudo'] = np.where(
            m7.notna() & m3.notna() & (m7 != 0),
            ((m3 - m7) / m7.abs()) * 100,
            np.nan
        )

        # βCrónico — % mudança média 7d vs média 28d (janela calendário)
        src['beta_cronico'] = np.where(
            src['bm28'].notna() & m7.notna() & (src['bm28'] != 0),
            ((m7 - src['bm28']) / src['bm28'].abs()) * 100,
            np.nan
        )

        return src[['LnrMSSD', 'bm28', 'bs28', 'beta', 'beta_agudo', 'beta_cronico']].tail(90)

    def _regra_convergencia(beta, b_agudo, b_cronico, hrv_hoje_notna):
        """
        Regra do paper: actuar só quando ≥2 de 3 indicadores convergem.
        NaN em qualquer indicador = incerteza = prescrição conservadora.

        Retorna: (prescricao, cor, n_sinais_pos, n_sinais_neg, n_incertos, detalhe)
        """
        sinais = []  # +1 positivo, -1 negativo, 0 incerto

        # Sinal 1: β actual
        if pd.isna(beta):
            sinais.append(('β actual', 0, 'NaN — sem medição hoje', '#888'))
        elif beta >= 60:
            sinais.append(('β actual', +1, f'{beta:.0f} ≥ 60 ✅', '#27ae60'))
        elif beta <= 40:
            sinais.append(('β actual', -1, f'{beta:.0f} ≤ 40 ⚠️', '#e74c3c'))
        else:
            sinais.append(('β actual', 0, f'{beta:.0f} zona neutra (40-60)', '#f39c12'))

        # Sinal 2: βAgudo (3d)
        if pd.isna(b_agudo):
            sinais.append(('βAgudo 3d', 0, 'NaN — dados insuficientes', '#888'))
        elif b_agudo >= 1.0:
            sinais.append(('βAgudo 3d', +1, f'{b_agudo:+.1f}% ≥ +1% ✅', '#27ae60'))
        elif b_agudo <= -1.0:
            sinais.append(('βAgudo 3d', -1, f'{b_agudo:+.1f}% ≤ -1% ⚠️', '#e74c3c'))
        else:
            sinais.append(('βAgudo 3d', 0, f'{b_agudo:+.1f}% zona neutra', '#f39c12'))

        # Sinal 3: βCrónico (7d)
        if pd.isna(b_cronico):
            sinais.append(('βCrónico 7d', 0, 'NaN — dados insuficientes', '#888'))
        elif b_cronico >= 1.0:
            sinais.append(('βCrónico 7d', +1, f'{b_cronico:+.1f}% ≥ +1% ✅', '#27ae60'))
        elif b_cronico <= -1.0:
            sinais.append(('βCrónico 7d', -1, f'{b_cronico:+.1f}% ≤ -1% ⚠️', '#e74c3c'))
        else:
            sinais.append(('βCrónico 7d', 0, f'{b_cronico:+.1f}% zona neutra', '#f39c12'))

        n_pos = sum(1 for _, s, _, _ in sinais if s == +1)
        n_neg = sum(1 for _, s, _, _ in sinais if s == -1)
        n_inc = sum(1 for _, s, _, _ in sinais if s == 0)

        # REGRA CRÍTICA: dado ausente hoje = não confirmar HIIT
        if not hrv_hoje_notna:
            prescricao = "⚠️ SEM MEDIÇÃO HOJE — Não prescrever HIIT"
            cor_pres   = "#e67e22"
            return prescricao, cor_pres, n_pos, n_neg, n_inc, sinais

        # Regra ≥2 convergem
        if n_pos >= 2:
            prescricao = "✅ HIIT / Alta intensidade — ≥2 sinais positivos"
            cor_pres   = "#27ae60"
        elif n_neg >= 2:
            prescricao = "🔴 Recuperação activa — ≥2 sinais negativos"
            cor_pres   = "#e74c3c"
        elif n_neg >= 1 and n_inc >= 1:
            prescricao = "🟠 Sessão moderada Z1/Z2 — 1 sinal negativo + incerteza"
            cor_pres   = "#e67e22"
        elif n_pos == 1 and n_inc >= 2:
            prescricao = "🟡 Sessão moderada Z1/Z2 — sinais insuficientes para HIIT"
            cor_pres   = "#f39c12"
        else:
            prescricao = "🟡 Zona neutra — manter intensidade planeada"
            cor_pres   = "#f39c12"

        return prescricao, cor_pres, n_pos, n_neg, n_inc, sinais

    # ── Calcular Modelo β ──────────────────────────────────────────────────
    beta_df = _calcular_modelo_beta(wc_full)

    if beta_df is None or beta_df.empty or beta_df['beta'].isna().all():
        st.info("Dados insuficientes para calcular Modelo β (mínimo 14 dias de HRV).")
    else:
        # Valores actuais (hoje = último registo calendário)
        ult = beta_df.iloc[-1]
        beta_hoje    = ult['beta']
        b_agudo_hoje = ult['beta_agudo']
        b_cron_hoje  = ult['beta_cronico']

        # Verificar se hoje tem medição real
        # "hoje" = data mais recente no índice do beta_df
        data_ultimo_idx = beta_df.index[-1]
        hrv_hoje_notna  = pd.notna(ult['LnrMSSD'])

        # Quantos dias desde última medição
        # Procurar último dia com LnrMSSD não-NaN
        ultima_med = beta_df['LnrMSSD'].dropna()
        dias_sem_medicao = 0
        if not ultima_med.empty:
            ultima_data_med = ultima_med.index[-1]
            dias_sem_medicao = (data_ultimo_idx - ultima_data_med).days

        # ── Aviso de dado ausente — CRÍTICO ──────────────────────────────
        if dias_sem_medicao > 0:
            st.error(
                f"⚠️ **ATENÇÃO — {dias_sem_medicao} dia(s) sem medição de HRV.** "
                f"Última medição: {ultima_data_med.strftime('%d/%m/%Y')}. "
                f"Sem sinal autonómico actual, o sistema não pode confirmar "
                f"estado de readiness. **Não prescrever HIIT por precaução.** "
                f"O bug de ontem (HIIT sugerido sem medição) era exactamente este cenário: "
                f"o sistema usou o último valor disponível ({ultima_data_med.strftime('%d/%m/%Y')}) "
                f"como se fosse hoje."
            )

        # ── Regra de convergência ─────────────────────────────────────────
        prescricao, cor_pres, n_pos, n_neg, n_inc, sinais_detalhe = _regra_convergencia(
            beta_hoje, b_agudo_hoje, b_cron_hoje, hrv_hoje_notna
        )

        # ── Cards principais ──────────────────────────────────────────────
        cb1, cb2, cb3, cb4 = st.columns(4)

        # β actual
        beta_label = f"{beta_hoje:.0f}/100" if pd.notna(beta_hoje) else "— (sem dados)"
        beta_delta = (
            "Alta frescura ✅" if pd.notna(beta_hoje) and beta_hoje >= 65
            else ("Zona funcional" if pd.notna(beta_hoje) and beta_hoje >= 50
            else ("Possível fadiga ⚠️" if pd.notna(beta_hoje) else "Sem medição hoje"))
        )
        cb1.metric(
            "β Frescura actual",
            beta_label,
            delta=beta_delta,
            delta_color="normal" if pd.notna(beta_hoje) and beta_hoje >= 50 else "inverse",
            help=(
                "Score 0-100 baseado no z-score de LnrMSSD vs baseline 28d. "
                ">65: Alta frescura | 50-65: Zona funcional | <50: Possível fadiga. "
                "NaN = sem medição hoje → incerteza."
            )
        )

        # βAgudo 3d
        ba_label = f"{b_agudo_hoje:+.1f}%" if pd.notna(b_agudo_hoje) else "— (NaN)"
        ba_delta = (
            "Tendência +3d ↗" if pd.notna(b_agudo_hoje) and b_agudo_hoje >= 1
            else ("Estável" if pd.notna(b_agudo_hoje) and b_agudo_hoje >= -1
            else ("Queda aguda ↘ ⚠️" if pd.notna(b_agudo_hoje) else "Incerto"))
        )
        cb2.metric(
            "βAgudo (3d)",
            ba_label,
            delta=ba_delta,
            delta_color="normal" if pd.notna(b_agudo_hoje) and b_agudo_hoje >= 0 else "inverse",
            help=(
                "% mudança da média LnrMSSD 3d vs 7d (janela calendário, não sessões). "
                "Capta aceleração de fadiga antes do β diário. "
                "NaN se <2 medições nos últimos 3 dias."
            )
        )

        # βCrónico 7d
        bc_label = f"{b_cron_hoje:+.1f}%" if pd.notna(b_cron_hoje) else "— (NaN)"
        bc_delta = (
            "Adaptação positiva ↗" if pd.notna(b_cron_hoje) and b_cron_hoje >= 1
            else ("Estável" if pd.notna(b_cron_hoje) and b_cron_hoje >= -1
            else ("Declínio crónico ↘ ⚠️" if pd.notna(b_cron_hoje) else "Incerto"))
        )
        cb3.metric(
            "βCrónico (7d)",
            bc_label,
            delta=bc_delta,
            delta_color="normal" if pd.notna(b_cron_hoje) and b_cron_hoje >= 0 else "inverse",
            help=(
                "% mudança da média LnrMSSD 7d vs baseline 28d (janela calendário). "
                "Tendência de adaptação de médio prazo. "
                "NaN se <4 medições nos últimos 7 dias."
            )
        )

        # Sinais convergentes
        cb4.metric(
            "Sinais convergentes",
            f"{max(n_pos, n_neg)}/3",
            delta=f"+{n_pos} pos | -{n_neg} neg | ~{n_inc} inc",
            delta_color="normal" if n_pos >= 2 else ("inverse" if n_neg >= 2 else "off"),
            help="Número de indicadores que convergem na mesma direcção. ≥2 = sinal accionável."
        )

        # ── Prescrição — card destacado ───────────────────────────────────
        h_r, h_g, h_b = (
            int(cor_pres[1:3], 16), int(cor_pres[3:5], 16), int(cor_pres[5:7], 16)
        )
        st.markdown(
            f'<div style="padding:16px 20px; border-radius:10px; margin:12px 0; '
            f'background:rgba({h_r},{h_g},{h_b},0.10); '
            f'border-left:6px solid {cor_pres};">'
            f'<span style="font-size:1.15em; font-weight:700; color:{cor_pres};">'
            f'Prescrição Modelo β: {prescricao}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Detalhe dos sinais ────────────────────────────────────────────
        with st.expander("🔍 Detalhe dos 3 indicadores β", expanded=False):
            st.markdown("**Regra de decisão:** actuar apenas quando ≥2 dos 3 indicadores convergem na mesma direcção.")
            st.markdown("**Dado ausente = incerteza = prescrição conservadora** (não prescrever HIIT)")
            st.markdown("")

            for nome, sinal, desc, cor_s in sinais_detalhe:
                hs_r = int(cor_s[1:3], 16)
                hs_g = int(cor_s[3:5], 16)
                hs_b = int(cor_s[5:7], 16)
                icone = "✅" if sinal == +1 else ("⚠️" if sinal == -1 else "⬜")
                st.markdown(
                    f'<div style="padding:8px 14px; margin:4px 0; border-radius:6px; '
                    f'background:rgba({hs_r},{hs_g},{hs_b},0.10); '
                    f'border-left:4px solid {cor_s};">'
                    f'<b>{icone} {nome}:</b> {desc}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("")
            st.caption(
                "**Por que NaN não é zero?** Um dia sem medição de HRV não significa "
                "que a HRV estava normal — significa que não sabemos. O SNA pode estar "
                "a responder a factores invisíveis (glucogénio baixo, microinflamação, "
                "deshidratação) que só a medição confirmaria. "
                "Tratar NaN como 'baseline' foi o que causou o HIIT sugerido ontem."
            )

        # ── Gráfico β — série 90 dias ─────────────────────────────────────
        st.markdown("#### Evolução β — últimos 90 dias")

        beta_plot = beta_df.dropna(subset=['bm28']).copy()
        beta_plot.index.name = 'Data'
        beta_plot = beta_plot.reset_index()

        fig_b = go.Figure()

        # Banda: dias sem medição (LnrMSSD NaN) → fundo cinzento
        sem_med = beta_df[beta_df['LnrMSSD'].isna()].copy()
        sem_med = sem_med.reset_index()
        for _, row_sm in sem_med.iterrows():
            fig_b.add_vrect(
                x0=row_sm['Data'] - pd.Timedelta(hours=12),
                x1=row_sm['Data'] + pd.Timedelta(hours=12),
                fillcolor="rgba(150,150,150,0.15)",
                line_width=0,
                annotation_text="sem HRV",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color="#aaa",
            )

        # Zonas β
        fig_b.add_hrect(y0=65, y1=100,
            fillcolor="rgba(39,174,96,0.07)", line_width=0,
            annotation_text="Alta frescura (>65)", annotation_position="left",
            annotation_font_size=10, annotation_font_color="#27ae60")
        fig_b.add_hrect(y0=40, y1=65,
            fillcolor="rgba(243,156,18,0.05)", line_width=0,
            annotation_text="Zona funcional", annotation_position="left",
            annotation_font_size=10, annotation_font_color="#f39c12")
        fig_b.add_hrect(y0=0, y1=40,
            fillcolor="rgba(231,76,60,0.07)", line_width=0,
            annotation_text="Fadiga possível (<40)", annotation_position="left",
            annotation_font_size=10, annotation_font_color="#e74c3c")

        # β diário
        fig_b.add_trace(go.Scatter(
            x=beta_plot['Data'], y=beta_plot['beta'],
            mode='lines+markers',
            name='β (frescura)',
            line=dict(color='#2471A3', width=2.5),
            marker=dict(size=6),
            hovertemplate='%{x|%d/%m/%Y}<br>β: <b>%{y:.0f}</b><extra></extra>'
        ))

        # βAgudo (eixo y2)
        fig_b.add_trace(go.Scatter(
            x=beta_plot['Data'], y=beta_plot['beta_agudo'],
            mode='lines',
            name='βAgudo 3d (%)',
            line=dict(color='#E74C3C', width=1.5, dash='dot'),
            yaxis='y2',
            hovertemplate='%{x|%d/%m/%Y}<br>βAgudo: <b>%{y:+.1f}%</b><extra></extra>'
        ))

        # βCrónico (eixo y2)
        fig_b.add_trace(go.Scatter(
            x=beta_plot['Data'], y=beta_plot['beta_cronico'],
            mode='lines',
            name='βCrónico 7d (%)',
            line=dict(color='#9B59B6', width=1.5, dash='dash'),
            yaxis='y2',
            hovertemplate='%{x|%d/%m/%Y}<br>βCrónico: <b>%{y:+.1f}%</b><extra></extra>'
        ))

        # Linha y=0 no eixo y2
        fig_b.add_hline(y=0, line_dash='solid', line_color='rgba(150,150,150,0.4)',
                        line_width=1, yref='y2')

        fig_b.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=12),
            height=420,
            hovermode='x unified',
            margin=dict(t=40, b=70, l=60, r=80),
            legend=dict(orientation='h', y=-0.18,
                        font=dict(color='#111', size=10),
                        bgcolor='rgba(255,255,255,0.9)'),
            yaxis=dict(
                title='β (0–100)',
                range=[0, 100],
                showgrid=True, gridcolor='#eee',
                tickfont=dict(color='#2471A3'),
                title_font=dict(color='#2471A3'),
            ),
            yaxis2=dict(
                title='βAgudo / βCrónico (%)',
                overlaying='y', side='right',
                showgrid=False,
                zeroline=True, zerolinecolor='rgba(150,150,150,0.4)',
                tickfont=dict(color='#888'),
                title_font=dict(color='#888'),
            ),
            xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'))
        )

        st.plotly_chart(fig_b, use_container_width=True,
                        config={'displayModeBar': False, 'responsive': True,
                                'scrollZoom': False},
                        key="rec_beta_chart")

        # ── Nota metodológica ─────────────────────────────────────────────
        with st.expander("ℹ️ Metodologia — Modelo β e tratamento de NAs"):
            st.markdown(f"""
**O problema com dados ausentes na tab_recovery actual**

O código actual (linha 132) faz `df = df.dropna(subset=['LnrMSSD'])` — remove todos
os dias sem medição **antes** de qualquer cálculo. Consequência: `.iloc[-1]` (prescrição
de hoje) usa o último dia **com dados**, que pode ser há 2-3 dias. Se esse dia estava
dentro do baseline → sistema diz HIIT. Isto foi exactamente o que aconteceu ontem.

**Como o Modelo β trata NAs**

Em vez de eliminar, usa `reindex()` para manter o calendário completo com NaN onde
não há medição. O rolling usa `min_periods` explícito (não `min_periods=1`) para
garantir que não calcula com dados insuficientes.

| Situação | β | βAgudo | βCrónico | Prescrição |
|---|---|---|---|---|
| Medição normal | calculado | calculado | calculado | por convergência |
| Sem medição hoje | **NaN** | impactado | calculado | **conservadora** |
| 2+ dias sem medição | NaN | **NaN** | calculado | **conservadora** |
| Semana irregular (<4 medições) | parcial | NaN | **NaN** | **conservadora** |

**Fórmulas**

- `β = Φ(z) × 100` onde `z = (LnrMSSD_hoje - bm28) / bs28`
- `βAgudo = (mean_3d - mean_7d) / |mean_7d| × 100`
- `βCrónico = (mean_7d - bm28) / |bm28| × 100`
- Thresholds: +1% / -1% (sinal detectável acima do ruído de CV 8-12%)

**Referências**

Della Mattia G (2025). *VFC y Sistema Nervioso Autónomo — Lo que no podemos sentir.*
ednacore AI. | Plews et al. (2013). Training adaptation and HRV in elite endurance athletes.
*Sports Medicine.* | Buchheit M (2014). Monitoring training status with HR measures.
*Frontiers in Physiology.*
            """)

    st.markdown("---")


    # ════════════════════════════════════════════════════════════════════════
    # INTEGRAR RPE DAS ATIVIDADES (padrão tab_correlacoes)
    # ════════════════════════════════════════════════════════════════════════
    
    # Helper para remover outliers (igual ao da tab_correlacoes)
    def _remove_outliers_iqr(series, factor=1.5):
        s = pd.to_numeric(series, errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s < q1 - factor*iqr) | (s > q3 + factor*iqr)
        s[mask] = np.nan
        return s

    # Helper para classificar RPE
    def classificar_rpe(val):
        if pd.isna(val): return None
        if val < 5: return 'Leve'
        elif val < 7: return 'Moderado'
        else: return 'Pesado'

    # Encontrar coluna de RPE
    rpe_col = next((c for c in ['icu_rpe', 'rpe', 'RPE'] if c in da.columns), None) if not da.empty else None
    
    if rpe_col:
        # Processar datas
        da_proc = da.copy()
        da_proc['Data'] = pd.to_datetime(da_proc['Data']).dt.normalize()
        dw['Data'] = pd.to_datetime(dw['Data']).dt.normalize()
        
        # Remover outliers do RPE
        da_proc[rpe_col] = _remove_outliers_iqr(da_proc[rpe_col])
        da_proc = da_proc.dropna(subset=[rpe_col])
        
        # Agrupar por dia (média dos treinos do dia)
        rpe_diario = da_proc.groupby('Data')[rpe_col].agg([
            ('icu_rpe', 'mean'),  # média do dia
            ('icu_rpe_max', 'max'),
            ('treinos_count', 'count')
        ]).reset_index()
        
        # Classificar intensidade
        rpe_diario['rpe_cat'] = rpe_diario['icu_rpe'].apply(classificar_rpe)
        
        # Merge com wellness
        dw = dw.merge(rpe_diario, on='Data', how='left')
        
        # Preencher dias sem treino como descanso (Rest) ou deixar NaN conforme sua preferência
        # Na tab_correlacoes você usa 'Rest' para dias sem atividade
        # Aqui vamos deixar NaN para indicar "sem treino" mas criar flag de descanso
        dw['treino_pesado'] = (dw['icu_rpe'] >= 7).astype(int)
        dw['treino_moderado'] = ((dw['icu_rpe'] >= 5) & (dw['icu_rpe'] < 7)).astype(int)
        dw['treino_leve'] = (dw['icu_rpe'] < 5).astype(int)
        dw['descanso'] = dw['icu_rpe'].isna().astype(int)  # 1 = dia sem treino
        
        st.success(f"✅ Dados de carga integrados: {dw['icu_rpe'].notna().sum()} dias com RPE")
    else:
        # Criar colunas vazias para não quebrar o código posterior
        dw['icu_rpe'] = np.nan
        dw['icu_rpe_max'] = np.nan
        dw['treinos_count'] = 0
        dw['rpe_cat'] = 'Sem dados'
        dw['treino_pesado'] = 0
        dw['treino_moderado'] = 0
        dw['treino_leve'] = 0
        dw['descanso'] = 1  # Assume descanso se não tem dados
        if not da.empty:
            st.info("ℹ️ Atividades disponíveis mas sem coluna de RPE (icu_rpe/rpe)")

    # Calcular recovery com dw já contendo RPE integrado
    # (dw já foi limpo por preproc_wellness: zscore, zeros→NaN, lookback fill)
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

        # ── MUDANÇA: reindex para calendário completo em vez de dropna ──────
        # Mantém dias sem HRV como NaN — não os elimina
        # Baseline rolling calcula sobre dias COM dados (min_periods)
        # Dias sem medição aparecem no gráfico com estrela ⭐ na posição do baseline
        df_hg = df_hg.set_index('Data')
        date_range_hg = pd.date_range(df_hg.index.min(), df_hg.index.max(), freq='D')
        df_hg = df_hg.reindex(date_range_hg)
        df_hg.index.name = 'Data'
        df_hg = df_hg.reset_index()

        df_hg['LnrMSSD'] = np.where(
            df_hg['hrv'].notna() & (df_hg['hrv'] > 0),
            np.log(df_hg['hrv']),
            np.nan
        )
        # Flag: dia sem medição de HRV
        df_hg['sem_medicao'] = df_hg['LnrMSSD'].isna()

        hg_c1, hg_c2 = st.columns(2)
        dias_fam = hg_c1.slider("Dias baseline rolling", 7, 28, 14, key="hg_baseline")
        n_hg_max = max(14, df_hg['LnrMSSD'].notna().sum())
        n_hg     = hg_c2.slider("Dias a mostrar", 14, min(len(df_hg), 180),
                                  min(60, len(df_hg)), key="hg_dias")

        # Baseline: rolling sobre dias COM dados (NaN são ignorados pelo rolling)
        # min_periods=max(5, dias_fam//2) — permite calcular mesmo com alguns NaN
        _mp = max(5, dias_fam // 2)
        df_hg['bm']    = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=_mp).mean()
        df_hg['bs']    = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=_mp).std()
        df_hg['linf']  = df_hg['bm'] - 0.5 * df_hg['bs']
        df_hg['lsup']  = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio_dp'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs'].replace(0, np.nan)

        # Classificação:
        # - Dia COM medição: HIIT / Recuperação / Sem dados (baseline insuf.)
        # - Dia SEM medição: 'Sem medição' → prescrição conservadora
        def _classif_hg(r):
            if r['sem_medicao']:
                return 'Sem medição ⭐'
            if pd.isna(r['bm']):
                return 'Sem dados'
            if r['linf'] <= r['LnrMSSD'] <= r['lsup']:
                return 'HIIT'
            return 'Recuperação'

        df_hg['intens'] = df_hg.apply(_classif_hg, axis=1)

        # Guardar para análise de correlação posterior
        df_analysis = df_hg.copy()
        df_hg_raw = dw.copy().sort_values('Data')
        df_hg_raw['Data'] = pd.to_datetime(df_hg_raw['Data'])
        df_hg_raw['RMSSD'] = df_hg_raw['hrv'].where(df_hg_raw['hrv'] > 0)
        # Manter calendário completo também para RMSSD bruto
        df_hg_raw = df_hg_raw.set_index('Data')
        _dr_raw = pd.date_range(df_hg_raw.index.min(), df_hg_raw.index.max(), freq='D')
        df_hg_raw = df_hg_raw.reindex(_dr_raw)
        df_hg_raw.index.name = 'Data'
        df_hg_raw = df_hg_raw.reset_index()
        _mp_raw = max(5, dias_fam // 2)
        df_hg_raw['bm_raw'] = df_hg_raw['RMSSD'].rolling(dias_fam, min_periods=_mp_raw).mean()
        df_hg_raw['bs_raw'] = df_hg_raw['RMSSD'].rolling(dias_fam, min_periods=_mp_raw).std()
        df_hg_raw['desvio_dp_raw'] = (df_hg_raw['RMSSD'] - df_hg_raw['bm_raw']) / df_hg_raw['bs_raw'].replace(0, np.nan)
        df_hg_raw['intens_raw'] = df_hg_raw.apply(
            lambda r: ('Sem medição ⭐' if pd.isna(r['RMSSD'])
                       else 'HIIT' if pd.notna(r['bm_raw']) and (r['bm_raw'] - 0.5*r['bs_raw']) <= r['RMSSD'] <= (r['bm_raw'] + 0.5*r['bs_raw'])
                       else ('Recuperação' if pd.notna(r['bm_raw']) else 'Sem dados')), axis=1)

        df_p = df_hg.tail(n_hg).copy()

        COR_MAP = {'HIIT': '#27ae60', 'Recuperação': '#f39c12', 'Sem dados': '#95a5a6', 'Sem medição ⭐': '#cccccc'}

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

            if intensidade == 'Sem medição ⭐':
                # Dias sem HRV: estrela no valor do baseline (ou 0 se baseline tbm NaN)
                y_vals = df_i['bm'].fillna(df_i['LnrMSSD'].mean())
                _fig_hg.add_trace(go.Scatter(
                    x=df_i['Data'], y=y_vals,
                    mode='markers', name='Sem medição ⭐',
                    marker=dict(
                        color='#aaaaaa', size=14,
                        symbol='star',
                        line=dict(width=1.5, color='white')
                    ),
                    hovertemplate='<b>⭐ Sem medição HRV</b><br>%{x|%d/%m/%Y}<br>Prescrição: Recuperação (por precaução)<extra></extra>'
                ), row=1, col=1)
            else:
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
            # Prescrição HOJE — verificar se hoje tem medição real
            _ultimo_hg = df_val.iloc[-1]
            _data_ultimo = _ultimo_hg['Data']
            _data_hoje = pd.Timestamp('today').normalize()
            _dias_sem = (_data_hoje - _data_ultimo).days if pd.notna(_data_ultimo) else 999

            if _ultimo_hg['intens'] == 'Sem medição ⭐' or _dias_sem > 0:
                _pres_hoje = f'⭐ Sem medição ({_dias_sem}d) — Recuperação'
            elif _ultimo_hg['intens'] == 'HIIT':
                _pres_hoje = '✅ HIIT'
            else:
                _pres_hoje = '🟠 Recuperação'
            m3.metric("Prescrição HOJE", _pres_hoje,
                      help=(
                          "⭐ = sem medição de HRV hoje ou nos últimos dias. "
                          "Sem sinal autonómico, a prescrição conservadora é Recuperação. "
                          "O baseline não é substituto da medição real."
                      ))

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
                _slope_now = df_corr['slope_7'].dropna().iloc[-1] if df_corr['slope_7'].notna().any() else 0.0
                st.metric("Slope Atual", f"{_slope_now:.4f}")
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
            # Gráfico começa onde o slope TEM dados (não onde o df começa)
            _df_slope_plot = df_corr[df_corr['slope_7'].notna()].copy()
            if len(_df_slope_plot) > 0:
                _slope_start = _df_slope_plot['Data'].min()
            else:
                _slope_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=86)
            _df_slope_30 = df_corr[df_corr['Data'] >= _slope_start].copy()

            fig_slope_ind = go.Figure()
            
            # Linha do slope — alinhado com início dos dados reais
            fig_slope_ind.add_trace(go.Scatter(
                x=_df_slope_30['Data'],
                y=_df_slope_30['slope_7'],
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
        # ANÁLISE 2: HRV-Guided Protocol (Javaloyes et al. 2018/2020)
        # Protocolo publicado: LnRMSSD 7d rolling avg + SWC + limite consecutivos
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Análise 2: HRV-Guided Protocol — Javaloyes et al.")
        st.caption(
            "Protocolo publicado: Javaloyes A et al. (2018/2020). "
            "LnRMSSD rolling 7 dias vs SWC (mean ± 0.5×SD do baseline). "
            "Máx 2 sessões consecutivas de alta intensidade. Máx 2 dias REST consecutivos. "
            "Referência: Plews et al. (2013); Kiviniemi et al. (2007)."
        )

        with st.expander("📖 Metodologia — como funciona", expanded=False):
            st.markdown("""
**Protocolo Javaloyes et al. 2018/2020 (IJSPP / JSCR)**

1. **Métrica**: LnRMSSD rolling 7 dias (`LnRMSSD₇ₐᵥ`)
2. **Banda SWC**: calculada no período baseline → `mean ± 0.5 × SD`
3. **Decisão diária**:
   - LnRMSSD₇ₐᵥ **acima do SWC** → Alta intensidade (HIGH/HIIT)
   - LnRMSSD₇ₐᵥ **dentro da banda** → Baixa intensidade (LOW)
   - LnRMSSD₇ₐᵥ **abaixo do SWC** → Descanso (REST)
4. **Restrições de consecutivos**:
   - Máx **2 sessões consecutivas** de alta intensidade → força LOW
   - Máx **2 dias consecutivos** de REST → força LOW
5. **SWC actualizado**: a cada 4 semanas com os dados das últimas 4 semanas
            """)

        # ── Reutilizar df_hg já construído pelo HRV-Guided ────────────────
        # df_hg tem: Data, LnrMSSD, bm, bs, linf, lsup, intens, sem_medicao
        # Usar o mesmo LnrMSSD e o mesmo período — alinhado com o gráfico HRV-Guided
        # Rolling 7 dias sobre LnrMSSD (Javaloyes usa 7d, HRV-Guided usa 14d)
        _wc_j = df_hg.copy()
        _wc_j['Data'] = pd.to_datetime(_wc_j['Data']).dt.normalize()

        if _wc_j['LnrMSSD'].notna().sum() < 14:
            st.info("Dados insuficientes para o protocolo Javaloyes (mínimo 14 dias).")
        else:
            # ── Rolling 7 dias (Javaloyes usa 7d, não 14d do HRV-Guided) ────
            _wc_j['ln7'] = _wc_j['LnrMSSD'].rolling(7, min_periods=4).mean()

            # ── Aviso dias sem medição real ───────────────────────────────────
            _n_sem_med = int(_wc_j['sem_medicao'].sum()) if 'sem_medicao' in _wc_j.columns else 0
            # Só mostrar se há dias sem medição COM ln7 calculado (dias dentro do período)
            _n_sem_med_valido = int(
                (_wc_j['sem_medicao'] & _wc_j['ln7'].notna()).sum()
            ) if 'sem_medicao' in _wc_j.columns else 0
            if _n_sem_med_valido > 0:
                st.warning(
                    f"⚠️ **SEM MEDIÇÃO**: {_n_sem_med_valido} dias sem HRV real "
                    "incluídos no rolling 7d. LnRMSSD₇ nesses dias é estimado."
                )

            # ── SWC: calculado sobre os primeiros 28 dias COM medição real ───
            # Javaloyes 2020: baseline weeks (4 semanas iniciais)
            _ln_real = _wc_j[~_wc_j['sem_medicao']]['LnrMSSD'].dropna() if 'sem_medicao' in _wc_j.columns else _wc_j['LnrMSSD'].dropna()
            _ln_base = _ln_real.head(28) if len(_ln_real) >= 7 else _ln_real
            _swc_mean = float(_ln_base.mean())
            _swc_sd   = float(_ln_base.std())
            _swc_sup  = _swc_mean + 0.5 * _swc_sd   # acima → HIGH
            _swc_inf  = _swc_mean - 0.5 * _swc_sd   # abaixo → REST
            # dentro [inf, sup] → LOW

            # ── Máquina de estados Javaloyes/Kiviniemi ────────────────────────
            # 3 zonas com AS DUAS bandas do SWC:
            #   ln7 > SWC sup            → HIGH (HIIT / acima VT2)
            #   SWC inf <= ln7 <= SWC sup → LOW  (abaixo VT1)
            #   ln7 < SWC inf            → REST
            # Restrições:
            #   Máx 2 HIGH consecutivos → força LOW no 3º dia
            #   Máx 2 REST consecutivos → força LOW no 3º dia
            _prescricoes = []
            _consec_high = 0
            _consec_rest = 0

            for _, row in _wc_j.iterrows():
                ln7 = row['ln7']
                if pd.isna(ln7):
                    _pres = 'LOW'
                    _consec_high = 0
                    _consec_rest = 0
                elif ln7 > _swc_sup:
                    if _consec_high >= 2:
                        _pres = 'LOW'
                        _consec_high = 0
                        _consec_rest = 0
                    else:
                        _pres = 'HIGH'
                        _consec_high += 1
                        _consec_rest = 0
                elif ln7 >= _swc_inf:
                    # Dentro da banda → LOW
                    _pres = 'LOW'
                    _consec_high = 0
                    _consec_rest = 0
                else:
                    # Abaixo → REST (máx 2 consecutivos)
                    _consec_high = 0
                    if _consec_rest >= 2:
                        _pres = 'LOW'
                        _consec_rest = 0
                    else:
                        _pres = 'REST'
                        _consec_rest += 1

                _prescricoes.append(_pres)

            _wc_j['prescricao'] = _prescricoes

            # ── Labels e cores ────────────────────────────────────────────────
            _COR_MAP = {'HIGH': '#27ae60', 'LOW': '#3498db', 'REST': '#e74c3c'}
            _LABEL_MAP = {
                'HIGH': '🟢 HIGH — treino intenso',
                'LOW':  '🔵 LOW — treino leve',
                'REST': '🔴 REST — descanso activo',
            }
            _ZONA_MAP = {
                'HIGH': 'Acima SWC sup',
                'LOW':  'Dentro da banda',
                'REST': 'Abaixo SWC inf',
            }

            # ── Cards estado actual ───────────────────────────────────────────
            _wc_j_val = _wc_j[_wc_j['ln7'].notna()]
            if len(_wc_j_val) > 0:
                _ult_j     = _wc_j_val.iloc[-1]
                _pres_hoje = _ult_j['prescricao']
                _ln7_hoje  = _ult_j['ln7']
                _sem_hoje  = bool(_ult_j.get('sem_medicao', False))

                _cj1, _cj2, _cj3, _cj4 = st.columns(4)
                _cj1.metric("LnRMSSD₇ avg",
                            f"{_ln7_hoje:.3f}" + (" ⚠️" if _sem_hoje else ""),
                            help="Rolling 7 dias. ⚠️ = estimado.")
                _cj2.metric("SWC superior", f"{_swc_sup:.3f}",
                            help=f"mean {_swc_mean:.3f} + 0.5×SD → HIGH acima daqui")
                _cj3.metric("SWC inferior", f"{_swc_inf:.3f}",
                            help=f"mean {_swc_mean:.3f} - 0.5×SD → REST abaixo daqui")
                _cj4.metric("Prescrição HOJE", _LABEL_MAP.get(_pres_hoje, _pres_hoje))

                _cor_j = _COR_MAP.get(_pres_hoje, '#888')
                _rj,_gj,_bj = int(_cor_j[1:3],16), int(_cor_j[3:5],16), int(_cor_j[5:7],16)
                st.markdown(
                    f'<div style="padding:14px 20px;border-radius:8px;margin:8px 0;'
                    f'background:rgba({_rj},{_gj},{_bj},0.10);'
                    f'border-left:5px solid {_cor_j};">'
                    f'<b style="font-size:1.1em;color:{_cor_j};">'
                    f'{_LABEL_MAP.get(_pres_hoje, _pres_hoje)}</b>'
                    f'<span style="color:#888;margin-left:12px;font-size:0.9em;">'
                    f'LnRMSSD₇={_ln7_hoje:.3f} | SWC [{_swc_inf:.3f}, {_swc_sup:.3f}]'
                    f'</span></div>',
                    unsafe_allow_html=True
                )

            # ── Tabela últimos 5 dias ─────────────────────────────────────────
            # Período igual ao HRV-Guided (n_hg dias)
            _ult5_j = _wc_j_val.tail(5).copy()
            _rows5_j = []
            for _, r in _ult5_j.iterrows():
                _sem = bool(r.get('sem_medicao', False))
                _rows5_j.append({
                    'Data':       r['Data'].strftime('%d/%m') + (' ⚠️' if _sem else ''),
                    'LnRMSSD₇':  f"{r['ln7']:.3f}" + (' (est.)' if _sem else ''),
                    'Zona':       _ZONA_MAP.get(r['prescricao'], '—'),
                    'Prescrição': _LABEL_MAP.get(r['prescricao'], r['prescricao']),
                })
            st.markdown("**Últimos 5 dias — protocolo Javaloyes:**")
            st.dataframe(pd.DataFrame(_rows5_j), hide_index=True, use_container_width=True)
            st.caption(
                f"SWC baseline (primeiros 28 dias reais): "
                f"mean={_swc_mean:.3f} ± 0.5×SD={0.5*_swc_sd:.3f} → "
                f"banda [{_swc_inf:.3f}, {_swc_sup:.3f}]. "
                "Máx 2 HIGH consecutivos → força LOW. "
                "Máx 2 REST consecutivos → força LOW."
            )

            # ── Gráfico — mesmo período do HRV-Guided (n_hg dias) ───────────
            _df_plot = _wc_j_val.tail(n_hg).copy()
            import plotly.graph_objects as _go_j
            _fig_j = _go_j.Figure()

            # Banda SWC
            _fig_j.add_hrect(
                y0=_swc_inf, y1=_swc_sup,
                fillcolor='rgba(52,152,219,0.08)', line_width=0,
                annotation_text="LOW (dentro banda)",
                annotation_position="right",
                annotation_font_size=9, annotation_font_color='#3498db'
            )
            _fig_j.add_hline(y=_swc_sup, line_dash='dash',
                              line_color='rgba(39,174,96,0.6)', line_width=1.2,
                              annotation_text="SWC sup", annotation_position="right",
                              annotation_font_color='#27ae60', annotation_font_size=9)
            _fig_j.add_hline(y=_swc_inf, line_dash='dash',
                              line_color='rgba(231,76,60,0.6)', line_width=1.2,
                              annotation_text="SWC inf", annotation_position="right",
                              annotation_font_color='#e74c3c', annotation_font_size=9)

            # Linha contínua LnRMSSD₇
            _fig_j.add_trace(_go_j.Scatter(
                x=_df_plot['Data'], y=_df_plot['ln7'],
                mode='lines',
                line=dict(color='rgba(44,62,80,0.35)', width=1.5),
                showlegend=False, hoverinfo='skip'
            ))

            # Pontos coloridos por prescrição
            for _pg, _cg in _COR_MAP.items():
                _sg = _df_plot[_df_plot['prescricao'] == _pg]
                if len(_sg) > 0:
                    _fig_j.add_trace(_go_j.Scatter(
                        x=_sg['Data'], y=_sg['ln7'],
                        mode='markers',
                        name=_LABEL_MAP[_pg],
                        marker=dict(color=_cg, size=7,
                                    line=dict(width=1.5, color='white')),
                        hovertemplate='%{x|%d/%m}<br>LnRMSSD₇: %{y:.3f}<extra></extra>'
                    ))

            # Dias sem medição — marcador ⭐
            _sem_plot = _df_plot[_df_plot.get('sem_medicao', pd.Series(False, index=_df_plot.index)).astype(bool)] if 'sem_medicao' in _df_plot.columns else pd.DataFrame()
            if len(_sem_plot) > 0:
                _fig_j.add_trace(_go_j.Scatter(
                    x=_sem_plot['Data'], y=_sem_plot['ln7'],
                    mode='markers', name='Sem medição',
                    marker=dict(symbol='star', color='gray', size=9,
                                line=dict(width=1, color='white')),
                    hovertemplate='%{x|%d/%m}<br>Estimado: %{y:.3f}<extra></extra>'
                ))

            _fig_j.update_layout(
                height=380,
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=11),
                hovermode='x unified',
                margin=dict(t=30, b=60, l=60, r=130),
                legend=dict(orientation='h', y=-0.22, font=dict(size=10)),
                title=dict(
                    text='LnRMSSD rolling 7 dias — protocolo Javaloyes/Kiviniemi',
                    font=dict(size=13)
                ),
                xaxis=dict(showgrid=True, gridcolor='#eee'),
                yaxis=dict(title='LnRMSSD₇', showgrid=True, gridcolor='#eee')
            )
            st.plotly_chart(_fig_j, use_container_width=True,
                            config={'displayModeBar': False},
                            key='javaloyes_hrv_chart')

            # ── Correlação com HRV-Guided ─────────────────────────────────────
            # HRV-Guided "HIIT"        ↔  Javaloyes "HIGH"
            # HRV-Guided "Recuperação" ↔  Javaloyes "LOW" ou "REST"
            st.markdown("**Correlação: protocolo Javaloyes vs HRV-Guided (Altini/Plews):**")
            try:
                # df_hg tem 'intens' com 'HIIT' / 'Recuperação'
                # _wc_j tem 'prescricao' com 'HIGH' / 'LOW' / 'REST'
                _df_hg_corr = df_hg[['Data', 'intens']].copy()
                _df_hg_corr['Data'] = pd.to_datetime(_df_hg_corr['Data']).dt.normalize()
                _wc_j_corr = _wc_j[_wc_j['ln7'].notna()][['Data', 'prescricao']].copy()
                _wc_j_corr['Data'] = pd.to_datetime(_wc_j_corr['Data']).dt.normalize()

                _merged_j = _wc_j_corr.merge(_df_hg_corr, on='Data', how='inner')
                _merged_j = _merged_j[
                    _merged_j['intens'].isin(['HIIT', 'Recuperação'])
                ].copy()
                _merged_j['high_hrv'] = (_merged_j['intens'] == 'HIIT').astype(int)
                _merged_j['high_jav'] = (_merged_j['prescricao'] == 'HIGH').astype(int)

                if len(_merged_j) >= 10:
                    from scipy import stats as _sst_j
                    _n_j = len(_merged_j)
                    _concord_j = int((_merged_j['high_hrv'] == _merged_j['high_jav']).sum())
                    _pct_j = _concord_j / _n_j * 100
                    _r_j, _p_j = _sst_j.pearsonr(_merged_j['high_hrv'],
                                                   _merged_j['high_jav'])

                    _ct_j = pd.crosstab(
                        _merged_j['intens'].map({
                            'HIIT':        'HRV-Guided: HIIT',
                            'Recuperação': 'HRV-Guided: Recuperação'
                        }),
                        _merged_j['prescricao'].map({
                            'HIGH': 'Javaloyes: HIGH',
                            'LOW':  'Javaloyes: LOW',
                            'REST': 'Javaloyes: REST'
                        })
                    )
                    st.dataframe(_ct_j, use_container_width=True)

                    _cc1, _cc2, _cc3 = st.columns(3)
                    _cc1.metric("Concordância", f"{_pct_j:.0f}%",
                                f"{_concord_j}/{_n_j} dias")
                    _cc2.metric("Correlação r", f"{_r_j:.3f}",
                                "p<0.05" if _p_j < 0.05 else f"p={_p_j:.3f}")
                    _cc3.metric("Dias analisados", _n_j)

                    _hh = int(((_merged_j['high_hrv']==1)&(_merged_j['high_jav']==1)).sum())
                    _hr = int(((_merged_j['high_hrv']==1)&(_merged_j['high_jav']==0)).sum())
                    _rh = int(((_merged_j['high_hrv']==0)&(_merged_j['high_jav']==1)).sum())
                    _rr = int(((_merged_j['high_hrv']==0)&(_merged_j['high_jav']==0)).sum())
                    st.caption(
                        f"Ambos HIGH/HIIT: {_hh}d | "
                        f"Ambos Rec/LOW/REST: {_rr}d | "
                        f"HRV-G HIIT mas Jav LOW/REST: {_hr}d | "
                        f"HRV-G Rec mas Jav HIGH: {_rh}d"
                    )
                else:
                    st.info(f"Dados insuficientes ({len(_merged_j)} dias comuns).")
            except Exception as _j_err:
                st.info(f"Correlação indisponível: {_j_err}")

    # ── DOWNLOAD CSV — sinais de recovery diários (histórico completo) ────────
    st.markdown("---")
    st.subheader("📥 Export Recovery CSV")
    st.caption(
        "Histórico **completo** de recovery — independente do filtro global do sidebar. "
        "Inclui: recovery_score, HRV baseline, CV, zonas Altini/Plews, slope LnrMSSD.")

    try:
        # Usar wc_full (histórico completo) se disponível; senão usa dw (filtrado)
        _dw_csv = wc_full if (wc_full is not None and len(wc_full) > 0) else dw
        # Integrar RPE do histórico completo de atividades
        _da_csv = da_full if (da_full is not None and len(da_full) > 0) else (da if da is not None else pd.DataFrame())

        # Integrar RPE no _dw_csv
        _rpe_col_csv = next((c for c in ['icu_rpe', 'rpe', 'RPE'] if c in _da_csv.columns), None) if not _da_csv.empty else None
        if _rpe_col_csv:
            _da_rpe = _da_csv.copy()
            _da_rpe['Data'] = pd.to_datetime(_da_rpe['Data']).dt.normalize()
            _rpe_agg = _da_rpe.groupby('Data')[_rpe_col_csv].mean().reset_index()
            _rpe_agg.columns = ['Data', 'rpe_diario']
            _dw_csv = _dw_csv.copy()
            _dw_csv['Data'] = pd.to_datetime(_dw_csv['Data']).dt.normalize()
            _dw_csv = _dw_csv.merge(_rpe_agg, on='Data', how='left')

        _rec_dl = calcular_recovery(_dw_csv)
        if len(_rec_dl) > 0:
            _dl_cols_rec = [c for c in [
                'Data', 'hrv', 'rhr', 'recovery_score',
                'hrv_baseline', 'hrv_cv_7d',
                'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness',
            ] if c in _rec_dl.columns]
            _df_rec_export = _rec_dl[_dl_cols_rec].copy()

            # Adicionar rpe_diario se disponível
            if 'rpe_diario' in _dw_csv.columns:
                _rpe_merge = _dw_csv[['Data', 'rpe_diario']].copy()
                _rpe_merge['Data'] = pd.to_datetime(_rpe_merge['Data'])
                _df_rec_export['Data'] = pd.to_datetime(_df_rec_export['Data'])
                _df_rec_export = _df_rec_export.merge(_rpe_merge, on='Data', how='left')

            _df_rec_export['Data'] = _df_rec_export['Data'].astype(str)
            _df_rec_export = _df_rec_export.round(4)

            _c_dl1, _c_dl2 = st.columns([2, 1])
            with _c_dl1:
                st.dataframe(_df_rec_export.tail(14), use_container_width=True, hide_index=True)
                st.caption(f"Mostrando últimos 14 de {len(_df_rec_export)} dias totais no CSV")
            with _c_dl2:
                st.metric("Dias no CSV", len(_df_rec_export))
                st.metric("Sidebar (gráficos)", len(dw))
                st.caption("CSV = histórico completo\nGráficos = período sidebar")
                st.download_button(
                    label="📥 Download Recovery CSV",
                    data=_df_rec_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    file_name="atheltica_recovery.csv",
                    mime="text/csv",
                    key="rec_dl_csv",
                )
    except Exception as _rec_err:
        st.info(f"Export não disponível: {_rec_err}")
