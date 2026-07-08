"""
tab_hrv_analyzer.py — Recovery Pattern Analyzer
================================================
Módulo N=1 de análise fisiológica: o que o HRV (rMSSD) diz sobre o treino
em períodos específicos vs períodos anteriores.

Arquitectura:
  1. Construção do sinal HRV normalizado (rMSSD, ln_rMSSD, rMSSD_norm, ratio HRV/RHR)
  2. Detecção automática de períodos (HRV↑ vs HRV↓)
  3. Event window analysis: 14d antes, período, 7d depois
  4. Lag correlation: qual variável de treino antecede as mudanças de HRV
  5. Comparação Before/After: quais variáveis mudaram e quando
  6. Padrões recorrentes: "top 10% HRV days — o que aconteceu antes"
  7. Fingerprints de recovery vs suppression

Métricas HRV usadas:
  rMSSD       — sinal base
  ln_rMSSD    — log-normalizado (literatura padrão)
  AVNN        — 60000 / HR  (espaço temporal por batimento)
  rMSSD_norm  — (rMSSD / AVNN) × 100  = variabilidade relativa à FC
  HRV_RHR_r   — coupling autonómico

Análises estatísticas:
  rolling mean / z-score / EWMA / slope
  cross-correlação com lag 1-14d
  Cohen's d entre períodos
  event windows (alinhamento em torno de mudanças)
"""

from utils.config import *
from utils.helpers import *
from utils.data import *
import utils.hrv_analyzer as _hra
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

MC = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}
_C = {'primary': '#2980b9', 'hrv_up': '#27ae60', 'hrv_dn': '#e74c3c',
      'neutral': '#7f8c8d', 'load': '#e67e22', 'accent': '#8e44ad',
      'bg': 'white', 'grid': '#eee', 'font': '#111'}


# ── A. Construção do sinal HRV ────────────────────────────────────────────────

_build_hrv_signal = _hra._build_hrv_signal


# ── B. Detecção automática de períodos ───────────────────────────────────────

_detect_hrv_periods = _hra._detect_hrv_periods


# ── C. Construir DataFrame de treino diário ───────────────────────────────────

_build_training_signal = _hra._build_training_signal


# ── D. Event Window Analysis ──────────────────────────────────────────────────

_event_window = _hra._event_window


# ── E. Lag Correlation ────────────────────────────────────────────────────────

_lag_correlations = _hra._lag_correlations


# ── F. Comparação Before/After ───────────────────────────────────────────────

_compare_periods = _hra._compare_periods


# ── G. Fingerprint: Top vs Bottom HRV days ───────────────────────────────────

_hrv_fingerprint = _hra._hrv_fingerprint


# ══════════════════════════════════════════════════════════════════════════════
# EXPLICAÇÕES — o que cada análise faz e a intenção (dropdowns "ℹ️ O que é isto?")
# ══════════════════════════════════════════════════════════════════════════════

_EXPLICACOES = {
    'periodo_manual': (
        "**O que faz:** compara um período que escolhes (ex.: as últimas 3 semanas) "
        "com o período imediatamente anterior, mostrando o que mudou no HRV e no treino.\n\n"
        "**Intenção:** responder a *'este bloco de treino melhorou ou piorou a minha "
        "recuperação face ao anterior?'*. Útil depois de uma fase específica (carga, taper, "
        "estágio) para ver o efeito autonómico."),
    'deteccao': (
        "**O que faz:** encontra automaticamente os períodos em que o teu HRV esteve "
        "significativamente acima (HRV↑) ou abaixo (HRV↓) do normal, usando o z-score de 28 dias.\n\n"
        "**Intenção:** identificar sozinho as fases de boa e má forma autonómica, sem teres "
        "de as procurar à mão. Cada período detectado é um candidato a investigar (o que "
        "aconteceu no treino nessa altura?)."),
    'lag': (
        "**O que faz:** testa vários desfasamentos (lags) entre cada variável de treino e o "
        "HRV, e encontra em quantos dias o treino melhor 'prevê' a mudança de HRV. Mostra o "
        "rMSSD e o HF power lado a lado.\n\n"
        "**Intenção:** responder a *'quantos dias depois de um treino duro é que o meu HRV "
        "reage?'*. Se o rMSSD e o HF power apontam o mesmo lag → o efeito é robusto. O lag "
        "óptimo vem do Auto-Runner."),
    'fingerprint': (
        "**O que faz:** pega nos teus melhores e piores dias de HRV e olha para trás — o que "
        "fizeste nos X dias antes de cada um? Revela o padrão de treino que antecede a boa "
        "vs a má forma.\n\n"
        "**Intenção:** descobrir a tua 'impressão digital' de recuperação: que combinação de "
        "carga/intensidade costuma preceder os teus melhores dias autonómicos."),
    'ari': (
        "**O que faz:** o Autonomic Readiness Index — um score 0-100 que funde 5 sinais "
        "autonómicos (HRV, RHR, tendência, etc.) numa só métrica de prontidão.\n\n"
        "**Intenção:** um 'resumo executivo' do teu estado: >60 = pronto para carga, "
        "<40 = precisa de atenção. Simplifica a decisão diária."),
    'estados': (
        "**O que faz:** classifica cada dia num de vários estados fisiológicos (fadiga "
        "acumulada, rebote parassimpático, resposta ao taper, estado resiliente, baseline...).\n\n"
        "**Intenção:** dar um 'rótulo' interpretável a cada dia, para perceberes em que "
        "regime autonómico estás e como evoluis ao longo do tempo."),
    'elasticidade': (
        "**O que faz:** mede quanto tempo o teu HRV demora a recuperar depois de uma queda "
        "(supressão). O τ (tau) é o tempo mediano de retorno ao normal.\n\n"
        "**Intenção:** quantificar a tua capacidade de recuperação. Um τ baixo = recuperas "
        "rápido; τ alto = a fadiga persiste. O τ alimenta a janela do Directional."),
    'lag_avancado': (
        "**O que faz:** como o Lag Correlation, mas com 3 métodos (Pearson, Spearman, "
        "informação mútua) para captar também relações não-lineares.\n\n"
        "**Intenção:** confirmar de forma mais robusta as relações treino→HRV. Se os 3 "
        "métodos concordam, a relação é sólida."),
    'directional': (
        "**O que faz:** testa padrões específicos (ex.: 'carga muito elevada', 'TSB positivo') "
        "e mede em que % das vezes o HRV melhorou nos dias seguintes.\n\n"
        "**Intenção:** validar regras accionáveis do tipo *'quando faço X, recupero bem?'*. "
        "A janela usa o τ da elasticidade. ⚠️ Consistência alta só com histórico longo pode "
        "ser artefacto do N grande — compara sempre com 1 ano."),
    'dose_response': (
        "**O que faz:** traça a relação dose-efeito entre uma variável de treino e o HRV — "
        "mais carga leva a mais ou menos HRV?\n\n"
        "**Intenção:** encontrar o teu 'ponto óptimo' de carga: até onde podes empurrar antes "
        "de a recuperação começar a sofrer."),
    'semanas': (
        "**O que faz:** agrupa as tuas semanas por perfil de treino (load, monotonia, "
        "frequência, %Z3, strain) via K-means, e colore cada grupo pelo HRV médio da semana "
        "seguinte.\n\n"
        "**Intenção:** descobrir 'tipos' de semana que tens e qual o seu efeito na recuperação "
        "— que tipo de semana costuma ser seguido de bom HRV, e qual de fadiga."),
    'transicoes': (
        "**O que faz:** calcula a probabilidade de passar de cada estado para outro, e "
        "destaca o próximo estado mais provável a partir do teu estado de hoje.\n\n"
        "**Intenção:** antecipar a tua trajectória autonómica: *'estou em fadiga — qual a "
        "probabilidade de amanhã estar recuperado vs continuar em fadiga?'*."),
    'autorunner': (
        "**O que faz:** corre todas as análises acima para 7 períodos (60d a todo histórico) "
        "testando muitas combinações de parâmetros, e encontra os valores óptimos de cada.\n\n"
        "**Intenção:** em vez de mexeres nos sliders à mão, o Auto-Runner descobre "
        "automaticamente o melhor lag, janela, nº de clusters, etc. — que depois pré-preenchem "
        "as análises individuais."),
}


def _dropdown_explica(chave):
    """Mostra um expander 'ℹ️ O que é isto?' com a explicação da análise."""
    txt = _EXPLICACOES.get(chave)
    if txt:
        with st.expander("ℹ️ O que é isto? (o que faz e para que serve)"):
            st.markdown(txt)


# ══════════════════════════════════════════════════════════════════════════════
# TAB PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def tab_hrv_analyzer(dw: pd.DataFrame, da: pd.DataFrame,
                     wc_full: pd.DataFrame = None, da_full: pd.DataFrame = None):
    """
    Recovery Pattern Analyzer — tab principal.
    dw / wc_full : DataFrame de wellness
    da / da_full : DataFrame de actividades
    """
    st.subheader("🔬 Recovery Pattern Analyzer")
    st.caption(
        "Análise N=1 longitudinal: o que o HRV (rMSSD) diz sobre o treino em "
        "diferentes períodos. Causalidade temporal, lag correlation, event windows, "
        "fingerprints de recovery vs suppression."
    )

    # Dados
    _dw = wc_full if wc_full is not None else dw
    _da = da_full if da_full is not None else da

    if _dw is None or len(_dw) == 0:
        st.warning("Sem dados de wellness. Verifica a ligação à Google Sheet.")
        return
    if 'hrv' not in _dw.columns or _dw['hrv'].notna().sum() < 14:
        st.warning("Sem dados suficientes de HRV (mínimo 14 dias).")
        return

    # ── Filtro de datas ───────────────────────────────────────────────────────
    _dw_all = _dw.copy()
    _dw_all['Data'] = pd.to_datetime(_dw_all['Data'])
    _da_all = _da.copy() if _da is not None else None
    if _da_all is not None:
        _da_all['Data'] = pd.to_datetime(_da_all['Data'])

    _date_min_data = _dw_all['Data'].min().date()
    _date_max_data = _dw_all['Data'].max().date()

    _fc1, _fc2 = st.columns(2)
    with _fc1:
        _filter_from = st.date_input(
            "📅 Analisar a partir de",
            value=max(_date_min_data, pd.Timestamp('2023-01-01').date()),
            min_value=_date_min_data,
            max_value=_date_max_data,
            key="hrv_filter_from",
            help="Exclui dados anteriores a esta data de TODAS as análises. "
                 "Útil para ignorar períodos com dados incompletos ou de treino muito diferente."
        )
    with _fc2:
        _filter_to = st.date_input(
            "Até",
            value=_date_max_data,
            min_value=_date_min_data,
            max_value=_date_max_data,
            key="hrv_filter_to",
            help="Data final da análise."
        )

    # Aplicar filtro
    _dw = _dw_all[
        (_dw_all['Data'].dt.date >= _filter_from) &
        (_dw_all['Data'].dt.date <= _filter_to)
    ].reset_index(drop=True)
    if _da_all is not None:
        _da = _da_all[
            (_da_all['Data'].dt.date >= _filter_from) &
            (_da_all['Data'].dt.date <= _filter_to)
        ].reset_index(drop=True)

    _n_hrv = _dw['hrv'].notna().sum()
    if _n_hrv < 14:
        st.warning(f"Apenas {_n_hrv} dias de HRV no período seleccionado (mínimo 14). "
                   "Alarga o intervalo de datas.")
        return

    st.caption(
        f"📅 Período de análise: **{_filter_from}** → **{_filter_to}** "
        f"({(_filter_to - _filter_from).days} dias | {_n_hrv} medições HRV)"
    )

    # ── Construir sinais ──────────────────────────────────────────────────────
    with st.spinner("A construir sinais HRV e treino..."):
        sig_hrv   = _build_hrv_signal(_dw)
        sig_train = _build_training_signal(_da) if _da is not None else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # GATE — nada de pesado corre até o utilizador clicar "▶ Rodar análises".
    # Isto evita que o auto-runner (pesado) corra a cada interação da app inteira
    # (o Streamlit corre todas as tabs sempre; sem este gate, mexer num dropdown
    #  noutra tab dispararia o auto-runner e empilharia a renderização).
    # O resultado fica guardado em session_state — as análises leem de lá.
    # ══════════════════════════════════════════════════════════════════════════
    # Chave de identidade dos dados (deteta se mudaram desde a última corrida)
    try:
        _dkey_gate = str((
            len(sig_hrv), str(sig_hrv['Data'].iloc[0]) if len(sig_hrv) else '',
            str(sig_hrv['Data'].iloc[-1]) if len(sig_hrv) else '',
            len(sig_train), str(sig_train['Data'].iloc[-1]) if len(sig_train) else ''))
    except Exception:
        _dkey_gate = str((len(sig_hrv), len(sig_train)))

    _gate = st.session_state.get('_hrv_gate')
    _ja_correu = (_gate is not None and _gate.get('key') == _dkey_gate)

    # ══════════════════════════════════════════════════════════════════════════
    # GATE — controla APENAS o auto-runner (pesado, ~100s) e as Análises Avançadas.
    # As análises leves (Detecção, Lag, Fingerprint) correm sempre — não travam.
    # O "▶ Rodar" calcula o auto-runner (óptimos + avançadas) e guarda em session.
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    _gc1, _gc2 = st.columns([1, 3])
    with _gc1:
        _rodar = st.button("▶ Rodar Auto-Runner + Avançadas", type="primary",
                           key="hrv_gate_run", use_container_width=True)
    with _gc2:
        if _ja_correu:
            st.caption("✅ Auto-Runner calculado. Os sliders abaixo usam os óptimos "
                       "encontrados; as Análises Avançadas estão disponíveis. "
                       "Clica novamente para actualizar.")
        else:
            st.caption("⏸️ **Detecção, Lag e Fingerprint já correm automaticamente** abaixo. "
                       "Clica em **▶ Rodar** para calcular o Auto-Runner (óptimos por "
                       "período + Análises Avançadas: ARI, Estados, Elasticidade, etc.). "
                       "Só o Auto-Runner é pesado — por isso fica atrás do botão.")

    if _rodar:
        with st.spinner("A calcular Auto-Runner (varredura de todos os períodos)..."):
            _hoje_g = pd.Timestamp.now().normalize()
            _res_g = _hra.run_autorunner(sig_hrv, sig_train,
                                         da_full=_da if _da is not None else None,
                                         hoje_ar=_hoje_g)
            _otimos_g = _hra.extrair_otimos(_res_g.get('runner_results', []),
                                            periodo_pref='1 ano')
            st.session_state['_hrv_gate'] = {
                'key': _dkey_gate,
                'autorunner': _res_g,
                'otimos': _otimos_g,
            }
        _ja_correu = True
        _gate = st.session_state['_hrv_gate']

    # Óptimos disponíveis (vazios até clicar "▶ Rodar" → sliders usam defaults)
    _AR_RESULT = _gate.get('autorunner', {}) if _ja_correu and _gate else {}
    _AR_OTIMOS = _gate.get('otimos', {}) if _ja_correu and _gate else {}

    # ── Selector de análise (radio horizontal — evita bug de secções empilhadas) ──
    st.markdown("---")
    _analyses = ["📅 Período Manual", "🔍 Detecção Automática",
                 "🔗 Lag Correlation", "🧬 Fingerprint HRV"]
    _mode_lbl = st.radio(
        "Análise", _analyses, horizontal=True,
        label_visibility="collapsed", key="hrv_mode_radio")
    _mode = _analyses.index(_mode_lbl)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PAINEL SUPERIOR — Sinal HRV completo (sempre visível)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📊 Sinal HRV — visão geral completa", expanded=True):
        _metric_choice = st.radio(
            "Métrica HRV a visualizar",
            ["rMSSD absoluto", "ln(rMSSD)", "rMSSD normalizado (÷AVNN×100)",
             "HRV/RHR coupling"],
            horizontal=True, key="hrv_metric_choice"
        )
        _col_map = {
            "rMSSD absoluto":                  "hrv",
            "ln(rMSSD)":                       "ln_hrv",
            "rMSSD normalizado (÷AVNN×100)":   "hrv_norm",
            "HRV/RHR coupling":                "hrv_rhr_ratio",
        }
        _yvar = _col_map[_metric_choice]

        if _yvar not in sig_hrv.columns:
            st.warning(f"Coluna {_yvar} não disponível (falta RHR?).")
        else:
            _fig_hrv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.65, 0.35],
                                     vertical_spacing=0.04)

            # Sinal principal
            _fig_hrv.add_trace(go.Scatter(
                x=sig_hrv['Data'], y=sig_hrv[_yvar],
                mode='lines', name=_metric_choice,
                line=dict(color=_C['hrv_up'], width=1.5),
                hovertemplate='%{x|%d/%m/%Y}<br>HRV: <b>%{y:.1f}</b><extra></extra>'
            ), row=1, col=1)

            # EWMA
            _ewma_col = 'ln_hrv_ewma' if 'ln' in _yvar else 'hrv_ewma'
            if _ewma_col in sig_hrv.columns:
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'], y=sig_hrv[_ewma_col],
                    mode='lines', name='EWMA (span=19)',
                    line=dict(color=_C['accent'], width=2, dash='dash'),
                    hovertemplate='%{x|%d/%m/%Y}<br>EWMA: <b>%{y:.1f}</b><extra></extra>'
                ), row=1, col=1)

            # Banda baseline ±1 std (28d)
            if 'hrv_mean_28d' in sig_hrv.columns and _yvar == 'hrv':
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'],
                    y=sig_hrv['hrv_mean_28d'] + sig_hrv['hrv_std_28d'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ), row=1, col=1)
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'],
                    y=sig_hrv['hrv_mean_28d'] - sig_hrv['hrv_std_28d'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(39,174,96,0.12)',
                    name='Banda ±1σ (28d)', hoverinfo='skip'
                ), row=1, col=1)

            # RHR no painel inferior (se disponível)
            if 'rhr' in sig_hrv.columns:
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'], y=sig_hrv['rhr'],
                    mode='lines', name='RHR (bpm)',
                    line=dict(color=_C['hrv_dn'], width=1.5),
                    hovertemplate='%{x|%d/%m/%Y}<br>RHR: <b>%{y:.0f}</b> bpm<extra></extra>'
                ), row=2, col=1)

            # Z-score overlay (eixo secundário seria complexo — usar cor de fundo)
            # Marcar zonas z>0.5 a verde e z<-0.5 a vermelho
            if 'hrv_z28' in sig_hrv.columns:
                _hv = sig_hrv[sig_hrv['hrv_z28'] > 0.5]
                _hd = sig_hrv[sig_hrv['hrv_z28'] < -0.5]
                for _row_mark, _df_mark, _col_mark in [
                    (1, _hv, 'rgba(39,174,96,0.15)'),
                    (1, _hd, 'rgba(231,76,60,0.15)'),
                ]:
                    if len(_df_mark) > 0:
                        _dates_m = _df_mark['Data']
                        _y_m     = _df_mark[_yvar]
                        _fig_hrv.add_trace(go.Scatter(
                            x=_dates_m, y=_y_m,
                            mode='markers',
                            marker=dict(size=5, color=_col_mark.replace('0.15', '0.6'),
                                        line=dict(width=0)),
                            name='HRV↑ (z>0.5)' if 'verde' in _col_mark or '174' in _col_mark
                                 else 'HRV↓ (z<-0.5)',
                            showlegend=True, hoverinfo='skip'
                        ), row=_row_mark, col=1)

            _fig_hrv.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=11),
                height=420, hovermode='x unified',
                margin=dict(t=20, b=50, l=55, r=20),
                legend=dict(orientation='h', y=-0.16,
                            font=dict(color='#111', size=10)),
            )
            _fig_hrv.update_xaxes(showgrid=True, gridcolor='#eee',
                                   tickfont=dict(color='#111'))
            _fig_hrv.update_yaxes(showgrid=True, gridcolor='#eee',
                                   tickfont=dict(color='#111'))
            st.plotly_chart(_fig_hrv, use_container_width=True,
                            config=MC, key='hrv_main_plot')

            # Cards de resumo
            _c1, _c2, _c3, _c4, _c5 = st.columns(5)
            _hrv_now  = sig_hrv['hrv'].dropna().iloc[-1] if sig_hrv['hrv'].notna().any() else np.nan
            _hrv_b28  = sig_hrv['hrv_mean_28d'].dropna().iloc[-1] if 'hrv_mean_28d' in sig_hrv.columns else np.nan
            _hrv_z    = sig_hrv['hrv_z28'].dropna().iloc[-1] if 'hrv_z28' in sig_hrv.columns else np.nan
            _hrv_slp  = sig_hrv['hrv_slope_7d'].dropna().iloc[-1] if 'hrv_slope_7d' in sig_hrv.columns else np.nan
            _ln_now   = sig_hrv['ln_hrv'].dropna().iloc[-1] if 'ln_hrv' in sig_hrv.columns else np.nan
            _norm_now = sig_hrv['hrv_norm'].dropna().iloc[-1] if 'hrv_norm' in sig_hrv.columns else np.nan
            _rhr_now  = sig_hrv['rhr'].dropna().iloc[-1] if 'rhr' in sig_hrv.columns else np.nan

            _c1.metric("rMSSD hoje", f"{_hrv_now:.0f} ms" if not np.isnan(_hrv_now) else "—",
                       delta=f"base: {_hrv_b28:.0f}" if not np.isnan(_hrv_b28) else None,
                       help="rMSSD absoluto. Baseline = média 28d.")
            _c2.metric("ln(rMSSD)", f"{_ln_now:.2f}" if not np.isnan(_ln_now) else "—",
                       help="Logaritmo natural do rMSSD — distribuição mais normal.")
            _c3.metric("rMSSD norm.", f"{_norm_now:.2f}" if not np.isnan(_norm_now) else "—",
                       help="(rMSSD÷AVNN)×100 — variabilidade relativa à FC de repouso.")
            _c4.metric("z-score 28d", f"{_hrv_z:+.2f}" if not np.isnan(_hrv_z) else "—",
                       delta="acima baseline" if (not np.isnan(_hrv_z) and _hrv_z > 0) else "abaixo baseline",
                       delta_color="normal" if (not np.isnan(_hrv_z) and _hrv_z > 0) else "inverse",
                       help="Desvios-padrão acima/abaixo da média 28d.")
            _c5.metric("Slope 7d", f"{_hrv_slp:+.1f} ms/d" if not np.isnan(_hrv_slp) else "—",
                       delta="→ melhorando" if (not np.isnan(_hrv_slp) and _hrv_slp > 0.3) else
                             ("→ estável" if not np.isnan(_hrv_slp) and abs(_hrv_slp) <= 0.3 else "→ a cair"),
                       delta_color="normal" if (not np.isnan(_hrv_slp) and _hrv_slp > 0.3) else
                                   "off" if not np.isnan(_hrv_slp) and abs(_hrv_slp) <= 0.3 else "inverse",
                       help="Slope da regressão linear dos últimos 7 dias de HRV.")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MODO: PERÍODO MANUAL
    # ══════════════════════════════════════════════════════════════════════════
    if _mode == 0:
        st.markdown("### 📅 Análise por período manual")
        _dropdown_explica('periodo_manual')
        st.caption("Selecciona o período alvo e compara com o período anterior.")

        _hrv_dates = sig_hrv['Data'].dropna()
        _date_min  = _hrv_dates.min().date()
        _date_max  = _hrv_dates.max().date()

        _col_d1, _col_d2, _col_d3 = st.columns(3)
        with _col_d1:
            _p_start = st.date_input("Início do período", value=_date_max - pd.Timedelta(days=21),
                                     min_value=_date_min, max_value=_date_max, key="hrv_pstart")
        with _col_d2:
            _p_end = st.date_input("Fim do período", value=_date_max,
                                   min_value=_date_min, max_value=_date_max, key="hrv_pend")
        with _col_d3:
            _ref_days = st.number_input("Dias de referência (anterior)", value=21,
                                        min_value=7, max_value=90, step=7, key="hrv_refdays")

        # Período Manual só corre ao clicar (tu escolhes as datas primeiro)
        _run_manual = st.button("▶ Comparar período", type="primary", key="hrv_run_manual_btn")
        if _run_manual:
            st.session_state['_hrv_manual_done'] = True

        if st.session_state.get('_hrv_manual_done'):
            _ts = pd.Timestamp(_p_start)
            _te = pd.Timestamp(_p_end)

            if len(sig_train) == 0:
                st.warning("Sem dados de treino para comparar.")
            else:
                cmp = _compare_periods(sig_hrv, sig_train, _ts, _te, _ref_days)
                if cmp.empty:
                    st.warning("Sem dados suficientes no período.")
                else:
                    st.markdown(f"#### Comparação: {_ref_days}d antes vs [{_p_start} → {_p_end}]")

                    # Separar HRV de treino
                    _hrv_rows   = cmp[cmp['col'].isin(['hrv','ln_hrv','hrv_norm',
                                                        'hrv_rhr_ratio','rhr'])]
                    _train_rows = cmp[~cmp['col'].isin(['hrv','ln_hrv','hrv_norm',
                                                         'hrv_rhr_ratio','rhr'])]

                    # ── HRV: o que mudou ────────────────────────────────────
                    st.markdown("**❤️ HRV — o que mudou neste período**")
                    _hrv_display = _hrv_rows[['Variável','Antes','Período','Δ%',"Cohen's d",'sig']].copy()
                    _hrv_display['Δ%'] = _hrv_display['Δ%'].apply(
                        lambda x: f"{x:+.1f}%" if not pd.isna(x) else '—')
                    _hrv_display['sig'] = _hrv_display['sig'].map({True: '✅', False: ''})
                    st.dataframe(_hrv_display.rename(columns={'sig': 'Sig.'}),
                                 hide_index=True, use_container_width=True)

                    # ── Treino: o que mudou ─────────────────────────────────
                    st.markdown("**🏋️ Treino — o que antecedeu / acompanhou**")
                    _train_rows_s = _train_rows.sort_values('Δ%', key=lambda x:
                        pd.to_numeric(x, errors='coerce').abs(), ascending=False)
                    _tr_display = _train_rows_s[['Variável','Antes','Período','Δ%',
                                                  "Cohen's d",'sig']].copy()
                    _tr_display['Δ%'] = _tr_display['Δ%'].apply(
                        lambda x: f"{x:+.1f}%" if not pd.isna(x) else '—')
                    _tr_display['sig'] = _tr_display['sig'].map({True: '✅', False: ''})
                    st.dataframe(_tr_display.rename(columns={'sig': 'Sig.'}),
                                 hide_index=True, use_container_width=True)

                    # ── Narrativa automática ────────────────────────────────
                    st.markdown("**💡 Interpretação automática**")
                    _sig_changes = cmp[cmp['sig'] & (cmp['Δ%'].abs() > 5)].copy()
                    _sig_changes['Δ%_num'] = pd.to_numeric(_sig_changes['Δ%'], errors='coerce')

                    _hrv_delta = cmp[cmp['col']=='hrv']['Δ%'].values
                    _hrv_delta = float(_hrv_delta[0]) if len(_hrv_delta) > 0 else 0
                    _hrv_dir   = "subiu" if _hrv_delta > 0 else "desceu"
                    _hrv_mag   = "significativamente" if abs(_hrv_delta) > 10 else "ligeiramente"

                    _narrativa = [f"**HRV {_hrv_dir} {abs(_hrv_delta):.1f}%** "
                                  f"({_hrv_mag}) neste período."]

                    _top_pos = _sig_changes[_sig_changes['Δ%_num'] > 10] \
                        .nlargest(3, 'Δ%_num')
                    _top_neg = _sig_changes[_sig_changes['Δ%_num'] < -10] \
                        .nsmallest(3, 'Δ%_num')

                    if len(_top_pos) > 0:
                        _items = ', '.join(
                            f"**{r['Variável']}** ({r['Δ%']:+.1f}%)"
                            for _, r in _top_pos.iterrows()
                        )
                        _narrativa.append(f"Variáveis que subiram: {_items}.")
                    if len(_top_neg) > 0:
                        _items = ', '.join(
                            f"**{r['Variável']}** ({r['Δ%']:+.1f}%)"
                            for _, r in _top_neg.iterrows()
                        )
                        _narrativa.append(f"Variáveis que desceram: {_items}.")

                    for _n in _narrativa:
                        st.markdown(f"→ {_n}")

                    # ── Radar chart Before vs After ─────────────────────────
                    _radar_vars = ['load', 'dur_min', 'pct_z3', 'freq_7d',
                                   'mono_7d', 'tsb', 'atl']
                    _radar_rows = cmp[cmp['col'].isin(_radar_vars)].copy()
                    if len(_radar_rows) >= 4:
                        _before_n = (_radar_rows['Antes'] /
                                     _radar_rows['Antes'].replace(0, np.nan)).fillna(1)
                        _target_n = (_radar_rows['Período'] /
                                     _radar_rows['Antes'].replace(0, np.nan)).fillna(1)
                        _labels   = _radar_rows['Variável'].tolist()

                        _fig_r = go.Figure()
                        _fig_r.add_trace(go.Scatterpolar(
                            r=list(_before_n) + [_before_n.iloc[0]],
                            theta=_labels + [_labels[0]],
                            fill='toself', name='Antes',
                            line_color=_C['neutral'],
                            fillcolor='rgba(127,140,141,0.2)'
                        ))
                        _fig_r.add_trace(go.Scatterpolar(
                            r=list(_target_n) + [_target_n.iloc[0]],
                            theta=_labels + [_labels[0]],
                            fill='toself', name='Período',
                            line_color=_C['primary'],
                            fillcolor='rgba(41,128,185,0.2)'
                        ))
                        _fig_r.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 2],
                                                tickfont=dict(color='#111', size=9))
                            ),
                            paper_bgcolor='white', font=dict(color='#111', size=11),
                            height=380, margin=dict(t=30, b=30, l=40, r=40),
                            legend=dict(orientation='h', y=-0.08,
                        font=dict(color='#111', size=10))
                        )
                        st.plotly_chart(_fig_r, use_container_width=True,
                                        config=MC, key='hrv_radar')

                    # ── Download ────────────────────────────────────────────
                    st.download_button(
                        "📥 Download comparação período",
                        cmp[['Variável','Antes','Período','Δ%',"Cohen's d",'p-valor']].to_csv(
                            index=False, sep=';', decimal=','
                        ).encode('utf-8'),
                        f"hrv_comparacao_{_p_start}_{_p_end}.csv",
                        "text/csv", key="hrv_dl_manual"
                    )
        else:
            st.info("👆 Escolhe o período e clica em **▶ Comparar período** para ver a análise.")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MODO: DETECÇÃO AUTOMÁTICA
    # ══════════════════════════════════════════════════════════════════════════
    elif _mode == 1:
        st.markdown("### 🔍 Detecção automática de períodos HRV")
        _dropdown_explica('deteccao')
        st.caption("Detecta automaticamente períodos de HRV↑ e HRV↓ com base no z-score 28d.")

        _cz1, _cz2, _cz3 = st.columns(3)
        _z_thresh = _cz1.slider("Threshold z-score", 0.3, 1.5, 0.5, 0.1, key="hrv_zthresh")
        _min_len  = _cz2.slider("Duração mínima (dias)", 3, 14, 5, 1, key="hrv_minlen")
        _show_n   = _cz3.number_input("Mostrar últimos N períodos", 3, 20, 6, 1, key="hrv_shown")

        periods = _detect_hrv_periods(sig_hrv, _min_len, _z_thresh)

        if not periods:
            st.info("Sem períodos detectados com os critérios actuais.")
        else:
            _periods_df = pd.DataFrame(periods)
            _periods_df['duração'] = (_periods_df['end'] - _periods_df['start']).dt.days + 1
            _periods_df['start'] = _periods_df['start'].dt.strftime('%Y-%m-%d')
            _periods_df['end']   = _periods_df['end'].dt.strftime('%Y-%m-%d')
            _periods_df['mean_z'] = _periods_df['mean_z'].round(2)
            _periods_df['delta_hrv'] = _periods_df['delta_hrv'].round(1)

            st.markdown(f"**{len(periods)} períodos detectados** "
                        f"({sum(1 for p in periods if p['tipo']=='HRV↑')} ↑ | "
                        f"{sum(1 for p in periods if p['tipo']=='HRV↓')} ↓)")

            _pu = _periods_df[_periods_df['tipo']=='HRV↑'].tail(int(_show_n//2))
            _pd = _periods_df[_periods_df['tipo']=='HRV↓'].tail(int(_show_n//2))

            _ca, _cb = st.columns(2)
            with _ca:
                st.markdown("**HRV↑ — períodos de recuperação/adaptação**")
                if len(_pu) > 0:
                    st.dataframe(_pu[['start','end','duração','mean_z','delta_hrv']],
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Sem períodos HRV↑")
            with _cb:
                st.markdown("**HRV↓ — períodos de supressão/fadiga**")
                if len(_pd) > 0:
                    st.dataframe(_pd[['start','end','duração','mean_z','delta_hrv']],
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Sem períodos HRV↓")

            # Seleccionar um período para analisar
            st.markdown("---")
            st.markdown("**Analisar em detalhe um período detectado:**")

            # Construir lista local dos períodos a mostrar (mesmos que nas tabelas)
            _n_show   = int(_show_n)
            _show_list = periods[-_n_show:] if len(periods) >= _n_show else periods[:]
            _sel_opts  = [
                f"{p['tipo']} {p['start'].strftime('%d/%m/%Y')} → {p['end'].strftime('%d/%m/%Y')}"
                for p in _show_list
            ]
            _sel = st.selectbox("Seleccionar período", _sel_opts,
                                key="hrv_auto_sel")
            if _sel and len(sig_train) > 0:
                _idx  = _sel_opts.index(_sel)
                # Índice directo na lista local — sem aritmética negativa
                _per  = _show_list[_idx]
                _ts   = pd.Timestamp(_per['start'])
                _te   = pd.Timestamp(_per['end'])
                cmp   = _compare_periods(sig_hrv, sig_train, _ts, _te, 14)
                if not cmp.empty:
                    _sig_rows = cmp[cmp['sig']].sort_values(
                        'Δ%', key=lambda x: pd.to_numeric(x, errors='coerce').abs(),
                        ascending=False).head(10)
                    if len(_sig_rows) > 0:
                        st.markdown(f"**Top mudanças significativas — {_sel}:**")
                        _disp = _sig_rows[['Variável','Antes','Período','Δ%',"Cohen's d"]].copy()
                        _disp['Δ%'] = _disp['Δ%'].apply(
                            lambda x: f"{float(x):+.1f}%" if pd.notna(x) else '—')
                        st.dataframe(_disp, hide_index=True, use_container_width=True)
                    else:
                        st.info("Sem mudanças estatisticamente significativas neste período.")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MODO: LAG CORRELATION
    # ══════════════════════════════════════════════════════════════════════════
    elif _mode == 2:
        st.markdown("### 🔗 Lag Correlation")
        _dropdown_explica('lag')
        st.caption(
            "Qual variável de treino antecede as mudanças de HRV e com quantos dias? "
            "Lag positivo = variável de treino precede HRV."
        )

        if len(sig_train) == 0:
            st.warning("Sem dados de treino.")
        else:
            # Óptimo do lag — lido do gate (já calculado ao clicar "Rodar")
            _lag_opt = 14
            try:
                if _AR_OTIMOS.get('lag_correlation'):
                    _lag_opt = min(max(int(_AR_OTIMOS['lag_correlation']), 3), 21)
                    st.caption(f"🔧 Lag óptimo do Auto-Runner: **{_lag_opt}d** "
                               "(pré-preenchido; podes ajustar).")
            except Exception:
                pass

            _max_lag = st.slider("Lag máximo (dias)", 3, 21, _lag_opt, 1, key="hrv_lag_max")

            # ── Tabela principal: rMSSD vs HF power (sem dropdown de escolha) ──
            try:
                _dual = _hra.lag_correlations_dual(sig_hrv, sig_train, max_lag=_max_lag)
                st.markdown("#### 📊 rMSSD vs HF power — principais achados")
                if _dual.get('tem_hf'):
                    st.caption(
                        "Melhor lag e correlação de cada variável de treino, para o "
                        "rMSSD (rHRV) e para o HF power (mesmo sinal da tab Recovery). "
                        "Se ambos apontam o mesmo lag/sinal → efeito robusto; "
                        "se divergem → o sinal depende da métrica.")
                else:
                    st.caption(
                        "Melhor lag e correlação de cada variável de treino para o rMSSD. "
                        "ℹ️ HF power não disponível na sheet — adiciona a coluna 'hf_power' "
                        "ao wellness para a comparação lado a lado.")
                st.dataframe(_dual['tabela'], hide_index=True, use_container_width=True)
            except Exception as _e_dual:
                st.caption(f"Comparação rMSSD/HF indisponível: {_e_dual}")

            # Detalhe completo (todos os lags) para o rMSSD
            _hrv_target = 'hrv'
            if True:  # corre automaticamente ao ver a secção
                with st.spinner("A calcular correlações com lag..."):
                    lag_df = _cx_lag_correlations(sig_hrv, sig_train,
                                               hrv_var=_hrv_target,
                                               max_lag=_max_lag)

                if lag_df.empty:
                    st.warning("Sem dados suficientes.")
                else:
                    # Melhor lag por variável — groupby().apply() com idxmax()
                    # perde colunas no Pandas 2.x → usar merge explícito
                    _sig_df = lag_df[lag_df['sig']].copy()
                    if len(_sig_df) > 0:
                        _best_idx = _sig_df.groupby('var')['r_abs'].idxmax()
                        best = _sig_df.loc[_best_idx].reset_index(drop=True) \
                                      .sort_values('r_abs', ascending=False)
                    else:
                        best = pd.DataFrame(columns=lag_df.columns)

                    if len(best) > 0:
                        st.markdown("**Top correlações significativas por variável:**")
                        _best_disp = best[['var','lag','r','p']].copy()
                        _best_disp['r']   = _best_disp['r'].round(3)
                        _best_disp['p']   = _best_disp['p'].apply(lambda x: f"{x:.3f}")
                        _best_disp['lag'] = _best_disp['lag'].apply(lambda x: f"{x}d")
                        _best_disp['direcção'] = best['r'].apply(
                            lambda x: '↑ HRV com ↑ variável' if x > 0
                                      else '↑ HRV com ↓ variável')
                        st.dataframe(_best_disp.rename(columns={'var': 'Variável treino',
                                                                  'lag': 'Lag óptimo',
                                                                  'r':   'r Pearson',
                                                                  'p':   'p-valor'}),
                                     hide_index=True, use_container_width=True)

                    # Heatmap lag × variável
                    _lag_pivot = lag_df.pivot(index='var', columns='lag', values='r')
                    if not _lag_pivot.empty:
                        _fig_heat = go.Figure(go.Heatmap(
                            z=_lag_pivot.values,
                            x=[f"lag {l}d" for l in _lag_pivot.columns],
                            y=_lag_pivot.index.tolist(),
                            colorscale='RdBu', zmid=0,
                            zmin=-1, zmax=1,
                            colorbar=dict(title='r', tickfont=dict(color='#111')),
                            hovertemplate='%{y} @ lag %{x}<br>r = <b>%{z:.2f}</b><extra></extra>'
                        ))
                        _fig_heat.update_layout(
                            paper_bgcolor='white', plot_bgcolor='white',
                            font=dict(color='#111', size=10),
                            height=max(300, len(_lag_pivot) * 28 + 80),
                            margin=dict(t=20, b=60, l=120, r=30),
                            xaxis_tickangle=-45,
                        )
                        st.plotly_chart(_fig_heat, use_container_width=True,
                                        config=MC, key='hrv_lag_heat')

                        st.caption(
                            "Azul escuro = correlação positiva forte (↑ variável → ↑ HRV). "
                            "Vermelho escuro = correlação negativa forte (↑ variável → ↓ HRV). "
                            "Lag Xd = variável X dias antes do HRV."
                        )

                    # Download
                    st.download_button(
                        "📥 Download lag correlations",
                        lag_df[lag_df['sig']].round(3).to_csv(
                            index=False, sep=';', decimal=','
                        ).encode('utf-8'),
                        "hrv_lag_correlations.csv", "text/csv",
                        key="hrv_dl_lag"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. MODO: FINGERPRINT HRV
    # ══════════════════════════════════════════════════════════════════════════
    elif _mode == 3:
        st.markdown("### 🧬 Fingerprint — top vs bottom HRV days")
        _dropdown_explica('fingerprint')
        st.caption(
            "O que aconteceu nos X dias antes dos melhores e piores dias de HRV? "
            "Identifica o padrão de treino que antecede a boa forma autonómica."
        )

        if len(sig_train) == 0:
            st.warning("Sem dados de treino.")
        else:
            # Óptimo do fingerprint — lido do gate (já calculado ao clicar "Rodar")
            _fp_opt_dias = 7
            try:
                if _AR_OTIMOS.get('fingerprint_dias'):
                    _fp_opt_dias = min(max(int(_AR_OTIMOS['fingerprint_dias']), 3), 14)
                    st.caption(f"🔧 Óptimo do Auto-Runner: **{_fp_opt_dias}d** antes "
                               "(pré-preenchido; podes ajustar).")
            except Exception:
                pass

            _fp1, _fp2 = st.columns(2)
            _fp_pct  = _fp1.slider("Percentil top/bottom (%)", 5, 25, 10, 5,
                                    key="hrv_fp_pct")
            _fp_pre  = _fp2.slider("Dias antes a analisar", 3, 14, _fp_opt_dias, 1,
                                    key="hrv_fp_pre")

            if True:  # corre automaticamente ao ver a secção
                with st.spinner("A calcular fingerprints..."):
                    fp = _cx_fingerprint(sig_hrv, sig_train,
                                          pct=_fp_pct/100, pre_days=_fp_pre)

                if not fp:
                    st.warning("Sem dados suficientes.")
                else:
                    _var_labels = {
                        'load': 'Carga (TSS)',
                        'kj': 'kJ',
                        'dur_min': 'Duração (min)',
                        'n_sess': 'Nº sessões',
                        'pct_z3': '% Z3',
                        'freq_7d': 'Freq. semanal',
                        'mono_7d': 'Monotonia',
                        'strain_7d': 'Strain',
                        'tsb': 'TSB',
                        'atl': 'ATL',
                    }

                    _fp_rows = []
                    for var, vals in fp.items():
                        _fp_rows.append({
                            'Variável': _var_labels.get(var, var),
                            f'Top {_fp_pct}% HRV': round(vals['top'], 2) if not np.isnan(vals['top']) else '—',
                            f'Bottom {_fp_pct}% HRV': round(vals['bot'], 2) if not np.isnan(vals['bot']) else '—',
                            'Diferença %': f"{vals['diff_pct']:+.1f}%" if not np.isnan(vals['diff_pct']) else '—',
                            '_diff': vals['diff_pct'],
                        })

                    _fp_df = pd.DataFrame(_fp_rows)
                    _fp_df_s = _fp_df.dropna(subset=['_diff']).sort_values('_diff', ascending=False)

                    # Interpretação
                    _pos_patterns = _fp_df_s[_fp_df_s['_diff'] > 15]
                    _neg_patterns = _fp_df_s[_fp_df_s['_diff'] < -15]

                    st.markdown(f"#### Nos {_fp_pre} dias antes dos top {_fp_pct}% HRV:")
                    _fcp1, _fcp2 = st.columns(2)
                    with _fcp1:
                        st.markdown("🟢 **Mais alto nos dias de bom HRV:**")
                        for _, r in _pos_patterns.iterrows():
                            st.markdown(f"→ **{r['Variável']}**: {r['Diferença %']}")
                    with _fcp2:
                        st.markdown("🔴 **Mais baixo nos dias de bom HRV:**")
                        for _, r in _neg_patterns.iterrows():
                            st.markdown(f"→ **{r['Variável']}**: {r['Diferença %']}")

                    # Tabela
                    _fp_disp = _fp_df_s.drop(columns=['_diff'])
                    st.dataframe(_fp_disp, hide_index=True, use_container_width=True)

                    # Barras horizontais
                    _fig_bar = go.Figure()
                    _colors  = [_C['hrv_up'] if d >= 0 else _C['hrv_dn']
                                 for d in _fp_df_s['_diff']]
                    _fig_bar.add_trace(go.Bar(
                        y=_fp_df_s['Variável'],
                        x=_fp_df_s['_diff'],
                        orientation='h',
                        marker_color=_colors,
                        text=[f"{v:+.1f}%" for v in _fp_df_s['_diff']],
                        textposition='outside',
                        hovertemplate='%{y}<br>Diferença: <b>%{x:+.1f}%</b><extra></extra>'
                    ))
                    _fig_bar.add_vline(x=0, line_color='#aaa', line_width=1)
                    _fig_bar.update_layout(
                        paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(color='#111', size=11),
                        height=max(280, len(_fp_df_s) * 32 + 60),
                        margin=dict(t=20, b=40, l=120, r=60),
                        xaxis_title=f"Diferença % (top vs bottom {_fp_pct}% HRV)",
                        yaxis_title=None,
                    )
                    st.plotly_chart(_fig_bar, use_container_width=True,
                                    config=MC, key='hrv_fp_bar')

                    st.caption(
                        f"Verde = variável mais alta nos {_fp_pre}d antes de dias de HRV alto. "
                        f"Vermelho = variável mais baixa antes de dias de HRV alto. "
                        f"Interpretação: os gatilhos positivos são as variáveis verdes."
                    )

                    st.download_button(
                        "📥 Download fingerprint HRV",
                        _fp_disp.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                        f"hrv_fingerprint_top{_fp_pct}.csv", "text/csv",
                        key="hrv_dl_fp"
                    )

    # ── Análises avançadas (só depois de clicar "▶ Rodar" — precisam do auto-runner) ──
    if len(sig_train) == 0:
        st.markdown("---")
        st.info("Conecta os dados de actividade para aceder às análises avançadas "
                "(ARI, Estados, Elasticidade, Lag Avançado, etc.).")
    elif not _ja_correu:
        st.markdown("---")
        st.info("🔬 As **Análises Avançadas** (ARI, Estados, Elasticidade, Lag Avançado, "
                "Directional, Dose-Response, Semanas, Transições) aparecem depois de "
                "clicares em **▶ Rodar Auto-Runner + Avançadas** no topo.")
    else:
        tab_hrv_advanced(sig_hrv, sig_train, da_full=_da,
                         ar_result=_AR_RESULT, ar_otimos=_AR_OTIMOS)

    # ── Nota metodológica ─────────────────────────────────────────────────────
    with st.expander("ℹ️ Metodologia — Recovery Pattern Analyzer"):
        st.markdown(f"""
**Métricas HRV utilizadas:**

| Métrica | Fórmula | Interpretação |
|---|---|---|
| rMSSD | directo | Actividade parassimpática absoluta |
| ln(rMSSD) | log(rMSSD) | Distribuição normal; padrão na literatura |
| AVNN | 60000 / RHR | Espaço temporal por batimento (ms) |
| rMSSD norm. | (rMSSD / AVNN) × 100 | Variabilidade relativa à FC de repouso |
| HRV/RHR ratio | rMSSD / RHR | Coupling autonómico |
| z-score 28d | (HRV - média28d) / std28d | Desvio relativo ao baseline |

**rMSSD normalizado — porquê importa:**
Um rMSSD de 60ms com RHR=60bpm (AVNN=1000ms) dá norm=6.
O mesmo rMSSD=60ms com RHR=40bpm (AVNN=1500ms) dá norm=4.
A variabilidade relativa piorou mesmo com rMSSD estável.

**Lag correlation:**
Correlação de Pearson entre variável de treino (dia t-lag) e HRV (dia t).
Lag óptimo = o lag com maior |r| significativo (p<0.05).

**Event windows:**
Para cada evento detectado, alinha os dados em torno do dia 0 e calcula
a média normalizada de cada variável na janela [-14d, +7d].

**Fingerprint HRV:**
Compara a média de cada variável de treino nos X dias antes dos top 10% HRV days
vs os X dias antes dos bottom 10% HRV days.
Diferença positiva = esta variável está associada a melhor HRV.

**Referências:** Kiviniemi et al. (2007), Hautala et al. (2010),
Plews et al. (2013), Buchheit (2014), Flatt & Esco (2016).
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULOS AVANÇADOS — adicionados após revisão de arquitectura
# ══════════════════════════════════════════════════════════════════════════════

# ── H. Estados Fisiológicos Heurísticos ──────────────────────────────────────

_STATES = {
    'autonomic_suppression': {
        'label': '🔴 Autonomic Suppression',
        'color': '#c0392b',
        'desc': 'HRV colapsado + RHR elevada + strain alto + slope negativo.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) < -1.0 and
            r.get('rhr_z', 0) > 0.8 and
            r.get('strain_7d', 0) > 0 and
            r.get('hrv_slope_7d', 0) < -0.3
        ),
    },
    'accumulated_fatigue': {
        'label': '🟠 Accumulated Fatigue',
        'color': '#e67e22',
        'desc': 'HRV abaixo baseline + ATL elevada + coupling deteriorando.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) < -0.5 and
            r.get('rhr_z', 0) > 0.3 and
            r.get('atl', 0) > r.get('ctl', 1) * 1.1
        ),
    },
    'functional_overreach': {
        'label': '🟡 Functional Overreach',
        'color': '#f39c12',
        'desc': 'HRV variável + monotonia alta + strain elevado. Precisa de unload.',
        'rules': lambda r: (
            r.get('mono_7d', 0) > 2.0 and
            r.get('strain_7d', 0) > 0 and
            abs(r.get('ln_hrv_z', 0)) < 1.0
        ),
    },
    'taper_response': {
        'label': '🟢 Taper Response',
        'color': '#27ae60',
        'desc': 'HRV a subir + RHR a cair + carga reduzida. Forma a emergir.',
        'rules': lambda r: (
            r.get('hrv_slope_7d', 0) > 0.3 and
            r.get('ln_hrv_z', 0) > 0.0 and
            r.get('load_7d', 1) < r.get('load_28d', 1) / 4 * 0.85
        ),
    },
    'parasympathetic_rebound': {
        'label': '💚 Parasympathetic Rebound',
        'color': '#1abc9c',
        'desc': 'HRV bem acima baseline + slope positivo + RHR baixa. Óptimo.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) > 1.0 and
            r.get('hrv_slope_7d', 0) > 0.2 and
            r.get('rhr_z', 0) < 0.0
        ),
    },
    'resilient_state': {
        'label': '🔵 Resilient State',
        'color': '#2980b9',
        'desc': 'HRV estável e acima baseline com carga normal. Adaptado.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) > 0.3 and
            abs(r.get('hrv_slope_7d', 0)) < 0.3 and
            r.get('rhr_z', 0) < 0.3
        ),
    },
    'maladaptation': {
        'label': '⚫ Maladaptation Risk',
        'color': '#2c3e50',
        'desc': 'HRV cronicamente baixo + RHR alta + strain persistente.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) < -0.8 and
            r.get('rhr_z', 0) > 0.5 and
            r.get('mono_7d', 0) > 1.5
        ),
    },
    'baseline': {
        'label': '⚪ Baseline',
        'color': '#95a5a6',
        'desc': 'Estado neutro — sem padrão fisiológico dominante.',
        'rules': lambda r: True,   # fallback
    },
}


_classify_states = _hra._classify_states


# ── I. ARI — Autonomic Readiness Index ───────────────────────────────────────

_ARI_WEIGHTS = {
    'ln_hrv_z':       +0.35,   # HRV logarítmico normalizado
    'rhr_z':          -0.30,   # RHR (negativo: RHR alta = ARI baixo)
    'hrv_norm_z':     +0.20,   # rMSSD norm (variabilidade relativa)
    'instability_z':  -0.10,   # instabilidade HRV 7d (negativo)
    'slope_z':        +0.05,   # slope positivo = melhorando
}

_compute_ari = _hra._compute_ari


# ── J. Recovery Elasticity ────────────────────────────────────────────────────

_recovery_elasticity = _hra._recovery_elasticity


# ── K. Lag Correlation Avançada (Pearson + Spearman + MI) ─────────────────────

_normalized_mi = _hra._normalized_mi


_lag_correlations_advanced = _hra._lag_correlations_advanced


# ── L. Directional Analysis ───────────────────────────────────────────────────

_directional_analysis = _hra._directional_analysis


# ── M. Dose-Response Curves (LOWESS) ─────────────────────────────────────────

_dose_response = _hra._dose_response


# ── N. K-means de semanas ─────────────────────────────────────────────────────

_cluster_weeks = _hra._cluster_weeks


# ── O. Transition Matrix ──────────────────────────────────────────────────────

_transition_matrix = _hra._transition_matrix


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPERS CACHEADOS — auto-run sem recalcular a cada interação
# As análises correm automaticamente (sem botões); o cache garante que só
# recalculam quando os dados de entrada mudam. Nas visitas seguintes = instantâneo.
# ══════════════════════════════════════════════════════════════════════════════

def _cx_lag_correlations(sig_hrv, sig_train, hrv_var='hrv', max_lag=14):
    return _hra._lag_correlations(sig_hrv, sig_train, hrv_var=hrv_var, max_lag=max_lag)

def _cx_fingerprint(sig_hrv, sig_train, pct=0.10, pre_days=10):
    return _hra._hrv_fingerprint(sig_hrv, sig_train, pct=pct, pre_days=pre_days)

def _cx_compute_ari(sig_hrv):
    return _hra._compute_ari(sig_hrv)

def _cx_classify_states(sig_hrv, sig_train):
    return _hra._classify_states(sig_hrv, sig_train)

def _cx_recovery_elasticity(sig_hrv, sig_train, z_suppress=-1.0, z_recover=-0.3):
    return _hra._recovery_elasticity(sig_hrv, sig_train, z_suppress=z_suppress, z_recover=z_recover)

def _cx_lag_advanced(sig_hrv, sig_train, hrv_var='hrv', max_lag=28, train_vars=None):
    return _hra._lag_correlations_advanced(sig_hrv, sig_train, hrv_var=hrv_var,
                                           max_lag=max_lag, train_vars=train_vars)

def _cx_dose_response(sig_hrv, sig_train, x_var, y_var='hrv', lag=0):
    return _hra._dose_response(sig_hrv, sig_train, x_var, y_var=y_var, lag=lag)

# _directional recebe uma lista de padrões (dicts) — chamamos direto (sem cache)
# para evitar problemas de hash; a função é rápida.
def _cx_directional(sig_hrv, sig_train, patterns, outcome_lag=5):
    return _hra._directional_analysis(sig_hrv, sig_train, patterns, outcome_lag=outcome_lag)

def _cx_cluster_weeks(sig_hrv, sig_train, n_clusters=4):
    return _hra._cluster_weeks(sig_hrv, sig_train, n_clusters=n_clusters)

def _cx_autorunner(sig_hrv, sig_train, da_full=None, hoje_ar=None):
    """
    Lê o resultado do auto-runner guardado pelo gate em session_state.
    NÃO recalcula — o cálculo só acontece quando o utilizador clica "▶ Rodar".
    Se ainda não houver resultado guardado, devolve estrutura vazia.
    """
    _gate = st.session_state.get('_hrv_gate')
    if _gate is not None and _gate.get('autorunner') is not None:
        return _gate['autorunner']
    return {'runner_results': [], 'summary_rows': []}


# ══════════════════════════════════════════════════════════════════════════════
# TAB AVANÇADA — adicionar à função tab_hrv_analyzer existente
# ══════════════════════════════════════════════════════════════════════════════

def tab_hrv_advanced(sig_hrv: pd.DataFrame,
                      sig_train: pd.DataFrame,
                      da_full: pd.DataFrame = None,
                      ar_result: dict = None,
                      ar_otimos: dict = None):
    """
    Secção avançada da tab HRV — chamada dentro de tab_hrv_analyzer.
    Contém: ARI, Estados, Elasticidade, Lag Avançado, Directional,
            Dose-Response, K-means, Transition Matrix.
    ar_result/ar_otimos vêm do gate (já calculados ao clicar "▶ Rodar").
    """
    st.markdown("---")
    st.subheader("🧠 Análises Avançadas")

    # ── Óptimos do Auto-Runner — lidos do gate (não recalcula) ────────────────
    # Os sliders usam estes valores como default — o utilizador pode na mesma mexer.
    _otimos = ar_otimos if ar_otimos else {}
    try:
        if _otimos.get('lag_max'):
            st.caption(
                "🔧 Sliders pré-preenchidos com os óptimos do Auto-Runner (período 1 ano). "
                "Podes ajustar à mão. "
                f"Lag máx: **{_otimos.get('lag_max','—')}d** · "
                f"Fingerprint: **{_otimos.get('fingerprint_dias','—')}d** · "
                f"Dose-resp lag: **{_otimos.get('dose_response_lag','—')}d** · "
                f"Directional: **{_otimos.get('directional_janela','—')}d** · "
                f"Clusters: **{_otimos.get('clustering_n','—')}**")
    except Exception:
        _otimos = {}

    def _opt(chave, fallback):
        """Devolve o óptimo do auto-runner para a chave, ou o fallback."""
        v = _otimos.get(chave)
        return v if v is not None else fallback

    _adv_tabs = st.tabs([
        "🎯 ARI",
        "🏷️ Estados",
        "⚡ Elasticidade",
        "🔗 Lag Avançado",
        "➡️ Directional",
        "📈 Dose-Response",
        "🗂️ Semanas",
        "🔄 Transições",
    ])

    # ── ARI ──────────────────────────────────────────────────────────────────
    with _adv_tabs[0]:
        st.markdown("#### 🎯 Autonomic Readiness Index (ARI)")
        _dropdown_explica('ari')
        st.caption(
            "Score composto 0-100 que integra 5 sinais autonómicos. "
            "Média histórica = 50. ARI>60 = boa readiness. ARI<40 = atenção."
        )
        _ari_df = _cx_compute_ari(sig_hrv)

        # Cards actuais
        _ari_now   = _ari_df['ARI'].dropna().iloc[-1] if _ari_df['ARI'].notna().any() else np.nan
        _ari_conf  = _ari_df['ARI_confidence'].dropna().iloc[-1] if len(_ari_df) > 0 else '—'
        _ari_nalign= int(_ari_df['ARI_n_aligned'].dropna().iloc[-1]) if _ari_df['ARI_n_aligned'].notna().any() else 0
        _ari_navail= int(_ari_df['ARI_n_signals'].dropna().iloc[-1]) if _ari_df['ARI_n_signals'].notna().any() else 0

        _ac1, _ac2, _ac3 = st.columns(3)
        _ari_color = ('🟢' if not np.isnan(_ari_now) and _ari_now > 60 else
                      '🟡' if not np.isnan(_ari_now) and _ari_now > 40 else '🔴')
        _ac1.metric("ARI hoje",
                    f"{_ari_color} {_ari_now:.0f}/100" if not np.isnan(_ari_now) else "—",
                    help="0-100. Média histórica=50. >60=boa readiness. <40=atenção.")
        _ac2.metric("Confidence",
                    str(_ari_conf),
                    delta=f"Sinais alinhados: {_ari_nalign}/{_ari_navail}",
                    delta_color="normal" if _ari_nalign >= 3 else "off",
                    help="Quantos dos 5 sinais estão alinhados na mesma direcção.")
        _ac3.metric("Pesos ARI",
                    "ln(HRV)×0.35 | RHR×0.30",
                    help="ARI = 0.35z(ln_rMSSD) - 0.30z(RHR) + 0.20z(norm) - 0.10z(instab) + 0.05z(slope)")

        # Série ARI
        _fig_ari = go.Figure()
        _fig_ari.add_hrect(y0=60, y1=100, fillcolor='rgba(39,174,96,0.08)',
                            line_width=0, name='Zona óptima')
        _fig_ari.add_hrect(y0=0, y1=40, fillcolor='rgba(231,76,60,0.08)',
                            line_width=0, name='Zona de atenção')
        _fig_ari.add_hline(y=50, line_dash='dot', line_color='#aaa', line_width=1)
        _fig_ari.add_trace(go.Scatter(
            x=_ari_df['Data'], y=_ari_df['ARI'],
            mode='lines', name='ARI',
            line=dict(color=_C['primary'], width=2.5),
            fill='tozeroy', fillcolor='rgba(41,128,185,0.08)',
            hovertemplate='%{x|%d/%m/%Y}<br>ARI: <b>%{y:.0f}</b><extra></extra>'
        ))
        _fig_ari.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=11), height=320,
            margin=dict(t=20, b=50, l=50, r=20),
            yaxis=dict(range=[0, 100], title='ARI', tickfont=dict(color='#111'),
                       showgrid=True, gridcolor='#eee'),
            xaxis=dict(tickfont=dict(color='#111'), showgrid=True, gridcolor='#eee'),
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.18,
                        font=dict(color='#111', size=10)),
        )
        st.plotly_chart(_fig_ari, use_container_width=True, config=MC, key='ari_series')

        # Download ARI
        _ari_dl = _ari_df[['Data','hrv','ln_hrv','rhr','ARI',
                              'ARI_n_signals','ARI_n_aligned','ARI_confidence']].copy()
        _ari_dl['Data'] = _ari_dl['Data'].astype(str)
        st.download_button("📥 Download ARI diário",
                           _ari_dl.round(3).to_csv(index=False,sep=';',decimal=',').encode(),
                           "atheltica_ari.csv","text/csv", key="ari_dl")

        with st.expander("ℹ️ Fórmula ARI"):
            st.markdown("""
| Componente | Peso | Interpretação |
|---|---|---|
| z(ln_rMSSD) | +0.35 | HRV logarítmico — sinal principal |
| z(RHR) | **-0.30** | RHR alta = ARI baixo |
| z(rMSSD_norm) | +0.20 | Variabilidade relativa à FC |
| z(instabilidade_7d) | **-0.10** | Instabilidade HRV = stress |
| z(slope_7d) | +0.05 | Tendência positiva = melhorando |

Escalado: média 90d = 50, ±2σ ≈ ±30 pontos.
Confidence = nº de sinais alinhados na mesma direcção (0-5).
            """)

    # ── ESTADOS ──────────────────────────────────────────────────────────────
    with _adv_tabs[1]:
        st.markdown("#### 🏷️ Estados Fisiológicos Heurísticos")
        _dropdown_explica('estados')
        st.caption(
            "7 estados detectados por regras fisiológicas. "
            "Interpretáveis e accionáveis sem modelo probabilístico."
        )

        _state_df = _cx_classify_states(sig_hrv, sig_train)

        # Timeline de estados
        _fig_st = go.Figure()
        for state_key, state_info in _STATES.items():
            _mask = _state_df['state'] == state_key
            if _mask.sum() == 0:
                continue
            _sub = _state_df[_mask]
            _fig_st.add_trace(go.Scatter(
                x=_sub['Data'], y=[state_info['label']] * len(_sub),
                mode='markers',
                marker=dict(size=8, color=state_info['color'],
                            line=dict(width=1, color='white')),
                name=state_info['label'],
                hovertemplate='%{x|%d/%m/%Y}<br>%{y}<extra></extra>'
            ))

        _fig_st.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10), height=320,
            margin=dict(t=20, b=50, l=200, r=20),
            xaxis=dict(tickfont=dict(color='#111'), showgrid=True, gridcolor='#eee'),
            yaxis=dict(tickfont=dict(color='#111')),
            legend=dict(orientation='h', y=-0.22,
                        font=dict(color='#111', size=9)),
            hovermode='closest',
        )
        st.plotly_chart(_fig_st, use_container_width=True, config=MC, key='state_timeline')

        # Distribuição de estados
        _st_counts = _state_df['state'].value_counts().reset_index()
        _st_counts.columns = ['state','n']
        _st_counts['label'] = _st_counts['state'].map(
            {k: v['label'] for k, v in _STATES.items()})
        _st_counts['color'] = _st_counts['state'].map(
            {k: v['color'] for k, v in _STATES.items()})
        _st_counts['pct']   = (_st_counts['n'] / len(_state_df) * 100).round(1)

        _fc1, _fc2 = st.columns([2, 3])
        with _fc1:
            st.markdown("**Distribuição**")
            st.dataframe(
                _st_counts[['label','n','pct']].rename(
                    columns={'label':'Estado','n':'Dias','pct':'%'}),
                hide_index=True, use_container_width=True)

        with _fc2:
            st.markdown("**Definições**")
            for k, v in _STATES.items():
                if k == 'baseline': continue
                st.markdown(f"**{v['label']}** — {v['desc']}")

        # Estado actual
        _today_state = _state_df['state_label'].dropna().iloc[-1] if len(_state_df) > 0 else '—'
        _today_desc  = _STATES.get(_state_df['state'].iloc[-1], {}).get('desc', '')
        st.info(f"**Estado actual:** {_today_state}\n\n{_today_desc}")

        # Download estados
        _st_dl = _state_df[['Data','state_label','state',
                              'hrv','ln_hrv','hrv_norm','rhr',
                              'hrv_z28','hrv_slope_7d']].copy()
        _st_dl['Data'] = _st_dl['Data'].astype(str)
        st.download_button(
            "📥 Download estados fisiológicos diários",
            _st_dl.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
            "atheltica_hrv_estados.csv", "text/csv", key="adv_estados_dl"
        )

    # ── ELASTICIDADE ─────────────────────────────────────────────────────────
    with _adv_tabs[2]:
        st.markdown("#### ⚡ Recovery Elasticity")
        _dropdown_explica('elasticidade')
        st.caption(
            "τ_recovery = dias até HRV voltar ao baseline após supressão. "
            "Assinatura individual: τ baixo = recuperas rápido."
        )

        _ez1, _ez2 = st.columns(2)
        _z_supp = _ez1.slider("z supressão (trigger)", -2.0, -0.5,
                              -abs(_opt("elasticidade_z", 1.0)) if _opt("elasticidade_z", None) else -1.0,
                              0.1, key="elast_z_supp")
        _z_rec  = _ez2.slider("z recuperação (target)", -0.5, 0.5, -0.3, 0.1,
                               key="elast_z_rec")

        elast = _cx_recovery_elasticity(sig_hrv, sig_train,
                                      z_suppress=_z_supp, z_recover=_z_rec)

        if elast['n_events'] == 0:
            st.info("Sem eventos de supressão detectados com estes critérios.")
        else:
            _ec1, _ec2, _ec3, _ec4 = st.columns(4)
            _ec1.metric("τ mediana",
                        f"{elast['tau_median']}d" if elast['tau_median'] else "—",
                        help="Mediana de dias para recuperar após supressão.")
            _ec2.metric("τ média",
                        f"{elast['tau_mean']}d" if elast['tau_mean'] else "—")
            _ec3.metric("Eventos",
                        f"{elast['n_events']}",
                        delta=f"{elast['n_recovered']} recuperados",
                        delta_color="normal")
            _ec4.metric("Taxa recuperação",
                        f"{elast['n_recovered']/elast['n_events']*100:.0f}%"
                        if elast['n_events'] > 0 else "—")

            # Tabela de eventos
            if elast['events']:
                _ev_df = pd.DataFrame(elast['events'])
                _ev_df['date'] = _ev_df['date'].astype(str)
                _ev_df['recovered'] = _ev_df['recovered'].map({True: '✅', False: '❌'})
                st.dataframe(
                    _ev_df.rename(columns={
                        'date': 'Data',
                        'days_to_recovery': 'Dias até recuperar',
                        'suppression_depth': 'Profundidade z',
                        'recovered': 'Recuperou',
                    }),
                    hide_index=True, use_container_width=True)

            # Histogram de τ
            _rec_days = [e['days_to_recovery'] for e in elast['events'] if e['recovered']]
            if _rec_days:
                _fig_hist = go.Figure(go.Histogram(
                    x=_rec_days, nbinsx=min(10, len(_rec_days)),
                    marker_color=_C['primary'],
                    marker_line_color='white', marker_line_width=1,
                    hovertemplate='%{x}d: <b>%{y}</b> eventos<extra></extra>'
                ))
                if elast['tau_median']:
                    _fig_hist.add_vline(x=elast['tau_median'],
                                        line_dash='dash', line_color='#e74c3c',
                                        line_width=2,
                                        annotation_text=f"τ={elast['tau_median']}d",
                                        annotation_font=dict(color='#e74c3c'))
                _fig_hist.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=11), height=260,
                    margin=dict(t=20, b=50, l=50, r=20),
                    xaxis=dict(title='Dias até recuperar',
                               tickfont=dict(color='#111'), showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='Nº eventos', tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                )
                st.plotly_chart(_fig_hist, use_container_width=True,
                                config=MC, key='elast_hist')
                st.caption(
                    f"τ mediana = {elast['tau_median']}d. "
                    "Este é o teu tempo típico de recuperação HRV. "
                    "Usa-o para planear o intervalo entre blocos de carga intensos."
                )
                # Download elasticidade
                _el_dl = pd.DataFrame(elast['events'])
                if not _el_dl.empty:
                    _el_dl['date'] = _el_dl['date'].astype(str)
                    _el_meta = pd.DataFrame([{
                        'metrica': 'tau_mediana_dias', 'valor': elast['tau_median']},
                        {'metrica': 'tau_media_dias',   'valor': elast['tau_mean']},
                        {'metrica': 'n_eventos',         'valor': elast['n_events']},
                        {'metrica': 'n_recuperados',     'valor': elast['n_recovered']},
                        {'metrica': 'taxa_recuperacao_pct',
                         'valor': round(elast['n_recovered']/elast['n_events']*100,1)
                                  if elast['n_events'] > 0 else 0},
                    ])
                    st.download_button(
                        "📥 Download Recovery Elasticity",
                        (_el_dl.to_csv(index=False, sep=';', decimal=',') +
                         '\n--- Métricas ---\n' +
                         _el_meta.to_csv(index=False, sep=';', decimal=',')
                        ).encode('utf-8'),
                        "atheltica_hrv_elasticidade.csv", "text/csv",
                        key="adv_elast_dl"
                    )

    # ── LAG AVANÇADO ─────────────────────────────────────────────────────────
    with _adv_tabs[3]:
        st.markdown("#### 🔗 Lag Correlation Avançada (Pearson + Spearman + MI)")
        _dropdown_explica('lag_avancado')
        st.caption(
            "Pearson: magnitude linear. "
            "Spearman: robusto a outliers. "
            "MI normalizada: detecta relações não-lineares (U-shape dose-resposta)."
        )

        _lv1, _lv2 = st.columns(2)
        _lv_tgt  = _lv1.selectbox(
            "HRV alvo", [v for v in ['hrv','ln_hrv','hrv_norm'] if v in sig_hrv.columns],
            key="adv_lag_tgt")
        _lv_max  = _lv2.slider("Lag máximo (d)", 3, 35, min(max(_opt("lag_max",10),3),35), 1, key="adv_lag_max")

        if True:  # corre automaticamente ao ver a secção
            with st.spinner("Pearson + Spearman + MI..."):
                adv_lag = _cx_lag_advanced(
                    sig_hrv, sig_train, hrv_var=_lv_tgt, max_lag=_lv_max)

            if adv_lag.empty:
                st.warning("Sem dados suficientes.")
            else:
                # Top por cada método
                st.markdown("**Melhor lag por variável — comparação dos 3 métodos:**")

                def _best_per_var(df, r_col, sig_col):
                    sub = df[df[sig_col]].copy() if sig_col in df.columns else df.copy()
                    if sub.empty: return pd.DataFrame()
                    idx = sub.groupby('var')[r_col].apply(lambda x: x.abs().idxmax())
                    return sub.loc[idx].reset_index(drop=True)

                _bp = _best_per_var(adv_lag, 'r_pearson',  'sig_pearson')
                _bs = _best_per_var(adv_lag, 'r_spearman', 'sig_spearman')

                _comp_rows = []
                for var in adv_lag['var'].unique():
                    _rp  = _bp[_bp['var']==var]['r_pearson'].values
                    _lp  = _bp[_bp['var']==var]['lag'].values
                    _rs  = _bs[_bs['var']==var]['r_spearman'].values
                    _ls  = _bs[_bs['var']==var]['lag'].values
                    _mi  = adv_lag[adv_lag['var']==var]['mi_norm'].max()
                    _comp_rows.append({
                        'Variável': var,
                        'r Pearson': f"{_rp[0]:+.3f} @{_lp[0]}d" if len(_rp) else '—',
                        'r Spearman': f"{_rs[0]:+.3f} @{_ls[0]}d" if len(_rs) else '—',
                        'MI norm max': f"{_mi:.3f}" if not np.isnan(_mi) else '—',
                        'MI>Pearson?': ('✅' if not np.isnan(_mi) and
                                         _mi > (abs(_rp[0]) if len(_rp) else 0) else ''),
                    })

                st.dataframe(pd.DataFrame(_comp_rows), hide_index=True,
                             use_container_width=True)
                st.caption(
                    "MI>Pearson? = a MI detectou relação mais forte que Pearson "
                    "(possível não-linearidade / dose-resposta em U)."
                )

                st.download_button(
                    "📥 Download lag correlation avançada",
                    adv_lag.to_csv(index=False, sep=';', decimal=',').encode(),
                    "hrv_lag_advanced.csv", "text/csv", key="adv_lag_dl")

    # ── DIRECTIONAL ───────────────────────────────────────────────────────────
    with _adv_tabs[4]:
        st.markdown("#### ➡️ Directional Analysis")
        _dropdown_explica('directional')
        st.caption(
            "Padrões de treino → probabilidade de HRV melhorar nos X dias seguintes. "
            "⚠️ Ferramenta de geração de hipóteses, não inferência causal. "
            "N pequeno → confidence baixa."
        )

        # Padrões pré-definidos fisiologicamente
        _DEFAULT_PATTERNS = [
            {'name': 'Monotonia↓ >15%',
             'conditions': [{'var': 'mono_7d_delta', 'op': '<', 'val': -0.15}]},
            {'name': 'Carga↓ + Freq.↓',
             'conditions': [{'var': 'load_7d_delta', 'op': '<', 'val': -0.20},
                             {'var': 'freq_7d', 'op': '<', 'val': 4}]},
            {'name': 'Alta Monotonia (>2.0)',
             'conditions': [{'var': 'mono_7d', 'op': '>', 'val': 2.0}]},
            {'name': 'Strain↓ + Z2↑ (pct_z3<30%)',
             'conditions': [{'var': 'strain_7d_delta', 'op': '<', 'val': -0.15},
                             {'var': 'pct_z3', 'op': '<', 'val': 30}]},
            {'name': 'TSB positivo (>+5)',
             'conditions': [{'var': 'tsb', 'op': '>', 'val': 5}]},
            {'name': 'Carga muito elevada (ATL>CTL×1.2)',
             'conditions': [{'var': 'atl', 'op': '>', 'val': 0}]},  # proxy
        ]

        # ── Janela default = tau da elasticidade (tempo típico de recuperação) ──
        # Fisiologicamente, a janela de outcome deve ser o tempo que o HRV demora
        # a recuperar. Usamos o tau mediano da elasticidade como default.
        _tau_default = _opt("directional_janela", 5)
        try:
            _elast_dir = _cx_recovery_elasticity(sig_hrv, sig_train)
            _tau_med = _elast_dir.get('tau_median')
            if _tau_med and not (isinstance(_tau_med, float) and np.isnan(_tau_med)):
                _tau_default = int(min(max(round(_tau_med), 3), 14))
                st.caption(f"🔧 Janela pré-preenchida com o **τ da elasticidade** "
                           f"(tempo mediano de recuperação = **{_tau_default}d**). "
                           "Fisiologicamente, é o tempo que o HRV demora a recuperar. "
                           "Podes ajustar.")
        except Exception:
            pass

        _dp_lag = st.slider("Janela de outcome (dias)", 3, 14,
                            min(max(_tau_default, 3), 14), 1, key="dir_lag")

        if True:  # corre automaticamente ao ver a secção
            _dir_res = _cx_directional(
                sig_hrv, sig_train, _DEFAULT_PATTERNS, outcome_lag=_dp_lag)

            _dir_df = pd.DataFrame([{
                'Padrão':       r['pattern'],
                'Ocorrências':  r['n_occur'],
                'HRV melhorou': r['n_improve'],
                'Consistência': f"{r['consistency']}%",
                'Confidence':   r['confidence'],
            } for r in _dir_res])

            if not _dir_df.empty:
                st.dataframe(_dir_df, hide_index=True, use_container_width=True)
                st.caption(
                    "Consistência = % das ocorrências seguidas de HRV melhorado "
                    f"nos {_dp_lag} dias seguintes. "
                    "Confidence indica o tamanho amostral — N<10 é exploratório."
                )
                st.download_button(
                    "📥 Download Directional Analysis",
                    _dir_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    f"atheltica_hrv_directional_lag{_dp_lag}d.csv", "text/csv",
                    key="adv_dir_dl"
                )

    # ── DOSE-RESPONSE ─────────────────────────────────────────────────────────
    with _adv_tabs[5]:
        st.markdown("#### 📈 Dose-Response Curves (LOWESS)")
        _dropdown_explica('dose_response')
        st.caption(
            "Relação não-linear entre variável de treino e HRV. "
            "LOWESS detecta U-shapes (ex: carga baixa/alta → HRV ruim, carga moderada → HRV óptimo)."
        )

        _dr1, _dr2, _dr3 = st.columns(3)
        _dr_xvar = _dr1.selectbox(
            "Variável de treino (X)",
            [v for v in ['mono_7d','strain_7d','load_7d','pct_z3','freq_7d','atl','tsb']
             if v in sig_train.columns],
            key="dr_xvar")
        _dr_lag  = _dr2.slider("Lag (dias)", 0, 28, min(max(_opt("dose_response_lag",3),0),28), 1, key="dr_lag")
        _dr_yvar = _dr3.selectbox(
            "HRV alvo (Y)",
            [v for v in ['hrv','ln_hrv','hrv_norm'] if v in sig_hrv.columns],
            key="dr_yvar")

        if True:  # corre automaticamente ao ver a secção
            dr = _cx_dose_response(sig_hrv, sig_train, _dr_xvar, _dr_yvar, _dr_lag)

            if dr.empty:
                st.warning("Dados insuficientes.")
            else:
                _fig_dr = go.Figure()
                # Scatter pontos reais
                _fig_dr.add_trace(go.Scatter(
                    x=dr['x_raw'], y=dr['y_raw'],
                    mode='markers', name='Dados',
                    marker=dict(size=5, color='rgba(41,128,185,0.3)',
                                line=dict(width=0)),
                    hovertemplate=f'{_dr_xvar}: %{{x:.2f}}<br>{_dr_yvar}: %{{y:.1f}}<extra></extra>'
                ))
                # Curva LOWESS
                _fig_dr.add_trace(go.Scatter(
                    x=dr['x'], y=dr['y_smooth'],
                    mode='lines', name='LOWESS',
                    line=dict(color='#e74c3c', width=3),
                    hovertemplate=f'{_dr_xvar}: %{{x:.2f}}<br>HRV smooth: %{{y:.1f}}<extra></extra>'
                ))
                _fig_dr.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=11), height=360,
                    margin=dict(t=20, b=60, l=60, r=20),
                    xaxis=dict(title=_dr_xvar, tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title=f'{_dr_yvar} (lag={_dr_lag}d)',
                               tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                    legend=dict(orientation='h', y=-0.18,
                                font=dict(color='#111', size=10)),
                )
                st.plotly_chart(_fig_dr, use_container_width=True,
                                config=MC, key='dr_plot')
                st.caption(
                    f"X = {_dr_xvar} no dia t. "
                    f"Y = {_dr_yvar} no dia t+{_dr_lag}. "
                    "Curva LOWESS capta relações não-lineares — pico = zona óptima."
                )
                st.download_button(
                    "📥 Download Dose-Response (dados + curva LOWESS)",
                    dr.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    f"atheltica_hrv_dose_response_{_dr_xvar}_lag{_dr_lag}d.csv",
                    "text/csv", key="adv_dr_dl"
                )

    # ── K-MEANS ───────────────────────────────────────────────────────────────
    with _adv_tabs[6]:
        st.markdown("#### 🗂️ Clustering de Semanas")
        _dropdown_explica('semanas')
        st.caption(
            "K-means sobre variáveis de TREINO (sem HRV). "
            "Clusters coloridos pelo HRV médio da semana seguinte. "
            "⚠️ N pequeno (~100 semanas) — interpretar com cautela."
        )

        _kk1, _kk2 = st.columns(2)
        _n_clust = _kk1.slider("Nº clusters", 2, 7, min(max(_opt("clustering_n",4),2),7), 1, key="km_n")

        try:
            import sklearn
            _has_sklearn = True
        except ImportError:
            _has_sklearn = False

        if not _has_sklearn:
            st.warning("sklearn não disponível. Instala scikit-learn para usar esta análise.")
        elif _has_sklearn:  # auto-run (era botão)
            with st.spinner("K-means..."):
                wk_df = _cx_cluster_weeks(sig_hrv, sig_train, n_clusters=_n_clust)

            if wk_df.empty:
                st.warning("Dados insuficientes (mínimo 12 semanas completas).")
            else:
                # Tabela de características por cluster
                _feat_cols = ['load_total','mono_mean','freq','pct_z3','strain_mean','hrv_next']
                _clust_summary = wk_df.groupby('cluster_label')[_feat_cols].mean().round(2)
                st.markdown("**Características médias por cluster:**")
                st.dataframe(_clust_summary, use_container_width=True)

                # Scatter semanas ao longo do tempo
                _cmap = {'🟢 Semana Óptima': '#27ae60', '🟡 Semana Boa': '#f39c12',
                          '🟠 Semana de Atenção': '#e67e22', '🔴 Semana Difícil': '#e74c3c'}
                _fig_km = go.Figure()
                for lbl, color in _cmap.items():
                    _sub = wk_df[wk_df['cluster_label'] == lbl]
                    if len(_sub) == 0: continue
                    _fig_km.add_trace(go.Scatter(
                        x=_sub['week'].astype(str), y=_sub['hrv_next'],
                        mode='markers', name=lbl,
                        marker=dict(size=9, color=color,
                                    line=dict(width=1, color='white')),
                        hovertemplate='Semana %{x}<br>HRV seguinte: <b>%{y:.1f}</b><extra></extra>'
                    ))
                _fig_km.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=10), height=320,
                    margin=dict(t=20, b=80, l=60, r=20),
                    xaxis=dict(tickangle=-45, tickfont=dict(color='#111', size=8),
                               showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='HRV semana seguinte', tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                    legend=dict(orientation='h', y=-0.30,
                                font=dict(color='#111', size=9)),
                )
                st.plotly_chart(_fig_km, use_container_width=True,
                                config=MC, key='km_scatter')

                _current_week = pd.Timestamp.now().to_period('W')
                _current_row  = wk_df[wk_df['week'] == _current_week]
                if len(_current_row) > 0:
                    _clbl = _current_row['cluster_label'].values[0]
                    st.info(f"**Semana actual:** {_clbl}")

                # Download K-means
                _km_dl = wk_df[['week','cluster_label','load_total','mono_mean',
                                  'freq','pct_z3','strain_mean',
                                  'hrv_mean','hrv_next']].copy()
                _km_dl['week'] = _km_dl['week'].astype(str)
                st.download_button(
                    "📥 Download clustering de semanas",
                    _km_dl.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    f"atheltica_hrv_clusters_{_n_clust}k.csv", "text/csv",
                    key="adv_km_dl"
                )

    # ── TRANSIÇÕES ───────────────────────────────────────────────────────────
    with _adv_tabs[7]:
        st.markdown("#### 🔄 Probabilistic Transition Matrix")
        _dropdown_explica('transicoes')
        st.caption(
            "P(estado_amanhã | estado_hoje). "
            "Alternativa ao Sankey — mostra probabilidades reais entre estados."
        )

        _state_df2 = _cx_classify_states(sig_hrv, sig_train)
        _state_labels = _state_df2['state_label'].dropna()

        if len(_state_labels) < 10:
            st.warning("Dados insuficientes para transition matrix.")
        else:
            _tm = _transition_matrix(_state_df2['state_label'])

            # ── Transição a partir do estado de HOJE (o mais prático) ─────────
            try:
                _th = _hra.transicao_de_hoje(_state_df2['state_label'], top_n=3)
                if _th.get('estado_hoje'):
                    st.markdown("##### 🎯 A partir do teu estado de HOJE")
                    st.markdown(f"Estado actual: **{_th['estado_hoje']}**")
                    if _th['proximos']:
                        _prox_df = pd.DataFrame([
                            {'Próximo estado mais provável': p['estado'],
                             'Probabilidade': f"{p['prob']*100:.0f}%"}
                            for p in _th['proximos']
                        ])
                        st.dataframe(_prox_df, hide_index=True, use_container_width=True)
                        _top = _th['proximos'][0]
                        st.caption(f"➡️ Amanhã, o estado mais provável é "
                                   f"**{_top['estado']}** ({_top['prob']*100:.0f}%).")
                    st.markdown("---")
            except Exception:
                pass

            if not _tm.empty:
                # Heatmap da transition matrix
                _fig_tm = go.Figure(go.Heatmap(
                    z=_tm.values,
                    x=list(_tm.columns),
                    y=list(_tm.index),
                    colorscale='Blues',
                    zmin=0, zmax=1,
                    text=_tm.round(2).values,
                    texttemplate='%{text}',
                    colorbar=dict(title='P', tickfont=dict(color='#111')),
                    hovertemplate='De: %{y}<br>Para: %{x}<br>P = <b>%{z:.2f}</b><extra></extra>'
                ))
                _fig_tm.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=9),
                    height=max(320, len(_tm) * 40 + 100),
                    margin=dict(t=20, b=120, l=220, r=20),
                    xaxis=dict(tickangle=-35, tickfont=dict(color='#111', size=8)),
                    yaxis=dict(tickfont=dict(color='#111', size=8)),
                )
                st.plotly_chart(_fig_tm, use_container_width=True,
                                config=MC, key='tm_heat')

                # Insights das transições mais prováveis
                st.markdown("**Transições mais prováveis (P > 0.40):**")
                _trans_rows = []
                for frm in _tm.index:
                    for to in _tm.columns:
                        p = _tm.loc[frm, to]
                        if p > 0.40 and frm != to:
                            _trans_rows.append({
                                'De': frm, 'Para': to, 'P': f"{p:.2f}"})
                if _trans_rows:
                    st.dataframe(pd.DataFrame(_trans_rows), hide_index=True,
                                 use_container_width=True)
                else:
                    st.info("Sem transições com P>0.40 (estados muito distribuídos).")

                st.caption(
                    "Lê-se por linha: dado que hoje estás em estado X, "
                    "qual a probabilidade de amanhã estar em Y? "
                    "Diagonal = auto-persistência do estado."
                )
                st.download_button(
                    "📥 Download Transition Matrix",
                    _tm.to_csv(sep=';', decimal=',').encode('utf-8'),
                    "atheltica_hrv_transition_matrix.csv", "text/csv",
                    key="adv_tm_dl"
                )

    # ════════════════════════════════════════════════════════════════════════
    # AUTO-RUNNER — optimização automática de todos os parâmetros
    # ════════════════════════════════════════════════════════════════════════
    _adv_tabs_full = st.tabs(["🔬 Auto-Runner — parâmetros óptimos por período"])

    with _adv_tabs_full[0]:
        st.markdown("#### 🔬 Auto-Runner — Optimização automática de parâmetros")
        _dropdown_explica('autorunner')
        st.caption(
            "Roda todas as análises para **7 períodos** (60d / 90d / 180d / 1ano / "
            "2anos / 3anos / todo histórico) testando automaticamente múltiplas "
            "combinações de parâmetros. Detecta os valores óptimos por variável e período. "
            "Output: CSV consolidado com todos os resultados + comparação entre períodos."
        )
        st.caption("✅ Já calculado ao clicar em **▶ Rodar Auto-Runner + Avançadas** — resultados abaixo.")

        # Lê o resultado guardado pelo gate (não recalcula)
        _res_ar = ar_result if ar_result else {}
        _runner_results = _res_ar.get('runner_results', [])
        _summary_rows   = _res_ar.get('summary_rows', [])

        if True:  # bloco de display (indentação preservada)
            if True:
                # ── Display resumo ────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📊 Resumo — parâmetros óptimos por período")
                with st.expander("ℹ️ O que é esta tabela e o que significa cada coluna"):
                    st.markdown(
                        "Cada **linha** é um período de análise (60d, 90d, ..., todo o "
                        "histórico). Cada **coluna** é o valor óptimo encontrado nesse período:\n\n"
                        "- **N dias HRV** — dias com dados nesse período\n"
                        "- **Lag máx óptimo** — desfasamento (dias) em que o treino melhor "
                        "prevê o HRV\n"
                        "- **N clusters óptimo** — nº de 'tipos de semana' distintos\n"
                        "- **Janela directional (d)** — janela onde os padrões têm mais efeito\n"
                        "- **Consist. directional** — % de vezes que os padrões acertaram\n"
                        "- **N eventos directional** — quantos eventos entraram nessa % (o N)\n"
                        "- **Target Z / Tau elast.** — gatilho de supressão e tempo de recuperação\n"
                        "- **Melhor preditor ↘ HRV** — variável de treino que mais baixa o HRV "
                        "(com r e lag)\n"
                        "- **FP: var mais discriminante** — variável que mais separa bons de maus dias\n"
                        "- **N lags sig p<0.05** — nº de correlações estatisticamente significativas\n\n"
                        "**⚠️ Cuidado com o N grande (o aviso mais importante):**\n"
                        "A coluna *Consist. directional* depende muito do nº de eventos. Com "
                        "**'todo histórico'** (milhares de dias) a consistência sai alta (83-96%) "
                        "— mas isso pode ser só por teres muitos dados. Se a mesma análise cai "
                        "para **~52% com '1 ano'** (basicamente moeda ao ar), então **não é um "
                        "sinal causal real** — é artefacto do N grande. Um efeito verdadeiro "
                        "mantém-se forte em ambos os períodos. **Regra:** desconfia de padrões "
                        "que só são fortes com todo o histórico; compara sempre com 1 ano."
                    )

                if _summary_rows:
                    _df_sum = pd.DataFrame(_summary_rows)
                    st.dataframe(_df_sum, hide_index=True, use_container_width=True)

                    # ── Fingerprint top por período ────────────────────────────────
                    _fp_all = [r for r in _runner_results
                               if r['analise']=='fingerprint' and r.get('r_abs')]
                    if _fp_all:
                        st.markdown("### 👆 Fingerprint — variáveis mais discriminantes (1 ano)")
                        with st.expander("ℹ️ O que mostra esta tabela"):
                            st.markdown(
                                "As variáveis de treino que **mais distinguem** os teus melhores "
                                "dias de HRV dos piores, nos dias que os antecedem. Um valor alto "
                                "(%) significa que essa variável estava muito diferente antes dos "
                                "bons dias vs antes dos maus dias — logo, é um bom 'marcador' "
                                "antecipatório da tua recuperação.")
                        _df_fp = pd.DataFrame(_fp_all)
                        _fp_1a = (_df_fp[_df_fp['periodo']=='1 ano']
                                  .nlargest(10,'r_abs')
                                  [['variavel','param_val','r_pearson','n','nota']]
                                  .rename(columns={
                                      'variavel':'Variável','param_val':'Dias antes',
                                      'r_pearson':'Diff% HRV alto vs baixo','n':'N dias'}))
                        st.dataframe(_fp_1a, hide_index=True, use_container_width=True)

                    # ── Dose-Response por período ──────────────────────────────────
                    _dr_all = [r for r in _runner_results if r['analise']=='dose_response']
                    if _dr_all:
                        st.markdown("### 📈 Dose-Response — quartil óptimo de carga (1 ano)")
                        with st.expander("ℹ️ O que mostra esta tabela"):
                            st.markdown(
                                "Divide a carga em quartis (Q1=baixa … Q4=alta) e mostra o HRV "
                                "médio associado a cada nível. Revela o teu **'ponto óptimo'**: "
                                "o quartil de carga que coincide com melhor HRV. Ajuda a perceber "
                                "até onde podes empurrar a carga antes de a recuperação sofrer.")
                        _df_dr = pd.DataFrame(_dr_all)
                        _df_dr_1a = _df_dr[_df_dr['periodo']=='1 ano']
                        if len(_df_dr_1a) > 0:
                            st.dataframe(
                                _df_dr_1a[['variavel','param_val','r_spearman','n','nota']]
                                .rename(columns={
                                    'variavel':'Variável','param_val':'Lag óptimo (d)',
                                    'r_spearman':'r Spearman','n':'N pares'}),
                                hide_index=True, use_container_width=True)

                    # ── Análise de divergências automática ────────────────────────
                    st.markdown("### 🔍 Divergências entre períodos")
                    with st.expander("ℹ️ O que mostra esta tabela"):
                        st.markdown(
                            "Compara os resultados entre períodos curtos e longos e assinala "
                            "onde **divergem muito**. É a ferramenta-chave contra o efeito do N "
                            "grande: se um achado é forte no histórico todo mas fraco em 1 ano, "
                            "aparece aqui como divergência — sinal de que pode não ser um efeito "
                            "real. Achados que se mantêm em todos os períodos são os mais fiáveis.")
                    _div_rows = []

                    # Lag máximo
                    _lag_vals = [r['Lag máx óptimo'] for r in _summary_rows]
                    if max(_lag_vals) - min(_lag_vals) >= 7:
                        _div_rows.append({
                            'Parâmetro': 'Lag máximo óptimo',
                            'Min': f"{min(_lag_vals)}d",
                            'Max': f"{max(_lag_vals)}d",
                            'Divergência': '⚠️ Alta — lag de resposta ao HRV mudou ao longo do tempo',
                        })
                    else:
                        _div_rows.append({
                            'Parâmetro': 'Lag máximo óptimo',
                            'Min': f"{min(_lag_vals)}d",
                            'Max': f"{max(_lag_vals)}d",
                            'Divergência': '✅ Estável entre períodos',
                        })

                    # Target Z — usar chave correcta 'Target Z'
                    _tz_vals = [r['Target Z'] for r in _summary_rows
                                if r['Período'] != 'Todo histórico']
                    if _tz_vals and len(set(_tz_vals)) > 1:
                        _div_rows.append({
                            'Parâmetro': 'Target Z (limiar supressão HRV)',
                            'Min': str(min(_tz_vals)),
                            'Max': str(max(_tz_vals)),
                            'Divergência': '⚠️ Sensibilidade HRV mudou — atleta mais/menos resiliente recentemente',
                        })
                    elif _tz_vals:
                        _div_rows.append({
                            'Parâmetro': 'Target Z',
                            'Min': str(_tz_vals[0]),
                            'Max': str(_tz_vals[0]),
                            'Divergência': '✅ Consistente',
                        })

                    # N clusters
                    _nc_vals = [r['N clusters óptimo'] for r in _summary_rows
                                if r['Período'] != 'Todo histórico']
                    if _nc_vals and max(_nc_vals) - min(_nc_vals) >= 2:
                        _div_rows.append({
                            'Parâmetro': 'N clusters óptimo',
                            'Min': str(min(_nc_vals)),
                            'Max': str(max(_nc_vals)),
                            'Divergência': '⚠️ Complexidade dos padrões de treino mudou',
                        })
                    elif _nc_vals:
                        _div_rows.append({
                            'Parâmetro': 'N clusters',
                            'Min': str(min(_nc_vals)),
                            'Max': str(max(_nc_vals)),
                            'Divergência': '✅ Estável',
                        })

                    # Directional: todo histórico vs 1 ano
                    _dir_hist = next((r for r in _summary_rows
                                      if r['Período'] == 'Todo histórico'), None)
                    _dir_1a   = next((r for r in _summary_rows
                                      if r['Período'] == '1 ano'), None)
                    if _dir_hist and _dir_1a:
                        _div_rows.append({
                            'Parâmetro': 'Directional consistência',
                            'Min': _dir_1a['Consist. directional'],
                            'Max': _dir_hist['Consist. directional'],
                            'Divergência': (
                                f"⚠️ Efeito N: histór. "
                                f"{_dir_hist['Consist. directional']} "
                                f"(N={_dir_hist['N eventos directional']}) vs "
                                f"1ano {_dir_1a['Consist. directional']} "
                                f"(N={_dir_1a['N eventos directional']})")
                        })

                    st.dataframe(pd.DataFrame(_div_rows), hide_index=True,
                                 use_container_width=True)

                    # ── Insights síntese — 180 dias ───────────────────────────────
                    st.markdown("### 💡 Insights — período mais recente (180 dias)")
                    _rec = next((r for r in _summary_rows if r['Período']=='180 dias'), {})
                    if _rec:
                        st.markdown(f"""
- **Lag de resposta HRV**: {_rec.get('Lag máx óptimo','—')}d — carga hoje afecta HRV daqui a **{_rec.get('Lag máx óptimo','?')} dias**
- **Preditor que mais suprime HRV**: {_rec.get('Melhor preditor ↘ HRV','—')}
- **Fingerprint (var mais discriminante)**: {_rec.get('FP: var mais discriminante','—')}
- **Limiar de supressão (Z)**: {_rec.get('Target Z','—')} | Tau recuperação: {_rec.get('Tau elast. (d)','—')}d
- **Directional (180d)**: {_rec.get('Consist. directional','—')} (N={_rec.get('N eventos directional','—')})
- **Clusters óptimos**: {_rec.get('N clusters óptimo','—')} tipos de semana nos últimos 180d
- **N variáveis sig. (p<0.05)**: {_rec.get('N lags sig p<0.05','—')}
                        """)

                # ── Download CSV completo ─────────────────────────────────────────
                if _runner_results:
                    _df_full = pd.DataFrame(_runner_results)
                    _ordem_p = {"180 dias":0,"1 ano":1,"2 anos":2,"3 anos":3,"Todo histórico":4}
                    _df_full['_ordem'] = _df_full['periodo'].map(_ordem_p).fillna(9)
                    _df_full = _df_full.sort_values(['_ordem','analise','variavel']).drop(columns=['_ordem'])

                    _csv_full = _df_full.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                    st.download_button(
                        "📥 Download completo — todos os parâmetros óptimos por período",
                        _csv_full,
                        "atheltica_hrv_autorunner_completo.csv",
                        "text/csv",
                        key="autorunner_dl_full"
                    )
                    if _summary_rows:
                        _csv_sum = pd.DataFrame(_summary_rows).to_csv(
                            index=False, sep=';', decimal=',').encode('utf-8')
                        st.download_button(
                            "📥 Download resumo — parâmetros óptimos (síntese)",
                            _csv_sum,
                            "atheltica_hrv_autorunner_resumo.csv",
                            "text/csv",
                            key="autorunner_dl_sum"
                        )
                else:
                    st.warning("Sem resultados — dados insuficientes para todos os períodos.")
