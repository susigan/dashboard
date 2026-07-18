"""
ATHELTICA — Tab de análise fisiológica de ficheiros FIT
========================================================
Upload de .fit → análise MOXY/SmO2/THb/DFA-α1/respiração/FC.

Inclui:
  • Deteção automática de laps trabalho/recuperação (com correção manual)
  • Séries temporais das métricas fisiológicas
  • Cinética de restauração (trabalho → recuperação)
  • Limiares de SmO2 (3 métodos)
  • Decoupling FC/potência
  • Indicadores de fadiga
  • Histórico de sessões para comparação longitudinal
"""

from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from utils.fit_analyzer import (
    analisar_fit, resumir_para_historico, NOMES_METRICAS,
)

# Cores por métrica (paleta Wong 2011, colorblind-safe)
_CORES_METRICA = {
    'smo2':        '#0072B2',
    'thb':         '#009E73',
    'dfa1':        '#CC79A7',
    'respiration': '#E69F00',
    'heart_rate':  '#D55E00',
    'power':       '#56B4E9',
    'cadence':     '#999999',
}

_CHAVE_HIST = '_fit_historico'


def _mmss(segundos):
    """Formata segundos como mm:ss."""
    if segundos is None or (isinstance(segundos, float) and np.isnan(segundos)):
        return '—'
    m, s = divmod(int(segundos), 60)
    return f"{m}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def _grafico_series(df, colunas, lap_stats, metricas_sel):
    """Séries temporais das métricas seleccionadas, com bandas dos laps de trabalho."""
    if not metricas_sel:
        return None

    fig = make_subplots(
        rows=len(metricas_sel), cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[NOMES_METRICAS.get(m, m) for m in metricas_sel])

    tmin = df['time_seconds'].min()

    for i, metrica in enumerate(metricas_sel, start=1):
        col = colunas[metrica]
        serie = pd.to_numeric(df[col], errors='coerce')
        fig.add_trace(go.Scatter(
            x=df['time_seconds'] / 60.0, y=serie,
            mode='lines', name=NOMES_METRICAS.get(metrica, metrica),
            line=dict(color=_CORES_METRICA.get(metrica, '#333'), width=1.4),
            showlegend=False,
            hovertemplate='%{y:.1f}<extra></extra>'), row=i, col=1)

    # Sombrear os laps de trabalho em todos os painéis
    for l in lap_stats:
        if l.get('phase') != 'work':
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        fig.add_vrect(x0=x0, x1=x1, fillcolor='rgba(214,39,40,0.07)',
                      line_width=0, layer='below')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=180 * len(metricas_sel) + 60, hovermode='x unified',
        margin=dict(t=50, b=45, l=55, r=20), showlegend=False,
        font=dict(size=11))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(title_text='Tempo (min)', row=len(metricas_sel), col=1)
    return fig


def _grafico_limiares(limiares):
    """SmO2 médio por lap vs intensidade, com os limiares marcados."""
    p = limiares['pontos']
    unidade = limiares.get('unidade', 'W')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p['intensidade'], y=p['smo2'], mode='markers+lines',
        marker=dict(size=11, color=_CORES_METRICA['smo2']),
        line=dict(color=_CORES_METRICA['smo2'], width=2),
        name='SmO₂ por lap',
        text=[f"Lap {int(l)}" for l in p['lap']],
        hovertemplate='%{text}<br>%{x:.0f}' + unidade + '<br>SmO₂: %{y:.1f}%<extra></extra>'))

    cores_lim = {'dmax': '#e74c3c', 'quebra': '#f39c12', 'deflexao': '#8e44ad'}
    nomes_lim = {'dmax': 'Dmax', 'quebra': 'Quebra inclinação', 'deflexao': 'Deflexão 1%'}
    for chave, cor in cores_lim.items():
        v = limiares.get(chave)
        if v is not None:
            fig.add_vline(x=v, line_dash='dash', line_color=cor, line_width=2,
                          annotation_text=f"{nomes_lim[chave]}: {v:.0f}",
                          annotation_position='top', annotation_font_size=10,
                          annotation_font_color=cor)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=60, b=50, l=55, r=30),
        xaxis_title=f'Intensidade ({unidade})', yaxis_title='SmO₂ médio (%)',
        font=dict(size=11), showlegend=False,
        title=dict(text='Limiares de SmO₂ — desoxigenação vs intensidade', font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_restauracao(tabela):
    """Tempo de restauração por sequência, por métrica."""
    metricas = [c.replace('tempo_', '').replace('_s', '')
                for c in tabela.columns if c.startswith('tempo_')]
    if not metricas:
        return None

    fig = go.Figure()
    for m in metricas:
        col = f'tempo_{m}_s'
        if col not in tabela.columns:
            continue
        fig.add_trace(go.Scatter(
            x=tabela['sequencia'], y=tabela[col],
            mode='lines+markers', name=NOMES_METRICAS.get(m, m),
            line=dict(color=_CORES_METRICA.get(m, '#333'), width=2.2),
            marker=dict(size=9),
            hovertemplate='Seq %{x}<br>%{y:.0f}s<extra></extra>'))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=360, hovermode='x unified',
        margin=dict(t=55, b=50, l=55, r=20), font=dict(size=11),
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        xaxis_title='Sequência trabalho→recuperação',
        yaxis_title='Tempo até 80% da restauração (s)',
        title=dict(text='Cinética de restauração ao longo da sessão', font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_decoupling(dec):
    """Decoupling FC/potência por lap de trabalho."""
    cores = ['#e74c3c' if v > 5 else '#f39c12' if v > 0 else '#27ae60'
             for v in dec['decoupling_pct']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Lap {int(l)}" for l in dec['lap']], y=dec['decoupling_pct'],
        marker_color=cores, text=[f"{v:+.1f}%" for v in dec['decoupling_pct']],
        textposition='outside',
        hovertemplate='%{x}<br>Decoupling: %{y:+.1f}%<extra></extra>'))
    fig.add_hline(y=5, line_dash='dash', line_color='#e74c3c', line_width=1.5,
                  annotation_text='5% (limiar de deriva)', annotation_font_size=10)
    fig.add_hline(y=0, line_color='rgba(128,128,128,0.5)', line_width=1)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=330, margin=dict(t=55, b=50, l=55, r=20), font=dict(size=11),
        yaxis_title='Decoupling (%)', showlegend=False,
        title=dict(text='Decoupling FC/potência (vs primeiro intervalo)', font=dict(size=13)))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def tab_fit_analise():
    st.header("🫁 Análise FIT — Fisiologia (MOXY / SmO₂ / DFA-α1 / Respiração)")
    st.caption(
        "Carrega um ficheiro .fit de uma sessão intervalada para analisar a resposta "
        "fisiológica: desoxigenação muscular (SmO₂/THb), complexidade autonómica (DFA-α1), "
        "respiração e cinética de recuperação entre intervalos.")

    ficheiro = st.file_uploader(
        "Ficheiro .fit", type=['fit'], key='fit_upload',
        help="A sessão deve ter laps definidos (intervalos de trabalho e recuperação). "
             "Métricas MOXY/DFA-α1 são detectadas automaticamente se existirem no ficheiro.")

    if ficheiro is None:
        _mostrar_historico()
        return

    # ── Análise ───────────────────────────────────────────────────────────────
    bytes_fit = ficheiro.getvalue()
    chave_manual = f'_fit_laps_manual_{ficheiro.name}'
    laps_manual = st.session_state.get(chave_manual)

    with st.spinner("A analisar o ficheiro..."):
        res = analisar_fit(bytes_fit, laps_trabalho_manual=laps_manual)

    if 'erro' in res:
        st.error(f"❌ {res['erro']}")
        return

    colunas = res['colunas']
    lap_stats = res['lap_stats']

    # ── Cabeçalho da sessão ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessão", res.get('data_sessao') or '—')
    c2.metric("Duração", _mmss(res['duracao_total_s']))
    c3.metric("Laps", f"{len(lap_stats)}")
    c4.metric("Laps de trabalho", f"{sum(1 for l in lap_stats if l['phase'] == 'work')}")

    # Métricas fisiológicas encontradas
    fisiologicas = [m for m in ['smo2', 'thb', 'dfa1', 'respiration'] if m in colunas]
    if fisiologicas:
        st.success("✅ Métricas fisiológicas encontradas: " +
                   ", ".join(NOMES_METRICAS.get(m, m) for m in fisiologicas))
    else:
        st.warning(
            "⚠️ Não foram encontradas métricas MOXY/DFA-α1 neste ficheiro. "
            "A análise fica limitada a FC/potência. Se o teu sensor grava estes dados, "
            "verifica se o ficheiro foi exportado com os campos de developer data.")

    st.markdown("---")

    # ── Laps: deteção automática + correção manual ───────────────────────────
    st.markdown("### 🔧 Laps de trabalho")
    st.caption("Detecção automática: intensidade ≥70% da mediana e duração entre 60s e 600s. "
               "Podes corrigir manualmente se a detecção não estiver correcta.")

    tabela_laps = []
    for l in lap_stats:
        linha = {
            'Lap': l['lap_number'],
            'Fase': '🏃 Trabalho' if l['phase'] == 'work' else '🛌 Recuperação',
            'Duração': _mmss(l['duration']),
        }
        for m in ['power', 'heart_rate', 'smo2', 'dfa1', 'respiration']:
            if f'avg_{m}' in l:
                linha[NOMES_METRICAS.get(m, m)] = round(l[f'avg_{m}'], 1)
        tabela_laps.append(linha)
    st.dataframe(pd.DataFrame(tabela_laps), hide_index=True, use_container_width=True)

    with st.expander("✏️ Corrigir manualmente quais laps são de trabalho"):
        auto_work = [l['lap_number'] for l in lap_stats if l['phase'] == 'work']
        escolha = st.multiselect(
            "Laps de trabalho", options=[l['lap_number'] for l in lap_stats],
            default=auto_work, key=f'ms_{ficheiro.name}')
        cA, cB = st.columns(2)
        if cA.button("Aplicar selecção", key=f'aplicar_{ficheiro.name}'):
            st.session_state[chave_manual] = escolha
            st.rerun()
        if cB.button("Voltar à detecção automática", key=f'auto_{ficheiro.name}'):
            st.session_state.pop(chave_manual, None)
            st.rerun()
        if laps_manual is not None:
            st.info(f"A usar selecção manual: laps {sorted(laps_manual)}")

    st.markdown("---")

    # ── Séries temporais ──────────────────────────────────────────────────────
    st.markdown("### 📈 Séries temporais")
    disponiveis = [m for m in ['smo2', 'thb', 'dfa1', 'respiration', 'heart_rate', 'power']
                   if m in colunas]
    default = [m for m in ['smo2', 'heart_rate', 'power'] if m in colunas] or disponiveis[:3]
    sel = st.multiselect(
        "Métricas a mostrar", options=disponiveis, default=default,
        format_func=lambda m: NOMES_METRICAS.get(m, m), key=f'series_{ficheiro.name}')
    if sel:
        fig = _grafico_series(res['df'], colunas, lap_stats, sel)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': False}, key=f'g_series_{ficheiro.name}')
            st.caption("As bandas vermelhas marcam os laps de trabalho.")

    # ── Cinética de restauração ───────────────────────────────────────────────
    rest = res['restauracao']
    if rest['sequencias']:
        st.markdown("---")
        st.markdown("### ♻️ Cinética de restauração")
        st.caption("Tempo que cada métrica demora a atingir 80% da sua recuperação, "
                   "em cada intervalo de recuperação. Tempos crescentes ao longo da "
                   "sessão sugerem fadiga acumulada.")

        cols_resumo = st.columns(max(len(rest['resumo']), 1))
        for i, (metrica, r) in enumerate(rest['resumo'].items()):
            with cols_resumo[i % len(cols_resumo)]:
                st.metric(NOMES_METRICAS.get(metrica, metrica),
                          _mmss(r['media']),
                          f"±{r['std']:.0f}s (n={r['n']})")

        fig_r = _grafico_restauracao(rest['tabela'])
        if fig_r:
            st.plotly_chart(fig_r, use_container_width=True,
                            config={'displayModeBar': False}, key=f'g_rest_{ficheiro.name}')

        with st.expander("📋 Detalhe por sequência"):
            st.dataframe(rest['tabela'], hide_index=True, use_container_width=True)

    # ── Limiares SmO2 ─────────────────────────────────────────────────────────
    lim = res['limiares']
    if lim and lim.get('media') is not None:
        st.markdown("---")
        st.markdown("### 🎯 Limiares de SmO₂")
        st.caption("Ponto de inflexão na desoxigenação muscular — a intensidade a partir da "
                   "qual o SmO₂ cai de forma acelerada. Três métodos independentes.")
        u = lim.get('unidade', 'W')
        lc = st.columns(4)
        lc[0].metric("Dmax", f"{lim['dmax']:.0f} {u}" if lim['dmax'] else "—")
        lc[1].metric("Quebra inclinação", f"{lim['quebra']:.0f} {u}" if lim['quebra'] else "—")
        lc[2].metric("Deflexão 1%", f"{lim['deflexao']:.0f} {u}" if lim['deflexao'] else "—")
        lc[3].metric("**Média**", f"{lim['media']:.0f} {u}")

        st.plotly_chart(_grafico_limiares(lim), use_container_width=True,
                        config={'displayModeBar': False}, key=f'g_lim_{ficheiro.name}')

        _vals = [v for v in (lim['dmax'], lim['quebra'], lim['deflexao']) if v is not None]
        if len(_vals) >= 2:
            _disp = (max(_vals) - min(_vals)) / lim['media'] * 100
            if _disp > 15:
                st.warning(f"⚠️ Os três métodos divergem {_disp:.0f}% entre si — interpreta a "
                           "média com cautela. Mais laps em degraus dariam uma estimativa melhor.")
            else:
                st.success(f"✅ Os três métodos concordam (dispersão {_disp:.0f}%) — "
                           "estimativa fiável.")

    # ── Decoupling ────────────────────────────────────────────────────────────
    dec = res['decoupling']
    if dec is not None and len(dec) >= 2:
        st.markdown("---")
        st.markdown("### 💓 Decoupling FC/potência")
        st.caption("Deriva do custo cardíaco: quanto mais FC é necessária para a mesma potência "
                   "ao longo da sessão. Acima de 5% indica deriva cardiovascular relevante.")
        st.plotly_chart(_grafico_decoupling(dec), use_container_width=True,
                        config={'displayModeBar': False}, key=f'g_dec_{ficheiro.name}')

    # ── Fadiga ────────────────────────────────────────────────────────────────
    fad = res['fadiga']
    if fad:
        st.markdown("---")
        st.markdown("### 🔋 Indicadores de fadiga")
        cor = fad.get('veredicto_cor', '#888')
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:8px;"
            f"background:{cor}1A;border-left:5px solid {cor}'>"
            f"<b style='color:{cor};font-size:16px'>Fadiga: {fad['veredicto']}</b> "
            f"<span style='font-size:13px'>({fad['n_alertas']}/3 sinais de alerta)</span></div>",
            unsafe_allow_html=True)

        fc = st.columns(3)
        if 'tendencia_fc' in fad:
            fc[0].metric("Restauração da FC", fad['tendencia_fc'],
                         f"slope {fad.get('slope_restauracao_fc', 0):+.1f}s/seq")
        if 'consistencia' in fad:
            fc[1].metric("Consistência", fad['consistencia'],
                         f"CV {fad.get('cv_restauracao_fc', 0):.0%}")
        if 'deriva_cardiovascular' in fad:
            fc[2].metric("Deriva cardiovascular", fad['deriva_cardiovascular'],
                         f"{fad.get('decoupling_final_pct', 0):+.1f}%")

        st.caption("Os três sinais: (1) o tempo de recuperação da FC aumenta ao longo da sessão, "
                   "(2) a recuperação é inconsistente entre intervalos, (3) há deriva do custo "
                   "cardíaco. Dois ou mais sinais indicam fadiga elevada.")

    # ── Tempo até à falha ─────────────────────────────────────────────────────
    tf = res.get('tempo_falha')
    if tf is not None and len(tf) > 0:
        with st.expander("⏱️ Estimativa de tempo até à falha (extrapolação SmO₂)"):
            st.caption("Extrapolação da taxa de queda do SmO₂ até ao mínimo observado na sessão. "
                       "É uma estimativa grosseira — usa-a como ordem de grandeza, não valor exacto.")
            st.dataframe(tf, hide_index=True, use_container_width=True)

    # ── Guardar no histórico ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Guardar no histórico")
    resumo = resumir_para_historico(res, ficheiro.name)
    cg1, cg2 = st.columns([1, 2])
    if cg1.button("➕ Adicionar esta sessão ao histórico", key=f'guardar_{ficheiro.name}'):
        hist = st.session_state.get(_CHAVE_HIST, [])
        if not any(h.get('ficheiro') == resumo['ficheiro'] and h.get('data') == resumo['data']
                   for h in hist):
            hist.append(resumo)
            st.session_state[_CHAVE_HIST] = hist
            st.success("Sessão adicionada ao histórico.")
        else:
            st.info("Esta sessão já está no histórico.")
    cg2.caption("O histórico permite comparar a evolução entre sessões. "
                "Fica guardado durante a sessão do dashboard — exporta o CSV para o manteres.")

    _mostrar_historico()


def _mostrar_historico():
    """Histórico de sessões analisadas, para comparação longitudinal."""
    hist = st.session_state.get(_CHAVE_HIST, [])
    if not hist:
        return

    st.markdown("---")
    st.markdown("### 📚 Histórico de sessões")
    dfh = pd.DataFrame(hist).sort_values('data')
    st.dataframe(dfh, hide_index=True, use_container_width=True)

    # Evolução de uma métrica à escolha
    numericas = [c for c in dfh.columns
                 if dfh[c].dtype.kind in 'if' and dfh[c].notna().sum() >= 2]
    if len(dfh) >= 2 and numericas:
        metrica = st.selectbox("Evolução de", options=numericas, key='hist_metrica')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfh['data'], y=dfh[metrica], mode='lines+markers',
            line=dict(color='#0072B2', width=2.5), marker=dict(size=10),
            hovertemplate='%{x}<br>%{y:.1f}<extra></extra>'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=40, b=50, l=55, r=20), font=dict(size=11),
            yaxis_title=metrica, showlegend=False,
            title=dict(text=f'Evolução — {metrica}', font=dict(size=13)))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False}, key='g_hist')

    ch1, ch2 = st.columns([1, 2])
    ch1.download_button(
        "📥 Descarregar histórico (CSV)",
        dfh.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
        "atheltica_fit_historico.csv", "text/csv", key='hist_dl')
    if ch2.button("🗑️ Limpar histórico", key='hist_limpar'):
        st.session_state[_CHAVE_HIST] = []
        st.rerun()
