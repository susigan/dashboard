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
    analisar_fit, resumir_para_historico, parse_intervalos, NOMES_METRICAS,
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

def _grafico_multi_eixo(df, colunas, lap_stats, y1, y2=None, y3=None,
                        suavizar=0):
    """
    Gráfico com até 3 eixos Y independentes (estilo fitfileviewer):
      Y1 → eixo esquerdo
      Y2 → eixo direito
      Y3 → segundo eixo direito, deslocado

    Cada grupo pode ter várias métricas (partilham a escala do seu eixo).
    """
    if not y1 and not y2 and not y3:
        return None

    fig = go.Figure()
    tmin = df['time_seconds'].min()
    x = (df['time_seconds'] - tmin) / 60.0

    def _serie(metrica):
        s = pd.to_numeric(df[colunas[metrica]], errors='coerce')
        if suavizar and suavizar > 1:
            s = s.rolling(int(suavizar), min_periods=1, center=True).mean()
        return s

    grupos = [
        (y1 or [], 'y',  None),
        (y2 or [], 'y2', None),
        (y3 or [], 'y3', None),
    ]
    for metricas, eixo, _ in grupos:
        for m in metricas:
            if m not in colunas:
                continue
            fig.add_trace(go.Scatter(
                x=x, y=_serie(m), mode='lines',
                name=NOMES_METRICAS.get(m, m), yaxis=eixo,
                line=dict(color=_CORES_METRICA.get(m, '#333'), width=1.6),
                hovertemplate='%{y:.1f}<extra>' + NOMES_METRICAS.get(m, m) + '</extra>'))

    # Sombrear laps de trabalho (vermelho) e excluídos (cinzento)
    for l in lap_stats:
        fase = l.get('phase')
        if fase not in ('work', 'excluded'):
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        cor = ('rgba(214,39,40,0.07)' if fase == 'work' else 'rgba(128,128,128,0.16)')
        fig.add_vrect(x0=x0, x1=x1, fillcolor=cor, line_width=0, layer='below')

    def _titulo(metricas):
        return ' / '.join(NOMES_METRICAS.get(m, m) for m in metricas)

    def _cor_eixo(metricas):
        return _CORES_METRICA.get(metricas[0], '#333') if metricas else '#333'

    # Com 3 eixos, encolhe o domínio do X para o 3º eixo caber à direita
    dominio_x = [0.0, 0.88] if y3 else [0.0, 1.0]

    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=460, hovermode='x unified', font=dict(size=11),
        margin=dict(t=50, b=55, l=60, r=40 if not y3 else 90),
        legend=dict(orientation='h', y=-0.16, font=dict(size=10)),
        xaxis=dict(title=dict(text='Tempo (min)'), domain=dominio_x,
                   showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(title=dict(text=_titulo(y1) if y1 else None,
                              font=dict(color=_cor_eixo(y1))),
                   tickfont=dict(color=_cor_eixo(y1)),
                   showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
    )
    if y2:
        layout['yaxis2'] = dict(
            title=dict(text=_titulo(y2), font=dict(color=_cor_eixo(y2))),
            tickfont=dict(color=_cor_eixo(y2)),
            overlaying='y', side='right', showgrid=False)
    if y3:
        layout['yaxis3'] = dict(
            title=dict(text=_titulo(y3), font=dict(color=_cor_eixo(y3))),
            tickfont=dict(color=_cor_eixo(y3)),
            overlaying='y', side='right', position=0.97,
            anchor='free', showgrid=False)

    fig.update_layout(**layout)
    return fig


def _grafico_series(df, colunas, lap_stats, metricas_sel):
    """Séries temporais empilhadas (um painel por métrica)."""
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

    # Sombrear laps: trabalho (vermelho) e excluídos (cinzento)
    for l in lap_stats:
        fase = l.get('phase')
        if fase not in ('work', 'excluded'):
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        cor = ('rgba(214,39,40,0.07)' if fase == 'work'
               else 'rgba(128,128,128,0.16)')
        fig.add_vrect(x0=x0, x1=x1, fillcolor=cor, line_width=0, layer='below')

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
    chave_excl = f'_fit_laps_excl_{ficheiro.name}'
    laps_manual = st.session_state.get(chave_manual)
    laps_excl = st.session_state.get(chave_excl, [])

    # ── Definições da análise ────────────────────────────────────────────────
    with st.expander("⚙️ Definições da análise", expanded=False):
        janela = st.slider(
            "Janela de estado estacionário (segundos finais de cada lap)",
            min_value=0, max_value=180, value=60, step=10,
            key=f'janela_{ficheiro.name}',
            help="As médias de cada lap são calculadas só sobre os últimos N segundos. "
                 "Métricas como o SmO₂ têm cinética lenta (~30-60s) e no início do lap "
                 "ainda estão em transição da intensidade anterior. Usar o lap inteiro "
                 "sobrestima o SmO₂ e distorce os limiares. 0 = usar o lap inteiro.")
        if janela == 0:
            st.warning("⚠️ A usar o lap inteiro — as médias incluem a fase de transição, "
                       "o que tende a sobrestimar o SmO₂ e a deslocar os limiares.")

        st.markdown("---")
        st.markdown("**Como identificar os intervalos de trabalho**")
        modo = st.radio(
            "Modo",
            options=['auto', 'corte', 'intervalos'],
            format_func=lambda m: {
                'auto': '🤖 Automático (laps do ficheiro, ou detecção pelo sinal)',
                'corte': '📉 Detecção por corte de intensidade (defino a %)',
                'intervalos': '⏱️ Eu defino os tempos de trabalho',
            }[m],
            key=f'modo_{ficheiro.name}',
            help="Se a detecção automática não acertar no teu ficheiro, usa um dos "
                 "outros modos.")

        frac_corte = None
        intervalos = None
        min_dur_seg = 45

        if modo == 'corte':
            pct = st.slider(
                "Recuperação = abaixo de X% da intensidade de trabalho",
                min_value=20, max_value=90, value=50, step=5,
                key=f'pct_{ficheiro.name}',
                help="Tudo o que estiver abaixo desta percentagem da potência (ou FC) "
                     "típica de trabalho é considerado recuperação.")
            frac_corte = pct / 100.0
            min_dur_seg = st.slider(
                "Duração mínima de um bloco (s)", 10, 180, 45, 5,
                key=f'mindur_{ficheiro.name}',
                help="Blocos mais curtos são fundidos com o anterior, para evitar "
                     "dezenas de micro-intervalos por causa de oscilações.")

        elif modo == 'intervalos':
            st.caption("Escreve um intervalo de **trabalho** por linha. Tudo o que ficar "
                       "fora (os 'buracos') passa automaticamente a recuperação.")
            texto = st.text_area(
                "Intervalos de trabalho",
                value=st.session_state.get(f'txt_iv_{ficheiro.name}', ''),
                placeholder="10:00-13:00\n14:00-17:00\n18:00-21:00",
                height=140, key=f'txt_iv_{ficheiro.name}',
                help="Formatos aceites: mm:ss-mm:ss, h:mm:ss-h:mm:ss, ou segundos "
                     "(600-780). Um intervalo por linha, ou separados por ';'.")
            intervalos, erros_iv = parse_intervalos(texto)
            if erros_iv:
                st.error("Não consegui interpretar: " + ", ".join(f"`{e}`" for e in erros_iv))
            if intervalos:
                st.success(f"✅ {len(intervalos)} intervalos de trabalho: " +
                           ", ".join(f"{_mmss(a)}–{_mmss(b)}" for a, b in intervalos))
            elif texto.strip():
                st.warning("Nenhum intervalo válido — a usar detecção automática.")

    _modo_seg = {'auto': 'auto', 'corte': 'forcar', 'intervalos': 'intervalos'}[modo]

    with st.spinner("A analisar o ficheiro..."):
        res = analisar_fit(
            bytes_fit, laps_trabalho_manual=laps_manual, laps_excluidos=laps_excl,
            janela_final_s=janela, modo_segmentacao=_modo_seg,
            intervalos_trabalho=intervalos, frac_corte=frac_corte,
            min_dur_segmento=min_dur_seg)

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

    # ── Laps: deteção automática + correção manual + aquecimento ─────────────
    st.markdown("### 🔧 Laps")

    # Origem dos laps: marcados no ficheiro vs segmentados automaticamente
    _trigger0 = lap_stats[0].get('lap_trigger') if lap_stats else None
    _auto_seg = _trigger0 == 'auto_segmentado'
    _sem_seg = _trigger0 == 'auto_none'
    _por_tempo = _trigger0 == 'manual_tempo'
    if _por_tempo:
        st.success(
            "⏱️ **Intervalos definidos por ti** — os blocos que indicaste são o trabalho; "
            "os períodos entre eles foram convertidos automaticamente em recuperação.")
    elif _auto_seg:
        st.info(
            "🤖 **O ficheiro não tinha laps marcados** — os intervalos foram detectados "
            "automaticamente a partir do sinal de intensidade (blocos alto/baixo). "
            "Confirma na tabela abaixo se a segmentação corresponde ao teu protocolo; "
            "se não, ajusta manualmente.")
    elif _sem_seg:
        st.warning(
            "⚠️ **Ficheiro sem laps e sem estrutura de intervalos detectável** — a sessão "
            "foi tratada como um bloco único. As análises que dependem de pares "
            "trabalho→recuperação não estarão disponíveis.")

    _metodo = lap_stats[0].get('metodo_classificacao') if lap_stats else None
    _msg_metodo = ""
    if _metodo == 'FIT intensity':
        _msg_metodo = ("Classificação a partir do campo `intensity` gravado no próprio "
                       "ficheiro FIT (mais fiável que inferir). ")
    elif _metodo and _metodo.startswith('auto'):
        _msg_metodo = ("Classificação inferida da intensidade medida (separação entre os "
                       "blocos de trabalho e de recuperação). ")

    _msg_janela = (f"Médias calculadas sobre os últimos {janela}s de cada lap "
                   "(estado estacionário)." if janela > 0 else
                   "Médias calculadas sobre o lap inteiro.")
    st.caption(f"{_msg_metodo}{_msg_janela} "
               "Podes corrigir a classificação e excluir o aquecimento abaixo.")

    _FASE_LBL = {'work': '🏃 Trabalho', 'recovery': '🛌 Recuperação',
                 'excluded': '⚪ Excluído'}
    tabela_laps = []
    for l in lap_stats:
        linha = {
            'Lap': l['lap_number'],
            'Fase': _FASE_LBL.get(l['phase'], l['phase']),
            'Duração': _mmss(l['duration']),
        }
        for m in ['power', 'heart_rate', 'smo2', 'dfa1', 'respiration']:
            if f'avg_{m}' in l:
                linha[NOMES_METRICAS.get(m, m)] = round(l[f'avg_{m}'], 1)
        # Mostrar os campos nativos do FIT quando existem e são informativos
        if l.get('intensity') and not _auto_seg:
            linha['FIT intensity'] = l['intensity']
        if l.get('lap_trigger') and l['lap_trigger'] not in ('auto_segmentado', 'auto_none'):
            linha['FIT trigger'] = l['lap_trigger']
        tabela_laps.append(linha)
    st.dataframe(pd.DataFrame(tabela_laps), hide_index=True, use_container_width=True)

    if laps_excl:
        st.info(f"⚪ Laps excluídos da análise: {sorted(laps_excl)} "
                "(não entram nos limiares, restauração, decoupling nem fadiga).")

    todos_laps = [l['lap_number'] for l in lap_stats]

    with st.expander("✏️ Ajustar laps — aquecimento e classificação"):
        st.markdown("**1. Excluir aquecimento / arrefecimento**")
        st.caption("Laps excluídos são ignorados em toda a análise. A mediana de "
                   "referência da detecção automática também passa a ignorá-los, "
                   "para o aquecimento não puxar o limiar para baixo.")
        escolha_excl = st.multiselect(
            "Laps a excluir", options=todos_laps, default=laps_excl,
            key=f'ms_excl_{ficheiro.name}')

        st.markdown("**2. Laps de trabalho**")
        st.caption("A recuperação é inferida: tudo o que não for trabalho nem "
                   "estiver excluído conta como recuperação.")
        auto_work = [l['lap_number'] for l in lap_stats if l['phase'] == 'work']
        opcoes_work = [n for n in todos_laps if n not in escolha_excl]
        escolha_work = st.multiselect(
            "Laps de trabalho", options=opcoes_work,
            default=[n for n in auto_work if n not in escolha_excl],
            key=f'ms_work_{ficheiro.name}')

        cA, cB = st.columns(2)
        if cA.button("Aplicar", key=f'aplicar_{ficheiro.name}'):
            st.session_state[chave_excl] = escolha_excl
            st.session_state[chave_manual] = escolha_work
            st.rerun()
        if cB.button("Repor detecção automática", key=f'auto_{ficheiro.name}'):
            st.session_state.pop(chave_manual, None)
            st.session_state.pop(chave_excl, None)
            st.rerun()
        if laps_manual is not None:
            st.caption(f"A usar selecção manual de trabalho: laps {sorted(laps_manual)}")

    st.markdown("---")

    # ── Séries temporais ──────────────────────────────────────────────────────
    st.markdown("### 📈 Séries temporais")
    disponiveis = [m for m in ['smo2', 'thb', 'dfa1', 'respiration',
                               'heart_rate', 'power', 'cadence'] if m in colunas]

    estilo = st.radio(
        "Estilo do gráfico", options=['multi', 'empilhado'],
        format_func=lambda e: {'multi': '📊 Eixos combinados (Y1/Y2/Y3)',
                               'empilhado': '📑 Painéis empilhados'}[e],
        horizontal=True, key=f'estilo_{ficheiro.name}')

    if estilo == 'multi':
        st.caption("Escolhe que métricas vão em cada eixo. Métricas no mesmo eixo "
                   "partilham a escala — junta as que têm grandezas parecidas.")
        ce1, ce2, ce3 = st.columns(3)
        _def1 = [m for m in ['smo2'] if m in colunas] or disponiveis[:1]
        _def2 = [m for m in ['power'] if m in colunas]
        _def3 = [m for m in ['heart_rate'] if m in colunas]
        y1 = ce1.multiselect("Eixo Y1 (esquerda)", disponiveis, default=_def1,
                             format_func=lambda m: NOMES_METRICAS.get(m, m),
                             key=f'y1_{ficheiro.name}')
        y2 = ce2.multiselect("Eixo Y2 (direita)", disponiveis, default=_def2,
                             format_func=lambda m: NOMES_METRICAS.get(m, m),
                             key=f'y2_{ficheiro.name}')
        y3 = ce3.multiselect("Eixo Y3 (extra)", disponiveis, default=_def3,
                             format_func=lambda m: NOMES_METRICAS.get(m, m),
                             key=f'y3_{ficheiro.name}')
        suav = st.slider("Suavização (média móvel, segundos)", 0, 60, 0, 5,
                         key=f'suav_{ficheiro.name}',
                         help="0 = dados brutos a 1Hz. Suavizar ajuda a ver a tendência "
                              "em métricas ruidosas como o DFA-α1.")
        fig = _grafico_multi_eixo(res['df'], colunas, lap_stats, y1, y2, y3, suav)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': True, 'scrollZoom': True},
                            key=f'g_multi_{ficheiro.name}')
            st.caption("Bandas vermelhas = laps de trabalho · cinzentas = excluídos. "
                       "Podes fazer zoom e arrastar no gráfico.")
        else:
            st.info("Escolhe pelo menos uma métrica.")
    else:
        default = [m for m in ['smo2', 'heart_rate', 'power'] if m in colunas] or disponiveis[:3]
        sel = st.multiselect(
            "Métricas a mostrar", options=disponiveis, default=default,
            format_func=lambda m: NOMES_METRICAS.get(m, m), key=f'series_{ficheiro.name}')
        if sel:
            fig = _grafico_series(res['df'], colunas, lap_stats, sel)
            if fig:
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_series_{ficheiro.name}')
                st.caption("Bandas vermelhas = laps de trabalho · cinzentas = excluídos.")

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
