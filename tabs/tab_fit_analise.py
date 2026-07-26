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
import hashlib
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from utils.fit_analyzer import (
    preparar_fit, analisar_completo, resumir_para_historico, parse_intervalos,
    sugerir_offset, sugerir_offset_por_laps, NOMES_METRICAS,
    DFA1_HRVT2, DFA1_HRVT1, LOA_LITERATURA, ler_fit, calcular_wbal,
    comparar_com_historico,
)

# Cores por métrica (paleta Wong 2011, colorblind-safe)
_CORES_METRICA = {
    'smo2':          '#0072B2',
    'thb':           '#009E73',
    'dfa1':          '#CC79A7',
    'respiration':   '#E69F00',
    'resp_enhanced': '#B8860B',
    'artifacts':     '#8B0000',
    'hhb':           '#8E1600',
    'o2hb':          '#1B7837',
    'rr_ratio':      '#7B68EE',
    'hr_alphahrv':   '#FF8C69',
    'heart_rate':    '#D55E00',
    'power':         '#56B4E9',
    'cadence':       '#999999',
    'speed':         '#4682B4',
    'distance':      '#708090',
    'cycle_length':  '#A0522D',
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

def _grafico_breakpoint_hhb(bp_smo2, bp_hhb):
    """
    HHb e SmO₂ vs intensidade, lado a lado, cada um com o seu ajuste
    double-linear. É a visualização que a literatura NIRS usa: o HHb sobe
    (desoxigenação) e o SmO₂ desce, e o breakpoint deve coincidir.
    """
    from plotly.subplots import make_subplots as _msp
    fig = _msp(rows=1, cols=2, horizontal_spacing=0.10,
               subplot_titles=['HHb (hemoglobina desoxigenada)', 'SmO₂ (%)'])

    for _col, _bp, _cor_pt in ((1, bp_hhb, '#8E1600'), (2, bp_smo2, '#0072B2')):
        if _bp is None:
            continue
        p = _bp['pontos']
        u = _bp['unidade']
        fig.add_trace(go.Scatter(
            x=p['intensidade'], y=p['smo2'], mode='markers',
            marker=dict(size=5, color=_cor_pt, opacity=0.55),
            showlegend=False,
            hovertemplate='%{x:.0f}' + u + '<br>%{y:.2f}<extra></extra>'),
            row=1, col=_col)

        x = p['intensidade'].values
        xb = _bp['breakpoint']
        x1 = np.linspace(x.min(), xb, 30)
        x2 = np.linspace(xb, x.max(), 30)
        fig.add_trace(go.Scatter(
            x=x1, y=np.polyval(_bp['coef_antes'], x1), mode='lines',
            line=dict(color='#27ae60', width=2.5), showlegend=(_col == 1),
            name='Antes do breakpoint'), row=1, col=_col)
        fig.add_trace(go.Scatter(
            x=x2, y=np.polyval(_bp['coef_depois'], x2), mode='lines',
            line=dict(color='#e74c3c', width=2.5), showlegend=(_col == 1),
            name='Depois do breakpoint'), row=1, col=_col)
        fig.add_vline(x=xb, line_dash='dash', line_color='#333', line_width=2,
                      annotation_text=f"{xb:.0f}{u}", annotation_position='top',
                      row=1, col=_col)

    _u = (bp_hhb or bp_smo2)['unidade'] if (bp_hhb or bp_smo2) else 'W'
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=60, b=55, l=55, r=25), font=dict(size=11),
        legend=dict(orientation='h', y=-0.20, font=dict(size=10)),
        title=dict(text='Breakpoint nos dois sinais — validação cruzada',
                   font=dict(size=13)))
    fig.update_xaxes(title_text=f'Intensidade ({_u})', showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text='HHb', row=1, col=1)
    fig.update_yaxes(title_text='SmO₂ (%)', row=1, col=2)
    return fig


def _grafico_hhb_temporal(df, colunas, lap_stats):
    """HHb e SmO₂ ao longo do tempo, com as fases sombreadas."""
    fig = go.Figure()
    tmin = df['time_seconds'].min()
    x = (df['time_seconds'] - tmin) / 60.0

    if 'hhb' in colunas:
        fig.add_trace(go.Scatter(
            x=x, y=pd.to_numeric(df[colunas['hhb']], errors='coerce'),
            mode='lines', name='HHb (desoxi)',
            line=dict(color='#8E1600', width=1.6),
            hovertemplate='HHb %{y:.2f}<extra></extra>'))
    if 'o2hb' in colunas:
        fig.add_trace(go.Scatter(
            x=x, y=pd.to_numeric(df[colunas['o2hb']], errors='coerce'),
            mode='lines', name='O₂Hb (oxi)',
            line=dict(color='#1B7837', width=1.6),
            hovertemplate='O₂Hb %{y:.2f}<extra></extra>'))
    if 'thb' in colunas:
        fig.add_trace(go.Scatter(
            x=x, y=pd.to_numeric(df[colunas['thb']], errors='coerce'),
            mode='lines', name='THb (total)',
            line=dict(color='#009E73', width=1.2, dash='dot'),
            hovertemplate='THb %{y:.2f}<extra></extra>'))

    for l in lap_stats:
        fase = l.get('phase')
        if fase not in ('work', 'excluded', 'recovery'):
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        cor = {'work': 'rgba(214,39,40,0.09)',
               'excluded': 'rgba(128,128,128,0.16)',
               'recovery': 'rgba(52,152,219,0.07)'}[fase]
        fig.add_vrect(x0=x0, x1=x1, fillcolor=cor, line_width=0, layer='below')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=340, hovermode='x unified', font=dict(size=11),
        margin=dict(t=50, b=50, l=55, r=25),
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        xaxis_title='Tempo (min)', yaxis_title='Hemoglobina',
        title=dict(text='Distribuição da hemoglobina ao longo da sessão',
                   font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_curva_dfa1(serie, hrvt2=None):
    """Curva DFA-α1 vs FC, com as linhas de referência e o ajuste do HRVT2."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie['fc_media'], y=serie['dfa1'], mode='markers',
        marker=dict(size=5, color='rgba(204,121,167,0.5)'),
        name='DFA-α1 (janelas 2 min)',
        hovertemplate='FC %{x:.0f} bpm<br>α1 %{y:.2f}<extra></extra>'))

    for alvo, cor, txt in [(DFA1_HRVT1, '#f39c12', 'α1=0.75 (≈VT1)'),
                           (DFA1_HRVT2, '#e74c3c', 'α1=0.50 (HRVT2 ≈ RCP/MLSS)')]:
        fig.add_hline(y=alvo, line_dash='dash', line_color=cor, line_width=1.5,
                      annotation_text=txt, annotation_position='right',
                      annotation_font_size=9, annotation_font_color=cor)

    if hrvt2 and 'erro' not in hrvt2 and hrvt2.get('coef'):
        pl = hrvt2.get('pontos_linear')
        if pl is not None and len(pl) > 1:
            xr = np.linspace(pl['fc_media'].min(), pl['fc_media'].max(), 40)
            fig.add_trace(go.Scatter(
                x=xr, y=np.polyval(hrvt2['coef'], xr), mode='lines',
                line=dict(color='#CC79A7', width=2.5),
                name=f"Ajuste linear (R²={hrvt2['r2']:.2f})"))
        if hrvt2.get('fiavel'):
            fig.add_vline(x=hrvt2['fc'], line_dash='dot', line_color='#e74c3c',
                          line_width=2,
                          annotation_text=f"HRVT2 {hrvt2['fc']:.0f} bpm",
                          annotation_position='top')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=55, b=50, l=55, r=160), font=dict(size=11),
        xaxis_title='FC (bpm)', yaxis_title='DFA-α1',
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        title=dict(text='DFA-α1 vs FC — recalculado dos intervalos RR',
                   font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_double_linear(bp):
    """SmO2 vs intensidade com as duas rectas do ajuste double-linear."""
    p = bp['pontos']
    u = bp['unidade']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p['intensidade'], y=p['smo2'], mode='markers',
        marker=dict(size=6, color='rgba(0,114,178,0.55)'),
        name='SmO₂ (estado estacionário)',
        hovertemplate='%{x:.0f}' + u + '<br>SmO₂ %{y:.1f}%<extra></extra>'))

    x = p['intensidade'].values
    xb = bp['breakpoint']
    c1, c2 = bp['coef_antes'], bp['coef_depois']
    x1 = np.linspace(x.min(), xb, 30)
    x2 = np.linspace(xb, x.max(), 30)
    fig.add_trace(go.Scatter(x=x1, y=np.polyval(c1, x1), mode='lines',
        line=dict(color='#27ae60', width=2.5), name='Antes do breakpoint'))
    fig.add_trace(go.Scatter(x=x2, y=np.polyval(c2, x2), mode='lines',
        line=dict(color='#e74c3c', width=2.5), name='Depois do breakpoint'))
    fig.add_vline(x=xb, line_dash='dash', line_color='#333', line_width=2,
                  annotation_text=f"MLSS ≈ {xb:.0f}{u}", annotation_position='top')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=55, b=50, l=55, r=25), font=dict(size=11),
        xaxis_title=f'Intensidade ({u})', yaxis_title='SmO₂ (%)',
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        title=dict(text='Breakpoint de SmO₂ — ajuste double-linear', font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_triplo_linear(bp):
    """SmO2 vs intensidade com as três rectas do ajuste (LT1 + LT2)."""
    p = bp['pontos']
    u = bp['unidade']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p['intensidade'], y=p['smo2'], mode='markers',
        marker=dict(size=6, color='rgba(0,114,178,0.55)'),
        name='SmO₂ (estado estacionário)',
        hovertemplate='%{x:.0f}' + u + '<br>SmO₂ %{y:.1f}%<extra></extra>'))

    x = p['intensidade'].values
    lt1, lt2 = bp['breakpoint_lt1'], bp['breakpoint_lt2']
    c1, c2, c3 = bp['coef_1'], bp['coef_2'], bp['coef_3']
    x1 = np.linspace(x.min(), lt1, 20)
    x2 = np.linspace(lt1, lt2, 20)
    x3 = np.linspace(lt2, x.max(), 20)
    fig.add_trace(go.Scatter(x=x1, y=np.polyval(c1, x1), mode='lines',
        line=dict(color='#27ae60', width=2.5), name='Antes do LT1 (moderado)'))
    fig.add_trace(go.Scatter(x=x2, y=np.polyval(c2, x2), mode='lines',
        line=dict(color='#f39c12', width=2.5), name='Entre LT1-LT2 (pesado)'))
    fig.add_trace(go.Scatter(x=x3, y=np.polyval(c3, x3), mode='lines',
        line=dict(color='#e74c3c', width=2.5), name='Depois do LT2 (severo)'))
    fig.add_vline(x=lt1, line_dash='dash', line_color='#333', line_width=2,
                  annotation_text=f"LT1 ≈ {lt1:.0f}{u}", annotation_position='top')
    fig.add_vline(x=lt2, line_dash='dash', line_color='#333', line_width=2,
                  annotation_text=f"LT2 ≈ {lt2:.0f}{u}", annotation_position='bottom')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=55, b=50, l=55, r=25), font=dict(size=11),
        xaxis_title=f'Intensidade ({u})', yaxis_title='SmO₂ (%)',
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        title=dict(text='LT1 + LT2 via SmO₂ — ajuste de 3 segmentos', font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_dfa1(ld):
    """DFA-α1 vs intensidade com as linhas de referência 0.75 / 0.70 / 0.50."""
    p = ld['pontos']
    u = ld['unidade']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p['intensidade'], y=p['dfa1'], mode='markers+text',
        marker=dict(size=12, color='#CC79A7'),
        text=[f"L{int(l)}" for l in p['lap']], textposition='top center',
        textfont=dict(size=9), name='DFA-α1 por lap',
        hovertemplate='%{x:.0f}' + u + '<br>DFA-α1 %{y:.2f}<extra></extra>'))

    x = p['intensidade'].values
    xr = np.linspace(x.min() * 0.9, x.max() * 1.1, 50)
    fig.add_trace(go.Scatter(x=xr, y=np.polyval(ld['coef'], xr), mode='lines',
        line=dict(color='#CC79A7', width=2, dash='dot'),
        name=f"Ajuste (R²={ld['r2']:.2f})"))

    for alvo, cor, txt in [(0.75, '#27ae60', 'α1=0.75 (≈VT1)'),
                           (0.70, '#f39c12', 'α1=0.70 (limite Z1)'),
                           (0.50, '#e74c3c', 'α1=0.50 (ruído branco)')]:
        fig.add_hline(y=alvo, line_dash='dash', line_color=cor, line_width=1.5,
                      annotation_text=txt, annotation_position='right',
                      annotation_font_size=9, annotation_font_color=cor)
        v = ld['limiares'].get(alvo)
        if v and not v['extrapolado']:
            fig.add_vline(x=v['intensidade'], line_dash='dot', line_color=cor,
                          line_width=1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=55, b=50, l=55, r=140), font=dict(size=11),
        xaxis_title=f'Intensidade ({u})', yaxis_title='DFA-α1',
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        title=dict(text='DFA-α1 vs intensidade — estimativa do VT1', font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


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
        if fase not in ('work', 'excluded', 'recovery'):
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        cor = {'work': 'rgba(214,39,40,0.09)',
               'excluded': 'rgba(128,128,128,0.16)',
               'recovery': 'rgba(52,152,219,0.07)'}[fase]
        fig.add_vrect(x0=x0, x1=x1, fillcolor=cor, line_width=0, layer='below')

    def _titulo(metricas):
        return ' / '.join(NOMES_METRICAS.get(m, m) for m in metricas)

    def _cor_eixo(metricas):
        return _CORES_METRICA.get(metricas[0], '#333') if metricas else '#333'

    def _intervalo(metricas, folga=0.08):
        """
        Calcula o intervalo do eixo a partir dos dados reais, com uma folga.

        Sem isto, o Plotly inclui o zero por defeito, o que esmaga métricas de
        amplitude pequena: o THb, por exemplo, varia entre 12.4 e 12.8 — se o
        eixo for de 0 a 13, toda a variação real aparece como uma linha recta.
        Ajustando o eixo à amplitude real, a variação torna-se visível.
        """
        vals = []
        for m in metricas:
            if m not in colunas:
                continue
            s = pd.to_numeric(df[colunas[m]], errors='coerce')
            if suavizar and suavizar > 1:
                s = s.rolling(int(suavizar), min_periods=1, center=True).mean()
            s = s.dropna()
            if len(s) > 0:
                vals.append((float(s.min()), float(s.max())))
        if not vals:
            return None
        lo = min(v[0] for v in vals)
        hi = max(v[1] for v in vals)
        amp = hi - lo
        if amp <= 0:
            margem = max(abs(lo) * 0.01, 0.5)
            return [lo - margem, hi + margem]
        _lo_f, _hi_f = lo - amp * folga, hi + amp * folga
        # Métricas que não podem ser negativas: não deixar o eixo passar abaixo de
        # zero só por causa da folga (potência a -22W não faz sentido físico).
        _NAO_NEGATIVAS = {'power', 'cadence', 'speed', 'distance', 'heart_rate',
                          'hr_alphahrv', 'smo2', 'thb', 'respiration',
                          'resp_enhanced', 'artifacts', 'cycle_length'}
        if lo >= 0 and any(m in _NAO_NEGATIVAS for m in metricas):
            _lo_f = max(0.0, _lo_f)
        return [_lo_f, _hi_f]

    # Com 3 eixos, encolhe o domínio do X para o 3º eixo caber à direita
    dominio_x = [0.0, 0.88] if y3 else [0.0, 1.0]

    _r1 = _intervalo(y1) if y1 else None
    _r2 = _intervalo(y2) if y2 else None
    _r3 = _intervalo(y3) if y3 else None

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
                   range=_r1,
                   showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
    )
    if y2:
        layout['yaxis2'] = dict(
            title=dict(text=_titulo(y2), font=dict(color=_cor_eixo(y2))),
            tickfont=dict(color=_cor_eixo(y2)),
            range=_r2,
            overlaying='y', side='right', showgrid=False)
    if y3:
        layout['yaxis3'] = dict(
            title=dict(text=_titulo(y3), font=dict(color=_cor_eixo(y3))),
            tickfont=dict(color=_cor_eixo(y3)),
            range=_r3,
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

        # Ajustar o eixo à amplitude real desta métrica. Sem isto, métricas de
        # amplitude pequena (ex.: THb, que varia ~0.4 unidades) ficam esmagadas
        # porque o Plotly inclui o zero por defeito.
        _s = serie.dropna()
        if len(_s) > 0:
            _lo, _hi = float(_s.min()), float(_s.max())
            _amp = _hi - _lo
            if _amp > 0:
                _f0, _f1 = _lo - _amp * 0.08, _hi + _amp * 0.08
                if _lo >= 0:
                    _f0 = max(0.0, _f0)
                fig.update_yaxes(range=[_f0, _f1], row=i, col=1)
            else:
                _mg = max(abs(_lo) * 0.01, 0.5)
                fig.update_yaxes(range=[_lo - _mg, _hi + _mg], row=i, col=1)

    # Sombrear laps: trabalho (vermelho) e excluídos (cinzento)
    for l in lap_stats:
        fase = l.get('phase')
        if fase not in ('work', 'excluded', 'recovery'):
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        cor = {'work': 'rgba(214,39,40,0.09)',
               'excluded': 'rgba(128,128,128,0.16)',
               'recovery': 'rgba(52,152,219,0.07)'}[fase]
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


def _grafico_limitador_bruto(df, colunas, lap_stats):
    """
    Traço bruto de SmO2 (eixo esquerdo) + THb (eixo direito) ao longo do
    tempo, com sombreado de trabalho/descanso — o mesmo estilo de gráfico
    usado nos casos de estudo do fórum Moxy/NNOXX (SmO2 + THb + fase),
    para o utilizador ver visualmente o que gerou a classificação.
    """
    if 'smo2' not in colunas or colunas['smo2'] not in df.columns:
        return None

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    tmin = df['time_seconds'].min()
    t_min_rel = (df['time_seconds'] - tmin) / 60.0

    smo2 = pd.to_numeric(df[colunas['smo2']], errors='coerce')
    fig.add_trace(go.Scatter(
        x=t_min_rel, y=smo2, mode='lines', name='SmO₂ (%)',
        line=dict(color=_CORES_METRICA['smo2'], width=1.6),
        hovertemplate='SmO₂: %{y:.1f}%<extra></extra>'), secondary_y=False)

    if 'thb' in colunas and colunas['thb'] in df.columns:
        thb = pd.to_numeric(df[colunas['thb']], errors='coerce')
        fig.add_trace(go.Scatter(
            x=t_min_rel, y=thb, mode='lines', name='THb',
            line=dict(color=_CORES_METRICA['thb'], width=1.4),
            hovertemplate='THb: %{y:.2f}<extra></extra>'), secondary_y=True)
        _t = thb.dropna()
        if len(_t) > 0:
            _lo, _hi = float(_t.min()), float(_t.max())
            _amp = max(_hi - _lo, 1e-6)
            fig.update_yaxes(range=[_lo - _amp * 0.15, _hi + _amp * 0.15],
                             secondary_y=True)

    # Sombrear trabalho (vermelho) / descanso (azul) — mesmo padrão do
    # _grafico_series, para reconhecimento visual consistente
    for l in lap_stats:
        fase = l.get('phase')
        if fase not in ('work', 'recovery'):
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) == 0:
            continue
        x0 = (d['time_seconds'].iloc[0] - tmin) / 60.0
        x1 = (d['time_seconds'].iloc[-1] - tmin) / 60.0
        cor = 'rgba(214,39,40,0.09)' if fase == 'work' else 'rgba(52,152,219,0.07)'
        fig.add_vrect(x0=x0, x1=x1, fillcolor=cor, line_width=0, layer='below')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=320, hovermode='x unified',
        margin=dict(t=30, b=45, l=55, r=55),
        legend=dict(orientation='h', y=1.08),
        font=dict(size=11))
    fig.update_xaxes(title_text='Tempo (min)', showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text='SmO₂ (%)', showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)', secondary_y=False)
    fig.update_yaxes(title_text='THb', showgrid=False, secondary_y=True)
    return fig


def _grafico_limitador_tendencias(lap_stats):
    """
    As 4 séries que o classificador de limitador realmente pontua, por lap
    sucessivo — para o utilizador confirmar visualmente a tendência (ou não)
    por trás do resultado, em vez de confiar só no texto.
    """
    laps_trabalho = [l for l in lap_stats if l.get('phase') == 'work']
    laps_descanso = [l for l in lap_stats if l.get('phase') == 'recovery']
    if len(laps_trabalho) < 2:
        return None

    def _v(l, est, todo):
        return l.get(est, l.get(todo))

    n_trab = list(range(1, len(laps_trabalho) + 1))
    n_desc = list(range(1, len(laps_descanso) + 1))
    min_smo2_trab = [_v(l, 'min_smo2_est', 'min_smo2') for l in laps_trabalho]
    max_smo2_desc = [_v(l, 'max_smo2_est', 'max_smo2') for l in laps_descanso]
    max_thb_trab  = [_v(l, 'max_thb_est', 'max_thb') for l in laps_trabalho]
    max_thb_desc  = [_v(l, 'max_thb_est', 'max_thb') for l in laps_descanso]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'SmO₂ — mínimo em trabalho / máximo em descanso',
        'THb — máximo em trabalho / máximo em descanso'))

    fig.add_trace(go.Scatter(
        x=n_trab, y=min_smo2_trab, mode='lines+markers', name='SmO₂ mín. (trabalho)',
        line=dict(color=_CORES_METRICA['smo2'], width=2), marker=dict(size=8),
        hovertemplate='Lap trabalho %{x}<br>SmO₂ mín: %{y:.1f}%<extra></extra>'),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=n_desc, y=max_smo2_desc, mode='lines+markers', name='SmO₂ máx. (descanso)',
        line=dict(color=_CORES_METRICA['smo2'], width=2, dash='dot'), marker=dict(size=8, symbol='diamond'),
        hovertemplate='Lap descanso %{x}<br>SmO₂ máx: %{y:.1f}%<extra></extra>'),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=n_trab, y=max_thb_trab, mode='lines+markers', name='THb máx. (trabalho)',
        line=dict(color=_CORES_METRICA['thb'], width=2), marker=dict(size=8),
        hovertemplate='Lap trabalho %{x}<br>THb máx: %{y:.2f}<extra></extra>'),
        row=1, col=2)
    fig.add_trace(go.Scatter(
        x=n_desc, y=max_thb_desc, mode='lines+markers', name='THb máx. (descanso)',
        line=dict(color=_CORES_METRICA['thb'], width=2, dash='dot'), marker=dict(size=8, symbol='diamond'),
        hovertemplate='Lap descanso %{x}<br>THb máx: %{y:.2f}<extra></extra>'),
        row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=320, margin=dict(t=40, b=45, l=50, r=20),
        legend=dict(orientation='h', y=-0.2), font=dict(size=11))
    fig.update_xaxes(title_text='Nº do lap (sucessivo)', showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig


def _grafico_wbal(wbal_res):
    """W' balance ao longo da sessão — % de W' restante, com a potência por baixo."""
    s = wbal_res['serie']
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(
        x=s['tempo_s'] / 60.0, y=s['wbal_pct'], mode='lines', name="W′ balance (%)",
        line=dict(color='#8E44AD', width=2),
        hovertemplate="W′ restante: %{y:.0f}%<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=s['tempo_s'] / 60.0, y=s['potencia'], mode='lines', name='Potência (W)',
        line=dict(color=_CORES_METRICA['power'], width=1.2),
        opacity=0.6, hovertemplate="Potência: %{y:.0f}W<extra></extra>"), secondary_y=True)
    fig.add_hline(y=5, line_dash='dot', line_color='#E74C3C', secondary_y=False,
                  annotation_text="quase vazio (≤5%)")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=320, hovermode='x unified', margin=dict(t=30, b=45, l=55, r=55),
        legend=dict(orientation='h', y=1.08), font=dict(size=11))
    fig.update_xaxes(title_text='Tempo (min)', showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="W′ balance (%)", range=[0, 105], showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)', secondary_y=False)
    fig.update_yaxes(title_text='Potência (W)', showgrid=False, secondary_y=True)
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

def tab_fit_analise(ac_full=None):
    st.header("🫁 Análise FIT — Fisiologia (MOXY / SmO₂ / DFA-α1 / Respiração)")
    st.caption(
        "Carrega um ficheiro .fit de uma sessão intervalada para analisar a resposta "
        "fisiológica: desoxigenação muscular (SmO₂/THb), complexidade autonómica (DFA-α1), "
        "respiração e cinética de recuperação entre intervalos.")

    _col_up, _col_mod = st.columns([3, 1])
    with _col_up:
        ficheiro = st.file_uploader(
            "Ficheiro .fit", type=['fit'], key='fit_upload',
            help="A sessão deve ter laps definidos (intervalos de trabalho e recuperação). "
                 "Métricas MOXY/DFA-α1 são detectadas automaticamente se existirem no ficheiro.")
    with _col_mod:
        modalidade_fit = st.selectbox(
            "Modalidade", ['Bike', 'Row', 'Ski', 'Run'], key='fit_modalidade',
            help="Usada para a secção 'Comparação com histórico' — para saber que dados "
                 "do Intervals.icu (HRVT1, HRVT2, PBP, etc.) procurar para esta modalidade.")

    with st.expander("⚡ CP e W′ (para o W′ Balance) — opcional", expanded=False):
        st.caption(
            "Introduz o CP e W′ já calculados (ex.: aba CP Model do dashboard principal) "
            "para veres como a reserva anaeróbia (W′) esgotou e recuperou ao longo desta "
            "sessão. Deixa a 0 para não calcular esta secção.")
        _cwc1, _cwc2 = st.columns(2)
        cp_wbal = _cwc1.number_input("CP (W)", min_value=0, value=0, step=5, key='fit_cp_wbal')
        wprime_wbal = _cwc2.number_input("W′ (Joules)", min_value=0, value=0, step=500,
                                         key='fit_wprime_wbal')

    with st.expander("📖 Que protocolos são suportados e o que fazer em cada um",
                     expanded=False):
        st.markdown("""
**📈 Rampa contínua** — a intensidade sobe sem paragens (ex.: +15 a 30 W/min).
É o protocolo dos estudos publicados e o que dá as melhores estimativas de limiares.
*Análises:* breakpoint SmO₂ (double-linear), HRVT2 (α1=0.50), HRVT1c, Combo.
*Dica:* rampas mais lentas (5-15 W/min) dão limiares de **potência** mais fiáveis.
A FC dos limiares não é afectada pela inclinação — a potência é.

**🪜 Degraus incrementais** — patamares de 3-5 min com intensidade crescente.
*Análises:* as mesmas, mas usando o estado estacionário do fim de cada degrau.
*Dica:* degraus de 5 min permitem também o método de estabilidade do SmO₂.

**🔁 Intervalos repetidos** — blocos à mesma intensidade com recuperações.
*Análises:* cinética de restauração, decoupling, fadiga. Os limiares são menos
fiáveis porque não há progressão de intensidade.

**➡️ Sessão contínua** (tempo run, zona 2 longa) — intensidade estável.
*Análises:* **durabilidade** (deriva de FC, respiração e DFA-α1 ao longo do tempo),
decoupling. Não há limiares a estimar.

---
**Para resultados fiáveis, em qualquer protocolo:**
- Estar **fresco** — um teste no dia seguinte a um esforço duro dá valores errados
  por supressão autonómica
- **Posição da cinta cardíaca**: o pico R deve ser maior que a onda S. Esta é uma
  das causas mais comuns de erro no DFA-α1
- Artefactos HRV **abaixo de 5%** (o painel de fiabilidade verifica isto)
        """)

    if ficheiro is None:
        _mostrar_historico()
        return

    # ── Análise ───────────────────────────────────────────────────────────────
    bytes_fit = ficheiro.getvalue()
    # Identificador por CONTEÚDO, não só pelo nome — ficheiros de teste com nomes
    # repetidos ou enganadores (ex.: dois ficheiros diferentes chamados
    # "moxy_remo_2026.fit") não devem herdar laps corrigidos nem análise já
    # feita de uma sessão anterior. Sem isto, reabrir um ficheiro com o mesmo
    # nome de outro já analisado disparava a Fase 2 (a mais pesada) logo no
    # upload, usando estado de uma sessão diferente.
    _fid = f'{ficheiro.name}_{hashlib.md5(bytes_fit).hexdigest()[:10]}'
    chave_manual = f'_fit_laps_manual_{_fid}'
    chave_excl = f'_fit_laps_excl_{_fid}'
    chave_edit_iv = f'_fit_edit_iv_{_fid}'
    chave_offsets = f'_fit_offsets_{_fid}'
    laps_manual = st.session_state.get(chave_manual)
    laps_excl = st.session_state.get(chave_excl, [])
    iv_editados = st.session_state.get(chave_edit_iv)

    # ── Definições da análise ────────────────────────────────────────────────
    with st.expander("⚙️ Definições da análise", expanded=False):
        janela = st.slider(
            "Janela de estado estacionário (segundos finais de cada lap)",
            min_value=0, max_value=180, value=60, step=10,
            key=f'janela_{_fid}',
            help="As médias de cada lap são calculadas só sobre os últimos N segundos. "
                 "Métricas como o SmO₂ têm cinética lenta (~30-60s) e no início do lap "
                 "ainda estão em transição da intensidade anterior. Usar o lap inteiro "
                 "sobrestima o SmO₂ e distorce os limiares. 0 = usar o lap inteiro.")
        if janela == 0:
            st.warning("⚠️ A usar o lap inteiro — as médias incluem a fase de transição, "
                       "o que tende a sobrestimar o SmO₂ e a deslocar os limiares.")

        zerar_pot = st.checkbox(
            "Zerar potência nos períodos de recuperação",
            value=False, key=f'zerar_{_fid}',
            help="Alguns ergómetros registam potência residual durante a pausa "
                 "(inércia do volante, movimento leve), o que inflaciona as médias "
                 "de recuperação e distorce o decoupling. Esta opção força a "
                 "potência e a cadência a zero nos laps de recuperação.")

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
            key=f'modo_{_fid}',
            help="Se a detecção automática não acertar no teu ficheiro, usa um dos "
                 "outros modos.")

        frac_corte = None
        intervalos = None
        min_dur_seg = 45

        if modo == 'corte':
            pct = st.slider(
                "Recuperação = abaixo de X% da intensidade de trabalho",
                min_value=20, max_value=90, value=50, step=5,
                key=f'pct_{_fid}',
                help="Tudo o que estiver abaixo desta percentagem da potência (ou FC) "
                     "típica de trabalho é considerado recuperação.")
            frac_corte = pct / 100.0
            min_dur_seg = st.slider(
                "Duração mínima de um bloco (s)", 10, 180, 45, 5,
                key=f'mindur_{_fid}',
                help="Blocos mais curtos são fundidos com o anterior, para evitar "
                     "dezenas de micro-intervalos por causa de oscilações.")

        elif modo == 'intervalos':
            st.caption("Escreve um intervalo de **trabalho** por linha. Tudo o que ficar "
                       "fora (os 'buracos') passa automaticamente a recuperação.")
            texto = st.text_area(
                "Intervalos de trabalho",
                value=st.session_state.get(f'txt_iv_{_fid}', ''),
                placeholder="10:00-13:00\n14:00-17:00\n18:00-21:00",
                height=140, key=f'txt_iv_{_fid}',
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
    # Intervalos editados na tabela de laps têm prioridade
    if iv_editados:
        _modo_seg = 'intervalos'
        intervalos = iv_editados

    _offsets = st.session_state.get(chave_offsets, {})

    # ── Cache em DUAS camadas ─────────────────────────────────────────────────
    # Camada A — leitura bruta do binário FIT (fitdecode). É a parte cara desta
    # fase, e NÃO depende dos laps nem de nenhuma definição — só do conteúdo do
    # ficheiro. Corrigir laps e clicar "Aplicar" nunca deveria obrigar a reler o
    # ficheiro outra vez, só a reprocessar os dados já decodificados.
    _chave_raw = f'_fit_raw_{_fid}'
    if _chave_raw not in st.session_state:
        with st.spinner("A ler o ficheiro..."):
            st.session_state[_chave_raw] = ler_fit(bytes_fit)
    _raw = st.session_state[_chave_raw]

    if 'erro' in _raw:
        st.error(f"❌ {_raw['erro']}")
        return

    # Camada B — segmentação/classificação de laps + definições (janela, modo,
    # offsets, zerar potência). Esta sim depende dos laps corrigidos — mas
    # parte sempre do _raw já decodificado (camada A), nunca relê o ficheiro.
    _chave_prep = f'_fit_prep_{_fid}'
    _assinatura_prep = (
        str(sorted(laps_manual or [])), str(sorted(laps_excl)),
        janela, _modo_seg, str(intervalos), frac_corte, min_dur_seg,
        zerar_pot, str(sorted(_offsets.items())),
    )
    _prep_cache = st.session_state.get(_chave_prep)
    if _prep_cache is not None and _prep_cache.get('_sig') == _assinatura_prep:
        res = _prep_cache['res']
    else:
        with st.spinner("A aplicar laps e definições..."):
            res = preparar_fit(
                bytes_fit, raw=_raw,
                laps_trabalho_manual=laps_manual, laps_excluidos=laps_excl,
                janela_final_s=janela, modo_segmentacao=_modo_seg,
                intervalos_trabalho=intervalos, frac_corte=frac_corte,
                min_dur_segmento=min_dur_seg, zerar_potencia_descanso=zerar_pot,
                offsets=_offsets)
        st.session_state[_chave_prep] = {'_sig': _assinatura_prep, 'res': res}

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

    # ── Qualidade do sinal HRV (Artifacts) ───────────────────────────────────
    # Artifacts = % de batimentos corrigidos/interpolados. Acima de ~5% o DFA-α1
    # desse período torna-se pouco fiável, porque é muito sensível a erros de RR.
    if 'artifacts' in colunas and 'dfa1' in colunas:
        _art_laps = [(l['lap_number'], l['avg_artifacts'])
                     for l in lap_stats
                     if l.get('phase') == 'work' and 'avg_artifacts' in l]
        _maus = [(n, v) for n, v in _art_laps if v > 5]
        if _maus:
            _txt_maus = ", ".join(f"lap {n} ({v:.0f}%)" for n, v in _maus)
            st.warning(
                f"⚠️ **Qualidade do sinal HRV** — artefactos acima de 5% em: {_txt_maus}. "
                "O DFA-α1 é muito sensível a erros de intervalo RR: nestes laps, "
                "interpreta-o com reserva. Podes adicionar 'Artifacts' ao gráfico para ver "
                "onde o sinal degradou.")
        elif _art_laps:
            _media_art = np.mean([v for _, v in _art_laps])
            st.caption(f"✅ Qualidade do sinal HRV boa (artefactos médios: {_media_art:.1f}%).")

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

    # Tabela editável: o utilizador marca aquecimento/trabalho e corrige os tempos.
    # Nada é recalculado enquanto edita — só ao carregar em "Aplicar".
    _t0_sessao = res['df']['time_seconds'].min()
    _linhas_edit = []
    for l in lap_stats:
        _ini_s = float(l.get('_t_ini', 0))
        _fim_s = float(l.get('_t_fim', l['duration']))
        _linhas_edit.append({
            'Lap': l['lap_number'],
            'Aquecimento': l['phase'] == 'excluded',
            'Trabalho': l['phase'] == 'work',
            'Fase actual': _FASE_LBL.get(l['phase'], l['phase']),
            'Início': _mmss(_ini_s),
            'Fim': _mmss(_fim_s),
            'Duração': _mmss(l['duration']),
            **{NOMES_METRICAS.get(m, m): round(l[f'avg_{m}'], 1)
               for m in ['power', 'heart_rate', 'smo2', 'dfa1', 'artifacts']
               if f'avg_{m}' in l},
        })
    _df_edit = pd.DataFrame(_linhas_edit)

    st.caption(
        "Marca os laps de **aquecimento** (excluídos da análise) e de **trabalho**. "
        "Tudo o que não for marcado conta automaticamente como **recuperação**. "
        "Marcar caixas não altera os tempos dos laps — só muda a classificação. "
        "Se precisares de corrigir as fronteiras, edita os campos de início/fim "
        "(mm:ss ou h:mm:ss); só nesse caso a sessão é re-segmentada. "
        "Nada é recalculado enquanto editas — só ao carregar no botão.")

    _editado = st.data_editor(
        _df_edit,
        hide_index=True, use_container_width=True,
        key=f'editor_{_fid}',
        column_config={
            'Lap': st.column_config.NumberColumn('Lap', disabled=True, width='small'),
            'Aquecimento': st.column_config.CheckboxColumn(
                '⚪ Aquec.', help='Excluir este lap de toda a análise', width='small'),
            'Trabalho': st.column_config.CheckboxColumn(
                '🏃 Trab.', help='Marcar como intervalo de trabalho', width='small'),
            'Fase actual': st.column_config.TextColumn(
                'Fase', disabled=True, width='small',
                help='Resultado: o que não for aquecimento nem trabalho é recuperação'),
            'Início': st.column_config.TextColumn(
                'Início', help='mm:ss ou h:mm:ss desde o início da sessão', width='small'),
            'Fim': st.column_config.TextColumn(
                'Fim', help='mm:ss ou h:mm:ss desde o início da sessão', width='small'),
            'Duração': st.column_config.TextColumn('Duração', disabled=True, width='small'),
        },
        disabled=[c for c in _df_edit.columns
                  if c not in ('Aquecimento', 'Trabalho', 'Início', 'Fim')])

    _ca, _cb = st.columns([1, 3])
    if _ca.button("✅ Aplicar alterações", key=f'aplicar_ed_{_fid}',
                  type='primary'):
        # IMPORTANTE: marcar aquecimento/trabalho NÃO re-segmenta a sessão.
        # As fronteiras dos laps ficam como estão — só mudam os rótulos. Assim os
        # tempos de início/fim/duração de cada lap nunca se alteram entre cliques.
        # Só se o utilizador editar de facto os tempos é que a sessão é
        # re-segmentada a partir dos novos intervalos.
        _novos_excl, _novos_work = [], []
        _tempos_alterados = False
        _erros_t = []
        for _, row in _editado.iterrows():
            _lp = int(row['Lap'])
            if bool(row['Aquecimento']):
                _novos_excl.append(_lp)
            elif bool(row['Trabalho']):
                _novos_work.append(_lp)
            # Detectar se os tempos foram editados face aos originais
            _orig = _df_edit[_df_edit['Lap'] == _lp]
            if len(_orig) == 1:
                if (str(row['Início']).strip() != str(_orig.iloc[0]['Início']).strip() or
                        str(row['Fim']).strip() != str(_orig.iloc[0]['Fim']).strip()):
                    _tempos_alterados = True

        if _tempos_alterados:
            # O utilizador mexeu nos tempos → re-segmentar pelos intervalos de trabalho
            _novos_iv = []
            for _, row in _editado.iterrows():
                if not bool(row['Trabalho']) or bool(row['Aquecimento']):
                    continue
                _iv, _er = parse_intervalos(f"{row['Início']}-{row['Fim']}")
                if _iv:
                    _novos_iv.append(_iv[0])
                else:
                    _erros_t.append(f"lap {int(row['Lap'])}: "
                                    f"{row['Início']}–{row['Fim']}")
            if _erros_t:
                st.error("Tempos inválidos em: " + ", ".join(_erros_t))
            else:
                st.session_state[chave_edit_iv] = _novos_iv
                st.session_state[chave_excl] = []
                st.session_state.pop(chave_manual, None)
                st.rerun()
        else:
            # Só rótulos → manter a segmentação actual, mudar apenas as fases
            st.session_state[chave_excl] = _novos_excl
            st.session_state[chave_manual] = _novos_work
            st.session_state.pop(chave_edit_iv, None)
            st.rerun()

    if _cb.button("↩️ Repor detecção automática", key=f'repor_{_fid}'):
        for _k in (chave_edit_iv, chave_excl, chave_manual):
            st.session_state.pop(_k, None)
        st.rerun()

    if st.session_state.get(chave_edit_iv):
        _n_iv = len(st.session_state[chave_edit_iv])
        st.info(f"⏱️ A usar {_n_iv} intervalos de trabalho com os tempos que definiste. "
                "Os períodos entre eles contam como recuperação.")
    elif st.session_state.get(chave_manual) is not None:
        _nw = len(st.session_state[chave_manual])
        _nx = len(st.session_state.get(chave_excl, []))
        st.info(f"✅ Classificação manual aplicada: {_nw} laps de trabalho, "
                f"{_nx} de aquecimento. Os restantes contam como recuperação. "
                "Os tempos dos laps mantêm-se inalterados.")

    if laps_excl:
        st.info(f"⚪ Laps excluídos da análise: {sorted(laps_excl)} "
                "(não entram nos limiares, restauração, decoupling nem fadiga).")

    todos_laps = [l['lap_number'] for l in lap_stats]

    st.markdown("---")

    # ── Séries temporais ──────────────────────────────────────────────────────
    st.markdown("### 📈 Séries temporais")
    _ORDEM = ['smo2', 'thb', 'hhb', 'o2hb', 'dfa1', 'artifacts', 'rr_ratio', 'respiration',
              'resp_enhanced', 'heart_rate', 'hr_alphahrv', 'power', 'cadence',
              'speed', 'cycle_length', 'distance']
    disponiveis = ([m for m in _ORDEM if m in colunas] +
                   [m for m in colunas if m not in _ORDEM])

    estilo = st.radio(
        "Estilo do gráfico", options=['multi', 'empilhado'],
        format_func=lambda e: {'multi': '📊 Eixos combinados (Y1/Y2/Y3)',
                               'empilhado': '📑 Painéis empilhados'}[e],
        horizontal=True, key=f'estilo_{_fid}')

    # ── Correcção de sincronia entre métricas ────────────────────────────────
    with st.expander("🔧 Corrigir sincronia entre métricas", expanded=False):
        st.caption(
            "Se uma métrica aparecer desfasada no tempo (patamares rectos que não "
            "coincidem com as outras, típico de erro de gravação), podes deslocá-la. "
            "Valores positivos empurram para a **direita** (mais tarde), negativos "
            "para a **esquerda**. Nada é recalculado enquanto mexes — só ao aplicar.")

        _sync_sel = st.multiselect(
            "Métricas a deslocar", options=disponiveis,
            default=list(_offsets.keys()),
            format_func=lambda m: NOMES_METRICAS.get(m, m),
            key=f'sync_sel_{_fid}')

        _novos_off = {}
        if _sync_sel:
            _cols_sync = st.columns(min(len(_sync_sel), 3))
            for _i, _m in enumerate(_sync_sel):
                with _cols_sync[_i % len(_cols_sync)]:
                    _novos_off[_m] = st.slider(
                        NOMES_METRICAS.get(_m, _m),
                        min_value=-60, max_value=60,
                        value=int(_offsets.get(_m, 0)), step=1,
                        key=f'off_{_m}_{_fid}',
                        help="Segundos a deslocar (+ = mais tarde)")

            _cs1, _cs2 = st.columns(2)
            if _cs1.button("✅ Aplicar sincronia", key=f'aplicar_sync_{_fid}',
                           type='primary'):
                st.session_state[chave_offsets] = {k: v for k, v in _novos_off.items() if v}
                st.rerun()
            if _cs2.button("↩️ Repor", key=f'repor_sync_{_fid}'):
                st.session_state.pop(chave_offsets, None)
                st.rerun()

            # ── Sugestão automática ──────────────────────────────────────────
            if len(_sync_sel) == 1:
                _m_sel = _sync_sel[0]
                st.markdown("**💡 Sugerir alinhamento**")
                _sg1, _sg2 = st.columns(2)
                _base = _sg1.radio(
                    "Alinhar com:",
                    options=['laps', 'metrica'],
                    format_func=lambda b: {
                        'laps': '🏃 Laps de trabalho (recomendado)',
                        'metrica': '📊 Outra métrica'}[b],
                    key=f'base_sync_{_fid}',
                    help="Os laps de trabalho são marcadores temporais nítidos "
                         "(a intensidade sobe de forma abrupta), por isso costumam "
                         "dar um alinhamento mais fiável do que comparar duas séries.")
                _dir = _sg2.radio(
                    "Direcção:",
                    options=['ambas', 'frente', 'tras'],
                    format_func=lambda x: {'ambas': '↔️ Ambas',
                                           'frente': '➡️ Só para a frente',
                                           'tras': '⬅️ Só para trás'}[x],
                    key=f'dir_sync_{_fid}',
                    help="Restringe a procura ao sentido que sabes ser o correcto.")

                _sug = None
                if _base == 'laps':
                    _sug = sugerir_offset_por_laps(
                        res['df'], colunas, _m_sel, lap_stats, direcao=_dir)
                    _ref_txt = f"{_sug['n_laps_trabalho']} laps de trabalho" if _sug else ''
                else:
                    _ref_op = [m for m in disponiveis if m != _m_sel]
                    if _ref_op:
                        _ref = st.selectbox(
                            "Métrica de referência", options=_ref_op,
                            format_func=lambda m: NOMES_METRICAS.get(m, m),
                            key=f'ref_sync_{_fid}')
                        _sug = sugerir_offset(res['df'], colunas, _ref, _m_sel)
                        _ref_txt = NOMES_METRICAS.get(_ref, _ref)

                if _sug:
                    _r = _sug['r']
                    _forca = ('forte' if abs(_r) >= 0.5 else
                              'moderada' if abs(_r) >= 0.25 else 'fraca')
                    _msg = (f"Sugestão: **{_sug['offset']:+d}s** "
                            f"(correlação {_r:.2f} — {_forca}, vs {_ref_txt})")
                    if abs(_r) < 0.25:
                        st.warning(
                            f"{_msg}. Correlação fraca — pode não haver desfasamento "
                            "real a corrigir, ou o problema ser de outra natureza "
                            "(ex.: o gravador repetir valores em vez de os deslocar).")
                    elif _sug['offset'] == 0:
                        st.success(
                            f"✅ {_msg}. **A métrica já parece sincronizada** — os "
                            "patamares que vês no gráfico podem ser um problema de "
                            "resolução do sinal (valores repetidos), não de "
                            "desfasamento temporal. Nesse caso deslocar não resolve.")
                    else:
                        st.info(f"{_msg}")
                    st.caption(
                        "Confirma sempre visualmente: atrasos fisiológicos reais "
                        "(o SmO₂ responde 20-40s depois da potência) **não** devem ser "
                        "corrigidos — só erros de gravação.")

        if _offsets:
            _txt_off = ", ".join(f"{NOMES_METRICAS.get(k, k)} {v:+d}s"
                                 for k, v in _offsets.items())
            st.info(f"🔧 Sincronia aplicada: {_txt_off}. Todas as análises "
                    "(limiares, restauração, decoupling) usam os dados corrigidos.")

    if estilo == 'multi':
        st.caption("Escolhe que métricas vão em cada eixo. Métricas no mesmo eixo "
                   "partilham a escala — junta as que têm grandezas parecidas.")
        ce1, ce2, ce3 = st.columns(3)
        _def1 = [m for m in ['smo2'] if m in colunas] or disponiveis[:1]
        _def2 = [m for m in ['power'] if m in colunas]
        _def3 = [m for m in ['heart_rate'] if m in colunas]
        y1 = ce1.multiselect("Eixo Y1 (esquerda)", disponiveis, default=_def1,
                             format_func=lambda m: NOMES_METRICAS.get(m, m),
                             key=f'y1_{_fid}')
        y2 = ce2.multiselect("Eixo Y2 (direita)", disponiveis, default=_def2,
                             format_func=lambda m: NOMES_METRICAS.get(m, m),
                             key=f'y2_{_fid}')
        y3 = ce3.multiselect("Eixo Y3 (extra)", disponiveis, default=_def3,
                             format_func=lambda m: NOMES_METRICAS.get(m, m),
                             key=f'y3_{_fid}')
        suav = st.slider("Suavização (média móvel, segundos)", 0, 60, 0, 5,
                         key=f'suav_{_fid}',
                         help="0 = dados brutos a 1Hz. Suavizar ajuda a ver a tendência "
                              "em métricas ruidosas como o DFA-α1.")
        fig = _grafico_multi_eixo(res['df'], colunas, lap_stats, y1, y2, y3, suav)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': True, 'scrollZoom': True},
                            key=f'g_multi_{_fid}')
            st.caption("🔴 Bandas vermelhas = trabalho · 🔵 azuis = recuperação · "
                       "⚪ cinzentas = excluídos. Podes fazer zoom e arrastar.")
        else:
            st.info("Escolhe pelo menos uma métrica.")
    else:
        default = [m for m in ['smo2', 'heart_rate', 'power'] if m in colunas] or disponiveis[:3]
        sel = st.multiselect(
            "Métricas a mostrar", options=disponiveis, default=default,
            format_func=lambda m: NOMES_METRICAS.get(m, m), key=f'series_{_fid}')
        if sel:
            fig = _grafico_series(res['df'], colunas, lap_stats, sel)
            if fig:
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_series_{_fid}')
                st.caption("🔴 Trabalho · 🔵 recuperação · ⚪ excluídos.")

    # ══════════════════════════════════════════════════════════════════════
    # FASE 2 — Análises fisiológicas
    # Só corre depois de o utilizador confirmar que os laps e o alinhamento
    # das métricas estão correctos. Antes disso, calcular limiares seria
    # trabalhar sobre dados que ainda vão ser corrigidos.
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    _chave_run = f'_fit_run_{_fid}'
    _chave_res = f'_fit_res_{_fid}'   # guarda o RESULTADO, não só um sinalizador

    _assinatura = (str(sorted(laps_excl)), str(sorted(laps_manual or [])),
                   str(iv_editados), str(sorted(_offsets.items())),
                   janela, zerar_pot, modo)
    _prev = st.session_state.get(f'{_chave_run}_sig')
    if _prev is not None and _prev != _assinatura:
        # Os dados mudaram desde a última análise — invalidar o resultado
        st.session_state.pop(_chave_run, None)
        st.session_state.pop(f'{_chave_run}_sig', None)
        st.session_state.pop(_chave_res, None)

    _ja_analisado = st.session_state.get(_chave_run) is not None

    _cb1, _cb2 = st.columns([1, 3])
    if _cb1.button("🔬 Analisar" if not _ja_analisado else "🔄 Reanalisar",
                   key=f'run_{_fid}', type='primary'):
        st.session_state[_chave_run] = True
        st.session_state[f'{_chave_run}_sig'] = _assinatura
        st.session_state.pop(_chave_res, None)  # força recálculo já a seguir
        st.rerun()
    _cb2.caption(
        "Confirma primeiro que os laps estão bem classificados e que as métricas "
        "estão alinhadas no gráfico. Só depois carrega em Analisar — as análises "
        "(limiares, cinética, DFA-α1, fiabilidade) usarão exactamente estes dados."
        if not _ja_analisado else
        "As análises abaixo usam os dados actuais. Se alterares laps, sincronia ou "
        "definições, carrega em Reanalisar.")

    if not st.session_state.get(_chave_run):
        _mostrar_historico()
        return

    # Só recalcula se ainda não houver resultado em cache para esta assinatura —
    # sem isto, qualquer clique nesta página (mesmo sem tocar nos laps) fazia
    # correr a Fase 2 inteira outra vez, incluindo o DFA-α1.
    if _chave_res not in st.session_state:
        with st.spinner("A analisar..."):
            st.session_state[_chave_res] = analisar_completo(
                res, metodo_detrend='local', comparar_detrend=False)
    res = st.session_state[_chave_res]
    if 'erro' in res:
        st.error(f"❌ {res['erro']}")
        return
    lap_stats = res['lap_stats']

    # ── Protocolo detectado (a partir dos laps já corrigidos) ─────────────────
    _proto = res.get('protocolo')
    if _proto and _proto.get('tipo') != 'indefinido':
        _ICONE = {'rampa': '📈', 'degraus': '🪜', 'intervalos': '🔁', 'continuo': '➡️'}
        _NOME = {'rampa': 'Rampa contínua', 'degraus': 'Degraus incrementais',
                 'intervalos': 'Intervalos repetidos', 'continuo': 'Intensidade contínua'}
        _t = _proto['tipo']
        st.info(
            f"{_ICONE.get(_t, '📊')} **Protocolo detectado: {_NOME.get(_t, _t)}** — "
            f"{_proto['motivo']}.\n\n"
            f"Método aplicado: {_proto['metodo_recomendado']}.")
        if _t == 'continuo':
            st.caption("Numa sessão de intensidade constante não há limiares a "
                       "detectar — a análise foca-se na durabilidade e no decoupling.")

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
                            config={'displayModeBar': False}, key=f'g_rest_{_fid}')

        with st.expander("📋 Detalhe por sequência"):
            st.dataframe(rest['tabela'], hide_index=True, use_container_width=True)

    # ── Limiares SmO2 ─────────────────────────────────────────────────────────
    lim = res['limiares']
    if lim and lim.get('media') is not None:
        st.markdown("---")
        st.markdown("### 🎯 Limiares de SmO₂")
        _n_deg = len(lim['pontos'])
        _jan_txt = (f"a média dos últimos **{janela}s**" if janela > 0
                    else "a média do **lap inteiro**")
        st.caption(
            f"Ponto de inflexão na desoxigenação muscular. Calculado sobre os "
            f"**{_n_deg} laps de trabalho** (recuperações e laps excluídos não entram), "
            f"usando {_jan_txt} de cada um — o estado estacionário daquela intensidade. "
            f"Três métodos independentes.")
        u = lim.get('unidade', 'W')
        lc = st.columns(4)
        lc[0].metric("Dmax", f"{lim['dmax']:.0f} {u}" if lim['dmax'] else "—")
        lc[1].metric("Quebra inclinação", f"{lim['quebra']:.0f} {u}" if lim['quebra'] else "—")
        lc[2].metric("Deflexão 1%", f"{lim['deflexao']:.0f} {u}" if lim['deflexao'] else "—")
        lc[3].metric("**Média**", f"{lim['media']:.0f} {u}",
                     f"{lim['fc_media']:.0f} bpm" if lim.get('fc_media') else None)

        st.plotly_chart(_grafico_limiares(lim), use_container_width=True,
                        config={'displayModeBar': False}, key=f'g_lim_{_fid}')

        _vals = [v for v in (lim['dmax'], lim['quebra'], lim['deflexao']) if v is not None]
        if len(_vals) >= 2:
            _disp = (max(_vals) - min(_vals)) / lim['media'] * 100
            if _disp > 15:
                st.warning(f"⚠️ Os três métodos divergem {_disp:.0f}% entre si — interpreta a "
                           "média com cautela. Mais laps em degraus dariam uma estimativa melhor.")
            else:
                st.success(f"✅ Os três métodos concordam (dispersão {_disp:.0f}%) — "
                           "estimativa fiável.")

    # ── Resumo de zonas (FC e potência) ──────────────────────────────────────
    _z = res.get('zonas')
    if _z and (_z.get('baixo') or _z.get('alto')):
        st.markdown("---")
        st.markdown("### 🎯 Zonas de treino estimadas")
        _rel = _z.get('relacao_pot_fc')
        st.caption(
            "A **FC é a referência principal** — é praticamente independente do "
            "protocolo, ao contrário da potência (Physiological Reports 2023). "
            "Os valores de potência são convertidos a partir da relação "
            "potência↔FC desta sessão."
            + (f" Ajuste: R²={_rel['r2']:.2f} sobre {_rel['n']} pontos."
               if _rel else ""))

        _b, _a = _z.get('baixo'), _z.get('alto')

        def _fmt(v, u):
            return f"{v:.0f} {u}" if v is not None else "—"

        _zc1, _zc2, _zc3 = st.columns(3)
        with _zc1:
            st.markdown("**🟢 Zona 1** (fácil)")
            if _b:
                st.metric("até", _fmt(_b.get('fc'), 'bpm'),
                          _fmt(_b.get('pot'), 'W'))
            else:
                st.metric("até", "—", "sem limiar baixo")
        with _zc2:
            st.markdown("**🟡 Zona 2** (moderada)")
            if _b and _a:
                st.metric("entre",
                          f"{_b.get('fc', 0):.0f}–{_a.get('fc', 0):.0f} bpm",
                          f"{_b.get('pot', 0):.0f}–{_a.get('pot', 0):.0f} W")
            elif _a:
                st.metric("até", _fmt(_a.get('fc'), 'bpm'), _fmt(_a.get('pot'), 'W'))
            else:
                st.metric("entre", "—", "")
        with _zc3:
            st.markdown("**🔴 Zona 3** (intensa)")
            if _a:
                st.metric("acima de", _fmt(_a.get('fc'), 'bpm'),
                          _fmt(_a.get('pot'), 'W'))
            else:
                st.metric("acima de", "—", "sem limiar alto")

        _orig = []
        if _b:
            _orig.append(f"**Limiar baixo (Z1→Z2):** {_b['origem']}"
                         + ("" if _b.get('fiavel') else " ⚠️ com reservas"))
        if _a:
            _orig.append(f"**Limiar alto (Z2→Z3):** {_a['origem']}"
                         + ("" if _a.get('fiavel') else " ⚠️ com reservas"))
        if _orig:
            st.caption(" · ".join(_orig))

        if _z.get('coerente') is False:
            st.warning("⚠️ O limiar baixo ficou **acima** do alto, o que é "
                       "fisiologicamente improvável. Pelo menos uma das "
                       "estimativas não é de confiança — vê o painel de "
                       "fiabilidade abaixo.")
        elif not _b:
            st.info("ℹ️ Sem limiar baixo: precisa de DFA-α1 (intervalos RR no "
                    "ficheiro). Só a fronteira Z2→Z3 foi estimada.")

        # Todas as estimativas, para comparação
        _alts = _z.get('alternativas') or []
        if _alts:
            with st.expander("📋 Todas as estimativas do limiar alto"):
                _rows = []
                for _al in _alts:
                    _rows.append({
                        'Método': _al['origem'],
                        'FC (bpm)': round(_al['fc']) if _al.get('fc') else None,
                        'Potência (W)': round(_al['pot']) if _al.get('pot') else None,
                        'Fiável': '✅' if _al.get('fiavel') else '⚠️',
                    })
                st.dataframe(pd.DataFrame(_rows), hide_index=True,
                             use_container_width=True)
                _fcs = [a['fc'] for a in _alts if a.get('fc')]
                if len(_fcs) >= 2:
                    _amp = max(_fcs) - min(_fcs)
                    if _amp <= 8:
                        st.success(f"✅ Os métodos concordam (amplitude "
                                   f"{_amp:.0f} bpm) — estimativa robusta.")
                    else:
                        st.warning(f"⚠️ Os métodos divergem {_amp:.0f} bpm. "
                                   "Prefere os marcados como fiáveis.")

    # ── Painel de fiabilidade ────────────────────────────────────────────────
    _fi = res.get('fiabilidade')
    if _fi:
        st.markdown("---")
        st.markdown("### 🚦 Fiabilidade dos resultados")
        _ICO = {'ok': '✅', 'aviso': '⚠️', 'mau': '❌', 'ausente': '➖'}
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:8px;"
            f"background:{_fi['cor']}1A;border-left:5px solid {_fi['cor']}'>"
            f"<b style='color:{_fi['cor']};font-size:16px'>"
            f"Fiabilidade {_fi['nivel']}</b><br>"
            f"<span style='font-size:13px'>{_fi['texto']}</span></div>",
            unsafe_allow_html=True)

        _tb_fi = pd.DataFrame([{
            'Critério': c['criterio'],
            '': _ICO.get(c['estado'], ''),
            'Resultado': c['detalhe'],
        } for c in _fi['criterios']])
        st.dataframe(_tb_fi, hide_index=True, use_container_width=True)

        with st.expander("ℹ️ De onde vêm estes critérios"):
            for c in _fi['criterios']:
                if c.get('fonte'):
                    st.caption(f"**{c['criterio']}** — {c['fonte']}")
            st.markdown("---")
            st.markdown("**Margens de erro esperadas na literatura**")
            st.caption(
                "Mesmo quando tudo corre bem, estes métodos têm limites de "
                "concordância largos face aos padrões-ouro laboratoriais. "
                "Convém ter isto presente ao usar os números:")
            for k, v in _fi['loa'].items():
                st.caption(f"- **{k.replace('_', ' ')}**: {v}")

    # ── Limiares fisiológicos (métodos da literatura NIRS/HRV) ───────────────
    _bp = res.get('bp_continuo')
    _ldfa = res.get('limiar_dfa1')
    if _bp or (_ldfa and 'limiares' in _ldfa):
        st.markdown("---")
        st.markdown("### 🔬 Limiares fisiológicos")
        st.caption(
            "Métodos da literatura de NIRS e HRV, que estimam os limiares "
            "**sem análise de gases**. Referência: estudos do grupo Murias/Rogers "
            "(JSCR 2024, MSSE 2024, IJSPP 2024, JSCR 2025).")
        st.info(
            "💡 **Use a FC como referência principal.** O estudo Physiological "
            "Reports 2023 mostrou que a **FC** dos limiares é praticamente "
            "independente da inclinação da rampa (15, 30 ou 45 W/min dão o mesmo "
            "resultado), mas a **potência** varia bastante — até 60 W de diferença "
            "no limiar alto entre uma rampa lenta e uma rápida. Os valores de "
            "potência aqui apresentados são estimativas derivadas da relação "
            "FC↔potência desta sessão, e só são comparáveis entre testes com o "
            "mesmo protocolo.")

        _cl1, _cl2 = st.columns(2)

        # VT1 pelo DFA-α1
        with _cl1:
            st.markdown("**VT1 / topo da zona 1 — DFA-α1 (método fixo)**")
            st.caption(
                "⚠️ Estes valores usam os limiares **fixos** (0.75/0.70/0.50). "
                "O 0.50 tem base matemática — é o valor de um padrão "
                "não-correlacionado. Mas o **0.75 é um palpite empírico**, e o "
                "estudo MSSE 2024 mostrou que sobrestima o VT1 em +16 bpm. "
                "Vê a secção **HRVT1c** mais abaixo para a versão individualizada.")
            if _ldfa and 'limiares' in _ldfa:
                _u = _ldfa['unidade']
                _v075 = _ldfa['limiares'].get(0.75)
                _v070 = _ldfa['limiares'].get(0.70)
                _v050 = _ldfa['limiares'].get(0.50)
                if _v070:
                    _ex = " ⚠️ extrapolado" if _v070['extrapolado'] else ""
                    st.metric("Limite de zona 1 (α1 = 0.70)",
                              f"{_v070['intensidade']:.0f} {_u}{_ex}")
                if _v075:
                    st.caption(f"α1 = 0.75 (≈VT1): {_v075['intensidade']:.0f} {_u}")
                if _v050:
                    st.caption(f"α1 = 0.50 (ruído branco, já bem acima do VT1): "
                               f"{_v050['intensidade']:.0f} {_u}")
                st.caption(f"Ajuste sobre {_ldfa['n_usados']} laps · R² = {_ldfa['r2']:.2f}")
                if _ldfa.get('descartados_artifacts'):
                    _dd = ", ".join(f"lap {x['lap']} ({x['artifacts']:.0f}%)"
                                    for x in _ldfa['descartados_artifacts'])
                    st.caption(f"⚠️ Descartados por artefactos >5%: {_dd}")
                if _ldfa['n_usados'] < 4:
                    st.warning("Poucos laps válidos — estimativa pouco fiável. "
                               "Um R² alto com 3 pontos não significa precisão.")
            else:
                _msg = _ldfa.get('erro') if _ldfa else 'sem DFA-α1 no ficheiro'
                st.info(f"Não calculado ({_msg}).")

            # ── Versão recalculada a partir do RR, por lap (degraus/intervalos) ──
            _ldfa_rec = res.get('limiar_dfa1_recalculado')
            if _ldfa_rec is not None:
                with st.expander(
                        "🔬 Comparar com DFA-α1 recalculado do RR (um ponto por lap)"):
                    st.caption(
                        "O quadro acima usa o stream **cru** do dispositivo, sem "
                        "correcção de artefactos. Esta versão usa o DFA-α1 "
                        "**recalculado a partir dos RR** (o mesmo do HRVT2 acima), "
                        "agregado num único ponto por lap de trabalho — pensado "
                        "para protocolos de degraus/intervalos com descanso "
                        "genuíno entre cada intensidade: cada degrau fica isolado, "
                        "sem misturar dados do descanso anterior ou seguinte.")
                    if 'limiares' in _ldfa_rec:
                        _ur = _ldfa_rec['unidade']
                        _r075 = _ldfa_rec['limiares'].get(0.75)
                        _r070 = _ldfa_rec['limiares'].get(0.70)
                        _r050 = _ldfa_rec['limiares'].get(0.50)

                        def _fmt_r(_r):
                            if not _r:
                                return "—"
                            _avisos = []
                            if _r['extrapolado']:
                                _avisos.append("extrapolado")
                            if not _r.get('fisiologicamente_plausivel', True):
                                _avisos.append("valor implausível")
                            _sufixo = f" ⚠️ {', '.join(_avisos)}" if _avisos else ""
                            return f"{_r['intensidade']:.0f} {_ur}{_sufixo}"

                        if _ldfa_rec.get('r2_muito_baixo'):
                            st.error(
                                f"R² = {_ldfa_rec['r2']:.2f} — o ajuste linear é "
                                "essencialmente ruído (não descreve nenhuma relação real "
                                "entre intensidade e DFA-α1 nesta sessão). Os valores "
                                "abaixo NÃO devem ser usados, estejam ou não marcados "
                                "como extrapolados — um R² tão baixo já os invalida por "
                                "si só.")

                        if _r070:
                            st.metric("Limite de zona 1 (α1 = 0.70) — recalculado",
                                      _fmt_r(_r070))
                        if _r075:
                            st.caption(f"α1 = 0.75 (≈VT1): {_fmt_r(_r075)}")
                        if _r050:
                            st.caption(f"α1 = 0.50: {_fmt_r(_r050)}")
                        st.caption(f"Ajuste sobre {_ldfa_rec['n_usados']} laps · "
                                   f"R² = {_ldfa_rec['r2']:.2f}")
                        _pts_r = _ldfa_rec.get('pontos')
                        if _pts_r is not None and len(_pts_r):
                            st.dataframe(
                                _pts_r[['lap', 'intensidade', 'dfa1_recalculado',
                                       'n_janelas', 'janela_efetiva_media_s']]
                                .rename(columns={
                                    'lap': 'Lap', 'intensidade': f'Intensidade ({_ur})',
                                    'dfa1_recalculado': 'DFA-α1 recalculado',
                                    'n_janelas': 'Nº janelas de 5s',
                                    'janela_efetiva_media_s': 'Janela efetiva média (s)'}),
                                hide_index=True, use_container_width=True)
                            if (_pts_r['janela_efetiva_media_s'] < 90).any():
                                st.caption(
                                    "ℹ️ Alguns laps têm janela efetiva bem abaixo de "
                                    "120s — provavelmente laps de trabalho curtos, "
                                    "onde a maior parte dos primeiros ~2 min foi "
                                    "descartada para não misturar com o descanso "
                                    "anterior. O valor ainda é válido, só com "
                                    "menos pontos a suportá-lo.")
                    else:
                        st.info(f"Não calculado ({_ldfa_rec.get('erro')}).")

        # MLSS pelo breakpoint de SmO₂
        with _cl2:
            st.markdown("**MLSS / início da zona 3 — breakpoint SmO₂**")
            if _bp:
                _bph = res.get('bp_hhb')
                if _bph:
                    _bc1, _bc2 = st.columns(2)
                    _bc1.metric(f"Breakpoint — SmO₂",
                                f"{_bp['breakpoint']:.0f} {_bp['unidade']}",
                                f"{_bp['fc']:.0f} bpm" if _bp.get('fc') else None)
                    _bc2.metric(f"Breakpoint — HHb",
                                f"{_bph['breakpoint']:.0f} {_bph['unidade']}",
                                f"{_bph['fc']:.0f} bpm" if _bph.get('fc') else None,
                                help="HHb = hemoglobina desoxigenada, derivada de "
                                     "SmO₂ e THb. É a métrica que os estudos NIRS "
                                     "analisam — serve de verificação cruzada.")
                    _dif = abs(_bp['breakpoint'] - _bph['breakpoint'])
                    if _dif <= 5:
                        st.caption(f"✅ Os dois sinais concordam (diferença "
                                   f"{_dif:.0f} {_bp['unidade']}) — a estimativa "
                                   "é robusta.")
                    else:
                        st.caption(f"⚠️ Diferença de {_dif:.0f} {_bp['unidade']} "
                                   "entre sinais — interpreta com alguma reserva.")
                else:
                    st.metric("Breakpoint (double-linear)",
                              f"{_bp['breakpoint']:.0f} {_bp['unidade']}")
                st.caption(f"Declive antes: {_bp['slope_antes']:.3f} · "
                           f"depois: {_bp['slope_depois']:.3f} %/{_bp['unidade']}")
                st.caption(f"Ajuste sobre {_bp['n_pontos']} pontos · R² = {_bp['r2']:.2f}")
                if _bp['coerente_recto_femoral']:
                    st.caption("✅ Padrão de aceleração da desoxigenação — o esperado "
                               "no recto femoral.")
                else:
                    st.warning(f"⚠️ Padrão detectado: {_bp['padrao']}. No recto femoral "
                               "espera-se aceleração da queda; se o sensor estiver no "
                               "vasto lateral o padrão correcto seria um plateau.")
                if _bp['r2'] < 0.8:
                    st.warning("R² baixo — o modelo de duas rectas não descreve bem "
                               "estes dados. Interpreta com reserva.")
            else:
                st.info("Não calculado (sem SmO₂ ou dados insuficientes).")

        if _bp and _ldfa and 'limiares' in _ldfa and _ldfa['limiares'].get(0.70):
            _vt1 = _ldfa['limiares'][0.70]['intensidade']
            _mlss = _bp['breakpoint']
            if _bp['unidade'] == _ldfa['unidade']:
                if _vt1 < _mlss:
                    st.success(
                        f"✅ **Zonas estimadas:** Z1 até ~{_vt1:.0f}{_bp['unidade']} · "
                        f"Z2 entre {_vt1:.0f} e {_mlss:.0f} · Z3 acima de ~{_mlss:.0f}. "
                        "A ordem é fisiologicamente coerente (VT1 abaixo do MLSS).")
                else:
                    st.warning(
                        f"⚠️ O VT1 estimado ({_vt1:.0f}{_bp['unidade']}) ficou **acima** do "
                        f"MLSS ({_mlss:.0f}{_bp['unidade']}), o que é fisiologicamente "
                        "improvável. Provavelmente um dos ajustes não é fiável — verifica "
                        "o R² e o número de laps de cada um.")
        _gc1, _gc2 = st.columns(2)
        if _ldfa and 'limiares' in _ldfa and len(_ldfa.get('pontos', [])) >= 3:
            with _gc1:
                st.plotly_chart(_grafico_dfa1(_ldfa), use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_dfa1_{_fid}')
        if _bp:
            with _gc2:
                st.plotly_chart(_grafico_double_linear(_bp), use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_dbl_{_fid}')

        st.caption("Nota: o breakpoint corresponde à intensidade no momento da transição. "
                   "A literatura sugere subtrair 10-15 W para compensar o atraso da resposta "
                   "metabólica (MRT) em rampas rápidas; em degraus longos como estes o "
                   "efeito é menor (~2-10 W).")

        # ── LT1 + LT2 via SmO₂ (2 breakpoints no mesmo sinal) ────────────────
        _bp2 = res.get('bp_lt1_lt2')
        if _bp2:
            st.markdown("---")
            st.markdown("#### 📐 LT1 + LT2 via SmO₂ (2 breakpoints)")
            st.caption(
                "Andri Feldmann (fórum de developers da Moxy): \"vais encontrar duas "
                "mudanças distintas na inclinação/taxa do SmO₂. A primeira queda é o "
                "LT1; a segunda é uma segunda queda clara ou um achatamento — isso é "
                "o LT2.\" Dá um segundo ponto de vista sobre o LT2 (independente do "
                "DFA-α1/HRVT2), e uma primeira estimativa de LT1 a partir do SmO₂.")
            _l1c1, _l1c2 = st.columns(2)
            _l1c1.metric("LT1 (moderado→pesado)",
                        f"{_bp2['breakpoint_lt1']:.0f} {_bp2['unidade']}")
            _l1c2.metric("LT2 (pesado→severo)",
                        f"{_bp2['breakpoint_lt2']:.0f} {_bp2['unidade']}"
                        + ("" if _bp2['fiavel_lt2'] else " ⚠️"))
            st.caption(f"Declives: moderado {_bp2['slope_1']:.3f} → pesado "
                      f"{_bp2['slope_2']:.3f} → severo {_bp2['slope_3']:.3f} "
                      f"%/{_bp2['unidade']} · R² = {_bp2['r2']:.2f} · "
                      f"padrão no LT2: {_bp2['padrao_lt2']}")
            if not _bp2['fiavel_lt2']:
                st.warning(f"LT2 pouco fiável: {_bp2['motivo_lt2']}")
            if _bp2['r2'] < 0.8:
                st.warning("R² baixo — o modelo de 3 segmentos não descreve bem estes "
                          "dados. Interpreta com reserva (precisa de mais pontos/gama "
                          "de intensidade do que o ajuste de 1 breakpoint).")
            st.plotly_chart(_grafico_triplo_linear(_bp2), use_container_width=True,
                            config={'displayModeBar': False},
                            key=f'g_triplo_{_fid}')
            st.warning(_bp2['aviso'])
        elif res.get('protocolo') in ('degraus', 'intervalos'):
            st.markdown("---")
            st.markdown("#### 📐 LT1 + LT2 via SmO₂ (2 breakpoints)")
            st.caption(
                "Não calculado: dividir em 3 segmentos (2 quebras) precisa de mais "
                "degraus do que 1 quebra — com poucos, o último segmento fica preso a "
                "pontos de um só lap, o que dá um ajuste instável (às vezes até com o "
                "declive ao contrário). Repete com pelo menos 6 laps de trabalho para "
                "veres esta secção.")

        # ── HRVT2 pelo DFA-α1 recalculado + Combo (Murias 2023) ──────────────
        _h2 = res.get('hrvt2')
        _cb = res.get('combo')
        _sdfa = res.get('dfa1_serie')
        if _h2 or _cb:
            st.markdown("---")
            st.markdown("#### 🫀 HRVT2 e Combo (método Murias 2023)")
            st.caption(
                "O DFA-α1 é recalculado a partir dos intervalos RR do ficheiro "
                "(janelas de 2 min, passo 5 s, escalas 4-16 batimentos — os "
                "parâmetros do estudo). **α1 = 0.50 marca o limiar ALTO** "
                "(HRVT2 ≈ RCP/MLSS), não o VT1. O estudo mostrou que a média do "
                "HRVT2 com o breakpoint NIRS tem menor erro individual do que "
                "qualquer método isolado.")
            if res.get('protocolo', {}).get('tipo') in ('intervalos', 'degraus'):
                st.caption(
                    "ℹ️ Protocolo de intervalos: cada janela de 2 min é recortada para "
                    "não misturar batimentos de laps diferentes (ex.: recuperação + "
                    "trabalho seguinte). Perto do início de cada intervalo a janela "
                    "fica mais curta — e por isso menos precisa — até acumular tempo "
                    "suficiente dentro do mesmo lap.")

            _q = res.get('dfa1_qualidade')
            if _q:
                _cq1, _cq2, _cq3 = st.columns(3)
                _cq1.metric("Intervalos RR", f"{_q['n_total']}")
                _cq2.metric("Artefactos corrigidos",
                            f"{_q['pct_artefactos']:.1f}%",
                            delta="acima de 5%" if _q['pct_artefactos'] > 5 else "aceitável",
                            delta_color="inverse" if _q['pct_artefactos'] > 5 else "normal")
                _cq3.metric("Janelas de α1", f"{len(_sdfa) if _sdfa is not None else 0}")
                if _q['pct_artefactos'] > 5:
                    st.warning(
                        f"⚠️ {_q['pct_artefactos']:.1f}% de artefactos corrigidos. O estudo "
                        "exclui registos acima de 5% — o DFA-α1 é muito sensível a erros "
                        "de intervalo RR. Verifica a posição da cinta cardíaca.")

            if _h2 and 'erro' not in _h2:
                if _h2.get('fiavel'):
                    _cm1, _cm2 = st.columns(2)
                    _cm1.metric("HRVT2 — FC", f"{_h2['fc']:.0f} bpm",
                                help="Valor principal: a FC do limiar é estável "
                                     "entre protocolos")
                    if _h2.get('potencia'):
                        _cm2.metric("HRVT2 — Potência (estimada)",
                                    f"{_h2['potencia']:.0f} W",
                                    help="Secundário: depende da inclinação da rampa")
                    st.caption(f"Regressão na secção linear da curva α1 vs FC · "
                               f"R²={_h2['r2']:.2f} · n={_h2['n_pontos']} janelas")
                else:
                    st.error(
                        "❌ **HRVT2 não fiável neste teste** — não deve ser usado:\n\n"
                        + "\n".join(f"- {a}" for a in _h2.get('avisos', []))
                        + "\n\nPara obter um HRVT2 válido, o protocolo precisa de levar "
                          "o α1 claramente abaixo de 0.5 (intensidade suficiente) com sinal "
                          "de HRV limpo. Rampas contínuas funcionam melhor do que degraus "
                          "curtos, porque o α1 precisa de janelas de 2 min.")
            elif _h2:
                st.info(f"HRVT2 não calculado: {_h2.get('erro')}")
            elif res.get('rr_info') is None:
                st.info("Este ficheiro não contém intervalos RR — o DFA-α1 não pode ser "
                        "recalculado. Só ficheiros que gravam RR (Garmin, apps com Polar H10) "
                        "permitem esta análise.")

            # Combo
            if _cb:
                st.markdown("**🎯 Estimativa combinada do limiar alto**")
                _cc1, _cc2, _cc3 = st.columns(3)
                _cc1.metric("HRVT2 (DFA-α1)",
                            f"{_cb['hrvt2']:.0f} W" if _cb['hrvt2'] else "—")
                _cc2.metric("NIRS breakpoint",
                            f"{_cb['nirs']:.0f} W" if _cb['nirs'] else "—")
                _cc3.metric("**COMBO**", f"{_cb['combo']:.0f} W",
                            f"{_cb['fc']:.0f} bpm" if _cb.get('fc') else None)
                if _h2 and 'erro' not in _h2 and _h2.get('fc'):
                    st.caption(f"Em FC (mais estável entre protocolos): "
                               f"HRVT2 = {_h2['fc']:.0f} bpm")

                if _cb.get('hrv_descartado'):
                    st.info(
                        "ℹ️ O HRVT2 foi **excluído do combo** por não ser fiável neste teste. "
                        "A estimativa usa apenas o NIRS. Isto é exactamente o cenário de "
                        "'falha técnica' que o estudo descreve: quando um método falha, o "
                        "outro ainda dá um resultado utilizável.")
                elif _cb['estado'] == 'concordante':
                    st.success(
                        f"✅ Os dois métodos concordam (divergência {_cb['divergencia']:.0f} W, "
                        f"{_cb['divergencia_pct']:.0f}%). Segundo o estudo, a média tem menor "
                        "erro individual do que qualquer um isolado.")
                elif _cb['estado'] == 'divergente':
                    st.warning(
                        f"⚠️ Os métodos divergem {_cb['divergencia']:.0f} W "
                        f"({_cb['divergencia_pct']:.0f}%). Uma divergência grande sugere que "
                        "pelo menos um dos sinais tem problemas. A média continua a ser a "
                        "melhor aposta, mas com menos confiança — vale a pena repetir o teste.")

            # Gráfico da curva α1
            if _sdfa is not None and len(_sdfa) >= 10:
                st.plotly_chart(_grafico_curva_dfa1(_sdfa, _h2),
                                use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_curva_dfa1_{_fid}')

        # ── HRVT1c — ponto médio individual (IJSPP 2024) ─────────────────────
        _h1c = res.get('hrvt1c')
        if _h1c and 'erro' not in _h1c:
            st.markdown("---")
            st.markdown("#### 🎯 HRVT1c — limiar baixo com ponto médio individual")
            st.caption(
                "**Porque é que 0.75 é arbitrário e 0.50 não é:** o valor 0.50 tem "
                "significado matemático — corresponde a um padrão de batimentos "
                "**não-correlacionado** (ruído branco), com perda das propriedades "
                "fractais. Por isso a concordância HRVT2↔RCP é forte (viés <1 bpm). "
                "Já o 0.75 foi escolhido como ponto médio hipotético entre 1.0 e "
                "0.5 — o próprio autor lhe chama *\"palpite empírico\"*. Como nem "
                "toda a gente parte de 1.0, quem começa mais alto fica com o limiar "
                "sobrestimado: o estudo IJSPP 2024 mediu um viés de **+16 bpm**.\n\n"
                "**A correcção:** em vez de 0.75 fixo, usa-se o ponto médio "
                "individual `(α1_máximo_inicial + 0.50) / 2`. No estudo, isso "
                "reduziu o viés de +16 para **+2 bpm** e estreitou os limites de "
                "concordância de ±35 para ±26 bpm.")
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("α1 máximo inicial", f"{_h1c['max_inicial_dfa1']:.2f}")
            _c2.metric("Alvo individual", f"{_h1c['alvo_individual']:.2f}",
                       delta=f"vs 0.75 fixo")
            _c3.metric("HRVT1c", f"{_h1c['fc']:.0f} bpm")
            st.caption(
                f"Cálculo: (α1 máximo inicial **{_h1c['max_inicial_dfa1']:.2f}** + "
                f"0.50) ÷ 2 = **{_h1c['alvo_individual']:.2f}** — este é o teu ponto "
                "médio entre um padrão bem correlacionado e um não-correlacionado.")
            if _h1c.get('diferenca_vs_fixo') is not None:
                _d = _h1c['diferenca_vs_fixo']
                st.caption(
                    f"O método fixo (α1=0.75) daria **{_h1c['fc_metodo_fixo']:.0f} bpm** — "
                    f"uma diferença de **{_d:+.0f} bpm**. "
                    + ("A correcção individual é a que a literatura recomenda."
                       if abs(_d) > 5 else
                       "Neste caso os dois métodos quase coincidem."))
            if not _h1c.get('fiavel', True):
                st.warning("⚠️ " + "; ".join(_h1c.get('avisos', [])))

        # ── HRVT2 submáximo (JSCR 2025) ──────────────────────────────────────
        _sub = res.get('hrvt2_submax')
        if _sub:
            st.markdown("---")
            st.markdown("#### 📉 HRVT2 previsto por dados submáximos")
            st.caption(
                "Prevê o limiar alto **sem chegar à exaustão**, extrapolando a recta "
                "do α1 no troço 1.5→0.75 (que se atinge dentro da zona 2). "
                "Vantagem: pode repetir-se com frequência sem afectar o treino.")
            if 'erro' in _sub:
                st.info(f"Não calculado: {_sub['erro']}")
            else:
                _s1, _s2, _s3 = st.columns(3)
                _s1.metric("HRVT2 previsto", f"{_sub['fc']:.0f} bpm")
                _s2.metric("FC máxima (janela submáxima)", f"{_sub['fc_max_medida']:.0f} bpm")
                _s3.metric("Extrapolação", f"{_sub['extrapolacao_bpm']:+.0f} bpm")
                st.caption(f"Ajuste sobre {_sub['n_pontos']} janelas · "
                           f"R²={_sub['r2']:.2f} · ondulação {_sub['ondulacao_pct']:.0f}% · "
                           "nota: esta FC máxima é só dentro da janela submáxima (α1 "
                           "0.75-1.5) usada para a recta — pode ser menor do que a FC "
                           "máxima medida na sessão inteira, referida no aviso de "
                           "plausibilidade abaixo.")
                if _sub['fiavel']:
                    st.success("✅ Previsão fiável — a recta é inequívoca e a "
                               "extrapolação é curta.")
                else:
                    st.warning("⚠️ **Previsão pouco fiável:**\n\n"
                               + "\n".join(f"- {a}" for a in _sub['avisos']))

        # ── Análise NIRS: HHb e SmO₂ ─────────────────────────────────────────
        _bph = res.get('bp_hhb')
        if _bph or ('hhb' in colunas):
            st.markdown("---")
            st.markdown("#### 🩸 Análise NIRS — HHb e SmO₂")
            st.caption(
                "O **HHb** (hemoglobina desoxigenada) é a métrica que os estudos "
                "de NIRS analisam, não o SmO₂ directamente. Deriva-se dos dois "
                "sinais do sensor: `HHb = THb × (1 − SmO₂/100)`.\n\n"
                "**Porque importa:** o SmO₂ é uma *proporção* (satura em "
                "intensidades altas); o HHb é uma *quantidade absoluta* e mantém "
                "amplitude dinâmica precisamente onde o SmO₂ começa a achatar. "
                "Como são derivados um do outro, o breakpoint deve coincidir — "
                "e isso serve de validação cruzada.")

            st.plotly_chart(
                _grafico_hhb_temporal(res['df'], colunas, lap_stats),
                use_container_width=True, config={'displayModeBar': False},
                key=f'g_hhb_temp_{_fid}')
            st.caption("🔴 Trabalho · 🔵 recuperação · ⚪ excluídos. "
                       "O HHb sobe com a intensidade (mais extracção de O₂), "
                       "o O₂Hb desce, e o THb mantém-se relativamente estável.")

            if _bph and _bp:
                st.plotly_chart(
                    _grafico_breakpoint_hhb(_bp, _bph),
                    use_container_width=True, config={'displayModeBar': False},
                    key=f'g_bp_hhb_{_fid}')
                _dif = abs(_bp['breakpoint'] - _bph['breakpoint'])
                _cn1, _cn2, _cn3 = st.columns(3)
                _cn1.metric("Breakpoint HHb",
                            f"{_bph['breakpoint']:.0f} {_bph['unidade']}")
                _cn2.metric("Breakpoint SmO₂",
                            f"{_bp['breakpoint']:.0f} {_bp['unidade']}")
                _cn3.metric("Diferença", f"{_dif:.0f} {_bp['unidade']}",
                            delta="concordante" if _dif <= 5 else "divergente",
                            delta_color="normal" if _dif <= 5 else "inverse")
                st.caption(
                    f"HHb: declive {_bph['slope_antes']:+.3f} → "
                    f"{_bph['slope_depois']:+.3f} · R²={_bph['r2']:.2f} · "
                    f"{_bph['padrao']}")

            # Amplitude dinâmica dos dois sinais nos laps de trabalho
            _w = [l for l in lap_stats if l.get('phase') == 'work']
            if _w and 'avg_hhb' in _w[0]:
                _hh = [l['avg_hhb'] for l in _w if 'avg_hhb' in l]
                _ss = [l['avg_smo2'] for l in _w if 'avg_smo2' in l]
                if len(_hh) >= 2 and len(_ss) >= 2:
                    st.caption(
                        f"Amplitude nos degraus de trabalho — "
                        f"HHb: {min(_hh):.2f} a {max(_hh):.2f} "
                        f"(Δ {max(_hh)-min(_hh):.2f}) · "
                        f"SmO₂: {min(_ss):.1f}% a {max(_ss):.1f}% "
                        f"(Δ {max(_ss)-min(_ss):.1f} pontos)")

        # ── MLSS por intervalos longos (artigo 2019/03) ──────────────────────
        _mi = res.get('mlss_intervalos')
        if _mi:
            st.markdown("---")
            st.markdown("#### 🎯 MLSS por intervalos longos — método de referência")
            st.caption(
                f"Compara o comportamento do **{_mi['sinal']}** dentro de cada bloco de "
                "intensidade constante. Abaixo do MLSS o sinal estabiliza; acima, "
                "deriva continuamente. O MLSS fica entre a intensidade mais alta "
                "estável e a mais baixa instável.\n\n"
                "**Porquê este método:** a literatura mostra que os breakpoints por "
                "rampa têm erro acima de 10 W — e exercitar apenas +10 W acima do "
                "MLSS (~3-5%) já provoca subida progressiva do lactato e prejudica o "
                "desempenho. Este método por blocos é o que o autor considera mais "
                "fiável.")

            if _mi['usa_hhb']:
                st.caption("ℹ️ A análise usa o **HHb** (hemoglobina desoxigenada, "
                           "derivado de SmO₂ e THb) — é a métrica dos estudos, e "
                           "mantém amplitude dinâmica em intensidades altas onde o "
                           "SmO₂ começa a achatar.")

            _tbm = _mi['tabela'].copy()
            _cols_show = [c for c in _tbm.columns if c not in
                          ('estavel', 'tendencia_credivel')]
            st.dataframe(_tbm[_cols_show], hide_index=True, use_container_width=True)

            _lo, _hi = _mi['mlss_entre']
            if _mi['estado'] == 'enquadrado':
                _txt_fc = (f" · em FC: ~{_mi['mlss_fc']:.0f} bpm"
                           if _mi.get('mlss_fc') else "")
                _msg = (f"✅ **MLSS entre {_lo:.0f} e {_hi:.0f} {_mi['unidade']}** — "
                        f"estimativa **{_mi['mlss_estimado']:.0f} {_mi['unidade']}**"
                        f"{_txt_fc}. Janela de ±{_mi['largura_janela']/2:.0f} "
                        f"{_mi['unidade']} (precisão {_mi['precisao']}).")
                if _mi['precisao'] == 'boa':
                    st.success(_msg + " A janela está dentro dos ±10 W que a "
                               "literatura aponta como limite crítico.")
                else:
                    st.warning(_msg + " Para estreitar a janela, inclui blocos com "
                               "intensidades mais próximas entre si.")
            elif _mi['estado'] == 'abaixo_do_testado':
                st.warning(f"⚠️ Todos os blocos derivam — o MLSS estará **abaixo de "
                           f"{_hi:.0f} {_mi['unidade']}**. Repete incluindo blocos "
                           "mais fáceis.")
            elif _mi['estado'] == 'acima_do_testado':
                st.info(f"ℹ️ Todos os blocos estáveis — o MLSS estará **acima de "
                        f"{_lo:.0f} {_mi['unidade']}**. Repete incluindo blocos mais "
                        "intensos.")
            else:
                st.warning("⚠️ Resposta inconsistente: há blocos instáveis abaixo de "
                           "blocos estáveis. Pode indicar variação de cadência, "
                           "pacing irregular, ou blocos curtos demais para o padrão "
                           "se manifestar.")

            st.caption(
                f"Ignorados os primeiros {_mi['ignorar_inicio_s']}s de cada bloco "
                "(transição da intensidade anterior). Um bloco só é classificado "
                "como instável se a tendência for consistente (R²≥0.25) — evita "
                "confundir ruído com deriva real.")

        # ── Método alternativo: estabilidade do SmO₂ dentro de cada intervalo ──
        _est = res.get('estabilidade_smo2')
        if _est:
            st.markdown("---")
            st.markdown("#### 📉 MLSS por estabilidade intra-intervalo")
            st.caption(
                "Método alternativo (o preferido do blog para intervalos a potência "
                "constante): em vez de olhar para a curva SmO₂-vs-potência **entre** "
                "degraus, olha para o comportamento **dentro** de cada degrau. "
                "Se o SmO₂ estabiliza, estás abaixo do MLSS; se desce continuamente, "
                "estás acima. O MLSS fica entre os dois.")

            _tb = _est['tabela'].copy()
            _tb['comportamento'] = _tb['estavel'].map(
                {True: '✅ estável', False: '📉 declínio contínuo'})
            st.dataframe(
                _tb[['lap', 'intensidade', 'smo2_inicio', 'smo2_fim', 'delta_smo2',
                     'slope_pct_min', 'comportamento']].rename(columns={
                        'lap': 'Lap', 'intensidade': f"Intensidade ({_est['unidade']})",
                        'smo2_inicio': 'SmO₂ início', 'smo2_fim': 'SmO₂ fim',
                        'delta_smo2': 'Δ SmO₂', 'slope_pct_min': 'Declive (%/min)',
                        'comportamento': 'Comportamento'}),
                hide_index=True, use_container_width=True)

            _lo, _hi = _est['mlss_entre']
            if _est['confianca'] == 'boa':
                st.success(
                    f"✅ **MLSS entre {_lo:.0f} e {_hi:.0f} {_est['unidade']}** "
                    f"(estimativa: {_est['mlss_estimado']:.0f} {_est['unidade']}) — "
                    f"último degrau estável a {_lo:.0f}, primeiro instável a {_hi:.0f}.")
            elif _est['confianca'] == 'todos instáveis':
                st.warning(
                    f"⚠️ Todos os intervalos em declínio contínuo — o MLSS estará "
                    f"**abaixo de {_hi:.0f} {_est['unidade']}** (a intensidade mais baixa "
                    f"testada). Para o localizar, inclui intervalos mais fáceis. "
                    f"Nota: intervalos curtos favorecem este resultado, porque o SmO₂ "
                    f"pode ainda estar em transição — aqui a duração mediana analisada "
                    f"foi {_est['duracao_mediana_s']:.0f}s.")
            elif _est['confianca'] == 'todos estáveis':
                st.info(
                    "ℹ️ Todos os degraus estáveis — o MLSS estará **acima** da intensidade "
                    "mais alta testada. Repete incluindo degraus mais intensos.")
            else:
                st.warning(
                    "⚠️ Resposta inconsistente: há degraus instáveis abaixo de degraus "
                    "estáveis. Pode indicar variação de cadência, deriva do sensor, ou "
                    "pacing irregular.")

    # ── Durabilidade (EJAP 2025) ──────────────────────────────────────────────
    _dur = res.get('durabilidade')
    if _dur:
        st.markdown("---")
        st.markdown("### 🏋️ Durabilidade / resiliência fisiológica")
        st.caption(
            "Deterioração das características fisiológicas ao longo da sessão. "
            "Num esforço abaixo do MMSS, o metabolismo estabiliza — mas a FC e a "
            "respiração sobem e o DFA-α1 desce progressivamente. Essa deriva é o "
            "sinal de perda de durabilidade, e é repetível entre sessões "
            "(ICC 0.73-0.94 no estudo EJAP 2025). A sessão é dividida em quartos "
            "para comparar o início com o fim.")

        st.markdown(
            f"<div style='padding:12px 16px;border-radius:8px;"
            f"background:{_dur['cor']}1A;border-left:5px solid {_dur['cor']}'>"
            f"<b style='color:{_dur['cor']};font-size:15px'>{_dur['veredicto']}</b> "
            f"<span style='font-size:13px'>({_dur['n_sinais']}/3 marcadores com "
            f"deriva significativa)</span></div>", unsafe_allow_html=True)

        if _dur['detalhe']:
            st.caption("Do primeiro ao último quarto: " + " · ".join(_dur['detalhe']))

        _cd = st.columns(max(len(_dur['derivas']), 1))
        for _i, (_k, _v) in enumerate(_dur['derivas'].items()):
            if _k in ('power', 'smo2'):
                continue
            with _cd[_i % len(_cd)]:
                st.metric(_v['nome'], f"{_v['fim']:.1f}",
                          delta=f"{_v['delta']:+.2f} vs início")

        st.dataframe(_dur['tabela'].rename(columns={
            'bloco': 'Quarto', 'inicio_min': 'Início (min)',
            **{k: NOMES_METRICAS.get(k, k) for k in _dur['tabela'].columns}}),
            hide_index=True, use_container_width=True)
        st.caption("Os três marcadores devem ser lidos em conjunto: alguém pode ter "
                   "pouca deriva da respiração mas queda normal do α1 — olhar só para "
                   "um levaria a concluir erradamente que não houve degradação.")

    # ── Decoupling ────────────────────────────────────────────────────────────
    dec = res['decoupling']
    if dec is not None and len(dec) >= 2:
        st.markdown("---")
        st.markdown("### 💓 Decoupling FC/potência")
        st.caption("Deriva do custo cardíaco: quanto mais FC é necessária para a mesma potência "
                   "ao longo da sessão. Acima de 5% indica deriva cardiovascular relevante.")
        st.plotly_chart(_grafico_decoupling(dec), use_container_width=True,
                        config={'displayModeBar': False}, key=f'g_dec_{_fid}')
    elif res.get('protocolo', {}).get('tipo') == 'degraus':
        st.markdown("---")
        st.markdown("### 💓 Decoupling FC/potência")
        st.caption(
            "Não calculado para protocolos de degraus: a potência sobe deliberadamente "
            "a cada lap, e a relação FC-Potência tem um intercepto (uma FC de base que "
            "não escala com a potência) — por isso o rácio FC/Potência desce sempre com "
            "a potência a subir, mesmo sem nenhuma deriva cardiovascular real. Este "
            "cálculo só é válido quando a potência é aproximadamente constante entre os "
            "laps comparados (ex.: intervalos repetidos à mesma intensidade).")

    # ── Fadiga ────────────────────────────────────────────────────────────────
    fad = res['fadiga']
    if fad:
        st.markdown("---")
        st.markdown("### 🔋 Indicadores de fadiga")
        cor = fad.get('veredicto_cor', '#888')
        _n_sinais_possiveis = sum(k in fad for k in
                                  ('tendencia_fc', 'consistencia', 'deriva_cardiovascular'))
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:8px;"
            f"background:{cor}1A;border-left:5px solid {cor}'>"
            f"<b style='color:{cor};font-size:16px'>Fadiga: {fad['veredicto']}</b> "
            f"<span style='font-size:13px'>({fad['n_alertas']}/{_n_sinais_possiveis} "
            f"sinais de alerta)</span></div>",
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

        st.caption(f"Sinais avaliados nesta sessão ({_n_sinais_possiveis}): "
                   "(1) o tempo de recuperação da FC aumenta ao longo da sessão, "
                   "(2) a recuperação é inconsistente entre intervalos"
                   + (", (3) há deriva do custo cardíaco." if 'deriva_cardiovascular' in fad
                      else ". A deriva do custo cardíaco não é avaliada em protocolos de "
                           "degraus — ver nota na secção de Decoupling acima.")
                   + f" {fad['n_alertas']} ou mais sinais (de {_n_sinais_possiveis}) "
                     "indicam fadiga elevada.")

    # ── Limitador fisiológico exploratório (SmO2/THb) ────────────────────────
    lim_smo2 = res.get('limitador_smo2')
    if lim_smo2 is not None:
        st.markdown("---")
        st.markdown("### 🔍 Limitador fisiológico provável (exploratório)")
        st.caption(
            "Baseado no '5-1-5 Assessment' (Moxy) e nos artigos de Evan Peikon "
            "(Emergent Performance Lab) — usa só SmO₂/THb, sem sensor de NO/CO₂. "
            "Compara como o SmO₂ e o THb evoluem entre laps sucessivos de "
            "intensidade crescente.")
        if 'erro' in lim_smo2:
            st.info(f"Não calculado: {lim_smo2['erro']}")
        else:
            _nomes_lim = {'muscular': '💪 Muscular / Utilização',
                          'cardiaco': '❤️ Cardíaco / Delivery',
                          'pulmonar': '🫁 Pulmonar',
                          'inconclusivo': '❓ Inconclusivo'}
            _prov = lim_smo2['limitador_provavel']
            st.markdown(f"**Limitador mais provável: {_nomes_lim.get(_prov, _prov)}**")

            _lc1, _lc2, _lc3 = st.columns(3)
            _pt = lim_smo2['pontuacao']
            _lc1.metric("💪 Muscular", _pt.get('muscular', 0))
            _lc2.metric("❤️ Cardíaco", _pt.get('cardiaco', 0))
            _lc3.metric("🫁 Pulmonar", _pt.get('pulmonar', 0))

            _fig_bruto = _grafico_limitador_bruto(res['df'], colunas, lap_stats)
            if _fig_bruto is not None:
                st.plotly_chart(_fig_bruto, use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_limitador_bruto_{ficheiro.name}')
                st.caption("Traço bruto de SmO₂/THb com trabalho (vermelho) e "
                          "descanso (azul) sombreados — o mesmo estilo de "
                          "gráfico usado nos casos de estudo Moxy/NNOXX.")

            if lim_smo2['sinais']:
                st.markdown("**Sinais encontrados:**")
                for s in lim_smo2['sinais']:
                    st.caption(f"• {s}")
            else:
                st.caption("Nenhum sinal claro encontrado — dados demasiado estáveis ou "
                           "sem tendência definida entre laps.")

            st.warning(lim_smo2['aviso'])

            _ctx = lim_smo2.get('contexto', {})
            if _ctx.get('nota_fc') or _ctx.get('nota_thb_forma') or _ctx.get('nota_consistencia_min_smo2'):
                st.markdown("**Contexto adicional (informativo — não entra na pontuação):**")
                if _ctx.get('nota_fc'):
                    st.caption(f"❤️ {_ctx['nota_fc']}")
                if _ctx.get('nota_thb_forma'):
                    st.caption(f"🩸 {_ctx['nota_thb_forma']}")
                if _ctx.get('nota_consistencia_min_smo2'):
                    st.caption(f"🔍 {_ctx['nota_consistencia_min_smo2']}")

            with st.expander("Ver tendências usadas na classificação"):
                _fig_tend = _grafico_limitador_tendencias(lap_stats)
                if _fig_tend is not None:
                    st.plotly_chart(_fig_tend, use_container_width=True,
                                    config={'displayModeBar': False},
                                    key=f'g_limitador_tend_{ficheiro.name}')
                for _nome_t, (_cat, _var) in lim_smo2['tendencias'].items():
                    st.caption(f"**{_nome_t}**: {_cat} ({_var:+.2f})")

    # ── Tempo até à falha ─────────────────────────────────────────────────────
    tf = res.get('tempo_falha')
    if tf is not None and len(tf) > 0:
        with st.expander("⏱️ Estimativa de tempo até à falha (extrapolação SmO₂)"):
            st.caption("Extrapolação da taxa de queda do SmO₂ até ao mínimo observado na sessão. "
                       "É uma estimativa grosseira — usa-a como ordem de grandeza, não valor exacto.")
            st.dataframe(tf, hide_index=True, use_container_width=True)

    # ── W′ Balance (modelo dinâmico, se CP/W' foram introduzidos) ────────────
    if cp_wbal and wprime_wbal:
        _wbal_res = calcular_wbal(res['df'], colunas, cp=cp_wbal, wprime=wprime_wbal)
        if _wbal_res is not None:
            st.markdown("---")
            st.markdown("### ⚡ W′ Balance")
            st.caption(
                "Modelo dinâmico (Skiba et al. 2012) — rastreia como a reserva anaeróbia "
                "(W′) esgota quando a potência excede o CP, e recupera quando fica abaixo "
                "(mais devagar quanto mais perto do CP). Diferente do CP/W′ estáticos da "
                "aba CP Model — isto mostra a evolução MOMENTO A MOMENTO nesta sessão.")
            _wc1, _wc2 = st.columns(2)
            _wc1.metric("W′ mínimo atingido", f"{_wbal_res['wbal_min']:.0f} J",
                       f"{_wbal_res['wbal_min_pct']:.0f}% de W′")
            _wc2.metric("Vezes quase vazio (≤5%)", _wbal_res['n_vezes_zero'])
            st.plotly_chart(_grafico_wbal(_wbal_res), use_container_width=True,
                            config={'displayModeBar': False}, key=f'g_wbal_{_fid}')
            if _wbal_res['wbal_min_pct'] < 5:
                st.warning(
                    "O W′ chegou a ficar praticamente esgotado nesta sessão — indica que "
                    "pelo menos um esforço foi mantido bem acima do CP, ou que as "
                    "recuperações entre esforços não foram suficientes para recarregar "
                    "antes do seguinte.")
        else:
            st.caption("⚡ W′ Balance: sem coluna de potência neste ficheiro, ou "
                      "CP/W′ inválidos.")

    # ── Comparação com histórico (Intervals.icu, por modalidade e por ano) ───
    if ac_full is not None and len(ac_full) > 0:
        st.markdown("---")
        st.markdown(f"### 📊 Comparação com histórico — {modalidade_fit}")
        st.caption(
            "Limiares calculados pelo Intervals.icu ao longo do tempo (custom fields), "
            "para esta modalidade, separados por ano — para veres se o teu range muda "
            "de ano para ano. Ao lado, os valores equivalentes que esta sessão encontrou.")

        _cmp_hist = comparar_com_historico(ac_full, modalidade_fit, res=res)

        if 'erro' in _cmp_hist:
            st.info(_cmp_hist['erro'])
        elif not _cmp_hist['linhas']:
            st.info("Sem dados suficientes para nenhuma métrica nesta modalidade.")
        else:
            _anos_h = _cmp_hist['anos']
            _linhas_tabela = []
            for _linha in _cmp_hist['linhas']:
                _row = {'Métrica': _linha['label']}
                for _ano in _anos_h:
                    _pa = _linha['por_ano'].get(_ano)
                    _row[str(_ano)] = (
                        f"{_pa['mediana']:.0f} [{_pa['q25']:.0f}-{_pa['q75']:.0f}] (n={_pa['n']})"
                        if _pa else "—")
                _mp = _linha['muda_por_ano']
                _row['Muda por ano?'] = ("⚠️ Sim" if _mp is True
                                         else ("Não" if _mp is False else "—"))
                if _linha['esta_sessao']:
                    _v, _u, _fonte = _linha['esta_sessao']
                    _row['Esta sessão'] = f"{_v:.0f} {_u}"
                else:
                    _row['Esta sessão'] = "—"
                _linhas_tabela.append(_row)

            st.dataframe(pd.DataFrame(_linhas_tabela), hide_index=True,
                        use_container_width=True)
            st.caption(
                "Cada célula: mediana [Q25-Q75] (n=nº de sessões), limpo por IQR×1.5. "
                "'Muda por ano?': sinaliza quando a mediana varia mais de 10% (ou "
                "mais de 5bpm/10W, o que for maior) entre o ano mais alto e o mais "
                "baixo — vale a pena investigar se é evolução real ou mudança de "
                "protocolo/sensor. 'Esta sessão': equivalente mais próximo calculado "
                "pela Análise FIT (LT1/LT2 via SmO₂, breakpoint/MLSS, HRVT2 via DFA-α1) "
                "— compara-o com o range histórico da mesma linha.")

    # ── Guardar no histórico ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Guardar no histórico")
    resumo = resumir_para_historico(res, ficheiro.name)
    cg1, cg2 = st.columns([1, 2])
    if cg1.button("➕ Adicionar esta sessão ao histórico", key=f'guardar_{_fid}'):
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
