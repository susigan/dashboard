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
    preparar_fit, analisar_completo, resumir_para_historico, parse_intervalos,
    sugerir_offset, sugerir_offset_por_laps, NOMES_METRICAS,
    DFA1_HRVT2, DFA1_HRVT1, LOA_LITERATURA,
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
    chave_manual = f'_fit_laps_manual_{ficheiro.name}'
    chave_excl = f'_fit_laps_excl_{ficheiro.name}'
    chave_edit_iv = f'_fit_edit_iv_{ficheiro.name}'
    chave_offsets = f'_fit_offsets_{ficheiro.name}'
    laps_manual = st.session_state.get(chave_manual)
    laps_excl = st.session_state.get(chave_excl, [])
    iv_editados = st.session_state.get(chave_edit_iv)

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

        zerar_pot = st.checkbox(
            "Zerar potência nos períodos de recuperação",
            value=False, key=f'zerar_{ficheiro.name}',
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
    # Intervalos editados na tabela de laps têm prioridade
    if iv_editados:
        _modo_seg = 'intervalos'
        intervalos = iv_editados

    _offsets = st.session_state.get(chave_offsets, {})

    with st.spinner("A ler o ficheiro..."):
        res = preparar_fit(
            bytes_fit, laps_trabalho_manual=laps_manual, laps_excluidos=laps_excl,
            janela_final_s=janela, modo_segmentacao=_modo_seg,
            intervalos_trabalho=intervalos, frac_corte=frac_corte,
            min_dur_segmento=min_dur_seg, zerar_potencia_descanso=zerar_pot,
            offsets=_offsets)

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
        key=f'editor_{ficheiro.name}',
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
    if _ca.button("✅ Aplicar alterações", key=f'aplicar_ed_{ficheiro.name}',
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

    if _cb.button("↩️ Repor detecção automática", key=f'repor_{ficheiro.name}'):
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
        horizontal=True, key=f'estilo_{ficheiro.name}')

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
            key=f'sync_sel_{ficheiro.name}')

        _novos_off = {}
        if _sync_sel:
            _cols_sync = st.columns(min(len(_sync_sel), 3))
            for _i, _m in enumerate(_sync_sel):
                with _cols_sync[_i % len(_cols_sync)]:
                    _novos_off[_m] = st.slider(
                        NOMES_METRICAS.get(_m, _m),
                        min_value=-60, max_value=60,
                        value=int(_offsets.get(_m, 0)), step=1,
                        key=f'off_{_m}_{ficheiro.name}',
                        help="Segundos a deslocar (+ = mais tarde)")

            _cs1, _cs2 = st.columns(2)
            if _cs1.button("✅ Aplicar sincronia", key=f'aplicar_sync_{ficheiro.name}',
                           type='primary'):
                st.session_state[chave_offsets] = {k: v for k, v in _novos_off.items() if v}
                st.rerun()
            if _cs2.button("↩️ Repor", key=f'repor_sync_{ficheiro.name}'):
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
                    key=f'base_sync_{ficheiro.name}',
                    help="Os laps de trabalho são marcadores temporais nítidos "
                         "(a intensidade sobe de forma abrupta), por isso costumam "
                         "dar um alinhamento mais fiável do que comparar duas séries.")
                _dir = _sg2.radio(
                    "Direcção:",
                    options=['ambas', 'frente', 'tras'],
                    format_func=lambda x: {'ambas': '↔️ Ambas',
                                           'frente': '➡️ Só para a frente',
                                           'tras': '⬅️ Só para trás'}[x],
                    key=f'dir_sync_{ficheiro.name}',
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
                            key=f'ref_sync_{ficheiro.name}')
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
            st.caption("🔴 Bandas vermelhas = trabalho · 🔵 azuis = recuperação · "
                       "⚪ cinzentas = excluídos. Podes fazer zoom e arrastar.")
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
                st.caption("🔴 Trabalho · 🔵 recuperação · ⚪ excluídos.")

    # ══════════════════════════════════════════════════════════════════════
    # FASE 2 — Análises fisiológicas
    # Só corre depois de o utilizador confirmar que os laps e o alinhamento
    # das métricas estão correctos. Antes disso, calcular limiares seria
    # trabalhar sobre dados que ainda vão ser corrigidos.
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    _chave_run = f'_fit_run_{ficheiro.name}'
    _chave_res = f'_fit_res_{ficheiro.name}'   # guarda o RESULTADO, não só um sinalizador

    _comparar_sp = st.checkbox(
        "🔬 Também comparar com Smoothness Priors global (mais lento)",
        value=False, key=f'cmp_sp_{ficheiro.name}',
        help="Recalcula o DFA-α1 uma SEGUNDA vez, usando Smoothness Priors "
             "(λ=500, estilo Kubios) em vez do detrending local, e mostra os "
             "dois lado a lado. Praticamente duplica o tempo da Fase 2 — "
             "deixa desligado para o dia-a-dia e liga só quando quiseres "
             "verificar a robustez de um resultado específico.")

    _assinatura = (str(sorted(laps_excl)), str(sorted(laps_manual or [])),
                   str(iv_editados), str(sorted(_offsets.items())),
                   janela, zerar_pot, modo, _comparar_sp)
    _prev = st.session_state.get(f'{_chave_run}_sig')
    if _prev is not None and _prev != _assinatura:
        # Os dados mudaram desde a última análise — invalidar o resultado
        st.session_state.pop(_chave_run, None)
        st.session_state.pop(f'{_chave_run}_sig', None)
        st.session_state.pop(_chave_res, None)

    _ja_analisado = st.session_state.get(_chave_run) is not None

    _cb1, _cb2 = st.columns([1, 3])
    if _cb1.button("🔬 Analisar" if not _ja_analisado else "🔄 Reanalisar",
                   key=f'run_{ficheiro.name}', type='primary'):
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
                res, metodo_detrend='local', comparar_detrend=_comparar_sp)
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
                            config={'displayModeBar': False}, key=f'g_rest_{ficheiro.name}')

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
                        if _r070:
                            _avisos_r = []
                            if _r070['extrapolado']:
                                _avisos_r.append("extrapolado")
                            if not _r070.get('fisiologicamente_plausivel', True):
                                _avisos_r.append("valor implausível")
                            _ex_r = f" ⚠️ {', '.join(_avisos_r)}" if _avisos_r else ""
                            st.metric("Limite de zona 1 (α1 = 0.70) — recalculado",
                                      f"{_r070['intensidade']:.0f} {_ur}{_ex_r}")
                        if _r075:
                            st.caption(f"α1 = 0.75 (≈VT1): {_r075['intensidade']:.0f} {_ur}")
                        if _r050:
                            st.caption(f"α1 = 0.50: {_r050['intensidade']:.0f} {_ur}")
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
                                key=f'g_dfa1_{ficheiro.name}')
        if _bp:
            with _gc2:
                st.plotly_chart(_grafico_double_linear(_bp), use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'g_dbl_{ficheiro.name}')

        st.caption("Nota: o breakpoint corresponde à intensidade no momento da transição. "
                   "A literatura sugere subtrair 10-15 W para compensar o atraso da resposta "
                   "metabólica (MRT) em rampas rápidas; em degraus longos como estes o "
                   "efeito é menor (~2-10 W).")

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
                                key=f'g_curva_dfa1_{ficheiro.name}')

        # ── Comparação de pré-processamento: local vs Smoothness Priors ───────
        _metodo_nomes = {'local': 'Detrending local (por janela)',
                          'sp_global': 'Smoothness Priors global (λ=500, estilo Kubios)'}
        _h2_alt = res.get('hrvt2_alt')
        _h1c_alt = res.get('hrvt1c_alt')
        _sub_alt = res.get('hrvt2_submax_alt')
        _metodo_pri = res.get('metodo_detrend', 'local')
        _metodo_sec = res.get('metodo_detrend_alt')
        if _metodo_sec and (_h2 or _h2_alt):
            with st.expander(
                    f"🔬 Comparar pré-processamento — "
                    f"{_metodo_nomes.get(_metodo_pri, _metodo_pri)} (principal) vs "
                    f"{_metodo_nomes.get(_metodo_sec, _metodo_sec)}"):
                st.caption(
                    "O post 'DFA a1 and ChatGPT interview' (muscleoxygentraining.com, "
                    "ago/2025) mostrou, com dados reais, que aplicar Smoothness Priors "
                    "DENTRO de cada janela de 2 min (em vez de ao tacograma inteiro) "
                    "pode desviar os limiares em várias dezenas de bpm. Esta tabela "
                    "mostra o efeito nos TEUS dados — se os dois métodos concordarem "
                    "(diferença de poucos bpm), o resultado é robusto ao "
                    "pré-processamento; se divergirem muito, vale a pena desconfiar "
                    "e olhar para a curva α1×FC de cada um.")

                def _fc_ou_traço(d):
                    if not d or 'erro' in d or d.get('fc') is None:
                        return "—"
                    txt = f"{d['fc']:.0f} bpm"
                    if not d.get('fiavel', True):
                        txt += " ⚠️"
                    return txt

                def _delta_bpm(d1, d2):
                    if (d1 and 'erro' not in d1 and d1.get('fc') is not None
                            and d2 and 'erro' not in d2 and d2.get('fc') is not None):
                        return d1['fc'] - d2['fc']
                    return None

                st.caption(
                    "⚠️ = o próprio método marcou este valor como não fiável "
                    "(R² fraco, extrapolação longa, ou FC fisiologicamente "
                    "implausível). Não uses um valor marcado só porque o outro "
                    "método também não é fiável — nesse caso, nenhum dos dois "
                    "deve ser usado para prescrição.")
                _linhas_cmp = [
                    ("HRVT2 (α1=0.50)", _h2, _h2_alt),
                    ("HRVT1c (ponto médio individual)", res.get('hrvt1c'), _h1c_alt),
                    ("HRVT2 previsto (submáximo)", res.get('hrvt2_submax'), _sub_alt),
                ]
                for _nome, _pri, _sec in _linhas_cmp:
                    _d = _delta_bpm(_pri, _sec)
                    _c1, _c2, _c3 = st.columns(3)
                    _c1.markdown(f"**{_nome}**")
                    _c2.metric(_metodo_nomes.get(_metodo_pri, _metodo_pri), _fc_ou_traço(_pri))
                    _c3.metric(_metodo_nomes.get(_metodo_sec, _metodo_sec), _fc_ou_traço(_sec),
                               delta=f"{-_d:+.0f} bpm" if _d is not None else None,
                               delta_color="off")
                    _avisos_linha = []
                    if _pri and not _pri.get('fiavel', True):
                        _avisos_linha += [f"{_metodo_nomes.get(_metodo_pri, _metodo_pri)}: {a}"
                                          for a in _pri.get('avisos', [])]
                    if _sec and not _sec.get('fiavel', True):
                        _avisos_linha += [f"{_metodo_nomes.get(_metodo_sec, _metodo_sec)}: {a}"
                                          for a in _sec.get('avisos', [])]
                    if _avisos_linha:
                        st.caption("⚠️ " + " · ".join(_avisos_linha))

                _sdfa_alt = res.get('dfa1_serie_alt')
                if _sdfa_alt is not None and len(_sdfa_alt) >= 10:
                    st.plotly_chart(
                        _grafico_curva_dfa1(_sdfa_alt, _h2_alt),
                        use_container_width=True,
                        config={'displayModeBar': False},
                        key=f'g_curva_dfa1_alt_{ficheiro.name}')
                    st.caption(f"Curva α1×FC com "
                               f"{_metodo_nomes.get(_metodo_sec, _metodo_sec).lower()}.")

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
                _s2.metric("FC máxima medida", f"{_sub['fc_max_medida']:.0f} bpm")
                _s3.metric("Extrapolação", f"{_sub['extrapolacao_bpm']:+.0f} bpm")
                st.caption(f"Ajuste sobre {_sub['n_pontos']} janelas · "
                           f"R²={_sub['r2']:.2f} · ondulação {_sub['ondulacao_pct']:.0f}%")
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
                key=f'g_hhb_temp_{ficheiro.name}')
            st.caption("🔴 Trabalho · 🔵 recuperação · ⚪ excluídos. "
                       "O HHb sobe com a intensidade (mais extracção de O₂), "
                       "o O₂Hb desce, e o THb mantém-se relativamente estável.")

            if _bph and _bp:
                st.plotly_chart(
                    _grafico_breakpoint_hhb(_bp, _bph),
                    use_container_width=True, config={'displayModeBar': False},
                    key=f'g_bp_hhb_{ficheiro.name}')
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
