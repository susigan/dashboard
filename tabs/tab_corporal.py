from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re as _re
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def tab_corporal(dc, da_full, wc=None):
    """
    Aba Composição Corporal & Nutrição.
    dc      : DataFrame Consolidado_Comida (pré-processado) — nutrição + Peso
    da_full : DataFrame atividades completo (para correlações)
    wc      : DataFrame wellness (opcional) — fonte primária do BF% (coluna bf_pct = FAT do formulário)
    """
    st.header("🧬 Composição Corporal & Nutrição")

    if dc is None or len(dc) == 0:
        st.warning("Sem dados corporais. Verifica a aba 'Consolidado_Comida' na planilha.")
        return

    dc = dc.copy()
    dc['Data'] = pd.to_datetime(dc['Data'])

    # ── BF% — fonte primária: wc['bf_pct'] (FAT do formulário wellness) ──────
    # Se wc disponível e tem bf_pct, substitui dc['BF'] pelos dados do formulário.
    # Fallback: dc['BF'] original (Consolidado_Comida) — mantido se wc não tiver dados.
    # 'Fat' em dc continua a ser gramas de gordura alimentar — não é tocado.
    if wc is not None and len(wc) > 0 and 'bf_pct' in wc.columns:
        _wc_bf = wc[['Data', 'bf_pct']].copy()
        _wc_bf['Data'] = pd.to_datetime(_wc_bf['Data'])
        _wc_bf = _wc_bf.dropna(subset=['bf_pct'])
        _wc_bf = _wc_bf.rename(columns={'bf_pct': '_BF_wc'})
        if len(_wc_bf) > 0:
            # Merge por data — wellness como fonte principal
            dc = dc.merge(_wc_bf, on='Data', how='left')
            # Preenche BF: usa _BF_wc onde disponível, senão mantém dc['BF'] original
            if 'BF' not in dc.columns:
                dc['BF'] = np.nan
            dc['BF'] = dc['_BF_wc'].combine_first(dc['BF'])
            dc = dc.drop(columns=['_BF_wc'])

    # ── Limitar cada coluna até ao seu último registo válido ─────────────────
    _num_cols = ['Peso','BF','Calorias','Carb','Fat','Ptn','Net']
    for _col in _num_cols:
        if _col not in dc.columns: continue
        _last_valid = dc.loc[dc[_col].notna(), 'Data'].max()
        if pd.isna(_last_valid): continue
        dc.loc[dc['Data'] > _last_valid, _col] = np.nan
    _nc = [c for c in _num_cols if c in dc.columns]
    if _nc:
        dc = dc[dc[_nc].notna().any(axis=1)].copy()

    # ── KPIs cobertura ────────────────────────────────────────────────────────
    n_total = len(dc)
    n_peso  = dc['Peso'].notna().sum()     if 'Peso'     in dc.columns else 0
    n_cal   = dc['Calorias'].notna().sum() if 'Calorias' in dc.columns else 0
    n_bf    = dc['BF'].notna().sum()       if 'BF'       in dc.columns else 0
    d_min   = dc['Data'].min().strftime('%d/%m/%Y')
    d_max   = dc['Data'].max().strftime('%d/%m/%Y')
    n_dias  = (dc['Data'].max() - dc['Data'].min()).days + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Período",             f"{d_min} → {d_max}")
    c2.metric("⚖️ Peso (registos)",     f"{n_peso}/{n_total}")
    c3.metric("🔥 Calorias (registos)", f"{n_cal}/{n_total}")
    c4.metric("🫁 BF (registos)",       f"{n_bf}/{n_total}")
    st.caption(f"Dados esparsos: {n_dias} dias no período, {n_total} entradas. "
               "Dias sem registo são ignorados em médias e correlações.")
    st.markdown("---")

    # ── Preparação de dados — fora do expander para uso global ───────────────
    _dc_all = dc.copy()
    _dc_all['_w'] = _dc_all['Data'].dt.to_period('W')
    _wk = _dc_all.groupby('_w')[['Peso','BF','Calorias','Net']].mean()
    _wk.index = _wk.index.to_timestamp()
    _wk = _wk.sort_index()

    # Rolling 7 dias (remove ruído diário de água/glicogénio)
    _peso_r7 = (_dc_all.set_index('Data')['Peso'].resample('D').mean()
                .rolling(7, min_periods=3).mean() if 'Peso' in _dc_all.columns else pd.Series(dtype=float))
    _bf_r7   = (_dc_all.set_index('Data')['BF'].resample('D').mean()
                .rolling(7, min_periods=3).mean() if 'BF' in _dc_all.columns else pd.Series(dtype=float))
    _cal_r7  = (_dc_all.set_index('Data')['Calorias'].resample('D').mean()
                .rolling(7, min_periods=3).mean() if 'Calorias' in _dc_all.columns else pd.Series(dtype=float))

    # Peso/BF actual = mediana rolling 7d das últimas 2 semanas
    _peso_atual = float(_peso_r7.dropna().tail(14).median()) if len(_peso_r7.dropna()) >= 3 else None
    _bf_atual   = float(_bf_r7.dropna().tail(14).median())   if len(_bf_r7.dropna()) >= 3 else None

    # Calcular lag óptimo kcal → Peso/BF (testa 0, 3, 5, 7, 10, 14, 21 dias)
    def _calc_lag_optimo(var_r7_series):
        from scipy.stats import spearmanr as _sr
        if len(_cal_r7.dropna()) < 15 or len(var_r7_series.dropna()) < 15:
            return 7, None
        best_lag, best_r = 7, 0.0
        for lag in [0, 3, 5, 7, 10, 14, 21]:
            cal_lagged = _cal_r7.shift(lag)
            pair = pd.DataFrame({'cal': cal_lagged, 'var': var_r7_series}).dropna()
            if len(pair) < 15: continue
            r, p = _sr(pair['cal'].values, pair['var'].values)
            if p < 0.10 and abs(r) > abs(best_r):
                best_lag, best_r = lag, r
        return best_lag, best_r if best_r != 0.0 else None

    _lag_peso, _r_lag_peso = _calc_lag_optimo(_peso_r7)
    _lag_bf,   _r_lag_bf   = _calc_lag_optimo(_bf_r7)

    # ════════════════════════════════════════════════════════════════════════
    # 🎯 MINI-CALCULADORA — usa TODO o histórico independente de filtros
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("🎯 Calculadora de Metas — Peso, BF e Calorias", expanded=True):

        st.caption(
            "Valores actuais = **mediana rolling 7d das últimas 2 semanas** (remove flutuações de água/glicogénio). "
            f"Lag calórico detectado: Peso={_lag_peso}d | BF={_lag_bf}d. "
            "Calorias estimadas pela relação histórica real com lag aplicado.")

        _ci1, _ci2, _ci3 = st.columns(3)
        _peso_alvo = _ci1.number_input(
            f"⚖️ Peso-alvo (kg)  {'[actual: ' + str(round(_peso_atual,1)) + ' kg]' if _peso_atual else ''}",
            min_value=30.0, max_value=200.0,
            value=float(round(_peso_atual,1)) if _peso_atual else 75.0,
            step=0.1, key="calc_peso_alvo")
        _bf_alvo = _ci2.number_input(
            f"🫁 BF-alvo (%)  {'[actual: ' + str(round(_bf_atual,1)) + '%]' if _bf_atual else ''}",
            min_value=3.0, max_value=50.0,
            value=float(round(_bf_atual,1)) if _bf_atual else 15.0,
            step=0.1, key="calc_bf_alvo")
        _usar_peso = _ci3.checkbox("Calcular meta Peso", value=True,  key="calc_use_peso")
        _usar_bf   = _ci3.checkbox("Calcular meta BF",   value=False, key="calc_use_bf")

        if not _usar_peso and not _usar_bf:
            st.info("Selecciona pelo menos uma meta (Peso ou BF).")
        else:
            def _cal_historicas_lag(target_col, lag_dias):
                """Regressão Calorias ~ target usando lag correcto."""
                from scipy.stats import linregress as _lr
                _d = _dc_all.set_index('Data')
                _cal_d = _d['Calorias'].resample('D').mean().rolling(7, min_periods=3).mean()
                _var_d = _d[target_col].resample('D').mean().rolling(7, min_periods=3).mean()
                # Calorias com lag: cal(t-lag) vs var(t)
                cal_lagged = _cal_d.shift(lag_dias)
                pair = pd.DataFrame({'cal': cal_lagged, 'var': _var_d}).dropna()
                if len(pair) < 15: return None
                # Regressão: cal ~ var (estimamos calorias dado o valor alvo)
                sl, ic, rv, pv, _ = _lr(pair['var'].values, pair['cal'].values)
                return {
                    'slope': sl, 'intercept': ic,
                    'r2': rv**2, 'pv': pv,
                    'cal_std': pair['cal'].std(),
                    'cal_media': pair['cal'].mean(),
                    'n': len(pair), 'lag': lag_dias,
                    'pair': pair,
                }

            for _var, _alvo, _atual, _usar, _lag in [
                ('Peso', _peso_alvo, _peso_atual, _usar_peso, _lag_peso),
                ('BF',   _bf_alvo,   _bf_atual,   _usar_bf,   _lag_bf),
            ]:
                if not _usar or _atual is None: continue
                if _var not in _dc_all.columns: continue

                diff_val = round(_alvo - _atual, 2)
                if diff_val == 0:
                    st.success(f"✅ {_var}: já está no alvo ({_atual:.1f})!")
                    continue

                unid    = 'kg' if _var == 'Peso' else '%'
                dir_lbl = ('⬆️ ganhar ' if diff_val > 0 else '⬇️ perder ') + f"{abs(diff_val):.1f} {unid}"

                st.markdown(f"#### {_var} — actual: **{_atual:.1f}** → alvo: **{_alvo:.1f}** ({dir_lbl})")

                c_rel = _cal_historicas_lag(_var, _lag)
                if c_rel:
                    qual = ("✅ Confiável" if c_rel['r2'] > 0.10 and c_rel['pv'] < 0.10
                            else "⚠️ Tendência fraca" if c_rel['n'] >= 15
                            else "⛔ Dados insuficientes")
                    st.caption(f"{qual} — R²={c_rel['r2']:.2f} | p={c_rel['pv']:.3f} | "
                               f"N={c_rel['n']} dias | Lag={_lag}d | "
                               f"Relação: cada 1 {unid} de {_var} ↔ "
                               f"{c_rel['slope']:+.0f} kcal (histórico com lag={_lag}d)")

                    cal_atual = c_rel['intercept'] + c_rel['slope'] * _atual
                    cal_alvo  = c_rel['intercept'] + c_rel['slope'] * _alvo
                    ajuste    = cal_alvo - cal_atual
                    dir_lbl2  = "défice" if diff_val < 0 else "superávit"

                    # Tempo estimado baseado no ritmo real de mudança observado
                    _par = c_rel['pair']
                    _var_changes = _par['var'].diff().dropna()
                    _cal_changes = _par['cal'].diff().dropna()
                    _change_mask = _cal_changes.abs() > 50  # só quando houve mudança calórica real
                    _ritmo_real  = None
                    if _change_mask.sum() >= 5:
                        # Ritmo: mudança de var por 100kcal de ajuste
                        from scipy.stats import spearmanr as _sr2
                        _rc = pd.DataFrame({'dc': _cal_changes[_change_mask],
                                            'dv': _var_changes[_change_mask]}).dropna()
                        if len(_rc) >= 5:
                            # g / dia por 100kcal de deficit estimado
                            _ritmo_real = float(np.polyfit(_rc['dc'], _rc['dv'], 1)[0])

                    tempo_semanas = None
                    if _ritmo_real and abs(_ritmo_real) > 1e-6:
                        # Semanas para atingir alvo com ajuste de 'ajuste' kcal/dia
                        delta_por_dia = _ritmo_real * ajuste
                        if abs(delta_por_dia) > 0.001:
                            tempo_dias = abs(diff_val / delta_por_dia)
                            tempo_semanas = tempo_dias / 7

                    rows_calc = [
                        {'Métrica': '📊 Cal. associadas ao estado actual (lag corrigido)',
                         'Valor': f"{cal_atual:.0f} kcal"},
                        {'Métrica': f'{"➕" if diff_val > 0 else "➖"} Ajuste necessário ({dir_lbl2})',
                         'Valor': f"{ajuste:+.0f} kcal/dia"},
                        {'Métrica': '🎯 Cal. alvo — central (dos dados)',
                         'Valor': f"{cal_alvo:.0f} kcal"},
                        {'Métrica': '📉 Cal. alvo — mínimo (–1σ histórico)',
                         'Valor': f"{cal_alvo - c_rel['cal_std']:.0f} kcal"},
                        {'Métrica': '📈 Cal. alvo — máximo (+1σ histórico)',
                         'Valor': f"{cal_alvo + c_rel['cal_std']:.0f} kcal"},
                    ]
                    if tempo_semanas:
                        rows_calc.append({
                            'Métrica': f'⏱️ Tempo estimado (ritmo histórico real deste atleta)',
                            'Valor': f"~{tempo_semanas:.0f} semanas (~{tempo_semanas*7:.0f} dias)"
                        })
                    else:
                        rows_calc.append({
                            'Métrica': '⏱️ Tempo estimado (referência genérica ±0.5kg/sem)',
                            'Valor': f"~{abs(diff_val)/0.5:.0f} semanas"
                        })

                    st.dataframe(pd.DataFrame(rows_calc), width="stretch", hide_index=True)
                else:
                    st.info(f"Sem dados suficientes (mín. 15 dias com lag={_lag}d) para estimar ({_var}).")

                st.markdown("---")

            st.caption(
                f"⚠️ Lag calórico: calorias de hoje afectam o peso em {_lag_peso}d (Peso) e {_lag_bf}d (BF). "
                "Semanas com treino intenso podem ter peso transitoriamente alto por retenção de água/glicogénio — "
                "não confundir com ganho de massa gorda. "
                "Calorias estimadas por regressão linear Calorias ~ Peso/BF com lag real detectado nos dados.")

    st.markdown("---")

    # ── Controlos: agrupamento + filtro de datas ──────────────────────────────
    st.subheader("⚙️ Filtros dos gráficos")
    _fc1, _fc2, _fc3, _fc4 = st.columns([1, 1, 1, 1])

    agrup_opts = {"Semana": "W", "Mês": "M", "Trimestre": "Q"}
    agrup_lbl  = _fc1.selectbox("Agrupar por", list(agrup_opts.keys()), key="corp_agrup")
    agrup_code = agrup_opts[agrup_lbl]
    roll_w     = _fc2.slider("Rolling (períodos)", 1, 12, 4, key="corp_roll")

    _d_min_hist = dc['Data'].min().date()
    _d_max_hist = dc['Data'].max().date()
    _tab_di = _fc3.date_input("Data início", value=_d_min_hist,
                               min_value=_d_min_hist, max_value=_d_max_hist,
                               key="corp_di")
    _tab_df = _fc4.date_input("Data fim", value=_d_max_hist,
                               min_value=_d_min_hist, max_value=_d_max_hist,
                               key="corp_df")

    dc_f = dc[(dc['Data'].dt.date >= _tab_di) & (dc['Data'].dt.date <= _tab_df)].copy()
    if len(dc_f) == 0:
        st.warning("Sem dados no período seleccionado.")
        return

    st.caption(f"📊 {len(dc_f)} registos | "
               f"{_tab_di.strftime('%d/%m/%Y')} → {_tab_df.strftime('%d/%m/%Y')}")

    # Agregação por período
    dc_f['_p'] = dc_f['Data'].dt.to_period(agrup_code)
    agg = dc_f.groupby('_p')[['Peso','BF','Calorias','Net','Carb','Fat','Ptn']].mean()
    agg.index = agg.index.to_timestamp()

    def _roll(series):
        return series.dropna().rolling(roll_w, min_periods=1).mean()

    # ── helper: formato tooltip de data por agrupamento ───────────────────────
    def _fmt_dates(idx):
        if agrup_code == 'W':
            return [d.strftime('Sem %d/%m/%y') for d in idx]
        elif agrup_code == 'M':
            return [d.strftime('%b %Y') for d in idx]
        else:
            return [f"Q{((d.month-1)//3)+1} {d.year}" for d in idx]

    # ── GRÁFICO 1: Peso + BF + Calorias ──────────────────────────────────────
    st.subheader("⚖️ Peso, % Gordura Corporal (BF) e Calorias")
    fig1 = go.Figure()

    if 'Peso' in agg.columns and agg['Peso'].notna().any():
        ps = agg['Peso'].dropna()
        pr = _roll(agg['Peso'])
        fig1.add_trace(go.Scatter(
            x=ps.index, y=ps.values,
            mode='markers', name='Peso (pontos)',
            marker=dict(color=CORES['azul'], size=6, opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>Peso: <b>%{y:.1f} kg</b><extra></extra>',
            yaxis='y1'))
        fig1.add_trace(go.Scatter(
            x=pr.index, y=pr.values,
            mode='lines', name=f'Peso roll({roll_w})',
            line=dict(color=CORES['azul'], width=2.5),
            hovertemplate='%{x|%d/%m/%Y}<br>Peso roll: <b>%{y:.1f} kg</b><extra></extra>',
            yaxis='y1'))

    if 'BF' in agg.columns and agg['BF'].notna().any():
        bs = agg['BF'].dropna()
        br = _roll(agg['BF'])
        fig1.add_trace(go.Scatter(
            x=bs.index, y=bs.values,
            mode='markers', name='BF% (pontos)',
            marker=dict(color=CORES['vermelho'], size=6, opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>BF: <b>%{y:.1f}%</b><extra></extra>',
            yaxis='y2'))
        fig1.add_trace(go.Scatter(
            x=br.index, y=br.values,
            mode='lines', name=f'BF roll({roll_w})',
            line=dict(color=CORES['vermelho'], width=2.5, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>BF roll: <b>%{y:.1f}%</b><extra></extra>',
            yaxis='y2'))

    if 'Calorias' in agg.columns and agg['Calorias'].notna().any():
        cs = agg['Calorias'].dropna()
        cr = _roll(agg['Calorias'])
        fig1.add_trace(go.Bar(
            x=cs.index, y=cs.values,
            name='Calorias',
            marker=dict(color=CORES['laranja'], opacity=0.25),
            hovertemplate='%{x|%d/%m/%Y}<br>Calorias: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y3'))
        fig1.add_trace(go.Scatter(
            x=cr.index, y=cr.values,
            mode='lines', name=f'Cal roll({roll_w})',
            line=dict(color=CORES['laranja'], width=1.8, dash='dot'),
            hovertemplate='%{x|%d/%m/%Y}<br>Cal roll: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y3'))

    fig1.update_layout(
        title=f'Peso, BF e Calorias — {agrup_lbl} | rolling={roll_w}',
        height=420,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        yaxis=dict(title='Peso (kg)', title_font=dict(color=CORES['azul']),
                   tickfont=dict(color=CORES['azul'])),
        yaxis2=dict(title='BF (%)', title_font=dict(color=CORES['vermelho']),
                    tickfont=dict(color=CORES['vermelho']),
                    overlaying='y', side='right'),
        yaxis3=dict(title='Calorias (kcal)', title_font=dict(color=CORES['laranja']),
                    tickfont=dict(color=CORES['laranja']),
                    overlaying='y', side='right', anchor='free', position=1.0,
                    showgrid=False),
        xaxis=dict(showgrid=True),
        margin=dict(r=80),
    )
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    # ── GRÁFICO 2: Calorias + Net ─────────────────────────────────────────────
    st.subheader("🔥 Calorias e Balanço Energético (Net)")
    fig2 = go.Figure()

    if 'Calorias' in agg.columns and agg['Calorias'].notna().any():
        cs2 = agg['Calorias'].dropna()
        cr2 = _roll(agg['Calorias'])
        fig2.add_trace(go.Bar(
            x=cs2.index, y=cs2.values,
            name='Calorias',
            marker=dict(color=CORES['laranja'], opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>Calorias: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y1'))
        fig2.add_trace(go.Scatter(
            x=cr2.index, y=cr2.values,
            mode='lines', name=f'Cal roll({roll_w})',
            line=dict(color=CORES['laranja'], width=2.5),
            hovertemplate='%{x|%d/%m/%Y}<br>Cal roll: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y1'))

    if 'Net' in agg.columns and agg['Net'].notna().any():
        nr = _roll(agg['Net'])
        fig2.add_trace(go.Scatter(
            x=nr.index, y=nr.values,
            mode='lines', name=f'Net roll({roll_w})',
            line=dict(color=CORES['roxo'], width=2.5, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>Net: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y2'))
        fig2.add_hline(y=0, line_dash='dot', line_color=CORES['cinza'],
                       annotation_text='Net = 0', annotation_position='bottom right',
                       yref='y2')

    fig2.update_layout(
        title=f'Calorias e Net — {agrup_lbl} | rolling={roll_w}',
        height=380,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        yaxis=dict(title='Calorias (kcal)', title_font=dict(color=CORES['laranja']),
                   tickfont=dict(color=CORES['laranja'])),
        yaxis2=dict(title='Net (kcal)', title_font=dict(color=CORES['roxo']),
                    tickfont=dict(color=CORES['roxo']),
                    overlaying='y', side='right'),
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    # ── GRÁFICO 3: Macros % stacked ───────────────────────────────────────────
    st.subheader("🥗 Distribuição de Macronutrientes (%)")
    macro_g = [c for c in ['Carb','Fat','Ptn'] if c in agg.columns]
    if macro_g:
        kcal_map = {'Carb': 4, 'Fat': 9, 'Ptn': 4}
        agg_k = agg[macro_g].copy()
        for m in macro_g:
            agg_k[m] = agg_k[m] * kcal_map.get(m, 4)
        total_k = agg_k[macro_g].sum(axis=1).replace(0, np.nan)
        agg_pct = (agg_k[macro_g].div(total_k, axis=0) * 100).dropna(how='all')

        if len(agg_pct) > 0:
            macro_cores = {'Carb': CORES['azul'], 'Fat': CORES['laranja'], 'Ptn': CORES['verde']}
            fig3 = go.Figure()
            for m in macro_g:
                vals = agg_pct[m].fillna(0)
                fig3.add_trace(go.Bar(
                    x=agg_pct.index, y=vals.values,
                    name=m,
                    marker=dict(color=macro_cores.get(m, CORES['cinza'])),
                    hovertemplate=f'%{{x|%d/%m/%Y}}<br>{m}: <b>%{{y:.1f}}%</b><extra></extra>',
                ))
            fig3.update_layout(
                barmode='stack',
                title=f'Macros % (Carb/Fat/Ptn em kcal) — {agrup_lbl}',
                height=360,
                hovermode='closest',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                yaxis=dict(title='% kcal de macros', range=[0, 115]),
            )
            fig3.add_hline(y=100, line_dash='dot', line_color=CORES['cinza'])
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    st.markdown("---")

    # ── GRÁFICO 4: Variação Peso e BF com bandas ─────────────────────────────
    st.subheader("📊 Variação de Peso e BF — bandas de ganho/perda esperado")
    st.caption("Peso (verde) e BF (azul) lado a lado. "
               "Bandas: limites calculados sobre o valor do período anterior. "
               "Peso: ±0.30%–0.70% | BF: ±0.25%–0.65%")

    dc_f2        = dc_f.copy()
    dc_f2['_p2'] = dc_f2['Data'].dt.to_period(agrup_code)
    agg_v        = dc_f2.groupby('_p2')[['Peso','BF']].mean()
    agg_v.index  = agg_v.index.to_timestamp()
    agg_v        = agg_v.sort_index()

    has_peso = 'Peso' in agg_v.columns and agg_v['Peso'].notna().sum() >= 3
    has_bf   = 'BF'   in agg_v.columns and agg_v['BF'].notna().sum()   >= 3

    if has_peso or has_bf:
        fig4 = go.Figure()

        if has_peso:
            peso_s     = agg_v['Peso'].dropna()
            peso_delta = peso_s.diff().dropna()
            prev_p     = peso_s.shift(1).dropna()
            xlbls      = _fmt_dates(peso_delta.index)

            fig4.add_trace(go.Bar(
                x=peso_delta.index, y=peso_delta.values,
                name='Δ Peso',
                marker=dict(color='#27ae60', opacity=0.85),
                width=1000*3600*24 * {'W':2,'M':8,'Q':20}[agrup_code],
                offset=-1000*3600*24 * {'W':1.2,'M':5,'Q':14}[agrup_code],
                customdata=np.stack([xlbls, peso_s.reindex(peso_delta.index).values], axis=-1),
                hovertemplate='%{customdata[0]}<br>Δ Peso: <b>%{y:+.2f} kg</b><br>Peso: %{customdata[1]:.1f} kg<extra></extra>',
                yaxis='y1'))

            for pct, col, lbl in [
                (0.0070, '#27ae60', '+max (+0.70%)'),
                (0.0030, '#82e0aa', '+min (+0.30%)'),
                (-0.0030, '#f1948a', '-min (-0.30%)'),
                (-0.0070, '#e74c3c', '-max (-0.70%)'),
            ]:
                fig4.add_trace(go.Scatter(
                    x=prev_p.index, y=prev_p.values * pct,
                    mode='lines', name=f'Peso {lbl}',
                    line=dict(color=col, width=1.5,
                              dash='dash' if abs(pct)==0.0070 else 'dot'),
                    hovertemplate=f'Limite Peso {lbl}: <b>%{{y:+.3f}} kg</b><extra></extra>',
                    yaxis='y1', showlegend=False))

            fig4.add_hline(y=0, line_color='black', line_width=0.8, opacity=0.5, yref='y')

        if has_bf:
            bf_s     = agg_v['BF'].dropna()
            bf_delta = bf_s.diff().dropna()
            prev_b   = bf_s.shift(1).dropna()
            xlbls_b  = _fmt_dates(bf_delta.index)

            fig4.add_trace(go.Bar(
                x=bf_delta.index, y=bf_delta.values,
                name='Δ BF',
                marker=dict(color='#2980b9', opacity=0.65),
                width=1000*3600*24 * {'W':2,'M':8,'Q':20}[agrup_code],
                offset=1000*3600*24 * {'W':0,'M':0,'Q':0}[agrup_code],
                customdata=np.stack([xlbls_b, bf_s.reindex(bf_delta.index).values], axis=-1),
                hovertemplate='%{customdata[0]}<br>Δ BF: <b>%{y:+.2f}%</b><br>BF: %{customdata[1]:.1f}%<extra></extra>',
                yaxis='y2'))

            for pct, col, lbl in [
                (0.0065, '#f39c12', '+max (+0.65%)'),
                (0.0025, '#fad7a0', '+min (+0.25%)'),
                (-0.0025, '#aed6f1', '-min (-0.25%)'),
                (-0.0065, '#2980b9', '-max (-0.65%)'),
            ]:
                fig4.add_trace(go.Scatter(
                    x=prev_b.index, y=prev_b.values * pct,
                    mode='lines', name=f'BF {lbl}',
                    line=dict(color=col, width=1.5,
                              dash='dash' if abs(pct)==0.0065 else 'dot'),
                    hovertemplate=f'Limite BF {lbl}: <b>%{{y:+.3f}}%</b><extra></extra>',
                    yaxis='y2', showlegend=False))

        fig4.update_layout(
            barmode='group',
            title=f'Variação Peso e BF por {agrup_lbl} — bandas de ganho/perda',
            height=450,
            hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            yaxis=dict(title='Δ Peso (kg)', title_font=dict(color='#27ae60'),
                       tickfont=dict(color='#27ae60'), zeroline=True),
            yaxis2=dict(title='Δ BF (%)', title_font=dict(color='#2980b9'),
                        tickfont=dict(color='#2980b9'),
                        overlaying='y', side='right', zeroline=True),
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
    else:
        st.info("Dados insuficientes de Peso/BF para o gráfico de variação.")

    st.markdown("---")

    # ── CORRELAÇÕES ───────────────────────────────────────────────────────────
    st.subheader("🔗 Correlações entre variáveis corporais e de treino")
    st.caption(
        "Correlação de Spearman semanal sobre **todo o histórico**. "
        "Variáveis de treino (KJ, Horas) usam **lag óptimo** para evitar confounding "
        "de retenção de água pós-treino. "
        "Só mostra moderada (|r|≥0.30) ou forte (|r|≥0.50)."
    )

    def _sem_mdc(series, icc=0.90):
        s = series.dropna()
        if len(s) < 5: return None, None
        sem = s.std(ddof=1) * np.sqrt(1 - icc)
        return sem, sem * 1.96 * np.sqrt(2)

    def _forca(r):
        a = abs(r)
        if a >= 0.50: return "★★★ Forte"
        if a >= 0.30: return "★★ Moderada"
        return None

    # ── Construir combined semanal ────────────────────────────────────────
    _dc_all2 = dc.copy()
    _dc_all2['_w'] = _dc_all2['Data'].dt.to_period('W')
    corp_agg = _dc_all2.groupby('_w')[['Peso','BF','Calorias','Net','Carb','Fat','Ptn']].mean()

    # % de macros (independente do volume calórico total)
    _macro_g = ['Carb','Fat','Ptn']
    _kcal_map = {'Carb':4,'Fat':9,'Ptn':4}
    _has_macros = all(c in corp_agg.columns for c in _macro_g)
    if _has_macros:
        _kcal_tot = sum(corp_agg[m].fillna(0) * _kcal_map[m] for m in _macro_g)
        _kcal_tot = _kcal_tot.replace(0, np.nan)
        for m in _macro_g:
            corp_agg[f'pct_{m}'] = corp_agg[m] * _kcal_map[m] / _kcal_tot * 100

    train_agg = pd.DataFrame()
    if da_full is not None and len(da_full) > 0:
        df_all = da_full.copy()
        df_all['Data'] = pd.to_datetime(df_all['Data'])
        df_all['_w']   = df_all['Data'].dt.to_period('W')
        df_all['_mt']  = pd.to_numeric(df_all['moving_time'], errors='coerce') / 3600
        df_cicl = df_all[df_all['type'].apply(norm_tipo) != 'WeightTraining'].copy()
        if 'icu_joules' in df_cicl.columns:
            df_cicl['_kj'] = pd.to_numeric(df_cicl['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in df_cicl.columns:
            df_cicl['_kj'] = (pd.to_numeric(df_cicl['power_avg'], errors='coerce') *
                              pd.to_numeric(df_cicl['moving_time'], errors='coerce') / 1000)
        else:
            df_cicl['_kj'] = np.nan
        df_cicl['_km'] = pd.to_numeric(
            df_cicl.get('distance', pd.Series(dtype=float)), errors='coerce') / 1000
        df_wt = df_all[df_all['type'].apply(norm_tipo) == 'WeightTraining'].copy()
        train_agg = df_cicl.groupby('_w').agg(
            Horas_cicl=('_mt','sum'), KJ_sem=('_kj','sum'), KM_sem=('_km','sum'))
        if len(df_wt) > 0:
            train_agg = train_agg.join(
                df_wt.groupby('_w').agg(Horas_WT=('_mt','sum')), how='outer')
        else:
            train_agg['Horas_WT'] = np.nan
        train_agg['Horas_total'] = (train_agg['Horas_cicl'].fillna(0) +
                                     train_agg['Horas_WT'].fillna(0))
        train_agg[train_agg == 0] = np.nan

    combined = (corp_agg.join(train_agg, how='outer')
                if len(train_agg) > 0 else corp_agg.copy())
    combined.index = combined.index.to_timestamp()
    combined = combined.sort_index()

    # ── Criar versões lagged das variáveis de treino ──────────────────────
    # KJ/Horas de semanas anteriores → Peso/BF desta semana
    # lag_treino = lag óptimo detectado (média de lag_peso e lag_bf, arredondado a semanas)
    _lag_treino_semanas = max(1, round((_lag_peso + _lag_bf) / 2 / 7))
    for col_t in ['KJ_sem','Horas_cicl','Horas_total','Horas_WT','KM_sem']:
        if col_t in combined.columns:
            combined[f'{col_t}_lag'] = combined[col_t].shift(_lag_treino_semanas)

    # ── Definir predictors e targets ──────────────────────────────────────
    _lag_sufx = f'_lag' if _lag_treino_semanas > 0 else ''
    targets = [c for c in ['Peso','BF'] if c in combined.columns]
    predictors_all = [c for c in
        # Nutrição (sem lag — são consumidas antes do efeito)
        ['Calorias','Net','Carb','Fat','Ptn','pct_Carb','pct_Fat','pct_Ptn'] +
        # Treino com lag
        [f'KJ_sem{_lag_sufx}', f'Horas_cicl{_lag_sufx}',
         f'Horas_total{_lag_sufx}', f'Horas_WT{_lag_sufx}',
         f'KM_sem{_lag_sufx}']
        if c in combined.columns]

    from scipy.stats import spearmanr as _sc_sp
    corr_rows = []
    for tgt in targets:
        for pred in predictors_all:
            if pred == tgt: continue
            pair = combined[[tgt, pred]].dropna()
            if len(pair) < 8: continue
            r, pv = _sc_sp(pair[pred].values, pair[tgt].values)
            if pv >= 0.10: continue
            f = _forca(r)
            if f is None: continue
            # Label mais legível
            pred_lbl = (pred.replace('_lag', f' (lag {_lag_treino_semanas}sem)')
                            .replace('pct_', '% ')
                            .replace('KJ_sem', 'KJ semanal')
                            .replace('Horas_cicl', 'Horas cíclico')
                            .replace('Horas_total', 'Horas total')
                            .replace('Horas_WT', 'Horas musculação')
                            .replace('KM_sem', 'KM semanal'))
            corr_rows.append({
                'Alvo': tgt,
                'Preditor': pred_lbl,
                'r': f"{r:+.2f}", 'p': f"{pv:.3f}",
                'N': len(pair),
                'Força': f,
                'Direcção': (f"↗ mais → {tgt}↑" if r > 0 else f"↘ mais → {tgt}↓"),
            })

    if corr_rows:
        df_c = pd.DataFrame(corr_rows)
        df_c['_ar'] = df_c['r'].str.replace('+','',regex=False).astype(float).abs()
        df_c = (df_c.sort_values(['Alvo','_ar'], ascending=[True, False])
                    .drop_duplicates(subset=['Alvo','Preditor'], keep='first')
                    .drop(columns=['_ar']))
        # Mostrar por alvo
        for tgt in targets:
            sub_c = df_c[df_c['Alvo'] == tgt]
            if len(sub_c) == 0: continue
            st.markdown(f"**{tgt}** — correlações significativas:")
            st.dataframe(sub_c[['Preditor','r','p','N','Força','Direcção']],
                         hide_index=True, use_container_width=True)
    else:
        st.info("Sem correlações moderadas/fortes detectadas.")

    if _lag_treino_semanas > 0:
        st.caption(
            f"KJ/Horas com lag={_lag_treino_semanas} semana(s) — "
            "evita confounding de retenção de água/glicogénio imediata pós-treino. "
            "% Carb/Fat/Ptn = distribuição em kcal, independente do volume calórico total.")

    st.markdown("---")

    # ── MACROS: efeito independente das calorias ──────────────────────────
    st.subheader("🥗 Distribuição de Macronutrientes — efeito independente das calorias")
    st.caption(
        "A pergunta: **uma distribuição diferente de macros (mais Carb, mais Fat, mais Ptn) "
        "faz diferença na composição corporal, independentemente das calorias totais?** "
        "Método: correlação parcial controlando Calorias, e quartis de % macro por quartil de BF/Peso."
    )

    if _has_macros and all(c in combined.columns for c in ['pct_Carb','pct_Fat','pct_Ptn']):
        # ── Correlação parcial: macro% → Peso/BF controlando Calorias ────
        from scipy.stats import spearmanr as _sp_m, rankdata as _rd
        st.markdown("**Correlação parcial: % Macro → Peso/BF (controlando Calorias totais)**")
        st.caption(
            "Correlação parcial de Spearman: remove o efeito das calorias totais antes de "
            "calcular a correlação entre distribuição de macros e composição corporal. "
            "r próximo de 0 = os macros não explicam variação de BF/Peso além das calorias."
        )

        def _spearman_parcial(x, y, z):
            """Correlação parcial de Spearman entre x e y controlando z."""
            df_xyz = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
            if len(df_xyz) < 10: return None, None, len(df_xyz)
            # Rankar tudo
            rx = _rd(df_xyz['x'].values).astype(float)
            ry = _rd(df_xyz['y'].values).astype(float)
            rz = _rd(df_xyz['z'].values).astype(float)
            # Residualizar x e y em relação a z via OLS
            from numpy.linalg import lstsq as _ls
            X_z = np.column_stack([np.ones(len(rz)), rz])
            res_rx = rx - X_z @ _ls(X_z, rx, rcond=None)[0]
            res_ry = ry - X_z @ _ls(X_z, ry, rcond=None)[0]
            # Correlação de Pearson nos resíduos
            if np.std(res_rx) == 0 or np.std(res_ry) == 0: return None, None, len(df_xyz)
            r_p = float(np.corrcoef(res_rx, res_ry)[0,1])
            # p-value aproximado (Fisher z → t)
            n = len(df_xyz)
            t_stat = r_p * np.sqrt(n - 3) / np.sqrt(max(1 - r_p**2, 1e-10))
            from scipy.stats import t as _t_dist
            pv = float(2 * _t_dist.sf(abs(t_stat), df=n-3))
            return round(r_p, 3), round(pv, 4), n

        parc_rows = []
        for tgt_m in ['BF','Peso']:
            if tgt_m not in combined.columns: continue
            if 'Calorias' not in combined.columns: continue
            for macro_pct, macro_lbl in [
                ('pct_Carb', '% Carb'),
                ('pct_Fat',  '% Gordura'),
                ('pct_Ptn',  '% Proteína'),
            ]:
                if macro_pct not in combined.columns: continue
                r_p, pv_p, n_p = _spearman_parcial(
                    combined[macro_pct], combined[tgt_m], combined['Calorias'])
                if r_p is None: continue
                sig = '✅ p<0.05' if pv_p < 0.05 else '~ p<0.10' if pv_p < 0.10 else '✗ ns'
                interp = ''
                if pv_p < 0.10:
                    if tgt_m == 'BF':
                        interp = (f"↘ mais {macro_lbl.replace('% ','')} → menos gordura"
                                  if r_p < 0 else
                                  f"↗ mais {macro_lbl.replace('% ','')} → mais gordura")
                    else:
                        interp = (f"↘ mais {macro_lbl.replace('% ','')} → menos peso"
                                  if r_p < 0 else
                                  f"↗ mais {macro_lbl.replace('% ','')} → mais peso")
                parc_rows.append({
                    'Alvo': tgt_m, 'Macro': macro_lbl, 'N': n_p,
                    'r parcial': f"{r_p:+.3f}", 'Sig': sig,
                    'Interpretação': interp if interp else '→ sem efeito independente',
                })
        if parc_rows:
            st.dataframe(pd.DataFrame(parc_rows), hide_index=True, use_container_width=True)

        # ── Quartis de BF por distribuição de macros ──────────────────────
        st.markdown("**Distribuição de macros por quartil de BF (semanas com dados completos)**")
        st.caption(
            "Semanas agrupadas por quartil de BF. "
            "Mostra se há diferença sistemática na distribuição de macros "
            "entre semanas de BF mais baixo vs mais alto."
        )

        _cols_q = ['BF','Calorias','pct_Carb','pct_Fat','pct_Ptn']
        _data_q  = combined[_cols_q].dropna()
        if len(_data_q) >= 16:
            try:
                _data_q = _data_q.copy()
                _data_q['_qbf'] = pd.qcut(_data_q['BF'], q=4,
                                           labels=['Q1 BF baixo','Q2','Q3','Q4 BF alto'],
                                           duplicates='drop')
                rows_macro_q = []
                for ql in ['Q1 BF baixo','Q2','Q3','Q4 BF alto']:
                    g = _data_q[_data_q['_qbf'] == ql]
                    if len(g) < 3: continue
                    rows_macro_q.append({
                        'Quartil BF': ql,
                        'N semanas': len(g),
                        'BF médio (%)': f"{g['BF'].mean():.1f}",
                        'Cal média': f"{g['Calorias'].mean():.0f} kcal",
                        '% Carb': f"{g['pct_Carb'].mean():.0f}%",
                        '% Gordura': f"{g['pct_Fat'].mean():.0f}%",
                        '% Proteína': f"{g['pct_Ptn'].mean():.0f}%",
                    })
                if rows_macro_q:
                    st.dataframe(pd.DataFrame(rows_macro_q), hide_index=True, use_container_width=True)
                    # Nota de interpretação
                    _q1 = _data_q[_data_q['_qbf']=='Q1 BF baixo']
                    _q4 = _data_q[_data_q['_qbf']=='Q4 BF alto']
                    if len(_q1) >= 3 and len(_q4) >= 3:
                        _d_ptn = _q1['pct_Ptn'].mean() - _q4['pct_Ptn'].mean()
                        _d_carb = _q1['pct_Carb'].mean() - _q4['pct_Carb'].mean()
                        _d_fat  = _q1['pct_Fat'].mean() - _q4['pct_Fat'].mean()
                        _d_cal  = _q1['Calorias'].mean() - _q4['Calorias'].mean()
                        st.info(
                            f"Q1 (BF mais baixo) vs Q4 (BF mais alto): "
                            f"Proteína {_d_ptn:+.0f}pp | Carb {_d_carb:+.0f}pp | "
                            f"Gordura {_d_fat:+.0f}pp | Calorias {_d_cal:+.0f} kcal. "
                            "Diferença nas calorias indica que parte do efeito pode ser calórico. "
                            "A correlação parcial acima isola o efeito independente."
                        )
            except Exception:
                st.info("Dados insuficientes para quartis de macros.")
        else:
            st.info(f"Dados insuficientes para análise de macros por quartil (N={len(_data_q)}, mín 16).")
    else:
        st.info("Sem dados de macronutrientes suficientes para análise de distribuição.")

    st.markdown("---")

    # ── TABELAS: base calórica por quartil de Peso e BF ───────────────────────
    st.subheader("📊 Base calórica por quartil de Peso e BF")
    st.caption(
        "Calorias com **lag aplicado** (calorias de N dias antes correspondem ao peso actual). "
        "Estratificado por carga de treino (kJ baixo vs alto) para remover confounding. "
        "Semanas com kJ alto têm peso transitoriamente mais alto por retenção de glicogénio/água."
    )

    # _cal_r7, _peso_r7, _bf_r7, _lag_peso, _lag_bf já calculados acima

    # kJ semanal para estratificação
    _kj_semanal = None
    if da_full is not None and len(da_full) > 0:
        _daf = da_full.copy()
        _daf['Data'] = pd.to_datetime(_daf['Data'])
        if 'icu_joules' in _daf.columns:
            _daf['_kj'] = pd.to_numeric(_daf['icu_joules'], errors='coerce') / 1000
        else:
            _daf['_kj'] = np.nan
        _daf['_w'] = _daf['Data'].dt.to_period('W')
        _kj_semanal = _daf.groupby('_w')['_kj'].sum()
        _kj_semanal.index = _kj_semanal.index.to_timestamp()
        # Normalizar: baixo = Q1+Q2, alto = Q3+Q4
        _kj_median = float(_kj_semanal.median()) if len(_kj_semanal) > 0 else None

    for alvo_q, alvo_lbl, unid, lag_d, var_r7 in [
        ('Peso', 'Peso', 'kg', _lag_peso, _peso_r7),
        ('BF',   'BF',   '%',  _lag_bf,   _bf_r7),
    ]:
        if alvo_q not in _dc_all.columns or 'Calorias' not in _dc_all.columns: continue
        if len(var_r7.dropna()) < 15 or len(_cal_r7.dropna()) < 15: continue

        # Calorias lagged
        cal_lagged = _cal_r7.shift(lag_d)

        # Construir dataframe diário para análise
        _df_lag = pd.DataFrame({
            'var': var_r7,
            'cal_lag': cal_lagged,
        }).dropna()
        if len(_df_lag) < 20:
            st.caption(f"Poucos dados para quartis de {alvo_lbl} com lag={lag_d}d ({len(_df_lag)} dias).")
            continue

        # Adicionar kJ semanal para estratificação
        if _kj_semanal is not None:
            _df_lag['_w'] = _df_lag.index.to_period('W').to_timestamp()
            _df_lag = _df_lag.merge(_kj_semanal.rename('kj_sem').reset_index()
                                    .rename(columns={'index':'_w'}),
                                    on='_w', how='left')
        else:
            _df_lag['kj_sem'] = np.nan

        # Quartis de var
        try:
            _df_lag['_q'], _bins = pd.qcut(
                _df_lag['var'], q=4,
                labels=['Q1 (baixo)','Q2','Q3','Q4 (alto)'],
                retbins=True, duplicates='drop')
        except Exception:
            continue

        # Tabela principal (todos os dados com lag)
        rows_q = []
        for ql in ['Q1 (baixo)','Q2','Q3','Q4 (alto)']:
            g = _df_lag[_df_lag['_q'] == ql]
            if len(g) < 3: continue
            cv = g['cal_lag']; av = g['var']
            _, mdc_c = _sem_mdc(cv) if len(cv) >= 5 else (None, None)

            row = {
                f'Quartil {alvo_lbl}':        ql,
                f'Range {alvo_lbl} ({unid})': f"{av.min():.1f}–{av.max():.1f}",
                f'Média {alvo_lbl}':          f"{av.mean():.1f} {unid}",
                'N dias':                     len(g),
                'Cal média (lag)':            f"{cv.mean():.0f} kcal",
                'Cal mediana (lag)':          f"{cv.median():.0f} kcal",
                'Cal Q1–Q3':                  f"{cv.quantile(0.25):.0f}–{cv.quantile(0.75):.0f} kcal",
                'MDC Cal':                    f"±{mdc_c:.0f} kcal" if mdc_c else '—',
            }
            rows_q.append(row)

        if rows_q:
            st.markdown(f"**Quartis de {alvo_lbl} — base calórica (lag={lag_d}d)**")
            st.dataframe(pd.DataFrame(rows_q), width="stretch", hide_index=True)

        # Tabela estratificada por kJ (se disponível)
        if _kj_semanal is not None and _kj_median is not None and 'kj_sem' in _df_lag.columns:
            _df_lag['_treino'] = np.where(_df_lag['kj_sem'] > _kj_median, 'Alto treino', 'Baixo treino')
            rows_strat = []
            for treino_g in ['Baixo treino', 'Alto treino']:
                for ql in ['Q1 (baixo)', 'Q2', 'Q3', 'Q4 (alto)']:
                    g2 = _df_lag[(_df_lag['_q'] == ql) & (_df_lag['_treino'] == treino_g)]
                    if len(g2) < 3: continue
                    cv2 = g2['cal_lag']
                    rows_strat.append({
                        'Treino': treino_g,
                        f'Quartil {alvo_lbl}': ql,
                        f'Média {alvo_lbl}': f"{g2['var'].mean():.1f} {unid}",
                        'N': len(g2),
                        'Cal média (lag)': f"{cv2.mean():.0f} kcal",
                        'Cal mediana': f"{cv2.median():.0f} kcal",
                    })
            if rows_strat:
                with st.expander(f"📊 Estratificado por carga de treino — {alvo_lbl}"):
                    st.caption(
                        "Baixo treino = kJ semanal abaixo da mediana histórica. "
                        "Alto treino = acima. Peso alto em semanas de alto treino pode ser "
                        "retenção de água/glicogénio — não indica que calorias são incorrectas.")
                    st.dataframe(pd.DataFrame(rows_strat), width="stretch", hide_index=True)

        st.markdown("")

    # ── Análise de lag calórico ────────────────────────────────────────────────
    st.subheader("⏱️ Análise de Lag Calórico")
    st.caption(
        "r Spearman entre calorias rolling 7d (com lag de 0 a 21 dias) e Peso/BF rolling 7d. "
        "O lag óptimo é onde a correlação é mais forte — mostra com quantos dias o teu peso "
        "responde a mudanças calóricas.")

    from scipy.stats import spearmanr as _sr_lag

    if len(_cal_r7.dropna()) >= 20:
        _lag_rows = []
        for lag in [0, 3, 5, 7, 10, 14, 21]:
            cal_lagged = _cal_r7.shift(lag)
            for var_col, var_r, var_lbl in [('Peso', _peso_r7, 'Peso'), ('BF', _bf_r7, 'BF')]:
                pair = pd.DataFrame({'cal': cal_lagged, 'var': var_r}).dropna()
                if len(pair) < 15: continue
                r, p = _sr_lag(pair['cal'].values, pair['var'].values)
                sig = '✅ p<0.05' if p < 0.05 else '~ p<0.10' if p < 0.10 else '✗ ns'
                _lag_rows.append({
                    'Lag (dias)': lag,
                    'Variável': var_lbl,
                    'N': len(pair),
                    'r Spearman': f"{r:+.3f}",
                    'Sig': sig,
                    'Força': '🔴 forte' if abs(r)>=0.5 else '🟡 moderada' if abs(r)>=0.3 else '🟢 fraca',
                    '★ Óptimo': '⭐' if (
                        (var_lbl == 'Peso' and lag == _lag_peso) or
                        (var_lbl == 'BF'   and lag == _lag_bf)
                    ) else '',
                })
        if _lag_rows:
            st.dataframe(pd.DataFrame(_lag_rows), hide_index=True, use_container_width=True)
            st.info(
                f"⭐ Lag óptimo detectado: **Peso = {_lag_peso} dias** | **BF = {_lag_bf} dias**. "
                "Isto significa que uma mudança calórica hoje só aparece de forma consistente "
                f"no peso após {_lag_peso} dias. Semanas com alto kJ podem ter lag diferente por "
                "retenção de água/glicogénio."
            )
