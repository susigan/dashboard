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

def calcular_polinomios_carga(df_act_full):
    """
    Polynomial fit CTL/ATL Overall e por modalidade.
    Usa a MESMA métrica de carga que o PMC:
      1ª escolha: icu_training_load (Intervals.icu)
      Fallback:   session_rpe = (moving_time_min) × RPE
    Assim os valores de CTL/ATL são comparáveis entre as duas tabs.
    """
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # Mesma lógica que tab_pmc
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['_load'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        _fonte = 'icu_training_load'
    elif 'moving_time' in df.columns and 'rpe' in df.columns and df['rpe'].notna().sum() > 10:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['_load'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        _fonte = 'session_rpe'
    else:
        return None

    df['_load'] = df['_load'].fillna(0)
    ld = df.groupby('Data')['_load'].sum().reset_index().sort_values('Data')
    idx = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7, adjust=False).mean()
    ld['dias_num'] = (ld['Data'] - ld['Data'].min()).dt.days.values
    ld['_fonte'] = _fonte  # passa para o gráfico poder mostrar na caption

    resultados = {'overall': {'CTL': {}, 'ATL': {}}}
    for metrica in ['CTL', 'ATL']:
        x, y = ld['dias_num'].values, ld[metrica].values
        for grau in [2, 3]:
            try:
                z = np.polyfit(x, y, grau)
                p = np.poly1d(z)
                r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                resultados['overall'][metrica][f'grau{grau}'] = {'coef': z, 'poly': p, 'r2': r2, 'x': x, 'y': y}
            except Exception:
                pass

    # Por modalidade
    for tipo in ['Bike', 'Run', 'Row', 'Ski']:
        df_tipo = df[df['type'] == tipo].copy()
        if len(df_tipo) < 5:
            continue
        ld_t = df_tipo.groupby('Data')['_load'].sum().reset_index().sort_values('Data')
        idx_t = pd.date_range(ld_t['Data'].min(), ld['Data'].max())
        ld_t = ld_t.set_index('Data').reindex(idx_t, fill_value=0).reset_index(); ld_t.columns = ['Data', 'load_val']
        ld_t['CTL'] = ld_t['load_val'].ewm(span=42, adjust=False).mean()
        ld_t['ATL'] = ld_t['load_val'].ewm(span=7, adjust=False).mean()
        ld_t['dias_num'] = (ld_t['Data'] - ld_t['Data'].min()).dt.days.values
        resultados[f'tipo_{tipo}'] = {'CTL': {}, 'ATL': {}}
        for metrica in ['CTL', 'ATL']:
            x, y = ld_t['dias_num'].values, ld_t[metrica].values
            for grau in [2, 3]:
                try:
                    z = np.polyfit(x, y, grau)
                    p = np.poly1d(z)
                    r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                    resultados[f'tipo_{tipo}'][metrica][f'grau{grau}'] = {'coef': z, 'poly': p, 'r2': r2, 'x': x, 'y': y}
                except Exception:
                    pass
    resultados['_ld'] = ld
    return resultados

def tabela_resumo_por_tipo_df(da):
    """Tabela resumo por tipo igual ao original."""
    df = filtrar_principais(da).copy()
    if len(df) == 0:
        return pd.DataFrame()
    df['type'] = df['type'].apply(norm_tipo)
    agg = {'Data': 'count'}
    if 'moving_time' in df.columns:
        df['horas'] = pd.to_numeric(df['moving_time'], errors='coerce') / 3600
        agg['horas'] = 'sum'
    if 'power_avg' in df.columns:
        df['power_avg'] = pd.to_numeric(df['power_avg'], errors='coerce')
        agg['power_avg'] = 'mean'
    if 'rpe' in df.columns:
        df['rpe'] = pd.to_numeric(df['rpe'], errors='coerce')
        agg['rpe'] = 'mean'
    resumo = df.groupby('type').agg(agg).round(1).reset_index()
    resumo.columns = ['Modalidade'] + [c.replace('Data', 'Sessões').replace('horas', 'Horas').replace('power_avg', 'Power (W)').replace('rpe', 'RPE') for c in resumo.columns[1:]]
    return resumo

def tabela_ranking_power_df(da, n=10):
    """Top N por power_avg."""
    df = filtrar_principais(da).copy()
    if len(df) == 0 or 'power_avg' not in df.columns:
        return pd.DataFrame()
    df['power_avg'] = pd.to_numeric(df['power_avg'], errors='coerce')
    df = df.dropna(subset=['power_avg'])
    if len(df) == 0:
        return pd.DataFrame()
    df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%Y-%m-%d')
    cols = [c for c in ['Data', 'type', 'name', 'power_avg', 'rpe', 'moving_time'] if c in df.columns]
    top = df.nlargest(n, 'power_avg')[cols].copy()
    if 'moving_time' in top.columns:
        top['moving_time'] = (pd.to_numeric(top['moving_time'], errors='coerce') / 3600).round(1)
        top.rename(columns={'moving_time': 'Horas'}, inplace=True)
    top.rename(columns={'power_avg': 'Power (W)', 'type': 'Tipo', 'name': 'Nome', 'rpe': 'RPE'}, inplace=True)
    return top.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 9 — ANÁLISES AVANÇADAS
# ════════════════════════════════════════════════════════════════════════════════
def tab_analises(da_full, dw, dfs_annual=None, df_annual=None):
    """
    Aba de Análises Avançadas — equivalente completo ao código Python original.
    Inclui: tabelas, training load, polynomial, BPE, falta de estímulo,
    Annual (aquecimentos), e TODAS as saídas escritas por modalidade (6 passos).
    """
    st.header("🔬 Análises Avançadas")
    if dfs_annual is None: dfs_annual = {}
    if df_annual is None:  df_annual  = pd.DataFrame()

    if len(da_full) == 0:
        st.warning("Sem dados de atividades para análise avançada.")
        return

    # ── Secção 1: Tabelas de Resumo ─────────────────────────────────────────
    st.subheader("📋 Resumo de Atividades por Modalidade")
    df_res = tabela_resumo_por_tipo_df(da_full)
    if len(df_res) > 0:
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.subheader("🏆 Top 10 Sessões por Potência Média")
    df_rank = tabela_ranking_power_df(da_full, n=10)
    if len(df_rank) > 0:
        st.dataframe(df_rank, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Secção 2: Training Load Mensal Stacked ──────────────────────────────
    st.subheader("📊 Training Load Mensal por Modalidade (TRIMP = min × RPE)")
    df_tl = filtrar_principais(da_full).copy()
    df_tl = add_tempo(df_tl)
    if 'moving_time' in df_tl.columns and 'rpe' in df_tl.columns:
        df_tl['rpe_fill']    = df_tl['rpe'].fillna(df_tl['rpe'].median())
        df_tl['session_rpe'] = (pd.to_numeric(df_tl['moving_time'], errors='coerce') / 60) * df_tl['rpe_fill']
        df_tl = df_tl[df_tl['type'].isin(['Bike', 'Run', 'Row', 'Ski', 'WeightTraining'])]
        pivot_tl = df_tl.pivot_table(index='mes', columns='type', values='session_rpe', aggfunc='sum', fill_value=0).sort_index()
        CORES_MOD = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo'], 'WeightTraining': CORES['laranja']}
        _fig_sb = go.Figure()
        if len(pivot_tl) > 0:
            for _tc in [c for c in pivot_tl.columns if c in CORES_MOD]:
                _fig_sb.add_trace(go.Bar(x=[str(x) for x in pivot_tl.index],
                    y=pivot_tl[_tc].tolist(), name=_tc,
                    marker_color=CORES_MOD.get(_tc,'gray'),
                    marker_line_width=0, opacity=0.85))
        _fig_sb.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), barmode='stack', height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(tickangle=-45, showgrid=False, gridcolor='#eee', tickfont=dict(color='#111')),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_sb, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="analises_training_load")

    st.markdown("---")

    # ── Secção 3: Polynomial CTL/ATL ────────────────────────────────────────
    st.subheader("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)")
    with st.spinner("Calculando polynomial fits..."):
        poli = calcular_polinomios_carga(da_full)

    if poli is None:
        st.warning("Sem dados suficientes para polynomial analysis.")
    else:
        _ld = poli['_ld']
        _MC_ANA = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}
        _POLY_COLORS = {'CTL': CORES['azul'], 'ATL': CORES['vermelho']}
        _POLY_DASH   = {2: 'dash', 3: 'dot'}

        def _poly_fig(res_met, ld_df, title):
            dates = pd.to_datetime(ld_df['Data']).tolist()
            fp = go.Figure()
            for met, cor in _POLY_COLORS.items():
                y_raw = ld_df[met].tolist() if met in ld_df.columns else []
                if y_raw:
                    fp.add_trace(go.Scatter(
                        x=dates, y=y_raw, mode='lines', name=met,
                        line=dict(color=cor, width=2), opacity=0.45,
                        hovertemplate=f'{met}: %{{y:.1f}}<extra></extra>'))
                for grau_k, gd in res_met.get(met, {}).items():
                    grau_n = int(grau_k[-1])
                    r2 = gd['r2']; xarr = gd['x']; poly = gd['poly']
                    fp.add_trace(go.Scatter(
                        x=dates[:len(xarr)], y=poly(xarr).tolist(),
                        mode='lines',
                        name=f'{met} G{grau_n} R²={r2:.3f}',
                        line=dict(color=cor, width=2.5,
                                  dash=_POLY_DASH.get(grau_n, 'solid')),
                        hovertemplate=f'{met} poly: %{{y:.1f}}<extra></extra>'))
            fp.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=11), height=380,
                margin=dict(t=55, b=80, l=55, r=20), hovermode='x unified',
                title=dict(text=title, font=dict(size=13, color='#111')),
                legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=10)),
                xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')),
                yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'),
                           title='CTL / ATL'))
            return fp

        # Overall
        _fonte_lbl = poli['_ld'].get('_fonte', poli['_ld']['_fonte'].iloc[0] if '_fonte' in poli['_ld'].columns else 'session_rpe')
        st.markdown(f"**Overall CTL vs ATL** — fonte: `{_fonte_lbl}`")
        st.caption("Mesma métrica de carga do PMC — valores comparáveis.")
        st.plotly_chart(
            _poly_fig(poli['overall'], _ld, 'CTL/ATL Overall — Polynomial Fit'),
            use_container_width=True, config=_MC_ANA, key="ana_poly_overall")

        # Per modality — 2-column grid
        tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
        if tipos_poli:
            st.markdown("**Por Modalidade**")
            _mods_ord = [m for m in ['Bike', 'Row', 'Ski', 'Run'] if m in tipos_poli]
            for _mi in range(0, len(_mods_ord), 2):
                _cols = st.columns(2)
                for _ci, mod in enumerate(_mods_ord[_mi:_mi+2]):
                    res_mod = poli[tipos_poli[mod]]
                    # Get x range from first available metric
                    _xarr = None
                    for _m in ['CTL', 'ATL']:
                        for _gd in res_mod.get(_m, {}).values():
                            _xarr = _gd['x']; break
                        if _xarr is not None: break
                    if _xarr is None: continue
                    # Build ld_mod with raw CTL/ATL values from stored 'y'
                    _ld_mod = _ld[['Data']].iloc[:len(_xarr)].copy().reset_index(drop=True)
                    for _m in ['CTL', 'ATL']:
                        _best = (res_mod.get(_m, {}).get('grau2') or
                                 res_mod.get(_m, {}).get('grau3'))
                        _ld_mod[_m] = pd.Series(_best['y'][:len(_xarr)] if _best else
                                                  np.zeros(len(_xarr)))
                    with _cols[_ci]:
                        st.plotly_chart(
                            _poly_fig(res_mod, _ld_mod,
                                      f'{mod} — CTL/ATL Polynomial Fit'),
                            use_container_width=True,
                            config=_MC_ANA, key=f"ana_poly_{mod}")
    st.markdown("---")

    # ── Secção 4: BPE Heatmap ───────────────────────────────────────────────
    st.subheader("🗓️ BPE — Mapa de Estados Semanal")
    if len(dw) >= 14:
        mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness']
                    if m in dw.columns and dw[m].notna().sum() >= 14]
        n_sem_max = max(4, len(dw) // 7)
        n_sem_bpe = st.slider("Semanas BPE", 4, min(52, n_sem_max), min(16, n_sem_max), key="bpe_an")
        dados_bpe = {m: calcular_bpe(dw, m, 60).tail(n_sem_bpe) for m in mets_bpe}
        dados_bpe = {k: v for k, v in dados_bpe.items() if len(v) > 0}
        if dados_bpe:
            semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
            nm = len(dados_bpe)
            mat = np.zeros((nm, len(semanas)))
            nomes_bpe = {'hrv': 'HRV', 'rhr': 'RHR (inv)', 'sleep_quality': 'Sono',
                         'fatiga': 'Energia', 'stress': 'Relaxamento', 'humor': 'Humor', 'soreness': 'Sem Dor'}
            for i, met in enumerate(dados_bpe):
                z = dados_bpe[met]['zscore'].values
                mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
            from matplotlib.colors import LinearSegmentedColormap as _LSC
            cmap_bpe = _LSC.from_list('bpe', [CORES['vermelho'], CORES['amarelo'], CORES['verde']], N=100)
            import numpy as _npIM
            _zIM = [[float(mat[r][c]) if not _npIM.isnan(mat[r][c]) else None
                     for c in range(mat.shape[1])] for r in range(mat.shape[0])]
            _yIM = list(nomes_bpe.values())
            _fIM = go.Figure(go.Heatmap(z=_zIM,
                x=[str(s) for s in semanas],
                y=_yIM, colorscale='RdYlGn', zmid=0, zmin=-2, zmax=2,
                colorbar=dict(title='Z', tickfont=dict(color='#111'))))
            _fIM.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=max(280, len(_yIM)*35),
                title=dict(text='BPE — Z-Score com SWC', font=dict(size=13,color='#111')),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9,color='#111')),
                yaxis=dict(tickfont=dict(size=9,color='#111')))
            st.plotly_chart(_fIM, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False}, key="analises_bpe")
    else:
        st.info("Mínimo 14 dias de wellness para BPE.")

    st.markdown("---")

    # ── Secção 5: Falta de Estímulo ─────────────────────────────────────────
    st.subheader("🎯 Análise de Falta de Estímulo por Modalidade")
    st.caption(
        "Need Score v4 — A=Share(25%) B=Quality(25%) C=Load(20%) D=FTLM(20%) E=A×B(10%) | "
        "Need_volume=A+C | Need_intensity=B+D | Overload por score acumulado (≥2/3)")

    # ── Controlos de prioridade e preset K ───────────────────────────────────
    st.markdown("**⚙️ Configuração de Prioridades**")
    col_k, col_sp = st.columns([1, 3])
    with col_k:
        preset_k = st.selectbox("Preset influência",
            ["Conservador (K=6)", "Balanceado (K=10)", "Agressivo (K=15)"],
            index=1, key="prio_preset")
        K = {"Conservador (K=6)": 6, "Balanceado (K=10)": 10,
             "Agressivo (K=15)": 15}[preset_k]
        st.caption(f"K={K} — quanto a preferência influencia a ordenação")

    with col_sp:
        _mods_disp = ['Bike', 'Row', 'Ski', 'Run']
        pc1, pc2, pc3, pc4 = st.columns(4)
        prio_1 = pc1.selectbox("🥇 Prioridade 1 (Foco)", _mods_disp, index=0, key="prio1")
        prio_2 = pc2.selectbox("🥈 Prioridade 2 (Foco)", _mods_disp, index=1, key="prio2")
        prio_3 = pc3.selectbox("🥉 Prioridade 3 (Manutenção)", _mods_disp, index=2, key="prio3")
        prio_4 = pc4.selectbox("4️⃣  Prioridade 4 (Manutenção)", _mods_disp, index=3, key="prio4")

    prio_rank = {prio_1: 1, prio_2: 2, prio_3: 3, prio_4: 4}
    max_rank  = 4
    grupo_foco = {prio_1, prio_2}
    grupo_man  = {prio_3, prio_4}

    st.markdown("---")
    c1, c2 = st.columns(2)
    for col_w, janela, label in [(c1, 7, "7 dias"), (c2, 14, "14 dias")]:
        res, df_debug = analisar_falta_estimulo(da_full, janela_dias=janela)
        with col_w:
            st.markdown(f"**📅 Janela {label}**")
            if not res:
                st.info("Dados insuficientes.")
                continue

            # ── Calcular Need_final com bónus de prioridade ────────────────
            rows_foco = []
            rows_man  = []
            for mod, d in res.items():
                rank  = prio_rank.get(mod, 4)
                peso  = (max_rank + 1 - rank) / max_rank
                bonus = peso * K * (1 - d['need_score'] / 100)

                need_final = d['need_score'] + bonus

                # Overload: Need_final × 0.5
                ol_flag = ""
                intens  = "NORMAL"
                if d['overload']:
                    need_final *= 0.5
                    ol_flag = " ⚠️"
                    intens  = "LOW — reduzir intensidade"

                # Cap manutenção: nunca passa de 40
                if mod in grupo_man:
                    need_final = min(need_final, 40)

                # Piso mínimo
                need_final = max(need_final, 10)

                # Prioridade final
                pf = ('ALTA' if need_final >= 70 else
                      'MÉDIA' if need_final >= 40 else 'BAIXA')

                row_d = {
                    'Modalidade':   f"{'🎯' if mod in grupo_foco else '🔧'} {mod}{ol_flag}",
                    'Need base':    f"{d['need_score']:.1f}",
                    'Bónus prio':   f"+{bonus:.1f}",
                    'Need final':   f"{need_final:.1f}",
                    'Prioridade':   pf,
                    'Vol / Int':    f"{d['need_vol']:.0f} / {d['need_int_prescr']:.0f}",
                    'Prescrição':   d.get('prescricao','—'),
                }
                if mod in grupo_foco:
                    rows_foco.append((need_final, row_d))
                else:
                    rows_man.append((need_final, row_d))

            # Ordenar cada grupo por Need_final
            rows_foco.sort(key=lambda x: x[0], reverse=True)
            rows_man.sort( key=lambda x: x[0], reverse=True)

            # Mostrar separado por grupo
            if rows_foco:
                st.markdown("🎯 **Foco**")
                st.dataframe(pd.DataFrame([r for _, r in rows_foco]),
                             use_container_width=True, hide_index=True)
            if rows_man:
                st.markdown("🔧 **Manutenção**")
                st.dataframe(pd.DataFrame([r for _, r in rows_man]),
                             use_container_width=True, hide_index=True)

            # Recomendação: top do grupo foco
            if rows_foco:
                top_mod  = rows_foco[0][1]['Modalidade'].replace('🎯 ','').replace('🔧 ','').replace(' ⚠️','')
                top_need = rows_foco[0][0]
                top_ol   = res.get(top_mod, {}).get('overload', False)
                top_d = res.get(top_mod, {})
                top_prescr = top_d.get('prescricao', '—')
                if top_ol:
                    st.warning(f"⚠️ **{top_mod}**: overload — {top_prescr}")
                else:
                    st.info(f"🎯 **{top_mod}** (score {top_need:.1f}) — {top_prescr}")

            # Debug expandível
            if df_debug is not None and len(df_debug) > 0:
                with st.expander(f"🔬 Debug componentes — {label}"):
                    st.dataframe(df_debug, use_container_width=True, hide_index=True)
                    csv_b = df_debug.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"⬇️ Download debug CSV ({label})",
                        data=csv_b,
                        file_name=f"need_score_debug_{label.replace(' ','_')}.csv",
                        mime="text/csv",
                        key=f"dl_debug_{janela}")

    st.markdown("---")
    st.markdown("---")


    st.markdown("---")

    # ── Secção 7: SAÍDAS ESCRITAS POR MODALIDADE (igual ao Python original) ─
    st.subheader("📝 Análise Avançada por Modalidade (6 Passos)")
    st.caption("Equivalente às saídas print() do código Python original — CV, Tendências, Correlações, Sazonalidade, RPE, Metas")

    df_full = filtrar_principais(da_full).copy()
    df_full['Data'] = pd.to_datetime(df_full['Data'])
    if 'rpe' in df_full.columns and 'RPE' not in df_full.columns:
        df_full['RPE'] = pd.to_numeric(df_full['rpe'], errors='coerce')
    if 'moving_time' in df_full.columns:
        df_full['duration_hours'] = pd.to_numeric(df_full['moving_time'], errors='coerce') / 3600
    if 'icu_eftp' in df_full.columns:
        df_full['icu_eftp'] = pd.to_numeric(df_full['icu_eftp'], errors='coerce')
    if 'AllWorkFTP' in df_full.columns:
        df_full['AllWorkFTP'] = pd.to_numeric(df_full['AllWorkFTP'], errors='coerce')
    df_full['ano']          = df_full['Data'].dt.year
    df_full['mes']          = df_full['Data'].dt.month
    df_full['trimestre']    = df_full['Data'].dt.quarter
    df_full['ano_trimestre'] = df_full['ano'].astype(str) + '-Q' + df_full['trimestre'].astype(str)

    def _rpe_cat(v):
        try:
            v = float(v)
            if 1 <= v <= 4.5:  return 'leve'
            if 4.6 <= v <= 7:  return 'moderado'
            if 8 <= v <= 10:   return 'pesado'
        except: pass
        return None

    if 'RPE' in df_full.columns:
        df_full['RPE_categoria'] = df_full['RPE'].apply(_rpe_cat)

    modalidades = ['Ski', 'Row', 'Bike', 'Run']
    tabs_mod = st.tabs([f"🎿 Ski", f"🚣 Row", f"🚴 Bike", f"🏃 Run"])

    for tab_m, modalidade in zip(tabs_mod, modalidades):
        with tab_m:
            df_mod = df_full[df_full['type'] == modalidade].copy()
            n_ativ = len(df_mod)

            if n_ativ == 0:
                st.warning(f"Sem dados para {modalidade}.")
                continue

            periodo = (f"{df_mod['Data'].min().strftime('%b %Y')} → "
                       f"{df_mod['Data'].max().strftime('%b %Y')}")
            trimestres = sorted(df_mod['ano_trimestre'].unique())

            # ── Cabeçalho ──────────────────────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Atividades", n_ativ)
            col_b.metric("Período", periodo)
            if 'RPE' in df_mod.columns:
                rpe_m = df_mod['RPE'].dropna()
                col_c.metric("RPE médio", f"{rpe_m.mean():.2f}" if len(rpe_m) > 0 else "—")
            if 'duration_hours' in df_mod.columns:
                h = df_mod['duration_hours'].dropna()
                st.caption(f"⏱️ Horas totais: {h.sum():.1f}h  |  Média/sessão: {h.mean():.2f}h")

            # ── PASSO 1: CV ─────────────────────────────────────────────────
            with st.expander("📊 PASSO 1 — Coeficiente de Variação (CV)", expanded=True):
                cv_rows = []
                for var, label in [('icu_eftp','eFTP (W)'),('AllWorkFTP','AllWorkFTP (kJ)'),
                                   ('RPE','RPE'),('duration_hours','Duração (h)')]:
                    if var not in df_mod.columns: continue
                    s = df_mod[var].dropna()
                    if len(s) < 2 or s.mean() == 0: continue
                    cv = (s.std() / s.mean()) * 100
                    if var == 'RPE':
                        interp = "Muito consistente 🟢" if cv<20 else "Consistente 🟡" if cv<35 else "Variável 🔴"
                    else:
                        interp = "Muito consistente 🟢" if cv<15 else "Consistente 🟡" if cv<30 else "Variável 🔴"
                    cv_rows.append({'Variável': label, 'Média': f"{s.mean():.2f}",
                                    'STD': f"{s.std():.2f}", 'CV%': f"{cv:.1f}%", 'Interpretação': interp})
                if cv_rows:
                    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Sem dados suficientes para CV.")

            # ── PASSO 2: Tendências ─────────────────────────────────────────
            with st.expander("📈 PASSO 2 — Tendências (Slope)"):
                df_sort = df_mod.sort_values('Data')
                trend_rows = []
                for var, label in [('icu_eftp','eFTP (W)'),('AllWorkFTP','AllWorkFTP (kJ)'),
                                   ('RPE','RPE'),('duration_hours','Duração (h)')]:
                    if var not in df_sort.columns: continue
                    s = df_sort[var].dropna()
                    if len(s) < 3: continue
                    x = np.arange(len(s))
                    slope = np.polyfit(x, s.values, 1)[0]
                    if var == 'duration_hours': slope *= 60  # converter para min/atividade
                    unid = 'W/ativ' if var=='icu_eftp' else 'kJ/ativ' if var=='AllWorkFTP' else 'pts/ativ' if var=='RPE' else 'min/ativ'
                    if slope > 0.05:    tendencia = "↗️ Crescente"
                    elif slope < -0.05: tendencia = "↙️ Decrescente"
                    else:               tendencia = "→ Platô"
                    trend_rows.append({'Variável': label, f'Slope ({unid})': f"{slope:+.4f}", 'Tendência': tendencia})
                if trend_rows:
                    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

                # Correlações por trimestre (eFTP vs AllWorkFTP)
                if 'icu_eftp' in df_mod.columns and 'AllWorkFTP' in df_mod.columns and len(trimestres) > 0:
                    st.markdown("**eFTP e AllWorkFTP por trimestre:**")
                    trim_rows = []
                    for trim in trimestres:
                        dt = df_mod[df_mod['ano_trimestre'] == trim]
                        if len(dt) < 2: continue
                        trim_rows.append({'Trimestre': trim,
                                          'eFTP mediana (W)': f"{dt['icu_eftp'].median():.1f}" if dt['icu_eftp'].notna().any() else '—',
                                          'AllWorkFTP mediana (kJ)': f"{dt['AllWorkFTP'].median():.1f}" if dt['AllWorkFTP'].notna().any() else '—',
                                          'N': len(dt)})
                    if trim_rows:
                        st.dataframe(pd.DataFrame(trim_rows), use_container_width=True, hide_index=True)

            # ── PASSO 3: Correlações |r| > 0.4 ─────────────────────────────
            with st.expander("🔗 PASSO 3 — Correlações Avançadas (|r| > 0.4)"):
                variaveis = [v for v in ['icu_eftp','AllWorkFTP','WorkHour','RPE','duration_hours','mes']
                             if v in df_mod.columns]
                if len(variaveis) >= 2:
                    df_c = df_mod[variaveis].dropna()
                    if len(df_c) >= 3:
                        mc = df_c.corr()
                        corr_rows = []
                        for i in range(len(mc.columns)):
                            for j in range(i+1, len(mc.columns)):
                                cv = mc.iloc[i,j]
                                if abs(cv) > 0.4:
                                    v1, v2 = mc.columns[i], mc.columns[j]
                                    forca = "MUITO FORTE 🟢" if abs(cv)>0.7 else "FORTE 🟢" if abs(cv)>0.5 else "MODERADA 🟡"
                                    direcao = "positiva ↗️" if cv > 0 else "negativa ↘️"
                                    corr_rows.append({'Var 1': v1, 'Var 2': v2,
                                                      'r': f"{cv:.3f}", 'Força': forca, 'Direção': direcao})
                        if corr_rows:
                            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
                        else:
                            st.info("Nenhuma correlação > 0.4 encontrada.")
                    else:
                        st.info("Dados insuficientes para correlação.")

            # ── PASSO 4: Sazonalidade ───────────────────────────────────────
            with st.expander("📅 PASSO 4 — Sazonalidade (por trimestre)"):
                if len(trimestres) > 1:
                    saz_rows = []
                    for trim in trimestres:
                        dt = df_mod[df_mod['ano_trimestre'] == trim]
                        row = {'Trimestre': trim, 'N': len(dt)}
                        if 'icu_eftp' in dt.columns and dt['icu_eftp'].notna().any():
                            row['eFTP médio (W)']  = f"{dt['icu_eftp'].mean():.1f} ± {dt['icu_eftp'].std():.1f}"
                        if 'RPE' in dt.columns and dt['RPE'].notna().any():
                            row['RPE médio']       = f"{dt['RPE'].mean():.2f} ± {dt['RPE'].std():.2f}"
                        if 'duration_hours' in dt.columns and dt['duration_hours'].notna().any():
                            row['Horas total']     = f"{dt['duration_hours'].sum():.1f}h"
                            row['Horas/sessão']    = f"{dt['duration_hours'].mean():.2f}h"
                        saz_rows.append(row)
                    if saz_rows:
                        st.dataframe(pd.DataFrame(saz_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Apenas 1 trimestre de dados — sem análise sazonal.")

            # ── PASSO 5: RPE por Categoria ──────────────────────────────────
            with st.expander("🎯 PASSO 5 — Distribuição de RPE por Categoria"):
                if 'RPE_categoria' in df_mod.columns:
                    dist = df_mod['RPE_categoria'].value_counts()
                    total = len(df_mod)
                    rpe_rows = []
                    for cat in ['leve', 'moderado', 'pesado']:
                        n_cat = dist.get(cat, 0)
                        rpe_rows.append({'Categoria': cat.capitalize(),
                                         'N': n_cat, '%': f"{n_cat/total*100:.1f}%"})
                    st.dataframe(pd.DataFrame(rpe_rows), use_container_width=True, hide_index=True)

                    # Por trimestre
                    if len(trimestres) > 1:
                        st.markdown("**Por trimestre:**")
                        trim_rpe = []
                        for trim in trimestres:
                            dt = df_mod[df_mod['ano_trimestre'] == trim]
                            if len(dt) == 0: continue
                            dist_t = dt['RPE_categoria'].value_counts()
                            n_t = len(dt)
                            row = {'Trimestre': trim, 'N': n_t}
                            for cat in ['leve','moderado','pesado']:
                                pct = (dist_t.get(cat,0)/n_t*100) if n_t>0 else 0
                                row[cat.capitalize()+' %'] = f"{pct:.0f}%"
                            trim_rpe.append(row)
                        if trim_rpe:
                            st.dataframe(pd.DataFrame(trim_rpe), use_container_width=True, hide_index=True)
                else:
                    st.info("Sem dados de RPE para análise de categorias.")

            # ── PASSO 6: Metas baseadas em incrementos reais ────────────────
            with st.expander("🎯 PASSO 6 — Metas baseadas em incrementos reais"):
                if 'icu_eftp' in df_mod.columns and len(trimestres) > 1:
                    incrementos = []
                    for i in range(len(trimestres)-1):
                        e1 = df_mod[df_mod['ano_trimestre']==trimestres[i]]['icu_eftp'].median()
                        e2 = df_mod[df_mod['ano_trimestre']==trimestres[i+1]]['icu_eftp'].median()
                        if not pd.isna(e1) and not pd.isna(e2):
                            incrementos.append(e2 - e1)

                    if incrementos:
                        inc_med = np.mean(incrementos)
                        inc_std = np.std(incrementos)
                        ultimo  = trimestres[-1]
                        eftp_at = df_mod[df_mod['ano_trimestre']==ultimo]['icu_eftp'].median()

                        if not pd.isna(eftp_at):
                            meta_c = eftp_at + inc_med * 0.8
                            meta_m = eftp_at + inc_med
                            meta_a = eftp_at + inc_med * 1.2

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("eFTP actual",      f"{eftp_at:.1f} W",  f"({ultimo})")
                            col2.metric("Meta conservadora",f"{meta_c:.1f} W",   f"{meta_c-eftp_at:+.1f}W")
                            col3.metric("Meta moderada",    f"{meta_m:.1f} W",   f"{meta_m-eftp_at:+.1f}W")
                            col4.metric("Meta ambiciosa",   f"{meta_a:.1f} W",   f"{meta_a-eftp_at:+.1f}W")
                            st.caption(f"Incremento médio por trimestre: {inc_med:+.1f}W (±{inc_std:.1f}W) "
                                       f"baseado em {len(incrementos)} transições")

                            if 'RPE' in df_mod.columns:
                                rpe_m = df_mod['RPE'].mean()
                                if not pd.isna(rpe_m):
                                    if rpe_m < 5:
                                        rec = "💡 Intensidade baixa → meta conservadora recomendada"
                                    elif rpe_m > 7:
                                        rec = "💡 Intensidade alta → meta moderada recomendada"
                                    else:
                                        rec = "💡 Intensidade ideal → meta ambiciosa possível"
                                    st.info(rec)
                else:
                    st.info("Necessário eFTP com pelo menos 2 trimestres para calcular metas.")

    st.markdown("---")

    # ── Secção 8: Resumo Geral ──────────────────────────────────────────────
    st.subheader("📋 Resumo Geral (CTL/ATL/TSB actual)")
    ld_s, _ = calcular_series_carga(da_full)
    if len(ld_s) > 0:
        u_s = ld_s.iloc[-1]
        df7 = filtrar_principais(da_full).copy()
        df7['Data'] = pd.to_datetime(df7['Data'])
        df7 = df7[df7['Data'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
        horas7 = pd.to_numeric(df7.get('moving_time', pd.Series()), errors='coerce').sum() / 3600
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CTL (Fitness)",  f"{u_s['CTL']:.0f}")
        c2.metric("ATL (Fadiga)",   f"{u_s['ATL']:.0f}")
        c3.metric("TSB (Forma)",    f"{u_s['CTL']-u_s['ATL']:+.0f}")
        c4.metric("Atividades 7d",  len(df7))
        c5.metric("Horas 7d",       fmt_dur(horas7))
    if len(dw) > 0:
        cw1, cw2 = st.columns(2)
        if 'hrv' in dw.columns:
            hrv7 = pd.to_numeric(dw['hrv'], errors='coerce').dropna().tail(7).mean()
            if not pd.isna(hrv7): cw1.metric("HRV médio (7d)", f"{hrv7:.0f} ms")
        if 'rhr' in dw.columns:
            rhr_u = pd.to_numeric(dw['rhr'], errors='coerce').dropna()
            if len(rhr_u) > 0: cw2.metric("RHR último", f"{rhr_u.iloc[-1]:.0f} bpm")

    # Resumo final textual
    with st.expander("✅ ANÁLISE AVANÇADA — Resumo de Interpretação"):
        st.markdown("""
**O que cada métrica significa:**
- **CV baixo** → Consistência no treino (bom para progressão)
- **Slope positivo** → Progressão ao longo do tempo
- **Correlações fortes** → Relações significativas entre variáveis
- **Sazonalidade** → Padrões por trimestre / estação
- **Metas baseadas em dados** → Progressão realista baseada no histórico real
- **BPE Z-Score > +1 SWC** → Estado acima do baseline (boa recuperação)
- **BPE Z-Score < -1 SWC** → Estado abaixo do baseline (atenção à recuperação)

**Escalas:**
- CTL/ATL usa TRIMP = (moving_time_min × RPE) — escala ~300-500 (igual ao Python original)
- BPE usa SWC (Hopkins 2009) — mais sensível que Z-Score tradicional
        """)
