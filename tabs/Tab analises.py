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
warnings.filterwarnings('ignore')

def calcular_polinomios_carga(df_act_full):
    """
    Polynomial fit CTL/ATL Overall e por modalidade — igual ao original.
    Usa session_rpe para construir série CTL/ATL.
    """
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    if 'moving_time' not in df.columns or 'rpe' not in df.columns:
        return None
    df['rpe_fill'] = df['rpe'].fillna(df['rpe'].median())
    df['load_val'] = (df['moving_time'] / 60) * df['rpe_fill']
    ld = df.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
    idx = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx, fill_value=0).reset_index(); ld.columns = ['Data', 'load_val']
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7, adjust=False).mean()
    ld['dias_num'] = (ld['Data'] - ld['Data'].min()).dt.days.values

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
        ld_t = df_tipo.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
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
        st.dataframe(df_res, width="stretch", hide_index=True)

    st.subheader("🏆 Top 10 Sessões por Potência Média")
    df_rank = tabela_ranking_power_df(da_full, n=10)
    if len(df_rank) > 0:
        st.dataframe(df_rank, width="stretch", hide_index=True)

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
        if 'pivot' in dir() and len(pivot) > 0:
            for _tc in [c for c in pivot.columns if c in CORES_MOD]:
                _fig_sb.add_trace(go.Bar(x=[str(x) for x in pivot.index],
                    y=pivot[_tc].tolist(), name=_tc,
                    marker_color=CORES_MOD.get(_tc,'gray'),
                    marker_line_width=0, opacity=0.85))
        _fig_sb.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), barmode='stack', height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#eee', tickfont=dict(color='#111'), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_sb, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    st.markdown("---")

    # ── Secção 3: Polynomial CTL/ATL ────────────────────────────────────────
    st.subheader("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)")
    with st.spinner("Calculando polynomial fits..."):
        poli = calcular_polinomios_carga(da_full)

    if poli is None:
        st.warning("Sem dados suficientes para polynomial analysis.")
    else:
        # Overall
        st.markdown("**Overall CTL vs ATL**")
        _fig_gen = go.Figure()
        _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
        st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
        # TODO: chart content (converted from matplotlib)

        # Por modalidade — separado e combinado
        tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
        if tipos_poli:
            st.markdown("**Por Modalidade**")
            n_t = len(tipos_poli); ncols = 2; nrows = (n_t + 1) // 2
            _fig_gen = go.Figure()
            _fig_gen.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=340,
                legend=dict(orientation='h', y=-0.25, font=dict(color='#111')), hovermode='closest',
                xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')), yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#111')))
            st.plotly_chart(_fig_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
            # TODO: chart content (converted from matplotlib)

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
            _yIM = list(nomes.values()) if 'nomes' in dir() else [str(i) for i in range(mat.shape[0])]
            _fIM = go.Figure(go.Heatmap(z=_zIM,
                x=[str(s) for s in semanas.index] if 'semanas' in dir() else [],
                y=_yIM, colorscale='RdYlGn', zmid=0, zmin=-2, zmax=2,
                colorbar=dict(title='Z', tickfont=dict(color='#111'))))
            _fIM.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=max(280, len(_yIM)*35),
                title=dict(text='BPE — Z-Score com SWC', font=dict(size=13,color='#111')),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9,color='#111')),
                yaxis=dict(tickfont=dict(size=9,color='#111')))
            st.plotly_chart(_fIM, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
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
                             width="stretch", hide_index=True)
            if rows_man:
                st.markdown("🔧 **Manutenção**")
                st.dataframe(pd.DataFrame([r for _, r in rows_man]),
                             width="stretch", hide_index=True)

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
                    st.dataframe(df_debug, width="stretch", hide_index=True)
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
                    st.dataframe(pd.DataFrame(cv_rows), width="stretch", hide_index=True)
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
                    st.dataframe(pd.DataFrame(trend_rows), width="stretch", hide_index=True)

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
                        st.dataframe(pd.DataFrame(trim_rows), width="stretch", hide_index=True)

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
                            st.dataframe(pd.DataFrame(corr_rows), width="stretch", hide_index=True)
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
                        st.dataframe(pd.DataFrame(saz_rows), width="stretch", hide_index=True)
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
                    st.dataframe(pd.DataFrame(rpe_rows), width="stretch", hide_index=True)

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
                            st.dataframe(pd.DataFrame(trim_rpe), width="stretch", hide_index=True)
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




# ════════════════════════════════════════════════════════════════════════════════
# TAB 10 — AQUECIMENTO (Annual)
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# tabs/tab_aquecimento.py — ATHELTICA Dashboard — v4
# Aba de Aquecimento — análise completa dos dados Annual
#
# CORRECÇÕES v4:
#   - Slider rolling average DENTRO de cada secção temporal (HR e O2 independentes)
#   - Detecção robusta de Drag Factor (múltiplos nomes possíveis de coluna)
#   - Diagnóstico de colunas para debug
#   - Evolução temporal O2 com slider próprio
#   - Drag Factor para AquecSki E AquecRow
#
# Equivalente ao código original Python:
#   1. HR/O2 vs Potência (Z-Score, SEM, MDC)
#   2. Evolução temporal HR com slider rolling próprio (4 métodos)
#   3. Evolução temporal O2 com slider rolling próprio (4 métodos)
#   4. HR/Pwr ratio
#   5. Drag Factor evolução temporal (AquecRow E AquecSki)
#   6. Correlação Drag Factor vs HR e O2 por W (tabela + scatter)
#   7. SEM/MDC por grupo Drag Factor (3 partes)
# ════════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════




PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

def _axis(title='', color='#333333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#cccccc', linewidth=1, showline=True,
             zeroline=False)
    if secondary:
        d['overlaying'] = 'y'; d['side'] = 'right'; d['showgrid'] = False
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────


PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

def _axis(title='', color='#333333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#cccccc', linewidth=1, showline=True,
             zeroline=False)
    if secondary:
        d['overlaying'] = 'y'; d['side'] = 'right'; d['showgrid'] = False
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────


PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

LEGEND_STYLE = dict(orientation='h', y=1.02, x=0,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#aaa', borderwidth=1,
                    font=dict(color='#111111', size=11))

def _xaxis(title='Data', color='#333'):
    return dict(title=dict(text=title, font=dict(color=color, size=12)),
                tickfont=dict(color=color),
                showgrid=True, gridcolor='#e8e8e8',
                linecolor='#ccc', linewidth=1, showline=True)

def _yaxis(title='', color='#333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#ccc', linewidth=1, showline=True)
    if secondary:
        d.update({'overlaying':'y','side':'right','showgrid':False})
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────


PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

LEGEND_STYLE = dict(orientation='h', y=1.02, x=0,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#aaa', borderwidth=1,
                    font=dict(color='#111111', size=11))

def _xaxis(title='Data', color='#333'):
    return dict(title=dict(text=title, font=dict(color=color, size=12)),
                tickfont=dict(color=color),
                showgrid=True, gridcolor='#e8e8e8',
                linecolor='#ccc', linewidth=1, showline=True)

def _yaxis(title='', color='#333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#ccc', linewidth=1, showline=True)
    if secondary:
        d.update({'overlaying':'y','side':'right','showgrid':False})
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extrair_pot(col):
    m = re.search(r'(\d+)[_\s]*W', str(col).upper())
    return int(m.group(1)) if m else None

def _detectar_drag_col(df):
    for c in ['Drag_Factor','Drag Factor','DragFactor','drag_factor','Drag','drag','DF']:
        if c in df.columns: return c
    for c in df.columns:
        if 'DRAG' in str(c).upper(): return c
    return None

def _calcular_icc_sem_mdc(valores, tipo='HR'):
    """
    ICC calculado dos próprios dados via diferenças consecutivas.
    SEM de curto prazo: CV_curto = min(CV_dados_consec, CV_max_literatura)
      HR:   CV_max = 4%  (literatura: 1-3%, Achten & Jeukendrup 2003)
      SmO2: CV_max = 8%  (literatura: 3-8% NIRS, Davie 2018)
    MDC₉₅ = CV_curto × média × 1.96
    """
    v = np.array(valores, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 3:
        return None
    media = np.mean(v)
    std   = np.std(v, ddof=1)
    if media == 0:
        return None

    # ICC via variância intra (diferenças consecutivas) vs variância total
    if n >= 4:
        diffs    = np.diff(v)
        var_intra = np.var(diffs, ddof=1) / 2
        var_total = np.var(v, ddof=1)
        icc = max(0.0, min(1.0, (var_total - var_intra) / var_total))
    else:
        icc = max(0.0, 1.0 - (1.0 / max(n, 2)))

    # CV de curto prazo: dos próprios dados (diferenças consecutivas)
    if n >= 4:
        cv_curto_dados = (np.std(np.diff(v), ddof=1) / np.sqrt(2)) / media
    else:
        cv_curto_dados = std / media

    # Limitar pelo máximo da literatura
    cv_max = 0.04 if tipo == 'HR' else 0.08  # 4% HR | 8% SmO2
    cv_curto = min(cv_curto_dados, cv_max)

    # SEM e MDC baseados no CV de curto prazo
    sem    = cv_curto * media / np.sqrt(2)
    sem_pc = cv_curto * 100
    mdc95  = cv_curto * media * 1.96
    mdc_pc = (mdc95 / media) * 100

    if   icc >= 0.90: icc_qual = "✅ Excelente (≥0.90)"
    elif icc >= 0.75: icc_qual = "🟢 Boa (0.75–0.90)"
    elif icc >= 0.50: icc_qual = "🟡 Moderada (0.50–0.75)"
    else:              icc_qual = "⚠️ Fraca (<0.50)"

    return dict(n=n, media=media, std=std,
                icc=icc, icc_qual=icc_qual,
                cv_curto=cv_curto, cv_dados=cv_curto_dados, cv_max=cv_max,
                sem=sem, sem_pc=sem_pc,
                mdc95=mdc95, mdc_pc=mdc_pc)

def _limpar_ruido_sem(series, icc_dict, limiar_multiplo=2.0):
    """
    Remove pontos que desviam > limiar × SEM da média local (rolling 5).
    Retorna série limpa e máscara de ruído.
    """
    if icc_dict is None or len(series) < 4:
        return series, np.zeros(len(series), dtype=bool)
    sem   = icc_dict['sem']
    media_local = series.rolling(5, min_periods=2, center=True).mean().fillna(series.mean())
    ruido = (series - media_local).abs() > limiar_multiplo * sem
    limpa = series.copy()
    limpa[ruido] = np.nan
    return limpa, ruido.values

def _calcular_tendencia_com_mdc(df_col, col_data, col_val, mdc95):
    """
    Calcula tendência para uma janela temporal.
    Valida com MDC: |Δ| > MDC → usa 4 métodos para confirmar direcção.
    Retorna dict com classificação, delta, N, passa_mdc.
    """
    from scipy.stats import theilslopes
    df = df_col[[col_data, col_val]].dropna().sort_values(col_data)
    df[col_data] = pd.to_datetime(df[col_data])
    n = len(df)
    if n < 3:
        return {'classif': '— (N<3)', 'delta': None, 'passa_mdc': None, 'n': n}

    y   = df[col_val].values.astype(float)
    x   = (df[col_data] - df[col_data].min()).dt.days.values.astype(float)
    delta = y[-1] - y[0]  # mudança observada: último - primeiro

    # Valida com MDC
    if mdc95 is not None and abs(delta) <= mdc95:
        return {'classif': f'→ Estável (Δ{delta:+.1f} ≤ MDC{mdc95:.1f})',
                'delta': delta, 'passa_mdc': False, 'n': n}

    # 4 métodos
    try:
        sl, _, _, pv, _   = linregress(x, y)
        tau, p_k           = spearmanr(x, y)
        th_sl, _, _, _     = theilslopes(y, x)
        mid                = max(1, n // 2)
        _, p_t             = scipy_stats.ttest_ind(y[:mid], y[mid:]) if mid >= 2 else (0, 1)

        conf = 0
        if pv  < 0.05 and abs(sl) > 0:                              conf += 2
        if p_k < 0.05:                                               conf += 2
        if (th_sl > 0 and sl > 0) or (th_sl < 0 and sl < 0):        conf += 1
        if p_t < 0.05:                                               conf += 1

        if conf < 2:
            classif = f'→ Estável (Δ{delta:+.1f}, conf insuf.)'
        elif sl > 0:
            classif = f'↗ Aumentando (Δ{delta:+.1f} > MDC)'
        else:
            classif = f'↘ Diminuindo (Δ{delta:+.1f} > MDC)'
    except Exception:
        classif = f'→ Estável (Δ{delta:+.1f})'

    return {'classif': classif, 'delta': delta, 'passa_mdc': True, 'n': n}

def _controles_grafico(aba, secao, n_data):
    c1, c2 = st.columns([2, 1])
    agrup_lbl = c1.selectbox("Agrupar por",
        ["Sessão (sem agrup.)", "Mês", "Trimestre", "Ano"],
        key=f"agrup_{aba}_{secao}")
    agrup_map = {"Sessão (sem agrup.)": None, "Mês":"M", "Trimestre":"Q", "Ano":"A"}
    agrup_code = agrup_map[agrup_lbl]
    roll = c2.slider("Rolling (sessões)", 1, min(12, max(1, n_data-1)), 3,
                     key=f"roll_{aba}_{secao}")
    return agrup_code, agrup_lbl, roll

def _agrupar_serie(df, col_data, col_val, agrup_code):
    d = df[[col_data, col_val]].dropna().copy()
    d[col_data] = pd.to_datetime(d[col_data])
    d['_p'] = d[col_data].dt.to_period(agrup_code)
    agg = d.groupby('_p')[col_val].agg(['mean','std','count']).reset_index()
    agg['_ts'] = agg['_p'].dt.to_timestamp()
    if agrup_code == 'M':
        agg['_lbl'] = agg['_p'].dt.strftime('%b %Y')
    elif agrup_code == 'Q':
        agg['_lbl'] = agg['_p'].apply(
            lambda p: f"Q{((p.start_time.month-1)//3)+1} {p.start_time.year}")
    else:
        agg['_lbl'] = agg['_p'].dt.strftime('%Y')
    return agg


# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════
