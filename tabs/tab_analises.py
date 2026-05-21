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

    # ══════════════════════════════════════════════════════════════════════════
    # §07 GLYCOGEN DEPLETION — Della Mattia, Fatigue Curves (2025)
    # Estima o custo glicogénico por zona de intensidade relativa ao FTP
    # Inputs: z1_kj (Z1KJ), z2_kj (Z2KJ), z3_kj (Z3KJ), icu_weight, FTP=223W Bike
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🍬 §07 Depleção Glicogénica — Fatigue Curves (Della Mattia 2025)")
    with st.expander("📖 Como é calculado", expanded=False):
        st.markdown("""
**Depleção Glicogénica por Zona (§07 — Della Mattia 2025)**

Estima o custo em CHO (gramas) por sessão, proporcional à intensidade relativa ao FTP:

```
G_dep = k_z1 × kJ_z1 + k_z2 × kJ_z2 + k_z3 × kJ_z3
G_dep_norm = G_dep / peso_kg   [g CHO / kg]
```

Coeficientes de custo glicogénico por zona (relativos ao FTP Bike=223W):
- **Z1** (<55% FTP) → predominância aeróbica, baixo custo glicogénico (k=0.10)
- **Z2** (55–75% FTP) → zona mista, custo moderado (k=0.25)
- **Z3** (>75% FTP) → predominância anaeróbica/glicogénica, alto custo (k=0.55)

A **dívida glicogénica rolling 7d** indica a acumulação semanal — valores acima
do p75 histórico sinalizam necessidade de repleção activa (CHO strategy).

*Calibrado nos dados deste atleta — percentis relativos ao histórico individual.*
        """)

    _FTP_BIKE = 223.0  # W — FTP Bike deste atleta
    _K_ZONES  = {'z1': 0.10, 'z2': 0.25, 'z3': 0.55}  # coeficientes por zona

    _gly_df = filtrar_principais(da_full).copy()
    _gly_df['Data'] = pd.to_datetime(_gly_df['Data']).dt.normalize()
    _gly_df['type'] = _gly_df['type'].apply(norm_tipo)
    _gly_df = _gly_df[_gly_df['type'].isin(['Bike', 'Row', 'Run', 'Ski'])]

    # Colunas kJ por zona
    _z1_col = next((c for c in ['z1_kj','Z1KJ','z1kj'] if c in _gly_df.columns), None)
    _z2_col = next((c for c in ['z2_kj','Z2KJ','z2kj'] if c in _gly_df.columns), None)
    _z3_col = next((c for c in ['z3_kj','Z3KJ','z3kj'] if c in _gly_df.columns), None)
    _wt_col = next((c for c in ['icu_weight','weight'] if c in _gly_df.columns), None)

    if _z1_col and _z2_col and _z3_col:
        _gly_df['_z1'] = pd.to_numeric(_gly_df[_z1_col], errors='coerce').fillna(0)
        _gly_df['_z2'] = pd.to_numeric(_gly_df[_z2_col], errors='coerce').fillna(0)
        _gly_df['_z3'] = pd.to_numeric(_gly_df[_z3_col], errors='coerce').fillna(0)

        # Peso: rolling 30d ou mediana global
        if _wt_col:
            _gly_df['_peso'] = (pd.to_numeric(_gly_df[_wt_col], errors='coerce')
                                  .replace(0, np.nan)
                                  .fillna(method='ffill')
                                  .fillna(method='bfill')
                                  .fillna(75.0))
        else:
            _gly_df['_peso'] = 75.0

        # Custo glicogénico por sessão (g CHO) — normalizado por peso
        _gly_df['G_dep']      = (_K_ZONES['z1'] * _gly_df['_z1'] +
                                  _K_ZONES['z2'] * _gly_df['_z2'] +
                                  _K_ZONES['z3'] * _gly_df['_z3'])
        _gly_df['G_dep_norm'] = (_gly_df['G_dep'] / _gly_df['_peso']).round(2)

        # Série diária
        _gly_daily = (_gly_df.groupby('Data')
                              .agg(G_dep=('G_dep','sum'),
                                   G_norm=('G_dep_norm','sum'),
                                   z1_kj=('_z1','sum'),
                                   z2_kj=('_z2','sum'),
                                   z3_kj=('_z3','sum'))
                              .reset_index()
                              .sort_values('Data'))

        # Rolling 7d — dívida glicogénica
        _gly_idx  = pd.date_range(_gly_daily['Data'].min(),
                                   pd.Timestamp.now().normalize(), freq='D')
        _gly_full = _gly_daily.set_index('Data').reindex(_gly_idx, fill_value=0).reset_index()
        _gly_full.columns = ['Data'] + list(_gly_full.columns[1:])
        _gly_full['G_roll7'] = _gly_full['G_norm'].rolling(7, min_periods=1).sum()

        # Percentis históricos para semáforo
        _p25 = float(_gly_full['G_roll7'].quantile(0.25)) if len(_gly_full) > 14 else 5.0
        _p75 = float(_gly_full['G_roll7'].quantile(0.75)) if len(_gly_full) > 14 else 15.0
        _g_hoje     = float(_gly_full['G_norm'].iloc[-1])
        _g_roll7    = float(_gly_full['G_roll7'].iloc[-1])
        _g_max_hist = float(_gly_full['G_roll7'].max())
        _g_pct_max  = _g_roll7 / max(_g_max_hist, 0.01) * 100

        # Cards de resumo
        _gc1, _gc2, _gc3, _gc4 = st.columns(4)
        _gc1.metric("G_dep hoje (g CHO/kg)", f"{_g_hoje:.1f}",
                    help="Custo glicogénico estimado da sessão de hoje.")
        _gc2.metric("G_dep rolling 7d (g CHO/kg)", f"{_g_roll7:.1f}",
                    help="Acumulação glicogénica dos últimos 7 dias.")
        _gc3.metric("% do máximo histórico", f"{_g_pct_max:.0f}%",
                    help="Percentagem do pico máximo de dívida glicogénica do atleta.")
        _semaforo = ("🔴 Alta" if _g_roll7 > _p75 else
                     "🟡 Média" if _g_roll7 > _p25 else "🟢 Baixa")
        _gc4.metric("Dívida glicogénica", _semaforo,
                    help=f"Calibrado nos percentis deste atleta: p25={_p25:.1f} / p75={_p75:.1f}")

        # Gráfico stacked: Z1, Z2, Z3 por sessão (últimas 60d)
        _gly_plot = _gly_full[_gly_full['G_norm'] > 0].tail(60).copy()
        _fig_gly = go.Figure()

        _ZONE_COLS = {'Z1': ('#3498db', 'z1_kj'), 'Z2': ('#f39c12', 'z2_kj'), 'Z3': ('#e74c3c', 'z3_kj')}
        for _zn, (_zc, _zcol) in _ZONE_COLS.items():
            # Convert kJ to g CHO contribution
            _k = _K_ZONES[_zn.lower()]
            _vals = (_gly_plot[_zcol] * _k / _gly_plot['Data'].map(
                _gly_df.set_index('Data')['_peso'].to_dict()).fillna(75)).fillna(0)
            _fig_gly.add_trace(go.Bar(
                x=_gly_plot['Data'].tolist(), y=_vals.round(2).tolist(),
                name=f"{_zn} (<55% FTP)" if _zn=='Z1' else
                      f"{_zn} (55–75%)" if _zn=='Z2' else f"{_zn} (>75% FTP)",
                marker_color=_zc, marker_line_width=0, opacity=0.85,
                hovertemplate=f"{_zn}: %{{y:.1f}} g CHO/kg<extra></extra>"))

        # Rolling 7d overlay
        _gly_full_plot = _gly_full.tail(60)
        _fig_gly.add_trace(go.Scatter(
            x=_gly_full_plot['Data'].tolist(),
            y=_gly_full_plot['G_roll7'].round(2).tolist(),
            name='Rolling 7d',
            line=dict(color='#2c3e50', width=2.5),
            hovertemplate='Rolling 7d: %{y:.1f} g CHO/kg<extra></extra>'))

        # Linhas de threshold p25/p75
        _fig_gly.add_hline(y=_p75, line_dash='dash', line_color='#e74c3c', line_width=1,
                            annotation_text=f'p75={_p75:.1f}', annotation_font_size=10)
        _fig_gly.add_hline(y=_p25, line_dash='dash', line_color='#27ae60', line_width=1,
                            annotation_text=f'p25={_p25:.1f}', annotation_font_size=10)

        _fig_gly.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            barmode='stack', height=360,
            margin=dict(t=40, b=70, l=65, r=20),
            font=dict(color='#222', size=11),
            hovermode='x unified',
            title=dict(text='Depleção Glicogénica por Sessão (g CHO/kg) — últimas 60 sessões',
                       font=dict(size=13, color='#222')),
            legend=dict(orientation='h', y=-0.22, font=dict(color='#333', size=11)),
            xaxis=dict(tickfont=dict(color='#333'), gridcolor='rgba(0,0,0,0.04)',
                       tickangle=-30),
            yaxis=dict(title='g CHO / kg', tickfont=dict(color='#333'),
                       gridcolor='rgba(0,0,0,0.05)'))
        st.plotly_chart(_fig_gly, use_container_width=True,
                        config={'displayModeBar': False}, key='gly_dep_chart')

        # Dívida por modalidade
        st.markdown("##### Dívida glicogénica 7d por modalidade")
        _gly_mod = (_gly_df.assign(
            week=lambda d: d['Data'].dt.to_period('W').apply(lambda p: p.start_time)
        ).groupby(['week','type'])['G_dep_norm'].sum().reset_index())
        _last_week = _gly_mod['week'].max()
        _gly_lw = _gly_mod[_gly_mod['week'] == _last_week].set_index('type')['G_dep_norm']

        _mod_cols = st.columns(4)
        for _mi, _mod in enumerate(['Bike','Row','Ski','Run']):
            _val = float(_gly_lw.get(_mod, 0))
            _all_vals = _gly_mod[_gly_mod['type']==_mod]['G_dep_norm']
            _p75m = float(_all_vals.quantile(0.75)) if len(_all_vals) > 4 else 10.0
            _p25m = float(_all_vals.quantile(0.25)) if len(_all_vals) > 4 else 3.0
            _sem  = "🔴 Alta" if _val > _p75m else ("🟡 Média" if _val > _p25m else "🟢 Baixa")
            _mod_cols[_mi].metric(f"{_mod}", f"{_val:.1f} g/kg", delta=_sem, delta_color="off")

        # Download
        _gly_dl = _gly_full[['Data','G_norm','G_roll7','z1_kj','z2_kj','z3_kj']].copy()
        _gly_dl['Data'] = _gly_dl['Data'].astype(str)
        st.download_button(
            "📥 Download Depleção Glicogénica (.csv)",
            data=_gly_dl.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
            file_name="atheltica_glycogen_depletion.csv", mime="text/csv",
            key="gly_dl_btn")

    else:
        st.info(
            "Colunas z1_kj / z2_kj / z3_kj não encontradas. "
            "Confirma que as colunas Z1KJ, Z2KJ, Z3KJ estão na sheet de actividades."
        )

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
