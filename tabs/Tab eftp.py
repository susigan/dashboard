# tabs/tab_eftp.py — ATHELTICA Dashboard
# eFTP: evolução + RPE + tabelas históricas (eFTP/KM/kJ/Sessões) + correlações
# Tabelas e correlações usam da_full (histórico completo) independente do sidebar

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, spearmanr
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor, norm_tipo

def tab_eftp(da, mods_sel, da_full=None):
    st.header("⚡ Evolução do eFTP por Modalidade")
    ecol = next((c for c in ['icu_eftp', 'eFTP', 'eftp', 'EFTP'] if c in da.columns), None)
    if ecol is None: st.warning("Coluna eFTP não encontrada."); return
    df = filtrar_principais(da).copy(); df['Data'] = pd.to_datetime(df['Data']); df['ano'] = df['Data'].dt.year
    df[ecol] = pd.to_numeric(df[ecol], errors='coerce'); df = df[df['type'].isin(mods_sel)].dropna(subset=[ecol]); df = df[df[ecol] > 50]
    if len(df) == 0: st.warning("Sem dados de eFTP."); return
    anos = sorted(df['ano'].unique()); CANO = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
    mapa_cor = {a: CANO[i % len(CANO)] for i, a in enumerate(anos)}
    anos_sel = st.multiselect("Filtrar anos", anos, default=list(anos)); df = df[df['ano'].isin(anos_sel)]
    mods = [m for m in mods_sel if m in df['type'].values]
    if not mods: st.info("Nenhuma modalidade com eFTP."); return
    fig, axes = plt.subplots(1, len(mods), figsize=(7 * len(mods), 6))
    if len(mods) == 1: axes = [axes]
    for ax, mod in zip(axes, mods):
        dm = df[df['type'] == mod].sort_values('Data'); cm = get_cor(mod)
        for ano in anos_sel:
            da_ = dm[dm['ano'] == ano]
            if len(da_) == 0: continue
            ax.scatter(da_['Data'], da_[ecol], color=mapa_cor[ano], alpha=0.65, s=35, label=str(ano))
            if len(da_) >= 3:
                xn = (da_['Data'] - da_['Data'].min()).dt.days.values; coef = np.polyfit(xn, da_[ecol].values, 1)
                xp = np.array([xn.min(), xn.max()]); yp = np.poly1d(coef)(xp)
                dp = [da_['Data'].min() + pd.Timedelta(days=int(x)) for x in xp]
                ax.plot(dp, yp, color=mapa_cor[ano], linewidth=2, linestyle='--', alpha=0.9)
                sm = coef[0] * 30
                ax.annotate(f'{sm:+.1f}W/mês', xy=(dp[1], yp[1]), xytext=(5, 2), textcoords='offset points', fontsize=7.5, color=mapa_cor[ano], fontweight='bold')
        if len(dm) >= 5:
            roll = dm.set_index('Data')[ecol].resample('7D').mean().interpolate()
            ax.plot(roll.index, roll.values, color=cm, linewidth=2, alpha=0.4, label='Média 7d')
        if len(dm) > 0:
            mx = dm[ecol].max()
            ax.axhline(mx, color=cm, linestyle=':', linewidth=1.2, alpha=0.6)
            ax.annotate(f'Máx: {mx:.0f}W', xy=(dm.loc[dm[ecol].idxmax(), 'Data'], mx), xytext=(0, 6), textcoords='offset points', fontsize=8, color=cm, fontweight='bold', ha='center')
        ax.set_title(f'eFTP — {mod}', fontsize=13, fontweight='bold', color=cm)
        ax.set_xlabel('Data'); ax.set_ylabel('eFTP (W)'); ax.tick_params(axis='x', rotation=45); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.suptitle('Evolução eFTP por Modalidade', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📦 RPE por Modalidade")
    if 'rpe' in da.columns:
        df_r = filtrar_principais(da).copy(); df_r = add_tempo(df_r); df_r = df_r[df_r['type'].isin(mods_sel)].dropna(subset=['rpe'])
        if len(df_r) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            tipos = [t for t in mods_sel if t in df_r['type'].values]
            sns.boxplot(data=df_r, x='type', y='rpe', order=tipos, palette={t: get_cor(t) for t in tipos}, ax=ax1)
            ax1.set_title('RPE por Modalidade', fontweight='bold'); ax1.tick_params(axis='x', rotation=45)
            if 'mes' in df_r.columns:
                meses = sorted(df_r['mes'].unique())[-12:]; df_rm = df_r[df_r['mes'].isin(meses)]
                sns.violinplot(data=df_rm, x='mes', y='rpe', palette='Set2', ax=ax2)
                ax2.set_title('RPE por Mês', fontweight='bold'); ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    # ── TABELAS eFTP + KM/KJ — filtro próprio + agrupamento ─────────────────────
    st.markdown("---")
    st.subheader("📅 Tabelas históricas por modalidade")

    # Tabelas usam histórico COMPLETO (da_full) independente do filtro do sidebar
    # Assim o filtro próprio das tabelas funciona sobre todos os dados disponíveis
    # da_full param (passed from main) or session_state fallback or da
    _da_full_tab = da_full if da_full is not None and len(da_full) > 0 else st.session_state.get('da_full', da)
    _df_full_tab = filtrar_principais(_da_full_tab).copy()
    _df_full_tab['Data'] = pd.to_datetime(_df_full_tab['Data'])

    # ── Filtro de período (só para tabelas — gráficos usam filtro sidebar) ──
    _c1, _c2, _c3 = st.columns([2, 1, 1])
    _tab_opts = {
        "Últimos 3 meses": 90,  "Últimos 6 meses": 180,
        "Último ano": 365,      "Últimos 2 anos": 730,
        "Últimos 3 anos": 1095, "Todo histórico": 9999,
        "Datas manuais": -1,
    }
    _tab_sel  = _c1.selectbox("Período das tabelas",
                               list(_tab_opts.keys()), index=3,
                               key="pmc_tab_periodo")
    _tab_dias = _tab_opts[_tab_sel]
    if _tab_dias == -1:
        _tab_di  = _c2.date_input("Início", datetime(2017, 1, 1).date(), key="pmc_tab_di")
        _tab_df_ = _c3.date_input("Fim",    datetime.now().date(),       key="pmc_tab_df")
    else:
        _tab_df_ = datetime.now().date()
        _tab_di  = (_tab_df_ - timedelta(days=_tab_dias)
                    if _tab_dias < 9999 else datetime(2017, 1, 1).date())
        _c2.caption(f"De {_tab_di.strftime('%d/%m/%Y')}")
        _c3.caption(f"Até {_tab_df_.strftime('%d/%m/%Y')}")

    df_tab = _df_full_tab[
        (_df_full_tab['Data'] >= pd.Timestamp(_tab_di)) &
        (_df_full_tab['Data'] <= pd.Timestamp(_tab_df_))
    ].copy()
    df_tab = df_tab[df_tab['type'] != 'WeightTraining']
    st.caption(f"📊 {len(df_tab)} actividades "
               f"({_tab_di.strftime('%d/%m/%Y')} → {_tab_df_.strftime('%d/%m/%Y')})")

    # ── Função auxiliar: agregar período ─────────────────────────────────────
    def _agrupar(df_in, agrup):
        """Agrupa df_in por período (Ano / Mês / Semana) e devolve coluna label."""
        df_in = df_in.copy()
        if agrup == "Ano":
            df_in['_periodo'] = df_in['Data'].dt.to_period('A')
            fmt = lambda p: str(p.year)
        elif agrup == "Semana":
            df_in['_periodo'] = df_in['Data'].dt.to_period('W')
            fmt = lambda p: f"Sem {p.start_time.strftime('%d/%m/%y')}"
        else:  # Mês (default)
            df_in['_periodo'] = df_in['Data'].dt.to_period('M')
            fmt = lambda p: pd.to_datetime(str(p)).strftime('%B %Y').title()
        return df_in, fmt

    # ── Função auxiliar: linha de tendência ──────────────────────────────────
    def _tendencia(series_num):
        """Calcula slope da regressão linear sobre a série numérica. Retorna sinal."""
        s = series_num.dropna().reset_index(drop=True)
        if len(s) < 3: return None
        x = np.arange(len(s), dtype=float)
        from scipy.stats import linregress
        sl, _, _, pv, _ = linregress(x, s.values)
        pct = (sl / s.mean() * 100) if s.mean() != 0 else 0
        sig = pv < 0.10  # p<10% como threshold de tendência
        if not sig or abs(pct) < 2: return "→ Estável"
        return f"↗ +{abs(pct):.1f}%/período" if sl > 0 else f"↘ -{abs(pct):.1f}%/período"

    # ════════════════════════════════════════════════════════════════
    # TABELA eFTP — com agrupamento ao lado do título
    # ════════════════════════════════════════════════════════════════
    _hdr1, _agr1 = st.columns([3, 1])
    _hdr1.markdown("**eFTP por modalidade**")
    _agrup_eftp = _agr1.selectbox("Agrupar por", ["Mês", "Ano", "Semana"],
                                   key="pmc_agrup_eftp")

    if 'icu_eftp' in df_tab.columns and df_tab['icu_eftp'].notna().any():
        tipos_eftp = [t for t in ['Bike', 'Run', 'Ski', 'Row']
                      if t in df_tab['type'].unique()]

        # Agregar por período escolhido
        df_tab_e, fmt_e = _agrupar(df_tab, _agrup_eftp)
        eftp_pivot = {}
        for tipo in tipos_eftp:
            df_t = (df_tab_e[df_tab_e['type'] == tipo][['_periodo', 'icu_eftp']]
                    .dropna())
            if len(df_t) == 0: continue
            eftp_pivot[tipo] = (df_t.groupby('_periodo')['icu_eftp']
                                .max().reset_index()
                                .rename(columns={'icu_eftp': tipo}))

        if eftp_pivot:
            df_e = None
            for tipo, dft in eftp_pivot.items():
                df_e = dft if df_e is None else df_e.merge(dft, on='_periodo', how='outer')
            df_e = df_e.sort_values('_periodo', ascending=False)

            num_e = [t for t in tipos_eftp if t in df_e.columns]
            rows_e = []
            for _, r in df_e.iterrows():
                row = {'Período': fmt_e(r['_periodo'])}
                for t in num_e:
                    row[f'{t} eFTP'] = f"{r[t]:.0f}w" if pd.notna(r.get(t)) else '—'
                rows_e.append(row)

            # Avg
            avg_e = {'Período': 'Avg'}
            for t in num_e:
                v = df_e[t].dropna()
                avg_e[f'{t} eFTP'] = f"{v.mean():.0f}w" if len(v) > 0 else '—'
            rows_e.append(avg_e)

            # Tendência
            tend_e = {'Período': 'Tendência'}
            for t in num_e:
                tr = _tendencia(df_e[t])
                tend_e[f'{t} eFTP'] = tr if tr else '—'
            rows_e.append(tend_e)

            st.dataframe(pd.DataFrame(rows_e),
                         use_container_width=True, hide_index=True)
    else:
        st.caption("Sem dados de eFTP disponíveis.")

    # ════════════════════════════════════════════════════════════════
    # TABELAS KM / Moving Time / kJ / Sessions — uma por modalidade
    # ════════════════════════════════════════════════════════════════
    tipos_vol = [t for t in ['Bike', 'Ski', 'Row', 'Run']
                 if t in df_tab['type'].unique()]

    for tipo in tipos_vol:
        df_t = df_tab[df_tab['type'] == tipo].copy()
        if len(df_t) == 0: continue

        _hdr2, _agr2 = st.columns([3, 1])
        _hdr2.markdown(f"**{tipo} — Distância, Tempo, kJ e Sessões**")
        _agrup_vol = _agr2.selectbox("Agrupar por", ["Mês", "Ano", "Semana"],
                                      key=f"pmc_agrup_{tipo}")

        df_t_a, fmt_v = _agrupar(df_t, _agrup_vol)

        # kJ: icu_joules (J → kJ dividindo por 1000)
        if 'icu_joules' in df_t_a.columns and df_t_a['icu_joules'].notna().any():
            df_t_a['_kj'] = pd.to_numeric(df_t_a['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in df_t_a.columns and df_t_a['power_avg'].notna().any():
            df_t_a['_kj'] = (pd.to_numeric(df_t_a['power_avg'], errors='coerce') *
                              pd.to_numeric(df_t_a['moving_time'], errors='coerce') / 1000)
        else:
            df_t_a['_kj'] = np.nan

        if 'distance' in df_t_a.columns:
            df_t_a['_km'] = pd.to_numeric(df_t_a['distance'], errors='coerce') / 1000
        else:
            df_t_a['_km'] = np.nan

        df_t_a['_mt'] = pd.to_numeric(df_t_a['moving_time'], errors='coerce').fillna(0)

        agg = df_t_a.groupby('_periodo').agg(
            _km_s=('_km',  'sum'),
            _mt_s=('_mt',  'sum'),
            _kj_s=('_kj',  'sum'),
            _ses=('Data',  'count'),
        ).reset_index().sort_values('_periodo', ascending=False)

        rows_v = []
        for _, r in agg.iterrows():
            mt_h = int(r['_mt_s'] // 3600); mt_m = int((r['_mt_s'] % 3600) // 60)
            rows_v.append({
                'Período':       fmt_v(r['_periodo']),
                'Distance':      f"{r['_km_s']:.0f} km" if pd.notna(r['_km_s']) and r['_km_s'] > 0 else '—',
                'Moving Time':   f"{mt_h}h{mt_m:02d}m",
                'kJ':            f"{r['_kj_s']:.0f}" if pd.notna(r['_kj_s']) and r['_kj_s'] > 0 else '—',
                'Sessions':      int(r['_ses']),
            })

        if not rows_v: continue

        # Avg
        avg_km  = agg['_km_s'][agg['_km_s'] > 0].mean()   if (agg['_km_s'] > 0).any() else None
        avg_mt  = agg['_mt_s'].mean()
        avg_kj  = agg['_kj_s'][agg['_kj_s'] > 0].mean()   if (agg['_kj_s'] > 0).any() else None
        avg_ses = agg['_ses'].mean()
        avg_h   = int(avg_mt // 3600); avg_m = int((avg_mt % 3600) // 60)
        rows_v.append({
            'Período':     'Avg',
            'Distance':    f"{avg_km:.0f} km" if avg_km else '—',
            'Moving Time': f"{avg_h}h{avg_m:02d}m",
            'kJ':          f"{avg_kj:.0f}" if avg_kj else '—',
            'Sessions':    f"{avg_ses:.0f}",
        })

        # Tendência — regressão linear sobre série cronológica (ascending)
        agg_asc = agg.sort_values('_periodo', ascending=True)
        tend_km  = _tendencia(agg_asc['_km_s'])
        tend_mt  = _tendencia(agg_asc['_mt_s'])
        tend_kj  = _tendencia(agg_asc['_kj_s'])
        tend_ses = _tendencia(agg_asc['_ses'].astype(float))
        rows_v.append({
            'Período':     'Tendência',
            'Distance':    tend_km  if tend_km  else '—',
            'Moving Time': tend_mt  if tend_mt  else '—',
            'kJ':          tend_kj  if tend_kj  else '—',
            'Sessions':    tend_ses if tend_ses else '—',
        })

        st.dataframe(pd.DataFrame(rows_v),
                     use_container_width=True, hide_index=True)


    st.markdown("---")
    # ── CORRELAÇÕES: variáveis de carga vs eFTP ─────────────────────────────────
    st.subheader("🔗 O que está correlacionado com o eFTP?")
    st.caption("Correlação semanal, mensal e anual entre variáveis de carga e eFTP. "
               "Apenas correlações moderadas/fortes e estatisticamente significativas são mostradas.")

    # Correlações usam histórico completo
    _df_corr = _df_full_tab.copy() if '_df_full_tab' in locals() else df.copy()
    if 'icu_eftp' in _df_corr.columns and _df_corr['icu_eftp'].notna().any():
        from scipy.stats import spearmanr

        def _cv_ok(series, max_cv=50):
            """Retorna True se CV% da série for aceitável (não muito disperso)."""
            s = series.dropna()
            if len(s) < 3 or s.mean() == 0: return False
            return (s.std() / s.mean() * 100) < max_cv

        def _mdc_ok(eftp_series, icc=0.9):
            """Verifica se variação de eFTP excede MDC — confirma mudança real."""
            s = eftp_series.dropna()
            if len(s) < 3: return False
            std = s.std(ddof=1)
            sem = std * np.sqrt(1 - icc)
            mdc = sem * 1.96 * np.sqrt(2)
            return (s.max() - s.min()) > mdc

        def _forca(r):
            ar = abs(r)
            if ar >= 0.60: return "★★★ Forte"
            if ar >= 0.40: return "★★ Moderada"
            return None  # fraca — não mostrar

        def _corr_periodo(df_mod, periodo_label, periodo_code):
            """
            Agrega por período, filtra qualidade, calcula correlação Spearman
            entre variáveis de carga e eFTP.
            Retorna lista de resultados significativos.
            """
            results = []
            d = df_mod.copy()
            d['_p'] = d['Data'].dt.to_period(periodo_code)

            # eFTP: máximo do período
            eftp_agg = d.groupby('_p')['icu_eftp'].max()
            if eftp_agg.notna().sum() < 5: return results
            if not _mdc_ok(eftp_agg.dropna()): return results

            # Variáveis a testar
            vars_test = {}

            if 'icu_joules' in d.columns and d['icu_joules'].notna().any():
                kj = d.groupby('_p')['icu_joules'].sum() / 1000
                vars_test['KJ'] = kj

            if 'moving_time' in d.columns:
                hrs = d.groupby('_p')['moving_time'].sum() / 3600
                vars_test['Horas'] = hrs

            if 'distance' in d.columns and d['distance'].notna().any():
                km = d.groupby('_p')['distance'].sum() / 1000
                vars_test['KM'] = km

            sess = d.groupby('_p')['Data'].count()
            vars_test['Sessões'] = sess

            for var_name, var_series in vars_test.items():
                # Alinhar índices
                combined = pd.DataFrame({'eftp': eftp_agg, 'var': var_series}).dropna()
                if len(combined) < 5: continue
                # Filtro CV
                if not _cv_ok(combined['var']): continue
                # Correlação Spearman
                r, pv = spearmanr(combined['var'].values, combined['eftp'].values)
                if pv >= 0.10: continue  # não significativo
                forca = _forca(r)
                if forca is None: continue  # fraca — não mostrar
                results.append({
                    'Período': periodo_label,
                    'Variável': var_name,
                    'r (Spearman)': f"{r:+.2f}",
                    'p-value': f"{pv:.3f}",
                    'Força': forca,
                    'Correlação': (f"↗ {var_name} ↑ → eFTP ↑" if r > 0
                                   else f"↘ {var_name} ↑ → eFTP ↓"),
                })
            return results

        tipos_corr = [t for t in ['Bike', 'Run', 'Ski', 'Row']
                      if t in _df_corr['type'].unique()]

        for tipo in tipos_corr:
            df_mod = _df_corr[_df_corr['type'] == tipo].copy()
            if len(df_mod) < 10: continue

            all_results = []
            for label, code in [("Semanal", "W"), ("Mensal", "M"), ("Anual", "A")]:
                all_results.extend(_corr_periodo(df_mod, label, code))

            if not all_results:
                continue  # sem correlações relevantes — não mostrar nada

            st.markdown(f"**{tipo}**")
            df_res = pd.DataFrame(all_results)
            # Remover duplicados (mesma variável em múltiplos períodos — mostrar o mais forte)
            df_res['_ar'] = df_res['r (Spearman)'].str.replace('+','',regex=False).astype(float).abs()
            df_res = (df_res.sort_values('_ar', ascending=False)
                            .drop_duplicates(subset=['Variável'], keep='first')
                            .drop(columns=['_ar'])
                            .sort_values('Força', ascending=True))
            st.dataframe(df_res, use_container_width=True, hide_index=True)

        if not any(
            len(df[df['type']==t]) >= 10 and
            'icu_eftp' in df.columns
            for t in tipos_corr
        ):
            st.info("Dados insuficientes para análise de correlação.")
    else:
        st.info("Coluna icu_eftp não disponível para análise de correlação.")
