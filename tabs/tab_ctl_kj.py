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

def tab_ctl_kj(da_full):
    st.header("⚗️ CTL vs KJ — Coeficiente de Carga")
    st.caption(
        "Modelo comportamental: TRIMP ~ KJ_work × tipo. "
        "Bike: corrige warm-up (30min). IF removido (colinear com tipo). "
        "Densidade = IF×RPE. Eficiência = TRIMP/KJ rolling.")

    if da_full is None or len(da_full) == 0:
        st.warning("Sem dados de actividades.")
        return

    from scipy import stats as _scipy_stats

    # ── Preparar dados ────────────────────────────────────────────────────────
    df = filtrar_principais(da_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    for c in ['moving_time','icu_joules','power_avg','rpe','icu_eftp']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # KJ
    if 'icu_joules' in df.columns and df['icu_joules'].notna().any():
        df['KJ'] = df['icu_joules'] / 1000
        mask = df['KJ'].isna() | (df['KJ'] == 0)
        if 'power_avg' in df.columns:
            df.loc[mask,'KJ'] = (df['power_avg'] * df['moving_time'] / 1000)[mask]
    elif 'power_avg' in df.columns:
        df['KJ'] = df['power_avg'] * df['moving_time'] / 1000
    else:
        df['KJ'] = np.nan

    df['dur_min'] = df['moving_time'] / 60
    df['rpe_n']   = pd.to_numeric(df.get('rpe', np.nan), errors='coerce')

    # IF (apenas para densidade — não entra na regressão)
    if_cols = [c for c in df.columns if c.lower() in ('if','icu_if','intensity_factor')]
    if if_cols:
        df['IF'] = pd.to_numeric(df[if_cols[0]], errors='coerce')
    elif 'power_avg' in df.columns and 'icu_eftp' in df.columns:
        df['IF'] = (pd.to_numeric(df['power_avg'], errors='coerce') /
                    pd.to_numeric(df['icu_eftp'], errors='coerce').replace(0,np.nan)
                   ).clip(0.3, 1.8)
    else:
        df['IF'] = np.nan
    has_if = df['IF'].notna().sum() > len(df) * 0.2

    # ── Bike: correcção warm-up ───────────────────────────────────────────────
    df['work_fraction'] = ((df['moving_time'] - 1800) / df['moving_time']).clip(0.1, 1.0)
    df.loc[df['type'] != 'Bike', 'work_fraction'] = 1.0
    df['KJ_work'] = df['KJ'] * df['work_fraction']

    # ── Densidade = IF × RPE (proxy: tempo em alta intensidade) ──────────────
    if has_if:
        df['densidade'] = (df['IF'] * df['rpe_n']).clip(upper=15)
    else:
        df['densidade'] = df['rpe_n']   # fallback: só RPE

    # ── Tipo de sessão (RPE) — independente do IF ─────────────────────────────
    df['tipo'] = pd.cut(df['rpe_n'], bins=[0,5,7,10],
                        labels=['base','tempo','intervalado'], right=True)

    # ── TRIMP corrigido ───────────────────────────────────────────────────────
    if has_if:
        df['TRIMP_corr'] = df['dur_min'] * df['rpe_n'] * df['IF']
    else:
        df['TRIMP_corr'] = df['dur_min'] * df['rpe_n']

    # ── Eficiência por sessão: TRIMP / KJ_work ────────────────────────────────
    kj_ref = df['KJ_work'].where(df['KJ_work'] > 0, np.nan)
    df['eff_sessao'] = df['TRIMP_corr'] / kj_ref   # TRIMP por kJ

    # ── Filtros de qualidade ──────────────────────────────────────────────────
    n_raw = len(df)
    df = df[df['dur_min'] >= 10]
    df = df[df['rpe_n'].between(1, 10)]
    df = df[df['KJ'] > 0]
    if has_if:
        df = df[df['IF'].between(0.5, 1.5) | df['IF'].isna()]

    for col in ['KJ_work','TRIMP_corr']:
        q99 = df[col].quantile(0.99)
        df  = df[df[col] <= q99]

    st.info(f"Sessões: **{len(df)}** de {n_raw}. "
            f"IF disponível: {'✅' if has_if else '❌ — densidade usa só RPE'}")

    if len(df) < 20:
        st.warning("Dados insuficientes.")
        return

    # ── CTL/ATL real ──────────────────────────────────────────────────────────
    all_dates = pd.date_range(df['Data'].min(), pd.Timestamp.now().normalize(), freq='D')
    load_d    = df.groupby('Data')['TRIMP_corr'].sum().reindex(all_dates, fill_value=0)
    ctl_s     = load_d.ewm(span=42, adjust=False).mean()
    atl_s     = load_d.ewm(span=7,  adjust=False).mean()
    load_raw  = (df['dur_min']*df['rpe_n']).groupby(df['Data']).sum().reindex(all_dates, fill_value=0)
    ctl_raw_s = load_raw.ewm(span=42, adjust=False).mean()

    # Fadiga D-1
    ratio_s = (atl_s / ctl_s.replace(0, np.nan)).shift(1)
    df_ctx  = pd.DataFrame({'Data': all_dates, 'ATL_CTL_d1': ratio_s.values})
    df      = df.merge(df_ctx, on='Data', how='left')

    # ── 4 tabs internas ───────────────────────────────────────────────────────
    t_coef, t_eff, t_ctl, t_serie, t_debug = st.tabs([
        "📊 Coeficientes", "📈 Eficiência", "🎯 CTL pred vs real", "📉 Série CTL", "🔬 Debug"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — COEFICIENTES (modelo sem IF — TRIMP ~ KJ_work + tipo)
    # ════════════════════════════════════════════════════════════════════════
    with t_coef:
        st.subheader("dTRIMP/dKJ — Modelo: TRIMP ~ KJ_work + densidade")
        st.caption(
            "IF removido da regressão (colinear com tipo). "
            "Densidade (IF×RPE) adicionada como variável independente. "
            "Modelo: TRIMP ~ KJ_work + densidade por tipo de sessão.")

        rows_coef = []

        for mod in sorted(df['type'].unique()):
            dm = df[df['type'] == mod].copy()
            kj_col = 'KJ_work' if mod == 'Bike' else 'KJ'

            # RUN: modelo separado dur×RPE
            if mod == 'Run':
                dm_run = dm.dropna(subset=['rpe_n','dur_min'])
                if len(dm_run) >= 5:
                    x = dm_run['dur_min'].values
                    y = (dm_run['dur_min'] * dm_run['rpe_n']).values
                    sl,ic,r,p,se = _scipy_stats.linregress(x, y)
                    rows_coef.append({
                        'Modalidade':'Run','Tipo':'dur×RPE','Modelo':'dur→TRIMP',
                        'N':len(dm_run),'dTRIMP/dKJ':'—','coef_dens':'—',
                        'R²':round(r**2,3),'RMSE':round(np.sqrt(np.mean(((ic+sl*x)-y)**2)),1),
                        'Bias':round(np.mean((ic+sl*x)-y),2),
                        'KJ médio':'—','TRIMP médio':round(y.mean(),1),
                        'eff médio':'—',
                    })
                continue

            dm = dm.dropna(subset=[kj_col,'TRIMP_corr','tipo','densidade'])
            if len(dm) < 8: continue

            for tipo_seg in ['todos'] + list(dm['tipo'].dropna().unique()):
                ds = dm if tipo_seg == 'todos' else dm[dm['tipo']==tipo_seg]
                if len(ds) < 5: continue

                x_kj   = ds[kj_col].values
                x_dens = ds['densidade'].values
                y      = ds['TRIMP_corr'].values

                # OLS: TRIMP ~ intercept + KJ_work + densidade
                X = np.column_stack([np.ones(len(y)), x_kj, x_dens])
                try:
                    beta  = np.linalg.lstsq(X, y, rcond=None)[0]
                    ic_m, coef_kj, coef_dens = beta
                    y_pred = X @ beta
                    ss_res = np.sum((y - y_pred)**2)
                    ss_tot = np.sum((y - y.mean())**2)
                    r2     = max(0.0, 1 - ss_res/ss_tot) if ss_tot > 0 else 0
                    rmse   = np.sqrt(ss_res/len(y))
                    bias   = float(np.mean(y_pred - y))
                except Exception:
                    continue

                eff_m = float(ds['eff_sessao'].dropna().mean())
                rows_coef.append({
                    'Modalidade':  mod,
                    'Tipo':        str(tipo_seg),
                    'Modelo':      'KJ+dens',
                    'N':           len(ds),
                    'dTRIMP/dKJ':  round(coef_kj, 4),
                    'coef_dens':   round(coef_dens, 3),
                    'R²':          round(r2, 3),
                    'RMSE':        round(rmse, 1),
                    'Bias':        round(bias, 2),
                    'KJ médio':    f"{ds[kj_col].mean():.0f}kJ",
                    'TRIMP médio': round(ds['TRIMP_corr'].mean(), 1),
                    'eff médio':   round(eff_m, 3),
                })

        if rows_coef:
            df_coef = pd.DataFrame(rows_coef)

            def _color_coef(val):
                try:
                    v = float(val)
                    if   v < 0.3: return 'background-color:#d5f5e3'
                    elif v < 0.5: return 'background-color:#fef9e7'
                    else:         return 'background-color:#fdecea'
                except: return ''

            num_cols = [c for c in ['dTRIMP/dKJ'] if c in df_coef.columns]
            st.dataframe(df_coef.style.map(_color_coef, subset=num_cols),
                         width='stretch', hide_index=True)
            st.caption("🟢 dTRIMP/dKJ < 0.3 eficiente | 🟡 0.3–0.5 normal | 🔴 > 0.5 caro")

            csv_coef = df_coef.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download coeficientes CSV",
                               csv_coef, "dtrimp_dkj_v3.csv", "text/csv",
                               key="dl_coef_v3")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — EFICIÊNCIA FISIOLÓGICA ROLLING
    # ════════════════════════════════════════════════════════════════════════
    with t_eff:
        st.subheader("📈 Eficiência Fisiológica — TRIMP / KJ_work")
        st.caption(
            "eff = TRIMP / KJ_work por sessão. Rolling 4 semanas por modalidade. "
            "↑ Subindo = fadiga acumulada (mesmo kJ custa mais). "
            "↓ Descendo = adaptação (mesmo kJ custa menos).")

        mods_eff = [m for m in ['Bike','Row','Ski'] if m in df['type'].values]
        if not mods_eff:
            st.info("Sem dados suficientes.")
        else:
            fig_eff = go.Figure()
            CORES_MOD = {'Bike':'#e74c3c','Row':'#2980b9','Ski':'#27ae60','Run':'#8e44ad'}

            for mod in mods_eff:
                dm = df[df['type']==mod].dropna(subset=['eff_sessao']).copy()
                kj_col = 'KJ_work' if mod == 'Bike' else 'KJ'
                dm = dm[dm['eff_sessao'].between(
                    dm['eff_sessao'].quantile(0.05),
                    dm['eff_sessao'].quantile(0.95))]
                if len(dm) < 4: continue

                # Agrupar por semana e calcular mediana
                dm['_sem'] = dm['Data'].dt.to_period('W').dt.start_time
                eff_sem = (dm.groupby('_sem')['eff_sessao']
                           .median().reset_index()
                           .rename(columns={'_sem':'Data','eff_sessao':'eff'}))
                eff_sem = eff_sem.sort_values('Data')

                # Rolling 4 semanas
                eff_sem['eff_roll4'] = eff_sem['eff'].rolling(4, min_periods=2).mean()

                # Tendência recente (últimas 8 semanas)
                rec = eff_sem.tail(8)
                if len(rec) >= 4:
                    x_t = np.arange(len(rec))
                    sl_t,_,r_t,_,_ = _scipy_stats.linregress(x_t, rec['eff_roll4'].ffill())
                    tend = ('↑ fadiga' if sl_t > 0.005 else
                            '↓ adaptação' if sl_t < -0.005 else '→ estável')
                else:
                    tend = '—'

                fig_eff.add_trace(go.Scatter(
                    x=eff_sem['Data'], y=eff_sem['eff'],
                    name=f"{mod} (sessão)",
                    mode='markers',
                    marker=dict(color=CORES_MOD.get(mod,'#888'), size=4, opacity=0.4)))
                fig_eff.add_trace(go.Scatter(
                    x=eff_sem['Data'], y=eff_sem['eff_roll4'],
                    name=f"{mod} (4sem) {tend}",
                    line=dict(color=CORES_MOD.get(mod,'#888'), width=2.5)))

            fig_eff.update_layout(
                paper_bgcolor='white', plot_bgcolor='white', height=400,
                legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                xaxis=dict(showgrid=True, gridcolor='#eee'),
                yaxis=dict(title='eff = TRIMP/KJ', showgrid=True, gridcolor='#eee'),
                title=dict(text='Eficiência Fisiológica (TRIMP/KJ) — rolling 4 semanas',
                           font=dict(size=13)))
            st.plotly_chart(fig_eff, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

            # Tabela resumo de eficiência actual e tendência
            st.markdown("**Eficiência actual por modalidade × tipo:**")
            rows_eff = []
            for mod in mods_eff:
                dm = df[df['type']==mod].dropna(subset=['eff_sessao','tipo'])
                for tipo_seg in ['base','tempo','intervalado']:
                    ds = dm[dm['tipo']==tipo_seg]
                    if len(ds) < 3: continue
                    eff_all  = ds['eff_sessao'].median()
                    eff_rec  = ds[ds['Data'] >= ds['Data'].max()-pd.Timedelta(weeks=8)]['eff_sessao'].median()
                    delta    = eff_rec - eff_all
                    tend_sym = ('↑' if delta >  0.05 else '↓' if delta < -0.05 else '→')
                    rows_eff.append({
                        'Modalidade': mod,
                        'Tipo':       tipo_seg,
                        'eff hist':   round(eff_all, 3),
                        'eff 8sem':   round(eff_rec, 3),
                        'Δ':          f"{delta:+.3f}",
                        'Tendência':  tend_sym + (' fadiga' if delta>0.05 else
                                                   ' adaptação' if delta<-0.05 else ' estável'),
                    })
            if rows_eff:
                st.dataframe(pd.DataFrame(rows_eff), width='stretch', hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — CTL pred vs real
    # ════════════════════════════════════════════════════════════════════════
    with t_ctl:
        st.subheader("CTL predito vs real por modalidade")

        # Toggle: calibração histórica completa vs rolling 90d
        _calib_mode = st.radio(
            "Janela de calibração dos coeficientes β",
            ["Histórico completo", "Rolling 90 dias (re-calibração recente)"],
            horizontal=True, index=0, key="ctl_calib_mode",
        )
        st.caption(
            "**Histórico completo**: β calibrado sobre toda a série — estável mas pode divergir se "
            "o atleta mudou muito recentemente. "
            "**Rolling 90d**: β re-calibrado com as últimas 90 dias — acompanha evolução rápida "
            "(ex: Row que melhorou muito em 2024-2026)."
        )

        _usar_rolling = "Rolling" in _calib_mode
        _janela_calib = 90  # dias

        rows_met = []
        for mod in sorted(df['type'].unique()):
            dm_full = df[df['type']==mod].copy()
            kj_col  = 'KJ_work' if mod == 'Bike' else 'KJ'

            if mod == 'Run':
                dm_full['TRIMP_pred_m'] = dm_full['dur_min'] * dm_full['rpe_n'].fillna(5)
            else:
                # Seleccionar janela de calibração
                if _usar_rolling:
                    _corte_calib = dm_full['Data'].max() - pd.Timedelta(days=_janela_calib)
                    dm_ok = dm_full[dm_full['Data'] >= _corte_calib].dropna(
                        subset=[kj_col,'TRIMP_corr','tipo','densidade'])
                    _calib_label = f"(β calibrado nos últimos {_janela_calib}d)"
                else:
                    dm_ok = dm_full.dropna(subset=[kj_col,'TRIMP_corr','tipo','densidade'])
                    _calib_label = "(β calibrado no histórico completo)"

                if len(dm_ok) < 8:
                    # Fallback para histórico completo se rolling tem poucos dados
                    dm_ok = dm_full.dropna(subset=[kj_col,'TRIMP_corr','tipo','densidade'])
                    _calib_label = "(β fallback — dados insuficientes no rolling)"

                if len(dm_ok) < 8: continue
                X_ok = np.column_stack([np.ones(len(dm_ok)),
                                        dm_ok[kj_col].values,
                                        dm_ok['densidade'].values])
                try:
                    beta = np.linalg.lstsq(X_ok, dm_ok['TRIMP_corr'].values, rcond=None)[0]
                except Exception:
                    continue
                dm_full['_kj_f']   = dm_full[kj_col].fillna(0)
                dm_full['_dens_f'] = dm_full['densidade'].fillna(dm_full['densidade'].median())
                dm_full['TRIMP_pred_m'] = (beta[0] + beta[1]*dm_full['_kj_f'] +
                                           beta[2]*dm_full['_dens_f']).clip(lower=0)

                # Nota de calibração para Row (onde bias é mais relevante)
                if mod == 'Row' and _usar_rolling:
                    _n_calib = len(dm_ok)
                    _beta_kj  = round(beta[1], 4)
                    st.info(
                        f"**Row — β re-calibrado**: dTRIMP/dKJ={_beta_kj} "
                        f"(últimos {_janela_calib}d, n={_n_calib} sessões). "
                        f"O bias de Row varia temporalmente (2023: +2.4, 2024: +4.9, 2025: -0.5, 2026: -16) "
                        f"porque o modelo histórico não acompanha a melhoria rápida do atleta em Row."
                    )

            pred_d = dm_full.groupby('Data')['TRIMP_pred_m'].sum().reindex(all_dates, fill_value=0)
            real_d = dm_full.groupby('Data')['TRIMP_corr'].sum().reindex(all_dates, fill_value=0)
            ctl_p  = pred_d.ewm(span=42, adjust=False).mean()
            ctl_r  = real_d.ewm(span=42, adjust=False).mean()
            diff   = ctl_p - ctl_r
            valid  = ctl_r > 0
            # Bias últimos 30d vs global
            diff_30d = diff[valid].tail(30)
            rows_met.append({
                'Modalidade':     mod,
                'CTL real':       round(float(ctl_r.iloc[-1]), 1),
                'CTL pred':       round(float(ctl_p.iloc[-1]), 1),
                'MAE':            round(float(diff[valid].abs().mean()), 2),
                'Bias global':    round(float(diff[valid].mean()), 3),
                'Bias 30d':       round(float(diff_30d.mean()), 3) if len(diff_30d) > 0 else '—',
                'Err%':           round(float((diff[valid].abs()/ctl_r[valid]*100).mean()), 1),
            })
        if rows_met:
            st.dataframe(pd.DataFrame(rows_met), use_container_width=True, hide_index=True)
            st.caption(
                "Bias positivo = sobreestima. Bias negativo = subestima. "
                "**Bias 30d** é o mais relevante — mostra o estado actual do modelo. "
                "RMSE < 15 = bom ajuste. Row: bias muda de sinal ao longo do tempo (deriva temporal)."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — SÉRIE CTL
    # ════════════════════════════════════════════════════════════════════════
    with t_serie:
        st.subheader("Série CTL — raw vs IF corrigido")
        df_ts = pd.DataFrame({
            'Data':    all_dates,
            'CTL_raw': ctl_raw_s.values.round(2),
            'CTL_IF':  ctl_s.values.round(2),
            'ATL':     atl_s.values.round(2),
            'TSB':     (ctl_s - atl_s).values.round(2),
        })
        fig_ctl = go.Figure()
        fig_ctl.add_trace(go.Scatter(
            x=df_ts['Data'], y=df_ts['CTL_raw'],
            name='CTL raw (dur×RPE)',
            line=dict(color='#95a5a6', dash='dot', width=1.5)))
        fig_ctl.add_trace(go.Scatter(
            x=df_ts['Data'], y=df_ts['CTL_IF'],
            name='CTL IF corrigido',
            line=dict(color='#2980b9', width=2)))
        fig_ctl.add_trace(go.Scatter(
            x=df_ts['Data'], y=df_ts['ATL'],
            name='ATL (7d)',
            line=dict(color='#e74c3c', width=1.5)))
        fig_ctl.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=380,
            legend=dict(orientation='h', y=-0.22, font=dict(color='#111')),
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            yaxis=dict(title='Carga', showgrid=True, gridcolor='#eee'),
            title=dict(text='CTL raw vs IF corrigido', font=dict(size=13)))
        st.plotly_chart(fig_ctl, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
        csv_ts = df_ts.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download série CTL (agrupada)", csv_ts,
                           "series_ctl_v3.csv", "text/csv", key="dl_ts_v3")

        st.markdown("---")
        st.subheader("⬇️ Export sessões — dados por actividade")
        st.caption(
            "date | modality | duration_min | kj_total | z1_kj | z2_kj | z3_kj | "
            "rpe | trimp | ctl | atl + power_avg | IF (opcionais)")

        # Construir DataFrame de sessões com CTL/ATL por sessão
        _df_exp = df.copy()
        _df_exp = _df_exp.sort_values("Data").reset_index(drop=True)

        # CTL/ATL no dia de cada sessão (join com série diária)
        _ctl_map = pd.Series(ctl_s.values,   index=all_dates).to_dict()
        _atl_map = pd.Series(atl_s.values,   index=all_dates).to_dict()
        _df_exp["ctl"] = _df_exp["Data"].map(_ctl_map).round(2)
        _df_exp["atl"] = _df_exp["Data"].map(_atl_map).round(2)

        # Colunas essenciais
        _exp_cols = {
            "date":         _df_exp["Data"].dt.strftime("%Y-%m-%d"),
            "modality":     _df_exp["type"].apply(norm_tipo),
            "duration_min": _df_exp["dur_min"].round(1),
            "kj_total":     _df_exp["KJ"].round(1),
            "z1_kj":        pd.to_numeric(_df_exp.get("z1_kj", np.nan), errors="coerce").round(1)
                            if "z1_kj" in _df_exp.columns else np.nan,
            "z2_kj":        pd.to_numeric(_df_exp.get("z2_kj", np.nan), errors="coerce").round(1)
                            if "z2_kj" in _df_exp.columns else np.nan,
            "z3_kj":        pd.to_numeric(_df_exp.get("z3_kj", np.nan), errors="coerce").round(1)
                            if "z3_kj" in _df_exp.columns else np.nan,
            "rpe":          pd.to_numeric(_df_exp["rpe_n"], errors="coerce").round(1),
            "trimp":        _df_exp["TRIMP_corr"].round(1),
            "ctl":          _df_exp["ctl"],
            "atl":          _df_exp["atl"],
        }

        # Opcionais
        if "power_avg" in _df_exp.columns:
            _exp_cols["power_avg"] = pd.to_numeric(_df_exp["power_avg"], errors="coerce").round(1)
        if "IF" in _df_exp.columns and _df_exp["IF"].notna().any():
            _exp_cols["if"] = _df_exp["IF"].round(3)

        _df_export = pd.DataFrame(_exp_cols)

        # Preview últimas 5 sessões
        st.dataframe(_df_export.tail(5), hide_index=True)
        st.caption(f"Total: {len(_df_export)} sessões | "
                   f"Período: {_df_export['date'].iloc[0]} → {_df_export['date'].iloc[-1]}")

        _csv_exp = _df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download sessões CSV",
            _csv_exp,
            "atheltica_sessions_export.csv",
            "text/csv",
            key="dl_sessions_exp")

    # ════════════════════════════════════════════════════════════════════════
    # TAB DEBUG — 5 testes visuais do sistema de sugestão
    # ════════════════════════════════════════════════════════════════════════
    with t_debug:
        st.subheader("🔬 Debug Visual — Sistema de Sugestão")
        st.caption(
            "Diagnóstico do comportamento dos 3 sinais: eff_delta, KJ/h, pwr_inc. "
            "Baseado nas sessões filtradas pelo modelo (pool por zona/RPE).")

        if len(df) < 10:
            st.warning("Dados insuficientes para debug.")
        else:
            # ── Calcular sinais por sessão ────────────────────────────────
            _debug_rows = []
            _MODS_DEBUG = [m for m in ['Bike','Row','Ski','Run'] if m in df['type'].values]

            for _mod in _MODS_DEBUG:
                _dm = df[df['type'].apply(norm_tipo) == _mod].copy()
                if len(_dm) < 5: continue
                _kj_col = 'KJ_work' if _mod == 'Bike' else 'KJ'

                # Pool completo desta modalidade (sem filtro de zona — ver evolução total)
                _dm2 = _dm.dropna(subset=[_kj_col, 'TRIMP_corr', 'rpe_n']).copy()
                _dm2 = _dm2.sort_values('Data')

                # eff por sessão
                _kj_w = _dm2[_kj_col].replace(0, np.nan)
                _dm2['_eff'] = _dm2['TRIMP_corr'] / _kj_w

                # KJ/h por sessão
                _dm2['_kjh'] = (_dm2[_kj_col] / (_dm2['dur_min'] / 60)).replace([np.inf, -np.inf], np.nan)

                # eff_delta rolling (eff_rec / eff_baseline - 1) — janela deslizante 8 semanas
                _dm2['_eff_roll']  = _dm2['_eff'].rolling(8, min_periods=3).mean()
                _dm2['_eff_base']  = _dm2['_eff'].rolling(40, min_periods=10).mean()
                _dm2['_eff_delta'] = (_dm2['_eff_roll'] / _dm2['_eff_base'] - 1).clip(-0.5, 0.5)
                # Garantir que _kjh_roll e _kjh_base existem sempre
                _dm2['_kjh_roll']  = _dm2['_kjh'].rolling(5,  min_periods=2).mean()
                _dm2['_kjh_base']  = _dm2['_kjh'].rolling(30, min_periods=8).mean()
                _dm2['_kjh_ratio'] = (_dm2['_kjh_roll'] / _dm2['_kjh_base']).clip(0.5, 1.5)

                # KJ/h rolling e baseline
                _dm2['_kjh_roll']  = _dm2['_kjh'].rolling(5, min_periods=2).mean()
                _dm2['_kjh_base']  = _dm2['_kjh'].rolling(30, min_periods=8).mean()
                _dm2['_kjh_ratio'] = (_dm2['_kjh_roll'] / _dm2['_kjh_base']).clip(0.5, 1.5)

                # pwr_inc simulado (base por zona)
                try:
                    _ni_mod = _ni_cache.get(_mod, 50)
                except NameError:
                    _ni_mod = 50
                if _ni_mod >= 75:    _pwr_base = 0.01
                elif _ni_mod >= 40:  _pwr_base = 0.02
                else:                _pwr_base = 0.02

                def _calc_pwr_inc(row):
                    ed = row['_eff_delta'] if pd.notna(row['_eff_delta']) else 0.0
                    kr = row['_kjh_ratio'] if pd.notna(row['_kjh_ratio']) else 1.0
                    p = _pwr_base
                    if   ed < -0.05: p *= 1.2
                    elif ed < 0.05:  p *= 1.0
                    elif ed < 0.12:  p *= 0.9
                    else:            p *= 0.7
                    if   kr >= 1.0:  p *= 1.1
                    elif kr >= 0.95: p *= 1.0
                    return round(p * 100, 2)  # em %

                _dm2['_pwr_inc_pct'] = _dm2.apply(_calc_pwr_inc, axis=1)
                _dm2['_mod'] = _mod
                # Garantir todas as colunas necessárias
                for _c in ['_kjh_roll','_kjh_base','_kjh_ratio','_eff_delta','_pwr_inc_pct']:
                    if _c not in _dm2.columns: _dm2[_c] = np.nan
                _debug_rows.append(_dm2[['Data','_mod','_eff','_eff_delta','_kjh',
                                          '_kjh_roll','_kjh_base','_kjh_ratio',
                                          '_pwr_inc_pct','TRIMP_corr',
                                          _kj_col, 'rpe_n']].copy())

            if not _debug_rows:
                st.info("Sem dados para debug.")
            else:
                _df_dbg = pd.concat(_debug_rows, ignore_index=True).sort_values('Data')
                _CORES_M = {'Bike':'#e74c3c','Row':'#2980b9','Ski':'#27ae60','Run':'#8e44ad'}

                # ── GRÁFICO 1: eff_delta ao longo do tempo ────────────────
                st.markdown("**Gráfico 1 — eff_delta (custo interno relativo ao baseline)**")
                st.caption(
                    "< −5%: adaptação | −5% a +5%: normal | "
                    "+5% a +12%: fadiga produtiva | > +12%: fadiga alta")

                fig_d1 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_eff_delta'])
                    if len(_s) < 3: continue
                    fig_d1.add_trace(go.Scatter(
                        x=_s['Data'], y=(_s['_eff_delta'] * 100).round(1),
                        mode='lines+markers', name=_mod,
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=2),
                        marker=dict(size=4),
                        hovertemplate='%{x|%d/%m/%y}<br>eff_delta: <b>%{y:.1f}%%</b><extra></extra>'))
                # Zonas de fundo
                for y0, y1, col, lbl in [
                    (-50, -5, 'rgba(39,174,96,0.08)',  'adaptação'),
                    (-5,   5, 'rgba(52,152,219,0.06)', 'normal'),
                    (5,   12, 'rgba(243,156,18,0.10)', 'fadiga prod.'),
                    (12,  50, 'rgba(231,76,60,0.10)',  'fadiga alta'),
                ]:
                    fig_d1.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0,
                                     annotation_text=lbl, annotation_position='top left',
                                     annotation_font=dict(size=10, color='#555'))
                fig_d1.add_hline(y=0, line_dash='dash', line_color='#aaa', line_width=1)
                fig_d1.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=320,
                    legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                    xaxis=dict(showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='eff_delta (%)', showgrid=True, gridcolor='#eee'),
                    title=dict(text='eff_delta — custo interno relativo ao baseline', font=dict(size=12)))
                st.plotly_chart(fig_d1, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

                # ── GRÁFICO 2: KJ/h ao longo do tempo ────────────────────
                st.markdown("**Gráfico 2 — KJ/h (densidade de trabalho)**")
                st.caption("Rolling 5 sessões vs baseline 30 sessões. Ratio abaixo de 1.0 → A reduzida.")

                fig_d2 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_kjh_roll'])
                    if len(_s) < 3: continue
                    fig_d2.add_trace(go.Scatter(
                        x=_s['Data'], y=_s['_kjh_roll'].round(0),
                        mode='lines', name=f'{_mod} roll',
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=2.5),
                        hovertemplate='%{x|%d/%m/%y}<br>KJ/h: <b>%{y:.0f}</b><extra></extra>'))
                    fig_d2.add_trace(go.Scatter(
                        x=_s['Data'], y=_s['_kjh_base'].round(0),
                        mode='lines', name=f'{_mod} baseline',
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=1, dash='dot'),
                        hovertemplate='%{x|%d/%m/%y}<br>KJ/h baseline: <b>%{y:.0f}</b><extra></extra>'))
                fig_d2.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=320,
                    legend=dict(orientation='h', y=-0.30, font=dict(color='#111')),
                    xaxis=dict(showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='KJ/h', showgrid=True, gridcolor='#eee'),
                    title=dict(text='KJ/h rolling vs baseline por modalidade', font=dict(size=12)))
                st.plotly_chart(fig_d2, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

                # ── GRÁFICO 3: pwr_inc simulado ───────────────────────────
                st.markdown("**Gráfico 3 — pwr_inc simulado (%)**")
                st.caption("Incremento de power que seria sugerido com base nos 3 sinais naquele dia.")

                fig_d3 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_pwr_inc_pct'])
                    if len(_s) < 3: continue
                    fig_d3.add_trace(go.Scatter(
                        x=_s['Data'], y=_s['_pwr_inc_pct'],
                        mode='lines+markers', name=_mod,
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=2),
                        marker=dict(size=4),
                        hovertemplate='%{x|%d/%m/%y}<br>pwr_inc: <b>%{y:.2f}%%</b><extra></extra>'))
                fig_d3.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=300,
                    legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                    xaxis=dict(showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='pwr_inc (%)', showgrid=True, gridcolor='#eee'),
                    title=dict(text='pwr_inc simulado por sessão', font=dict(size=12)))
                st.plotly_chart(fig_d3, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

                st.markdown("---")

                # ── TESTE 1: log eff_delta / kjh_ratio / pwr_inc ─────────
                st.markdown("**Teste 1 — Log: eff_delta | kjh_ratio | pwr_inc por sessão**")
                _cols_log = ['Data','_mod','rpe_n','_eff_delta','_kjh_ratio','_pwr_inc_pct']
                _df_log = _df_dbg[_cols_log].dropna().tail(50).copy()
                _df_log.columns = ['Data','Modalidade','RPE','eff_delta%','kjh_ratio','pwr_inc%']
                _df_log['eff_delta%'] = (_df_log['eff_delta%'] * 100).round(1)
                _df_log['kjh_ratio']  = _df_log['kjh_ratio'].round(3)
                _df_log['Data'] = _df_log['Data'].dt.strftime('%d/%m/%y')
                st.dataframe(_df_log.iloc[::-1], hide_index=True, width="stretch")

                st.markdown("---")

                # ── TESTE 2: A vs B dominância ────────────────────────────
                st.markdown("**Teste 2 — Dominância A vs B por modalidade**")
                st.caption("Simulação: quando B (intensidade) seria maior que A (volume)?")

                _rows_ab = []
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(
                        subset=['_eff_delta','_kjh_ratio','_pwr_inc_pct'])
                    if len(_s) < 3: continue

                    _kj_col = 'KJ_work' if _mod == 'Bike' else 'KJ'
                    _n_total = len(_s)

                    # B = mesmo tempo, mais power → KJ_B = ref_kj * (1 + pwr_inc/100)
                    # A = mais tempo, mesmo power → limitado por cap
                    # Simular: B > A quando pwr_inc efectivo suficiente
                    _n_b_dom = (_s['_pwr_inc_pct'] >= 1.0).sum()   # B produz ganho real
                    _n_a_dom = (_s['_kjh_ratio']   <  0.95).sum()  # A reduzida
                    _n_normal = _n_total - _n_b_dom - _n_a_dom + min(_n_b_dom, _n_a_dom)

                    _rows_ab.append({
                        'Modalidade':  _mod,
                        'N sessões':   _n_total,
                        'B dominante (pwr≥1%)':  f"{_n_b_dom} ({_n_b_dom/_n_total*100:.0f}%)",
                        'A reduzida (kjh<95%)':  f"{_n_a_dom} ({_n_a_dom/_n_total*100:.0f}%)",
                        'pwr_inc médio':         f"{_s['_pwr_inc_pct'].mean():.2f}%",
                        'pwr_inc max':           f"{_s['_pwr_inc_pct'].max():.2f}%",
                        'pwr_inc min':           f"{_s['_pwr_inc_pct'].min():.2f}%",
                    })
                if _rows_ab:
                    st.dataframe(pd.DataFrame(_rows_ab), hide_index=True, width="stretch")

                st.markdown("---")

                # ── TESTE 3: KJ/h rolling 7–14 dias ─────────────────────
                st.markdown("**Teste 3 — KJ/h rolling 7 e 14 sessões**")
                _rows_kjh = []
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_kjh']).copy()
                    if len(_s) < 5: continue
                    _s = _s.sort_values('Data')
                    _kjh7  = _s['_kjh'].rolling(7,  min_periods=3).mean().iloc[-1]
                    _kjh14 = _s['_kjh'].rolling(14, min_periods=5).mean().iloc[-1]
                    _kjh_all = _s['_kjh'].mean()
                    _rows_kjh.append({
                        'Modalidade':    _mod,
                        'KJ/h histórico': round(_kjh_all, 0),
                        'KJ/h roll 7':   round(_kjh7, 0),
                        'KJ/h roll 14':  round(_kjh14, 0),
                        'Ratio 7/hist':  f"{_kjh7/_kjh_all:.2f}" if _kjh_all > 0 else '—',
                        'Ratio 14/hist': f"{_kjh14/_kjh_all:.2f}" if _kjh_all > 0 else '—',
                    })
                if _rows_kjh:
                    st.dataframe(pd.DataFrame(_rows_kjh), hide_index=True, width="stretch")

                st.markdown("---")

                # ── TESTE 4: eff_delta vs pwr_inc scatter ─────────────────
                st.markdown("**Teste 4 — eff_delta vs pwr_inc (fadiga vs resposta)**")
                st.caption("Esperado: correlação negativa — mais fadiga → menor pwr_inc.")

                fig_d4 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(
                        subset=['_eff_delta','_pwr_inc_pct'])
                    if len(_s) < 5: continue
                    fig_d4.add_trace(go.Scatter(
                        x=(_s['_eff_delta'] * 100).round(1),
                        y=_s['_pwr_inc_pct'],
                        mode='markers', name=_mod,
                        marker=dict(color=_CORES_M.get(_mod,'#888'), size=6, opacity=0.6),
                        hovertemplate=f'{_mod}<br>eff_delta: <b>%{{x:.1f}}%%</b><br>pwr_inc: <b>%{{y:.2f}}%%</b><extra></extra>'))
                # Linhas verticais das zonas
                for xv, col in [(-5,'#27ae60'),(5,'#f39c12'),(12,'#e74c3c')]:
                    fig_d4.add_vline(x=xv, line_dash='dot', line_color=col, line_width=1.5)
                fig_d4.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=320,
                    legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                    xaxis=dict(title='eff_delta (%)', showgrid=True, gridcolor='#eee', zeroline=True),
                    yaxis=dict(title='pwr_inc (%)', showgrid=True, gridcolor='#eee'),
                    title=dict(text='eff_delta vs pwr_inc — fadiga vs resposta', font=dict(size=12)))
                st.plotly_chart(fig_d4, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

                st.markdown("---")

                # ── TESTE 5: prioridade vs pwr_inc ───────────────────────
                st.markdown("**Teste 5 — Prioridade vs pwr_inc médio**")
                st.caption(
                    "Esperado: modalidades P1/P2 (Foco) com ni mais alto → "
                    "pwr_inc maior que P3/P4 (Manutenção).")

                _rows_prio = []
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_pwr_inc_pct'])
                    if len(_s) < 3: continue
                    try:
                        _ni_m  = _ni_cache.get(_mod, 50)
                        _ol_m  = _ol_cache.get(_mod, False)
                        _need  = _need_cache.get(_mod, 40)
                    except NameError:
                        _ni_m, _ol_m, _need = 50, False, 40
                    _p1 = st.session_state.get('vg_prio1','Bike')
                    _p2 = st.session_state.get('vg_prio2','Row')
                    _grupo = '🎯 Foco' if _mod in {_p1, _p2} else '🔧 Manutenção'
                    _rows_prio.append({
                        'Modalidade':   _mod,
                        'Grupo':        _grupo,
                        'Need_int':     _ni_m,
                        'Need_score':   round(_need, 1),
                        'Overload':     '⚠️' if _ol_m else '—',
                        'pwr_inc médio': f"{_s['_pwr_inc_pct'].mean():.2f}%",
                        'pwr_inc últimas 5': f"{_s['_pwr_inc_pct'].tail(5).mean():.2f}%",
                    })
                if _rows_prio:
                    _rows_prio.sort(key=lambda x: float(x['pwr_inc médio'].replace('%','')),
                                    reverse=True)
                    st.dataframe(pd.DataFrame(_rows_prio), hide_index=True, width="stretch")
                    st.caption(
                        "pwr_inc é determinado por eff_delta e kjh_ratio (dados históricos), "
                        "NÃO directamente pela prioridade. "
                        "A prioridade influencia Need_int → zona alvo → pwr_inc base.")

                # Download debug CSV
                st.markdown("---")
                _df_dbg_dl = _df_dbg.copy()
                _df_dbg_dl['Data'] = _df_dbg_dl['Data'].dt.strftime('%Y-%m-%d')
                _df_dbg_dl['_eff_delta_pct'] = (_df_dbg_dl['_eff_delta'] * 100).round(2)
                _df_dbg_dl = _df_dbg_dl.rename(columns={
                    '_mod':'modality','_eff':'eff','_eff_delta_pct':'eff_delta_pct',
                    '_kjh':'kjh','_kjh_ratio':'kjh_ratio','_pwr_inc_pct':'pwr_inc_pct',
                    'rpe_n':'rpe','TRIMP_corr':'trimp'})
                _cols_dl = [c for c in ['Data','modality','rpe','eff','eff_delta_pct',
                                         'kjh','kjh_ratio','pwr_inc_pct','trimp']
                            if c in _df_dbg_dl.columns]
                _csv_dbg = _df_dbg_dl[_cols_dl].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download debug CSV",
                    _csv_dbg, "atheltica_debug_signals.csv",
                    "text/csv", key="dl_dbg_signals")
