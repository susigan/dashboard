# ════════════════════════════════════════════════════════════════════════════════
# tab_analises.py — Análises Avançadas (8 secções)
# ════════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from scipy import stats as scipy_stats
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

from config import CORES, CORES_ATIV, TYPE_MAP, VALID_TYPES, CICLICOS, ANNUAL_SPREADSHEET_ID, ANNUAL_SHEETS
from utils.helpers import (
    filtrar_principais, add_tempo, norm_tipo, get_cor, norm_serie,
    cvr, conv_15, norm_range, calcular_swc, classificar_rpe, remove_zscore,
    calcular_series_carga, calcular_bpe, calcular_recovery,
    calcular_polinomios_carga, analisar_falta_estimulo,
    tabela_resumo_por_tipo_df, tabela_ranking_power_df,
)
from data_loader import carregar_wellness, carregar_atividades, carregar_annual, preproc_wellness, preproc_ativ, filtrar_datas

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

def analisar_falta_estimulo(df_act_full, janela_dias=14):
    """Análise de falta de estímulo por modalidade — igual ao original."""
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

    data_limite = pd.Timestamp.now() - pd.Timedelta(days=janela_dias)
    carga_rec = ld[ld['Data'] >= data_limite].copy()
    if len(carga_rec) == 0:
        return None

    resultados = {}
    for mod in ['Bike', 'Run', 'Row', 'Ski']:
        df_mod = df[(df['type'] == mod) & (df['Data'] >= data_limite)]
        dias_ativ = df_mod['Data'].nunique()
        freq = dias_ativ / max(janela_dias, 1)

        atl_m = carga_rec['ATL'].mean()
        ctl_m = carga_rec['CTL'].mean()
        gap = ((ctl_m - atl_m) / ctl_m * 100) if ctl_m > 0 else 0
        dias_atl_baixo = (carga_rec['ATL'] < carga_rec['CTL']).sum()

        x_s = np.arange(len(carga_rec))
        slope = np.polyfit(x_s, carga_rec['ATL'].values, 1)[0] if len(carga_rec) > 1 else 0
        slope_norm = max(0, min(1, (slope + 5) / 10))

        need = (
            min(1, max(0, gap / 50)) * 100 * 0.4 +
            min(1, dias_atl_baixo / max(len(carga_rec), 1)) * 100 * 0.3 +
            (1 - slope_norm) * 100 * 0.2 +
            (1 - freq) * 100 * 0.1
        )
        prio = 'ALTA' if need >= 70 else 'MÉDIA' if need >= 40 else 'BAIXA'
        resultados[mod] = {
            'need_score': need, 'prioridade': prio,
            'gap_relativo': gap, 'dias_atl_menor_ctl': int(dias_atl_baixo),
            'dias_com_atividade': dias_ativ, 'atl_medio': atl_m, 'ctl_medio': ctl_m
        }
    return dict(sorted(resultados.items(), key=lambda x: x[1]['need_score'], reverse=True))

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
        fig, ax = plt.subplots(figsize=(16, 6))
        bottom = np.zeros(len(pivot_tl))
        for tipo in [t for t in ['Bike', 'Row', 'Ski', 'Run', 'WeightTraining'] if t in pivot_tl.columns]:
            vals = pivot_tl[tipo].values
            ax.bar(range(len(pivot_tl)), vals, bottom=bottom, label=tipo,
                   color=CORES_MOD.get(tipo, '#888'), alpha=0.85, edgecolor='white', linewidth=0.5)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 50:
                    ax.text(i, b + v/2, f'{v:.0f}', ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            bottom += vals
        totais = pivot_tl.sum(axis=1).values
        for i, t in enumerate(totais):
            if t > 0: ax.text(i, t + 5, f'{t:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(pivot_tl)))
        ax.set_xticklabels(pivot_tl.index, rotation=45, ha='right')
        ax.axhline(totais.mean(), color='black', linestyle='--', alpha=0.5, label=f'Média: {totais.mean():.0f}')
        ax.set_ylabel('Training Load (TRIMP)'); ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3, axis='y')
        ax.set_title('Training Load Mensal por Modalidade', fontsize=13, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

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
        fig, ax = plt.subplots(figsize=(16, 7))
        for metrica, (cor_s, cor_l, sty) in [
            ('CTL', (CORES['azul'],    CORES['azul_escuro'],    '-')),
            ('ATL', (CORES['vermelho'],CORES['vermelho_escuro'],'--'))]:
            if metrica not in poli.get('overall', {}): continue
            dm = poli['overall'][metrica]
            gk = 'grau3' if 'grau3' in dm else 'grau2'
            if gk not in dm: continue
            d = dm[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
            xs = np.linspace(x.min(), x.max(), 200)
            ax.scatter(x, y, alpha=0.3, s=40, color=cor_s, edgecolors='white', linewidths=1, label=f'{metrica} dados')
            ax.plot(xs, poly(xs), linewidth=3, color=cor_l, linestyle=sty,
                    label=f'{metrica} Poly{gk.replace("grau","")} (R²={r2:.3f})')
        ax.set_xlabel('Dias'); ax.set_ylabel('Carga (TRIMP)')
        ax.set_title('CTL vs ATL Overall — Polynomial Fit', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Por modalidade — separado e combinado
        tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
        if tipos_poli:
            st.markdown("**Por Modalidade**")
            n_t = len(tipos_poli); ncols = 2; nrows = (n_t + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6*nrows))
            axes_flat = axes.flatten() if n_t > 1 else [axes] if ncols*nrows == 1 else axes.flatten()
            for idx, (tipo_n, tipo_k) in enumerate(sorted(tipos_poli.items())):
                ax = axes_flat[idx]
                for metrica, cor, sty in [('CTL', CORES['azul'], '-'), ('ATL', CORES['vermelho'], '--')]:
                    if metrica not in poli[tipo_k]: continue
                    dm = poli[tipo_k][metrica]
                    gk = 'grau3' if 'grau3' in dm else 'grau2'
                    if gk not in dm: continue
                    d = dm[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
                    xs = np.linspace(x.min(), x.max(), 150)
                    ax.scatter(x, y, alpha=0.35, s=40, color=cor, edgecolors='white', linewidths=1)
                    ax.plot(xs, poly(xs), linewidth=2.5, color=cor, linestyle=sty,
                            label=f'{metrica} R²={r2:.3f}')
                ax.set_title(f'{tipo_n} — CTL/ATL Polynomial', fontsize=11, fontweight='bold')
                ax.set_xlabel('Dias'); ax.set_ylabel('Carga'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            for idx in range(n_t, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            plt.suptitle('CTL/ATL por Modalidade — Polynomial Fit', fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

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
            fig, ax = plt.subplots(figsize=(max(14, len(semanas)*0.9), max(5, nm*1.2)))
            im = ax.imshow(mat, cmap=cmap_bpe, aspect='auto', vmin=-2, vmax=2)
            ax.set_yticks(range(nm))
            ax.set_yticklabels([nomes_bpe.get(m, m) for m in dados_bpe], fontsize=11)
            sem_labels = [str(s).split('-W')[1] if '-W' in str(s) else str(s) for s in semanas]
            ax.set_xticks(range(len(semanas)))
            ax.set_xticklabels([f'S{s}' for s in sem_labels], rotation=45, fontsize=10)
            for i in range(nm):
                for j in range(len(semanas)):
                    v = mat[i, j]; cor_t = 'white' if abs(v) > 1 else 'black'
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=cor_t)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Z-Score BPE (múltiplos de SWC)')
            cbar.set_ticks([-2,-1,0,1,2]); cbar.set_ticklabels(['🔴-2','-1','0','+1','🟢+2'])
            ax.set_title('BPE — Blocos de Padrão Específico (Z-Score com SWC)', fontsize=13, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Mínimo 14 dias de wellness para BPE.")

    st.markdown("---")

    # ── Secção 5: Falta de Estímulo ─────────────────────────────────────────
    st.subheader("🎯 Análise de Falta de Estímulo por Modalidade")
    c1, c2 = st.columns(2)
    for col_w, janela, label in [(c1, 7, "7 dias"), (c2, 14, "14 dias")]:
        res = analisar_falta_estimulo(da_full, janela)
        with col_w:
            st.markdown(f"**📅 Janela {label}**")
            if res:
                rows_fe = []
                for mod, d in res.items():
                    pe = "🔴" if d['prioridade']=='ALTA' else "🟡" if d['prioridade']=='MÉDIA' else "🟢"
                    rows_fe.append({'Modalidade': mod, 'Need Score': f"{d['need_score']:.1f}",
                                    'Prioridade': f"{pe} {d['prioridade']}",
                                    'Gap CTL-ATL': f"{d['gap_relativo']:.1f}%",
                                    'Dias ATL<CTL': d['dias_atl_menor_ctl'],
                                    'Dias c/ Ativ.': d['dias_com_atividade']})
                st.dataframe(pd.DataFrame(rows_fe), use_container_width=True, hide_index=True)
                top = list(res.keys())[0]
                pe = "🔴" if res[top]['prioridade']=='ALTA' else "🟡" if res[top]['prioridade']=='MÉDIA' else "🟢"
                st.info(f"{pe} Foco recomendado: **{top}** (Score: {res[top]['need_score']:.1f})")
            else:
                st.info("Dados insuficientes.")

    st.markdown("---")

    # ── Secção 6: Annual — Aquecimentos ─────────────────────────────────────
    st.subheader("📅 Dados Annual — Aquecimentos por Modalidade")
    MODAL_ABA = {'Ski': 'AquecSki', 'Bike': 'AquecBike', 'Row': 'AquecRow'}
    MODAL_HR  = {'Ski':  ['HR_140W','HR_160W','HR_180W','HR_200W'],
                 'Bike': ['HR_140W','HR_160W','HR_180W','HR_200W'],
                 'Row':  ['HR_140W','HR_160W','HR_180W','HR_200W']}
    MODAL_O2  = {'Row': ['O2_140W','O2_160W','O2_180W']}

    if not df_annual.empty:
        tabs_aq = st.tabs(["🎿 Ski", "🚴 Bike", "🚣 Row"])
        for tab_aq, (modal, aba) in zip(tabs_aq, MODAL_ABA.items()):
            with tab_aq:
                df_a = dfs_annual.get(aba, pd.DataFrame())
                if df_a.empty:
                    st.info(f"Sem dados para {aba}.")
                    continue
                st.write(f"**{aba}** — {len(df_a)} registos")
                st.dataframe(df_a.head(10), use_container_width=True)

                hr_cols = [c for c in MODAL_HR.get(modal, []) if c in df_a.columns]
                if hr_cols and df_a[hr_cols].notna().any().any():
                    fig, ax = plt.subplots(figsize=(14, 5))
                    for col in hr_cols:
                        vals = df_a[col].dropna()
                        if len(vals) > 0:
                            ax.plot(range(len(vals)), vals.values, marker='o', label=col, linewidth=2, alpha=0.8)
                    ax.set_title(f'{aba} — HR por Potência ao Longo do Tempo', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Sessão'); ax.set_ylabel('HR (bpm)')
                    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                o2_cols = [c for c in MODAL_O2.get(modal, []) if c in df_a.columns]
                if o2_cols and df_a[o2_cols].notna().any().any():
                    fig, ax = plt.subplots(figsize=(14, 4))
                    for col in o2_cols:
                        vals = df_a[col].dropna()
                        if len(vals) > 0:
                            ax.plot(range(len(vals)), vals.values, marker='s', label=col, linewidth=2, alpha=0.8)
                    ax.set_title(f'{aba} — SmO2 por Potência', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Sessão'); ax.set_ylabel('SmO2 (%)')
                    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Planilha Annual não carregada (verificar ANNUAL_SPREADSHEET_ID e permissões).")

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
        c5.metric("Horas 7d",       f"{horas7:.1f}h")
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

# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: app.py (sidebar + main)
# ════════════════════════════════════════════════════════════════════════════

# DEVE ser a primeira chamada Streamlit — define layout wide global
st.set_page_config(
    page_title="ATHELTICA",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)
