# AUTO-GENERATED — edit this file directly
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

from config import CORES, CORES_ATIV, CICLICOS
from utils.helpers import (
    filtrar_principais, add_tempo, norm_tipo, get_cor, norm_serie,
    cvr, conv_15, norm_range, calcular_swc, classificar_rpe,
    calcular_series_carga, calcular_bpe, calcular_recovery,
    calcular_polinomios_carga, analisar_falta_estimulo
)

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES — ANÁLISES AVANÇADAS (do original Python/SQLite)
# ════════════════════════════════════════════════════════════════════════════════

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
def tab_analises(da_full, dw):
    st.header("🔬 Análises Avançadas")
    if len(da_full) == 0:
        st.warning("Sem dados de atividades para análise avançada.")
        return

    # ── Secção 1: Tabelas de Resumo ─────────────────────────────────────────
    st.subheader("📋 Resumo de Atividades por Modalidade")
    df_res = tabela_resumo_por_tipo_df(da_full)
    if len(df_res) > 0:
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.info("Sem dados para tabela de resumo.")

    st.subheader("🏆 Top 10 Sessões por Potência Média")
    df_rank = tabela_ranking_power_df(da_full, n=10)
    if len(df_rank) > 0:
        st.dataframe(df_rank, use_container_width=True, hide_index=True)
    else:
        st.info("Sem dados de potência para ranking.")

    st.markdown("---")

    # ── Secção 2: Training Load Mensal Stacked ──────────────────────────────
    st.subheader("📊 Training Load Mensal por Modalidade (Stacked)")
    df_tl = filtrar_principais(da_full).copy()
    df_tl = add_tempo(df_tl)
    if 'moving_time' in df_tl.columns and 'rpe' in df_tl.columns:
        df_tl['rpe_fill'] = df_tl['rpe'].fillna(df_tl['rpe'].median())
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
                    ax.text(i, b + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            bottom += vals
        totais = pivot_tl.sum(axis=1).values
        for i, t in enumerate(totais):
            if t > 0:
                ax.text(i, t + 5, f'{t:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(pivot_tl)))
        ax.set_xticklabels(pivot_tl.index, rotation=45, ha='right')
        ax.axhline(totais.mean(), color='black', linestyle='--', alpha=0.5, label=f'Média: {totais.mean():.0f}')
        ax.set_ylabel('Training Load (TRIMP = min × RPE)', fontweight='bold')
        ax.set_title('Training Load Mensal por Modalidade', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Sem dados de moving_time e RPE para Training Load.")

    st.markdown("---")

    # ── Secção 3: Polynomial CTL/ATL ────────────────────────────────────────
    st.subheader("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)")
    with st.spinner("Calculando polynomial fits..."):
        poli = calcular_polinomios_carga(da_full)

    if poli is None:
        st.warning("Sem dados suficientes para polynomial analysis.")
    else:
        ld = poli.get('_ld')

        # Overall
        st.markdown("**Overall CTL vs ATL**")
        fig, ax = plt.subplots(figsize=(16, 7))
        CORES_POLI = {'CTL': (CORES['azul'], CORES['azul_escuro']), 'ATL': (CORES['vermelho'], CORES['vermelho_escuro'])}
        for metrica, (cor_s, cor_l) in CORES_POLI.items():
            if metrica not in poli.get('overall', {}): continue
            dados_m = poli['overall'][metrica]
            gk = 'grau3' if 'grau3' in dados_m else 'grau2'
            if gk not in dados_m: continue
            d = dados_m[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
            xs = np.linspace(x.min(), x.max(), 200)
            ax.scatter(x, y, alpha=0.3, s=40, color=cor_s, edgecolors='white', linewidths=1, label=f'{metrica} dados')
            ax.plot(xs, poly(xs), linewidth=3, color=cor_l, linestyle='-' if metrica == 'CTL' else '--',
                    label=f'{metrica} Poly{gk.replace("grau","")} (R²={r2:.3f})')
        ax.set_xlabel('Dias desde início', fontweight='bold'); ax.set_ylabel('Carga (TRIMP)', fontweight='bold')
        ax.set_title('CTL vs ATL Overall — Polynomial Fit', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Por modalidade
        tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
        if tipos_poli:
            st.markdown("**Por Modalidade (CTL/ATL separados)**")
            n_t = len(tipos_poli); ncols = 2; nrows = (n_t + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows))
            axes_flat = axes.flatten() if nrows > 1 else (axes if ncols == 1 else axes.flatten())
            for idx, (tipo_n, tipo_k) in enumerate(sorted(tipos_poli.items())):
                ax = axes_flat[idx]
                for metrica, cor, sty in [('CTL', CORES['azul'], '-'), ('ATL', CORES['vermelho'], '--')]:
                    if metrica not in poli[tipo_k]: continue
                    dados_m = poli[tipo_k][metrica]
                    gk = 'grau3' if 'grau3' in dados_m else 'grau2'
                    if gk not in dados_m: continue
                    d = dados_m[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
                    xs = np.linspace(x.min(), x.max(), 150)
                    ax.scatter(x, y, alpha=0.35, s=40, color=cor, edgecolors='white', linewidths=1)
                    ax.plot(xs, poly(xs), linewidth=2.5, color=cor, linestyle=sty,
                            label=f'{metrica} Poly{gk.replace("grau","")} R²={r2:.3f}')
                ax.set_title(f'{tipo_n} — CTL/ATL Polynomial', fontsize=11, fontweight='bold')
                ax.set_xlabel('Dias'); ax.set_ylabel('Carga'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            for idx in range(n_t, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            plt.suptitle('CTL/ATL por Modalidade — Polynomial Fit', fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Combinado (todos tipos no mesmo gráfico)
            st.markdown("**CTL e ATL — Todos os Tipos Combinados**")
            fig, axes2 = plt.subplots(1, 2, figsize=(18, 7))
            CORES_TIPO = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo']}
            for metrica, ax_m in zip(['CTL', 'ATL'], axes2):
                for tipo_n in sorted(tipos_poli):
                    tipo_k = tipos_poli[tipo_n]
                    if metrica not in poli[tipo_k]: continue
                    dados_m = poli[tipo_k][metrica]
                    gk = 'grau3' if 'grau3' in dados_m else 'grau2'
                    if gk not in dados_m: continue
                    d = dados_m[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
                    cor = CORES_TIPO.get(tipo_n, CORES['cinza'])
                    xs = np.linspace(x.min(), x.max(), 150)
                    ax_m.scatter(x, y, alpha=0.25, s=30, color=cor, edgecolors='white', linewidths=0.5)
                    ax_m.plot(xs, poly(xs), linewidth=2.5, color=cor, label=f'{tipo_n} R²={r2:.3f}')
                ax_m.set_title(f'{metrica} — Todos os Tipos', fontsize=12, fontweight='bold')
                ax_m.set_xlabel('Dias'); ax_m.set_ylabel(metrica)
                ax_m.legend(fontsize=10); ax_m.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Secção 4: BPE Heatmap (mapa de estados) ─────────────────────────────
    st.subheader("🗓️ BPE — Mapa de Estados Semanal (Heatmap)")
    if len(dw) < 14:
        st.info("Mínimo 14 dias de wellness necessários para BPE.")
    else:
        mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness']
                    if m in dw.columns and dw[m].notna().sum() >= 14]
        if not mets_bpe:
            st.info("Sem métricas wellness com dados suficientes para BPE.")
        else:
            n_sem_max = max(4, len(dw) // 7)
            n_sem_bpe = st.slider("Semanas BPE (heatmap)", 4, min(52, n_sem_max), min(16, n_sem_max), key="bpe_heat")
            dados_bpe = {}
            for met in mets_bpe:
                s = calcular_bpe(dw, met, 60)
                if len(s) > 0:
                    dados_bpe[met] = s.tail(n_sem_bpe)

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
                fig, ax = plt.subplots(figsize=(max(14, len(semanas) * 0.9), max(6, nm * 1.3)))
                im = ax.imshow(mat, cmap=cmap_bpe, aspect='auto', vmin=-2, vmax=2)
                ax.set_yticks(range(nm))
                ax.set_yticklabels([nomes_bpe.get(m, m) for m in dados_bpe], fontsize=11)
                sem_labels = [s.split('-W')[1] if '-W' in str(s) else str(s) for s in semanas]
                ax.set_xticks(range(len(semanas)))
                ax.set_xticklabels([f'S{s}' for s in sem_labels], rotation=45, fontsize=10)
                for i in range(nm):
                    for j in range(len(semanas)):
                        v = mat[i, j]
                        cor_txt = 'white' if abs(v) > 1 else 'black'
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=cor_txt)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Z-Score BPE (múltiplos de SWC)', fontsize=10)
                cbar.set_ticks([-2, -1, 0, 1, 2])
                cbar.set_ticklabels(['🔴 -2', '-1', '0', '+1', '🟢 +2'])
                ax.set_title('BPE — Blocos de Padrão Específico (Z-Score com SWC)', fontsize=13, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

                # BPE Timeline por métrica
                st.markdown("**BPE — Timeline por Métrica**")
                met_sel = st.selectbox("Métrica", list(dados_bpe.keys()), format_func=lambda m: nomes_bpe.get(m, m), key="bpe_timeline_met")
                df_tl_bpe = dados_bpe[met_sel]
                fig2, ax2 = plt.subplots(figsize=(14, 5))
                ax2.axhspan(1, 3, alpha=0.1, color=CORES['verde'], label='Acima (+1 SWC)')
                ax2.axhspan(-1, 1, alpha=0.1, color=CORES['amarelo'], label='Normal (±1 SWC)')
                ax2.axhspan(-3, -1, alpha=0.1, color=CORES['vermelho'], label='Abaixo (-1 SWC)')
                ax2.axhline(0, color='black', linewidth=1, linestyle='--')
                ax2.axhline(1, color=CORES['verde'], linewidth=1, linestyle=':')
                ax2.axhline(-1, color=CORES['vermelho'], linewidth=1, linestyle=':')
                z_vals = (-df_tl_bpe['zscore'].values if met_sel == 'rhr' else df_tl_bpe['zscore'].values)
                cores_z = [CORES['verde'] if v > 1 else CORES['vermelho'] if v < -1 else CORES['amarelo'] for v in z_vals]
                ax2.bar(range(len(df_tl_bpe)), z_vals, color=cores_z, alpha=0.8, edgecolor='white')
                ax2.plot(range(len(df_tl_bpe)), z_vals, 'o-', color='black', linewidth=1.5, markersize=5)
                for j, v in enumerate(z_vals):
                    ax2.text(j, v + (0.1 if v >= 0 else -0.15), f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8, fontweight='bold')
                ax2.set_xticks(range(len(df_tl_bpe)))
                ax2.set_xticklabels([f'S{s.split("-W")[1] if "-W" in str(s) else s}' for s in df_tl_bpe['ano_semana']], rotation=45, fontsize=10)
                ax2.set_ylabel('Z-Score BPE (SWC)', fontweight='bold')
                ax2.set_title(f'BPE Timeline — {nomes_bpe.get(met_sel, met_sel)}', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')
                plt.tight_layout(); st.pyplot(fig2); plt.close()

                # Resumo textual BPE
                with st.expander("📖 Interpretação BPE & Metodologia"):
                    st.markdown("""
**Fórmula BPE (Blocos de Padrão Específico):**
- **Baseline** = Média dos últimos 60 dias do período total
- **CV%** = (STD / Média) × 100 do baseline
- **SWC** = 0.5 × CV% × Baseline / 100 *(Smallest Worthwhile Change — Hopkins et al. 2009)*
- **Z-Score** = (Média_semanal − Baseline) / SWC

**Interpretação:**
| Z-Score | Estado |
|---|---|
| > +2 SWC | 🟢🟢 Muito Acima — Excelente recuperação |
| > +1 SWC | 🟢 Acima — Boa recuperação |
| ±1 SWC | 🟡 Normal — variação esperada |
| < -1 SWC | 🟠 Abaixo — atenção |
| < -2 SWC | 🔴 Muito Abaixo — alerta! |

> **Nota:** RHR é invertido (baixo = bom). O Z-Score BPE usa SWC (< STD) → detecta mudanças menores que o Z-Score tradicional.
                    """)

    st.markdown("---")

    # ── Secção 5: Análise de Falta de Estímulo ──────────────────────────────
    st.subheader("🎯 Análise de Falta de Estímulo por Modalidade")
    col_j1, col_j2 = st.columns(2)
    res_7 = analisar_falta_estimulo(da_full, 7)
    res_14 = analisar_falta_estimulo(da_full, 14)

    for janela_label, res in [("7 dias", res_7), ("14 dias", res_14)]:
        with (col_j1 if janela_label == "7 dias" else col_j2):
            st.markdown(f"**📅 Janela {janela_label}**")
            if res:
                rows_fe = []
                for mod, d in res.items():
                    prio_emoji = "🔴" if d['prioridade'] == 'ALTA' else "🟡" if d['prioridade'] == 'MÉDIA' else "🟢"
                    rows_fe.append({
                        'Modalidade': mod,
                        'Need Score': f"{d['need_score']:.1f}",
                        'Prioridade': f"{prio_emoji} {d['prioridade']}",
                        'Gap CTL-ATL': f"{d['gap_relativo']:.1f}%",
                        'Dias ATL<CTL': d['dias_atl_menor_ctl'],
                        'Dias c/ Atividade': d['dias_com_atividade']
                    })
                st.dataframe(pd.DataFrame(rows_fe), use_container_width=True, hide_index=True)
                top = list(res.keys())[0]
                prio_top = res[top]['prioridade']
                prio_e = "🔴" if prio_top == 'ALTA' else "🟡" if prio_top == 'MÉDIA' else "🟢"
                st.info(f"{prio_e} Foco recomendado ({janela_label}): **{top}** (Need Score: {res[top]['need_score']:.1f})")
            else:
                st.info("Dados insuficientes.")

    if res_7 and res_14:
        top7, top14 = list(res_7.keys())[0], list(res_14.keys())[0]
        if top7 == top14:
            st.success(f"✅ Foco consistente em **{top7}** nas duas janelas temporais.")
        else:
            st.warning(f"⚠️ Prioridade divergente: **{top7}** (7d, urgência) vs **{top14}** (14d, tendência)")

    st.markdown("---")

    # ── Secção 6: Saídas Textuais — Resumo Semanal ──────────────────────────
    st.subheader("📋 Resumo Geral (CTL/ATL/TSB + Atividades 7d)")
    if da_full is not None and len(da_full) > 0:
        df_s = filtrar_principais(da_full).copy()
        df_s['Data'] = pd.to_datetime(df_s['Data'])
        if 'moving_time' in df_s.columns and 'rpe' in df_s.columns:
            df_s['rpe_fill'] = df_s['rpe'].fillna(df_s['rpe'].median())
            df_s['load_val'] = (df_s['moving_time'] / 60) * df_s['rpe_fill']
            ld_s = df_s.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
            idx_s = pd.date_range(ld_s['Data'].min(), datetime.now().date())
            ld_s = ld_s.set_index('Data').reindex(idx_s, fill_value=0).reset_index(); ld_s.columns = ['Data', 'load_val']
            ld_s['CTL'] = ld_s['load_val'].ewm(span=42, adjust=False).mean()
            ld_s['ATL'] = ld_s['load_val'].ewm(span=7, adjust=False).mean()
            u_s = ld_s.iloc[-1]
            df_7d = df_s[df_s['Data'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
            horas_7d = pd.to_numeric(df_7d['moving_time'], errors='coerce').sum() / 3600
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("CTL (Fitness)", f"{u_s['CTL']:.0f}")
            col2.metric("ATL (Fadiga)", f"{u_s['ATL']:.0f}")
            col3.metric("TSB (Forma)", f"{u_s['CTL']-u_s['ATL']:+.0f}")
            col4.metric("Atividades 7d", len(df_7d))
            col5.metric("Horas 7d", f"{horas_7d:.1f}h")
    if dw is not None and len(dw) > 0:
        hrv_7 = pd.to_numeric(dw['hrv'], errors='coerce').dropna().tail(7).mean() if 'hrv' in dw.columns else None
        rhr_u = pd.to_numeric(dw['rhr'], errors='coerce').dropna().iloc[-1] if 'rhr' in dw.columns and len(dw) > 0 else None
        col1w, col2w = st.columns(2)
        if hrv_7: col1w.metric("HRV médio (7d)", f"{hrv_7:.0f} ms")
        if rhr_u: col2w.metric("RHR último", f"{rhr_u:.0f} bpm")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
