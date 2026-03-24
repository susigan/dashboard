# ════════════════════════════════════════════════════════════════════════════════
# tab_correlacoes.py — Correlações
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

def tab_correlacoes(da, dw):
    st.header("🧠 Correlações & Impacto RPE")
    if len(da) == 0 or len(dw) == 0: st.warning("Sem dados suficientes."); return
    rpe_col = next((c for c in ['rpe', 'RPE', 'icu_rpe'] if c in da.columns), None)

    st.subheader("💚 Impacto do RPE no HRV/RHR (dia seguinte)")
    if rpe_col and 'hrv' in dw.columns:
        da2 = filtrar_principais(da).copy(); da2['Data'] = pd.to_datetime(da2['Data']).dt.normalize()
        dw2 = dw.copy(); dw2['Data'] = pd.to_datetime(dw2['Data']).dt.normalize()
        rpe_daily = da2.groupby('Data')[rpe_col].mean().reset_index(); rpe_daily.columns = ['Data', 'rpe_avg']
        rpe_daily['rpe_cat'] = rpe_daily['rpe_avg'].apply(classificar_rpe)
        dw2_shift = dw2[['Data', 'hrv'] + (['rhr'] if 'rhr' in dw2.columns else [])].copy()
        dw2_shift['Data_prev'] = dw2_shift['Data'] - pd.Timedelta(days=1)
        merged = rpe_daily.merge(dw2_shift.rename(columns={'Data': 'Data_hrv', 'Data_prev': 'Data'}), on='Data', how='inner')
        merged = merged.dropna(subset=['hrv', 'rpe_cat'])
        if len(merged) >= 5:
            cats = ['Leve', 'Moderado', 'Pesado']; baseline_hrv = merged['hrv'].mean()
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 4))
                medias_hrv = {c: merged[merged['rpe_cat'] == c]['hrv'].mean() for c in cats if c in merged['rpe_cat'].values}
                vals = [((medias_hrv.get(c, baseline_hrv) - baseline_hrv) / baseline_hrv * 100) for c in cats]
                ax.bar(cats, vals, color=[CORES['verde'] if v > 0 else CORES['vermelho'] for v in vals], alpha=0.8, edgecolor='white')
                ax.axhline(0, color=CORES['cinza'], linestyle='--'); ax.set_title('Δ HRV% (dia+1) por RPE', fontweight='bold')
                for i, v in enumerate(vals): ax.text(i, v + (1 if v >= 0 else -1.5), f'{v:+.1f}%', ha='center', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                if 'rhr' in merged.columns:
                    fig, ax = plt.subplots(figsize=(7, 4)); baseline_rhr = merged['rhr'].mean()
                    medias_rhr = {c: merged[merged['rpe_cat'] == c]['rhr'].mean() for c in cats if c in merged['rpe_cat'].values}
                    vals_r = [medias_rhr.get(c, baseline_rhr) - baseline_rhr for c in cats]
                    ax.bar(cats, vals_r, color=[CORES['vermelho'] if v > 0 else CORES['verde'] for v in vals_r], alpha=0.8, edgecolor='white')
                    ax.axhline(0, color=CORES['cinza'], linestyle='--'); ax.set_title('Δ RHR (bpm, dia+1) por RPE', fontweight='bold')
                    for i, v in enumerate(vals_r): ax.text(i, v + (0.3 if v >= 0 else -0.6), f'{v:+.1f}', ha='center', fontsize=9, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); st.pyplot(fig); plt.close()
            tabela = []
            for cat in cats:
                sub = merged[merged['rpe_cat'] == cat]
                if len(sub) > 0:
                    dhrv = ((sub['hrv'].mean() - baseline_hrv) / baseline_hrv * 100)
                    drhr = (sub['rhr'].mean() - (merged['rhr'].mean() if 'rhr' in merged.columns else 0)) if 'rhr' in sub.columns else 0
                    tabela.append({'Categoria': cat, 'Δ HRV (%)': f'{dhrv:+.1f}%', 'Δ RHR (bpm)': f'{drhr:+.1f}', 'n': len(sub)})
            if tabela: st.dataframe(pd.DataFrame(tabela), use_container_width=True, hide_index=True)

    st.subheader("🔍 Scatter: RPE → HRV | HRV → RHR")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if rpe_col and 'hrv' in dw.columns and 'merged' in dir() and len(merged) > 5:
        m_clean = merged[['rpe_avg', 'hrv']].dropna()
        if len(m_clean) > 5:
            ax1.scatter(m_clean['rpe_avg'], m_clean['hrv'], c=CORES['azul'], alpha=0.5, s=30)
            z = np.polyfit(m_clean['rpe_avg'], m_clean['hrv'], 1); xl = np.linspace(m_clean['rpe_avg'].min(), m_clean['rpe_avg'].max(), 100)
            ax1.plot(xl, np.poly1d(z)(xl), '--', color=CORES['vermelho'])
            r, _ = pearsonr(m_clean['rpe_avg'], m_clean['hrv'])
            ax1.text(0.05, 0.95, f'r = {r:.3f}', transform=ax1.transAxes, fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.set_xlabel('RPE'); ax1.set_ylabel('HRV (dia+1)'); ax1.set_title('RPE → HRV', fontweight='bold'); ax1.grid(True, alpha=0.3)
    if 'hrv' in dw.columns and 'rhr' in dw.columns:
        dw3 = dw.dropna(subset=['hrv', 'rhr'])
        if len(dw3) > 5:
            ax2.scatter(dw3['hrv'], dw3['rhr'], c=CORES['roxo'], alpha=0.5, s=30)
            z2 = np.polyfit(dw3['hrv'], dw3['rhr'], 1); xl2 = np.linspace(dw3['hrv'].min(), dw3['hrv'].max(), 100)
            ax2.plot(xl2, np.poly1d(z2)(xl2), '--', color=CORES['vermelho'])
            r2 = dw3['hrv'].corr(dw3['rhr'])
            ax2.text(0.05, 0.95, f'r = {r2:.3f}', transform=ax2.transAxes, fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.set_xlabel('HRV (ms)'); ax2.set_ylabel('RHR (bpm)'); ax2.set_title('HRV vs RHR', fontweight='bold'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 Correlações entre Métricas Wellness")
    mets_num = [c for c in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness'] if c in dw.columns and dw[c].notna().any()]
    if len(mets_num) >= 3:
        corr_mat = dw[mets_num].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlações Wellness (Pearson)', fontsize=13, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("🏃 Impacto por Tipo de Atividade → HRV/RHR (dia+1)")
    if rpe_col and 'type' in da.columns and 'hrv' in dw.columns:
        da3 = filtrar_principais(da).copy(); da3['Data'] = pd.to_datetime(da3['Data']).dt.normalize()
        tipo_daily = da3.groupby('Data')['type'].agg(lambda x: x.mode()[0] if len(x) > 0 else None).reset_index()
        dw2b = dw.copy(); dw2b['Data'] = pd.to_datetime(dw2b['Data']).dt.normalize()
        dw2b_s = dw2b[['Data', 'hrv'] + (['rhr'] if 'rhr' in dw2b.columns else [])].copy()
        dw2b_s['Data_prev'] = dw2b_s['Data'] - pd.Timedelta(days=1)
        merged2 = tipo_daily.merge(dw2b_s.rename(columns={'Data': 'Data_hrv', 'Data_prev': 'Data'}), on='Data', how='inner')
        merged2 = merged2.dropna(subset=['hrv', 'type'])
        if len(merged2) >= 5:
            tipos_disp = [t for t in ['Bike', 'Row', 'Run', 'Ski'] if t in merged2['type'].values]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            baseline_h = merged2['hrv'].mean()
            vals_h = [((merged2[merged2['type'] == t]['hrv'].mean() - baseline_h) / baseline_h * 100) for t in tipos_disp]
            ax1.bar(tipos_disp, vals_h, color=[get_cor(t) for t in tipos_disp], alpha=0.8, edgecolor='white')
            ax1.axhline(0, color=CORES['cinza'], linestyle='--'); ax1.set_title('Atividade → HRV (dia+1)', fontweight='bold')
            ax1.set_ylabel('Δ HRV (%)'); ax1.grid(True, alpha=0.3, axis='y')
            if 'rhr' in merged2.columns:
                baseline_r = merged2['rhr'].mean()
                vals_r = [merged2[merged2['type'] == t]['rhr'].mean() - baseline_r for t in tipos_disp]
                ax2.bar(tipos_disp, vals_r, color=[get_cor(t) for t in tipos_disp], alpha=0.8, edgecolor='white')
                ax2.axhline(0, color=CORES['cinza'], linestyle='--'); ax2.set_title('Atividade → RHR (dia+1)', fontweight='bold')
                ax2.set_ylabel('Δ RHR (bpm)'); ax2.grid(True, alpha=0.3, axis='y')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_recovery.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════════
