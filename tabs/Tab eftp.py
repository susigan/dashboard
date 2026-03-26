# tabs/tab_eftp.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor, conv_15

def tab_eftp(da, mods_sel):
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — HR ZONES + RPE ZONES + CORRELAÇÃO
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_zones.py
# ════════════════════════════════════════════════════════════════════════════
