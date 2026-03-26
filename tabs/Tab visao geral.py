# tabs/tab_visao_geral.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor

def tab_visao_geral(dw, da, di, df_):
    st.header("📊 Visão Geral")
    c1, c2, c3, c4 = st.columns(4)
    horas = (da['moving_time'].sum() / 3600) if 'moving_time' in da.columns and len(da) > 0 else None
    hrv_m = dw['hrv'].dropna().tail(7).mean() if 'hrv' in dw.columns and len(dw) > 0 else None
    rhr_u = dw['rhr'].dropna().iloc[-1] if 'rhr' in dw.columns and len(dw) > 0 and dw['rhr'].notna().any() else None
    c1.metric("🏋️ Sessões", f"{len(da)}")
    c2.metric("⏱️ Horas", f"{horas:.1f}h" if horas else "—")
    c3.metric("💚 HRV (7d)", f"{hrv_m:.0f} ms" if hrv_m else "—")
    c4.metric("❤️ RHR", f"{rhr_u:.0f} bpm" if rhr_u else "—")
    st.markdown("---")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("📈 Performance Overview")
        fig, ax = plt.subplots(figsize=(13, 4))
        if 'moving_time' in da.columns and 'rpe' in da.columns and len(da) > 0:
            dl = da.copy(); dl['Data'] = pd.to_datetime(dl['Data'])
            dl['load'] = (dl['moving_time'] / 60) * dl['rpe'].fillna(0)
            ld = dl.groupby('Data')['load'].sum().reset_index().sort_values('Data')
            ax.bar(ld['Data'], norm_serie(ld['load']), color=CORES['cinza'], alpha=0.3, label='Load (norm)', width=0.8)
        if 'hrv' in dw.columns and len(dw) > 0:
            dw2 = dw.dropna(subset=['hrv']).copy(); dw2['Data'] = pd.to_datetime(dw2['Data'])
            ax.plot(dw2['Data'], norm_serie(dw2['hrv']), color=CORES['verde'], linewidth=2, linestyle='--', label='HRV (norm)')
        ax.set_title('Performance Overview', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9); ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col_r:
        st.subheader("🎯 Distribuição")
        df_d = filtrar_principais(da).copy()
        if len(df_d) > 0:
            cnt = df_d['type'].apply(norm_tipo).value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(cnt.values, labels=cnt.index, autopct='%1.0f%%',
                    colors=[get_cor(t) for t in cnt.index], startangle=90,
                    pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
            ax2.text(0, 0, f'{cnt.sum()}', fontsize=28, fontweight='bold', ha='center', va='center')
            plt.tight_layout(); st.pyplot(fig2); plt.close()
    st.markdown("---")
    st.subheader("📋 Atividades Recentes")
    df_tab = filtrar_principais(da).sort_values('Data', ascending=False).head(10)
    if len(df_tab) > 0:
        cs = [c for c in ['Data', 'type', 'name', 'moving_time', 'rpe', 'power_avg', 'icu_eftp'] if c in df_tab.columns]
        ds = df_tab[cs].copy()
        if 'moving_time' in ds.columns:
            ds['moving_time'] = ds['moving_time'].apply(lambda x: f"{int(x/3600)}h{int((x%3600)/60):02d}m" if pd.notna(x) else '—')
        ds.columns = [c.replace('_', ' ').title() for c in ds.columns]
        st.dataframe(ds, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("📋 Resumo Semanal")
    col1, col2, col3 = st.columns(3)
    if len(da) > 0:
        dw7 = da[pd.to_datetime(da['Data']).dt.date >= (datetime.now().date() - timedelta(days=7))]
        col1.metric("Sessões (7d)", len(dw7))
        if 'moving_time' in dw7.columns: col2.metric("Horas (7d)", f"{dw7['moving_time'].sum()/3600:.1f}h")
        if 'rpe' in dw7.columns and dw7['rpe'].notna().any(): col3.metric("RPE médio (7d)", f"{dw7['rpe'].mean():.1f}")
    df_rank = filtrar_principais(da).copy()
    if 'power_avg' in df_rank.columns and df_rank['power_avg'].notna().any():
        st.subheader("🏆 Top 10 por Potência")
        top = df_rank.nlargest(10, 'power_avg')[['Data', 'type', 'name', 'power_avg', 'rpe']].copy()
        top['Data'] = pd.to_datetime(top['Data']).dt.strftime('%Y-%m-%d')
        top.columns = ['Data', 'Tipo', 'Nome', 'Power (W)', 'RPE']
        st.dataframe(top, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC + FTLM
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_pmc.py
# ════════════════════════════════════════════════════════════════════════════
