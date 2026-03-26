# tabs/tab_volume.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor

def tab_volume(da, dw):
    st.header("📦 Volume & Carga")
    if len(da) == 0: st.warning("Sem dados de atividades."); return
    df = filtrar_principais(da).copy()
    df = add_tempo(df); df['Data'] = pd.to_datetime(df['Data'])
    df['horas'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 3600).fillna(0)
    ciclicos = ['Bike', 'Run', 'Row', 'Ski']
    CORES_MOD = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo'], 'WeightTraining': CORES['laranja']}

    st.subheader("🚴 Volume Mensal — Atividades Cíclicas (horas)")
    df_cic = df[df['type'].isin(ciclicos)].copy()
    if len(df_cic) > 0:
        pivot = df_cic.pivot_table(index='mes', columns='type', values='horas', aggfunc='sum', fill_value=0).sort_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        bottom = np.zeros(len(pivot))
        for tipo in [t for t in ciclicos if t in pivot.columns]:
            vals = pivot[tipo].values
            ax.bar(range(len(pivot)), vals, bottom=bottom, label=tipo, color=CORES_MOD.get(tipo, 'gray'), alpha=0.85, edgecolor='white')
            bottom += vals
        totais = pivot.sum(axis=1).values
        for i, t in enumerate(totais):
            if t > 0: ax.text(i, t + 0.1, f'{t:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(pivot))); ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        media = totais.mean(); ax.axhline(media, color='black', linestyle='--', alpha=0.5, label=f'Média: {media:.1f}h')
        ax.set_ylabel('Horas', fontweight='bold'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        c1, c2 = st.columns(2)
        c1.metric("Total horas cíclicos", f"{pivot.values.sum():.1f}h")
        c2.metric("Média mensal", f"{media:.1f}h")

    st.subheader("🏋️ Volume Mensal — WeightTraining (horas)")
    df_wt = da[da['type'] == 'WeightTraining'].copy()
    if len(df_wt) > 0:
        df_wt = add_tempo(df_wt); df_wt['horas'] = (pd.to_numeric(df_wt['moving_time'], errors='coerce') / 3600).fillna(0)
        mensal = df_wt.groupby('mes').agg(horas=('horas', 'sum'), sessoes=('Data', 'count')).reset_index().sort_values('mes')
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(mensal)), mensal['horas'], color=CORES['laranja'], alpha=0.8, edgecolor='white')
        for i, (h, s) in enumerate(zip(mensal['horas'], mensal['sessoes'])):
            if h > 0: ax.text(i, h + 0.05, f'{h:.1f}h\n({s}x)', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(mensal))); ax.set_xticklabels(mensal['mes'], rotation=45, ha='right')
        media_wt = mensal['horas'].mean()
        ax.axhline(media_wt, color='red', linestyle='--', alpha=0.7, label=f'Média: {media_wt:.1f}h')
        ax.set_ylabel('Horas', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Sem sessões de WeightTraining no período.")

    st.subheader("💥 Strain Score (XSS)")
    xss_col = next((c for c in ['xss', 'SS', 'XSS'] if c in df.columns and df[c].notna().any()), None)
    if xss_col:
        df_xss = df[df['type'].isin(ciclicos)].dropna(subset=[xss_col]).copy()
        if len(df_xss) > 3:
            df_xss = df_xss.sort_values('Data'); df_xss['xss_s'] = pd.to_numeric(df_xss[xss_col], errors='coerce').rolling(7, min_periods=1).mean()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(df_xss['Data'], pd.to_numeric(df_xss[xss_col], errors='coerce'), alpha=0.4, label='XSS')
            ax1.plot(df_xss['Data'], df_xss['xss_s'], linewidth=2.5, label='XSS 7d')
            ax1.set_title('Evolução XSS', fontweight='bold'); ax1.legend(); ax1.tick_params(axis='x', rotation=45)
            comp = [c for c in ['glycolytic', 'aerobic', 'pmax'] if c in df_xss.columns]
            if comp:
                med = df_xss.groupby('type')[comp].mean().fillna(0)
                med.plot(kind='bar', stacked=True, ax=ax2, color=[CORES['vermelho'], CORES['verde'], CORES['laranja']][:len(comp)])
                ax2.set_title('Componentes por Tipo', fontweight='bold'); ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 Volume de Horas por Intensidade (Trimestral)")
    if 'rpe' in df.columns and 'moving_time' in df.columns:
        df_rpe = df[df['type'].isin(ciclicos)].copy()
        df_rpe['rpe_cat'] = df_rpe['rpe'].apply(classificar_rpe); df_rpe = df_rpe.dropna(subset=['rpe_cat'])
        if len(df_rpe) > 0:
            piv = df_rpe.pivot_table(index='trimestre', columns='rpe_cat', values='horas', aggfunc='sum', fill_value=0).sort_index()
            CORES_RPE = {'Leve': CORES['verde'], 'Moderado': CORES['laranja'], 'Pesado': CORES['vermelho']}
            fig, ax = plt.subplots(figsize=(13, 5))
            bottom = np.zeros(len(piv))
            for cat in ['Leve', 'Moderado', 'Pesado']:
                if cat in piv.columns:
                    vals = piv[cat].values
                    ax.bar(range(len(piv)), vals, bottom=bottom, label=cat, color=CORES_RPE.get(cat, 'gray'), alpha=0.85, edgecolor='white')
                    for i, (v, b) in enumerate(zip(vals, bottom)):
                        if v > 0.5: ax.text(i, b + v / 2, f'{v:.1f}h', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                    bottom += vals
            ax.set_xticks(range(len(piv))); ax.set_xticklabels(piv.index, rotation=45, ha='right')
            ax.set_ylabel('Horas', fontweight='bold'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, axis='y')
            ax.set_title('Volume de Horas por Intensidade RPE (Trimestral)', fontsize=12, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — eFTP
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_eftp.py
# ════════════════════════════════════════════════════════════════════════════
