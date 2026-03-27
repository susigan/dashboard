# tabs/tab_visao_geral.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor, norm_tipo, norm_serie

def tab_visao_geral(dw, da, di, df_):
    st.header("📊 Visão Geral")

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    horas = (da['moving_time'].sum() / 3600) if 'moving_time' in da.columns and len(da) > 0 else None
    hrv_m = dw['hrv'].dropna().tail(7).mean() if 'hrv' in dw.columns and len(dw) > 0 else None
    rhr_u = dw['rhr'].dropna().iloc[-1] if 'rhr' in dw.columns and len(dw) > 0 and dw['rhr'].notna().any() else None
    c1.metric("🏋️ Sessões",   f"{len(da)}")
    c2.metric("⏱️ Horas",     f"{horas:.1f}h" if horas else "—")
    c3.metric("💚 HRV (7d)", f"{hrv_m:.0f} ms" if hrv_m else "—")
    c4.metric("❤️ RHR",       f"{rhr_u:.0f} bpm" if rhr_u else "—")
    st.markdown("---")

    # ── Performance Overview + pizza Sessões ──
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("📈 Performance Overview")
        fig, ax = plt.subplots(figsize=(13, 4))
        if 'moving_time' in da.columns and 'rpe' in da.columns and len(da) > 0:
            dl = da.copy(); dl['Data'] = pd.to_datetime(dl['Data'])
            dl['load'] = (dl['moving_time'] / 60) * dl['rpe'].fillna(0)
            ld = dl.groupby('Data')['load'].sum().reset_index().sort_values('Data')
            ax.bar(ld['Data'], norm_serie(ld['load']), color=CORES['cinza'],
                   alpha=0.3, label='Load (norm)', width=0.8)
        if 'hrv' in dw.columns and len(dw) > 0:
            dw2 = dw.dropna(subset=['hrv']).copy()
            dw2['Data'] = pd.to_datetime(dw2['Data'])
            ax.plot(dw2['Data'], norm_serie(dw2['hrv']),
                    color=CORES['verde'], linewidth=2, linestyle='--', label='HRV (norm)')
        ax.set_title('Performance Overview', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("🎯 Sessões")
        df_pie = filtrar_principais(da).copy()
        # Excluir WeightTraining de todos os pizzas
        df_pie = df_pie[df_pie['type'].apply(norm_tipo) != 'WeightTraining']
        if len(df_pie) > 0:
            cnt = df_pie['type'].apply(norm_tipo).value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(cnt.values, labels=cnt.index, autopct='%1.0f%%',
                    colors=[get_cor(t) for t in cnt.index], startangle=90,
                    pctdistance=0.75,
                    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
            ax2.text(0, 0, f'{cnt.sum()}', fontsize=22, fontweight='bold',
                     ha='center', va='center')
            ax2.set_title('Sessões (excl. WT)', fontsize=9, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")

    # ── Pizzas: Horas | KM | RPE ──
    st.subheader("🎯 Distribuição por Horas, KM e RPE")
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        if 'moving_time' in df_pie.columns and df_pie['moving_time'].notna().any():
            hrs_t = (df_pie.groupby(df_pie['type'].apply(norm_tipo))['moving_time']
                     .sum() / 3600)
            hrs_t = hrs_t[hrs_t > 0]
            if len(hrs_t) > 0:
                fig_h, ax_h = plt.subplots(figsize=(5, 5))
                ax_h.pie(hrs_t.values, labels=hrs_t.index, autopct='%1.0f%%',
                         colors=[get_cor(t) for t in hrs_t.index], startangle=90,
                         pctdistance=0.75,
                         wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
                ax_h.text(0, 0, f'{hrs_t.sum():.0f}h', fontsize=18,
                          fontweight='bold', ha='center', va='center')
                ax_h.set_title('Horas', fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_h); plt.close()

    with pc2:
        if 'distance' in df_pie.columns and df_pie['distance'].notna().any():
            df_kmt = df_pie.copy()
            df_kmt['_t'] = df_kmt['type'].apply(norm_tipo)
            km_t = df_kmt.groupby('_t')['distance'].sum() / 1000
            km_t = km_t[km_t > 0]
            if len(km_t) > 0:
                fig_k, ax_k = plt.subplots(figsize=(5, 5))
                ax_k.pie(km_t.values, labels=km_t.index, autopct='%1.0f%%',
                         colors=[get_cor(t) for t in km_t.index], startangle=90,
                         pctdistance=0.75,
                         wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
                ax_k.text(0, 0, str(int(km_t.sum())) + ' km', fontsize=14,
                          fontweight='bold', ha='center', va='center')
                ax_k.set_title('KM', fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_k); plt.close()

    with pc3:
        if 'rpe' in df_pie.columns and df_pie['rpe'].notna().any():
            df_rpe = df_pie.dropna(subset=['rpe']).copy()
            df_rpe['rpe_n'] = pd.to_numeric(df_rpe['rpe'], errors='coerce')
            df_rpe = df_rpe.dropna(subset=['rpe_n'])
            df_rpe['rpe_cat'] = pd.cut(df_rpe['rpe_n'],
                                        bins=[0, 4.9, 6.9, 10],
                                        labels=['Leve (1-5)', 'Moderado (5-7)', 'Forte (7-10)'])
            df_rpe = df_rpe.dropna(subset=['rpe_cat'])
            rpe_cnt = df_rpe['rpe_cat'].value_counts().sort_index()
            if len(rpe_cnt) > 0:
                rpe_cores = {'Leve (1-5)': CORES['verde'],
                             'Moderado (5-7)': CORES['laranja'],
                             'Forte (7-10)': CORES['vermelho']}
                fig_r, ax_r = plt.subplots(figsize=(5, 5))
                ax_r.pie(rpe_cnt.values, labels=rpe_cnt.index, autopct='%1.0f%%',
                         colors=[rpe_cores.get(str(l), CORES['cinza'])
                                 for l in rpe_cnt.index],
                         startangle=90, pctdistance=0.75,
                         wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
                ax_r.text(0, 0, f'{rpe_cnt.sum()}', fontsize=18,
                          fontweight='bold', ha='center', va='center')
                ax_r.set_title('RPE Geral', fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_r); plt.close()

    st.markdown("---")

    # ── Tabela % KM por modalidade ──
    st.subheader("📏 Distribuição de KM por modalidade")
    if 'distance' in da.columns and da['distance'].notna().any():
        _agr_col, _ = st.columns([1, 3])
        _agr = _agr_col.selectbox("Agrupar por", ["Semana", "Mês", "Ano"],
                                   key="vg_agrup_km")
        _code = {"Semana": "W", "Mês": "M", "Ano": "A"}[_agr]

        df_km = da.copy()
        df_km['_t'] = df_km['type'].apply(norm_tipo)
        df_km = df_km[df_km['_t'] != 'WeightTraining']
        df_km['km'] = pd.to_numeric(df_km['distance'], errors='coerce') / 1000
        df_km = df_km[df_km['km'].notna() & (df_km['km'] > 0)]
        df_km['Data'] = pd.to_datetime(df_km['Data'])
        df_km['_p'] = df_km['Data'].dt.to_period(_code)

        tipos_km = [t for t in ['Bike', 'Row', 'Ski', 'Run']
                    if t in df_km['_t'].unique()]

        if len(df_km) > 0 and tipos_km:
            piv = (df_km.groupby(['_p', '_t'])['km'].sum()
                   .unstack(fill_value=0)
                   .reindex(columns=tipos_km, fill_value=0))
            piv['Total'] = piv[tipos_km].sum(axis=1)

            rows_km = []
            for p, r in piv.sort_index(ascending=False).iterrows():
                if _agr == "Ano":    lbl = str(p.year)
                elif _agr == "Semana": lbl = p.start_time.strftime('%d/%m/%y')
                else: lbl = pd.to_datetime(str(p)).strftime('%B %Y').title()
                row = {'Período': lbl}
                tot = r['Total']
                for t in tipos_km:
                    v = r[t]
                    pct = (v / tot * 100) if tot > 0 else 0
                    row[t] = f"{v:.0f} km ({pct:.0f}%)" if v > 0 else '—'
                row['Total'] = f"{tot:.0f} km"
                rows_km.append(row)

            if rows_km:
                st.dataframe(pd.DataFrame(rows_km),
                             use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Atividades Recentes ──
    st.subheader("📋 Atividades Recentes")
    df_tab = filtrar_principais(da).sort_values('Data', ascending=False).head(10)
    if len(df_tab) > 0:
        cs = [c for c in ['Data', 'type', 'name', 'moving_time',
                           'rpe', 'power_avg', 'icu_eftp'] if c in df_tab.columns]
        ds = df_tab[cs].copy()
        if 'moving_time' in ds.columns:
            ds['moving_time'] = ds['moving_time'].apply(
                lambda x: f"{int(x/3600)}h{int((x%3600)/60):02d}m"
                if pd.notna(x) else '—')
        ds.columns = [c.replace('_', ' ').title() for c in ds.columns]
        st.dataframe(ds, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Resumo Semanal ──
    st.subheader("📋 Resumo Semanal")
    col1, col2, col3 = st.columns(3)
    if len(da) > 0:
        dw7 = da[pd.to_datetime(da['Data']).dt.date >=
                 (datetime.now().date() - timedelta(days=7))]
        col1.metric("Sessões (7d)", len(dw7))
        if 'moving_time' in dw7.columns:
            col2.metric("Horas (7d)", f"{dw7['moving_time'].sum()/3600:.1f}h")
        if 'rpe' in dw7.columns and dw7['rpe'].notna().any():
            col3.metric("RPE médio (7d)", f"{dw7['rpe'].mean():.1f}")

    # ── Top 10 por Potência ──
    df_rank = filtrar_principais(da).copy()
    if 'power_avg' in df_rank.columns and df_rank['power_avg'].notna().any():
        st.subheader("🏆 Top 10 por Potência")
        top = df_rank.nlargest(10, 'power_avg')[
            ['Data', 'type', 'name', 'power_avg', 'rpe']].copy()
        top['Data'] = pd.to_datetime(top['Data']).dt.strftime('%Y-%m-%d')
        top.columns = ['Data', 'Tipo', 'Nome', 'Power (W)', 'RPE']
        st.dataframe(top, use_container_width=True, hide_index=True)
