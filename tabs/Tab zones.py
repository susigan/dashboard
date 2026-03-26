# tabs/tab_zones.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor

def _zonas_bp(ax_bar, ax_pie, por_tipo, total_seg, cores_zona, tit_bar, tit_pie):
    zonas = list(cores_zona.keys()); bottom = np.zeros(len(por_tipo))
    for zona in zonas:
        if zona not in por_tipo.columns: continue
        vals = por_tipo[zona].values
        ax_bar.bar(por_tipo.index, vals, bottom=bottom, color=cores_zona[zona], label=zona, edgecolor='white', linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5: ax_bar.text(i, b + v / 2, f'{v:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        bottom += vals
    ax_bar.set_ylim(0, 100); ax_bar.set_ylabel('% sessões', fontweight='bold')
    ax_bar.set_title(tit_bar, fontweight='bold'); ax_bar.legend(loc='upper right', fontsize=8); ax_bar.grid(True, alpha=0.2, axis='y')
    lp = [l for l, v in total_seg.items() if v > 0]; sp = [v for v in total_seg.values() if v > 0]
    if sum(sp) > 0:
        _, _, ats = ax_pie.pie(sp, labels=lp, autopct='%1.1f%%', colors=[cores_zona[l] for l in lp], startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2), pctdistance=0.75)
        for at in ats: at.set_fontweight('bold'); at.set_fontsize(10)
    ax_pie.set_title(tit_pie, fontweight='bold')

def tab_zones(da, mods_sel):
    st.header("❤️ HR Zones & RPE Zones")
    df = filtrar_principais(da).copy(); df['Data'] = pd.to_datetime(df['Data']); df['ano'] = df['Data'].dt.year
    df = df[df['type'].isin(mods_sel)]
    if len(df) == 0: st.warning("Sem dados."); return
    anos = sorted(df['ano'].unique()); ano_sel = st.selectbox("Ano", anos, index=len(anos) - 1)
    df_ano = df[df['ano'] == ano_sel].copy()
    zonas_hr = [c for c in df.columns if c.lower().startswith('hr_z') and c.lower().endswith('_secs')]
    rpe_col = next((c for c in ['rpe', 'RPE', 'icu_rpe'] if c in df.columns), None)
    def gzn(col): m = re.search(r'hr_z(\d+)_secs', col.lower()); return int(m.group(1)) if m else 0
    CORES_HR = {'Baixa (Z1+Z2)': '#2ECC71', 'Moderada (Z3+Z4)': '#F39C12', 'Alta (Z5+Z6+Z7)': '#E74C3C'}
    CORES_RPE = {'Leve (1–4)': '#3498DB', 'Moderado (5–6)': '#F39C12', 'Forte (7–10)': '#E74C3C'}
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    if zonas_hr:
        dh = df_ano.copy()
        bc = [c for c in zonas_hr if gzn(c) in (1, 2)]; mc = [c for c in zonas_hr if gzn(c) in (3, 4)]; ac = [c for c in zonas_hr if gzn(c) in (5, 6, 7)]
        for cols in [bc, mc, ac]:
            for c in cols: dh[c] = pd.to_numeric(dh[c], errors='coerce').fillna(0)
        dh['zb'] = dh[bc].sum(axis=1) if bc else 0; dh['zm'] = dh[mc].sum(axis=1) if mc else 0; dh['za'] = dh[ac].sum(axis=1) if ac else 0
        dh['zt'] = dh['zb'] + dh['zm'] + dh['za']; dh = dh[dh['zt'] > 0]
        if len(dh) > 0:
            for z, p in [('zb', 'pb'), ('zm', 'pm'), ('za', 'pa')]: dh[p] = dh[z] / dh['zt'] * 100
            pt = dh.groupby('type')[['pb', 'pm', 'pa']].mean(); pt.columns = list(CORES_HR.keys())
            ts = {'Baixa (Z1+Z2)': dh['zb'].sum(), 'Moderada (Z3+Z4)': dh['zm'].sum(), 'Alta (Z5+Z6+Z7)': dh['za'].sum()}
            _zonas_bp(axes[0], axes[1], pt, ts, CORES_HR, f'❤️ HR Zones — {ano_sel}', f'❤️ HR Geral — {ano_sel}')
    if rpe_col:
        dr = df_ano.dropna(subset=[rpe_col]).copy(); dr[rpe_col] = pd.to_numeric(dr[rpe_col], errors='coerce')
        dr['rz'] = pd.cut(dr[rpe_col], bins=[0, 4.9, 6.9, 10], labels=list(CORES_RPE.keys()), right=True); dr = dr.dropna(subset=['rz'])
        if len(dr) > 0:
            piv = dr.groupby(['type', 'rz'], observed=True).size().unstack(fill_value=0)
            for z in CORES_RPE.keys():
                if z not in piv.columns: piv[z] = 0
            piv = piv[list(CORES_RPE.keys())]; pct = piv.div(piv.sum(axis=1), axis=0) * 100
            tr = {z: piv[z].sum() for z in CORES_RPE.keys()}
            _zonas_bp(axes[2], axes[3], pct, tr, CORES_RPE, f'🎯 RPE Zones — {ano_sel}', f'🎯 RPE Geral — {ano_sel}')
    plt.suptitle(f'HR Zones · RPE Zones — {ano_sel}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("🔗 Correlação HR Zones × RPE")
    if zonas_hr and rpe_col:
        df_ok = df.copy()
        for c in [c for c in zonas_hr]: df_ok[c] = pd.to_numeric(df_ok[c], errors='coerce').fillna(0)
        df_ok['zb'] = df_ok[[c for c in zonas_hr if gzn(c) in (1, 2)]].sum(axis=1)
        df_ok['zm'] = df_ok[[c for c in zonas_hr if gzn(c) in (3, 4)]].sum(axis=1)
        df_ok['za'] = df_ok[[c for c in zonas_hr if gzn(c) in (5, 6, 7)]].sum(axis=1)
        df_ok['zt'] = df_ok['zb'] + df_ok['zm'] + df_ok['za']; mhr = df_ok['zt'] > 0
        df_ok.loc[mhr, 'pb'] = df_ok.loc[mhr, 'zb'] / df_ok.loc[mhr, 'zt'] * 100
        df_ok.loc[mhr, 'pm'] = df_ok.loc[mhr, 'zm'] / df_ok.loc[mhr, 'zt'] * 100
        df_ok.loc[mhr, 'pa'] = df_ok.loc[mhr, 'za'] / df_ok.loc[mhr, 'zt'] * 100
        df_ok[rpe_col] = pd.to_numeric(df_ok[rpe_col], errors='coerce')
        df_ok['rpe_cat'] = pd.cut(df_ok[rpe_col], bins=[0, 4.9, 6.9, 10], labels=['Leve (1–4)', 'Moderado (5–6)', 'Forte (7–10)'], right=True)
        df_ok = df_ok.dropna(subset=[rpe_col, 'pb', 'pm', 'pa', 'rpe_cat']); df_ok = df_ok[df_ok['zt'] > 0]
        mods_corr = [m for m in mods_sel if m in df_ok['type'].values]
        if mods_corr and len(df_ok) >= 5:
            HR_VARS = ['pb', 'pm', 'pa']; HR_LABELS = ['HR Baixa\n(Z1+Z2)', 'HR Moderada\n(Z3+Z4)', 'HR Alta\n(Z5+Z6+Z7)']
            CORES_RPE_CAT = {'Leve (1–4)': CORES['azul'], 'Moderado (5–6)': CORES['laranja'], 'Forte (7–10)': CORES['vermelho']}
            fig, axes2 = plt.subplots(1, len(mods_corr), figsize=(5 * len(mods_corr), 5))
            if len(mods_corr) == 1: axes2 = [axes2]
            for ax, mod in zip(axes2, mods_corr):
                dm = df_ok[df_ok['type'] == mod]
                if len(dm) < 5: ax.text(0.5, 0.5, f'{mod}\nn insuficiente', ha='center', va='center', transform=ax.transAxes); continue
                cm_mat = np.zeros((3, 1)); annot = np.empty((3, 1), dtype=object)
                for i, hv in enumerate(HR_VARS):
                    x = dm[rpe_col].values; y = dm[hv].values
                    if np.std(y) > 0 and len(x) >= 5:
                        r, p = pearsonr(x, y); cm_mat[i, 0] = r
                        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                        annot[i, 0] = f'r={r:.2f}\n{sig}'
                    else: annot[i, 0] = 'n/a'
                im = ax.imshow(cm_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
                ax.set_xticks([0]); ax.set_xticklabels(['RPE'], fontsize=10)
                ax.set_yticks(range(3)); ax.set_yticklabels(HR_LABELS, fontsize=9)
                for i in range(3):
                    tc = 'white' if abs(cm_mat[i, 0]) > 0.5 else 'black'
                    ax.text(0, i, annot[i, 0], ha='center', va='center', fontsize=10, fontweight='bold', color=tc)
                ax.set_title(f'{mod} (n={len(dm)})', fontsize=11, fontweight='bold', color=get_cor(mod))
                plt.colorbar(im, ax=ax, shrink=0.7, label='r de Pearson')
            plt.suptitle('Correlação Pearson: RPE × HR Zones', fontsize=13, fontweight='bold', y=1.03)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            rows = []
            forca = lambda r: 'Muito Forte' if abs(r) >= 0.7 else ('Forte' if abs(r) >= 0.5 else ('Moderada' if abs(r) >= 0.3 else 'Fraca'))
            for mod in mods_corr:
                dm = df_ok[df_ok['type'] == mod]
                for hv, hl in zip(HR_VARS, ['HR Baixa', 'HR Moderada', 'HR Alta']):
                    x = dm[rpe_col].values; y = dm[hv].values
                    if np.std(y) > 0 and len(x) >= 5:
                        r, p = pearsonr(x, y)
                        rows.append({'Modalidade': mod, 'HR Zone': hl, 'r': f'{r:+.3f}', 'p-value': f'{p:.4f}', 'n': len(x),
                                     'Sig': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')), 'Força': forca(r)})
            if rows:
                st.subheader("📋 Tabela de Correlações")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — CORRELAÇÕES & IMPACTO RPE
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_correlacoes.py
# ════════════════════════════════════════════════════════════════════════════
