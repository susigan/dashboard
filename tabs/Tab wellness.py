# tabs/tab_wellness.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta

from config import CORES, MAPA_WELLNESS
from utils.helpers import (filtrar_principais, calcular_bpe,
                            calcular_swc, norm_range, cvr)

def tab_wellness(dw):
    st.header("🧘 Wellness")
    if len(dw) == 0: st.warning("Sem dados de wellness."); return
    mets = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness'] if m in dw.columns and dw[m].notna().any()]
    if not mets: st.warning("Sem métricas wellness."); return
    sel = st.multiselect("Métricas", mets, default=mets[:5])
    if not sel: return
    fig, axes = plt.subplots(len(sel), 1, figsize=(14, 3 * len(sel)), sharex=True)
    if len(sel) == 1: axes = [axes]
    x = range(len(dw)); datas = pd.to_datetime(dw['Data']).dt.strftime('%d/%m')
    CM = {'hrv': CORES['verde'], 'rhr': CORES['vermelho'], 'sleep_quality': CORES['roxo'],
          'fatiga': CORES['laranja'], 'stress': CORES['vermelho_escuro'], 'humor': CORES['verde_escuro'], 'soreness': CORES['azul']}
    for ax, met in zip(axes, sel):
        v = pd.to_numeric(dw[met], errors='coerce').values
        ax.plot(x, v, color=CM.get(met, CORES['azul']), linewidth=2, marker='o', markersize=4)
        if len(v) >= 7: ax.plot(x, pd.Series(v).rolling(7, min_periods=3).mean(), color=CORES['preto'], linewidth=1.5, linestyle='--', alpha=0.5, label='Média 7d')
        ax.set_ylabel(met.replace('_', ' ').title(), fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    step = max(1, len(x) // 12); axes[-1].set_xticks(list(x)[::step])
    axes[-1].set_xticklabels([datas.iloc[i] for i in range(0, len(datas), step)], rotation=45)
    plt.suptitle('Métricas Wellness', fontsize=14, fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
    st.subheader("📋 Resumo (últimos 7 dias)")
    if len(dw) >= 7:
        u7 = dw.tail(7); rows = []
        for m in mets:
            col = pd.to_numeric(u7[m], errors='coerce')
            rows.append({'Métrica': m.replace('_', ' ').title(), 'Média': f"{col.mean():.1f}", 'Mín': f"{col.min():.0f}", 'Máx': f"{col.max():.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



# ════════════════════════════════════════════════════════════════════════════
