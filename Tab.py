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

def tab_pmc(da):
    """
    PMC usa SEMPRE o histórico completo (histórico carregado = 730d) para calcular
    CTL/ATL/FTLM correctamente — o filtro de período apenas controla o que é exibido.
    """
    st.header("📈 PMC — Performance Management Chart")
    if len(da) == 0: st.warning("Sem dados de atividades."); return

    # Usa todos os dados disponíveis (não só o período filtrado)
    # da já vem do carregar_atividades(days_back=730) via st.session_state
    da_full = st.session_state.get('da_full', da)
    df = filtrar_principais(da_full).copy(); df['Data'] = pd.to_datetime(df['Data'])

    # MÉTRICA: session_rpe = (moving_time_min) × RPE — igual ao código original Python/SQLite
    # CTL=~350-400 com esta escala (60min×RPE6=360). icu_training_load daria CTL~40-80 (escala errada).
    if 'moving_time' in df.columns and 'rpe' in df.columns and df['rpe'].notna().sum() > 10:
        df['rpe_fill'] = df['rpe'].fillna(df['rpe'].median())
        df['load_val'] = (df['moving_time'] / 60) * df['rpe_fill']
        _load_metrica = "session_rpe = (moving_time_min × RPE) — igual ao Python/SQLite original"
    elif 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['load_val'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        _load_metrica = "icu_training_load — escala diferente (CTL ~40-80 em vez de ~350-400)"
    else:
        st.warning("Sem dados de load (rpe ou icu_training_load necessários)."); return

    ld = df.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
    idx_full = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx_full, fill_value=0).reset_index(); ld.columns = ['Data', 'load_val']

    # CTL/ATL sobre TODO o histórico (para que os valores actuais sejam correctos)
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB'] = ld['CTL'] - ld['ATL']

    # FTLM — gamma otimizado sobre todo o histórico
    best_g, best_r = 0.30, -1
    for g in np.arange(0.25, 0.36, 0.01):
        ema = ld['load_val'].ewm(alpha=g, adjust=False).mean()
        if ema.std() > 0:
            r = abs(np.corrcoef(ld['load_val'].values, ema.values)[0, 1])
            if r > best_r: best_r, best_g = r, g
    ld['FTLM'] = ld['load_val'].ewm(alpha=best_g, adjust=False).mean()

    # Filtro de exibição — controla o período mostrado no gráfico
    st.info(f"📊 Métrica de load: **{_load_metrica}** | Histórico: {len(ld)} dias")
    if "session_rpe" in _load_metrica:
        st.warning("⚠️ Para resultados equivalentes ao Python/SQLite: usa icu_training_load. Exporta o histórico completo do Intervals.icu para a Google Sheet.")

    col1, col2, col3 = st.columns(3)
    dias_exib_opts = {"30 dias": 30, "60 dias": 60, "90 dias": 90, "180 dias": 180, "1 ano": 365, "Todo histórico": len(ld)}
    dias_exib_lbl = col1.selectbox("Período exibido", list(dias_exib_opts.keys()), index=2)
    dias_exib = dias_exib_opts[dias_exib_lbl]
    ld_plot = ld.tail(dias_exib).copy()

    smooth = col2.checkbox("Suavizar CTL/ATL (3d)", value=False)
    show_ftlm = col3.checkbox("Mostrar FTLM", value=True)
    if smooth:
        ld_plot = ld_plot.copy()
        ld_plot['CTL'] = ld_plot['CTL'].rolling(3, min_periods=1).mean()
        ld_plot['ATL'] = ld_plot['ATL'].rolling(3, min_periods=1).mean()

    idx = ld_plot['Data']  # para as barras de load por tipo

    fig, (ax_pmc, ax_load) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ax_pmc.plot(ld_plot['Data'], ld_plot['CTL'], label='CTL', color=CORES['azul'], linewidth=2.5)
    ax_pmc.plot(ld_plot['Data'], ld_plot['ATL'], label='ATL', color=CORES['vermelho'], linewidth=2.5)
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'], where=(ld_plot['TSB'] >= 0), color=CORES['verde'], alpha=0.25, label='TSB+')
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'], where=(ld_plot['TSB'] < 0), color=CORES['vermelho'], alpha=0.20, label='TSB-')
    ax_pmc.axhline(0, color=CORES['cinza'], linestyle='--', linewidth=0.8)
    ax_pmc.set_ylabel('CTL / ATL / TSB', fontweight='bold'); ax_pmc.grid(True, alpha=0.3)
    if show_ftlm:
        ax2 = ax_pmc.twinx()
        ax2.plot(ld_plot['Data'], ld_plot['FTLM'], label=f'FTLM (gamma={best_g:.2f})', color=CORES['laranja'], linewidth=2, linestyle='--', alpha=0.85)
        ax2.set_ylabel('FTLM', color=CORES['laranja'], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=CORES['laranja'])
        l1, lb1 = ax_pmc.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        ax_pmc.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=9)
    else:
        ax_pmc.legend(loc='upper left', fontsize=9)
    # Valores actuais = sempre ultimo dia de todo o historico
    u = ld.iloc[-1]
    ax_pmc.text(0.99, 0.97, f"CTL: {u['CTL']:.0f}  |  ATL: {u['ATL']:.0f}  |  TSB: {u['TSB']:+.0f}",
                transform=ax_pmc.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax_pmc.set_title('PMC — CTL / ATL / TSB / FTLM / Training Load', fontsize=14, fontweight='bold')
    # Stacked bars por modalidade — igual ao original plot_training_load
    if 'type' in df.columns:
        tipos_ord = [t for t in ['Bike', 'Row', 'Ski', 'Run', 'WeightTraining'] if t in df['type'].unique()]
        tipos_ord += [t for t in df['type'].unique() if t not in tipos_ord]
        bottom_vals = np.zeros(len(ld_plot))
        for tipo in tipos_ord:
            dt = df[df['type'] == tipo].groupby('Data')['load_val'].sum().reset_index()
            dt.columns = ['Data', 'lv']
            dt['Data'] = pd.to_datetime(dt['Data'])
            # Alinhar com ld_plot usando merge
            merged_bar = ld_plot[['Data']].merge(dt, on='Data', how='left').fillna(0)
            vals = merged_bar['lv'].values
            ax_load.bar(ld_plot['Data'], vals, bottom=bottom_vals,
                        color=get_cor(tipo), alpha=0.85, width=0.8, label=tipo, edgecolor='white', linewidth=0.3)
            bottom_vals += vals
        ax_load.legend(loc='upper left', fontsize=8, ncol=min(5, len(tipos_ord)))
    else:
        ax_load.bar(ld_plot['Data'], ld_plot['load_val'], color=CORES['roxo'], alpha=0.65, width=0.8, label='Training Load')
        ax_load.legend(loc='upper left', fontsize=8)
    ax_load.set_ylabel('Load\n(TRIMP)', fontweight='bold', fontsize=9); ax_load.grid(True, alpha=0.2, axis='y')
    ax_load.tick_params(axis='x', rotation=45)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.subheader("Resumo PMC")
    res = pd.DataFrame({'Metrica': ['CTL (atual)', 'ATL (atual)', 'TSB (atual)', 'CTL (max hist)', 'ATL (max hist)', 'FTLM (atual)', 'FTLM gamma'],
                        'Valor': [f"{u['CTL']:.1f}", f"{u['ATL']:.1f}", f"{u['TSB']:+.1f}",
                                  f"{ld['CTL'].max():.1f}", f"{ld['ATL'].max():.1f}", f"{u['FTLM']:.1f}", f"{best_g:.3f}"]})
    st.dataframe(res, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOLUME & CARGA
# ════════════════════════════════════════════════════════════════════════════════
