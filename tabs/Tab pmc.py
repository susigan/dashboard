# tabs/tab_pmc.py — ATHELTICA Dashboard
# PMC: CTL/ATL/FTLM + TRIMP bars + Resumo + FTLM
# (tabelas eFTP/KM/Correlações movidas para tab_eftp)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from utils.helpers import filtrar_principais, add_tempo, get_cor

def tab_pmc(da):
    """
    PMC — icu_training_load para CTL/ATL/FTLM.
    Barras de Load = TRIMP (session_rpe).
    Tabelas mensais: eFTP por modalidade + KM/KJ por modalidade (como Intervals.icu).
    Gráfico KM semanal stacked com linha de média por modalidade.
    """
    st.header("📈 PMC — Performance Management Chart")
    if len(da) == 0: st.warning("Sem dados de atividades."); return

    da_full = st.session_state.get('da_full', da)
    _mods = st.session_state.get('mods_sel', None)
    if _mods and 'type' in da_full.columns:
        da_full = da_full[da_full['type'].isin(_mods + ['WeightTraining'])]
    df = filtrar_principais(da_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # ── CTL/ATL: icu_training_load primeiro, session_rpe fallback ──
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['icu_tl'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        _metrica_ctl = "icu_training_load (Intervals.icu)"
    elif 'moving_time' in df.columns and 'rpe' in df.columns and df['rpe'].notna().sum() > 10:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['icu_tl'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        _metrica_ctl = "session_rpe (fallback)"
    else:
        st.warning("Sem icu_training_load nem RPE para calcular CTL/ATL."); return

    # ── Load bars: TRIMP = session_rpe ──
    _rpe2 = pd.to_numeric(df['rpe'], errors='coerce') if 'rpe' in df.columns else pd.Series(dtype=float)
    _mt   = pd.to_numeric(df['moving_time'], errors='coerce') if 'moving_time' in df.columns else pd.Series(dtype=float)
    if _rpe2.notna().sum() > 5:
        df['trimp_val'] = (_mt / 60) * _rpe2.fillna(_rpe2.median())
    else:
        df['trimp_val'] = df['icu_tl']

    # ── Série diária CTL/ATL ──
    ld = df.groupby('Data')['icu_tl'].sum().reset_index().sort_values('Data')
    idx_full = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx_full, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']
    ld['CTL']  = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL']  = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB']  = ld['CTL'] - ld['ATL']

    best_g, best_r = 0.30, -1.0
    for g in np.arange(0.25, 0.36, 0.01):
        ema = ld['load_val'].ewm(alpha=g, adjust=False).mean()
        if ema.std() > 0:
            r = abs(np.corrcoef(ld['load_val'].values, ema.values)[0, 1])
            if r > best_r: best_r, best_g = r, g
    ld['FTLM'] = ld['load_val'].ewm(alpha=best_g, adjust=False).mean()
    u = ld.iloc[-1]

    st.caption(f"CTL/ATL/FTLM: **{_metrica_ctl}** | "
               f"Barras Load: **TRIMP (session_rpe)** | Histórico: {len(ld)} dias")

    # ── Controlos ──
    col1, col2, col3 = st.columns(3)
    dias_opts = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
                 "180 dias": 180, "1 ano": 365, "Todo histórico": len(ld)}
    dias_exib = dias_opts[col1.selectbox("Período exibido", list(dias_opts.keys()), index=2)]
    ld_plot   = ld.tail(dias_exib).copy()
    smooth    = col2.checkbox("Suavizar CTL/ATL (3d)", value=False)
    show_ftlm = col3.checkbox("Mostrar FTLM", value=True)
    if smooth:
        ld_plot['CTL'] = ld_plot['CTL'].rolling(3, min_periods=1).mean()
        ld_plot['ATL'] = ld_plot['ATL'].rolling(3, min_periods=1).mean()

    # ── GRÁFICO 1: PMC + Load (TRIMP) ──
    fig, (ax_pmc, ax_load) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ax_pmc.plot(ld_plot['Data'], ld_plot['CTL'], label='CTL (Fitness)',
                color=CORES['azul'], linewidth=2.5)
    ax_pmc.plot(ld_plot['Data'], ld_plot['ATL'], label='ATL (Fadiga)',
                color=CORES['vermelho'], linewidth=2.5)
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'],
                        where=(ld_plot['TSB'] >= 0),
                        color=CORES['verde'], alpha=0.25, label='TSB+ (Forma)')
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'],
                        where=(ld_plot['TSB'] < 0),
                        color=CORES['vermelho'], alpha=0.20, label='TSB- (Fadiga)')
    ax_pmc.axhline(0, color=CORES['cinza'], linestyle='--', linewidth=0.8)
    ax_pmc.set_ylabel('CTL / ATL / TSB', fontweight='bold')
    ax_pmc.grid(True, alpha=0.3)
    if show_ftlm:
        ax2 = ax_pmc.twinx()
        ax2.plot(ld_plot['Data'], ld_plot['FTLM'],
                 label=f'FTLM (gamma={best_g:.2f})',
                 color=CORES['laranja'], linewidth=2, linestyle='--', alpha=0.85)
        ax2.set_ylabel('FTLM', color=CORES['laranja'], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=CORES['laranja'])
        l1, lb1 = ax_pmc.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax_pmc.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
    else:
        ax_pmc.legend(loc='upper left', fontsize=9)
    ax_pmc.text(0.99, 0.97,
                f"CTL: {u['CTL']:.1f}  |  ATL: {u['ATL']:.1f}  |  TSB: {u['TSB']:+.1f}",
                transform=ax_pmc.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax_pmc.set_title('PMC — CTL / ATL / TSB / FTLM', fontsize=14, fontweight='bold')

    trimp_d = df.groupby(['Data', 'type'])['trimp_val'].sum().reset_index()
    trimp_d['Data'] = pd.to_datetime(trimp_d['Data'])
    tipos_ord = [t for t in ['Bike', 'Row', 'Ski', 'Run', 'WeightTraining']
                 if t in trimp_d['type'].unique()]
    tipos_ord += [t for t in trimp_d['type'].unique() if t not in tipos_ord]
    bot = np.zeros(len(ld_plot))
    for tipo in tipos_ord:
        dt = trimp_d[trimp_d['type'] == tipo][['Data', 'trimp_val']]
        merged = ld_plot[['Data']].merge(dt, on='Data', how='left').fillna(0)
        ax_load.bar(ld_plot['Data'], merged['trimp_val'].values, bottom=bot,
                    color=get_cor(tipo), alpha=0.85, width=0.8, label=tipo,
                    edgecolor='white', linewidth=0.3)
        bot += merged['trimp_val'].values
    ax_load.legend(loc='upper left', fontsize=8, ncol=min(5, len(tipos_ord)))
    ax_load.set_ylabel('Load\n(TRIMP)', fontweight='bold', fontsize=9)
    ax_load.grid(True, alpha=0.2, axis='y')
    ax_load.tick_params(axis='x', rotation=45)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── RESUMO PMC ──
    st.subheader("📊 Resumo PMC")
    tsb_v = u['TSB']
    if   tsb_v >  25: tsb_i = "🟢 Forma — pronto para competir/esforço máximo"
    elif tsb_v >   5: tsb_i = "🟡 Fresco — bom equilíbrio treino/recuperação"
    elif tsb_v > -10: tsb_i = "🟠 Neutro — zona de manutenção"
    elif tsb_v > -25: tsb_i = "🔴 Fatigado — carga elevada, monitorar recuperação"
    else:              tsb_i = "⛔ Sobrecarregado — reduzir carga imediatamente"
    resumo = pd.DataFrame([
        {'Métrica': 'CTL (Fitness atual)',  'Valor': f"{u['CTL']:.1f}",
         'Interpretação': 'Capacidade aeróbica crónica (42d). Maior = melhor condição.'},
        {'Métrica': 'ATL (Fadiga atual)',   'Valor': f"{u['ATL']:.1f}",
         'Interpretação': 'Fadiga aguda (7d). Maior = mais fatigado recentemente.'},
        {'Métrica': 'TSB (Forma atual)',    'Valor': f"{u['TSB']:+.1f}",
         'Interpretação': tsb_i},
        {'Métrica': 'CTL max histórico',    'Valor': f"{ld['CTL'].max():.1f}",
         'Interpretação': 'Pico de fitness no período carregado.'},
        {'Métrica': 'ATL max histórico',    'Valor': f"{ld['ATL'].max():.1f}",
         'Interpretação': 'Pico de fadiga no período carregado.'},
    ])
    st.dataframe(resumo, use_container_width=True, hide_index=True)

    # ── FTLM — explicação + resultado atual ──
    st.subheader("🔁 FTLM — Fast Training Load Monitor")
    ftlm_v = u['FTLM']
    ctl_v  = u['CTL']
    pct    = (ftlm_v / ctl_v * 100) if ctl_v > 0 else 0
    if   pct > 110: ftlm_i = "⚠️ Carga muito acima do crónico — risco de overreaching"
    elif pct > 100: ftlm_i = "🔴 Carga acima do CTL — fase de acumulação/sobrecarga"
    elif pct >  90: ftlm_i = "🟡 Ligeiramente abaixo do CTL — manutenção/tapering leve"
    elif pct >  75: ftlm_i = "🟢 Tapering activo — carga a baixar, forma a subir"
    else:            ftlm_i = "⬇️ Destreino — carga muito abaixo do nível crónico"

    with st.expander("📖 O que é o FTLM e como interpretar", expanded=True):
        st.markdown(f"""
**FTLM (Fast Training Load Monitor)** é uma média exponencial da carga diária com
um factor gamma (γ) **optimizado automaticamente** por correlação com os teus dados.

| Parâmetro | Valor actual |
|---|---|
| Gamma (γ) | `{best_g:.3f}` |
| FTLM actual | `{ftlm_v:.1f}` |
| CTL actual | `{ctl_v:.1f}` |
| FTLM / CTL | `{pct:.0f}%` |
| Interpretação | {ftlm_i} |

**Como interpretar:**
- **FTLM > CTL (>100%)** — Carga recente **acima** da capacidade crónica. Bloco de carga intenso. Monitorar recuperação.
- **FTLM ≈ CTL (90–110%)** — Carga estável. Manutenção do nível actual.
- **FTLM < CTL (<90%)** — Carga recente **abaixo** do crónico. Tapering intencional ou destreino.

**Diferença para o ATL:**
O ATL usa sempre span=7 (fixo). O FTLM usa γ=`{best_g:.3f}`
(equivalente a span≈{int(round(2/best_g - 1))}), **optimizado para os teus dados**,
tornando-o mais sensível ao teu padrão específico de treino.
        """)
