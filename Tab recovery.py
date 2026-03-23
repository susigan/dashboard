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
# TAB 7 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════════
def calcular_recovery(dw):
    if len(dw) == 0: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['hrv_b14'] = df['hrv'].rolling(14, min_periods=7).mean()
    df['rhr_b14'] = df['rhr'].rolling(14, min_periods=7).mean() if 'rhr' in df.columns else np.nan
    df['cv7'] = cvr(df['hrv'], 7); df['cv30'] = cvr(df['hrv'], 30)
    rows = []
    for _, row in df.iterrows():
        hv = row.get('hrv'); hb = row.get('hrv_b14'); cv = row.get('cv7'); hs = 50
        if pd.notna(hv) and pd.notna(hb) and hb > 0 and pd.notna(cv):
            inf, sup = norm_range(hb, cv)
            if inf and sup:
                band = sup - inf
                if hv >= sup: hs = min(100, 75 + (hv - sup) / band * 25 if band > 0 else 75)
                elif hv <= inf: hs = max(0, 40 - (inf - hv) / band * 40 if band > 0 else 40)
                else: hs = 50 + ((hv - inf) / band * 25 if band > 0 else 0)
        rv = row.get('rhr'); rb = row.get('rhr_b14'); rs = 50
        if pd.notna(rv) and pd.notna(rb) and rb > 0:
            pct = (rv - rb) / rb * 100; rs = 90 if pct < -10 else 75 if pct < -5 else 55 if pct < 5 else 35 if pct < 10 else 20
        sl = conv_15(row.get('sleep_quality')); fa = conv_15(row.get('fatiga'))
        st_ = conv_15(row.get('stress')); hu = conv_15(row.get('humor')); so = conv_15(row.get('soreness'))
        score = hs * 0.30 + rs * 0.15 + sl * 0.20 + fa * 0.10 + st_ * 0.10 + hu * 0.05 + so * 0.05 + 50 * 0.05
        inf2, sup2 = norm_range(hb if pd.notna(hb) else 0, cv if pd.notna(cv) else 10)
        rows.append({'Data': row['Data'], 'recovery_score': score, 'hrv': hv, 'hrv_baseline': hb,
                     'hrv_cv7': cv, 'hrv_cv30': row.get('cv30'), 'normal_range_inf': inf2, 'normal_range_sup': sup2,
                     'hrv_comp': hs, 'rhr_comp': rs, 'sleep_comp': sl, 'fatiga_comp': fa, 'stress_comp': st_})
    return pd.DataFrame(rows)

def calcular_bpe(dw, metrica='hrv', baseline_dias=60):
    """
    BPE Z-Score semanal usando metodologia do artigo (igual ao Python original):
    - Baseline = média dos últimos N dias do período TOTAL (fixo, não rolling)
    - CV%      = (STD / Média) × 100 do mesmo período de baseline
    - SWC      = 0.5 × CV% × Baseline / 100
    - Z-Score  = (Média_semanal - Baseline) / SWC
    """
    if metrica not in dw.columns or len(dw) < 14: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
    df_clean = df.dropna(subset=[metrica])
    if len(df_clean) < 14: return pd.DataFrame()

    # BASELINE FIXO: últimos baseline_dias do período total (igual ao original)
    n_base = min(baseline_dias, len(df_clean))
    baseline_data = df_clean[metrica].tail(n_base)
    base = baseline_data.mean()
    std_base = baseline_data.std()
    if pd.isna(base) or base <= 0 or pd.isna(std_base) or std_base <= 0:
        return pd.DataFrame()
    cv = (std_base / base) * 100
    swc = calcular_swc(base, cv)
    if swc is None or swc <= 0: return pd.DataFrame()

    # Agrupar por semana e calcular Z-Score com baseline fixo
    df_clean['semana'] = df_clean['Data'].dt.to_period('W')
    rows = []
    for sem in sorted(df_clean['semana'].unique()):
        df_sem = df_clean[df_clean['semana'] == sem]
        media_sem = df_sem[metrica].mean()
        if pd.isna(media_sem): continue
        zscore = (media_sem - base) / swc
        rows.append({
            'ano_semana': str(sem),
            'media_semanal': media_sem,
            'baseline': base,
            'swc': swc,
            'cv_percent': cv,
            'zscore': zscore,
            'n_dias': len(df_sem)
        })
    return pd.DataFrame(rows)

def tab_recovery(dw):
    st.header("🔋 Recovery Score & HRV Analysis")
    if len(dw) == 0 or 'hrv' not in dw.columns: st.warning("Sem dados de wellness/HRV."); return
    rec = calcular_recovery(dw)
    if len(rec) == 0: return
    u = rec.iloc[-1]; score = u['recovery_score']
    cat = ('🟢 Excelente' if score >= 80 else '🟡 Bom' if score >= 60 else '🟠 Moderado' if score >= 40 else '🔴 Baixo')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{score:.0f}/100", delta=cat)
    c2.metric("HRV atual", f"{u['hrv']:.0f} ms" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline HRV", f"{u['hrv_baseline']:.0f} ms" if pd.notna(u['hrv_baseline']) else "—")
    c4.metric("CV% 7d", f"{u['hrv_cv7']:.1f}%" if pd.notna(u['hrv_cv7']) else "—")
    st.markdown("---")
    n_dias = st.slider("Dias a mostrar", 14, min(len(rec), 365), min(90, len(rec)))
    df_tl = rec.tail(n_dias).copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(80, 100, alpha=0.15, color=CORES['verde'], label='Excelente (80–100)')
    ax.axhspan(60, 80, alpha=0.15, color=CORES['amarelo'], label='Bom (60–79)')
    ax.axhspan(40, 60, alpha=0.15, color=CORES['laranja'], label='Moderado (40–59)')
    ax.axhspan(0, 40, alpha=0.15, color=CORES['vermelho'], label='Baixo (0–39)')
    x = range(len(df_tl)); sc = df_tl['recovery_score'].values
    cpts = [CORES['verde'] if s >= 80 else CORES['amarelo'] if s >= 60 else CORES['laranja'] if s >= 40 else CORES['vermelho'] for s in sc]
    ax.plot(x, sc, color=CORES['azul_escuro'], linewidth=2, alpha=0.7)
    ax.scatter(x, sc, c=cpts, s=70, edgecolors='white', linewidths=2, zorder=5)
    if len(df_tl) >= 7: ax.plot(x, pd.Series(sc).rolling(7, min_periods=3).mean(), color=CORES['roxo'], linewidth=2.5, linestyle='--', label='Média 7d', alpha=0.8)
    datas = df_tl['Data'].dt.strftime('%d/%m'); step = max(1, len(x) // 10)
    ax.set_xticks(list(x)[::step]); ax.set_xticklabels([datas.iloc[i] for i in range(0, len(datas), step)], rotation=45)
    ax.set_ylim(0, 105); ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Score — Timeline', fontsize=14, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 HRV com Normal Range")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1.2])
    xr = range(len(df_tl)); dr = df_tl['Data'].dt.strftime('%d/%m'); sr = max(1, len(xr) // 10)
    if df_tl['normal_range_inf'].notna().any():
        ax1.fill_between(xr, df_tl['normal_range_inf'], df_tl['normal_range_sup'], alpha=0.25, color=CORES['azul'], label='Normal Range HRV')
    if df_tl['hrv_baseline'].notna().any():
        ax1.plot(xr, df_tl['hrv_baseline'], color=CORES['roxo'], linestyle='--', linewidth=2, label='Baseline 14d')
    hv = df_tl['hrv'].values
    chr_list = [CORES['verde'] if (not pd.isna(df_tl['normal_range_sup'].iloc[i]) and not pd.isna(hv[i]) and hv[i] > df_tl['normal_range_sup'].iloc[i])
                else CORES['vermelho'] if (not pd.isna(df_tl['normal_range_inf'].iloc[i]) and not pd.isna(hv[i]) and hv[i] < df_tl['normal_range_inf'].iloc[i])
                else CORES['azul'] for i in range(len(df_tl))]
    ax1.plot(xr, hv, color=CORES['preto'], linewidth=2, alpha=0.6)
    ax1.scatter(xr, hv, c=chr_list, s=70, edgecolors='white', linewidths=2, zorder=5)
    ax1.legend(loc='upper right', fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylabel('HRV (ms)', fontweight='bold')
    ax1.set_title('HRV com Normal Range (HRV4Training)', fontsize=13, fontweight='bold')
    cv_s = df_tl['hrv_cv7'].copy()
    if cv_s.notna().sum() >= 5:
        cv_b = cv_s.rolling(14, min_periods=5).mean(); cv_sd = cv_s.rolling(14, min_periods=5).std()
        cv_inf = cv_b - 0.5 * cv_sd; cv_sup = cv_b + 0.5 * cv_sd
        ax2.fill_between(xr, cv_inf, cv_sup, alpha=0.25, color=CORES['laranja'], label='Normal Range CV%')
        ax2.plot(xr, cv_b, color=CORES['roxo'], linestyle='--', linewidth=1.8, alpha=0.8)
        cv_v = cv_s.values
        ccv = [CORES['verde'] if (not pd.isna(cv_sup.iloc[i]) and not pd.isna(cv_v[i]) and cv_v[i] > cv_sup.iloc[i])
               else CORES['vermelho'] if (not pd.isna(cv_inf.iloc[i]) and not pd.isna(cv_v[i]) and cv_v[i] < cv_inf.iloc[i])
               else CORES['azul'] for i in range(len(df_tl))]
        ax2.plot(xr, cv_v, color=CORES['preto'], linewidth=1.8, alpha=0.6)
        ax2.scatter(xr, cv_v, c=ccv, s=55, edgecolors='white', linewidths=1.5, zorder=5)
        ax2.axhline(3, color=CORES['cinza'], linestyle=':', alpha=0.5); ax2.axhline(10, color=CORES['cinza'], linestyle=':', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=8)
    for axr in [ax1, ax2]:
        axr.set_xticks(list(xr)[::sr]); axr.set_xticklabels([dr.iloc[i] for i in range(0, len(dr), sr)], rotation=45); axr.grid(True, alpha=0.3)
    ax2.set_ylabel('CV% HRV', fontweight='bold'); ax2.set_title('CV% com Normal Range', fontsize=12, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.subheader("📊 BPE — Z-Score Semanal (Método SWC)")
    mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] if m in dw.columns and dw[m].notna().any()]
    n_semanas_disp = max(1, len(dw) // 7)
    _skip_bpe = n_semanas_disp < 4
    if _skip_bpe:
        st.info(f"Dados insuficientes para BPE (min 4 semanas, disponivel: {n_semanas_disp}).")
        n_sem = n_semanas_disp
    else:
        _slider_max = min(52, n_semanas_disp)
        _slider_val = min(16, _slider_max)
        if _slider_max > 4:
            n_sem = st.slider("Semanas (BPE)", 4, _slider_max, _slider_val)
        else:
            n_sem = _slider_max
            st.caption(f"BPE: {n_sem} semanas disponíveis")
    dados_bpe = {}
    if not _skip_bpe:
        for met in mets_bpe:
            s = calcular_bpe(dw, met, 60)
            if len(s) > 0: dados_bpe[met] = s.tail(n_sem)
    if dados_bpe:
        semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
        nm = len(dados_bpe); mat = np.zeros((nm, len(semanas)))
        for i, met in enumerate(dados_bpe.keys()):
            z = dados_bpe[met]['zscore'].values; mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
        cmap = LinearSegmentedColormap.from_list('bpe', [CORES['vermelho'], CORES['amarelo'], CORES['verde']], N=100)
        fig, ax = plt.subplots(figsize=(max(14, len(semanas) * 0.9), max(5, nm * 1.1)))
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
        nomes = {'hrv': 'HRV', 'rhr': 'RHR (inv)', 'sleep_quality': 'Sono', 'fatiga': 'Energia', 'stress': 'Relaxamento'}
        ax.set_yticks(range(nm)); ax.set_yticklabels([nomes.get(m, m) for m in dados_bpe.keys()], fontsize=11)
        slbls = [s.split('-W')[1] if '-W' in s else s for s in semanas]
        ax.set_xticks(range(len(semanas))); ax.set_xticklabels([f'S{s}' for s in slbls], rotation=45, fontsize=9)
        for i in range(nm):
            for j in range(len(semanas)):
                v = mat[i, j]; tc = 'white' if abs(v) > 1 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('Z-Score BPE (múltiplos de SWC)')
        cbar.set_ticks([-2, -1, 0, 1, 2]); cbar.set_ticklabels(['🔴 -2', '🟠 -1', '0', '🟡 +1', '🟢 +2'])
        ax.set_title('BPE — Blocos de Padrão Específico (Z-Score com SWC)', fontsize=13, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")
    if len(dw) >= 14 and 'hrv' in dw.columns and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data'); df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'] > 0, np.log(df_hg['hrv']), np.nan); df_hg = df_hg.dropna(subset=['LnrMSSD'])
        dias_fam = st.slider("Dias baseline rolling", 7, 28, 14)
        df_hg['bm'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg['bs'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg['linf'] = df_hg['bm'] - 0.5 * df_hg['bs']; df_hg['lsup'] = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs']
        df_hg['intens'] = df_hg.apply(lambda r: 'HIIT' if pd.notna(r['bm']) and r['linf'] <= r['LnrMSSD'] <= r['lsup'] else ('Recuperação' if pd.notna(r['bm']) else 'Sem dados'), axis=1)
        n_hg = st.slider("Dias HRV-Guided", 14, min(len(df_hg), 180), min(60, len(df_hg)))
        df_p = df_hg.tail(n_hg).copy(); xh = range(len(df_p)); dh = df_p['Data'].dt.strftime('%d/%m'); sh = max(1, len(xh) // 15)
        ch = [CORES['verde'] if i == 'HIIT' else CORES['laranja'] if i == 'Recuperação' else CORES['cinza'] for i in df_p['intens']]
        fig3, (ah1, ah2) = plt.subplots(2, 1, figsize=(15, 10))
        ah1.plot(xh, df_p['LnrMSSD'], '-', alpha=0.6, color=CORES['azul'], linewidth=2, label='LnrMSSD')
        ah1.scatter(xh, df_p['LnrMSSD'], c=ch, s=80, edgecolors='white', linewidths=2, zorder=5)
        ah1.plot(xh, df_p['bm'], color=CORES['verde_escuro'], linestyle='--', linewidth=2.5, label=f'Baseline ({dias_fam}d)', alpha=0.8)
        ah1.plot(xh, df_p['lsup'], color=CORES['laranja'], linestyle=':', linewidth=2, label='±0.5 DP', alpha=0.7)
        ah1.plot(xh, df_p['linf'], color=CORES['laranja'], linestyle=':', linewidth=2, alpha=0.7)
        ah1.fill_between(xh, df_p['linf'], df_p['lsup'], alpha=0.15, color=CORES['verde'], label='Zona HIIT')
        ah1.legend(loc='best', fontsize=9); ah1.grid(True, alpha=0.3); ah1.set_ylabel('LnrMSSD', fontweight='bold')
        ah1.set_title(f'HRV-Guided Training — Baseline Rolling ({dias_fam}d)', fontsize=13, fontweight='bold')
        ah1.set_xticks(list(xh)[::sh]); ah1.set_xticklabels([dh.iloc[i] for i in range(0, len(dh), sh)], rotation=45)
        ah2.axhline(0, color=CORES['verde_escuro'], linestyle='-', linewidth=2, alpha=0.8)
        ah2.axhline(0.5, color=CORES['laranja'], linestyle=':', linewidth=2); ah2.axhline(-0.5, color=CORES['laranja'], linestyle=':', linewidth=2)
        ah2.fill_between(xh, -0.5, 0.5, alpha=0.2, color=CORES['verde'], label='Zona HIIT')
        ah2.scatter(xh, df_p['desvio'], c=ch, alpha=0.7, s=60, edgecolors='white', linewidths=1)
        ah2.plot(xh, df_p['desvio'], color=CORES['cinza'], alpha=0.4, linewidth=1)
        ah2.set_ylim(-3, 3); ah2.legend(loc='best', fontsize=9); ah2.grid(True, alpha=0.3)
        ah2.set_ylabel('Desvio (DP)', fontweight='bold'); ah2.set_title('Desvio LnrMSSD do Baseline', fontsize=12, fontweight='bold')
        ah2.set_xticks(list(xh)[::sh]); ah2.set_xticklabels([dh.iloc[i] for i in range(0, len(dh), sh)], rotation=45)
        plt.tight_layout(); st.pyplot(fig3); plt.close()
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum(); rec_n = (df_val['intens'] == 'Recuperação').sum(); total_n = len(df_val)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            c2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            c3.metric("Prescrição HOJE", '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT' else '🟠 Recuperação')

# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — WELLNESS
# ════════════════════════════════════════════════════════════════════════════════
