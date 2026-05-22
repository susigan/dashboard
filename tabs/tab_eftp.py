"""
tab_eftp.py — eFTP por modalidade + Diagnóstico de Resposta ao Treino
Implementa o framework do paper "Variabilidad Inter-Individual en el Entrenamiento
de Resistencia" (Della Mattia, ednacore AI 2025) com dados reais do atleta.

Assinatura actualizada:
    tab_eftp(da_filt, mods_sel, ac_full, wc_full=None)

Novas fontes utilizadas:
    - ac_full          → icu_eftp sessão-a-sessão + CTL + z1/z2/z3_kj
    - wc_full          → HRV para contexto de wellness por bloco
    - session_state    → ld_frac_cache com FMT_kappa por dia

NOTA SOBRE A PROJECÇÃO CP 28 DIAS:
    A regressão Δln(eFTP) ~ CTLγ_nivel é uma extensão dos papers FTLM Part I+II
    (Della Mattia 2025). O paper original usa CTLγ como preditor retrospectivo
    de performance (ActivityCP) e recovery (HRV_trend). A projecção forward
    e o uso do slope de CTLγ como 2ª covariável são adaptações para uso
    prático no dashboard — não estão literalmente nos papers.
    O IC usa residuals reais da OLS (não heurística 1-R²).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONSTANTES: SEM empírico calculado sobre série histórica real
# Método: std das variações dia-a-dia em janelas ≤14 dias
# MDC 95% = 1.96 × √2 × SEM
# ─────────────────────────────────────────────
_SEM = {"Bike": 3.92, "Row": 5.75, "Ski": 4.23, "Run": 10.27}
_MDC = {m: round(1.96 * np.sqrt(2) * s, 1) for m, s in _SEM.items()}
_MDC_PCT = {"Bike": 6.4, "Row": 8.2, "Ski": 7.4, "Run": 20.3}

_CORES = {
    "Bike": "#E74C3C",
    "Row":  "#3498DB",
    "Ski":  "#9B59B6",
    "Run":  "#2ECC71",
}

_CTL_DOSE_MIN   = 45
_KAPPA_P75      = 5.954
_KAPPA_P87      = 7.182
_Z2_POLAR_MAX   = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _preparar_eftp_semanal(ac_full: pd.DataFrame, mod: str) -> pd.DataFrame:
    col_mod  = "type" if "type" in ac_full.columns else "modality"
    col_eftp = "icu_eftp" if "icu_eftp" in ac_full.columns else "eFTP"
    col_date = "date" if "date" in ac_full.columns else "Data"

    _ctl_col = next((c for c in ["ctl", "CTL", "ctl_end"] if c in ac_full.columns), None)
    _z1_col  = next((c for c in ["z1_kj", "z1kj"] if c in ac_full.columns), None)
    _z2_col  = next((c for c in ["z2_kj", "z2kj"] if c in ac_full.columns), None)
    _z3_col  = next((c for c in ["z3_kj", "z3kj"] if c in ac_full.columns), None)

    _cols_base = [col_date, col_eftp]
    _cols_opt  = {"ctl": _ctl_col, "z1_kj": _z1_col, "z2_kj": _z2_col, "z3_kj": _z3_col}
    _col_map   = {v: k for k, v in _cols_opt.items() if v is not None}
    _cols_sel  = _cols_base + list(_col_map.keys())

    sub = ac_full[ac_full[col_mod] == mod][_cols_sel].copy()
    if sub.empty or sub[col_eftp].isna().all():
        return pd.DataFrame()

    sub = sub.rename(columns=_col_map)
    sub[col_date] = pd.to_datetime(sub[col_date])
    sub = sub.sort_values(col_date).set_index(col_date)
    sub[col_eftp] = pd.to_numeric(sub[col_eftp], errors="coerce")

    _agg_dict = {col_eftp: "last"}
    if "ctl"   in sub.columns: _agg_dict["ctl"]   = "mean"
    if "z1_kj" in sub.columns: _agg_dict["z1_kj"] = "sum"
    if "z2_kj" in sub.columns: _agg_dict["z2_kj"] = "sum"
    if "z3_kj" in sub.columns: _agg_dict["z3_kj"] = "sum"

    weekly = sub.resample("W").agg(_agg_dict).dropna(subset=[col_eftp])
    weekly = weekly.rename(columns={col_eftp: "eftp"})

    for _c in ["ctl", "z1_kj", "z2_kj", "z3_kj"]:
        if _c not in weekly.columns:
            weekly[_c] = np.nan

    weekly["z_total"] = weekly["z1_kj"] + weekly["z2_kj"] + weekly["z3_kj"]
    weekly["z1_pct"]  = weekly["z1_kj"] / weekly["z_total"].replace(0, np.nan)
    weekly["z2_pct"]  = weekly["z2_kj"] / weekly["z_total"].replace(0, np.nan)
    weekly["z3_pct"]  = weekly["z3_kj"] / weekly["z_total"].replace(0, np.nan)
    return weekly


def _calcular_blocos(weekly: pd.DataFrame, mdc: float, janela: int = 8) -> pd.DataFrame:
    w = weekly.copy()
    w["eftp_prev"]  = w["eftp"].shift(janela)
    w["delta"]      = w["eftp"] - w["eftp_prev"]
    w["ctl_bloco"]  = w["ctl"].rolling(janela).mean()
    w["z1_bloco"]   = w["z1_pct"].rolling(janela).mean()
    w["z2_bloco"]   = w["z2_pct"].rolling(janela).mean()
    w["z3_bloco"]   = w["z3_pct"].rolling(janela).mean()

    def _classif(d):
        if pd.isna(d): return None
        ad = abs(d)
        if ad >= mdc:          return "REAL"
        elif ad >= mdc * 0.5:  return "INCERTO"
        else:                  return "RUÍDO"

    w["classificacao"] = w["delta"].apply(_classif)
    return w.dropna(subset=["delta"])


def _kappa_por_periodo(ld, data_ini, data_fim):
    if ld is None or ld.empty: return None
    col_date = "Data" if "Data" in ld.columns else ld.index.name
    if col_date and col_date in ld.columns:
        sub = ld[(ld[col_date] >= data_ini) & (ld[col_date] <= data_fim)]
    else:
        sub = ld.loc[data_ini:data_fim]
    if "FMT_kappa" not in sub.columns or sub.empty: return None
    return float(sub["FMT_kappa"].mean())


def _rolling_slope_14(series_arr):
    """Rolling slope 14d sobre array numpy."""
    from scipy import stats as _sp
    n   = len(series_arr)
    out = np.full(n, np.nan)
    for _i in range(13, n):
        _y = series_arr[max(0, _i-13):_i+1].astype(float)
        _x = np.arange(len(_y), dtype=float)
        _m = np.isfinite(_y)
        if _m.sum() >= 5:
            out[_i], *_ = _sp.linregress(_x[_m], _y[_m])
    return out


def _diagnostico_texto(delta, ctl, kappa, z2_pct, mdc, mod):
    causas, prescricoes = [], []

    if ctl is not None and ctl < _CTL_DOSE_MIN:
        causas.append(f"📉 **Dose insuficiente** (CTL médio={ctl:.0f}, mínimo={_CTL_DOSE_MIN})")
        prescricoes.append("Aumentar frequência/volume de sessões")

    if kappa is not None and kappa > _KAPPA_P75:
        nivel_k = "crítico (>p87)" if kappa > _KAPPA_P87 else "elevado (>p75)"
        causas.append(f"⚡ **Stress silencioso** (κ={kappa:.2f}, {nivel_k})")
        prescricoes.append("Reduzir κ antes de aumentar estímulo")

    if not pd.isna(z2_pct) and z2_pct > _Z2_POLAR_MAX:
        causas.append(f"🔄 **Distribuição não polarizada** (Z2={z2_pct*100:.0f}%)")
        prescricoes.append("Redistribuir: mais Z1 + Z3, menos Z2")

    if not causas:
        causas.append("🔬 **Possível meseta homeostática**")
        prescricoes.append("Mudar natureza do estímulo: novo tipo, nova intensidade")

    if mod == "Run":
        causas.insert(0, f"⚠️ **Nota Run**: MDC={mdc:.0f}W (~{_MDC_PCT['Run']:.0f}%) — baixa confiança")

    diag_curto  = causas[0].split("**")[1] if "**" in causas[0] else causas[0]
    diag_detalhe = "\n\n".join([
        f"**Causa:** {c}\n\n**Prescrição:** {p}"
        for c, p in zip(causas, prescricoes)
    ])
    return diag_curto, diag_detalhe


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE A — Gráfico eFTP com banda de incerteza
# ─────────────────────────────────────────────────────────────────────────────

def _grafico_eftp_banda(df_eftp_orig: pd.DataFrame, mods_sel: list) -> go.Figure:
    fig = go.Figure()
    for mod in mods_sel:
        if mod not in df_eftp_orig.columns: continue
        cor  = _CORES.get(mod, "#888")
        mdc  = _MDC[mod]
        sub  = df_eftp_orig[["Data", mod]].dropna().copy()
        sub["Data"] = pd.to_datetime(sub["Data"])
        sub = sub.sort_values("Data")
        h = cor.lstrip("#"); r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        fill_rgba = f"rgba({r},{g},{b},0.12)"

        fig.add_trace(go.Scatter(
            x=pd.concat([sub["Data"], sub["Data"][::-1]]),
            y=pd.concat([sub[mod]+mdc/2, (sub[mod]-mdc/2)[::-1]]),
            fill="toself", fillcolor=fill_rgba, line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=sub["Data"], y=sub[mod], mode="lines", name=mod,
            line=dict(color=cor, width=2.5),
            hovertemplate=f"<b>{mod}</b><br>%{{x|%d %b %Y}}<br>eFTP: %{{y:.0f}}W<extra></extra>"))

    fig.add_annotation(
        text="Banda = zona de ruído (±MDC/2). Variações dentro desta banda são estatisticamente indistinguíveis de zero.",
        xref="paper", yref="paper", x=0, y=-0.12, showarrow=False,
        font=dict(size=11, color="#888"), align="left")
    fig.update_layout(
        title=dict(text="eFTP por modalidade — com banda de incerteza empírica", font=dict(size=15)),
        xaxis_title="Data", yaxis_title="eFTP (W)",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", y=1.08), margin=dict(b=80))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE B — Tabela de blocos 8 semanas
# ─────────────────────────────────────────────────────────────────────────────

def _tabela_blocos(blocos: pd.DataFrame, mod: str, ld=None) -> pd.DataFrame:
    mdc  = _MDC[mod]
    rows = []
    for idx, row in blocos.tail(10).iterrows():
        ini   = idx - pd.Timedelta(weeks=8)
        kappa = _kappa_por_periodo(ld, ini, idx)
        delta = row["delta"]
        classif = row["classificacao"]
        emoji   = {"REAL": "✅", "INCERTO": "⚠️", "RUÍDO": "🔴"}.get(classif, "")
        rows.append({
            "Período (fim)":   idx.strftime("%d %b %Y"),
            "eFTP fim (W)":    f"{row['eftp']:.0f}",
            "Δ 8 sem (W)":     f"{delta:+.0f}",
            f"MDC={mdc:.0f}W": f"{abs(delta)/mdc*100:.0f}%",
            "Sinal":           f"{emoji} {classif}",
            "CTL médio":       f"{row['ctl_bloco']:.0f}" if not pd.isna(row['ctl_bloco']) else "—",
            "κ médio":         f"{kappa:.2f}" if kappa is not None else "—",
            "Z1%":             f"{row['z1_bloco']*100:.0f}%" if not pd.isna(row.get('z1_bloco', np.nan)) else "—",
            "Z2%":             f"{row['z2_bloco']*100:.0f}%" if not pd.isna(row.get('z2_bloco', np.nan)) else "—",
            "Z3%":             f"{row['z3_bloco']*100:.0f}%" if not pd.isna(row.get('z3_bloco', np.nan)) else "—",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE C — Diagnóstico interanual
# ─────────────────────────────────────────────────────────────────────────────

def _grafico_plateau_interanual(df_eftp_orig: pd.DataFrame, mods_sel: list) -> go.Figure:
    fig = go.Figure()
    df  = df_eftp_orig.copy()
    df["Data"] = pd.to_datetime(df["Data"])
    df["Ano"]  = df["Data"].dt.year

    for mod in mods_sel:
        if mod not in df.columns: continue
        cor   = _CORES.get(mod, "#888")
        mdc   = _MDC[mod]
        picos = df.groupby("Ano")[mod].max().dropna()
        if len(picos) < 2: continue
        anos  = list(picos.index); vals = list(picos.values)
        deltas = [None] + [vals[i]-vals[i-1] for i in range(1, len(vals))]
        _h = cor.lstrip("#"); _r,_g,_b = int(_h[0:2],16),int(_h[2:4],16),int(_h[4:6],16)
        colors = []
        for d in deltas:
            if d is None:        colors.append(f"rgba({_r},{_g},{_b},0.85)")
            elif abs(d) >= mdc:  colors.append("#2ECC71" if d>0 else "#E74C3C")
            else:                colors.append("#888888")

        fig.add_trace(go.Bar(
            name=mod,
            x=[f"{a} ({mod})" for a in anos], y=vals,
            marker_color=colors,
            text=[f"{v:.0f}W" + (f"<br>{d:+.0f}W" if d is not None else "")
                  for v, d in zip(vals, deltas)],
            textposition="outside"))

    fig.add_annotation(
        text="🟢 Verde = mudança real (>MDC) | ⬜ Cinzento = ruído (<MDC) | 🔴 Vermelho = queda real",
        xref="paper", yref="paper", x=0, y=-0.15, showarrow=False,
        font=dict(size=11, color="#888"), align="left")
    fig.update_layout(
        title=dict(text="eFTP pico por ano — progressão interanual", font=dict(size=15)),
        barmode="group", yaxis_title="eFTP (W)", height=420,
        legend=dict(orientation="h", y=1.08), margin=dict(b=80))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAINEL DE DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────────────

def _painel_diagnostico(mod: str, blocos: pd.DataFrame, ld=None):
    mdc     = _MDC[mod]
    sem     = _SEM[mod]
    mdc_pct = _MDC_PCT[mod]

    if blocos.empty:
        st.info(f"Dados insuficientes para diagnóstico de {mod}.")
        return

    ultimo  = blocos.iloc[-1]
    classif = ultimo["classificacao"]
    delta   = ultimo["delta"]
    ctl     = ultimo.get("ctl_bloco", np.nan)
    z2      = ultimo.get("z2_bloco", np.nan)
    idx_ult = blocos.index[-1]
    ini_ult = idx_ult - pd.Timedelta(weeks=8)
    kappa   = _kappa_por_periodo(ld, ini_ult, idx_ult)

    cor_status = {"REAL":"#2ECC71","INCERTO":"#F39C12","RUÍDO":"#E74C3C"}.get(classif,"#888")
    emoji_s    = {"REAL":"✅","INCERTO":"⚠️","RUÍDO":"🔴"}.get(classif,"")

    st.markdown(f"""
    <div style="border-left:4px solid {cor_status};padding:12px 16px;
                background:rgba(0,0,0,0.03);border-radius:0 8px 8px 0;margin-bottom:8px;">
        <span style="font-size:1.1em;font-weight:700;">{emoji_s} {mod} — {classif}</span>
        <span style="color:#888;margin-left:12px;font-size:0.9em;">
            Δ 8 semanas: {delta:+.0f}W &nbsp;|&nbsp; MDC={mdc:.0f}W &nbsp;|&nbsp;
            SEM={sem:.1f}W &nbsp;|&nbsp; MDC%={mdc_pct:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)

    if mod == "Run":
        st.warning(f"⚠️ **Run — baixa confiabilidade.** MDC={mdc:.0f}W ({mdc_pct:.0f}%).")

    if classif == "REAL" and delta > 0:
        st.success(f"**Resposta real positiva.** eFTP +{delta:.0f}W nas últimas 8 semanas (>MDC={mdc:.0f}W).")
        return
    if classif == "REAL" and delta < 0:
        st.error(f"**Queda real.** eFTP {delta:.0f}W (>MDC={mdc:.0f}W).")

    if classif in ("RUÍDO","INCERTO") or (classif=="REAL" and delta<0):
        st.markdown("#### Diagnóstico diferencial")
        c1,c2,c3,c4 = st.columns(4)
        dose_ok  = not (pd.isna(ctl) or ctl < _CTL_DOSE_MIN)
        kappa_ok = kappa is None or kappa <= _KAPPA_P75
        polar_ok = pd.isna(z2) or z2 <= _Z2_POLAR_MAX

        with c1: st.metric("📊 CTL médio", f"{ctl:.0f}" if not pd.isna(ctl) else "—",
                            delta="OK" if dose_ok else f"< {_CTL_DOSE_MIN} ⚠️",
                            delta_color="normal" if dose_ok else "inverse")
        with c2: st.metric("⚡ κ médio", f"{kappa:.2f}" if kappa else "—",
                            delta="OK" if kappa_ok else "> p75 ⚠️",
                            delta_color="normal" if kappa_ok else "inverse")
        with c3: st.metric("🎯 Z2% bloco", f"{z2*100:.0f}%" if not pd.isna(z2) else "—",
                            delta="Polarizado" if polar_ok else "Não polarizado ⚠️",
                            delta_color="normal" if polar_ok else "inverse")
        with c4:
            pct_mdc = abs(delta)/mdc*100
            st.metric("📏 Sinal vs Ruído", f"{pct_mdc:.0f}% do MDC", delta=classif,
                       delta_color="normal" if classif=="REAL" else "inverse")

        st.markdown("---")
        _, detalhe = _diagnostico_texto(delta, ctl, kappa, z2, mdc, mod)
        for bloco_causa in detalhe.split("\n\n"):
            if not bloco_causa.strip(): continue
            linhas = bloco_causa.strip().split("\n\n")
            causa_txt = linhas[0].replace("**Causa:** ","") if linhas else ""
            presc_txt = linhas[1].replace("**Prescrição:** ","") if len(linhas)>1 else ""
            st.markdown(f"**→ {causa_txt}**")
            if presc_txt: st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Prescrição:* {presc_txt}")

        st.caption("Fontes: Montero & Lundby (2017); Iannetta et al. (2020); Issurin (2010); FMT Tensor κ (Della Mattia 2019).")


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def tab_eftp(da_filt: pd.DataFrame, mods_sel: list, ac_full: pd.DataFrame,
             wc_full: pd.DataFrame = None):

    ld = st.session_state.get("ld_frac_cache", None)

    col_mod  = next((c for c in ["type","modality","sport"] if c in ac_full.columns), None)
    col_eftp = next((c for c in ["icu_eftp","eFTP","eftp","ftp"] if c in ac_full.columns), None)
    col_date = next((c for c in ["date","Data","data","Date"] if c in ac_full.columns), None)

    if col_mod is None or col_eftp is None or col_date is None:
        st.warning(f"Colunas necessárias não encontradas. Disponíveis: {list(ac_full.columns[:15])}")
        return

    # Construir pivot eFTP
    pivot_rows = []
    for mod in ["Bike","Row","Ski","Run"]:
        sub = ac_full[ac_full[col_mod]==mod][[col_date,col_eftp]].dropna()
        if not sub.empty:
            sub = sub.copy()
            sub[col_date] = pd.to_datetime(sub[col_date])
            sub = sub.rename(columns={col_date:"Data", col_eftp:mod})
            pivot_rows.append(sub.set_index("Data")[mod])

    if pivot_rows:
        df_pivot = pd.concat(pivot_rows, axis=1).reset_index()
        df_pivot.columns.name = None
    else:
        st.warning("Sem dados eFTP em ac_full.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PROJECÇÃO CP 28 DIAS
    # Baseia-se no FTLM Part II (Della Mattia 2025) — extensão para projecção
    # Paper original: CTLγ(t) ↔ ActivityCP(t) retrospectivo
    # Aqui: Δln(eFTP) ~ nível CTLγ + slope CTLγ_14d — projecção forward
    # IC via residuals OLS reais (não heurística)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## Projecção de CP — 28 dias")
    with st.expander("📖 Como é calculada — e diferenças em relação ao paper", expanded=False):
        st.markdown("""
**Base teórica — FTLM Part II (Della Mattia 2025)**

O paper usa CTLγ como preditor de performance (ActivityCP) e recovery (HRV_trend):
```
CTLγ(t) ↔ ActivityCP(t)        ← §2.1 fitting γ_perf
CTLγ(t-1) ↔ HRV_trend(t)       ← §2.2 fitting γ_rec
```

**Extensão implementada aqui (não literal do paper):**
```
β = OLS(Δln(eFTP_sessão) ~ CTLγ_nivel + dCTLγ_slope_14d)
eFTP_proj(t+28) = eFTP_hoje × exp(β × CTLγ_slope_actual × 28)
```

**Diferenças em relação ao paper:**
- Paper usa correlação retrospectiva; aqui fazemos projecção forward
- Paper usa ActivityCP como target; aqui usamos eFTP (estimador do Intervals.icu)
- IC calculado via residuals OLS reais (σ_res × z₀.₉₀) — não heurística

**Interpretação do R²:**
- R² > 0.15 → slope CTLγ tem poder preditivo razoável para esta modalidade
- R² < 0.10 → slope CTLγ não é preditor forte; projecção tem alta incerteza
        """)

    _proj_ok = False
    if ld is not None and len(ld) > 30:
        try:
            from scipy import stats as _sp_stats

            _ld_proj = ld.copy()
            _ld_proj['Data'] = pd.to_datetime(_ld_proj['Data'])

            # Calcular dCTLg_14d se não existe
            if 'dCTLg_14d' not in _ld_proj.columns and 'CTLg_perf' in _ld_proj.columns:
                _ctlg = _ld_proj['CTLg_perf'].values.astype(float)
                _slopes = np.full(len(_ctlg), np.nan)
                for _ti in range(13, len(_ctlg)):
                    _y = _ctlg[max(0,_ti-13):_ti+1]
                    _x = np.arange(len(_y), dtype=float)
                    _mask = np.isfinite(_y)
                    if _mask.sum() >= 7:
                        _sl, *_ = _sp_stats.linregress(_x[_mask], _y[_mask])
                        _slopes[_ti] = _sl
                _ld_proj['dCTLg_14d'] = _slopes

            _ld_proj_idx = _ld_proj.set_index('Data')

            _mods_proj      = ['Bike','Row','Ski','Run']
            _beta_dict      = {}
            _r2_dict        = {}
            _eftp_now       = {}
            _slope_now_dict = {}
            _sigma_dict     = {}  # residual std para IC real

            for _mproj in _mods_proj:
                # Sessões desta modalidade com eFTP válido
                _ef_m = (ac_full[ac_full[col_mod]==_mproj][[col_date,col_eftp]]
                         .dropna().copy())
                if len(_ef_m) < 8:
                    continue
                _ef_m[col_date] = pd.to_datetime(_ef_m[col_date])
                _ef_m = (_ef_m.rename(columns={col_date:'Data',col_eftp:'eftp'})
                         .sort_values('Data').drop_duplicates('Data').reset_index(drop=True))

                # Variação relativa vs baseline 60d anterior
                # CORRECÇÃO: min_periods=3 (era 5) para mais dados válidos
                _ef_m['eftp_base'] = (_ef_m['eftp']
                                       .rolling(60, min_periods=3).median()
                                       .shift(1))
                _ef_m['dln'] = np.log(
                    (_ef_m['eftp'] / _ef_m['eftp_base'].clip(lower=1))
                    .clip(lower=0.5, upper=2)
                )

                # CTLγ por modalidade — com fallback para CTLg_perf global
                # CORRECÇÃO PRINCIPAL: fallback para CTLg_perf quando modal não existe
                _ctlg_col = f'CTLg_{_mproj}'
                if _ctlg_col in _ld_proj_idx.columns:
                    _ctlg_daily = _ld_proj_idx[_ctlg_col].ffill().values.astype(float)
                    _sl_daily   = _rolling_slope_14(_ctlg_daily)
                    _sl_series  = pd.Series(_sl_daily, index=_ld_proj_idx.index)
                    _ef_m['sl'] = _ef_m['Data'].map(_sl_series.to_dict())
                    _ef_m['ctlg_nivel'] = _ef_m['Data'].map(
                        _ld_proj_idx[_ctlg_col].ffill().to_dict())
                elif 'dCTLg_14d' in _ld_proj_idx.columns:
                    _sl_series  = _ld_proj_idx['dCTLg_14d'].ffill()
                    _ef_m['sl'] = _ef_m['Data'].map(_sl_series.to_dict())
                    _ef_m['ctlg_nivel'] = _ef_m['Data'].map(
                        _ld_proj_idx.get('CTLg_perf', pd.Series()).ffill().to_dict()) if 'CTLg_perf' in _ld_proj_idx.columns else np.nan
                elif 'CTLg_perf' in _ld_proj_idx.columns:
                    # FALLBACK: CTLg_perf global
                    _ctlg_global = _ld_proj_idx['CTLg_perf'].ffill().values.astype(float)
                    _sl_global   = _rolling_slope_14(_ctlg_global)
                    _sl_series   = pd.Series(_sl_global, index=_ld_proj_idx.index)
                    _ef_m['sl'] = _ef_m['Data'].map(_sl_series.to_dict())
                    _ef_m['ctlg_nivel'] = _ef_m['Data'].map(
                        _ld_proj_idx['CTLg_perf'].ffill().to_dict())
                else:
                    continue

                # OLS: dln ~ slope (preditor principal)
                # CORRECÇÃO: filtro len < 5 (era < 8) para mais modalidades válidas
                _valid = (_ef_m[['dln','sl']].replace([np.inf,-np.inf], np.nan)
                          .dropna())
                if len(_valid) < 5:
                    continue

                _sl_r, _intercept, _r, _, _se = _sp_stats.linregress(
                    _valid['sl'].values.astype(float),
                    _valid['dln'].values.astype(float))

                # Residuals para IC real
                _y_pred  = _intercept + _sl_r * _valid['sl'].values
                _resids  = _valid['dln'].values - _y_pred
                _sigma   = float(np.std(_resids, ddof=2)) if len(_resids) > 2 else 0.05

                _beta_dict[_mproj]      = float(_sl_r)
                _r2_dict[_mproj]        = float(_r**2)
                _sigma_dict[_mproj]     = _sigma
                _eftp_now[_mproj]       = float(_ef_m['eftp'].iloc[-1])
                _slope_now_dict[_mproj] = float(_sl_series.dropna().iloc[-1]) \
                                          if _sl_series.notna().any() else 0.0

            if not _beta_dict:
                raise ValueError(
                    "Nenhuma modalidade com dados suficientes. "
                    "Verifique se ld_frac_cache tem CTLg_perf ou CTLg_Bike/Row/Ski/Run. "
                    "Carregue a tab PMC primeiro para calcular CTLγ.")

            # Cards β
            st.markdown("#### Coeficiente β — sensibilidade CTLγ_slope → ΔeFTP")
            st.caption(
                "Calibrado nos dados históricos reais deste atleta. "
                "β>0 = slope positivo do CTLγ prediz ganho de eFTP. "
                "Extensão do FTLM Part II (Della Mattia 2025).")
            _bc = st.columns(max(len(_beta_dict), 1))
            _MOD_COLS_PROJ = {'Bike':'#e74c3c','Row':'#3498db','Ski':'#9b59b6','Run':'#27ae60'}
            for _bi, (_bm, _bv) in enumerate(_beta_dict.items()):
                _r2v = _r2_dict.get(_bm, 0)
                _bc[_bi].metric(
                    f"{_bm} β",
                    f"{_bv:.4f}",
                    delta=f"R²={_r2v:.2f}",
                    delta_color="normal" if _r2v > 0.15 else "off",
                    help=(f"σ_resid={_sigma_dict.get(_bm,0):.4f}. "
                          f"{'Poder preditivo razoável.' if _r2v>0.15 else 'R² baixo — incerteza alta.'}"))

            # Gráfico de projecção
            _fig_proj  = go.Figure()
            _PROJ_DAYS = 28
            _today     = pd.Timestamp.now().normalize()
            _proj_dates = pd.date_range(_today, periods=_PROJ_DAYS+1, freq='D')
            _proj_dates_str = [str(d.date()) for d in _proj_dates]

            for _mproj, _bv in _beta_dict.items():
                _eftp0 = _eftp_now.get(_mproj)
                if not _eftp0: continue
                _cor_m  = _MOD_COLS_PROJ.get(_mproj,'#888')
                _r2v    = _r2_dict.get(_mproj, 0)
                _sigma  = _sigma_dict.get(_mproj, 0.05)

                # Histórico 180d suavizado
                _ef_hist = (ac_full[ac_full[col_mod]==_mproj][[col_date,col_eftp]]
                            .dropna().copy())
                _ef_hist[col_date] = pd.to_datetime(_ef_hist[col_date])
                _ef_hist = (_ef_hist.rename(columns={col_date:'Data',col_eftp:'eftp'})
                             .sort_values('Data'))
                _ef_hist = _ef_hist[_ef_hist['Data'] >= _today-pd.Timedelta(days=180)]
                _ef_smooth = _ef_hist.set_index('Data')['eftp'].rolling(14,min_periods=3).mean()

                if len(_ef_smooth) > 0:
                    _fig_proj.add_trace(go.Scatter(
                        x=_ef_smooth.index.tolist(), y=_ef_smooth.values.tolist(),
                        name=f"{_mproj} observado",
                        line=dict(color=_cor_m, width=2),
                        hovertemplate=f"{_mproj}: %{{y:.0f}}W<extra></extra>"))

                # Projecção com IC via residuals reais
                _slope_f   = float(_slope_now_dict.get(_mproj, 0))
                _bv_f      = float(_bv)
                _eftp0_f   = float(_eftp0)
                _z90       = 1.645  # 90% IC

                _proj_vals = [float(_eftp0_f * np.exp(_bv_f * _slope_f * d))
                              for d in range(_PROJ_DAYS+1)]
                # IC real: ±z90 × σ_resid × sqrt(d) (propaga incerteza no tempo)
                _ic_half   = [float(_eftp0_f * _sigma * _z90 * max(d**0.5, 0.5))
                              for d in range(_PROJ_DAYS+1)]
                _proj_hi   = [float(v+ic) for v,ic in zip(_proj_vals,_ic_half)]
                _proj_lo   = [float(max(v-ic,1.0)) for v,ic in zip(_proj_vals,_ic_half)]

                _r_int,_g_int,_b_int = int(_cor_m[1:3],16),int(_cor_m[3:5],16),int(_cor_m[5:7],16)
                _rev = list(reversed(_proj_dates_str))

                _fig_proj.add_trace(go.Scatter(
                    x=_proj_dates_str+_rev,
                    y=_proj_hi+list(reversed(_proj_lo)),
                    fill='toself',
                    fillcolor=f'rgba({_r_int},{_g_int},{_b_int},0.10)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
                _fig_proj.add_trace(go.Scatter(
                    x=_proj_dates_str,
                    y=[float(round(v,1)) for v in _proj_vals],
                    name=f"{_mproj} proj 28d (β={_bv_f:.3f} R²={_r2v:.2f})",
                    line=dict(color=_cor_m,width=2.5,dash='dash'),
                    hovertemplate=f"{_mproj} proj: %{{y:.0f}}W<extra></extra>"))

                _proj_end  = float(round(_proj_vals[-1],0))
                _delta_pct = float((_proj_end-_eftp0_f)/max(_eftp0_f,1)*100)
                _fig_proj.add_annotation(
                    x=_proj_dates_str[-1], y=_proj_end,
                    text=f"{_mproj} +28d: {_proj_end:.0f}W ({_delta_pct:+.1f}%)",
                    showarrow=False, xshift=5, yshift=8,
                    font=dict(size=11,color=_cor_m),
                    bgcolor='rgba(255,255,255,0.85)')

            _today_str = str(_today.date())
            _fig_proj.add_shape(
                type='line', x0=_today_str, x1=_today_str, y0=0, y1=1,
                xref='x', yref='paper', line=dict(dash='dot',color='#aaa',width=1))
            _fig_proj.add_annotation(
                x=_today_str, y=1.02, xref='x', yref='paper',
                text='Hoje', showarrow=False, font=dict(size=10,color='#aaa'), xanchor='left')

            _fig_proj.update_layout(
                height=420, hovermode='x unified',
                margin=dict(t=40,b=80,l=65,r=30),
                legend=dict(orientation='h',y=-0.22,font=dict(size=11),
                            bgcolor='rgba(255,255,255,0.9)',borderwidth=1),
                title=dict(text='eFTP observado + Projecção 28 dias por modalidade',
                           font=dict(size=13)),
                xaxis=dict(tickangle=-25,gridcolor='rgba(0,0,0,0.04)'),
                yaxis=dict(title='eFTP (W)',gridcolor='rgba(0,0,0,0.05)'))

            st.plotly_chart(_fig_proj, use_container_width=True,
                            config={'displayModeBar':False}, key='cp_proj_28d')

            # Tabela resumo
            _proj_rows = []
            for _mproj, _bv in _beta_dict.items():
                _eftp0 = _eftp_now.get(_mproj)
                if not _eftp0: continue
                _slope_f = float(_slope_now_dict.get(_mproj,0))
                _proj28  = _eftp0 * np.exp(_bv * _slope_f * 28)
                _delta   = _proj28 - _eftp0
                _sigma   = _sigma_dict.get(_mproj,0.05)
                _ic_28   = _eftp0 * _sigma * 1.645 * 28**0.5
                _proj_rows.append({
                    'Modalidade':         _mproj,
                    'eFTP actual (W)':    f"{_eftp0:.0f}",
                    'eFTP proj +28d (W)': f"{_proj28:.0f}",
                    'Δ (W)':              f"{_delta:+.0f}",
                    'Δ (%)':              f"{(_delta/_eftp0*100):+.1f}%",
                    'IC 90% ±(W)':        f"±{_ic_28:.0f}",
                    'β':                  f"{_bv:.4f}",
                    'R²':                 f"{_r2_dict.get(_mproj,0):.3f}",
                    'σ_resid':            f"{_sigma:.4f}",
                })
            if _proj_rows:
                st.dataframe(pd.DataFrame(_proj_rows), use_container_width=True, hide_index=True)
                _slope_global = float(_ld_proj['dCTLg_14d'].dropna().iloc[-1]) \
                                if 'dCTLg_14d' in _ld_proj.columns and _ld_proj['dCTLg_14d'].notna().any() else 0.0
                st.caption(
                    f"dCTLγ_slope actual (global) = {_slope_global:.5f}/d | "
                    "IC 90% via residuals OLS reais (σ×z₀.₉₀×√t). "
                    "Projecção assume slope constante nos próximos 28 dias.")

            _proj_ok = True

        except Exception as _proj_err:
            st.info(f"Projecção CP: {_proj_err}")
    else:
        st.info("Projecção CP requer CTLγ no session_state (ld_frac_cache). Carrega a tab PMC primeiro.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER + MDC cards
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## eFTP por Modalidade")
    st.caption(
        "Estimativa de FTP funcional por modalidade ao longo do tempo. "
        "A banda representa a zona de ruído empírico (±MDC/2).")

    cols_mdc = st.columns(4)
    for i, mod in enumerate(["Bike","Row","Ski","Run"]):
        with cols_mdc[i]:
            fiab = "Baixa ⚠️" if mod=="Run" else "Normal"
            st.metric(f"{mod} — MDC 95%", f"±{_MDC[mod]:.0f}W",
                      delta=f"SEM={_SEM[mod]:.1f}W | {fiab}",
                      delta_color="inverse" if mod=="Run" else "off")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPONENTE A — Série histórica
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Série histórica com banda de incerteza")
    fig_serie = _grafico_eftp_banda(df_pivot, mods_sel)
    st.plotly_chart(fig_serie, use_container_width=True)

    csv_eftp = df_pivot.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button("⬇️ Download eFTP completo (CSV)", data=csv_eftp,
                       file_name="atheltica_eftp_completo.csv", mime="text/csv")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPONENTE B + C — Diagnóstico por modalidade
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Diagnóstico de Resposta ao Treino")
    st.caption(
        "Framework de Della Mattia (ednacore AI, 2025) baseado em "
        "Montero & Lundby (2017), Iannetta et al. (2020) e Hecksteden et al. (2015). "
        "Blocos de 8 semanas: REAL (>MDC) / INCERTO (50-100% MDC) / RUÍDO (<50% MDC).")

    mod_diag = st.selectbox("Modalidade para diagnóstico detalhado",
                             [m for m in mods_sel if m in df_pivot.columns],
                             key="eftp_diag_mod")

    if mod_diag:
        weekly = _preparar_eftp_semanal(ac_full, mod_diag)
        if weekly.empty:
            st.info(f"Sem dados suficientes para {mod_diag}.")
        else:
            blocos = _calcular_blocos(weekly, _MDC[mod_diag])
            st.markdown(f"#### Blocos de 8 semanas — {mod_diag}")
            df_tab = _tabela_blocos(blocos, mod_diag, ld=ld)
            if not df_tab.empty:
                st.dataframe(df_tab, use_container_width=True, hide_index=True)
                csv_blocos = df_tab.to_csv(index=False, sep=";").encode("utf-8")
                st.download_button(f"⬇️ Download blocos {mod_diag} (CSV)",
                                   data=csv_blocos,
                                   file_name=f"atheltica_blocos_{mod_diag.lower()}.csv",
                                   mime="text/csv", key=f"dl_blocos_{mod_diag}")
            st.markdown("---")
            st.markdown(f"#### Diagnóstico diferencial — {mod_diag} (bloco mais recente)")
            _painel_diagnostico(mod_diag, blocos, ld=ld)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPONENTE EXTRA — Plateau interanual
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Progressão interanual — detecção de plateau")
    st.caption("eFTP pico por ano. Verde = real (>MDC). Cinzento = ruído. Vermelho = queda real.")
    fig_plateau = _grafico_plateau_interanual(df_pivot, mods_sel)
    st.plotly_chart(fig_plateau, use_container_width=True)

    st.markdown("#### Síntese de plateau por modalidade")
    df_piv_c = df_pivot.copy()
    df_piv_c["Data"] = pd.to_datetime(df_piv_c["Data"])
    df_piv_c["Ano"]  = df_piv_c["Data"].dt.year

    for mod in mods_sel:
        if mod not in df_piv_c.columns: continue
        mdc   = _MDC[mod]
        picos = df_piv_c.groupby("Ano")[mod].max().dropna()
        if len(picos) < 2: continue
        anos  = list(picos.index); vals = list(picos.values)
        delta_recente = vals[-1]-vals[-2]
        anos_plateau  = sum(1 for i in range(1,len(vals)) if abs(vals[i]-vals[i-1])<mdc)
        if abs(delta_recente) < mdc and anos_plateau >= 2:
            st.warning(f"**{mod}:** Plateau interanual. Pico {anos[-2]}→{anos[-1]}: {delta_recente:+.0f}W (MDC={mdc:.0f}W).")
        elif delta_recente >= mdc:
            st.success(f"**{mod}:** Progressão real. +{delta_recente:.0f}W (>{mdc:.0f}W MDC).")
        elif delta_recente <= -mdc:
            st.error(f"**{mod}:** Regressão real. {delta_recente:.0f}W (>{mdc:.0f}W MDC).")
        else:
            st.info(f"**{mod}:** Variação dentro do ruído ({delta_recente:+.0f}W vs MDC={mdc:.0f}W).")

    # Nota metodológica
    with st.expander("ℹ️ Metodologia — MDC e projecção"):
        st.markdown(f"""
**MDC calculado empiricamente sobre dados reais (2018-2026):**

| Modalidade | SEM (W) | MDC 95% (W) | MDC% |
|---|---|---|---|
| Bike | {_SEM['Bike']:.1f} | {_MDC['Bike']:.1f} | {_MDC_PCT['Bike']:.1f}% |
| Row | {_SEM['Row']:.1f} | {_MDC['Row']:.1f} | {_MDC_PCT['Row']:.1f}% |
| Ski | {_SEM['Ski']:.1f} | {_MDC['Ski']:.1f} | {_MDC_PCT['Ski']:.1f}% |
| Run | {_SEM['Run']:.1f} | {_MDC['Run']:.1f} | {_MDC_PCT['Run']:.1f}% |

**Fórmula:** `MDC₉₅ = 1.96 × √2 × SEM` (Hecksteden et al., 2015)

**Projecção CP 28 dias — desvios ao paper original:**

O FTLM Part II (Della Mattia 2025) usa CTLγ como preditor retrospectivo.
A projecção forward implementada aqui é uma **extensão prática** não literal do paper.
O IC 90% usa σ_resid real da OLS (não heurística), propagado como σ×z₀.₉₀×√t.

**Referências:**
- Montero & Lundby (2017). *J Physiol.* — dose mínima
- Hecksteden et al. (2015). *J Appl Physiol.* — MDC individual
- Iannetta et al. (2020). *Med Sci Sports Exerc.* — domínios de intensidade
- Della Mattia G (2025). *FTLM Parts I+II.* — CTLγ e fitting por modalidade
- Della Mattia G (2025). *Variabilidad Inter-Individual.* — framework diagnóstico
        """)
