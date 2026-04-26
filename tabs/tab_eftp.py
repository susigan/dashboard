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
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONSTANTES: SEM empírico calculado sobre série histórica real
# Método: std das variações dia-a-dia em janelas ≤14 dias
# (representa ruído puro do estimador icu_eftp, não mudança real)
# MDC 95% = 1.96 × √2 × SEM
# ─────────────────────────────────────────────
_SEM = {"Bike": 3.92, "Row": 5.75, "Ski": 4.23, "Run": 10.27}
_MDC = {m: round(1.96 * np.sqrt(2) * s, 1) for m, s in _SEM.items()}

# Run tem MDC=28.5W sobre média 140W (20.3%) → praticamente não interpretável
_MDC_PCT = {"Bike": 6.4, "Row": 8.2, "Ski": 7.4, "Run": 20.3}

# Cores por modalidade (config.py)
_CORES = {
    "Bike": "#E74C3C",
    "Row":  "#3498DB",
    "Ski":  "#9B59B6",
    "Run":  "#2ECC71",
}

# Thresholds diagnóstico (baseados em dados do atleta, handoff Abril 2026)
_CTL_DOSE_MIN   = 45   # CTL médio abaixo = dose insuficiente
_KAPPA_P75      = 5.954
_KAPPA_P87      = 7.182
_Z2_POLAR_MAX   = 0.35  # Z2% acima → não polarizado


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _preparar_eftp_semanal(ac_full: pd.DataFrame, mod: str) -> pd.DataFrame:
    """
    Extrai eFTP semanal para uma modalidade a partir de ac_full.
    Usa coluna 'icu_eftp' filtrada por modality == mod.
    Retorna df semanal com last() value — o mais recente da semana.
    """
    col_mod = "type" if "type" in ac_full.columns else "modality"
    col_eftp = "icu_eftp" if "icu_eftp" in ac_full.columns else "eFTP"
    col_date = "date" if "date" in ac_full.columns else "Data"

    # Detectar colunas disponíveis em ac_full de forma defensiva
    # Nomes alternativos possíveis para as colunas
    _ctl_col  = next((c for c in ["ctl", "CTL", "ctl_end"] if c in ac_full.columns), None)
    _z1_col   = next((c for c in ["z1_kj", "z1kj", "trimp_z1", "trimp1"] if c in ac_full.columns), None)
    _z2_col   = next((c for c in ["z2_kj", "z2kj", "trimp_z2", "trimp2"] if c in ac_full.columns), None)
    _z3_col   = next((c for c in ["z3_kj", "z3kj", "trimp_z3", "trimp3"] if c in ac_full.columns), None)

    # Colunas base obrigatórias
    _cols_base = [col_date, col_eftp]
    # Colunas opcionais — só adicionar se existirem
    _cols_opt = {
        "ctl":   _ctl_col,
        "z1_kj": _z1_col,
        "z2_kj": _z2_col,
        "z3_kj": _z3_col,
    }
    _col_map = {v: k for k, v in _cols_opt.items() if v is not None}
    _cols_sel = _cols_base + [c for c in _col_map.keys()]

    sub = ac_full[ac_full[col_mod] == mod][_cols_sel].copy()
    if sub.empty or sub[col_eftp].isna().all():
        return pd.DataFrame()

    # Renomear colunas opcionais para nomes canónicos
    sub = sub.rename(columns=_col_map)

    sub[col_date] = pd.to_datetime(sub[col_date])
    sub = sub.sort_values(col_date).set_index(col_date)
    sub[col_eftp] = pd.to_numeric(sub[col_eftp], errors="coerce")

    # Agregação semanal — só colunas que existem
    _agg_dict = {col_eftp: "last"}
    if "ctl"   in sub.columns: _agg_dict["ctl"]   = "mean"
    if "z1_kj" in sub.columns: _agg_dict["z1_kj"] = "sum"
    if "z2_kj" in sub.columns: _agg_dict["z2_kj"] = "sum"
    if "z3_kj" in sub.columns: _agg_dict["z3_kj"] = "sum"

    weekly = sub.resample("W").agg(_agg_dict).dropna(subset=[col_eftp])
    weekly = weekly.rename(columns={col_eftp: "eftp"})

    # Garantir colunas mesmo que não existam (com NaN)
    for _c in ["ctl", "z1_kj", "z2_kj", "z3_kj"]:
        if _c not in weekly.columns:
            weekly[_c] = np.nan

    weekly["z_total"] = weekly["z1_kj"] + weekly["z2_kj"] + weekly["z3_kj"]
    weekly["z1_pct"] = weekly["z1_kj"] / weekly["z_total"].replace(0, np.nan)
    weekly["z2_pct"] = weekly["z2_kj"] / weekly["z_total"].replace(0, np.nan)
    weekly["z3_pct"] = weekly["z3_kj"] / weekly["z_total"].replace(0, np.nan)
    return weekly


def _calcular_blocos(weekly: pd.DataFrame, mdc: float, janela: int = 8) -> pd.DataFrame:
    """
    Calcula blocos de N semanas (rolante).
    Para cada semana t: compara eftp[t] vs eftp[t-janela].
    Classifica: REAL / INCERTO / RUÍDO.
    """
    w = weekly.copy()
    w["eftp_prev"]  = w["eftp"].shift(janela)
    w["delta"]      = w["eftp"] - w["eftp_prev"]
    w["ctl_bloco"]  = w["ctl"].rolling(janela).mean()
    w["z1_bloco"]   = w["z1_pct"].rolling(janela).mean()
    w["z2_bloco"]   = w["z2_pct"].rolling(janela).mean()
    w["z3_bloco"]   = w["z3_pct"].rolling(janela).mean()

    def _classif(d):
        if pd.isna(d):
            return None
        ad = abs(d)
        if ad >= mdc:
            return "REAL"
        elif ad >= mdc * 0.5:
            return "INCERTO"
        else:
            return "RUÍDO"

    w["classificacao"] = w["delta"].apply(_classif)
    return w.dropna(subset=["delta"])


def _kappa_por_periodo(ld: pd.DataFrame, data_ini, data_fim) -> float | None:
    """
    Retorna κ médio de um período. ld vem do session_state['ld_frac_cache'].
    """
    if ld is None or ld.empty:
        return None
    col_date = "Data" if "Data" in ld.columns else ld.index.name
    if col_date and col_date in ld.columns:
        sub = ld[(ld[col_date] >= data_ini) & (ld[col_date] <= data_fim)]
    else:
        sub = ld.loc[data_ini:data_fim]
    if "FMT_kappa" not in sub.columns or sub.empty:
        return None
    return float(sub["FMT_kappa"].mean())


def _diagnostico_texto(delta: float, ctl: float, kappa, z2_pct: float,
                       mdc: float, mod: str) -> tuple[str, str]:
    """
    Retorna (diagnóstico_curto, diagnóstico_detalhe) baseado no framework do paper.
    Só chamado quando classificacao == 'RUÍDO' ou INCERTO sem crescimento.
    """
    causas = []
    prescricoes = []

    # 1. Dose (Montero & Lundby 2017)
    if ctl is not None and ctl < _CTL_DOSE_MIN:
        causas.append(f"📉 **Dose insuficiente** (CTL médio={ctl:.0f}, mínimo recomendado={_CTL_DOSE_MIN})")
        prescricoes.append("Aumentar frequência/volume de sessões antes de mudar qualidade")

    # 2. Qualidade do estímulo via κ (stress silencioso)
    if kappa is not None and kappa > _KAPPA_P75:
        nivel_k = "crítico (>p87)" if kappa > _KAPPA_P87 else "elevado (>p75)"
        causas.append(f"⚡ **Stress silencioso** (κ={kappa:.2f}, {nivel_k})")
        prescricoes.append("SNA em modo defensivo → adaptações suprimidas. Reduzir κ antes de aumentar estímulo")

    # 3. Distribuição de intensidade (Iannetta et al. 2020)
    if not pd.isna(z2_pct) and z2_pct > _Z2_POLAR_MAX:
        causas.append(f"🔄 **Distribuição não polarizada** (Z2={z2_pct*100:.0f}% — zona 'moderada dura')")
        prescricoes.append("Redistribuir: mais Z1 (<4 RPE) + mais Z3 (≥7 RPE), menos Z2")

    # 4. Meseta homeostática (Issurin / mTOR)
    if not causas:
        causas.append("🔬 **Possível meseta homeostática** — estímulo familiar, sinalização mTOR/PGC-1α saturada")
        prescricoes.append("Mudar natureza do estímulo: novo tipo de sessão, nova intensidade-alvo, bloco neuromuscular")

    # Run — aviso especial
    if mod == "Run":
        causas.insert(0, f"⚠️ **Nota Run**: MDC={mdc:.0f}W sobre média ~140W (20%). eFTP Run tem baixa confiança — requer teste controlado")

    diag_curto = causas[0].split("**")[1] if "**" in causas[0] else causas[0]
    diag_detalhe = "\n\n".join([f"**Causa:** {c}\n\n**Prescrição:** {p}"
                                 for c, p in zip(causas, prescricoes)])
    return diag_curto, diag_detalhe


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE A — Gráfico eFTP com banda de incerteza
# ─────────────────────────────────────────────────────────────────────────────

def _grafico_eftp_banda(df_eftp_orig: pd.DataFrame, mods_sel: list) -> go.Figure:
    """
    Gráfico da série eFTP por modalidade com banda ±MDC/2 (zona de ruído).
    df_eftp_orig: o DataFrame pivot original (Data, Bike, Row, Ski, Run).
    """
    fig = go.Figure()

    for mod in mods_sel:
        if mod not in df_eftp_orig.columns:
            continue
        cor = _CORES.get(mod, "#888")
        mdc = _MDC[mod]
        sub = df_eftp_orig[["Data", mod]].dropna().copy()
        sub["Data"] = pd.to_datetime(sub["Data"])
        sub = sub.sort_values("Data")

        # Converter hex para rgba para a banda de incerteza
        h = cor.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        fill_rgba = f"rgba({r},{g},{b},0.12)"

        # Banda de ruído (±MDC/2 em volta da série)
        fig.add_trace(go.Scatter(
            x=pd.concat([sub["Data"], sub["Data"][::-1]]),
            y=pd.concat([sub[mod] + mdc/2, (sub[mod] - mdc/2)[::-1]]),
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"{mod} banda ruído",
        ))

        # Linha principal
        fig.add_trace(go.Scatter(
            x=sub["Data"],
            y=sub[mod],
            mode="lines",
            name=mod,
            line=dict(color=cor, width=2.5),
            hovertemplate=f"<b>{mod}</b><br>%{{x|%d %b %Y}}<br>eFTP: %{{y:.0f}}W<extra></extra>",
        ))

    # Linha MDC annotation — legenda
    fig.add_annotation(
        text="Banda cinzenta = zona de ruído (±MDC/2). Mudanças dentro desta banda são estatisticamente indistinguíveis de zero.",
        xref="paper", yref="paper",
        x=0, y=-0.12, showarrow=False,
        font=dict(size=11, color="#888"),
        align="left",
    )

    fig.update_layout(
        title=dict(text="eFTP por modalidade — com banda de incerteza empírica", font=dict(size=15)),
        xaxis_title="Data",
        yaxis_title="eFTP (W)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", y=1.08),
        margin=dict(b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE B — Tabela de blocos 8 semanas
# ─────────────────────────────────────────────────────────────────────────────

def _tabela_blocos(blocos: pd.DataFrame, mod: str, ld=None) -> pd.DataFrame:
    """
    Prepara tabela de blocos para st.dataframe().
    Mostra últimos 8 blocos com: período, eFTP, Δ, vs MDC, dose CTL, κ, Z1/Z2/Z3, classificação.
    """
    mdc = _MDC[mod]
    rows = []
    for idx, row in blocos.tail(10).iterrows():
        ini = idx - pd.Timedelta(weeks=8)
        kappa = _kappa_por_periodo(ld, ini, idx)

        delta = row["delta"]
        classif = row["classificacao"]
        emoji = {"REAL": "✅", "INCERTO": "⚠️", "RUÍDO": "🔴"}.get(classif, "")

        rows.append({
            "Período (fim)":     idx.strftime("%d %b %Y"),
            "eFTP fim (W)":      f"{row['eftp']:.0f}",
            "Δ 8 sem (W)":       f"{delta:+.0f}",
            f"MDC={mdc:.0f}W":   f"{abs(delta)/mdc*100:.0f}%",
            "Sinal":             f"{emoji} {classif}",
            "CTL médio":         f"{row['ctl_bloco']:.0f}" if not pd.isna(row['ctl_bloco']) else "—",
            "κ médio":           f"{kappa:.2f}" if kappa is not None else "—",
            "Z1%":               f"{row['z1_bloco']*100:.0f}%" if not pd.isna(row.get('z1_bloco', np.nan)) else "—",
            "Z2%":               f"{row['z2_bloco']*100:.0f}%" if not pd.isna(row.get('z2_bloco', np.nan)) else "—",
            "Z3%":               f"{row['z3_bloco']*100:.0f}%" if not pd.isna(row.get('z3_bloco', np.nan)) else "—",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE C — Diagnóstico interanual (plateau)
# ─────────────────────────────────────────────────────────────────────────────

def _grafico_plateau_interanual(df_eftp_orig: pd.DataFrame, mods_sel: list) -> go.Figure:
    """
    Barras de eFTP pico por ano, com classificação REAL/RUÍDO vs MDC.
    Implementa directamente Paper §5 "Ausência de Periodização Longitudinal".
    """
    fig = go.Figure()
    df = df_eftp_orig.copy()
    df["Data"] = pd.to_datetime(df["Data"])
    df["Ano"] = df["Data"].dt.year

    for mod in mods_sel:
        if mod not in df.columns:
            continue
        cor = _CORES.get(mod, "#888")
        mdc = _MDC[mod]
        picos = df.groupby("Ano")[mod].max().dropna()
        if len(picos) < 2:
            continue

        anos = list(picos.index)
        vals = list(picos.values)
        deltas = [None] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
        # Converter cor base para rgba (Plotly não aceita hex+alpha)
        _h = cor.lstrip("#")
        _r, _g, _b = int(_h[0:2],16), int(_h[2:4],16), int(_h[4:6],16)
        cor_rgba_base = f"rgba({_r},{_g},{_b},0.85)"

        colors = []
        for d in deltas:
            if d is None:
                colors.append(cor_rgba_base)
            elif abs(d) >= mdc:
                colors.append("#2ECC71" if d > 0 else "#E74C3C")
            else:
                colors.append("#888888")

        fig.add_trace(go.Bar(
            name=mod,
            x=[f"{a} ({mod})" for a in anos],
            y=vals,
            marker_color=colors,
            text=[f"{v:.0f}W" + (f"<br>{d:+.0f}W" if d is not None else "")
                  for v, d in zip(vals, deltas)],
            textposition="outside",
            hovertemplate=(
                f"<b>{mod}</b> %{{x}}<br>"
                f"eFTP pico: %{{y:.0f}}W<extra></extra>"
            ),
        ))

    fig.add_annotation(
        text="🟢 Verde = mudança real (>MDC) | ⬜ Cinzento = ruído (<MDC) | 🔴 Vermelho = queda real",
        xref="paper", yref="paper",
        x=0, y=-0.15, showarrow=False,
        font=dict(size=11, color="#888"), align="left",
    )

    fig.update_layout(
        title=dict(text="eFTP pico por ano — progressão interanual (verde=real, cinzento=ruído)", font=dict(size=15)),
        barmode="group",
        yaxis_title="eFTP (W)",
        height=420,
        legend=dict(orientation="h", y=1.08),
        margin=dict(b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAINEL DE DIAGNÓSTICO COMPLETO POR MODALIDADE
# ─────────────────────────────────────────────────────────────────────────────

def _painel_diagnostico(mod: str, blocos: pd.DataFrame, ld=None):
    """
    Caixa de diagnóstico diferencial para uma modalidade.
    Analisa o bloco mais recente e produz texto accionável.
    """
    mdc = _MDC[mod]
    sem = _SEM[mod]
    mdc_pct = _MDC_PCT[mod]

    if blocos.empty:
        st.info(f"Dados insuficientes para diagnóstico de {mod}.")
        return

    ultimo = blocos.iloc[-1]
    classif = ultimo["classificacao"]
    delta = ultimo["delta"]
    ctl = ultimo.get("ctl_bloco", np.nan)
    z2 = ultimo.get("z2_bloco", np.nan)
    idx_ult = blocos.index[-1]
    ini_ult = idx_ult - pd.Timedelta(weeks=8)
    kappa = _kappa_por_periodo(ld, ini_ult, idx_ult)

    # Header do card
    cor_status = {"REAL": "#2ECC71", "INCERTO": "#F39C12", "RUÍDO": "#E74C3C"}.get(classif, "#888")
    emoji_status = {"REAL": "✅", "INCERTO": "⚠️", "RUÍDO": "🔴"}.get(classif, "")

    st.markdown(f"""
    <div style="border-left: 4px solid {cor_status}; padding: 12px 16px; 
                background: rgba(0,0,0,0.03); border-radius: 0 8px 8px 0; margin-bottom: 8px;">
        <span style="font-size: 1.1em; font-weight: 700;">{emoji_status} {mod} — {classif}</span>
        <span style="color: #888; margin-left: 12px; font-size: 0.9em;">
            Δ 8 semanas: {delta:+.0f}W &nbsp;|&nbsp; MDC={mdc:.0f}W &nbsp;|&nbsp; 
            SEM={sem:.1f}W &nbsp;|&nbsp; MDC%={mdc_pct:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Aviso especial Run
    if mod == "Run":
        st.warning(
            f"⚠️ **Run — baixa confiabilidade do eFTP estimado.** "
            f"MDC = {mdc:.0f}W sobre média histórica ~140W ({mdc_pct:.0f}%). "
            f"Qualquer variação abaixo de {mdc:.0f}W é estatisticamente ruído. "
            f"Recomendado: teste controlado (TT de 20-30 min) para medir eFTP real."
        )

    if classif == "REAL" and delta > 0:
        st.success(
            f"**Resposta real positiva confirmada.** eFTP aumentou {delta:+.0f}W nas últimas 8 semanas "
            f"(>{mdc:.0f}W MDC). O estímulo está a gerar adaptação mensurável."
        )
        return

    if classif == "REAL" and delta < 0:
        st.error(
            f"**Queda real confirmada.** eFTP diminuiu {delta:.0f}W nas últimas 8 semanas "
            f"(>{mdc:.0f}W MDC). Não é ruído — requer diagnóstico activo."
        )

    # Diagnóstico diferencial (para RUÍDO, INCERTO, ou queda REAL)
    if classif in ("RUÍDO", "INCERTO") or (classif == "REAL" and delta < 0):
        st.markdown("#### Diagnóstico diferencial")

        col1, col2, col3, col4 = st.columns(4)

        # 1. Dose
        dose_ok = not (pd.isna(ctl) or ctl < _CTL_DOSE_MIN)
        with col1:
            st.metric(
                "📊 Dose (CTL médio)",
                f"{ctl:.0f}" if not pd.isna(ctl) else "—",
                delta=f"{'OK' if dose_ok else f'< {_CTL_DOSE_MIN} ⚠️'}",
                delta_color="normal" if dose_ok else "inverse",
            )

        # 2. κ stress silencioso
        kappa_ok = kappa is None or kappa <= _KAPPA_P75
        with col2:
            st.metric(
                "⚡ κ médio bloco",
                f"{kappa:.2f}" if kappa is not None else "—",
                delta=f"{'OK' if kappa_ok else f'> p75 ⚠️'}",
                delta_color="normal" if kappa_ok else "inverse",
            )

        # 3. Polarização Z2
        polar_ok = pd.isna(z2) or z2 <= _Z2_POLAR_MAX
        with col3:
            st.metric(
                "🎯 Z2% bloco",
                f"{z2*100:.0f}%" if not pd.isna(z2) else "—",
                delta=f"{'Polarizado' if polar_ok else 'Não polarizado ⚠️'}",
                delta_color="normal" if polar_ok else "inverse",
            )

        # 4. Classificação MDC
        with col4:
            pct_mdc = abs(delta) / mdc * 100
            st.metric(
                "📏 Sinal vs Ruído",
                f"{pct_mdc:.0f}% do MDC",
                delta=classif,
                delta_color="normal" if classif == "REAL" else "inverse",
            )

        # Texto de diagnóstico principal
        st.markdown("---")
        _, detalhe = _diagnostico_texto(delta, ctl, kappa, z2, mdc, mod)

        causas_lista = detalhe.split("\n\n")
        for bloco_causa in causas_lista:
            if bloco_causa.strip():
                linhas = bloco_causa.strip().split("\n\n")
                causa_txt = linhas[0].replace("**Causa:** ", "") if linhas else ""
                presc_txt = linhas[1].replace("**Prescrição:** ", "") if len(linhas) > 1 else ""
                st.markdown(f"**→ {causa_txt}**")
                if presc_txt:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Prescrição:* {presc_txt}")

        # Referências do paper
        st.caption(
            "Fontes: Montero & Lundby (2017) — dose; Iannetta et al. (2020) — prescrição por domínio; "
            "Issurin (2010) — meseta homeostática; FMT Tensor κ (Della Mattia 2019) — stress silencioso."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL — tab_eftp()
# ─────────────────────────────────────────────────────────────────────────────

def tab_eftp(da_filt: pd.DataFrame, mods_sel: list, ac_full: pd.DataFrame,
             wc_full: pd.DataFrame = None):
    """
    Tab eFTP completa com Diagnóstico de Resposta ao Treino.

    Parâmetros:
        da_filt   : actividades filtradas pelo sidebar
        mods_sel  : modalidades seleccionadas
        ac_full   : actividades completas (sem filtro de data) — necessário para histórico
        wc_full   : wellness completo (opcional — usado para contexto HRV)
    """

    # ── Carregar κ do session_state ──
    ld = st.session_state.get("ld_frac_cache", None)

    # ── Construir DataFrame pivot eFTP a partir de ac_full ──
    # (replica o que o CSV atheltica_eftp.csv contém, mas em tempo real)
    col_mod  = next((c for c in ["type", "modality", "sport"] if c in ac_full.columns), None)
    col_eftp = next((c for c in ["icu_eftp", "eFTP", "eftp", "ftp"] if c in ac_full.columns), None)
    col_date = next((c for c in ["date", "Data", "data", "Date"] if c in ac_full.columns), None)

    if col_mod is None or col_eftp is None or col_date is None:
        st.warning(
            f"Colunas necessárias não encontradas em ac_full. "
            f"Disponíveis: {list(ac_full.columns[:15])}. "
            f"Necessárias: type/modality, icu_eftp/eFTP, date/Data"
        )
        return

    pivot_rows = []
    for mod in ["Bike", "Row", "Ski", "Run"]:
        sub = ac_full[ac_full[col_mod] == mod][[col_date, col_eftp]].dropna()
        if not sub.empty:
            sub = sub.copy()
            sub[col_date] = pd.to_datetime(sub[col_date])
            sub = sub.rename(columns={col_date: "Data", col_eftp: mod})
            pivot_rows.append(sub.set_index("Data")[mod])

    if pivot_rows:
        df_pivot = pd.concat(pivot_rows, axis=1).reset_index()
        df_pivot.columns.name = None
    else:
        st.warning(
            f"Sem dados eFTP em ac_full para modalidades Bike/Row/Ski/Run. "
            f"Coluna '{col_mod}' valores únicos: {list(ac_full[col_mod].unique()[:10])}"
        )
        return

    # ═══════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════
    st.markdown("## eFTP por Modalidade")
    st.caption(
        "Estimativa de FTP funcional por modalidade ao longo do tempo. "
        "A banda cinzenta representa a zona de ruído empírico (±MDC/2) — "
        "variações dentro desta zona são estatisticamente indistinguíveis de zero."
    )

    # ── MDC por modalidade — cards informativos ──
    cols_mdc = st.columns(4)
    for i, mod in enumerate(["Bike", "Row", "Ski", "Run"]):
        with cols_mdc[i]:
            fiab = "Baixa ⚠️" if mod == "Run" else "Normal"
            st.metric(
                f"{mod} — MDC 95%",
                f"±{_MDC[mod]:.0f}W",
                delta=f"SEM={_SEM[mod]:.1f}W | Fiab.:{fiab}",
                delta_color="inverse" if mod == "Run" else "off",
                help=(
                    f"Mínima Diferença Detectável a 95% de confiança para {mod}.\n\n"
                    f"Calculado empiricamente sobre {545 if mod=='Bike' else 449 if mod=='Row' else 196 if mod=='Ski' else 202} "
                    f"observações reais (2018-2026).\n\n"
                    f"Fórmula: MDC = 1.96 × √2 × SEM = {_MDC[mod]:.1f}W\n\n"
                    f"Qualquer variação abaixo de {_MDC[mod]:.0f}W pode ser ruído do estimador icu_eftp."
                )
            )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # COMPONENTE A — Gráfico com banda de incerteza
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Série histórica com banda de incerteza")
    fig_serie = _grafico_eftp_banda(df_pivot, mods_sel)
    st.plotly_chart(fig_serie, use_container_width=True)

    # ── Download CSV eFTP completo ──
    csv_eftp = df_pivot.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button(
        "⬇️ Download eFTP completo (CSV)",
        data=csv_eftp,
        file_name="atheltica_eftp_completo.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # COMPONENTE B + C — Diagnóstico por modalidade
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Diagnóstico de Resposta ao Treino")
    st.caption(
        "Implementa o framework de Della Mattia (ednacore AI, 2025) baseado em "
        "Montero & Lundby (2017), Iannetta et al. (2020) e Hecksteden et al. (2015). "
        "Cada bloco de 8 semanas é classificado como REAL (>MDC), INCERTO (50-100% MDC) "
        "ou RUÍDO (<50% MDC). O diagnóstico diferencial integra dose (CTL), "
        "qualidade do estímulo (κ tensor) e distribuição de domínios (Z1/Z2/Z3)."
    )

    # Selector de modalidade para diagnóstico
    mod_diag = st.selectbox(
        "Modalidade para diagnóstico detalhado",
        [m for m in mods_sel if m in df_pivot.columns],
        key="eftp_diag_mod",
    )

    if mod_diag:
        weekly = _preparar_eftp_semanal(ac_full, mod_diag)

        if weekly.empty:
            st.info(f"Sem dados suficientes de ac_full para {mod_diag}.")
        else:
            blocos = _calcular_blocos(weekly, _MDC[mod_diag])

            # Tabela de blocos
            st.markdown(f"#### Blocos de 8 semanas — {mod_diag}")
            df_tab = _tabela_blocos(blocos, mod_diag, ld=ld)
            if not df_tab.empty:
                st.dataframe(df_tab, use_container_width=True, hide_index=True)

                # Download tabela
                csv_blocos = df_tab.to_csv(index=False, sep=";").encode("utf-8")
                st.download_button(
                    f"⬇️ Download blocos {mod_diag} (CSV)",
                    data=csv_blocos,
                    file_name=f"atheltica_blocos_{mod_diag.lower()}.csv",
                    mime="text/csv",
                    key=f"dl_blocos_{mod_diag}",
                )

            st.markdown("---")

            # Painel de diagnóstico
            st.markdown(f"#### Diagnóstico diferencial — {mod_diag} (bloco mais recente)")
            _painel_diagnostico(mod_diag, blocos, ld=ld)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # COMPONENTE EXTRA — Plateau interanual
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Progressão interanual — detecção de plateau")
    st.caption(
        "eFTP pico por ano por modalidade. Cor verde = mudança real (>MDC). "
        "Cor cinzenta = ruído (<MDC, variação não significativa). "
        "Cor vermelha = queda real confirmada. "
        "Implementa Paper §5: 'tecto da temporada anterior como piso da seguinte'."
    )
    fig_plateau = _grafico_plateau_interanual(df_pivot, mods_sel)
    st.plotly_chart(fig_plateau, use_container_width=True)

    # Insight automático plateau
    st.markdown("#### Síntese de plateau por modalidade")
    df_pivot_c = df_pivot.copy()
    df_pivot_c["Data"] = pd.to_datetime(df_pivot_c["Data"])
    df_pivot_c["Ano"] = df_pivot_c["Data"].dt.year

    for mod in mods_sel:
        if mod not in df_pivot_c.columns:
            continue
        mdc = _MDC[mod]
        picos = df_pivot_c.groupby("Ano")[mod].max().dropna()
        if len(picos) < 2:
            continue
        anos = list(picos.index)
        vals = list(picos.values)
        # Verificar últimos 2 anos
        delta_recente = vals[-1] - vals[-2]
        anos_plateau = sum(1 for i in range(1, len(vals)) if abs(vals[i] - vals[i-1]) < mdc)

        if abs(delta_recente) < mdc and anos_plateau >= 2:
            st.warning(
                f"**{mod}:** Plateau interanual detectado. "
                f"Pico {anos[-2]}: {vals[-2]:.0f}W → Pico {anos[-1]}: {vals[-1]:.0f}W "
                f"(Δ={delta_recente:+.0f}W, MDC={mdc:.0f}W). "
                f"Mudança de natureza do estímulo indicada (Paper §1 — meseta homeostática)."
            )
        elif delta_recente >= mdc:
            st.success(
                f"**{mod}:** Progressão interanual real. "
                f"{anos[-2]}→{anos[-1]}: +{delta_recente:.0f}W (>{mdc:.0f}W MDC)."
            )
        elif delta_recente <= -mdc:
            st.error(
                f"**{mod}:** Regressão interanual real. "
                f"{anos[-2]}→{anos[-1]}: {delta_recente:.0f}W (>{mdc:.0f}W MDC)."
            )
        else:
            st.info(
                f"**{mod}:** Variação recente dentro do ruído "
                f"({delta_recente:+.0f}W vs MDC={mdc:.0f}W). Monitorizar."
            )

    # ═══════════════════════════════════════════════════════════
    # NOTA METODOLÓGICA
    # ═══════════════════════════════════════════════════════════
    with st.expander("ℹ️ Metodologia — Como foi calculado o MDC"):
        st.markdown(f"""
**Erro Típico de Medição (SEM) — calculado empiricamente sobre dados reais**

O SEM foi estimado a partir da variabilidade intrínseca do estimador `icu_eftp` 
do Intervals.icu, usando a série histórica do próprio atleta (2018-2026).

**Método:** desvio padrão das variações dia-a-dia (`Δ eFTP`) em janelas de ≤14 dias,
que representam ruído puro do estimador (sem mudança fisiológica real esperada).

| Modalidade | N obs | SEM (W) | MDC 95% (W) | MDC% da média |
|---|---|---|---|---|
| Bike | 521 | {_SEM['Bike']:.1f} | {_MDC['Bike']:.1f} | {_MDC_PCT['Bike']:.1f}% |
| Row | 449 | {_SEM['Row']:.1f} | {_MDC['Row']:.1f} | {_MDC_PCT['Row']:.1f}% |
| Ski | 196 | {_SEM['Ski']:.1f} | {_MDC['Ski']:.1f} | {_MDC_PCT['Ski']:.1f}% |
| Run | 202 | {_SEM['Run']:.1f} | {_MDC['Run']:.1f} | {_MDC_PCT['Run']:.1f}% |

**Fórmula MDC:** `MDC₉₅ = 1.96 × √2 × SEM`
(Hecksteden et al., 2015; Atkinson & Batterham, 2015)

**Interpretação:**
- `|Δ| ≥ MDC` → **REAL** (95% de confiança de mudança verdadeira)
- `|Δ| ≥ MDC/2` → **INCERTO** (sinal fraco, monitorizar)
- `|Δ| < MDC/2` → **RUÍDO** (indistinguível de zero estatisticamente)

**Nota Run:** MDC = {_MDC['Run']:.0f}W sobre média histórica ~140W ({_MDC_PCT['Run']:.0f}%).
O eFTP estimado em Run tem muito maior variabilidade que as modalidades com potenciómetro.
Para diagnóstico de Run, recomenda-se teste controlado (TT 20-30 min em condições fixas).

**Referências:**
- Montero D & Lundby C (2017). Refuting the myth of non-response to exercise training. *J Physiol.*
- Hecksteden A et al. (2015). Individual response to exercise training — a statistical perspective. *J Appl Physiol.*
- Iannetta D et al. (2020). A critical evaluation of current methods for exercise prescription. *Med Sci Sports Exerc.*
- Della Mattia G (2025). Variabilidad Inter-Individual en el Entrenamiento de Resistencia. *ednacore AI.*
        """)
