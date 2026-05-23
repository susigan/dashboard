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

    # ══════════════════════════════════════════════════════════════════════════
    # PROJECÇÃO CP 28 DIAS
    # β adimensional: OLS(Δln(eFTP) ~ CTLγ_norm) — escala correcta, sem absurdos
    # Cap ±25% em 28 dias. IC 90% via σ_resid em ln-escala.
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## Projecção de CP — 28 dias")
    with st.expander("📖 Como é calculada", expanded=False):
        st.markdown("""
**Base: FTLM Part II (Della Mattia 2025)**

```
x = CTLγ_norm = (CTLγ / mediana_CTLγ) − 1   [adimensional, centrado em 0]
y = Δln(eFTP) = ln(eFTP / eFTP_ref_90d)       [log-variação vs baseline]

β = OLS(y ~ x)                                 [β adimensional]
```

β=0.5 → +10% CTLγ prediz +5% eFTP. Valores razoáveis: 0.1–1.5.
Projecção: CTLγ evolui linearmente ao ritmo do slope actual (últimos 14d).
IC 90% via σ_resid em ln-escala → convertido para Watts. Cap ±25% em 28d.
        """)

    _proj_ok    = False
    ld_for_proj = ld

    # Fallback CTLγ simples se session_state vazio
    if ld_for_proj is None or len(ld_for_proj) < 30:
        try:
            _col_d_p = next((c for c in ["date","Data"] if c in ac_full.columns), None)
            _col_kj  = next((c for c in [
                "icu_trimp","trimp","AllWorkFTP","icu_training_load",
                "training_load","icu_load","ctl","icu_ctl","z1_kj"
            ] if c in ac_full.columns), None)
            if _col_d_p and _col_kj:
                _df_ld = ac_full[[_col_d_p, _col_kj, col_mod]].copy()
                _df_ld[_col_d_p] = pd.to_datetime(_df_ld[_col_d_p])
                _df_ld = _df_ld.rename(columns={_col_d_p:"Data", _col_kj:"load"})
                _df_ld["load"] = pd.to_numeric(_df_ld["load"], errors="coerce").fillna(0)
                _daily = (_df_ld.groupby("Data")["load"].sum()
                          .reindex(pd.date_range(_df_ld["Data"].min(),
                                                 _df_ld["Data"].max(), freq="D"),
                                   fill_value=0))
                _ctl_g = _daily.ewm(span=42).mean()
                ld_for_proj = pd.DataFrame({"Data": _ctl_g.index, "CTLg_perf": _ctl_g.values})
                for _mp in ["Bike","Row","Ski","Run"]:
                    _dm = _df_ld[_df_ld[col_mod]==_mp].groupby("Data")["load"].sum()
                    if len(_dm) >= 5:
                        ld_for_proj[f"CTLg_{_mp}"] = (_dm.reindex(_ctl_g.index, fill_value=0)
                                                         .ewm(span=42).mean().values)
                st.caption("CTLγ local (τ=42d EWM). Carrega tab PMC para valores calibrados.")
        except Exception as _ld_e:
            st.caption(f"Fallback CTLγ: {_ld_e}")

    if ld_for_proj is not None and len(ld_for_proj) > 30:
        try:
            from scipy import stats as _sp_stats

            _ld_p = ld_for_proj.copy()
            _ld_p["Data"] = pd.to_datetime(_ld_p["Data"]).dt.normalize()
            _ld_idx = _ld_p.set_index("Data")

            _mods_proj   = ["Bike","Row","Ski","Run"]
            _beta_d      = {}
            _r2_d        = {}
            _eftp_now_d  = {}
            _ctlg_now_d  = {}
            _slope_pct_d = {}
            _sigma_ln_d  = {}

            for _mp in _mods_proj:
                # Sessões com eFTP válido
                _ef = (ac_full[ac_full[col_mod]==_mp][[col_date,col_eftp]]
                       .dropna().copy())
                if len(_ef) < 8:
                    continue
                _ef[col_date] = pd.to_datetime(_ef[col_date]).dt.normalize()
                _ef = (_ef.rename(columns={col_date:"Data",col_eftp:"eftp"})
                        .sort_values("Data").drop_duplicates("Data").reset_index(drop=True))

                # CTLγ desta modalidade
                _cc = (f"CTLg_{_mp}" if f"CTLg_{_mp}" in _ld_idx.columns
                       else "CTLg_perf" if "CTLg_perf" in _ld_idx.columns else None)
                if _cc is None:
                    continue

                _cr   = _ld_idx[_cc].ffill().bfill()
                _cmed = float(_cr.median())
                if _cmed < 0.01:
                    continue

                # Normalizar: adimensional centrado em 0
                _cn = (_cr / _cmed) - 1.0
                _ef["ctlg_norm"] = _ef["Data"].map(_cn.to_dict())

                # Δln(eFTP) vs mediana rolling 90d anterior
                _ef["eftp_ref"] = _ef["eftp"].rolling(90, min_periods=5).median().shift(1)
                _ef = _ef.dropna(subset=["eftp_ref","ctlg_norm"])
                if len(_ef) < 8:
                    continue
                # No clip on dln — let OLS see real variance (outliers handled by R²)
                _ratio = _ef["eftp"] / _ef["eftp_ref"].clip(lower=1.0)
                _ef["dln"] = np.log(_ratio.clip(lower=0.5, upper=2.0))  # ±100% max plausível fisiologicamente

                _v = _ef[["dln","ctlg_norm"]].replace([np.inf,-np.inf],np.nan).dropna()
                if len(_v) < 8:
                    continue

                _bv, _ic_ols, _r, _, _ = _sp_stats.linregress(
                    _v["ctlg_norm"].values.astype(float),
                    _v["dln"].values.astype(float))

                _res   = _v["dln"].values - (_ic_ols + _bv*_v["ctlg_norm"].values)
                _sig_l = float(np.std(_res, ddof=2))

                _beta_d[_mp]     = float(_bv)
                _r2_d[_mp]       = float(_r**2)
                _sigma_ln_d[_mp] = _sig_l
                _eftp_now_d[_mp] = float(_ef["eftp"].iloc[-1])

                _ca_now = float(_cr.dropna().iloc[-1])
                _cn_now = float(_cn.dropna().iloc[-1])
                _ctlg_now_d[_mp] = {"abs":_ca_now,"norm":_cn_now,"med":_cmed}

                # Slope %/dia (últimos 14d)
                _c14 = _cr.dropna().tail(14)
                if len(_c14) >= 7:
                    _xx  = np.arange(len(_c14), dtype=float)
                    _sl, *_ = _sp_stats.linregress(_xx, _c14.values.astype(float))
                    _slope_pct_d[_mp] = float(_sl / max(_ca_now, 0.01))
                else:
                    _slope_pct_d[_mp] = 0.0

            if not _beta_d:
                raise ValueError("Nenhuma modalidade com dados suficientes. Carrega tab PMC primeiro.")

            # ── Cards β ────────────────────────────────────────────────────────
            st.markdown("#### Coeficiente β — CTLγ_norm → Δln(eFTP)")
            st.caption("β adimensional. β=0.5 → +10% CTLγ prediz +5% eFTP. Razoável: 0.1–1.5.")
            _bc = st.columns(max(len(_beta_d),1))
            _MC = {"Bike":"#e74c3c","Row":"#3498db","Ski":"#9b59b6","Run":"#27ae60"}
            for _bi, (_bm, _bv) in enumerate(_beta_d.items()):
                _r2v = _r2_d.get(_bm,0)
                _bc[_bi].metric(f"{_bm} β", f"{_bv:.3f}",
                    delta=f"R²={_r2v:.2f}",
                    delta_color="normal" if _r2v>0.10 else "off",
                    help=f"σ_ln={_sigma_ln_d.get(_bm,0):.3f}. {'OK.' if _r2v>0.10 else 'R² baixo.'}")

            # ── Gráfico ────────────────────────────────────────────────────────
            _fig_p   = go.Figure()
            _PD      = 28
            _today   = pd.Timestamp.now().normalize()
            _pdates  = pd.date_range(_today, periods=_PD+1, freq="D")
            _pdstr   = [str(d.date()) for d in _pdates]
            _pdrev   = list(reversed(_pdstr))

            for _mp, _bv in _beta_d.items():
                _e0   = _eftp_now_d.get(_mp)
                if not _e0: continue
                _cor  = _MC.get(_mp,"#888")
                _r2v  = _r2_d.get(_mp,0)
                _sln  = _sigma_ln_d.get(_mp,0.05)
                _ci   = _ctlg_now_d.get(_mp,{})
                _ca   = _ci.get("abs",1.0)
                _cn   = _ci.get("norm",0.0)
                _cm   = _ci.get("med",1.0)
                _spc  = _slope_pct_d.get(_mp,0.0)

                # Histórico 90d
                _eh = (ac_full[ac_full[col_mod]==_mp][[col_date,col_eftp]].dropna().copy())
                _eh[col_date] = pd.to_datetime(_eh[col_date])
                _eh = _eh.rename(columns={col_date:"Data",col_eftp:"eftp"}).sort_values("Data")
                _eh = _eh[_eh["Data"] >= _today-pd.Timedelta(days=90)]
                _es = _eh.set_index("Data")["eftp"].rolling(14,min_periods=2).mean()
                if len(_es) > 0:
                    _fig_p.add_trace(go.Scatter(
                        x=_es.index.tolist(), y=[float(v) for v in _es.values],
                        name=f"{_mp} observado",
                        line=dict(color=_cor,width=2.5),
                        hovertemplate=f"{_mp}: %{{y:.0f}}W<extra></extra>"))

                # Projecção: CTLγ evolui linearmente
                _pvs = []
                for _d in range(_PD+1):
                    _cad = _ca*(1.0+_spc*_d)
                    _cnd = (_cad/max(_cm,0.01))-1.0
                    _dn  = _cnd-_cn
                    _ev  = float(_e0 * np.exp(float(_bv) * _dn))
                    _pvs.append(_ev)

                _z90  = 1.645
                _icw  = float(min(_e0*(np.exp(_sln*_z90)-1.0), _e0*0.20))
                _phi  = [v + _icw for v in _pvs]
                _plo  = [max(v - _icw, 1.0) for v in _pvs]

                _ri,_gi,_bi2 = int(_cor[1:3],16),int(_cor[3:5],16),int(_cor[5:7],16)
                _fig_p.add_trace(go.Scatter(
                    x=_pdstr+_pdrev,
                    y=[float(v) for v in _phi]+[float(v) for v in reversed(_plo)],
                    fill="toself",fillcolor=f"rgba({_ri},{_gi},{_bi2},0.10)",
                    line=dict(width=0),showlegend=False,hoverinfo="skip"))
                # Estilo da linha baseado na fiabilidade (R²)
                _line_opacity = 1.0 if _r2v >= 0.20 else (0.7 if _r2v >= 0.08 else 0.4)
                _line_dash    = "dash" if _r2v >= 0.08 else "dot"
                _line_width   = 2.5 if _r2v >= 0.20 else 2.0
                _fig_p.add_trace(go.Scatter(
                    x=_pdstr, y=[float(round(v,1)) for v in _pvs],
                    name=f"{_mp} proj 28d (β={_bv:.2f} R²={_r2v:.2f})",
                    opacity=_line_opacity,
                    line=dict(color=_cor, width=_line_width, dash=_line_dash),
                    hovertemplate=f"{_mp} proj: %{{y:.0f}}W<extra></extra>"))

                _pe   = float(round(_pvs[-1],0))
                _dpct = float((_pe-_e0)/max(_e0,1)*100)
                # Fiabilidade no texto da anotação
                _fiab_icon = "🟢" if _r2v >= 0.20 else ("🟡" if _r2v >= 0.08 else "🔴")
                _fig_p.add_annotation(
                    x=_pdstr[-1],y=_pe,
                    text=f"{_fiab_icon} <b>{_mp} +28d: {_pe:.0f}W ({_dpct:+.1f}%)</b>",
                    showarrow=False,xshift=4,yshift=10,
                    font=dict(size=11,color=_cor),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor=_cor,borderwidth=1,borderpad=3)

            _tstr = str(_today.date())
            _fig_p.add_shape(type="line",x0=_tstr,x1=_tstr,y0=0,y1=1,
                xref="x",yref="paper",line=dict(dash="dot",color="#888",width=1))
            _fig_p.add_annotation(x=_tstr,y=1.02,xref="x",yref="paper",
                text="Hoje",showarrow=False,font=dict(size=10,color="#555"),
                bgcolor="rgba(255,255,255,0.88)",xanchor="left")
            _fig_p.update_layout(
                height=440,hovermode="x unified",
                paper_bgcolor="white",plot_bgcolor="white",
                margin=dict(t=40,b=90,l=65,r=150),
                legend=dict(orientation="h",y=-0.22,
                    font=dict(size=11,color="#333"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#ddd",borderwidth=1),
                title=dict(text="eFTP observado (90d) + Projecção 28 dias",
                    font=dict(size=13,color="#222")),
                xaxis=dict(tickangle=-25,gridcolor="rgba(0,0,0,0.04)",
                    tickfont=dict(color="#333")),
                yaxis=dict(title="eFTP (W)",gridcolor="rgba(0,0,0,0.05)",
                    tickfont=dict(color="#333"),zeroline=False))
            st.plotly_chart(_fig_p,use_container_width=True,
                config={"displayModeBar":False},key="cp_proj_28d")

            # Tabela
            _rows = []
            for _mp, _bv in _beta_d.items():
                _e0  = _eftp_now_d.get(_mp)
                if not _e0: continue
                _ci  = _ctlg_now_d.get(_mp,{})
                _ca  = _ci.get("abs",1.0); _cn=_ci.get("norm",0.0); _cm=_ci.get("med",1.0)
                _spc = _slope_pct_d.get(_mp,0.0)
                _sln = _sigma_ln_d.get(_mp,0.05)
                _ca28= _ca*(1.0+_spc*28)
                _cn28= (_ca28/max(_cm,0.01))-1.0
                _p28 = float(_e0 * np.exp(float(_bv) * (_cn28 - _cn)))
                _dw  = _p28-_e0
                _icw = float(min(_e0*(np.exp(_sln*1.645)-1.0),_e0*0.20))
                _rows.append({
                    "Modalidade":_mp,
                    "eFTP actual (W)":f"{_e0:.0f}",
                    "eFTP proj +28d (W)":f"{_p28:.0f}",
                    "Δ (W)":f"{_dw:+.0f}",
                    "Δ (%)":f"{(_dw/_e0*100):+.1f}%",
                    "IC 90% ±(W)":f"±{_icw:.0f}",
                    "β":f"{_bv:.3f}",
                    "R²":f"{_r2_d.get(_mp,0):.3f}",
                    "CTLγ":f"{_ca:.1f}",
                    "slope %/d":f"{_spc*100:+.3f}%",
                })
            if _rows:
                _df_rows = pd.DataFrame(_rows)
                st.dataframe(_df_rows, use_container_width=True, hide_index=True)

                # ── Sinalização de fiabilidade por modalidade ──────────────────
                st.markdown("##### Diagnóstico de fiabilidade da projecção")
                _diag_cols = st.columns(len(_rows))
                for _di, _row in enumerate(_rows):
                    _r2v    = float(_row["R²"])
                    _bv_abs = abs(float(_row["β"]))
                    _mp     = _row["Modalidade"]
                    _dw     = float(_row["Δ (W)"].replace("+",""))
                    _dpct   = float(_row["Δ (%)"].replace("%","").replace("+",""))
                    _ca     = float(_row["CTLγ"])
                    _spc    = float(_row["slope %/d"].replace("%","").replace("+",""))

                    # Classificação de fiabilidade
                    if _r2v >= 0.20:
                        _fcolor = "#27ae60"; _flabel = "🟢 Fiável"
                        _fdesc  = f"R²={_r2v:.2f} — modelo tem poder preditivo."
                    elif _r2v >= 0.08:
                        _fcolor = "#f39c12"; _flabel = "🟡 Incerto"
                        _fdesc  = f"R²={_r2v:.2f} — direcção indicativa, magnitude incerta."
                    else:
                        _fcolor = "#e74c3c"; _flabel = "🔴 Baixa"
                        _fdesc  = f"R²={_r2v:.2f} — CTLγ não explica variação de eFTP."

                    # Consistência β vs direcção
                    _beta_v = float(_row["β"])
                    if abs(_beta_v) < 0.01:
                        _fdesc += " β≈0 — sem sensibilidade detectada."

                    # CTLγ slope: está a crescer ou cair?
                    if abs(_spc) < 0.001:
                        _slope_txt = "CTLγ estável"
                    elif _spc > 0:
                        _slope_txt = f"CTLγ ↑ +{_spc:.3f}%/d"
                    else:
                        _slope_txt = f"CTLγ ↓ {_spc:.3f}%/d"

                    with _diag_cols[_di]:
                        _hx = _fcolor.lstrip("#")
                        _rr,_gg,_bb = int(_hx[0:2],16),int(_hx[2:4],16),int(_hx[4:6],16)
                        st.markdown(
                            f"<div style='border:1.5px solid {_fcolor};"
                            f"border-radius:8px;padding:10px 12px;"
                            f"background:rgba({_rr},{_gg},{_bb},0.06)'>"
                            f"<div style='font-size:13px;font-weight:600;"
                            f"color:{_fcolor};margin-bottom:4px'>{_flabel}</div>"
                            f"<div style='font-size:14px;font-weight:500;"
                            f"color:#222;margin-bottom:2px'>{_mp}</div>"
                            f"<div style='font-size:12px;color:#555'>{_fdesc}</div>"
                            f"<div style='font-size:11px;color:#888;margin-top:4px'>"
                            f"{_slope_txt} | CTLγ={_ca:.1f}"
                            f"</div></div>",
                            unsafe_allow_html=True)

                # Nota interpretativa global
                _n_fiaveis = sum(1 for r in _rows if float(r["R²"]) >= 0.20)
                _n_incertos = sum(1 for r in _rows if 0.08 <= float(r["R²"]) < 0.20)
                if _n_fiaveis == 0:
                    st.warning(
                        "⚠️ **Nenhuma modalidade com R²≥0.20.** "
                        "O CTLγ não é um preditor forte do eFTP para este atleta neste período. "
                        "Possíveis causas: (1) composição do treino mudou recentemente "
                        "(ex: mudança de intensidade sem mudança de volume), "
                        "(2) poucos testes de eFTP para calibração, "
                        "(3) outros factores dominam (sono, nutrição, viagens).")
                elif _n_fiaveis >= 2:
                    st.success(
                        f"✅ {_n_fiaveis} modalidade(s) com R²≥0.20. "
                        "Projecção tem poder preditivo razoável para essas modalidades.")

                st.caption(
                    "β adimensional. IC 90% σ_ln×z₀.₉₀. 🟢 R²≥0.20 | 🟡 R²=0.08–0.20 | 🔴 R²<0.08")
            _proj_ok = True

            # ── Guardar α no session_state para tab_visao_geral ──────────────
            # Formato: {'Bike': {'ok':True,'alpha_z3':0.94,...}, ...}
            try:
                from utils.data import calcular_alpha_polar
                _gamma_from_ld = {}
                _ld_info = st.session_state.get('ld_frac_info', {})
                for _mi in ['Bike','Row','Ski','Run']:
                    _gamma_from_ld[_mi] = (_ld_info.get('mods',{}).get(_mi,{})
                                            .get('gamma_perf', 0.5))
                _alpha_result = calcular_alpha_polar(ac_full, gamma_map=_gamma_from_ld)
                st.session_state['alpha_polar_cache'] = _alpha_result
            except Exception as _ae:
                # Fallback: usar os α já calculados no Modelo 2
                _alpha_result = {}
                for _pr in _polar_rows:
                    _mp2x = _pr['Modalidade']
                    _alpha_result[_mp2x] = {
                        'ok': True,
                        'alpha_z3': float(_pr['α_Z3 (intenso)']),
                        'alpha_z2': float(_pr['α_Z2 (limiar)']),
                        'alpha_z1': float(_pr['α_Z1 (base)']),
                        'r2': float(_pr['R² Modelo 2']),
                        'eftp_now': _eftp_z_now.get(_mp2x, 0),
                        'cz3_now': _ctlg_z_now.get(_mp2x, {}).get('Z3', 0),
                        'cz2_now': _ctlg_z_now.get(_mp2x, {}).get('Z2', 0),
                        'cz1_now': _ctlg_z_now.get(_mp2x, {}).get('Z1', 0),
                    }
                st.session_state['alpha_polar_cache'] = _alpha_result

        except Exception as _pe:
            import traceback as _tb
            st.info(f"Projecção CP: {_pe}")
            with st.expander("Traceback"):
                st.code(_tb.format_exc())
    else:
        st.info("Projecção CP requer CTLγ. Carrega tab PMC primeiro.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MODELO 2 — FTLM POLAR: CTLγ decomposto por zonas de intensidade
    # Extensão do FTLM Part II (Della Mattia 2025) — não foge ao paper:
    # o mesmo γ modal é aplicado separadamente a kJ_Z1, kJ_Z2, kJ_Z3
    # eFTP ~ α_Z3·CTLγ_Z3 + α_Z2·CTLγ_Z2 + α_Z1·CTLγ_Z1   (OLS por modalidade)
    # Espírito do paper: "carga fraccionária por domínio de intensidade"
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Modelo 2 — FTLM Polar (CTLγ por zona de intensidade)")
    st.caption(
        "Extensão do FTLM Part II (Della Mattia 2025): o mesmo γ modal é aplicado "
        "separadamente a kJ_Z1 / kJ_Z2 / kJ_Z3, decompondo a carga por domínio de intensidade. "
        "Permite distinguir 'muito volume Z1' de 'muito estímulo Z3' — o CTLγ total não distingue.")

    with st.expander("📖 Diferenças vs Modelo 1 e ligação ao paper", expanded=False):
        st.markdown("""
**Modelo 1 (CTLγ total)** — fiel ao paper FTLM Part II §2.1:
```
CTLγ(t) = EWM(carga_total, τ=42d/γ)
eFTP ~ β × CTLγ_norm     [R² tipicamente baixo: 0.01–0.31]
```

**Modelo 2 (FTLM Polar)** — extensão natural do mesmo paper:
```
CTLγ_Z3(t) = EWM(kJ_Z3, τ=42d/γ_modal)   ← estímulo de alta intensidade
CTLγ_Z2(t) = EWM(kJ_Z2, τ=42d/γ_modal)   ← zona de limiar
CTLγ_Z1(t) = EWM(kJ_Z1, τ=42d/γ_modal)   ← base aeróbica
eFTP ~ α_Z3·CTLγ_Z3 + α_Z2·CTLγ_Z2 + α_Z1·CTLγ_Z1   [OLS múltipla]
```

**Porquê não foge ao paper:**
- Usa o mesmo γ modal já calibrado (não inventa novos parâmetros)
- É a mesma decomposição fraccionária do FTLM, agora por zona em vez de modalidade
- O paper §3.2 menciona explicitamente que CTLγ subestima o impacto de sessões Z3 isoladas

**Interpretação dos coeficientes α:**
- α_Z3 > 0, grande → treino intenso prediz ganho de eFTP (resposta anaeróbica)
- α_Z1 > 0, pequeno → base aeróbica contribui mas menos
- α_Z2 ≈ 0 → zona de limiar tem contribuição ambígua (zona "lixo" para eFTP)
        """)

    # Verificar disponibilidade das colunas z1/z2/z3_kj
    _z1c = next((c for c in ["z1_kj","Z1KJ","z1kj"] if c in ac_full.columns), None)
    _z2c = next((c for c in ["z2_kj","Z2KJ","z2kj"] if c in ac_full.columns), None)
    _z3c = next((c for c in ["z3_kj","Z3KJ","z3kj"] if c in ac_full.columns), None)

    if not (_z1c and _z2c and _z3c):
        st.info(
            f"Modelo 2 requer colunas z1_kj, z2_kj, z3_kj. "
            f"Disponíveis em ac_full: {[c for c in ac_full.columns if 'kj' in c.lower() or 'KJ' in c]}. "
            "Verifica que as colunas Z1KJ/Z2KJ/Z3KJ estão na sheet de actividades.")
    else:
        try:
            from scipy import stats as _sp_stats

            # Gamma por modalidade (do ld_frac_cache se disponível)
            _gamma_map = {}
            if ld is not None and len(ld) > 0:
                # Tentar extrair info de gamma do session_state
                _info = st.session_state.get("ld_frac_info", {})
                _mods_info = _info.get("mods", {})
                for _mp in ["Bike","Row","Ski","Run"]:
                    _gamma_map[_mp] = _mods_info.get(_mp, {}).get("gamma_perf", 0.5)

            # Fallback gammas calibrados (do tab_pmc)
            _gamma_defaults = {"Bike":0.250,"Row":0.900,"Ski":0.600,"Run":0.900}
            for _mp in ["Bike","Row","Ski","Run"]:
                if _mp not in _gamma_map or _gamma_map[_mp] == 0.5:
                    _gamma_map[_mp] = _gamma_defaults[_mp]

            _mods_z     = ["Bike","Row","Ski","Run"]
            _alpha_dict = {}   # {mod: {"Z1":α1,"Z2":α2,"Z3":α3}}
            _r2_z_dict  = {}   # R² do modelo polar
            _eftp_z_now = {}   # eFTP actual por modal
            _ctlg_z_now = {}   # CTLγ_Z3/Z2/Z1 actuais
            _proj_z_28d = {}   # projecção 28d por modal

            _polar_rows = []   # para tabela comparativa

            for _mp in _mods_z:
                # Sessões desta modalidade
                _ef = (ac_full[ac_full[col_mod]==_mp]
                       [[col_date, col_eftp, _z1c, _z2c, _z3c]]
                       .copy())
                _ef[col_date] = pd.to_datetime(_ef[col_date]).dt.normalize()
                _ef = (_ef.rename(columns={col_date:"Data", col_eftp:"eftp",
                                           _z1c:"z1", _z2c:"z2", _z3c:"z3"})
                        .sort_values("Data").drop_duplicates("Data").reset_index(drop=True))
                _ef[["eftp","z1","z2","z3"]] = _ef[["eftp","z1","z2","z3"]].apply(
                    pd.to_numeric, errors="coerce")
                _ef = _ef.dropna(subset=["eftp"])
                _ef[["z1","z2","z3"]] = _ef[["z1","z2","z3"]].fillna(0)

                if len(_ef) < 10:
                    continue

                # γ modal
                _gam = _gamma_map.get(_mp, 0.5)
                _tau = max(42.0 * (1.0 - _gam) + 7.0 * _gam, 7.0)  # interpola entre 42d e 7d
                _span = int(round(_tau))

                # Construir série diária de kJ por zona
                _date_range = pd.date_range(_ef["Data"].min(), pd.Timestamp.now().normalize(), freq="D")
                _ef_idx = _ef.set_index("Data")

                _z1_daily = _ef_idx["z1"].reindex(_date_range, fill_value=0)
                _z2_daily = _ef_idx["z2"].reindex(_date_range, fill_value=0)
                _z3_daily = _ef_idx["z3"].reindex(_date_range, fill_value=0)

                # CTLγ por zona — EWM com span modal
                _ctlg_z1 = _z1_daily.ewm(span=_span).mean()
                _ctlg_z2 = _z2_daily.ewm(span=_span).mean()
                _ctlg_z3 = _z3_daily.ewm(span=_span).mean()

                # Mapear CTLγ de cada zona para as datas das sessões
                _ef["cz1"] = _ef["Data"].map(_ctlg_z1.to_dict())
                _ef["cz2"] = _ef["Data"].map(_ctlg_z2.to_dict())
                _ef["cz3"] = _ef["Data"].map(_ctlg_z3.to_dict())
                _ef = _ef.dropna(subset=["cz1","cz2","cz3"])

                if len(_ef) < 10:
                    continue

                # OLS múltipla: eFTP ~ α_Z3·cz3 + α_Z2·cz2 + α_Z1·cz1
                _X = np.column_stack([
                    _ef["cz3"].values.astype(float),
                    _ef["cz2"].values.astype(float),
                    _ef["cz1"].values.astype(float),
                    np.ones(len(_ef))  # intercept
                ])
                _y = _ef["eftp"].values.astype(float)

                # OLS via lstsq (mais robusto que linregress para multi-variável)
                _coef, _res, _rank, _ = np.linalg.lstsq(_X, _y, rcond=None)
                _a_z3, _a_z2, _a_z1, _intc = float(_coef[0]), float(_coef[1]), float(_coef[2]), float(_coef[3])

                _y_pred = _X @ _coef
                _ss_res = float(np.sum((_y - _y_pred)**2))
                _ss_tot = float(np.sum((_y - _y.mean())**2))
                _r2_z   = float(1 - _ss_res / _ss_tot) if _ss_tot > 0 else 0.0
                _sigma_w = float(np.std(_y - _y_pred, ddof=4))

                _alpha_dict[_mp] = {"Z3":_a_z3,"Z2":_a_z2,"Z1":_a_z1,"intc":_intc}
                _r2_z_dict[_mp]  = _r2_z
                _eftp_z_now[_mp] = float(_ef["eftp"].iloc[-1])

                # CTLγ actual por zona
                _cz3_now = float(_ctlg_z3.iloc[-1])
                _cz2_now = float(_ctlg_z2.iloc[-1])
                _cz1_now = float(_ctlg_z1.iloc[-1])
                _ctlg_z_now[_mp] = {"Z3":_cz3_now,"Z2":_cz2_now,"Z1":_cz1_now}

                # Projecção 28d: assume CTLγ de cada zona estável (slope 14d)
                def _zone_slope(series, n=14):
                    s = series.dropna().tail(n)
                    if len(s) < 5: return 0.0
                    xx = np.arange(len(s), dtype=float)
                    sl, *_ = _sp_stats.linregress(xx, s.values.astype(float))
                    return float(sl)

                _sl_z3 = _zone_slope(_ctlg_z3); _sl_z2 = _zone_slope(_ctlg_z2); _sl_z1 = _zone_slope(_ctlg_z1)

                # eFTP proj(t+28) = α_Z3·(cz3+sl_z3×28) + α_Z2·(cz2+sl_z2×28) + α_Z1·(cz1+sl_z1×28) + intc
                _cz3_28 = _cz3_now + _sl_z3*28; _cz2_28 = _cz2_now + _sl_z2*28; _cz1_28 = _cz1_now + _sl_z1*28
                _eftp_28 = float(_a_z3*_cz3_28 + _a_z2*_cz2_28 + _a_z1*_cz1_28 + _intc)
                _proj_z_28d[_mp] = _eftp_28

                # R² comparison with model 1
                _r2_m1 = _r2_d.get(_mp, 0.0) if "_r2_d" in dir() else _r2_dict.get(_mp, 0.0)

                _polar_rows.append({
                    "Modalidade":          _mp,
                    "γ modal":             f"{_gam:.3f}",
                    "α_Z3 (intenso)":      f"{_a_z3:.3f}",
                    "α_Z2 (limiar)":       f"{_a_z2:.3f}",
                    "α_Z1 (base)":         f"{_a_z1:.3f}",
                    "R² Modelo 2":         f"{_r2_z:.3f}",
                    "R² Modelo 1 (CTLγ)":  f"{_r2_m1:.3f}",
                    "ΔR²":                 f"{(_r2_z-_r2_m1):+.3f}",
                    "eFTP proj +28d (W)":  f"{_eftp_28:.0f}",
                    "eFTP M1 proj (W)":    _rows[_mods_z.index(_mp)]["eFTP proj +28d (W)"] if "_rows" in dir() and _mp in _mods_z and _mods_z.index(_mp) < len(_rows) else "—",
                    "CTLγ_Z3 actual":      f"{_cz3_now:.2f}",
                    "slope Z3 (%/sem)":    f"{_sl_z3*7:+.3f}",
                })

            if not _polar_rows:
                st.info("Modelo 2: dados insuficientes (requer z1_kj/z2_kj/z3_kj por sessão).")
            else:
                # ── Cards α por modalidade ─────────────────────────────────────
                st.markdown("#### Coeficientes α por zona — sensibilidade ao estímulo")
                _ac = st.columns(len(_polar_rows))
                for _pi, _pr in enumerate(_polar_rows):
                    _mp2  = _pr["Modalidade"]
                    _r2z  = float(_pr["R² Modelo 2"])
                    _r2m1 = float(_pr["R² Modelo 1 (CTLγ)"])
                    _a3   = float(_pr["α_Z3 (intenso)"])
                    _a2   = float(_pr["α_Z2 (limiar)"])
                    _a1   = float(_pr["α_Z1 (base)"])
                    _cor2 = _MC.get(_mp2, "#888") if "_MC" in dir() else {"Bike":"#e74c3c","Row":"#3498db","Ski":"#9b59b6","Run":"#27ae60"}.get(_mp2,"#888")
                    _hx2  = _cor2.lstrip("#"); _rr2,_gg2,_bb2 = int(_hx2[0:2],16),int(_hx2[2:4],16),int(_hx2[4:6],16)

                    _flab = ("🟢 Fiável" if _r2z>=0.20 else ("🟡 Incerto" if _r2z>=0.08 else "🔴 Baixa"))
                    _imp  = ("🟢 Melhora" if _r2z > _r2m1+0.02 else ("⚖️ Igual" if abs(_r2z-_r2m1)<=0.02 else "🔴 Piora"))

                    with _ac[_pi]:
                        st.markdown(
                            f"<div style='border:1.5px solid {_cor2};border-radius:8px;"
                            f"padding:10px 12px;background:rgba({_rr2},{_gg2},{_bb2},0.06)'>"
                            f"<div style='font-size:13px;font-weight:600;color:{_cor2}'>{_mp2}</div>"
                            f"<div style='font-size:11px;color:#555;margin:4px 0'>"
                            f"α_Z3={_a3:.3f} | α_Z2={_a2:.3f} | α_Z1={_a1:.3f}</div>"
                            f"<div style='font-size:11px;color:#555'>"
                            f"R²={_r2z:.3f} {_flab}</div>"
                            f"<div style='font-size:11px;color:#888;margin-top:3px'>"
                            f"vs M1: {_imp} (ΔR²={_r2z-_r2m1:+.3f})</div>"
                            f"</div>",
                            unsafe_allow_html=True)

                # ── Gráfico comparativo M1 vs M2 ──────────────────────────────
                st.markdown("#### Projecção: Modelo 1 (CTLγ) vs Modelo 2 (FTLM Polar)")

                _fig_z = go.Figure()
                _MCOLS = {"Bike":"#e74c3c","Row":"#3498db","Ski":"#9b59b6","Run":"#27ae60"}
                _pdates_z    = pd.date_range(pd.Timestamp.now().normalize(), periods=29, freq="D")
                _pdates_z_str= [str(d.date()) for d in _pdates_z]
                _today_z     = str(pd.Timestamp.now().date())

                for _pr in _polar_rows:
                    _mp2  = _pr["Modalidade"]
                    _e0   = _eftp_z_now.get(_mp2,0)
                    _p28z = _proj_z_28d.get(_mp2, _e0)
                    _cor2 = _MCOLS.get(_mp2,"#888")
                    _r2z  = float(_pr["R² Modelo 2"])

                    # Histórico 90d
                    _eh2 = (ac_full[ac_full[col_mod]==_mp2][[col_date,col_eftp]].dropna().copy())
                    _eh2[col_date] = pd.to_datetime(_eh2[col_date])
                    _eh2 = _eh2.rename(columns={col_date:"Data",col_eftp:"eftp"}).sort_values("Data")
                    _eh2 = _eh2[_eh2["Data"] >= pd.Timestamp.now()-pd.Timedelta(days=90)]
                    _es2 = _eh2.set_index("Data")["eftp"].rolling(14,min_periods=2).mean()
                    if len(_es2) > 0:
                        _fig_z.add_trace(go.Scatter(
                            x=_es2.index.tolist(), y=[float(v) for v in _es2.values],
                            name=f"{_mp2} obs",
                            line=dict(color=_cor2, width=2),
                            hovertemplate=f"{_mp2}: %{{y:.0f}}W<extra></extra>"))

                    # M1 proj (já calculado)
                    # eFTP actual da modalidade — usar _eftp_z_now (calculado no M2)
                    # ou _eftp_now_d (do Modelo 1, se disponível)
                    _p28m1 = (_eftp_now_d.get(_mp2) if "_eftp_now_d" in dir() and _eftp_now_d.get(_mp2)
                              else _eftp_z_now.get(_mp2, _e0))
                    # Projecção M1 do mesmo modelo se disponível via _rows
                    if "_rows" in dir() and isinstance(_rows, list):
                        for _rr2x in _rows:
                            if isinstance(_rr2x, dict) and _rr2x.get("Modalidade") == _mp2:
                                try: _p28m1 = float(_rr2x["eFTP proj +28d (W)"])
                                except: pass

                    # Linha M1 (tracejada, 50% opacidade)
                    _ri2,_gi2,_bi2x = int(_cor2[1:3],16),int(_cor2[3:5],16),int(_cor2[5:7],16)
                    _proj_m1_vals = [float(_e0 + (_p28m1-_e0)*d/28) for d in range(29)]
                    _fig_z.add_trace(go.Scatter(
                        x=_pdates_z_str, y=[round(v,1) for v in _proj_m1_vals],
                        name=f"{_mp2} M1",
                        line=dict(color=_cor2, width=1.5, dash="dot"), opacity=0.5,
                        hovertemplate=f"{_mp2} M1: %{{y:.0f}}W<extra></extra>"))

                    # Linha M2 (tracejada sólida, mais proeminente)
                    _proj_m2_vals = [float(_e0 + (_p28z-_e0)*d/28) for d in range(29)]
                    _opac2 = 1.0 if _r2z>=0.20 else (0.75 if _r2z>=0.08 else 0.5)
                    _fig_z.add_trace(go.Scatter(
                        x=_pdates_z_str, y=[round(v,1) for v in _proj_m2_vals],
                        name=f"{_mp2} M2",
                        line=dict(color=_cor2, width=2.5, dash="dash"), opacity=_opac2,
                        hovertemplate=f"{_mp2} M2: %{{y:.0f}}W<extra></extra>"))

                    # Anotação M2 no dia 28
                    _d28 = _p28z-_e0
                    _icon = "🟢" if _r2z>=0.20 else ("🟡" if _r2z>=0.08 else "🔴")
                    _fig_z.add_annotation(
                        x=_pdates_z_str[-1], y=_p28z,
                        text=f"{_icon}<b>{_mp2} M2: {_p28z:.0f}W ({_d28:+.0f}W)</b>",
                        showarrow=False, xshift=4, yshift=10,
                        font=dict(size=11,color=_cor2),
                        bgcolor="rgba(255,255,255,0.88)",
                        bordercolor=_cor2,borderwidth=1,borderpad=3)

                # Linha Hoje
                _fig_z.add_shape(type="line",x0=_today_z,x1=_today_z,y0=0,y1=1,
                    xref="x",yref="paper",line=dict(dash="dot",color="#888",width=1))
                _fig_z.add_annotation(x=_today_z,y=1.02,xref="x",yref="paper",
                    text="Hoje",showarrow=False,font=dict(size=10,color="#555"),
                    bgcolor="rgba(255,255,255,0.88)",xanchor="left")

                _fig_z.update_layout(
                    height=440, hovermode="x unified",
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(t=40,b=90,l=65,r=150),
                    legend=dict(orientation="h",y=-0.22,
                        font=dict(size=11,color="#333"),
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#ddd",borderwidth=1),
                    title=dict(text="M1 (···) vs M2 FTLM Polar (- - -) — Projecção 28 dias",
                        font=dict(size=13,color="#222")),
                    xaxis=dict(tickangle=-25,gridcolor="rgba(0,0,0,0.04)",
                        tickfont=dict(color="#333")),
                    yaxis=dict(title="eFTP (W)",gridcolor="rgba(0,0,0,0.05)",
                        tickfont=dict(color="#333"),zeroline=False))

                st.plotly_chart(_fig_z,use_container_width=True,
                    config={"displayModeBar":False},key="cp_proj_polar")

                # ── Tabela comparativa ──────────────────────────────────────────
                st.markdown("#### Tabela comparativa — Modelo 1 vs Modelo 2")
                _df_polar = pd.DataFrame(_polar_rows)[[
                    "Modalidade","γ modal",
                    "α_Z3 (intenso)","α_Z2 (limiar)","α_Z1 (base)",
                    "R² Modelo 2","R² Modelo 1 (CTLγ)","ΔR²",
                    "eFTP proj +28d (W)","CTLγ_Z3 actual","slope Z3 (%/sem)"]]
                st.dataframe(_df_polar,use_container_width=True,hide_index=True)

                # Insight automático
                _best_mod = max(_polar_rows, key=lambda x: float(x["R² Modelo 2"]))
                _worst_mod= min(_polar_rows, key=lambda x: float(x["R² Modelo 2"]))
                _improved = [r["Modalidade"] for r in _polar_rows if float(r["ΔR²"])>0.02]
                _worse    = [r["Modalidade"] for r in _polar_rows if float(r["ΔR²"])<-0.02]

                _insight_parts = [
                    f"**{_best_mod['Modalidade']}** tem o melhor ajuste do Modelo 2 "
                    f"(R²={float(_best_mod['R² Modelo 2']):.3f})."]
                if _improved:
                    _insight_parts.append(
                        f"O modelo polar **melhora** vs CTLγ em: {', '.join(_improved)} — "
                        "a decomposição por zona captura variação que o volume total não capta.")
                if _worse:
                    _insight_parts.append(
                        f"O modelo polar **não melhora** em: {', '.join(_worse)} — "
                        "possível que nessas modalidades o volume total seja mais consistente "
                        "que a composição de zonas.")

                _a3_max = max(_polar_rows, key=lambda x: float(x["α_Z3 (intenso)"]))
                if float(_a3_max["α_Z3 (intenso)"]) > 1.0:
                    _insight_parts.append(
                        f"**{_a3_max['Modalidade']}** mostra alta sensibilidade ao treino "
                        f"intenso (α_Z3={float(_a3_max['α_Z3 (intenso)']):.2f}): "
                        "sessões Z3 têm impacto desproporcional no eFTP.")

                st.info("💡 " + " ".join(_insight_parts))
                st.caption(
                    "M1 (···) = Modelo 1 CTLγ total | M2 (- - -) = FTLM Polar por zona. "
                    "🟢 R²≥0.20 | 🟡 R²=0.08–0.20 | 🔴 R²<0.08. "
                    "γ modal calibrado no tab PMC. Sem cap — projecção reflecte o modelo sem truncagem.")

        except Exception as _ze:
            import traceback as _ztb
            st.info(f"Modelo 2 (FTLM Polar): {_ze}")
            with st.expander("Traceback"):
                st.code(_ztb.format_exc())


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
