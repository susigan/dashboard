# ══════════════════════════════════════════════════════════════════════════════
# tab_metab.py — ATHELTICA
# Metabolic Profiling: VLamax · MLSS · FatMax · CHO/Fat · Lactato · Glicogénio
# Modelo: Mader/Heck (1986), Mader (2003), Hauser (2014), konaendu/vlamax
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constantes fisiológicas (Mader/Hauser) ────────────────────────────────────
_Ks1        = 0.0631   # constante ADP/VO2 — Hauser 2014
_Ks2        = 1.331    # constante Michaelis-Menten — Hauser 2014
_VolRel     = 0.40     # espaço de distribuição de lactato (40% peso) — Mader 1986
_Kel        = 4.0      # actividade PDH — Mader & Heck 1986
_VO2rest    = 5.0      # VO2 repouso (ml/min/kg)
_Watt_O2    = 11.685   # slope ml/min/W — calibrado contra Excel MLSSc Hauser
_LAC_O2     = 0.01576  # mmol lactato por ml O2
_LIVER_GLY  = 90.0     # g glicogénio hepático
_MUSCLE_PCT = 0.70     # % lean mass como músculo

# ── Paleta estilo powerlab.icu ────────────────────────────────────────────────
_C_FAT   = "#00C896"   # verde — gordura
_C_CHO   = "#FF6B35"   # laranja — carbohidratos
_C_LAC   = "#E63946"   # vermelho — lactato
_C_VLAM  = "#A855F7"   # roxo — VLamax
_C_AT    = "#FFD166"   # amarelo — AT/MLSS
_C_FATMX = _C_FAT
_C_BG    = "rgba(0,0,0,0)"
_FONT    = "Inter, system-ui, sans-serif"
_AXIS    = dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                zeroline=False, tickfont=dict(size=11))
_BASE_LAYOUT = dict(
    plot_bgcolor=_C_BG, paper_bgcolor=_C_BG,
    font=dict(family=_FONT, color="#EEEEEE", size=12),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


def _compute_metab(vo2max: float, vlamax: float, bw: float,
                   watt_o2: float = _Watt_O2):
    """
    Calcula todo o perfil metabólico.
    Returns dict com arrays e valores escalares.
    """
    VO2ss = np.arange(0.5, vo2max - 0.05, 0.01)

    # ADP (activação glicolítica)
    with np.errstate(divide='ignore', invalid='ignore'):
        ADP = np.sqrt(np.maximum(0, (_Ks1 * VO2ss) / (vo2max - VO2ss)))

    # Taxa bruta de formação de lactato (mmol/L/min)
    vLass = 60 * vlamax / (1 + (_Ks2 / np.maximum(ADP**3, 1e-12)))

    # Taxa de combustão de lactato (mmol/L/min)
    LaComb = (_LAC_O2 / _VolRel) * VO2ss

    # vLanet: diferença absoluta → AT onde vLanet = 0
    vLanet = vLass - LaComb  # signed (positivo = produção > combustão)
    vLanet_abs = np.abs(vLanet)

    # AT: cruzamento vLass = LaComb (mínimo de vLanet_abs)
    arg_AT = int(np.argmin(vLanet_abs))

    # Demanda global (ml/min/kg)
    overall = (vLass * (_VolRel * bw) * ((1 / 4.3) * 22.4) / bw) + VO2ss

    # Intensidade (Watts) — subtrair VO2rest, multiplicar por bw, dividir pelo slope
    Intensity = np.maximum(0, (overall - _VO2rest) * bw / watt_o2)

    # FatMax: máximo de vLanet negativo ANTES do AT (falta de piruvato máxima)
    vLanet_before = vLanet[:arg_AT]
    arg_FatMax = int(np.argmax(-vLanet_before)) if arg_AT > 1 else 0

    # Macronutrientes
    CHO_util = np.full(len(VO2ss[:arg_AT + 400]),
                       (bw * _VolRel) * 60 / 1000 / 2 * 162.14)
    Fat_util = np.maximum(0,
        (-vLanet[:arg_AT]) * _VolRel / _LAC_O2 * bw * 60 * 4.65 / 9.5 / 1000)

    # Lactato estacionário abaixo do AT
    with np.errstate(divide='ignore', invalid='ignore'):
        _denom = ((_LAC_O2 / _VolRel) * VO2ss[:arg_AT] *
                  (1 + (_Ks2 / np.maximum(
                      (_Ks1 * VO2ss[:arg_AT]) / np.maximum(vo2max - VO2ss[:arg_AT], 0.01),
                      1e-9)) ** 1.5) - vlamax * 60)
        CLass = np.where(_denom > 0,
                         np.sqrt(np.maximum(0, (vlamax * _Kel * 60) / _denom)),
                         np.nan)

    return dict(
        VO2ss         = VO2ss,
        vLass         = vLass,
        LaComb        = LaComb,
        vLanet        = vLanet,
        overall       = overall,
        Intensity     = Intensity,
        arg_AT        = arg_AT,
        arg_FatMax    = arg_FatMax,
        CHO_util      = CHO_util,
        Fat_util      = Fat_util,
        CLass         = CLass,
        # Escalares chave
        VO2_AT        = float(VO2ss[arg_AT]),
        W_AT          = float(Intensity[arg_AT]),
        pct_VO2_AT    = float(VO2ss[arg_AT] / vo2max * 100),
        VO2_FatMax    = float(VO2ss[arg_FatMax]),
        W_FatMax      = float(Intensity[arg_FatMax]),
        pct_VO2_FatMax= float(VO2ss[arg_FatMax] / vo2max * 100),
        Fat_at_FatMax = float(Fat_util[arg_FatMax]) if arg_FatMax < len(Fat_util) else 0.0,
        CHO_at_AT     = float(CHO_util[0]) if len(CHO_util) > 0 else 0.0,
    )


def _glycogen(vo2max, vlamax, bw, body_fat_pct=14.0):
    fat_mass    = bw * body_fat_pct / 100
    lean_mass   = bw - fat_mass
    muscle_mass = lean_mass * _MUSCLE_PCT
    fitness     = ('elite'        if vo2max >= 65 and vlamax <= 0.5 else
                   'advanced'     if vo2max >= 50 and vlamax <= 0.7 else
                   'intermediate' if vo2max >= 40 and vlamax <= 0.9 else
                   'beginner')
    gly_per_kg  = {'elite':17,'advanced':15,'intermediate':14,'beginner':13}[fitness]
    total       = _LIVER_GLY + muscle_mass * gly_per_kg
    return dict(total=total, liver=_LIVER_GLY,
                muscle=muscle_mass*gly_per_kg,
                fitness=fitness, gly_per_kg=gly_per_kg,
                muscle_mass=muscle_mass)


def _fig_substrate(res, bw, modalidade):
    """Curva CHO + Fat vs Potência — estilo powerlab.icu."""
    W    = res['Intensity']
    n_AT = res['arg_AT']
    n_FM = res['arg_FatMax']

    fig = go.Figure()

    # Fat utilisation
    W_fat = W[:n_AT]
    fig.add_trace(go.Scatter(
        x=W_fat, y=res['Fat_util'],
        mode='lines', name='Fat (g/h)',
        line=dict(color=_C_FAT, width=3),
        fill='tozeroy',
        fillcolor=f"rgba(0,200,150,0.12)",
    ))

    # CHO utilisation
    n_cho = min(n_AT + 400, len(W))
    W_cho = W[:n_cho]
    fig.add_trace(go.Scatter(
        x=W_cho, y=res['CHO_util'][:len(W_cho)],
        mode='lines', name='CHO (g/h)',
        line=dict(color=_C_CHO, width=3),
        fill='tozeroy',
        fillcolor=f"rgba(255,107,53,0.12)",
    ))

    # FatMax
    fig.add_vline(x=res['W_FatMax'], line_dash='dot', line_color=_C_FAT, line_width=1.5,
                  annotation_text=f"FatMax {res['W_FatMax']:.0f}W",
                  annotation_font_color=_C_FAT, annotation_position="top right")

    # AT/MLSS
    fig.add_vline(x=res['W_AT'], line_dash='dot', line_color=_C_AT, line_width=1.5,
                  annotation_text=f"MLSS {res['W_AT']:.0f}W",
                  annotation_font_color=_C_AT, annotation_position="top right")

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"Utilização de Substratos — {modalidade}",
                   font=dict(size=14, color="#EEEEEE")),
        xaxis=dict(**_AXIS, title="Potência (W)"),
        yaxis=dict(**_AXIS, title="g / hora"),
    )
    return fig


def _fig_lactate(res, bw, vlamax):
    """Curva de lactato estacionário vs Potência — abaixo do AT."""
    W_below = res['Intensity'][:res['arg_AT']]
    CLass   = res['CLass']
    valid   = np.isfinite(CLass) & (CLass > 0) & (CLass < 20)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=W_below[valid], y=CLass[valid],
        mode='lines', name='[La] estacionário',
        line=dict(color=_C_LAC, width=3),
    ))

    # Linha 2 mmol/L (LT1 proxy) e 4 mmol/L (LT2 proxy)
    for y_val, lbl, col in [(2.0, 'LT1 ~2mmol','rgba(255,209,102,0.7)'),
                             (4.0, 'LT2 ~4mmol','rgba(230,57,70,0.7)')]:
        fig.add_hline(y=y_val, line_dash='dot', line_color=col, line_width=1.2,
                      annotation_text=lbl, annotation_font_color=col,
                      annotation_position="right")

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Lactato Estacionário vs Potência",
                   font=dict(size=14, color="#EEEEEE")),
        xaxis=dict(**_AXIS, title="Potência (W)"),
        yaxis=dict(**_AXIS, title="[Lactato] (mmol/L)", range=[0, 8]),
    )
    return fig


def _fig_vlamax_sensitivity(vo2max, bw, cp_watts, modalidade):
    """Sensibilidade: como VLamax afecta MLSS e FatMax."""
    vla_range = np.linspace(0.1, 1.2, 50)
    mlss_vals, fatmax_vals = [], []

    for vla in vla_range:
        try:
            r = _compute_metab(vo2max, vla, bw)
            mlss_vals.append(r['W_AT'])
            fatmax_vals.append(r['W_FatMax'])
        except Exception:
            mlss_vals.append(np.nan)
            fatmax_vals.append(np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vla_range, y=mlss_vals, mode='lines',
        name='MLSS (W)', line=dict(color=_C_AT, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=vla_range, y=fatmax_vals, mode='lines',
        name='FatMax (W)', line=dict(color=_C_FAT, width=2.5),
    ))

    # Linha CP
    fig.add_hline(y=cp_watts, line_dash='dot', line_color='#A855F7', line_width=1.5,
                  annotation_text=f"CP={cp_watts:.0f}W",
                  annotation_font_color='#A855F7', annotation_position="right")

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Sensibilidade ao VLamax",
                   font=dict(size=14, color="#EEEEEE")),
        xaxis=dict(**_AXIS, title="VLamax (mmol/L/s)"),
        yaxis=dict(**_AXIS, title="Potência (W)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
def tab_metab(ac_full: pd.DataFrame = None, wc_full: pd.DataFrame = None):
    """
    Tab de Perfil Metabólico.
    Entradas: CP (da tab_cp_model), Peso (wellness média mês),
              Pmax (p_max sheet), MMP3/MMP5 (ac_full), Altura, Idade.
    """

    st.markdown("""
    <style>
    .metab-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metab-val { font-size: 26px; font-weight: 600; margin: 0; }
    .metab-lbl { font-size: 11px; opacity: .6; margin: 0 0 3px; }
    .metab-sub { font-size: 11px; opacity: .5; margin: 3px 0 0; }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("🧪 Perfil Metabólico — Mader / Hauser / konaendu")
    with st.expander("ℹ️ Sobre este modelo", expanded=False):
        st.markdown("""
**Modelo:** Mader & Heck (1986), Mader (2003), Hauser (2014), konaendu/vlamax

**Fluxo de cálculo:**
1. **VLamax** é estimado a partir de CP, MMP3/5 e Pmax (sprint) via konaendu
2. **MLSS** (Limiar Anaeróbio) = ponto onde produção de La = combustão
3. **FatMax** = intensidade de máxima oxidação de gordura
4. **CHO/Fat** = utilização de substratos por intensidade (g/h)
5. **Lactato estacionário** = [La] abaixo do MLSS
6. **Glicogénio total** = reservas hepáticas + musculares

**Constantes:** Ks1=0.0631, Ks2=1.331, VolRel=0.40, Kel=4 (Hauser 2014)
        """)

    st.markdown("---")

    # ── 1. INPUTS ──────────────────────────────────────────────────────────────
    st.markdown("**⚙️ Inputs**")

    _col_mod = next((c for c in ['type','modality'] if ac_full is not None
                     and c in ac_full.columns), None)
    _mods_avail = (['Bike','Run','Row','Ski'] if _col_mod is None
                   else [m for m in ['Bike','Run','Row','Ski']
                         if m in ac_full[_col_mod].values])

    _ic1, _ic2, _ic3 = st.columns(3)
    with _ic1:
        modalidade = st.selectbox("Modalidade", _mods_avail, key="metab_mod")
    with _ic2:
        altura_cm = st.number_input("Altura (cm)", 140, 220, 178, 1, key="metab_h")
    with _ic3:
        idade = st.number_input("Idade (anos)", 15, 80, 40, 1, key="metab_age")

    # Peso: média do mês da coluna Peso da sheet Wellness
    _peso_default = 85.3
    if wc_full is not None and 'Peso' in wc_full.columns:
        _wc_p = wc_full.copy()
        _wc_p['Data'] = pd.to_datetime(_wc_p['Data'])
        _cutoff_peso  = pd.Timestamp.now() - pd.Timedelta(days=30)
        _peso_mes     = _wc_p[_wc_p['Data'] >= _cutoff_peso]['Peso'].dropna()
        if len(_peso_mes) >= 3:
            _peso_default = float(_peso_mes.mean())

    _ip1, _ip2, _ip3, _ip4 = st.columns(4)
    with _ip1:
        peso = st.number_input("Peso (kg) — média 30d", 40.0, 150.0,
                               round(_peso_default, 1), 0.5, key="metab_peso",
                               help="Média do mês da coluna Peso na sheet Wellness")

    # CP: input manual — idealmente vem do resultado da tab_cp_model
    with _ip2:
        cp_default = 223.0
        cp_watts = st.number_input(
            "CP (W)", 50, 600, int(cp_default), 5, key="metab_cp",
            help="CP do melhor modelo M1/M2/M3 na tab CP Model — Ranking SEE%"
        )

    # MMP3 e MMP5 da sheet de actividades, por modalidade
    _mmp3_val = _mmp5_val = None
    if ac_full is not None and _col_mod is not None:
        _ac_m = ac_full[ac_full[_col_mod] == modalidade].copy()
        _col_date = next((c for c in ['date','Data'] if c in _ac_m.columns), None)
        for _mc, _dur_t in [('MMP3', '_mmp3_val'), ('MMP5', '_mmp5_val')]:
            if _mc in _ac_m.columns and _col_date:
                _s = (_ac_m[[_mc, _col_date]]
                      .dropna(subset=[_mc])
                      .sort_values(_col_date, ascending=False))
                if not _s.empty:
                    try:
                        v = float(str(_s[_mc].iloc[0]).replace(',', '.'))
                        if v > 0:
                            if _mc == 'MMP3': _mmp3_val = v
                            else: _mmp5_val = v
                    except Exception:
                        pass

    with _ip3:
        mmp3 = st.number_input(
            "MMP3 (W)", 50, 1000,
            int(_mmp3_val) if _mmp3_val else int(cp_watts * 1.4),
            5, key="metab_mmp3",
            help="Potência máxima 3 minutos — puxado da sheet de actividades"
        )
    with _ip4:
        mmp5 = st.number_input(
            "MMP5 (W)", 50, 900,
            int(_mmp5_val) if _mmp5_val else int(cp_watts * 1.25),
            5, key="metab_mmp5",
            help="Potência máxima 5 minutos — puxado da sheet de actividades"
        )

    # Pmax: coluna p_max da sheet de actividades
    _pmax_val = None
    if ac_full is not None and _col_mod is not None and 'p_max' in ac_full.columns:
        _ac_m2 = ac_full[ac_full[_col_mod] == modalidade]
        _col_d2 = next((c for c in ['date','Data'] if c in _ac_m2.columns), None)
        if _col_d2:
            _px_s = (_ac_m2[['p_max', _col_d2]]
                     .dropna(subset=['p_max'])
                     .sort_values(_col_d2, ascending=False))
            if not _px_s.empty:
                try:
                    _pmax_val = float(_px_s['p_max'].iloc[0])
                except Exception:
                    pass

    _ip5, _ip6 = st.columns(2)
    with _ip5:
        pmax = st.number_input(
            "Pmax / Sprint (W)", 100, 3000,
            int(_pmax_val) if _pmax_val else int(cp_watts * 5),
            10, key="metab_pmax",
            help="p_max da sheet de actividades — potência máxima de sprint"
        )
    with _ip6:
        body_fat_pct = st.number_input(
            "% Gordura corporal", 5.0, 40.0, 14.0, 0.5, key="metab_bf",
            help="Padrão: 14% para triatleta fit masculino"
        )

    st.markdown("---")

    # ── 2. CALCULAR VLamax ─────────────────────────────────────────────────────
    # konaendu formula (README):
    # workload = (MMP3 + MMP5) / 3
    # volRel_vlamax = workload / bw  ← só para VLamax
    # VolRel = 0.40  ← constante fisiológica para o resto
    bmi              = peso / ((altura_cm / 100) ** 2)
    workload         = (mmp3 + mmp5) / 3
    volRel_vlamax    = workload / peso

    mader_formula      = (0.02049 / volRel_vlamax * (cp_watts / peso * 10.8 + 7)
                          * (bmi / 22)
                          * (1 + 0.000025 * idade - 0.0000001 * peso))
    sprint_contrib     = (0.000004 / volRel_vlamax * pmax
                          * (1 + 0.0000001 * idade - 0.0000001 * peso))
    vlamax_calc        = mader_formula + sprint_contrib
    vo2max_calc        = cp_watts / peso * 10.8 + 7  # Hawley & Noakes via CP

    # Override manual de VLamax se o utilizador quiser
    _vla1, _vla2, _vla3 = st.columns(3)
    with _vla1:
        st.metric("VO2max estimado", f"{vo2max_calc:.1f} ml/min/kg",
                  help="Via CP: VO2max ≈ (CP/peso) × 10.8 + 7 (Hawley & Noakes)")
    with _vla2:
        st.metric("VLamax calculado", f"{vlamax_calc:.3f} mmol/L/s",
                  help="konaendu formula — Mader/Hauser")
    with _vla3:
        vlamax = st.number_input(
            "VLamax (usar este)", 0.05, 2.0,
            round(float(np.clip(vlamax_calc, 0.05, 1.8)), 3),
            0.01, key="metab_vla",
            help="Podes ajustar manualmente se tiveres medição laboratorial"
        )

    st.markdown("---")

    # ── 3. COMPUTAR MODELO ────────────────────────────────────────────────────
    try:
        res  = _compute_metab(vo2max_calc, vlamax, peso)
        gly  = _glycogen(vo2max_calc, vlamax, peso, body_fat_pct)
    except Exception as e:
        st.error(f"Erro no modelo: {e}")
        return

    # ── 4. MÉTRICAS RESUMO (estilo powerlab.icu) ──────────────────────────────
    st.markdown("**📊 Resultados**")

    def _card(label, val, sub="", color="#EEEEEE"):
        return (f"<div class='metab-card'>"
                f"<p class='metab-lbl'>{label}</p>"
                f"<p class='metab-val' style='color:{color}'>{val}</p>"
                f"<p class='metab-sub'>{sub}</p></div>")

    _mc = st.columns(4)
    with _mc[0]:
        st.markdown(_card("MLSS / AT", f"{res['W_AT']:.0f} W",
                          f"{res['VO2_AT']:.1f} ml/min/kg · {res['pct_VO2_AT']:.0f}% VO2max",
                          _C_AT), unsafe_allow_html=True)
    with _mc[1]:
        st.markdown(_card("FatMax", f"{res['W_FatMax']:.0f} W",
                          f"{res['VO2_FatMax']:.1f} ml/min/kg · {res['pct_VO2_FatMax']:.0f}% VO2max",
                          _C_FAT), unsafe_allow_html=True)
    with _mc[2]:
        st.markdown(_card("Fat @ FatMax", f"{res['Fat_at_FatMax']:.0f} g/h",
                          "Oxidação máxima de gordura",
                          _C_FAT), unsafe_allow_html=True)
    with _mc[3]:
        st.markdown(_card("Glicogénio total", f"{gly['total']:.0f} g",
                          f"Fígado {gly['liver']:.0f}g + Músculo {gly['muscle']:.0f}g · {gly['fitness']}",
                          _C_CHO), unsafe_allow_html=True)

    _mc2 = st.columns(4)
    with _mc2[0]:
        st.markdown(_card("VLamax", f"{vlamax:.3f} mmol/L/s",
                          "Taxa máx. produção de lactato",
                          _C_VLAM), unsafe_allow_html=True)
    with _mc2[1]:
        st.markdown(_card("VO2max (est.)", f"{vo2max_calc:.1f} ml/min/kg",
                          f"Via CP={cp_watts}W, {peso}kg",
                          "#60A5FA"), unsafe_allow_html=True)
    with _mc2[2]:
        st.markdown(_card("CP vs MLSS", f"{cp_watts - res['W_AT']:+.0f} W",
                          f"CP={cp_watts}W · MLSS={res['W_AT']:.0f}W",
                          "#AAAAAA"), unsafe_allow_html=True)
    with _mc2[3]:
        st.markdown(_card("BMI", f"{bmi:.1f}",
                          f"{altura_cm}cm · {peso}kg",
                          "#AAAAAA"), unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. GRÁFICOS ────────────────────────────────────────────────────────────
    _g1, _g2 = st.columns(2)
    with _g1:
        st.plotly_chart(_fig_substrate(res, peso, modalidade),
                        use_container_width=True)
    with _g2:
        st.plotly_chart(_fig_lactate(res, peso, vlamax),
                        use_container_width=True)

    st.plotly_chart(_fig_vlamax_sensitivity(vo2max_calc, peso, cp_watts, modalidade),
                    use_container_width=True)

    # ── 6. TABELA DE ZONAS ────────────────────────────────────────────────────
    st.markdown("**📋 Zonas de Intensidade por Substrato**")
    st.caption("Potências calculadas via modelo Mader para este atleta.")

    W     = res['Intensity']
    VO2ss = res['VO2ss']
    W_AT  = res['W_AT']
    W_FM  = res['W_FatMax']

    # Pontos de referência específicos (% MLSS)
    zonas = []
    for pct, nome in [(0.50,'Recuperação (50% MLSS)'), (0.65,'Z1 — Aeróbio leve'),
                       (0.75,'Z2 — Aeróbio moderado'), (0.85,'Z2-Z3 — FatMax'),
                       (0.92,'Z3 — Limiar (MLSS)'),    (1.00,'MLSS'),
                       (1.05,'Z4 — Acima do MLSS'),    (1.15,'Z5 — VO2max')]:
        w_zona = W_AT * pct
        # Fat e CHO neste ponto
        idx = np.argmin(np.abs(W - w_zona)) if w_zona < W_AT else res['arg_AT']
        fat_g  = float(res['Fat_util'][idx]) if idx < len(res['Fat_util']) else 0.0
        cho_g  = float(res['CHO_util'][0]) if len(res['CHO_util']) > 0 else 0.0
        lac_v  = float(res['CLass'][idx]) if idx < len(res['CLass']) and np.isfinite(res['CLass'][idx]) else None
        zonas.append({
            'Zona':       nome,
            'Potência':   f"{w_zona:.0f} W",
            '% MLSS':     f"{pct*100:.0f}%",
            'Fat (g/h)':  f"{fat_g:.0f}" if w_zona <= W_AT else "—",
            'CHO (g/h)':  f"{cho_g:.0f}" if w_zona <= W_AT else "—",
            '[La] mmol/L':f"{lac_v:.2f}" if lac_v and w_zona <= W_AT else "—",
        })
    st.dataframe(pd.DataFrame(zonas), hide_index=True, use_container_width=True)

    # ── 7. DOWNLOAD ───────────────────────────────────────────────────────────
    _df_out = pd.DataFrame({
        'VO2max_est_ml_min_kg': [round(vo2max_calc, 2)],
        'VLamax_mmol_L_s':      [round(vlamax, 4)],
        'CP_W':                 [cp_watts],
        'MLSS_W':               [round(res['W_AT'], 1)],
        'MLSS_VO2_ml_min_kg':   [round(res['VO2_AT'], 2)],
        'MLSS_pct_VO2max':      [round(res['pct_VO2_AT'], 1)],
        'FatMax_W':             [round(res['W_FatMax'], 1)],
        'FatMax_VO2_ml_min_kg': [round(res['VO2_FatMax'], 2)],
        'FatMax_pct_VO2max':    [round(res['pct_VO2_FatMax'], 1)],
        'Fat_at_FatMax_g_h':    [round(res['Fat_at_FatMax'], 1)],
        'Glycogen_total_g':     [round(gly['total'], 0)],
        'Glycogen_liver_g':     [_LIVER_GLY],
        'Glycogen_muscle_g':    [round(gly['muscle'], 0)],
        'fitness_level':        [gly['fitness']],
        'peso_kg':              [peso],
        'BMI':                  [round(bmi, 1)],
        'modalidade':           [modalidade],
    })
    st.download_button(
        "⬇️ Download resultados CSV",
        _df_out.to_csv(index=False, sep=';', decimal=','),
        file_name=f"atheltica_metab_{modalidade}.csv",
        mime="text/csv",
        key="metab_dl"
    )
