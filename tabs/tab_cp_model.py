from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re as _re
import warnings
warnings.filterwarnings('ignore')

def tab_cp_model(ac_full=None):
    """CP Model Comparison v4 — weighted fitting automático, Veloclinic correcto."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import linregress
    import plotly.graph_objects as go

    # ── Paleta ───────────────────────────────────────────────────────────────
    CORES_M = {"M1":"#e74c3c","M2":"#2980b9","M3":"#27ae60","M4":"#8e44ad"}
    CORES_W = {"none":"#2c3e50","1/t":"#e67e22","1/t²":"#16a085"}
    NOMES   = {"M1":"M1: P vs 1/t","M2":"M2: Work-Time","M3":"M3: Hiperbólico-t","M4":"M4: 3-Param"}
    W_MODES = ["none","1/t","1/t²"]

    # ── Durações MMP disponíveis na sheet (segundos) ─────────────────────────
    # MMP1=60s MMP3=180s MMP5=300s MMP12=720s MMP20=1200s MMP60=3600s
    MMP_COLS = {
        'MMP1':  60,   'MMP3':  180,  'MMP5':  300,
        'MMP12': 720,  'MMP20': 1200, 'MMP60': 3600,
    }
    # TCPmax = 1800s (30 min) — ponto de inflexão do OmPD (Puchowicz et al. 2020)
    TCP_MAX = 1800.0

    def parse_mmp(val):
        """
        Extrai watts de MMP no formato real da sheet.
        Formatos aceites:
            "Yes - 618w"   → 618.0  (season best atingido — USAR)
            "No (PR: 383w)" → None  (não atingido — IGNORAR)
        Só retorna valor quando a linha começa com "Yes".
        """
        import re as _re
        if not isinstance(val, str) or not val.strip():
            return None
        v = val.strip()
        if not v.lower().startswith('yes'):
            return None
        m = _re.search(r'-\s*(\d+(?:\.\d+)?)\s*w', v, _re.IGNORECASE)
        return float(m.group(1)) if m else None

    BASE = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#111111", size=12),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc",
                    borderwidth=1, font=dict(color="#111111", size=11),
                    orientation="h", y=-0.30),
        margin=dict(t=60,b=90,l=65,r=30),
    )
    AX = dict(color="#111111", showgrid=True, gridcolor="#eeeeee",
              linecolor="#bbbbbb", linewidth=1, showline=True,
              tickfont=dict(color="#111111", size=11))

    # ════════════════════════════════════════════════════════════════════════
    # WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    def make_w(t_obs, mode):
        t = np.array(t_obs, dtype=float)
        if mode == "1/t":   return 1.0/t
        if mode == "1/t²":  return 1.0/t**2
        return np.ones_like(t)

    # ════════════════════════════════════════════════════════════════════════
    # MODELOS
    # ════════════════════════════════════════════════════════════════════════
    def fit_m1(tests, w):
        """M1: P = W′·(1/t) + CP  — WLS no espaço P"""
        x = np.array([1/t for _,t in tests])
        y = np.array([p   for p,_ in tests])
        W = np.diag(w); X = np.column_stack([x, np.ones_like(x)])
        try:
            b = np.linalg.lstsq(W@X, W@y, rcond=None)[0]
            wp, cp = float(b[0]), float(b[1])
        except Exception:
            sl,ic,_,_,_ = linregress(x,y); wp,cp = float(sl),float(ic)
        pp = [wp/t+cp for _,t in tests]
        ss_res = float(np.sum(w*(y-np.array([wp/t+cp for _,t in tests]))**2))
        ss_tot = float(np.sum(w*(y-np.average(y,weights=w))**2))
        r2 = max(0.0,1-ss_res/ss_tot) if ss_tot>0 else 0.0
        return float(cp), float(wp), None, pp, r2, 2

    def fit_m2(tests, w):
        """M2: W = CP·t + W′  — WLS no espaço W"""
        x = np.array([t   for _,t in tests])
        y = np.array([p*t for p,t in tests])
        W = np.diag(w); X = np.column_stack([x, np.ones_like(x)])
        try:
            b = np.linalg.lstsq(W@X, W@y, rcond=None)[0]
            cp, wp = float(b[0]), float(b[1])
        except Exception:
            sl,ic,_,_,_ = linregress(x,y); cp,wp = float(sl),float(ic)
        pp = [cp+wp/t for _,t in tests]
        ss_res = float(np.sum(w*(y-np.array([cp*t+wp for _,t in tests]))**2))
        ss_tot = float(np.sum(w*(y-np.average(y,weights=w))**2))
        r2 = max(0.0,1-ss_res/ss_tot) if ss_tot>0 else 0.0
        return float(cp), float(wp), None, pp, r2, 2

    def fit_m3(tests, w):
        """M3: t = W′/(P-CP)  — minimiza erro em TEMPO"""
        p_obs = np.array([p for p,_ in tests])
        t_obs = np.array([t for _,t in tests])
        cp_max = float(min(p_obs))*0.99
        def _loss(params):
            cp,wp = params
            if wp<=0 or cp>=cp_max or cp<=0: return 1e12
            t_pred = wp/(p_obs-cp)
            return float(np.sum(w*(t_obs-t_pred)**2))
        best = None
        for cp0 in np.linspace(float(min(p_obs))*0.50, float(min(p_obs))*0.94, 8):
            wp0 = float(np.mean(t_obs))*float(min(p_obs)-cp0)*0.5
            if wp0<=0: continue
            try:
                r = minimize(_loss,[cp0,wp0],bounds=[(1,cp_max),(1,1e7)],method="L-BFGS-B")
                if best is None or r.fun < best.fun: best = r
            except Exception: pass
        if best is None or best.fun>1e10: return None,None,None,None,None,2
        cp,wp = float(best.x[0]),float(best.x[1])
        pp = [wp/t+cp for _,t in tests]
        ss_res = float(np.sum(w*(t_obs-wp/(p_obs-cp))**2))
        ss_tot = float(np.sum(w*(t_obs-np.average(t_obs,weights=w))**2))
        r2 = max(0.0,1-ss_res/ss_tot) if ss_tot>0 else 0.0
        return cp,wp,None,pp,r2,2

    def fit_m4(tests, w):
        """M4: t = W′/(P-CP)·(1-(P-CP)/(Pmax-CP))  — 3 parâmetros"""
        p_obs = np.array([p for p,_ in tests])
        t_obs = np.array([t for _,t in tests])
        cp_max  = float(min(p_obs))*0.99
        pmax_lb = float(max(p_obs))*1.01
        def _t3(p,cp,wp,pmax):
            d = p-cp
            if np.any(d<=0) or np.any(p>=pmax): return np.full_like(p,1e9)
            return (wp/d)*(1-d/(pmax-cp))
        def _loss3(params):
            cp,wp,pmax = params
            if wp<=0 or cp<=0 or cp>=cp_max or pmax<=float(max(p_obs)): return 1e12
            t_pred = _t3(p_obs,cp,wp,pmax)
            if np.any(t_pred<=0): return 1e12
            return float(np.sum(w*(t_obs-t_pred)**2))
        best = None
        for cp0 in np.linspace(float(min(p_obs))*0.50,float(min(p_obs))*0.92,4):
            for pm0 in [float(max(p_obs))*f for f in [1.05,1.10,1.20]]:
                wp0 = float(np.mean(t_obs))*float(min(p_obs)-cp0)*0.4
                if wp0<=0: continue
                try:
                    r = minimize(_loss3,[cp0,wp0,pm0],
                                 bounds=[(1,cp_max),(1,1e7),(pmax_lb,pmax_lb*3)],
                                 method="L-BFGS-B")
                    if best is None or r.fun<best.fun: best=r
                except Exception: pass
        if best is None or best.fun>1e10: return None,None,None,None,None,3
        cp,wp,pmax = [float(x) for x in best.x]
        pp = [wp/t+cp for _,t in tests]
        ss_res = float(np.sum(w*(t_obs-_t3(p_obs,cp,wp,pmax))**2))
        ss_tot = float(np.sum(w*(t_obs-np.average(t_obs,weights=w))**2))
        r2 = max(0.0,1-ss_res/ss_tot) if ss_tot>0 else 0.0
        return cp,wp,pmax,pp,r2,3

    def fit_ompd(tests, pmax_ext=None):
        """
        M5: OmPD — Omni-Domain Power-Duration (Puchowicz, Baker & Clarke 2020)

        Para t ≤ TCPmax (1800s):
            P(t) = W′/t × (1 - exp(-t×(Pmax-CP)/W′)) + CP

        Para t > TCPmax:
            P(t) = mesma equação - A × ln(t/TCPmax)

        Parâmetros: CP, W′, Pmax (fixo de p_max da sheet), A (se t>TCPmax disponível)

        Wʼeff(t) = W′ × (1 - exp(-t×(Pmax-CP)/W′))  → plateia ~110s → consistente com
        interpretação de capacidade anaeróbica fixa (diferença vs OmExp/Om3CP).

        Se pmax_ext=None → inferido como max(p_obs)*1.15 (estimativa conservadora).
        Se não há ponto t>TCPmax → A=0 (modelo reduz a 3 parâmetros para curtas durações).
        """
        from scipy.optimize import minimize as _minimize

        p_obs_arr = np.array([p for p, _ in tests])
        t_obs_arr = np.array([t for _, t in tests])

        # Pmax: usar valor externo (da sheet) se disponível, senão estimar
        if pmax_ext is not None and pmax_ext > float(max(p_obs_arr)):
            pmax = float(pmax_ext)
        else:
            pmax = float(max(p_obs_arr)) * 1.15

        # Separar testes curtos (≤TCPmax) e longos (>TCPmax)
        mask_long  = t_obs_arr > TCP_MAX
        has_long   = bool(np.any(mask_long))

        # Função OmPD P(t) com ou sem extensão longa
        def _ompd_p(t_arr, cp, wp, A=0.0):
            tau  = wp / max(pmax - cp, 1.0)
            base = wp / t_arr * (1 - np.exp(-t_arr / tau)) + cp
            if A > 0:
                decay = np.where(
                    t_arr > TCP_MAX,
                    A * np.log(t_arr / TCP_MAX),
                    0.0
                )
                return base - decay
            return base

        # Loss: minimiza erro quadrático ponderado em potência
        # Peso 1/t → mais peso em esforços curtos (onde o modelo é mais sensível)
        def _loss(params):
            if has_long:
                cp, wp, A = params
                if A < 0: return 1e12
            else:
                cp, wp = params; A = 0.0
            if wp <= 0 or cp <= 0 or cp >= float(min(p_obs_arr)) * 0.99: return 1e12
            if cp >= pmax: return 1e12
            p_pred = _ompd_p(t_obs_arr, cp, wp, A)
            w_vec  = 1.0 / t_obs_arr  # peso 1/t
            return float(np.sum(w_vec * (p_obs_arr - p_pred) ** 2))

        best = None
        cp_max = float(min(p_obs_arr)) * 0.99
        # Grid de arranques
        for cp0 in np.linspace(float(min(p_obs_arr)) * 0.50,
                               float(min(p_obs_arr)) * 0.93, 6):
            wp0 = float(np.mean(t_obs_arr)) * (float(min(p_obs_arr)) - cp0) * 0.5
            if wp0 <= 0: continue
            try:
                if has_long:
                    x0     = [cp0, wp0, 30.0]
                    bounds = [(1, cp_max), (1, 1e7), (0, 500)]
                else:
                    x0     = [cp0, wp0]
                    bounds = [(1, cp_max), (1, 1e7)]
                r = _minimize(_loss, x0, bounds=bounds, method='L-BFGS-B')
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                pass

        if best is None or best.fun > 1e10:
            return None, None, None, None, None, None, None

        if has_long:
            cp, wp, A = float(best.x[0]), float(best.x[1]), float(best.x[2])
        else:
            cp, wp = float(best.x[0]), float(best.x[1]); A = 0.0

        p_pred_arr = _ompd_p(t_obs_arr, cp, wp, A)
        pp         = list(p_pred_arr)

        # R² em potência
        ss_res = float(np.sum((p_obs_arr - p_pred_arr) ** 2))
        ss_tot = float(np.sum((p_obs_arr - float(np.mean(p_obs_arr))) ** 2))
        r2     = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Wʼeff(120s) — verificar que atinge plateia (paper: ~110s)
        tau_fit   = wp / max(pmax - cp, 1.0)
        weff_120  = wp * (1 - np.exp(-120.0 / tau_fit))
        weff_pct  = weff_120 / wp * 100  # deve ser ≈ 99%

        return cp, wp, pmax, A, pp, r2, weff_pct

    FIT_FNS = {"M1":fit_m1,"M2":fit_m2,"M3":fit_m3,"M4":fit_m4}

    # ── SEE ─────────────────────────────────────────────────────────────────
    def calc_see(p_obs, pp, k=2):
        n = len(p_obs)
        if n<=k: return None,None
        sse  = float(np.sum((np.array(p_obs)-np.array(pp))**2))
        see  = float(np.sqrt(sse/max(n-k,1)))
        seep = see/float(np.mean(p_obs))*100
        return round(see,2),round(seep,2)

    # ── Veloclinic ───────────────────────────────────────────────────────────
    def veloclinic_points(tests, cp):
        """
        Veloclinic: scatter P vs W′_point = t*(P-CP).
        SEM curva teórica — seria W′_point = W′ (linha horizontal trivial).
        O diagnóstico está na distribuição dos pontos reais.
        """
        p_pts  = [p for p,_ in tests]
        wp_pts = [t*(p-cp) for p,t in tests]
        return p_pts, wp_pts

    # ── Métricas ─────────────────────────────────────────────────────────────
    def vc_metrics(tests, cp, wp):
        wp_pts = [t*(p-cp) for p,t in tests if p>cp]
        if not wp_pts: return {"std":0,"cv":0,"mean":0,"slope":0}
        std_w  = float(np.std(wp_pts))
        mean_w = float(np.mean(wp_pts))
        cv_w   = std_w/mean_w*100 if mean_w>0 else 0.0
        p_pts  = [p for p,t in tests if p>cp]
        sl = 0.0
        # Proteger contra valores idênticos (linregress falha com std=0)
        if len(p_pts) >= 2 and len(set(p_pts)) > 1:
            try:
                sl,_,_,_,_ = linregress(p_pts, wp_pts)
            except Exception:
                sl = 0.0
        return {"std":round(std_w,1),"cv":round(cv_w,1),
                "mean":round(mean_w,0),"slope":round(float(sl),4)}

    def classify_fatigue(vm):
        cv,sl = vm["cv"],abs(vm["slope"])
        if cv<10 and sl<1:   return "✅ Bom fit — W′ consistente"
        if cv>30:             return "🔵 Fadiga central (variabilidade)"
        if vm["mean"]<vm["std"]*2 and vm["mean"]>0:
                              return "🔴 Fadiga periférica (W′ reduzido)"
        if cv>15:             return "🟠 Fadiga sistémica"
        return "⚠️ Dados inconsistentes"

    # ════════════════════════════════════════════════════════════════════════
    # UI
    # ════════════════════════════════════════════════════════════════════════
    st.header("🏁 CP Model Comparison")
    st.caption(
        "Fitting automático nos 3 cenários de weighting (none / 1/t / 1/t²). "
        "Selecção baseada em estabilidade + SEE + Veloclinic.")

    hdr1,hdr2 = st.columns(2)
    with hdr1: modalidade = st.selectbox("Modalidade",["Bike","Run","Row","Ski"],key="cp_mod")
    with hdr2: data_teste = st.date_input("Data",key="cp_data")


    # ════════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════════
    # MODELOS ADICIONAIS
    # ════════════════════════════════════════════════════════════════════════

    def fit_2p_hyperbolic(tests):
        """2P Hiperbólico: P = W′/t + CP  (trabalho-tempo linear)
        Janela recomendada: 2min – 60min. Mínimo: 2 pontos."""
        from scipy.stats import linregress
        if len(tests) < 2: return None, None, None, None
        x = np.array([1.0/t for _, t in tests])
        y = np.array([p for p, _ in tests])
        slope, intercept, r, _, _ = linregress(x, y)
        cp = float(intercept); wp = float(slope)
        if cp <= 0 or wp <= 0: return None, None, None, None
        pp = [wp/t + cp for _, t in tests]
        return cp, wp, None, pp

    def fit_3p_hyperbolic(tests, pmax_ext=None):
        """3P Hiperbólico: P(t) = (Pmax·W′) / (W′ + (Pmax-CP)·t)
        Se pmax_ext disponível → Pmax FIXO (apenas 2 parâmetros livres: CP, W′).
        Sem pmax_ext → Pmax como 3º parâmetro livre (precisa ponto curto <30s)."""
        from scipy.optimize import minimize as _min
        if len(tests) < 2: return None, None, None, None
        p_obs = np.array([p for p, _ in tests])
        t_obs = np.array([t for _, t in tests])

        # Usar Pmax externo fixo se disponível → reduz a 2 parâmetros, muito mais estável
        if pmax_ext and float(pmax_ext) > float(max(p_obs)):
            pmax_fixed = float(pmax_ext)
            def _p3f(t, cp, wp):
                return (pmax_fixed * wp) / (wp + (pmax_fixed - cp) * t)
            def _loss2(params):
                cp, wp = params
                if cp <= 0 or wp <= 0 or cp >= min(p_obs)*0.99 or cp >= pmax_fixed: return 1e12
                pred = _p3f(t_obs, cp, wp)
                return float(np.sum((p_obs - pred)**2))
            best = None
            for cp0 in np.linspace(float(min(p_obs))*0.50, float(min(p_obs))*0.93, 8):
                wp0 = float(np.mean(t_obs))*(float(min(p_obs))-cp0)*0.5
                if wp0 <= 0: continue
                try:
                    r = _min(_loss2, [cp0, max(wp0,1)],
                             bounds=[(1, float(min(p_obs))*0.98), (1, 1e7)],
                             method='L-BFGS-B')
                    if best is None or r.fun < best.fun: best = r
                except Exception: pass
            if best is None or best.fun > 1e10: return None, None, None, None
            cp, wp = float(best.x[0]), float(best.x[1])
            pp = [float(_p3f(np.array([t]), cp, wp)[0]) for _, t in tests]
            return cp, wp, pmax_fixed, pp

        # Sem Pmax externo → optimizar os 3 parâmetros (precisa ponto curto para Pmax)
        def _p3(t, cp, wp, pmax):
            return (pmax * wp) / (wp + (pmax - cp) * t)
        def _loss3(params):
            cp, wp, pmax = params
            if cp<=0 or wp<=0 or pmax<=max(p_obs) or cp>=min(p_obs)*0.99: return 1e12
            pred = _p3(t_obs, cp, wp, pmax)
            return float(np.sum((p_obs - pred)**2))
        best = None
        for cp0 in np.linspace(float(min(p_obs))*0.5, float(min(p_obs))*0.92, 5):
            for pm0 in [float(max(p_obs))*f for f in [1.05,1.10,1.20,1.50,2.0]]:
                wp0 = float(np.mean(t_obs))*(float(min(p_obs))-cp0)*0.5
                if wp0 <= 0: continue
                try:
                    r = _min(_loss3, [cp0, max(wp0,1), pm0],
                             bounds=[(1, float(min(p_obs))*0.98), (1, 1e7),
                                     (float(max(p_obs))*1.01, float(max(p_obs))*3)],
                             method='L-BFGS-B')
                    if best is None or r.fun < best.fun: best = r
                except Exception: pass
        if best is None or best.fun > 1e10: return None, None, None, None
        cp, wp, pmax = float(best.x[0]), float(best.x[1]), float(best.x[2])
        pp = [float(_p3(np.array([t]), cp, wp, pmax)[0]) for _, t in tests]
        return cp, wp, pmax, pp

    def fit_ward_smith(tests, pmax_ext=None):
        """Ward-Smith (1999): extensão 3P com decaimento fisiológico.
        P(t) = CP + (Pmax-CP)·exp(-t·(Pmax-CP)/W′)
        Requer Pmax externo; sem ele usa estimativa conservadora."""
        from scipy.optimize import minimize as _min
        if len(tests) < 3: return None, None, None, None
        p_obs = np.array([p for p, _ in tests])
        t_obs = np.array([t for _, t in tests])
        pmax  = float(pmax_ext) if pmax_ext and pmax_ext > max(p_obs) else float(max(p_obs)) * 1.2

        def _pws(t, cp, wp):
            return cp + (pmax - cp) * np.exp(-t * (pmax - cp) / max(wp, 1.0))

        def _loss(params):
            cp, wp = params
            if cp <= 0 or wp <= 0 or cp >= min(p_obs)*0.99: return 1e12
            return float(np.sum((p_obs - _pws(t_obs, cp, wp))**2))

        best = None
        for cp0 in np.linspace(float(min(p_obs))*0.5, float(min(p_obs))*0.92, 6):
            wp0 = float(np.mean(t_obs)) * (float(min(p_obs)) - cp0) * 0.5
            try:
                r = _min(_loss, [cp0, max(wp0, 1)],
                         bounds=[(1, float(min(p_obs))*0.98), (1, 1e7)],
                         method='L-BFGS-B')
                if best is None or r.fun < best.fun: best = r
            except Exception: pass
        if best is None or best.fun > 1e10: return None, None, None, None
        cp, wp = float(best.x[0]), float(best.x[1])
        pp = [float(_pws(np.array([t]), cp, wp)[0]) for _, t in tests]
        return cp, wp, pmax, pp

    def fit_om3cp(tests, pmax_ext=None):
        """Om3CP (Omni-3CP): OmPD com 3P base em vez de 2P.
        P(t) = W′/t × f(t,Pmax,CP) + CP, âncora em τ de 3P Pmax."""
        from scipy.optimize import minimize as _min
        if len(tests) < 2: return None, None, None, None
        p_obs = np.array([p for p, _ in tests])
        t_obs = np.array([t for _, t in tests])
        pmax  = float(pmax_ext) if pmax_ext and pmax_ext > max(p_obs) else float(max(p_obs)) * 1.15

        def _pom3(t, cp, wp, A_om=0.0):
            tau  = wp / max(pmax - cp, 1.0)
            base = wp / t * (1 - np.exp(-t / tau)) + cp
            if A_om > 0:
                decay = np.where(t > TCP_MAX, A_om * np.log(t / TCP_MAX), 0.0)
                return base - decay
            return base

        mask_long = t_obs > TCP_MAX
        has_long  = bool(np.any(mask_long))

        def _loss(params):
            cp, wp = params[0], params[1]
            A_om   = params[2] if has_long else 0.0
            if cp <= 0 or wp <= 0 or cp >= min(p_obs)*0.99 or cp >= pmax: return 1e12
            pred = _pom3(t_obs, cp, wp, A_om)
            return float(np.sum((1.0/t_obs) * (p_obs - pred)**2))

        best = None
        for cp0 in np.linspace(float(min(p_obs))*0.50, float(min(p_obs))*0.93, 6):
            wp0 = float(np.mean(t_obs)) * (float(min(p_obs)) - cp0) * 0.5
            if wp0 <= 0: continue
            try:
                x0 = [cp0, wp0, 30.0] if has_long else [cp0, wp0]
                bd = [(1, float(min(p_obs))*0.98), (1, 1e7)]
                if has_long: bd.append((0, 500))
                r = _min(_loss, x0, bounds=bd, method='L-BFGS-B')
                if best is None or r.fun < best.fun: best = r
            except Exception: pass
        if best is None or best.fun > 1e10: return None, None, None, None
        cp, wp = float(best.x[0]), float(best.x[1])
        A_om   = float(best.x[2]) if has_long else 0.0
        pp = [float(_pom3(np.array([t]), cp, wp, A_om)[0]) for _, t in tests]
        return cp, wp, pmax, pp

    def fit_omexp(tests, pmax_ext=None):
        """OmExp: variante OmPD com decaimento exponencial para t > TCPmax.
        P(t) = OmPD_base(t) para t≤TCPmax
        P(t) = OmPD_base(t) × exp(-A_e × (t-TCPmax)/TCPmax) para t>TCPmax"""
        from scipy.optimize import minimize as _min
        if len(tests) < 2: return None, None, None, None
        p_obs = np.array([p for p, _ in tests])
        t_obs = np.array([t for _, t in tests])
        pmax  = float(pmax_ext) if pmax_ext and pmax_ext > max(p_obs) else float(max(p_obs)) * 1.15

        def _pomexp(t, cp, wp, A_e=0.0):
            tau  = wp / max(pmax - cp, 1.0)
            base = wp / t * (1 - np.exp(-t / tau)) + cp
            if A_e > 0:
                decay = np.where(t > TCP_MAX,
                                 (1 - np.exp(-A_e * (t - TCP_MAX) / TCP_MAX)),
                                 0.0)
                return base * (1 - decay * 0.15)
            return base

        mask_long = t_obs > TCP_MAX
        has_long  = bool(np.any(mask_long))

        def _loss(params):
            cp, wp = params[0], params[1]
            A_e = params[2] if has_long else 0.0
            if cp <= 0 or wp <= 0 or cp >= min(p_obs)*0.99 or cp >= pmax: return 1e12
            pred = _pomexp(t_obs, cp, wp, A_e)
            return float(np.sum((1.0/t_obs) * (p_obs - pred)**2))

        best = None
        for cp0 in np.linspace(float(min(p_obs))*0.50, float(min(p_obs))*0.93, 6):
            wp0 = float(np.mean(t_obs)) * (float(min(p_obs)) - cp0) * 0.5
            if wp0 <= 0: continue
            try:
                x0 = [cp0, wp0, 1.0] if has_long else [cp0, wp0]
                bd = [(1, float(min(p_obs))*0.98), (1, 1e7)]
                if has_long: bd.append((0, 10))
                r = _min(_loss, x0, bounds=bd, method='L-BFGS-B')
                if best is None or r.fun < best.fun: best = r
            except Exception: pass
        if best is None or best.fun > 1e10: return None, None, None, None
        cp, wp = float(best.x[0]), float(best.x[1])
        A_e = float(best.x[2]) if has_long else 0.0
        pp = [float(_pomexp(np.array([t]), cp, wp, A_e)[0]) for _, t in tests]
        return cp, wp, pmax, pp

    def fit_power_law(tests):
        """Power Law: P = a × t^(-b). Sem CP explícito.
        log(P) = log(a) - b×log(t) — regressão linear no espaço log-log."""
        from scipy.stats import linregress
        if len(tests) < 2: return None, None, None, None
        x = np.log([t for _, t in tests])
        y = np.log([p for p, _ in tests])
        slope, intercept, r, _, _ = linregress(x, y)
        b = -float(slope); a = float(np.exp(intercept))
        if a <= 0 or b <= 0: return None, None, None, None
        pp = [a * t**(-b) for _, t in tests]
        # CP implícito ~ P(3600s)
        cp_impl = a * 3600.0**(-b)
        return cp_impl, a, b, pp  # (cp_proxy, a, b, pp)

    # ════════════════════════════════════════════════════════════════════════
    # GRID SEARCH: todas as combinações de MMPs por modelo
    # ════════════════════════════════════════════════════════════════════════

    def _grid_search_model(fit_fn, all_mmp_pts, min_pts, pmax_ext=None, k_params=2):
        """
        Testa todas as combinações de N pontos (N >= min_pts) dos MMPs disponíveis.
        Retorna a combinação com menor SEE%.
        fit_fn(tests, pmax_ext=None) → (cp, wp, pmax_or_extra, pp)
        """
        from itertools import combinations
        if len(all_mmp_pts) < min_pts:
            return None
        best = {'see_pct': 999, 'result': None, 'combo': None}
        for combo in combinations(range(len(all_mmp_pts)), min_pts):
            pts = [all_mmp_pts[i] for i in combo]
            try:
                if pmax_ext is not None:
                    res = fit_fn(pts, pmax_ext=pmax_ext)
                else:
                    res = fit_fn(pts)
                if res[0] is None or res[-1] is None: continue
                cp, pp = res[0], res[-1]
                p_obs  = [p for p, _ in pts]
                _, see_pct = calc_see(p_obs, pp, k=k_params)
                if see_pct is not None and see_pct < best['see_pct']:
                    best = {'see_pct': see_pct, 'result': res, 'combo': pts,
                            'n_pts': len(pts), 'cp': cp}
            except Exception:
                pass
        # Também testar com todos os pontos
        try:
            if pmax_ext is not None:
                res = fit_fn(all_mmp_pts, pmax_ext=pmax_ext)
            else:
                res = fit_fn(all_mmp_pts)
            if res[0] is not None and res[-1] is not None:
                p_obs = [p for p, _ in all_mmp_pts]
                _, see_pct = calc_see(p_obs, res[-1], k=k_params)
                if see_pct is not None and see_pct < best['see_pct']:
                    best = {'see_pct': see_pct, 'result': res, 'combo': all_mmp_pts,
                            'n_pts': len(all_mmp_pts), 'cp': res[0]}
        except Exception:
            pass
        return best if best['result'] is not None else None

    def _plot_model_curve(tests, pp, cp, wp, model_name, color="#2980b9",
                          extra_params=None, t_range=(1, 10800)):
        """Gráfico Plotly simples: pontos observados + curva do modelo."""
        t_curve = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[t for _, t in tests], y=[p for p, _ in tests],
            mode='markers', name='MMP observado',
            marker=dict(size=9, color='#e74c3c', symbol='circle'),
        ))
        fig.add_trace(go.Scatter(
            x=list(t_curve), y=pp if len(pp) == len(t_curve) else None,
            mode='lines', name=model_name,
            line=dict(color=color, width=2.5),
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**AX, title="Duração (s)", type='log'),
            yaxis=dict(**AX, title="Potência (W)"),
            title=dict(text=model_name, font=dict(size=14, color='#111111')),
        )
        return fig

    # ════════════════════════════════════════════════════════════════════════
    # ESTRUTURA DE TABS
    # ════════════════════════════════════════════════════════════════════════
    _tab_rank, _tab_ompd, _tab_m1, _tab_m2, _tab_m3, \
        _tab_2p, _tab_3p, _tab_ws, _tab_om3, \
        _tab_omexp, _tab_pl, _tab_manual = st.tabs([
        "🏆 Ranking SEE%",
        "🏅 OmPD",
        "📐 M1: P vs 1/t",
        "📐 M2: Work-Time",
        "📐 M3: Hiperbólico-t",
        "📐 2P Hiperbólico",
        "📐 3P Hiperbólico",
        "📐 Ward-Smith",
        "📐 Om3CP",
        "📐 OmExp",
        "📐 Power Law",
        "🔬 Testes Manuais",
    ])

    # ── Extrair MMPs da sheet ─────────────────────────────────────────────────
    # MMP60: sempre só validação. Nunca entra no fitting.
    # Row/Ski M1/M2/M3: usar MMP1, MMP5, MMP12 (sem MMP3 nem MMP20)
    # Row/Ski outros modelos: usar MMP1, MMP3, MMP5, MMP12, MMP20
    # Bike/Run: usar MMP1, MMP3, MMP5, MMP12, MMP20

    _all_mmp_pts      = []   # M1/M2/M3 Row/Ski: MMP1+MMP5+MMP12
    _all_mmp_pts_full = []   # outros modelos: todos excluindo MMP60
    _mmp60_val        = None
    _pmax_global      = None

    if ac_full is not None and len(ac_full) > 0:
        _col_mod  = next((c for c in ['type','modality'] if c in ac_full.columns), None)
        _col_date = next((c for c in ['date','Data'] if c in ac_full.columns), None)
        if _col_mod and _col_date:
            _ac_mod = ac_full[ac_full[_col_mod] == modalidade].copy()
            for _mc, _dur in MMP_COLS.items():
                if _mc not in _ac_mod.columns: continue
                _dur_f = float(_dur)
                _ac_mod_s = _ac_mod.sort_values(_col_date, ascending=False)

                # MMP60 → sempre só validação, nunca fitting
                if _mc == 'MMP60':
                    for _, _rr in _ac_mod_s.iterrows():
                        _mv = parse_mmp(str(_rr[_mc]))
                        if _mv is not None:
                            _mmp60_val = _mv; break
                    continue

                # Ler o valor
                _mv = None
                for _, _rr in _ac_mod_s.iterrows():
                    _mv = parse_mmp(str(_rr[_mc]))
                    if _mv is not None: break
                if _mv is None: continue

                # _all_mmp_pts — M1/M2/M3:
                #   Row/Ski → MMP3(180s) + MMP5(300s) + MMP12(720s)
                #   Bike/Run → todos exceto MMP60
                if modalidade in ('Row', 'Ski'):
                    if _dur_f in (180.0, 300.0, 720.0):
                        _all_mmp_pts.append((_mv, _dur_f))
                else:
                    _all_mmp_pts.append((_mv, _dur_f))

                # _all_mmp_pts_full — modelos não-clássicos:
                #   Row/Ski → MMP3+MMP5+MMP12+MMP20 (excluir MMP1 e MMP60)
                #   Bike/Run → todos exceto MMP60
                if modalidade in ('Row', 'Ski'):
                    if _dur_f != 60.0:   # excluir MMP1
                        _all_mmp_pts_full.append((_mv, _dur_f))
                else:
                    _all_mmp_pts_full.append((_mv, _dur_f))

            _all_mmp_pts      = sorted(set(_all_mmp_pts),      key=lambda x: x[1])
            _all_mmp_pts_full = sorted(set(_all_mmp_pts_full), key=lambda x: x[1])

            # Pmax
            if 'p_max' in _ac_mod.columns:
                _px = (_ac_mod[['p_max', _col_date]]
                       .dropna(subset=['p_max'])
                       .sort_values(_col_date, ascending=False))
                if not _px.empty:
                    _pmax_global = float(_px['p_max'].iloc[0])

    # Modelos clássicos para CP Row/Ski — definido aqui para uso no grid search
    _M_CLASSICOS = ('M1: P vs 1/t', 'M2: Work-Time', 'M3: Hiperbólico-t')

    # Cada modelo usa exactamente 3 pontos (papers)
    _FIXED_N_PTS = 3

    # ── Correr grid search para todos os modelos ──────────────────────────
    _MODELS = {
        # Modelos do Ranking automático (grid search 3 pontos)
        'M1: P vs 1/t':   {'fn': lambda pts, **kw: fit_m1(pts, np.ones(len(pts))),
                            'n_pts': 3, 'k': 2, 'color': '#e74c3c',
                            'needs_pmax': False,
                            'desc': 'P = W′/t + CP. Regressão linear. Monod & Scherrer 1965.'},
        'M2: Work-Time':  {'fn': lambda pts, **kw: fit_m2(pts, np.ones(len(pts))),
                            'n_pts': 3, 'k': 2, 'color': '#2980b9',
                            'needs_pmax': False,
                            'desc': 'W = CP·t + W′. Espaço trabalho-tempo. Morton 1986.'},
        'M3: Hiperbólico-t':{'fn': lambda pts, **kw: fit_m3(pts, np.ones(len(pts))),
                             'n_pts': 3, 'k': 2, 'color': '#27ae60',
                             'needs_pmax': False,
                             'desc': 't = W′/(P-CP). Minimiza erro em tempo. Mais robusto.'},
        'OmPD':          {'fn': fit_ompd,          'n_pts': 3, 'k': 2,
                          'color': '#8e44ad', 'needs_pmax': True,
                          'desc': '3 pts (incluindo ≤3min para Pmax). Puchowicz 2020.'},
        '2P Hiperbólico':{'fn': fit_2p_hyperbolic, 'n_pts': 3, 'k': 2,
                          'color': '#1abc9c', 'needs_pmax': False,
                          'desc': '3 pts entre 3-20min. Monod & Scherrer 1965.'},
        '3P Hiperbólico':{'fn': fit_3p_hyperbolic, 'n_pts': 3, 'k': 2,
                          'color': '#f39c12', 'needs_pmax': True,
                          'desc': '3 pts incluindo ≤3min. Morton 1996.'},
        'Ward-Smith':    {'fn': fit_ward_smith,    'n_pts': 3, 'k': 2,
                          'color': '#e67e22', 'needs_pmax': True,
                          'desc': '3 pts entre 3-20min (excluir <60s). Ward-Smith 1999.',
                          'exclude_short': True},
        'Om3CP':         {'fn': fit_om3cp,         'n_pts': 3, 'k': 2,
                          'color': '#16a085', 'needs_pmax': True,
                          'desc': '3 pts incluindo ≤3min. Puchowicz variante.'},
        'OmExp':         {'fn': fit_omexp,         'n_pts': 3, 'k': 2,
                          'color': '#d35400', 'needs_pmax': True,
                          'desc': '3 pts. Variante OmPD com decaimento exponencial.'},
        'Power Law':     {'fn': fit_power_law,     'n_pts': 3, 'k': 2,
                          'color': '#c0392b', 'needs_pmax': False,
                          'desc': '3 pts entre 3-20min. Sem CP explícito.'},
    }

    # Grid search: exactamente N=3 pontos (C(n,3) combinações)
    # Ward-Smith exclui pontos <60s
    _results = {}
    if _all_mmp_pts_full:
        for _mn, _mcfg in _MODELS.items():
            _px    = _pmax_global if _mcfg['needs_pmax'] else None
            # M1/M2/M3 Row/Ski: usar MMP1+MMP5+MMP12 apenas
            # Todos os outros: usar conjunto completo (MMP1+MMP3+MMP5+MMP12+MMP20)
            _pts_base = (_all_mmp_pts
                         if _mn in _M_CLASSICOS and modalidade in ('Row','Ski')
                         else _all_mmp_pts_full)
            _pts  = ([p for p in _pts_base if p[1] >= 60]
                     if _mcfg.get('exclude_short') else _pts_base)
            _n_pts = _mcfg['n_pts']
            if len(_pts) < _n_pts: continue

            from itertools import combinations as _comb
            _best = {'see_pct': 999, 'result': None, 'combo': None,
                     'cp_vals': [], 'see_vals': []}

            for _cb in _comb(range(len(_pts)), _n_pts):
                _combo_pts = [_pts[i] for i in _cb]
                try:
                    _res = (_mcfg['fn'](_combo_pts, pmax_ext=_px)
                            if _mcfg['needs_pmax']
                            else _mcfg['fn'](_combo_pts))
                    if _res[0] is None: continue
                    # Extrair pp conforme o tipo de modelo:
                    # OmPD: (cp,wp,pmax,A,pp,r2,weff) → idx 4
                    # M1/M2/M3: (cp,wp,None,pp,r2,k)  → idx 3
                    # Outros: (cp,wp,pmax,pp)           → idx -1 (= idx 3)
                    if _mn == 'OmPD':
                        _pp_gs = _res[4] if len(_res) > 4 else None
                    elif _mn in ('M1: P vs 1/t', 'M2: Work-Time', 'M3: Hiperbólico-t'):
                        _pp_gs = _res[3] if len(_res) > 3 else None
                    else:
                        _pp_gs = _res[-1]
                    if not isinstance(_pp_gs, (list, np.ndarray)) or len(_pp_gs) == 0: continue
                    _p_obs = [p for p,_ in _combo_pts]
                    _, _see = calc_see(_p_obs, _pp_gs, k=_mcfg['k'])
                    if _see is None: continue
                    _best['cp_vals'].append(float(_res[0]))
                    _best['see_vals'].append(float(_see))
                    if _see < _best['see_pct']:
                        _best.update({'see_pct': _see, 'result': _res,
                                      'combo': _combo_pts, 'n_pts': _n_pts,
                                      'cp': float(_res[0])})
                except Exception:
                    pass

            if _best['result'] is not None:
                # cp_var% = variação de CP entre combinações
                _cp_v = _best['cp_vals']
                _see_v = _best['see_vals']
                _cp_mean  = float(np.mean(_cp_v)) if _cp_v else 0
                # None quando só 1 fit — sem variação a medir (igual a N/A nos testes manuais)
                _cp_range = ((max(_cp_v)-min(_cp_v))/_cp_mean*100
                             if _cp_mean>0 and len(_cp_v)>1 else None)
                _see_mean = float(np.mean(_see_v)) if _see_v else _best['see_pct']
                _cv_pct   = (float(np.std(_cp_v)/_cp_mean*100)
                             if _cp_mean>0 and len(_cp_v)>1 else None)

                # Quando só há 1 combinação (ex: Row/Ski com C(3,3)=1),
                # usar variação cross-modelos (M1/M2/M3) como proxy de estabilidade
                if _cp_range is None or _cv_pct is None:
                    _cp_classicos = [
                        _results[m]['cp']
                        for m in ('M1: P vs 1/t','M2: Work-Time','M3: Hiperbólico-t')
                        if m in _results and _results[m].get('cp')
                    ]
                    if len(_cp_classicos) > 1:
                        _cm = float(np.mean(_cp_classicos))
                        if _cm > 0:
                            _cp_range = (max(_cp_classicos)-min(_cp_classicos))/_cm*100
                            _cv_pct   = float(np.std(_cp_classicos)/_cm*100)

                # Validação MMP60: erro relativo se disponível
                _mmp60_err = None
                if _mmp60_val and _best['result'][0]:
                    _cp_best = _best['result'][0]
                    _wp_best = _best['result'][1] if len(_best['result'])>1 else 0
                    _pred60  = (_wp_best/3600 + _cp_best) if isinstance(_wp_best, float) and _wp_best > 0 else None
                    if _pred60:
                        _mmp60_err = abs(_pred60 - _mmp60_val) / _mmp60_val * 100

                # Score composto (igual ao testes manuais)
                _sc = (0.40*((_cp_range or 0)/30) +
                       0.30*(_see_mean/20) +
                       0.20*((_cv_pct or 0)/20) +
                       0.10*((_mmp60_err or 0)/20))
                _best['cp_var_pct'] = round(_cp_range, 1) if _cp_range is not None else None
                _best['cv_pct']     = round(_cv_pct, 1)   if _cv_pct   is not None else None
                _best['see_mean']   = round(_see_mean, 2)
                _best['score']      = round(_sc*100, 1)
                _best['mmp60_err']  = round(_mmp60_err, 1) if _mmp60_err else None
                _results[_mn]       = _best

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — RANKING SEE%
    # ════════════════════════════════════════════════════════════════════════
    with _tab_rank:
        st.subheader(f"🏆 Ranking por SEE% — {modalidade}")
        st.caption(
            "Grid search automático: todas as combinações de MMPs disponíveis "
            "foram testadas para cada modelo. Mostra a melhor combinação de pontos "
            "para cada modelo (menor SEE%). SEE% < 2% = excelente | < 5% = bom."
        )

        if not _all_mmp_pts:
            st.warning("Sem pontos MMP disponíveis. Verifica se a modalidade tem dados 'Yes' na sheet.")
            st.stop()

        if modalidade in ('Row', 'Ski'):
            st.info(
                f"ℹ️ **{modalidade}:** "
                f"M1/M2/M3 usam MMP3+MMP5+MMP12 | "
                f"Outros modelos usam MMP3+MMP5+MMP12+MMP20 (sem MMP1). "
                "MMP60 sempre excluído do fitting (usado como validação)."
            )

        _pts_display = _all_mmp_pts_full if _all_mmp_pts_full else _all_mmp_pts
        st.info(f"**Pontos disponíveis ({modalidade}):** " +
                " · ".join([f"{int(t//60)}min={p:.0f}W" for p, t in _pts_display]) +
                (f" | **Pmax:** {_pmax_global:.0f}W" if _pmax_global
                 else " | ⚠️ Pmax não encontrado — modelos Omni usarão estimativa"))

        if not _results:
            st.error("Nenhum modelo convergiu com os dados disponíveis.")
        else:
            st.caption(
                "**CP var%** = amplitude de CP entre combinações (ou entre M1/M2/M3 quando só há 1 combinação). "
                "**CV%** = desvio-padrão/média (≤5% = consistente). "
                "**SEE% médio** = erro médio. "
                "**Score** = composto ponderado (menor = melhor)."
            )

            # Semáforos (definidos fora do loop)
            def _flag_cp_var(v):
                if not isinstance(v, float): return '—'
                return f"{'✅' if v<5 else '⚠️' if v<15 else '❌'} {v:.1f}%"
            def _flag_cv(v):
                if not isinstance(v, float): return '—'
                return f"{'✅' if v<5 else '⚠️' if v<10 else '❌'} {v:.1f}%"
            def _flag_see(v):
                if not isinstance(v, float): return '—'
                return f"{'✅' if v<2 else '⚠️' if v<5 else '❌'} {v:.2f}%"

            # Modelos clássicos para CP Row/Ski
            _M_CP_FILTER = _M_CLASSICOS if modalidade in ('Row', 'Ski') else None

            _rank_rows = []
            for _mn, _gr in sorted(_results.items(), key=lambda x: x[1]['score']):
                _res  = _gr['result']
                _cp_v = _gr.get('cp')
                _wp_v = _res[1] if len(_res) > 1 else None
                _pts_lbl = " + ".join([f"{int(t//60)}min" for _, t in _gr['combo']])
                _n_combos = len(_gr['cp_vals'])
                _cp_var  = _gr.get('cp_var_pct', '—')
                _cv      = _gr.get('cv_pct', '—')
                _see_b   = round(_gr['see_pct'], 2)
                _see_m   = _gr.get('see_mean', '—')
                _score   = _gr.get('score', '—')
                _m60_err = _gr.get('mmp60_err')

                # Classificação de fadiga via veloclinic
                _fat_lbl = '—'
                try:
                    if _cp_v and _wp_v and isinstance(_wp_v, float) and _wp_v > 0:
                        _vm = vc_metrics(_gr['combo'], _cp_v, _wp_v)
                        _fat_lbl = classify_fatigue(_vm)
                except Exception:
                    pass

                _rank_rows.append({
                    'Modelo':       _mn,
                    'CP (W)':       f"{_cp_v:.0f}" if _cp_v else "—",
                    "W′ (J)":       f"{_wp_v:.0f}" if isinstance(_wp_v, float) and _wp_v > 100 else "—",
                    'SEE% melhor':  _flag_see(_see_b),
                    'SEE% médio':   _flag_see(float(_see_m)) if isinstance(_see_m, float) else '—',
                    'CP var%':      _flag_cp_var(float(_cp_var)) if isinstance(_cp_var, float) else '—',
                    'CV%':          _flag_cv(float(_cv)) if isinstance(_cv, float) else '—',
                    'Fadiga':       _fat_lbl,
                    'Val. 60min':   f"{_m60_err:.1f}%" if _m60_err else ('sem dado' if _mmp60_val is None else '—'),
                    'N combos':     _n_combos,
                    'Score ↓':      _score,
                    'Melhores 3pts':_pts_lbl,
                })
            _rank_df = pd.DataFrame(_rank_rows)
            st.dataframe(_rank_df, hide_index=True, use_container_width=True)

            st.caption(
                "✅ Excelente | ⚠️ Aceitável | ❌ Problemático — "
                f"3 pontos por modelo | MMP60 excluído do fitting "
                + (f"(validação: 60min={_mmp60_val:.0f}W)" if _mmp60_val else "(MMP60 não disponível)")
                + (f" | **Row/Ski: CP seleccionado apenas entre M1, M2, M3**" if modalidade in ('Row','Ski') else "")
            )

            # ── Melhor modelo global ──────────────────────────────────────
            _best_mn  = sorted(_results.items(), key=lambda x: x[1]['score'])[0]
            _best_lbl, _best_gr = _best_mn

            # Para Row/Ski: melhor CP apenas entre M1/M2/M3
            if _M_CP_FILTER:
                _res_classicos = {k:v for k,v in _results.items() if k in _M_CP_FILTER}
                if _res_classicos:
                    _best_cp_mn   = sorted(_res_classicos.items(), key=lambda x: x[1]['score'])[0]
                    _best_cp_lbl, _best_cp_gr = _best_cp_mn
                    _best_cp_val  = _best_cp_gr.get('cp')
                    _best_cp_res  = _best_cp_gr['result']
                    _best_cp_wp   = _best_cp_res[1] if len(_best_cp_res) > 1 else None

                    # Guardar no session_state para tab_metab
                    if _best_cp_val:
                        _ss_key = f"cp_model_{modalidade}"
                        st.session_state[_ss_key] = {
                            'cp': float(_best_cp_val),
                            'wp': float(_best_cp_wp) if isinstance(_best_cp_wp, float) else None,
                            'modelo': _best_cp_lbl,
                            'score': _best_cp_gr.get('score'),
                        }
                    st.success(
                        f"**CP para {modalidade} (M1/M2/M3): {_best_cp_lbl}** | "
                        f"Score={_best_cp_gr['score']} | "
                        f"SEE% melhor={_best_cp_gr['see_pct']:.2f}% | "
                        f"CP={_best_cp_val:.0f}W" +
                        (f" | W′={_best_cp_wp:.0f}J" if isinstance(_best_cp_wp, float) and _best_cp_wp > 100 else "") +
                        f" | Pontos: " + " + ".join([f"{int(t//60)}min" for _, t in _best_cp_gr['combo']])
                    )
                    st.info(
                        f"**Melhor modelo global (todos):** {_best_lbl} | "
                        f"Score={_best_gr['score']} | CP={_best_gr.get('cp',0):.0f}W — "
                        "para Row/Ski, o CP recomendado é do modelo clássico acima."
                    )
                else:
                    st.warning("Nenhum modelo clássico (M1/M2/M3) convergiu.")
            else:
                _best_res = _best_gr['result']
                _best_cp  = _best_gr.get('cp')
                _best_wp  = _best_res[1] if len(_best_res) > 1 else None

                # Guardar no session_state para tab_metab
                if _best_cp:
                    st.session_state[f"cp_model_{modalidade}"] = {
                        'cp': float(_best_cp),
                        'wp': float(_best_wp) if isinstance(_best_wp, float) else None,
                        'modelo': _best_lbl,
                        'score': _best_gr.get('score'),
                    }
                st.success(
                    f"**Modelo mais confiável: {_best_lbl}** | "
                    f"Score={_best_gr['score']} | "
                    f"SEE%={_best_gr['see_pct']:.2f}% | "
                    f"CP={_best_cp:.0f}W" +
                    (f" | W′={_best_wp:.0f}J" if isinstance(_best_wp, float) and _best_wp > 100 else "") +
                    f" | Pontos: " + " + ".join([f"{int(t//60)}min" for _, t in _best_gr['combo']])
                )

            # ── Gráfico P-D comparativo (eixo X em MINUTOS) ──────────────
            _fig_comp = go.Figure()
            _t_comp_s = np.logspace(np.log10(30), np.log10(10800), 400)
            _t_comp_m = _t_comp_s / 60.0   # converter para minutos no eixo X
            _colors_rank = ['#e74c3c','#2980b9','#27ae60','#8e44ad',
                            '#e67e22','#16a085','#d35400','#c0392b',
                            '#f39c12','#1abc9c']

            # Pontos observados (X em minutos)
            _fig_comp.add_trace(go.Scatter(
                x=[t/60 for _, t in _all_mmp_pts],
                y=[p for p, _ in _all_mmp_pts],
                mode='markers+text', name='MMP',
                marker=dict(size=10, color='#111111'),
                text=[f"{int(t//60)}min" for _, t in _all_mmp_pts],
                textposition='top center', textfont=dict(size=9),
            ))
            if _mmp60_val:
                _fig_comp.add_trace(go.Scatter(
                    x=[60.0], y=[_mmp60_val],
                    mode='markers', name=f'MMP60 validação ({_mmp60_val:.0f}W)',
                    marker=dict(size=10, color='#e67e22', symbol='diamond'),
                ))

            for (_mn, _gr), _col in zip(
                    sorted(_results.items(), key=lambda x: x[1]['score']),
                    _colors_rank):
                _res = _gr['result']
                try:
                    if _mn in ('M1: P vs 1/t', 'M2: Work-Time',
                               'M3: Hiperbólico-t', '2P Hiperbólico'):
                        _y_comp = _res[1] / _t_comp_s + _res[0]
                    elif _mn == 'OmPD':
                        _cp2,_wp2,_pm2,_A2 = _res[0],_res[1],_res[2],_res[3]
                        _pm2 = _pm2 or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.15
                        _tau2 = _wp2/max(_pm2-_cp2,1.0)
                        _y_comp = _wp2/_t_comp_s*(1-np.exp(-_t_comp_s/_tau2))+_cp2
                        if _A2 and _A2>0:
                            _y_comp -= np.where(_t_comp_s>TCP_MAX, _A2*np.log(_t_comp_s/TCP_MAX),0)
                    elif _mn == '3P Hiperbólico':
                        _cp2,_wp2,_pm2 = _res[0],_res[1],_res[2]
                        _y_comp = (_pm2*_wp2)/(_wp2+(_pm2-_cp2)*_t_comp_s)
                    elif _mn == 'Ward-Smith':
                        _cp2,_wp2 = _res[0],_res[1]
                        _pm2 = _res[2] or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.2
                        _y_comp = _cp2+(_pm2-_cp2)*np.exp(-_t_comp_s*(_pm2-_cp2)/max(_wp2,1))
                    elif _mn == 'Power Law':
                        _a_pl,_b_pl = _res[1],_res[2]
                        _y_comp = _a_pl*_t_comp_s**(-_b_pl)
                    elif _mn in ('Om3CP','OmExp'):
                        _cp2,_wp2 = _res[0],_res[1]
                        _pm2 = _res[2] or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.15
                        _tau2 = _wp2/max(_pm2-_cp2,1.0)
                        _y_comp = _wp2/_t_comp_s*(1-np.exp(-_t_comp_s/_tau2))+_cp2
                    else:
                        continue

                    _is_cp_model = _mn in _M_CLASSICOS
                    _line_w = 2.5 if (_M_CP_FILTER and _is_cp_model) else 1.5
                    _dash   = 'solid' if (_M_CP_FILTER and _is_cp_model) or not _M_CP_FILTER else 'dot'

                    _fig_comp.add_trace(go.Scatter(
                        x=list(_t_comp_m), y=list(_y_comp),
                        mode='lines',
                        name=f"{_mn} (SEE={_gr['see_pct']:.1f}%)",
                        line=dict(color=_col, width=_line_w, dash=_dash),
                    ))
                except Exception:
                    pass

            # Linha horizontal CP do melhor modelo clássico (Row/Ski)
            if _M_CP_FILTER and _res_classicos:
                _fig_comp.add_hline(
                    y=_best_cp_val, line_dash='dot', line_color='#27ae60', line_width=1.5,
                    annotation_text=f"CP={_best_cp_val:.0f}W ({_best_cp_lbl})",
                    annotation_position="right",
                )

            _fig_comp.update_layout(
                **BASE,
                title=dict(text=f"Comparação de modelos — {modalidade} "
                           f"({'M1/M2/M3 a cheio, outros pontilhados' if _M_CP_FILTER else 'todos os modelos'})",
                           font=dict(size=13)),
                xaxis=dict(**AX, title="Duração (min)", type='log',
                           tickvals=[1,2,3,5,10,20,30,60,120,180],
                           ticktext=['1','2','3','5','10','20','30','60','120','180']),
                yaxis=dict(**AX, title="Potência (W)"),
            )
            st.plotly_chart(_fig_comp, use_container_width=True)

            # ── Veloclinic Plot ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("**🔬 Veloclinic Plot — diagnóstico de W′**")
            st.caption(
                "Eixo X = Potência (W) | Eixo Y = W′_ponto = t×(P−CP) para cada MMP. "
                "W′ consistente → pontos alinhados horizontalmente. "
                "Declive positivo = fadiga central (W′ cresce com P). "
                "Declive negativo = fadiga periférica (W′ cai com P)."
            )

            # Usar CP do melhor modelo clássico (Row/Ski) ou melhor global
            _cp_vc  = _best_cp_val  if _M_CP_FILTER and _res_classicos else _best_gr.get('cp', 0)
            _wp_vc  = _best_cp_gr['result'][1] if _M_CP_FILTER and _res_classicos else _best_gr['result'][1]
            _lbl_vc = _best_cp_lbl if _M_CP_FILTER and _res_classicos else _best_lbl

            if _cp_vc and _cp_vc > 0:
                _p_vc  = [p for p,_ in _all_mmp_pts if p > _cp_vc]
                _wp_vc_pts = [t*(p-_cp_vc) for p,t in _all_mmp_pts if p > _cp_vc]
                _lbl_vc_pts = [f"{int(t//60)}min" for p,t in _all_mmp_pts if p > _cp_vc]

                if len(_p_vc) >= 2:
                    _vm_vc = vc_metrics(_all_mmp_pts, _cp_vc, _wp_vc)
                    _fat_vc = classify_fatigue(_vm_vc)

                    _fig_vc = go.Figure()
                    _fig_vc.add_trace(go.Scatter(
                        x=_p_vc, y=_wp_vc_pts,
                        mode='markers+text',
                        name='W′ por ponto',
                        marker=dict(size=11, color='#8e44ad'),
                        text=_lbl_vc_pts,
                        textposition='top center',
                        textfont=dict(size=9),
                    ))
                    # Linha horizontal de W′ médio
                    _fig_vc.add_hline(
                        y=_vm_vc['mean'], line_dash='dash', line_color='#27ae60',
                        annotation_text=f"W′ médio={_vm_vc['mean']:.0f}J",
                        annotation_position="right",
                    )
                    # Linha de regressão se slope significativo
                    if len(_p_vc) >= 2 and len(set(_p_vc)) > 1:
                        _x_reg = np.array(_p_vc)
                        _y_reg = _vm_vc['slope']*_x_reg + (_vm_vc['mean'] - _vm_vc['slope']*np.mean(_x_reg))
                        _fig_vc.add_trace(go.Scatter(
                            x=list(_x_reg), y=list(_y_reg),
                            mode='lines', name='Tendência',
                            line=dict(color='#e74c3c', dash='dot', width=1.5),
                        ))
                    _fig_vc.update_layout(
                        **BASE,
                        title=dict(text=f"Veloclinic — {modalidade} | CP={_cp_vc:.0f}W ({_lbl_vc}) | {_fat_vc}",
                                   font=dict(size=13)),
                        xaxis=dict(**AX, title="Potência (W)"),
                        yaxis=dict(**AX, title="W′ pontual = t×(P−CP)  [J]"),
                    )
                    st.plotly_chart(_fig_vc, use_container_width=True)

                    _vc1, _vc2, _vc3, _vc4 = st.columns(4)
                    _vc1.metric("W′ médio", f"{_vm_vc['mean']:.0f} J")
                    _vc2.metric("CV%", f"{_vm_vc['cv']:.1f}%")
                    _vc3.metric("Declive", f"{_vm_vc['slope']:.1f}")
                    _vc4.metric("Fadiga", _fat_vc)
                else:
                    st.info("Poucos pontos acima do CP para o Veloclinic Plot.")

            # ── Calculadora P ↔ t ────────────────────────────────────────
            st.markdown("---")
            st.markdown("**🧮 Calculadora — Potência ↔ Tempo**")

            # CP e W′ a usar: melhor clássico (Row/Ski) ou melhor global
            _calc_cp  = (_best_cp_val  if _M_CP_FILTER and _res_classicos
                         else _best_gr.get('cp', 0))
            _calc_wp  = (_best_cp_res[1] if _M_CP_FILTER and _res_classicos
                         else (_best_gr['result'][1] if _best_gr.get('result') else 0))
            _calc_lbl = (_best_cp_lbl  if _M_CP_FILTER and _res_classicos
                         else _best_lbl)

            st.caption(
                f"Modelo: **{_calc_lbl}** | CP = **{_calc_cp:.0f} W** | W′ = **{_calc_wp:.0f} J** | "
                "Equação: W′ = (P − CP) × t"
            )

            if not _calc_cp or not _calc_wp or _calc_cp <= 0 or _calc_wp <= 0:
                st.warning("CP ou W′ não disponível.")
            else:
                def _fmt_duration(secs):
                    """Formata segundos em MM:SS ou HH:MM:SS se ≥ 3600s."""
                    secs = max(0, int(round(secs)))
                    h = secs // 3600
                    m = (secs % 3600) // 60
                    s = secs % 60
                    if h > 0:
                        return f"{h:02d}:{m:02d}:{s:02d}"
                    return f"{m:02d}:{s:02d}"

                def _parse_time(txt):
                    """Aceita MM, MM:SS ou HH:MM:SS. Retorna segundos ou None."""
                    try:
                        parts = [int(x) for x in txt.strip().split(":")]
                        if len(parts) == 1:   return parts[0] * 60
                        if len(parts) == 2:   return parts[0] * 60 + parts[1]
                        if len(parts) == 3:   return parts[0]*3600 + parts[1]*60 + parts[2]
                    except Exception:
                        return None

                # ── Linha 1: Tempo → Potência ─────────────────────────────
                _r1a, _r1b = st.columns(2)
                with _r1a:
                    _t_input = st.text_input(
                        "⏱️ Tempo (MM:SS ou HH:MM:SS)",
                        value="20:00", key="calc_t_input",
                        help="Ex: 5:00 = 5 min | 1:00:00 = 1 hora"
                    )
                with _r1b:
                    _t_secs = _parse_time(_t_input)
                    if _t_secs is None or _t_secs <= 0:
                        st.markdown("⚠️ Formato inválido (use MM:SS ou HH:MM:SS)", unsafe_allow_html=True)
                    else:
                        _p_res = _calc_cp + _calc_wp / _t_secs
                        _min_d = _t_secs // 60; _sec_d = _t_secs % 60
                        st.metric(
                            f"Potência para {_fmt_duration(_t_secs)}",
                            f"{_p_res:.0f} W",
                            delta=f"{_p_res - _calc_cp:+.0f}W vs CP | {_p_res/_calc_cp*100:.1f}% CP",
                            delta_color="off"
                        )

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                # ── Linha 2: Potência → Tempo ─────────────────────────────
                _r2a, _r2b = st.columns(2)
                with _r2a:
                    _p_input = st.number_input(
                        "⚡ Potência (W)",
                        min_value=1, max_value=5000,
                        value=int(_calc_cp) + 30, step=5,
                        key="calc_p_input"
                    )
                with _r2b:
                    if _p_input <= _calc_cp:
                        st.metric(
                            f"Duração a {_p_input}W",
                            "∞",
                            delta=f"{_p_input}W ≤ CP — abaixo do limiar anaeróbio",
                            delta_color="off"
                        )
                    else:
                        _t_res = _calc_wp / (_p_input - _calc_cp)
                        st.metric(
                            f"Duração a {_p_input}W",
                            _fmt_duration(_t_res),
                            delta=f"{_p_input - _calc_cp:.0f}W acima do CP | W′/s = {_p_input-_calc_cp:.0f}J/s",
                            delta_color="off"
                        )

            # ── Perfil Metabólico (Mader / Hauser / INSCYD / konaendu) ──────────
            st.markdown("---")
            st.markdown("**🧪 Perfil Metabólico — Mader / Hauser / konaendu**")

            with st.expander("ℹ️ Sobre o modelo", expanded=False):
                st.markdown("""
**Modelo:** Mader & Heck (1986), Mader (2003), Hauser (2014), konaendu/vlamax

**VLamax** é estimado automaticamente a partir de MMP3, MMP5 e Pmax — sem necessidade de laboratório.
Van Schuylenbergh et al. (2026) confirmam que o protocolo de sprint produz VLamax consistente
independentemente da resistência usada (bias ±0 entre 3 protocolos distintos).

**VO2max** é calculado via método INSCYD: subtrai a contribuição glicolítica (VLamax-dependente)
dos esforços TT3 e TT6, dando uma estimativa mais precisa do que a fórmula linear simples.
Bias vs espirometria: -0.21 ml/min/kg, 95% CI: -2.46 a +2.0 (Van Schuylenbergh 2026).

**Outputs:** MLSS/AT · FatMax · CHO/Fat utilização · Lactato estacionário · Glicogénio
                """)

            # ── Inputs ────────────────────────────────────────────────────────
            _mb1, _mb2, _mb3, _mb4 = st.columns(4)
            with _mb1:
                _mb_altura = st.number_input("Altura (cm)", 140, 220, 178, 1,
                                             key="mb_altura_rank")
            with _mb2:
                _mb_idade  = st.number_input("Idade (anos)", 15, 80, 40, 1,
                                             key="mb_idade_rank")
            with _mb3:
                _mb_peso = st.number_input("Peso (kg)", 40.0, 150.0, 85.0, 0.5,
                                           key="mb_peso_rank",
                                           help="Média do mês — sheet Wellness")
            with _mb4:
                _mb_bf = st.number_input("% Gordura", 5.0, 40.0, 14.0, 0.5,
                                         key="mb_bf_rank")

            # MMP3 e MMP5 — puxados da sheet
            _mb_mmp3 = _mb_mmp5 = None
            if ac_full is not None and _col_mod is not None:
                _ac_mb = ac_full[ac_full[_col_mod] == modalidade].copy()
                _col_date_mb = next((c for c in ['date','Data']
                                     if c in _ac_mb.columns), None)
                for _mc_mb in ['MMP3','MMP5']:
                    if _mc_mb in _ac_mb.columns and _col_date_mb:
                        _s_mb = (_ac_mb[[_mc_mb,_col_date_mb]]
                                 .dropna(subset=[_mc_mb])
                                 .sort_values(_col_date_mb, ascending=False))
                        if not _s_mb.empty:
                            try:
                                _v = float(str(_s_mb[_mc_mb].iloc[0]).replace(',','.'))
                                if _v > 0:
                                    if _mc_mb == 'MMP3': _mb_mmp3 = _v
                                    else: _mb_mmp5 = _v
                            except Exception: pass

            _mb5, _mb6 = st.columns(2)
            with _mb5:
                _mb_mmp3_in = st.number_input(
                    "MMP3 (W) — 3 min",
                    50, 1000,
                    int(_mb_mmp3) if _mb_mmp3 else int(_calc_cp * 1.4),
                    5, key="mb_mmp3_rank",
                    help="Potência máxima 3 min — puxado automaticamente")
            with _mb6:
                _mb_mmp5_in = st.number_input(
                    "MMP5 (W) — 5 min",
                    50, 900,
                    int(_mb_mmp5) if _mb_mmp5 else int(_calc_cp * 1.25),
                    5, key="mb_mmp5_rank",
                    help="Potência máxima 5 min — puxado automaticamente")

            # ── VLamax automático (konaendu) — sem override manual ────────────
            # volRel_vlamax ≠ VolRel (0.40). volRel_vlamax é só para a fórmula VLamax.
            _mb_bmi        = _mb_peso / ((_mb_altura / 100) ** 2)
            _mb_workload   = (_mb_mmp3_in + _mb_mmp5_in) / 3
            _mb_volrel_vla = _mb_workload / _mb_peso  # só para VLamax

            # VO2max corrigido pelo VLamax (método INSCYD):
            # Subtrai contribuição glicolítica dos TT3 e TT6
            # W_glicolítica (3min) ≈ VLamax × VolRel × bw × 60 × 180 × 5.5 ml/mmol
            # Mas VLamax é desconhecido → iterar: estimar VO2max inicial → VLamax → VO2max corrigido
            _mb_VO2rest = 5.0
            _mb_VolRel  = 0.40
            _mb_Ks1 = 0.0631; _mb_Ks2 = 1.331; _mb_Kel = 4.0
            _mb_LAC_O2 = 0.01576; _mb_Watt_O2 = 11.685

            # Iteração 1: VO2max inicial via Hawley
            _mb_vo2max_0 = _calc_cp / _mb_peso * 10.8 + 7

            # Iteração 2: calcular VLamax com VO2max_0
            _mb_mader  = (0.02049 / _mb_volrel_vla * _mb_vo2max_0 * (_mb_bmi / 22)
                          * (1 + 0.000025 * _mb_idade - 0.0000001 * _mb_peso))
            _mb_sprint = (0.000004 / _mb_volrel_vla * (_pmax_global or _calc_cp * 4)
                          * (1 + 0.0000001 * _mb_idade - 0.0000001 * _mb_peso))
            _mb_vlamax = float(np.clip(_mb_mader + _mb_sprint, 0.05, 1.8))

            # Iteração 3: VO2max corrigido (subtrai contribuição glicolítica TT3+TT5)
            # VLamax_contrib ≈ VLamax × 60 × t_s × VolRel × bw × 5.5 / bw / t_s
            #                = VLamax × 60 × VolRel × 5.5  [ml/min/kg]
            _mb_vla_o2_equiv   = _mb_vlamax * 60 * _mb_VolRel * 5.5  # ml/min/kg
            _mb_vo2_aerob_mmp3 = max(1.0, _mb_mmp3_in / _mb_peso * 10.8 + 7 - _mb_vla_o2_equiv)
            _mb_vo2_aerob_mmp5 = max(1.0, _mb_mmp5_in / _mb_peso * 10.8 + 7 - _mb_vla_o2_equiv * 0.85)
            _mb_vo2max         = float(np.clip(
                (_mb_vo2_aerob_mmp3 + _mb_vo2_aerob_mmp5) / 2,
                20, 95))

            # Iteração 4: recalcular VLamax com VO2max corrigido
            _mb_mader2  = (0.02049 / _mb_volrel_vla * _mb_vo2max * (_mb_bmi / 22)
                           * (1 + 0.000025 * _mb_idade - 0.0000001 * _mb_peso))
            _mb_sprint2 = (0.000004 / _mb_volrel_vla * (_pmax_global or _calc_cp * 4)
                           * (1 + 0.0000001 * _mb_idade - 0.0000001 * _mb_peso))
            _mb_vlamax  = float(np.clip(_mb_mader2 + _mb_sprint2, 0.05, 1.8))

            # Classificação do perfil fisiológico
            _mb_perfil = ('🏔️ Endurance puro'    if _mb_vlamax < 0.3 else
                          '⚖️ Endurance/Speed'   if _mb_vlamax < 0.5 else
                          '⚡ Speed/Power'        if _mb_vlamax < 0.8 else
                          '💥 Sprint/Anaeróbio')

            # Mostrar VLamax e VO2max como informação — sem input de override
            _mbc1, _mbc2, _mbc3, _mbc4 = st.columns(4)
            _mbc1.metric("VO2max estimado",
                         f"{_mb_vo2max:.1f} ml/min/kg",
                         help="INSCYD method: subtrai contribuição glicolítica dos TT3/TT5")
            _mbc2.metric("VLamax estimado",
                         f"{_mb_vlamax:.3f} mmol/L/s",
                         help="konaendu formula (Mader/Hauser) — automático via MMP3+MMP5+Pmax")
            _mbc3.metric("Perfil fisiológico", _mb_perfil)
            _mbc4.metric("VO2max (Hawley simples)",
                         f"{_mb_vo2max_0:.1f} ml/min/kg",
                         help="CP/peso × 10.8 + 7 — referência sem correcção VLamax")

            # ── Modelo Mader completo ─────────────────────────────────────────
            try:
                _mb_VO2ss  = np.arange(0.5, _mb_vo2max - 0.05, 0.01)
                _mb_ADP    = np.sqrt(np.maximum(0,
                    (_mb_Ks1 * _mb_VO2ss) / (_mb_vo2max - _mb_VO2ss)))
                _mb_vLass  = (60 * _mb_vlamax /
                              (1 + (_mb_Ks2 / np.maximum(_mb_ADP ** 3, 1e-12))))
                _mb_LaComb = (_mb_LAC_O2 / _mb_VolRel) * _mb_VO2ss
                _mb_vLanet = _mb_vLass - _mb_LaComb
                _mb_argAT  = int(np.argmin(np.abs(_mb_vLanet)))
                _mb_overall= ((_mb_vLass * (_mb_VolRel * _mb_peso) *
                               ((1 / 4.3) * 22.4) / _mb_peso) + _mb_VO2ss)
                _mb_Watts  = np.maximum(0,
                    (_mb_overall - _mb_VO2rest) * _mb_peso / _mb_Watt_O2)

                _mb_argFM  = (int(np.argmax(-_mb_vLanet[:_mb_argAT]))
                              if _mb_argAT > 1 else 0)
                _mb_Fat    = np.maximum(0,
                    (-_mb_vLanet[:_mb_argAT]) * _mb_VolRel /
                    _mb_LAC_O2 * _mb_peso * 60 * 4.65 / 9.5 / 1000)
                _mb_CHO_g  = (_mb_peso * _mb_VolRel) * 60 / 1000 / 2 * 162.14

                _mb_W_AT   = float(_mb_Watts[_mb_argAT])
                _mb_W_FM   = float(_mb_Watts[_mb_argFM])
                _mb_pct_AT = float(_mb_VO2ss[_mb_argAT] / _mb_vo2max * 100)
                _mb_pct_FM = float(_mb_VO2ss[_mb_argFM] / _mb_vo2max * 100)
                _mb_fat_FM = (float(_mb_Fat[_mb_argFM])
                              if _mb_argFM < len(_mb_Fat) else 0.0)

                # Lactato estacionário abaixo do AT
                with np.errstate(divide='ignore', invalid='ignore'):
                    _mb_denom = (
                        (_mb_LAC_O2 / _mb_VolRel) * _mb_VO2ss[:_mb_argAT] *
                        (1 + (_mb_Ks2 /
                              np.maximum((_mb_Ks1 * _mb_VO2ss[:_mb_argAT]) /
                              np.maximum(_mb_vo2max - _mb_VO2ss[:_mb_argAT], 0.01),
                              1e-9)) ** 1.5) - _mb_vlamax * 60)
                    _mb_CLass = np.where(
                        _mb_denom > 0,
                        np.sqrt(np.maximum(0,
                            (_mb_vlamax * _mb_Kel * 60) / _mb_denom)),
                        np.nan)

                # Glicogénio
                _mb_fat_kg    = _mb_peso * _mb_bf / 100
                _mb_lean      = _mb_peso - _mb_fat_kg
                _mb_muscle_kg = _mb_lean * 0.70
                _mb_fitness   = ('elite'         if _mb_vo2max >= 65 and _mb_vlamax <= 0.5 else
                                 'advanced'      if _mb_vo2max >= 50 and _mb_vlamax <= 0.7 else
                                 'intermediate'  if _mb_vo2max >= 40 and _mb_vlamax <= 0.9
                                 else 'beginner')
                _mb_gly_kg    = {'elite':17,'advanced':15,'intermediate':14,'beginner':13}[_mb_fitness]
                _mb_gly_total = 90 + _mb_muscle_kg * _mb_gly_kg

                # ── Métricas de output ────────────────────────────────────────
                st.markdown("**📊 Resultados**")
                _mr1, _mr2, _mr3, _mr4 = st.columns(4)
                _mr1.metric("MLSS / AT", f"{_mb_W_AT:.0f} W",
                            f"{_mb_VO2ss[_mb_argAT]:.1f} ml/min/kg · {_mb_pct_AT:.0f}% VO2max")
                _mr2.metric("FatMax", f"{_mb_W_FM:.0f} W",
                            f"{_mb_VO2ss[_mb_argFM]:.1f} ml/min/kg · {_mb_pct_FM:.0f}% VO2max")
                _mr3.metric("Fat @ FatMax", f"{_mb_fat_FM:.0f} g/h",
                            "Oxidação máxima de gordura")
                _mr4.metric("Glicogénio total", f"{_mb_gly_total:.0f} g",
                            f"Fígado 90g + Músculo {_mb_muscle_kg*_mb_gly_kg:.0f}g · {_mb_fitness}")

                _mr5, _mr6, _mr7, _mr8 = st.columns(4)
                _mr5.metric("CP vs MLSS", f"{_calc_cp - _mb_W_AT:+.0f} W",
                            f"CP={_calc_cp}W · MLSS={_mb_W_AT:.0f}W")
                _mr6.metric("CHO utilização", f"{_mb_CHO_g:.0f} g/h",
                            "Constante até AT")
                _mr7.metric("% MLSS @ CP",
                            f"{_calc_cp / _mb_W_AT * 100:.0f}%",
                            help="% do MLSS a que o CP ocorre")
                _mr8.metric("FatMax / MLSS",
                            f"{_mb_W_FM / _mb_W_AT * 100:.0f}%",
                            "FatMax como % do MLSS")

                # ── Gráficos ──────────────────────────────────────────────────
                _BASE_MB = dict(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11), margin=dict(l=45, r=20, t=45, b=40),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                _AX_MB = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                              zeroline=False)

                _gc1, _gc2 = st.columns(2)
                with _gc1:
                    # Curva Substratos — Fat + CHO vs Potência
                    _fig_sb = go.Figure()
                    _n_cho  = min(_mb_argAT + 400, len(_mb_Watts))
                    _fig_sb.add_trace(go.Scatter(
                        x=_mb_Watts[:_mb_argAT], y=_mb_Fat,
                        mode='lines', name='Fat (g/h)',
                        line=dict(color='#00C896', width=2.5),
                        fill='tozeroy', fillcolor='rgba(0,200,150,0.10)',
                    ))
                    _fig_sb.add_trace(go.Scatter(
                        x=_mb_Watts[:_n_cho],
                        y=np.full(_n_cho, _mb_CHO_g),
                        mode='lines', name='CHO (g/h)',
                        line=dict(color='#FF6B35', width=2.5),
                        fill='tozeroy', fillcolor='rgba(255,107,53,0.10)',
                    ))
                    # MMP3 e MMP5 marcados
                    for _mmp_w, _mmp_lbl in [(_mb_mmp3_in,'MMP3'),(_mb_mmp5_in,'MMP5')]:
                        if _mmp_w < _mb_W_AT:
                            _fig_sb.add_vline(x=_mmp_w, line_dash='dash',
                                              line_color='rgba(180,180,180,0.5)',
                                              line_width=1,
                                              annotation_text=_mmp_lbl,
                                              annotation_font_color='rgba(180,180,180,0.8)')
                    _fig_sb.add_vline(x=_mb_W_FM, line_dash='dot',
                                      line_color='#00C896', line_width=1.5,
                                      annotation_text=f"FatMax {_mb_W_FM:.0f}W",
                                      annotation_font_color='#00C896')
                    _fig_sb.add_vline(x=_mb_W_AT, line_dash='dot',
                                      line_color='#FFD166', line_width=1.5,
                                      annotation_text=f"MLSS {_mb_W_AT:.0f}W",
                                      annotation_font_color='#FFD166')
                    # CP marcado
                    _fig_sb.add_vline(x=_calc_cp, line_dash='dash',
                                      line_color='#A855F7', line_width=1.5,
                                      annotation_text=f"CP {_calc_cp}W",
                                      annotation_font_color='#A855F7',
                                      annotation_position="top left")
                    _fig_sb.update_layout(
                        **_BASE_MB,
                        title=dict(text=f"Substratos — {modalidade} | VLamax={_mb_vlamax:.3f}",
                                   font=dict(size=13)),
                        xaxis=dict(**_AX_MB, title="Potência (W)"),
                        yaxis=dict(**_AX_MB, title="g / hora"),
                    )
                    st.plotly_chart(_fig_sb, use_container_width=True)

                with _gc2:
                    # Curva Lactato estacionário
                    _mb_W_below = _mb_Watts[:_mb_argAT]
                    _valid_la   = (np.isfinite(_mb_CLass) & (_mb_CLass > 0)
                                   & (_mb_CLass < 20))
                    _fig_la = go.Figure()
                    if _valid_la.any():
                        _fig_la.add_trace(go.Scatter(
                            x=_mb_W_below[_valid_la], y=_mb_CLass[_valid_la],
                            mode='lines', name='[La] mmol/L',
                            line=dict(color='#E63946', width=2.5),
                        ))
                    for _y_la, _lbl_la, _col_la in [
                        (2.0, 'LT1 ~2 mmol', 'rgba(255,209,102,0.8)'),
                        (4.0, 'LT2 ~4 mmol', 'rgba(230,57,70,0.8)')]:
                        _fig_la.add_hline(y=_y_la, line_dash='dot',
                                          line_color=_col_la, line_width=1.2,
                                          annotation_text=_lbl_la,
                                          annotation_font_color=_col_la,
                                          annotation_position="right")
                    _fig_la.update_layout(
                        **_BASE_MB,
                        title=dict(text=f"Lactato Estacionário — {modalidade}",
                                   font=dict(size=13)),
                        xaxis=dict(**_AX_MB, title="Potência (W)"),
                        yaxis=dict(**_AX_MB, title="[La] mmol/L", range=[0, 8]),
                    )
                    st.plotly_chart(_fig_la, use_container_width=True)

                # Sensibilidade VLamax (mostra como MLSS e FatMax variam)
                with st.expander("📈 Sensibilidade VLamax", expanded=False):
                    _vla_range  = np.linspace(0.10, min(_mb_vlamax * 2, 1.2), 40)
                    _mlss_sens  = []
                    _fatm_sens  = []
                    for _vla_s in _vla_range:
                        try:
                            _ads  = np.sqrt(np.maximum(0, (_mb_Ks1 * _mb_VO2ss) /
                                                         (_mb_vo2max - _mb_VO2ss)))
                            _vls  = 60 * _vla_s / (1 + (_mb_Ks2 /
                                                          np.maximum(_ads ** 3, 1e-12)))
                            _lnet = _vls - _mb_LaComb
                            _arg  = int(np.argmin(np.abs(_lnet)))
                            _ovr  = (_vls * (_mb_VolRel * _mb_peso) *
                                     ((1 / 4.3) * 22.4) / _mb_peso) + _mb_VO2ss
                            _wts  = np.maximum(0, (_ovr - _mb_VO2rest) *
                                                    _mb_peso / _mb_Watt_O2)
                            _argf = int(np.argmax(-_lnet[:_arg])) if _arg > 1 else 0
                            _mlss_sens.append(float(_wts[_arg]))
                            _fatm_sens.append(float(_wts[_argf]))
                        except Exception:
                            _mlss_sens.append(np.nan)
                            _fatm_sens.append(np.nan)

                    _fig_sens = go.Figure()
                    _fig_sens.add_trace(go.Scatter(
                        x=list(_vla_range), y=_mlss_sens,
                        mode='lines', name='MLSS (W)',
                        line=dict(color='#FFD166', width=2.5)))
                    _fig_sens.add_trace(go.Scatter(
                        x=list(_vla_range), y=_fatm_sens,
                        mode='lines', name='FatMax (W)',
                        line=dict(color='#00C896', width=2.5)))
                    _fig_sens.add_vline(x=_mb_vlamax, line_dash='dot',
                                        line_color='#A855F7', line_width=1.5,
                                        annotation_text=f"VLamax={_mb_vlamax:.3f}",
                                        annotation_font_color='#A855F7')
                    _fig_sens.add_hline(y=_calc_cp, line_dash='dot',
                                        line_color='#A855F7', line_width=1,
                                        annotation_text=f"CP={_calc_cp}W",
                                        annotation_font_color='#A855F7',
                                        annotation_position="right")
                    _fig_sens.update_layout(
                        **_BASE_MB,
                        title=dict(text="Sensibilidade ao VLamax",
                                   font=dict(size=13)),
                        xaxis=dict(**_AX_MB, title="VLamax (mmol/L/s)"),
                        yaxis=dict(**_AX_MB, title="Potência (W)"),
                    )
                    st.plotly_chart(_fig_sens, use_container_width=True)

                # Tabela de zonas
                with st.expander("📋 Zonas de Intensidade por Substrato", expanded=False):
                    _mb_zonas = []
                    for _pct_z, _nome_z in [
                        (0.50, 'Recuperação'),
                        (0.65, 'Z1 — Aeróbio leve'),
                        (0.75, 'Z2 — Aeróbio moderado'),
                        (0.85, 'Z2-Z3 — FatMax'),
                        (0.92, 'Z3 — Limiar'),
                        (1.00, 'MLSS'),
                    ]:
                        _w_z   = _mb_W_AT * _pct_z
                        _idx_z = min(int(np.argmin(np.abs(_mb_Watts - _w_z))),
                                     _mb_argAT - 1)
                        _fat_z = (float(_mb_Fat[_idx_z])
                                  if _idx_z < len(_mb_Fat) else 0.0)
                        _la_z  = (float(_mb_CLass[_idx_z])
                                  if _idx_z < len(_mb_CLass)
                                  and np.isfinite(_mb_CLass[_idx_z]) else None)
                        _mb_zonas.append({
                            'Zona':         _nome_z,
                            'Potência':     f"{_w_z:.0f} W",
                            '% MLSS':       f"{_pct_z * 100:.0f}%",
                            'Fat (g/h)':    f"{_fat_z:.0f}",
                            'CHO (g/h)':    f"{_mb_CHO_g:.0f}",
                            '[La] mmol/L':  f"{_la_z:.2f}" if _la_z else "—",
                        })
                    st.dataframe(pd.DataFrame(_mb_zonas), hide_index=True,
                                 use_container_width=True)
                    st.caption(
                        f"⚙️ Mader/Hauser | Ks1={_mb_Ks1} · Ks2={_mb_Ks2} · "
                        f"VolRel={_mb_VolRel} · Kel={_mb_Kel} | "
                        f"VLamax={_mb_vlamax:.3f} · VO2max={_mb_vo2max:.1f} ml/min/kg"
                    )

            except Exception as _mb_err:
                st.warning(f"Erro no modelo metabólico: {_mb_err}")

            # ── VO2max: múltiplas fórmulas comparativas ───────────────────────
            st.markdown("---")
            st.markdown("**🫁 VO2max — Comparação de Estimativas (maior → menor)**")
            st.caption("Todas sem necessidade de laboratório. Ordenadas da maior para a menor.")

            _vo2_estimates = []

            # 1. INSCYD/konaendu (com VLamax) — já calculado acima
            _vo2_estimates.append({
                'Método': 'INSCYD / konaendu (com VLamax)',
                'VO2max': round(_mb_vo2max, 1),
                'Inputs': 'CP + MMP3 + MMP5 + Pmax + VLamax',
                'Precisão': 'Bias -0.21 ml/min/kg vs espirometria (N=11)',
                'Requer VLamax': '✅ Sim',
            })

            # 2. Hawley & Noakes via CP
            _vo2_hawley_cp = round(_calc_cp / _mb_peso * 10.8 + 7, 1)
            _vo2_estimates.append({
                'Método': 'Hawley & Noakes via CP',
                'VO2max': _vo2_hawley_cp,
                'Inputs': 'CP + peso',
                'Precisão': 'Boa para endurance; subestima se VLamax alto',
                'Requer VLamax': '❌ Não',
            })

            # 3. Hawley & Noakes via MMP5 (Sitko 2022, r=0.84)
            _vo2_hawley_mmp5 = round(_mb_mmp5_in / _mb_peso * 10.8 + 7, 1)
            _vo2_estimates.append({
                'Método': 'Hawley via MMP5 (Sitko 2022)',
                'VO2max': _vo2_hawley_mmp5,
                'Inputs': 'MMP5 + peso',
                'Precisão': 'r=0.84 em ciclistas treinados',
                'Requer VLamax': '❌ Não',
            })

            # 4. Hawley via MMP3 (Dexheimer 2020, r=0.82)
            _vo2_hawley_mmp3 = round(_mb_mmp3_in / _mb_peso * 10.8 + 7, 1)
            _vo2_estimates.append({
                'Método': 'Hawley via MMP3 (Dexheimer 2020)',
                'VO2max': _vo2_hawley_mmp3,
                'Inputs': 'MMP3 + peso',
                'Precisão': 'r=0.82 em atletas funcionais',
                'Requer VLamax': '❌ Não',
            })

            # 5. Pvo2max da sheet (se disponível)
            _pvo2_val = None
            if ac_full is not None and _col_mod is not None and 'Pvo2max' in ac_full.columns:
                _ac_pv = ac_full[ac_full[_col_mod] == modalidade]
                _col_d_pv = next((c for c in ['date','Data'] if c in _ac_pv.columns), None)
                if _col_d_pv:
                    _pv_s = (_ac_pv[['Pvo2max', _col_d_pv]]
                             .dropna(subset=['Pvo2max'])
                             .sort_values(_col_d_pv, ascending=False))
                    if not _pv_s.empty:
                        try:
                            _pvo2_val = float(str(_pv_s['Pvo2max'].iloc[0]).replace(',','.'))
                        except: pass
            if _pvo2_val:
                _vo2_pvo2 = round(_pvo2_val / _mb_peso * 10.8 + 7, 1)
                _vo2_estimates.append({
                    'Método': 'Via Pvo2max (sheet)',
                    'VO2max': _vo2_pvo2,
                    'Inputs': f'Pvo2max={_pvo2_val:.0f}W + peso',
                    'Precisão': 'Estimativa directa de potência @ VO2max',
                    'Requer VLamax': '❌ Não',
                })

            # Ordenar maior → menor
            _vo2_df = (pd.DataFrame(_vo2_estimates)
                       .sort_values('VO2max', ascending=False)
                       .reset_index(drop=True))
            # Highlight da maior
            _vo2_df.insert(0, 'Rank', [f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}º'}" for i in range(len(_vo2_df))])
            st.dataframe(_vo2_df, hide_index=True, use_container_width=True)
            st.caption(
                f"⚙️ Peso usado: {_mb_peso}kg | CP={_calc_cp}W | MMP3={_mb_mmp3_in}W | MMP5={_mb_mmp5_in}W"
            )

            # ── Limiares HR por modalidade (HRVT1/T1PLUS/TMSS/T2 + AeTHR + PBP) ───
            st.markdown("---")
            st.markdown("**❤️ Limiares Fisiológicos por HR — por Modalidade**")
            st.caption(
                "Baseado nas colunas do Intervals.icu calculadas por DFA-alpha1 piecewise fitting. "
                "Limpeza IQR×1.5. Valores em bpm (HR) ou W (potência)."
            )

            _THRESH_COLS = {
                'HRVT1':     {'label': 'LT1 / AeT',         'fisiologia': 'Z1→Z2 (72-80% FCmax)'},
                'HRVT1PLUS': {'label': 'LT1+ / Transição',  'fisiologia': 'Z2 superior (81-84% FCmax)'},
                'HRVTMSS':   {'label': 'MLSS / Limiar est.','fisiologia': 'Z2→Z3 (84-89% FCmax)'},
                'HRVT2':     {'label': 'LT2 / AnT',         'fisiologia': 'Z3→Z4 (89-95% FCmax)'},
                'AeTHR':     {'label': 'AeT HR',            'fisiologia': 'Limiar aeróbio (HR)'},
                'PBP':       {'label': 'PBP (W)',            'fisiologia': 'Power @ breakpoint LT1-LT2'},
                'Pvo2max':   {'label': 'Pvo2max (W)',        'fisiologia': 'Power @ VO2max estimado'},
            }

            def _clean_iqr(series, factor=1.5):
                """Remove outliers IQR×factor. Retorna série limpa."""
                s = pd.to_numeric(series, errors='coerce').dropna()
                if len(s) < 4: return s
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                return s[(s >= q1 - factor*iqr) & (s <= q3 + factor*iqr)]

            if ac_full is not None and _col_mod is not None:
                _ac_thr = ac_full[ac_full[_col_mod] == modalidade].copy()
                _cols_avail = [c for c in _THRESH_COLS if c in _ac_thr.columns]

                if not _cols_avail:
                    st.info("Colunas de limiares (HRVT1, HRVT2, etc.) não encontradas "
                            "nas actividades. Verifica se os custom fields estão configurados "
                            "no Intervals.icu.")
                else:
                    _thr_rows = []
                    _hr_zones = {}  # para o gráfico

                    for _tc in ['HRVT1','HRVT1PLUS','HRVTMSS','HRVT2','AeTHR','PBP','Pvo2max']:
                        if _tc not in _cols_avail: continue
                        _cfg = _THRESH_COLS[_tc]
                        _raw = _clean_iqr(_ac_thr[_tc])
                        if len(_raw) < 2: continue
                        _med = float(_raw.median())
                        _q25 = float(_raw.quantile(0.25))
                        _q75 = float(_raw.quantile(0.75))
                        _mn  = float(_raw.min())
                        _mx  = float(_raw.max())
                        _n   = len(_raw)
                        _unit = 'bpm' if _tc not in ('PBP','Pvo2max') else 'W'
                        _thr_rows.append({
                            'Métrica':     _cfg['label'],
                            'Zona':        _cfg['fisiologia'],
                            'Mediana':     f"{_med:.0f} {_unit}",
                            'IQR [Q25-Q75]': f"[{_q25:.0f} – {_q75:.0f}]",
                            'Range':       f"{_mn:.0f} – {_mx:.0f}",
                            'N':           _n,
                        })
                        _hr_zones[_tc] = {
                            'med': _med, 'q25': _q25, 'q75': _q75,
                            'label': _cfg['label'], 'unit': _unit
                        }

                    if _thr_rows:
                        st.dataframe(pd.DataFrame(_thr_rows), hide_index=True,
                                     use_container_width=True)

                        # Gráfico de zonas HR com ranges
                        _hr_keys = [k for k in ['HRVT1','HRVT1PLUS','HRVTMSS','HRVT2','AeTHR']
                                    if k in _hr_zones and _hr_zones[k]['unit'] == 'bpm']

                        if len(_hr_keys) >= 2:
                            # ════════════════════════════════════════════════
                            # GRÁFICO 1 — Eixo X: Potência (W)
                            # Y1: HR bpm (limiares horizontais)
                            # Y2: Potência — CP, MLSS, PBP, Pvo2max (linhas verticais)
                            # FatMax: linha vertical simples
                            # ════════════════════════════════════════════════
                            _fig_w = go.Figure()

                            # Faixas de zona (background HR)
                            _zone_colors = ['#60A5FA','#34D399','#FBBF24','#F87171']
                            _zone_defs_w = [
                                ('HRVT1', 'HRVTMSS', '#34D399', 'Z2 Aeróbio'),
                                ('HRVTMSS', 'HRVT2',  '#FBBF24', 'Z3 Tempo'),
                            ]
                            # Limiares HR como linhas horizontais
                            _hr_col_map = {
                                'HRVT1':     ('#60A5FA', 'LT1 / AeT'),
                                'HRVT1PLUS': ('#34D399', 'LT1+ Transição'),
                                'HRVTMSS':   ('#FBBF24', 'MLSS'),
                                'HRVT2':     ('#F87171', 'LT2 / AnT'),
                                'AeTHR':     ('#A78BFA', 'AeT HR'),
                            }
                            # Range de X: 0 a Pvo2max+10% ou CP*1.5
                            _x_max_w = max(
                                _hr_zones.get('Pvo2max', {}).get('med', 0) * 1.1,
                                (_calc_cp or 200) * 1.5,
                                300
                            )
                            _x_range_w = [0, _x_max_w]

                            # Bandas de cor entre zonas
                            _zone_band_pairs = [
                                (None,       'HRVT1',    '#60A5FA', 0.05, 'Z1 Rec'),
                                ('HRVT1',    'HRVTMSS',  '#34D399', 0.05, 'Z2 Aeróbio'),
                                ('HRVTMSS',  'HRVT2',    '#FBBF24', 0.05, 'Z3 Tempo'),
                                ('HRVT2',    None,        '#F87171', 0.05, 'Z4 Limiar'),
                            ]
                            _hr_min = min((_hr_zones[k]['q25'] for k in _hr_keys), default=100) - 5
                            _hr_max = max((_hr_zones[k]['q75'] for k in _hr_keys), default=200) + 5

                            for _zlo_k, _zhi_k, _zcl, _zop, _znm in _zone_band_pairs:
                                _y0z = _hr_zones[_zlo_k]['med'] if _zlo_k and _zlo_k in _hr_zones else _hr_min
                                _y1z = _hr_zones[_zhi_k]['med'] if _zhi_k and _zhi_k in _hr_zones else _hr_max
                                _fig_w.add_hrect(
                                    y0=_y0z, y1=_y1z,
                                    fillcolor=_zcl, opacity=_zop, line_width=0,
                                )
                                # Label da zona
                                _fig_w.add_annotation(
                                    x=_x_max_w * 0.02, y=(_y0z + _y1z) / 2,
                                    text=_znm, showarrow=False,
                                    font=dict(size=9, color=_zcl),
                                    xanchor='left',
                                )

                            # Limiares HR — linhas horizontais + IQR shading
                            for _hk in _hr_keys:
                                _hd = _hr_zones[_hk]
                                _hc, _hl = _hr_col_map.get(_hk, ('#AAAAAA', _hd['label']))
                                # IQR shading
                                _fig_w.add_hrect(
                                    y0=_hd['q25'], y1=_hd['q75'],
                                    fillcolor=_hc, opacity=0.12, line_width=0,
                                )
                                # Linha mediana
                                _fig_w.add_hline(
                                    y=_hd['med'],
                                    line_color=_hc, line_width=1.5, line_dash='solid',
                                    annotation_text=f"{_hl}: {_hd['med']:.0f} bpm",
                                    annotation_font_color=_hc,
                                    annotation_font_size=9,
                                    annotation_position="top right",
                                )

                            # Linhas verticais: CP, MLSS(Mader), FatMax, PBP, Pvo2max
                            _vlines_w = []
                            if _calc_cp:
                                _vlines_w.append((_calc_cp, '#A855F7', 'solid', f"CP {_calc_cp}W"))
                            if _mb_W_AT and _mb_W_AT > 10:
                                _vlines_w.append((_mb_W_AT, '#FFD166', 'dot', f"MLSS {_mb_W_AT:.0f}W"))
                            if _mb_W_FM and _mb_W_FM > 10:
                                _vlines_w.append((_mb_W_FM, '#00C896', 'dot', f"FatMax {_mb_W_FM:.0f}W"))
                            if 'PBP' in _hr_zones:
                                _vlines_w.append((_hr_zones['PBP']['med'], '#FF6B35', 'dash',
                                                  f"PBP {_hr_zones['PBP']['med']:.0f}W"))
                            if 'Pvo2max' in _hr_zones:
                                _vlines_w.append((_hr_zones['Pvo2max']['med'], '#60A5FA', 'dash',
                                                  f"Pvo2max {_hr_zones['Pvo2max']['med']:.0f}W"))

                            for _vx, _vc, _vd, _vl in _vlines_w:
                                _fig_w.add_vline(
                                    x=_vx, line_color=_vc,
                                    line_width=1.8, line_dash=_vd,
                                    annotation_text=_vl,
                                    annotation_font_color=_vc,
                                    annotation_font_size=9,
                                    annotation_position="top left",
                                )

                            _fig_w.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=11),
                                margin=dict(l=55, r=30, t=50, b=40),
                                showlegend=False,
                                title=dict(
                                    text=f"Limiares HR vs Potência — {modalidade}",
                                    font=dict(size=13)),
                                xaxis=dict(
                                    title="Potência (W)",
                                    range=_x_range_w,
                                    showgrid=True,
                                    gridcolor="rgba(128,128,128,0.12)",
                                    zeroline=False,
                                ),
                                yaxis=dict(
                                    title="HR (bpm)",
                                    range=[_hr_min, _hr_max],
                                    showgrid=True,
                                    gridcolor="rgba(128,128,128,0.12)",
                                    zeroline=False,
                                ),
                            )
                            st.plotly_chart(_fig_w, use_container_width=True)

                            # ════════════════════════════════════════════════
                            # GRÁFICO 2 — Eixo X: Duração (min)
                            # Y1: HR bpm (limiares)
                            # Linhas verticais: duração @ CP, MLSS, FatMax
                            # Usa equação W' = (P-CP)×t → t = W'/(P-CP)
                            # ════════════════════════════════════════════════
                            _fig_t = go.Figure()

                            # Durações estimadas via equação CP
                            _t_refs = []
                            if _calc_cp and _calc_wp and isinstance(_calc_wp, float) and _calc_wp > 0:
                                # MLSS
                                if _mb_W_AT and _mb_W_AT > _calc_cp:
                                    pass  # MLSS < CP → duração infinita no modelo
                                elif _mb_W_AT and _mb_W_AT > 0:
                                    _t_refs.append((60.0, '#FFD166', 'dot',
                                                    f"MLSS ~60min ({_mb_W_AT:.0f}W)"))
                                # CP
                                _t_refs.append((None, '#A855F7', 'solid', f"CP = ∞ ({_calc_cp}W)"))
                                # FatMax (estimativa: ~60-90min para maioria atletas)
                                if _mb_W_FM and _mb_W_FM > _calc_cp:
                                    pass
                                elif _mb_W_FM and _mb_W_FM > 0:
                                    _t_refs.append((None, '#00C896', 'dot',
                                                    f"FatMax ({_mb_W_FM:.0f}W ≤ CP)"))

                                # Potências acima do CP: calcular duração via W'/(P-CP)
                                for _pref, _plbl, _pcol in [
                                    ('PBP',    'PBP',    '#FF6B35'),
                                    ('Pvo2max','Pvo2max','#60A5FA'),
                                ]:
                                    if _pref in _hr_zones:
                                        _pw = _hr_zones[_pref]['med']
                                        if _pw > _calc_cp:
                                            _tt = _calc_wp / (_pw - _calc_cp) / 60
                                            _t_refs.append((
                                                _tt, _pcol, 'dash',
                                                f"{_plbl} {_pw:.0f}W → {_tt:.1f}min"
                                            ))

                            # Range X: 0 a 120 min
                            _x_max_t = 120.0

                            # Faixas de zona (mesmo background)
                            for _zlo_k, _zhi_k, _zcl, _zop, _znm in _zone_band_pairs:
                                _y0z = _hr_zones[_zlo_k]['med'] if _zlo_k and _zlo_k in _hr_zones else _hr_min
                                _y1z = _hr_zones[_zhi_k]['med'] if _zhi_k and _zhi_k in _hr_zones else _hr_max
                                _fig_t.add_hrect(
                                    y0=_y0z, y1=_y1z,
                                    fillcolor=_zcl, opacity=_zop, line_width=0,
                                )
                                _fig_t.add_annotation(
                                    x=1.0, y=(_y0z + _y1z) / 2,
                                    text=_znm, showarrow=False,
                                    font=dict(size=9, color=_zcl), xanchor='left',
                                )

                            # Limiares HR horizontais
                            for _hk in _hr_keys:
                                _hd = _hr_zones[_hk]
                                _hc, _hl = _hr_col_map.get(_hk, ('#AAAAAA', _hd['label']))
                                _fig_t.add_hrect(
                                    y0=_hd['q25'], y1=_hd['q75'],
                                    fillcolor=_hc, opacity=0.12, line_width=0,
                                )
                                _fig_t.add_hline(
                                    y=_hd['med'],
                                    line_color=_hc, line_width=1.5,
                                    annotation_text=f"{_hl}: {_hd['med']:.0f} bpm",
                                    annotation_font_color=_hc,
                                    annotation_font_size=9,
                                    annotation_position="top right",
                                )

                            # Linhas verticais: duração @ cada referência
                            for _tx, _tc, _td, _tl in _t_refs:
                                if _tx is None: continue
                                if _tx > _x_max_t: continue
                                _fig_t.add_vline(
                                    x=_tx, line_color=_tc,
                                    line_width=1.8, line_dash=_td,
                                    annotation_text=_tl,
                                    annotation_font_color=_tc,
                                    annotation_font_size=9,
                                    annotation_position="top left",
                                )

                            # Curva P-D no fundo (potência vs duração)
                            if _calc_cp and _calc_wp and isinstance(_calc_wp, float) and _calc_wp > 0:
                                _t_curve = np.linspace(1, _x_max_t, 200)
                                _p_curve = _calc_cp + _calc_wp / (_t_curve * 60)
                                _fig_t.add_trace(go.Scatter(
                                    x=_t_curve, y=_p_curve,
                                    mode='lines', name=f'P-D (CP={_calc_cp}W)',
                                    line=dict(color='rgba(168,85,247,0.4)', width=1.5,
                                              dash='dot'),
                                    yaxis='y',
                                ))

                            _fig_t.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=11),
                                margin=dict(l=55, r=30, t=50, b=40),
                                showlegend=False,
                                title=dict(
                                    text=f"Limiares HR vs Duração — {modalidade}",
                                    font=dict(size=13)),
                                xaxis=dict(
                                    title="Duração (min)",
                                    range=[0, _x_max_t],
                                    showgrid=True,
                                    gridcolor="rgba(128,128,128,0.12)",
                                    zeroline=False,
                                ),
                                yaxis=dict(
                                    title="HR (bpm)",
                                    range=[_hr_min, _hr_max],
                                    showgrid=True,
                                    gridcolor="rgba(128,128,128,0.12)",
                                    zeroline=False,
                                ),
                            )
                            st.plotly_chart(_fig_t, use_container_width=True)
                            st.caption(
                                "**Gráfico 1** — Linhas verticais = potência de cada limiar. "
                                "Linhas horizontais = HR mediana ± IQR histórico (bpm). "
                                "**Gráfico 2** — Duração estimada via W′/(P−CP). "
                                "Curva P-D a roxo claro."
                            )

                        # Tabela de zonas treino consolidada
                        if len(_hr_keys) >= 2:
                            st.markdown("**📋 Zonas de Treino Consolidadas**")
                            _z_treino = []
                            _z_defs = [
                                ('Z1 — Recuperação',  None,         'HRVT1',     '#60A5FA'),
                                ('Z2 — Aeróbio',      'HRVT1',      'HRVTMSS',   '#34D399'),
                                ('Z3 — Tempo / MLSS', 'HRVTMSS',    'HRVT2',     '#FBBF24'),
                                ('Z4 — Limiar',       'HRVT2',      None,        '#F87171'),
                            ]
                            for _znome, _zlow_k, _zhigh_k, _zcol in _z_defs:
                                _zlow  = f"< {_hr_zones[_zlow_k]['med']:.0f} bpm"  if _zlow_k  and _zlow_k  in _hr_zones else '—'
                                _zhigh = f"< {_hr_zones[_zhigh_k]['med']:.0f} bpm" if _zhigh_k and _zhigh_k in _hr_zones else '>'
                                _zrange = (f"{_hr_zones[_zlow_k]['med']:.0f} – {_hr_zones[_zhigh_k]['med']:.0f} bpm"
                                           if _zlow_k and _zhigh_k and _zlow_k in _hr_zones and _zhigh_k in _hr_zones
                                           else (_zlow if not _zlow_k else _zhigh))
                                _z_treino.append({'Zona': _znome, 'HR Range': _zrange})

                            if 'PBP' in _hr_zones:
                                _z_treino.append({
                                    'Zona': 'PBP (breakpoint)',
                                    'HR Range': f"Power @ LT1-LT2: {_hr_zones['PBP']['med']:.0f} W "
                                                f"[{_hr_zones['PBP']['q25']:.0f}–{_hr_zones['PBP']['q75']:.0f}]"
                                })
                            if 'Pvo2max' in _hr_zones:
                                _z_treino.append({
                                    'Zona': 'VO2max power',
                                    'HR Range': f"Power @ VO2max: {_hr_zones['Pvo2max']['med']:.0f} W "
                                                f"[{_hr_zones['Pvo2max']['q25']:.0f}–{_hr_zones['Pvo2max']['q75']:.0f}]"
                                })
                            st.dataframe(pd.DataFrame(_z_treino), hide_index=True,
                                         use_container_width=True)

            else:
                st.info("Dados de actividades não disponíveis.")
    with _tab_ompd:
        st.caption(
            "OmPD com os melhores 3 pontos MMP encontrados pelo grid search. "
            "MMP60 excluído do fitting (só validação). "
            "Row/Ski: máximo 12min."
        )
        # Mostrar resultado do grid search para OmPD (consistente com o Ranking)
        if 'OmPD' in _results:
            _gr_od = _results['OmPD']
            _res_od = _gr_od['result']
            _cp_od, _wp_od = _res_od[0], _res_od[1]
            _pm_od, _A_od  = _res_od[2], _res_od[3]
            _pp_od, _r2_od, _weff_od = _res_od[4], _res_od[5], _res_od[6]
            _pts_od = _gr_od['combo']

            _od1, _od2, _od3, _od4, _od5 = st.columns(5)
            _od1.metric("CP", f"{_cp_od:.0f}W")
            _od2.metric("W′", f"{_wp_od:.0f}J")
            _od3.metric("Pmax", f"{_pm_od:.0f}W" if _pm_od else "—")
            _, _see_od = calc_see([p for p,_ in _pts_od], _pp_od, k=2)
            _od4.metric("SEE%", f"{_see_od:.2f}%" if _see_od else "—")
            _od5.metric("W′eff @120s", f"{_weff_od:.1f}%" if _weff_od else "—")

            st.caption(f"Pontos usados: " +
                       " + ".join([f"{int(t//60)}min={p:.0f}W" for p, t in _pts_od]) +
                       f" | CP var%={_gr_od.get('cp_var_pct','—')}% | Score={_gr_od.get('score','—')}")

            _t_obs_od = [t for _, t in _pts_od]
            _trng_od  = np.logspace(np.log10(30), np.log10(max(_t_obs_od)*3), 400)
            def _ompd_p_od(t_arr, cp, wp, pmax_v, A):
                tau  = wp / max(pmax_v - cp, 1.0)
                base = wp / t_arr * (1 - np.exp(-t_arr / tau)) + cp
                if A and A > 0:
                    decay = np.where(t_arr > TCP_MAX, A * np.log(t_arr / TCP_MAX), 0.0)
                    return base - decay
                return base
            _ydd_od = _ompd_p_od(_trng_od, _cp_od, _wp_od,
                                  _pm_od or _pmax_global or max(p for p,_ in _pts_od)*1.15,
                                  _A_od or 0.0)
            _fig_od = go.Figure()
            _fig_od.add_trace(go.Scatter(
                x=[t for _,t in _all_mmp_pts], y=[p for p,_ in _all_mmp_pts],
                mode='markers+text', name='MMP (todos)',
                marker=dict(size=9, color='#e74c3c'),
                text=[f"{int(t//60)}min" for _,t in _all_mmp_pts],
                textposition='top center', textfont=dict(size=9),
            ))
            _fig_od.add_trace(go.Scatter(
                x=[t for _,t in _pts_od], y=[p for p,_ in _pts_od],
                mode='markers', name='Pontos usados (3)',
                marker=dict(size=13, color='#8e44ad', symbol='star'),
            ))
            _fig_od.add_trace(go.Scatter(
                x=list(_trng_od), y=list(_ydd_od),
                mode='lines', name='OmPD',
                line=dict(color='#8e44ad', width=2.5),
            ))
            if _mmp60_val:
                _fig_od.add_trace(go.Scatter(
                    x=[3600], y=[_mmp60_val],
                    mode='markers', name=f'MMP60 validação ({_mmp60_val:.0f}W)',
                    marker=dict(size=10, color='#e67e22', symbol='diamond'),
                ))
            _fig_od.add_hline(y=_cp_od, line_dash='dot', line_color='#888',
                               annotation_text=f"CP={_cp_od:.0f}W",
                               annotation_position="right")
            _fig_od.update_layout(
                **BASE,
                xaxis=dict(**AX, title="Duração (s)", type='log'),
                yaxis=dict(**AX, title="Potência (W)"),
                title=dict(text=f"OmPD — {modalidade} | 3 melhores pontos",
                           font=dict(size=13)),
            )
            st.plotly_chart(_fig_od, use_container_width=True)
        else:
            st.warning("OmPD não convergiu. Ver tab Ranking para diagnóstico.")
    # ════════════════════════════════════════════════════════════════════════
    # TABS DOS MODELOS INDIVIDUAIS — helper para mostrar resultado do grid
    # ════════════════════════════════════════════════════════════════════════
    def _show_model_tab(tab_obj, model_name, color):
        with tab_obj:
            st.subheader(f"{model_name} — {modalidade}")
            if model_name not in _results:
                if not _all_mmp_pts:
                    st.warning("Sem dados MMP disponíveis.")
                else:
                    st.warning(f"Modelo não convergiu com os dados disponíveis "
                               f"({len(_all_mmp_pts)} pontos MMP).")
                return
            _gr = _results[model_name]
            _res = _gr['result']
            _cp_m = _gr['cp']
            _wp_m = _res[1] if len(_res) > 1 else None
            _pm_m = _res[2] if len(_res) > 2 else None

            # Métricas principais
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric("SEE%", f"{_gr['see_pct']:.2f}%")
            if _cp_m: _mc2.metric("CP (W)", f"{_cp_m:.0f}")
            if isinstance(_wp_m, float) and _wp_m > 100:
                _mc3.metric("W′ (J)", f"{_wp_m:.0f}")
            if isinstance(_pm_m, float) and _pm_m > 100 and model_name != 'Power Law':
                _mc4.metric("Pmax (W)", f"{_pm_m:.0f}")

            st.caption(f"Melhores pontos: " +
                       " + ".join([f"{int(t//60)}min={p:.0f}W"
                                   for p, t in _gr['combo']]))

            # Grid search — mostrar todas as combinações de N=3 pontos
            _mcfg_i = _MODELS[model_name]
            _px_i   = _pmax_global if _mcfg_i['needs_pmax'] else None
            # Aplicar mesmas regras de filtragem da tab principal
            _pts_i  = ([p for p in _all_mmp_pts if p[1] >= 60]
                       if _mcfg_i.get('exclude_short') else _all_mmp_pts)
            _n_pts_i = _mcfg_i['n_pts']

            from itertools import combinations as _combos
            _all_combos_rows = []
            if len(_pts_i) >= _n_pts_i:
                for _cb in _combos(range(len(_pts_i)), _n_pts_i):
                    _pts_cb = [_pts_i[i] for i in _cb]
                    try:
                        _res_cb = (_mcfg_i['fn'](_pts_cb, pmax_ext=_px_i)
                                   if _mcfg_i['needs_pmax']
                                   else _mcfg_i['fn'](_pts_cb))
                        if _res_cb[0] is None: continue
                        _p_obs_cb = [p for p, _ in _pts_cb]
                        # OmPD: pp no índice 4
                        if model_name == 'OmPD':
                            _pp_cb = _res_cb[4] if len(_res_cb) > 4 else None
                        elif model_name in ('M1: P vs 1/t', 'M2: Work-Time', 'M3: Hiperbólico-t'):
                            _pp_cb = _res_cb[3] if len(_res_cb) > 3 else None
                        else:
                            _pp_cb = _res_cb[-1]
                        if not isinstance(_pp_cb, (list, np.ndarray)) or len(_pp_cb) == 0: continue
                        _, _see_cb = calc_see(_p_obs_cb, _pp_cb, k=_mcfg_i['k'])
                        if _see_cb is None: continue
                        _all_combos_rows.append({
                            'Pontos usados': " + ".join([f"{int(t//60)}min" for _, t in _pts_cb]),
                            'N': len(_pts_cb),
                            'SEE%': round(_see_cb, 3),
                            'CP (W)': round(float(_res_cb[0]), 1),
                        })
                    except Exception:
                        pass
            if _all_combos_rows:
                _df_combos = (pd.DataFrame(_all_combos_rows)
                              .sort_values('SEE%')
                              .reset_index(drop=True))
                with st.expander("📋 Todas as combinações testadas", expanded=False):
                    st.dataframe(_df_combos, hide_index=True, use_container_width=True)

            # Gráfico com melhor combinação
            _t_plot = np.logspace(np.log10(30), np.log10(12000), 400)
            _y_plot = None
            try:
                if model_name in ('M1: P vs 1/t', '2P Hiperbólico'):
                    _y_plot = _wp_m / _t_plot + _cp_m
                elif model_name in ('M2: Work-Time', 'M3: Hiperbólico-t'):
                    # M2 e M3 têm a mesma curva P = W′/t + CP
                    _y_plot = _wp_m / _t_plot + _cp_m
                elif model_name == '3P Hiperbólico':
                    _y_plot = (_pm_m * _wp_m) / (_wp_m + (_pm_m - _cp_m) * _t_plot)
                elif model_name == 'Ward-Smith':
                    _pm_ws = _pm_m or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.2
                    _y_plot = _cp_m + (_pm_ws - _cp_m) * np.exp(-_t_plot * (_pm_ws - _cp_m) / max(_wp_m, 1))
                elif model_name == 'Power Law':
                    _a_pl2, _b_pl2 = _res[1], _res[2]
                    _y_plot = _a_pl2 * _t_plot**(-_b_pl2)
                elif model_name in ('Om3CP', 'OmExp'):
                    _A3 = _res[3] if len(_res) > 3 and isinstance(_res[3], float) else 0.0
                    _pm3 = _pm_m or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.15
                    _tau3 = _wp_m / max(_pm3 - _cp_m, 1.0)
                    _y_plot = _wp_m / _t_plot * (1 - np.exp(-_t_plot / _tau3)) + _cp_m
                    if _A3 > 0:
                        _y_plot = _y_plot - np.where(_t_plot > TCP_MAX,
                                                      _A3 * np.log(_t_plot / TCP_MAX), 0.0)
            except Exception:
                pass

            if _y_plot is not None:
                _fig_m = go.Figure()
                _fig_m.add_trace(go.Scatter(
                    x=[t for _, t in _all_mmp_pts], y=[p for p, _ in _all_mmp_pts],
                    mode='markers+text', name='MMP',
                    marker=dict(size=9, color='#e74c3c'),
                    text=[f"{int(t//60)}min" for _, t in _all_mmp_pts],
                    textposition='top center', textfont=dict(size=9),
                ))
                _fig_m.add_trace(go.Scatter(
                    x=list(_t_plot), y=list(_y_plot),
                    mode='lines', name=model_name,
                    line=dict(color=color, width=2.5),
                ))
                if _cp_m:
                    _fig_m.add_hline(y=_cp_m, line_dash='dot',
                                     line_color='#888888',
                                     annotation_text=f"CP={_cp_m:.0f}W",
                                     annotation_position="right")
                _fig_m.update_layout(
                    **BASE,
                    xaxis=dict(**AX, title="Duração (s)", type='log'),
                    yaxis=dict(**AX, title="Potência (W)"),
                    title=dict(text=f"{model_name} — {modalidade}",
                               font=dict(size=13, color='#111111')),
                )
                st.plotly_chart(_fig_m, use_container_width=True)

    # Mostrar cada modelo na sua tab
    _show_model_tab(_tab_m1,    'M1: P vs 1/t',      '#e74c3c')
    _show_model_tab(_tab_m2,    'M2: Work-Time',      '#2980b9')
    _show_model_tab(_tab_m3,    'M3: Hiperbólico-t',  '#27ae60')
    _show_model_tab(_tab_2p,    '2P Hiperbólico',     '#1abc9c')
    _show_model_tab(_tab_3p,    '3P Hiperbólico',     '#f39c12')
    _show_model_tab(_tab_ws,    'Ward-Smith',         '#e67e22')
    _show_model_tab(_tab_om3,   'Om3CP',              '#16a085')
    _show_model_tab(_tab_omexp, 'OmExp',              '#d35400')
    _show_model_tab(_tab_pl,    'Power Law',          '#c0392b')

    with _tab_manual:
        st.subheader("📥 Testes Máximos")
        st.caption("TTE — esforço máximo até à falha a potência constante.")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Teste 1 ✅**")
            p1=st.number_input("Power T1 (W)",50,2000,400,key="cp_p1")
            t1=st.number_input("Tempo T1 (s)",10,7200,180,key="cp_t1")
        with c2:
            st.markdown("**Teste 2 ✅**")
            p2=st.number_input("Power T2 (W)",50,2000,320,key="cp_p2")
            t2=st.number_input("Tempo T2 (s)",10,7200,600,key="cp_t2")
        with c3:
            st.markdown("**Teste 3 (opcional)**")
            usar3=st.checkbox("Usar teste 3",True,key="cp_usar3")
            p3=st.number_input("Power T3 (W)",50,2000,270,key="cp_p3",disabled=not usar3)
            t3=st.number_input("Tempo T3 (s)",10,7200,1200,key="cp_t3",disabled=not usar3)

        # ════════════════════════════════════════════════════════════════════════
        if not st.button("⚡ Calcular",type="primary"):
            st.info("Configura os testes e clica em **Calcular**.")
            return

        tests = [(float(p1),float(t1)),(float(p2),float(t2))]
        if usar3: tests.append((float(p3),float(t3)))
        tests = sorted(tests,key=lambda x:x[1])
        n = len(tests)

        # ── OmPD: Pmax da sheet (coluna p_max — valor da modalidade) ────────────
        # Usar o valor mais recente disponível para a modalidade seleccionada
        _pmax_sheet = None
        if ac_full is not None and len(ac_full) > 0:
            _col_mod  = next((c for c in ['type','modality'] if c in ac_full.columns), None)
            _col_date = next((c for c in ['date','Data'] if c in ac_full.columns), None)
            if _col_mod and _col_date and 'p_max' in ac_full.columns:
                _pmax_sub = (ac_full[ac_full[_col_mod] == modalidade][['p_max', _col_date]]
                             .dropna(subset=['p_max'])
                             .sort_values(_col_date, ascending=False))
                if not _pmax_sub.empty:
                    _pmax_sheet = float(_pmax_sub['p_max'].iloc[0])

        # Input manual de Pmax se não disponível na sheet
        with st.expander("⚡ M5: OmPD — Pmax", expanded=False):
            st.caption(
                "O OmPD requer Pmax da modalidade (potência máxima de sprint, não de sessão). "
                "Carregado automaticamente da coluna `p_max` da sheet. "
                "Pode ajustar manualmente se necessário."
            )
            _pmax_default = int(_pmax_sheet) if _pmax_sheet else int(max(p for p,_ in tests)*1.20)
            pmax_input = st.number_input(
                f"Pmax {modalidade} (W)",
                min_value=int(max(p for p,_ in tests)*1.01),
                max_value=3000,
                value=_pmax_default,
                key="cp_pmax",
                help=(
                    f"Valor da sheet (p_max): {_pmax_sheet:.0f}W" if _pmax_sheet
                    else "Coluna p_max não encontrada — inserir manualmente"
                )
            )
            if _pmax_sheet:
                st.caption(f"✅ Pmax carregado da sheet: **{_pmax_sheet:.0f}W** ({modalidade}, mais recente)")
            else:
                st.warning("⚠️ p_max não encontrado na sheet. Usando estimativa ou valor manual.")

        errs=[]
        ts=[t for _,t in tests]
        if len(set(ts))<len(ts): errs.append("Dois testes com o mesmo tempo")
        for i,(p,t) in enumerate(tests,1):
            if p<=0: errs.append(f"T{i}: Power>0")
            if t<=0: errs.append(f"T{i}: Tempo>0")
        if errs: [st.error(e) for e in errs]; return

        if n<5:
            st.warning(f"⚠️ {n} testes — baixa confiabilidade. Ideal ≥5 esforços máximos.")

        p_obs = [p for p,_ in tests]

        # ════════════════════════════════════════════════════════════════════════
        # CALCULAR — 3 cenários × 4 modelos
        # ════════════════════════════════════════════════════════════════════════
        # all_fits[mk][wmode] = (cp,wp,pmax,pp,r2,k)
        all_fits = {mk:{} for mk in ["M1","M2","M3","M4"]}

        for mk,fn in FIT_FNS.items():
            skip_m4 = (mk=="M4" and n<=3)
            for wmode in W_MODES:
                if skip_m4:
                    all_fits[mk][wmode] = (None,None,None,None,None,3)
                    continue
                w = make_w([t for _,t in tests], wmode)
                all_fits[mk][wmode] = fn(tests, w)

        # ── M5: OmPD — um único fit (sem weighting alternativo) ────────────────
        ompd_result = fit_ompd(tests, pmax_ext=pmax_input)
        # ompd_result = (cp, wp, pmax, A, pp, r2, weff_pct)
        _ompd_cp, _ompd_wp, _ompd_pmax, _ompd_A, _ompd_pp, _ompd_r2, _ompd_weff = (
            ompd_result if ompd_result[0] is not None
            else (None, None, None, None, None, None, None)
        )

        # ════════════════════════════════════════════════════════════════════════
        # ESTABILIDADE — variação de CP e W′ entre os 3 cenários
        # ════════════════════════════════════════════════════════════════════════
        stability = {}   # mk → {cp_var, w_var, cp_mean, wp_mean, cp_vals, wp_vals}
        for mk in ["M1","M2","M3","M4"]:
            cp_vals = [all_fits[mk][wm][0] for wm in W_MODES
                       if all_fits[mk][wm][0] is not None]
            wp_vals = [all_fits[mk][wm][1] for wm in W_MODES
                       if all_fits[mk][wm][1] is not None]
            if not cp_vals:
                stability[mk] = None; continue
            cp_mean = float(np.mean(cp_vals))
            wp_mean = float(np.mean(wp_vals)) if wp_vals else 0
            cp_var  = (max(cp_vals)-min(cp_vals))/cp_mean*100 if cp_mean>0 else 0
            wp_var  = (max(wp_vals)-min(wp_vals))/wp_mean*100 if wp_mean>0 and wp_vals else 0
            stability[mk] = {
                "cp_mean":round(cp_mean,1),"wp_mean":round(wp_mean,0),
                "cp_var":round(cp_var,1),"wp_var":round(wp_var,1),
                "cp_vals":cp_vals,"wp_vals":wp_vals,
            }

        # ════════════════════════════════════════════════════════════════════════
        # SCORE FINAL — por modelo (usando fit "none" como referência)
        # ════════════════════════════════════════════════════════════════════════
        model_scores  = {}
        model_status  = {}   # 🟢 Estável | 🟡 Sensível | 🔴 Instável
        model_reject  = {}   # True = rejeitado por critério duro
        vm_ref  = {}
        fat_ref = {}

        for mk in ["M1","M2","M3","M4"]:
            cp,wp,pmax,pp,r2,k = all_fits[mk]["none"]
            stab = stability[mk]
            if cp is None or wp is None or stab is None:
                model_scores[mk]=999; model_status[mk]="❌"; model_reject[mk]=True; continue
            _,seep = calc_see(p_obs,pp,k)
            vm = vc_metrics(tests,cp,wp)
            vm_ref[mk]=vm; fat_ref[mk]=classify_fatigue(vm)
            cp_var = stab["cp_var"]
            wp_var = stab["wp_var"]
            cv_max = max(
                (vc_metrics(tests, all_fits[mk][wm][0], all_fits[mk][wm][1])["cv"]
                 for wm in W_MODES if all_fits[mk][wm][0] is not None),
                default=0)

            # ── Critérios duros de rejeição ──────────────────────────────────
            rejected = (cp_var > 15 or cv_max > 25 or (seep or 99) > 20)
            model_reject[mk] = rejected

            # ── Status visual ─────────────────────────────────────────────────
            if   cp_var < 5  and cv_max < 15: model_status[mk] = "🟢 Estável"
            elif cp_var < 10 and cv_max < 20: model_status[mk] = "🟡 Sensível"
            else:                              model_status[mk] = "🔴 Instável"
            if rejected: model_status[mk] = "🔴 Rejeitado"

            # ── Score — CP_var é critério principal (0.40) ───────────────────
            pen_k = 0.05*(k-2)
            sc = (0.40*(cp_var/30) +           # PRINCIPAL: estabilidade CP
                  0.30*(vm["cv"]/50 if vm["cv"] else 0) +   # consistência VC
                  0.20*((seep or 30)/30) +      # qualidade ajuste
                  0.10*(wp_var/30) +            # estabilidade W′
                  pen_k +
                  (0.50 if rejected else 0))    # penalidade dura
            model_scores[mk] = round(sc*100,1)

        # Preferir modelos não rejeitados; fallback ao menor score
        _candidates = {mk:sc for mk,sc in model_scores.items()
                       if not model_reject.get(mk,False)}
        if not _candidates: _candidates = model_scores   # fallback
        best_mk = min(_candidates, key=_candidates.get)
        best_cp,best_wp,best_pmax,_,_,_ = all_fits[best_mk]["none"]

        # ════════════════════════════════════════════════════════════════════════
        # TABELA PRINCIPAL — resumo por modelo
        # ════════════════════════════════════════════════════════════════════════
        rows_main = []
        for mk in ["M1","M2","M3","M4"]:
            cp,wp,pmax,pp,r2,k = all_fits[mk]["none"]
            stab = stability[mk]
            if cp is None or stab is None:
                rows_main.append({"Modelo":NOMES[mk],"CP médio (W)":"—",
                                  "CP var%":"—","W′ médio (J)":"—","W′ var%":"—",
                                  "CV W′%":"—","SEE%":"—","Score":"—","Robustez":"❌","Fadiga":"—"})
                continue
            _,seep = calc_see(p_obs,pp,k)
            vm = vm_ref.get(mk,{})
            cp_var = stab["cp_var"]
            rob = ("✅ Robusto" if cp_var<10 else
                   "⚠️ Sensível" if cp_var<20 else "❌ Instável")
            # SEE médio dos 3 cenários
            sees = [calc_see(p_obs,all_fits[mk][wm][3],k)[1]
                    for wm in W_MODES
                    if all_fits[mk][wm][3] is not None]
            see_mean = round(float(np.mean([s for s in sees if s])),2) if sees else None

            # CV médio dos 3 cenários
            cvs = [vc_metrics(tests,all_fits[mk][wm][0],all_fits[mk][wm][1])["cv"]
                   for wm in W_MODES if all_fits[mk][wm][0] is not None]
            cv_mean = round(float(np.mean(cvs)),1) if cvs else 0

            rows_main.append({
                "Modelo":        NOMES[mk],
                "Status":        model_status.get(mk,"—"),
                "CP médio (W)":  stab["cp_mean"],
                "CP var%":       f"{cp_var:.1f}%",
                "W′ médio (J)":  stab["wp_mean"],
                "W′ var%":       f"{stab['wp_var']:.1f}%",
                "CV médio%":     f"{cv_mean:.1f}%",
                "SEE médio%":    see_mean,
                "Score":         model_scores[mk],
                "Fadiga":        fat_ref.get(mk,"—"),
            })

        # ── Linha OmPD na tabela ────────────────────────────────────────────────
        if _ompd_cp is not None:
            _ompd_see, _ompd_seep = calc_see(p_obs, _ompd_pp, k=2)
            _ompd_vm  = vc_metrics(tests, _ompd_cp, _ompd_wp)
            _ompd_fat = classify_fatigue(_ompd_vm)
            _ompd_has_long = any(t > TCP_MAX for _, t in tests)
            _ompd_status = (
                "🟢 Estável" if (_ompd_seep or 99) < 5
                else "🟡 Sensível" if (_ompd_seep or 99) < 10
                else "🔴 Instável"
            )
            rows_main.append({
                "Modelo":        "M5: OmPD (Puchowicz 2020)",
                "Status":        _ompd_status,
                "CP médio (W)":  round(_ompd_cp, 1),
                "CP var%":       "N/A (1 fit)",
                "W′ médio (J)":  round(_ompd_wp, 0),
                "W′ var%":       "N/A",
                "CV médio%":     f"{_ompd_vm['cv']:.1f}%",
                "SEE médio%":    _ompd_seep,
                "Score":         "OmPD",
                "Fadiga":        _ompd_fat,
            })

        st.markdown("---")
        st.subheader("📋 Comparação por Modelo")
        st.dataframe(pd.DataFrame(rows_main),hide_index=True,use_container_width=True)

        # ── Card OmPD ────────────────────────────────────────────────────────────
        if _ompd_cp is not None:
            _has_A    = _ompd_A is not None and _ompd_A > 0.5
            _weff_ok  = _ompd_weff is not None and _ompd_weff > 95
            _h_r, _h_g, _h_b = 0x8e, 0x44, 0xad  # roxo
            st.markdown(
                f'<div style="padding:14px 18px; border-radius:10px; margin:10px 0; '
                f'background:rgba({_h_r},{_h_g},{_h_b},0.08); '
                f'border-left:6px solid #8e44ad;">'
                f'<b style="color:#8e44ad; font-size:1.05em;">M5: OmPD — Omni-Domain Power-Duration</b><br>'
                f'<span style="font-size:0.92em; color:#333;">'
                f'CP = <b>{_ompd_cp:.1f}W</b> &nbsp;|&nbsp; '
                f'W′ = <b>{_ompd_wp:.0f}J</b> &nbsp;|&nbsp; '
                f'Pmax = <b>{_ompd_pmax:.0f}W</b> &nbsp;|&nbsp; '
                f'A = <b>{_ompd_A:.1f}</b> {"(extensão longa activa)" if _has_A else "(sem pontos >30min)"}'
                f'<br>W′eff@120s = <b>{_ompd_weff:.1f}%</b> de W′ '
                f'{"✅ plateia atingida" if _weff_ok else "⚠️ plateia não atingida — testes muito curtos"}'
                f'<br><span style="color:#888; font-size:0.85em;">'
                f'Puchowicz MJ, Baker J & Clarke DC (2020). '
                f'Development and field validation of an omni-domain power-duration model. '
                f'<i>J Sports Sci.</i></span>'
                f'</span></div>',
                unsafe_allow_html=True
            )

        if best_cp and best_wp:
            stab_b = stability[best_mk]
            rob_b  = "✅ Robusto" if stab_b["cp_var"]<10 else "⚠️ Sensível"
            st.success(
                f"🏆 **Melhor:** {NOMES[best_mk]}  |  {rob_b}  |  "
                f"**{modalidade}** {data_teste}  |  "
                f"CP = **{best_cp:.1f} W**  |  W′ = **{best_wp:.0f} J**"
                +(f"  |  Pmax = **{best_pmax:.0f} W**" if best_pmax else ""))

            # Interpretação automática
            if stab_b["cp_var"] < 5:
                st.info("📊 **Perfil Endurance** — CP estável independente do weighting.")
            elif best_cp and stab_b["cp_vals"]:
                cp_range = max(stab_b["cp_vals"])-min(stab_b["cp_vals"])
                if cp_range > 10:
                    st.info("📊 **Perfil Anaeróbico dominante** — CP aumenta com 1/t² (sensível a testes curtos).")

        # ════════════════════════════════════════════════════════════════════════
        # TABELA DETALHADA — expandível
        # ════════════════════════════════════════════════════════════════════════
        with st.expander("🔍 Tabela detalhada — todos os modelos × todos os weightings"):
            rows_det = []
            for mk in ["M1","M2","M3","M4"]:
                for wmode in W_MODES:
                    cp,wp,pmax,pp,r2,k = all_fits[mk][wmode]
                    if cp is None: continue
                    _,seep = calc_see(p_obs,pp,k)
                    vm_d = vc_metrics(tests,cp,wp)
                    rows_det.append({
                        "Modelo":NOMES[mk],"Weight":wmode,
                        "CP (W)":round(cp,1),"W′ (J)":round(wp,0),
                        "Pmax":round(pmax,1) if pmax else "—",
                        "SEE%":seep,"CV W′%":f"{vm_d['cv']:.1f}%",
                        "R²":round(r2,4) if r2 else "—",
                    })
            st.dataframe(pd.DataFrame(rows_det),hide_index=True,use_container_width=True)

        # ════════════════════════════════════════════════════════════════════════
        # G1 — Power-Duration (melhor modelo, 3 weightings)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader(f"📈 Power-Duration — {NOMES[best_mk]}")
        st.caption("Curvas dos 3 cenários de weighting para o melhor modelo.")

        t_range = np.linspace(max(8,min(t for _,t in tests)*0.30),
                              max(t for _,t in tests)*3.0, 600)
        fig_pd = go.Figure()
        for wmode in W_MODES:
            cp,wp,pmax,pp,r2,k = all_fits[best_mk][wmode]
            if cp is None: continue
            fig_pd.add_trace(go.Scatter(
                x=t_range.tolist(),
                y=[max(0,wp/t+cp) for t in t_range],
                mode="lines",
                name=f"weight={wmode}  CP={cp:.0f}W",
                line=dict(color=CORES_W[wmode],width=2.5,
                          dash="solid" if wmode=="none" else
                               "dash"  if wmode=="1/t"  else "dot"),
                hovertemplate="t=%{x:.0f}s  P=%{y:.0f}W<extra></extra>"))

        # Curva OmPD no mesmo gráfico
        if _ompd_cp is not None:
            def _ompd_curve(t_arr, cp, wp, pmax_v, A):
                tau   = wp / max(pmax_v - cp, 1.0)
                base  = wp / t_arr * (1 - np.exp(-t_arr / tau)) + cp
                if A > 0.5:
                    decay = np.where(t_arr > TCP_MAX, A * np.log(t_arr / TCP_MAX), 0.0)
                    return base - decay
                return base
            y_ompd = _ompd_curve(t_range, _ompd_cp, _ompd_wp,
                                  _ompd_pmax, _ompd_A or 0.0)
            fig_pd.add_trace(go.Scatter(
                x=t_range.tolist(), y=np.maximum(y_ompd, 0).tolist(),
                mode="lines", name=f"M5: OmPD  CP={_ompd_cp:.0f}W",
                line=dict(color="#8e44ad", width=3, dash="longdash"),
                hovertemplate="t=%{x:.0f}s  P=%{y:.0f}W (OmPD)<extra></extra>"
            ))
            # Marcar TCPmax = 1800s se há extensão longa
            if _ompd_A and _ompd_A > 0.5:
                fig_pd.add_vline(x=TCP_MAX, line_dash="dot",
                                 line_color="#8e44ad", line_width=1.5,
                                 annotation_text="TCPmax=30min",
                                 annotation_font=dict(color="#8e44ad", size=10),
                                 annotation_position="top left")

        fig_pd.add_trace(go.Scatter(
            x=[t for _,t in tests],y=p_obs,
            mode="markers+text",
            text=[f"T{i+1}" for i in range(n)],
            textposition="top center",
            textfont=dict(color="#111",size=12,family="Arial Black"),
            marker=dict(size=14,color="#f39c12",symbol="circle",
                        line=dict(width=2,color="#2c3e50")),
            name="Testes reais"))
        fig_pd.update_layout(**BASE,
            title=dict(text=f"Power-Duration — {modalidade} ({data_teste})",
                       font=dict(size=14,color="#111")),
            height=430, hovermode='closest',
            xaxis=dict(title="Tempo (s)",**AX),
            yaxis=dict(title="Potência (W)",**AX))
        st.plotly_chart(fig_pd, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

        # ════════════════════════════════════════════════════════════════════════
        # G2 — Estabilidade CP × Weighting
        # ════════════════════════════════════════════════════════════════════════
        st.subheader("📊 Estabilidade de CP por modelo × weighting")
        st.caption("CP estável entre os 3 cenários = modelo robusto.")

        fig_stab = go.Figure()
        for mk in ["M1","M2","M3","M4"]:
            cp_vals = []
            for wmode in W_MODES:
                cp,_,_,_,_,_ = all_fits[mk][wmode]
                cp_vals.append(cp if cp else None)
            if all(v is None for v in cp_vals): continue
            fig_stab.add_trace(go.Scatter(
                x=W_MODES,
                y=[v for v in cp_vals],
                mode="lines+markers",
                name=NOMES[mk],
                line=dict(color=CORES_M[mk],width=2.5),
                marker=dict(size=10,color=CORES_M[mk]),
                hovertemplate="%{x}: CP=%{y:.1f}W<extra></extra>"))
        fig_stab.update_layout(**BASE,
            title=dict(text="CP por Weighting (linhas paralelas = estável)",
                       font=dict(size=13,color="#111")),
            height=320,
            xaxis=dict(title="Weighting",**AX),
            yaxis=dict(title="CP (W)",**AX))
        st.plotly_chart(fig_stab, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

        # ════════════════════════════════════════════════════════════════════════
        # G3 — VELOCLINIC (correcto — só pontos + linhas de referência)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader(f"🔬 Veloclinic Plot — {modalidade} — {data_teste}")

        with st.expander("📖 Como interpretar"):
            st.markdown("""
    **Fonte:** [veloclinic.com](https://veloclinic.com/veloclinic-plot-w-cp-subtraction-plot/)

    | Elemento | Descrição |
    |---|---|
    | **Eixo X** | Potência (W) |
    | **Eixo Y** | W′_point = t×(P−CP) |
    | **Sem curva** | A curva teórica = W′_point = W′ (linha horizontal trivial, sem informação) |
    | **Linha vermelha vert.** | CP do melhor modelo |
    | **Linha preta horiz.** | W′ do melhor modelo |
    | **Zona verde** | Janela 2–20 min (120–1200s) |

    **O diagnóstico está na posição dos pontos:**
    - Pontos próximos da linha W′ horizontal → W′ bem estimado
    - Pontos dispersos → fadiga central / pacing inconsistente
    - Pontos abaixo → W′ sobreestimado / fadiga periférica
            """)

        fig_vc = go.Figure()

        # Pontos reais — um scatter por modelo (usando CP de cada)
        for mk in ["M1","M2","M3","M4"]:
            cp,wp,_,_,_,_ = all_fits[mk]["none"]
            if cp is None or cp<=0: continue
            p_pts, wp_pts = veloclinic_points(tests, cp)
            hover_r = [f"T{i+1}: {tests[i][0]:.0f}W × {tests[i][1]:.0f}s" for i in range(n)]
            fig_vc.add_trace(go.Scatter(
                x=p_pts, y=wp_pts,
                mode="markers+text",
                text=[f"T{i+1}" for i in range(n)],
                textposition="top center",
                textfont=dict(color="#111",size=11,family="Arial Black"),
                marker=dict(size=15,color=CORES_M[mk],symbol="diamond",
                            line=dict(width=2,color="white")),
                name=NOMES[mk],
                customdata=hover_r,
                hovertemplate="%{customdata}<br>W′_point=%{y:.0f}J<extra></extra>"))

        if best_cp and best_wp:
            fig_vc.add_vline(x=best_cp,line_dash="dash",
                             line_color="#c0392b",line_width=2,
                             annotation_text=f"CP={best_cp:.0f}W",
                             annotation_font=dict(color="#c0392b",size=12),
                             annotation_position="top right")
            fig_vc.add_hline(y=best_wp,line_dash="dot",
                             line_color="#2c3e50",line_width=2,
                             annotation_text=f"W′={best_wp:.0f}J",
                             annotation_font=dict(color="#2c3e50",size=12),
                             annotation_position="right")
            p_2min  = best_wp/120  + best_cp
            p_20min = best_wp/1200 + best_cp
            if p_20min < p_2min:
                fig_vc.add_vrect(x0=p_20min,x1=p_2min,
                                 fillcolor="rgba(39,174,96,0.09)",line_width=0,
                                 annotation_text="Zona 2–20 min",
                                 annotation_position="top left",
                                 annotation_font=dict(size=10,color="#27ae60"))

        fig_vc.update_layout(**BASE,
            title=dict(text="Veloclinic — W′_point vs Potência (pontos reais + referências CP/W′)",
                       font=dict(size=13,color="#111")),
            height=440, hovermode="closest",
            xaxis=dict(title="Potência (W)",**AX),
            yaxis=dict(title="W′_point = t×(P−CP)  [J]",
                       zeroline=True,zerolinecolor="#aaaaaa",**AX))
        st.plotly_chart(fig_vc, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

        # Métricas Veloclinic
        vm_rows=[]
        for mk in ["M1","M2","M3","M4"]:
            cp,wp,_,_,_,_ = all_fits[mk]["none"]
            if cp is None: continue
            vm = vm_ref.get(mk,vc_metrics(tests,cp,wp))
            q_cv  = "✅" if vm["cv"]<10 else "⚠️" if vm["cv"]<25 else "❌"
            q_sl  = "✅" if abs(vm["slope"])<1 else "⚠️"
            vm_rows.append({
                "Modelo":NOMES[mk],
                "W′ médio (J)":vm["mean"],"Std W′":vm["std"],
                "CV W′%":f"{vm['cv']:.1f}% {q_cv}",
                "Slope":f"{vm['slope']:.4f} {q_sl}",
                "Fadiga":fat_ref.get(mk,"—"),
            })
        if vm_rows:
            st.dataframe(pd.DataFrame(vm_rows),hide_index=True,use_container_width=True)


        # Export
        st.markdown("---")
        _rows_exp = []
        for mk in ["M1","M2","M3","M4"]:
            for wmode in W_MODES:
                cp,wp,pmax,pp,r2,k = all_fits[mk][wmode]
                if cp is None: continue
                _,seep = calc_see(p_obs,pp,k)
                _rows_exp.append({"Modelo":NOMES[mk],"Weight":wmode,
                                  "CP":round(cp,1),"W′":round(wp,0),
                                  "Pmax":round(pmax,1) if pmax else "",
                                  "SEE%":seep,"Score":model_scores.get(mk,"")})
        _csv = pd.DataFrame(_rows_exp).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exportar CSV",_csv,
                           f"cp_{modalidade}_{data_teste}.csv",
                           "text/csv",key="dl_cp")
