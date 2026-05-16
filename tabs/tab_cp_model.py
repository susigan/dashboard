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

    # ── Extrair MMPs da sheet — excluindo MMP60 (3600s) dos pontos de fitting ──
    # MMP60 entra como validação externa, não como ponto de fitting
    # Papers recomendam 3-4 pontos entre 2-20min para CP (Monod, Morton, Puchowicz)
    _all_mmp_pts     = []   # todos sem MMP60 — para fitting
    _mmp60_val       = None # MMP60 guardado para validação
    _pmax_global     = None
    if ac_full is not None and len(ac_full) > 0:
        _col_mod  = next((c for c in ['type','modality'] if c in ac_full.columns), None)
        _col_date = next((c for c in ['date','Data'] if c in ac_full.columns), None)
        if _col_mod and _col_date:
            _ac_mod = ac_full[ac_full[_col_mod] == modalidade].copy()
            for _mc, _dur in MMP_COLS.items():
                if _mc not in _ac_mod.columns: continue
                # Row e Ski: usar MMP1(60s), MMP5(300s), MMP12(720s)
                # Excluir MMP3(180s), MMP20(1200s), MMP60(3600s)
                # MMP3 é redundante entre MMP1 e MMP5 nestas modalidades
                if modalidade in ('Row', 'Ski'):
                    if float(_dur) in (180.0, 1200.0, 3600.0): continue
                else:
                    # Bike/Run: excluir apenas MMP60
                    if float(_dur) == 3600.0:
                        pass  # tratado abaixo como validação
                if _mc == 'MMP60':           # sempre validação apenas
                    _ac_mod_s = _ac_mod.sort_values(_col_date, ascending=False)
                    for _, _rr in _ac_mod_s.iterrows():
                        _mv = parse_mmp(str(_rr[_mc]))
                        if _mv is not None:
                            _mmp60_val = _mv
                            break
                    continue
                _ac_mod_s = _ac_mod.sort_values(_col_date, ascending=False)
                for _, _rr in _ac_mod_s.iterrows():
                    _mv = parse_mmp(str(_rr[_mc]))
                    if _mv is not None:
                        _all_mmp_pts.append((_mv, float(_dur)))
                        break
            _all_mmp_pts = sorted(set(_all_mmp_pts), key=lambda x: x[1])
            # Pmax — mesma coluna do teste manual: p_max
            if 'p_max' in _ac_mod.columns:
                _px = (_ac_mod[['p_max', _col_date]]
                       .dropna(subset=['p_max'])
                       .sort_values(_col_date, ascending=False))
                if not _px.empty:
                    _pmax_global = float(_px['p_max'].iloc[0])

    # Cada modelo usa exactamente 3 pontos (papers) — testamos todas as C(n,3)
    # Ward-Smith: 3 pontos entre 2-20min (excluir ponto <60s para este modelo)
    _FIXED_N_PTS = 3   # número fixo de pontos por modelo (recomendação papers)

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
    if _all_mmp_pts:
        for _mn, _mcfg in _MODELS.items():
            _px   = _pmax_global if _mcfg['needs_pmax'] else None
            _pts  = ([p for p in _all_mmp_pts if p[1] >= 60]
                     if _mcfg.get('exclude_short') else _all_mmp_pts)
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
                _cp_mean = float(np.mean(_cp_v)) if _cp_v else 0
                _cp_range = (max(_cp_v)-min(_cp_v))/_cp_mean*100 if _cp_mean>0 and len(_cp_v)>1 else 0
                _see_mean = float(np.mean(_see_v)) if _see_v else _best['see_pct']
                _cv_pct   = float(np.std(_cp_v)/_cp_mean*100) if _cp_mean>0 and len(_cp_v)>1 else 0

                # Validação MMP60: erro relativo se disponível
                _mmp60_err = None
                if _mmp60_val and _best['result'][0]:
                    _cp_best = _best['result'][0]
                    _wp_best = _best['result'][1] if len(_best['result'])>1 else 0
                    _pred60  = (_wp_best/3600 + _cp_best) if isinstance(_wp_best, float) and _wp_best > 0 else None
                    if _pred60:
                        _mmp60_err = abs(_pred60 - _mmp60_val) / _mmp60_val * 100

                # Score composto (igual ao testes manuais)
                _sc = (0.40*(_cp_range/30) +
                       0.30*(_see_mean/20) +
                       0.20*(_cv_pct/20) +
                       0.10*((_mmp60_err or 0)/20))
                _best['cp_var_pct'] = round(_cp_range, 1)
                _best['cv_pct']     = round(_cv_pct, 1)
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
                f"ℹ️ **{modalidade}:** pontos usados: MMP1(1min), MMP5(5min), MMP12(12min). "
                "MMP3 excluído (redundante entre 1min e 5min). "
                "MMP20 e MMP60 excluídos."
            )

        st.info(f"**Pontos MMP disponíveis ({modalidade}):** " +
                " · ".join([f"{int(t//60)}min={p:.0f}W" for p, t in _all_mmp_pts]) +
                (f" | **Pmax:** {_pmax_global:.0f}W (da sheet)" if _pmax_global
                 else " | ⚠️ **Pmax não encontrado** — modelos Omni usarão estimativa (max×1.15). "
                      "Para melhores resultados, preencher coluna `p_max` ou `icu_pmax` na sheet."))

        if not _results:
            st.error("Nenhum modelo convergiu com os dados disponíveis.")
        else:
            st.caption(
                "**CP var%** = amplitude de CP entre todas as combinações de 3 pontos (≤5% = robusto). "
                "**CV%** = desvio-padrão de CP / média (≤5% = consistente). "
                "**SEE% médio** = erro médio entre todas as combinações. "
                "**Score** = composto ponderado (menor = melhor)."
            )
            _rank_rows = []
            for _mn, _gr in sorted(_results.items(), key=lambda x: x[1]['score']):
                _res  = _gr['result']
                _cp_v = _gr.get('cp')
                _wp_v = _res[1] if len(_res) > 1 else None
                _pts_lbl = " + ".join([f"{int(t//60)}min" for _, t in _gr['combo']])
                _n_combos = len(_gr['cp_vals'])

                _cp_var  = _gr.get('cp_var_pct', '—')
                _cv      = _gr.get('cv_pct', '—')
                _see_b   = round(_gr['see_pct'],   2)
                _see_m   = _gr.get('see_mean', '—')
                _score   = _gr.get('score', '—')
                _m60_err = _gr.get('mmp60_err')

                # Semáforos
                def _flag_cp_var(v):
                    if not isinstance(v, float): return '—'
                    return f"{'✅' if v<5 else '⚠️' if v<15 else '❌'} {v:.1f}%"
                def _flag_cv(v):
                    if not isinstance(v, float): return '—'
                    return f"{'✅' if v<5 else '⚠️' if v<10 else '❌'} {v:.1f}%"
                def _flag_see(v):
                    if not isinstance(v, float): return '—'
                    return f"{'✅' if v<2 else '⚠️' if v<5 else '❌'} {v:.2f}%"

                _rank_rows.append({
                    'Modelo':       _mn,
                    'CP (W)':       f"{_cp_v:.0f}" if _cp_v else "—",
                    "W′ (J)":       f"{_wp_v:.0f}" if isinstance(_wp_v, float) and _wp_v > 100 else "—",
                    'SEE% melhor':  _flag_see(_see_b),
                    'SEE% médio':   _flag_see(float(_see_m)) if isinstance(_see_m, float) else '—',
                    'CP var%':      _flag_cp_var(float(_cp_var)) if isinstance(_cp_var, float) else '—',
                    'CV%':          _flag_cv(float(_cv)) if isinstance(_cv, float) else '—',
                    'Val. 60min':   f"{_m60_err:.1f}%" if _m60_err else ('sem dado' if _mmp60_val is None else '—'),
                    'N combos':     _n_combos,
                    'Score ↓':      _score,
                    'Melhores 3pts':_pts_lbl,
                })
            _rank_df = pd.DataFrame(_rank_rows)

            # Colorir Score — menor = mais verde
            def _color_score(val):
                try:
                    v = float(val)
                    if v < 20: return 'background-color:#EAF3DE'
                    if v < 50: return 'background-color:#FEF3E2'
                    return 'background-color:#FDEAEA'
                except: return ''

            st.dataframe(_rank_df, hide_index=True, use_container_width=True)

            # Legenda
            st.caption(
                "✅ Excelente | ⚠️ Aceitável | ❌ Problemático — "
                f"3 pontos fixos por modelo (papers) | MMP60 excluído do fitting "
                + (f"(usado como validação: 60min={_mmp60_val:.0f}W)" if _mmp60_val else "(MMP60 não disponível)")
            )

            # Melhor modelo
            _best_mn  = sorted(_results.items(), key=lambda x: x[1]['score'])[0]
            _best_lbl, _best_gr = _best_mn
            _best_res = _best_gr['result']
            _best_cp  = _best_gr.get('cp')
            _best_wp  = _best_res[1] if len(_best_res)>1 else None

            st.success(
                f"**Modelo mais confiável: {_best_lbl}** | "
                f"Score={_best_gr['score']} | "
                f"SEE% melhor={_best_gr['see_pct']:.2f}% | "
                f"SEE% médio={_best_gr.get('see_mean','—')} | "
                f"CP var%={_best_gr.get('cp_var_pct','—')}% | "
                f"CV%={_best_gr.get('cv_pct','—')}% | "
                f"CP={_best_cp:.0f}W" + (f" | W′={_best_wp:.0f}J" if isinstance(_best_wp, float) and _best_wp>100 else "") +
                f" | Pontos: " + " + ".join([f"{int(t//60)}min" for _, t in _best_gr['combo']])
            )

            # Gráfico comparativo: todos os modelos na mesma curva
            _fig_comp = go.Figure()
            _t_comp   = np.logspace(np.log10(30), np.log10(10800), 400)
            _colors_rank = ['#8e44ad','#2980b9','#27ae60','#e67e22','#16a085','#d35400','#c0392b']

            # Pontos observados
            _fig_comp.add_trace(go.Scatter(
                x=[t for _, t in _all_mmp_pts],
                y=[p for p, _ in _all_mmp_pts],
                mode='markers+text',
                name='MMP',
                marker=dict(size=10, color='#e74c3c'),
                text=[f"{int(t//60)}min" for _, t in _all_mmp_pts],
                textposition='top center',
                textfont=dict(size=9),
            ))

            for (_mn, _gr), _col in zip(sorted(_results.items(),
                                                key=lambda x: x[1]['see_pct']),
                                         _colors_rank):
                _res = _gr['result']
                _cp_c = _gr['cp']
                _mcfg_c = _MODELS[_mn]
                _px_c = _pmax_global if _mcfg_c['needs_pmax'] else None
                # Recalcular curva completa com todos os pontos t_comp
                try:
                    if _mn in ('M1: P vs 1/t', '2P Hiperbólico',
                               'M2: Work-Time', 'M3: Hiperbólico-t'):
                        _cp_c2, _wp_c2 = _res[0], _res[1]
                        _y_comp = _wp_c2 / _t_comp + _cp_c2
                    elif _mn == 'OmPD':
                        _cp_c2, _wp_c2, _pm_c2, _A_c2 = _res[0], _res[1], _res[2], _res[3]
                        def _ompd_curve(t_arr, cp, wp, pmax_v, A):
                            tau  = wp / max(pmax_v - cp, 1.0)
                            base = wp / t_arr * (1 - np.exp(-t_arr / tau)) + cp
                            if A and A > 0:
                                decay = np.where(t_arr > TCP_MAX, A * np.log(t_arr / TCP_MAX), 0.0)
                                return base - decay
                            return base
                        _y_comp = _ompd_curve(_t_comp, _cp_c2, _wp_c2, _pm_c2 or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.15, _A_c2 or 0)
                    elif _mn == '2P Hiperbólico':
                        _cp_c2, _wp_c2 = _res[0], _res[1]
                        _y_comp = _wp_c2 / _t_comp + _cp_c2
                    elif _mn == '3P Hiperbólico':
                        _cp_c2, _wp_c2, _pm_c2 = _res[0], _res[1], _res[2]
                        _y_comp = (_pm_c2 * _wp_c2) / (_wp_c2 + (_pm_c2 - _cp_c2) * _t_comp)
                    elif _mn == 'Ward-Smith':
                        _cp_c2, _wp_c2 = _res[0], _res[1]
                        _pm_c2 = _res[2] or _pmax_global or max(p for p,_ in _all_mmp_pts)*1.2
                        _y_comp = _cp_c2 + (_pm_c2 - _cp_c2) * np.exp(-_t_comp * (_pm_c2 - _cp_c2) / max(_wp_c2, 1))
                    elif _mn == 'Power Law':
                        _a_pl, _b_pl = _res[1], _res[2]
                        _y_comp = _a_pl * _t_comp**(-_b_pl)
                    else:
                        continue
                    _fig_comp.add_trace(go.Scatter(
                        x=list(_t_comp), y=list(_y_comp),
                        mode='lines',
                        name=f"{_mn} ({_gr['see_pct']:.1f}%)",
                        line=dict(color=_col, width=2),
                    ))
                except Exception:
                    pass

            _fig_comp.update_layout(
                **BASE,
                title=dict(text=f"Comparação de modelos — {modalidade}", font=dict(size=14)),
                xaxis=dict(**AX, title="Duração (s)", type='log'),
                yaxis=dict(**AX, title="Potência (W)"),
            )
            st.plotly_chart(_fig_comp, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB OmPD (existente, mantida)
    # ════════════════════════════════════════════════════════════════════════
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
