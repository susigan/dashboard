# ══════════════════════════════════════════════════════════════════════════════
# utils/cp_model.py — ATHELTICA
# Módulo central de Critical Power (CP). Fonte ÚNICA de verdade para:
#   • Parsing de MMP real da sheet          → parse_mmp()
#   • Pesos WLS                             → make_w()
#   • Modelos de fit (M1-M4, OmPD, hyperb., → fit_m1 ... fit_power_law()
#     Ward-Smith, OM3CP, OMExp, power-law)
#   • Erro padrão (SEE)                     → calc_see()
#   • Veloclinic / classificação de fadiga  → veloclinic_points, vc_metrics, classify_fatigue
#   • Grid search do melhor subconjunto MMP → _grid_search_model()
#   • Orquestradora completa                → calcular_cp_completo()
#
# TODAS as funções são PURAS (sem Streamlit, sem BASE/AX de display) para
# poderem ser chamadas por qualquer tab (cp_model, pmc, visao_geral, etc.).
# Extraído de tab_cp_model.py — lógica idêntica, apenas reorganizada.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import linregress
from itertools import combinations

# Durações canónicas dos MMP (segundos) — usado no parsing
MMP_DURACOES = {'mmp1': 60, 'mmp3': 180, 'mmp5': 300,
                'mmp12': 720, 'mmp20': 1200, 'mmp60': 3600}

# Duração máxima considerada para o CP (segundos) — usada nos fits OM3CP/OMExp
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

def make_w(t_obs, mode):
    t = np.array(t_obs, dtype=float)
    if mode == "1/t":   return 1.0/t
    if mode == "1/t²":  return 1.0/t**2
    return np.ones_like(t)

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

def calc_see(p_obs, pp, k=2):
    n = len(p_obs)
    if n<=k: return None,None
    sse  = float(np.sum((np.array(p_obs)-np.array(pp))**2))
    see  = float(np.sqrt(sse/max(n-k,1)))
    seep = see/float(np.mean(p_obs))*100
    return round(see,2),round(seep,2)

def veloclinic_points(tests, cp):
    """
    Veloclinic: scatter P vs W′_point = t*(P-CP).
    SEM curva teórica — seria W′_point = W′ (linha horizontal trivial).
    O diagnóstico está na distribuição dos pontos reais.
    """
    p_pts  = [p for p,_ in tests]
    wp_pts = [t*(p-cp) for p,t in tests]
    return p_pts, wp_pts

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


# ══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DE DADOS — busca de MMP real das actividades
# ══════════════════════════════════════════════════════════════════════════════

# Colunas MMP na sheet (nome original) → duração em segundos
MMP_COLS = {'MMP1': 60, 'MMP3': 180, 'MMP5': 300,
            'MMP12': 720, 'MMP20': 1200, 'MMP60': 3600}


def preparar_mmp_pts(ac_full, modalidade):
    """
    Extrai os pontos MMP reais ("Yes - Xw") das actividades para uma modalidade.
    Réplica exacta da lógica da tab_cp_model (regras Row/Ski vs Bike/Run).

    Devolve dict:
      {
        'all_mmp_pts':      [(watts, dur_s), ...]  # M1/M2/M3
        'all_mmp_pts_full': [(watts, dur_s), ...]  # modelos não-clássicos
        'mmp60_val':        float|None             # validação (não entra no fit)
        'pmax':             float|None             # Pmax se existir
      }
    """
    out = {'all_mmp_pts': [], 'all_mmp_pts_full': [], 'mmp60_val': None, 'pmax': None}
    if ac_full is None or len(ac_full) == 0:
        return out
    _col_mod = next((c for c in ['type', 'modality'] if c in ac_full.columns), None)
    _col_date = next((c for c in ['date', 'Data'] if c in ac_full.columns), None)
    if not (_col_mod and _col_date):
        return out

    _ac_mod = ac_full[ac_full[_col_mod] == modalidade].copy()
    _all, _full = [], []
    for _mc, _dur in MMP_COLS.items():
        if _mc not in _ac_mod.columns:
            continue
        _dur_f = float(_dur)
        _ac_s = _ac_mod.sort_values(_col_date, ascending=False)
        if _mc == 'MMP60':
            for _, _rr in _ac_s.iterrows():
                _mv = parse_mmp(str(_rr[_mc]))
                if _mv is not None:
                    out['mmp60_val'] = _mv; break
            continue
        _mv = None
        for _, _rr in _ac_s.iterrows():
            _mv = parse_mmp(str(_rr[_mc]))
            if _mv is not None: break
        if _mv is None:
            continue
        if modalidade in ('Row', 'Ski'):
            if _dur_f in (60.0, 300.0, 720.0):
                _all.append((_mv, _dur_f))
            if _dur_f in (180.0, 300.0, 720.0, 1200.0):
                _full.append((_mv, _dur_f))
        else:
            _all.append((_mv, _dur_f))
            _full.append((_mv, _dur_f))
    out['all_mmp_pts'] = sorted(set(_all), key=lambda x: x[1])
    out['all_mmp_pts_full'] = sorted(set(_full), key=lambda x: x[1])

    if 'p_max' in _ac_mod.columns:
        _px = _ac_mod[['p_max', _col_date]].dropna(subset=['p_max']).sort_values(_col_date, ascending=False)
        if len(_px) > 0:
            try:
                out['pmax'] = float(_px.iloc[0]['p_max'])
            except Exception:
                pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRADORA — corre todos os modelos e selecciona o melhor
# ══════════════════════════════════════════════════════════════════════════════

def calcular_cp_completo(ac_full, modalidade, min_pts=3):
    """
    Pipeline completo de CP para uma modalidade, reutilizável por qualquer tab.
      1. Extrai MMP reais das actividades (preparar_mmp_pts)
      2. Corre os modelos principais via grid search (melhor subconjunto por SEE%)
      3. Selecciona o melhor modelo global (menor SEE%)

    Devolve dict:
      {
        'ok': bool, 'modalidade': str, 'n_mmp': int,
        'mmp_pts': [...], 'pmax': float|None, 'mmp60_val': float|None,
        'modelos': { nome: {cp, wp, pmax, see_pct, n_pts, pp} },
        'melhor': { 'nome':..., 'cp':..., 'wp':..., 'see_pct':... } | None,
      }
    """
    dados = preparar_mmp_pts(ac_full, modalidade)
    pts = dados['all_mmp_pts']
    pts_full = dados['all_mmp_pts_full']
    pmax = dados['pmax']

    if len(pts) < min_pts:
        return {'ok': False, 'modalidade': modalidade, 'n_mmp': len(pts),
                'reason': f'MMP insuficiente ({len(pts)} < {min_pts})',
                'mmp_pts': pts, 'pmax': pmax, 'mmp60_val': dados['mmp60_val'],
                'modelos': {}, 'melhor': None}

    # Modelos clássicos (2 params) usam all_mmp_pts; avançados usam full + pmax
    _modelos_def = [
        ('M1 (WLS-P)',      lambda t, **k: fit_m1(t, make_w([tt for _, tt in t], 'log')), pts,      2),
        ('M2 (WLS-1/t)',    lambda t, **k: fit_m2(t, make_w([tt for _, tt in t], 'log')), pts,      2),
        ('M3 (NL-2p)',      lambda t, **k: fit_m3(t, make_w([tt for _, tt in t], 'log')), pts,      2),
        ('2p hiperbólico',  lambda t, **k: fit_2p_hyperbolic(t),                          pts,      2),
        ('3p hiperbólico',  lambda t, pmax_ext=None: fit_3p_hyperbolic(t, pmax_ext),      pts_full, 3),
        ('Ward-Smith',      lambda t, pmax_ext=None: fit_ward_smith(t, pmax_ext),         pts_full, 3),
        ('OM3CP',           lambda t, pmax_ext=None: fit_om3cp(t, pmax_ext),              pts_full, 3),
        ('OMExp',           lambda t, pmax_ext=None: fit_omexp(t, pmax_ext),              pts_full, 3),
    ]

    modelos = {}
    for nome, fit_fn, base_pts, kp in _modelos_def:
        if len(base_pts) < min_pts:
            continue
        try:
            _pmax_ext = pmax if kp == 3 else None
            best = _grid_search_model(fit_fn, base_pts, min_pts=min(min_pts, len(base_pts)),
                                      pmax_ext=_pmax_ext, k_params=kp)
            if best and best['result']:
                res = best['result']
                modelos[nome] = {
                    'cp': float(res[0]) if res[0] is not None else None,
                    'wp': float(res[1]) if len(res) > 1 and res[1] is not None else None,
                    'see_pct': best['see_pct'],
                    'n_pts': best['n_pts'],
                }
        except Exception:
            pass

    # Melhor modelo global (menor SEE%)
    melhor = None
    if modelos:
        _bn = min(modelos, key=lambda n: modelos[n]['see_pct'])
        melhor = {'nome': _bn, **modelos[_bn]}

    return {'ok': len(modelos) > 0, 'modalidade': modalidade, 'n_mmp': len(pts),
            'mmp_pts': pts, 'pmax': pmax, 'mmp60_val': dados['mmp60_val'],
            'modelos': modelos, 'melhor': melhor}
