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
        if len(p_pts)>=2:
            sl,_,_,_,_ = linregress(p_pts, wp_pts)
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
    # DUAS TABS: OmPD MMP | Testes Manuais CP Model
    # ════════════════════════════════════════════════════════════════════════
    _main_tab_ompd, _main_tab_manual = st.tabs([
        "🏅 OmPD — MMP Automático",
        "🔬 Testes Manuais — CP Model",
    ])

    with _main_tab_ompd:
        st.caption(
            "Fitting OmPD automático com season bests (MMP) da sheet. "
            "Selecciona a modalidade no dropdown. "
            "**MMP1**=60s · **MMP3**=180s · **MMP5**=300s · "
            "**MMP12**=720s · **MMP20**=1200s · **MMP60**=3600s"
        )
        if ac_full is None or len(ac_full) == 0:
            st.info("Dados de actividades não disponíveis (ac_full).")
        else:
            _col_mod_dd  = next((c for c in ['type','modality'] if c in ac_full.columns), None)
            _col_date_dd = next((c for c in ['date','Data'] if c in ac_full.columns), None)
            if not _col_mod_dd or not _col_date_dd:
                st.warning("Colunas type/date não encontradas.")
            else:
                _mods_dd = [m for m in ['Bike','Row','Ski','Run']
                            if m in ac_full[_col_mod_dd].values]
                if not _mods_dd:
                    st.info("Sem modalidades em ac_full.")
                else:
                    _mod_dd = st.selectbox(
                        "Modalidade",
                        _mods_dd,
                        format_func=lambda m: {'Bike':'🚴 Bike','Row':'🚣 Row',
                                               'Ski':'🎿 Ski','Run':'🏃 Run'}.get(m,m),
                        key="ompd_dd_mod"
                    )
                    _sub_dd = (ac_full[ac_full[_col_mod_dd]==_mod_dd]
                               .sort_values(_col_date_dd, ascending=False))

                    # Pmax
                    _pm_dd = None
                    if 'p_max' in _sub_dd.columns:
                        for _v in _sub_dd['p_max']:
                            if _v is None or (isinstance(_v,float) and __import__('math').isnan(_v)):
                                continue
                            _p = parse_mmp(str(_v))
                            if _p:
                                _pm_dd = _p; break
                            try:
                                _f = float(_v)
                                if _f > 0:
                                    _pm_dd = _f; break
                            except: continue

                    # MMP
                    _mmp_dd = []
                    for _col, _dur in MMP_COLS.items():
                        if _col not in _sub_dd.columns: continue
                        for _rv in _sub_dd[_col]:
                            _mv = parse_mmp(_rv)
                            if _mv and _mv > 0:
                                _mmp_dd.append((_mv, float(_dur))); break
                    _mmp_dd = sorted(_mmp_dd, key=lambda x: x[1])

                    # Info cards
                    _ddc1, _ddc2 = st.columns(2)
                    _ddc1.metric("Pmax (p_max)", f"{_pm_dd:.0f}W" if _pm_dd else "—")
                    _ddc2.metric("Pontos MMP (Yes)", str(len(_mmp_dd)))

                    if len(_mmp_dd) < 2:
                        st.warning(f"⚠️ {_mod_dd}: menos de 2 pontos MMP com 'Yes'.")
                    else:
                        with st.expander(f"📋 Pontos MMP — {_mod_dd}", expanded=False):
                            _dd_rows = []
                            for _p, _t in _mmp_dd:
                                _lbl = next((k for k,v in MMP_COLS.items() if v==int(_t)),f"{int(_t)}s")
                                _m = int(_t)//60; _s = int(_t)%60
                                _dd_rows.append({
                                    'Coluna':_lbl,
                                    'Duração': f"{_m}min" if _s==0 else f"{_m}min{_s}s",
                                    'Potência (W)':int(_p),
                                    'Tipo':'Longo (>30min)' if _t>TCP_MAX else 'Curto (≤30min)'
                                })
                            st.dataframe(pd.DataFrame(_dd_rows), hide_index=True, use_container_width=True)

                        _res_dd = fit_ompd(_mmp_dd, pmax_ext=_pm_dd)
                        _cp_dd,_wp_dd,_pm2_dd,_A_dd,_pp_dd,_r2_dd,_weff_dd = _res_dd

                        if _cp_dd is None:
                            st.error(f"❌ OmPD fitting falhou para {_mod_dd}.")
                        else:
                            _ddr1,_ddr2,_ddr3,_ddr4,_ddr5 = st.columns(5)
                            _ddr1.metric("CP (W)", f"{_cp_dd:.1f}")
                            _ddr2.metric("W′ (J)", f"{_wp_dd:.0f}")
                            _ddr3.metric("Pmax (W)", f"{_pm2_dd:.0f}")
                            _ddr4.metric("A", f"{_A_dd:.1f}" if _A_dd and _A_dd>0.5 else "0")
                            _,_see_dd = calc_see([p for p,_ in _mmp_dd], _pp_dd, k=2) if _pp_dd else (None,None)
                            _ddr5.metric("SEE%", f"{_see_dd:.1f}%" if _see_dd else "—")

                            if _weff_dd and _weff_dd > 95:
                                st.success(f"✅ W′eff@120s = {_weff_dd:.1f}% — plateia atingida.")
                            else:
                                st.warning(f"⚠️ W′eff@120s = {_weff_dd:.1f}% — plateia não atingida.")

                            # Gráfico
                            _p_obs_dd = [p for p,_ in _mmp_dd]
                            _t_obs_dd = [t for _,t in _mmp_dd]
                            _trng_dd  = np.linspace(max(30.0,min(_t_obs_dd)*0.3),
                                                     max(_t_obs_dd)*1.5, 800)
                            def _ompd_p_dd(t_arr,cp,wp,pmax_v,A):
                                tau  = wp/max(pmax_v-cp,1.0)
                                base = wp/t_arr*(1-np.exp(-t_arr/tau))+cp
                                if A and A>0.5:
                                    return base-np.where(t_arr>TCP_MAX,A*np.log(t_arr/TCP_MAX),0.0)
                                return base
                            _ydd = _ompd_p_dd(_trng_dd,_cp_dd,_wp_dd,_pm2_dd,_A_dd or 0.0)

                            _fig_dd = go.Figure()
                            _fig_dd.add_trace(go.Scatter(
                                x=_trng_dd.tolist(), y=np.maximum(_ydd,0).tolist(),
                                mode='lines', name='OmPD',
                                line=dict(color='#8e44ad',width=3),
                                hovertemplate='t=%{x:.0f}s  P=%{y:.0f}W<extra></extra>'
                            ))
                            if _A_dd and _A_dd>0.5:
                                _fig_dd.add_vline(x=TCP_MAX,line_dash='dot',
                                                  line_color='#8e44ad',line_width=1.5,
                                                  annotation_text='TCPmax=30min',
                                                  annotation_font=dict(color='#8e44ad',size=10),
                                                  annotation_position='top left')
                            _m1cp,_m1wp,_,_m1pp,_,_ = fit_m1(_mmp_dd,make_w(_t_obs_dd,'none'))
                            if _m1cp:
                                _fig_dd.add_trace(go.Scatter(
                                    x=_trng_dd.tolist(),
                                    y=[max(0,_m1wp/t+_m1cp) for t in _trng_dd],
                                    mode='lines', name=f'M1 CP={_m1cp:.0f}W',
                                    line=dict(color='#e74c3c',width=1.5,dash='dash'),
                                    hovertemplate='t=%{x:.0f}s  P=%{y:.0f}W (M1)<extra></extra>'
                                ))
                            _fig_dd.add_trace(go.Scatter(
                                x=_t_obs_dd, y=_p_obs_dd,
                                mode='markers+text',
                                text=[next((k for k,v in MMP_COLS.items() if v==int(t)),str(int(t))+'s')
                                      for _,t in _mmp_dd],
                                textposition='top center',
                                textfont=dict(color='#111',size=11,family='Arial Black'),
                                marker=dict(size=13,color='#f39c12',symbol='circle',
                                            line=dict(width=2,color='#2c3e50')),
                                name='MMP (season bests)',
                                hovertemplate='%{text}: %{y:.0f}W<extra></extra>'
                            ))
                            _fig_dd.add_hline(y=_cp_dd,line_dash='dot',line_color='#2c3e50',
                                              line_width=1.5,
                                              annotation_text=f'CP={_cp_dd:.0f}W',
                                              annotation_font=dict(color='#2c3e50',size=11),
                                              annotation_position='right')
                            _fig_dd.update_layout(**BASE,
                                title=dict(
                                    text=f"OmPD — {_mod_dd} | CP={_cp_dd:.0f}W | W′={_wp_dd:.0f}J | Pmax={_pm2_dd:.0f}W | A={_A_dd:.1f}",
                                    font=dict(size=13,color='#111')
                                ),
                                height=430, hovermode='closest',
                                xaxis=dict(title='Duração (s)',
                                           type='log' if max(_t_obs_dd)>1200 else 'linear',**AX),
                                yaxis=dict(title='Potência (W)',**AX)
                            )
                            st.plotly_chart(_fig_dd,use_container_width=True,
                                            config={'displayModeBar':False,'responsive':True,'scrollZoom':False},
                                            key=f"ompd_dd_{_mod_dd}")

                            _has_long_dd = any(t>TCP_MAX for _,t in _mmp_dd)
                            if _has_long_dd and _A_dd and _A_dd>0.5:
                                st.info(
                                    f"📉 A={_A_dd:.1f} — "
                                    f"60min: P≈{_ompd_p_dd(np.array([3600.0]),_cp_dd,_wp_dd,_pm2_dd,_A_dd)[0]:.0f}W | "
                                    f"2h: P≈{_ompd_p_dd(np.array([7200.0]),_cp_dd,_wp_dd,_pm2_dd,_A_dd)[0]:.0f}W"
                                )
                            elif not _has_long_dd:
                                st.info(f"⚠️ Sem MMP60. CP={_cp_dd:.0f}W válido para 2–30min.")
                            if len(_mmp_dd)<4:
                                st.warning(f"⚠️ Apenas {len(_mmp_dd)} pontos MMP. Ideal ≥4.")

                            st.download_button(
                                f"⬇️ Export OmPD {_mod_dd}",
                                pd.DataFrame({
                                    'Modalidade':[_mod_dd],'CP (W)':[round(_cp_dd,1)],
                                    "W' (J)":[round(_wp_dd,0)],'Pmax (W)':[round(_pm2_dd,0)],
                                    'A':[round(_A_dd,2) if _A_dd else 0],
                                    'SEE%':[round(_see_dd,2) if _see_dd else None],
                                    "W'eff@120s%":[round(_weff_dd,1) if _weff_dd else None],
                                }).to_csv(index=False).encode('utf-8'),
                                f"ompd_{_mod_dd.lower()}.csv","text/csv",
                                key=f"dl_ompd_dd_{_mod_dd}"
                            )

    with _main_tab_manual:
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
