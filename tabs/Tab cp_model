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

def tab_cp_model():
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

    if not st.button("⚡ Calcular",type="primary"):
        st.info("Configura os testes e clica em **Calcular**.")
        return

    tests = [(float(p1),float(t1)),(float(p2),float(t2))]
    if usar3: tests.append((float(p3),float(t3)))
    tests = sorted(tests,key=lambda x:x[1])
    n = len(tests)

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

    st.markdown("---")
    st.subheader("📋 Comparação por Modelo")
    st.dataframe(pd.DataFrame(rows_main),hide_index=True,use_container_width=True)

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
