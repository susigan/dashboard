from utils.config import *
from utils.helpers import *
from utils.data import *
import utils.cp_model as _cpm
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re as _re
import warnings
warnings.filterwarnings('ignore')

# ── Modelos de CP (nível de módulo) ────────────────────────────────────────
# _MODELS e os wrappers _rank_m1/m2/m3 precisam de estar acessíveis tanto por
# _computar_resultados_cp_modelos() (cacheada) como por _show_model_tab()
# (dentro de tab_cp_model(), mostra o detalhe de grid search por modelo).
def _rank_m1(pts, **kw):
    """M1 no Ranking: CP médio dos 3 weightings — igual ao teste manual.
    SEE calculado com o weighting none (baseline)."""
    t_obs = np.array([t for _,t in pts])
    p_obs = np.array([p for p,_ in pts])
    cp_vals, wp_vals = [], []
    for mode in ["none","1/t","1/t²"]:
        w = _cpm.make_w(t_obs, mode)
        res = _cpm.fit_m1(pts, w)
        if res[0] is not None and res[1] is not None:
            cp_vals.append(res[0]); wp_vals.append(res[1])
    if not cp_vals: return _cpm.fit_m1(pts, np.ones(len(pts)))
    cp_mean = float(np.mean(cp_vals))
    wp_mean = float(np.mean(wp_vals))
    # pp calculado com CP médio
    pp = [wp_mean/t + cp_mean for _,t in pts]
    return cp_mean, wp_mean, None, pp, 0.0, 2

def _rank_m2(pts, **kw):
    """M2 no Ranking: CP médio dos 3 weightings — igual ao teste manual."""
    t_obs = np.array([t for _,t in pts])
    p_obs = np.array([p for p,_ in pts])
    cp_vals, wp_vals = [], []
    for mode in ["none","1/t","1/t²"]:
        w = _cpm.make_w(t_obs, mode)
        res = _cpm.fit_m2(pts, w)
        if res[0] is not None and res[1] is not None:
            cp_vals.append(res[0]); wp_vals.append(res[1])
    if not cp_vals: return _cpm.fit_m2(pts, np.ones(len(pts)))
    cp_mean = float(np.mean(cp_vals))
    wp_mean = float(np.mean(wp_vals))
    pp = [cp_mean + wp_mean/t for _,t in pts]
    return cp_mean, wp_mean, None, pp, 0.0, 2

def _rank_m3(pts, **kw):
    """M3 no Ranking: CP médio dos 3 weightings — igual ao teste manual."""
    t_obs = np.array([t for _,t in pts])
    p_obs = np.array([p for p,_ in pts])
    cp_vals, wp_vals = [], []
    for mode in ["none","1/t","1/t²"]:
        w = _cpm.make_w(t_obs, mode)
        res = _cpm.fit_m3(pts, w)
        if res[0] is not None and res[1] is not None and res[0] > 0 and res[1] > 0:
            cp_vals.append(res[0]); wp_vals.append(res[1])
    if not cp_vals: return _cpm.fit_m3(pts, np.ones(len(pts)))
    cp_mean = float(np.mean(cp_vals))
    wp_mean = float(np.mean(wp_vals))
    pp = [wp_mean/t + cp_mean for _,t in pts]
    return cp_mean, wp_mean, None, pp, 0.0, 2

# Cada modelo usa exactamente 3 pontos (papers)
_FIXED_N_PTS = 3

# Modelos clássicos para CP Row/Ski
_M_CLASSICOS = ('M1: P vs 1/t', 'M2: Work-Time', 'M3: Hiperbólico-t')

# Semáforos
def _flag_see(v):
    if not isinstance(v, float): return '—'
    return f"{'✅' if v<2 else '⚠️' if v<5 else '❌'} {v:.2f}%"
def _flag_cp_var(v):
    if not isinstance(v, float): return '—'
    return f"{'✅' if v<5 else '⚠️' if v<15 else '❌'} {v:.1f}%"
def _flag_cv(v):
    if not isinstance(v, float): return '—'
    return f"{'✅' if v<5 else '⚠️' if v<10 else '❌'} {v:.1f}%"

# ── Correr grid search para todos os modelos ──────────────────────────
_MODELS = {
    # M1/M2/M3: usam _rank_m1/m2/m3 — 3 weightings, igual ao teste manual
    'M1: P vs 1/t':   {'fn': _rank_m1,
                        'n_pts': 3, 'k': 2, 'color': '#e74c3c',
                        'needs_pmax': False,
                        'desc': 'P = W′/t + CP. 3 weightings. Monod & Scherrer 1965.'},
    'M2: Work-Time':  {'fn': _rank_m2,
                        'n_pts': 3, 'k': 2, 'color': '#2980b9',
                        'needs_pmax': False,
                        'desc': 'W = CP·t + W′. 3 weightings. Morton 1986.'},
    'M3: Hiperbólico-t':{'fn': _rank_m3,
                         'n_pts': 3, 'k': 2, 'color': '#27ae60',
                         'needs_pmax': False,
                         'desc': 't = W′/(P-CP). SEE em espaço t. 3 weightings.'},
    'OmPD':          {'fn': _cpm.fit_ompd,          'n_pts': 3, 'k': 2,
                      'color': '#8e44ad', 'needs_pmax': True,
                      'desc': '3 pts (incluindo ≤3min para Pmax). Puchowicz 2020.'},
    '2P Hiperbólico':{'fn': _cpm.fit_2p_hyperbolic, 'n_pts': 3, 'k': 2,
                      'color': '#1abc9c', 'needs_pmax': False,
                      'desc': '3 pts entre 3-20min. Monod & Scherrer 1965.'},
    '3P Hiperbólico':{'fn': _cpm.fit_3p_hyperbolic, 'n_pts': 3, 'k': 2,
                      'color': '#f39c12', 'needs_pmax': True,
                      'desc': '3 pts incluindo ≤3min. Morton 1996.'},
    'Ward-Smith':    {'fn': _cpm.fit_ward_smith,    'n_pts': 3, 'k': 2,
                      'color': '#e67e22', 'needs_pmax': True,
                      'desc': '3 pts entre 3-20min (excluir <60s). Ward-Smith 1999.',
                      'exclude_short': True},
    'Om3CP':         {'fn': _cpm.fit_om3cp,         'n_pts': 3, 'k': 2,
                      'color': '#16a085', 'needs_pmax': True,
                      'desc': '3 pts incluindo ≤3min. Puchowicz variante.'},
    'OmExp':         {'fn': _cpm.fit_omexp,         'n_pts': 3, 'k': 2,
                      'color': '#d35400', 'needs_pmax': True,
                      'desc': '3 pts. Variante OmPD com decaimento exponencial.'},
    'Power Law':     {'fn': _cpm.fit_power_law,     'n_pts': 3, 'k': 2,
                      'color': '#c0392b', 'needs_pmax': False,
                      'desc': '3 pts entre 3-20min. Sem CP explícito.'},
}


@st.cache_data(show_spinner="A ajustar modelos de Critical Power...", ttl=3600)
def _computar_resultados_cp_modelos(modalidade, _all_mmp_pts, _all_mmp_pts_full,
                                    _pmax_global, _mmp60_val):
    """
    Grid search de todos os modelos de CP (M1/M2/M3 + OmPD + hiperbolicos +
    Power Law etc.) sobre todas as combinacoes de pontos MMP disponiveis.

    Isto envolve dezenas a centenas de chamadas a optimizacao numerica
    (scipy.optimize.minimize, dentro das funcoes fit_*) — e por isso a parte
    mais pesada de toda a aba CP Model. Antes desta funcao existir, isto
    corria de novo em TODO rerun do Streamlit, mesmo sem os dados de entrada
    terem mudado (o que acontecia sempre que qualquer widget, em qualquer
    aba do dashboard, era tocado). Agora so recalcula quando modalidade ou
    os pontos MMP realmente mudam — o cache do Streamlit trata disso.

    Parametros: apenas dados simples (str, listas de tuplos de floats,
    float/None) para que o Streamlit consiga fazer o hash da chamada.

    Devolve o dict _results (uma entrada por modelo).
    """
    # M1/M2/M3 são os modelos classicos usados para Row/Ski
    _M_CLASSICOS = ('M1: P vs 1/t', 'M2: Work-Time', 'M3: Hiperbólico-t')

    # _rank_m1/m2/m3 e _MODELS agora são de nível de módulo (ver topo do
    # ficheiro) — acessíveis tanto por esta função como por _show_model_tab().


    # Grid search: exactamente N=3 pontos (C(n,3) combinações)
    # Ward-Smith exclui pontos <60s
    _results = {}
    if _all_mmp_pts_full:
        for _mn, _mcfg in _MODELS.items():
            _px    = _pmax_global if _mcfg['needs_pmax'] else None
            # M1/M2/M3 (clássicos): usam sempre _all_mmp_pts — Row/Ski restrito a
            # MMP1+MMP5+MMP12; Bike/Run sem restrição (todos os 5 pontos), tal
            # como já era antes desta correcção. A condição já não depende da
            # modalidade aqui — é o próprio _all_mmp_pts que já foi construído
            # por modalidade acima.
            # Todos os outros modelos: usam _all_mmp_pts_full — Row/Ski exclui
            # MMP1; Bike exclui MMP1+MMP3; Run sem restrição.
            _pts_base = (_all_mmp_pts if _mn in _M_CLASSICOS else _all_mmp_pts_full)
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
                    _, _see = _cpm.calc_see(_p_obs, _pp_gs, k=_mcfg['k'])
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

    return _results


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

    parse_mmp = _cpm.parse_mmp

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
    make_w = _cpm.make_w

    # ════════════════════════════════════════════════════════════════════════
    # MODELOS
    # ════════════════════════════════════════════════════════════════════════
    fit_m1 = _cpm.fit_m1

    fit_m2 = _cpm.fit_m2

    fit_m3 = _cpm.fit_m3

    fit_m4 = _cpm.fit_m4

    fit_ompd = _cpm.fit_ompd

    FIT_FNS = {"M1":fit_m1,"M2":fit_m2,"M3":fit_m3,"M4":fit_m4}

    # ── SEE ─────────────────────────────────────────────────────────────────
    calc_see = _cpm.calc_see

    # ── Veloclinic ───────────────────────────────────────────────────────────
    veloclinic_points = _cpm.veloclinic_points

    # ── Métricas ─────────────────────────────────────────────────────────────
    vc_metrics = _cpm.vc_metrics

    classify_fatigue = _cpm.classify_fatigue

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

    fit_2p_hyperbolic = _cpm.fit_2p_hyperbolic

    fit_3p_hyperbolic = _cpm.fit_3p_hyperbolic

    fit_ward_smith = _cpm.fit_ward_smith

    fit_om3cp = _cpm.fit_om3cp

    fit_omexp = _cpm.fit_omexp

    fit_power_law = _cpm.fit_power_law

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
    # Row/Ski outros modelos: usar MMP3, MMP5, MMP12, MMP20 (excluir MMP1)
    # Bike outros modelos: usar MMP5, MMP12, MMP20 (excluir MMP1 e MMP3 —
    #   esforços de 1-3min são dominados pela capacidade anaeróbia, não pela
    #   assíntota aeróbia que os modelos de CP pretendem capturar; sem isto o
    #   grid search podia escolher 1+3+5min só por dar menor SEE%, mesmo sendo
    #   fisiologicamente inadequado para estimar CP)
    # Run: usar MMP1, MMP3, MMP5, MMP12, MMP20 (mantém-se sem restrição)

    _all_mmp_pts      = []   # M1/M2/M3 Row/Ski: MMP1+MMP5+MMP12
    _all_mmp_pts_full = []   # outros modelos: todos excluindo MMP60 (e MMP1/MMP3 no Bike)
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
                #   Row/Ski → MMP1(60s) + MMP5(300s) + MMP12(720s)
                #   Bike/Run → todos exceto MMP60
                if modalidade in ('Row', 'Ski'):
                    if _dur_f in (60.0, 300.0, 720.0):
                        _all_mmp_pts.append((_mv, _dur_f))
                else:
                    _all_mmp_pts.append((_mv, _dur_f))

                # _all_mmp_pts_full — modelos não-clássicos:
                #   Row/Ski → MMP3+MMP5+MMP12+MMP20 (excluir MMP1 e MMP60)
                #   Bike    → MMP5+MMP12+MMP20 (excluir MMP1, MMP3 e MMP60)
                #   Run     → todos exceto MMP60
                if modalidade in ('Row', 'Ski'):
                    if _dur_f in (180.0, 300.0, 720.0, 1200.0):
                        _all_mmp_pts_full.append((_mv, _dur_f))
                elif modalidade == 'Bike':
                    if _dur_f in (300.0, 720.0, 1200.0):
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

    # Cada modelo usa exactamente 3 pontos (papers)
    _FIXED_N_PTS = 3

    # Modelos clássicos para CP Row/Ski
    _M_CLASSICOS = ('M1: P vs 1/t', 'M2: Work-Time', 'M3: Hiperbólico-t')

    # Semáforos
    def _flag_see(v):
        if not isinstance(v, float): return '—'
        return f"{'✅' if v<2 else '⚠️' if v<5 else '❌'} {v:.2f}%"
    def _flag_cp_var(v):
        if not isinstance(v, float): return '—'
        return f"{'✅' if v<5 else '⚠️' if v<15 else '❌'} {v:.1f}%"
    def _flag_cv(v):
        if not isinstance(v, float): return '—'
        return f"{'✅' if v<5 else '⚠️' if v<10 else '❌'} {v:.1f}%"

    # Grid search de todos os modelos — cacheado (ver _computar_resultados_cp_modelos
    # acima). Só recalcula quando a modalidade ou os pontos MMP mudam de facto,
    # em vez de recorrer em todo e qualquer rerun do Streamlit.
    _results = _computar_resultados_cp_modelos(
        modalidade, _all_mmp_pts, _all_mmp_pts_full, _pmax_global, _mmp60_val)

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

        # A tabela de Ranking continua a mostrar TODOS os modelos, para todas
        # as modalidades — Power Law, OmPD, etc. continuam visíveis para
        # comparação. Só a ESCOLHA do "melhor" (mais abaixo) fica restrita a
        # M1/M2/M3: o Power Law não tem assíntota física nenhuma (a potência
        # tende para zero com a duração, nunca estabiliza — ver "Modelling
        # human endurance: power laws vs critical power", Drake et al.), e
        # ajustado com 2 parâmetros a apenas 3 pontos ganha quase sempre por
        # artefacto estatístico (SEE%≈0%), não por ser fisiologicamente melhor.
        _results_rank = _results
        _candidatos_melhor = {k: v for k, v in _results.items() if k in _M_CLASSICOS}

        if not _all_mmp_pts:
            st.warning("Sem pontos MMP disponíveis.")
            st.stop()

        st.info(
            f"ℹ️ **{modalidade} — M1/M2/M3 (elegíveis a 'melhor') usam:** " +
            " · ".join([f"{int(t//60)}min={p:.0f}W" for p,t in _all_mmp_pts]) +
            " | **Outros modelos (informativos, não competem por 'melhor'):** " +
            " · ".join([f"{int(t//60)}min={p:.0f}W" for p,t in _all_mmp_pts_full]) +
            (f" | **Pmax:** {_pmax_global:.0f}W" if _pmax_global else "") +
            " | MMP60 excluído. Ranking mostra apenas M1/M2/M3."
        )

        if not _results:
            st.error("Nenhum modelo convergiu com os dados disponíveis.")
        else:
            st.caption(
                "**SEE%** = erro padrão (menor = melhor). "
                "**CP var%** = variação CP entre combinações (ou entre M1/M2/M3). "
                "**CV%** = desvio-padrão/média. "
                "Todos os modelos aparecem aqui para comparação, mas só M1/M2/M3 "
                "são elegíveis a 'melhor' — ver nota abaixo sobre o Power Law."
            )

            _rank_rows = []
            for _mn, _gr in sorted(_results_rank.items(),
                                   key=lambda x: x[1]['see_pct']):
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

                _rank_rows.append({
                    'Modelo':       ('⭐ ' if _mn in _M_CLASSICOS else '') + _mn,
                    'CP (W)':       f"{_cp_v:.0f}" if _cp_v else "—",
                    "W′ (J)":       f"{_wp_v:.0f}" if isinstance(_wp_v, float) and _wp_v > 100 else "—",
                    'SEE% melhor':  _flag_see(_see_b),
                    'SEE% médio':   _flag_see(float(_see_m)) if isinstance(_see_m, float) else '—',
                    'CP var%':      _flag_cp_var(float(_cp_var)) if isinstance(_cp_var, float) else '—',
                    'CV%':          _flag_cv(float(_cv)) if isinstance(_cv, float) else '—',
                    'Pontos (W)':   ", ".join([f"{p:.0f}W" for p,_ in _gr['combo']]),
                    'Val. 60min':   f"{_m60_err:.1f}%" if _m60_err else ('—' if _mmp60_val is None else '—'),
                    'Score ↓':      _score,
                })
            _rank_df = pd.DataFrame(_rank_rows)
            st.dataframe(_rank_df, hide_index=True, use_container_width=True)

            st.caption(
                "✅ Excelente | ⚠️ Aceitável | ❌ Problemático — "
                f"3 pontos por modelo | MMP60 excluído do fitting "
                + (f"(validação: 60min={_mmp60_val:.0f}W)" if _mmp60_val else "(MMP60 não disponível)")
                + " | ⭐ = elegível a 'melhor' (M1/M2/M3). "
                  "O Power Law não tem assíntota física (a potência tende para zero "
                  "com a duração, nunca estabiliza — Drake et al. 2022/2023) e, ajustado "
                  "com 2 parâmetros a 3 pontos, ganha quase sempre por artefacto "
                  "estatístico (SEE%≈0%) — por isso nunca compete pelo 'melhor', mesmo "
                  "aparecendo aqui para comparação."
            )

            # Melhor modelo — menor SEE% entre os mostrados
            _best_mn  = sorted(_candidatos_melhor.items(), key=lambda x: x[1]['see_pct'])[0]
            _best_lbl, _best_gr = _best_mn
            _best_res = _best_gr['result']
            _best_cp  = _best_gr.get('cp')
            _best_wp  = _best_res[1] if _best_res and len(_best_res) > 1 else None

            _best_cp_val   = _best_cp
            _best_cp_res   = _best_res
            _best_cp_lbl   = _best_lbl
            _best_cp_gr    = _best_gr
            _res_classicos = _candidatos_melhor
            _M_CP_FILTER   = True   # "melhor" sempre restrito a M1/M2/M3, em todas as modalidades

            # Guardar no session_state
            if _best_cp:
                st.session_state[f"cp_model_{modalidade}"] = {
                    'cp':      float(_best_cp),
                    'wp':      float(_best_wp) if isinstance(_best_wp, float) else None,
                    'modelo':  _best_lbl,
                    'see_pct': _best_gr.get('see_pct'),
                }

            st.success(
                f"**{'CP ' + modalidade + ' (menor SEE%)' if _M_CP_FILTER else 'Modelo mais confiável'}: "
                f"{_best_lbl}** | "
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

            # ── Calculadora Row/Ski: Watts ↔ Split ───────────────────────────
            if modalidade in ('Row', 'Ski'):
                st.markdown("---")
                st.markdown("**🚣 Calculadora de Performance — Row/Ski**")
                st.caption(
                    "Baseada no ergómetro Concept2. "
                    "Insere os Watts de cada teste — o valor de **2km é a referência central**. "
                    "Os outros são calculados automaticamente."
                )

                def _split_sec(w):
                    """Watts → segundos por 500m (Concept2)."""
                    if not w or w <= 0: return None
                    return ((2.8 / w) ** (1/3)) / 86400 * 500 * 86400

                def _fmt_split(t_sec):
                    if t_sec is None: return "—"
                    m = int(t_sec // 60)
                    s = t_sec % 60
                    return f"{m}:{s:05.2f}"

                def _w_from_split(split_str):
                    """MM:SS.ss → Watts."""
                    try:
                        parts = split_str.strip().replace(",",".").split(":")
                        t = float(parts[0])*60 + float(parts[1])
                        return 2.8 / (t/86400*86400/500)**3
                    except:
                        return None

                # Percentagens ideais por teste
                _pcts = {
                    "Power Peak": 173,
                    "60seg":      153,
                    "2km":        100,
                    "6km":         85,
                    "60min":       76,
                }

                # Input de Watts por teste — todos vazios (utilizador preenche)
                st.markdown("**Insere os Watts medidos (deixa 0 se não disponível):**")
                _ci1, _ci2, _ci3, _ci4, _ci5 = st.columns(5)
                _w_pp  = _ci1.number_input("Power Peak", 0, 3000, 0, 5, key="calc_w_pp")
                _w_60  = _ci2.number_input("60 seg",     0, 2000, 0, 5, key="calc_w_60")
                _w_2k  = _ci3.number_input("2km ★",     0, 1000, 0, 5, key="calc_w_2k",
                                            help="Referência central — todos os outros são calculados a partir daqui")
                _w_6k  = _ci4.number_input("6km",        0, 800,  0, 5, key="calc_w_6k")
                _w_60m = _ci5.number_input("60 min",     0, 700,  0, 5, key="calc_w_60m")

                _inputs = {
                    "Power Peak": _w_pp  if _w_pp  > 0 else None,
                    "60seg":      _w_60  if _w_60  > 0 else None,
                    "2km":        _w_2k  if _w_2k  > 0 else None,
                    "6km":        _w_6k  if _w_6k  > 0 else None,
                    "60min":      _w_60m if _w_60m > 0 else None,
                }

                if _w_2k > 0:
                    # Calcular tabela completa
                    _rows_calc = []
                    for _teste, _pct_ideal in _pcts.items():
                        _w_in   = _inputs.get(_teste)
                        _w_atual = _w_in or None

                        # Actual% = w_atual / w_2km
                        _pct_atual = (_w_atual / _w_2k * 100) if _w_atual else None

                        # Watts objectivo = %ideal × w_2km
                        _w_obj = _w_2k * _pct_ideal / 100

                        # Split objectivo
                        _split_obj = _fmt_split(_split_sec(_w_obj))

                        # Split actual (se tiver watts actual)
                        _split_atual = _fmt_split(_split_sec(_w_atual)) if _w_atual else "—"

                        _rows_calc.append({
                            "Teste":       _teste,
                            "Watts (real)":f"{_w_atual:.0f}W" if _w_atual else "—",
                            "Split (real)":_split_atual,
                            "% Actual":    f"{_pct_atual:.0f}%" if _pct_atual else "—",
                            "% Ideal":     f"{_pct_ideal}%",
                            "Watts (obj)": f"{_w_obj:.0f}W",
                            "Split (obj)": _split_obj,
                        })

                    _df_calc = pd.DataFrame(_rows_calc)

                    # Destacar linha 2km
                    st.dataframe(
                        _df_calc,
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        "★ 2km = referência central (100%). "
                        "**% Actual** = Watts reais / Watts 2km. "
                        "**Watts (obj)** = % Ideal × Watts 2km. "
                        "**Split** = fórmula Concept2: ((2.8/W)^(1/3)) × 500/86400 × 86400."
                    )

                    # Mini calculadora inversa: Split → Watts
                    st.markdown("**Converter Split → Watts:**")
                    _cs1, _cs2 = st.columns(2)
                    with _cs1:
                        _split_in = st.text_input("Split /500m (MM:SS.ss)", "2:00.00",
                                                   key="calc_split_in")
                    with _cs2:
                        _w_from_s = _w_from_split(_split_in)
                        if _w_from_s:
                            st.metric(f"Watts para {_split_in}",
                                      f"{_w_from_s:.1f} W",
                                      f"{_w_from_s/_w_2k*100:.1f}% do 2km")
                        else:
                            st.caption("Formato: M:SS.ss (ex: 1:58.50)")
                else:
                    st.info("Insere o valor de Watts do **2km** para calcular a tabela.")

            # ── Perfil Metabólico (Mader / Hauser / INSCYD / konaendu) ──────────
            _la_crossings = {}  # inicializar — preenchido no gráfico de lactato
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
            _mb_altura = 186  # fixo
            _mb1, _mb2, _mb3 = st.columns(3)
            with _mb1:
                _mb_idade = st.number_input("Idade (anos)", 15, 80, 40, 1,
                                            key="mb_idade_rank")
            with _mb2:
                _mb_peso = st.number_input("Peso (kg)", 40.0, 150.0, 85.0, 0.5,
                                           key="mb_peso_rank",
                                           help="Média do mês — sheet Wellness")
            with _mb3:
                _mb_bf = st.number_input("% Gordura", 5.0, 40.0, 14.0, 0.5,
                                         key="mb_bf_rank")

            # MMP3 e MMP5 — automáticos da sheet, sem input manual
            _mb_mmp3 = _mb_mmp5 = None
            if ac_full is not None and _col_mod is not None:
                _ac_mb = ac_full[ac_full[_col_mod] == modalidade].copy()
                _col_date_mb = next((c for c in ['date','Data']
                                     if c in _ac_mb.columns), None)
                # Usar parse_mmp e season best — igual ao Ranking SEE%
                for _mc_mb, _dur_mb in [('MMP3', 180), ('MMP5', 300)]:
                    if _mc_mb in _ac_mb.columns and _col_date_mb:
                        _s_mb = _ac_mb.sort_values(_col_date_mb, ascending=False)
                        for _, _rr_mb in _s_mb.iterrows():
                            _v = parse_mmp(str(_rr_mb[_mc_mb]))
                            if _v is not None and _v > 0:
                                if _mc_mb == 'MMP3': _mb_mmp3 = _v
                                else: _mb_mmp5 = _v
                                break

            _mb_mmp3_in = int(_mb_mmp3) if _mb_mmp3 else int(_calc_cp * 1.4)
            _mb_mmp5_in = int(_mb_mmp5) if _mb_mmp5 else int(_calc_cp * 1.25)
            st.caption(
                f"⚙️ Automático: Altura={_mb_altura}cm · "
                f"MMP3={_mb_mmp3_in}W · MMP5={_mb_mmp5_in}W · "
                f"Modalidade={modalidade}"
            )

            # ── VLamax automático (konaendu) — sem override manual ────────────
            # volRel_vlamax ≠ VolRel (0.40). volRel_vlamax é só para a fórmula VLamax.
            _mb_bmi        = _mb_peso / ((_mb_altura / 100) ** 2)
            _mb_workload   = (_mb_mmp3_in + _mb_mmp5_in) / 3
            _mb_volrel_vla = _mb_workload / _mb_peso  # só para VLamax

            # VO2max corrigido pelo VLamax (método INSCYD):
            # Subtrai contribuição glicolítica dos TT3 e TT6
            # W_glicolítica (3min) ≈ VLamax × VolRel × bw × 60 × 180 × 5.5 ml/mmol
            # Mas VLamax é desconhecido → iterar: estimar VO2max inicial → VLamax → VO2max corrigido
            # Constantes do modelo — seguir notebook Mader 1986 exactamente
            # Ks1 = 0.25² = 0.0625 (notebook), VolRel = 0.45 (notebook default)
            # Conversão W: Intensity = overall * bw / 12.5 (sem subtrair VO2rest)
            _mb_VolRel  = 0.45     # notebook: 0.45 (range 0.40-0.45)
            _mb_Ks1     = 0.25**2  # notebook: 0.0625
            _mb_Ks2     = 1.1**3   # notebook: 1.331
            _mb_Kel     = 2.0      # notebook: Kel=2 (Mader 1986), não 4 (Hauser)
            _mb_LAC_O2  = 0.01576
            _mb_Watt_O2 = 12.5     # notebook: 12.5 ml/min/W (não 11.685)

            # VO2max = média Hawley via MMP3 + MMP5 (mais robusto que via CP)
            _mb_vo2max = float(np.clip(
                ((_mb_mmp3_in / _mb_peso * 10.8 + 7) +
                 (_mb_mmp5_in / _mb_peso * 10.8 + 7)) / 2,
                20, 95))

            # VLamax via konaendu com VO2max Hawley (sem iteração INSCYD)
            _mb_mader  = (0.02049 / _mb_volrel_vla * _mb_vo2max * (_mb_bmi / 22)
                          * (1 + 0.000025 * _mb_idade - 0.0000001 * _mb_peso))
            _mb_sprint = (0.000004 / _mb_volrel_vla * (_pmax_global or _calc_cp * 4)
                          * (1 + 0.0000001 * _mb_idade - 0.0000001 * _mb_peso))
            _mb_vlamax = float(np.clip(_mb_mader + _mb_sprint, 0.05, 1.8))

            # Classificação do perfil fisiológico
            _mb_perfil = ('🏔️ Endurance puro'    if _mb_vlamax < 0.3 else
                          '⚖️ Endurance/Speed'   if _mb_vlamax < 0.5 else
                          '⚡ Speed/Power'        if _mb_vlamax < 0.8 else
                          '💥 Sprint/Anaeróbio')

            _mbc1, _mbc2, _mbc3 = st.columns(3)
            _mbc1.metric("VO2max (Hawley média MMP3+MMP5)",
                         f"{_mb_vo2max:.1f} ml/min/kg",
                         help=f"({_mb_mmp3_in}W + {_mb_mmp5_in}W) / 2 / {_mb_peso}kg × 10.8 + 7")
            _mbc2.metric("VLamax estimado",
                         f"{_mb_vlamax:.3f} mmol/L/s",
                         help="konaendu — Mader/Hauser via MMP3+MMP5+Pmax")
            _mbc3.metric("Perfil fisiológico", _mb_perfil)

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
                    _mb_overall * _mb_peso / _mb_Watt_O2)  # notebook: sem VO2rest

                _mb_argFM  = (int(np.argmax(-_mb_vLanet[:_mb_argAT]))
                              if _mb_argAT > 1 else 0)
                _mb_Fat    = np.maximum(0,
                    (-_mb_vLanet[:_mb_argAT]) * _mb_VolRel /
                    _mb_LAC_O2 * _mb_peso * 60 * 4.65 / 9.5 / 1000)
                # CHO: notebook usa vLass * constante (não constante fixa)
                _n_cho     = min(_mb_argAT + 400, len(_mb_Watts))
                _mb_CHO    = (_mb_vLass[:_n_cho] *
                              (_mb_peso * _mb_VolRel) * 60 / 1000 / 2 * 162.14)
                _mb_CHO_g  = float(_mb_CHO[_mb_argAT-1]) if _mb_argAT > 0 else float(_mb_CHO[0])

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
                        x=_mb_Watts[:_n_cho], y=_mb_CHO[:_n_cho],
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
                    _fig_sb.add_vline(x=_calc_cp, line_dash='dot',
                                      line_color='#A855F7', line_width=1.5,
                                      annotation_text=f"CP {_calc_cp}W",
                                      annotation_font_color='#A855F7',
                                      annotation_position="top left")
                    # PBP da sheet (se disponível)
                    if 'PBP' in (_hr_zones if '_hr_zones' in dir() else {}):
                        _pbp_w = _hr_zones['PBP']['med']
                        _fig_sb.add_vline(x=_pbp_w, line_dash='dot',
                                          line_color='#FF6B35', line_width=1.2,
                                          annotation_text=f"PBP {_pbp_w:.0f}W",
                                          annotation_font_color='#FF6B35',
                                          annotation_position="top right")
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

                    # Calcular watts de cruzamento com LT1 (~2mmol) e LT2 (~4mmol)
                    # Usar toda a curva disponível (não só abaixo do AT)
                    _la_crossings = {}
                    if _valid_la.any():
                        _W_la = _mb_W_below[_valid_la]
                        _C_la = _mb_CLass[_valid_la]
                        for _thr_la, _thr_name in [(2.0, 'LT1'), (4.0, 'LT2')]:
                            _cross_idx = np.where(np.diff(np.sign(_C_la - _thr_la)))[0]
                            if len(_cross_idx) > 0:
                                _i = _cross_idx[0]
                                if _i + 1 < len(_W_la):
                                    _w0, _w1 = float(_W_la[_i]), float(_W_la[_i+1])
                                    _c0, _c1 = float(_C_la[_i]), float(_C_la[_i+1])
                                    if _c1 != _c0:
                                        _w_cross = _w0 + (_thr_la-_c0)*(_w1-_w0)/(_c1-_c0)
                                        _la_crossings[_thr_name] = float(_w_cross)
                        # Se LT2 não encontrado: usar o valor máximo da curva
                        # (atleta de endurance com VLamax baixo pode não cruzar 4mmol antes do AT)
                        if 'LT2' not in _la_crossings and len(_C_la) > 0:
                            _max_la = float(np.nanmax(_C_la[np.isfinite(_C_la)]))
                            if _max_la >= 3.0:  # se chegou perto de 4mmol
                                _idx_max = int(np.nanargmax(_C_la))
                                _la_crossings['LT2'] = float(_W_la[_idx_max])

                    # Linhas horizontais LT1/LT2
                    for _y_la, _lbl_la, _col_la in [
                        (2.0, 'LT1 ~2 mmol', 'rgba(255,209,102,0.8)'),
                        (4.0, 'LT2 ~4 mmol', 'rgba(230,57,70,0.8)')]:
                        _fig_la.add_hline(y=_y_la, line_dash='dot',
                                          line_color=_col_la, line_width=1.2,
                                          annotation_text=_lbl_la,
                                          annotation_font_color=_col_la,
                                          annotation_position="right")

                    # Linhas verticais nos cruzamentos LT1/LT2
                    _la_colors = {'LT1': 'rgba(255,209,102,0.9)',
                                  'LT2': 'rgba(230,57,70,0.9)'}
                    for _thr_name, _w_cross in _la_crossings.items():
                        _fig_la.add_vline(
                            x=_w_cross, line_dash='dot',
                            line_color=_la_colors[_thr_name], line_width=1.5,
                            annotation_text=f"{_thr_name} {_w_cross:.0f}W",
                            annotation_font_color=_la_colors[_thr_name],
                            annotation_font_size=9,
                            annotation_position="top left",
                        )

                    # Linhas verticais: CP, FatMax, MLSS
                    for _vx_la, _vc_la, _vl_la in [
                        (_mb_W_FM,  '#00C896', f"FatMax {_mb_W_FM:.0f}W"),
                        (_mb_W_AT,  '#FFD166', f"MLSS {_mb_W_AT:.0f}W"),
                        (float(_calc_cp) if _calc_cp else None, '#A855F7', f"CP {_calc_cp}W"),
                    ]:
                        if _vx_la and _vx_la > 0 and _vx_la < float(_mb_W_AT) * 1.1:
                            _fig_la.add_vline(
                                x=_vx_la, line_dash='dot',
                                line_color=_vc_la, line_width=1.2,
                                annotation_text=_vl_la,
                                annotation_font_color=_vc_la,
                                annotation_font_size=9,
                                annotation_position="top right",
                            )

                    _fig_la.update_layout(
                        **_BASE_MB,
                        title=dict(text=f"Lactato Estacionário — {modalidade}"
                                   + (f" | LT1={_la_crossings.get('LT1',0):.0f}W"
                                      f" · LT2={_la_crossings.get('LT2',0):.0f}W"
                                      if _la_crossings else ""),
                                   font=dict(size=12)),
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

            # ── VO2max: média Hawley via MMP3 + MMP5 ─────────────────────────
            st.markdown("---")
            _vo2_mmp3 = _mb_mmp3_in / _mb_peso * 10.8 + 7
            _vo2_mmp5 = _mb_mmp5_in / _mb_peso * 10.8 + 7
            _vo2_media = (_vo2_mmp3 + _vo2_mmp5) / 2

            _vm1, _vm2, _vm3 = st.columns(3)
            _vm1.metric("VO2max via MMP3", f"{_vo2_mmp3:.1f} ml/min/kg",
                        f"({_mb_mmp3_in}W / {_mb_peso}kg) × 10.8 + 7")
            _vm2.metric("VO2max via MMP5", f"{_vo2_mmp5:.1f} ml/min/kg",
                        f"({_mb_mmp5_in}W / {_mb_peso}kg) × 10.8 + 7")
            _vm3.metric("VO2max (média Hawley)", f"{_vo2_media:.1f} ml/min/kg",
                        "MMP3 + MMP5 / 2 — Hawley & Noakes")
            st.caption(
                "Fórmula: VO2max ≈ (W/kg) × 10.8 + 7 (Hawley & Noakes). "
                f"MMP3={_mb_mmp3_in}W · MMP5={_mb_mmp5_in}W · Peso={_mb_peso}kg."
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
                            from plotly.subplots import make_subplots

                            # Paleta e labels
                            _hr_col_map = {
                                'HRVT1':     ('#60A5FA', 'LT1 / AeT'),
                                'HRVT1PLUS': ('#34D399', 'LT1+ Transição'),
                                'HRVTMSS':   ('#FBBF24', 'MLSS'),
                                'HRVT2':     ('#F87171', 'LT2 / AnT'),
                                'AeTHR':     ('#A78BFA', 'AeT HR'),
                            }

                            _hr_min = min((_hr_zones[k]['q25'] for k in _hr_keys), default=100) - 5
                            _hr_max = max((_hr_zones[k]['q75'] for k in _hr_keys), default=200) + 5

                            # Range X: 0 → Pvo2max*1.1 ou CP*1.5
                            _x_max_w = max(
                                _hr_zones.get('Pvo2max', {}).get('med', 0) * 1.1,
                                (_calc_cp or 200) * 1.5,
                                300,
                            )

                            # ── Subplot: cima=HR, baixo=Watts ─────────────────
                            # Calcular posição X (watts) para cada limiar via interpolação
                            _anchors = []
                            if _calc_cp and 'HRVTMSS' in _hr_zones:
                                _anchors.append((_hr_zones['HRVTMSS']['med'], float(_calc_cp)))
                            if 'PBP' in _hr_zones and 'HRVTMSS' in _hr_zones and 'HRVT2' in _hr_zones:
                                _hr_pbp_est = (_hr_zones['HRVTMSS']['med'] + _hr_zones['HRVT2']['med']) / 2
                                _anchors.append((_hr_pbp_est, float(_hr_zones['PBP']['med'])))
                            if 'Pvo2max' in _hr_zones and 'HRVT2' in _hr_zones:
                                _anchors.append((_hr_zones['HRVT2']['med'], float(_hr_zones['Pvo2max']['med'])))
                            _anchors.append((_hr_min + 5, 0.0))
                            _anchors = sorted(set(_anchors), key=lambda x: x[0])

                            def _hr_to_watts(hr_val):
                                if len(_anchors) < 2: return float(hr_val)
                                return float(np.interp(hr_val,
                                                       [a[0] for a in _anchors],
                                                       [a[1] for a in _anchors]))

                            _ordered_hr = sorted(
                                [(_hk, _hr_to_watts(_hr_zones[_hk]['med']))
                                 for _hk in _hr_keys if _hk in _hr_zones],
                                key=lambda t: t[1]
                            )
                            _line_x   = [t[1] for t in _ordered_hr]
                            _line_y   = [_hr_zones[t[0]]['med'] for t in _ordered_hr]
                            _line_q25 = [_hr_zones[t[0]]['q25'] for t in _ordered_hr]
                            _line_q75 = [_hr_zones[t[0]]['q75'] for t in _ordered_hr]
                            _line_lbl = [_hr_col_map.get(t[0], ('#AAAAAA', t[0]))[1]
                                         for t in _ordered_hr]
                            _line_col = [_hr_col_map.get(t[0], ('#AAAAAA', t[0]))[0]
                                         for t in _ordered_hr]

                            # ── GRÁFICO HR vs Potência ───────────────────────
                            # Eixo X = Potência (W) — partilhado por HR e Watts
                            # Círculos HR: posicionados no watts correspondente via âncoras
                            # Diamantes Watts: posicionados directamente no seu valor
                            _fig_hr = go.Figure()

                            # Zonas horizontais em HR
                            _zone_hr_defs = [
                                (None,      'HRVT1',   '#60A5FA', 0.06, 'Z1 Aeróbio'),
                                ('HRVT1',   'HRVTMSS', '#34D399', 0.06, 'Z2 Tempo'),
                                ('HRVTMSS', None,      '#FBBF24', 0.06, 'Z3 Limiar+'),
                            ]
                            for _zlo_k, _zhi_k, _zcl, _zop, _znm in _zone_hr_defs:
                                _y0z = (_hr_zones[_zlo_k]['med'] if _zlo_k and _zlo_k in _hr_zones else _hr_min)
                                _y1z = (_hr_zones[_zhi_k]['med'] if _zhi_k and _zhi_k in _hr_zones else _hr_max)
                                _fig_hr.add_hrect(y0=_y0z, y1=_y1z,
                                    fillcolor=_zcl, opacity=_zop, line_width=0)
                                _y0q = (_hr_zones[_zlo_k]['q25'] if _zlo_k and _zlo_k in _hr_zones else _hr_min)
                                _y1q = (_hr_zones[_zhi_k]['q75'] if _zhi_k and _zhi_k in _hr_zones else _hr_max)
                                _fig_hr.add_hrect(y0=_y0q, y1=_y1q,
                                    fillcolor=_zcl, opacity=0.09, line_width=0)

                            # Colectar pontos de Watts (eixo X real)
                            _w_refs_raw = []
                            if _mb_W_FM and _mb_W_FM > 10:
                                _w_refs_raw.append((float(_mb_W_FM), 'FatMax', '#00C896'))
                            if _mb_W_AT and _mb_W_AT > 10:
                                _w_refs_raw.append((float(_mb_W_AT), 'MLSS', '#FFD166'))
                            if _calc_cp:
                                _w_refs_raw.append((float(_calc_cp), 'CP', '#A855F7'))
                            if 'PBP' in _hr_zones:
                                _w_refs_raw.append((float(_hr_zones['PBP']['med']), 'PBP', '#FF6B35'))
                            if 'Pvo2max' in _hr_zones:
                                _w_refs_raw.append((float(_hr_zones['Pvo2max']['med']), 'Pvo2max', '#60A5FA'))
                            # Ordenar por Watts crescente (auto — sem sequência fixa)
                            _w_refs_raw.sort(key=lambda x: x[0])

                            # Range X baseado nos dados reais
                            _all_w_vals = [r[0] for r in _w_refs_raw]
                            if _all_w_vals:
                                _x_w_min = max(0, min(_all_w_vals) * 0.85)
                                _x_w_max = max(_all_w_vals) * 1.18
                            else:
                                _x_w_min, _x_w_max = 0, _x_max_w

                            # Âncoras HR↔Watts para interpolar posição X dos círculos HR
                            # Âncoras: HRVTMSS↔MLSS, HRVT2↔Pvo2max (ou CP), HRVT1↔FatMax
                            _hr_anchors = []  # (hr_bpm, watts)
                            if 'HRVTMSS' in _hr_zones and _mb_W_AT and _mb_W_AT > 0:
                                _hr_anchors.append((_hr_zones['HRVTMSS']['med'], float(_mb_W_AT)))
                            if 'HRVT2' in _hr_zones:
                                if 'Pvo2max' in _hr_zones:
                                    _hr_anchors.append((_hr_zones['HRVT2']['med'],
                                                        float(_hr_zones['Pvo2max']['med'])))
                                elif _calc_cp:
                                    _hr_anchors.append((_hr_zones['HRVT2']['med'], float(_calc_cp)))
                            if 'HRVT1' in _hr_zones and _mb_W_FM and _mb_W_FM > 0:
                                _hr_anchors.append((_hr_zones['HRVT1']['med'], float(_mb_W_FM)))
                            if 'AeTHR' in _hr_zones and _mb_W_FM and _mb_W_FM > 0:
                                _hr_anchors.append((_hr_zones['AeTHR']['med'], float(_mb_W_FM) * 0.95))
                            # Âncora de repouso
                            _hr_anchors.append((_hr_min, max(0, _x_w_min * 0.5)))
                            _hr_anchors.sort(key=lambda x: x[0])

                            def _hr_bpm_to_w(bpm):
                                if len(_hr_anchors) < 2: return _x_w_min
                                hrs = [a[0] for a in _hr_anchors]
                                ws  = [a[1] for a in _hr_anchors]
                                return float(np.interp(bpm, hrs, ws))

                            # Círculos HR — posição X = watts interpolado
                            _hr_order = ['HRVT1','AeTHR','HRVT1PLUS','HRVTMSS','HRVT2']
                            _hr_pts = []  # (x_watts, hr_bpm, color, label)
                            for _hk in _hr_order:
                                if _hk not in _hr_zones: continue
                                _hd = _hr_zones[_hk]
                                _hc, _hl = _hr_col_map.get(_hk, ('#AAAAAA', _hk))
                                _x_w = _hr_bpm_to_w(_hd['med'])
                                _hr_pts.append((_x_w, _hd['med'], _hc,
                                                f"{_hl}<br>{_hd['med']:.0f}bpm"))
                            _hr_pts.sort(key=lambda x: x[0])

                            if _hr_pts:
                                _hpx = [p[0] for p in _hr_pts]
                                _hpy = [p[1] for p in _hr_pts]
                                _hpc = [p[2] for p in _hr_pts]
                                _hpl = [p[3] for p in _hr_pts]
                                _fig_hr.add_trace(go.Scatter(
                                    x=_hpx, y=_hpy, mode='lines',
                                    line=dict(color='rgba(180,180,180,0.5)',width=1.5,dash='dot'),
                                    showlegend=False, hoverinfo='skip', yaxis='y'))
                                _fig_hr.add_trace(go.Scatter(
                                    x=_hpx, y=_hpy, mode='markers+text',
                                    marker=dict(size=12, color=_hpc,
                                                line=dict(width=2, color='white')),
                                    text=_hpl, textposition='top center',
                                    textfont=dict(size=8),
                                    showlegend=False, yaxis='y'))

                            # Diamantes Watts — posição X = valor real em Watts
                            if _w_refs_raw:
                                _wpx = [r[0] for r in _w_refs_raw]
                                _wpy = [r[0] for r in _w_refs_raw]  # Y2 = próprio valor W
                                _wpl = [f"{r[1]} {r[0]:.0f}W" for r in _w_refs_raw]
                                _wpc = [r[2] for r in _w_refs_raw]
                                _fig_hr.add_trace(go.Scatter(
                                    x=_wpx, y=_wpy, mode='lines',
                                    line=dict(color='rgba(150,100,220,0.4)',width=1.5,dash='dot'),
                                    showlegend=False, hoverinfo='skip', yaxis='y2'))
                                _fig_hr.add_trace(go.Scatter(
                                    x=_wpx, y=_wpy, mode='markers+text',
                                    marker=dict(size=12, color=_wpc, symbol='diamond',
                                                line=dict(width=2, color='white')),
                                    text=_wpl, textposition='bottom center',
                                    textfont=dict(size=8),
                                    showlegend=False, yaxis='y2'))

                            # LT1/LT2 horizontais em Watts (do lactato)
                            if '_la_crossings' in dir() and _la_crossings:
                                for _lt_nm, _lt_col in [('LT1','rgba(255,209,102,1.0)'),
                                                         ('LT2','rgba(230,57,70,1.0)')]:
                                    if _lt_nm in _la_crossings:
                                        _lt_w = _la_crossings[_lt_nm]
                                        _fig_hr.add_shape(type='line', yref='y2', xref='paper',
                                            x0=0, x1=1, y0=_lt_w, y1=_lt_w,
                                            line=dict(color=_lt_col, width=2, dash='dot'))
                                        _fig_hr.add_annotation(xref='paper', yref='y2',
                                            x=1.01, y=_lt_w, text=f"{_lt_nm} {_lt_w:.0f}W",
                                            showarrow=False, font=dict(size=9, color=_lt_col),
                                            xanchor='left')

                            # Linhas verticais nos valores de Watts
                            for _vw, _vlbl, _vc in _w_refs_raw:
                                _fig_hr.add_vline(x=_vw, line_color=_vc,
                                    line_width=1.5, line_dash='dot',
                                    annotation_text=_vlbl,
                                    annotation_font_color=_vc,
                                    annotation_font_size=8,
                                    annotation_position="top left")

                            _w_min_plot = max(0, min(_all_w_vals)*0.85) if _all_w_vals else 0
                            _w_max_plot = max(_all_w_vals)*1.15 if _all_w_vals else _x_max_w

                            _fig_hr.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=11), margin=dict(l=60,r=130,t=50,b=50),
                                height=520, showlegend=False,
                                title=dict(text=f"Limiares HR e Potência — {modalidade}  • HR  ◆ Watts",
                                           font=dict(size=13)),
                                xaxis=dict(
                                    title="Potência (W)",
                                    range=[_x_w_min, _x_w_max],
                                    showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                                    zeroline=False, tickfont=dict(size=10),
                                ),
                                yaxis=dict(
                                    title="HR (bpm)", range=[_hr_min, _hr_max],
                                    showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                                    zeroline=False, title_font_color="#AAAAAA",
                                    tickfont=dict(color='#AAAAAA')),
                                yaxis2=dict(
                                    title="Potência (W)",
                                    range=[_w_min_plot, _w_max_plot],
                                    overlaying='y', side='right',
                                    showgrid=False, zeroline=False,
                                    title_font_color="#A855F7",
                                    tickfont=dict(color='#A855F7', size=10)),
                            )
                            st.plotly_chart(_fig_hr, use_container_width=True)
                            st.caption(
                                "Eixo X = Potência (W). • Círculos = HR bpm (Y esq). "
                                "◆ Diamantes = Watts (Y dir, posição = valor real). "
                                "MLSS em HR e em Watts ficam na mesma posição X."
                            )

                            # ── GRÁFICO B: Referência de Potência ─────────────
                            _p_refs = []
                            if _calc_cp: _p_refs.append(('CP',_calc_cp,'#A855F7'))
                            if _mb_W_AT and _mb_W_AT>1: _p_refs.append(('MLSS',_mb_W_AT,'#FFD166'))
                            if _mb_W_FM and _mb_W_FM>1: _p_refs.append(('FatMax',_mb_W_FM,'#00C896'))
                            if 'PBP' in _hr_zones: _p_refs.append(('PBP',_hr_zones['PBP']['med'],'#FF6B35'))
                            if 'Pvo2max' in _hr_zones: _p_refs.append(('Pvo2max',_hr_zones['Pvo2max']['med'],'#60A5FA'))

                            if _p_refs:
                                _p_labels = [r[0] for r in _p_refs]
                                _p_vals   = [r[1] for r in _p_refs]
                                _p_colors = [r[2] for r in _p_refs]
                                _p_errs_lo,_p_errs_hi = [],[]
                                for _prl,_prv,_ in _p_refs:
                                    if _prl in ('PBP','Pvo2max') and _prl in _hr_zones:
                                        _pd = _hr_zones[_prl]
                                        _p_errs_lo.append(_prv-_pd['q25'])
                                        _p_errs_hi.append(_pd['q75']-_prv)
                                    else:
                                        _p_errs_lo.append(0); _p_errs_hi.append(0)

                                _fig_p = go.Figure()
                                _fig_p.add_trace(go.Bar(
                                    x=_p_vals, y=_p_labels, orientation='h',
                                    marker_color=_p_colors, marker_opacity=0.75,
                                    error_x=dict(type='data', array=_p_errs_hi,
                                                 arrayminus=_p_errs_lo, visible=True,
                                                 color='rgba(255,255,255,0.5)', thickness=2),
                                    text=[f"{v:.0f}W" for v in _p_vals],
                                    textposition='outside', textfont=dict(size=10),
                                    showlegend=False))
                                _fig_p.update_layout(
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(size=11), margin=dict(l=55,r=30,t=50,b=50),
                                    height=220, showlegend=False,
                                    title=dict(text="Referência de Potência", font=dict(size=13)),
                                    xaxis=dict(title="Potência (W)", range=[0,_x_max_w],
                                               showgrid=True,
                                               gridcolor="rgba(128,128,128,0.12)",
                                               zeroline=False, tickfont=dict(size=10)),
                                    yaxis=dict(showgrid=False, zeroline=False))
                                st.plotly_chart(_fig_p, use_container_width=True)


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

            # ── Botão Guardar no Google Drive DB ──────────────────────────────
            st.markdown("---")
            st.markdown("**💾 Guardar resultados**")

            if st.button("💾 Guardar no Drive", key="btn_save_cpmodel",
                         help="Guarda CP (todos modelos), perfil metabólico e limiares HR"):
                _saved_any = False
                _skipped   = []
                _errors    = []

                try:
                    from utils.drive_db import (
                        save_cp_result, save_metab_result, save_hr_thresholds,
                    )

                    # ── 1. CP — todos os modelos ──────────────────────────────
                    # Construir season bests com datas por MMP
                    _mmp_sb = {}
                    if ac_full is not None and _col_mod is not None:
                        _ac_sb = ac_full[ac_full[_col_mod] == modalidade].copy()
                        _col_d_sb = next((c for c in ['date','Data']
                                          if c in _ac_sb.columns), None)
                        for _mc_sb, _dur_sb in MMP_COLS.items():
                            if _mc_sb not in _ac_sb.columns or not _col_d_sb:
                                continue
                            _s_sb = (_ac_sb[[_mc_sb, _col_d_sb]]
                                     .dropna(subset=[_mc_sb])
                                     .sort_values(_col_d_sb, ascending=False))
                            if _s_sb.empty: continue
                            for _, _rr_sb in _s_sb.iterrows():
                                _v_sb = parse_mmp(str(_rr_sb[_mc_sb]))
                                if _v_sb:
                                    _mmp_sb[_mc_sb] = {
                                        'w':    _v_sb,
                                        'data': str(_rr_sb[_col_d_sb])[:10],
                                    }
                                    break

                    _r_cp = save_cp_result(
                        modalidade       = modalidade,
                        resultados_rank  = _results_rank,
                        all_mmp_pts      = _all_mmp_pts,
                        mmp_season_bests = _mmp_sb,
                        pmax             = _pmax_global,
                    )
                    if   _r_cp == "saved":   _saved_any = True
                    elif _r_cp == "skipped": _skipped.append("CP")
                    else:                    _errors.append("CP")

                    # ── 2. Metab ──────────────────────────────────────────────
                    if '_mb_W_AT' in dir() and _mb_W_AT and _mb_W_AT > 0:
                        # Zonas consolidadas
                        _z1_hr = _z2_hr = _z3_hr = ""
                        _z1_w  = _z2_w  = _z3_w  = ""
                        if '_hr_zones' in dir() and _hr_zones:
                            _h1 = _hr_zones.get('HRVT1',{}).get('med',0)
                            _hm = _hr_zones.get('HRVTMSS',{}).get('med',0)
                            _h2 = _hr_zones.get('HRVT2',{}).get('med',0)
                            if _h1: _z1_hr = f"< {_h1:.0f} bpm"
                            if _h1 and _hm: _z2_hr = f"{_h1:.0f} – {_hm:.0f} bpm"
                            if _hm: _z3_hr = f"> {_hm:.0f} bpm"
                        if _mb_W_FM: _z1_w = f"< {_mb_W_FM:.0f} W"
                        if _mb_W_FM and _mb_W_AT:
                            _z2_w = f"{_mb_W_FM:.0f} – {_mb_W_AT:.0f} W"
                        if _mb_W_AT: _z3_w = f"> {_mb_W_AT:.0f} W"

                        # CHO e Fat no MLSS
                        _cho_at_mlss = float(_mb_CHO[min(_mb_argAT, len(_mb_CHO)-1)]) \
                                       if '_mb_CHO' in dir() and len(_mb_CHO) > 0 else None
                        _fat_at_mlss = float(_mb_Fat[min(_mb_argAT-1, len(_mb_Fat)-1)]) \
                                       if '_mb_Fat' in dir() and _mb_argAT > 0 and len(_mb_Fat) > 0 else None

                        _vo2_mmp3 = _mb_mmp3_in / _mb_peso * 10.8 + 7
                        _vo2_mmp5 = _mb_mmp5_in / _mb_peso * 10.8 + 7
                        _vo2_media = (_vo2_mmp3 + _vo2_mmp5) / 2

                        # % VO2max via VO2ss (ml/min/kg no ponto)
                        _fatmax_pct_vo2 = float(_mb_VO2ss[_mb_argFM] / _vo2_media * 100) \
                                          if '_mb_VO2ss' in dir() and _mb_argFM < len(_mb_VO2ss) else 0
                        _mlss_pct_vo2   = float(_mb_VO2ss[_mb_argAT] / _vo2_media * 100) \
                                          if '_mb_VO2ss' in dir() and _mb_argAT < len(_mb_VO2ss) else 0

                        _r_m = save_metab_result(
                            modalidade       = modalidade,
                            vo2max_mmp3      = _vo2_mmp3,
                            vo2max_mmp5      = _vo2_mmp5,
                            vlamax           = _mb_vlamax,
                            perfil           = _mb_perfil if '_mb_perfil' in dir() else "",
                            fatmax_w         = _mb_W_FM or 0,
                            fatmax_pct_vo2max= _fatmax_pct_vo2,
                            mlss_w           = _mb_W_AT,
                            mlss_pct_vo2max  = _mlss_pct_vo2,
                            lt1_w            = _la_crossings.get('LT1'),
                            lt2_w            = _la_crossings.get('LT2'),
                            cp_watts         = float(_best_cp_val) if _best_cp_val else None,
                            fat_fatmax_g_h   = _mb_fat_FM if '_mb_fat_FM' in dir() else None,
                            cho_mlss_g_h     = _cho_at_mlss,
                            fat_mlss_g_h     = _fat_at_mlss,
                            glycogen_total   = _mb_gly_total if '_mb_gly_total' in dir() else None,
                            glycogen_liver   = 90.0,
                            glycogen_muscle  = _mb_muscle_kg * _mb_gly_kg
                                               if '_mb_muscle_kg' in dir() and '_mb_gly_kg' in dir() else None,
                            fitness_level    = _mb_fitness if '_mb_fitness' in dir() else None,
                            z1_hr=_z1_hr, z2_hr=_z2_hr, z3_hr=_z3_hr,
                            z1_w=_z1_w,   z2_w=_z2_w,   z3_w=_z3_w,
                            peso_kg          = _mb_peso,
                            altura_cm        = _mb_altura,
                            idade            = _mb_idade,
                        )
                        if   _r_m == "saved":   _saved_any = True
                        elif _r_m == "skipped": _skipped.append("Metab")
                        else:                   _errors.append("Metab")

                    # ── 3. HR Thresholds ──────────────────────────────────────
                    if '_hr_zones' in dir() and _hr_zones:
                        _n_sess = sum(
                            _hr_zones.get(k, {}).get('n', 0)
                            for k in ['HRVT1','HRVTMSS','HRVT2']
                        )
                        _r_h = save_hr_thresholds(
                            modalidade = modalidade,
                            hr_zones   = _hr_zones,
                            pbp_w      = _hr_zones.get('PBP', {}).get('med'),
                            pvo2max_w  = _hr_zones.get('Pvo2max', {}).get('med'),
                            n_sessions = _n_sess or None,
                        )
                        if   _r_h == "saved":   _saved_any = True
                        elif _r_h == "skipped": _skipped.append("HR")
                        else:                   _errors.append("HR")

                except ImportError:
                    st.error("❌ `utils/drive_db.py` não encontrado no GitHub.")
                except Exception as _save_err:
                    st.error(f"❌ Erro: {_save_err}")

                if _errors:
                    st.warning(f"⚠️ Falhou: {', '.join(_errors)}")
                if _saved_any:
                    st.success(
                        "✅ Guardado na Sheet" +
                        (f" | ignorados: {', '.join(_skipped)}" if _skipped else "")
                    )
                    # Tentar guardar também no .db do Drive (silencioso se falhar)
                    try:
                        from utils.drive_db import _upload_db, get_sqlite_conn, _find_db_id, _drive_svc
                        _conn_db = get_sqlite_conn()
                        _now_db  = datetime.now().strftime("%Y-%m-%d %H:%M")

                        # CP — todos os modelos (igual à Sheet)
                        if _results_rank:
                            _rank_db = 1
                            for _mn_db, _gr_db in sorted(_results_rank.items(),
                                                          key=lambda x: x[1].get('see_pct',999)):
                                _cp_db  = _gr_db.get('cp') or 0
                                _res_db = _gr_db.get('result',[None,None])
                                _wp_db  = _res_db[1] if _res_db and len(_res_db)>1 else None
                                _see_db = _gr_db.get('see_pct',0)
                                _combo_db = _gr_db.get('combo',[])
                                _pts_db = ",".join([f"{int(t//60)}min={p:.0f}W" for p,t in _combo_db])

                                # Verificar duplicado no SQLite
                                _dup = _conn_db.execute(
                                    "SELECT COUNT(*) FROM cp_results WHERE modalidade=? AND modelo=? "
                                    "AND ABS(cp_watts-?)<0.5 AND ABS(see_pct-?)<0.01",
                                    (modalidade, _mn_db, float(_cp_db), float(_see_db))
                                ).fetchone()[0]
                                if _dup > 0:
                                    _rank_db += 1
                                    continue

                                def _mdb(k):
                                    d = _mmp_sb.get(k,{})
                                    return d.get('w'), d.get('data','')
                                _conn_db.execute("""
                                    INSERT INTO cp_results
                                    (saved_at,modalidade,modelo,cp_watts,wp_joules,see_pct,
                                     pontos,rank_see,melhor_modelo,
                                     mmp1_w,mmp1_data,mmp3_w,mmp3_data,
                                     mmp5_w,mmp5_data,mmp12_w,mmp12_data,
                                     mmp20_w,mmp20_data,mmp60_w,mmp60_data,pmax)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                    (_now_db, modalidade, _mn_db,
                                     round(float(_cp_db),2),
                                     round(float(_wp_db),1) if _wp_db else None,
                                     round(float(_see_db),3), _pts_db,
                                     _rank_db, 1 if _rank_db==1 else 0,
                                     *_mdb('MMP1'), *_mdb('MMP3'),
                                     *_mdb('MMP5'), *_mdb('MMP12'),
                                     *_mdb('MMP20'), *_mdb('MMP60'),
                                     round(float(_pmax_global),1) if _pmax_global else None))
                                _rank_db += 1
                            _conn_db.commit()

                        _conn_db.close()

                        # Reabrir para metab e HR
                        _conn_db2 = get_sqlite_conn()

                        # Metab no SQLite
                        if '_mb_W_AT' in dir() and _mb_W_AT and _mb_W_AT > 0:
                            _dup_mb = _conn_db2.execute(
                                "SELECT COUNT(*) FROM metab_results WHERE modalidade=? "
                                "AND ABS(mlss_w-?)<0.5 AND ABS(vlamax-?)<0.001",
                                (modalidade, float(_mb_W_AT), float(_mb_vlamax))
                            ).fetchone()[0]
                            if _dup_mb == 0:
                                _vo2m3 = _mb_mmp3_in / _mb_peso * 10.8 + 7
                                _vo2m5 = _mb_mmp5_in / _mb_peso * 10.8 + 7
                                _vo2med = (_vo2m3 + _vo2m5) / 2
                                # % VO2max
                                _fm_pct = float(_mb_VO2ss[_mb_argFM]/_vo2med*100) \
                                          if '_mb_VO2ss' in dir() and _mb_argFM < len(_mb_VO2ss) else None
                                _ml_pct = float(_mb_VO2ss[_mb_argAT]/_vo2med*100) \
                                          if '_mb_VO2ss' in dir() and _mb_argAT < len(_mb_VO2ss) else None
                                # CP vs MLSS
                                _cp_vs_mlss_w   = round(float(_best_cp_val)-float(_mb_W_AT),1) \
                                                  if _best_cp_val else None
                                _cp_vs_mlss_pct = round((_cp_vs_mlss_w/float(_mb_W_AT))*100,1) \
                                                  if _cp_vs_mlss_w and _mb_W_AT else None
                                # CHO e Fat no MLSS
                                _cho_mlss = float(_mb_CHO[min(_mb_argAT, len(_mb_CHO)-1)]) \
                                            if '_mb_CHO' in dir() and len(_mb_CHO)>0 else None
                                _fat_mlss = float(_mb_Fat[min(_mb_argAT-1, len(_mb_Fat)-1)]) \
                                            if '_mb_Fat' in dir() and _mb_argAT>0 and len(_mb_Fat)>0 else None
                                # Glicogénio detalhado
                                _gly_liv = 90.0
                                _gly_mus = float(_mb_muscle_kg * _mb_gly_kg) \
                                           if '_mb_muscle_kg' in dir() and '_mb_gly_kg' in dir() else None
                                # W' validação 60min
                                _wp_val60 = None
                                if _mmp60_val and _best_cp_val and isinstance(_calc_wp, float):
                                    _wp_val60 = round((_mmp60_val - float(_best_cp_val)) * 3600, 0)

                                _conn_db2.execute("""
                                    INSERT INTO metab_results
                                    (saved_at,modalidade,vo2max_mmp3,vo2max_mmp5,vo2max_media,
                                     vlamax,perfil,
                                     fatmax_w,fatmax_pct_vo2,mlss_w,mlss_pct_vo2,
                                     lt1_w,lt2_w,cp_watts,
                                     cp_vs_mlss_w,cp_vs_mlss_pct,
                                     fat_fatmax_g_h,cho_mlss_g_h,fat_mlss_g_h,
                                     glycogen_total,glycogen_liver,glycogen_muscle,
                                     fitness_level,
                                     z1_hr,z2_hr,z3_hr,z1_w,z2_w,z3_w,
                                     peso_kg,altura_cm,idade)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                    (_now_db, modalidade,
                                     round(_vo2m3,2), round(_vo2m5,2), round(_vo2med,2),
                                     round(float(_mb_vlamax),4),
                                     _mb_perfil if '_mb_perfil' in dir() else "",
                                     round(float(_mb_W_FM),1) if _mb_W_FM else None,
                                     round(_fm_pct,1) if _fm_pct else None,
                                     round(float(_mb_W_AT),1),
                                     round(_ml_pct,1) if _ml_pct else None,
                                     _la_crossings.get('LT1'), _la_crossings.get('LT2'),
                                     round(float(_best_cp_val),1) if _best_cp_val else None,
                                     _cp_vs_mlss_w, _cp_vs_mlss_pct,
                                     round(float(_mb_fat_FM),1) if '_mb_fat_FM' in dir() and _mb_fat_FM else None,
                                     round(_cho_mlss,1) if _cho_mlss else None,
                                     round(_fat_mlss,1) if _fat_mlss else None,
                                     round(float(_mb_gly_total),0) if '_mb_gly_total' in dir() and _mb_gly_total else None,
                                     _gly_liv, round(_gly_mus,0) if _gly_mus else None,
                                     _mb_fitness if '_mb_fitness' in dir() else None,
                                     f"< {_hr_zones.get('HRVT1',{}).get('med',0):.0f} bpm" if '_hr_zones' in dir() and 'HRVT1' in _hr_zones else "",
                                     f"{_hr_zones.get('HRVT1',{}).get('med',0):.0f} – {_hr_zones.get('HRVTMSS',{}).get('med',0):.0f} bpm" if '_hr_zones' in dir() and 'HRVTMSS' in _hr_zones else "",
                                     f"> {_hr_zones.get('HRVTMSS',{}).get('med',0):.0f} bpm" if '_hr_zones' in dir() and 'HRVTMSS' in _hr_zones else "",
                                     f"< {_mb_W_FM:.0f} W" if _mb_W_FM else "",
                                     f"{_mb_W_FM:.0f} – {_mb_W_AT:.0f} W" if _mb_W_FM and _mb_W_AT else "",
                                     f"> {_mb_W_AT:.0f} W" if _mb_W_AT else "",
                                     _mb_peso, _mb_altura, _mb_idade))
                                _conn_db2.commit()

                        # HR Thresholds no SQLite
                        if '_hr_zones' in dir() and _hr_zones:
                            _mlss_hr = _hr_zones.get('HRVTMSS',{}).get('med')
                            if _mlss_hr:
                                _dup_hr = _conn_db2.execute(
                                    "SELECT COUNT(*) FROM hr_thresholds WHERE modalidade=? "
                                    "AND ABS(hrvtmss_bpm-?)<0.5",
                                    (modalidade, float(_mlss_hr))
                                ).fetchone()[0]
                                if _dup_hr == 0:
                                    def _ghz(k): return _hr_zones.get(k,{}).get('med')
                                    def _iqz(k): return f"[{_hr_zones.get(k,{}).get('q25',0):.0f}–{_hr_zones.get(k,{}).get('q75',0):.0f}]"
                                    _conn_db2.execute("""
                                        INSERT INTO hr_thresholds
                                        (saved_at,modalidade,
                                         hrvt1_bpm,hrvt1_iqr,hrvt1plus_bpm,hrvt1plus_iqr,
                                         hrvtmss_bpm,hrvtmss_iqr,hrvt2_bpm,hrvt2_iqr,
                                         aethr_bpm,aethr_iqr,pbp_w,pbp_iqr,pvo2max_w,pvo2max_iqr)
                                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                        (_now_db, modalidade,
                                         _ghz('HRVT1'),    _iqz('HRVT1'),
                                         _ghz('HRVT1PLUS'),_iqz('HRVT1PLUS'),
                                         _ghz('HRVTMSS'),  _iqz('HRVTMSS'),
                                         _ghz('HRVT2'),    _iqz('HRVT2'),
                                         _ghz('AeTHR'),    _iqz('AeTHR'),
                                         _hr_zones.get('PBP',{}).get('med'),
                                         f"[{_hr_zones.get('PBP',{}).get('q25',0):.0f}–{_hr_zones.get('PBP',{}).get('q75',0):.0f}]" if 'PBP' in _hr_zones else "",
                                         _hr_zones.get('Pvo2max',{}).get('med'),
                                         f"[{_hr_zones.get('Pvo2max',{}).get('q25',0):.0f}–{_hr_zones.get('Pvo2max',{}).get('q75',0):.0f}]" if 'Pvo2max' in _hr_zones else ""))
                                    _conn_db2.commit()

                        _conn_db2.close()

                        # Feedback do .db
                        _db_saved   = []
                        _db_skipped = []
                        _db_failed  = False

                        if _results_rank:
                            _ck = _conn_db2  # já fechado — reabrir para verificar
                            try:
                                _ck2 = get_sqlite_conn()
                                _n_cp = _ck2.execute(
                                    "SELECT COUNT(*) FROM cp_results WHERE modalidade=? AND saved_at=?",
                                    (modalidade, _now_db)).fetchone()[0]
                                _ck2.close()
                                if _n_cp > 0: _db_saved.append(f"CP ({_n_cp} modelos)")
                                else: _db_skipped.append("CP (já existe)")
                            except Exception: pass

                        if _upload_db():
                            _parts = []
                            if _db_saved:    _parts.append(f"✅ guardado: {', '.join(_db_saved)}")
                            if _db_skipped:  _parts.append(f"⏭️ ignorado: {', '.join(_db_skipped)}")
                            st.caption(f"📁 `.db` Drive — {' | '.join(_parts) if _parts else '✅ actualizado'}")
                        else:
                            st.caption("📁 `.db` — upload falhou (Sheet já guardou)")
                    except Exception:
                        pass  # silencioso — Sheet já guardou
                elif _skipped and not _errors:
                    st.info(f"ℹ️ Dados já existem — {', '.join(_skipped)}")
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
