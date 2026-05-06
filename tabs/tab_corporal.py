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
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def tab_corporal(dc, da_full, wc=None):
    """
    Aba Composição Corporal & Nutrição.
    dc      : DataFrame Consolidado_Comida (pré-processado) — nutrição + Peso
    da_full : DataFrame atividades completo (para correlações)
    wc      : DataFrame wellness (opcional) — fonte primária do BF% (coluna bf_pct = FAT do formulário)
    """
    st.header("🧬 Composição Corporal & Nutrição")

    if dc is None or len(dc) == 0:
        st.warning("Sem dados corporais. Verifica a aba 'Consolidado_Comida' na planilha.")
        return

    dc = dc.copy()
    dc['Data'] = pd.to_datetime(dc['Data'])

    # ── BF% — fonte primária: wc['bf_pct'] (FAT do formulário wellness) ──────
    # Se wc disponível e tem bf_pct, substitui dc['BF'] pelos dados do formulário.
    # Fallback: dc['BF'] original (Consolidado_Comida) — mantido se wc não tiver dados.
    # 'Fat' em dc continua a ser gramas de gordura alimentar — não é tocado.
    if wc is not None and len(wc) > 0 and 'bf_pct' in wc.columns:
        _wc_bf = wc[['Data', 'bf_pct']].copy()
        _wc_bf['Data'] = pd.to_datetime(_wc_bf['Data'])
        _wc_bf = _wc_bf.dropna(subset=['bf_pct'])
        _wc_bf = _wc_bf.rename(columns={'bf_pct': '_BF_wc'})
        if len(_wc_bf) > 0:
            # Merge por data — wellness como fonte principal
            dc = dc.merge(_wc_bf, on='Data', how='left')
            # Preenche BF: usa _BF_wc onde disponível, senão mantém dc['BF'] original
            if 'BF' not in dc.columns:
                dc['BF'] = np.nan
            dc['BF'] = dc['_BF_wc'].combine_first(dc['BF'])
            dc = dc.drop(columns=['_BF_wc'])

    # ── Limitar cada coluna até ao seu último registo válido ─────────────────
    _num_cols = ['Peso','BF','Calorias','Carb','Fat','Ptn','Net']
    for _col in _num_cols:
        if _col not in dc.columns: continue
        _last_valid = dc.loc[dc[_col].notna(), 'Data'].max()
        if pd.isna(_last_valid): continue
        dc.loc[dc['Data'] > _last_valid, _col] = np.nan
    _nc = [c for c in _num_cols if c in dc.columns]
    if _nc:
        dc = dc[dc[_nc].notna().any(axis=1)].copy()

    # ── KPIs cobertura ────────────────────────────────────────────────────────
    n_total = len(dc)
    n_peso  = dc['Peso'].notna().sum()     if 'Peso'     in dc.columns else 0
    n_cal   = dc['Calorias'].notna().sum() if 'Calorias' in dc.columns else 0
    n_bf    = dc['BF'].notna().sum()       if 'BF'       in dc.columns else 0
    d_min   = dc['Data'].min().strftime('%d/%m/%Y')
    d_max   = dc['Data'].max().strftime('%d/%m/%Y')
    n_dias  = (dc['Data'].max() - dc['Data'].min()).days + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Período",             f"{d_min} → {d_max}")
    c2.metric("⚖️ Peso (registos)",     f"{n_peso}/{n_total}")
    c3.metric("🔥 Calorias (registos)", f"{n_cal}/{n_total}")
    c4.metric("🫁 BF (registos)",       f"{n_bf}/{n_total}")
    st.caption(f"Dados esparsos: {n_dias} dias no período, {n_total} entradas. "
               "Dias sem registo são ignorados em médias e correlações.")
    st.markdown("---")

    # ── Preparação de dados — fora do expander para uso global ───────────────
    _dc_all = dc.copy()
    _dc_all['_w'] = _dc_all['Data'].dt.to_period('W')
    _wk = _dc_all.groupby('_w')[['Peso','BF','Calorias','Net']].mean()
    _wk.index = _wk.index.to_timestamp()
    _wk = _wk.sort_index()

    # Rolling 7 dias (remove ruído diário de água/glicogénio)
    _peso_r7 = (_dc_all.set_index('Data')['Peso'].resample('D').mean()
                .rolling(7, min_periods=3).mean() if 'Peso' in _dc_all.columns else pd.Series(dtype=float))
    _bf_r7   = (_dc_all.set_index('Data')['BF'].resample('D').mean()
                .rolling(7, min_periods=3).mean() if 'BF' in _dc_all.columns else pd.Series(dtype=float))
    _cal_r7  = (_dc_all.set_index('Data')['Calorias'].resample('D').mean()
                .rolling(7, min_periods=3).mean() if 'Calorias' in _dc_all.columns else pd.Series(dtype=float))

    # Peso/BF actual = mediana rolling 7d das últimas 2 semanas
    _peso_atual = float(_peso_r7.dropna().tail(14).median()) if len(_peso_r7.dropna()) >= 3 else None
    _bf_atual   = float(_bf_r7.dropna().tail(14).median())   if len(_bf_r7.dropna()) >= 3 else None

    # Calcular lag óptimo kcal → Peso/BF (testa 0, 3, 5, 7, 10, 14, 21 dias)
    def _calc_lag_optimo(var_r7_series):
        from scipy.stats import spearmanr as _sr
        if len(_cal_r7.dropna()) < 15 or len(var_r7_series.dropna()) < 15:
            return 7, None
        best_lag, best_r = 7, 0.0
        for lag in [0, 3, 5, 7, 10, 14, 21]:
            cal_lagged = _cal_r7.shift(lag)
            pair = pd.DataFrame({'cal': cal_lagged, 'var': var_r7_series}).dropna()
            if len(pair) < 15: continue
            r, p = _sr(pair['cal'].values, pair['var'].values)
            if p < 0.10 and abs(r) > abs(best_r):
                best_lag, best_r = lag, r
        return best_lag, best_r if best_r != 0.0 else None

    _lag_peso, _r_lag_peso = _calc_lag_optimo(_peso_r7)
    _lag_bf,   _r_lag_bf   = _calc_lag_optimo(_bf_r7)

    # ── KJ e Horas reais do atleta (últimas 8 semanas completas) ─────────
    # Usados como valores pré-preenchidos na calculadora — N=1 real
    _kj_semana_real    = 0
    _horas_total_real  = 0.0
    _horas_cicl_real   = 0.0
    _kj_semana_med_hist = 0  # mediana histórica para comparação
    if da_full is not None and len(da_full) > 0:
        try:
            _daf_pre = da_full.copy()
            _daf_pre['Data'] = pd.to_datetime(_daf_pre['Data'])
            _daf_pre['_w']   = _daf_pre['Data'].dt.to_period('W')
            _daf_pre['_mt']  = pd.to_numeric(_daf_pre['moving_time'], errors='coerce') / 3600
            _daf_cicl_pre = _daf_pre[_daf_pre['type'].apply(norm_tipo) != 'WeightTraining'].copy()
            if 'icu_joules' in _daf_cicl_pre.columns:
                _daf_cicl_pre['_kj'] = pd.to_numeric(_daf_cicl_pre['icu_joules'], errors='coerce') / 1000
            else:
                _daf_cicl_pre['_kj'] = np.nan
            _daf_wt_pre = _daf_pre[_daf_pre['type'].apply(norm_tipo) == 'WeightTraining']

            _kj_por_sem   = _daf_cicl_pre.groupby('_w')['_kj'].sum()
            _h_cicl_p_sem = _daf_cicl_pre.groupby('_w')['_mt'].sum()
            _h_wt_p_sem   = (_daf_wt_pre.groupby('_w')['_mt'].sum()
                              if len(_daf_wt_pre) > 0 else pd.Series(dtype=float))
            _h_tot_p_sem  = (_h_cicl_p_sem.add(_h_wt_p_sem, fill_value=0))

            _sem_atual_pre = pd.Timestamp.now().to_period('W')
            _sems_completas_pre = sorted(
                [s for s in _kj_por_sem.index if s < _sem_atual_pre],
                reverse=True)[:8]

            if _sems_completas_pre:
                _kj_semana_real   = int(_kj_por_sem[_sems_completas_pre].median())
                _horas_cicl_real  = float(_h_cicl_p_sem.reindex(_sems_completas_pre).median())
                _horas_total_real = float(_h_tot_p_sem.reindex(_sems_completas_pre).median())
                _kj_semana_med_hist = int(_kj_por_sem.median())
        except Exception:
            pass

    # ── Construir combined semanal (necessário para calculadora + correlações) ─
    _dc_all2 = dc.copy()
    _dc_all2['_w'] = _dc_all2['Data'].dt.to_period('W')
    corp_agg = _dc_all2.groupby('_w')[
        [c for c in ['Peso','BF','Calorias','Net','Carb','Fat','Ptn']
         if c in _dc_all2.columns]].mean()

    _macro_g  = ['Carb','Fat','Ptn']
    _kcal_map = {'Carb':4,'Fat':9,'Ptn':4}
    _has_macros = all(c in corp_agg.columns for c in _macro_g)
    if _has_macros:
        _kcal_tot = sum(corp_agg[m].fillna(0) * _kcal_map[m] for m in _macro_g)
        _kcal_tot = _kcal_tot.replace(0, np.nan)
        for m in _macro_g:
            corp_agg[f'pct_{m}'] = corp_agg[m] * _kcal_map[m] / _kcal_tot * 100

    train_agg = pd.DataFrame()
    if da_full is not None and len(da_full) > 0:
        _daf2 = da_full.copy()
        _daf2['Data'] = pd.to_datetime(_daf2['Data'])
        _daf2['_w']   = _daf2['Data'].dt.to_period('W')
        _daf2['_mt']  = pd.to_numeric(_daf2['moving_time'], errors='coerce') / 3600
        _df_cicl2 = _daf2[_daf2['type'].apply(norm_tipo) != 'WeightTraining'].copy()
        if 'icu_joules' in _df_cicl2.columns:
            _df_cicl2['_kj'] = pd.to_numeric(_df_cicl2['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _df_cicl2.columns:
            _df_cicl2['_kj'] = (pd.to_numeric(_df_cicl2['power_avg'], errors='coerce') *
                                 pd.to_numeric(_df_cicl2['moving_time'], errors='coerce') / 1000)
        else:
            _df_cicl2['_kj'] = np.nan
        _df_cicl2['_km'] = pd.to_numeric(
            _df_cicl2.get('distance', pd.Series(dtype=float)), errors='coerce') / 1000
        _df_wt2 = _daf2[_daf2['type'].apply(norm_tipo) == 'WeightTraining'].copy()
        train_agg = _df_cicl2.groupby('_w').agg(
            Horas_cicl=('_mt','sum'), KJ_sem=('_kj','sum'), KM_sem=('_km','sum'))
        if len(_df_wt2) > 0:
            train_agg = train_agg.join(
                _df_wt2.groupby('_w').agg(Horas_WT=('_mt','sum')), how='outer')
        else:
            train_agg['Horas_WT'] = np.nan
        train_agg['Horas_total'] = (train_agg['Horas_cicl'].fillna(0) +
                                     train_agg['Horas_WT'].fillna(0))
        train_agg[train_agg == 0] = np.nan

    combined = (corp_agg.join(train_agg, how='outer')
                if len(train_agg) > 0 else corp_agg.copy())
    combined.index = combined.index.to_timestamp()
    combined = combined.sort_index()

    # Lag de treino (semanas)
    _lag_treino_semanas = max(1, round((_lag_peso + _lag_bf) / 2 / 7))
    for _ct in ['KJ_sem','Horas_cicl','Horas_total','Horas_WT','KM_sem']:
        if _ct in combined.columns:
            combined[f'{_ct}_lag'] = combined[_ct].shift(_lag_treino_semanas)

    # ── Lookup histórico: semanas próximas dos alvos ──────────────────────
    # Encontra semanas onde Peso e BF estiveram próximos de qualquer alvo definido
    # Serve como estimativa directa N=1 sem necessidade de regressão separada
    def _lookup_historico(combined_df, peso_alvo, bf_alvo,
                          tol_peso=1.5, tol_bf=1.5):
        """Retorna semanas históricas onde Peso≈alvo E BF≈alvo."""
        if 'Peso' not in combined_df.columns or 'BF' not in combined_df.columns:
            return pd.DataFrame()
        d = combined_df[['Peso','BF','Calorias'] +
                         [c for c in ['Net','pct_Carb','pct_Fat','pct_Ptn',
                                       'KJ_sem','Horas_total','Horas_cicl']
                          if c in combined_df.columns]].dropna(subset=['Peso','BF'])
        mask = ((d['Peso'] - peso_alvo).abs() <= tol_peso) & \
               ((d['BF']   - bf_alvo).abs()   <= tol_bf)
        return d[mask]

    # (o lookup é chamado dentro da calculadora com os alvos definidos pelo user)
    # 🎯 CALCULADORA INTEGRADA — Peso + BF + Calorias + Macros
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("🎯 Calculadora de Metas — Peso, BF, Calorias e Macros", expanded=True):

        st.caption(
            "Valores actuais = **mediana rolling 7d das últimas 2 semanas** "
            "(remove flutuações de água/glicogénio). "
            f"Lag calórico detectado: Peso={_lag_peso}d | BF={_lag_bf}d.")

        # ── Inputs ────────────────────────────────────────────────────────
        _ci1, _ci2 = st.columns(2)
        with _ci1:
            st.markdown("**Estado actual (calculado dos dados)**")
            _lbm_atual = (_peso_atual * (1 - _bf_atual/100)
                          if _peso_atual and _bf_atual else None)
            _fm_atual  = (_peso_atual * (_bf_atual/100)
                          if _peso_atual and _bf_atual else None)
            if _peso_atual:
                st.metric("⚖️ Peso actual", f"{_peso_atual:.1f} kg")
            if _bf_atual:
                st.metric("🫁 BF actual", f"{_bf_atual:.1f}%")
            if _lbm_atual:
                st.metric("💪 Massa magra (LBM)", f"{_lbm_atual:.1f} kg")
            if _fm_atual:
                st.metric("🔴 Massa gorda", f"{_fm_atual:.1f} kg")
            # Treino real
            if _kj_semana_real > 0:
                st.metric("🚴 KJ/sem (mediana 8sem)", f"{_kj_semana_real} kJ",
                          delta=f"hist.: {_kj_semana_med_hist} kJ",
                          delta_color="off")
            if _horas_total_real > 0:
                st.metric("⏱️ Horas treino/sem (mediana 8sem)",
                          f"{_horas_total_real:.1f}h",
                          delta=f"cíclico: {_horas_cicl_real:.1f}h",
                          delta_color="off")

        with _ci2:
            st.markdown("**Definir alvos**")
            _peso_alvo = st.number_input(
                "⚖️ Peso-alvo (kg)",
                min_value=30.0, max_value=200.0,
                value=float(round(_peso_atual, 1)) if _peso_atual else 75.0,
                step=0.1, key="calc_peso_alvo")
            _bf_alvo = st.number_input(
                "🫁 BF-alvo (%)",
                min_value=3.0, max_value=50.0,
                value=float(round(_bf_atual, 1)) if _bf_atual else 15.0,
                step=0.1, key="calc_bf_alvo")

            st.markdown("**Ajustar treino planeado** *(pré-preenchido com os teus dados reais)*")
            _kj_semana = st.number_input(
                "🚴 KJ semanal cíclico",
                min_value=0, max_value=20000,
                value=int(_kj_semana_real) if _kj_semana_real > 0 else 0,
                step=100, key="calc_kj_sem",
                help=f"Mediana real das últimas 8 semanas: {_kj_semana_real} kJ. "
                     "Altera se planeas mudar o volume de treino.")
            _horas_total = st.number_input(
                "⏱️ Horas totais de treino/semana",
                min_value=0.0, max_value=40.0,
                value=float(round(_horas_total_real, 1)) if _horas_total_real > 0 else 0.0,
                step=0.5, key="calc_horas",
                help=f"Mediana real: {_horas_total_real:.1f}h (cíclico {_horas_cicl_real:.1f}h). "
                     "Usado para ajustar necessidade proteica.")

        # ── Cálculo de composição corporal alvo ───────────────────────────
        _lbm_alvo = _peso_alvo * (1 - _bf_alvo/100)
        _fm_alvo  = _peso_alvo * (_bf_alvo/100)

        _d_peso    = _peso_alvo - (_peso_atual or _peso_alvo)
        _d_bf_pp   = _bf_alvo  - (_bf_atual  or _bf_alvo)
        _d_lbm     = _lbm_alvo - (_lbm_atual or _lbm_alvo)
        _d_fm      = _fm_alvo  - (_fm_atual  or _fm_alvo)

        # Verificação de consistência
        _consistente = True
        _aviso_consistencia = ""
        if _peso_atual and _bf_atual:
            if _d_peso < 0 and _d_lbm > 0.5:
                _aviso_consistencia = (
                    f"⚠️ Meta implica perder {abs(_d_peso):.1f}kg de peso total "
                    f"mas GANHAR {_d_lbm:.1f}kg de massa magra — "
                    "requer perder {:.1f}kg de gordura. Possível mas exige periodização precisa.".format(
                        abs(_d_fm)))
            elif _d_peso > 0 and _d_fm < -0.5:
                _aviso_consistencia = (
                    f"⚠️ Meta implica ganhar {_d_peso:.1f}kg de peso "
                    f"mas PERDER {abs(_d_fm):.1f}kg de gordura — recomposição corporal. "
                    "Requer défice calórico moderado + alto treino de força.")

        st.markdown("---")
        st.markdown("**Composição corporal — actual vs alvo:**")
        _comp_rows = []
        for lbl, atual, alvo, delta, unid in [
            ("⚖️ Peso",       _peso_atual, _peso_alvo, _d_peso,  "kg"),
            ("🫁 BF",          _bf_atual,   _bf_alvo,   _d_bf_pp, "%"),
            ("💪 Massa magra", _lbm_atual,  _lbm_alvo,  _d_lbm,  "kg"),
            ("🔴 Massa gorda", _fm_atual,   _fm_alvo,   _d_fm,   "kg"),
        ]:
            if atual is None: continue
            emoji = "↗" if delta > 0.05 else ("↘" if delta < -0.05 else "→")
            _comp_rows.append({
                "": lbl,
                "Actual": f"{atual:.1f} {unid}",
                "Alvo":   f"{alvo:.1f} {unid}",
                "Δ":      f"{delta:+.1f} {unid} {emoji}",
            })
        if _comp_rows:
            st.dataframe(pd.DataFrame(_comp_rows), hide_index=True, use_container_width=True)
        if _aviso_consistencia:
            st.warning(_aviso_consistencia)

        # ── Estimativa calórica — Lookup histórico + Regressão ───────────────
        st.markdown("---")
        st.markdown("**Estimativa calórica alvo:**")
        st.caption(
            "**Método 1 (prioritário):** lookup das semanas históricas onde "
            "Peso e BF estiveram próximos dos alvos (±1.5kg / ±1.5pp). "
            "**Método 2 (fallback):** regressão Calorias~Peso e Calorias~BF com lag, "
            "combinados por R²."
        )

        def _cal_historicas_lag(target_col, lag_dias):
            from scipy.stats import linregress as _lr
            _d = _dc_all.set_index('Data')
            _cal_d = _d['Calorias'].resample('D').mean().rolling(7, min_periods=3).mean()
            _var_d = _d[target_col].resample('D').mean().rolling(7, min_periods=3).mean()
            cal_lagged = _cal_d.shift(lag_dias)
            pair = pd.DataFrame({'cal': cal_lagged, 'var': _var_d}).dropna()
            if len(pair) < 15: return None
            sl, ic, rv, pv, _ = _lr(pair['var'].values, pair['cal'].values)
            return {
                'slope': sl, 'intercept': ic, 'r2': rv**2, 'pv': pv,
                'cal_std': pair['cal'].std(), 'n': len(pair), 'lag': lag_dias,
                'pair': pair,
            }

        # ── Lookup histórico ──────────────────────────────────────────────
        _lookup = _lookup_historico(combined, _peso_alvo, _bf_alvo,
                                     tol_peso=1.5, tol_bf=1.5)
        # Alargar tolerância se poucos registos
        if len(_lookup) < 3:
            _lookup = _lookup_historico(combined, _peso_alvo, _bf_alvo,
                                         tol_peso=2.5, tol_bf=2.5)
            _tol_usada = "±2.5"
        else:
            _tol_usada = "±1.5"

        _cal_lookup = None
        _lookup_info = ""
        if len(_lookup) >= 3 and 'Calorias' in _lookup.columns:
            _cal_lookup     = float(_lookup['Calorias'].median())
            _cal_lookup_std = float(_lookup['Calorias'].std())
            _kj_lookup      = float(_lookup['KJ_sem'].median())  if 'KJ_sem'      in _lookup.columns else None
            _h_lookup       = float(_lookup['Horas_total'].median()) if 'Horas_total' in _lookup.columns else None
            _pct_c_lookup   = float(_lookup['pct_Carb'].median()) if 'pct_Carb' in _lookup.columns else None
            _pct_f_lookup   = float(_lookup['pct_Fat'].median())  if 'pct_Fat'  in _lookup.columns else None
            _pct_p_lookup   = float(_lookup['pct_Ptn'].median())  if 'pct_Ptn'  in _lookup.columns else None
            _lookup_info = (
                f"✅ **Lookup histórico**: {len(_lookup)} semanas com "
                f"Peso {_peso_alvo:.1f}±{_tol_usada}kg e BF {_bf_alvo:.1f}±{_tol_usada}%")
        else:
            _lookup_info = (
                f"⚠️ **Sem semanas históricas** com Peso≈{_peso_alvo:.1f}kg e BF≈{_bf_alvo:.1f}% "
                f"(tolerância ±2.5). A usar regressão como estimativa.")

        st.caption(_lookup_info)

        # ── Regressão (fallback ou complemento) ───────────────────────────
        _rel_peso = _cal_historicas_lag('Peso', _lag_peso) if 'Peso' in _dc_all.columns else None
        _rel_bf   = _cal_historicas_lag('BF',   _lag_bf)   if 'BF'   in _dc_all.columns else None

        _cal_base_peso = (_rel_peso['intercept'] + _rel_peso['slope'] * _peso_alvo
                          if _rel_peso else None)
        _cal_base_bf   = (_rel_bf['intercept']   + _rel_bf['slope']   * _bf_alvo
                          if _rel_bf   else None)

        if _cal_base_peso is not None and _cal_base_bf is not None:
            r2_p = _rel_peso['r2']; r2_b = _rel_bf['r2']
            soma_r2 = r2_p + r2_b if (r2_p + r2_b) > 0 else 1
            _cal_reg = (_cal_base_peso * r2_p + _cal_base_bf * r2_b) / soma_r2
            _cal_reg_std = float(np.mean([_rel_peso['cal_std'], _rel_bf['cal_std']]))
            _fonte_reg = f"Regressão ponderada R² (Peso={r2_p:.2f}, BF={r2_b:.2f})"
        elif _cal_base_peso is not None:
            _cal_reg = _cal_base_peso; _cal_reg_std = _rel_peso['cal_std']
            _fonte_reg = f"Regressão Peso (R²={_rel_peso['r2']:.2f})"
        elif _cal_base_bf is not None:
            _cal_reg = _cal_base_bf; _cal_reg_std = _rel_bf['cal_std']
            _fonte_reg = f"Regressão BF (R²={_rel_bf['r2']:.2f})"
        else:
            _cal_reg = None; _cal_reg_std = 300.0; _fonte_reg = "Sem dados"

        # ── Combinação final: lookup prioritário, regressão como âncora ──
        if _cal_lookup is not None and _cal_reg is not None:
            # Média ponderada: lookup (peso 2/3) + regressão (peso 1/3)
            _cal_central = (_cal_lookup * 2 + _cal_reg * 1) / 3
            _cal_std     = float(np.mean([_cal_lookup_std, _cal_reg_std]))
            _fonte_cal   = f"Lookup ({len(_lookup)} sem.) + {_fonte_reg}"
        elif _cal_lookup is not None:
            _cal_central = _cal_lookup; _cal_std = _cal_lookup_std
            _fonte_cal   = f"Lookup histórico ({len(_lookup)} semanas)"
        elif _cal_reg is not None:
            _cal_central = _cal_reg; _cal_std = _cal_reg_std
            _fonte_cal   = _fonte_reg
        else:
            _cal_central = None; _cal_std = 300.0; _fonte_cal = "Sem dados suficientes"

        # ── Ajuste calórico por treino — baseado nos dados reais N=1 ─────
        _ajuste_treino = 0.0
        _ajuste_treino_lbl = ""
        _eficiencia_real = 0.25  # default

        if _kj_semana > 0:
            # Tentar calcular eficiência metabólica real: slope(KJ_sem → Calorias)
            # slope em kcal/kJ → eficiência = slope × 0.239 (conversão J→cal)
            if da_full is not None and len(da_full) > 0 and 'Calorias' in _dc_all.columns:
                try:
                    from scipy.stats import linregress as _lr_kj
                    _daf_eff = da_full.copy()
                    _daf_eff['Data'] = pd.to_datetime(_daf_eff['Data'])
                    _daf_eff['_w'] = _daf_eff['Data'].dt.to_period('W')
                    if 'icu_joules' in _daf_eff.columns:
                        _daf_eff['_kj'] = pd.to_numeric(
                            _daf_eff['icu_joules'], errors='coerce') / 1000
                        _kj_w_eff = _daf_eff.groupby('_w')['_kj'].sum()
                        _dc_cal_w = _dc_all.copy()
                        _dc_cal_w['_w'] = _dc_cal_w['Data'].dt.to_period('W')
                        _cal_w_eff = _dc_cal_w.groupby('_w')['Calorias'].mean()
                        _pair_eff  = pd.DataFrame(
                            {'kj': _kj_w_eff, 'cal': _cal_w_eff}).dropna()
                        if len(_pair_eff) >= 10:
                            _sl_eff, _, _, _pv_eff, _ = _lr_kj(
                                _pair_eff['kj'].values, _pair_eff['cal'].values)
                            if _pv_eff < 0.15 and _sl_eff > 0:
                                # kcal_extra / kJ_extra → converte para eficiência
                                _eficiencia_real = max(0.15, min(0.40,
                                    float(_sl_eff * 0.239)))
                except Exception:
                    pass

            _gasto_diario = _kj_semana / _eficiencia_real * 0.239 / 7
            _ajuste_treino = _gasto_diario
            _ajuste_treino_lbl = (
                f"+{_ajuste_treino:.0f} kcal/dia "
                f"({_kj_semana} kJ/sem | efic. real={_eficiencia_real:.0%})")

        if _cal_central:
            _cal_com_treino = _cal_central + _ajuste_treino
            _cal_min = _cal_com_treino - _cal_std
            _cal_max = _cal_com_treino + _cal_std

            # Tempo estimado — usando ritmo real observado nos dados
            _ritmo_peso_real = None
            _ritmo_bf_real   = None
            if _rel_peso:
                _p = _rel_peso['pair']
                _rc = pd.DataFrame({
                    'dc': _p['cal'].diff(), 'dv': _p['var'].diff()
                }).dropna()
                _rc = _rc[_rc['dc'].abs() > 50]
                if len(_rc) >= 5:
                    _ritmo_peso_real = float(np.polyfit(_rc['dc'], _rc['dv'], 1)[0])
            if _rel_bf:
                _p = _rel_bf['pair']
                _rc = pd.DataFrame({
                    'dc': _p['cal'].diff(), 'dv': _p['var'].diff()
                }).dropna()
                _rc = _rc[_rc['dc'].abs() > 50]
                if len(_rc) >= 5:
                    _ritmo_bf_real = float(np.polyfit(_rc['dc'], _rc['dv'], 1)[0])

            _ajuste_cal = _cal_com_treino - (
                _rel_peso['intercept'] + _rel_peso['slope'] * (_peso_atual or _peso_alvo)
                if _rel_peso else _cal_com_treino)

            _tempo_peso_sem = None
            _tempo_bf_sem   = None
            if _ritmo_peso_real and abs(_ritmo_peso_real) > 1e-6 and _d_peso != 0:
                delta_dia = _ritmo_peso_real * _ajuste_cal
                if abs(delta_dia) > 0.001:
                    _tempo_peso_sem = abs(_d_peso / delta_dia) / 7
            if _ritmo_bf_real and abs(_ritmo_bf_real) > 1e-6 and _d_bf_pp != 0:
                delta_dia = _ritmo_bf_real * _ajuste_cal
                if abs(delta_dia) > 0.001:
                    _tempo_bf_sem = abs(_d_bf_pp / delta_dia) / 7

            _tempo_lbl = "—"
            if _tempo_peso_sem and _tempo_bf_sem:
                _tempo_max = max(_tempo_peso_sem, _tempo_bf_sem)
                _tempo_lbl = (f"~{_tempo_max:.0f} sem (Peso: {_tempo_peso_sem:.0f}sem | "
                              f"BF: {_tempo_bf_sem:.0f}sem — limitante: "
                              f"{'BF' if _tempo_bf_sem > _tempo_peso_sem else 'Peso'})")
            elif _tempo_peso_sem:
                _tempo_lbl = f"~{_tempo_peso_sem:.0f} semanas (pelo modelo Peso)"
            elif _tempo_bf_sem:
                _tempo_lbl = f"~{_tempo_bf_sem:.0f} semanas (pelo modelo BF)"
            else:
                # Fallback: energia por composição corporal
                _kcal_gordura = _d_fm * 7700  # kcal por kg de gordura
                _kcal_magra   = _d_lbm * 3000  # kcal por kg LBM (médio)
                _kcal_total   = _kcal_gordura + _kcal_magra
                if abs(_ajuste_cal) > 50:
                    _tempo_lbl = f"~{abs(_kcal_total / _ajuste_cal / 7):.0f} semanas (estimativa energética)"

            _rows_cal = [
                {'Métrica': f'🎯 Cal. alvo central ({_fonte_cal})',
                 'Valor': f"{_cal_com_treino:.0f} kcal/dia"},
                {'Métrica': '📊 Range (±1σ histórico)',
                 'Valor': f"{_cal_min:.0f} – {_cal_max:.0f} kcal/dia"},
                {'Métrica': '⏱️ Tempo estimado para atingir alvo',
                 'Valor': _tempo_lbl},
            ]
            if _ajuste_treino > 0:
                _rows_cal.insert(1, {
                    'Métrica': f'🚴 Ajuste por treino ({_kj_semana} kJ/sem)',
                    'Valor': f"+{_ajuste_treino:.0f} kcal/dia"
                })
            st.dataframe(pd.DataFrame(_rows_cal), hide_index=True, use_container_width=True)
        else:
            st.info("Sem dados suficientes para estimar calorias (mín. 15 dias com lag aplicado).")
            _cal_com_treino = 2000.0
            _cal_min = 1700.0
            _cal_max = 2300.0

        # ── Distribuição de macros recomendada ────────────────────────────
        st.markdown("---")
        st.markdown("**Distribuição de macros recomendada:**")
        st.caption(
            "Baseada nas % de macros encontradas nas semanas históricas com BF mais baixo (Q1). "
            "Proteína ajustada por peso corporal e horas de treino.")

        # % macros das semanas Q1 de BF (melhor composição observada)
        _pct_carb_hist = _pct_fat_hist = _pct_ptn_hist = None
        if _has_macros and 'pct_Carb' in combined.columns:
            _bf_col_c = 'BF' if 'BF' in combined.columns else None
            if _bf_col_c:
                _data_macros_hist = combined[
                    ['pct_Carb','pct_Fat','pct_Ptn','BF']].dropna()
                if len(_data_macros_hist) >= 12:
                    try:
                        _q1_bf_thresh = _data_macros_hist['BF'].quantile(0.25)
                        _q1_weeks = _data_macros_hist[
                            _data_macros_hist['BF'] <= _q1_bf_thresh]
                        if len(_q1_weeks) >= 3:
                            _pct_carb_hist = float(_q1_weeks['pct_Carb'].mean())
                            _pct_fat_hist  = float(_q1_weeks['pct_Fat'].mean())
                            _pct_ptn_hist  = float(_q1_weeks['pct_Ptn'].mean())
                    except Exception:
                        pass

        # Proteína: ajustada por LBM e treino
        # Base: 1.6g/kg LBM (sedentário) → 2.2g/kg LBM (alto volume treino)
        _ptn_por_kg = 1.6
        if _horas_total >= 10:
            _ptn_por_kg = 2.2
        elif _horas_total >= 6:
            _ptn_por_kg = 2.0
        elif _horas_total >= 3:
            _ptn_por_kg = 1.8

        _lbm_ref  = _lbm_alvo  # usar LBM alvo para proteína
        _ptn_g    = _lbm_ref * _ptn_por_kg
        _ptn_kcal = _ptn_g * 4

        # Se temos histórico Q1, usar % históricas como ponto de partida
        if _pct_ptn_hist is not None:
            # Balancear: 50% peso das % históricas + 50% do ajuste por treino
            _ptn_kcal_hist = (_pct_ptn_hist/100) * _cal_com_treino
            _ptn_kcal = (_ptn_kcal + _ptn_kcal_hist) / 2
            _ptn_g    = _ptn_kcal / 4

        _ptn_kcal = min(_ptn_kcal, _cal_com_treino * 0.40)  # cap 40% das calorias
        _ptn_g    = _ptn_kcal / 4

        # Gordura: mínimo fisiológico + histórico Q1
        _fat_kcal_min  = _peso_alvo * 0.8 * 9  # 0.8g/kg mínimo
        if _pct_fat_hist is not None:
            _fat_kcal = max(_fat_kcal_min, (_pct_fat_hist/100) * _cal_com_treino)
        else:
            _fat_kcal = max(_fat_kcal_min, _cal_com_treino * 0.25)
        _fat_g = _fat_kcal / 9

        # Carb: restante
        _carb_kcal = _cal_com_treino - _ptn_kcal - _fat_kcal
        _carb_kcal = max(_carb_kcal, 0)
        _carb_g    = _carb_kcal / 4

        # % reais
        _total_kcal_macros = _ptn_kcal + _fat_kcal + _carb_kcal
        _pct_ptn_calc  = _ptn_kcal  / _total_kcal_macros * 100 if _total_kcal_macros > 0 else 0
        _pct_fat_calc  = _fat_kcal  / _total_kcal_macros * 100 if _total_kcal_macros > 0 else 0
        _pct_carb_calc = _carb_kcal / _total_kcal_macros * 100 if _total_kcal_macros > 0 else 0

        _rows_macros = [
            {'Macro': '🥩 Proteína',
             'g/dia': f"{_ptn_g:.0f}g",
             'g/kg LBM': f"{_ptn_g/_lbm_ref:.1f} g/kg",
             'kcal': f"{_ptn_kcal:.0f} kcal",
             '% calorias': f"{_pct_ptn_calc:.0f}%",
             'Fonte': f"Ajustado por LBM ({_lbm_ref:.1f}kg) e treino ({_horas_total:.0f}h/sem)"},
            {'Macro': '🧈 Gordura',
             'g/dia': f"{_fat_g:.0f}g",
             'g/kg LBM': f"{_fat_g/_lbm_ref:.1f} g/kg",
             'kcal': f"{_fat_kcal:.0f} kcal",
             '% calorias': f"{_pct_fat_calc:.0f}%",
             'Fonte': ("Q1 BF histórico" if _pct_fat_hist else "25% calorias + mínimo fisiológico")},
            {'Macro': '🌾 Hidratos',
             'g/dia': f"{_carb_g:.0f}g",
             'g/kg LBM': f"{_carb_g/_lbm_ref:.1f} g/kg",
             'kcal': f"{_carb_kcal:.0f} kcal",
             '% calorias': f"{_pct_carb_calc:.0f}%",
             'Fonte': ("Q1 BF histórico (restante)" if _pct_carb_hist else "Restante das calorias")},
        ]
        st.dataframe(pd.DataFrame(_rows_macros), hide_index=True, use_container_width=True)

        # Range de macros (±15% de cada)
        st.caption(
            f"**Range sugerido (±1σ calórico → {_cal_min:.0f}–{_cal_max:.0f} kcal):** "
            f"Ptn {_ptn_g*(_cal_min/_cal_com_treino):.0f}–{_ptn_g*(_cal_max/_cal_com_treino):.0f}g | "
            f"Gordura {_fat_g*(_cal_min/_cal_com_treino):.0f}–{_fat_g*(_cal_max/_cal_com_treino):.0f}g | "
            f"Carb {_carb_g*(_cal_min/_cal_com_treino):.0f}–{_carb_g*(_cal_max/_cal_com_treino):.0f}g. "
            + (f"% históricas das semanas de BF baixo (Q1): "
               f"Carb {_pct_carb_hist:.0f}% | Gord {_pct_fat_hist:.0f}% | Ptn {_pct_ptn_hist:.0f}%."
               if _pct_carb_hist else
               "Histórico de macros insuficiente — usando referências fisiológicas."))

        if _pct_carb_hist or (_cal_lookup is not None and _pct_c_lookup):
            # Prioridade: lookup histórico (mesmas semanas do peso/bf alvo)
            # Fallback: Q1 BF histórico
            _pct_c_ref = _pct_c_lookup if (_cal_lookup and _pct_c_lookup) else _pct_carb_hist
            _pct_f_ref = _pct_f_lookup if (_cal_lookup and _pct_f_lookup) else _pct_fat_hist
            _pct_p_ref = _pct_p_lookup if (_cal_lookup and _pct_p_lookup) else _pct_ptn_hist
            _fonte_macro = ("lookup histórico" if (_cal_lookup and _pct_c_lookup)
                            else "Q1 BF histórico")
            st.info(
                f"📊 Nas semanas em que tiveste BF mais baixo (Q1 histórico), a distribuição típica foi: "
                f"**{_pct_carb_hist:.0f}% Carb | {_pct_fat_hist:.0f}% Gordura | {_pct_ptn_hist:.0f}% Proteína**. "
                "Estes valores foram incorporados no cálculo acima."
            )

        st.caption(
            f"⚠️ Lag calórico: Peso={_lag_peso}d | BF={_lag_bf}d. "
            "Proteína: 1.6–2.2g/kg LBM consoante volume de treino. "
            "Gordura: mínimo fisiológico 0.8g/kg. "
            "Calorias: média ponderada dos dois modelos (Peso e BF) pelo R² de cada um."
        )

    st.markdown("---")

    # ── Controlos: agrupamento + filtro de datas ──────────────────────────────
    st.subheader("⚙️ Filtros dos gráficos")
    _fc1, _fc2, _fc3, _fc4 = st.columns([1, 1, 1, 1])

    agrup_opts = {"Semana": "W", "Mês": "M", "Trimestre": "Q"}
    agrup_lbl  = _fc1.selectbox("Agrupar por", list(agrup_opts.keys()), key="corp_agrup")
    agrup_code = agrup_opts[agrup_lbl]
    roll_w     = _fc2.slider("Rolling (períodos)", 1, 12, 4, key="corp_roll")

    _d_min_hist = dc['Data'].min().date()
    _d_max_hist = dc['Data'].max().date()
    _tab_di = _fc3.date_input("Data início", value=_d_min_hist,
                               min_value=_d_min_hist, max_value=_d_max_hist,
                               key="corp_di")
    _tab_df = _fc4.date_input("Data fim", value=_d_max_hist,
                               min_value=_d_min_hist, max_value=_d_max_hist,
                               key="corp_df")

    dc_f = dc[(dc['Data'].dt.date >= _tab_di) & (dc['Data'].dt.date <= _tab_df)].copy()
    if len(dc_f) == 0:
        st.warning("Sem dados no período seleccionado.")
        return

    st.caption(f"📊 {len(dc_f)} registos | "
               f"{_tab_di.strftime('%d/%m/%Y')} → {_tab_df.strftime('%d/%m/%Y')}")

    # Agregação por período
    dc_f['_p'] = dc_f['Data'].dt.to_period(agrup_code)
    agg = dc_f.groupby('_p')[['Peso','BF','Calorias','Net','Carb','Fat','Ptn']].mean()
    agg.index = agg.index.to_timestamp()

    def _roll(series):
        return series.dropna().rolling(roll_w, min_periods=1).mean()

    # ── helper: formato tooltip de data por agrupamento ───────────────────────
    def _fmt_dates(idx):
        if agrup_code == 'W':
            return [d.strftime('Sem %d/%m/%y') for d in idx]
        elif agrup_code == 'M':
            return [d.strftime('%b %Y') for d in idx]
        else:
            return [f"Q{((d.month-1)//3)+1} {d.year}" for d in idx]

    # ── GRÁFICO 1: Peso + BF + Calorias ──────────────────────────────────────
    st.subheader("⚖️ Peso, % Gordura Corporal (BF) e Calorias")
    fig1 = go.Figure()

    if 'Peso' in agg.columns and agg['Peso'].notna().any():
        ps = agg['Peso'].dropna()
        pr = _roll(agg['Peso'])
        fig1.add_trace(go.Scatter(
            x=ps.index, y=ps.values,
            mode='markers', name='Peso (pontos)',
            marker=dict(color=CORES['azul'], size=6, opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>Peso: <b>%{y:.1f} kg</b><extra></extra>',
            yaxis='y1'))
        fig1.add_trace(go.Scatter(
            x=pr.index, y=pr.values,
            mode='lines', name=f'Peso roll({roll_w})',
            line=dict(color=CORES['azul'], width=2.5),
            hovertemplate='%{x|%d/%m/%Y}<br>Peso roll: <b>%{y:.1f} kg</b><extra></extra>',
            yaxis='y1'))

    if 'BF' in agg.columns and agg['BF'].notna().any():
        bs = agg['BF'].dropna()
        br = _roll(agg['BF'])
        fig1.add_trace(go.Scatter(
            x=bs.index, y=bs.values,
            mode='markers', name='BF% (pontos)',
            marker=dict(color=CORES['vermelho'], size=6, opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>BF: <b>%{y:.1f}%</b><extra></extra>',
            yaxis='y2'))
        fig1.add_trace(go.Scatter(
            x=br.index, y=br.values,
            mode='lines', name=f'BF roll({roll_w})',
            line=dict(color=CORES['vermelho'], width=2.5, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>BF roll: <b>%{y:.1f}%</b><extra></extra>',
            yaxis='y2'))

    if 'Calorias' in agg.columns and agg['Calorias'].notna().any():
        cs = agg['Calorias'].dropna()
        cr = _roll(agg['Calorias'])
        fig1.add_trace(go.Bar(
            x=cs.index, y=cs.values,
            name='Calorias',
            marker=dict(color=CORES['laranja'], opacity=0.25),
            hovertemplate='%{x|%d/%m/%Y}<br>Calorias: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y3'))
        fig1.add_trace(go.Scatter(
            x=cr.index, y=cr.values,
            mode='lines', name=f'Cal roll({roll_w})',
            line=dict(color=CORES['laranja'], width=1.8, dash='dot'),
            hovertemplate='%{x|%d/%m/%Y}<br>Cal roll: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y3'))

    fig1.update_layout(
        title=f'Peso, BF e Calorias — {agrup_lbl} | rolling={roll_w}',
        height=420,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        yaxis=dict(title='Peso (kg)', title_font=dict(color=CORES['azul']),
                   tickfont=dict(color=CORES['azul'])),
        yaxis2=dict(title='BF (%)', title_font=dict(color=CORES['vermelho']),
                    tickfont=dict(color=CORES['vermelho']),
                    overlaying='y', side='right'),
        yaxis3=dict(title='Calorias (kcal)', title_font=dict(color=CORES['laranja']),
                    tickfont=dict(color=CORES['laranja']),
                    overlaying='y', side='right', anchor='free', position=1.0,
                    showgrid=False),
        xaxis=dict(showgrid=True),
        margin=dict(r=80),
    )
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    # ── GRÁFICO 2: Calorias + Net ─────────────────────────────────────────────
    st.subheader("🔥 Calorias e Balanço Energético (Net)")
    fig2 = go.Figure()

    if 'Calorias' in agg.columns and agg['Calorias'].notna().any():
        cs2 = agg['Calorias'].dropna()
        cr2 = _roll(agg['Calorias'])
        fig2.add_trace(go.Bar(
            x=cs2.index, y=cs2.values,
            name='Calorias',
            marker=dict(color=CORES['laranja'], opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>Calorias: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y1'))
        fig2.add_trace(go.Scatter(
            x=cr2.index, y=cr2.values,
            mode='lines', name=f'Cal roll({roll_w})',
            line=dict(color=CORES['laranja'], width=2.5),
            hovertemplate='%{x|%d/%m/%Y}<br>Cal roll: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y1'))

    if 'Net' in agg.columns and agg['Net'].notna().any():
        nr = _roll(agg['Net'])
        fig2.add_trace(go.Scatter(
            x=nr.index, y=nr.values,
            mode='lines', name=f'Net roll({roll_w})',
            line=dict(color=CORES['roxo'], width=2.5, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>Net: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y2'))
        fig2.add_hline(y=0, line_dash='dot', line_color=CORES['cinza'],
                       annotation_text='Net = 0', annotation_position='bottom right',
                       yref='y2')

    fig2.update_layout(
        title=f'Calorias e Net — {agrup_lbl} | rolling={roll_w}',
        height=380,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        yaxis=dict(title='Calorias (kcal)', title_font=dict(color=CORES['laranja']),
                   tickfont=dict(color=CORES['laranja'])),
        yaxis2=dict(title='Net (kcal)', title_font=dict(color=CORES['roxo']),
                    tickfont=dict(color=CORES['roxo']),
                    overlaying='y', side='right'),
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    # ── GRÁFICO 3: Macros % stacked ───────────────────────────────────────────
    st.subheader("🥗 Distribuição de Macronutrientes (%)")
    macro_g = [c for c in ['Carb','Fat','Ptn'] if c in agg.columns]
    if macro_g:
        kcal_map = {'Carb': 4, 'Fat': 9, 'Ptn': 4}
        agg_k = agg[macro_g].copy()
        for m in macro_g:
            agg_k[m] = agg_k[m] * kcal_map.get(m, 4)
        total_k = agg_k[macro_g].sum(axis=1).replace(0, np.nan)
        agg_pct = (agg_k[macro_g].div(total_k, axis=0) * 100).dropna(how='all')

        if len(agg_pct) > 0:
            macro_cores = {'Carb': CORES['azul'], 'Fat': CORES['laranja'], 'Ptn': CORES['verde']}
            fig3 = go.Figure()
            for m in macro_g:
                vals = agg_pct[m].fillna(0)
                fig3.add_trace(go.Bar(
                    x=agg_pct.index, y=vals.values,
                    name=m,
                    marker=dict(color=macro_cores.get(m, CORES['cinza'])),
                    hovertemplate=f'%{{x|%d/%m/%Y}}<br>{m}: <b>%{{y:.1f}}%</b><extra></extra>',
                ))
            fig3.update_layout(
                barmode='stack',
                title=f'Macros % (Carb/Fat/Ptn em kcal) — {agrup_lbl}',
                height=360,
                hovermode='closest',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                yaxis=dict(title='% kcal de macros', range=[0, 115]),
            )
            fig3.add_hline(y=100, line_dash='dot', line_color=CORES['cinza'])
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    st.markdown("---")

    # ── GRÁFICO 4: Variação Peso e BF com bandas ─────────────────────────────
    st.subheader("📊 Variação de Peso e BF — bandas de ganho/perda esperado")
    st.caption("Peso (verde) e BF (azul) lado a lado. "
               "Bandas: limites calculados sobre o valor do período anterior. "
               "Peso: ±0.30%–0.70% | BF: ±0.25%–0.65%")

    dc_f2        = dc_f.copy()
    dc_f2['_p2'] = dc_f2['Data'].dt.to_period(agrup_code)
    agg_v        = dc_f2.groupby('_p2')[['Peso','BF']].mean()
    agg_v.index  = agg_v.index.to_timestamp()
    agg_v        = agg_v.sort_index()

    has_peso = 'Peso' in agg_v.columns and agg_v['Peso'].notna().sum() >= 3
    has_bf   = 'BF'   in agg_v.columns and agg_v['BF'].notna().sum()   >= 3

    if has_peso or has_bf:
        fig4 = go.Figure()

        if has_peso:
            peso_s     = agg_v['Peso'].dropna()
            peso_delta = peso_s.diff().dropna()
            prev_p     = peso_s.shift(1).dropna()
            xlbls      = _fmt_dates(peso_delta.index)

            fig4.add_trace(go.Bar(
                x=peso_delta.index, y=peso_delta.values,
                name='Δ Peso',
                marker=dict(color='#27ae60', opacity=0.85),
                width=1000*3600*24 * {'W':2,'M':8,'Q':20}[agrup_code],
                offset=-1000*3600*24 * {'W':1.2,'M':5,'Q':14}[agrup_code],
                customdata=np.stack([xlbls, peso_s.reindex(peso_delta.index).values], axis=-1),
                hovertemplate='%{customdata[0]}<br>Δ Peso: <b>%{y:+.2f} kg</b><br>Peso: %{customdata[1]:.1f} kg<extra></extra>',
                yaxis='y1'))

            for pct, col, lbl in [
                (0.0070, '#27ae60', '+max (+0.70%)'),
                (0.0030, '#82e0aa', '+min (+0.30%)'),
                (-0.0030, '#f1948a', '-min (-0.30%)'),
                (-0.0070, '#e74c3c', '-max (-0.70%)'),
            ]:
                fig4.add_trace(go.Scatter(
                    x=prev_p.index, y=prev_p.values * pct,
                    mode='lines', name=f'Peso {lbl}',
                    line=dict(color=col, width=1.5,
                              dash='dash' if abs(pct)==0.0070 else 'dot'),
                    hovertemplate=f'Limite Peso {lbl}: <b>%{{y:+.3f}} kg</b><extra></extra>',
                    yaxis='y1', showlegend=False))

            fig4.add_hline(y=0, line_color='black', line_width=0.8, opacity=0.5, yref='y')

        if has_bf:
            bf_s     = agg_v['BF'].dropna()
            bf_delta = bf_s.diff().dropna()
            prev_b   = bf_s.shift(1).dropna()
            xlbls_b  = _fmt_dates(bf_delta.index)

            fig4.add_trace(go.Bar(
                x=bf_delta.index, y=bf_delta.values,
                name='Δ BF',
                marker=dict(color='#2980b9', opacity=0.65),
                width=1000*3600*24 * {'W':2,'M':8,'Q':20}[agrup_code],
                offset=1000*3600*24 * {'W':0,'M':0,'Q':0}[agrup_code],
                customdata=np.stack([xlbls_b, bf_s.reindex(bf_delta.index).values], axis=-1),
                hovertemplate='%{customdata[0]}<br>Δ BF: <b>%{y:+.2f}%</b><br>BF: %{customdata[1]:.1f}%<extra></extra>',
                yaxis='y2'))

            for pct, col, lbl in [
                (0.0065, '#f39c12', '+max (+0.65%)'),
                (0.0025, '#fad7a0', '+min (+0.25%)'),
                (-0.0025, '#aed6f1', '-min (-0.25%)'),
                (-0.0065, '#2980b9', '-max (-0.65%)'),
            ]:
                fig4.add_trace(go.Scatter(
                    x=prev_b.index, y=prev_b.values * pct,
                    mode='lines', name=f'BF {lbl}',
                    line=dict(color=col, width=1.5,
                              dash='dash' if abs(pct)==0.0065 else 'dot'),
                    hovertemplate=f'Limite BF {lbl}: <b>%{{y:+.3f}}%</b><extra></extra>',
                    yaxis='y2', showlegend=False))

        fig4.update_layout(
            barmode='group',
            title=f'Variação Peso e BF por {agrup_lbl} — bandas de ganho/perda',
            height=450,
            hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            yaxis=dict(title='Δ Peso (kg)', title_font=dict(color='#27ae60'),
                       tickfont=dict(color='#27ae60'), zeroline=True),
            yaxis2=dict(title='Δ BF (%)', title_font=dict(color='#2980b9'),
                        tickfont=dict(color='#2980b9'),
                        overlaying='y', side='right', zeroline=True),
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})
    else:
        st.info("Dados insuficientes de Peso/BF para o gráfico de variação.")

    st.markdown("---")

    # ── CORRELAÇÕES ───────────────────────────────────────────────────────────
    st.subheader("🔗 Correlações entre variáveis corporais e de treino")
    st.caption(
        "Correlação de Spearman semanal sobre **todo o histórico**. "
        "Variáveis de treino (KJ, Horas) usam **lag óptimo** para evitar confounding "
        "de retenção de água pós-treino. "
        "Só mostra moderada (|r|≥0.30) ou forte (|r|≥0.50)."
    )

    def _sem_mdc(series, icc=0.90):
        s = series.dropna()
        if len(s) < 5: return None, None
        sem = s.std(ddof=1) * np.sqrt(1 - icc)
        return sem, sem * 1.96 * np.sqrt(2)

    def _forca(r):
        a = abs(r)
        if a >= 0.50: return "★★★ Forte"
        if a >= 0.30: return "★★ Moderada"
        return None

    # ── Definir predictors e targets ──────────────────────────────────────
    _lag_sufx = f'_lag' if _lag_treino_semanas > 0 else ''
    targets = [c for c in ['Peso','BF'] if c in combined.columns]
    predictors_all = [c for c in
        # Nutrição (sem lag — são consumidas antes do efeito)
        ['Calorias','Net','Carb','Fat','Ptn','pct_Carb','pct_Fat','pct_Ptn'] +
        # Treino com lag
        [f'KJ_sem{_lag_sufx}', f'Horas_cicl{_lag_sufx}',
         f'Horas_total{_lag_sufx}', f'Horas_WT{_lag_sufx}',
         f'KM_sem{_lag_sufx}']
        if c in combined.columns]

    from scipy.stats import spearmanr as _sc_sp
    corr_rows = []
    for tgt in targets:
        for pred in predictors_all:
            if pred == tgt: continue
            pair = combined[[tgt, pred]].dropna()
            if len(pair) < 8: continue
            r, pv = _sc_sp(pair[pred].values, pair[tgt].values)
            if pv >= 0.10: continue
            f = _forca(r)
            if f is None: continue
            # Label mais legível
            pred_lbl = (pred.replace('_lag', f' (lag {_lag_treino_semanas}sem)')
                            .replace('pct_', '% ')
                            .replace('KJ_sem', 'KJ semanal')
                            .replace('Horas_cicl', 'Horas cíclico')
                            .replace('Horas_total', 'Horas total')
                            .replace('Horas_WT', 'Horas musculação')
                            .replace('KM_sem', 'KM semanal'))
            corr_rows.append({
                'Alvo': tgt,
                'Preditor': pred_lbl,
                'r': f"{r:+.2f}", 'p': f"{pv:.3f}",
                'N': len(pair),
                'Força': f,
                'Direcção': (f"↗ mais → {tgt}↑" if r > 0 else f"↘ mais → {tgt}↓"),
            })

    if corr_rows:
        df_c = pd.DataFrame(corr_rows)
        df_c['_ar'] = df_c['r'].str.replace('+','',regex=False).astype(float).abs()
        df_c = (df_c.sort_values(['Alvo','_ar'], ascending=[True, False])
                    .drop_duplicates(subset=['Alvo','Preditor'], keep='first')
                    .drop(columns=['_ar']))
        # Mostrar por alvo
        for tgt in targets:
            sub_c = df_c[df_c['Alvo'] == tgt]
            if len(sub_c) == 0: continue
            st.markdown(f"**{tgt}** — correlações significativas:")
            st.dataframe(sub_c[['Preditor','r','p','N','Força','Direcção']],
                         hide_index=True, use_container_width=True)
    else:
        st.info("Sem correlações moderadas/fortes detectadas.")

    if _lag_treino_semanas > 0:
        st.caption(
            f"KJ/Horas com lag={_lag_treino_semanas} semana(s) — "
            "evita confounding de retenção de água/glicogénio imediata pós-treino. "
            "% Carb/Fat/Ptn = distribuição em kcal, independente do volume calórico total.")

    st.markdown("---")

    # ── MACROS: efeito independente das calorias ──────────────────────────
    st.subheader("🥗 Distribuição de Macronutrientes — efeito independente das calorias")
    st.caption(
        "A pergunta: **uma distribuição diferente de macros (mais Carb, mais Fat, mais Ptn) "
        "faz diferença na composição corporal, independentemente das calorias totais?** "
        "Método: correlação parcial controlando Calorias, e quartis de % macro por quartil de BF/Peso."
    )

    if _has_macros and all(c in combined.columns for c in ['pct_Carb','pct_Fat','pct_Ptn']):
        # ── Correlação parcial: macro% → Peso/BF controlando Calorias ────
        from scipy.stats import spearmanr as _sp_m, rankdata as _rd
        st.markdown("**Correlação parcial: % Macro → Peso/BF (controlando Calorias totais)**")
        st.caption(
            "Correlação parcial de Spearman: remove o efeito das calorias totais antes de "
            "calcular a correlação entre distribuição de macros e composição corporal. "
            "r próximo de 0 = os macros não explicam variação de BF/Peso além das calorias."
        )

        def _spearman_parcial(x, y, z):
            """Correlação parcial de Spearman entre x e y controlando z."""
            df_xyz = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
            if len(df_xyz) < 10: return None, None, len(df_xyz)
            # Rankar tudo
            rx = _rd(df_xyz['x'].values).astype(float)
            ry = _rd(df_xyz['y'].values).astype(float)
            rz = _rd(df_xyz['z'].values).astype(float)
            # Residualizar x e y em relação a z via OLS
            from numpy.linalg import lstsq as _ls
            X_z = np.column_stack([np.ones(len(rz)), rz])
            res_rx = rx - X_z @ _ls(X_z, rx, rcond=None)[0]
            res_ry = ry - X_z @ _ls(X_z, ry, rcond=None)[0]
            # Correlação de Pearson nos resíduos
            if np.std(res_rx) == 0 or np.std(res_ry) == 0: return None, None, len(df_xyz)
            r_p = float(np.corrcoef(res_rx, res_ry)[0,1])
            # p-value aproximado (Fisher z → t)
            n = len(df_xyz)
            t_stat = r_p * np.sqrt(n - 3) / np.sqrt(max(1 - r_p**2, 1e-10))
            from scipy.stats import t as _t_dist
            pv = float(2 * _t_dist.sf(abs(t_stat), df=n-3))
            return round(r_p, 3), round(pv, 4), n

        parc_rows = []
        for tgt_m in ['BF','Peso']:
            if tgt_m not in combined.columns: continue
            if 'Calorias' not in combined.columns: continue
            for macro_pct, macro_lbl in [
                ('pct_Carb', '% Carb'),
                ('pct_Fat',  '% Gordura'),
                ('pct_Ptn',  '% Proteína'),
            ]:
                if macro_pct not in combined.columns: continue
                r_p, pv_p, n_p = _spearman_parcial(
                    combined[macro_pct], combined[tgt_m], combined['Calorias'])
                if r_p is None: continue
                sig = '✅ p<0.05' if pv_p < 0.05 else '~ p<0.10' if pv_p < 0.10 else '✗ ns'
                interp = ''
                if pv_p < 0.10:
                    if tgt_m == 'BF':
                        interp = (f"↘ mais {macro_lbl.replace('% ','')} → menos gordura"
                                  if r_p < 0 else
                                  f"↗ mais {macro_lbl.replace('% ','')} → mais gordura")
                    else:
                        interp = (f"↘ mais {macro_lbl.replace('% ','')} → menos peso"
                                  if r_p < 0 else
                                  f"↗ mais {macro_lbl.replace('% ','')} → mais peso")
                parc_rows.append({
                    'Alvo': tgt_m, 'Macro': macro_lbl, 'N': n_p,
                    'r parcial': f"{r_p:+.3f}", 'Sig': sig,
                    'Interpretação': interp if interp else '→ sem efeito independente',
                })
        if parc_rows:
            st.dataframe(pd.DataFrame(parc_rows), hide_index=True, use_container_width=True)

        # ── Quartis de BF por distribuição de macros ──────────────────────
        st.markdown("**Distribuição de macros por quartil de BF (semanas com dados completos)**")
        st.caption(
            "Semanas agrupadas por quartil de BF. "
            "Mostra se há diferença sistemática na distribuição de macros "
            "entre semanas de BF mais baixo vs mais alto."
        )

        _cols_q = ['BF','Calorias','pct_Carb','pct_Fat','pct_Ptn']
        _data_q  = combined[_cols_q].dropna()
        if len(_data_q) >= 16:
            try:
                _data_q = _data_q.copy()
                _data_q['_qbf'] = pd.qcut(_data_q['BF'], q=4,
                                           labels=['Q1 BF baixo','Q2','Q3','Q4 BF alto'],
                                           duplicates='drop')
                rows_macro_q = []
                for ql in ['Q1 BF baixo','Q2','Q3','Q4 BF alto']:
                    g = _data_q[_data_q['_qbf'] == ql]
                    if len(g) < 3: continue
                    rows_macro_q.append({
                        'Quartil BF': ql,
                        'N semanas': len(g),
                        'BF médio (%)': f"{g['BF'].mean():.1f}",
                        'Cal média': f"{g['Calorias'].mean():.0f} kcal",
                        '% Carb': f"{g['pct_Carb'].mean():.0f}%",
                        '% Gordura': f"{g['pct_Fat'].mean():.0f}%",
                        '% Proteína': f"{g['pct_Ptn'].mean():.0f}%",
                    })
                if rows_macro_q:
                    st.dataframe(pd.DataFrame(rows_macro_q), hide_index=True, use_container_width=True)
                    # Nota de interpretação
                    _q1 = _data_q[_data_q['_qbf']=='Q1 BF baixo']
                    _q4 = _data_q[_data_q['_qbf']=='Q4 BF alto']
                    if len(_q1) >= 3 and len(_q4) >= 3:
                        _d_ptn = _q1['pct_Ptn'].mean() - _q4['pct_Ptn'].mean()
                        _d_carb = _q1['pct_Carb'].mean() - _q4['pct_Carb'].mean()
                        _d_fat  = _q1['pct_Fat'].mean() - _q4['pct_Fat'].mean()
                        _d_cal  = _q1['Calorias'].mean() - _q4['Calorias'].mean()
                        st.info(
                            f"Q1 (BF mais baixo) vs Q4 (BF mais alto): "
                            f"Proteína {_d_ptn:+.0f}pp | Carb {_d_carb:+.0f}pp | "
                            f"Gordura {_d_fat:+.0f}pp | Calorias {_d_cal:+.0f} kcal. "
                            "Diferença nas calorias indica que parte do efeito pode ser calórico. "
                            "A correlação parcial acima isola o efeito independente."
                        )
            except Exception:
                st.info("Dados insuficientes para quartis de macros.")
        else:
            st.info(f"Dados insuficientes para análise de macros por quartil (N={len(_data_q)}, mín 16).")
    else:
        st.info("Sem dados de macronutrientes suficientes para análise de distribuição.")

    st.markdown("---")

    # ── TABELAS: base calórica por quartil de Peso e BF ───────────────────────
    st.subheader("📊 Base calórica por quartil de Peso e BF")
    st.caption(
        "Calorias com **lag aplicado** (calorias de N dias antes correspondem ao peso actual). "
        "Estratificado por carga de treino (kJ baixo vs alto) para remover confounding. "
        "Semanas com kJ alto têm peso transitoriamente mais alto por retenção de glicogénio/água."
    )

    # _cal_r7, _peso_r7, _bf_r7, _lag_peso, _lag_bf já calculados acima

    # kJ semanal para estratificação
    _kj_semanal = None
    if da_full is not None and len(da_full) > 0:
        _daf = da_full.copy()
        _daf['Data'] = pd.to_datetime(_daf['Data'])
        if 'icu_joules' in _daf.columns:
            _daf['_kj'] = pd.to_numeric(_daf['icu_joules'], errors='coerce') / 1000
        else:
            _daf['_kj'] = np.nan
        _daf['_w'] = _daf['Data'].dt.to_period('W')
        _kj_semanal = _daf.groupby('_w')['_kj'].sum()
        _kj_semanal.index = _kj_semanal.index.to_timestamp()
        # Normalizar: baixo = Q1+Q2, alto = Q3+Q4
        _kj_median = float(_kj_semanal.median()) if len(_kj_semanal) > 0 else None

    for alvo_q, alvo_lbl, unid, lag_d, var_r7 in [
        ('Peso', 'Peso', 'kg', _lag_peso, _peso_r7),
        ('BF',   'BF',   '%',  _lag_bf,   _bf_r7),
    ]:
        if alvo_q not in _dc_all.columns or 'Calorias' not in _dc_all.columns: continue
        if len(var_r7.dropna()) < 15 or len(_cal_r7.dropna()) < 15: continue

        # Calorias lagged
        cal_lagged = _cal_r7.shift(lag_d)

        # Construir dataframe diário para análise
        _df_lag = pd.DataFrame({
            'var': var_r7,
            'cal_lag': cal_lagged,
        }).dropna()
        if len(_df_lag) < 20:
            st.caption(f"Poucos dados para quartis de {alvo_lbl} com lag={lag_d}d ({len(_df_lag)} dias).")
            continue

        # Adicionar kJ semanal para estratificação
        if _kj_semanal is not None:
            _df_lag['_w'] = _df_lag.index.to_period('W').to_timestamp()
            _df_lag = _df_lag.merge(_kj_semanal.rename('kj_sem').reset_index()
                                    .rename(columns={'index':'_w'}),
                                    on='_w', how='left')
        else:
            _df_lag['kj_sem'] = np.nan

        # Quartis de var
        try:
            _df_lag['_q'], _bins = pd.qcut(
                _df_lag['var'], q=4,
                labels=['Q1 (baixo)','Q2','Q3','Q4 (alto)'],
                retbins=True, duplicates='drop')
        except Exception:
            continue

        # Tabela principal (todos os dados com lag)
        rows_q = []
        for ql in ['Q1 (baixo)','Q2','Q3','Q4 (alto)']:
            g = _df_lag[_df_lag['_q'] == ql]
            if len(g) < 3: continue
            cv = g['cal_lag']; av = g['var']
            _, mdc_c = _sem_mdc(cv) if len(cv) >= 5 else (None, None)

            row = {
                f'Quartil {alvo_lbl}':        ql,
                f'Range {alvo_lbl} ({unid})': f"{av.min():.1f}–{av.max():.1f}",
                f'Média {alvo_lbl}':          f"{av.mean():.1f} {unid}",
                'N dias':                     len(g),
                'Cal média (lag)':            f"{cv.mean():.0f} kcal",
                'Cal mediana (lag)':          f"{cv.median():.0f} kcal",
                'Cal Q1–Q3':                  f"{cv.quantile(0.25):.0f}–{cv.quantile(0.75):.0f} kcal",
                'MDC Cal':                    f"±{mdc_c:.0f} kcal" if mdc_c else '—',
            }
            rows_q.append(row)

        if rows_q:
            st.markdown(f"**Quartis de {alvo_lbl} — base calórica (lag={lag_d}d)**")
            st.dataframe(pd.DataFrame(rows_q), width="stretch", hide_index=True)

        # Tabela estratificada por kJ (se disponível)
        if _kj_semanal is not None and _kj_median is not None and 'kj_sem' in _df_lag.columns:
            _df_lag['_treino'] = np.where(_df_lag['kj_sem'] > _kj_median, 'Alto treino', 'Baixo treino')
            rows_strat = []
            for treino_g in ['Baixo treino', 'Alto treino']:
                for ql in ['Q1 (baixo)', 'Q2', 'Q3', 'Q4 (alto)']:
                    g2 = _df_lag[(_df_lag['_q'] == ql) & (_df_lag['_treino'] == treino_g)]
                    if len(g2) < 3: continue
                    cv2 = g2['cal_lag']
                    rows_strat.append({
                        'Treino': treino_g,
                        f'Quartil {alvo_lbl}': ql,
                        f'Média {alvo_lbl}': f"{g2['var'].mean():.1f} {unid}",
                        'N': len(g2),
                        'Cal média (lag)': f"{cv2.mean():.0f} kcal",
                        'Cal mediana': f"{cv2.median():.0f} kcal",
                    })
            if rows_strat:
                with st.expander(f"📊 Estratificado por carga de treino — {alvo_lbl}"):
                    st.caption(
                        "Baixo treino = kJ semanal abaixo da mediana histórica. "
                        "Alto treino = acima. Peso alto em semanas de alto treino pode ser "
                        "retenção de água/glicogénio — não indica que calorias são incorrectas.")
                    st.dataframe(pd.DataFrame(rows_strat), width="stretch", hide_index=True)

        st.markdown("")

    # ── Análise de lag calórico ────────────────────────────────────────────────
    st.subheader("⏱️ Análise de Lag Calórico")
    st.caption(
        "r Spearman entre calorias rolling 7d (com lag de 0 a 21 dias) e Peso/BF rolling 7d. "
        "O lag óptimo é onde a correlação é mais forte — mostra com quantos dias o teu peso "
        "responde a mudanças calóricas.")

    from scipy.stats import spearmanr as _sr_lag

    if len(_cal_r7.dropna()) >= 20:
        _lag_rows = []
        for lag in [0, 3, 5, 7, 10, 14, 21]:
            cal_lagged = _cal_r7.shift(lag)
            for var_col, var_r, var_lbl in [('Peso', _peso_r7, 'Peso'), ('BF', _bf_r7, 'BF')]:
                pair = pd.DataFrame({'cal': cal_lagged, 'var': var_r}).dropna()
                if len(pair) < 15: continue
                r, p = _sr_lag(pair['cal'].values, pair['var'].values)
                sig = '✅ p<0.05' if p < 0.05 else '~ p<0.10' if p < 0.10 else '✗ ns'
                _lag_rows.append({
                    'Lag (dias)': lag,
                    'Variável': var_lbl,
                    'N': len(pair),
                    'r Spearman': f"{r:+.3f}",
                    'Sig': sig,
                    'Força': '🔴 forte' if abs(r)>=0.5 else '🟡 moderada' if abs(r)>=0.3 else '🟢 fraca',
                    '★ Óptimo': '⭐' if (
                        (var_lbl == 'Peso' and lag == _lag_peso) or
                        (var_lbl == 'BF'   and lag == _lag_bf)
                    ) else '',
                })
        if _lag_rows:
            st.dataframe(pd.DataFrame(_lag_rows), hide_index=True, use_container_width=True)
            st.info(
                f"⭐ Lag óptimo detectado: **Peso = {_lag_peso} dias** | **BF = {_lag_bf} dias**. "
                "Isto significa que uma mudança calórica hoje só aparece de forma consistente "
                f"no peso após {_lag_peso} dias. Semanas com alto kJ podem ter lag diferente por "
                "retenção de água/glicogénio."
            )
