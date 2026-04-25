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

def tab_visao_geral(dw, da, di, df_, da_full=None, wc_full=None, dc=None):
    st.header("📊 Visão Geral")

    # ── KPIs — mês corrente fixo (independente do filtro global) ──
    _kpi_hoje    = pd.Timestamp.now().normalize()
    _kpi_mc_ini  = _kpi_hoje.replace(day=1)
    _kpi_mp_fim  = _kpi_mc_ini - pd.Timedelta(days=1)
    _kpi_mp_ini  = _kpi_mp_fim.replace(day=1)

    # Atividades do mês corrente e mês anterior (usa da_full se disponível)
    _src_kpi = da_full if da_full is not None and len(da_full) > 0 else da
    if len(_src_kpi) > 0 and 'Data' in _src_kpi.columns:
        _kpi_df = _src_kpi.copy()
        _kpi_df['Data'] = pd.to_datetime(_kpi_df['Data'])
        _kpi_df = _kpi_df[_kpi_df['type'].apply(norm_tipo) != 'WeightTraining']
        _kpi_mc = _kpi_df[_kpi_df['Data'] >= _kpi_mc_ini]
        _kpi_mp = _kpi_df[(_kpi_df['Data'] >= _kpi_mp_ini) & (_kpi_df['Data'] <= _kpi_mp_fim)]
    else:
        _kpi_mc = _kpi_mp = pd.DataFrame()

    _sess_mc  = len(_kpi_mc)
    _sess_mp  = len(_kpi_mp)
    _horas_mc = _kpi_mc['moving_time'].sum() / 3600 if 'moving_time' in _kpi_mc.columns and len(_kpi_mc) > 0 else 0
    _horas_mp = _kpi_mp['moving_time'].sum() / 3600 if 'moving_time' in _kpi_mp.columns and len(_kpi_mp) > 0 else 0

    def _delta_pct(vc, vp):
        if not vp or vp == 0: return None
        p = (vc - vp) / vp * 100
        return f"{p:+.0f}% vs mês ant."

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏋️ Sessões (mês)", f"{_sess_mc}",
              _delta_pct(_sess_mc, _sess_mp))
    c2.metric("⏱️ Horas (mês)", fmt_dur(_horas_mc) if _horas_mc else "—",
              _delta_pct(_horas_mc, _horas_mp))
    # HRV e RHR — média mês corrente vs mês anterior (de wc_full)
    _src_wkpi = wc_full if wc_full is not None and len(wc_full) > 0 else dw
    if _src_wkpi is not None and len(_src_wkpi) > 0 and 'hrv' in _src_wkpi.columns:
        _wkpi = _src_wkpi.copy()
        _wkpi['Data'] = pd.to_datetime(_wkpi['Data'])
        _hrv_mc = _wkpi[_wkpi['Data'] >= _kpi_mc_ini]['hrv'].dropna()
        _hrv_mp = _wkpi[(_wkpi['Data'] >= _kpi_mp_ini) & (_wkpi['Data'] <= _kpi_mp_fim)]['hrv'].dropna()
        _rhr_mc = _wkpi[_wkpi['Data'] >= _kpi_mc_ini]['rhr'].dropna() if 'rhr' in _wkpi.columns else pd.Series()
        _rhr_mp = _wkpi[(_wkpi['Data'] >= _kpi_mp_ini) & (_wkpi['Data'] <= _kpi_mp_fim)]['rhr'].dropna() if 'rhr' in _wkpi.columns else pd.Series()
        _hrv_mc_v = float(_hrv_mc.mean()) if len(_hrv_mc) > 0 else None
        _hrv_mp_v = float(_hrv_mp.mean()) if len(_hrv_mp) > 0 else None
        _rhr_mc_v = float(_rhr_mc.mean()) if len(_rhr_mc) > 0 else None
        _rhr_mp_v = float(_rhr_mp.mean()) if len(_rhr_mp) > 0 else None
    else:
        _hrv_mc_v = _hrv_mp_v = _rhr_mc_v = _rhr_mp_v = None

    c3.metric("💚 HRV (mês)", f"{_hrv_mc_v:.0f} ms" if _hrv_mc_v else "—",
              _delta_pct(_hrv_mc_v, _hrv_mp_v) if _hrv_mc_v and _hrv_mp_v else None)
    c4.metric("❤️ RHR (mês)", f"{_rhr_mc_v:.0f} bpm" if _rhr_mc_v else "—",
              _delta_pct(_rhr_mc_v, _rhr_mp_v) if _rhr_mc_v and _rhr_mp_v else None)
    st.markdown("---")

    # ── Semana ACTUAL (seg→hoje) ──────────────────────────────────────────
    if da_full is not None and len(da_full) > 0:
        _df_sw = da_full.copy()
        _df_sw['Data'] = pd.to_datetime(_df_sw['Data'])
        _df_sw = _df_sw[_df_sw['type'].apply(norm_tipo) != 'WeightTraining']
        _df_sw = filtrar_principais(_df_sw)
        if 'icu_joules' in _df_sw.columns:
            _df_sw['_kj'] = pd.to_numeric(_df_sw['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _df_sw.columns:
            _df_sw['_kj'] = (pd.to_numeric(_df_sw['power_avg'], errors='coerce') *
                              pd.to_numeric(_df_sw['moving_time'], errors='coerce') / 1000)
        else:
            _df_sw['_kj'] = np.nan
        _df_sw['_mt']  = pd.to_numeric(_df_sw['moving_time'], errors='coerce') / 3600
        _df_sw['_rpe'] = pd.to_numeric(_df_sw['rpe'], errors='coerce')                          if 'rpe' in _df_sw.columns else np.nan

        _hoje_sw  = pd.Timestamp.now().normalize()
        _dow_sw   = _hoje_sw.weekday()
        _sem_ini_sw = _hoje_sw - pd.Timedelta(days=_dow_sw)  # segunda desta semana
        _df_sw_cur = _df_sw[_df_sw['Data'] >= _sem_ini_sw].copy()

        if len(_df_sw_cur) > 0:
            _df_sw_cur['_dia'] = _df_sw_cur['Data'].dt.strftime('%a %d/%m')
            rows_sw = []
            for _, r in _df_sw_cur.sort_values('Data').iterrows():
                rows_sw.append({
                    'Dia':        r['_dia'],
                    'Modalidade': r['type'],
                    'RPE':        f"{r['_rpe']:.0f}" if pd.notna(r['_rpe']) else '—',
                    'KJ':         f"{r['_kj']:.0f}" if pd.notna(r['_kj']) and r['_kj']>0 else '—',
                    'Horas':      fmt_dur(r['_mt']) if pd.notna(r['_mt']) else '—',
                })
            # ── Resumo Semanal — semana Seg→Dom actual ───────────────────
            st.subheader("📋 Resumo Semanal")

            # Semana actual: Seg → hoje
            _rs_hoje     = pd.Timestamp.now().normalize()
            _rs_dow      = _rs_hoje.weekday()
            _rs_sem_ini  = _rs_hoje - pd.Timedelta(days=_rs_dow)   # segunda
            # Semana anterior: Seg → Dom anterior
            _rs_sp_fim   = _rs_sem_ini - pd.Timedelta(days=1)      # dom passado
            _rs_sp_ini   = _rs_sp_fim  - pd.Timedelta(days=6)      # seg passada

            # Filtra da_full para as duas semanas (exclui WeightTraining)
            _rs_src = da_full.copy()
            _rs_src['Data'] = pd.to_datetime(_rs_src['Data'])
            _rs_src = _rs_src[_rs_src['type'].apply(norm_tipo) != 'WeightTraining']

            _rs_cur = _rs_src[_rs_src['Data'] >= _rs_sem_ini]
            _rs_prev= _rs_src[(_rs_src['Data'] >= _rs_sp_ini) & (_rs_src['Data'] <= _rs_sp_fim)]

            # Sessões e Horas
            _rs_sess_c = len(_rs_cur)
            _rs_sess_p = len(_rs_prev)
            _rs_h_c = _rs_cur['moving_time'].sum() / 3600 if 'moving_time' in _rs_cur.columns else 0
            _rs_h_p = _rs_prev['moving_time'].sum() / 3600 if 'moving_time' in _rs_prev.columns else 0
            _rs_kj_c = pd.to_numeric(_rs_cur.get('icu_joules', pd.Series()), errors='coerce').sum() / 1000 if 'icu_joules' in _rs_cur.columns else 0
            _rs_kj_p = pd.to_numeric(_rs_prev.get('icu_joules', pd.Series()), errors='coerce').sum() / 1000 if 'icu_joules' in _rs_prev.columns else 0

            def _rs_delta(vc, vp):
                if not vp or vp == 0: return None
                return f"{(vc-vp)/vp*100:+.0f}% vs sem.ant"

            # RPE distribuição Leve/Moderado/Forte — mesma escala de todas as outras tabs:
            # Leve: 1–4 | Moderado: 4.1–6.9 | Forte: 7–10
            _rs_rpe = pd.to_numeric(_rs_cur.get('rpe', pd.Series()), errors='coerce').dropna() if 'rpe' in _rs_cur.columns else pd.Series()
            _rs_rpe_str = "—"
            if len(_rs_rpe) > 0:
                _n_total = len(_rs_rpe)
                _n_leve  = int((_rs_rpe <= 4.0).sum())
                _n_mod   = int(((_rs_rpe > 4.0) & (_rs_rpe < 7.0)).sum())
                _n_forte = int((_rs_rpe >= 7.0).sum())
                _p_leve  = _n_leve / _n_total * 100
                _p_mod   = _n_mod  / _n_total * 100
                _p_forte = _n_forte/ _n_total * 100
                _rs_rpe_str = f"{_p_leve:.0f}% / {_p_mod:.0f}% / {_p_forte:.0f}%"

            # HRV e RHR — semana actual vs semana anterior
            _rs_wc = (wc_full if wc_full is not None and len(wc_full) > 0 else dw).copy() if (wc_full is not None or len(dw) > 0) else pd.DataFrame()
            _rs_hrv_c = _rs_hrv_p = _rs_rhr_c = _rs_rhr_p = None
            if len(_rs_wc) > 0 and 'hrv' in _rs_wc.columns:
                _rs_wc['Data'] = pd.to_datetime(_rs_wc['Data'])
                _wc_cur  = _rs_wc[_rs_wc['Data'] >= _rs_sem_ini]
                _wc_prev = _rs_wc[(_rs_wc['Data'] >= _rs_sp_ini) & (_rs_wc['Data'] <= _rs_sp_fim)]
                _rs_hrv_c = float(_wc_cur['hrv'].dropna().mean())  if _wc_cur['hrv'].notna().any()  else None
                _rs_hrv_p = float(_wc_prev['hrv'].dropna().mean()) if _wc_prev['hrv'].notna().any() else None
                if 'rhr' in _rs_wc.columns:
                    _rs_rhr_c = float(_wc_cur['rhr'].dropna().mean())  if _wc_cur['rhr'].notna().any()  else None
                    _rs_rhr_p = float(_wc_prev['rhr'].dropna().mean()) if _wc_prev['rhr'].notna().any() else None

            # CTL actual + ΔCTL semana + projetado (cálculo rápido com icu_training_load)
            _rs_ctl_str = _rs_dctl_str = _rs_ctl_proj_str = None
            try:
                _rs_pf = da_full.copy()
                _rs_pf['Data'] = pd.to_datetime(_rs_pf['Data'])
                if 'icu_training_load' in _rs_pf.columns:
                    _rs_pf['_load'] = pd.to_numeric(_rs_pf['icu_training_load'], errors='coerce').fillna(0)
                    _rs_dates = pd.date_range(_rs_pf['Data'].min(), _rs_hoje, freq='D')
                    _rs_load_d = _rs_pf.groupby('Data')['_load'].sum().reindex(_rs_dates, fill_value=0)
                    _rs_ctl_s  = _rs_load_d.ewm(span=42, adjust=False).mean()
                    _rs_ctl_hoje_v = float(_rs_ctl_s.iloc[-1])
                    # ΔCTL desta semana = diferença entre CTL hoje e CTL de segunda-feira
                    _rs_ctl_seg_v  = float(_rs_ctl_s.loc[_rs_sem_ini]) if _rs_sem_ini in _rs_ctl_s.index else _rs_ctl_hoje_v
                    _rs_dctl       = _rs_ctl_hoje_v - _rs_ctl_seg_v
                    # Projetar CTL para domingo (assumindo mesmo ritmo de carga)
                    _rs_dias_rest  = 6 - _rs_dow  # dias restantes até domingo
                    _rs_load_cur_w = float(_rs_pf[_rs_pf['Data'] >= _rs_sem_ini]['_load'].sum())
                    _rs_load_dia   = _rs_load_cur_w / max(_rs_dow + 1, 1)
                    _rs_ctl_dom_v  = _rs_ctl_hoje_v  # approx simples
                    for _ in range(_rs_dias_rest):
                        _rs_ctl_dom_v = _rs_ctl_dom_v + (_rs_load_dia - _rs_ctl_dom_v) / 42
                    _rs_ctl_str      = f"{_rs_ctl_hoje_v:.1f}"
                    _rs_dctl_str     = f"{_rs_dctl:+.2f} esta sem."
                    _rs_ctl_proj_str = f"→ {_rs_ctl_dom_v:.1f} (dom)"
            except Exception:
                pass

            # Layout: 4 colunas principais + linha adicional para RPE e CTL
            _rs1, _rs2, _rs3, _rs4 = st.columns(4)
            _rs1.metric("Sessões (sem.)",
                        str(_rs_sess_c),
                        _rs_delta(_rs_sess_c, _rs_sess_p))
            _rs2.metric("Horas (sem.)",
                        fmt_dur(_rs_h_c) if _rs_h_c else "—",
                        _rs_delta(_rs_h_c, _rs_h_p))
            _rs3.metric("HRV médio (sem.)",
                        f"{_rs_hrv_c:.0f} ms" if _rs_hrv_c else "—",
                        _rs_delta(_rs_hrv_c, _rs_hrv_p) if _rs_hrv_c and _rs_hrv_p else None)
            _rs4.metric("RHR médio (sem.)",
                        f"{_rs_rhr_c:.0f} bpm" if _rs_rhr_c else "—",
                        _rs_delta(_rs_rhr_c, _rs_rhr_p) if _rs_rhr_c and _rs_rhr_p else None)

            # Linha 2: KJ + RPE distribuição + CTL
            _rs5, _rs6, _rs7 = st.columns(3)
            _rs5.metric("KJ (sem.)",
                        f"{_rs_kj_c:.0f}" if _rs_kj_c else "—",
                        _rs_delta(_rs_kj_c, _rs_kj_p))
            with _rs6:
                st.metric("Intensidade (Leve/Mod/Forte)",
                          _rs_rpe_str,
                          help="% sessões por zona RPE: Leve≤4 / Moderado 5–7 / Forte≥8")
                if len(_rs_rpe) > 0:
                    st.caption(
                        f"<span style='color:#2ecc71'>●</span> Leve {_n_leve} &nbsp;"
                        f"<span style='color:#f39c12'>●</span> Mod {_n_mod} &nbsp;"
                        f"<span style='color:#e74c3c'>●</span> Forte {_n_forte} sessões",
                        unsafe_allow_html=True)
            with _rs7:
                if _rs_ctl_str:
                    st.metric("CTL actual",
                              _rs_ctl_str,
                              _rs_dctl_str)
                    st.caption(_rs_ctl_proj_str or "")

            st.markdown("---")

            # ── Semana actual | Semana anterior (lado a lado) ─────────────
            _col_sa, _col_sp = st.columns(2)
            with _col_sa:
                st.subheader("📅 Semana actual")
                st.dataframe(pd.DataFrame(rows_sw), width="stretch", hide_index=True)
            with _col_sp:
                _hoje_sw = pd.Timestamp.now().normalize()
                _sp_fim  = _hoje_sw - pd.Timedelta(days=_hoje_sw.weekday()+1)
                _sp_ini  = _sp_fim - pd.Timedelta(days=6)
                st.subheader(f"📅 Sem. anterior ({_sp_ini.strftime('%d/%m')}→{_sp_fim.strftime('%d/%m')})")
                if da_full is not None and len(da_full) > 0:
                    _dfsp = da_full.copy()
                    _dfsp['Data'] = pd.to_datetime(_dfsp['Data'])
                    _dfsp['_mt'] = pd.to_numeric(_dfsp['moving_time'], errors='coerce') / 3600
                    _dfsp['_kj'] = (pd.to_numeric(_dfsp['icu_joules'], errors='coerce') / 1000
                                    if 'icu_joules' in _dfsp.columns else np.nan)
                    _dfsp['_km'] = (pd.to_numeric(_dfsp['distance'], errors='coerce') / 1000
                                    if 'distance' in _dfsp.columns else np.nan)
                    _dfsp['_rpe'] = pd.to_numeric(_dfsp.get('rpe', pd.Series(dtype=float)),
                                                   errors='coerce')
                    _sub_sp = _dfsp[(_dfsp['Data'] >= _sp_ini) & (_dfsp['Data'] <= _sp_fim)]
                    _sub_sp = _sub_sp[_sub_sp['type'].apply(norm_tipo) != 'WeightTraining']
                    if len(_sub_sp) > 0:
                        _rows_sp = []
                        for _tp in sorted(_sub_sp['type'].apply(norm_tipo).unique()):
                            _s = _sub_sp[_sub_sp['type'].apply(norm_tipo) == _tp]
                            _rows_sp.append({
                                'Modal.':  _tp,
                                'Sess.':   len(_s),
                                'Horas':   fmt_dur(_s['_mt'].sum()),
                                'KM':      f"{_s['_km'].sum():.0f}" if _s['_km'].notna().any() else '—',
                                'KJ':      f"{_s['_kj'].sum():.0f}" if _s['_kj'].notna().any() else '—',
                                'RPE≥7':   int((_s['_rpe'] >= 7).sum()),
                            })
                        st.dataframe(pd.DataFrame(_rows_sp), width="stretch", hide_index=True)
                    else:
                        st.info("Sem actividades na semana anterior.")
        else:
            st.subheader("📋 Resumo Semanal")
            st.info("Sem actividades desde segunda-feira.")
            st.subheader("📅 Semana actual")
            st.info("Sem actividades desde segunda-feira.")
        st.markdown("---")

    # ── HRV-Guided + Recovery Score + Peso/BF ────────────────────────────
    vg_r1, vg_r2, vg_r3, vg_r4, vg_r5 = st.columns(5)

    # ── HRV-Guided Training (LnrMSSD, baseline 14d, ±0.5 SD) ────────────
    hrv_hoje    = None
    hrv_class   = "Sem dados"
    hrv_emoji   = "⚪"
    rec_score   = None
    rec_trend   = ""

    if wc_full is not None and len(wc_full) > 0 and 'hrv' in wc_full.columns:
        _wc = wc_full.copy()
        _wc['Data'] = pd.to_datetime(_wc['Data'])
        _wc = _wc.sort_values('Data')
        _wc['LnrMSSD'] = np.where(_wc['hrv'] > 0, np.log(_wc['hrv']), np.nan)
        _wc = _wc.dropna(subset=['LnrMSSD'])
        if len(_wc) >= 7:
            _wc['bm']  = _wc['LnrMSSD'].rolling(14, min_periods=7).mean()
            _wc['bs']  = _wc['LnrMSSD'].rolling(14, min_periods=7).std()
            _wc['linf']= _wc['bm'] - 0.5 * _wc['bs']
            _wc['lsup']= _wc['bm'] + 0.5 * _wc['bs']
            last = _wc.dropna(subset=['bm']).iloc[-1]
            hrv_hoje = last['hrv'] if 'hrv' in last else None
            if pd.notna(last['bm']) and pd.notna(last['LnrMSSD']):
                if last['linf'] <= last['LnrMSSD'] <= last['lsup']:
                    hrv_class = "HIIT"; hrv_emoji = "🟢"
                else:
                    hrv_class = "Recuperação"; hrv_emoji = "🔴"

        # Recovery Score trend (7d)
        _rec = calcular_recovery(_wc.rename(columns={'Data':'Data'}))
        if len(_rec) >= 7:
            rec_vals = _rec['recovery_score'].dropna()
            if len(rec_vals) >= 7:
                rec_score = rec_vals.iloc[-1]
                rec_mean7 = rec_vals.tail(7).mean()
                rec_mean_prev = rec_vals.iloc[-14:-7].mean() if len(rec_vals) >= 14 else rec_vals.mean()
                if rec_mean7 > rec_mean_prev * 1.03:
                    rec_trend = "↗"
                elif rec_mean7 < rec_mean_prev * 0.97:
                    rec_trend = "↘"
                else:
                    rec_trend = "→"

    with vg_r1:
        st.metric("🧠 HRV-Guided",
                  f"{hrv_emoji} {hrv_class}",
                  f"HRV {hrv_hoje:.0f} ms" if hrv_hoje else None)
    with vg_r2:
        st.metric("🔋 Recovery Score",
                  f"{rec_score:.0f}/100" if rec_score else "—",
                  rec_trend if rec_score else None)

    # ── Peso e BF (rolling 7d, só dias com dados reais) ────────────────
    # Fonte primária: wc/dw (formulário wellness — colunas 'peso' e 'bf_pct')
    # Fallback: dc (Consolidado_Comida — colunas 'Peso' e 'BF')
    peso_7d = peso_atual = peso_trend = bf_7d = bf_atual = bf_trend = None

    _hoje_p   = pd.Timestamp.now().normalize()
    _mes_ini_p  = _hoje_p.replace(day=1)
    _mes_p_fim2 = _mes_ini_p - pd.Timedelta(days=1)
    _mes_p_ini2 = _mes_p_fim2.replace(day=1)

    # Tenta primeiro wc (wellness form: 'peso' e 'bf_pct')
    # Se não tiver, usa dc (food diary: 'Peso' e 'BF')
    _sources = []
    if wc_full is not None and len(wc_full) > 0:
        _wc2 = wc_full.copy()
        _wc2['Data'] = pd.to_datetime(_wc2['Data'])
        # Mapa: (col_no_df, var_peso, var_bf)
        _peso_col = 'peso'   if 'peso'   in _wc2.columns else None
        _bf_col   = 'bf_pct' if 'bf_pct' in _wc2.columns else None
        if _peso_col or _bf_col:
            _sources.append((_wc2, _peso_col, _bf_col, 'wellness'))
    if dc is not None and len(dc) > 0:
        _dc2 = dc.copy()
        _dc2['Data'] = pd.to_datetime(_dc2['Data'])
        _sources.append((_dc2, 'Peso', 'BF', 'food'))

    for _src_df, _pcol, _bcol, _src_name in _sources:
        # Peso
        if _pcol and _pcol in _src_df.columns and peso_7d is None:
            _s = _src_df[['Data', _pcol]].dropna(subset=[_pcol]).copy()
            _s = _s[pd.to_numeric(_s[_pcol], errors='coerce').between(30, 200)]
            if len(_s) >= 2:
                _v_atual = float(_s[_pcol].iloc[-1])
                _ult7    = _s[_s['Data'] >= _hoje_p - pd.Timedelta(days=14)][_pcol]
                _v7      = float(_ult7.tail(7).mean()) if len(_ult7) >= 1 else _v_atual
                _mp      = _s[(_s['Data'] >= _mes_p_ini2) & (_s['Data'] <= _mes_p_fim2)][_pcol]
                _vmp     = float(_mp.mean()) if len(_mp) >= 1 else None
                if _vmp and _vmp > 0:
                    _pct = (_v7 - _vmp) / _vmp * 100
                    peso_trend = (f"↗ +{_pct:.1f}% vs mês ant." if _pct > 0.5
                                  else f"↘ {_pct:.1f}% vs mês ant." if _pct < -0.5
                                  else "→ estável vs mês ant.")
                peso_7d = _v7; peso_atual = _v_atual
        # BF %
        if _bcol and _bcol in _src_df.columns and bf_7d is None:
            _s = _src_df[['Data', _bcol]].dropna(subset=[_bcol]).copy()
            _s = _s[pd.to_numeric(_s[_bcol], errors='coerce').between(3, 50)]
            if len(_s) >= 2:
                _v_atual = float(_s[_bcol].iloc[-1])
                _ult7    = _s[_s['Data'] >= _hoje_p - pd.Timedelta(days=14)][_bcol]
                _v7      = float(_ult7.tail(7).mean()) if len(_ult7) >= 1 else _v_atual
                _mp      = _s[(_s['Data'] >= _mes_p_ini2) & (_s['Data'] <= _mes_p_fim2)][_bcol]
                _vmp     = float(_mp.mean()) if len(_mp) >= 1 else None
                if _vmp and _vmp > 0:
                    _pct = (_v7 - _vmp) / _vmp * 100
                    bf_trend = (f"↗ +{_pct:.1f}% vs mês ant." if _pct > 0.5
                                else f"↘ {_pct:.1f}% vs mês ant." if _pct < -0.5
                                else "→ estável vs mês ant.")
                bf_7d = _v7; bf_atual = _v_atual
        if peso_7d is not None and bf_7d is not None:
            break  # ambos encontrados, não precisa de fallback

    with vg_r3:
        st.metric("⚖️ Peso",
                  f"{peso_7d:.1f} kg (7d)" if peso_7d else "—",
                  f"Actual: {peso_atual:.1f} kg" if 'peso_atual' in dir() and peso_atual else None)
        if peso_trend: st.caption(peso_trend)
    with vg_r4:
        st.metric("🫁 BF",
                  f"{bf_7d:.1f}% (7d)" if bf_7d else "—",
                  f"Actual: {bf_atual:.1f}%" if 'bf_atual' in dir() and bf_atual else None)
        if bf_trend: st.caption(bf_trend)
    with vg_r5:
        pass  # coluna vazia para espaçamento

    st.markdown("---")

    # ── Prioridades + Need Score (visão rápida) ──────────────────────────
    st.subheader("🎯 Próxima Sessão — Prioridades e Necessidade")

    # Controlos de prioridade — partilhados com a aba Análises via keys únicas
    _mods_vg = ['Bike', 'Row', 'Ski', 'Run']
    # Defaults com session_state — lembra as últimas escolhas do utilizador
    _def_preset = st.session_state.get("vg_prio_preset", "Balanceado (K=10)")
    _def_p1     = st.session_state.get("vg_prio1", "Bike")
    _def_p2     = st.session_state.get("vg_prio2", "Row")
    _def_p3     = st.session_state.get("vg_prio3", "Ski")
    _def_p4     = st.session_state.get("vg_prio4", "Run")
    _presets    = ["Conservador (K=6)", "Balanceado (K=10)", "Agressivo (K=15)"]

    vg_c0, vg_c1, vg_c2, vg_c3, vg_c4 = st.columns([1, 1, 1, 1, 1])
    with vg_c0:
        vg_preset = st.selectbox("Preset K", _presets,
            index=_presets.index(_def_preset) if _def_preset in _presets else 1,
            key="vg_prio_preset")
        vg_K = {"Conservador (K=6)":6,"Balanceado (K=10)":10,"Agressivo (K=15)":15}[vg_preset]
    with vg_c1:
        vg_p1 = st.selectbox("🥇 P1 Foco", _mods_vg,
            index=_mods_vg.index(_def_p1) if _def_p1 in _mods_vg else 0,
            key="vg_prio1")
    with vg_c2:
        vg_p2 = st.selectbox("🥈 P2 Foco", _mods_vg,
            index=_mods_vg.index(_def_p2) if _def_p2 in _mods_vg else 1,
            key="vg_prio2")
    with vg_c3:
        vg_p3 = st.selectbox("🥉 P3 Manutenção", _mods_vg,
            index=_mods_vg.index(_def_p3) if _def_p3 in _mods_vg else 2,
            key="vg_prio3")
    with vg_c4:
        vg_p4 = st.selectbox("4️⃣  P4 Manutenção", _mods_vg,
            index=_mods_vg.index(_def_p4) if _def_p4 in _mods_vg else 3,
            key="vg_prio4")

    vg_prio_rank  = {vg_p1:1, vg_p2:2, vg_p3:3, vg_p4:4}
    vg_grupo_foco = {vg_p1, vg_p2}
    vg_grupo_man  = {vg_p3, vg_p4}

    if da_full is not None and len(da_full) > 0:
        vg_res, _ = analisar_falta_estimulo(da_full, janela_dias=7)
        if vg_res:
            rows_f, rows_m = [], []
            for mod, d in vg_res.items():
                rank   = vg_prio_rank.get(mod, 4)
                peso   = (4 + 1 - rank) / 4
                bonus  = peso * vg_K * (1 - d['need_score'] / 100)
                nf     = d['need_score'] + bonus
                ol_flag= d.get('overload', False)
                if ol_flag:   nf *= 0.5
                if mod in vg_grupo_man: nf = min(nf, 40)
                nf = max(nf, 10)
                pf = ('ALTA' if nf>=70 else 'MÉDIA' if nf>=40 else 'BAIXA')
                row_d = {
                    'Modalidade': f"{'🎯' if mod in vg_grupo_foco else '🔧'} {mod}"
                                  + (' ⚠️' if ol_flag else ''),
                    'Need':       f"{d['need_score']:.0f}",
                    'Final':      f"{nf:.0f}",
                    'Vol/Int':    f"{d.get('need_vol',0):.0f}/{d.get('need_int_prescr',0):.0f}",
                    'Prescrição': d.get('prescricao','—'),
                }
                if mod in vg_grupo_foco: rows_f.append((nf, row_d))
                else:                    rows_m.append((nf, row_d))

            rows_f.sort(key=lambda x: x[0], reverse=True)
            rows_m.sort(key=lambda x: x[0], reverse=True)

            col_f, col_m = st.columns(2)
            with col_f:
                st.markdown("**🎯 Foco**")
                if rows_f:
                    st.dataframe(pd.DataFrame([r for _,r in rows_f]),
                                 width="stretch", hide_index=True)
                    # Sugestão principal
                    top    = rows_f[0]
                    top_mod= top[1]['Modalidade'].replace('🎯 ','').replace('🔧 ','').replace(' ⚠️','')
                    top_d  = vg_res.get(top_mod, {})
                    if top_d.get('overload'):
                        st.warning(f"⚠️ **{top_mod}**: {top_d.get('prescricao','—')}")
                    else:
                        st.info(f"🎯 **{top_mod}** — {top_d.get('prescricao','—')}")
            with col_m:
                st.markdown("**🔧 Manutenção**")
                if rows_m:
                    st.dataframe(pd.DataFrame([r for _,r in rows_m]),
                                 width="stretch", hide_index=True)
        else:
            st.info("Dados insuficientes para análise de necessidade.")
    else:
        st.info("Dados de actividade completos não disponíveis.")

    # ── Semana anterior + Comparação mensal ─────────────────────────────
    if da_full is not None and len(da_full) > 0:
        _df_all = da_full.copy()
        _df_all['Data'] = pd.to_datetime(_df_all['Data'])
        _df_all = _df_all[_df_all['type'].apply(norm_tipo) != 'WeightTraining']

        # KJ
        if 'icu_joules' in _df_all.columns:
            _df_all['_kj'] = pd.to_numeric(_df_all['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _df_all.columns and 'moving_time' in _df_all.columns:
            _df_all['_kj'] = (pd.to_numeric(_df_all['power_avg'], errors='coerce') *
                              pd.to_numeric(_df_all['moving_time'], errors='coerce') / 1000)
        else:
            _df_all['_kj'] = np.nan
        _df_all['_km']  = pd.to_numeric(_df_all['distance'], errors='coerce') / 1000                           if 'distance' in _df_all.columns else np.nan
        _df_all['_mt']  = pd.to_numeric(_df_all['moving_time'], errors='coerce') / 3600
        _df_all['_rpe'] = pd.to_numeric(_df_all['rpe'], errors='coerce')                           if 'rpe' in _df_all.columns else np.nan

        hoje      = pd.Timestamp.now().normalize()
        dow       = hoje.weekday()
        sem_fim   = hoje - pd.Timedelta(days=dow + 1)
        sem_ini   = sem_fim - pd.Timedelta(days=6)
        mes_c_ini = hoje.replace(day=1)
        mes_p_fim = mes_c_ini - pd.Timedelta(days=1)
        mes_p_ini = mes_p_fim.replace(day=1)

        def _vg_agg(df, d_ini, d_fim, semana=False):
            sub = df[(df['Data'] >= d_ini) & (df['Data'] <= d_fim)]
            if len(sub) == 0: return {}
            res = {}
            for mod in sorted(sub['type'].apply(norm_tipo).unique()):
                s = sub[sub['type'].apply(norm_tipo) == mod]
                d = {'kj': s['_kj'].sum() if s['_kj'].notna().any() else 0,
                     'km': s['_km'].sum() if '_km' in s and s['_km'].notna().any() else 0,
                     'horas': s['_mt'].sum() if s['_mt'].notna().any() else 0}
                if semana:
                    d['sessoes']   = len(s)
                    d['rpe_altas'] = int((s['_rpe'] >= 7).sum())                                      if '_rpe' in s.columns else 0
                res[mod] = d
            return res

        sem_data    = _vg_agg(_df_all, sem_ini, sem_fim, semana=True)
        mes_p_p_fim = mes_p_ini - pd.Timedelta(days=1)
        mes_p_p_ini = mes_p_p_fim.replace(day=1)
        mes_pp      = _vg_agg(_df_all, mes_p_p_ini, mes_p_p_fim)
        mes_p       = _vg_agg(_df_all, mes_p_ini, mes_p_fim)
        mes_c       = _vg_agg(_df_all, mes_c_ini, hoje)

        # ── Gráfico barras horizontais KJ ────────────────────────────────
        all_mods_m = sorted(set(list(mes_p.keys()) + list(mes_c.keys())))
        if all_mods_m:
            st.subheader(f"⚡ KJ por Modalidade — "
                         f"{mes_p_ini.strftime('%b %Y')} vs {mes_c_ini.strftime('%b %Y')}")
            kj_p = [mes_p.get(m,{}).get('kj',0) for m in all_mods_m]
            kj_c = [mes_c.get(m,{}).get('kj',0) for m in all_mods_m]
            fig_kj = go.Figure()
            fig_kj.add_trace(go.Bar(
                y=all_mods_m, x=kj_p, name=mes_p_ini.strftime('%b %Y'),
                orientation='h', marker_color='#95a5a6',
                text=[f"{v:.0f}" if v>0 else '' for v in kj_p],
                textposition='outside',
                hovertemplate='%{y}: <b>%{x:.0f} kJ</b><extra></extra>'))
            fig_kj.add_trace(go.Bar(
                y=all_mods_m, x=kj_c,
                name=f"{mes_c_ini.strftime('%b %Y')} (corrente)",
                orientation='h', marker_color='#2ecc71',
                text=[f"{v:.0f}" if v>0 else '' for v in kj_c],
                textposition='outside',
                hovertemplate='%{y}: <b>%{x:.0f} kJ</b><extra></extra>'))
            fig_kj.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                barmode='group', height=max(200, len(all_mods_m)*60+80),
                font=dict(color='#222222'),
                margin=dict(l=60, r=80, t=30, b=20),
                xaxis=dict(title='kJ', showgrid=True, gridcolor='#eeeeee',
                           tickfont=dict(color='#333333')),
                yaxis=dict(tickfont=dict(color='#333333')),
                legend=dict(orientation='h', y=1.08,
                            font=dict(color='#111111', size=11)))
            st.plotly_chart(fig_kj, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

        # ── Tabela comparativa mensal — 2 blocos com cores ──────────────
        if all_mods_m:
            st.subheader("📊 Comparação Mensal por Modalidade")

            def _di(vc, vp, lim=3.0):
                # icon + color delta
                if not vp or vp == 0: return '—', '#888'
                pct = (vc - vp) / vp * 100
                if   pct >  lim: return f'↗ +{pct:.0f}%', '#27ae60'
                elif pct < -lim: return f'↘ {pct:.0f}%',  '#e74c3c'
                else:             return f'→ {pct:+.0f}%', '#7f8c8d'

            def _mes_rows(mod_list, mc, mr):
                rows = []
                for mod in mod_list:
                    dc_ = mc.get(mod, {}); dp_ = mr.get(mod, {})
                    kj_i, _ = _di(dc_.get('kj',0),    dp_.get('kj',0))
                    km_i, _ = _di(dc_.get('km',0),    dp_.get('km',0))
                    h_i,  _ = _di(dc_.get('horas',0), dp_.get('horas',0))
                    rows.append({
                        'Modal.': mod,
                        'KJ':    f"{dc_.get('kj',0):.0f}"    if dc_.get('kj',0)>0    else '—',
                        'ΔKJ':   kj_i,
                        'KM':    f"{dc_.get('km',0):.0f}"    if dc_.get('km',0)>0    else '—',
                        'ΔKM':   km_i,
                        'Horas': fmt_dur(dc_.get('horas',0)) if dc_.get('horas',0)>0 else '—',
                        'ΔH':    h_i,
                    })
                return rows

            def _color_df(df_s):
                def _c(v):
                    v = str(v)
                    if '↗' in v: return 'color:#27ae60;font-weight:bold'
                    if '↘' in v: return 'color:#e74c3c;font-weight:bold'
                    if '→' in v: return 'color:#7f8c8d'
                    return ''
                dcols = [c for c in df_s.columns if c.startswith('Δ')]
                return df_s.style.map(_c, subset=dcols) if dcols else df_s.style

            all_mods_pp = sorted(set(list(mes_p.keys()) + list(mes_pp.keys())))
            _cb1, _cb2 = st.columns(2)
            with _cb1:
                st.caption(f"**{mes_p_ini.strftime('%b %Y')}** vs {mes_p_p_ini.strftime('%b %Y')}")
                _rb1 = _mes_rows(all_mods_pp, mes_p, mes_pp)
                if _rb1:
                    st.dataframe(_color_df(pd.DataFrame(_rb1)),
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Sem dados para mês anterior")
            with _cb2:
                st.caption(f"**{mes_c_ini.strftime('%b %Y')} ▶** vs {mes_p_ini.strftime('%b %Y')}")
                _rb2 = _mes_rows(all_mods_m, mes_c, mes_p)
                if _rb2:
                    st.dataframe(_color_df(pd.DataFrame(_rb2)),
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Sem dados para mês corrente")


    # ── Camada de Progressão de Carga ────────────────────────────────────
    if da_full is not None and len(da_full) > 0:
        st.subheader("📈 Progressão de Carga Semanal")
        st.caption(
            "Camada independente do Need Score — controla QUANTO treinar. "
            "Fator modulado pelo Need (leitura apenas). Cap +12% vs ano anterior.")

        _pf = da_full.copy()
        _pf['Data'] = pd.to_datetime(_pf['Data'])
        _pf = _pf[_pf['type'].apply(norm_tipo) != 'WeightTraining']

        # KJ
        if 'icu_joules' in _pf.columns:
            _pf['_kj'] = pd.to_numeric(_pf['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _pf.columns and 'moving_time' in _pf.columns:
            _pf['_kj'] = (pd.to_numeric(_pf['power_avg'], errors='coerce') *
                          pd.to_numeric(_pf['moving_time'], errors='coerce') / 1000)
        else:
            _pf['_kj'] = np.nan
        _pf['_km']  = pd.to_numeric(_pf['distance'], errors='coerce') / 1000                       if 'distance' in _pf.columns else np.nan
        _pf['_mt']  = pd.to_numeric(_pf['moving_time'], errors='coerce') / 3600
        _pf['_rpe_n']   = pd.to_numeric(_pf['rpe'], errors='coerce') if 'rpe' in _pf.columns else np.nan
        _pf['_dur_min'] = _pf['_mt'] * 60

        hoje_pf   = pd.Timestamp.now().normalize()
        ano_atual = hoje_pf.year
        ano_ant   = ano_atual - 1
        dia_ano   = hoje_pf.timetuple().tm_yday
        # Semana actual: segunda até hoje
        dow_pf    = hoje_pf.weekday()
        sem_ini_pf= hoje_pf - pd.Timedelta(days=dow_pf)

        # Buscar Need scores e overload do modelo (se disponível)
        _need_cache = {}
        _ol_cache   = {}
        _ni_cache   = {}
        try:
            _res_prog, _ = analisar_falta_estimulo(_pf, janela_dias=14)
            if _res_prog:
                for _m, _d in _res_prog.items():
                    _need_cache[_m] = _d.get('need_score', 40)
                    _ol_cache[_m]   = _d.get('overload', False)
                    _ni_cache[_m]   = _d.get('need_int_prescr', 50)
        except Exception:
            pass

        # Need_7d cache — para calcular delta 7d vs 14d
        _need7_cache = {}
        try:
            _res_prog7, _ = analisar_falta_estimulo(_pf, janela_dias=7)
            if _res_prog7:
                for _m, _d in _res_prog7.items():
                    _need7_cache[_m] = _d.get('need_score', 40)
        except Exception:
            pass

        def _calc_f_delta(need_7d, need_14d, ol, n_sess_7d):
            """
            Opção B — thresholds fixos calibrados.
            Delta = Need_7d - Need_14d.
            Positivo = deficit recente maior que baseline = aumentar kj_target.
            Negativo = excesso recente = reduzir kj_target.
            Limites: F_MAX=1.10, F_MIN=0.85 (alinhados com prog_cap).
            """
            # Poucos dados recentes → neutro
            if n_sess_7d < 3:
                return 1.0
            # Overload → nunca aumentar
            delta = float(need_7d) - float(need_14d)
            if   delta >= 25:  f = 1.10
            elif delta >= 15:  f = 1.06
            elif delta >= 5:   f = 1.02
            elif delta <= -25: f = 0.85
            elif delta <= -15: f = 0.90
            elif delta <= -5:  f = 0.95
            else:              f = 1.00
            if ol:
                f = min(f, 1.0)
            return max(0.85, min(f, 1.10))

        # eFTP por modalidade (último valor disponível)
        _eftp = {}
        # icu_ftp = FTP testado (estável, para zonas) — preferido
        # icu_eftp = estimado pelo modelo (para acompanhamento de forma)
        _ftp_col = 'icu_ftp' if 'icu_ftp' in _pf.columns else 'icu_eftp'
        if _ftp_col in _pf.columns:
            for _m in ['Bike','Row','Ski','Run']:
                _s = _pf[_pf['type'].apply(norm_tipo)==_m][_ftp_col]
                _s = pd.to_numeric(_s, errors='coerce').dropna()
                if len(_s) > 0: _eftp[_m] = float(_s.iloc[-1])

        def _med_semanas(df, col, n_sem=8):
            """Mediana das últimas n semanas com dados (col > 0)."""
            df = df.copy()
            df['_sem'] = df['Data'].dt.to_period('W')
            agg = df.groupby('_sem')[col].sum().reset_index()
            agg = agg.sort_values('_sem').tail(n_sem)
            vals = agg[col][agg[col] > 0]
            return float(vals.median()) if len(vals) > 0 else 0.0

        def _sugestao_sessao(kj_rest, h_rest, km_rest, mod, eftp, ni, ol, df_hist=None, f_delta=1.0):
            """
            Retorna (df_opcoes, ref_line, ol_warn).
            5 tipos de estimulo com mesmo KJ_target. Principal por Need_intensity.
            """
            _TIPOS = [
                {"key":"anaerobio",  "label":"⚫ Anaeróbio",  "pct":(1.15,1.30), "rpe_lbl":"9–10"},
                {"key":"vo2",        "label":"🔴 VO2max",     "pct":(0.95,1.10), "rpe_lbl":"8–9"},
                {"key":"threshold",  "label":"🟡 Threshold",  "pct":(0.83,0.90), "rpe_lbl":"6–7"},
                {"key":"sweetspot",  "label":"🟢 Sweet Spot", "pct":(0.78,0.83), "rpe_lbl":"5–6"},
                {"key":"leve",       "label":"🔵 Leve",       "pct":(0.55,0.68), "rpe_lbl":"3–4"},
            ]

            def _estrutura(tipo_key, dur_work_min):
                d = max(1.0, float(dur_work_min))
                structs = []
                if tipo_key == "anaerobio":
                    for reps, on_s, rest_s in [(10,20,120),(8,25,150),(6,30,180),(5,30,180),(4,40,240)]:
                        if d > 0 and abs(reps*on_s/60 - d)/d < 0.30:
                            structs.append(f"{reps}x{on_s}s rest {rest_s//60}:{rest_s%60:02d}")
                    if not structs:
                        structs = [f"{max(4,round(d*60/25))}x25s rest 2:30"]
                elif tipo_key == "vo2":
                    for reps, on_s in [(4,240),(5,180),(6,180),(5,240),(4,300),(6,240),(3,300),(8,150),(10,120)]:
                        if d > 0 and abs(reps*on_s/60 - d)/d < 0.20:
                            ms = on_s//60; ss_r = on_s%60
                            structs.append(f"{reps}x{ms}:{ss_r:02d} rest {ms}:{ss_r:02d}")
                    reps_30 = round(d*2)
                    if 6 <= reps_30 <= 20:
                        n_ser = max(1, reps_30//8); rpp = reps_30//n_ser
                        structs.append(f"{n_ser}x{rpp}x30/30")
                    structs = structs[:3]
                    if not structs:
                        structs = [f"{max(3,round(d/4))}x4:00 rest 4:00"]
                elif tipo_key == "threshold":
                    for reps, on_m in [(3,10),(4,8),(2,15),(4,10),(3,12),(5,8),(2,20)]:
                        if d > 0 and abs(reps*on_m - d)/d < 0.25:
                            structs.append(f"{reps}x{on_m}min rest 3–4min")
                    structs = structs[:2]
                    if not structs:
                        n = max(2, round(d/10))
                        structs = [f"{n}x{round(d/n)}min rest 3min"]
                elif tipo_key == "sweetspot":
                    for reps, on_m in [(1,40),(2,20),(3,15),(1,30),(2,25),(1,50)]:
                        if d > 0 and abs(reps*on_m - d)/d < 0.25:
                            structs.append(f"1x{on_m}min contínuo" if reps==1 else f"{reps}x{on_m}min rest 4–5min")
                    structs = structs[:2]
                    if not structs:
                        structs = [f"1x{round(d)}min contínuo"]
                else:
                    structs = [f"Contínuo {round(d)}min"]
                return "  /  ".join(structs)

            # ── Histórico ─────────────────────────────────────────────────
            _ref_kj = _ref_dur = _ref_pwr = None
            _eff_delta = 0.0
            _kjh_ref = _kjh_baseline = None
            kj_target_A_adj = 1.0
            _PROG_CAP = {"Z3":1.07, "Z2":1.10, "Z1":1.12}
            _TEMPO_CAP = 1.10
            pwr_inc_base = 0.01 if ni >= 70 else 0.02

            if df_hist is not None and len(df_hist) >= 2:
                _df_mod = df_hist[df_hist["type"].apply(norm_tipo) == mod].copy()
                if len(_df_mod) > 0:
                    _has_zona = "z3_kj" in _df_mod.columns and "z2_kj" in _df_mod.columns
                    if _has_zona:
                        _z3n = pd.to_numeric(_df_mod["z3_kj"], errors="coerce").fillna(0)
                        _z2n = pd.to_numeric(_df_mod["z2_kj"], errors="coerce").fillna(0)
                        _tot = pd.to_numeric(_df_mod["_kj"], errors="coerce").replace(0,np.nan).fillna(1)
                        _df_mod["_zona_dom"] = "Z1"
                        _df_mod.loc[_z2n/_tot > 0.30, "_zona_dom"] = "Z2"
                        _df_mod.loc[_z3n/_tot > 0.20, "_zona_dom"] = "Z3"
                        _df_mod.loc[(_z3n==0)&(_z2n==0), "_zona_dom"] = None
                    else:
                        _df_mod["_zona_dom"] = None
                    _df_mod["_rpe_n_num"] = pd.to_numeric(_df_mod["_rpe_n"], errors="coerce")
                    _df_mod["_match_rpe"]  = _df_mod["_rpe_n_num"].between(5,10).astype(float)
                    _df_mod["_match_zona"] = (_df_mod["_zona_dom"]=="Z3").astype(float)*0.5
                    _nt = len(_df_mod)
                    _df_mod["_recency"] = (np.arange(_nt)/max(_nt-1,1))*0.2
                    _df_mod["_score"]   = _df_mod["_match_rpe"]+_df_mod["_match_zona"]+_df_mod["_recency"]
                    _pool = _df_mod[_df_mod["_match_rpe"] >= 1.0].copy()
                    _dh   = _pool.sort_values("_score", ascending=False).head(5)
                    if len(_dh) >= 2:
                        _ref_kj  = float(_dh["_kj"].median())
                        _ref_dur = float(_dh["_dur_min"].median())
                        if "z3_pwr" in _dh.columns:
                            _zp = pd.to_numeric(_dh["z3_pwr"], errors="coerce").replace(0,np.nan)
                            _ref_pwr = float(_zp.median()) if _zp.notna().any() else None
                        _pool2 = _pool.copy()
                        _kj_w  = pd.to_numeric(_pool2["_kj"], errors="coerce").replace(0,np.nan)
                        _trimp = (pd.to_numeric(_pool2["_dur_min"], errors="coerce") *
                                  pd.to_numeric(_pool2["_rpe_n"],   errors="coerce"))
                        _pool2["_eff"] = _trimp / _kj_w
                        _eff_bl  = float(_pool2["_eff"].median()) if _pool2["_eff"].notna().any() else None
                        _cut8w   = pd.Timestamp.now() - pd.Timedelta(weeks=8)
                        _prec    = _pool2[_pool2["Data"] >= _cut8w]
                        _eff_rec = float(_prec["_eff"].median()) if (len(_prec)>=2 and _prec["_eff"].notna().any()) else None
                        if _eff_bl and _eff_rec and _eff_bl > 0:
                            _eff_delta = (_eff_rec/_eff_bl) - 1.0
                        _pool2["_kjh"] = (pd.to_numeric(_pool2["_kj"], errors="coerce") /
                                          (pd.to_numeric(_pool2["_dur_min"], errors="coerce")/60))
                        _kjh_baseline = float(_pool2["_kjh"].median()) if _pool2["_kjh"].notna().any() else None
                        _kjh_ref = (_ref_kj/(_ref_dur/60)) if (_ref_kj and _ref_dur and _ref_dur>0) else None

            # ── pwr_inc ────────────────────────────────────────────────────
            pwr_inc = pwr_inc_base
            if   _eff_delta < -0.05: pwr_inc *= 1.2
            elif _eff_delta <  0.05: pwr_inc *= 1.0
            elif _eff_delta <  0.12: pwr_inc *= 0.9
            else:                    pwr_inc *= 0.7
            if _kjh_baseline and _kjh_ref:
                _kjh_ratio = _kjh_ref/_kjh_baseline
                if   _kjh_ratio >= 1.0:  pwr_inc *= 1.1
                elif _kjh_ratio >= 0.90: kj_target_A_adj = 0.95
                else:                    kj_target_A_adj = 0.90

            # ── zona e KJ_target ───────────────────────────────────────────
            _prog_cap = _PROG_CAP["Z3" if ni>=70 else "Z2" if ni>=40 else "Z1"]
            if ol:
                kj_target = (_ref_kj*0.65) if _ref_kj else max(kj_rest*0.65, 60)
            elif _ref_kj and _ref_kj > 0:
                kj_target = min(kj_rest if kj_rest>0 else _ref_kj*1.03,
                                _ref_kj*_prog_cap) * kj_target_A_adj
            else:
                kj_target = max(kj_rest, 80)

            # ── f_delta: ajuste 7d vs 14d ─────────────────────────────────
            # Overload já capturado em kj_target*0.65 — f_delta clampado a ≤1.0 por _calc_f_delta
            kj_target = kj_target * f_delta

            # Clamps de segurança pós f_delta
            if _ref_kj and _ref_kj > 0:
                kj_target = min(kj_target, _ref_kj * 1.15)  # nunca +15% do histórico
            kj_target = max(kj_target, 40)                   # mínimo absoluto 40 kJ

            tempo_max = (_ref_dur*_TEMPO_CAP) if _ref_dur else 90
            watts_ftp = eftp if eftp else 200

            # ── Semana anterior — Z1/Z2/Z3 por zona ───────────────────────
            _sa_z3_kj = _sa_z3_dur = _sa_z3_pwr = None
            _sa_z2_kj = _sa_z2_dur = _sa_z2_pwr = None
            _sa_z1_kj = _sa_z1_dur = _sa_z1_rpe_med = None
            hoje_sa    = pd.Timestamp.now().normalize()
            sem_ini_sa = hoje_sa - pd.Timedelta(days=hoje_sa.weekday())
            sa_ini     = sem_ini_sa - pd.Timedelta(weeks=1)
            sa_fim     = sem_ini_sa - pd.Timedelta(days=1)

            if df_hist is not None and len(df_hist) > 0 and "_rpe_n" in df_hist.columns:
                _df_sa_all = df_hist[
                    (df_hist["type"].apply(norm_tipo) == mod) &
                    (df_hist["Data"] >= sa_ini) &
                    (df_hist["Data"] <= sa_fim)
                ].copy()
                _rpe_sa = pd.to_numeric(_df_sa_all.get("_rpe_n", pd.Series(dtype=float)), errors="coerce")
                _kj_sa  = pd.to_numeric(_df_sa_all.get("_kj",    pd.Series(dtype=float)), errors="coerce").replace(0,np.nan)

                # Z3 — RPE >= 7 AND z3_kj/total_kj > 20%
                if "z3_kj" in _df_sa_all.columns:
                    _z3_sa  = pd.to_numeric(_df_sa_all["z3_kj"], errors="coerce").replace(0,np.nan)
                    _z3_dom = (_z3_sa / _kj_sa.fillna(1) > 0.20) & (_rpe_sa >= 7)
                    _df_z3  = _df_sa_all[_z3_dom].copy()
                    if len(_df_z3) > 0:
                        _sa_z3_kj  = float(pd.to_numeric(_df_z3["z3_kj"],  errors="coerce").replace(0,np.nan).sum())   if "z3_kj"  in _df_z3.columns else None
                        _sa_z3_dur = float(pd.to_numeric(_df_z3["z3_sec"], errors="coerce").replace(0,np.nan).sum()/60) if "z3_sec" in _df_z3.columns else None
                        _sa_z3_pwr = float(pd.to_numeric(_df_z3["z3_pwr"], errors="coerce").replace(0,np.nan).mean())  if "z3_pwr" in _df_z3.columns else None

                # Z2 — RPE 5-7 AND z2_kj/total_kj > 30%
                if "z2_kj" in _df_sa_all.columns:
                    _z2_sa  = pd.to_numeric(_df_sa_all["z2_kj"], errors="coerce").replace(0,np.nan)
                    _z2_dom = (_z2_sa / _kj_sa.fillna(1) > 0.30) & (_rpe_sa.between(5,7))
                    _df_z2  = _df_sa_all[_z2_dom].copy()
                    if len(_df_z2) > 0:
                        _sa_z2_kj  = float(pd.to_numeric(_df_z2["z2_kj"],  errors="coerce").replace(0,np.nan).sum())   if "z2_kj"  in _df_z2.columns else None
                        _sa_z2_dur = float(pd.to_numeric(_df_z2["z2_sec"], errors="coerce").replace(0,np.nan).sum()/60) if "z2_sec" in _df_z2.columns else None
                        _sa_z2_pwr = float(pd.to_numeric(_df_z2["z2_pwr"], errors="coerce").replace(0,np.nan).mean())  if "z2_pwr" in _df_z2.columns else None

                # Z1 — RPE <= 4 (qualquer sessão leve)
                _df_z1 = _df_sa_all[_rpe_sa <= 4].copy()
                if len(_df_z1) > 0:
                    _sa_z1_kj      = float(_kj_sa[_rpe_sa<=4].sum())         if _kj_sa[_rpe_sa<=4].notna().any() else None
                    _sa_z1_dur     = float(pd.to_numeric(_df_z1["_dur_min"], errors="coerce").sum()) if "_dur_min" in _df_z1.columns else None
                    _sa_z1_rpe_med = float(_rpe_sa[_rpe_sa<=4].median())

            # ── Principal ──────────────────────────────────────────────────
            if ol:
                _pk = "leve"
            elif ni >= 90:
                _pk = "anaerobio"
            elif ni >= 70:
                _pk = "vo2"
            elif ni >= 60:
                _pk = "threshold"
            elif ni >= 40:
                _pk = "sweetspot"
            else:
                _pk = "leve"

            # ── Gerar linhas ───────────────────────────────────────────────
            _rows = []
            for _t in _TIPOS:
                _key  = _t["key"]
                _pct  = (_t["pct"][0]+_t["pct"][1])/2
                _pwr_z = watts_ftp * _pct
                if _key in ("vo2","anaerobio") and _ref_pwr:
                    _pwr_z = _ref_pwr
                _inc = pwr_inc if _key==_pk else (0.0 if _key=="anaerobio" else pwr_inc*0.5)
                _pwr_f = _pwr_z*(1+_inc)*(0.95 if ol else 1.0)

                # ── f_delta power adjustment — ajuste leve com clamp por zona ──
                # 50% do efeito do f_delta, para não sobrepor o tipo de sessão
                _pwr_adj = 1.0 + (f_delta - 1.0) * 0.5
                # Clamp por zona (garante que Threshold não vaza para VO2)
                _clamp_zona = {"anaerobio": 1.01, "vo2": 1.02,
                               "threshold": 1.03, "sweetspot": 1.03, "leve": 1.05}
                _pwr_adj = min(_pwr_adj, _clamp_zona.get(_key, 1.03))
                _pwr_adj = max(_pwr_adj, 0.90)  # nunca reduzir mais de 10%
                # Limites de zona para não mudar tipo de sessão
                _zona_pct_min, _zona_pct_max = {
                    "anaerobio":  (1.10, 1.35),
                    "vo2":        (0.90, 1.12),
                    "threshold":  (0.80, 0.92),
                    "sweetspot":  (0.75, 0.84),
                    "leve":       (0.50, 0.70),
                }.get(_key, (0.50, 1.35))
                _pwr_f = _pwr_f * _pwr_adj
                _ftp_ref = (eftp if eftp else 200)
                _pwr_f = max(_ftp_ref * _zona_pct_min,
                             min(_pwr_f, _ftp_ref * _zona_pct_max))
                _kj_z  = (kj_target*0.35 if _key=="anaerobio" else
                           kj_target*1.10 if _key=="leve" else kj_target)
                _dw = min((_kj_z*1000/(_pwr_f*60)) if _pwr_f>0 else 40, tempo_max)
                _rr = {"anaerobio":8.0,"vo2":1.0,"threshold":0.35,"sweetspot":0.20}.get(_key,0.0)
                _dt = _dw*(1+_rr)
                _kj_r = _pwr_f*_dw*60/1000
                _kjh  = _kj_r/(_dt/60) if _dt>0 else 0
                _struct = _estrutura(_key, _dw)
                # ── vs semana anterior por zona ──────────────────────────
                _vs = ""
                if _key in ("vo2","anaerobio"):
                    # Z3: comparar kJ, duração e power em Z3
                    if _sa_z3_kj and _sa_z3_kj > 5:
                        _dk = _kj_r - _sa_z3_kj
                        _vs = f"{_dk:+.0f}kJ Z3"
                        if _sa_z3_pwr and _sa_z3_pwr > 0:
                            _dp = _pwr_f - _sa_z3_pwr
                            _vs += f" | {_dp:+.0f}W"
                    elif _sa_z3_dur and _sa_z3_dur > 2:
                        _vs = f"{_dw-_sa_z3_dur:+.1f}min Z3"

                elif _key in ("threshold","sweetspot"):
                    # Z2: comparar kJ e duração em Z2
                    if _sa_z2_kj and _sa_z2_kj > 10:
                        _dk2 = _kj_r - _sa_z2_kj
                        _vs = f"{_dk2:+.0f}kJ Z2"
                        if _sa_z2_pwr and _sa_z2_pwr > 0:
                            _dp2 = _pwr_f - _sa_z2_pwr
                            _vs += f" | {_dp2:+.0f}W"
                    elif _sa_z2_dur and _sa_z2_dur > 5:
                        _vs = f"{_dw-_sa_z2_dur:+.1f}min Z2"
                    elif not _sa_z2_kj and not _sa_z2_dur:
                        # Sem Z2 na semana anterior — novo estímulo
                        _vs = "novo estímulo Z2"

                elif _key == "leve":
                    # Z1: opção B — se RPE médio < 4 e dur > 45min, sugerir subir para Z2
                    if _sa_z1_kj and _sa_z1_dur:
                        _dk1 = _kj_r - _sa_z1_kj
                        _vs = f"{_dk1:+.0f}kJ Z1"
                        # Sugerir upgrade se leve bem tolerado
                        if (_sa_z1_rpe_med is not None and _sa_z1_rpe_med < 3.5
                                and _sa_z1_dur > 45):
                            _vs += " → considera Sweet Spot"
                    elif _sa_z1_dur and _sa_z1_dur > 0:
                        _dd1 = _dw - _sa_z1_dur
                        _vs = f"{_dd1:+.0f}min Z1"
                        if (_sa_z1_rpe_med is not None and _sa_z1_rpe_med < 3.5
                                and _sa_z1_dur > 45):
                            _vs += " → considera Sweet Spot"
                _kjh_str = f"{_kjh:.0f}"
                if _kjh_ref and _kjh_ref>0:
                    _kjh_str += f" ({(_kjh-_kjh_ref)/_kjh_ref*100:+.0f}%)"
                _rows.append({
                    "Tipo":       ("★ " if _key==_pk else "  ")+_t["label"],
                    "Estrutura":  _struct,
                    "Watts":      f"{round(_pwr_f)}W",
                    "Work":       f"{_dw:.0f}min",
                    "Total":      f"{_dt:.0f}min",
                    "KJ":         f"{_kj_r:.0f}",
                    "KJ/h":       _kjh_str,
                    "RPE":        _t["rpe_lbl"],
                    "vs sem.ant": _vs,
                })

            if not _rows:
                return None, None, None

            _df_out = pd.DataFrame(_rows)
            _ref_line = ""
            if _ref_kj and _ref_dur:
                _kjh_r = _ref_kj/(_ref_dur/60)
                _ref_line = f"Ref: {_ref_kj:.0f} kJ | {_ref_dur:.0f} min | {_kjh_r:.0f} kJ/h"
                if _sa_z3_kj:
                    _ref_line += f"  |  Sem.ant Z3: {_sa_z3_kj:.0f} kJ"
                    if _sa_z3_dur: _ref_line += f" / {_sa_z3_dur:.0f} min"
            _ol_warn = "⚠️ EM OVERLOAD — power reduzido 5%" if ol else ""
            # Mostrar f_delta no ref_line para transparência
            if f_delta != 1.0:
                _fd_str = f"+{(f_delta-1)*100:.0f}%" if f_delta > 1 else f"{(f_delta-1)*100:.0f}%"
                _ref_line += f"  |  Δ7d/14d: {_fd_str} KJ"
            return _df_out, _ref_line, _ol_warn

        rows_prog = []
        for mod in ['Bike','Row','Ski','Run']:
            _sub = _pf[_pf['type'].apply(norm_tipo)==mod].copy()
            if len(_sub) == 0: continue

            need  = _need_cache.get(mod, 40)
            ol    = _ol_cache.get(mod, False)
            ni    = _ni_cache.get(mod, 50)
            eftp  = _eftp.get(mod)

            # Baseline mediana 8 semanas (só semanas com dados)
            kj_base  = _med_semanas(_sub, '_kj')
            h_base   = _med_semanas(_sub, '_mt')
            km_base  = _med_semanas(_sub, '_km') if '_km' in _sub.columns else 0.0
            has_kj   = kj_base > 0
            has_km   = km_base > 0

            # Fator progressão
            if ol:                fator = 0.98
            elif need > 60:       fator = 1.04
            elif need < 30:       fator = 1.01
            else:                 fator = 1.02

            # Sem anterior (última semana completa)
            sem_ant_ini = sem_ini_pf - pd.Timedelta(weeks=1)
            sem_ant_fim = sem_ini_pf - pd.Timedelta(days=1)
            _sem_ant = _sub[(_sub['Data']>=sem_ant_ini)&(_sub['Data']<=sem_ant_fim)]
            kj_sem_ant = float(_sem_ant['_kj'].sum()) if has_kj else 0.0
            h_sem_ant  = float(_sem_ant['_mt'].sum())

            # Meta semana (com suavização anti-salto)
            kj_meta = min(kj_base * fator, max(kj_sem_ant, kj_base) * 1.08) if has_kj else 0.0
            h_meta  = min(h_base  * fator, max(h_sem_ant,  h_base)  * 1.08)
            km_meta = km_base * fator if has_km else 0.0

            # Informativo: comparação com ano anterior (sem cap — só leitura)
            _ano_ant = _sub[_sub['Data'].dt.year == ano_ant]
            _ano_cur = _sub[_sub['Data'].dt.year == ano_atual]
            h_2025   = float(_ano_ant['_mt'].sum())
            kj_2025  = float(_ano_ant['_kj'].sum()) if has_kj else 0.0
            km_2025  = float(_ano_ant['_km'].sum()) if has_km else 0.0
            h_acum   = float(_ano_cur['_mt'].sum())
            kj_acum  = float(_ano_cur['_kj'].sum()) if has_kj else 0.0
            km_acum  = float(_ano_cur['_km'].sum()) if has_km else 0.0
            h_proj   = h_acum  / dia_ano * 365 if h_acum  > 0 else 0
            kj_proj  = kj_acum / dia_ano * 365 if has_kj and kj_acum > 0 else 0
            cap_atingido = False  # cap removido — só informativo

            # Status horas vs ano anterior (informativo)
            if h_2025 > 0 and h_proj > 0:
                h_delta_pct = (h_proj - h_2025) / h_2025 * 100
                if h_delta_pct < 0:
                    status_ano = f"⚠️ Abaixo ({h_delta_pct:+.0f}% vs {ano_ant})"
                elif h_delta_pct <= 3:
                    status_ano = f"→ Manutenção ({h_delta_pct:+.0f}% vs {ano_ant})"
                elif h_delta_pct <= 12:
                    status_ano = f"✅ No range ({h_delta_pct:+.0f}% vs {ano_ant})"
                else:
                    status_ano = f"📈 Acima ({h_delta_pct:+.0f}% vs {ano_ant})"
            elif h_2025 == 0:
                status_ano = "— (sem ano anterior)"
            else:
                status_ano = "— (sem proj.)"

            # Semana actual (Seg → hoje)
            _sem_cur = _sub[_sub['Data'] >= sem_ini_pf]
            kj_feito = float(_sem_cur['_kj'].sum()) if has_kj else 0.0
            h_feito  = float(_sem_cur['_mt'].sum())
            km_feito = float(_sem_cur['_km'].sum()) if has_km else 0.0

            # Restante real (sem forçar 0 por overload — overload indica-se no display)
            kj_rest_real  = max(0.0, kj_meta - kj_feito) if has_kj else 0.0
            h_rest_real   = max(0.0, h_meta  - h_feito)
            km_rest_real  = max(0.0, km_meta - km_feito) if has_km else 0.0

            # Para o engine de sugestão: overload → não adicionar carga
            if ol:
                kj_rest = 0.0; h_rest = 0.0; km_rest = 0.0
            else:
                kj_rest = kj_rest_real; h_rest = h_rest_real; km_rest = km_rest_real

            # Display do Restante: se overload, mostra aviso em vez de "✅ 0"
            if ol:
                _restante_display = "⚠️ Overload"
            elif has_kj:
                _restante_display = "✅ 0" if kj_rest_real == 0 else f"{kj_rest_real:.0f} kJ"
            elif has_km:
                _restante_display = ("✅ 0" if km_rest_real == 0 and h_rest_real == 0
                                     else f"{km_rest_real:.0f} km | {fmt_dur(h_rest_real)}")
            else:
                _restante_display = fmt_dur(h_rest_real)

            # Sugestão — retorna (df, ref_line, ol_warn)
            # Calcular f_delta para esta modalidade
            _need7_m   = _need7_cache.get(mod, _need_cache.get(mod, 40))
            _need14_m  = _need_cache.get(mod, 40)
            _cut7d = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
            _n7d   = len(_sub[_sub['Data'] >= _cut7d])
            _f_delta_m = _calc_f_delta(_need7_m, _need14_m, ol, _n7d)

            _sug_df, _sug_ref, _sug_ol = _sugestao_sessao(
                kj_rest, h_rest, km_rest, mod, eftp, ni, ol,
                df_hist=_pf, f_delta=_f_delta_m)

            # Fator label
            if ol:       fl = "↓ 0.98 overload"
            elif cap_atingido: fl = "→ 1.00 cap"
            elif need>60: fl = "↑ 1.04"
            elif need<30: fl = "↑ 1.01"
            else:         fl = "↑ 1.02"

            row = {
                'Modalidade':    mod,
                'Métrica':       'KJ' if has_kj else ('KM' if has_km else 'Horas'),
                'Base (med.8s)': f"{kj_base:.0f} kJ" if has_kj else
                                  (f"{km_base:.0f} km | {fmt_dur(h_base)}" if has_km
                                   else fmt_dur(h_base)),
                'Fator':         fl,
                'Meta semana':   f"{kj_meta:.0f} kJ" if has_kj else
                                  (f"{km_meta:.0f} km | {fmt_dur(h_meta)}" if has_km
                                   else fmt_dur(h_meta)),
                'Feito':         f"{kj_feito:.0f} kJ" if has_kj else
                                  (f"{km_feito:.0f} km | {fmt_dur(h_feito)}" if has_km
                                   else fmt_dur(h_feito)),
                'Restante':      _restante_display,
                'Proj. Horas 2026': fmt_dur(h_proj) if h_proj>0 else "—",
                'Range Horas (+3–12%)': (
                    f"{fmt_dur(h_2025)} → {fmt_dur(h_2025*1.03)}–{fmt_dur(h_2025*1.12)}"
                    if h_2025 > 0 else "—"),
                'Status Horas':  status_ano,
                '_sug_df': _sug_df, '_sug_ref': _sug_ref, '_sug_ol': _sug_ol,
            }
            rows_prog.append(row)

        if rows_prog:
            # Remover colunas internas antes de mostrar tabela
            _rows_prog_display = [{k:v for k,v in r.items()
                                   if k not in ("_sug_df","_sug_ref","_sug_ol")}
                                  for r in rows_prog]
            df_prog = pd.DataFrame(_rows_prog_display)

            # ── Deload / Taper detector ───────────────────────────────────
            # baseline = mediana últimas 3 semanas com dados (KJ ou Horas)
            _sem_ini_det  = hoje_pf - pd.Timedelta(weeks=1)
            _sem3_ini_det = hoje_pf - pd.Timedelta(weeks=4)
            _kj_total_cur = 0.0
            _kj_total_b3  = []
            for _m3 in ['Bike','Row','Ski','Run']:
                _s3 = _pf[_pf['type'].apply(norm_tipo)==_m3].copy()
                if len(_s3) == 0: continue
                _s3['_ksem'] = _s3['Data'].dt.to_period('W')
                _agg3 = _s3.groupby('_ksem')['_kj'].sum()
                # Últimas 3 semanas completas (antes da actual)
                _prev3 = _agg3[_agg3.index < _s3['Data'].max().to_period('W')].tail(3)
                _kj_total_b3.extend(_prev3.values)
                # Semana actual desta modalidade
                _kj_total_cur += float(_s3[_s3['Data'] >= sem_ini_pf]['_kj'].sum())

            _kj_b3_mean = float(np.mean(_kj_total_b3)) if _kj_total_b3 else 0.0

            # ── Semana anterior KJ (para guard início de semana) ──────────
            _sem_ant_ini_det = sem_ini_pf - pd.Timedelta(weeks=1)
            _sem_ant_fim_det = sem_ini_pf - pd.Timedelta(days=1)
            _kj_sem_ant_det  = 0.0
            for _m3 in ['Bike','Row','Ski','Run']:
                _s3b = _pf[_pf['type'].apply(norm_tipo)==_m3]
                _kj_sem_ant_det += float(_s3b[
                    (_s3b['Data'] >= _sem_ant_ini_det) &
                    (_s3b['Data'] <= _sem_ant_fim_det)
                ]['_kj'].sum())
            _ratio_sem_ant = (_kj_sem_ant_det / _kj_b3_mean) if _kj_b3_mean > 0 else 1.0

            # Guard: dias decorridos desta semana (Seg=0, Dom=6)
            _dias_semana = hoje_pf.weekday() + 1   # 1=seg, 7=dom
            # KJ projectado para semana completa (pro-rata)
            _kj_proj_semana = (_kj_total_cur / _dias_semana * 7) if _dias_semana > 0 else 0

            # Guard início de semana: se <3 dias decorridos E semana anterior Normal → não julgar
            _semana_iniciando = _dias_semana < 3

            # Ratio a usar: projectado se início de semana, actual se ≥3 dias
            _load_ratio = ((_kj_proj_semana / _kj_b3_mean) if (_kj_b3_mean > 0 and _semana_iniciando)
                           else (_kj_total_cur / _kj_b3_mean) if _kj_b3_mean > 0 else 1.0)

            # Se semana anterior já era Taper/Deload → não alarmar novamente
            _sem_ant_era_taper = _ratio_sem_ant < 0.80

            # Limiares
            _kj_normal_min = _kj_b3_mean * 0.80
            _kj_deload_min = _kj_b3_mean * 0.60
            _kj_taper_min  = _kj_b3_mean * 0.30
            _kj_taper_max  = _kj_b3_mean * 0.60
            _kj_deload_max = _kj_b3_mean * 0.80

            if _kj_b3_mean > 0:
                st.markdown("**📊 Estado de carga — semana actual**")
                _c_det1, _c_det2, _c_det3 = st.columns(3)
                with _c_det1:
                    _kj_show = _kj_proj_semana if _semana_iniciando else _kj_total_cur
                    _lbl_kj  = f"{_kj_show:.0f} kJ {'(proj.)' if _semana_iniciando else ''}"
                    st.metric("Carga actual (semana)", _lbl_kj,
                              f"{(_load_ratio-1)*100:+.0f}% vs baseline")
                with _c_det2:
                    st.metric("Baseline (média 3 sem)", f"{_kj_b3_mean:.0f} kJ",
                              f"Sem.ant: {_ratio_sem_ant*100:.0f}% baseline")
                with _c_det3:
                    if _semana_iniciando and not _sem_ant_era_taper:
                        _fase = "⏳ Início semana"
                    elif _load_ratio >= 0.80:
                        _fase = "✅ Normal"
                    elif _load_ratio >= 0.60:
                        _fase = "📉 Deload"
                    elif _load_ratio >= 0.30:
                        _fase = "🔵 Taper"
                    else:
                        _fase = "⚠️ Muito baixo"
                    st.metric("Fase detectada", _fase, f"ratio={_load_ratio:.2f}")

                # Ranges informativos
                st.caption(
                    f"**Ranges** (baseline={_kj_b3_mean:.0f} kJ/sem) — "
                    f"Normal: >{_kj_normal_min:.0f} | "
                    f"Deload: {_kj_deload_min:.0f}–{_kj_deload_max:.0f} | "
                    f"Taper: {_kj_taper_min:.0f}–{_kj_taper_max:.0f} kJ  "
                    f"{'⏳ Início semana — usando projecção pro-rata' if _semana_iniciando else ''}"
                    f"{'  |  Sem.ant já era Taper/Deload' if _sem_ant_era_taper else ''}")

                # Alertas — só se semana tem dados suficientes OU sem.ant era taper
                if not _semana_iniciando or _sem_ant_era_taper:
                    if _load_ratio < 0.80:
                        if _load_ratio >= 0.60:
                            _taper_cont = " (continuação)" if _sem_ant_era_taper else ""
                            st.info(f"📉 **Deload{_taper_cont}** — volume 20–40% abaixo. "
                                    "Manter intensidade. Duração típica: 3–7 dias.")
                        elif _load_ratio >= 0.30:
                            _taper_cont = " (continuação)" if _sem_ant_era_taper else ""
                            st.info(f"🔵 **Taper{_taper_cont}** — volume 40–70% abaixo. "
                                    "Manter/aumentar intensidade. Duração típica: 7–14 dias.")
                        else:
                            st.warning("⚠️ Carga muito baixa (>70% redução). "
                                       "Confirma se é intencional.")

            st.markdown("---")

            # ── ΔCTL estimado — usa icu_training_load (mesma escala do PMC) ──
            # CTL/ATL calculados com icu_training_load → consistente com Tab PMC
            # ΔCTL/sessão = load_sessão / 42  (definição EMA span=42)
            # Threshold: +1 a +5 CTL por semana (válido na escala TSS-like)
            _pf2 = _pf.copy()

            # Usar icu_training_load se disponível, fallback session_rpe
            _tl_col = None
            if 'icu_training_load' in _pf2.columns:
                _pf2['_load_icu'] = pd.to_numeric(_pf2['icu_training_load'], errors='coerce').fillna(0)
                _tl_col = 'icu_training_load'
            else:
                _pf2['_load_icu'] = (_pf2['_mt'] * 60 *
                    pd.to_numeric(_pf2.get('rpe', pd.Series(dtype=float)), errors='coerce').fillna(5))
                _tl_col = 'session_rpe (fallback)'

            _dates_ctl = pd.date_range(_pf2['Data'].min(), hoje_pf, freq='D')
            _load_ctl  = _pf2.groupby('Data')['_load_icu'].sum().reindex(_dates_ctl, fill_value=0)
            _ctl_hoje  = float(_load_ctl.ewm(span=42, adjust=False).mean().iloc[-1])
            _atl_hoje  = float(_load_ctl.ewm(span=7,  adjust=False).mean().iloc[-1])
            _tsb_hoje  = _ctl_hoje - _atl_hoje

            # ΔCTL por modalidade:
            # Estimar load da próxima sessão = mediana icu_training_load das sessões comparáveis
            # ΔCTL_sessão = load_estimada / 42
            _delta_ctl_total = 0.0
            _delta_rows = []
            for r_p in rows_prog:
                _mod_p = r_p['Modalidade']
                _sug_df_p = r_p.get('_sug_df')

                # Load estimada: mediana das sessões comparáveis desta modalidade
                _load_est = 0.0
                _load_src = "—"
                if 'icu_training_load' in _pf2.columns:
                    _df_mod_p = _pf2[_pf2['type'].apply(norm_tipo) == _mod_p]
                    _ni_p     = _ni_cache.get(_mod_p, 50)
                    # Filtrar por RPE compatível com Need_intensity
                    _rpe_min  = 7 if _ni_p >= 75 else 5 if _ni_p >= 40 else 1
                    _rpe_max  = 10 if _ni_p >= 60 else 7 if _ni_p >= 30 else 5
                    _rpe_f    = pd.to_numeric(_df_mod_p.get('rpe', pd.Series(dtype=float)), errors='coerce')
                    _df_comp  = _df_mod_p[_rpe_f.between(_rpe_min, _rpe_max)]
                    if len(_df_comp) >= 3:
                        _tl_vals  = pd.to_numeric(_df_comp['icu_training_load'], errors='coerce').dropna()
                        if len(_tl_vals) >= 2:
                            _load_est = float(_tl_vals.tail(10).median())
                            _load_src = f"mediana {min(len(_tl_vals),10)} sessões RPE {_rpe_min}–{_rpe_max}"
                    if _load_est == 0:
                        # Fallback: mediana geral desta modalidade
                        _tl_all = pd.to_numeric(_df_mod_p['icu_training_load'], errors='coerce').dropna()
                        if len(_tl_all) >= 2:
                            _load_est = float(_tl_all.tail(10).median())
                            _load_src = "mediana geral (fallback)"

                # ΔCTL = load / 42
                _delta_p = _load_est / 42.0 if _load_est > 0 else 0.0
                _delta_ctl_total += _delta_p
                _delta_rows.append({
                    'Modalidade':    _mod_p,
                    'Load estimada': f"{_load_est:.0f}" if _load_est > 0 else "—",
                    'Fonte':         _load_src,
                    'ΔCTL est.':     f"{_delta_p:+.2f}",
                })

            # CTL/TSB projectados
            _ctl_proj = _ctl_hoje + _delta_ctl_total
            _tsb_proj = _tsb_hoje - _delta_ctl_total

            # Threshold +1 a +5 CTL/semana (escala icu_training_load)
            _dentro_range = 1.0 <= _delta_ctl_total <= 5.0

            st.markdown("**⚡ Impacto CTL estimado — semana actual**")
            st.caption(f"Métrica: **{_tl_col}** — consistente com Tab PMC")
            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            with _cc1: st.metric("CTL actual",     f"{_ctl_hoje:.1f}")
            with _cc2: st.metric("ΔCTL estimado",  f"{_delta_ctl_total:+.2f}",
                                 "✅ no range 1–5" if _dentro_range else
                                 ("⚠️ abaixo de 1" if _delta_ctl_total < 1 else "⚠️ acima de 5"))
            with _cc3: st.metric("CTL projectado", f"{_ctl_proj:.1f}")
            with _cc4: st.metric("TSB projectado", f"{_tsb_proj:.1f}")

            if _delta_rows:
                with st.expander("🔍 Detalhe ΔCTL por modalidade"):
                    st.dataframe(pd.DataFrame(_delta_rows), width="stretch", hide_index=True)
                    st.caption(
                        "Load estimada = mediana icu_training_load das sessões comparáveis (RPE range). "
                        "ΔCTL = load / 42. Threshold: +1 a +5 CTL/semana.")

            st.markdown("---")
            st.dataframe(df_prog,
                         width="stretch", hide_index=True)
            st.markdown("**💡 Sugestões de sessão (semana actual)**")

            # Mini-tabs por modalidade
            _mods_sug = [r for r in rows_prog if r.get("_sug_df") is not None]
            if _mods_sug:
                _emj_map = {"Bike":"🚴 Bike","Row":"🚣 Row","Ski":"🎿 Ski","Run":"🏃 Run"}
                _tab_labels = [_emj_map.get(r["Modalidade"], r["Modalidade"])
                               for r in _mods_sug]
                _sug_tabs = st.tabs(_tab_labels)
                for _stab, r in zip(_sug_tabs, _mods_sug):
                    with _stab:
                        _df_s  = r["_sug_df"]
                        _ref_s = r.get("_sug_ref","")
                        _ol_s  = r.get("_sug_ol","")
                        # Linha de referência + overload
                        if _ref_s: st.caption(_ref_s)
                        if _ol_s:  st.warning(_ol_s)
                        # Tabela de opções — principal marcada com ★ na coluna Tipo
                        # Não usar style (causa texto branco no tema escuro do Streamlit)
                        st.dataframe(
                            _df_s,
                            hide_index=True,
                            use_container_width=True)
                        # KJ restante e meta abaixo da tabela
                        _kj_r_val = r.get("Restante","")
                        _meta_val = r.get("Meta semana","")
                        _feito_val = r.get("Feito","")
                        if _kj_r_val or _meta_val:
                            st.caption(
                                f"Meta semana: **{_meta_val}** | "
                                f"Feito: **{_feito_val}** | "
                                f"Restante: **{_kj_r_val}**")
            st.caption(
                "⚠️ Quantidade de carga: esta camada. Tipo de treino: Need Score acima. "
                "Cap horas +12% vs " + str(ano_ant) + ".")

    st.markdown("---")

    st.markdown("---")

    st.markdown("---")

    # ── Tabela % KM por modalidade ──
    st.subheader("📏 Distribuição de KM por modalidade")
    if 'distance' in da.columns and da['distance'].notna().any():
        _agr_col, _ = st.columns([1, 3])
        _agr = _agr_col.selectbox("Agrupar por", ["Semana", "Mês", "Ano"],
                                   key="vg_agrup_km")
        _code = {"Semana": "W", "Mês": "M", "Ano": "Y"}[_agr]

        df_km = da.copy()
        df_km['_t'] = df_km['type'].apply(norm_tipo)
        df_km = df_km[df_km['_t'] != 'WeightTraining']
        df_km['km'] = pd.to_numeric(df_km['distance'], errors='coerce') / 1000
        df_km = df_km[df_km['km'].notna() & (df_km['km'] > 0)]
        df_km['Data'] = pd.to_datetime(df_km['Data'])
        df_km['_p'] = df_km['Data'].dt.to_period(_code)

        tipos_km = [t for t in ['Bike', 'Row', 'Ski', 'Run']
                    if t in df_km['_t'].unique()]

        if len(df_km) > 0 and tipos_km:
            piv = (df_km.groupby(['_p', '_t'])['km'].sum()
                   .unstack(fill_value=0)
                   .reindex(columns=tipos_km, fill_value=0))
            piv['Total'] = piv[tipos_km].sum(axis=1)

            rows_km = []
            for p, r in piv.sort_index(ascending=False).iterrows():
                if _agr == "Ano":    lbl = str(p.year)
                elif _agr == "Semana": lbl = p.start_time.strftime('%d/%m/%y')
                else: lbl = pd.to_datetime(str(p)).strftime('%B %Y').title()
                row = {'Período': lbl}
                tot = r['Total']
                for t in tipos_km:
                    v = r[t]
                    pct = (v / tot * 100) if tot > 0 else 0
                    row[t] = f"{v:.0f} km ({pct:.0f}%)" if v > 0 else '—'
                row['Total'] = f"{tot:.0f} km"
                rows_km.append(row)

            if rows_km:
                st.dataframe(pd.DataFrame(rows_km),
                             width="stretch", hide_index=True)

    st.markdown("---")

    # ── Atividades Recentes ──
    st.subheader("📋 Atividades Recentes")
    _df_fp = filtrar_principais(da)
    df_tab = (_df_fp.sort_values('Data', ascending=False).head(10)
              if 'Data' in _df_fp.columns and len(_df_fp) > 0
              else _df_fp.head(10))
    if len(df_tab) > 0:
        cs = [c for c in ['Data', 'type', 'name', 'moving_time',
                           'rpe', 'power_avg', 'icu_eftp'] if c in df_tab.columns]
        ds = df_tab[cs].copy()
        if 'moving_time' in ds.columns:
            ds['moving_time'] = ds['moving_time'].apply(
                lambda x: f"{int(x/3600)}h{int((x%3600)/60):02d}m"
                if pd.notna(x) else '—')
        ds.columns = [c.replace('_', ' ').title() for c in ds.columns]
        st.dataframe(ds, width="stretch", hide_index=True)

    st.markdown("---")

    # Resumo Semanal movido para cima (acima da tabela Semana actual)

    # ── Top 10 por Potência ──
    df_rank = filtrar_principais(da).copy()
    if 'power_avg' in df_rank.columns and df_rank['power_avg'].notna().any():
        st.subheader("🏆 Top 10 por Potência")
        top = df_rank.nlargest(10, 'power_avg')[
            ['Data', 'type', 'name', 'power_avg', 'rpe']].copy()
        top['Data'] = pd.to_datetime(top['Data']).dt.strftime('%Y-%m-%d')
        top.columns = ['Data', 'Tipo', 'Nome', 'Power (W)', 'RPE']
        st.dataframe(top, width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC + FTLM
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_pmc.py
# ════════════════════════════════════════════════════════════════════════════
