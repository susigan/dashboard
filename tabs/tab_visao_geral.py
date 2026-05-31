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
    _kj_mc = pd.to_numeric(_kpi_mc.get('icu_joules', pd.Series()), errors='coerce').sum() / 1000 if 'icu_joules' in _kpi_mc.columns and len(_kpi_mc) > 0 else 0
    _kj_mp = pd.to_numeric(_kpi_mp.get('icu_joules', pd.Series()), errors='coerce').sum() / 1000 if 'icu_joules' in _kpi_mp.columns and len(_kpi_mp) > 0 else 0

    # Delta absoluto (não %)
    def _delta_abs_sess(vc, vp):
        if not vp or vp == 0: return None
        d = vc - vp
        return f"{d:+.0f} vs mês ant."

    def _delta_abs_h(vc, vp):
        if not vp or vp == 0: return None
        d = vc - vp
        return f"{d:+.1f}h vs mês ant."

    def _delta_abs_kj(vc, vp):
        if not vp or vp == 0: return None
        d = vc - vp
        return f"{d:+.0f} kJ vs mês ant."

    # Intensidade do mês — RPE / HR / Power
    def _calc_intensidade(df_src):
        """Retorna dicts com distribuição leve/mod/forte por RPE, HR e Power."""
        result = {}
        if len(df_src) == 0:
            return result

        # RPE: Leve≤4 / Mod 4.1-6.9 / Forte≥7
        _rpe = pd.to_numeric(df_src.get('rpe', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(_rpe) > 0:
            n = len(_rpe)
            result['rpe'] = {
                'L': int((_rpe <= 4.0).sum()),
                'M': int(((_rpe > 4.0) & (_rpe < 7.0)).sum()),
                'F': int((_rpe >= 7.0).sum()),
                'n': n,
            }

        # HR: Leve<120 / Mod 120-149 / Forte≥150 bpm (médias por sessão)
        _hr = pd.to_numeric(df_src.get('hr_avg', df_src.get('average_heartrate', pd.Series(dtype=float))), errors='coerce').dropna()
        if len(_hr) > 0:
            n = len(_hr)
            result['hr'] = {
                'L': int((_hr < 120).sum()),
                'M': int(((_hr >= 120) & (_hr < 150)).sum()),
                'F': int((_hr >= 150).sum()),
                'n': n,
            }

        # Power: Leve<0.75IF / Mod 0.75-0.90IF / Forte>0.90IF
        # Proxy: usar IF se disponível, ou usar power_avg vs eFTP
        _if = pd.to_numeric(df_src.get('IF', df_src.get('icu_intensity', pd.Series(dtype=float))), errors='coerce').dropna()
        if len(_if) == 0 and 'power_avg' in df_src.columns and 'icu_eftp' in df_src.columns:
            _pwr = pd.to_numeric(df_src['power_avg'], errors='coerce')
            _ftp = pd.to_numeric(df_src['icu_eftp'],  errors='coerce')
            _if  = (_pwr / _ftp.replace(0, np.nan)).dropna()
        if len(_if) > 0:
            n = len(_if)
            result['pwr'] = {
                'L': int((_if < 0.75).sum()),
                'M': int(((_if >= 0.75) & (_if < 0.90)).sum()),
                'F': int((_if >= 0.90).sum()),
                'n': n,
            }

        return result

    def _fmt_int(d, key):
        """Formata distribuição como 'X%/Y%/Z%' com legenda de cores."""
        if key not in d or d[key]['n'] == 0:
            return '—', ''
        n  = d[key]['n']
        pl = d[key]['L'] / n * 100
        pm = d[key]['M'] / n * 100
        pf = d[key]['F'] / n * 100
        s  = f"{pl:.0f}%/{pm:.0f}%/{pf:.0f}%"
        cap = (f"<span style='color:#2ecc71'>●</span> {d[key]['L']} &nbsp;"
               f"<span style='color:#f39c12'>●</span> {d[key]['M']} &nbsp;"
               f"<span style='color:#e74c3c'>●</span> {d[key]['F']}")
        return s, cap

    _int_mc = _calc_intensidade(_kpi_mc)
    _rpe_str_mc, _rpe_cap_mc = _fmt_int(_int_mc, 'rpe')
    _hr_str_mc,  _hr_cap_mc  = _fmt_int(_int_mc, 'hr')
    _pwr_str_mc, _pwr_cap_mc = _fmt_int(_int_mc, 'pwr')

    # Linha 1: Proj Horas / Status Horas / Sessões / Horas / KJ
    _ct1, _ct2, _ct3, _ct4, _ct5 = st.columns(5)
    # Cards de horas anuais — calculados antecipadamente para o topo
    # (os valores h_proj/h_2025/status_ano são calculados mais abaixo por modalidade;
    #  aqui fazemos um cálculo rápido global para o card do topo)
    try:
        _src_h = (da_full if da_full is not None and len(da_full) > 0 else da).copy()
        _src_h['Data'] = pd.to_datetime(_src_h['Data'])
        _src_h = _src_h[_src_h['type'].apply(norm_tipo) != 'WeightTraining']
        _src_h['_mt_h'] = pd.to_numeric(_src_h['moving_time'], errors='coerce') / 3600
        _hoje_top = pd.Timestamp.now().normalize()
        _dia_ano_top = _hoje_top.timetuple().tm_yday
        _ano_cur_top = _hoje_top.year
        _ano_ant_top = _ano_cur_top - 1
        _h_acum_top  = float(_src_h[_src_h['Data'].dt.year==_ano_cur_top]['_mt_h'].sum())
        _h_2025_top  = float(_src_h[_src_h['Data'].dt.year==_ano_ant_top]['_mt_h'].sum())
        _h_proj_top  = _h_acum_top / _dia_ano_top * 365 if _h_acum_top > 0 else 0
        if _h_2025_top > 0 and _h_proj_top > 0:
            _h_pct_top = (_h_proj_top - _h_2025_top) / _h_2025_top * 100
            _status_top = (f"📈 +{_h_pct_top:.0f}% vs {_ano_ant_top}"
                           if _h_pct_top > 3 else
                           f"📉 {_h_pct_top:.0f}% vs {_ano_ant_top}"
                           if _h_pct_top < -3 else
                           f"→ estável vs {_ano_ant_top}")
            _range_top = f"{fmt_dur(_h_2025_top*1.03)} – {fmt_dur(_h_2025_top*1.12)}"
        else:
            _status_top = "—"; _range_top = "—"
    except Exception:
        _h_proj_top = 0; _h_2025_top = 0; _status_top = "—"; _range_top = "—"

    _ct1.metric("📈 Proj. Horas 2026", fmt_dur(_h_proj_top) if _h_proj_top else "—",
                _status_top)
    _ct2.metric("📊 Range Horas (+3–12%)", _range_top)
    _ct3.metric("🏋️ Sessões (mês)", f"{_sess_mc}",
                _delta_abs_sess(_sess_mc, _sess_mp))
    _ct4.metric("⏱️ Horas (mês)", fmt_dur(_horas_mc) if _horas_mc else "—",
                _delta_abs_h(_horas_mc, _horas_mp))
    _ct5.metric("⚡ KJ (mês)", f"{_kj_mc:.0f}" if _kj_mc else "—",
                _delta_abs_kj(_kj_mc, _kj_mp))

    # Linha 2: Intensidade RPE / HR / Power
    _ci1, _ci2, _ci3 = st.columns(3)
    with _ci1:
        st.metric("Intensidade RPE (L/M/F)", _rpe_str_mc,
                  help="Leve ≤4 / Moderado 4.1–6.9 / Forte ≥7")
        if _rpe_cap_mc:
            st.caption(_rpe_cap_mc + " sess.", unsafe_allow_html=True)
    with _ci2:
        st.metric("Intensidade HR (L/M/F)", _hr_str_mc,
                  help="Leve <120 bpm / Moderado 120–149 bpm / Forte ≥150 bpm")
        if _hr_cap_mc:
            st.caption(_hr_cap_mc + " sess.", unsafe_allow_html=True)
    with _ci3:
        st.metric("Intensidade Power (L/M/F)", _pwr_str_mc,
                  help="Leve IF<0.75 / Moderado IF 0.75–0.89 / Forte IF≥0.90")
        if _pwr_cap_mc:
            st.caption(_pwr_cap_mc + " sess.", unsafe_allow_html=True)

    # HRV e RHR
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

    def _delta_abs_ms(vc, vp):
        if vc is None or vp is None: return None
        return f"{vc-vp:+.0f} ms vs mês ant."

    def _delta_abs_bpm(vc, vp):
        if vc is None or vp is None: return None
        return f"{vc-vp:+.0f} bpm vs mês ant."

    _ch1, _ch2 = st.columns(2)
    _ch1.metric("💚 HRV (mês)", f"{_hrv_mc_v:.0f} ms" if _hrv_mc_v else "—",
                _delta_abs_ms(_hrv_mc_v, _hrv_mp_v))
    _ch2.metric("❤️ RHR (mês)", f"{_rhr_mc_v:.0f} bpm" if _rhr_mc_v else "—",
                _delta_abs_bpm(_rhr_mc_v, _rhr_mp_v))
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

        rows_sw = []  # inicializar — sem actividades esta semana
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

            def _rs_delta_abs_sess(vc, vp):
                if not vp: return None
                return f"{vc-vp:+.0f} vs sem.ant"

            def _rs_delta_abs_h(vc, vp):
                if not vp: return None
                return f"{vc-vp:+.1f}h vs sem.ant"

            def _rs_delta_abs_kj(vc, vp):
                if not vp: return None
                return f"{vc-vp:+.0f} kJ vs sem.ant"

            # RPE distribuição Leve/Moderado/Forte — Leve≤4 / Mod 4.1–6.9 / Forte≥7
            _rs_rpe = pd.to_numeric(_rs_cur.get('rpe', pd.Series()), errors='coerce').dropna() if 'rpe' in _rs_cur.columns else pd.Series()
            _rs_rpe_str = "—"
            _n_leve = _n_mod = _n_forte = 0
            if len(_rs_rpe) > 0:
                _n_total = len(_rs_rpe)
                _n_leve  = int((_rs_rpe <= 4.0).sum())
                _n_mod   = int(((_rs_rpe > 4.0) & (_rs_rpe < 7.0)).sum())
                _n_forte = int((_rs_rpe >= 7.0).sum())
                _p_leve  = _n_leve / _n_total * 100
                _p_mod   = _n_mod  / _n_total * 100
                _p_forte = _n_forte/ _n_total * 100
                _rs_rpe_str = f"{_p_leve:.0f}%/{_p_mod:.0f}%/{_p_forte:.0f}%"

            # HR intensidade (média por sessão): Leve<120 / Mod 120-149 / Forte≥150
            _hr_col = next((c for c in ['hr_avg','average_heartrate'] if c in _rs_cur.columns), None)
            _rs_hr_str = "—"; _rs_hr_cap = ""
            _nh_l = _nh_m = _nh_f = 0
            if _hr_col:
                _rs_hr = pd.to_numeric(_rs_cur[_hr_col], errors='coerce').dropna()
                if len(_rs_hr) > 0:
                    _nh_l = int((_rs_hr < 120).sum())
                    _nh_m = int(((_rs_hr >= 120) & (_rs_hr < 150)).sum())
                    _nh_f = int((_rs_hr >= 150).sum())
                    _nt   = len(_rs_hr)
                    _rs_hr_str = f"{_nh_l/_nt*100:.0f}%/{_nh_m/_nt*100:.0f}%/{_nh_f/_nt*100:.0f}%"
                    _rs_hr_cap = (f"<span style='color:#2ecc71'>●</span> {_nh_l} &nbsp;"
                                  f"<span style='color:#f39c12'>●</span> {_nh_m} &nbsp;"
                                  f"<span style='color:#e74c3c'>●</span> {_nh_f}")

            # Power intensidade via IF: Leve IF<0.75 / Mod 0.75-0.89 / Forte≥0.90
            _rs_pwr_str = "—"; _rs_pwr_cap = ""
            _np_l = _np_m = _np_f = 0
            _if_col = next((c for c in ['IF','icu_intensity'] if c in _rs_cur.columns), None)
            _rs_if = pd.Series(dtype=float)
            if _if_col:
                _rs_if = pd.to_numeric(_rs_cur[_if_col], errors='coerce').dropna()
            elif 'power_avg' in _rs_cur.columns and 'icu_eftp' in _rs_cur.columns:
                _p = pd.to_numeric(_rs_cur['power_avg'], errors='coerce')
                _e = pd.to_numeric(_rs_cur['icu_eftp'],  errors='coerce')
                _rs_if = (_p / _e.replace(0, np.nan)).dropna()
            if len(_rs_if) > 0:
                _np_l = int((_rs_if < 0.75).sum())
                _np_m = int(((_rs_if >= 0.75) & (_rs_if < 0.90)).sum())
                _np_f = int((_rs_if >= 0.90).sum())
                _nt_p = len(_rs_if)
                _rs_pwr_str = f"{_np_l/_nt_p*100:.0f}%/{_np_m/_nt_p*100:.0f}%/{_np_f/_nt_p*100:.0f}%"
                _rs_pwr_cap = (f"<span style='color:#2ecc71'>●</span> {_np_l} &nbsp;"
                               f"<span style='color:#f39c12'>●</span> {_np_m} &nbsp;"
                               f"<span style='color:#e74c3c'>●</span> {_np_f}")

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

            # Layout: linha 1 — Sessões / Horas / KJ (delta absoluto)
            _rs1, _rs2, _rs3 = st.columns(3)
            _rs1.metric("Sessões (sem.)",
                        str(_rs_sess_c),
                        _rs_delta_abs_sess(_rs_sess_c, _rs_sess_p))
            _rs2.metric("Horas (sem.)",
                        fmt_dur(_rs_h_c) if _rs_h_c else "—",
                        _rs_delta_abs_h(_rs_h_c, _rs_h_p))
            _rs3.metric("KJ (sem.)",
                        f"{_rs_kj_c:.0f}" if _rs_kj_c else "—",
                        _rs_delta_abs_kj(_rs_kj_c, _rs_kj_p))

            # Linha 2 — HRV / RHR / CTL
            _rs4, _rs5, _rs6 = st.columns(3)
            _rs4.metric("HRV médio (sem.)",
                        f"{_rs_hrv_c:.0f} ms" if _rs_hrv_c else "—",
                        (f"{_rs_hrv_c-_rs_hrv_p:+.0f} ms vs sem.ant"
                         if _rs_hrv_c and _rs_hrv_p else None))
            _rs5.metric("RHR médio (sem.)",
                        f"{_rs_rhr_c:.0f} bpm" if _rs_rhr_c else "—",
                        (f"{_rs_rhr_c-_rs_rhr_p:+.0f} bpm vs sem.ant"
                         if _rs_rhr_c and _rs_rhr_p else None))
            with _rs6:
                if _rs_ctl_str:
                    st.metric("CTL actual", _rs_ctl_str, _rs_dctl_str)
                    st.caption(_rs_ctl_proj_str or "")

            # Linha 3 — Intensidade RPE / HR / Power (L/M/F)
            _ri1, _ri2, _ri3 = st.columns(3)
            with _ri1:
                st.metric("Intensidade RPE (L/M/F)", _rs_rpe_str,
                          help="Leve ≤4 / Moderado 4.1–6.9 / Forte ≥7")
                if len(_rs_rpe) > 0:
                    st.caption(
                        f"<span style='color:#2ecc71'>●</span> {_n_leve} &nbsp;"
                        f"<span style='color:#f39c12'>●</span> {_n_mod} &nbsp;"
                        f"<span style='color:#e74c3c'>●</span> {_n_forte} sess.",
                        unsafe_allow_html=True)
            with _ri2:
                st.metric("Intensidade HR (L/M/F)", _rs_hr_str,
                          help="Leve <120 bpm / Moderado 120–149 bpm / Forte ≥150 bpm")
                if _rs_hr_cap:
                    st.caption(_rs_hr_cap + " sess.", unsafe_allow_html=True)
            with _ri3:
                st.metric("Intensidade Power (L/M/F)", _rs_pwr_str,
                          help="Leve IF<0.75 / Moderado IF 0.75–0.89 / Forte IF≥0.90")
                if _rs_pwr_cap:
                    st.caption(_rs_pwr_cap + " sess.", unsafe_allow_html=True)

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
            # Use last row for hrv_hoje; use last row WITH bm for classification
            # This ensures today's HRV is classified even if bm not yet propagated
            last_hrv = _wc.iloc[-1]   # actual last HRV entry (today if measured)
            last_bm  = _wc.dropna(subset=['bm']).iloc[-1]  # last with rolling bm
            hrv_hoje = float(last_hrv['hrv']) if pd.notna(last_hrv.get('hrv')) else None
            # Classify using today's LnrMSSD vs the most recent baseline
            _lnr_today = float(last_hrv['LnrMSSD']) if pd.notna(last_hrv.get('LnrMSSD')) else None
            _linf_ref  = float(last_bm['linf']) if pd.notna(last_bm.get('linf')) else None
            _lsup_ref  = float(last_bm['lsup']) if pd.notna(last_bm.get('lsup')) else None
            if _lnr_today is not None and _linf_ref is not None:
                if _linf_ref <= _lnr_today <= _lsup_ref:
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

    # Tenta primeiro wc (wellness form — colunas 'peso'/'fat'/'bf_pct')
    # Fallback: dc (Consolidado_Comida — colunas 'Peso'/'BF')
    _sources = []
    if wc_full is not None and len(wc_full) > 0:
        _wc2 = wc_full.copy()
        _wc2['Data'] = pd.to_datetime(_wc2['Data'])
        _peso_col = next((c for c in ['peso','Peso','weight'] if c in _wc2.columns), None)
        # BF: aceitar FAT, fat, bf_pct, BF, body_fat
        _bf_col   = next((c for c in ['FAT','fat','bf_pct','BF','body_fat']
                          if c in _wc2.columns), None)
        if _peso_col or _bf_col:
            _sources.append((_wc2, _peso_col, _bf_col, 'wellness'))
    if dc is not None and len(dc) > 0:
        _dc2 = dc.copy()
        _dc2['Data'] = pd.to_datetime(_dc2['Data'])
        _dc_peso = next((c for c in ['Peso','peso'] if c in _dc2.columns), None)
        _dc_bf   = next((c for c in ['BF','FAT','fat','bf_pct'] if c in _dc2.columns), None)
        if _dc_peso or _dc_bf:
            _sources.append((_dc2, _dc_peso, _dc_bf, 'food'))

    for _src_df, _pcol, _bcol, _src_name in _sources:
        # Peso
        if _pcol and _pcol in _src_df.columns and peso_7d is None:
            _s = (_src_df[['Data', _pcol]].copy()
                  .assign(**{_pcol: pd.to_numeric(_src_df[_pcol], errors='coerce')})
                  .dropna(subset=[_pcol])
                  .query(f"30 <= {_pcol} <= 200")
                  .sort_values('Data'))
            if len(_s) >= 2:
                _v_atual = float(_s[_pcol].iloc[-1])   # último dia COM dado real
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
            _s = (_src_df[['Data', _bcol]].copy()
                  .assign(**{_bcol: pd.to_numeric(_src_df[_bcol], errors='coerce')})
                  .dropna(subset=[_bcol])
                  .query(f"3 <= {_bcol} <= 50")
                  .sort_values('Data'))
            if len(_s) >= 2:
                _v_atual = float(_s[_bcol].iloc[-1])   # último dia COM dado real
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
            break

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

        # ── Gráfico por modalidade — KJ / KM / Horas: mês actual vs anterior ─
        all_mods_m = sorted(set(list(mes_p.keys()) + list(mes_c.keys())))
        if all_mods_m:
            _lbl_mes_p = mes_p_ini.strftime('%b %Y')
            _lbl_mes_c = mes_c_ini.strftime('%b %Y')
            st.subheader(f"📊 KJ · KM · Horas por Modalidade — {_lbl_mes_p} vs {_lbl_mes_c}")

            # Sub-tabs por métrica
            _gt1, _gt2, _gt3 = st.tabs(["⚡ KJ", "🛣️ KM", "⏱️ Horas"])

            def _bar_chart(metric, unit, color_p, color_c):
                vals_p = [mes_p.get(m,{}).get(metric,0) for m in all_mods_m]
                vals_c = [mes_c.get(m,{}).get(metric,0) for m in all_mods_m]
                if metric == 'horas':
                    txt_p = [fmt_dur(v) if v>0 else '' for v in vals_p]
                    txt_c = [fmt_dur(v) if v>0 else '' for v in vals_c]
                    hover = '%{y}: <b>%{text}</b><extra></extra>'
                else:
                    txt_p = [f"{v:.0f}" if v>0 else '' for v in vals_p]
                    txt_c = [f"{v:.0f}" if v>0 else '' for v in vals_c]
                    hover = f'%{{y}}: <b>%{{x:.0f}} {unit}</b><extra></extra>'
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=all_mods_m, x=vals_p, name=_lbl_mes_p,
                    orientation='h', marker_color=color_p,
                    text=txt_p, textposition='outside',
                    hovertemplate=hover))
                fig.add_trace(go.Bar(
                    y=all_mods_m, x=vals_c, name=f"{_lbl_mes_c} ▶",
                    orientation='h', marker_color=color_c,
                    text=txt_c, textposition='outside',
                    hovertemplate=hover))
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    barmode='group', height=max(200, len(all_mods_m)*65+80),
                    font=dict(color='#222222'),
                    margin=dict(l=60, r=90, t=20, b=20),
                    xaxis=dict(title=unit, showgrid=True, gridcolor='#eeeeee',
                               tickfont=dict(color='#333333')),
                    yaxis=dict(tickfont=dict(color='#333333')),
                    legend=dict(orientation='h', y=1.08,
                                font=dict(color='#111111', size=11)))
                return fig

            with _gt1:
                st.plotly_chart(_bar_chart('kj','kJ','#95a5a6','#2ecc71'),
                                use_container_width=True,
                                config={'displayModeBar':False})
            with _gt2:
                st.plotly_chart(_bar_chart('km','km','#95a5a6','#3498db'),
                                use_container_width=True,
                                config={'displayModeBar':False})
            with _gt3:
                st.plotly_chart(_bar_chart('horas','h','#95a5a6','#e67e22'),
                                use_container_width=True,
                                config={'displayModeBar':False})


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
                # kJ Z3 desta sessão — fracção do trabalho em zona 3 (>90% FTP)
                # Anaeróbio: intervalos curtos acima de 120% FTP — ~40% do total
                # VO2max: 95-110% FTP — ~55% do trabalho total em Z3
                # Threshold: 83-90% FTP — zona limiar, Z2/Z3 fronteiriço ~10%
                # Sweetspot: 78-83% FTP — Z2, minimal Z3 ~3%
                # Leve: Z1 — 0% Z3
                _z3_frac = {
                    'anaerobio': 0.40,
                    'vo2':       0.55,
                    'threshold': 0.10,
                    'sweetspot': 0.03,
                    'leve':      0.00,
                }.get(_key, 0.05)
                _kj_z3_sess = _kj_r * _z3_frac
                _kj_label = (f"{_kj_r:.0f} ({_kj_z3_sess:.0f} Z3)"
                             if _kj_z3_sess >= 1 else f"{_kj_r:.0f}")

                _rows.append({
                    "Tipo":       ("★ " if _key==_pk else "  ")+_t["label"],
                    "Estrutura":  _struct,
                    "Watts":      f"{round(_pwr_f)}W",
                    "Work":       f"{_dw:.0f}min",
                    "Total":      f"{_dt:.0f}min",
                    "KJ (z3 kJ)": _kj_label,
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

        # ── κ actual — lido do session_state (populado pelo tab_pmc via calcular_series_carga) ──
        # Os percentis são calculados DINAMICAMENTE sobre a série histórica real do atleta.
        # Fallback: se tab_pmc ainda não correu nesta sessão, calcula aqui (sem cache interno).
        _kappa_s           = None   # série completa de κ
        _kappa_now         = None   # valor actual
        _kappa_pct         = None   # percentil actual no histórico do atleta
        _kappa_consec_alert = 0     # dias consecutivos com κ > p87

        # Percentis dinâmicos — inicializam com valores do histórico do atleta
        # (serão recalculados abaixo assim que tivermos a série real)
        _KAPPA_P25 = 3.954
        _KAPPA_P75 = 5.954
        _KAPPA_P87 = 7.182

        # 1ª tentativa: session_state (populado pelo tab_pmc quando é visitado)
        _ld_ss = st.session_state.get('ld_frac_cache', None)
        if _ld_ss is not None and 'FMT_kappa' in _ld_ss.columns:
            _kappa_s = pd.to_numeric(_ld_ss['FMT_kappa'], errors='coerce').dropna()

        # 2ª tentativa: calcular directamente (tab_pmc ainda não foi visitada)
        if _kappa_s is None or len(_kappa_s) < 10:
            try:
                _ld_direct, _ = calcular_series_carga(
                    da_full, df_wellness=wc_full, ate_hoje=True)
                if _ld_direct is not None and 'FMT_kappa' in _ld_direct.columns:
                    _kappa_s = pd.to_numeric(
                        _ld_direct['FMT_kappa'], errors='coerce').dropna()
                    # Guardar no session_state para evitar recálculo noutras tabs
                    st.session_state['ld_frac_cache'] = _ld_direct
            except Exception:
                _kappa_s = None

        # Calcular percentis REAIS do atleta e estado actual
        if _kappa_s is not None and len(_kappa_s) >= 10:
            # Percentis dinâmicos sobre o histórico real
            _KAPPA_P25 = float(_kappa_s.quantile(0.25))
            _KAPPA_P75 = float(_kappa_s.quantile(0.75))
            _KAPPA_P87 = float(_kappa_s.quantile(0.87))

            _kappa_now = float(_kappa_s.dropna().iloc[-1]) if _kappa_s.notna().any() else 0.0
            _kappa_pct = float((_kappa_s < _kappa_now).mean() * 100)

            # Dias consecutivos acima de p87
            for _kv in _kappa_s.iloc[::-1]:
                if _kv > _KAPPA_P87:
                    _kappa_consec_alert += 1
                else:
                    break

        # ── ALERTA AUTOMÁTICO κ > p87 por 3+ dias consecutivos ─────────────────
        if _kappa_consec_alert >= 3:
            st.error(
                f"⚠️ **Alerta FMT Tensor** — κ acima do limiar de sobrecarga "
                f"(>{_KAPPA_P87:.2f}, p87 do histórico do atleta) "
                f"há **{_kappa_consec_alert} dias consecutivos**.\n\n"
                f"κ actual: **{_kappa_now:.3f}** (p{_kappa_pct:.0f} do teu histórico). "
                f"Sugestão: reduzir volume total 15-20% e eliminar sessões intervaladas por 5-7 dias."
            )
        elif _kappa_now is not None and _kappa_now > _KAPPA_P75:
            st.warning(
                f"⚠️ **κ elevado** ({_kappa_now:.3f}, p{_kappa_pct:.0f} do teu histórico) — "
                f"fadiga silenciosa possível. {_kappa_consec_alert} dia(s) acima de p87 "
                f"({_KAPPA_P87:.2f})."
            )

        # ── Carregar α do Modelo 2 (FTLM Polar) ───────────────────────────────
        # Primeiro tenta o cache (populado pelo tab_eftp).
        # Se vazio (utilizador ainda não visitou tab_eftp), calcula aqui.
        _alpha_p = st.session_state.get('alpha_polar_cache', {})
        if not _alpha_p:
            try:
                from utils.data import calcular_alpha_polar as _cap
                _gamma_map_vg = st.session_state.get('gamma_map', {})
                _alpha_p = _cap(_pf, gamma_map=_gamma_map_vg)
                if _alpha_p:
                    st.session_state['alpha_polar_cache'] = _alpha_p
            except Exception:
                _alpha_p = {}

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

            # Fator progressão — calibrado com κ e dados reais do atleta
            # Percentis κ: p25=3.954 p75=5.954 p87=7.182
            # Base empírica: análise de fator_real × rec_next semana seguinte (299 semanas)
            if ol:
                # Overload confirmado: -15% (dados mostram rec_next=56.8 em κ>p87,
                # fator real em semanas de mau recovery = 0.880)
                fator = 0.85
            elif _kappa_now is not None and _kappa_now > _KAPPA_P87:
                # Regra de segurança absoluta: κ > p87 força redução independente do need
                fator = 0.90
            elif _kappa_now is not None and _kappa_now > _KAPPA_P75:
                # κ alto sobrepõe need — conter progressão
                if need > 60:   fator = 0.98  # need alto mas κ alto — prevalecer cautela
                else:           fator = 0.97
            elif _kappa_now is not None and _kappa_now < _KAPPA_P25:
                # κ baixo = máxima janela de ganho fisiológico
                if need > 60:   fator = 1.08  # melhor janela: empurrar aqui
                elif need > 30: fator = 1.05  # espaço de ganho moderado
                else:           fator = 1.00  # manutenção mesmo com κ baixo
            else:
                # κ na zona média (p25-p75) — lógica original com ajustes
                if need > 60:   fator = 1.04  # confirmado pelos dados
                elif need < 30: fator = 0.97  # conter quando need baixo + κ médio
                else:           fator = 1.02  # manter (dados confirmam)

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

            # ── Alvo de zonas (FTLM Polar) ────────────────────────────────────
            _ap_mod = _alpha_p.get(mod, {})
            _zone_prescription = None
            if _ap_mod.get('ok'):
                # HRV-Guided define o tecto de intensidade
                _hrv_class_local = hrv_class if 'hrv_class' in dir() else 'Sem dados'
                # Recuperação → Z1, Z2 curto, ou Descanso (nao Z3)
                # HIIT        → Z2 (threshold/sweetspot) ou Z3 (VO2max/anaer)
                if _hrv_class_local == 'Recuperação':
                    _zona_permitidas = ['Z1', 'Z2']
                    _zona_max        = 'Z2'
                    _zona_primaria   = 'Z1'
                    _hrv_note        = "HRV↓ Recuperação — Z1 prioritário, Z2 curto ou Descanso"
                elif _hrv_class_local == 'HIIT':
                    _zona_permitidas = ['Z2', 'Z3']
                    _zona_max        = 'Z3'
                    _zona_primaria   = 'Z2' if ni < 60 else 'Z3'
                    _hrv_note        = "HRV✓ HIIT — Z2 (threshold/sweetspot) ou Z3 (VO2max)"
                else:
                    _zona_permitidas = ['Z1', 'Z2', 'Z3']
                    _zona_max        = 'Z3' if ni >= 70 else 'Z2'
                    _zona_primaria   = 'Z2'
                    _hrv_note        = "HRV sem dados — Z2 por defeito"

                # kJ actuais vs kJ necessários (alvo 3 meses)
                _alvo_3m = _ap_mod.get('alvos', {}).get('3m', {})
                _kj_z3_need = _alvo_3m.get('kj_z3_semana', 0)
                _kj_z2_need = _alvo_3m.get('kj_z2_semana', 0)
                _kj_z1_need = _alvo_3m.get('kj_z1_semana', 0)
                _kj_z3_act  = _ap_mod.get('kj_z3_semana_actual', 0)
                _kj_z2_act  = _ap_mod.get('kj_z2_semana_actual', 0)
                _kj_z1_act  = _ap_mod.get('kj_z1_semana_actual', 0)
                _eftp_tgt   = _alvo_3m.get('eftp_proj', eftp)
                _delta_tgt  = _alvo_3m.get('delta_w', 0)

                # Gap entre actual e necessário (para selecção de zona)
                _gap_z3 = max(0, _kj_z3_need - _kj_z3_act)
                _gap_z2 = max(0, _kj_z2_need - _kj_z2_act)

                # Feito esta semana e alvo desta semana (do plano DB)
                _z3_feito_sem = 0.0  # feito esta semana (real)
                _z3_alvo_sem  = _kj_z3_need  # fallback: alvo M2 estático
                try:
                    from utils.plano_db import get_semana_atual as _gsa
                    _si = _gsa(mod)
                    _sd = _si.get('dados')
                    if _sd:
                        _z3_feito_sem = float(_sd.get('kj_z3_feito', 0) or 0)
                        _z3_alvo_sem  = float(_sd.get('kj_z3_alvo',  _kj_z3_need) or _kj_z3_need)
                    # Actualizar feito com dados reais se tiver colunas Z3
                    _col_z3_inline = next((c for c in ['Z3KJ','z3_kj','z3kj'] if c in _pf.columns), None)
                    if _col_z3_inline and _si.get('semana_iso'):
                        _pf_sem_inline = _pf[
                            (_pf['type'].apply(norm_tipo)==mod) &
                            (_pf['Data'] >= pd.Timestamp(_si['semana_iso']))]
                        _z3_feito_sem = float(pd.to_numeric(
                            _pf_sem_inline[_col_z3_inline],errors='coerce').fillna(0).sum())
                except Exception:
                    pass

                _gap_z3_sem = max(0, _z3_alvo_sem - _z3_feito_sem)  # gap real desta semana

                # Selecção de zona — usa gap desta semana
                if ol or (_kappa_now is not None and _kappa_now > _KAPPA_P87):
                    _zona_semana = 'Z1'
                    _zona_note   = "κ alto/overload → Z1 (independente HRV)"
                elif _gap_z3_sem > 5 and 'Z3' in _zona_permitidas:
                    _zona_semana = 'Z3'
                    _zona_note   = f"Gap Z3: {_gap_z3_sem:.0f} kJ/sem | {_hrv_note}"
                elif _gap_z3_sem > 5 and 'Z3' not in _zona_permitidas:
                    _zona_semana = 'Z2' if 'Z2' in _zona_permitidas else 'Z1'
                    _zona_note   = f"Gap Z3 ({_gap_z3_sem:.0f} kJ) mas HRV→{_zona_semana} | {_hrv_note}"
                elif _gap_z2 > 5 and 'Z2' in _zona_permitidas:
                    _zona_semana = 'Z2'
                    _zona_note   = f"Gap Z2: {_gap_z2:.0f} kJ/sem | {_hrv_note}"
                else:
                    _zona_semana = _zona_primaria
                    _zona_note   = f"Zonas no alvo — {_hrv_note}"

                _zone_prescription = {
                    'zona_primaria':  _zona_semana,
                    'nota':           _zona_note,
                    'hrv_note':       _hrv_note,
                    'r2':             _ap_mod.get('r2', 0),
                    'kj_z3_feito':    _z3_feito_sem,   # feito esta semana
                    'kj_z3_alvo':     _z3_alvo_sem,    # alvo desta semana (plano DB)
                    'kj_z3_act':      _kj_z3_act,      # histórico 4sem (para gap M2)
                    'kj_z3_need':     _kj_z3_need,
                    'kj_z2_act':      _kj_z2_act,
                    'kj_z2_need':     _kj_z2_need,
                    'kj_z1_act':      _kj_z1_act,
                    'kj_z1_need':     _kj_z1_need,
                    'eftp_tgt':       _eftp_tgt,
                    'delta_tgt':      _delta_tgt,
                    'gap_z3':         _gap_z3_sem,
                    'gap_z2':         _gap_z2,
                    'mmp_label':      _ap_mod.get('mmp_label','MMP'),
                }

            # Fator label
            if ol:       fl = "↓ 0.98 overload"
            elif cap_atingido: fl = "→ 1.00 cap"
            elif need>60: fl = "↑ 1.04"
            elif need<30: fl = "↑ 1.01"
            else:         fl = "↑ 1.02"

            # Zone targets for row display
            _z3_lbl = 'a calcular...'; _z_alvo_eftp = 'a calcular...'; _z_zona_rec = '—'
            if _zone_prescription:
                _zp = _zone_prescription
                _r2_z = _zp.get('r2', 0)
                _r2_icon = '🟢' if _r2_z >= 0.20 else ('🟡' if _r2_z >= 0.08 else '🔴')
                # feito esta semana / alvo desta semana / gap restante
                _z3f = _zp['kj_z3_feito']
                _z3a = _zp['kj_z3_alvo']
                _z3g = _zp['gap_z3']
                _pct_z3 = min(100, _z3f / max(_z3a, 1) * 100)
                _z3_lbl = (f"feito:{_z3f:.0f} alvo:{_z3a:.0f} kJ "
                           f"({'✅' if _z3g == 0 else f'falta:{_z3g:.0f}'}) "
                           f"{_pct_z3:.0f}%")
                _z_alvo_eftp = (f"{_zp['eftp_tgt']:.0f}W ({_zp['delta_tgt']:+.0f}W 3m) "
                                f"{_r2_icon}R²={_r2_z:.2f}")
                _z_zona_rec = f"{_zp['zona_primaria']} — {_zp['nota'][:40]}"

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
                'Zona (esta sem)': _z_zona_rec,
                'kJ Z3 act→alvo': _z3_lbl,
                'eFTP alvo 3m':  _z_alvo_eftp,
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

            # ── CTL real desta semana — usa icu_training_load real ────────
            _pf2 = _pf.copy()
            if 'icu_training_load' in _pf2.columns:
                _pf2['_load_icu'] = pd.to_numeric(_pf2['icu_training_load'], errors='coerce').fillna(0)
            else:
                _pf2['_load_icu'] = 0.0

            # CTL e ATL reais — série histórica completa
            _dates_ctl  = pd.date_range(_pf2['Data'].min(), hoje_pf, freq='D')
            _load_ctl   = _pf2.groupby('Data')['_load_icu'].sum().reindex(_dates_ctl, fill_value=0)
            _ctl_serie  = _load_ctl.ewm(span=42, adjust=False).mean()
            _atl_serie  = _load_ctl.ewm(span=7,  adjust=False).mean()
            _ctl_hoje   = float(_ctl_serie.iloc[-1])
            _atl_hoje   = float(_atl_serie.iloc[-1])
            _tsb_hoje   = _ctl_hoje - _atl_hoje

            # CTL e ATL na segunda-feira desta semana (início da semana)
            _ctl_seg = float(_ctl_serie.get(sem_ini_pf, _ctl_serie.iloc[-1]))
            _atl_seg = float(_atl_serie.get(sem_ini_pf, _atl_serie.iloc[-1]))

            # CTL semana anterior (domingo passado)
            _sem_ant_dom = sem_ini_pf - pd.Timedelta(days=1)
            _ctl_sem_ant = float(_ctl_serie.get(_sem_ant_dom, _ctl_serie.iloc[-1]))
            _atl_sem_ant = float(_atl_serie.get(_sem_ant_dom, _atl_serie.iloc[-1]))

            # Variação real desta semana
            _dctl_real = _ctl_hoje - _ctl_seg
            _datl_real = _atl_hoje - _atl_seg

            # CTL real por modalidade esta semana
            _ctl_mod_rows = []
            for _mod_p in ['Bike','Row','Ski','Run']:
                _df_mod_sem = _pf2[
                    (_pf2['type'].apply(norm_tipo) == _mod_p) &
                    (_pf2['Data'] >= sem_ini_pf)
                ]
                _load_mod = float(_df_mod_sem['_load_icu'].sum()) if len(_df_mod_sem) > 0 else 0
                _sess_mod = len(_df_mod_sem)
                if _load_mod > 0 or _sess_mod > 0:
                    _ctl_mod_rows.append({
                        'Modalidade':       _mod_p,
                        'Sessões sem.':     _sess_mod,
                        'Load total sem.':  f"{_load_mod:.0f}",
                        'Contribuição CTL': f"{_load_mod/42:.2f}",
                    })

            # Mostrar
            st.markdown("**📈 CTL — semana actual**")
            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            with _cc1:
                st.metric("CTL actual",
                          f"{_ctl_hoje:.1f}",
                          f"{_dctl_real:+.2f} vs início sem.")
            with _cc2:
                st.metric("CTL sem. anterior",
                          f"{_ctl_sem_ant:.1f}",
                          f"{_ctl_hoje - _ctl_sem_ant:+.2f} variação")
            with _cc3:
                st.metric("ATL actual",
                          f"{_atl_hoje:.1f}",
                          f"{_datl_real:+.2f} vs início sem.")
            with _cc4:
                st.metric("TSB actual",
                          f"{_tsb_hoje:.1f}",
                          f"{'⚡ Fresco' if _tsb_hoje > 5 else '😴 Fatigado' if _tsb_hoje < -10 else '✅ Neutro'}")

            # CTL por modalidade esta semana
            if _ctl_mod_rows:
                with st.expander("🔍 CTL por modalidade — semana actual"):
                    st.dataframe(pd.DataFrame(_ctl_mod_rows),
                                 hide_index=True, use_container_width=True)
                    st.caption("Load = soma icu_training_load. "
                               "Contribuição CTL = load/42 (mesma escala do PMC).")

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

            # ── Planeador de Progressão Z3 com DB ────────────────────────
            st.markdown("---")
            st.markdown("**📅 Planeador de Progressão Z3 — por modalidade**")
            st.caption("Plano criado automaticamente. Muda prazo ou eFTP alvo → novo plano.")

            _pc1, _pc2, _pc3 = st.columns(3)
            _prazo_sem = _pc1.slider("Prazo (semanas)", 4, 24, 12, 2,
                                     key="prazo_sem_planeador")
            _delta_eftp_input = _pc2.slider("eFTP alvo (ganho em W)", 0, 30, 10, 1,
                                            key="delta_eftp_planeador")
            _sem_override = _pc3.number_input("Semana actual (auto)", 1, _prazo_sem, 1,
                                              key="sem_override_planeador",
                                              help="Auto-detectado. Ajusta só se necessário.")

            _plan_rows = []
            for _mv_p in ['Bike','Row','Ski','Run']:
                _ap_p = _alpha_p.get(_mv_p, {})
                if not _ap_p.get('ok'): continue
                _eftp_p   = _ap_p.get('eftp_now', 0)
                _a3p      = _ap_p.get('alpha_z3', 0)
                _a2p      = _ap_p.get('alpha_z2', 0)
                _a1p      = _ap_p.get('alpha_z1', 0)
                _intcp    = _eftp_p - (_a3p*_ap_p.get('cz3_now',0)
                                       + _a2p*_ap_p.get('cz2_now',0)
                                       + _a1p*_ap_p.get('cz1_now',0))
                _kj3_act  = _ap_p.get('kj_z3_semana_actual', 0)
                _kj2_act  = _ap_p.get('kj_z2_semana_actual', 0)
                _kj1_act  = _ap_p.get('kj_z1_semana_actual', 0)
                _cz3_now  = _ap_p.get('cz3_now', 0)
                _cz2_now  = _ap_p.get('cz2_now', 0)
                _cz1_now  = _ap_p.get('cz1_now', 0)
                _r2_p     = _ap_p.get('r2', 0)
                _r2_icon  = '🟢' if _r2_p >= 0.20 else ('🟡' if _r2_p >= 0.08 else '🔴')
                _eftp_tgt_p = _eftp_p + _delta_eftp_input

                # CTLγ_Z3 alvo
                if abs(_a3p) > 0.01:
                    _cz3_tgt = (_eftp_tgt_p - _a2p*_cz2_now - _a1p*_cz1_now - _intcp) / _a3p
                    _cz3_tgt = max(_cz3_tgt, _cz3_now)
                else:
                    _cz3_tgt = _cz3_now * 1.10
                # Inversão correcta CTLγ → kJ/semana via EWM
                # α_EWM = 2/(span+1); kJ_diário = CTLγ × α_EWM; kJ_sem = × 7
                _span_p = _ap_p.get('span', 28)
                _alpha_ewm_p = 2.0 / (_span_p + 1)
                _kj3_alvo_final = float(_cz3_tgt) * _alpha_ewm_p * 7
                # Sanity: max 3× o actual (evita valores irreais)
                _kj3_act_safe = _kj3_act if _kj3_act > 5 else 0
                if _kj3_act_safe > 0:
                    _kj3_alvo_final = min(_kj3_alvo_final, _kj3_act_safe * 3.0)

                # Ler/criar plano no DB
                _z3_feito_show = 0
                _sem_num_show  = 1
                _concluido     = False
                try:
                    from utils.plano_db import (get_semana_atual, plano_mudou,
                                                criar_plano, actualizar_feito)
                    if plano_mudou(_mv_p, _prazo_sem, _delta_eftp_input):
                        criar_plano(
                            modalidade=_mv_p, prazo_semanas=_prazo_sem,
                            eftp_alvo_delta=_delta_eftp_input, eftp_actual=_eftp_p,
                            kj_z3_inicial=_kj3_act, kj_z2_inicial=_kj2_act,
                            kj_z1_inicial=_kj1_act,
                            alpha_z3=_a3p, alpha_z2=_a2p, alpha_z1=_a1p,
                            intercept=_intcp,
                            cz3_now=_cz3_now, cz2_now=_cz2_now, cz1_now=_cz1_now,
                            span=_ap_p.get('span', 28),
                            r2_modelo=_ap_p.get('r2', 0.0))
                    _sem_info     = get_semana_atual(_mv_p)
                    _sem_num_db   = _sem_info.get('semana_num') or 1
                    # Usar override do user se for diferente do auto-detectado
                    _sem_num_show = int(_sem_override) if _sem_override != _sem_num_db else _sem_num_db
                    _concluido    = _sem_num_show > _prazo_sem
                    _dados        = _sem_info.get('dados')
                    # Actualizar feito com dados reais desta semana
                    _col_z3_p = next((c for c in ['Z3KJ','z3_kj','z3kj'] if c in _pf.columns), None)
                    _col_z2_p = next((c for c in ['Z2KJ','z2_kj','z2kj'] if c in _pf.columns), None)
                    _col_z1_p = next((c for c in ['Z1KJ','z1_kj','z1kj'] if c in _pf.columns), None)
                    # Segunda-feira da semana actual (baseada no plano)
                    _plano_info = _sem_info.get('plano') or {}
                    _semana_inicio_str = _plano_info.get('semana_inicio', '')
                    try:
                        from datetime import date, timedelta
                        _d_ini_p = date.fromisoformat(_semana_inicio_str) if _semana_inicio_str else date.today()
                        _seg_atual = str(_d_ini_p + timedelta(weeks=_sem_num_show - 1))
                        _seg_fim   = str(_d_ini_p + timedelta(weeks=_sem_num_show))
                    except Exception:
                        _seg_atual = _sem_info.get('semana_iso', '')
                        _seg_fim   = ''
                    _sem_iso_p = _seg_atual
                    if _col_z3_p and _sem_iso_p:
                        _pf_sem = _pf[
                            (_pf['type'].apply(norm_tipo)==_mv_p) &
                            (_pf['Data'] >= pd.Timestamp(_sem_iso_p)) &
                            (_pf['Data'] < pd.Timestamp(_seg_fim) if _seg_fim else True)]
                        _z3f = float(pd.to_numeric(_pf_sem[_col_z3_p],errors='coerce').fillna(0).sum()) if len(_pf_sem)>0 else 0
                        _z2f = float(pd.to_numeric(_pf_sem[_col_z2_p],errors='coerce').fillna(0).sum()) if len(_pf_sem)>0 and _col_z2_p else 0
                        _z1f = float(pd.to_numeric(_pf_sem[_col_z1_p],errors='coerce').fillna(0).sum()) if len(_pf_sem)>0 and _col_z1_p else 0
                        _eftp_r = float(_pf[_pf['type'].apply(norm_tipo)==_mv_p]['icu_eftp'].dropna().iloc[-1]) if 'icu_eftp' in _pf.columns else None
                        actualizar_feito(_mv_p, _sem_iso_p, _z3f, _z2f, _z1f, _eftp_r)
                        if _dados: _dados['kj_z3_feito'] = _z3f
                    _z3_feito_show = float(_dados['kj_z3_feito']) if _dados and _dados.get('kj_z3_feito') else 0
                except Exception:
                    pass

                # Alvo desta semana (rampa linear)
                _frac = _sem_num_show / _prazo_sem
                _kj3_esta_sem = _kj3_act + (_kj3_alvo_final - _kj3_act) * _frac
                _z3_pct = min(100, _z3_feito_show / max(_kj3_esta_sem, 1) * 100)
                _sem_lbl = "✅ Concluído" if _concluido else f"Sem {_sem_num_show}/{_prazo_sem}"

                _plan_rows.append({
                    'Modalidade':         _mv_p,
                    'eFTP actual':        f"{_eftp_p:.0f}W",
                    f'+{_delta_eftp_input}W alvo': f"{_eftp_tgt_p:.0f}W",
                    'Z3 histórico':       f"{_kj3_act:.0f} kJ/sem",
                    'Z3 esta sem (alvo)': f"{_kj3_esta_sem:.0f} kJ",
                    'Z3 feito':           f"{_z3_feito_show:.0f} kJ ({_z3_pct:.0f}%)",
                    'Z3 alvo final':      f"{_kj3_alvo_final:.0f} kJ/sem",
                    'Semana':             _sem_lbl,
                    'R²':                 f"{_r2_icon}{_r2_p:.2f}",
                })

            if _plan_rows:
                st.dataframe(pd.DataFrame(_plan_rows), hide_index=True,
                             use_container_width=True)
                st.caption(f"Rampa linear {_prazo_sem} semanas. "
                           "Semana detectada automaticamente pelo DB. "
                           "Novo plano ao mudar prazo ou eFTP alvo. "
                           "🔴R²<0.08 = modelo pouco fiável.")

            # Contexto fisiológico dTRIMP/dKJ
            _eff_kj = st.session_state.get('eff_kj_cache', {})
            if _eff_kj:
                st.markdown("**Contexto fisiológico para aumentar Z3:**")
                _ctx_rows = []
                for _mv_ctx in ['Bike','Row','Ski','Run']:
                    _ec = _eff_kj.get(_mv_ctx, {})
                    if not _ec: continue
                    _ctx_rows.append({
                        'Modal.':        _mv_ctx,
                        'dTRIMP/dKJ':    f"{_ec['dtrimp_dkj']:.3f} {_ec['dtrimp_lbl']}",
                        'Eff delta 28d': f"{_ec['eff_delta']:+.1%} {_ec['eff_delta_lbl']}",
                        'Aumentar Z3?':  '✅ Seguro' if _ec.get('aumentar_z3_ok') else '⚠️ Cautela',
                    })
                if _ctx_rows:
                    st.dataframe(pd.DataFrame(_ctx_rows), hide_index=True,
                                 use_container_width=True)

            # Histórico completo
            try:
                from utils.plano_db import get_historico_plano
                with st.expander("📊 Histórico do plano — todas as semanas"):
                    _ht1,_ht2,_ht3,_ht4 = st.tabs(['Bike','Row','Ski','Run'])
                    for _htab, _mh in zip([_ht1,_ht2,_ht3,_ht4],['Bike','Row','Ski','Run']):
                        with _htab:
                            _df_hist_p = get_historico_plano(_mh)
                            if not _df_hist_p.empty:
                                _df_hist_p['% Z3'] = (
                                    _df_hist_p['kj_z3_feito'] /
                                    _df_hist_p['kj_z3_alvo'].replace(0,1)*100
                                ).round(0).astype(int).astype(str)+'%'
                                st.dataframe(
                                    _df_hist_p[['semana_num','semana_iso',
                                                'kj_z3_alvo','kj_z3_feito','% Z3',
                                                'kj_z2_alvo','kj_z2_feito','eftp_real']],
                                    hide_index=True, use_container_width=True)
                            else:
                                st.info(f"Sem plano activo para {_mh}.")
            except Exception:
                pass


    # ════════════════════════════════════════════════════════════════════════
    # ÍNDICE DE MONOTONIA DE FRY — Card de alerta na Visão Geral
    # Fry RW et al. (1992). Periodisation and Prevention of Overtraining.
    # IM = carga_média_7d / std_carga_7d (dias com treino=kJ, dias sem=0)
    # IM > 2.0 → supressão imune provável | IM > 1.5 → monitorizar
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔄 Monotonia de Treino — Índice de Fry (7 dias)")
    st.caption(
        "Fry et al. (1992) | Seiler & Tønnessen (2009). "
        "IM = carga média diária / desvio padrão da carga. "
        "Dias sem treino contam como 0 kJ — o descanso reduz a monotonia."
    )

    _src_fry = da_full if da_full is not None and len(da_full) > 0 else da
    if _src_fry is not None and len(_src_fry) > 0:
        # Usar mesma pipeline do tab_volume: filtrar_principais + add_tempo + norm_tipo
        # Garante mesma lógica de filtragem e mesmos nomes de colunas
        _fry_df = filtrar_principais(_src_fry).copy()
        _fry_df = add_tempo(_fry_df)
        _fry_df['Data'] = pd.to_datetime(_fry_df['Data']).dt.normalize()
        _fry_df['type'] = _fry_df['type'].apply(norm_tipo)
        _CICLICOS_FRY = ['Bike', 'Run', 'Row', 'Ski']
        _fry_df = _fry_df[_fry_df['type'].isin(_CICLICOS_FRY)]

        # kJ por dia (soma de todas modalidades) — dias sem treino = 0
        _kj_col = next((c for c in ['icu_joules','joules','kj_total']
                        if c in _fry_df.columns and _fry_df[c].notna().any()), None)
        if _kj_col:
            _fry_daily_raw = _fry_df.groupby('Data')[_kj_col].sum()
            # Converter J → kJ se necessário (uma vez, sobre a série inteira)
            if _kj_col == 'icu_joules':
                _fry_daily_raw = _fry_daily_raw / 1000

            # Reindex para calendário completo — dias sem treino = 0 (não NaN)
            _fry_idx = pd.date_range(_fry_daily_raw.index.min(),
                                      pd.Timestamp.now().normalize(), freq='D')
            _fry_daily = _fry_daily_raw.reindex(_fry_idx, fill_value=0)

            # IM = média 7d / std 7d
            _fry_m7 = _fry_daily.rolling(7, min_periods=4).mean()
            _fry_s7 = _fry_daily.rolling(7, min_periods=4).std()
            _fry_im = (_fry_m7 / _fry_s7.replace(0, np.nan)).round(2)

            # Training Strain = IM × carga_média
            _fry_strain = (_fry_im * _fry_m7).round(1)

            _im_hoje    = float(_fry_im.iloc[-1])  if _fry_im.notna().any()    else None
            _m7_hoje    = float(_fry_m7.iloc[-1])  if _fry_m7.notna().any()    else None
            _strain_hoje= float(_fry_strain.iloc[-1]) if _fry_strain.notna().any() else None
            _im_7d_ant  = float(_fry_im.iloc[-8])  if len(_fry_im) > 8 and pd.notna(_fry_im.iloc[-8]) else None

            # Semáforo
            if _im_hoje is None:
                _im_cor = "#888"; _im_emoji = "⬜"; _im_lbl = "Sem dados"
            elif _im_hoje > 2.0:
                _im_cor = "#e74c3c"; _im_emoji = "🔴"; _im_lbl = "Supressão imune provável"
            elif _im_hoje > 1.5:
                _im_cor = "#f39c12"; _im_emoji = "🟡"; _im_lbl = "Monitorizar"
            else:
                _im_cor = "#27ae60"; _im_emoji = "✅"; _im_lbl = "Polarizado"

            _fi1, _fi2, _fi3, _fi4 = st.columns(4)
            _fi1.metric(
                "IM (Índice de Monotonia)",
                f"{_im_hoje:.2f}" if _im_hoje is not None else "—",
                delta=f"{_im_emoji} {_im_lbl}",
                delta_color="off",
                help=(
                    "IM = carga_média_7d / std_carga_7d. "
                    "Dias sem treino = 0 kJ. "
                    "<1.5 ✅ | 1.5–2.0 🟡 | >2.0 🔴 supressão imune"
                )
            )
            _fi2.metric(
                "Carga média 7d (kJ/dia)",
                f"{_m7_hoje:.0f}" if _m7_hoje is not None else "—",
                help="Média de kJ por dia nos últimos 7 dias (dias sem treino = 0)"
            )
            _fi3.metric(
                "Training Strain",
                f"{_strain_hoje:.0f}" if _strain_hoje is not None else "—",
                help="Training Strain = IM × carga_média_7d. Combina quantidade e monotonia."
            )
            _delta_im = (
                f"{_im_hoje - _im_7d_ant:+.2f} vs semana ant."
                if _im_hoje is not None and _im_7d_ant is not None else None
            )
            _fi4.metric(
                "Δ IM vs semana anterior",
                _delta_im or "—",
                delta_color="inverse" if _im_hoje is not None and _im_hoje > 1.5 else "off"
            )

            # Card de alerta se IM > 2.0
            if _im_hoje is not None and _im_hoje > 2.0:
                _h_r, _h_g, _h_b = int(_im_cor[1:3],16), int(_im_cor[3:5],16), int(_im_cor[5:7],16)
                st.markdown(
                    f'<div style="padding:12px 16px; border-radius:8px; margin:8px 0; '
                    f'background:rgba({_h_r},{_h_g},{_h_b},0.10); '
                    f'border-left:5px solid {_im_cor};">'
                    f'<b style="color:{_im_cor};">⚠️ Monotonia elevada (IM={_im_hoje:.2f})</b> — '
                    f'Treino demasiado uniforme nos últimos 7 dias. '
                    f'Introduzir dias de carga muito baixa (Z1) entre sessões de alta intensidade. '
                    f'Ver análise completa em Tab Volume.'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif _im_hoje is not None and _im_hoje > 1.5:
                st.info(
                    f"🟡 IM={_im_hoje:.2f} — zona de atenção. "
                    f"Verificar distribuição Z1/Z2/Z3 na Tab Volume."
                )

        else:
            st.info("Coluna kJ não disponível para cálculo de monotonia.")
    else:
        st.info("Sem dados de actividades para cálculo.")


    # ══════════════════════════════════════════════════════════════════════════
    # MARKOV CHAIN + SUGESTAO HRV-GUIDED x MONOTONIA x PADRAO HISTORICO
    # Estado = Modalidade x Zona RPE (Leve/Moderado/Forte) + Descanso
    # Recalcula a cada carregamento sem cache
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔗 Markov Chain — Padrão e Sugestão HRV-Guided")
    with st.expander("Como funciona", expanded=False):
        st.markdown("""
**Markov Chain 1a e 2a ordem — calibrada nos dados deste atleta**

Estado = Modalidade_Zona (ex: Bike_Moderado, Row_Forte) ou Descanso.
A cadeia aprende a probabilidade de cada transição e o HRV médio t+1 e t+2.

**Sugestão integrada:**
1. HRV-Guided: Recuperação → Z1/Z2/Descanso | HIIT → Z2/Z3
2. Monotonia (IM): se IM>1.5 forçar variação de zona/modalidade
3. Markov: selecciona a transição com melhor HRV histórico dentro das opcoes permitidas

Recalcula sem cache a cada carregamento.
        """)

    if da_full is not None and len(da_full) > 0 and wc_full is not None and len(wc_full) > 0:
        try:
            _mk_da = filtrar_principais(da_full).copy()
            _mk_da['Data'] = pd.to_datetime(_mk_da['Data']).dt.normalize()
            _mk_da['type'] = _mk_da['type'].apply(norm_tipo)
            _mk_da = _mk_da[_mk_da['type'].isin(['Bike','Run','Row','Ski','WeightTraining'])]

            _mk_wc = wc_full.copy()
            _mk_wc['Data'] = pd.to_datetime(_mk_wc['Data']).dt.normalize()

            def _zona_rpe(rpe_val):
                try:
                    r = float(rpe_val)
                    if r <= 4.0:   return 'Leve'
                    elif r <= 7.0: return 'Moderado'
                    else:          return 'Forte'
                except: return 'Moderado'

            _mk_da['rpe_n']  = pd.to_numeric(
                _mk_da.get('rpe', pd.Series(dtype=float)), errors='coerce').fillna(5)
            _mk_da['zona']   = _mk_da['rpe_n'].apply(_zona_rpe)
            _mk_da['estado'] = _mk_da['type'] + '_' + _mk_da['zona']

            _mk_daily = (_mk_da.sort_values('rpe_n', ascending=False)
                               .groupby('Data')['estado'].first().reset_index())

            _mk_idx = pd.date_range(
                _mk_daily['Data'].min(), pd.Timestamp.now().normalize(), freq='D')
            _mk_estados = (_mk_daily.set_index('Data')
                                    .reindex(_mk_idx).fillna('Descanso').reset_index())
            _mk_estados.columns = ['Data', 'estado']

            _hrv_col_mk = next((c for c in ['hrv','HRV'] if c in _mk_wc.columns), None)
            if _hrv_col_mk:
                _mk_hrv   = _mk_wc[['Data', _hrv_col_mk]].copy()
                _mk_hrv.columns = ['Data', 'hrv']
                _hrv_base = float(_mk_hrv['hrv'].median()) if len(_mk_hrv) > 5 else 50.0
            else:
                _mk_hrv   = pd.DataFrame(columns=['Data','hrv'])
                _hrv_base = 50.0

            def _hrv_delta_str(val):
                if val is None or (isinstance(val,float) and np.isnan(val)): return '—'
                d = (val - _hrv_base) / max(_hrv_base,1) * 100
                return f"{d:+.1f}%"

            from collections import defaultdict
            _trans  = defaultdict(int)
            _hrv1   = defaultdict(list)
            _trans2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            _estados_list = _mk_estados['estado'].tolist()
            _datas_list   = _mk_estados['Data'].tolist()
            _hrv_dict     = dict(zip(_mk_hrv['Data'], _mk_hrv['hrv'])) if len(_mk_hrv) > 0 else {}

            for _i in range(len(_estados_list)-2):
                _ei = _estados_list[_i]; _ej = _estados_list[_i+1]; _ek = _estados_list[_i+2]
                _d_next = _datas_list[_i+1]; _d_next2 = _datas_list[_i+2]
                _trans[(_ei,_ej)] += 1
                _h1 = _hrv_dict.get(_d_next)
                if _h1 is not None and not np.isnan(float(_h1)):
                    _hrv1[(_ei,_ej)].append(float(_h1))
                _h2 = _hrv_dict.get(_d_next2)
                if _h2 is not None and not np.isnan(float(_h2)):
                    _trans2[_ei][_ej][_ek].append(float(_h2))

            _ei_counts = defaultdict(int)
            for (_ei,_ej), cnt in _trans.items():
                _ei_counts[_ei] += cnt
            _prob = {(_ei,_ej): cnt/_ei_counts[_ei]
                     for (_ei,_ej),cnt in _trans.items() if _ei_counts[_ei] > 0}

            # Check today's state directly from da_full (more reliable than reindex)
            _hoje_mk = pd.Timestamp.now().normalize()
            _today_sessions = _mk_da[_mk_da['Data'] == _hoje_mk]
            if len(_today_sessions) > 0:
                # Today has training — use the highest RPE session
                _best_today = _today_sessions.sort_values('rpe_n', ascending=False).iloc[0]
                _estado_hoje = str(_best_today['estado'])
            else:
                _estado_hoje = _mk_estados.iloc[-1]['estado']

            _CORES_MK = {'Bike':'#e74c3c','Run':'#27ae60','Row':'#3498db',
                         'Ski':'#9b59b6','WeightTraining':'#f39c12','Descanso':'#95a5a6'}

            _mc1, _mc2 = st.columns([1,2])
            with _mc1:
                _partes_h = _estado_hoje.split('_'); _mod_h = _partes_h[0]
                _zona_h   = '_'.join(_partes_h[1:]) if len(_partes_h) > 1 else ''
                _cor_h    = _CORES_MK.get(_mod_h,'#888')
                _hx = _cor_h.lstrip('#'); _rh,_gh,_bh = int(_hx[0:2],16),int(_hx[2:4],16),int(_hx[4:6],16)
                st.markdown(
                    f"<div style='background:rgba({_rh},{_gh},{_bh},0.10);"
                    f"border-left:4px solid {_cor_h};border-radius:6px;padding:10px 14px;'>"
                    f"<div style='font-size:11px;color:#666;margin-bottom:2px'>Estado actual</div>"
                    f"<div style='font-size:18px;font-weight:500;color:{_cor_h}'>{_mod_h}</div>"
                    f"<div style='font-size:12px;color:#555'>{_zona_h}</div>"
                    f"</div>", unsafe_allow_html=True)

            with _mc2:
                st.caption("Top transicoes a partir do estado actual (por HRV t+1)")
                _rows_1 = []
                for (_ei,_ej),_p in _prob.items():
                    if _ei != _estado_hoje or _p < 0.04: continue
                    _hvals = _hrv1.get((_ei,_ej),[])
                    _hmed  = float(np.mean(_hvals)) if len(_hvals) >= 2 else None
                    _rows_1.append({'Proxima sessao':_ej,'Probabilidade':f"{_p*100:.0f}%",
                                    'HRV t+1':_hrv_delta_str(_hmed),'n obs':len(_hvals),
                                    '_hmed':_hmed if _hmed else -999,'_prob':_p})
                _rows_1.sort(key=lambda x: x['_hmed'], reverse=True)
                if _rows_1:
                    st.dataframe(pd.DataFrame([
                        {k:v for k,v in r.items() if not k.startswith('_')}
                        for r in _rows_1[:6]
                    ]), hide_index=True, use_container_width=True)
                else:
                    st.info("Sem historico de transicoes a partir deste estado.")

            # Sugestao integrada
            st.markdown("#### Sugestao — HRV x Monotonia x Markov")

            # HRV-Guided — reutilizar hrv_class já calculado no topo da função
            # (mesmo valor que aparece no card HRV-Guided, consistência garantida)
            if hrv_class == 'Recuperação':
                _zona_perm_mk = ['Leve', 'Moderado']
                _hrv_guid_mk  = "Recuperação — Z1/Descanso, Z2 curto se necessário"
            elif hrv_class == 'HIIT':
                _zona_perm_mk = ['Moderado', 'Forte']
                _hrv_guid_mk  = "HIIT — Z2 (threshold/sweetspot) ou Z3 (VO2max)"
            else:
                _zona_perm_mk = ['Leve', 'Moderado', 'Forte']
                _hrv_guid_mk  = "HRV sem dados — todas as zonas permitidas" 

            # IM actual
            _im_mk = None
            try:
                _kj_c_mk = next((c for c in ['icu_joules','joules','kj_total']
                                  if c in _mk_da.columns and _mk_da[c].notna().any()), None)
                if _kj_c_mk:
                    _kj_d2 = _mk_da.groupby('Data')[_kj_c_mk].sum()
                    if _kj_c_mk == 'icu_joules': _kj_d2 = _kj_d2/1000
                    _kj_i2 = pd.date_range(_kj_d2.index.min(),pd.Timestamp.now().normalize(),freq='D')
                    _kj_f2 = _kj_d2.reindex(_kj_i2,fill_value=0)
                    _m7_2  = _kj_f2.rolling(7,min_periods=4).mean()
                    _s7_2  = _kj_f2.rolling(7,min_periods=4).std()
                    _im_s2 = _m7_2/_s7_2.replace(0,np.nan)
                    _im_mk = float(_im_s2.iloc[-1]) if _im_s2.notna().any() else None
            except Exception: pass

            _forcar_var = _im_mk is not None and _im_mk > 1.5

            # Filtrar candidatos
            _cands = []
            for r in _rows_1:
                _ej2    = r['Proxima sessao']
                _partes = _ej2.split('_')
                _zona_c = '_'.join(_partes[1:]) if len(_partes) > 1 else ''
                # Mapear zona RPE para zona_perm
                _zona_rpe_map = {'Leve':'Leve','Moderado':'Moderado','Forte':'Forte'}
                if _ej2 != 'Descanso' and _zona_c not in _zona_perm_mk:
                    continue
                if _forcar_var and _ej2 == _estado_hoje:
                    continue
                _cands.append(r)
            _cands.sort(key=lambda x: x['_hmed'], reverse=True)

            _sc1,_sc2,_sc3 = st.columns(3)
            _sc1.metric("HRV-Guided", _hrv_guid_mk[:30])
            _sc2.metric("Indice Monotonia",
                        f"{_im_mk:.2f}" if _im_mk is not None else "—",
                        delta="variar" if _forcar_var else "ok repetir",
                        delta_color="inverse" if _forcar_var else "off")
            _sc3.metric("Estado actual", _estado_hoje)

            if _cands:
                _best    = _cands[0]
                _best_ej = _best['Proxima sessao']
                _partes_b= _best_ej.split('_'); _mod_b=_partes_b[0]
                _zona_b  = '_'.join(_partes_b[1:]) if len(_partes_b)>1 else ''
                _cor_b   = _CORES_MK.get(_mod_b,'#27ae60')
                _hxb = _cor_b.lstrip('#'); _rb,_gb,_bb = int(_hxb[0:2],16),int(_hxb[2:4],16),int(_hxb[4:6],16)
                st.markdown(
                    f"<div style='background:rgba({_rb},{_gb},{_bb},0.08);"
                    f"border:1.5px solid {_cor_b};border-radius:8px;padding:14px 18px;margin:8px 0'>"
                    f"<div style='font-size:12px;color:#666;margin-bottom:4px'>"
                    f"Sugestao Markov (HRV x IM x padrao historico)</div>"
                    f"<div style='font-size:22px;font-weight:500;color:{_cor_b}'>"
                    f"{_mod_b} — {_zona_b}</div>"
                    f"<div style='font-size:12px;color:#555;margin-top:4px'>"
                    f"P={_best['Probabilidade']} | HRV t+1: {_best['HRV t+1']} "
                    f"{'| IM alto — variar' if _forcar_var else ''}"
                    f"</div></div>", unsafe_allow_html=True)
                if len(_cands) > 1:
                    with st.expander("Alternativas"):
                        st.dataframe(pd.DataFrame([
                            {'Opcao':r['Proxima sessao'],'P':r['Probabilidade'],'HRV t+1':r['HRV t+1']}
                            for r in _cands[1:4]
                        ]), hide_index=True, use_container_width=True)
            else:
                st.info("Sem transicao compativel. Sugestao: Descanso ou sessao Leve.")

            # 2a ordem
            st.markdown("#### Planeamento 2 dias — Markov 2a ordem")
            st.caption("Sequencia hoje→amanha que maximiza HRV t+2.")
            _d2_rows = []
            if _estado_hoje in _trans2:
                for _ej_o in sorted(_trans2[_estado_hoje].keys()):
                    _ek_opts = _trans2[_estado_hoje][_ej_o]
                    _best_ek = None; _best_h2 = -999
                    for _ek, _hl in _ek_opts.items():
                        if len(_hl) < 2: continue
                        _m2 = float(np.mean(_hl))
                        if _m2 > _best_h2: _best_h2=_m2; _best_ek=_ek
                    if _best_ek:
                        _p_ej = _prob.get((_estado_hoje,_ej_o),0)
                        if _p_ej < 0.03: continue
                        _d2_rows.append({'Hoje (escolha)':_ej_o,'P(hoje→amanha)':f"{_p_ej*100:.0f}%",
                                         'Melhor amanha':_best_ek,'HRV t+2':_hrv_delta_str(_best_h2),
                                         '_h2':_best_h2})
                if _d2_rows:
                    _d2_rows.sort(key=lambda x: x['_h2'], reverse=True)
                    st.dataframe(pd.DataFrame([
                        {k:v for k,v in r.items() if k!='_h2'} for r in _d2_rows[:6]
                    ]), hide_index=True, use_container_width=True)
                else:
                    st.info("Dados insuficientes para 2a ordem.")
            else:
                st.info("Sem historico para 2a ordem.")

            _n_total_mk = sum(_trans.values())
            _n_est_mk   = len(set(e for (e,_) in _trans.keys()))
            with st.expander("Sobre o modelo"):
                st.markdown(f"""
**Markov Chain — {_n_total_mk} transicoes | {_n_est_mk} estados**

- Estado = Modalidade x Zona RPE (Leve<=4 / Moderado 4-7 / Forte >7) + Descanso
- HRV: Recuperacao → Z1/Descanso/Z2curto | HIIT → Z2/Z3
- IM Fry: se >1.5 forcar variacao de zona/modalidade
- Recalcula sem cache
                """)

        except Exception as _mk_err:
            st.warning(f"Markov Chain — erro: {_mk_err}")
    else:
        st.info("Markov Chain requer actividades + wellness (da_full + wc_full).")


    st.markdown("---")

    # Resumo Semanal movido para cima (acima da tabela Semana actual)




# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC + FTLM
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_pmc.py
# ════════════════════════════════════════════════════════════════════════════
