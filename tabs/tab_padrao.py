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

# CORREÇÃO: Definição da variável faltante
DIAS_SEMANA = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']

def _zona_rpe(v):
    try:
        v = float(v)
        if 1 <= v <= 4.9:  return 'Z1 (Leve)'
        if 5 <= v <= 6.9:  return 'Z2 (Mod.)'
        if 7 <= v <= 10:   return 'Z3 (Pesado)'
    except Exception:
        pass
    return None

def _periodos(da, dw):
    """
    Gera lista de (label, data_inicio, data_fim) para todos os períodos.
    Fixos: 15d, 30d, 90d, 180d
    Por ano: anos com dados, do mais recente para o mais antigo
    Todo histórico
    """
    hoje = pd.Timestamp.now().normalize()
    periodos = []

    for dias, lbl in [(15,'15 dias'),(30,'30 dias'),(90,'90 dias'),(180,'180 dias')]:
        periodos.append((lbl, hoje - pd.Timedelta(days=dias), hoje))

    # Anos com dados (actividades + wellness)
    anos_set = set()
    for df in [da, dw]:
        if df is not None and len(df) > 0 and 'Data' in df.columns:
            anos_set.update(pd.to_datetime(df['Data']).dt.year.unique())
    for ano in sorted(anos_set, reverse=True):
        ini = pd.Timestamp(ano, 1, 1)
        fim = min(pd.Timestamp(ano, 12, 31), hoje)
        periodos.append((str(ano), ini, fim))

    # Todo histórico
    todos = []
    for df in [da, dw]:
        if df is not None and len(df) > 0 and 'Data' in df.columns:
            todos.append(pd.to_datetime(df['Data']).min())
    if todos:
        periodos.append(('Todo histórico', min(todos), hoje))

    return periodos


def _filtrar(df, d_ini, d_fim):
    if df is None or len(df) == 0: return df
    d = df.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    return d[(d['Data'] >= d_ini) & (d['Data'] <= d_fim)]


def _cell_atividade(df_p, dia_num):
    """
    Para um dia da semana (0=Seg), retorna 'Tipo1 (X%) / Tipo2' ou '—'.
    Conta todas as datas desse dia da semana no período.
    """
    if df_p is None or len(df_p) == 0:
        return '—'
    d = df_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek
    sub = d[d['dow'] == dia_num]

    # Conta por data única (não por actividade — um dia pode ter 2 sessões)
    datas_unicas = sub['Data'].dt.normalize().unique()
    total_datas = len(pd.date_range(
        df_p['Data'].min(), df_p['Data'].max(),
        freq='W-' + ['MON','TUE','WED','THU','FRI','SAT','SUN'][dia_num]
    ))
    total_datas = max(total_datas, 1)

    if len(sub) == 0:
        return f'Rest (100%)'

    # Conta por tipo (inclui Rest para dias sem actividade)
    # Só actividades cíclicas (excluir WeightTraining)
    sub_cicl = sub[sub['type'].apply(norm_tipo) != 'WeightTraining']
    contagem = sub_cicl['type'].apply(norm_tipo).value_counts()

    # Dias sem actividade cíclica
    datas_com_ativ = sub_cicl['Data'].dt.normalize().nunique()
    datas_sem_ativ = total_datas - datas_com_ativ
    if datas_sem_ativ > 0:
        contagem['Rest'] = datas_sem_ativ

    total = contagem.sum()
    top = contagem.nlargest(2)
    itens = list(top.items())

    if len(itens) == 0:
        return '—'

    t1, n1 = itens[0]
    pct1 = int(round(n1 / total * 100))
    cell = f'{t1} ({pct1}%)'
    if len(itens) > 1:
        t2, _ = itens[1]
        cell += f' / {t2}'
    return cell


def _cell_rpe(df_p, dia_num):
    """
    Para um dia da semana, retorna a zona RPE mais frequente
    como padrão dominante (moda da zona por semana).
    """
    if df_p is None or len(df_p) == 0:
        return 'Rest'
    d = df_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek
    sub = d[d['dow'] == dia_num].copy()

    if len(sub) == 0 or 'rpe' not in sub.columns:
        return 'Rest'

    sub['_zona'] = pd.to_numeric(sub['rpe'], errors='coerce').apply(_zona_rpe)
    sub['_semana'] = sub['Data'].dt.to_period('W')

    # Para cada semana, qual foi a zona desse dia (moda se múltiplas sessões)
    padroes_semana = []
    for sem, grp in sub.groupby('_semana'):
        zonas = grp['_zona'].dropna()
        if len(zonas) == 0:
            padroes_semana.append('Rest')
        else:
            padroes_semana.append(zonas.mode().iloc[0])

    if not padroes_semana:
        return 'Rest'

    # Zona dominante ao longo das semanas
    from collections import Counter
    cnt = Counter(padroes_semana)
    return cnt.most_common(1)[0][0]


def _contagem_zonas_padrao(df_p):
    """
    Contagem de dias com cada zona dominante (pelo padrão semanal),
    mais a média de sessões Z3 por semana real.
    Retorna dict {zona: count, 'z3_por_semana': float}.
    """
    if df_p is None or len(df_p) == 0:
        return {}
    d = df_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek

    if 'rpe' not in d.columns:
        return {}

    d['_zona'] = pd.to_numeric(d['rpe'], errors='coerce').apply(_zona_rpe)

    # Padrão dominante por dia da semana
    contagem = {'Z1 (Leve)': 0, 'Z2 (Mod.)': 0, 'Z3 (Pesado)': 0, 'Rest': 0}
    for dia_num in range(7):
        zona_dom = _cell_rpe(d, dia_num)
        if zona_dom in contagem:
            contagem[zona_dom] += 1
        else:
            contagem['Rest'] += 1

    # Z3 por semana real: conta sessões Z3 por semana e tira a média
    d['_semana'] = d['Data'].dt.to_period('W')
    d_z3 = d[d['_zona'] == 'Z3 (Pesado)']
    n_semanas = d['_semana'].nunique()
    if n_semanas > 0:
        z3_por_sem = d_z3.groupby('_semana')['_zona'].count()
        media_z3 = z3_por_sem.reindex(d['_semana'].unique(), fill_value=0).mean()
    else:
        media_z3 = 0.0

    contagem['z3_por_semana'] = round(media_z3, 1)
    return contagem


def _cell_recovery(dw_p, dia_num):
    """
    Média ± DP do Recovery Score nesse dia da semana.
    """
    if dw_p is None or len(dw_p) == 0:
        return '—'
    d = dw_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek

    # Calcular recovery score
    rec = calcular_recovery(d)
    if len(rec) == 0:
        return '—'
    rec['Data'] = pd.to_datetime(rec['Data'])
    rec['dow'] = rec['Data'].dt.dayofweek
    sub = rec[rec['dow'] == dia_num]['recovery_score'].dropna()

    if len(sub) < 2:
        return f'{sub.mean():.0f}' if len(sub) == 1 else '—'
    return f'{sub.mean():.0f} ±{sub.std():.0f}'


def _cell_hrv_rhr(dw_p, dia_num, baseline_hrv, baseline_rhr, dp_hrv, dp_rhr):
    """
    Média de HRV e RHR nesse dia da semana.
    Cor: verde se dentro de ±0.5 DP do baseline, vermelho se fora.
    Retorna (texto_hrv, texto_rhr, cor_hrv, cor_rhr).
    """
    if dw_p is None or len(dw_p) == 0:
        return '—', '—', 'normal', 'normal'

    d = dw_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek
    sub = d[d['dow'] == dia_num]

    def _stats(col):
        if col not in sub.columns: return None, None
        vals = pd.to_numeric(sub[col], errors='coerce').dropna()
        if len(vals) == 0: return None, None
        return vals.mean(), vals.std()

    hrv_m, hrv_s = _stats('hrv')
    rhr_m, rhr_s = _stats('rhr')

    def _cor_hrv(m, base, dp):
        if m is None or base is None or dp is None or dp == 0: return 'normal'
        return 'green' if abs(m - base) <= 0.5 * dp else 'red'

    def _cor_rhr(m, base, dp):
        # RHR: mais alto = pior → invertido
        if m is None or base is None or dp is None or dp == 0: return 'normal'
        return 'green' if abs(m - base) <= 0.5 * dp else 'red'

    hrv_txt = f'{hrv_m:.0f}' if hrv_m else '—'
    rhr_txt = f'{rhr_m:.0f}' if rhr_m else '—'
    cor_hrv = _cor_hrv(hrv_m, baseline_hrv, dp_hrv)
    cor_rhr = _cor_rhr(rhr_m, baseline_rhr, dp_rhr)

    return hrv_txt, rhr_txt, cor_hrv, cor_rhr


def _render_hrv_rhr_table(rows_hrv):
    """
    Renderiza tabela HRV/RHR com cores usando HTML.
    rows_hrv: list of dicts com keys: Período, + dias da semana (cada com tuple (hrv,rhr,c_h,c_r))
    """
    cols = ['Período'] + DIAS_SEMANA

    def _cell_html(hrv_txt, rhr_txt, c_h, c_r):
        cor_h = '#27ae60' if c_h == 'green' else '#e74c3c' if c_h == 'red' else '#333333'
        cor_r = '#27ae60' if c_r == 'green' else '#e74c3c' if c_r == 'red' else '#333333'
        return (f'<span style="color:{cor_h};font-weight:bold">{hrv_txt}</span>'
                f'<span style="color:#666666"> | </span>'
                f'<span style="color:{cor_r};font-weight:bold">{rhr_txt}</span>')

    html = ('<table style="border-collapse:collapse;width:100%;font-size:12px;'
             'background:#ffffff;color:#222222">')
    # Header
    html += ('<tr style="background:#e8e8e8">')
    for c in cols:
        html += (f'<th style="border:1px solid #ccc;padding:6px 8px;'
                 f'text-align:center;color:#111111;font-weight:bold">{c}</th>')
    html += '</tr>'

    for i, row in enumerate(rows_hrv):
        bg = '#ffffff' if i % 2 == 0 else '#f9f9f9'
        html += f'<tr style="background:{bg}">'
        html += (f'<td style="border:1px solid #ccc;padding:5px 8px;'
                 f'font-weight:bold;white-space:nowrap;color:#111111">'
                 f'{row["Período"]}</td>')
        for dia in DIAS_SEMANA:
            val = row.get(dia, ('—','—','normal','normal'))
            if isinstance(val, tuple):
                h, r, ch, cr = val
                cell = _cell_html(h, r, ch, cr)
            else:
                cell = str(val)
            html += (f'<td style="border:1px solid #ccc;padding:5px 8px;'
                     f'text-align:center;color:#222222">{cell}</td>')
        html += '</tr>'
    html += '</table>'
    return html


# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def tab_padrao(da_full, dw_full):
    """
    Aba Padrão — padrão semanal típico do atleta.
    da_full : DataFrame actividades completo (ac_full)
    dw_full : DataFrame wellness completo (wc — preproc_wellness sobre histórico total)
    """
    st.header("🔄 Padrão Semanal do Atleta")
    st.caption(
        "Análise do padrão típico por dia da semana em múltiplos períodos. "
        "Todos os períodos usam o histórico completo disponível.")

    if (da_full is None or len(da_full) == 0) and (dw_full is None or len(dw_full) == 0):
        st.warning("Sem dados disponíveis para análise de padrão.")
        return

    # Preparar dados
    da = da_full.copy() if da_full is not None and len(da_full) > 0 else pd.DataFrame()
    dw = dw_full.copy() if dw_full is not None and len(dw_full) > 0 else pd.DataFrame()
    if len(da) > 0:
        da['Data'] = pd.to_datetime(da['Data'])
    if len(dw) > 0:
        dw['Data'] = pd.to_datetime(dw['Data'])

    periodos = _periodos(da if len(da) > 0 else None,
                         dw if len(dw) > 0 else None)

    # ════════════════════════════════════════════════════════════════════
    # TABELA 1 — Padrão de actividade por dia da semana
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🏃 Padrão de Actividade por Dia da Semana")
    st.caption("Top 1 com %, Top 2 sem %. 'Rest' quando maioria dos dias sem actividade.")

    if len(da) > 0:
        rows_ativ = []
        for lbl, d_ini, d_fim in periodos:
            da_p = _filtrar(da, d_ini, d_fim)
            if len(da_p) == 0:
                continue
            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                row[dia] = _cell_atividade(da_p, n)
            rows_ativ.append(row)

        if rows_ativ:
            # CORREÇÃO: use_container_width em vez de width="stretch"
            st.dataframe(pd.DataFrame(rows_ativ), use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados de actividade suficientes.")
    else:
        st.info("Sem dados de actividade.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA 2 — Padrão de RPE por dia da semana + contagem de zonas
    # ════════════════════════════════════════════════════════════════════
    st.subheader("⚡ Padrão de RPE por Dia da Semana")
    st.caption(
        "Zona dominante por dia. "
        "Z1=RPE 1–5 | Z2=RPE 5–7 | Z3=RPE 7–10 | Rest=sem actividade. "
        "Contagem = nº de dias da semana com cada zona como dominante.")

    if len(da) > 0 and 'rpe' in da.columns:
        # RPE: usar só dados a partir de 2023
        da_rpe = da[pd.to_datetime(da['Data']).dt.year >= 2023].copy()
        rows_rpe = []
        for lbl, d_ini, d_fim in periodos:
            da_p = _filtrar(da_rpe, d_ini, d_fim)
            if len(da_p) == 0:
                continue
            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                row[dia] = _cell_rpe(da_p, n)
            # Contagem de zonas pelo padrão + Z3 real por semana
            cnt = _contagem_zonas_padrao(da_p)
            row['Z1 dias'] = cnt.get('Z1 (Leve)', 0)
            row['Z2 dias'] = cnt.get('Z2 (Mod.)', 0)
            row['Z3 dias'] = cnt.get('Z3 (Pesado)', 0)
            row['Rest dias'] = cnt.get('Rest', 0)
            row['Z3/semana'] = cnt.get('z3_por_semana', 0.0)
            rows_rpe.append(row)

        if rows_rpe:
            # CORREÇÃO: use_container_width em vez de width="stretch"
            st.dataframe(pd.DataFrame(rows_rpe), use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados de RPE suficientes.")
    else:
        st.info("Sem dados de RPE.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA 3 — Padrão de Recovery Score por dia da semana
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🔋 Padrão de Recovery Score por Dia da Semana")
    st.caption("Média ± DP do Recovery Score por dia da semana em cada período.")

    if len(dw) > 0:
        rows_rec = []
        for lbl, d_ini, d_fim in periodos:
            dw_p = _filtrar(dw, d_ini, d_fim)
            if len(dw_p) < 3:
                continue
            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                row[dia] = _cell_recovery(dw_p, n)
            rows_rec.append(row)

        if rows_rec:
            # CORREÇÃO: use_container_width em vez de width="stretch"
            st.dataframe(pd.DataFrame(rows_rec), use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados de wellness suficientes.")
    else:
        st.info("Sem dados de wellness.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA 4 — Padrão HRV / RHR por dia da semana (com cores)
    # ════════════════════════════════════════════════════════════════════
    st.subheader("💚 Padrão HRV / RHR por Dia da Semana")
    st.caption(
        "Formato: **HRV | RHR**  "
        "🟢 verde = dentro de ±0.5 DP do baseline do período  "
        "🔴 vermelho = fora do baseline")

    if len(dw) > 0 and 'hrv' in dw.columns:
        rows_hrv = []
        for lbl, d_ini, d_fim in periodos:
            dw_p = _filtrar(dw, d_ini, d_fim)
            if len(dw_p) < 3:
                continue

            # Baseline = média e DP do período
            hrv_vals = pd.to_numeric(dw_p['hrv'], errors='coerce').dropna()
            rhr_vals  = (pd.to_numeric(dw_p['rhr'], errors='coerce').dropna()
                         if 'rhr' in dw_p.columns else pd.Series(dtype=float))

            if len(hrv_vals) == 0:
                continue

            base_hrv = hrv_vals.mean()
            dp_hrv   = hrv_vals.std()
            base_rhr = rhr_vals.mean() if len(rhr_vals) > 0 else None
            dp_rhr   = rhr_vals.std()  if len(rhr_vals) > 0 else None

            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                h, r, ch, cr = _cell_hrv_rhr(dw_p, n,
                                               base_hrv, base_rhr,
                                               dp_hrv, dp_rhr)
                row[dia] = (h, r, ch, cr)
            rows_hrv.append(row)

        if rows_hrv:
            html = _render_hrv_rhr_table(rows_hrv)
            st.markdown(html, unsafe_allow_html=True)
            st.caption("HRV em ms | RHR em bpm")
        else:
            st.info("Sem dados de HRV suficientes.")
    else:
        st.info("Sem dados de HRV/RHR.")
