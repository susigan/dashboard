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

def tab_correlacoes(da, dw):
    st.header("🧠 Correlações & Impacto")
    st.caption("Análise sobre todo o histórico disponível — independente do filtro de período do sidebar.")
    if len(da) == 0 or len(dw) == 0: st.warning("Sem dados suficientes."); return

    rpe_col   = next((c for c in ['rpe','RPE','icu_rpe'] if c in da.columns), None)
    CICLICOS_T = ['Bike','Row','Run','Ski']
    # Cores fortes para boa visibilidade mobile
    CORES_T  = {'Bike':'#e74c3c','Row':'#2980b9','Ski':'#8e44ad',
                'Run':'#27ae60','WeightTraining':'#e67e22','Rest':'#7f8c8d'}
    CORES_CAT = {'Leve':'#27ae60','Moderado':'#e67e22','Pesado':'#c0392b','Rest':'#7f8c8d'}
    LAYOUT_BASE = dict(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='#111111', size=13),
        margin=dict(l=45, r=20, t=50, b=50))

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _remove_outliers_iqr(series, factor=1.5):
        """Remove outliers IQR 1.5x — retorna série com NaN nos extremos."""
        s = pd.to_numeric(series, errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s < q1 - factor*iqr) | (s > q3 + factor*iqr)
        s[mask] = np.nan
        return s

    def _prep_dw_clean(dw_in, data_min='2020-01-01'):
        """Wellness limpo: filtro 2020+, outliers IQR removidos."""
        d = dw_in.copy()
        d['Data'] = pd.to_datetime(d['Data']).dt.normalize()
        d = d[d['Data'] >= pd.Timestamp(data_min)]
        if 'hrv' in d.columns:
            d['hrv'] = _remove_outliers_iqr(d['hrv'])
        if 'rhr' in d.columns:
            d['rhr'] = _remove_outliers_iqr(d['rhr'])
        return d.dropna(subset=['hrv'])

    def _dias_com_atividade(da_in, data_min='2020-01-01'):
        """
        Retorna set de datas onde houve QUALQUER actividade
        (cíclica OU WeightTraining), a partir de data_min.
        Usado para definir Rest de forma uniforme em ambas as análises.
        """
        d = da_in.copy()
        d['Data'] = pd.to_datetime(d['Data']).dt.normalize()
        d = d[d['Data'] >= pd.Timestamp(data_min)]
        d['_tipo'] = d['type'].apply(norm_tipo)
        # Qualquer actividade reconhecida (cíclica OU WT)
        d_ativ = d[d['_tipo'].isin(CICLICOS_T + ['WeightTraining'])]
        return set(d_ativ['Data'].unique())

    def _prep_merged_rpe(da_in, data_min='2020-01-01'):
        """
        Cruza RPE diário (só cíclicas, com RPE válido) com HRV/RHR dia seguinte.

        REGRA REST (uniforme):
          Rest = dia de wellness onde NÃO houve NENHUMA actividade
                 (nem cíclica, nem WeightTraining).
          Dias com actividade mas sem RPE → ficam como Rest para o scatter
          mas são excluídos da análise RPE (não têm categoria Leve/Mod/Pesado).
        """
        da2 = da_in.copy()
        da2['_tipo'] = da2['type'].apply(norm_tipo)
        da2['Data'] = pd.to_datetime(da2['Data']).dt.normalize()
        da2 = da2[da2['Data'] >= pd.Timestamp(data_min)]

        # Dias com QUALQUER actividade (para definir Rest)
        _todos_ativ = _dias_com_atividade(da_in, data_min)

        # Só cíclicas COM RPE válido → para classificar zona
        da_cicl = da2[da2['_tipo'].isin(CICLICOS_T)].copy()
        if not rpe_col:
            return pd.DataFrame()
        da_cicl = da_cicl.dropna(subset=[rpe_col])
        da_cicl[rpe_col] = pd.to_numeric(da_cicl[rpe_col], errors='coerce')
        da_cicl = da_cicl.dropna(subset=[rpe_col])

        rpe_d = da_cicl.groupby('Data')[rpe_col].mean().reset_index()
        rpe_d.columns = ['Data', 'rpe_avg']
        rpe_d['rpe_cat'] = rpe_d['rpe_avg'].apply(classificar_rpe)
        rpe_d = rpe_d.dropna(subset=['rpe_cat'])

        dw_clean = _prep_dw_clean(dw, data_min)
        all_days = dw_clean[['Data']].copy()
        all_days = all_days.merge(rpe_d[['Data','rpe_cat','rpe_avg']], on='Data', how='left')

        # Rest = sem NENHUMA actividade nesse dia (cíclica OU WT)
        all_days['rpe_cat'] = all_days.apply(
            lambda r: r['rpe_cat'] if pd.notna(r['rpe_cat'])
            else ('Rest' if r['Data'] not in _todos_ativ else None),
            axis=1)
        # Dias com actividade mas sem RPE válido → excluir (None → drop)
        all_days = all_days.dropna(subset=['rpe_cat'])

        cols_dw = ['Data','hrv'] + (['rhr'] if 'rhr' in dw_clean.columns else [])
        dw_shift = dw_clean[cols_dw].copy()
        dw_shift['Data'] = dw_shift['Data'] - pd.Timedelta(days=1)
        merged = all_days.merge(dw_shift, on='Data', how='inner')
        return merged.dropna(subset=['hrv'])

    def _prep_merged_tipo(da_in, data_min='2020-01-01'):
        """
        Cruza tipo de actividade com HRV/RHR dia seguinte.

        REGRA REST (uniforme, igual ao _prep_merged_rpe):
          Rest = dia de wellness onde NÃO houve NENHUMA actividade
                 (nem cíclica, nem WeightTraining).
          WT só quando sozinho (sem cíclica no mesmo dia).
        """
        da3 = da_in.copy()
        da3['_tipo'] = da3['type'].apply(norm_tipo)
        da3['Data']  = pd.to_datetime(da3['Data']).dt.normalize()
        da3 = da3[da3['Data'] >= pd.Timestamp(data_min)]

        # Dias com QUALQUER actividade (mesma definição do RPE)
        _todos_ativ = _dias_com_atividade(da_in, data_min)

        dias_cicl = set(da3[da3['_tipo'].isin(CICLICOS_T)]['Data'])

        da3_f = da3[
            da3['_tipo'].isin(CICLICOS_T) |
            ((da3['_tipo'] == 'WeightTraining') & (~da3['Data'].isin(dias_cicl)))
        ].copy()

        tipo_d = (da3_f.groupby('Data')['_tipo']
                  .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
                  .reset_index())

        dw_clean = _prep_dw_clean(dw, data_min)
        all_days = dw_clean[['Data']].copy()
        all_days = all_days.merge(tipo_d, on='Data', how='left')

        # Rest = sem NENHUMA actividade (cíclica OU WT) — mesma regra
        all_days['_tipo'] = all_days.apply(
            lambda r: r['_tipo'] if pd.notna(r['_tipo'])
            else ('Rest' if r['Data'] not in _todos_ativ else None),
            axis=1)
        all_days = all_days.dropna(subset=['_tipo'])

        cols_dw = ['Data','hrv'] + (['rhr'] if 'rhr' in dw_clean.columns else [])
        dw_shift = dw_clean[cols_dw].copy()
        dw_shift['Data'] = dw_shift['Data'] - pd.Timedelta(days=1)
        merged = all_days.merge(dw_shift, on='Data', how='inner')
        return merged.dropna(subset=['hrv'])

    def _prep_merged_rpe_modal(da_in, data_min='2020-01-01'):
        """
        Cruza RPE por modalidade × dia com HRV/RHR do dia seguinte.
        Para cada dia: modalidade dominante + RPE médio desse dia.
        Só dias com cíclica + RPE válido.
        """
        da2 = da_in.copy()
        da2['_tipo'] = da2['type'].apply(norm_tipo)
        da2['Data']  = pd.to_datetime(da2['Data']).dt.normalize()
        da2 = da2[da2['Data'] >= pd.Timestamp(data_min)]
        da2 = da2[da2['_tipo'].isin(CICLICOS_T)]
        if not rpe_col: return pd.DataFrame()
        da2 = da2.dropna(subset=[rpe_col])
        da2[rpe_col] = pd.to_numeric(da2[rpe_col], errors='coerce')
        da2 = da2.dropna(subset=[rpe_col])
        if len(da2) == 0: return pd.DataFrame()

        # Agrupar: modalidade dominante + RPE médio por dia
        grp = da2.groupby('Data').agg(
            modalidade=('_tipo', lambda x: x.mode()[0]),
            rpe_avg=(rpe_col, 'mean')
        ).reset_index()
        grp['rpe_cat'] = grp['rpe_avg'].apply(classificar_rpe)
        grp = grp.dropna(subset=['rpe_cat'])

        dw_clean = _prep_dw_clean(dw, data_min)
        cols_dw = ['Data','hrv'] + (['rhr'] if 'rhr' in dw_clean.columns else [])
        dw_shift = dw_clean[cols_dw].copy()
        dw_shift['Data'] = dw_shift['Data'] - pd.Timedelta(days=1)
        merged = grp.merge(dw_shift, on='Data', how='inner')
        return merged.dropna(subset=['hrv'])

    def _stat_kruskal(merged, grupo_col, grupos):
        """
        Estatísticas de confiabilidade por grupo:
        - Kruskal-Wallis: diferença global entre grupos (p-value)
        - Eta² (η²): variância explicada — tamanho do efeito (sinal vs ruído)
        - Cohen's d: tamanho do efeito entre grupo e restante
        - CV% intra-grupo: variabilidade interna (ruído)
        """
        from scipy.stats import kruskal as _kruskal
        results = {}
        N = len(merged)
        k = len([g for g in grupos if (merged[grupo_col]==g).sum() >= 3])
        for metric in ['hrv','rhr']:
            if metric not in merged.columns: continue
            vals = [merged[merged[grupo_col]==g][metric].dropna().values
                    for g in grupos if (merged[grupo_col]==g).sum() >= 3]
            if len(vals) < 2: continue
            try:
                H, p = _kruskal(*vals)
                # Eta² — variância explicada pelo grupo
                eta2 = max(0.0, (H - k + 1) / (N - k)) if N > k else 0.0
                if   eta2 >= 0.14: eta2_lbl = "grande (≥14%)"
                elif eta2 >= 0.06: eta2_lbl = "médio (6–14%)"
                elif eta2 >= 0.01: eta2_lbl = "pequeno (1–6%)"
                else:               eta2_lbl = "negligenciável (<1%)"
                sig = ('✅ SIG p<0.05' if p < 0.05 else
                       '~ marginal p<0.10' if p < 0.10 else '✗ ns')
                results[metric] = {
                    'H': round(H,2), 'p': round(p,4), 'sig': sig,
                    'eta2': round(eta2,3), 'eta2_lbl': eta2_lbl,
                    'N': N,
                }
            except Exception:
                pass
        return results

    def _cohen_d(g1, g2):
        """Cohen's d entre dois grupos (pooled SD)."""
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2: return None
        s_pooled = np.sqrt(((n1-1)*np.var(g1,ddof=1) + (n2-1)*np.var(g2,ddof=1)) / (n1+n2-2))
        if s_pooled == 0: return None
        d = (np.mean(g1) - np.mean(g2)) / s_pooled
        if   abs(d) >= 0.8: lbl = "grande"
        elif abs(d) >= 0.5: lbl = "médio"
        elif abs(d) >= 0.2: lbl = "pequeno"
        else:               lbl = "negligenciável"
        return round(d, 2), lbl

    def _tabela_delta(merged, grupo_col, grupos):
        rows = []
        base_hrv = merged['hrv'].mean()
        base_rhr = merged['rhr'].mean() if 'rhr' in merged.columns else None
        kw = _stat_kruskal(merged, grupo_col, grupos)
        for g in grupos:
            sub = merged[merged[grupo_col] == g]
            if len(sub) < 2: continue
            d_hrv = (sub['hrv'].mean() - base_hrv) / base_hrv * 100
            # CV% intra-grupo HRV
            cv_hrv = (sub['hrv'].std() / sub['hrv'].mean() * 100
                      if sub['hrv'].mean() > 0 and len(sub) >= 3 else None)
            cv_lbl = ('baixo ✅' if cv_hrv and cv_hrv < 10 else
                      'médio ⚠️' if cv_hrv and cv_hrv < 20 else
                      'alto 🔴' if cv_hrv else '—')
            row = {
                'Grupo':            g,
                'N':                len(sub),
                'HRV médio':        f"{sub['hrv'].mean():.0f} ms",
                'Δ HRV%':           f"{d_hrv:+.1f}%",
                'CV% HRV':          f"{cv_hrv:.0f}% {cv_lbl}" if cv_hrv else '—',
                'Interpretação HRV':('↗ recuperação' if d_hrv > 3
                                     else '↘ stress' if d_hrv < -3 else '→ neutro'),
            }
            if base_rhr and 'rhr' in merged.columns:
                d_rhr = sub['rhr'].mean() - base_rhr
                cv_rhr = (sub['rhr'].std() / sub['rhr'].mean() * 100
                          if sub['rhr'].mean() > 0 and len(sub) >= 3 else None)
                row['RHR médio']      = f"{sub['rhr'].mean():.0f} bpm"
                row['Δ RHR']          = f"{d_rhr:+.1f} bpm"
                row['Interpret. RHR'] = ('↘ recuperação' if d_rhr < -2
                                          else '↗ stress' if d_rhr > 2 else '→ neutro')
            rows.append(row)
        df_tab = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Linha de estatísticas globais (KW + Eta²)
        if len(df_tab) > 0 and kw:
            for metric, col_hrv, col_delta in [
                ('hrv', 'HRV médio', 'Δ HRV%'),
                ('rhr', 'RHR médio', 'Δ RHR'),
            ]:
                if metric not in kw: continue
                kw_m = kw[metric]
                stat_row = {
                    'Grupo': f"— KW {'HRV' if metric=='hrv' else 'RHR'}",
                    'N': kw_m['N'],
                    col_hrv: f"H={kw_m['H']}  p={kw_m['p']}",
                    col_delta: kw_m['sig'],
                    'CV% HRV' if metric=='hrv' else 'Δ RHR': f"η²={kw_m['eta2']} ({kw_m['eta2_lbl']})",
                    'Interpretação HRV' if metric=='hrv' else 'Interpret. RHR':
                        ("Sinal forte" if kw_m['eta2'] >= 0.06 else
                         "Sinal fraco" if kw_m['eta2'] >= 0.01 else "Ruído"),
                }
                df_tab = pd.concat([df_tab, pd.DataFrame([stat_row])], ignore_index=True)
        return df_tab

    def _tabela_impacto(merged, grupo_col, grupos):
        """Tabela compacta: Grupo | N | HRV médio | Δ HRV% | RHR | Interpretação."""
        base_hrv = merged['hrv'].mean()
        base_rhr = merged['rhr'].mean() if 'rhr' in merged.columns else None
        rows = []
        for g in grupos:
            sub = merged[merged[grupo_col] == g]
            if len(sub) < 2: continue
            d_hrv = (sub['hrv'].mean() - base_hrv) / base_hrv * 100
            row = {'Grupo': g, 'N': len(sub),
                   'HRV médio (ms)': f"{sub['hrv'].mean():.0f}",
                   'Δ HRV%': f"{d_hrv:+.1f}%",
                   'Interpretação': '↗ recuperação' if d_hrv > 3 else '↘ stress' if d_hrv < -3 else '→ neutro'}
            if base_rhr and 'rhr' in merged.columns:
                d_rhr = sub['rhr'].mean() - base_rhr
                row['RHR (bpm)'] = f"{sub['rhr'].mean():.0f}"
                row['Δ RHR'] = f"{d_rhr:+.1f} bpm"
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _bar_impacto(grupos, deltas_hrv, cores_map, titulo):
        fig = go.Figure()
        for g, d in zip(grupos, deltas_hrv):
            if d is None: continue
            fig.add_trace(go.Bar(
                x=[g], y=[round(d, 1)],
                marker_color=cores_map.get(g, '#555'),
                text=[f"{d:+.1f}%"], textposition='outside',
                textfont=dict(color='#111', size=11), width=0.55,
                hovertemplate=f'{g}<br>Δ HRV: <b>{d:+.1f}%</b><extra></extra>'))
        fig.add_hline(y=0, line_dash='dash', line_color='#888', line_width=1)
        fig.update_layout(**LAYOUT_BASE, height=300, showlegend=False,
                          title=dict(text=titulo, font=dict(size=12, color='#111')),
                          xaxis=dict(tickfont=dict(size=10, color='#111'), showgrid=False),
                          yaxis=dict(title='Δ HRV%', tickfont=dict(color='#111'),
                                     showgrid=True, gridcolor='#ddd', zeroline=True, zerolinecolor='#888'))
        return fig

    MC_LOC = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}

    def _bar_chart(grupos, deltas_hrv, deltas_rhr, cores_map, title_hrv, title_rhr):
        """Dois gráficos HRV% + RHR bpm lado a lado, mobile-friendly."""
        col_h, col_r = st.columns(2)
        with col_h:
            fig = go.Figure()
            for g, d in zip(grupos, deltas_hrv):
                if d is None: continue
                fig.add_trace(go.Bar(
                    x=[g], y=[round(d, 1)],
                    marker_color=cores_map.get(g, '#555555'),
                    text=[f"{d:+.1f}%"], textposition='outside',
                    textfont=dict(color='#111111', size=12), width=0.55,
                    hovertemplate=f'{g}<br>Δ HRV: <b>{d:+.1f}%</b><extra></extra>'))
            fig.add_hline(y=0, line_dash='dash', line_color='#555', line_width=1)
            fig.update_layout(**LAYOUT_BASE,
                title=dict(text=title_hrv, font=dict(size=13, color='#111')),
                height=340, showlegend=False,
                xaxis=dict(title='', tickfont=dict(size=11, color='#111'),
                           showgrid=False, linecolor='#ccc', showline=True),
                yaxis=dict(title='Δ HRV (%)', tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd', zeroline=True,
                           zerolinecolor='#888', zerolinewidth=1.5))
            st.plotly_chart(fig, use_container_width=True, config=MC_LOC)
        with col_r:
            if any(d is not None for d in deltas_rhr):
                fig2 = go.Figure()
                for g, d in zip(grupos, deltas_rhr):
                    if d is None: continue
                    fig2.add_trace(go.Bar(
                        x=[g], y=[round(d, 1)],
                        marker_color=cores_map.get(g, '#555555'),
                        text=[f"{d:+.1f}"], textposition='outside',
                        textfont=dict(color='#111111', size=12), width=0.55,
                        hovertemplate=f'{g}<br>Δ RHR: <b>{d:+.1f} bpm</b><extra></extra>'))
                fig2.add_hline(y=0, line_dash='dash', line_color='#555', line_width=1)
                fig2.update_layout(**LAYOUT_BASE,
                    title=dict(text=title_rhr, font=dict(size=13, color='#111')),
                    height=340, showlegend=False,
                    xaxis=dict(title='', tickfont=dict(size=11, color='#111'),
                               showgrid=False, linecolor='#ccc', showline=True),
                    yaxis=dict(title='Δ RHR (bpm)', tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#ddd', zeroline=True,
                               zerolinecolor='#888', zerolinewidth=1.5))
                st.plotly_chart(fig2, use_container_width=True, config=MC_LOC)
            else:
                st.info("Sem dados de RHR.")

    # ── Dados base para análises (2020+, filtrados) ──────────────────────
    _da_use = filtrar_principais(da).copy()
    _da_use['Data'] = pd.to_datetime(_da_use['Data'])
    _da_use = _da_use[_da_use['Data'] >= pd.Timestamp('2020-01-01')]

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 1 — Impacto RPE → HRV/RHR (dia seguinte)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("💚 Impacto do RPE → HRV/RHR (dia seguinte)")
    st.caption(
        "Leve RPE 1–4 | Moderado RPE 5–6 | Pesado RPE 7–10. "
        "Sessões duplas no mesmo dia: RPE médio → categoria. "
        "Rest = dias sem actividade."
    )

    merged_rpe = _prep_merged_rpe(_da_use)

    if len(merged_rpe) >= 5:
        cats      = ['Leve','Moderado','Pesado','Rest']
        base_hrv_r = merged_rpe['hrv'].mean()
        base_rhr_r = merged_rpe['rhr'].mean() if 'rhr' in merged_rpe.columns else None
        grupos_r   = [c for c in cats if (merged_rpe['rpe_cat']==c).sum() >= 2]
        d_hrv_r    = [(merged_rpe[merged_rpe['rpe_cat']==g]['hrv'].mean()-base_hrv_r)/base_hrv_r*100
                      for g in grupos_r]
        _c1, _c2 = st.columns([3, 2])
        with _c1:
            st.plotly_chart(_bar_impacto(grupos_r, d_hrv_r, CORES_CAT,
                                          "Δ HRV% — dia seguinte por RPE"),
                            use_container_width=True, config=MC_LOC)
        with _c2:
            df_r = _tabela_impacto(merged_rpe, 'rpe_cat', grupos_r)
            if len(df_r): st.dataframe(df_r, hide_index=True, use_container_width=True)
            kw_r = _stat_kruskal(merged_rpe, 'rpe_cat', grupos_r)
            if 'hrv' in kw_r:
                st.caption(f"KW: H={kw_r['hrv']['H']}  {kw_r['hrv']['sig']}  η²={kw_r['hrv']['eta2']}")
    else:
        st.info("Dados insuficientes.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 1b — RPE por Modalidade → HRV/RHR (dia seguinte)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🚴 RPE por Modalidade → HRV/RHR (dia seguinte)")
    st.caption(
        "Cada dia: modalidade dominante × zona RPE → HRV/RHR do dia seguinte. "
        "Permite ver se Bike pesado afecta diferente de Row pesado.")

    merged_modal = _prep_merged_rpe_modal(_da_use)

    if len(merged_modal) >= 5:
        mods_modal = [m for m in CICLICOS_T if m in merged_modal['modalidade'].values]
        base_hrv_m = merged_modal['hrv'].mean()
        base_rhr_m = merged_modal['rhr'].mean() if 'rhr' in merged_modal.columns else None

        # Gráfico: barras agrupadas por modalidade + zona RPE
        _grupos_modal = []
        _dhrv_modal   = []
        _drhr_modal   = []
        for mod in mods_modal:
            for cat in ['Leve','Moderado','Pesado']:
                sub = merged_modal[(merged_modal['modalidade']==mod) &
                                   (merged_modal['rpe_cat']==cat)]
                if len(sub) < 2: continue
                lbl = f"{mod} {cat}"
                _grupos_modal.append(lbl)
                _dhrv_modal.append((sub['hrv'].mean() - base_hrv_m) / base_hrv_m * 100)
                _drhr_modal.append((sub['rhr'].mean() - base_rhr_m)
                                   if base_rhr_m else None)

        CORES_MODAL_RPE = {
            'Bike Leve':'#fadbd8','Bike Moderado':'#f1948a','Bike Pesado':'#e74c3c',
            'Row Leve':'#d6eaf8','Row Moderado':'#5dade2','Row Pesado':'#2980b9',
            'Ski Leve':'#e8daef','Ski Moderado':'#a569bd','Ski Pesado':'#8e44ad',
            'Run Leve':'#d5f5e3','Run Moderado':'#58d68d','Run Pesado':'#27ae60',
        }

        if _grupos_modal:
            _c1m, _c2m = st.columns([3, 2])
            with _c1m:
                st.plotly_chart(_bar_impacto(_grupos_modal, _dhrv_modal, CORES_MODAL_RPE,
                                              "Δ HRV% por Modalidade × RPE"),
                                use_container_width=True, config=MC_LOC)

            # Tabela compacta
            rows_modal = []
            kw_modal = _stat_kruskal(merged_modal, 'modalidade', mods_modal)
            for mod in mods_modal:
                for cat in ['Leve','Moderado','Pesado']:
                    sub = merged_modal[(merged_modal['modalidade']==mod) &
                                       (merged_modal['rpe_cat']==cat)]
                    if len(sub) < 2: continue
                    d_hrv = (sub['hrv'].mean() - base_hrv_m) / base_hrv_m * 100
                    row = {'Modalidade': mod, 'Zona': cat, 'N': len(sub),
                           'HRV médio (ms)': f"{sub['hrv'].mean():.0f}",
                           'Δ HRV%': f"{d_hrv:+.1f}%",
                           'Interpretação': '↗ rec.' if d_hrv>3 else '↘ stress' if d_hrv<-3 else '→'}
                    if base_rhr_m and 'rhr' in sub.columns:
                        d_rhr = sub['rhr'].mean() - base_rhr_m
                        row['Δ RHR'] = f"{d_rhr:+.1f} bpm"
                    rows_modal.append(row)
            if rows_modal:
                df_modal_tab = pd.DataFrame(rows_modal)
                st.dataframe(df_modal_tab, hide_index=True, use_container_width=True)

            # Significância Kruskal-Wallis entre modalidades
            if kw_modal:
                parts = []
                if 'hrv' in kw_modal:
                    parts.append(f"HRV: H={kw_modal['hrv']['H']} {kw_modal['hrv']['sig']}")
                if 'rhr' in kw_modal:
                    parts.append(f"RHR: H={kw_modal['rhr']['H']} {kw_modal['rhr']['sig']}")
                st.caption("Kruskal-Wallis entre modalidades: " + "  |  ".join(parts))
    else:
        st.info("Dados insuficientes para análise por modalidade.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 2 — Scatter RPE→HRV | HRV↔RHR | KJ→HRV | KJ→RHR
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔍 Relações: RPE, KJ global e HRV/RHR")
    col1, col2 = st.columns(2)

    with col1:
        if rpe_col and len(merged_rpe) > 0 and 'rpe_avg' in merged_rpe.columns:
            m_cl = merged_rpe[['rpe_avg','hrv']].dropna()
            # Excluir Rest do scatter (rpe_avg=NaN para Rest — já dropna acima)
            m_cl = m_cl[m_cl['rpe_avg'].notna()]
            st.caption(f"N pontos scatter RPE→HRV: **{len(m_cl)}**")
            if len(m_cl) >= 3:
                r_val, _ = pearsonr(m_cl['rpe_avg'].astype(float),
                                    m_cl['hrv'].astype(float))
                xr = np.linspace(float(m_cl['rpe_avg'].min()),
                                 float(m_cl['rpe_avg'].max()), 50)
                z  = np.polyfit(m_cl['rpe_avg'].astype(float),
                                m_cl['hrv'].astype(float), 1)
                fig_sc1 = go.Figure()
                fig_sc1.add_trace(go.Scatter(
                    x=m_cl['rpe_avg'].tolist(), y=m_cl['hrv'].tolist(),
                    mode='markers',
                    marker=dict(color='#2980b9', size=7, opacity=0.5),
                    hovertemplate='RPE: %{x:.1f}<br>HRV: <b>%{y:.0f} ms</b><extra></extra>'))
                fig_sc1.add_trace(go.Scatter(
                    x=xr.tolist(), y=np.poly1d(z)(xr).tolist(),
                    mode='lines', line=dict(color='#e74c3c', width=2),
                    hoverinfo='skip'))
                fig_sc1.update_layout(**LAYOUT_BASE,
                    title=dict(text=f'RPE → HRV (r={r_val:.2f})', font=dict(size=12,color='#111')),
                    height=260,
                    xaxis=dict(title='RPE', tickfont=dict(color='#111'), showgrid=True, gridcolor='#ddd'),
                    yaxis=dict(title='HRV (ms)', tickfont=dict(color='#111'), showgrid=True, gridcolor='#ddd'),
                    showlegend=False)
                st.plotly_chart(fig_sc1, use_container_width=True, config=MC_LOC)

    with col2:
        if 'hrv' in dw.columns and 'rhr' in dw.columns:
            dw3 = dw[['hrv','rhr']].dropna()
            if len(dw3) >= 5:
                r2   = float(dw3['hrv'].astype(float).corr(dw3['rhr'].astype(float)))
                xr2  = np.linspace(float(dw3['hrv'].min()), float(dw3['hrv'].max()), 50)
                z2   = np.polyfit(dw3['hrv'].astype(float), dw3['rhr'].astype(float), 1)
                fig_sc2 = go.Figure()
                fig_sc2.add_trace(go.Scatter(
                    x=dw3['hrv'].tolist(), y=dw3['rhr'].tolist(),
                    mode='markers',
                    marker=dict(color='#8e44ad', size=7, opacity=0.5),
                    hovertemplate='HRV: %{x:.0f}<br>RHR: <b>%{y:.0f} bpm</b><extra></extra>'))
                fig_sc2.add_trace(go.Scatter(
                    x=xr2.tolist(), y=np.poly1d(z2)(xr2).tolist(),
                    mode='lines', line=dict(color='#e74c3c', width=2),
                    hoverinfo='skip'))
                fig_sc2.update_layout(**LAYOUT_BASE,
                    title=dict(text=f'HRV vs RHR (r={r2:.2f})', font=dict(size=12,color='#111')),
                    height=260,
                    xaxis=dict(title='HRV (ms)', tickfont=dict(color='#111'), showgrid=True, gridcolor='#ddd'),
                    yaxis=dict(title='RHR (bpm)', tickfont=dict(color='#111'), showgrid=True, gridcolor='#ddd'),
                    showlegend=False)
                st.plotly_chart(fig_sc2, use_container_width=True, config=MC_LOC)


    # ── KJ Global → HRV e KJ Global → RHR ───────────────────────────────
    _da_kj = _da_use.copy()
    _da_kj['_tipo'] = _da_kj['type'].apply(norm_tipo)
    _da_kj = _da_kj[_da_kj['_tipo'].isin(CICLICOS_T)]
    if 'icu_joules' in _da_kj.columns:
        _da_kj['_kj'] = pd.to_numeric(_da_kj['icu_joules'], errors='coerce') / 1000
    elif 'power_avg' in _da_kj.columns:
        _da_kj['_kj'] = (pd.to_numeric(_da_kj['power_avg'], errors='coerce') *
                          pd.to_numeric(_da_kj['moving_time'], errors='coerce') / 1000)
    else:
        _da_kj['_kj'] = np.nan
    _kj_day = _da_kj.groupby('Data')['_kj'].sum().reset_index()
    _kj_day.columns = ['Data', 'kj']
    _kj_day['Data'] = pd.to_datetime(_kj_day['Data'])
    _dw_sc = _prep_dw_clean(dw)[['Data','hrv'] + (['rhr'] if 'rhr' in dw.columns else [])].copy()
    _dw_sc['Data'] = pd.to_datetime(_dw_sc['Data'])
    _kj_hrv = _kj_day.merge(_dw_sc, on='Data', how='inner').dropna(subset=['hrv','kj'])

    if len(_kj_hrv) >= 5:
        from scipy.stats import pearsonr as _pr_kj
        _sc3, _sc4 = st.columns(2)
        for col_c, col_v, lbl_v, color_v, key_v in [
            (_sc3, 'hrv', 'HRV (ms)', '#27ae60', 'kj_hrv_sc'),
            (_sc4, 'rhr', 'RHR (bpm)', '#e67e22', 'kj_rhr_sc'),
        ]:
            if col_v not in _kj_hrv.columns: continue
            _d = _kj_hrv[['kj', col_v]].dropna()
            if len(_d) < 5: continue
            _r, _ = _pr_kj(_d['kj'].astype(float), _d[col_v].astype(float))
            _z  = np.polyfit(_d['kj'].astype(float), _d[col_v].astype(float), 1)
            _xr = np.linspace(float(_d['kj'].min()), float(_d['kj'].max()), 50)
            _fig_kj = go.Figure()
            _fig_kj.add_trace(go.Scatter(
                x=_d['kj'].tolist(), y=_d[col_v].tolist(), mode='markers',
                marker=dict(color=color_v, size=5, opacity=0.4),
                hovertemplate=f'kJ: %{{x:.0f}}<br>{lbl_v}: <b>%{{y:.0f}}</b><extra></extra>'))
            _fig_kj.add_trace(go.Scatter(
                x=_xr.tolist(), y=np.poly1d(_z)(_xr).tolist(),
                mode='lines', line=dict(color='#e74c3c', width=2), hoverinfo='skip'))
            _fig_kj.update_layout(**LAYOUT_BASE, height=240, showlegend=False,
                title=dict(text=f'kJ global vs {lbl_v} (r={_r:.2f})',
                           font=dict(size=11, color='#111')),
                xaxis=dict(title='kJ global', tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd'),
                yaxis=dict(title=lbl_v, tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd'))
            col_c.plotly_chart(_fig_kj, use_container_width=True, config=MC_LOC)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 3 — Correlações Wellness (heatmap + tabela)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📊 Correlações Wellness")

    mets_num = [c for c in ['hrv','rhr','sleep_quality','fatiga','stress','humor','soreness']
                if c in dw.columns and dw[c].notna().any()]

    if len(mets_num) >= 3:
        corr_mat = dw[mets_num].apply(pd.to_numeric, errors='coerce').corr(method='pearson')

        import matplotlib.pyplot as plt
        import seaborn as sns
        n = len(mets_num)
        import numpy as _npH
        _mskH = _npH.triu(_npH.ones_like(corr_mat.values, dtype=bool), k=1)
        _cHM  = corr_mat.columns.tolist()
        _zHM  = [[float(corr_mat.values[r][c]) if not _mskH[r][c] else None
                  for c in range(len(_cHM))] for r in range(len(_cHM))]
        _tHM  = [[f'{corr_mat.values[r][c]:.2f}' if not _mskH[r][c] else ''
                  for c in range(len(_cHM))] for r in range(len(_cHM))]
        _figHM = go.Figure(go.Heatmap(z=_zHM, x=_cHM, y=_cHM,
            text=_tHM, texttemplate='%{text}', textfont=dict(size=9,color='#111'),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title='r', tickfont=dict(color='#111'))))
        _figHM.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#111'), margin=dict(t=50,b=70,l=55,r=20), height=max(320, n*35),
            title=dict(text='Correlações Wellness', font=dict(size=13,color='#111')),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9,color='#111')),
            yaxis=dict(tickfont=dict(size=9,color='#111'), autorange='reversed'))
        st.plotly_chart(_figHM, use_container_width=True, config=MC_LOC)

        def _forca_str(r):
            a = abs(r)
            if a >= 0.70: return "★★★ Forte"
            if a >= 0.50: return "★★ Moderada"
            if a >= 0.30: return "★ Fraca"
            return "— Nenhuma"

        rows_corr = []
        for i in range(n):
            for j in range(i+1, n):
                r = float(corr_mat.iloc[i,j])
                if abs(r) < 0.20: continue
                rows_corr.append({
                    'Variável A': mets_num[i],
                    'Variável B': mets_num[j],
                    'r':          f"{r:+.2f}",
                    'r_num':      round(abs(r), 3),
                    'Força':      _forca_str(r),
                    'Direcção':   ('↗ positiva' if r>0 else '↘ negativa'),
                })
        if rows_corr:
            df_ct = (pd.DataFrame(rows_corr)
                     .sort_values('r_num', ascending=False)
                     .drop(columns=['r_num']))
            st.dataframe(df_ct, hide_index=True, use_container_width=True)
            st.caption("★★★ Forte |r|≥0.70 | ★★ Moderada ≥0.50 | ★ Fraca ≥0.30")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 4 — Impacto por Tipo de Actividade
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏃 Impacto por Tipo de Actividade → HRV/RHR (dia seguinte)")
    st.caption("Cíclicas: todas. WeightTraining: só dias sem outra cíclica. Rest: dias sem actividade.")

    merged_tipo = _prep_merged_tipo(_da_use)

    if len(merged_tipo) >= 5:
        base_hrv_t = merged_tipo['hrv'].mean()
        base_rhr_t = merged_tipo['rhr'].mean() if 'rhr' in merged_tipo.columns else None

        tipos_disp = [t for t in CICLICOS_T + ['WeightTraining','Rest']
                      if (merged_tipo['_tipo']==t).sum() >= 2]
        deltas_hrv_t = [(merged_tipo[merged_tipo['_tipo']==g]['hrv'].mean()-base_hrv_t)/base_hrv_t*100
                        for g in tipos_disp]
        deltas_rhr_t = [(merged_tipo[merged_tipo['_tipo']==g]['rhr'].mean()-base_rhr_t)
                        if base_rhr_t else None
                        for g in tipos_disp]

        _bar_chart(tipos_disp, deltas_hrv_t, deltas_rhr_t, CORES_T,
                   "Δ HRV% — dia seguinte (por Modalidade)",
                   "Δ RHR bpm — dia seguinte (por Modalidade)")

        _c1t, _c2t = st.columns([3, 2])
        with _c1t:
            _d_hrv_t = [(merged_tipo[merged_tipo['_tipo']==g]['hrv'].mean()-base_hrv_t)/base_hrv_t*100
                        for g in tipos_disp]
            st.plotly_chart(_bar_impacto(tipos_disp, _d_hrv_t, CORES_T,
                                          "Δ HRV% — dia seguinte por Modalidade"),
                            use_container_width=True, config=MC_LOC)
        with _c2t:
            df_t_comp = _tabela_impacto(merged_tipo, '_tipo', tipos_disp)
            if len(df_t_comp): st.dataframe(df_t_comp, hide_index=True, use_container_width=True)
            kw_t = _stat_kruskal(merged_tipo, '_tipo', tipos_disp)
            if 'hrv' in kw_t:
                st.caption(f"KW: H={kw_t['hrv']['H']}  {kw_t['hrv']['sig']}  η²={kw_t['hrv']['eta2']}")
    else:
        st.info("Dados insuficientes.")




    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 6 — Análise Avançada: Carga, Fadiga e HRV
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("\U0001f9ea Análise Avançada — Carga, Fadiga e HRV")
    st.caption(
        "Usa icu_training_load (mesma escala do PMC). "
        "WeightTraining excluído. Actividades disponíveis a partir de 2023-01-25 (RPE). "
        "ATL/CTL calculados sobre training_load historico completo. "
        "lag(ATL/CTL) evita leakage: ATL(t-1) = fadiga ANTES da sessao de hoje.")

    # ── Série diária de carga (icu_training_load, sem WT) ────────────────
    _adv_da = _da_use.copy()
    _adv_da = _adv_da[_adv_da['type'].apply(norm_tipo) != 'WeightTraining']
    _adv_dw = _prep_dw_clean(dw, '2020-01-01').copy()
    _adv_dw['Data'] = pd.to_datetime(_adv_dw['Data']).dt.normalize()

    # Usar icu_training_load como carga primária (fallback: dur_min × rpe)
    _has_icu = ('icu_training_load' in _adv_da.columns and
                _adv_da['icu_training_load'].notna().sum() > 20)
    if _has_icu:
        _adv_da['_load'] = pd.to_numeric(_adv_da['icu_training_load'], errors='coerce').fillna(0)
        _load_src = "icu_training_load"
    else:
        _rpe_c = pd.to_numeric(_adv_da.get('rpe', pd.Series(dtype=float)), errors='coerce')
        _adv_da['_load'] = (pd.to_numeric(_adv_da['moving_time'], errors='coerce') / 60 * _rpe_c).fillna(0)
        _load_src = "TRIMP (fallback)"

    # KJ por dia (para impacto agudo)
    if 'icu_joules' in _adv_da.columns and _adv_da['icu_joules'].notna().any():
        _adv_da['_kj'] = pd.to_numeric(_adv_da['icu_joules'], errors='coerce') / 1000
    elif 'power_avg' in _adv_da.columns:
        _adv_da['_kj'] = (pd.to_numeric(_adv_da['power_avg'], errors='coerce') *
                          pd.to_numeric(_adv_da['moving_time'], errors='coerce') / 1000)
    else:
        _adv_da['_kj'] = np.nan

    # Agregar por dia
    _daily = _adv_da.groupby('Data').agg(
        load=('_load', 'sum'),
        kj=('_kj',    'sum'),
        n_sess=('_load', 'count'),
    ).reset_index()
    _daily['Data'] = pd.to_datetime(_daily['Data'])
    _daily['kj']   = _daily['kj'].replace(0, np.nan)
    _daily['load'] = _daily['load'].replace(0, np.nan)

    # ATL/CTL sobre série completa (histórico desde 2020)
    _all_d = pd.date_range(_daily['Data'].min(), pd.Timestamp.now().normalize(), freq='D')
    _load_ser = _daily.set_index('Data')['load'].reindex(_all_d, fill_value=0)
    _atl_s = _load_ser.ewm(span=7,  adjust=False).mean()
    _ctl_s = _load_ser.ewm(span=42, adjust=False).mean()

    # ⚠️ CRÍTICO: lag(ATL/CTL) — usar ATL(t-1) para evitar data leakage
    _atl_lag = _atl_s.shift(1)
    _ctl_lag = _ctl_s.shift(1)
    _atl_ctl_lag = (_atl_lag / _ctl_lag.replace(0, np.nan)).round(3)

    _ld_df = pd.DataFrame({
        'Data': _all_d,
        'ATL': _atl_s.values,
        'CTL': _ctl_s.values,
        'ATL_lag': _atl_lag.values,
        'CTL_lag': _ctl_lag.values,
        'ATL_CTL_lag': _atl_ctl_lag.values,
    })
    _daily = _daily.merge(_ld_df, on='Data', how='left')

    # Log-transform de KJ e load (reduz outliers, captura não-linearidade)
    _daily['log_kj']   = np.log1p(_daily['kj'].fillna(0))
    _daily['log_load'] = np.log1p(_daily['load'].fillna(0))

    # Cruzar com HRV(t) e HRV(t+1)
    _hrv_today = _adv_dw[['Data','hrv']].rename(columns={'hrv':'hrv_t'})
    _hrv_next  = _adv_dw[['Data','hrv']].copy()
    _hrv_next['Data'] = _hrv_next['Data'] - pd.Timedelta(days=1)
    _hrv_next  = _hrv_next.rename(columns={'hrv':'hrv_t1'})
    _daily = (_daily
              .merge(_hrv_today, on='Data', how='inner')
              .merge(_hrv_next,  on='Data', how='left'))

    # HRV relativo: HRV / rolling_mean(7d) — reduz ruído individual
    _hrv_roll7 = _adv_dw.set_index('Data')['hrv'].rolling(7, min_periods=3).mean()
    _hrv_roll7_df = pd.DataFrame({'Data': _hrv_roll7.index, 'hrv_roll7': _hrv_roll7.values})
    _hrv_roll7_df['Data'] = pd.to_datetime(_hrv_roll7_df['Data'])
    _daily = _daily.merge(_hrv_roll7_df, on='Data', how='left')
    _daily['hrv_rel'] = (_daily['hrv_t'] / _daily['hrv_roll7'].replace(0, np.nan)).round(4)
    # HRV relativo dia seguinte
    _hrv_roll7_next = _hrv_roll7_df.copy()
    _hrv_roll7_next['Data'] = _hrv_roll7_next['Data'] - pd.Timedelta(days=1)
    _hrv_roll7_next = _hrv_roll7_next.rename(columns={'hrv_roll7':'hrv_roll7_next'})
    _daily = _daily.merge(_hrv_roll7_next, on='Data', how='left')
    _hrv_t1_abs = _adv_dw[['Data','hrv']].copy()
    _hrv_t1_abs['Data'] = _hrv_t1_abs['Data'] - pd.Timedelta(days=1)
    _hrv_t1_abs = _hrv_t1_abs.rename(columns={'hrv':'hrv_t1_abs'})
    _daily = _daily.merge(_hrv_t1_abs, on='Data', how='left')
    _daily['hrv_t1_rel'] = (_daily['hrv_t1_abs'] / _daily['hrv_roll7_next'].replace(0, np.nan)).round(4)

    # Filtros de segmentação
    _atl_p33 = float(_daily['ATL_lag'].quantile(0.33))
    _atl_p66 = float(_daily['ATL_lag'].quantile(0.66))
    _isolated = _daily[_daily['ATL_lag'] <= _atl_p33].copy()
    _fatigued  = _daily[_daily['ATL_lag'] >= _atl_p66].copy()

    st.caption(
        f"Metrica load: **{_load_src}** | "
        f"Dias com actividade: {len(_daily)} | "
        f"ATL-lag tercis: baixo ≤{_atl_p33:.0f} | alto ≥{_atl_p66:.0f} | "
        f"Isolados: {len(_isolated)} | Fatigados: {len(_fatigued)}")

    # ── Função de correlação robusta ──────────────────────────────────────
    def _corr_row(df_in, x_col, y_col, label_x, label_y, grupo="Todos"):
        d = df_in[[x_col, y_col]].dropna()
        if len(d) < 8: return None
        x = d[x_col].values.astype(float)
        y = d[y_col].values.astype(float)
        from scipy.stats import spearmanr, pearsonr, linregress
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        sl, ic, _, _, _ = linregress(x, y)
        sig = ("✅ p<0.05" if min(p_p,p_s)<0.05 else
               "~ p<0.10" if min(p_p,p_s)<0.10 else "✗ ns")
        forca = ("forte" if abs(r_s)>=0.5 else
                 "moderada" if abs(r_s)>=0.3 else
                 "fraca" if abs(r_s)>=0.1 else "negligenciavel")
        return {
            'Grupo': grupo, 'X': label_x, 'Y': label_y, 'N': len(d),
            'r Spearman': round(r_s, 3), 'r Pearson': round(r_p, 3),
            'p': round(min(p_p,p_s), 4), 'Sig': sig,
            'Slope': round(sl, 4), 'Forca': forca,
            'Direcao': ("↗" if r_s > 0 else "↘"),
        }

    # ════════════════════════════════════════════════════════════════════
    # BLOCO 1 — Impacto Agudo: HRV(t+1) ~ load(t) / KJ(t)
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("**1️⃣  Impacto Agudo — HRV amanhã em função da carga hoje**")
    st.caption(
        "Usa HRV relativo (HRV/rolling7d) e log(KJ+1) para reduzir ruído e outliers. "
        "ATL(t-1) = fadiga ANTES da sessão. "
        "Isolados = ATL-lag baixo (≤P33) — efeito limpo sem fadiga acumulada.")

    _rows_b1 = []
    for _df_g, _lbl_g in [
        (_daily,    "Todos"),
        (_isolated, "Isolados (ATL-lag baixo)"),
        (_fatigued, "Fatigados (ATL-lag alto)"),
    ]:
        # HRV relativo dia seguinte ~ log_load
        r = _corr_row(_df_g, 'log_load', 'hrv_t1_rel', f'log({_load_src})', 'HRV_rel(t+1)', _lbl_g)
        if r: _rows_b1.append(r)
        # HRV relativo dia seguinte ~ log_kj
        r2 = _corr_row(_df_g, 'log_kj', 'hrv_t1_rel', 'log(KJ)', 'HRV_rel(t+1)', _lbl_g)
        if r2: _rows_b1.append(r2)

    if _rows_b1:
        st.dataframe(pd.DataFrame(_rows_b1), hide_index=True, use_container_width=True)

    # Scatter com LOWESS para dias isolados
    _sc1 = _isolated[['log_kj','hrv_t1_rel']].dropna()
    if len(_sc1) >= 8:
        from scipy.stats import pearsonr as _pr1
        _r1, _ = _pr1(_sc1['log_kj'].astype(float), _sc1['hrv_t1_rel'].astype(float))
        _z1 = np.polyfit(_sc1['log_kj'].astype(float), _sc1['hrv_t1_rel'].astype(float), 1)
        _xr1 = np.linspace(float(_sc1['log_kj'].min()), float(_sc1['log_kj'].max()), 50)
        # LOWESS via pandas rolling (simples)
        _sc1_s = _sc1.sort_values('log_kj').copy()
        _sc1_s['_roll'] = _sc1_s['hrv_t1_rel'].rolling(max(3, len(_sc1_s)//8),
                                                         min_periods=2, center=True).mean()
        fig_b1 = go.Figure()
        fig_b1.add_trace(go.Scatter(
            x=_sc1_s['log_kj'].tolist(), y=_sc1_s['hrv_t1_rel'].tolist(),
            mode='markers', name='Pontos',
            marker=dict(color='#2980b9', size=6, opacity=0.45),
            hovertemplate='log(KJ): %{x:.2f}<br>HRV_rel(t+1): <b>%{y:.3f}</b><extra></extra>'))
        fig_b1.add_trace(go.Scatter(
            x=_sc1_s['log_kj'].tolist(), y=np.poly1d(_z1)(_sc1_s['log_kj'].values).tolist(),
            mode='lines', name='Regressão linear',
            line=dict(color='#e74c3c', width=2)))
        fig_b1.add_trace(go.Scatter(
            x=_sc1_s['log_kj'].tolist(), y=_sc1_s['_roll'].tolist(),
            mode='lines', name='Tendência (rolling)',
            line=dict(color='#27ae60', width=2, dash='dash')))
        fig_b1.update_layout(**LAYOUT_BASE,
            title=dict(text=f'log(KJ) vs HRV_rel(t+1) — Dias Isolados (r={_r1:.2f})',
                       font=dict(size=13, color='#111')),
            height=310, showlegend=True,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(title='log(KJ+1)', tickfont=dict(color='#111'),
                       showgrid=True, gridcolor='#ddd'),
            yaxis=dict(title='HRV relativo dia seguinte', tickfont=dict(color='#111'),
                       showgrid=True, gridcolor='#ddd'))
        st.plotly_chart(fig_b1, use_container_width=True, config=MC_LOC)
        st.caption("HRV_rel = HRV / média_7d (remove tendência individual). "
                   "log(KJ) = escala logarítmica (captura não-linearidade).")

    # ════════════════════════════════════════════════════════════════════
    # BLOCO 2 — Fadiga Acumulada: HRV(t) ~ ATL_lag
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("**2️⃣  Fadiga Acumulada — HRV hoje em função do ATL do dia anterior**")
    st.caption(
        "ATL_lag = ATL calculado até ONTEM (shift 1). "
        "Evita que a sessão de hoje contamine o ATL que tentamos correlacionar com HRV de hoje.")

    _rows_b2 = []
    for _df_g, _lbl_g in [(_daily,"Todos"),(_isolated,"ATL-lag baixo"),(_fatigued,"ATL-lag alto")]:
        r = _corr_row(_df_g, 'ATL_lag', 'hrv_rel', 'ATL(t-1)', 'HRV_rel(t)', _lbl_g)
        if r: _rows_b2.append(r)
        r2 = _corr_row(_df_g, 'ATL_CTL_lag', 'hrv_rel', 'ATL/CTL(t-1)', 'HRV_rel(t)', _lbl_g)
        if r2: _rows_b2.append(r2)

    if _rows_b2:
        st.dataframe(pd.DataFrame(_rows_b2), hide_index=True, use_container_width=True)

    # Scatter ATL_lag vs HRV_rel
    _sc2 = _daily[['ATL_lag','hrv_rel']].dropna()
    if len(_sc2) >= 8:
        from scipy.stats import pearsonr as _pr2
        _r2, _ = _pr2(_sc2['ATL_lag'].astype(float), _sc2['hrv_rel'].astype(float))
        _z2 = np.polyfit(_sc2['ATL_lag'].astype(float), _sc2['hrv_rel'].astype(float), 1)
        _sc2_s = _sc2.sort_values('ATL_lag').copy()
        _sc2_s['_roll'] = _sc2_s['hrv_rel'].rolling(max(3, len(_sc2_s)//8),
                                                      min_periods=2, center=True).mean()
        fig_b2 = go.Figure()
        fig_b2.add_trace(go.Scatter(
            x=_sc2_s['ATL_lag'].tolist(), y=_sc2_s['hrv_rel'].tolist(),
            mode='markers', name='Pontos',
            marker=dict(color='#e74c3c', size=6, opacity=0.4),
            hovertemplate='ATL(t-1): %{x:.1f}<br>HRV_rel: <b>%{y:.3f}</b><extra></extra>'))
        fig_b2.add_trace(go.Scatter(
            x=_sc2_s['ATL_lag'].tolist(), y=np.poly1d(_z2)(_sc2_s['ATL_lag'].values).tolist(),
            mode='lines', name='Regressão linear', line=dict(color='#2c3e50', width=2)))
        fig_b2.add_trace(go.Scatter(
            x=_sc2_s['ATL_lag'].tolist(), y=_sc2_s['_roll'].tolist(),
            mode='lines', name='Tendência (rolling)',
            line=dict(color='#f39c12', width=2, dash='dash')))
        fig_b2.update_layout(**LAYOUT_BASE,
            title=dict(text=f'ATL(t-1) vs HRV_rel(t) (r={_r2:.2f})',
                       font=dict(size=13, color='#111')),
            height=310, showlegend=True,
            legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
            xaxis=dict(title=f'ATL(t-1) [{_load_src}]', tickfont=dict(color='#111'),
                       showgrid=True, gridcolor='#ddd'),
            yaxis=dict(title='HRV relativo (t)', tickfont=dict(color='#111'),
                       showgrid=True, gridcolor='#ddd'))
        st.plotly_chart(fig_b2, use_container_width=True, config=MC_LOC)
        st.caption("ATL↑ → HRV_rel↓ esperado. Se r < -0.2 com p<0.05 = fadiga detectável.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCO 3 — Robustez: modelo combinado + interação
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("**3️⃣  Robustez — Modelo combinado: HRV_rel(t+1) ~ log(KJ) + ATL_lag + KJ×ATL**")
    st.caption(
        "Variáveis z-scored (mesma escala). "
        "Interação KJ×ATL: testa se o impacto do KJ depende do nível de fadiga. "
        "R² = % da variação de HRV explicada pelo modelo.")

    # Verificar zonas
    _has_kj_zones_b3 = all(c in _daily.columns for c in ['kj_z1','kj_z2','kj_z3'])
    if _has_kj_zones_b3:
        _daily['kj_weighted'] = (_daily['kj_z1'].fillna(0)*1 +
                                  _daily['kj_z2'].fillna(0)*2 +
                                  _daily['kj_z3'].fillna(0)*3)
        _daily['log_kj_z3']  = np.log1p(_daily['kj_z3'].fillna(0))
        _daily['log_kj_w']   = np.log1p(_daily['kj_weighted'])

    from scipy.stats import zscore as _zsc, f as _f_dist

    # M1-M4: dataset completo (todas as linhas com KJ+ATL+HRV)
    _dm3_base = _daily[['log_kj','ATL_lag','hrv_t1_rel']].dropna().copy()
    # M5-M8: dataset com zonas (linhas que têm dados de zona)
    _dm3_z = (pd.DataFrame() if not _has_kj_zones_b3
              else _daily[['log_kj','log_kj_z3','log_kj_w',
                            'ATL_lag','hrv_t1_rel']].dropna().copy())

    def _run_b3(dm, suffix=""):
        if len(dm) < 10: return []
        dm = dm.copy()
        dm['z_logkj'] = _zsc(dm['log_kj'].astype(float))
        dm['z_atl']   = _zsc(dm['ATL_lag'].astype(float))
        dm['z_inter'] = dm['z_logkj'] * dm['z_atl']
        if 'log_kj_z3' in dm.columns:
            dm['z_kj_z3']    = _zsc(dm['log_kj_z3'].astype(float))
            dm['z_kj_w']     = _zsc(dm['log_kj_w'].astype(float))
            dm['z_inter_z3'] = dm['z_kj_z3'] * dm['z_atl']
            dm['z_inter_w']  = dm['z_kj_w']  * dm['z_atl']
        y = dm['hrv_t1_rel'].values.astype(float)
        models = [
            ("M1: log(KJ_total)",         ['z_logkj']),
            ("M2: ATL_lag",               ['z_atl']),
            ("M3: log(KJ)+ATL_lag",       ['z_logkj','z_atl']),
            ("M4: log(KJ)+ATL+KJ*ATL",   ['z_logkj','z_atl','z_inter']),
        ]
        if 'z_kj_z3' in dm.columns:
            models += [
                ("M5: log(KJ_Z3)+ATL",        ['z_kj_z3','z_atl']),
                ("M6: log(KJ_Z3)+ATL+Z3*ATL", ['z_kj_z3','z_atl','z_inter_z3']),
                ("M7: KJ_w+ATL",               ['z_kj_w','z_atl']),
                ("M8: KJ_w+ATL+Kw*ATL",        ['z_kj_w','z_atl','z_inter_w']),
            ]
        rows = []
        for mlbl, mcols in models:
            if not all(c in dm.columns for c in mcols): continue
            X = np.column_stack([np.ones(len(dm))] + [dm[c].values for c in mcols])
            try:
                beta   = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - y.mean())**2)
                r2 = max(0.0, 1 - ss_res/ss_tot) if ss_tot > 0 else 0
                k, n = len(mcols), len(dm)
                F  = (r2/k)/((1-r2)/(n-k-1)) if r2 < 1 and n > k+1 else 0
                pf = 1 - _f_dist.cdf(F, k, n-k-1)
                coef_s = "  ".join(f"{c.replace('z_','')}:{b:+.3f}"
                                    for c, b in zip(mcols, beta[1:]))
                rows.append({
                    'Modelo': mlbl + suffix, 'N': n,
                    'R²': round(r2,4), 'F': round(F,2),
                    'p(F)': round(pf,4),
                    'Sig': "✅" if pf < 0.05 else "✗ ns",
                    'Coeficientes (z)': coef_s,
                })
            except Exception: pass
        return rows

    # Correr base (M1-M4) + zonas (M5-M8 com N próprio)
    _rows_b3 = _run_b3(_dm3_base)
    if len(_dm3_z) >= 10:
        _rows_b3 += [r for r in _run_b3(_dm3_z, f" [N={len(_dm3_z)}]")
                     if any(r['Modelo'].startswith(m) for m in ['M5','M6','M7','M8'])]

    if _rows_b3:
        st.dataframe(pd.DataFrame(_rows_b3), hide_index=True, use_container_width=True)
        st.caption(
            "M1-M4: todo o histórico. M5-M8: só dias com dados de zonas. "
            "Coeficientes z-scored. KJ neg = mais treino → HRV↓ amanhã. "
            "KJ*ATL positivo = efeito do KJ diminui com fadiga alta.")
    elif len(_dm3_base) < 10:
        st.info(f"Dados insuficientes (N={len(_dm3_base)} < 10).")

    st.markdown("---")
    _all_e2 = []
    for _dfe3, _lbl3 in [(df_tr,'RPE'), (df_tt,'Tipo'), (df_tm,'Modal')]:
        if len(_dfe3) > 0:
            _dfe3c = _dfe3.copy()
            _dfe3c.insert(0, 'Tabela', _lbl3)
            _all_e2.append(_dfe3c)
    if _all_e2:
        _df_all2 = pd.concat(_all_e2, ignore_index=True)
        _csv2 = _df_all2.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV -- Correlacoes completo",
                           _csv2, "atheltica_correlacoes.csv",
                           "text/csv", key="dl_corr_all2")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # ANÁLISES AVANÇADAS DE RESPOSTA AO TREINO
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔬 Análises Avançadas de Resposta ao Treino")
    st.caption(
        "Três perspectivas complementares: duração do efeito (multi-lag), "
        "efeito limpo sem confounding (event response), "
        "impacto da acumulação de carga (load ecology)."
    )

    _adv_tabs = st.tabs(["⏱️ Acute Multi-lag", "🧪 Event Response", "🌊 Load Ecology"])
    _dw_clean_adv = _prep_dw_clean(dw)
    _dw_idx_adv   = _dw_clean_adv.set_index('Data')['hrv']
    _base_hrv_adv = float(_dw_clean_adv['hrv'].mean())

    # HRV multi-lag no merged_rpe
    _mr_ml = merged_rpe.copy() if len(merged_rpe) >= 10 else pd.DataFrame()
    if len(_mr_ml) > 0:
        _mr_ml['Data'] = pd.to_datetime(_mr_ml['Data'])
        for _k in [1, 2, 3, 5, 7]:
            _mr_ml[f'hrv_lag{_k}'] = _mr_ml['Data'].map(
                lambda d, k=_k: _dw_idx_adv.get(d + pd.Timedelta(days=k), np.nan))

    CORES_CAT_ADV = {'Leve':'#27ae60','Moderado':'#e67e22','Pesado':'#c0392b','Rest':'#7f8c8d'}

    # ── ACUTE MULTI-LAG ──────────────────────────────────────────────────
    with _adv_tabs[0]:
        st.markdown("**Como evolui o HRV nos dias após cada tipo de estímulo?**")
        if len(_mr_ml) == 0:
            st.info("Dados insuficientes.")
        else:
            _lag_days = [1, 2, 3, 5, 7]
            _ml_rows, _fig_ml = [], go.Figure()
            for cat in ['Leve','Moderado','Pesado','Rest']:
                sub_c = _mr_ml[_mr_ml['rpe_cat'] == cat]
                if len(sub_c) < 3: continue
                _ys = []
                row_ml = {'Categoria': cat, 'N': len(sub_c)}
                for _k in _lag_days:
                    v = sub_c[f'hrv_lag{_k}'].dropna()
                    if len(v) >= 3:
                        delta = (v.mean() - _base_hrv_adv) / _base_hrv_adv * 100
                        _ys.append(delta)
                        row_ml[f't+{_k}d'] = f"{delta:+.1f}%"
                    else:
                        _ys.append(None)
                        row_ml[f't+{_k}d'] = '—'
                _ml_rows.append(row_ml)
                _fig_ml.add_trace(go.Scatter(
                    x=[f't+{k}d' for k in _lag_days], y=_ys,
                    mode='lines+markers', name=cat,
                    line=dict(color=CORES_CAT_ADV.get(cat, '#555'), width=2.5),
                    marker=dict(size=8), connectgaps=True,
                    hovertemplate=f'{cat} %{{x}}: <b>%{{y:+.1f}}%</b><extra></extra>'
                ))
            _fig_ml.add_hline(y=0, line_dash='dash', line_color='#888', line_width=1)
            _fig_ml.update_layout(**LAYOUT_BASE, height=320, hovermode='x unified',
                title=dict(text='Δ HRV% após estímulo — evolução temporal',
                           font=dict(size=12, color='#111')),
                xaxis=dict(title='Dias após estímulo', tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd'),
                yaxis=dict(title='Δ HRV% vs baseline', tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd', zeroline=True, zerolinecolor='#888'),
                legend=dict(orientation='h', y=-0.22, font=dict(color='#111', size=11)))
            st.plotly_chart(_fig_ml, use_container_width=True, config=MC_LOC)
            if _ml_rows:
                df_ml = pd.DataFrame(_ml_rows)
                st.dataframe(df_ml, hide_index=True, use_container_width=True)
                st.download_button("📥 Download Acute Multi-lag",
                                   df_ml.to_csv(index=False, sep=';', decimal=',').encode(),
                                   "atheltica_acute_multilag.csv", "text/csv", key="dl_ml")

    # ── EVENT RESPONSE ───────────────────────────────────────────────────
    with _adv_tabs[1]:
        st.markdown("**Efeito real de cada zona — sem confounding temporal**")
        st.caption(
            "Sólido = janela limpa (dias seguintes leves/descanso). "
            "Tracejado = com treino intenso nos dias seguintes.")
        _er_window = st.slider("Dias seguintes que devem ser leves/descanso",
                                1, 4, 2, 1, key="er_clean_window")
        if len(_mr_ml) == 0:
            st.info("Dados insuficientes.")
        else:
            _er_lags = [1, 2, 3, 4, 5]
            _mr_er   = _mr_ml.copy()
            for _k in _er_lags:
                if f'hrv_lag{_k}' not in _mr_er.columns:
                    _mr_er[f'hrv_lag{_k}'] = _mr_er['Data'].map(
                        lambda d, k=_k: _dw_idx_adv.get(d + pd.Timedelta(days=k), np.nan))
            _rpe_by_date = dict(zip(pd.to_datetime(merged_rpe['Data']), merged_rpe['rpe_cat']))

            def _is_clean(date, n):
                for k in range(1, n + 1):
                    if _rpe_by_date.get(date + pd.Timedelta(days=k)) in ('Moderado', 'Pesado'):
                        return False
                return True

            _mr_er['clean'] = _mr_er['Data'].apply(lambda d: _is_clean(d, _er_window))
            _er_rows, _fig_er = [], go.Figure()
            _er_styles = {
                ('Pesado', True): ('#c0392b','solid'), ('Pesado', False): ('#c0392b','dot'),
                ('Leve',   True): ('#27ae60','solid'), ('Leve',   False): ('#27ae60','dot'),
                ('Rest',   True): ('#7f8c8d','solid'),
            }
            for cat in ['Leve','Moderado','Pesado','Rest']:
                for clean_flag, lbl_c in [(True, f'✅ limpa ({_er_window}d)'), (False, '❌ confounding')]:
                    sub_er = _mr_er[(_mr_er['rpe_cat']==cat) & (_mr_er['clean']==clean_flag)]
                    if len(sub_er) < 3: continue
                    _ys_er = []
                    row_er = {'Categoria': cat, 'Janela': lbl_c, 'N': len(sub_er)}
                    for _k in _er_lags:
                        v = sub_er[f'hrv_lag{_k}'].dropna()
                        if len(v) >= 3:
                            delta = (v.mean() - _base_hrv_adv) / _base_hrv_adv * 100
                            _ys_er.append(delta); row_er[f't+{_k}d'] = f"{delta:+.1f}%"
                        else:
                            _ys_er.append(None); row_er[f't+{_k}d'] = '—'
                    _er_rows.append(row_er)
                    clr, dash = _er_styles.get((cat, clean_flag), ('#555','solid'))
                    _fig_er.add_trace(go.Scatter(
                        x=[f't+{k}d' for k in _er_lags], y=_ys_er,
                        mode='lines+markers', name=f"{cat} {lbl_c}",
                        line=dict(color=clr, dash=dash, width=2),
                        marker=dict(size=7), connectgaps=True,
                        hovertemplate=f'{cat} %{{x}}: <b>%{{y:+.1f}}%</b><extra></extra>'
                    ))
            _fig_er.add_hline(y=0, line_dash='dash', line_color='#888', line_width=1)
            _fig_er.update_layout(**LAYOUT_BASE, height=340, hovermode='x unified',
                title=dict(text='Efeito limpo vs confounding', font=dict(size=12, color='#111')),
                xaxis=dict(title='Dias após o evento', tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd'),
                yaxis=dict(title='Δ HRV% vs baseline', tickfont=dict(color='#111'),
                           showgrid=True, gridcolor='#ddd', zeroline=True, zerolinecolor='#888'),
                legend=dict(orientation='h', y=-0.28, font=dict(color='#111', size=9)))
            st.plotly_chart(_fig_er, use_container_width=True, config=MC_LOC)
            if _er_rows:
                df_er = pd.DataFrame(_er_rows)
                _cols_er = ['Categoria','Janela','N'] + [f't+{k}d' for k in _er_lags]
                st.dataframe(df_er[_cols_er], hide_index=True, use_container_width=True)
                st.download_button("📥 Download Event Response",
                                   df_er[_cols_er].to_csv(index=False, sep=';', decimal=',').encode(),
                                   f"atheltica_event_{_er_window}d.csv", "text/csv", key="dl_er")

    # ── LOAD ECOLOGY ─────────────────────────────────────────────────────
    with _adv_tabs[2]:
        st.markdown("**Como a acumulação de carga (3d/5d/7d) se relaciona com o HRV?**")
        _da_eco = _da_use[_da_use['type'].apply(norm_tipo) != 'WeightTraining'].copy()
        _da_eco['Data'] = pd.to_datetime(_da_eco['Data'])
        if rpe_col and rpe_col in _da_eco.columns:
            _da_eco[rpe_col] = pd.to_numeric(_da_eco[rpe_col], errors='coerce')
            _eco_rpe = (_da_eco.dropna(subset=[rpe_col]).groupby('Data')[rpe_col]
                        .mean().reset_index().rename(columns={rpe_col:'rpe_day'}))
        else:
            _eco_rpe = pd.DataFrame(columns=['Data','rpe_day'])
        if 'icu_joules' in _da_eco.columns:
            _da_eco['_kje'] = pd.to_numeric(_da_eco['icu_joules'], errors='coerce') / 1000
            _eco_kj = _da_eco.groupby('Data').agg(kj_day=('_kje','sum')).reset_index()
        else:
            _eco_kj = pd.DataFrame(columns=['Data','kj_day'])

        _eco_ser = pd.DataFrame({'Data': pd.date_range('2020-01-01',
                                  pd.Timestamp.now().normalize(), freq='D')})
        if len(_eco_rpe): _eco_ser = _eco_ser.merge(_eco_rpe, on='Data', how='left')
        if len(_eco_kj):  _eco_ser = _eco_ser.merge(_eco_kj,  on='Data', how='left')
        for col in ['rpe_day','kj_day']:
            if col in _eco_ser.columns: _eco_ser[col] = _eco_ser[col].fillna(0)
        for _w in [3, 5, 7]:
            if 'rpe_day' in _eco_ser.columns:
                _eco_ser[f'rpe_roll{_w}d'] = _eco_ser['rpe_day'].rolling(_w, min_periods=1).mean()
            if 'kj_day' in _eco_ser.columns:
                _eco_ser[f'kj_roll{_w}d']  = _eco_ser['kj_day'].rolling(_w, min_periods=1).sum()
        if 'rpe_day' in _eco_ser.columns:
            _rm7 = _eco_ser['rpe_day'].rolling(7, min_periods=4).mean()
            _rs7 = _eco_ser['rpe_day'].rolling(7, min_periods=4).std()
            _eco_ser['monotonia_rpe'] = (_rm7 / _rs7.replace(0, np.nan)).round(3)
        _eco_ser = _eco_ser.merge(
            _dw_clean_adv[['Data','hrv']].rename(columns={'hrv':'hrv_hoje'}),
            on='Data', how='inner').dropna(subset=['hrv_hoje'])

        if len(_eco_ser) < 20:
            st.info("Dados insuficientes.")
        else:
            from scipy.stats import spearmanr as _eco_sr2, pearsonr as _eco_pr2
            _eco_vars = {
                'rpe_roll3d':'RPE médio 3d', 'rpe_roll5d':'RPE médio 5d',
                'rpe_roll7d':'RPE médio 7d', 'kj_roll3d':'kJ acum. 3d',
                'kj_roll5d':'kJ acum. 5d',   'kj_roll7d':'kJ acum. 7d',
                'monotonia_rpe':'Monotonia RPE (7d)',
            }
            _eco_pairs = []
            for var, lbl in _eco_vars.items():
                if var not in _eco_ser.columns: continue
                _d = _eco_ser[[var,'hrv_hoje']].dropna()
                if len(_d) < 15: continue
                x, y = _d[var].values.astype(float), _d['hrv_hoje'].values.astype(float)
                try:
                    r_s, p_s = _eco_sr2(x, y)
                    r_p, p_p = _eco_pr2(x, y)
                    sig = '✅ p<0.05' if min(p_s,p_p)<0.05 else '~ p<0.10' if min(p_s,p_p)<0.10 else '✗ ns'
                    _eco_pairs.append({'Variável':lbl, 'N':len(_d),
                                        'r Spearman':round(r_s,3), 'Sig':sig,
                                        'Interpretação':'↘ mais carga → HRV↓' if r_s<0 else '↗',
                                        '_r_abs':abs(r_s)})
                except Exception:
                    pass
            if _eco_pairs:
                df_eco = (pd.DataFrame(_eco_pairs).sort_values('_r_abs', ascending=False)
                          .drop(columns=['_r_abs']))
                _fig_eco = go.Figure(go.Bar(
                    y=df_eco['Variável'].tolist(), x=df_eco['r Spearman'].tolist(),
                    orientation='h',
                    marker_color=['#c0392b' if r<0 else '#27ae60' for r in df_eco['r Spearman']],
                    text=[f"{r:+.3f}" for r in df_eco['r Spearman']],
                    textposition='outside',
                    hovertemplate='%{y}<br>r: <b>%{x:+.3f}</b><extra></extra>'))
                _fig_eco.add_vline(x=0, line_color='#888', line_width=1)
                _fig_eco.update_layout(**LAYOUT_BASE, showlegend=False,
                    height=max(260, len(df_eco)*36+80),
                    title=dict(text='r Spearman: carga acumulada → HRV',
                               font=dict(size=12, color='#111')),
                    xaxis=dict(title='r Spearman', tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#ddd', range=[-0.6,0.6]),
                    yaxis=dict(tickfont=dict(color='#111', size=10)))
                st.plotly_chart(_fig_eco, use_container_width=True, config=MC_LOC)
                st.dataframe(df_eco[['Variável','N','r Spearman','Sig','Interpretação']],
                             hide_index=True, use_container_width=True)
                st.download_button("📥 Download Load Ecology",
                                   df_eco.to_csv(index=False, sep=';', decimal=',').encode(),
                                   "atheltica_load_ecology.csv", "text/csv", key="dl_eco")
            else:
                st.info("Dados insuficientes para Load Ecology.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # PERFIL DE RESPOSTA DO ATLETA (síntese final)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏅 Perfil de Resposta do Atleta")
    st.caption(
        "Síntese automática de todas as análises: impacto RPE, impacto por modalidade, "
        "análise de carga e fadiga, multi-lag e load ecology."
    )

    _perfil = []

    # A — Impacto RPE
    if len(merged_rpe) >= 5:
        base_p = merged_rpe['hrv'].mean()
        _perfil.append("**Impacto RPE → HRV (dia seguinte):**")
        for cat, emoji in [('Pesado','🔴'),('Moderado','🟡'),('Leve','🟢'),('Rest','🔵')]:
            sub_p = merged_rpe[merged_rpe['rpe_cat']==cat]
            if len(sub_p) < 2: continue
            d = (sub_p['hrv'].mean() - base_p) / base_p * 100
            interp = 'stress' if d<-3 else ('recuperação' if d>3 else 'neutro')
            _perfil.append(f"  {emoji} **{cat}**: {d:+.1f}% → {interp}")
        kw_pf = _stat_kruskal(merged_rpe, 'rpe_cat', ['Leve','Moderado','Pesado','Rest'])
        if 'hrv' in kw_pf:
            eta2 = kw_pf['hrv']['eta2']
            lbl_e = "forte ✅" if eta2>=0.14 else "médio ✅" if eta2>=0.06 else "pequeno ⚠️" if eta2>=0.01 else "negligenciável ❌"
            _perfil.append(f"  📊 Força do sinal: η²={eta2:.3f} ({lbl_e})")

    # B — Por modalidade
    if len(merged_modal) >= 5:
        _perfil.append("")
        _perfil.append("**Por modalidade (sessões pesadas):**")
        base_pm = merged_modal['hrv'].mean()
        for mod in CICLICOS_T:
            sub_pm = merged_modal[(merged_modal['modalidade']==mod)&(merged_modal['rpe_cat']=='Pesado')]
            if len(sub_pm) < 2: continue
            d = (sub_pm['hrv'].mean() - base_pm) / base_pm * 100
            _perfil.append(f"  → **{mod} Pesado**: {d:+.1f}% HRV")

    # C — Multi-lag
    if '_ml_rows' in dir() and _ml_rows:
        _perfil.append("")
        _perfil.append("**Duração do efeito (multi-lag):**")
        df_ml_pf = pd.DataFrame(_ml_rows)
        for cat in ['Pesado','Leve']:
            row_ml = df_ml_pf[df_ml_pf['Categoria']==cat]
            if len(row_ml)==0: continue
            row_ml = row_ml.iloc[0]
            crossings = []
            for k in [1,2,3,5,7]:
                try:
                    v = float(str(row_ml.get(f't+{k}d','—')).replace('%','').replace('+',''))
                    if abs(v) < 2.0: crossings.append(k)
                except: pass
            if crossings:
                _perfil.append(f"  → **{cat}**: efeito aproxima-se de neutro em t+{crossings[0]}d")
            else:
                _perfil.append(f"  → **{cat}**: efeito persiste até t+7d")

    # D — Load Ecology
    if '_eco_pairs' in dir() and _eco_pairs:
        df_eco_pf = pd.DataFrame(_eco_pairs).sort_values('_r_abs', ascending=False)
        best = df_eco_pf.iloc[0]
        _perfil.append("")
        _perfil.append("**Carga acumulada (Load Ecology):**")
        _perfil.append(f"  → Melhor preditor: **{best['Variável']}** r={best['r Spearman']:+.3f} {best['Sig']}")

    if _perfil:
        for ln in _perfil:
            st.markdown(ln if ln else "---")
    else:
        st.info("Dados insuficientes para gerar perfil automático.")

    st.info(
        "Δ HRV% negativo = HRV mais baixo = stress/fadiga. "
        "η² > 0.06 = sinal real. "
        "r Spearman: |r|>0.5 forte | |r|>0.3 moderado."
    )

    # Download consolidado
    _all_dl = []
    if len(merged_rpe) > 0: _all_dl.append(merged_rpe.assign(tabela='RPE'))
    if len(merged_tipo) > 0: _all_dl.append(merged_tipo.assign(tabela='Tipo'))
    if _all_dl:
        st.download_button(
            "📥 Download consolidado completo",
            pd.concat(_all_dl, ignore_index=True)
            .to_csv(index=False, sep=';', decimal=',').encode(),
            "atheltica_correlacoes_completo.csv", "text/csv", key="dl_corr_all")
