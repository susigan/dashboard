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

def tab_zones(da, mods_sel):
    st.header("❤️ HR Zones & RPE Zones")
    df = filtrar_principais(da).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df['ano']  = df['Data'].dt.year
    df = df[df['type'].isin(mods_sel)]
    if len(df) == 0:
        st.warning("Sem dados.")
        return

    zonas_hr = [c for c in df.columns if c.lower().startswith('hr_z') and c.lower().endswith('_secs')]
    rpe_col  = next((c for c in ['rpe','RPE','icu_rpe'] if c in df.columns), None)

    def gzn(col):
        m = re.search(r'hr_z(\d+)_secs', col.lower())
        return int(m.group(1)) if m else 0

    CORES_HR  = {'Baixa (Z1+Z2)':'#2ECC71', 'Moderada (Z3+Z4)':'#F39C12', 'Alta (Z5+Z6+Z7)':'#E74C3C'}
    CORES_RPE = {'Leve (1–4)':'#3498DB',    'Moderado (5–7)':'#F39C12',    'Forte (7–10)':'#E74C3C'}

    def _prep_hr(df_in):
        dh = df_in.copy()
        bc = [c for c in zonas_hr if gzn(c) in (1,2)]
        mc = [c for c in zonas_hr if gzn(c) in (3,4)]
        ac = [c for c in zonas_hr if gzn(c) in (5,6,7)]
        for cols in [bc, mc, ac]:
            for c in cols:
                dh[c] = pd.to_numeric(dh[c], errors='coerce').fillna(0)
        dh['zb'] = dh[bc].sum(axis=1) if bc else 0
        dh['zm'] = dh[mc].sum(axis=1) if mc else 0
        dh['za'] = dh[ac].sum(axis=1) if ac else 0
        dh['zt'] = dh['zb'] + dh['zm'] + dh['za']
        dh = dh[dh['zt'] > 0].copy()
        for z, p in [('zb','pb'),('zm','pm'),('za','pa')]:
            dh[p] = dh[z] / dh['zt'] * 100
        return dh

    def _prep_rpe(df_in):
        if not rpe_col: return pd.DataFrame()
        dr = df_in.dropna(subset=[rpe_col]).copy()
        dr[rpe_col] = pd.to_numeric(dr[rpe_col], errors='coerce')
        dr = dr.dropna(subset=[rpe_col])
        dr['rz'] = pd.cut(dr[rpe_col], bins=[0,4.9,6.9,10],
                          labels=list(CORES_RPE.keys()), right=True)
        return dr.dropna(subset=['rz'])

    LEGEND = dict(orientation='h', y=-0.22, font=dict(color='#111111', size=11),
                  bgcolor='rgba(255,255,255,0.9)', bordercolor='#ccc', borderwidth=1)
    PW = dict(paper_bgcolor='white', plot_bgcolor='white',
              font=dict(color='#222222', family='Arial'))

    # ── Botões de ano (substituem dropdown) ──────────────────────────────
    anos = sorted(df['ano'].unique())
    # Estado persistente
    if 'zones_ano_sel' not in st.session_state or st.session_state['zones_ano_sel'] not in anos:
        st.session_state['zones_ano_sel'] = anos[-1]

    st.markdown("**📅 Seleccionar ano:**")
    _btn_cols = st.columns(len(anos))
    for _i, _ano in enumerate(anos):
        _is_active = (st.session_state['zones_ano_sel'] == _ano)
        _bg  = "#1a73e8" if _is_active else "#f0f2f6"
        _clr = "white"   if _is_active else "#333"
        if _btn_cols[_i].button(
            str(_ano),
            key=f"zones_btn_{_ano}",
            help=f"Ver dados de {_ano}",
            use_container_width=True,
            type="primary" if _is_active else "secondary"
        ):
            st.session_state['zones_ano_sel'] = _ano
            st.rerun()

    ano_sel = st.session_state['zones_ano_sel']
    df_ano  = df[df['ano'] == ano_sel].copy()

    # ════════════════════════════════════════════════════════════════════
    # GRÁFICOS — ano seleccionado
    # ════════════════════════════════════════════════════════════════════
    dh_ano = _prep_hr(df_ano)  if zonas_hr else pd.DataFrame()
    dr_ano = _prep_rpe(df_ano) if rpe_col   else pd.DataFrame()

    col1, col2 = st.columns(2)

    # HR por modalidade — barras stacked com % dentro
    if len(dh_ano) > 0:
        pt_hr = dh_ano.groupby('type')[['pb','pm','pa']].mean().reset_index()
        pt_hr.columns = ['type'] + list(CORES_HR.keys())

        fig_hr_mod = go.Figure()
        for zona, cor in CORES_HR.items():
            vals = pt_hr[zona]
            fig_hr_mod.add_trace(go.Bar(
                x=pt_hr['type'], y=vals, name=zona,
                marker_color=cor,
                text=[f"{v:.0f}%" for v in vals],
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial'),
                hovertemplate=f'{zona}: <b>%{{y:.1f}}%</b><extra></extra>'))
        fig_hr_mod.update_layout(**PW,
            barmode='stack',
            title=dict(text=f'❤️ HR Zones por Modalidade — {ano_sel}',
                       font=dict(color='#222', size=13)),
            height=380, legend=LEGEND,
            xaxis=dict(title='', tickfont=dict(color='#333333'), showgrid=False,
                       linecolor='#ccc', showline=True),
            yaxis=dict(title='%', tickfont=dict(color='#333333'),
                       showgrid=True, gridcolor='#eeeeee', range=[0,108]))
        with col1:
            st.plotly_chart(fig_hr_mod, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

        # HR Geral pizza
        ts_hr = {z: dh_ano[c].sum()
                 for z, c in zip(CORES_HR.keys(), ['zb','zm','za'])}
        fig_hr_gen = go.Figure(go.Pie(
            labels=list(ts_hr.keys()), values=list(ts_hr.values()),
            marker=dict(colors=list(CORES_HR.values())),
            textinfo='label+percent',
            textfont=dict(color='white', size=11),
            hovertemplate='%{label}: <b>%{percent}</b><extra></extra>'))
        fig_hr_gen.update_layout(**PW,
            title=dict(text=f'❤️ HR Geral — {ano_sel}', font=dict(color='#222', size=13)),
            height=380, showlegend=True,
            legend=dict(font=dict(color='#111111', size=11)))
        with col2:
            st.plotly_chart(fig_hr_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    # RPE por modalidade
    col3, col4 = st.columns(2)
    if len(dr_ano) > 0:
        piv_r = (dr_ano.groupby(['type','rz'], observed=True)
                 .size().unstack(fill_value=0))
        for z in CORES_RPE.keys():
            if z not in piv_r.columns: piv_r[z] = 0
        piv_r   = piv_r[list(CORES_RPE.keys())]
        pct_r   = piv_r.div(piv_r.sum(axis=1), axis=0) * 100
        pct_r   = pct_r.reset_index()

        fig_rpe_mod = go.Figure()
        for zona, cor in CORES_RPE.items():
            vals = pct_r[zona]
            fig_rpe_mod.add_trace(go.Bar(
                x=pct_r['type'], y=vals, name=zona,
                marker_color=cor,
                text=[f"{v:.0f}%" for v in vals],
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial'),
                hovertemplate=f'{zona}: <b>%{{y:.1f}}%</b><extra></extra>'))
        fig_rpe_mod.update_layout(**PW,
            barmode='stack',
            title=dict(text=f'🎯 RPE Zones por Modalidade — {ano_sel}',
                       font=dict(color='#222', size=13)),
            height=380, legend=LEGEND,
            xaxis=dict(title='', tickfont=dict(color='#333333'), showgrid=False,
                       linecolor='#ccc', showline=True),
            yaxis=dict(title='%', tickfont=dict(color='#333333'),
                       showgrid=True, gridcolor='#eeeeee', range=[0,108]))
        with col3:
            st.plotly_chart(fig_rpe_mod, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

        ts_rpe = {z: piv_r[z].sum() for z in CORES_RPE.keys()}
        fig_rpe_gen = go.Figure(go.Pie(
            labels=list(ts_rpe.keys()), values=list(ts_rpe.values()),
            marker=dict(colors=list(CORES_RPE.values())),
            textinfo='label+percent',
            textfont=dict(color='white', size=11),
            hovertemplate='%{label}: <b>%{percent}</b><extra></extra>'))
        fig_rpe_gen.update_layout(**PW,
            title=dict(text=f'🎯 RPE Geral — {ano_sel}', font=dict(color='#222', size=13)),
            height=380, showlegend=True,
            legend=dict(font=dict(color='#111111', size=11)))
        with col4:
            st.plotly_chart(fig_rpe_gen, use_container_width=True, config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False})

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA ANUAL — separada por ano, linha Geral em cinza
    # ════════════════════════════════════════════════════════════════════
    st.subheader("📊 HR Zones & RPE Zones — todos os anos")
    st.caption(
        "% média por modalidade. "
        "🟢 Verde = maioria Z1 (polarizado) | 🟡 Amarelo = Z2 | 🔴 Vermelho = Z3 dominante. "
        "Linha 'Geral' em cinza claro."
    )

    rows_anual = []
    for ano in sorted(df['ano'].unique(), reverse=True):
        df_a  = df[df['ano'] == ano]
        dh_a  = _prep_hr(df_a)  if zonas_hr else pd.DataFrame()
        rpe_a = _prep_rpe(df_a) if rpe_col   else pd.DataFrame()

        mods_ano = sorted(set(
            (dh_a['type'].unique().tolist() if len(dh_a) > 0 else []) +
            (rpe_a['type'].unique().tolist() if len(rpe_a) > 0 else [])))

        for mod in mods_ano + ['Geral']:
            is_geral = (mod == 'Geral')
            row = {'Ano': str(ano) if not is_geral else '', 'Modalidade': mod}

            sub_h = dh_a  if is_geral else (dh_a[dh_a['type']==mod]   if len(dh_a)  > 0 else pd.DataFrame())
            sub_r = rpe_a if is_geral else (rpe_a[rpe_a['type']==mod]  if len(rpe_a) > 0 else pd.DataFrame())

            if len(sub_h) > 0:
                row['HR Z1+Z2%'] = f"{sub_h['pb'].mean():.0f}%"
                row['HR Z3+Z4%'] = f"{sub_h['pm'].mean():.0f}%"
                row['HR Z5+%']   = f"{sub_h['pa'].mean():.0f}%"
            else:
                row['HR Z1+Z2%'] = row['HR Z3+Z4%'] = row['HR Z5+%'] = '—'

            if len(sub_r) > 0:
                t = len(sub_r)
                for z, lbl in [('Leve (1–4)','RPE Z1%'),
                                ('Moderado (5–7)','RPE Z2%'),
                                ('Forte (7–10)','RPE Z3%')]:
                    row[lbl] = f"{(sub_r['rz']==z).sum()/t*100:.0f}%"
            else:
                row['RPE Z1%'] = row['RPE Z2%'] = row['RPE Z3%'] = '—'

            row['_geral'] = is_geral
            rows_anual.append(row)

    if rows_anual:
        # Combinar HR|RPE
        for row in rows_anual:
            for z_hr, z_rpe, z_lbl in [
                ('HR Z1+Z2%','RPE Z1%','Z1 HR|RPE'),
                ('HR Z3+Z4%','RPE Z2%','Z2 HR|RPE'),
                ('HR Z5+%',  'RPE Z3%','Z3 HR|RPE'),
            ]:
                row[z_lbl] = f"{row.get(z_hr,'—')} | {row.get(z_rpe,'—')}"

        def _pct_to_int(s):
            m = re.search(r'(\d+)%', str(s))
            return int(m.group(1)) if m else 0

        def _zone_cell_color(pct_val, zone_idx):
            """Cor de fundo baseada na % da zona."""
            if pct_val == 0: return '#ffffff'
            if zone_idx == 0:   # Z1: mais verde = mais polarizado
                i = min(pct_val, 80) / 80
                return f'rgb({int(255-i*80)},{int(240-i*30)},{int(210-i*90)})'
            elif zone_idx == 1: # Z2: amarelo suave
                i = min(pct_val, 50) / 50
                return f'rgb(255,{int(248-i*50)},{int(205-i*100)})'
            else:               # Z3: vermelho quanto mais alto
                i = min(pct_val, 40) / 40
                return f'rgb(255,{int(240-i*130)},{int(240-i*130)})'

        df_anual     = pd.DataFrame(rows_anual)
        display_cols = ['Ano','Modalidade','Z1 HR|RPE','Z2 HR|RPE','Z3 HR|RPE']
        zone_cols    = ['Z1 HR|RPE','Z2 HR|RPE','Z3 HR|RPE']

        html  = '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        html += '<tr style="background:#2c3e50;color:#fff">'
        for c in display_cols:
            html += (f'<th style="border:1px solid #444;padding:7px 12px;'
                     f'text-align:center;font-weight:600">{c}</th>')
        html += '</tr>'
        for _, row in df_anual.iterrows():
            is_geral = row.get('_geral', False)
            row_bg   = '#ecf0f1' if is_geral else '#fff'
            fw       = '700'    if is_geral else '400'
            html += '<tr>'
            for c in display_cols:
                val = row.get(c, '—')
                if c in zone_cols and not is_geral:
                    zi  = zone_cols.index(c)
                    bg  = _zone_cell_color(_pct_to_int(val), zi)
                    html += (f'<td style="border:1px solid #ddd;padding:5px 10px;'
                             f'text-align:center;background:{bg};font-weight:{fw}">{val}</td>')
                else:
                    html += (f'<td style="border:1px solid #ddd;padding:5px 10px;'
                             f'text-align:center;background:{row_bg};color:#222;font-weight:{fw}">{val}</td>')
            html += '</tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # CORRELAÇÃO HR Zone × RPE Zone — tabela simplificada
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🔗 HR Zone × RPE — Distribuição e Correlação")

    if zonas_hr and rpe_col:
        dh_ok = _prep_hr(df.copy())
        if len(dh_ok) > 0:
            dh_ok[rpe_col] = pd.to_numeric(dh_ok[rpe_col], errors='coerce')
            dh_ok = dh_ok.dropna(subset=[rpe_col,'pb','pm','pa'])
            dh_ok = dh_ok[dh_ok['zt'] > 0]
            dh_ok['rpe_int'] = dh_ok[rpe_col].round(0).astype(int).clip(1, 10)

            HR_VARS  = [('pb','Z1+Z2','#2ECC71'), ('pm','Z3+Z4','#F39C12'), ('pa','Z5+','#E74C3C')]
            forca_fn = lambda r: ('Muito Forte' if abs(r) >= 0.7 else
                                   'Forte'      if abs(r) >= 0.5 else
                                   'Moderada'   if abs(r) >= 0.3 else 'Fraca')

            mods_disp = sorted(dh_ok['type'].unique().tolist())

            # ── Gráfico compacto: boxplot HR Zone por RPE inteiro × modalidade ──
            st.caption(
                "**Range de HR Zone (%) por nível de RPE** — "
                "boxplot por modalidade. Cada cor = zona HR.")

            _n_mods = len(mods_disp)
            _fig_box = make_subplots(
                rows=1, cols=_n_mods,
                subplot_titles=[f"<b>{m}</b>" for m in mods_disp],
                horizontal_spacing=0.06
            )
            _rpe_levels = sorted(dh_ok['rpe_int'].unique())

            for _ci, mod in enumerate(mods_disp, 1):
                dm = dh_ok[dh_ok['type'] == mod]
                if len(dm) < 3: continue
                for hv, lbl, cor in HR_VARS:
                    y_vals, x_vals = [], []
                    for rpe_v in _rpe_levels:
                        vals = dm[dm['rpe_int'] == rpe_v][hv].dropna().values
                        if len(vals) >= 2:
                            y_vals.extend(vals.tolist())
                            x_vals.extend([f"RPE {rpe_v}"] * len(vals))
                    if y_vals:
                        _fig_box.add_trace(go.Box(
                            x=x_vals, y=y_vals,
                            name=lbl, legendgroup=lbl,
                            marker_color=cor, line_color=cor,
                            showlegend=(_ci == 1),
                            boxmean=False,
                            hovertemplate=f'{lbl}<br>RPE: %{{x}}<br>HR%: <b>%{{y:.0f}}%</b><extra></extra>'
                        ), row=1, col=_ci)

            _fig_box.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=10),
                height=320,
                boxmode='group',
                margin=dict(l=30, r=10, t=40, b=40),
                legend=dict(orientation='h', y=-0.18,
                            font=dict(color='#111', size=10)),
                showlegend=True
            )
            _fig_box.update_yaxes(title_text='HR Zone %', tickfont=dict(size=9, color='#333'),
                                   showgrid=True, gridcolor='#eee', range=[0, 105])
            _fig_box.update_xaxes(tickfont=dict(size=8, color='#333'), showgrid=False)
            st.plotly_chart(_fig_box, use_container_width=True,
                            config={'displayModeBar': False, 'responsive': True})

            st.markdown("---")

            # ── Tabs por modalidade: correlação por ano ────────────────────────
            st.caption(
                "**Correlação HR Zone × RPE Zone por modalidade e ano.** "
                "Força: Muito Forte ≥0.7 | Forte ≥0.5 | Moderada ≥0.3 | Fraca <0.3")

            _mod_tabs = st.tabs([f"🚴 {m}" if m=='Bike' else
                                  f"🚣 {m}" if m=='Row'  else
                                  f"🎿 {m}" if m=='Ski'  else
                                  f"🏃 {m}" if m=='Run'  else m
                                  for m in mods_disp])

            _rpe_zone_labels = [
                ('pb', 'Z1+Z2 (baixa)',  'Leve (RPE 1–4)'),
                ('pm', 'Z3+Z4 (média)',  'Moderado (RPE 5–6)'),
                ('pa', 'Z5+ (alta)',     'Forte (RPE 7–10)'),
            ]

            for _ti, mod in enumerate(mods_disp):
                with _mod_tabs[_ti]:
                    dm_all = dh_ok[dh_ok['type'] == mod]
                    if len(dm_all) < 5:
                        st.info(f"Poucos dados para {mod}.")
                        continue

                    _anos_mod = sorted(dm_all['ano'].dropna().unique().astype(int).tolist())
                    _corr_rows = []
                    for _ano_c in _anos_mod + ['Total']:
                        dm_c = dm_all if _ano_c == 'Total' else dm_all[dm_all['ano'] == _ano_c]
                        if len(dm_c) < 5: continue
                        for hv, hr_lbl, rpe_lbl in _rpe_zone_labels:
                            x = dm_c[rpe_col].dropna().values.astype(float)
                            y = dm_c[hv].dropna().values.astype(float)
                            # alinhar índices
                            _df_xy = pd.DataFrame({'x': dm_c[rpe_col], 'y': dm_c[hv]}).dropna()
                            if len(_df_xy) < 5: continue
                            from scipy.stats import pearsonr as _pr_zn
                            r_v, _ = _pr_zn(_df_xy['x'].astype(float), _df_xy['y'].astype(float))
                            forca_v = forca_fn(r_v)
                            # Emoji força
                            emoji_f = ('🔴' if forca_v == 'Muito Forte' else
                                       '🟠' if forca_v == 'Forte' else
                                       '🟡' if forca_v == 'Moderada' else '🟢')
                            # Direcção
                            dir_v = '↗ RPE↑ → mais HR nesta zona' if r_v > 0 else '↘ RPE↑ → menos HR nesta zona'
                            _corr_rows.append({
                                'Ano': str(_ano_c),
                                'HR Zone': hr_lbl,
                                'RPE Zone': rpe_lbl,
                                'Força': f"{emoji_f} {forca_v}",
                                'Direcção': dir_v,
                            })

                    if _corr_rows:
                        _df_corr = pd.DataFrame(_corr_rows)
                        # Pivotar: HR Zone como linhas, Anos como colunas
                        try:
                            _pivot = _df_corr.pivot_table(
                                index=['HR Zone','RPE Zone'],
                                columns='Ano',
                                values='Força',
                                aggfunc='first'
                            ).reset_index()
                            _pivot.columns.name = None
                            st.dataframe(_pivot, hide_index=True, use_container_width=True)
                        except Exception:
                            # Fallback: tabela plana
                            st.dataframe(_df_corr, hide_index=True, use_container_width=True)

                        st.caption(
                            "🔴 Muito Forte (r≥0.7) | 🟠 Forte (r≥0.5) | "
                            "🟡 Moderada (r≥0.3) | 🟢 Fraca (r<0.3). "
                            "↗ = RPE alto correlaciona com mais tempo nesta HR zone. "
                            "↘ = RPE alto correlaciona com menos tempo nesta HR zone.")
