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

    # ── Dropdown ano (só para gráficos) ──────────────────────────────────
    anos = sorted(df['ano'].unique())
    ano_sel = st.selectbox("📅 Ano (gráficos)", anos, index=len(anos)-1)
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
    st.caption("% média por modalidade cíclica. Linha 'Geral' em cinza claro.")

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
        # 1. Combinar HR|RPE ANTES de criar DataFrame
        for row in rows_anual:
            for z_hr, z_rpe, z_lbl in [
                ('HR Z1+Z2%','RPE Z1%','Z1 (HR|RPE)'),
                ('HR Z3+Z4%','RPE Z2%','Z2 (HR|RPE)'),
                ('HR Z5+%',  'RPE Z3%','Z3 (HR|RPE)'),
            ]:
                row[z_lbl] = f"{row.get(z_hr,'—')} | {row.get(z_rpe,'—')}"

        # 2. Criar DataFrame depois da combinação
        df_anual     = pd.DataFrame(rows_anual)
        display_cols = ['Ano','Modalidade',
                        'Z1 (HR|RPE)','Z2 (HR|RPE)','Z3 (HR|RPE)']

        # 3. Render HTML com linha Geral em cinza
        html  = ('<table style="border-collapse:collapse;width:100%;'
                 'font-size:12px;background:#fff;color:#222">')
        html += '<tr style="background:#e0e0e0">'
        for c in display_cols:
            html += (f'<th style="border:1px solid #ccc;padding:6px 10px;'
                     f'text-align:center;color:#111;font-weight:bold">{c}</th>')
        html += '</tr>'
        for _, row in df_anual.iterrows():
            bg = '#f0f0f0' if row.get('_geral', False) else '#ffffff'
            fw = 'bold'    if row.get('_geral', False) else 'normal'
            html += f'<tr style="background:{bg}">'
            for c in display_cols:
                html += (f'<td style="border:1px solid #ddd;padding:5px 10px;'
                         f'text-align:center;color:#222;font-weight:{fw}">'
                         f'{row.get(c,"—")}</td>')
            html += '</tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # CORRELAÇÃO HR Zone × RPE Zone — tabela simplificada
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🔗 Correlação HR Zone × RPE Zone")
    st.caption(
        "Pearson r entre % de tempo em cada HR Zone e RPE numérico. "
        "**Força Forte/Muito Forte = HR Zone e RPE altamente alinhados** "
        "(quando RPE sobe, tempo nessa HR Zone sobe ou desce consistentemente).")

    if zonas_hr and rpe_col:
        dh_ok = _prep_hr(df.copy())
        if len(dh_ok) > 0:
            dh_ok[rpe_col] = pd.to_numeric(dh_ok[rpe_col], errors='coerce')
            dh_ok = dh_ok.dropna(subset=[rpe_col,'pb','pm','pa'])
            dh_ok = dh_ok[dh_ok['zt'] > 0]

            HR_VARS   = [('pb','Baixa (Z1+Z2)', 'Leve (1–4)'),
                         ('pm','Moderada (Z3+Z4)', 'Moderado (5–7)'),
                         ('pa','Alta (Z5+Z6+Z7)', 'Forte (7–10)')]
            forca_fn  = lambda r: ('Muito Forte' if abs(r) >= 0.7 else
                                   'Forte'       if abs(r) >= 0.5 else
                                   'Moderada'    if abs(r) >= 0.3 else 'Fraca')

            mods_corr = sorted(dh_ok['type'].unique().tolist())  # todos os tipos com dados
            rows_corr = []
            for mod in mods_corr:
                dm = dh_ok[dh_ok['type'] == mod]
                if len(dm) < 3: continue
                for hv, hr_lbl, rpe_lbl in HR_VARS:
                    x = dm[rpe_col].values
                    y = dm[hv].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() < 5 or np.std(y[mask]) == 0:
                        continue
                    r, p = pearsonr(x[mask], y[mask])
                    sig  = ('***' if p<0.001 else '**' if p<0.01
                            else '*' if p<0.05 else 'ns')
                    # Ranges observados
                    hr_v  = dm[hv].dropna()
                    rpe_v = dm[rpe_col].dropna()
                    hr_rng  = (f"{hr_v.min():.0f}–{hr_v.max():.0f}%"
                               if len(hr_v) > 0 else '—')
                    rpe_rng = (f"{rpe_v.min():.1f}–{rpe_v.max():.1f}"
                               if len(rpe_v) > 0 else '—')
                    rows_corr.append({
                        'Modalidade': mod,
                        'HR Zone':    f"{hr_lbl} [{hr_rng}]",
                        'RPE Zone':   f"{rpe_lbl} [{rpe_rng}]",
                        'r':          f"{r:+.3f}",
                        'p':          f"{p:.4f}",
                        'Sig.':       sig,
                        'n':          int(mask.sum()),
                        'Força':      forca_fn(r),
                    })

            if rows_corr:
                st.dataframe(pd.DataFrame(rows_corr), width="stretch", hide_index=True)
