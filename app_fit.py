"""
ATHELTICA FIT — app dedicada só à análise de ficheiros .fit
=============================================================
App separada do dashboard principal (susigan/dashboard), para que a análise
FIT corra isolada — sem as outras 16 abas do dashboard principal a
re-executarem em cada interação (era essa a causa da lentidão).

Reutiliza tal e qual:
  - utils/fit_analyzer.py  (toda a lógica de leitura/análise do .fit)
  - tabs/tab_fit_analise.py (toda a interface já construída e testada)

Se precisares de dados do Google Sheets/Drive aqui (ex.: para guardar o
histórico de sessões de forma persistente, em vez de só na sessão), integra
utils/data.py da mesma forma que o dashboard principal já faz — ver notas no
fim deste ficheiro.
"""

import streamlit as st

st.set_page_config(
    page_title="ATHELTICA — Análise FIT",
    page_icon="🔬",
    layout="wide",
)

from tabs.tab_fit_analise import tab_fit_analise

tab_fit_analise()

# ─────────────────────────────────────────────────────────────────────────
# NOTA — Google Sheets / Drive (opcional, para persistência do histórico)
# ─────────────────────────────────────────────────────────────────────────
# tab_fit_analise() não precisa de nenhuma ligação ao Google Sheets/Drive
# para funcionar — o upload, a correção de laps e a análise são
# completamente autossuficientes a partir do ficheiro .fit.
#
# Se quiseres que o histórico de sessões fique persistente entre sessões
# (hoje só vive em st.session_state, perde-se ao fechar o browser), a forma
# mais fiel de fazer isso é reutilizar exatamente as mesmas funções de
# ligação que o dashboard principal já usa em utils/data.py — e os mesmos
# "Secrets" (credenciais da service account) configurados nesta nova app no
# Streamlit Cloud (em Settings → Secrets), copiados do dashboard principal.
