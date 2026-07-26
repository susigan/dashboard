"""
ATHELTICA FIT — app dedicada só à análise de ficheiros .fit
=============================================================
App separada do dashboard principal (susigan/dashboard), para que a análise
FIT corra isolada — sem as outras 16 abas do dashboard principal a
re-executarem em cada interação (era essa a causa da lentidão).

Reutiliza tal e qual:
  - utils/fit_analyzer.py  (toda a lógica de leitura/análise do .fit)
  - tabs/tab_fit_analise.py (toda a interface já construída e testada)

Agora TAMBÉM carrega o histórico de atividades (ac_full) do Intervals.icu,
com a mesma cadeia que o dashboard principal usa (app.py):
    ac_full = preproc_ativ(carregar_atividades(9999))
Isto alimenta a secção "Comparação com histórico" em tab_fit_analise() —
os limiares (HRVT1, HRVT2, PBP, etc.) calculados pelo Intervals.icu ao
longo do tempo, por modalidade e por ano, ao lado dos valores que a
Análise FIT encontra NESTA sessão específica.

Precisa dos mesmos Secrets (credenciais da service account) já configurados
no dashboard principal — confirmado que já estão copiados para esta app.
"""

import streamlit as st

st.set_page_config(
    page_title="ATHELTICA — Análise FIT",
    page_icon="🔬",
    layout="wide",
)

from utils.data import carregar_atividades, preproc_ativ
from tabs.tab_fit_analise import tab_fit_analise


@st.cache_data(ttl=3600, show_spinner="A carregar histórico de atividades...")
def _carregar_ac_full():
    """
    Mesma cadeia que o app.py principal usa para construir ac_full —
    histórico completo (9999 dias), pré-processado. Cacheado 1h: não é
    preciso recarregar isto em cada interação com a Análise FIT.
    """
    return preproc_ativ(carregar_atividades(9999))


try:
    _ac_full = _carregar_ac_full()
except Exception as _e:
    st.warning(
        f"⚠️ Não foi possível carregar o histórico de atividades do Intervals.icu "
        f"({_e}) — a Análise FIT continua a funcionar normalmente, só a secção "
        "'Comparação com histórico' fica indisponível nesta sessão."
    )
    _ac_full = None

tab_fit_analise(ac_full=_ac_full)

# ─────────────────────────────────────────────────────────────────────────
# NOTA — Google Sheets / Drive (opcional, para persistência do histórico)
# ─────────────────────────────────────────────────────────────────────────
# tab_fit_analise() não precisa de nenhuma ligação ao Google Sheets/Drive
# para a análise em si (upload, correção de laps, limiares) — só a nova
# secção "Comparação com histórico" (acima) depende de ac_full.
#
# Se quiseres que o histórico de SESSÕES FIT fique persistente entre
# sessões do browser (hoje só vive em st.session_state), a forma mais fiel
# de fazer isso é reutilizar as mesmas funções de utils/data.py para
# escrever numa aba/sheet dedicada.
# "Secrets" (credenciais da service account) configurados nesta nova app no
# Streamlit Cloud (em Settings → Secrets), copiados do dashboard principal.
