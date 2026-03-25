#!/usr/bin/env python3
"""
build.py — ATHELTICA Dashboard
════════════════════════════════════════════════════════════════════════════════
Junta todos os módulos num único ficheiro app_bundle.py.

Usar quando:
  - O Streamlit Cloud não suporta imports locais (pasta sem __init__.py)
  - Queres distribuir um único ficheiro
  - Estás a depurar e queres um ficheiro monolítico

Uso:
    python build.py
    → gera app_bundle.py (deploy este ficheiro como app.py no GitHub)

Fluxo de desenvolvimento recomendado:
    1. Edita os ficheiros modulares (tabs/tab_pmc.py, utils/helpers.py, etc.)
    2. Corre: python build.py
    3. Faz commit do app_bundle.py como app.py no GitHub
    4. O Streamlit Cloud reconstrui automaticamente
════════════════════════════════════════════════════════════════════════════════
"""

import re
import os
import ast
from datetime import datetime

# ── Ordem dos ficheiros a injetar ─────────────────────────────────────────────
BUILD_ORDER = [
    "config.py",
    "utils/helpers.py",
    "data_loader.py",
    "tabs/tab_visao_geral.py",
    "tabs/tab_pmc.py",
    "tabs/tab_volume.py",
    "tabs/tab_eftp.py",
    "tabs/tab_zones.py",
    "tabs/tab_correlacoes.py",
    "tabs/tab_recovery.py",
    "tabs/tab_wellness.py",
    "tabs/tab_analises.py",
    "tabs/tab_aquecimento.py",
]

# Imports que devem aparecer uma só vez no bundle
GLOBAL_IMPORTS = """\
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, linregress, spearmanr, theilslopes
from itertools import combinations
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
"""

# Padrões a remover dos ficheiros individuais (imports locais + duplicados)
REMOVE_PATTERNS = [
    r'^import streamlit.*\n',
    r'^import pandas.*\n',
    r'^import numpy.*\n',
    r'^import matplotlib.*\n',
    r'^from matplotlib.*\n',
    r'^import seaborn.*\n',
    r'^import gspread.*\n',
    r'^from gspread.*\n',
    r'^from google.*\n',
    r'^from scipy.*\n',
    r'^import re\n',
    r'^import warnings.*\n',
    r'^warnings\.filterwarnings.*\n',
    r'^plt\.style\.use.*\n',
    r'^from datetime.*\n',
    r'^from config import.*\n',
    r'^from utils\.helpers import.*\n',
    r'^from data_loader import.*\n',
    r'^from tabs\.\w+ import.*\n',
    r'^# AUTO-GENERATED.*\n',
    r'^import os\n',
]


def strip_local_imports(src: str) -> str:
    """Remove imports locais e duplicados de um ficheiro."""
    # First strip multiline from ... import (\n    ...) patterns
    src = re.sub(r'^from \w[^\n]+\(\n(?:.*?\n)*?\)\n?', '', src, flags=re.MULTILINE)
    for pat in REMOVE_PATTERNS:
        src = re.sub(pat, '', src, flags=re.MULTILINE)
    # Remove linhas em branco excessivas (>2 consecutivas)
    src = re.sub(r'\n{3,}', '\n\n', src)
    return src.strip()


def build():
    base = os.path.dirname(os.path.abspath(__file__))
    parts = []

    # Cabeçalho
    parts.append(f"""\
# ════════════════════════════════════════════════════════════════════════════════
# app_bundle.py — ATHELTICA Dashboard (ficheiro único gerado por build.py)
# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# NÃO EDITAR DIRECTAMENTE — edita os módulos e corre python build.py
# ════════════════════════════════════════════════════════════════════════════════
""")

    # Imports globais
    parts.append(GLOBAL_IMPORTS)

    # Injectar cada módulo
    for rel_path in BUILD_ORDER:
        fpath = os.path.join(base, rel_path)
        if not os.path.exists(fpath):
            print(f"⚠️  Ficheiro não encontrado: {rel_path}")
            continue
        with open(fpath) as f:
            content = f.read()
        clean = strip_local_imports(content)
        if not clean.strip():
            continue
        parts.append(f"\n# {'═'*76}\n# MÓDULO: {rel_path}\n# {'═'*76}\n")
        parts.append(clean)
        parts.append("\n")
        print(f"✅ {rel_path}")

    # Injectar o main() do app.py (apenas a função render_sidebar e main)
    app_path = os.path.join(base, "app.py")
    with open(app_path) as f:
        app_src = f.read()
    # Extrair apenas render_sidebar + main + if __name__
    main_match = re.search(
        r'(^def render_sidebar.*)',
        app_src, re.DOTALL | re.MULTILINE)
    if main_match:
        main_src = strip_local_imports(main_match.group(1))
        parts.append(f"\n# {'═'*76}\n# MÓDULO: app.py (sidebar + main)\n# {'═'*76}\n")
        parts.append(main_src)
        parts.append("\n")
        print("✅ app.py (sidebar + main)")

    # Escrever bundle
    bundle = "\n".join(parts)
    out_path = os.path.join(base, "app_bundle.py")
    with open(out_path, "w") as f:
        f.write(bundle)

    # Verificar sintaxe
    try:
        ast.parse(bundle)
        print(f"\n✅ Sintaxe OK")
    except SyntaxError as e:
        print(f"\n❌ SyntaxError: {e}")
        lines = bundle.splitlines()
        s = max(0, e.lineno - 5); end = min(len(lines), e.lineno + 5)
        for i, l in enumerate(lines[s:end], s + 1):
            print(f"  {'>>>' if i == e.lineno else '   '} {i}: {l}")
        return False

    size_kb = os.path.getsize(out_path) / 1024
    n_lines = bundle.count('\n')
    print(f"📦 {out_path} — {n_lines} linhas, {size_kb:.0f} KB")
    print("\n→ Para deploy: copia app_bundle.py para app.py no GitHub")
    return True


if __name__ == "__main__":
    build()
