# ══════════════════════════════════════════════════════════════════════════════
# utils/plano_db.py — ATHELTICA
# Gestão do plano semanal de zonas (Z1/Z2/Z3) por modalidade
# Persiste em SQLite no Google Drive (atheltica_plano.db)
# ══════════════════════════════════════════════════════════════════════════════

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_DB_NAME   = "atheltica_plano.db"
_LOCAL_DB  = f"/tmp/{_DB_NAME}"
_FOLDER_ID = "11oXQPkFrG6ZBCsvjDqb8RAiE_VfwBSfV"

# ── Drive helpers ─────────────────────────────────────────────────────────────

def _drive_svc():
    import streamlit as st
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    _SCOPES = ["https://spreadsheets.google.com/feeds",
               "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_SCOPES)
    return build("drive", "v3", credentials=creds)

def _find_file_id(svc) -> str | None:
    try:
        r = svc.files().list(
            q=f"name='{_DB_NAME}' and '{_FOLDER_ID}' in parents and trashed=false",
            fields="files(id)", supportsAllDrives=True,
            includeItemsFromAllDrives=True).execute()
        files = r.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None

def _download_db() -> bool:
    """Download do .db do Drive para /tmp/. Cria se não existir."""
    try:
        from googleapiclient.http import MediaIoBaseDownload
        svc = _drive_svc()
        file_id = _find_file_id(svc)
        if file_id:
            req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
            with open(_LOCAL_DB, "wb") as f:
                dl = MediaIoBaseDownload(f, req)
                done = False
                while not done: _, done = dl.next_chunk()
    except Exception:
        pass
    _init_db()
    return True

def _upload_db() -> bool:
    """Upload do .db local para o Drive."""
    if not os.path.exists(_LOCAL_DB):
        return False
    try:
        from googleapiclient.http import MediaFileUpload
        svc = _drive_svc()
        file_id = _find_file_id(svc)
        media = MediaFileUpload(_LOCAL_DB, mimetype="application/x-sqlite3", resumable=False)
        if file_id:
            svc.files().update(fileId=file_id, media_body=media,
                               supportsAllDrives=True).execute()
        else:
            svc.files().create(
                body={"name": _DB_NAME, "parents": [_FOLDER_ID]},
                media_body=media, supportsAllDrives=True, fields="id").execute()
        return True
    except Exception:
        return False

def _init_db():
    conn = sqlite3.connect(_LOCAL_DB)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS planos (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        modalidade      TEXT NOT NULL,
        criado_em       TEXT NOT NULL,
        prazo_semanas   INTEGER NOT NULL,
        eftp_alvo_delta INTEGER NOT NULL,
        eftp_actual     REAL,
        semana_inicio   TEXT NOT NULL,
        kj_z3_inicial   REAL,
        kj_z2_inicial   REAL,
        kj_z1_inicial   REAL,
        ativo           INTEGER DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS semanas_plano (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        plano_id        INTEGER NOT NULL,
        semana_num      INTEGER NOT NULL,
        semana_iso      TEXT NOT NULL,
        modalidade      TEXT NOT NULL,
        kj_z3_alvo      REAL,
        kj_z2_alvo      REAL,
        kj_z1_alvo      REAL,
        kj_z3_feito     REAL DEFAULT 0,
        kj_z2_feito     REAL DEFAULT 0,
        kj_z1_feito     REAL DEFAULT 0,
        eftp_real       REAL,
        updated_at      TEXT,
        FOREIGN KEY(plano_id) REFERENCES planos(id)
    );
    CREATE INDEX IF NOT EXISTS idx_semanas_plano_id ON semanas_plano(plano_id);
    CREATE INDEX IF NOT EXISTS idx_semanas_iso ON semanas_plano(semana_iso, modalidade);
    """)
    conn.commit()
    conn.close()

def _get_conn() -> sqlite3.Connection:
    if not os.path.exists(_LOCAL_DB):
        _download_db()
    _init_db()
    return sqlite3.connect(_LOCAL_DB)

# ── Segunda-feira da semana actual ────────────────────────────────────────────

def _segunda_da_semana(dt=None) -> str:
    """Retorna data ISO (YYYY-MM-DD) da segunda-feira da semana de dt."""
    dt = dt or datetime.now().date()
    if hasattr(dt, 'date'):
        dt = dt.date()
    return str(dt - timedelta(days=dt.weekday()))

# ── API pública ───────────────────────────────────────────────────────────────

def get_plano_ativo(modalidade: str) -> dict | None:
    """Retorna o plano activo para uma modalidade, ou None."""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM planos WHERE modalidade=? AND ativo=1 ORDER BY id DESC LIMIT 1",
            (modalidade,)).fetchone()
        conn.close()
        if not row:
            return None
        cols = ['id','modalidade','criado_em','prazo_semanas','eftp_alvo_delta',
                'eftp_actual','semana_inicio','kj_z3_inicial','kj_z2_inicial',
                'kj_z1_inicial','ativo']
        return dict(zip(cols, row))
    except Exception:
        return None

def get_semana_atual(modalidade: str) -> dict:
    """
    Retorna info sobre a semana actual do plano activo:
    - semana_num: qual semana (1..N) ou None se não há plano
    - semana_iso: data ISO da segunda-feira actual
    - total_semanas: prazo total
    - concluido: True se passou todas as semanas
    - dados: row da tabela semanas_plano
    """
    plano = get_plano_ativo(modalidade)
    if not plano:
        return {'semana_num': None, 'plano': None}

    hoje_seg = _segunda_da_semana()
    inicio   = plano['semana_inicio']
    prazo    = plano['prazo_semanas']

    # Calcular número da semana actual
    from datetime import date
    d_inicio = date.fromisoformat(inicio)
    d_hoje   = date.fromisoformat(hoje_seg)
    delta_sem = (d_hoje - d_inicio).days // 7 + 1

    if delta_sem < 1:
        delta_sem = 1  # ainda antes de começar

    concluido = delta_sem > prazo

    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM semanas_plano WHERE plano_id=? AND semana_num=?",
            (plano['id'], min(delta_sem, prazo))).fetchone()
        conn.close()
        if row:
            cols = ['id','plano_id','semana_num','semana_iso','modalidade',
                    'kj_z3_alvo','kj_z2_alvo','kj_z1_alvo',
                    'kj_z3_feito','kj_z2_feito','kj_z1_feito',
                    'eftp_real','updated_at']
            dados = dict(zip(cols, row))
        else:
            dados = None
    except Exception:
        dados = None

    return {
        'semana_num':    delta_sem,
        'total_semanas': prazo,
        'concluido':     concluido,
        'semana_iso':    hoje_seg,
        'plano':         plano,
        'dados':         dados,
    }

def criar_plano(modalidade: str, prazo_semanas: int, eftp_alvo_delta: int,
                eftp_actual: float, kj_z3_inicial: float, kj_z2_inicial: float,
                kj_z1_inicial: float, alpha_z3: float, alpha_z2: float,
                alpha_z1: float, intercept: float, cz3_now: float,
                cz2_now: float, cz1_now: float, span: int = 28) -> int:
    """
    Cria um novo plano e gera todas as semanas da rampa.
    Desactiva planos anteriores desta modalidade.
    Retorna o id do novo plano.
    """
    hoje_seg   = _segunda_da_semana()
    criado_em  = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Calcular CTLγ_Z3 alvo via modelo inverso
    eftp_tgt = eftp_actual + eftp_alvo_delta
    if abs(alpha_z3) > 0.01:
        cz3_alvo = (eftp_tgt - alpha_z2*cz2_now - alpha_z1*cz1_now - intercept) / alpha_z3
        cz3_alvo = max(cz3_alvo, cz3_now)
    else:
        cz3_alvo = cz3_now * 1.10

    # Inversão correcta do CTLγ → kJ/semana
    # CTLγ é uma EWM: estado estacionário → kJ_diário = CTLγ × α_EWM
    # α_EWM = 2/(span+1) para EWM span equivalente
    _alpha_ewm = 2.0 / (span + 1)
    _kj_diario_alvo = float(cz3_alvo) * _alpha_ewm
    kj_z3_alvo_final = _kj_diario_alvo * 7  # kJ/semana
    # Sanity check: alvo não pode ser mais que 3× o actual
    kj_z3_alvo_final = min(kj_z3_alvo_final, kj_z3_inicial * 3.0 if kj_z3_inicial > 5 else kj_z3_alvo_final)

    try:
        conn = _get_conn()
        # Desactivar planos anteriores
        conn.execute("UPDATE planos SET ativo=0 WHERE modalidade=?", (modalidade,))
        # Criar novo plano
        cur = conn.execute(
            """INSERT INTO planos
               (modalidade,criado_em,prazo_semanas,eftp_alvo_delta,eftp_actual,
                semana_inicio,kj_z3_inicial,kj_z2_inicial,kj_z1_inicial,ativo)
               VALUES (?,?,?,?,?,?,?,?,?,1)""",
            (modalidade, criado_em, prazo_semanas, eftp_alvo_delta, eftp_actual,
             hoje_seg, kj_z3_inicial, kj_z2_inicial, kj_z1_inicial))
        plano_id = cur.lastrowid

        # Gerar semanas da rampa linear
        from datetime import date, timedelta
        d_inicio = date.fromisoformat(hoje_seg)
        for sem in range(1, prazo_semanas + 1):
            frac = sem / prazo_semanas
            kj_z3_sem = kj_z3_inicial + (kj_z3_alvo_final - kj_z3_inicial) * frac
            kj_z2_sem = kj_z2_inicial * (1 + 0.05 * frac)
            kj_z1_sem = kj_z1_inicial * (1 + 0.05 * frac)
            sem_iso   = str(d_inicio + timedelta(weeks=sem-1))
            conn.execute(
                """INSERT INTO semanas_plano
                   (plano_id,semana_num,semana_iso,modalidade,
                    kj_z3_alvo,kj_z2_alvo,kj_z1_alvo)
                   VALUES (?,?,?,?,?,?,?)""",
                (plano_id, sem, sem_iso, modalidade,
                 round(kj_z3_sem, 1), round(kj_z2_sem, 1), round(kj_z1_sem, 1)))
        conn.commit()
        conn.close()
        _upload_db()
        return plano_id
    except Exception as e:
        return -1

def actualizar_feito(modalidade: str, semana_iso: str,
                     kj_z3_feito: float, kj_z2_feito: float,
                     kj_z1_feito: float, eftp_real: float = None) -> bool:
    """Actualiza o kJ feito numa semana do plano activo."""
    plano = get_plano_ativo(modalidade)
    if not plano:
        return False
    try:
        conn = _get_conn()
        conn.execute(
            """UPDATE semanas_plano
               SET kj_z3_feito=?, kj_z2_feito=?, kj_z1_feito=?,
                   eftp_real=?, updated_at=?
               WHERE plano_id=? AND semana_iso=?""",
            (round(kj_z3_feito,1), round(kj_z2_feito,1), round(kj_z1_feito,1),
             eftp_real, datetime.now().strftime("%Y-%m-%d %H:%M"),
             plano['id'], semana_iso))
        conn.commit()
        conn.close()
        _upload_db()
        return True
    except Exception:
        return False

def get_historico_plano(modalidade: str) -> pd.DataFrame:
    """Retorna todas as semanas do plano activo como DataFrame."""
    plano = get_plano_ativo(modalidade)
    if not plano:
        return pd.DataFrame()
    try:
        conn = _get_conn()
        df = pd.read_sql_query(
            "SELECT * FROM semanas_plano WHERE plano_id=? ORDER BY semana_num",
            conn, params=(plano['id'],))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def plano_mudou(modalidade: str, prazo_semanas: int, eftp_alvo_delta: int) -> bool:
    """Verifica se os parâmetros mudaram em relação ao plano activo."""
    plano = get_plano_ativo(modalidade)
    if not plano:
        return True  # não há plano → criar
    return (plano['prazo_semanas'] != prazo_semanas or
            plano['eftp_alvo_delta'] != eftp_alvo_delta)
