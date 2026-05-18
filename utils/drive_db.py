# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db.py — ATHELTICA
# Persistência via SQLite no Google Drive pessoal
# Requer: google-api-python-client, google-auth
# Service account partilhada como Editor na pasta do Drive
# ══════════════════════════════════════════════════════════════════════════════

import os, io, sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

_DB_NAME   = "atheltica_cpmodel.db"
_LOCAL     = f"/tmp/{_DB_NAME}"
_SCOPES    = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ── Auth ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def _svc():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=_SCOPES)
    return build("drive", "v3", credentials=creds)

def _folder() -> str:
    return st.secrets.get("drive", {}).get("folder_id", "")

# ── Drive helpers ─────────────────────────────────────────────────────────────
def _find_id(svc, folder_id: str) -> str | None:
    try:
        r = svc.files().list(
            q=f"name='{_DB_NAME}' and '{folder_id}' in parents and trashed=false",
            fields="files(id)",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files = r.get("files", [])
        return files[0]["id"] if files else None
    except HttpError:
        return None

def _download() -> bool:
    svc = _svc(); fid = _folder()
    if not fid: return False
    file_id = _find_id(svc, fid)
    if file_id:
        try:
            req = svc.files().get_media(fileId=file_id,
                                        supportsAllDrives=True)
            with open(_LOCAL, "wb") as f:
                dl = MediaIoBaseDownload(f, req)
                done = False
                while not done: _, done = dl.next_chunk()
            return True
        except HttpError as e:
            st.error(f"[drive_db] Download falhou: {e}")
            return False
    else:
        # Primeira vez — criar DB local vazio
        _init_db()
        return True

def _upload() -> bool:
    if not os.path.exists(_LOCAL): return False
    svc = _svc(); fid = _folder()
    if not fid: return False
    file_id = _find_id(svc, fid)
    media = MediaFileUpload(_LOCAL, mimetype="application/x-sqlite3",
                            resumable=False)
    try:
        if file_id:
            svc.files().update(
                fileId=file_id, media_body=media,
                supportsAllDrives=True,
            ).execute()
        else:
            svc.files().create(
                body={"name": _DB_NAME, "parents": [fid]},
                media_body=media,
                supportsAllDrives=True,
                fields="id",
            ).execute()
        return True
    except HttpError as e:
        st.error(f"[drive_db] Upload falhou: {e}")
        return False

# ── Schema ────────────────────────────────────────────────────────────────────
def _init_db():
    conn = sqlite3.connect(_LOCAL)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS cp_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, modalidade TEXT, modelo TEXT,
        cp_watts REAL, wp_joules REAL, see_pct REAL, pontos TEXT,
        mmp1 REAL, mmp3 REAL, mmp5 REAL, mmp12 REAL, mmp20 REAL, pmax REAL);
    CREATE TABLE IF NOT EXISTS metab_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, modalidade TEXT,
        vo2max REAL, vlamax REAL, mlss_w REAL, fatmax_w REAL,
        lt1_w REAL, lt2_w REAL, fat_fatmax REAL, glycogen_g REAL, perfil TEXT);
    CREATE TABLE IF NOT EXISTS hr_thresholds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, modalidade TEXT,
        hrvt1_bpm REAL, hrvt1plus_bpm REAL, hrvtmss_bpm REAL,
        hrvt2_bpm REAL, aethr_bpm REAL, pbp_w REAL, pvo2max_w REAL);
    """)
    conn.commit(); conn.close()

def _conn() -> sqlite3.Connection | None:
    if not os.path.exists(_LOCAL):
        if not _download(): return None
    _init_db()
    return sqlite3.connect(_LOCAL)

# ── Deduplicação ──────────────────────────────────────────────────────────────
def _exists(table: str, modalidade: str, checks: dict, tol=0.5) -> bool:
    c = _conn()
    if not c: return False
    where = " AND ".join(
        f"ABS(COALESCE({k},0)-COALESCE(?,0))<{tol}" for k in checks)
    try:
        n = c.execute(
            f"SELECT COUNT(*) FROM {table} WHERE modalidade=? AND {where}",
            (modalidade,) + tuple(checks.values())
        ).fetchone()[0]
        c.close(); return n > 0
    except Exception:
        c.close(); return False

# ── API pública ───────────────────────────────────────────────────────────────
def save_cp_result(modalidade, modelo, cp_watts, wp_joules, see_pct,
                   combo_pts, mmp_dict=None, pmax=None) -> str:
    """Retorna 'saved' | 'skipped' | 'error'"""
    checks = {"cp_watts": round(cp_watts,1),
              "wp_joules": round(wp_joules,0),
              "see_pct": round(see_pct,2)}
    if _exists("cp_results", modalidade, checks):
        return "skipped"
    c = _conn()
    if not c: return "error"
    mmp = mmp_dict or {}
    pontos = ",".join([f"{int(t//60)}min={p:.0f}W" for p,t in combo_pts])
    try:
        c.execute("""INSERT INTO cp_results
            (saved_at,modalidade,modelo,cp_watts,wp_joules,see_pct,pontos,
             mmp1,mmp3,mmp5,mmp12,mmp20,pmax) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), modalidade, modelo,
             round(cp_watts,2), round(wp_joules,1), round(see_pct,3), pontos,
             mmp.get("MMP1"), mmp.get("MMP3"), mmp.get("MMP5"),
             mmp.get("MMP12"), mmp.get("MMP20"),
             round(pmax,1) if pmax else None))
        c.commit(); c.close()
        return "saved" if _upload() else "error"
    except Exception as e:
        c.close(); st.error(f"[drive_db] {e}"); return "error"

def save_metab_result(modalidade, vo2max, vlamax, mlss_w, fatmax_w,
                      lt1_w=None, lt2_w=None, fat_fatmax=None,
                      glycogen_g=None, perfil=None) -> str:
    checks = {"vo2max": round(vo2max,1), "vlamax": round(vlamax,3),
              "mlss_w": round(mlss_w,1),
              "fatmax_w": round(fatmax_w,1) if fatmax_w else 0}
    if _exists("metab_results", modalidade, checks):
        return "skipped"
    c = _conn()
    if not c: return "error"
    try:
        c.execute("""INSERT INTO metab_results
            (saved_at,modalidade,vo2max,vlamax,mlss_w,fatmax_w,
             lt1_w,lt2_w,fat_fatmax,glycogen_g,perfil) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), modalidade,
             round(vo2max,2), round(vlamax,4), round(mlss_w,1),
             round(fatmax_w,1) if fatmax_w else None,
             round(lt1_w,1) if lt1_w else None,
             round(lt2_w,1) if lt2_w else None,
             round(fat_fatmax,1) if fat_fatmax else None,
             round(glycogen_g,0) if glycogen_g else None,
             perfil))
        c.commit(); c.close()
        return "saved" if _upload() else "error"
    except Exception as e:
        c.close(); st.error(f"[drive_db] {e}"); return "error"

def save_hr_thresholds(modalidade, hr_zones, pbp_w=None, pvo2max_w=None) -> str:
    def _g(k): return hr_zones.get(k, {}).get("med")
    mlss = _g("HRVTMSS")
    if not mlss: return "skipped"
    checks = {"hrvtmss_bpm": round(mlss,0),
              "hrvt1_bpm": round(_g("HRVT1") or 0, 0),
              "hrvt2_bpm": round(_g("HRVT2") or 0, 0)}
    if _exists("hr_thresholds", modalidade, checks):
        return "skipped"
    c = _conn()
    if not c: return "error"
    try:
        c.execute("""INSERT INTO hr_thresholds
            (saved_at,modalidade,hrvt1_bpm,hrvt1plus_bpm,hrvtmss_bpm,
             hrvt2_bpm,aethr_bpm,pbp_w,pvo2max_w) VALUES (?,?,?,?,?,?,?,?,?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), modalidade,
             _g("HRVT1"), _g("HRVT1PLUS"), mlss,
             _g("HRVT2"), _g("AeTHR"), pbp_w, pvo2max_w))
        c.commit(); c.close()
        return "saved" if _upload() else "error"
    except Exception as e:
        c.close(); st.error(f"[drive_db] {e}"); return "error"

def load_cp_history(modalidade=None, n=20) -> pd.DataFrame:
    c = _conn()
    if not c: return pd.DataFrame()
    q = "SELECT * FROM cp_results"
    p = []
    if modalidade: q += " WHERE modalidade=?"; p.append(modalidade)
    q += " ORDER BY saved_at DESC LIMIT ?"
    p.append(n)
    try:
        df = pd.read_sql_query(q, c, params=p)
        c.close(); return df
    except Exception:
        c.close(); return pd.DataFrame()

def load_metab_history(modalidade=None, n=20) -> pd.DataFrame:
    c = _conn()
    if not c: return pd.DataFrame()
    q = "SELECT * FROM metab_results"
    p = []
    if modalidade: q += " WHERE modalidade=?"; p.append(modalidade)
    q += " ORDER BY saved_at DESC LIMIT ?"
    p.append(n)
    try:
        df = pd.read_sql_query(q, c, params=p)
        c.close(); return df
    except Exception:
        c.close(); return pd.DataFrame()

def load_latest_cp(modalidade: str) -> dict:
    df = load_cp_history(modalidade, n=1)
    return df.iloc[0].to_dict() if not df.empty else {}

def load_latest_metab(modalidade: str) -> dict:
    df = load_metab_history(modalidade, n=1)
    return df.iloc[0].to_dict() if not df.empty else {}
