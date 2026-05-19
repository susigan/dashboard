# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db.py — ATHELTICA
# Storage via abas na sheet existente — schema completo
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import traceback
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import WorksheetNotFound

_SPREADSHEET_ID = "10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI"
_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# ── Headers completos ─────────────────────────────────────────────────────────
_HEADERS = {

    # Todos os modelos CP (uma linha por modelo, não só o melhor)
    "cp_results": [
        "saved_at", "modalidade",
        # Modelo
        "modelo", "cp_watts", "wp_joules", "see_pct", "pontos",
        # Ranking relativo
        "rank_see", "melhor_modelo",         # 1/2/3 e True/False
        # MMPs com data do season best
        "mmp1_w",  "mmp1_data",
        "mmp3_w",  "mmp3_data",
        "mmp5_w",  "mmp5_data",
        "mmp12_w", "mmp12_data",
        "mmp20_w", "mmp20_data",
        "mmp60_w", "mmp60_data",             # validação externa
        "pmax",
        # Ratios
        "cp_pct_mmp5",                        # CP / MMP5 %
        "cp_pct_mmp20",                       # CP / MMP20 %
    ],

    # Perfil metabólico completo
    "metab_results": [
        "saved_at", "modalidade",
        # VO2max
        "vo2max_hawley_mmp3",
        "vo2max_hawley_mmp5",
        "vo2max_media",
        # VLamax
        "vlamax",
        "perfil_fisiologico",                 # Endurance puro / Speed etc.
        # Limiares de potência
        "fatmax_w", "fatmax_pct_vo2max",
        "mlss_w",   "mlss_pct_vo2max",
        "lt1_w",    "lt2_w",
        # CP vs MLSS
        "cp_watts",
        "cp_vs_mlss_w",                       # CP - MLSS (watts)
        "cp_vs_mlss_pct",                     # (CP-MLSS)/MLSS %
        "mlss_vs_fatmax_pct",                 # (MLSS-FatMax)/FatMax %
        # Substratos
        "fat_at_fatmax_g_h",
        "cho_at_mlss_g_h",
        "fat_at_mlss_g_h",                    # gordura ainda oxidada no MLSS
        # Glicogénio
        "glycogen_total_g",
        "glycogen_liver_g",
        "glycogen_muscle_g",
        "fitness_level",                      # elite/advanced/intermediate/beginner
        # Zonas consolidadas
        "z1_hr_range",                        # ex: "< 141 bpm"
        "z2_hr_range",                        # ex: "141 – 161 bpm"
        "z3_hr_range",                        # ex: "> 161 bpm"
        "z1_w_range",                         # ex: "< 135 W"
        "z2_w_range",                         # ex: "135 – 213 W"
        "z3_w_range",                         # ex: "> 213 W"
        # Inputs usados
        "peso_kg", "altura_cm", "idade",
    ],

    # Limiares HR por modalidade
    "hr_thresholds": [
        "saved_at", "modalidade",
        "hrvt1_bpm",  "hrvt1_iqr",           # mediana e [Q25-Q75]
        "hrvt1plus_bpm", "hrvt1plus_iqr",
        "hrvtmss_bpm", "hrvtmss_iqr",
        "hrvt2_bpm",  "hrvt2_iqr",
        "aethr_bpm",  "aethr_iqr",
        "pbp_w",      "pbp_iqr",
        "pvo2max_w",  "pvo2max_iqr",
        "n_sessions_usadas",                  # N actividades usadas para calcular
    ],
}

# ── Auth ──────────────────────────────────────────────────────────────────────
def _gc():
    try:
        from utils.data import get_gc
        return get_gc()
    except Exception:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=_SCOPES)
        return gspread.authorize(creds)

def _sh():
    return _gc().open_by_key(_SPREADSHEET_ID)

def _ws(name: str):
    sh = _sh()
    try:
        return sh.worksheet(name)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=2000, cols=len(_HEADERS[name]))
        ws.append_row(_HEADERS[name])
        return ws

def _exists(ws, modalidade: str, checks: dict, tol=0.5) -> bool:
    try:
        records = ws.get_all_records()
        if not records: return False
        for row in records:
            if str(row.get("modalidade","")).strip() != str(modalidade).strip():
                continue
            first_key = list(checks.keys())[0]
            if not row.get(first_key) and row.get(first_key) != 0:
                continue
            if all(abs(float(row.get(k) or 0) - float(v or 0)) < tol
                   for k, v in checks.items()):
                return True
        return False
    except Exception:
        return False

def _fmt_iqr(d: dict) -> str:
    """Formata IQR como '[Q25-Q75]'."""
    if not d: return ""
    return f"[{d.get('q25',0):.0f}–{d.get('q75',0):.0f}]"

def _r(v, n=1):
    """round seguro."""
    try: return round(float(v), n) if v is not None else ""
    except: return ""

# ── API pública ───────────────────────────────────────────────────────────────

def save_cp_result(
    modalidade: str,
    resultados_rank: dict,      # {modelo: {cp, wp, see_pct, combo, result}} — todos
    all_mmp_pts: list,          # [(watts, secs), ...] — pontos disponíveis
    mmp_season_bests: dict,     # {"MMP1": {"w": 388, "data": "2025-03-12"}, ...}
    pmax: float = None,
) -> str:
    """
    Guarda TODOS os modelos do Ranking, um por linha.
    O melhor (menor SEE%) é marcado como melhor_modelo=True.
    """
    if not resultados_rank: return "skipped"
    try:
        ws = _ws("cp_results")

        # Verificar se já existe (pelo melhor modelo desta sessão)
        melhor = min(resultados_rank.items(), key=lambda x: x[1].get('see_pct', 999))
        melhor_lbl, melhor_gr = melhor
        melhor_cp = melhor_gr.get('cp', 0)

        if _exists(ws, modalidade, {
            "cp_watts": _r(melhor_cp, 1),
            "see_pct":  _r(melhor_gr.get('see_pct', 0), 2),
        }): return "skipped"

        sb = mmp_season_bests or {}
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Uma linha por modelo
        rows_to_add = []
        rank = 1
        for modelo, gr in sorted(resultados_rank.items(),
                                  key=lambda x: x[1].get('see_pct', 999)):
            cp  = gr.get('cp') or 0
            wp  = gr.get('result', [None,None])[1] if gr.get('result') else None
            see = gr.get('see_pct', 0)
            combo = gr.get('combo', [])
            pontos = ",".join([f"{int(t//60)}min={p:.0f}W" for p,t in combo])

            # MMPs com datas
            def _mmp(k):
                d = sb.get(k, {})
                return _r(d.get('w')), d.get('data','')

            m1w,m1d   = _mmp('MMP1')
            m3w,m3d   = _mmp('MMP3')
            m5w,m5d   = _mmp('MMP5')
            m12w,m12d = _mmp('MMP12')
            m20w,m20d = _mmp('MMP20')
            m60w,m60d = _mmp('MMP60')

            cp_pct_mmp5  = _r(cp / float(m5w)  * 100, 1) if m5w  else ""
            cp_pct_mmp20 = _r(cp / float(m20w) * 100, 1) if m20w else ""

            rows_to_add.append([
                now, modalidade,
                modelo, _r(cp,2), _r(wp,1), _r(see,3), pontos,
                rank, str(rank == 1),
                m1w, m1d, m3w, m3d, m5w, m5d,
                m12w, m12d, m20w, m20d, m60w, m60d,
                _r(pmax,1),
                cp_pct_mmp5, cp_pct_mmp20,
            ])
            rank += 1

        for row in rows_to_add:
            ws.append_row(row)
        return "saved"

    except Exception as e:
        st.error(f"[drive_db] save_cp: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        return "error"


def save_metab_result(
    modalidade: str,
    # VO2max
    vo2max_mmp3: float, vo2max_mmp5: float,
    # VLamax
    vlamax: float, perfil: str,
    # Limiares
    fatmax_w: float, fatmax_pct_vo2max: float,
    mlss_w: float, mlss_pct_vo2max: float,
    lt1_w: float = None, lt2_w: float = None,
    # CP para comparação
    cp_watts: float = None,
    # Substratos
    fat_fatmax_g_h: float = None,
    cho_mlss_g_h: float = None,
    fat_mlss_g_h: float = None,
    # Glicogénio
    glycogen_total: float = None,
    glycogen_liver: float = None,
    glycogen_muscle: float = None,
    fitness_level: str = None,
    # Zonas consolidadas
    z1_hr: str = "", z2_hr: str = "", z3_hr: str = "",
    z1_w: str = "",  z2_w: str = "",  z3_w: str = "",
    # Inputs
    peso_kg: float = None, altura_cm: int = None, idade: int = None,
) -> str:
    try:
        ws = _ws("metab_results")
        vo2max_media = (vo2max_mmp3 + vo2max_mmp5) / 2

        if _exists(ws, modalidade, {
            "vo2max_media": _r(vo2max_media, 1),
            "vlamax":       _r(vlamax, 3),
            "mlss_w":       _r(mlss_w, 1),
        }): return "skipped"

        cp_vs_mlss_w   = _r(cp_watts - mlss_w, 1) if cp_watts and mlss_w else ""
        cp_vs_mlss_pct = _r((cp_watts - mlss_w) / mlss_w * 100, 1) if cp_watts and mlss_w else ""
        mlss_fat_pct   = _r((mlss_w - fatmax_w) / fatmax_w * 100, 1) if mlss_w and fatmax_w else ""

        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"), modalidade,
            _r(vo2max_mmp3,2), _r(vo2max_mmp5,2), _r(vo2max_media,2),
            _r(vlamax,4), perfil or "",
            _r(fatmax_w,1), _r(fatmax_pct_vo2max,1),
            _r(mlss_w,1),   _r(mlss_pct_vo2max,1),
            _r(lt1_w,1), _r(lt2_w,1),
            _r(cp_watts,1), cp_vs_mlss_w, cp_vs_mlss_pct, mlss_fat_pct,
            _r(fat_fatmax_g_h,1), _r(cho_mlss_g_h,1), _r(fat_mlss_g_h,1),
            _r(glycogen_total,0), _r(glycogen_liver,0), _r(glycogen_muscle,0),
            fitness_level or "",
            z1_hr, z2_hr, z3_hr, z1_w, z2_w, z3_w,
            _r(peso_kg,1), altura_cm or "", idade or "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_metab: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        return "error"


def save_hr_thresholds(
    modalidade: str,
    hr_zones: dict,
    pbp_w: float = None,
    pvo2max_w: float = None,
    n_sessions: int = None,
) -> str:
    def _g(k):  return hr_zones.get(k, {}).get("med")
    def _iq(k): return _fmt_iqr(hr_zones.get(k, {}))

    mlss = _g("HRVTMSS")
    if not mlss: return "skipped"
    try:
        ws = _ws("hr_thresholds")
        if _exists(ws, modalidade, {
            "hrvtmss_bpm": round(mlss, 0),
            "hrvt1_bpm":   round(_g("HRVT1") or 0, 0),
            "hrvt2_bpm":   round(_g("HRVT2") or 0, 0),
        }): return "skipped"

        pbp_iqr   = _fmt_iqr(hr_zones.get('PBP', {}))    if 'PBP'    in hr_zones else ""
        pvo2_iqr  = _fmt_iqr(hr_zones.get('Pvo2max',{})) if 'Pvo2max' in hr_zones else ""

        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"), modalidade,
            _r(_g("HRVT1"),0),     _iq("HRVT1"),
            _r(_g("HRVT1PLUS"),0), _iq("HRVT1PLUS"),
            _r(mlss,0),            _iq("HRVTMSS"),
            _r(_g("HRVT2"),0),     _iq("HRVT2"),
            _r(_g("AeTHR"),0),     _iq("AeTHR"),
            _r(pbp_w,1),           pbp_iqr,
            _r(pvo2max_w,1),       pvo2_iqr,
            n_sessions or "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_hr: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        return "error"


# ── Leitura ───────────────────────────────────────────────────────────────────
def load_cp_history(modalidade=None, n=50) -> pd.DataFrame:
    try:
        ws = _ws("cp_results")
        df = pd.DataFrame(ws.get_all_records())
        if df.empty: return df
        if modalidade: df = df[df["modalidade"] == modalidade]
        return df.sort_values("saved_at", ascending=False).head(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def load_metab_history(modalidade=None, n=20) -> pd.DataFrame:
    try:
        ws = _ws("metab_results")
        df = pd.DataFrame(ws.get_all_records())
        if df.empty: return df
        if modalidade: df = df[df["modalidade"] == modalidade]
        return df.sort_values("saved_at", ascending=False).head(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def load_latest_cp(modalidade: str) -> dict:
    df = load_cp_history(modalidade, n=1)
    return df[df["melhor_modelo"] == "True"].iloc[0].to_dict() \
           if not df.empty and "melhor_modelo" in df.columns else \
           (df.iloc[0].to_dict() if not df.empty else {})

def load_latest_metab(modalidade: str) -> dict:
    df = load_metab_history(modalidade, n=1)
    return df.iloc[0].to_dict() if not df.empty else {}


# ══════════════════════════════════════════════════════════════════════════════
# SQLite no Google Drive — funciona se a pasta estiver partilhada como Editor
# ══════════════════════════════════════════════════════════════════════════════

import os, io, sqlite3
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

_DB_NAME     = "atheltica_cpmodel.db"
_LOCAL_DB    = f"/tmp/{_DB_NAME}"
_FOLDER_ID   = "11oXQPkFrG6ZBCsvjDqb8RAiE_VfwBSfV"
_DRIVE_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

def _drive_svc():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)

def _find_db_id(svc) -> str | None:
    try:
        r = svc.files().list(
            q=f"name='{_DB_NAME}' and '{_FOLDER_ID}' in parents and trashed=false",
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = r.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None

def _init_sqlite():
    conn = sqlite3.connect(_LOCAL_DB)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS cp_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, modalidade TEXT, modelo TEXT,
        cp_watts REAL, wp_joules REAL, see_pct REAL,
        pontos TEXT, rank_see INTEGER, melhor_modelo INTEGER,
        mmp1_w REAL, mmp1_data TEXT, mmp3_w REAL, mmp3_data TEXT,
        mmp5_w REAL, mmp5_data TEXT, mmp12_w REAL, mmp12_data TEXT,
        mmp20_w REAL, mmp20_data TEXT, mmp60_w REAL, mmp60_data TEXT,
        pmax REAL, cp_pct_mmp5 REAL, cp_pct_mmp20 REAL);
    CREATE TABLE IF NOT EXISTS metab_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, modalidade TEXT,
        vo2max_mmp3 REAL, vo2max_mmp5 REAL, vo2max_media REAL,
        vlamax REAL, perfil TEXT,
        fatmax_w REAL, fatmax_pct_vo2 REAL,
        mlss_w REAL, mlss_pct_vo2 REAL,
        lt1_w REAL, lt2_w REAL, cp_watts REAL,
        cp_vs_mlss_w REAL, cp_vs_mlss_pct REAL, mlss_vs_fatmax_pct REAL,
        fat_fatmax_g_h REAL, cho_mlss_g_h REAL, fat_mlss_g_h REAL,
        glycogen_total REAL, glycogen_liver REAL, glycogen_muscle REAL,
        fitness_level TEXT,
        z1_hr TEXT, z2_hr TEXT, z3_hr TEXT,
        z1_w TEXT, z2_w TEXT, z3_w TEXT,
        peso_kg REAL, altura_cm INTEGER, idade INTEGER);
    CREATE TABLE IF NOT EXISTS hr_thresholds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, modalidade TEXT,
        hrvt1_bpm REAL, hrvt1_iqr TEXT,
        hrvt1plus_bpm REAL, hrvt1plus_iqr TEXT,
        hrvtmss_bpm REAL, hrvtmss_iqr TEXT,
        hrvt2_bpm REAL, hrvt2_iqr TEXT,
        aethr_bpm REAL, aethr_iqr TEXT,
        pbp_w REAL, pbp_iqr TEXT,
        pvo2max_w REAL, pvo2max_iqr TEXT,
        n_sessions INTEGER);
    """)
    conn.commit()
    conn.close()

def _upload_db() -> bool:
    """Tenta upload do .db para o Drive. Retorna True se OK."""
    if not os.path.exists(_LOCAL_DB): return False
    try:
        svc = _drive_svc()
        file_id = _find_db_id(svc)
        media = MediaFileUpload(_LOCAL_DB, mimetype="application/x-sqlite3",
                                resumable=False)
        if file_id:
            svc.files().update(
                fileId=file_id, media_body=media,
                supportsAllDrives=True,
            ).execute()
        else:
            svc.files().create(
                body={"name": _DB_NAME, "parents": [_FOLDER_ID]},
                media_body=media,
                supportsAllDrives=True,
                fields="id",
            ).execute()
        return True
    except HttpError as e:
        # Não mostra erro — é opcional, o Sheets já guardou
        return False
    except Exception:
        return False

def _download_db() -> bool:
    """Download do .db do Drive para /tmp/. Cria se não existir."""
    try:
        svc = _drive_svc()
        file_id = _find_db_id(svc)
        if file_id:
            req = svc.files().get_media(fileId=file_id,
                                        supportsAllDrives=True)
            with open(_LOCAL_DB, "wb") as f:
                dl = MediaIoBaseDownload(f, req)
                done = False
                while not done: _, done = dl.next_chunk()
    except Exception:
        pass
    # Sempre inicializar (cria tabelas se não existirem)
    _init_sqlite()
    return True

def get_sqlite_conn() -> sqlite3.Connection:
    """Retorna conexão SQLite. Faz download se necessário."""
    if not os.path.exists(_LOCAL_DB):
        _download_db()
    _init_sqlite()
    return sqlite3.connect(_LOCAL_DB)
