# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db.py — ATHELTICA
# Storage via abas na sheet existente — mesmos SCOPES e auth do data.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import traceback
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import WorksheetNotFound

_SPREADSHEET_ID = "10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI"

# Exactamente os mesmos SCOPES do config.py
_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

_HEADERS = {
    "cp_results": [
        "saved_at","modalidade","modelo","cp_watts","wp_joules","see_pct",
        "pontos","mmp1","mmp3","mmp5","mmp12","mmp20","pmax"
    ],
    "metab_results": [
        "saved_at","modalidade","vo2max","vlamax","mlss_w","fatmax_w",
        "lt1_w","lt2_w","fat_fatmax","glycogen_g","perfil"
    ],
    "hr_thresholds": [
        "saved_at","modalidade","hrvt1_bpm","hrvt1plus_bpm","hrvtmss_bpm",
        "hrvt2_bpm","aethr_bpm","pbp_w","pvo2max_w"
    ],
}

def _gc():
    """Usa get_gc() do data.py — mesma autenticação."""
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
        if not records:
            return False
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

def save_cp_result(modalidade, modelo, cp_watts, wp_joules, see_pct,
                   combo_pts, mmp_dict=None, pmax=None) -> str:
    try:
        ws = _ws("cp_results")
        if _exists(ws, modalidade, {
            "cp_watts":  round(cp_watts, 1),
            "wp_joules": round(wp_joules, 0),
            "see_pct":   round(see_pct, 2),
        }): return "skipped"
        mmp = mmp_dict or {}
        pontos = ",".join([f"{int(t//60)}min={p:.0f}W" for p,t in combo_pts])
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            modalidade, modelo,
            round(cp_watts,2), round(wp_joules,1), round(see_pct,3),
            pontos,
            mmp.get("MMP1",""), mmp.get("MMP3",""), mmp.get("MMP5",""),
            mmp.get("MMP12",""), mmp.get("MMP20",""),
            round(pmax,1) if pmax else "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_cp: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        return "error"

def save_metab_result(modalidade, vo2max, vlamax, mlss_w, fatmax_w,
                      lt1_w=None, lt2_w=None, fat_fatmax=None,
                      glycogen_g=None, perfil=None) -> str:
    try:
        ws = _ws("metab_results")
        if _exists(ws, modalidade, {
            "vo2max":   round(vo2max, 1),
            "vlamax":   round(vlamax, 3),
            "mlss_w":   round(mlss_w, 1),
            "fatmax_w": round(fatmax_w, 1) if fatmax_w else 0,
        }): return "skipped"
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            modalidade,
            round(vo2max,2), round(vlamax,4),
            round(mlss_w,1), round(fatmax_w,1) if fatmax_w else "",
            round(lt1_w,1) if lt1_w else "",
            round(lt2_w,1) if lt2_w else "",
            round(fat_fatmax,1) if fat_fatmax else "",
            round(glycogen_g,0) if glycogen_g else "",
            perfil or "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_metab: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        return "error"

def save_hr_thresholds(modalidade, hr_zones, pbp_w=None, pvo2max_w=None) -> str:
    def _g(k): return hr_zones.get(k, {}).get("med")
    mlss = _g("HRVTMSS")
    if not mlss: return "skipped"
    try:
        ws = _ws("hr_thresholds")
        if _exists(ws, modalidade, {
            "hrvtmss_bpm": round(mlss, 0),
            "hrvt1_bpm":   round(_g("HRVT1") or 0, 0),
            "hrvt2_bpm":   round(_g("HRVT2") or 0, 0),
        }): return "skipped"
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            modalidade,
            round(_g("HRVT1") or 0, 0),
            round(_g("HRVT1PLUS") or 0, 0),
            round(mlss, 0),
            round(_g("HRVT2") or 0, 0),
            round(_g("AeTHR") or 0, 0),
            round(pbp_w, 1) if pbp_w else "",
            round(pvo2max_w, 1) if pvo2max_w else "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_hr: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        return "error"

def load_cp_history(modalidade=None, n=20) -> pd.DataFrame:
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
    return df.iloc[0].to_dict() if not df.empty else {}

def load_latest_metab(modalidade: str) -> dict:
    df = load_metab_history(modalidade, n=1)
    return df.iloc[0].to_dict() if not df.empty else {}
