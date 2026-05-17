# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db.py — ATHELTICA
# Persistência via Google Sheets (evita problema de quota da Service Account)
# Usa a mesma autenticação do Sheets já configurada
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_SHEET_NAME = "atheltica_db"   # nome da Google Sheet a criar/usar

# ── Autenticação ──────────────────────────────────────────────────────────────
@st.cache_resource
def _get_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=_SCOPES
    )
    return gspread.authorize(creds)


def _get_spreadsheet():
    """Abre ou cria a spreadsheet de DB."""
    gc = _get_client()
    try:
        sh = gc.open(_SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(_SHEET_NAME)
        # Partilhar com o utilizador para que consiga ver no Drive
        try:
            folder_id = st.secrets.get("drive", {}).get("folder_id")
            if folder_id:
                # Mover para a pasta do utilizador
                from googleapiclient.discovery import build
                creds2 = Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes=_SCOPES
                )
                drive = build("drive", "v3", credentials=creds2)
                drive.files().update(
                    fileId=sh.id,
                    addParents=folder_id,
                    fields="id,parents",
                    supportsAllDrives=True,
                ).execute()
        except Exception:
            pass
    return sh


def _get_worksheet(sh, name: str, headers: list):
    """Abre ou cria uma worksheet com headers."""
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=1000, cols=len(headers))
        ws.append_row(headers)
    return ws


# ── Headers das tabelas ───────────────────────────────────────────────────────
_CP_HEADERS = [
    "saved_at","modalidade","modelo","cp_watts","wp_joules","see_pct",
    "pontos","mmp1","mmp3","mmp5","mmp12","mmp20","pmax"
]
_METAB_HEADERS = [
    "saved_at","modalidade","vo2max","vlamax","mlss_w","fatmax_w",
    "lt1_w","lt2_w","fat_fatmax","glycogen_g","perfil"
]
_HR_HEADERS = [
    "saved_at","modalidade","hrvt1_bpm","hrvt1plus_bpm","hrvtmss_bpm",
    "hrvt2_bpm","aethr_bpm","pbp_w","pvo2max_w"
]


# ── Deduplicação ──────────────────────────────────────────────────────────────
def _already_exists(ws, checks: dict, tol=0.5) -> bool:
    """Verifica se já existe linha com valores iguais (tolerância tol)."""
    try:
        records = ws.get_all_records()
        for row in records:
            match = all(
                abs(float(row.get(k, 0) or 0) - float(v or 0)) < tol
                for k, v in checks.items()
            )
            if match:
                return True
        return False
    except Exception:
        return False


def _load_df(ws) -> pd.DataFrame:
    """Carrega todos os registos como DataFrame."""
    try:
        records = ws.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── API pública ───────────────────────────────────────────────────────────────
def save_cp_result(modalidade, modelo, cp_watts, wp_joules, see_pct,
                   combo_pts, mmp_dict=None, pmax=None) -> str:
    """
    Guarda resultado de CP.
    Retorna: 'saved' | 'skipped' | 'error'
    """
    try:
        sh = _get_spreadsheet()
        ws = _get_worksheet(sh, "cp_results", _CP_HEADERS)
        checks = {
            "cp_watts":  round(cp_watts, 1),
            "wp_joules": round(wp_joules, 0),
            "see_pct":   round(see_pct, 2),
        }
        if _already_exists(ws, checks):
            return "skipped"
        mmp = mmp_dict or {}
        pontos = ",".join([f"{int(t//60)}min={p:.0f}W" for p,t in combo_pts])
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            modalidade, modelo,
            round(cp_watts, 2), round(wp_joules, 1), round(see_pct, 3),
            pontos,
            mmp.get("MMP1",""), mmp.get("MMP3",""), mmp.get("MMP5",""),
            mmp.get("MMP12",""), mmp.get("MMP20",""),
            round(pmax, 1) if pmax else "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_cp: {e}")
        return "error"


def save_metab_result(modalidade, vo2max, vlamax, mlss_w, fatmax_w,
                      lt1_w=None, lt2_w=None, fat_fatmax=None,
                      glycogen_g=None, perfil=None) -> str:
    try:
        sh = _get_spreadsheet()
        ws = _get_worksheet(sh, "metab_results", _METAB_HEADERS)
        checks = {
            "vo2max":   round(vo2max, 1),
            "vlamax":   round(vlamax, 3),
            "mlss_w":   round(mlss_w, 1),
            "fatmax_w": round(fatmax_w, 1) if fatmax_w else 0,
        }
        if _already_exists(ws, checks):
            return "skipped"
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            modalidade,
            round(vo2max, 2), round(vlamax, 4),
            round(mlss_w, 1), round(fatmax_w, 1) if fatmax_w else "",
            round(lt1_w, 1) if lt1_w else "",
            round(lt2_w, 1) if lt2_w else "",
            round(fat_fatmax, 1) if fat_fatmax else "",
            round(glycogen_g, 0) if glycogen_g else "",
            perfil or "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_metab: {e}")
        return "error"


def save_hr_thresholds(modalidade, hr_zones, pbp_w=None, pvo2max_w=None) -> str:
    def _g(k): return hr_zones.get(k, {}).get("med")
    try:
        sh = _get_spreadsheet()
        ws = _get_worksheet(sh, "hr_thresholds", _HR_HEADERS)
        mlss_bpm = _g("HRVTMSS")
        if not mlss_bpm:
            return "skipped"
        checks = {
            "hrvtmss_bpm": round(mlss_bpm, 0),
            "hrvt1_bpm":   round(_g("HRVT1") or 0, 0),
            "hrvt2_bpm":   round(_g("HRVT2") or 0, 0),
        }
        if _already_exists(ws, checks):
            return "skipped"
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            modalidade,
            round(_g("HRVT1") or 0, 0),
            round(_g("HRVT1PLUS") or 0, 0),
            round(mlss_bpm, 0),
            round(_g("HRVT2") or 0, 0),
            round(_g("AeTHR") or 0, 0),
            round(pbp_w, 1) if pbp_w else "",
            round(pvo2max_w, 1) if pvo2max_w else "",
        ])
        return "saved"
    except Exception as e:
        st.error(f"[drive_db] save_hr: {e}")
        return "error"


def load_cp_history(modalidade=None, n=20) -> pd.DataFrame:
    """Lê histórico CP. Retorna DataFrame ordenado do mais recente."""
    try:
        sh = _get_spreadsheet()
        ws = _get_worksheet(sh, "cp_results", _CP_HEADERS)
        df = _load_df(ws)
        if df.empty: return df
        if modalidade:
            df = df[df["modalidade"] == modalidade]
        df = df.sort_values("saved_at", ascending=False).head(n)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_metab_history(modalidade=None, n=20) -> pd.DataFrame:
    try:
        sh = _get_spreadsheet()
        ws = _get_worksheet(sh, "metab_results", _METAB_HEADERS)
        df = _load_df(ws)
        if df.empty: return df
        if modalidade:
            df = df[df["modalidade"] == modalidade]
        return df.sort_values("saved_at", ascending=False).head(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_latest_cp(modalidade: str) -> dict:
    """Retorna o CP mais recente de uma modalidade como dict."""
    df = load_cp_history(modalidade, n=1)
    return df.iloc[0].to_dict() if not df.empty else {}


def load_latest_metab(modalidade: str) -> dict:
    """Retorna o perfil metabólico mais recente como dict."""
    df = load_metab_history(modalidade, n=1)
    return df.iloc[0].to_dict() if not df.empty else {}
