"""
ATHELTICA — Motor de análise fisiológica de ficheiros FIT
==========================================================
Adaptado do script cp_e_cs.py (Colab) para uso em Streamlit.
Foco EXCLUSIVO em métricas fisiológicas: SmO2, THb, DFA-alpha1, respiração, FC.
(A parte de Critical Power / CS / veloclinic do script original NÃO é usada aqui.)

Diferenças em relação ao script original:
  • Lê o FIT a partir de bytes em memória (upload), não de um caminho no disco
  • Sem input()/print() — devolve estruturas de dados
  • Deteção de colunas mais flexível (substring, não só nomes exactos)
  • Funções puras e testáveis
"""

import io
import numpy as np
import pandas as pd

try:
    import fitdecode
    _TEM_FITDECODE = True
except ImportError:
    _TEM_FITDECODE = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. LEITURA DO FICHEIRO FIT
# ══════════════════════════════════════════════════════════════════════════════

def ler_fit(file_bytes):
    """
    Lê um ficheiro FIT a partir de bytes (upload do Streamlit).

    Devolve dict:
      {'records': [...], 'session': {...}, 'laps': [...], 'activity_name': str}
    ou {'erro': str} em caso de falha.
    """
    if not _TEM_FITDECODE:
        return {'erro': "Biblioteca 'fitdecode' não instalada. "
                        "Adiciona 'fitdecode' ao requirements.txt."}

    records_data, session_data, lap_data = [], {}, []
    activity_name = None

    try:
        with fitdecode.FitReader(io.BytesIO(file_bytes)) as fit:
            for frame in fit:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue

                if frame.name == 'record':
                    rec = {f.name: f.value for f in frame.fields if f.value is not None}
                    if rec:
                        records_data.append(rec)

                elif frame.name == 'session':
                    for f in frame.fields:
                        if f.value is not None:
                            session_data[f.name] = f.value
                            if f.name == 'sport' and not activity_name:
                                activity_name = str(f.value)

                elif frame.name == 'lap':
                    lap_info = {f.name: f.value for f in frame.fields if f.value is not None}
                    if lap_info:
                        lap_data.append(lap_info)
    except Exception as e:
        return {'erro': f"Erro ao ler o ficheiro FIT: {e}"}

    if not records_data:
        return {'erro': "O ficheiro não contém registos de dados (records)."}

    return {
        'records': records_data,
        'session': session_data,
        'laps': lap_data,
        'activity_name': activity_name or 'Atividade',
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. DETEÇÃO DE COLUNAS FISIOLÓGICAS
# ══════════════════════════════════════════════════════════════════════════════

# Padrões de procura por métrica. A deteção é por substring (case-insensitive),
# o que torna o sistema tolerante a nomes de sensor diferentes
# (ex.: "1st SmO2 Sensor 4503 on R. Quad" vs "SmO2" vs "smo2_left").
_PADROES_METRICAS = {
    'smo2':         ['smo2', 'sm o2', 'saturated hemoglobin', 'muscle oxygen'],
    'thb':          ['thb', 't hb', 'total hemoglobin'],
    'dfa1':         ['alpha1', 'alpha 1', 'dfa1', 'dfa a1', 'dfa_alpha1'],
    'respiration':  ['respirationrate', 'respiration rate', 'respiration', 'resp_rate'],
    'rr_ratio':     ['rra1 ratio', 'rr_ratio', 'rr ratio'],
    'power':        ['power'],
    'heart_rate':   ['heart_rate', 'heartrate', 'heart rate'],
    'cadence':      ['cadence'],
}

# Nomes "bonitos" para mostrar na interface
NOMES_METRICAS = {
    'smo2':        'SmO₂ (%)',
    'thb':         'THb',
    'dfa1':        'DFA-α1',
    'respiration': 'Respiração (rpm)',
    'rr_ratio':    'RR ratio',
    'power':       'Potência (W)',
    'heart_rate':  'FC (bpm)',
    'cadence':     'Cadência',
}


def detectar_colunas(df):
    """
    Mapeia as métricas fisiológicas para os nomes reais das colunas do DataFrame.
    Procura por substring (case-insensitive), com preferência por correspondência exacta.

    Devolve dict {metrica: nome_da_coluna}.
    """
    encontradas = {}
    cols = list(df.columns)
    cols_lower = {c: str(c).lower().strip() for c in cols}

    for metrica, padroes in _PADROES_METRICAS.items():
        achou = None
        # 1ª passagem: correspondência exacta com um padrão
        for col, cl in cols_lower.items():
            if cl in padroes:
                achou = col
                break
        # 2ª passagem: substring
        if achou is None:
            for padrao in padroes:
                for col, cl in cols_lower.items():
                    if padrao in cl and col not in encontradas.values():
                        # evitar apanhar 'power' dentro de 'hf_power' etc.
                        achou = col
                        break
                if achou:
                    break
        if achou is not None:
            # a coluna tem de ter dados numéricos utilizáveis
            serie = pd.to_numeric(df[achou], errors='coerce')
            if serie.notna().sum() >= 5:
                encontradas[metrica] = achou

    return encontradas


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRUÇÃO DO DATAFRAME E ATRIBUIÇÃO DE LAPS
# ══════════════════════════════════════════════════════════════════════════════

def construir_dataframe(fit_data):
    """
    Constrói o DataFrame 1Hz a partir dos records, com time_seconds e lap_number.

    Devolve (df, laps_info) onde laps_info é uma lista de dicts com
    lap_number, start_time, end_time, duration.
    """
    df = pd.DataFrame(fit_data['records'])
    if 'timestamp' not in df.columns:
        return None, []

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if df.empty:
        return None, []

    t0 = df['timestamp'].iloc[0]
    df['time_seconds'] = (df['timestamp'] - t0).dt.total_seconds()

    # ── Atribuir lap_number a cada record ────────────────────────────────────
    laps = fit_data.get('laps', [])
    laps_info = []
    df['lap_number'] = np.nan

    if laps:
        # Cada lap tem 'timestamp' (fim do lap) e opcionalmente 'start_time'
        fronteiras = []
        for i, lap in enumerate(laps):
            fim = lap.get('timestamp')
            ini = lap.get('start_time')
            if fim is None:
                continue
            fim = pd.to_datetime(fim, utc=True, errors='coerce')
            ini = pd.to_datetime(ini, utc=True, errors='coerce') if ini is not None else None
            if pd.isna(fim):
                continue
            fronteiras.append((i + 1, ini, fim, lap))

        anterior_fim = t0
        for lap_num, ini, fim, lap_raw in fronteiras:
            inicio = ini if (ini is not None and not pd.isna(ini)) else anterior_fim
            mask = (df['timestamp'] > inicio - pd.Timedelta(seconds=1)) & \
                   (df['timestamp'] <= fim)
            df.loc[mask, 'lap_number'] = lap_num
            dur = (fim - inicio).total_seconds()
            laps_info.append({
                'lap_number': lap_num,
                'start_time': inicio,
                'end_time': fim,
                'duration': max(dur, 0),
            })
            anterior_fim = fim

    # Records sem lap (ou ficheiro sem laps) → lap único
    if df['lap_number'].isna().all():
        df['lap_number'] = 1
        laps_info = [{
            'lap_number': 1,
            'start_time': df['timestamp'].iloc[0],
            'end_time': df['timestamp'].iloc[-1],
            'duration': float(df['time_seconds'].iloc[-1]),
        }]
    else:
        df['lap_number'] = df['lap_number'].ffill().fillna(1).astype(int)

    return df, laps_info


# ══════════════════════════════════════════════════════════════════════════════
# 4. ESTATÍSTICAS POR LAP + CLASSIFICAÇÃO TRABALHO/RECUPERAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def estatisticas_por_lap(df, laps_info, colunas):
    """
    Calcula avg/max/min de cada métrica encontrada, por lap.
    Devolve lista de dicts (lap_stats).
    """
    lap_stats = []
    for info in laps_info:
        d = df[df['lap_number'] == info['lap_number']]
        if len(d) == 0:
            continue
        s = {
            'lap_number': info['lap_number'],
            'start_time': info['start_time'],
            'end_time': info['end_time'],
            'duration': info['duration'],
            'n_pontos': len(d),
        }
        for metrica, col in colunas.items():
            if col in d.columns:
                vals = pd.to_numeric(d[col], errors='coerce').dropna()
                if len(vals) > 0:
                    s[f'avg_{metrica}'] = float(vals.mean())
                    s[f'max_{metrica}'] = float(vals.max())
                    s[f'min_{metrica}'] = float(vals.min())
        lap_stats.append(s)
    return lap_stats


def classificar_laps(lap_stats, dur_min=60, dur_max=600, frac_mediana=0.7):
    """
    Classifica cada lap como 'work' ou 'recovery'.
    Critério (do script original): potência >= frac_mediana × mediana das potências
    E duração entre dur_min e dur_max segundos.

    Se não houver potência, usa a FC como alternativa.
    Modifica lap_stats in-place e devolve-o.
    """
    chave = None
    if any('avg_power' in l for l in lap_stats):
        chave = 'avg_power'
    elif any('avg_heart_rate' in l for l in lap_stats):
        chave = 'avg_heart_rate'

    if chave is None:
        for l in lap_stats:
            l['phase'] = 'work'
        return lap_stats

    valores = [l[chave] for l in lap_stats if chave in l]
    if not valores:
        for l in lap_stats:
            l['phase'] = 'work'
        return lap_stats

    limiar = float(np.median(valores)) * frac_mediana

    for l in lap_stats:
        if chave in l:
            por_intensidade = l[chave] >= limiar
            por_duracao = dur_min <= l['duration'] <= dur_max
            l['phase'] = 'work' if (por_intensidade and por_duracao) else 'recovery'
        else:
            l['phase'] = 'recovery'
    return lap_stats


def identificar_sequencias(lap_stats):
    """
    Identifica pares consecutivos trabalho→recuperação.
    Devolve lista de dicts {'work_lap': ..., 'recovery_lap': ...}.
    """
    seqs = []
    for i in range(len(lap_stats) - 1):
        if lap_stats[i].get('phase') == 'work' and \
           lap_stats[i + 1].get('phase') == 'recovery':
            seqs.append({'work_lap': lap_stats[i], 'recovery_lap': lap_stats[i + 1]})
    return seqs


# ══════════════════════════════════════════════════════════════════════════════
# 5. CINÉTICA DE RESTAURAÇÃO (trabalho → recuperação)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_restauracao(df, sequencia, coluna, sentido='sobe', pct_alvo=0.8):
    """
    Calcula o tempo e a taxa de restauração de uma métrica durante a recuperação.

    Réplica da lógica do script original: define o alvo como
        início + pct_alvo × (fim − início)
    e mede quanto tempo demora a atingi-lo.

    sentido='sobe'  → métricas que sobem na recuperação (SmO2, DFA1)
    sentido='desce' → métricas que descem na recuperação (FC, respiração)

    Devolve (tempo_segundos, taxa_por_segundo) ou (None, None).
    """
    rec_lap = sequencia['recovery_lap']
    d = df[df['lap_number'] == rec_lap['lap_number']]
    if len(d) < 3 or coluna not in d.columns:
        return None, None

    vals = pd.to_numeric(d[coluna], errors='coerce')
    tempos = d['time_seconds'].values
    mask = vals.notna().values
    if mask.sum() < 3:
        return None, None

    vals = vals.values[mask]
    tempos = tempos[mask]

    v_ini, v_fim = float(vals[0]), float(vals[-1])
    if v_ini == v_fim:
        return None, None

    alvo = v_ini + pct_alvo * (v_fim - v_ini)

    for j in range(len(vals)):
        atingiu = (vals[j] >= alvo) if sentido == 'sobe' else (vals[j] <= alvo)
        if atingiu:
            t = float(tempos[j] - tempos[0])
            taxa = (alvo - v_ini) / t if t > 0 else 0.0
            return t, taxa

    # Não atingiu dentro do lap → devolve a duração total e a taxa média
    dur = float(rec_lap['duration']) if rec_lap['duration'] > 0 else float(tempos[-1] - tempos[0])
    return dur, ((v_fim - v_ini) / dur if dur > 0 else 0.0)


# Sentido de recuperação por métrica (sobe ou desce durante a recuperação)
_SENTIDO_RECUPERACAO = {
    'smo2': 'sobe',          # reoxigenação muscular
    'thb': 'sobe',
    'dfa1': 'sobe',          # regresso da complexidade autonómica
    'heart_rate': 'desce',   # FC baixa na recuperação
    'respiration': 'desce',  # respiração normaliza
}


def analisar_restauracao_completa(df, lap_stats, colunas):
    """
    Corre a cinética de restauração para todas as métricas disponíveis,
    em todas as sequências trabalho→recuperação.

    Devolve dict:
      {'sequencias': [ {...} ], 'resumo': {metrica: {media, std, n}} }
    """
    seqs = identificar_sequencias(lap_stats)
    if not seqs:
        return {'sequencias': [], 'resumo': {}}

    linhas = []
    for idx, seq in enumerate(seqs, start=1):
        linha = {
            'sequencia': idx,
            'lap_trabalho': seq['work_lap']['lap_number'],
            'lap_recuperacao': seq['recovery_lap']['lap_number'],
            'dur_trabalho_s': seq['work_lap']['duration'],
            'dur_recup_s': seq['recovery_lap']['duration'],
        }
        if 'avg_power' in seq['work_lap']:
            linha['potencia_trabalho_W'] = seq['work_lap']['avg_power']

        for metrica, sentido in _SENTIDO_RECUPERACAO.items():
            if metrica not in colunas:
                continue
            t, taxa = calcular_restauracao(df, seq, colunas[metrica], sentido=sentido)
            if t is not None:
                linha[f'tempo_{metrica}_s'] = round(t, 1)
                linha[f'taxa_{metrica}'] = round(taxa, 4)
        linhas.append(linha)

    # Resumo estatístico por métrica
    resumo = {}
    dfl = pd.DataFrame(linhas)
    for metrica in _SENTIDO_RECUPERACAO:
        col = f'tempo_{metrica}_s'
        if col in dfl.columns:
            vals = dfl[col].dropna()
            if len(vals) > 0:
                resumo[metrica] = {
                    'media': float(vals.mean()),
                    'std': float(vals.std()) if len(vals) > 1 else 0.0,
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'n': int(len(vals)),
                }

    return {'sequencias': linhas, 'resumo': resumo, 'tabela': dfl}


# ══════════════════════════════════════════════════════════════════════════════
# 6. LIMIARES DE SmO2 (breakpoints)
# ══════════════════════════════════════════════════════════════════════════════

def _dmax_smo2(x, y):
    """Ponto de máxima curvatura (Dmax adaptado). Requer scipy."""
    try:
        from scipy.interpolate import interp1d
        if len(x) < 4:
            return None
        f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
        xi = np.linspace(x.min(), x.max(), 200)
        yi = f(xi)
        dy = np.gradient(yi, xi)
        d2y = np.gradient(dy, xi)
        curvatura = np.abs(d2y) / (1 + dy ** 2) ** 1.5
        return float(xi[int(np.argmax(curvatura))])
    except Exception:
        return None


def _quebra_inclinacao(x, y):
    """Ponto de quebra que maximiza o R² combinado de duas regressões."""
    try:
        melhor_r2, melhor = -np.inf, None
        for i in range(2, len(x) - 2):
            x1, y1 = x[:i], y[:i]
            x2, y2 = x[i:], y[i:]
            if len(x1) < 2 or len(x2) < 2:
                continue
            c1 = np.polyfit(x1, y1, 1)
            r2_1 = 1 - np.sum((y1 - np.polyval(c1, x1)) ** 2) / max(np.sum((y1 - y1.mean()) ** 2), 1e-9)
            c2 = np.polyfit(x2, y2, 1)
            r2_2 = 1 - np.sum((y2 - np.polyval(c2, x2)) ** 2) / max(np.sum((y2 - y2.mean()) ** 2), 1e-9)
            r2 = (r2_1 + r2_2) / 2
            if r2 > melhor_r2:
                melhor_r2, melhor = r2, float(x[i])
        return melhor
    except Exception:
        return None


def _deflexao_smo2(x, y, taxa_lim=-0.1):
    """Primeiro ponto onde a queda de SmO2 excede taxa_lim (%/W)."""
    try:
        if len(x) < 3:
            return None
        dy = np.diff(y)
        dx = np.diff(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            taxa = np.where(dx != 0, dy / dx, np.nan)
        idx = np.where(taxa < taxa_lim)[0]
        return float(x[idx[0]]) if len(idx) > 0 else None
    except Exception:
        return None


def calcular_limiares_smo2(lap_stats, colunas):
    """
    Calcula limiares de SmO2 vs intensidade a partir das médias por lap de trabalho.
    Usa três métodos independentes e faz a média dos que produzirem resultado.

    Devolve dict {'dmax':..., 'quebra':..., 'deflexao':..., 'media':...,
                  'pontos': DataFrame} ou None.
    """
    if 'smo2' not in colunas:
        return None

    intensidade = 'avg_power' if any('avg_power' in l for l in lap_stats) else 'avg_heart_rate'
    work = [l for l in lap_stats
            if l.get('phase') == 'work' and 'avg_smo2' in l and intensidade in l]
    if len(work) < 3:
        return None

    pontos = pd.DataFrame([
        {'lap': l['lap_number'], 'intensidade': l[intensidade], 'smo2': l['avg_smo2']}
        for l in work
    ]).sort_values('intensidade').reset_index(drop=True)

    x = pontos['intensidade'].values.astype(float)
    y = pontos['smo2'].values.astype(float)

    dmax = _dmax_smo2(x, y)
    quebra = _quebra_inclinacao(x, y)
    defl = _deflexao_smo2(x, y)

    validos = [v for v in (dmax, quebra, defl) if v is not None]
    media = float(np.mean(validos)) if validos else None

    return {
        'dmax': dmax,
        'quebra': quebra,
        'deflexao': defl,
        'media': media,
        'pontos': pontos,
        'unidade': 'W' if intensidade == 'avg_power' else 'bpm',
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. DECOUPLING FC/POTÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

def calcular_decoupling(lap_stats):
    """
    Decoupling = variação do rácio FC/potência ao longo dos laps de trabalho,
    normalizada ao primeiro lap. Valores positivos = deriva cardiovascular.

    Devolve DataFrame ou None.
    """
    work = [l for l in lap_stats
            if l.get('phase') == 'work' and 'avg_power' in l and 'avg_heart_rate' in l
            and l['avg_power'] > 0]
    if len(work) < 2:
        return None

    linhas = []
    for l in work:
        linhas.append({
            'lap': l['lap_number'],
            'potencia': l['avg_power'],
            'fc': l['avg_heart_rate'],
            'ratio_fc_pot': l['avg_heart_rate'] / l['avg_power'],
        })
    d = pd.DataFrame(linhas)
    base = d['ratio_fc_pot'].iloc[0]
    d['decoupling_pct'] = ((d['ratio_fc_pot'] / base) - 1) * 100 if base > 0 else np.nan
    return d


# ══════════════════════════════════════════════════════════════════════════════
# 8. INDICADORES DE FADIGA
# ══════════════════════════════════════════════════════════════════════════════

def classificar_fadiga(restauracao, decoupling_df=None):
    """
    Indicadores de fadiga, adaptados do script original:
      1. Tendência do tempo de restauração da FC (slope ao longo das sequências)
      2. Consistência da restauração (std vs média)
      3. Deriva cardiovascular (decoupling final)

    Devolve dict com os indicadores e um veredicto textual.
    """
    ind = {}
    tabela = restauracao.get('tabela')

    # 1. Tendência do tempo de restauração da FC
    if tabela is not None and 'tempo_heart_rate_s' in tabela.columns:
        vals = tabela['tempo_heart_rate_s'].dropna()
        if len(vals) >= 3:
            slope = float(np.polyfit(np.arange(len(vals)), vals.values, 1)[0])
            ind['slope_restauracao_fc'] = round(slope, 2)
            ind['tendencia_fc'] = 'A PIORAR' if slope > 0 else 'ESTÁVEL/A MELHORAR'

    # 2. Consistência
    resumo_fc = restauracao.get('resumo', {}).get('heart_rate')
    if resumo_fc and resumo_fc['media'] > 0:
        cv = resumo_fc['std'] / resumo_fc['media']
        ind['cv_restauracao_fc'] = round(cv, 3)
        ind['consistencia'] = 'INCONSISTENTE' if cv > 0.30 else 'CONSISTENTE'

    # 3. Decoupling
    if decoupling_df is not None and 'decoupling_pct' in decoupling_df.columns:
        final = float(decoupling_df['decoupling_pct'].iloc[-1])
        ind['decoupling_final_pct'] = round(final, 1)
        if final > 10:
            ind['deriva_cardiovascular'] = 'ELEVADA'
        elif final > 5:
            ind['deriva_cardiovascular'] = 'MODERADA'
        else:
            ind['deriva_cardiovascular'] = 'BAIXA'

    # Veredicto: conta sinais de alerta
    alertas = 0
    if ind.get('tendencia_fc') == 'A PIORAR':
        alertas += 1
    if ind.get('consistencia') == 'INCONSISTENTE':
        alertas += 1
    if ind.get('deriva_cardiovascular') in ('ELEVADA', 'MODERADA'):
        alertas += 1

    ind['n_alertas'] = alertas
    if alertas >= 2:
        ind['veredicto'] = 'ELEVADA'
        ind['veredicto_cor'] = '#e74c3c'
    elif alertas == 1:
        ind['veredicto'] = 'MODERADA'
        ind['veredicto_cor'] = '#f39c12'
    else:
        ind['veredicto'] = 'BAIXA'
        ind['veredicto_cor'] = '#27ae60'

    return ind


def tempo_ate_falha(lap_stats):
    """
    Estimativa de tempo até à falha (CMR), do script original: extrapola a taxa de
    queda do SmO2 de cada lap de trabalho até ao SmO2 mínimo observado na sessão.

    Devolve DataFrame ou None.
    """
    work = [l for l in lap_stats
            if l.get('phase') == 'work' and 'avg_smo2' in l and 'min_smo2' in l
            and l['duration'] > 0]
    if len(work) < 2:
        return None

    smo2_min_global = min(l['min_smo2'] for l in work)
    linhas = []
    for l in work:
        delta_por_min = (l['avg_smo2'] - l['min_smo2']) / (l['duration'] / 60.0)
        if delta_por_min > 0:
            t = (l['min_smo2'] - smo2_min_global) / delta_por_min
            linhas.append({
                'lap': l['lap_number'],
                'potencia': l.get('avg_power'),
                'smo2_min': l['min_smo2'],
                'queda_smo2_por_min': round(delta_por_min, 2),
                'tempo_falha_min': round(t, 1) if t > 0 else None,
            })
    return pd.DataFrame(linhas) if linhas else None


# ══════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def analisar_fit(file_bytes, laps_trabalho_manual=None):
    """
    Pipeline completo: bytes do FIT → análise fisiológica completa.

    laps_trabalho_manual : lista opcional de lap_numbers que o utilizador
                           marcou como trabalho (sobrepõe a deteção automática).

    Devolve dict com tudo, ou {'erro': str}.
    """
    fit = ler_fit(file_bytes)
    if 'erro' in fit:
        return fit

    df, laps_info = construir_dataframe(fit)
    if df is None or df.empty:
        return {'erro': "Não foi possível construir a série temporal do ficheiro."}

    colunas = detectar_colunas(df)
    lap_stats = estatisticas_por_lap(df, laps_info, colunas)
    if not lap_stats:
        return {'erro': "Nenhum lap com dados utilizáveis."}

    # Classificação automática, depois override manual se fornecido
    lap_stats = classificar_laps(lap_stats)
    if laps_trabalho_manual is not None:
        manual = set(laps_trabalho_manual)
        for l in lap_stats:
            l['phase'] = 'work' if l['lap_number'] in manual else 'recovery'

    restauracao = analisar_restauracao_completa(df, lap_stats, colunas)
    limiares = calcular_limiares_smo2(lap_stats, colunas)
    decoupling = calcular_decoupling(lap_stats)
    fadiga = classificar_fadiga(restauracao, decoupling)
    falha = tempo_ate_falha(lap_stats)

    return {
        'df': df,
        'colunas': colunas,
        'lap_stats': lap_stats,
        'laps_info': laps_info,
        'restauracao': restauracao,
        'limiares': limiares,
        'decoupling': decoupling,
        'fadiga': fadiga,
        'tempo_falha': falha,
        'activity_name': fit.get('activity_name', 'Atividade'),
        'session': fit.get('session', {}),
        'duracao_total_s': float(df['time_seconds'].iloc[-1]) if len(df) else 0.0,
        'data_sessao': (df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')
                        if 'timestamp' in df.columns and len(df) else None),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. RESUMO PARA HISTÓRICO (comparar sessões ao longo do tempo)
# ══════════════════════════════════════════════════════════════════════════════

def resumir_para_historico(resultado, nome_ficheiro=''):
    """
    Extrai uma linha-resumo da análise, para guardar e comparar sessões.
    Devolve dict achatado (uma linha de CSV).
    """
    if 'erro' in resultado:
        return None

    r = {
        'data': resultado.get('data_sessao'),
        'ficheiro': nome_ficheiro,
        'atividade': resultado.get('activity_name'),
        'duracao_min': round(resultado.get('duracao_total_s', 0) / 60, 1),
        'n_laps': len(resultado.get('lap_stats', [])),
        'n_laps_trabalho': sum(1 for l in resultado.get('lap_stats', [])
                               if l.get('phase') == 'work'),
    }

    # Métricas médias dos laps de trabalho
    work = [l for l in resultado.get('lap_stats', []) if l.get('phase') == 'work']
    if work:
        for metrica in ['power', 'heart_rate', 'smo2', 'thb', 'dfa1', 'respiration']:
            vals = [l[f'avg_{metrica}'] for l in work if f'avg_{metrica}' in l]
            if vals:
                r[f'{metrica}_medio'] = round(float(np.mean(vals)), 2)
        mins = [l['min_smo2'] for l in work if 'min_smo2' in l]
        if mins:
            r['smo2_min'] = round(float(np.min(mins)), 1)

    # Tempos de restauração médios
    for metrica, res in resultado.get('restauracao', {}).get('resumo', {}).items():
        r[f'restauracao_{metrica}_s'] = round(res['media'], 1)

    # Limiares
    lim = resultado.get('limiares')
    if lim and lim.get('media') is not None:
        r['limiar_smo2'] = round(lim['media'], 1)
        r['limiar_unidade'] = lim.get('unidade', '')

    # Decoupling e fadiga
    dec = resultado.get('decoupling')
    if dec is not None and len(dec) > 0:
        r['decoupling_pct'] = round(float(dec['decoupling_pct'].iloc[-1]), 1)
    fad = resultado.get('fadiga', {})
    r['fadiga'] = fad.get('veredicto')
    r['fadiga_alertas'] = fad.get('n_alertas')

    return r
