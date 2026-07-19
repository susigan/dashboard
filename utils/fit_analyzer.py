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
    'resp_enhanced': ['enhanced_respiration_rate', 'enhanced respiration rate'],
    # Qualidade do sinal HRV: percentagem de batimentos corrigidos/interpolados.
    # Valores altos (>5%) indicam que o DFA-α1 desse período é pouco fiável.
    'artifacts':    ['artifacts', 'artefacts', 'artifact'],
    # Rácio RR do algoritmo alpha1 (métrica auxiliar do sensor)
    'rr_ratio':     ['rra1_ratio', 'rra1 ratio', 'rr_ratio', 'rr ratio'],
    'hr_alphahrv':  ['heartrate_alphahrv', 'heartrate alphahrv'],
    'power':        ['power'],
    'heart_rate':   ['heart_rate', 'heartrate', 'heart rate'],
    'cadence':      ['cadence'],
    'speed':        ['enhanced_speed', 'speed'],
    'distance':     ['distance'],
    'cycle_length': ['cycle_length16', 'cycle_length'],
}

# Nomes "bonitos" para mostrar na interface
NOMES_METRICAS = {
    'smo2':          'SmO₂ (%)',
    'thb':           'THb',
    'dfa1':          'DFA-α1',
    'respiration':   'Respiração (rpm)',
    'resp_enhanced': 'Respiração enhanced',
    'artifacts':     'Artifacts (%)',
    'rr_ratio':      'RRa1 ratio',
    'hr_alphahrv':   'FC (alphaHRV)',
    'power':         'Potência (W)',
    'heart_rate':    'FC (bpm)',
    'cadence':       'Cadência',
    'speed':         'Velocidade',
    'distance':      'Distância',
    'cycle_length':  'Comprimento de ciclo',
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

    # Colunas que nunca devem ser apanhadas por substring (derivadas/acumuladas
    # que confundiriam com a métrica instantânea — ex.: accumulated_power vs power)
    _EXCLUIR = ('accumulated', 'total_', 'avg_', 'max_', 'min_', 'norm', 'fractional')

    for metrica, padroes in _PADROES_METRICAS.items():
        achou = None
        # 1ª passagem: correspondência exacta com um padrão
        for col, cl in cols_lower.items():
            if cl in padroes:
                achou = col
                break
        # 2ª passagem: substring, evitando derivadas e colunas já usadas
        if achou is None:
            for padrao in padroes:
                for col, cl in cols_lower.items():
                    if col in encontradas.values():
                        continue
                    if any(x in cl for x in _EXCLUIR):
                        continue
                    if padrao in cl:
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

def _txt(valor):
    """
    Normaliza um valor de enum do FIT para string minúscula.
    O fitdecode devolve por vezes o nome do enum ('active'), por vezes o inteiro
    bruto (0). Esta função trata ambos e devolve None se não houver valor.
    """
    if valor is None:
        return None
    try:
        if isinstance(valor, (int, np.integer)) and not isinstance(valor, bool):
            return str(int(valor))
        s = str(valor).strip().lower()
        return s if s and s not in ('nan', 'none') else None
    except Exception:
        return None


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
        # ── Determinar as fronteiras de cada lap ─────────────────────────────
        # Nem todos os gravadores preenchem o 'timestamp' do lap correctamente:
        # há ficheiros em que TODOS os laps partilham o mesmo timestamp (o do
        # início da sessão), o que faria colapsar tudo num único lap. Por isso a
        # ordem de preferência para o fim do lap é:
        #   1) start_time + total_elapsed_time (o mais fiável quando existe)
        #   2) start_time do lap seguinte
        #   3) timestamp do lap (só se for coerente)
        brutos = []
        for i, lap in enumerate(laps):
            ini = pd.to_datetime(lap.get('start_time'), utc=True, errors='coerce') \
                if lap.get('start_time') is not None else pd.NaT
            ts = pd.to_datetime(lap.get('timestamp'), utc=True, errors='coerce') \
                if lap.get('timestamp') is not None else pd.NaT
            elapsed = lap.get('total_elapsed_time')
            if elapsed is None:
                elapsed = lap.get('total_timer_time')
            try:
                elapsed = float(elapsed) if elapsed is not None else None
            except (TypeError, ValueError):
                elapsed = None
            brutos.append({'idx': i + 1, 'ini': ini, 'ts': ts,
                           'elapsed': elapsed, 'raw': lap})

        # O 'timestamp' é fiável se os laps tiverem timestamps distintos entre si
        ts_validos = [b['ts'] for b in brutos if pd.notna(b['ts'])]
        ts_fiavel = len(set(ts_validos)) > 1 and len(ts_validos) == len(brutos)

        fronteiras = []
        for i, b in enumerate(brutos):
            inicio = b['ini']
            if pd.isna(inicio):
                # Sem start_time: usa o fim do lap anterior, ou o início da sessão
                inicio = fronteiras[-1][2] if fronteiras else t0

            fim = pd.NaT
            if b['elapsed'] is not None and b['elapsed'] > 0:
                fim = inicio + pd.Timedelta(seconds=b['elapsed'])
            if pd.isna(fim) and i + 1 < len(brutos) and pd.notna(brutos[i + 1]['ini']):
                fim = brutos[i + 1]['ini']
            if pd.isna(fim) and ts_fiavel and pd.notna(b['ts']):
                fim = b['ts']
            if pd.isna(fim):
                fim = df['timestamp'].iloc[-1]

            if fim > inicio:
                fronteiras.append((b['idx'], inicio, fim, b['raw']))

        anterior_fim = t0
        for lap_num, inicio, fim, lap_raw in fronteiras:
            mask = (df['timestamp'] >= inicio - pd.Timedelta(seconds=0.5)) & \
                   (df['timestamp'] <= fim)
            df.loc[mask, 'lap_number'] = lap_num
            dur = (fim - inicio).total_seconds()
            laps_info.append({
                'lap_number': lap_num,
                'start_time': inicio,
                'end_time': fim,
                'duration': max(dur, 0),
                # Campos nativos do FIT que ajudam a classificar sem adivinhar:
                #   intensity   → 'active' | 'rest' | 'warmup' | 'cooldown' | 'recovery'
                #   lap_trigger → 'manual' | 'time' | 'distance' | 'session_end' | ...
                #   event/event_type → confirmam que é um evento de lap
                'intensity': _txt(lap_raw.get('intensity')),
                'lap_trigger': _txt(lap_raw.get('lap_trigger')),
                'event': _txt(lap_raw.get('event')),
                'event_type': _txt(lap_raw.get('event_type')),
                'avg_power_fit': lap_raw.get('avg_power'),
                'avg_hr_fit': lap_raw.get('avg_heart_rate'),
            })
            anterior_fim = fim

    # Records sem lap (ou ficheiro sem laps) → tentar segmentar automaticamente
    if df['lap_number'].isna().all():
        df, laps_info = _segmentar_sem_laps(df)
    else:
        df['lap_number'] = df['lap_number'].ffill().fillna(1).astype(int)

    return df, laps_info


def segmentar_por_intervalos(df, intervalos):
    """
    Cria laps a partir de intervalos de TRABALHO definidos manualmente pelo
    utilizador (em segundos desde o início da sessão). Tudo o que fica fora dos
    intervalos ("buracos") passa a ser recuperação.

    intervalos : lista de tuplos (inicio_s, fim_s) — os blocos de trabalho.

    Exemplo: [(600, 780), (840, 1020)] numa sessão de 1200s produz:
        1. 0-600s      → recuperação (aquecimento)
        2. 600-780s    → trabalho
        3. 780-840s    → recuperação
        4. 840-1020s   → trabalho
        5. 1020-1200s  → recuperação

    Devolve (df com lap_number, laps_info) com 'intensity' já definido
    ('active' para trabalho, 'rest' para os buracos).
    """
    d = df.copy()
    t_min = float(d['time_seconds'].min())
    t_max = float(d['time_seconds'].max())

    # Normalizar, validar e ordenar os intervalos
    limpos = []
    for par in (intervalos or []):
        try:
            ini, fim = float(par[0]), float(par[1])
        except (TypeError, ValueError, IndexError):
            continue
        ini, fim = max(ini, t_min), min(fim, t_max)
        if fim - ini >= 5:  # ignorar intervalos degenerados
            limpos.append((ini, fim))
    limpos.sort()

    # Fundir intervalos sobrepostos
    fundidos = []
    for ini, fim in limpos:
        if fundidos and ini <= fundidos[-1][1]:
            fundidos[-1] = (fundidos[-1][0], max(fundidos[-1][1], fim))
        else:
            fundidos.append((ini, fim))

    if not fundidos:
        return _segmentar_sem_laps(df)

    # Construir a sequência completa: buracos + trabalho, por ordem temporal
    blocos = []
    cursor = t_min
    for ini, fim in fundidos:
        if ini - cursor >= 5:
            blocos.append((cursor, ini, 'rest'))
        blocos.append((ini, fim, 'active'))
        cursor = fim
    if t_max - cursor >= 5:
        blocos.append((cursor, t_max, 'rest'))

    d['lap_number'] = np.nan
    laps_info = []
    for n, (ini, fim, tipo) in enumerate(blocos, start=1):
        mask = (d['time_seconds'] >= ini) & (d['time_seconds'] <= fim)
        if mask.sum() == 0:
            continue
        d.loc[mask, 'lap_number'] = n
        sub = d[mask]
        laps_info.append({
            'lap_number': n,
            'start_time': sub['timestamp'].iloc[0],
            'end_time': sub['timestamp'].iloc[-1],
            'duration': float(fim - ini),
            'intensity': tipo,
            'lap_trigger': 'manual_tempo',
            'event': None, 'event_type': None,
            'avg_power_fit': None, 'avg_hr_fit': None,
        })

    d['lap_number'] = d['lap_number'].ffill().bfill().fillna(1).astype(int)
    # Renumerar de forma contígua (caso algum bloco tenha ficado sem records)
    mapa = {v: i + 1 for i, v in enumerate(sorted(d['lap_number'].unique()))}
    d['lap_number'] = d['lap_number'].map(mapa)
    for info in laps_info:
        info['lap_number'] = mapa.get(info['lap_number'], info['lap_number'])
    laps_info = [i for i in laps_info if i['lap_number'] in set(d['lap_number'])]
    laps_info.sort(key=lambda x: x['lap_number'])

    return d, laps_info


def _segmentar_sem_laps(df, min_dur=45, suavizacao=15, frac_corte=None):
    """
    Segmenta a sessão automaticamente quando o ficheiro NÃO tem laps marcados.

    Estratégia: usa o sinal de intensidade (potência, ou FC como alternativa),
    suaviza-o, e separa em blocos "alto" vs "baixo" pelo ponto médio entre os dois
    modos da distribuição. Blocos contíguos do mesmo tipo formam um segmento;
    segmentos mais curtos que min_dur são fundidos com o vizinho, para não gerar
    dezenas de micro-laps por causa de oscilações.

    Devolve (df com lap_number, laps_info) — cada segmento é tratado como um lap.
    Se não houver sinal utilizável, devolve um único lap com toda a sessão.
    """
    col = None
    for c in ['power', 'Power', 'heart_rate', 'HeartRate']:
        if c in df.columns and pd.to_numeric(df[c], errors='coerce').notna().sum() > 30:
            col = c
            break

    def _lap_unico():
        d = df.copy()
        d['lap_number'] = 1
        info = [{
            'lap_number': 1,
            'start_time': d['timestamp'].iloc[0],
            'end_time': d['timestamp'].iloc[-1],
            'duration': float(d['time_seconds'].iloc[-1]),
            'intensity': None, 'lap_trigger': 'auto_none',
            'event': None, 'event_type': None,
            'avg_power_fit': None, 'avg_hr_fit': None,
        }]
        return d, info

    if col is None:
        return _lap_unico()

    sinal = pd.to_numeric(df[col], errors='coerce').interpolate(limit=5)
    sinal = sinal.rolling(suavizacao, min_periods=1, center=True).mean()
    vals = sinal.dropna().values
    if len(vals) < 60:
        return _lap_unico()

    # Ponto médio entre os dois modos (mesma lógica da classificação de laps)
    med = float(np.median(vals))
    altos = vals[vals >= med]
    baixos = vals[vals < med]
    if len(altos) == 0 or len(baixos) == 0:
        return _lap_unico()
    modo_alto, modo_baixo = float(np.median(altos)), float(np.median(baixos))
    separacao = (modo_alto - modo_baixo) / modo_alto if modo_alto > 0 else 0

    if frac_corte is not None:
        # Corte explícito pedido pelo utilizador: X% do valor típico de trabalho.
        # Ex.: frac_corte=0.5 → tudo abaixo de 50% da potência de trabalho conta
        # como recuperação. Útil quando a separação automática não acerta.
        corte = modo_alto * float(frac_corte)
    else:
        if separacao < 0.20:
            # Intensidade demasiado constante → não é uma sessão intervalada
            return _lap_unico()
        corte = (modo_alto + modo_baixo) / 2.0

    alto = (sinal >= corte).fillna(False).values

    # Blocos contíguos
    blocos = []
    ini = 0
    for i in range(1, len(alto)):
        if alto[i] != alto[i - 1]:
            blocos.append([ini, i - 1, bool(alto[ini])])
            ini = i
    blocos.append([ini, len(alto) - 1, bool(alto[ini])])

    # Fundir blocos curtos com o vizinho anterior
    tempos = df['time_seconds'].values
    fundidos = []
    for b in blocos:
        dur = tempos[b[1]] - tempos[b[0]]
        if fundidos and dur < min_dur:
            fundidos[-1][1] = b[1]
        else:
            fundidos.append(b)
    # Segunda passagem (fundir curtos que sobraram no início)
    if len(fundidos) > 1:
        dur0 = tempos[fundidos[0][1]] - tempos[fundidos[0][0]]
        if dur0 < min_dur:
            fundidos[1][0] = fundidos[0][0]
            fundidos.pop(0)

    d = df.copy()
    d['lap_number'] = 0
    laps_info = []
    for n, (i0, i1, eh_alto) in enumerate(fundidos, start=1):
        d.iloc[i0:i1 + 1, d.columns.get_loc('lap_number')] = n
        laps_info.append({
            'lap_number': n,
            'start_time': d['timestamp'].iloc[i0],
            'end_time': d['timestamp'].iloc[i1],
            'duration': float(tempos[i1] - tempos[i0]),
            'intensity': 'active' if eh_alto else 'rest',
            'lap_trigger': 'auto_segmentado',
            'event': None, 'event_type': None,
            'avg_power_fit': None, 'avg_hr_fit': None,
        })
    d['lap_number'] = d['lap_number'].replace(0, np.nan).ffill().bfill().astype(int)
    return d, laps_info


# ══════════════════════════════════════════════════════════════════════════════
# 4. ESTATÍSTICAS POR LAP + CLASSIFICAÇÃO TRABALHO/RECUPERAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def estatisticas_por_lap(df, laps_info, colunas, janela_final_s=60):
    """
    Calcula avg/max/min de cada métrica encontrada, por lap.

    janela_final_s : usa apenas os últimos N segundos de cada lap para calcular as
        MÉDIAS. Isto é importante porque métricas como o SmO2 têm cinética lenta
        (tau ~30-60s): no início de um degrau ainda estão em transição a partir da
        intensidade anterior, e só no final atingem o estado estacionário daquela
        intensidade. Usar o lap inteiro sobrestima o SmO2 em vários pontos
        percentuais, e o erro cresce com a intensidade — o que distorce a forma da
        curva SmO2-vs-intensidade e desloca os limiares.
        Passar None ou 0 usa o lap inteiro (comportamento antigo).

        max/min continuam a ser calculados sobre o lap INTEIRO (interessa o
        extremo atingido, não o estado estacionário).

    Devolve lista de dicts (lap_stats).
    """
    lap_stats = []
    for info in laps_info:
        d = df[df['lap_number'] == info['lap_number']]
        if len(d) == 0:
            continue

        # Subconjunto de estado estacionário (últimos N segundos do lap)
        if janela_final_s and janela_final_s > 0 and 'time_seconds' in d.columns:
            t_fim = d['time_seconds'].iloc[-1]
            d_est = d[d['time_seconds'] >= t_fim - janela_final_s]
            # Se o lap for curto demais, usa pelo menos os últimos 25% dos pontos
            if len(d_est) < 3:
                d_est = d.tail(max(3, int(len(d) * 0.25)))
        else:
            d_est = d

        s = {
            'lap_number': info['lap_number'],
            'start_time': info['start_time'],
            'end_time': info['end_time'],
            'duration': info['duration'],
            # Tempos em segundos desde o início da sessão (para a tabela editável)
            '_t_ini': float(d['time_seconds'].iloc[0]),
            '_t_fim': float(d['time_seconds'].iloc[-1]),
            'n_pontos': len(d),
            'n_pontos_estacionario': len(d_est),
            'janela_final_s': janela_final_s if janela_final_s else None,
            # Campos nativos do FIT (podem ser None se o dispositivo não os grava)
            'intensity': info.get('intensity'),
            'lap_trigger': info.get('lap_trigger'),
            'event': info.get('event'),
        }
        for metrica, col in colunas.items():
            if col in d.columns:
                # Média sobre o estado estacionário
                vals_est = pd.to_numeric(d_est[col], errors='coerce').dropna()
                if len(vals_est) > 0:
                    s[f'avg_{metrica}'] = float(vals_est.mean())
                # max/min sobre o lap inteiro
                vals_all = pd.to_numeric(d[col], errors='coerce').dropna()
                if len(vals_all) > 0:
                    s[f'max_{metrica}'] = float(vals_all.max())
                    s[f'min_{metrica}'] = float(vals_all.min())
                    s[f'avg_lap_inteiro_{metrica}'] = float(vals_all.mean())
        lap_stats.append(s)
    return lap_stats


def classificar_laps(lap_stats, dur_min=60, dur_max=600, frac_mediana=0.7,
                     laps_excluidos=None):
    """
    Classifica cada lap como 'work', 'recovery' ou 'excluded'.
    Critério (do script original): potência >= frac_mediana × mediana das potências
    E duração entre dur_min e dur_max segundos.

    laps_excluidos : lista de lap_numbers a excluir da análise (ex.: aquecimento,
        arrefecimento). Ficam com phase='excluded' e são ignorados em todos os
        cálculos — a mediana de referência também é calculada sem eles, para o
        aquecimento não puxar o limiar para baixo.

    Se não houver potência, usa a FC como alternativa.
    Modifica lap_stats in-place e devolve-o.
    """
    excluidos = set(laps_excluidos or [])

    # Marcar os excluídos primeiro
    for l in lap_stats:
        if l['lap_number'] in excluidos:
            l['phase'] = 'excluded'

    considerados = [l for l in lap_stats if l['lap_number'] not in excluidos]
    if not considerados:
        return lap_stats

    # ── Prioridade 1: campo 'intensity' nativo do FIT ────────────────────────
    # Muitos dispositivos gravam o tipo de lap directamente. Se estiver presente
    # E distinguir de facto os laps (não for tudo 'active'), é mais fiável do que
    # inferir pela potência. Valores possíveis: active, rest, warmup, cooldown,
    # recovery, interval.
    _MAP_INTENSITY = {
        'active': 'work', 'interval': 'work',
        'rest': 'recovery', 'recovery': 'recovery',
        'warmup': 'excluded', 'cooldown': 'excluded',
    }
    intensidades = [l.get('intensity') for l in considerados if l.get('intensity')]
    usou_intensity = False
    if len(intensidades) == len(considerados) and len(set(intensidades)) > 1:
        # O campo existe em todos e distingue — usar directamente
        if all(i in _MAP_INTENSITY for i in intensidades):
            for l in considerados:
                l['phase'] = _MAP_INTENSITY[l['intensity']]
                l['metodo_classificacao'] = 'FIT intensity'
            usou_intensity = True

    if usou_intensity:
        return lap_stats

    # ── Prioridade 2: inferir pela intensidade medida ────────────────────────
    chave = None
    if any('avg_power' in l for l in considerados):
        chave = 'avg_power'
    elif any('avg_heart_rate' in l for l in considerados):
        chave = 'avg_heart_rate'

    if chave is None:
        for l in considerados:
            l['phase'] = 'work'
            l['metodo_classificacao'] = 'sem sinal'
        return lap_stats

    # Limiar de separação trabalho/recuperação.
    # Numa sessão intervalada a distribuição de intensidades é bimodal (trabalho
    # alto vs recuperação baixo). Usar a mediana × fração é frágil quando há poucos
    # laps ou quando se excluem alguns — a mediana desloca-se e a classificação
    # muda toda. Usamos o ponto médio entre os dois modos (via mediana dos valores
    # acima e abaixo da mediana global), que é estável a exclusões, e recorremos à
    # mediana × fração apenas se a separação bimodal não for clara.
    valores = np.array([l[chave] for l in considerados if chave in l], dtype=float)
    if len(valores) == 0:
        for l in considerados:
            l['phase'] = 'work'
        return lap_stats

    med = float(np.median(valores))
    altos = valores[valores >= med]
    baixos = valores[valores < med]

    if len(altos) > 0 and len(baixos) > 0:
        modo_alto = float(np.median(altos))
        modo_baixo = float(np.median(baixos))
        separacao = (modo_alto - modo_baixo) / modo_alto if modo_alto > 0 else 0
        # Bimodalidade clara → ponto médio entre os dois modos
        if separacao >= 0.25:
            limiar = (modo_alto + modo_baixo) / 2.0
        else:
            limiar = med * frac_mediana
    else:
        limiar = med * frac_mediana

    for l in considerados:
        if chave in l:
            por_intensidade = l[chave] >= limiar
            por_duracao = dur_min <= l['duration'] <= dur_max
            l['phase'] = 'work' if (por_intensidade and por_duracao) else 'recovery'
            l['metodo_classificacao'] = f'auto ({chave.replace("avg_", "")})'
        else:
            l['phase'] = 'recovery'
            l['metodo_classificacao'] = 'sem dados'
    return lap_stats


def identificar_sequencias(lap_stats):
    """
    Identifica pares consecutivos trabalho→recuperação.
    Laps excluídos (aquecimento/arrefecimento) são ignorados: a sequência é
    procurada entre os laps considerados, para que um aquecimento no meio não
    quebre nem crie pares falsos.

    Devolve lista de dicts {'work_lap': ..., 'recovery_lap': ...}.
    """
    validos = [l for l in lap_stats if l.get('phase') in ('work', 'recovery')]
    seqs = []
    for i in range(len(validos) - 1):
        if validos[i].get('phase') == 'work' and validos[i + 1].get('phase') == 'recovery':
            seqs.append({'work_lap': validos[i], 'recovery_lap': validos[i + 1]})
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

def analisar_fit(file_bytes, laps_trabalho_manual=None, laps_excluidos=None,
                 janela_final_s=60, modo_segmentacao='auto',
                 intervalos_trabalho=None, frac_corte=None,
                 min_dur_segmento=45, zerar_potencia_descanso=False,
                 offsets=None):
    """
    Pipeline completo: bytes do FIT → análise fisiológica completa.

    laps_trabalho_manual : lap_numbers marcados como trabalho (sobrepõe a detecção).
    laps_excluidos       : lap_numbers a excluir (aquecimento/arrefecimento).
    janela_final_s       : segundos finais de cada lap usados nas médias.

    modo_segmentacao :
      'auto'      → usa os laps do ficheiro; se não houver, segmenta pelo sinal
      'forcar'    → ignora os laps do ficheiro e segmenta sempre pelo sinal
      'intervalos'→ usa intervalos_trabalho definidos pelo utilizador

    intervalos_trabalho : lista de (inicio_s, fim_s) — blocos de TRABALHO. O que
                          ficar fora torna-se recuperação automaticamente.
    frac_corte          : fracção da intensidade típica de trabalho abaixo da qual
                          se considera recuperação (ex.: 0.5 = 50%). None = auto.
    min_dur_segmento    : duração mínima de um segmento na detecção automática.
    zerar_potencia_descanso : se True, força potência e trabalho (kJ) a zero nos
                          laps de recuperação. Útil quando o ergómetro regista
                          valores residuais durante a pausa (ex.: inércia do
                          volante) que inflacionam as médias e o decoupling.
    offsets             : dict {metrica: segundos} para corrigir métricas fora de
                          sincronia no ficheiro. Aplicado ANTES de qualquer
                          cálculo, para que todas as análises usem os dados
                          alinhados.

    Devolve dict com tudo, ou {'erro': str}.
    """
    fit = ler_fit(file_bytes)
    if 'erro' in fit:
        return fit

    df, laps_info = construir_dataframe(fit)
    if df is None or df.empty:
        return {'erro': "Não foi possível construir a série temporal do ficheiro."}

    # ── Re-segmentação conforme o modo escolhido ─────────────────────────────
    if modo_segmentacao == 'intervalos' and intervalos_trabalho:
        df, laps_info = segmentar_por_intervalos(df, intervalos_trabalho)
    elif modo_segmentacao == 'forcar':
        df, laps_info = _segmentar_sem_laps(
            df, min_dur=min_dur_segmento, frac_corte=frac_corte)
    elif modo_segmentacao == 'auto' and frac_corte is not None:
        # Auto mas com corte explícito: só re-segmenta se os laps forem automáticos
        if laps_info and laps_info[0].get('lap_trigger') in ('auto_segmentado', 'auto_none'):
            df, laps_info = _segmentar_sem_laps(
                df, min_dur=min_dur_segmento, frac_corte=frac_corte)

    colunas = detectar_colunas(df)

    # Correcção de sincronia — aplicada ANTES de tudo o resto, para que as
    # estatísticas, limiares, restauração e decoupling usem os dados alinhados.
    offsets_aplicados = []
    if offsets:
        df, offsets_aplicados = aplicar_offsets(df, colunas, offsets)

    lap_stats = estatisticas_por_lap(df, laps_info, colunas,
                                     janela_final_s=janela_final_s)
    if not lap_stats:
        return {'erro': "Nenhum lap com dados utilizáveis."}

    excluidos = set(laps_excluidos or [])

    # Classificação automática (já ignora os excluídos no cálculo da mediana)
    lap_stats = classificar_laps(lap_stats, laps_excluidos=excluidos)

    # Override manual dos laps de trabalho, preservando as exclusões
    if laps_trabalho_manual is not None:
        manual = set(laps_trabalho_manual)
        for l in lap_stats:
            if l['lap_number'] in excluidos:
                l['phase'] = 'excluded'
            else:
                l['phase'] = 'work' if l['lap_number'] in manual else 'recovery'

    # ── Zerar potência/trabalho nos laps de recuperação (opcional) ────────────
    # Alguns ergómetros registam potência residual durante a pausa (inércia do
    # volante, movimento leve). Isso inflaciona as médias de recuperação e
    # distorce o decoupling. Com esta opção, força-se tudo a zero nesses laps e
    # recalculam-se as estatísticas para reflectir a correcção.
    if zerar_potencia_descanso:
        laps_rec = {l['lap_number'] for l in lap_stats if l.get('phase') == 'recovery'}
        if laps_rec:
            _fases_atuais = {l['lap_number']: l['phase'] for l in lap_stats}
            mask_rec = df['lap_number'].isin(laps_rec)
            for _c in ('power', 'cadence'):
                if _c in colunas and colunas[_c] in df.columns:
                    df.loc[mask_rec, colunas[_c]] = 0.0
            # Recalcular estatísticas com os valores zerados
            lap_stats = estatisticas_por_lap(df, laps_info, colunas,
                                             janela_final_s=janela_final_s)
            for l in lap_stats:
                l['phase'] = _fases_atuais.get(l['lap_number'], 'recovery')
                l['metodo_classificacao'] = 'manual/auto (potência zerada no descanso)'

    restauracao = analisar_restauracao_completa(df, lap_stats, colunas)
    limiares = calcular_limiares_smo2(lap_stats, colunas)
    # Métodos da literatura NIRS (muscleoxygentraining.com / Murias et al.)
    bp_continuo = breakpoint_smo2_continuo(df, colunas, lap_stats)
    lim_dfa1 = limiar_dfa1(lap_stats, colunas)
    estab_smo2 = estabilidade_smo2_intervalos(df, colunas, lap_stats)

    # ── DFA-alpha1 a partir dos RR brutos + HRVT2 + Combo (Murias 2023) ──────
    rr_info = extrair_rr(file_bytes)
    dfa1_serie = pd.DataFrame()
    dfa1_qualidade = None
    hrvt2 = None
    if rr_info is not None:
        rr_lim, dfa1_qualidade = limpar_rr(rr_info['rr_ms'])
        if rr_lim is not None:
            dfa1_serie = calcular_dfa1_serie(rr_lim, rr_info['tempo_s'])
            if len(dfa1_serie) >= 10:
                hrvt2 = calcular_hrvt(dfa1_serie, df_metricas=df, colunas=colunas,
                                      alvo=DFA1_HRVT2, lap_stats=lap_stats,
                                      df_tempo=df)
    combo = combo_limiares(hrvt2, bp_continuo)
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
        'bp_continuo': bp_continuo,
        'limiar_dfa1': lim_dfa1,
        'estabilidade_smo2': estab_smo2,
        'rr_info': rr_info,
        'dfa1_serie': dfa1_serie,
        'dfa1_qualidade': dfa1_qualidade,
        'hrvt2': hrvt2,
        'combo': combo,
        'decoupling': decoupling,
        'fadiga': fadiga,
        'tempo_falha': falha,
        'activity_name': fit.get('activity_name', 'Atividade'),
        'session': fit.get('session', {}),
        'janela_final_s': janela_final_s,
        'laps_excluidos': sorted(excluidos),
        'offsets_aplicados': offsets_aplicados,
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
        'janela_estacionario_s': resultado.get('janela_final_s'),
        'laps_excluidos': ','.join(str(x) for x in resultado.get('laps_excluidos', [])) or None,
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


def parse_intervalos(texto):
    """
    Converte texto de intervalos de trabalho em lista de (inicio_s, fim_s).

    Aceita, uma linha por intervalo (ou separados por ';'):
        10:00-13:00      → mm:ss
        1:05:00-1:08:00  → h:mm:ss
        600-780          → segundos
        10:00 13:00      → separado por espaço

    Devolve (intervalos, erros) onde erros é uma lista de linhas não reconhecidas.
    """
    def _seg(tok):
        tok = tok.strip()
        if not tok:
            return None
        if ':' in tok:
            partes = tok.split(':')
            try:
                partes = [float(p) for p in partes]
            except ValueError:
                return None
            if len(partes) == 2:      # mm:ss
                return partes[0] * 60 + partes[1]
            if len(partes) == 3:      # h:mm:ss
                return partes[0] * 3600 + partes[1] * 60 + partes[2]
            return None
        try:
            return float(tok)
        except ValueError:
            return None

    intervalos, erros = [], []
    if not texto:
        return intervalos, erros

    linhas = []
    for bloco in str(texto).replace(';', '\n').split('\n'):
        if bloco.strip():
            linhas.append(bloco.strip())

    for linha in linhas:
        # separadores aceites: '-', '–', 'a', espaço
        tokens = None
        for sep in ['-', '–', ' a ', '\t']:
            if sep in linha:
                tokens = [t for t in linha.split(sep) if t.strip()]
                break
        if tokens is None:
            tokens = linha.split()
        if len(tokens) != 2:
            erros.append(linha)
            continue
        ini, fim = _seg(tokens[0]), _seg(tokens[1])
        if ini is None or fim is None or fim <= ini:
            erros.append(linha)
            continue
        intervalos.append((ini, fim))

    return intervalos, erros


# ══════════════════════════════════════════════════════════════════════════════
# BREAKPOINT DOUBLE-LINEAR (método muscleoxygentraining.com / Murias et al.)
# ══════════════════════════════════════════════════════════════════════════════

def _ajuste_double_linear(x, y, margem=0.15):
    """
    Ajusta um modelo de duas rectas que se intersectam ("double linear"), o modelo
    usado na literatura de NIRS para localizar o breakpoint de desoxigenação.

    Testa cada ponto candidato como ponto de quebra, ajusta uma recta antes e outra
    depois, e escolhe o que minimiza a soma dos quadrados dos resíduos (SSE).

    margem : fracção dos extremos a ignorar como candidatos (evita quebras logo no
             início ou no fim, que ajustam bem mas não têm significado fisiológico).

    Devolve dict com o ponto de quebra, os declives antes/depois, R² e as rectas
    para desenhar, ou None se não for possível ajustar.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 12:
        return None

    ini = max(4, int(n * margem))
    fim = min(n - 4, int(n * (1 - margem)))
    if fim <= ini:
        return None

    melhor = None
    sse_total = np.sum((y - y.mean()) ** 2)

    for i in range(ini, fim):
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]
        if len(x1) < 3 or len(x2) < 3:
            continue
        # Rectas exigem variação em x
        if np.ptp(x1) < 1e-9 or np.ptp(x2) < 1e-9:
            continue
        try:
            c1 = np.polyfit(x1, y1, 1)
            c2 = np.polyfit(x2, y2, 1)
        except Exception:
            continue
        sse = (np.sum((y1 - np.polyval(c1, x1)) ** 2) +
               np.sum((y2 - np.polyval(c2, x2)) ** 2))
        if melhor is None or sse < melhor['sse']:
            melhor = {'idx': i, 'sse': sse, 'c1': c1, 'c2': c2}

    if melhor is None:
        return None

    c1, c2 = melhor['c1'], melhor['c2']
    # Ponto de intersecção das duas rectas (o breakpoint propriamente dito)
    if abs(c1[0] - c2[0]) > 1e-9:
        x_bp = (c2[1] - c1[1]) / (c1[0] - c2[0])
        # Se a intersecção cair fora do intervalo, usa o ponto candidato
        if not (x.min() <= x_bp <= x.max()):
            x_bp = float(x[melhor['idx']])
    else:
        x_bp = float(x[melhor['idx']])

    r2 = 1 - melhor['sse'] / sse_total if sse_total > 0 else np.nan

    return {
        'breakpoint': float(x_bp),
        'idx': melhor['idx'],
        'slope_antes': float(c1[0]),
        'slope_depois': float(c2[0]),
        'coef_antes': c1.tolist(),
        'coef_depois': c2.tolist(),
        'r2': float(r2),
        'n_pontos': n,
    }


def breakpoint_smo2_continuo(df, colunas, lap_stats=None, janela_media=10,
                             usar_apenas_trabalho=True, so_estado_estacionario=True,
                             janela_estavel_s=90):
    """
    Breakpoint de SmO2 pelo método contínuo (muscleoxygentraining.com):
      • média móvel de `janela_media` segundos do SmO2
      • amostragem a cada `janela_media` segundos (reduz autocorrelação)
      • regressão double-linear SmO2 vs intensidade

    IMPORTANTE — protocolos por degraus vs rampa contínua:
    O método original foi desenhado para uma RAMPA CONTÍNUA (20-30 W/min), onde a
    intensidade sobe suavemente e cada ponto tem uma intensidade distinta. Num
    protocolo por DEGRAUS (ex.: 3 min a potência fixa), a potência é constante
    dentro do degrau mas o SmO2 desce ao longo dele — o que produz "nuvens
    verticais" (o mesmo x com muitos valores de y) que fazem o ajuste seguir o
    ruído em vez do sinal.
    Por isso, com `so_estado_estacionario=True` usa-se apenas a parte final de
    cada lap (os últimos `janela_estavel_s` segundos), onde o SmO2 já estabilizou
    naquela intensidade. Isto torna a estimativa muito mais robusta em degraus.

    NOTA sobre o músculo: assume-se sensor no RECTO FEMORAL, onde o padrão
    esperado é uma descida gradual seguida de ACELERAÇÃO da queda no MLSS.
    (No vasto lateral o padrão é o oposto — descida linear seguida de plateau.)

    Devolve dict com o breakpoint (em W ou bpm), os declives, R², e os pontos
    usados — ou None.
    """
    if 'smo2' not in colunas:
        return None

    col_smo2 = colunas['smo2']
    col_int = colunas.get('power') or colunas.get('heart_rate')
    if col_int is None:
        return None
    unidade = 'W' if col_int == colunas.get('power') else 'bpm'

    d = df[['time_seconds', 'lap_number', col_smo2, col_int]].copy()
    d.columns = ['t', 'lap', 'smo2', 'intensidade']
    d['smo2'] = pd.to_numeric(d['smo2'], errors='coerce')
    d['intensidade'] = pd.to_numeric(d['intensidade'], errors='coerce')

    if usar_apenas_trabalho and lap_stats:
        laps_ok = {l['lap_number'] for l in lap_stats if l.get('phase') == 'work'}
        if laps_ok:
            d = d[d['lap'].isin(laps_ok)]

    # Restringir à parte estável de cada lap (essencial em protocolos por degraus)
    if so_estado_estacionario and janela_estavel_s and janela_estavel_s > 0:
        partes = []
        for _, g in d.groupby('lap'):
            if len(g) == 0:
                continue
            t_fim = g['t'].max()
            sub = g[g['t'] >= t_fim - janela_estavel_s]
            # Se o lap for curto, usa pelo menos a segunda metade
            if len(sub) < 10:
                sub = g.tail(max(10, len(g) // 2))
            partes.append(sub)
        if partes:
            d = pd.concat(partes, ignore_index=True)

    d = d.dropna(subset=['smo2', 'intensidade'])
    if len(d) < 40:
        return None

    # Média móvel + amostragem a cada janela_media segundos
    d = d.sort_values('t').reset_index(drop=True)
    _mp = max(3, janela_media // 2)
    d['smo2_ma'] = d['smo2'].rolling(janela_media, min_periods=_mp).mean()
    d['int_ma'] = d['intensidade'].rolling(janela_media, min_periods=_mp).mean()
    amostra = d.iloc[::janela_media].dropna(subset=['smo2_ma', 'int_ma'])
    if len(amostra) < 12:
        return None

    # Ordenar por intensidade (a relação é SmO2 vs intensidade, não vs tempo)
    amostra = amostra.sort_values('int_ma').reset_index(drop=True)

    res = _ajuste_double_linear(amostra['int_ma'].values, amostra['smo2_ma'].values)
    if res is None:
        return None

    # Interpretação do padrão (recto femoral: espera-se aceleração da queda)
    s1, s2 = res['slope_antes'], res['slope_depois']
    if s2 < s1:
        padrao = 'aceleração da desoxigenação'
        coerente = True
    elif abs(s2) < abs(s1) * 0.5:
        padrao = 'plateau (padrão típico de vasto lateral)'
        coerente = False
    else:
        padrao = 'sem mudança clara de declive'
        coerente = False

    res.update({
        'unidade': unidade,
        'pontos': amostra[['int_ma', 'smo2_ma', 't']].rename(
            columns={'int_ma': 'intensidade', 'smo2_ma': 'smo2', 't': 'tempo_s'}),
        'janela_media': janela_media,
        'padrao': padrao,
        'coerente_recto_femoral': coerente,
    })
    return res


# ══════════════════════════════════════════════════════════════════════════════
# LIMIAR VT1 PELO DFA-alpha1 (método Gronwald / muscleoxygentraining.com)
# ══════════════════════════════════════════════════════════════════════════════

# Valores de referência do DFA-alpha1 (Gronwald et al.; blog muscleoxygentraining)
DFA1_VT1 = 0.75      # aproximação do VT1 / topo da zona 1
DFA1_LIMITE = 0.70   # limite de segurança recomendado para sessões fáceis
DFA1_RUIDO = 0.50    # ruído branco (não correlacionado) — já bem acima do VT1


def limiar_dfa1(lap_stats, colunas, alvos=(0.75, 0.70, 0.50), max_artifacts=5.0):
    """
    Estima a intensidade (potência ou FC) correspondente a valores-alvo de DFA-α1,
    por regressão linear de DFA-α1 vs intensidade nos laps de trabalho.

    Base: o DFA-α1 decresce com a intensidade. Um valor de ~0.75 aproxima o VT1
    (topo da zona 1); 0.5 é ruído branco, já bem acima do VT1. O blog recomenda
    0.7 como limite prático de segurança para sessões de baixa intensidade.

    max_artifacts : se a métrica 'artifacts' existir, laps com mais do que esta
        percentagem de artefactos são EXCLUÍDOS — o DFA-α1 é muito sensível a
        erros de intervalo RR e valores contaminados distorceriam a recta.

    Devolve dict com a recta, os limiares estimados por alvo, e os pontos usados.
    """
    if 'dfa1' not in colunas:
        return None

    intensidade = 'avg_power' if any('avg_power' in l for l in lap_stats) else 'avg_heart_rate'
    unidade = 'W' if intensidade == 'avg_power' else 'bpm'

    usados, descartados = [], []
    for l in lap_stats:
        if l.get('phase') != 'work':
            continue
        if 'avg_dfa1' not in l or intensidade not in l:
            continue
        art = l.get('avg_artifacts')
        if art is not None and max_artifacts is not None and art > max_artifacts:
            descartados.append({'lap': l['lap_number'], 'artifacts': round(art, 1)})
            continue
        usados.append({
            'lap': l['lap_number'],
            'intensidade': l[intensidade],
            'dfa1': l['avg_dfa1'],
            'artifacts': art,
        })

    if len(usados) < 3:
        return {'erro': 'poucos laps válidos', 'descartados': descartados,
                'n_usados': len(usados)}

    pontos = pd.DataFrame(usados).sort_values('intensidade').reset_index(drop=True)
    x = pontos['intensidade'].values.astype(float)
    y = pontos['dfa1'].values.astype(float)

    if np.ptp(x) < 1e-9:
        return {'erro': 'intensidade sem variação', 'descartados': descartados}

    coef = np.polyfit(x, y, 1)
    y_pred = np.polyval(coef, x)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / sst if sst > 0 else np.nan

    # Resolver para cada alvo: intensidade onde DFA-α1 = alvo
    limiares = {}
    for alvo in alvos:
        if abs(coef[0]) > 1e-12:
            xi = (alvo - coef[1]) / coef[0]
            # Só reportar como fiável se cair dentro (ou perto) do intervalo testado
            dentro = x.min() - 0.1 * np.ptp(x) <= xi <= x.max() + 0.1 * np.ptp(x)
            limiares[alvo] = {'intensidade': float(xi), 'extrapolado': not dentro}
        else:
            limiares[alvo] = None

    return {
        'limiares': limiares,
        'coef': coef.tolist(),
        'r2': float(r2),
        'unidade': unidade,
        'pontos': pontos,
        'descartados_artifacts': descartados,
        'n_usados': len(usados),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MLSS POR ESTABILIDADE INTRA-INTERVALO
# (método preferido do blog para intervalos de 5 min a potência constante)
# ══════════════════════════════════════════════════════════════════════════════

def estabilidade_smo2_intervalos(df, colunas, lap_stats, ignorar_inicio_s=60,
                                 limiar_slope=-0.5):
    """
    Analisa, DENTRO de cada lap de trabalho, se o SmO2 estabiliza ou continua a
    descer. É o método que o blog prefere para intervalos a potência constante:

        "procuramos a transição de estabilidade do SmO2 para um declínio
         contínuo ao longo de 5 minutos"

    Interpretação:
      • SmO2 ESTABILIZA no intervalo  → intensidade ABAIXO do MLSS
      • SmO2 DESCE continuamente      → intensidade ACIMA do MLSS
      • O MLSS está entre a intensidade mais alta estável e a mais baixa instável.

    Diferença face ao breakpoint double-linear: aqui não se olha para a curva
    SmO2-vs-potência entre degraus, mas para o COMPORTAMENTO TEMPORAL dentro de
    cada degrau. São métodos complementares.

    ignorar_inicio_s : segundos iniciais a ignorar (a queda inicial é a transição
        da intensidade anterior, não o comportamento estacionário).
    limiar_slope : declive (em % de SmO2 por minuto) abaixo do qual se considera
        que o SmO2 está em declínio contínuo. -0.5 %/min é um valor conservador.

    Devolve dict com a tabela por intervalo e a estimativa de MLSS, ou None.
    """
    if 'smo2' not in colunas:
        return None

    col_smo2 = colunas['smo2']
    col_int = colunas.get('power') or colunas.get('heart_rate')
    if col_int is None:
        return None
    unidade = 'W' if col_int == colunas.get('power') else 'bpm'

    linhas = []
    for l in lap_stats:
        if l.get('phase') != 'work':
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) < 60:
            continue
        t0 = d['time_seconds'].iloc[0]
        d = d[d['time_seconds'] >= t0 + ignorar_inicio_s]
        if len(d) < 30:
            continue

        y = pd.to_numeric(d[col_smo2], errors='coerce')
        t = d['time_seconds']
        mask = y.notna()
        if mask.sum() < 20:
            continue
        y, t = y[mask].values, t[mask].values

        # Declive em % de SmO2 por minuto
        slope_por_s = float(np.polyfit(t, y, 1)[0])
        slope_min = slope_por_s * 60.0
        # R² do ajuste linear (quão consistente é a tendência)
        y_pred = np.polyval(np.polyfit(t, y, 1), t)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum((y - y_pred) ** 2) / sst) if sst > 0 else np.nan

        estavel = slope_min > limiar_slope
        linhas.append({
            'lap': l['lap_number'],
            'intensidade': l.get('avg_power', l.get('avg_heart_rate')),
            'smo2_inicio': round(float(y[0]), 1),
            'smo2_fim': round(float(y[-1]), 1),
            'delta_smo2': round(float(y[-1] - y[0]), 1),
            'slope_pct_min': round(slope_min, 2),
            'r2': round(r2, 2) if not np.isnan(r2) else None,
            'comportamento': 'estável' if estavel else 'declínio contínuo',
            'estavel': estavel,
            'duracao_analisada_s': int(t[-1] - t[0]),
        })

    if len(linhas) < 2:
        return None

    tabela = pd.DataFrame(linhas).sort_values('intensidade').reset_index(drop=True)

    # Duração típica analisada (informativa; a análise corre com qualquer duração,
    # que varia naturalmente conforme a modalidade e o protocolo).
    dur_mediana = float(tabela['duracao_analisada_s'].median())

    # MLSS entre a intensidade mais alta ESTÁVEL e a mais baixa INSTÁVEL
    estaveis = tabela[tabela['estavel']]
    instaveis = tabela[~tabela['estavel']]
    mlss_min = float(estaveis['intensidade'].max()) if len(estaveis) else None
    mlss_max = float(instaveis['intensidade'].min()) if len(instaveis) else None

    if mlss_min is not None and mlss_max is not None and mlss_min < mlss_max:
        estimativa = (mlss_min + mlss_max) / 2.0
        confianca = 'boa'
    elif mlss_min is not None and mlss_max is not None:
        # Sobreposição: há instáveis abaixo de estáveis — resposta inconsistente
        estimativa = None
        confianca = 'inconsistente'
    elif mlss_max is not None:
        estimativa = None
        confianca = 'todos instáveis'   # MLSS abaixo do intervalo testado
    else:
        estimativa = None
        confianca = 'todos estáveis'    # MLSS acima do intervalo testado

    # Aviso: com intervalos curtos, "todos instáveis" é o resultado esperado
    # mesmo abaixo do MLSS — não é uma conclusão fisiológica válida.

    return {
        'tabela': tabela,
        'mlss_entre': (mlss_min, mlss_max),
        'mlss_estimado': estimativa,
        'confianca': confianca,
        'unidade': unidade,
        'limiar_slope': limiar_slope,
        'ignorar_inicio_s': ignorar_inicio_s,
        'duracao_mediana_s': dur_mediana,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO DE SINCRONIZAÇÃO ENTRE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def aplicar_offsets(df, colunas, offsets):
    """
    Desloca métricas no tempo para corrigir desfasamentos de gravação.

    Alguns ficheiros FIT têm métricas fora de sincronia: sensores diferentes
    (potenciómetro, Moxy, cinta cardíaca) podem ter latências distintas, ou o
    gravador pode ter interpolado/repetido valores. O resultado são patamares
    rectos ou picos que não coincidem entre séries.

    offsets : dict {metrica: segundos}. Positivo = empurra para a DIREITA
        (o valor passa a aparecer mais tarde); negativo = para a esquerda.

    NOTA: isto altera o alinhamento dos dados. É uma correcção legítima quando o
    desfasamento é claramente um artefacto de gravação, mas deve ser usada com
    critério — deslocar métricas até "encaixarem" pode criar correlações falsas.

    Devolve (df_corrigido, aplicados) onde `aplicados` lista o que foi deslocado.
    """
    if not offsets:
        return df, []

    d = df.copy()
    aplicados = []

    for metrica, seg in offsets.items():
        if not seg or metrica not in colunas:
            continue
        col = colunas[metrica]
        if col not in d.columns:
            continue
        try:
            seg = int(round(float(seg)))
        except (TypeError, ValueError):
            continue
        if seg == 0:
            continue
        # A série é 1 Hz, por isso deslocar N linhas = deslocar N segundos.
        # shift positivo em pandas move os valores para baixo = mais tarde.
        d[col] = pd.to_numeric(d[col], errors='coerce').shift(seg)
        aplicados.append({'metrica': metrica, 'coluna': col, 'segundos': seg})

    return d, aplicados


def sugerir_offset(df, colunas, metrica_ref, metrica_alvo, max_lag=30):
    """
    Sugere o deslocamento que maximiza a correlação entre duas métricas.

    Usa correlação cruzada: testa deslocamentos de -max_lag a +max_lag segundos e
    devolve o que maximiza |r| entre as duas séries (após diferenciação, para
    alinhar as MUDANÇAS e não os níveis absolutos).

    Útil como ponto de partida, mas confirma sempre visualmente — o máximo de
    correlação nem sempre corresponde ao alinhamento fisiologicamente correcto,
    sobretudo entre métricas com cinéticas diferentes (ex.: potência muda de
    imediato, SmO2 responde com 20-40s de atraso REAL, que não é para corrigir).

    Devolve dict {'offset': int, 'r': float, 'curva': DataFrame} ou None.
    """
    if metrica_ref not in colunas or metrica_alvo not in colunas:
        return None

    a = pd.to_numeric(df[colunas[metrica_ref]], errors='coerce')
    b = pd.to_numeric(df[colunas[metrica_alvo]], errors='coerce')
    if a.notna().sum() < 60 or b.notna().sum() < 60:
        return None

    # Diferenciar para focar nas transições, não nos níveis
    da = a.diff().fillna(0.0)
    db = b.diff().fillna(0.0)

    linhas = []
    for lag in range(-max_lag, max_lag + 1):
        bb = db.shift(lag)
        m = da.notna() & bb.notna()
        if m.sum() < 60:
            continue
        x, y = da[m].values, bb[m].values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        linhas.append({'offset': lag, 'r': r})

    if not linhas:
        return None

    curva = pd.DataFrame(linhas)
    melhor = curva.loc[curva['r'].abs().idxmax()]
    return {
        'offset': int(melhor['offset']),
        'r': float(melhor['r']),
        'curva': curva,
    }


def sugerir_offset_por_laps(df, colunas, metrica, lap_stats, max_lag=60,
                            direcao='ambas'):
    """
    Sugere o deslocamento que melhor alinha uma métrica com as FRONTEIRAS DOS LAPS
    de trabalho — em vez de a alinhar com outra métrica.

    Porquê: a correlação cruzada entre duas métricas pode falhar quando ambas
    estão ruidosas ou têm cinéticas diferentes. As transições trabalho↔recuperação
    são marcadores temporais muito mais nítidos: no início de cada lap de trabalho
    a intensidade sobe de forma abrupta, e é esse degrau que se procura alinhar.

    Método: constrói um sinal de referência quadrado (1 durante os laps de
    trabalho, 0 nos restantes), diferencia-o para marcar as transições, e testa
    deslocamentos da métrica até maximizar a correlação com essas transições.

    direcao : 'ambas' (−max_lag a +max_lag), 'frente' (só positivos, empurra para
        a direita/mais tarde) ou 'tras' (só negativos, para a esquerda/mais cedo).

    Devolve dict {'offset', 'r', 'curva', 'direcao'} ou None.
    """
    if metrica not in colunas or not lap_stats:
        return None

    serie = pd.to_numeric(df[colunas[metrica]], errors='coerce')
    if serie.notna().sum() < 60:
        return None

    laps_work = {l['lap_number'] for l in lap_stats if l.get('phase') == 'work'}
    if not laps_work:
        return None

    # Sinal quadrado de referência: 1 no trabalho, 0 fora
    ref = df['lap_number'].isin(laps_work).astype(float)
    if ref.sum() < 30 or ref.sum() == len(ref):
        return None

    # Diferenciar: marca as transições (subidas e descidas)
    d_ref = ref.diff().fillna(0.0)
    d_met = serie.diff().fillna(0.0)

    if direcao == 'frente':
        lags = range(0, max_lag + 1)
    elif direcao == 'tras':
        lags = range(-max_lag, 1)
    else:
        lags = range(-max_lag, max_lag + 1)

    linhas = []
    for lag in lags:
        mm = d_met.shift(lag)
        m = d_ref.notna() & mm.notna()
        if m.sum() < 60:
            continue
        x, y = d_ref[m].values, mm[m].values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        linhas.append({'offset': lag, 'r': float(np.corrcoef(x, y)[0, 1])})

    if not linhas:
        return None

    curva = pd.DataFrame(linhas)
    # Queremos correlação POSITIVA: a métrica deve subir quando o trabalho começa
    melhor = curva.loc[curva['r'].idxmax()]
    return {
        'offset': int(melhor['offset']),
        'r': float(melhor['r']),
        'curva': curva,
        'direcao': direcao,
        'n_laps_trabalho': len(laps_work),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DFA-alpha1 A PARTIR DOS INTERVALOS RR
# Método: Peng et al. (algoritmo DFA), com os parâmetros do estudo
# Fleitas-Paniagua/Murias 2023 (JSCR) e do Kubios: janela alpha1 = 4-16 batimentos,
# janelas móveis de 2 min recalculadas a cada 5 s.
# ══════════════════════════════════════════════════════════════════════════════

def extrair_rr(file_bytes):
    """
    Extrai os intervalos RR brutos das mensagens 'hrv' do ficheiro FIT.

    Muitos gravadores (Garmin, apps com Polar H10) guardam os RR além das
    métricas por segundo. Tê-los permite recalcular o DFA-alpha1 em vez de
    depender do valor pré-calculado pelo sensor.

    Devolve dict {'rr_ms': array, 'tempo_s': array (tempo cumulativo), 'n': int}
    ou None se o ficheiro não tiver RR.
    """
    if not _TEM_FITDECODE:
        return None
    rr = []
    try:
        with fitdecode.FitReader(io.BytesIO(file_bytes)) as fit:
            for frame in fit:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                if frame.name != 'hrv':
                    continue
                for f in frame.fields:
                    if f.name != 'time' or f.value is None:
                        continue
                    v = f.value
                    if isinstance(v, (list, tuple)):
                        rr.extend([x for x in v if x is not None])
                    else:
                        rr.append(v)
    except Exception:
        return None

    if len(rr) < 100:
        return None

    rr = np.array([float(x) for x in rr], dtype=float)
    # O FIT guarda os RR em segundos; converter para ms se necessário
    rr_ms = rr * 1000.0 if np.nanmax(rr) < 10 else rr
    tempo = np.cumsum(rr_ms) / 1000.0  # segundos desde o início
    return {'rr_ms': rr_ms, 'tempo_s': tempo, 'n': len(rr_ms)}


def limpar_rr(rr_ms, low=300, high=2000, malik_pct=20):
    """
    Pré-processamento dos intervalos RR antes do DFA.

    Réplica do pipeline descrito no artigo Kubios-vs-Python:
      1. remove_outliers  → fora de [low, high] ms
      2. remove_ectopic_beats (regra de Malik) → um RR difere >20% do anterior
      3. interpolação linear dos removidos

    ATENÇÃO: o artigo mostra que o método de correcção de artefactos é a maior
    fonte de divergência face ao Kubios (R²=0.85, viés ~20% abaixo de alpha1=0.75).
    Isto é uma aproximação razoável, não uma réplica exacta do Kubios.

    Devolve (rr_limpo, info) com a percentagem de artefactos corrigidos.
    """
    x = np.array(rr_ms, dtype=float)
    n0 = len(x)
    mask_bad = (x < low) | (x > high) | ~np.isfinite(x)

    # Regra de Malik: comparar cada intervalo com o anterior válido
    for i in range(1, len(x)):
        if mask_bad[i] or mask_bad[i - 1]:
            continue
        if abs(x[i] - x[i - 1]) > (malik_pct / 100.0) * x[i - 1]:
            mask_bad[i] = True

    n_bad = int(mask_bad.sum())
    xc = x.copy()
    xc[mask_bad] = np.nan
    # Interpolação linear
    idx = np.arange(len(xc))
    ok = ~np.isnan(xc)
    if ok.sum() < 10:
        return None, {'pct_artefactos': 100.0, 'n_corrigidos': n_bad, 'n_total': n0}
    xc = np.interp(idx, idx[ok], xc[ok])

    return xc, {
        'pct_artefactos': round(n_bad / n0 * 100, 2),
        'n_corrigidos': n_bad,
        'n_total': n0,
    }


def _dfa_alpha(serie, n_min=4, n_max=16):
    """
    Detrended Fluctuation Analysis — expoente de escala.

    Implementação do algoritmo clássico (Peng et al.), com a mesma estrutura do
    dokato/dfa mas com as escalas do alpha1 usado em HRV:

        n_min=4, n_max=16 batimentos  ← alpha1 (curto prazo)

    NOTA sobre o dokato/dfa: o seu default é scale_lim=[5,9], ou seja escalas de
    2^5=32 a 2^9=512 amostras. Isso NÃO é o alpha1 — corresponde a escalas muito
    maiores (alpha2 e além). Usar o default daria um valor sem relação com os
    limiares de intensidade. Daqui a escolha explícita de 4-16 batimentos.

    Passos:
      1. y = soma cumulativa do sinal centrado
      2. para cada escala n: dividir y em janelas de n pontos, remover a
         tendência linear de cada uma, calcular o RMS
      3. F(n) = RMS médio; alpha = declive de log F(n) vs log n
    """
    x = np.asarray(serie, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < n_max * 4:
        return None

    y = np.cumsum(x - np.mean(x))
    escalas = np.arange(n_min, n_max + 1)
    flut = []
    escalas_ok = []

    for n in escalas:
        n_janelas = len(y) // n
        if n_janelas < 2:
            continue
        seg = y[:n_janelas * n].reshape(n_janelas, n)
        eixo = np.arange(n)
        rms = np.empty(n_janelas)
        for i in range(n_janelas):
            coef = np.polyfit(eixo, seg[i], 1)
            rms[i] = np.sqrt(np.mean((seg[i] - np.polyval(coef, eixo)) ** 2))
        f = np.sqrt(np.mean(rms ** 2))
        if f > 0:
            flut.append(f)
            escalas_ok.append(n)

    if len(flut) < 3:
        return None

    coef = np.polyfit(np.log(escalas_ok), np.log(flut), 1)
    return float(coef[0])


def calcular_dfa1_serie(rr_ms, tempo_s, janela_s=120, passo_s=5,
                        n_min=4, n_max=16):
    """
    Calcula o DFA-alpha1 ao longo do tempo, com janelas móveis.

    Parâmetros do estudo Murias 2023:
      "the DFA a1 was calculated over time using 2 min HRV measurement windows
       with a recalculation every 5 s"

    Devolve DataFrame com tempo_s, dfa1, n_batimentos, fc_media.
    """
    rr = np.asarray(rr_ms, dtype=float)
    t = np.asarray(tempo_s, dtype=float)
    if len(rr) < 50:
        return pd.DataFrame()

    linhas = []
    t_fim = t[-1]
    t_ini = janela_s
    for centro in np.arange(t_ini, t_fim + 0.001, passo_s):
        m = (t > centro - janela_s) & (t <= centro)
        if m.sum() < n_max * 4:
            continue
        seg = rr[m]
        a1 = _dfa_alpha(seg, n_min=n_min, n_max=n_max)
        if a1 is None:
            continue
        linhas.append({
            'tempo_s': float(centro),
            'dfa1': round(a1, 4),
            'n_batimentos': int(m.sum()),
            'fc_media': round(60000.0 / np.mean(seg), 1),
        })

    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# HRVT2 — segundo limiar pelo DFA-alpha1 (Fleitas-Paniagua/Murias 2023)
# ══════════════════════════════════════════════════════════════════════════════

# Valores de referência do DFA-alpha1 na literatura
DFA1_HRVT2 = 0.50   # HRVT2 ≈ RCP / MLSS / limiar ALTO (Murias 2023, Rogers et al.)
DFA1_HRVT1 = 0.75   # aproximação do VT1 / limiar BAIXO (Gronwald, Rogers 2020)


def calcular_hrvt(serie_dfa1, df_metricas=None, colunas=None, alvo=0.50,
                  janela_ajuste=(0.4, 1.0), lap_stats=None, df_tempo=None,
                  so_trabalho=True):
    """
    Estima o limiar associado a um valor-alvo de DFA-alpha1, pelo método do estudo:

        "The relationship generally showed a reverse sigmoidal curve, with a
         stable area above 1.0 at low work rates, a rapid, near linear drop
         reaching below 0.5 at higher intensity, then flattening without major
         change. A linear regression line was drawn through the appropriate
         section with the HRVT2 defined as the RI time or HR where DFA a1
         equaled 0.5"

    Ou seja: a regressão é feita SÓ na secção de queda quase-linear, não em toda
    a curva (que é sigmoidal invertida e achataria a estimativa).

    alvo=0.50 → HRVT2 (limiar ALTO, ≈ RCP/MLSS)
    alvo=0.75 → aproximação do VT1 (limiar baixo)

    janela_ajuste : intervalo de alpha1 considerado "secção linear de queda".

    df_metricas/colunas : opcionais; se fornecidos, converte o tempo do limiar
        na potência correspondente.

    Devolve dict com o limiar em FC (e potência, se disponível), a recta, R² e
    os pontos usados — ou {'erro': ...}.
    """
    if serie_dfa1 is None or len(serie_dfa1) < 10:
        return {'erro': 'série DFA-α1 insuficiente'}

    s = serie_dfa1.dropna(subset=['dfa1', 'fc_media']).copy()

    # Em protocolos INTERVALADOS, as recuperações produzem pontos com FC baixa e
    # alpha1 alto que não pertencem à curva intensidade→alpha1. O método original
    # foi desenhado para rampas contínuas; restringir aos períodos de trabalho
    # recupera a relação monotónica que o ajuste pressupõe.
    n_antes = len(s)
    if so_trabalho and lap_stats and df_tempo is not None:
        janelas = [(l['_t_ini'], l['_t_fim']) for l in lap_stats
                   if l.get('phase') == 'work' and '_t_ini' in l]
        if janelas:
            m = pd.Series(False, index=s.index)
            for t0, t1 in janelas:
                m |= (s['tempo_s'] >= t0) & (s['tempo_s'] <= t1)
            if m.sum() >= 10:
                s = s[m].copy()

    lo, hi = janela_ajuste
    linear = s[(s['dfa1'] >= lo) & (s['dfa1'] <= hi)]

    # Se a janela não apanhar pontos suficientes, alargar progressivamente
    if len(linear) < 6:
        for margem in (0.1, 0.2, 0.3):
            linear = s[(s['dfa1'] >= lo - margem) & (s['dfa1'] <= hi + margem)]
            if len(linear) >= 6:
                break
    if len(linear) < 6:
        return {'erro': f'poucos pontos na secção linear (n={len(linear)})',
                'serie': s}

    x = linear['fc_media'].values.astype(float)
    y = linear['dfa1'].values.astype(float)
    if np.ptp(x) < 1e-9:
        return {'erro': 'FC sem variação na secção linear', 'serie': s}

    coef = np.polyfit(x, y, 1)
    y_pred = np.polyval(coef, x)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - y_pred) ** 2) / sst) if sst > 0 else np.nan

    if abs(coef[0]) < 1e-12:
        return {'erro': 'declive nulo', 'serie': s}

    fc_limiar = float((alvo - coef[1]) / coef[0])
    extrapolado = not (x.min() - 0.1 * np.ptp(x) <= fc_limiar <= x.max() + 0.1 * np.ptp(x))

    # Converter FC em potência, se houver dados
    pot_limiar = None
    if df_metricas is not None and colunas and 'power' in colunas and 'heart_rate' in colunas:
        try:
            dm = df_metricas[[colunas['heart_rate'], colunas['power']]].copy()
            dm.columns = ['fc', 'pot']
            dm = dm.apply(pd.to_numeric, errors='coerce').dropna()
            dm = dm[dm['pot'] > 0]
            if len(dm) > 30 and np.ptp(dm['fc'].values) > 5:
                cfp = np.polyfit(dm['fc'].values, dm['pot'].values, 1)
                pot_limiar = float(np.polyval(cfp, fc_limiar))
        except Exception:
            pot_limiar = None

    # Tempo em que o alpha1 cruza o alvo (primeira travessia descendente)
    tempo_limiar = None
    abaixo = s[s['dfa1'] <= alvo]
    if len(abaixo) > 0:
        tempo_limiar = float(abaixo['tempo_s'].iloc[0])

    # ── Diagnóstico de fiabilidade ───────────────────────────────────────────
    # O método pressupõe que o alpha1 atravessa o alvo de forma consistente
    # durante o teste. Se isso não acontece, a "estimativa" é uma extrapolação
    # da recta muito para lá dos dados observados — e não deve ser usada.
    n_abaixo = int((s['dfa1'] <= alvo).sum())
    pct_abaixo = n_abaixo / len(s) * 100 if len(s) else 0.0
    avisos = []
    if pct_abaixo < 5:
        avisos.append(
            f"o α1 só desceu abaixo de {alvo} em {n_abaixo}/{len(s)} janelas "
            f"({pct_abaixo:.1f}%) — o teste pode não ter atingido o limiar")
    if r2 < 0.5:
        avisos.append(f"ajuste fraco na secção linear (R²={r2:.2f})")
    if extrapolado:
        avisos.append("o valor está extrapolado para fora do intervalo medido")
    fiavel = (not avisos)

    return {
        'alvo': alvo,
        'fiavel': fiavel,
        'avisos': avisos,
        'pct_abaixo_alvo': round(pct_abaixo, 1),
        'fc': fc_limiar,
        'potencia': pot_limiar,
        'tempo_s': tempo_limiar,
        'coef': coef.tolist(),
        'r2': r2,
        'n_pontos': len(linear),
        'extrapolado': extrapolado,
        'pontos_linear': linear,
        'serie': s,
        'janela_ajuste': (lo, hi),
        'so_trabalho': bool(so_trabalho and len(s) < n_antes),
        'n_serie_usada': len(s),
        'n_serie_total': n_antes,
    }


def combo_limiares(hrvt2, bp_nirs, tolerancia_pct=15):
    """
    Combo HRVT2 + NIRS breakpoint (Fleitas-Paniagua, Murias et al., JSCR 2023).

    O estudo mostrou que a média das duas estimativas tem menor viés e limites de
    concordância mais estreitos face ao padrão-ouro (RCP) do que qualquer uma
    isolada — porque derivam de subsistemas fisiológicos diferentes e os erros
    tendem a cancelar-se:

        HRVT2 sozinho : 4/19 casos com erro ≥10 bpm (21%)
        NIRS sozinho  : 5/16 (31%)
        Combo         : 3/21 (14%)

    Vantagem adicional: se um dos métodos falhar tecnicamente (artefactos no HRV,
    sinal fraco no NIRS), o outro ainda dá um resultado utilizável.

    Devolve dict com a estimativa combinada, a divergência entre métodos e um
    aviso quando essa divergência é grande.
    """
    v_hrv = None
    hrv_descartado = False
    if hrvt2 and 'erro' not in hrvt2 and hrvt2.get('potencia') is not None:
        if hrvt2.get('fiavel', True):
            v_hrv = float(hrvt2['potencia'])
        else:
            # HRVT2 pouco fiável: não entra no combo, para não contaminar a
            # estimativa. O estudo assume que ambos os métodos são válidos.
            hrv_descartado = True
    v_nirs = float(bp_nirs['breakpoint']) if bp_nirs else None

    disponiveis = [v for v in (v_hrv, v_nirs) if v is not None]
    if not disponiveis:
        return None

    combo = float(np.mean(disponiveis))
    divergencia = abs(v_hrv - v_nirs) if len(disponiveis) == 2 else None
    div_pct = (divergencia / combo * 100) if divergencia is not None and combo > 0 else None

    if len(disponiveis) == 1:
        estado = 'metodo_unico'
    elif div_pct is not None and div_pct > tolerancia_pct:
        estado = 'divergente'
    else:
        estado = 'concordante'

    return {
        'combo': combo,
        'hrvt2': v_hrv,
        'nirs': v_nirs,
        'divergencia': divergencia,
        'divergencia_pct': div_pct,
        'estado': estado,
        'hrv_descartado': hrv_descartado,
        'n_metodos': len(disponiveis),
        'tolerancia_pct': tolerancia_pct,
    }
