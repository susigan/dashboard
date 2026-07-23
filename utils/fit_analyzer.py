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

    Faz UMA ÚNICA passagem pelo ficheiro inteiro, extraindo tudo o que é
    preciso nas duas fases (record/session/lap para a Fase 1, e as mensagens
    'hrv' com os RR brutos para a Fase 2) — evitar uma segunda passagem
    completa é importante porque o fitdecode tem de descodificar TODAS as
    mensagens do ficheiro (incluindo as 'hrv', que em gravações com RR
    batimento-a-batimento podem ser milhares), mesmo que uma função só
    aproveite uma parte delas. Ler duas vezes duplicava esse custo.

    Devolve dict:
      {'records': [...], 'session': {...}, 'laps': [...], 'activity_name': str,
       'rr_bruto': [...]}
    ou {'erro': str} em caso de falha.
    """
    if not _TEM_FITDECODE:
        return {'erro': "Biblioteca 'fitdecode' não instalada. "
                        "Adiciona 'fitdecode' ao requirements.txt."}

    records_data, session_data, lap_data, rr_bruto = [], {}, [], []
    activity_name = None

    try:
        with fitdecode.FitReader(io.BytesIO(file_bytes), check_crc=fitdecode.CrcCheck.DISABLED) as fit:
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

                elif frame.name == 'hrv':
                    for f in frame.fields:
                        if f.name != 'time' or f.value is None:
                            continue
                        v = f.value
                        if isinstance(v, (list, tuple)):
                            rr_bruto.extend([x for x in v if x is not None])
                        else:
                            rr_bruto.append(v)
    except Exception as e:
        return {'erro': f"Erro ao ler o ficheiro FIT: {e}"}

    if not records_data:
        return {'erro': "O ficheiro não contém registos de dados (records)."}

    return {
        'records': records_data,
        'session': session_data,
        'laps': lap_data,
        'activity_name': activity_name or 'Atividade',
        'rr_bruto': rr_bruto,
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
                    # max/min também sobre o estado estacionário — útil quando o que
                    # interessa é "onde estabilizou", não o pico/vale do lap inteiro
                    # (que pode ser ruído de medição ou um transiente de início de lap).
                    # Ver classificar_limitador_smo2().
                    s[f'max_{metrica}_est'] = float(vals_est.max())
                    s[f'min_{metrica}_est'] = float(vals_est.min())
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


def calcular_limiares_smo2(lap_stats, colunas, df=None, protocolo=None,
                           n_bins=8):
    """Wrapper que adapta o cálculo ao protocolo (ver _limiares_smo2_laps)."""
    # Numa RAMPA não há laps de trabalho: divide-se a rampa em faixas de
    # intensidade e usa-se a média de SmO2 em cada faixa como "ponto",
    # replicando a estrutura que os métodos de breakpoint esperam.
    if protocolo in ('rampa', 'continuo') and df is not None:
        return _limiares_smo2_bins(df, colunas, n_bins=n_bins)
    return _limiares_smo2_laps(lap_stats, colunas)


def _limiares_smo2_bins(df, colunas, n_bins=8):
    """
    Limiares de SmO2 para protocolos SEM laps de trabalho (rampa contínua).

    Divide o intervalo de intensidade em `n_bins` faixas de igual largura e
    calcula o SmO2 médio de cada uma. Isto produz uma relação
    intensidade→SmO2 comparável à que se obtém com degraus, permitindo aplicar
    os mesmos três métodos de detecção de breakpoint.
    """
    if 'smo2' not in colunas:
        return None
    col_int = colunas.get('power') or colunas.get('heart_rate')
    if col_int is None:
        return None
    unidade = 'W' if col_int == colunas.get('power') else 'bpm'

    d = df[[col_int, colunas['smo2']]].copy()
    d.columns = ['intensidade', 'smo2']
    d = d.apply(pd.to_numeric, errors='coerce').dropna()
    # Ignorar intensidades muito baixas (arranque/aquecimento)
    if len(d) > 60:
        lim_baixo = d['intensidade'].quantile(0.05)
        d = d[d['intensidade'] >= lim_baixo]
    if len(d) < 60 or np.ptp(d['intensidade'].values) < 1e-6:
        return None

    bins = np.linspace(d['intensidade'].min(), d['intensidade'].max(), n_bins + 1)
    d['bin'] = pd.cut(d['intensidade'], bins, include_lowest=True)
    g = d.groupby('bin', observed=True).agg(
        intensidade=('intensidade', 'mean'), smo2=('smo2', 'mean'),
        n=('smo2', 'size')).reset_index(drop=True)
    g = g[g['n'] >= 10].dropna()
    if len(g) < 3:
        return None
    g['lap'] = range(1, len(g) + 1)
    pontos = g[['lap', 'intensidade', 'smo2']].sort_values('intensidade').reset_index(drop=True)

    x = pontos['intensidade'].values.astype(float)
    y = pontos['smo2'].values.astype(float)
    dmax = _dmax_smo2(x, y)
    quebra = _quebra_inclinacao(x, y)
    defl = _deflexao_smo2(x, y)
    validos = [v for v in (dmax, quebra, defl) if v is not None]

    return {
        'dmax': dmax, 'quebra': quebra, 'deflexao': defl,
        'media': float(np.mean(validos)) if validos else None,
        'pontos': pontos, 'unidade': unidade,
        'metodo': f'faixas de intensidade (n={len(pontos)})',
    }


def _limiares_smo2_laps(lap_stats, colunas):
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

def preparar_fit(file_bytes, laps_trabalho_manual=None, laps_excluidos=None,
                 janela_final_s=60, modo_segmentacao='auto',
                 intervalos_trabalho=None, frac_corte=None,
                 min_dur_segmento=45, zerar_potencia_descanso=False,
                 offsets=None, raw=None):
    """
    FASE 1 — preparação dos dados. Leve e rápida.

    Faz apenas o necessário para o utilizador poder VER e CORRIGIR os dados:
    lê o ficheiro, segmenta em laps, classifica trabalho/recuperação/aquecimento,
    aplica correcções de sincronia e o zeramento da potência em recuperação.

    NÃO faz nenhuma análise fisiológica. Isso é a fase 2 (analisar_completo),
    executada só depois de o utilizador confirmar que os laps e o alinhamento
    das métricas estão correctos — caso contrário estaríamos a calcular limiares
    sobre dados que ainda vão ser corrigidos.

    raw : resultado já calculado de ler_fit(file_bytes) — passa isto quando só
        os laps/definições mudaram (ex.: o utilizador corrigiu um lap e clicou
        "Aplicar") para NÃO reanalisar o binário do ficheiro outra vez. O
        parsing do FIT (fitdecode) é a parte mais cara desta fase; a
        segmentação/classificação de laps é barata em comparação. Se None,
        lê o ficheiro do zero (comportamento antigo, backward-compatible).

    Devolve dict com df, colunas, lap_stats e metadados, ou {'erro': str}.
    """
    fit = raw if raw is not None else ler_fit(file_bytes)
    if 'erro' in fit:
        return fit

    df, laps_info = construir_dataframe(fit)
    if df is None or df.empty:
        return {'erro': "Não foi possível construir a série temporal do ficheiro."}

    # Re-segmentação conforme o modo escolhido
    if modo_segmentacao == 'intervalos' and intervalos_trabalho:
        df, laps_info = segmentar_por_intervalos(df, intervalos_trabalho)
    elif modo_segmentacao == 'forcar':
        df, laps_info = _segmentar_sem_laps(
            df, min_dur=min_dur_segmento, frac_corte=frac_corte)
    elif modo_segmentacao == 'auto' and frac_corte is not None:
        if laps_info and laps_info[0].get('lap_trigger') in ('auto_segmentado', 'auto_none'):
            df, laps_info = _segmentar_sem_laps(
                df, min_dur=min_dur_segmento, frac_corte=frac_corte)

    colunas = detectar_colunas(df)

    # Derivar HHb/O2Hb a partir de SmO2 e THb. Fica disponível para os gráficos
    # e para todas as análises — é a métrica que a literatura NIRS usa.
    df, colunas = derivar_hhb(df, colunas)

    # Correcção de sincronia — antes de calcular estatísticas
    offsets_aplicados = []
    if offsets:
        df, offsets_aplicados = aplicar_offsets(df, colunas, offsets)

    lap_stats = estatisticas_por_lap(df, laps_info, colunas,
                                     janela_final_s=janela_final_s)
    if not lap_stats:
        return {'erro': "Nenhum lap com dados utilizáveis."}

    excluidos = set(laps_excluidos or [])
    lap_stats = classificar_laps(lap_stats, laps_excluidos=excluidos)

    if laps_trabalho_manual is not None:
        manual = set(laps_trabalho_manual)
        for l in lap_stats:
            if l['lap_number'] in excluidos:
                l['phase'] = 'excluded'
            else:
                l['phase'] = 'work' if l['lap_number'] in manual else 'recovery'

    # Zerar potência nos laps de recuperação (opcional)
    if zerar_potencia_descanso:
        laps_rec = {l['lap_number'] for l in lap_stats if l.get('phase') == 'recovery'}
        if laps_rec:
            _fases = {l['lap_number']: l['phase'] for l in lap_stats}
            mask_rec = df['lap_number'].isin(laps_rec)
            for _c in ('power', 'cadence'):
                if _c in colunas and colunas[_c] in df.columns:
                    df.loc[mask_rec, colunas[_c]] = 0.0
            lap_stats = estatisticas_por_lap(df, laps_info, colunas,
                                             janela_final_s=janela_final_s)
            for l in lap_stats:
                l['phase'] = _fases.get(l['lap_number'], 'recovery')
                l['metodo_classificacao'] = 'manual/auto (potência zerada no descanso)'

    return {
        'df': df,
        'colunas': colunas,
        'lap_stats': lap_stats,
        'laps_info': laps_info,
        'file_bytes': file_bytes,
        'rr_bruto': fit.get('rr_bruto', []),
        'activity_name': fit.get('activity_name', 'Atividade'),
        'session': fit.get('session', {}),
        'janela_final_s': janela_final_s,
        'laps_excluidos': sorted(excluidos),
        'offsets_aplicados': offsets_aplicados,
        'zerar_potencia_descanso': zerar_potencia_descanso,
        'duracao_total_s': float(df['time_seconds'].iloc[-1]) if len(df) else 0.0,
        'data_sessao': (df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')
                        if 'timestamp' in df.columns and len(df) else None),
    }


def analisar_completo(prep, metodo_detrend='local', comparar_detrend=False, lam_sp=500):
    """
    FASE 2 — análises fisiológicas, sobre os dados JÁ CORRIGIDOS pelo utilizador.

    Recebe o resultado de preparar_fit() e corre todas as análises: detecção de
    protocolo, cinética de restauração, limiares de SmO₂, DFA-α1 e HRVTs,
    decoupling, fadiga, durabilidade e avaliação de fiabilidade.

    A detecção de protocolo acontece AQUI (não na preparação), porque depende da
    classificação final dos laps — que o utilizador pode ter corrigido.

    metodo_detrend : 'local' (default, o que já estava implementado) ou
        'sp_global' (Smoothness Priors λ=lam_sp aplicado ao tacograma inteiro,
        estilo Kubios — ver detrend_sp()). Define qual dos dois é o resultado
        PRINCIPAL (hrvt2, hrvt1c, hrvt2_submax, dfa1_serie).

    comparar_detrend : se True (default), corre TAMBÉM o método alternativo e
        guarda os resultados em '..._alt' — permite comparar os dois lado a
        lado sem correr a análise duas vezes. Custo extra: mais um DFA sobre
        a mesma série (poucos segundos).
    """
    if not prep or 'erro' in prep:
        return prep

    df = prep['df']
    colunas = prep['colunas']
    lap_stats = prep['lap_stats']
    file_bytes = prep['file_bytes']
    janela_final_s = prep.get('janela_final_s', 60)

    # Protocolo detectado a partir dos laps FINAIS (após correcção do utilizador)
    protocolo = detectar_protocolo(df, colunas, lap_stats)
    _tipo = protocolo.get('tipo')

    restauracao = analisar_restauracao_completa(df, lap_stats, colunas)
    limiares = calcular_limiares_smo2(lap_stats, colunas, df=df, protocolo=_tipo)
    bp_continuo = breakpoint_smo2_continuo(df, colunas, lap_stats, protocolo=_tipo,
                                           sinal='smo2')
    # O mesmo breakpoint calculado sobre o HHb — é a métrica dos estudos, e
    # serve de verificação cruzada ao resultado obtido com SmO2.
    bp_hhb = (breakpoint_smo2_continuo(df, colunas, lap_stats, protocolo=_tipo,
                                       sinal='hhb') if 'hhb' in colunas else None)
    # LT1+LT2 (2 breakpoints no mesmo sinal SmO2) — Andri Feldmann, fórum Moxy.
    # LT2 aqui é um segundo ponto de vista, independente, sobre o mesmo
    # território do HRVT2/RCP (via DFA-α1); LT1 é novo (não tínhamos nenhuma
    # estimativa de VT1/LT1 a partir do SmO2 antes disto).
    bp_lt1_lt2 = breakpoint_smo2_lt1_lt2(df, colunas, lap_stats, protocolo=_tipo,
                                        sinal='smo2')
    lim_dfa1 = limiar_dfa1(lap_stats, colunas)
    estab_smo2 = (estabilidade_smo2_intervalos(df, colunas, lap_stats)
                  if _tipo in ('degraus', 'intervalos') else None)
    # Método dos intervalos longos (o que a literatura considera mais fiável
    # que os breakpoints por rampa, pelo erro >10 W destes)
    mlss_longos = (mlss_intervalos_longos(df, colunas, lap_stats)
                   if _tipo in ('degraus', 'intervalos') else None)
    decoupling = calcular_decoupling(lap_stats)
    fadiga = classificar_fadiga(restauracao, decoupling)
    falha = tempo_ate_falha(lap_stats)

    # DFA-alpha1 a partir dos RR brutos + HRVTs + Combo
    rr_info = extrair_rr(file_bytes, rr_bruto=prep.get('rr_bruto'))
    dfa1_serie = pd.DataFrame()
    dfa1_qualidade = None
    hrvt2 = None
    hrvt1c = None
    hrvt2_sub = None

    metodo_alt = 'sp_global' if metodo_detrend == 'local' else 'local'
    dfa1_serie_alt = pd.DataFrame()
    hrvt2_alt = None
    hrvt1c_alt = None
    hrvt2_sub_alt = None

    if rr_info is not None:
        rr_lim, dfa1_qualidade = limpar_rr(rr_info['rr_ms'])
        if rr_lim is not None:
            # Só faz sentido "respeitar fases" em protocolos com laps curtos
            # de trabalho/descanso; numa rampa ou esforço contínuo há
            # tipicamente um único lap efectivo e a janela de 120s nunca
            # atravessa fronteira nenhuma que importe.
            _respeitar_fases = _tipo in ('intervalos', 'degraus')
            dfa1_serie = calcular_dfa1_serie(rr_lim, rr_info['tempo_s'],
                                             metodo_detrend=metodo_detrend,
                                             lam_sp=lam_sp,
                                             lap_stats=lap_stats,
                                             respeitar_fases=_respeitar_fases)
            if len(dfa1_serie) >= 10:
                hrvt2 = calcular_hrvt(dfa1_serie, df_metricas=df, colunas=colunas,
                                      alvo=DFA1_HRVT2, lap_stats=lap_stats,
                                      df_tempo=df, protocolo=_tipo)
                hrvt1c = calcular_hrvt1c(dfa1_serie, df_metricas=df, colunas=colunas)
                hrvt2_sub = hrvt2_submaximo(dfa1_serie, df_metricas=df, colunas=colunas)

            if comparar_detrend:
                dfa1_serie_alt = calcular_dfa1_serie(rr_lim, rr_info['tempo_s'],
                                                     metodo_detrend=metodo_alt,
                                                     lam_sp=lam_sp,
                                                     lap_stats=lap_stats,
                                                     respeitar_fases=_respeitar_fases)
                if len(dfa1_serie_alt) >= 10:
                    hrvt2_alt = calcular_hrvt(dfa1_serie_alt, df_metricas=df, colunas=colunas,
                                             alvo=DFA1_HRVT2, lap_stats=lap_stats,
                                             df_tempo=df, protocolo=_tipo)
                    hrvt1c_alt = calcular_hrvt1c(dfa1_serie_alt, df_metricas=df, colunas=colunas)
                    hrvt2_sub_alt = hrvt2_submaximo(dfa1_serie_alt, df_metricas=df, colunas=colunas)

    durabilidade = (analisar_durabilidade(df, colunas, dfa1_serie, lap_stats)
                    if _tipo in ('continuo', 'intervalos', 'degraus') else None)
    combo = combo_limiares(hrvt2, bp_continuo)

    # Limiar por DFA-α1 RECALCULADO, um ponto por lap de trabalho — pensado
    # para DEGRAUS/intervalos com descanso entre cada intensidade crescente,
    # onde faz mais sentido um ponto limpo por degrau (já beneficiando de
    # respeitar_fases) do que uma regressão contínua ao longo da rampa.
    lim_dfa1_recalc = (limiar_dfa1_recalculado(dfa1_serie, lap_stats, colunas,
                                               janela_final_s=janela_final_s)
                      if _tipo in ('degraus', 'intervalos') and len(dfa1_serie) >= 10
                      else None)

    # Classificação exploratória do limitador fisiológico (SmO2/THb) — só faz
    # sentido em protocolos de degraus/intervalos, onde há laps sucessivos de
    # intensidade crescente com descanso entre eles para ver uma tendência.
    limitador_smo2 = (classificar_limitador_smo2(lap_stats, df=df, colunas=colunas)
                      if _tipo in ('degraus', 'intervalos') else None)

    # Relação potência↔FC desta sessão, para reportar os limiares nas duas
    # unidades. A FC é mais estável entre protocolos do que a potência
    # (Physiological Reports 2023), por isso convém ter ambas.
    relacao_pf = _relacao_pot_fc(df, colunas, lap_stats)
    for _b in (bp_continuo, bp_hhb):
        if _b and _b.get('breakpoint') is not None and relacao_pf:
            if _b.get('unidade') == 'W':
                _b['fc'] = pot_para_fc(_b['breakpoint'], relacao_pf)
                _b['potencia'] = _b['breakpoint']
            else:
                _b['fc'] = _b['breakpoint']
                _b['potencia'] = fc_para_pot(_b['breakpoint'], relacao_pf)
    if limiares and limiares.get('media') is not None and relacao_pf:
        if limiares.get('unidade') == 'W':
            limiares['fc_media'] = pot_para_fc(limiares['media'], relacao_pf)
        else:
            limiares['fc_media'] = limiares['media']
    if combo and combo.get('combo') is not None and relacao_pf:
        combo['fc'] = pot_para_fc(combo['combo'], relacao_pf)

    _res = dict(prep)
    _res.update({
        'protocolo': protocolo,
        'restauracao': restauracao,
        'limiares': limiares,
        'bp_continuo': bp_continuo,
        'bp_hhb': bp_hhb,
        'bp_lt1_lt2': bp_lt1_lt2,
        'limiar_dfa1': lim_dfa1,
        'limiar_dfa1_recalculado': lim_dfa1_recalc,
        'limitador_smo2': limitador_smo2,
        'estabilidade_smo2': estab_smo2,
        'mlss_intervalos': mlss_longos,
        'decoupling': decoupling,
        'fadiga': fadiga,
        'tempo_falha': falha,
        'rr_info': rr_info,
        'dfa1_serie': dfa1_serie,
        'dfa1_qualidade': dfa1_qualidade,
        'hrvt2': hrvt2,
        'hrvt1c': hrvt1c,
        'hrvt2_submax': hrvt2_sub,
        'durabilidade': durabilidade,
        'combo': combo,
        'relacao_pot_fc': relacao_pf,
        # ── Comparação de pré-processamento DFA-α1 (local vs SP global) ──────
        'metodo_detrend': metodo_detrend,
        'metodo_detrend_alt': metodo_alt if comparar_detrend else None,
        'dfa1_serie_alt': dfa1_serie_alt,
        'hrvt2_alt': hrvt2_alt,
        'hrvt1c_alt': hrvt1c_alt,
        'hrvt2_submax_alt': hrvt2_sub_alt,
    })
    _res['zonas'] = resumir_zonas(_res)
    _res['fiabilidade'] = avaliar_fiabilidade(_res)
    return _res


def analisar_fit(file_bytes, **kwargs):
    """
    Pipeline completo em duas fases (retrocompatível).
    Para controlar as fases separadamente, usa preparar_fit() + analisar_completo().
    """
    prep = preparar_fit(file_bytes, **kwargs)
    if 'erro' in prep:
        return prep
    return analisar_completo(prep)

# ══════════════════════════════════════════════════════════════════════════════
# 9b. CLASSIFICAÇÃO EXPLORATÓRIA DO LIMITADOR (SmO2/THb — sem NO/CO2)
# ══════════════════════════════════════════════════════════════════════════════
# PROTÓTIPO — ainda não ligado à interface (tab_fit_analise.py). Baseado em:
#   - "5-1-5 Assessment" (Moxy Muscle Oxygen Monitor) — framework de 3 limitadores
#     (Muscular/Utilização, Cardíaco, Pulmonar), pensado para funcionar só com
#     SmO2+THb de um único sensor, sem sensor de NO/CO2.
#   - Evan Peikon, "Applied Bioenergetics" e "A Unified Theory of Bioenergetic
#     Demands in Sport" (Emergent Performance Lab) — mesma lógica de sinais.
#
# IMPORTANTE: isto é uma classificação de TREINO — identifica qual sistema
# fisiológico está a limitar a performance NESTE esforço específico, para
# orientar a prescrição. NÃO é um diagnóstico médico de doença cardíaca,
# respiratória ou muscular.
#
# As regras seguem as DESCRIÇÕES qualitativas dos documentos (inequívocas),
# não a tabela numérica de pontos do Apêndice A do "5-1-5 Assessment" — essa
# tabela veio com a formatação degradada na extracção do PDF e a atribuição
# exacta de cada coluna de pontos (U/S vs P/C) ficou ambígua o suficiente
# para arriscar mal-classificar. Preferi uma versão mais simples e
# transparente a replicar uma fórmula que podia estar errada.

def _classificar_tendencia(valores, limiar_ligeiro=2.0, limiar_claro=6.0):
    """
    Classifica a tendência de uma série de valores (um por lap sucessivo) via
    regressão linear simples. Os limiares por defeito (2 e 6 pontos de
    variação total ao longo da série) foram calibrados para SmO2/THb em
    escala 0-100%; para THb (escala diferente, ex. g/dL) ajusta os limiares
    ao chamar a função.

    Devolve (categoria, variação_total) onde categoria é um de:
    'clara_subida', 'ligeira_subida', 'estavel', 'ligeira_descida', 'clara_descida'.
    """
    y = np.asarray([v for v in valores if v is not None], dtype=float)
    if len(y) < 3:
        return 'dados_insuficientes', 0.0
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0])
    variacao_total = slope * (len(y) - 1)
    if variacao_total > limiar_claro:
        return 'clara_subida', variacao_total
    elif variacao_total > limiar_ligeiro:
        return 'ligeira_subida', variacao_total
    elif variacao_total < -limiar_claro:
        return 'clara_descida', variacao_total
    elif variacao_total < -limiar_ligeiro:
        return 'ligeira_descida', variacao_total
    else:
        return 'estavel', variacao_total


def _forma_thb_inicio_descanso(df, colunas, t_ini, t_fim, segundos=15):
    """
    Olha à FORMA do THb nos primeiros `segundos` de um lap de descanso —
    não só à direção. Feldmann (fórum Moxy): vasodilatação genuína faz o THb
    subir de forma limpa; uma saída de oclusão venosa faz o THb DESCER
    primeiro (a pressão mecânica ainda a libertar-se) e só depois subir. A
    segunda forma é sobretudo um artefacto mecânico/postural, não um sinal
    fisiológico do limitador.

    Devolve 'subida_limpa', 'possivel_oclusao', ou 'indeterminado' (dados
    insuficientes ou sem coluna de THb). Best-effort/heurístico — reportado
    como contexto informativo, nunca pontuado no scoring.
    """
    col = colunas.get('thb')
    if not col or col not in df.columns or 'time_seconds' not in df.columns:
        return 'indeterminado'
    janela = df[(df['time_seconds'] >= t_ini) & (df['time_seconds'] <= min(t_ini + segundos, t_fim))]
    vals = pd.to_numeric(janela[col], errors='coerce').dropna().values
    if len(vals) < 6:
        return 'indeterminado'
    metade = max(3, len(vals) // 2)
    v_ini, v_min_inicio, v_fim = vals[0], vals[:metade].min(), vals[-1]
    if v_min_inicio < v_ini - 0.02 and v_fim > v_min_inicio + 0.02:
        return 'possivel_oclusao'
    return 'subida_limpa'


def classificar_limitador_smo2(lap_stats, min_laps_trabalho=3, df=None, colunas=None,
                               atleta_treinado=True):
    """
    Classificação exploratória do limitador fisiológico dominante num
    protocolo de degraus/intervalos de intensidade crescente, usando só
    SmO2/THb já calculados por estatisticas_por_lap() — sem sensor de NO/CO2.

    IMPORTANTE (revisto após ler discussões do fórum de desenvolvimento da
    Moxy, incl. Andri Feldmann e o investigador de NIRS Jem Arnold): o SmO2
    é consideravelmente mais fiável do que o THb para interpretação
    individual — o THb varia pouco em magnitude face à sua própria baseline
    (~±0.1 numa baseline de 11-13) e não é comparável de forma fiável entre
    sessões. Por isso os sinais baseados em THb aqui pesam MENOS no score do
    que os baseados em SmO2, e a forma da curva do THb (não só a direção) é
    usada como contexto adicional, nunca como prova isolada. Um investigador
    de NIRS envolvido nesse fórum foi explícito: "não teria confiança para
    afirmar que esta apresentação está exclusiva, predominante, ou mais
    provavelmente relacionada com uma limitação X" — mantemos essa cautela.

    Segue a terminologia do "5-1-5 Assessment" (Moxy): usa sempre o MÁXIMO ou
    MÍNIMO real de cada lap (nunca a média), sobre o estado estacionário
    (últimos ~60s, não o lap inteiro — ver estatisticas_por_lap).

    O sinal de utilização/muscular usa agora um critério RELATIVO à própria
    sessão (quanto desceu o SmO2 mínimo do 1º ao último lap de trabalho),
    não um corte absoluto (ex. ">60%") — confirmado no fórum que esses
    cortes variam muito com a posição do sensor, mesmo a poucos cm de
    distância no mesmo músculo (Andri Feldmann).

    df, colunas (opcionais): se fornecidos, verifica também a FORMA do THb
    no início de cada lap de descanso (subida limpa vs possível oclusão) —
    informativo, não entra no score.

    atleta_treinado (default True — perfil do R, anos de treino de endurance):
    calibra a confiança do sinal muscular. Andri Feldmann (fórum Moxy):
    atletas NÃO treinados tendem a ter o SmO2 a DERIVAR PARA CIMA com a
    fadiga (menos capacidade de extrair); atletas TREINADOS mostram o
    oposto — o SmO2 deriva para BAIXO com a fadiga. Por isso, "SmO2 não
    desceu apesar da intensidade a subir" é um sinal mais fiável e direto de
    limitação de utilização num atleta treinado (o principal confundidor —
    ser só o efeito normal da fadiga num atleta pouco treinado — não se
    aplica). Passa False só se estiveres a analisar um atleta iniciante.

    RESSALVA (Evan Peikon, NNOXX): esta calibração por "treinado/não-
    treinado" é uma simplificação de uma variável só. Segundo o Peikon, o
    que realmente determina se o SmO2 sobe ou desce com a fadiga é se há
    SUBSTITUIÇÃO DE RECRUTAMENTO MUSCULAR possível naquele movimento — ele
    próprio observa os dois padrões dependendo da modalidade (desce a
    pedalar, sobe a escalar, mesma pessoa). Sem um 2º sensor num músculo
    não-primário não há forma de confirmar se houve essa substituição —
    "atleta_treinado" fica como aproximação razoável, não uma explicação
    completa.

    Precisa de pelo menos `min_laps_trabalho` laps de trabalho (idealmente
    com descanso entre eles) para conseguir ver uma TENDÊNCIA entre laps
    sucessivos — um único lap não chega. As tendências SÃO SEMPRE uma
    comparação entre laps sucessivos (não um valor isolado) — todas as 8
    séries abaixo (min/max SmO2 e THb, em trabalho e descanso) são
    comparadas lap-a-lap via regressão linear (_classificar_tendencia).

    Devolve dict com 'limitador_provavel' ('muscular'|'cardiaco'|'pulmonar'|
    'inconclusivo'), 'pontuacao' (dict com os 3 scores), 'sinais' (lista de
    strings explicando cada sinal encontrado), 'tendencias' (dados brutos),
    'contexto' (FC e forma do THb, informativo, não pontuado), e um 'aviso'
    que deve ser sempre mostrado junto ao resultado.
    """
    laps_trabalho = [l for l in lap_stats if l.get('phase') == 'work']
    laps_descanso = [l for l in lap_stats if l.get('phase') == 'recovery']

    if len(laps_trabalho) < min_laps_trabalho:
        return {'erro': f'poucos laps de trabalho ({len(laps_trabalho)}) — são '
                        f'precisos pelo menos {min_laps_trabalho} intensidades '
                        'crescentes para ver uma tendência'}

    # Trabalho: extremos do ESTADO ESTACIONÁRIO (últimos ~60s do lap, por
    # defeito — ver janela_final_s em estatisticas_por_lap) em vez do lap
    # inteiro. Isto evita que um pico/vale de ruído no início do lap (ainda
    # em transição da fase anterior, cinética lenta ~30-60s) seja confundido
    # com o valor "estabilizado" que realmente interessa para o limitador.
    # Cai para o lap inteiro se o campo _est não existir (compatibilidade).
    def _v(l, campo_est, campo_todo):
        return l.get(campo_est, l.get(campo_todo))

    min_work_smo2 = [_v(l, 'min_smo2_est', 'min_smo2') for l in laps_trabalho]
    max_work_smo2 = [_v(l, 'max_smo2_est', 'max_smo2') for l in laps_trabalho]
    max_work_thb  = [_v(l, 'max_thb_est', 'max_thb') for l in laps_trabalho]
    min_work_thb  = [_v(l, 'min_thb_est', 'min_thb') for l in laps_trabalho]
    avg_work_hr   = [l.get('avg_heart_rate') for l in laps_trabalho]

    # Descanso: idem — "recupera acima do descanso anterior?" usa o MÁXIMO
    # de SmO2 do estado estacionário de cada lap de descanso, comparado entre
    # si (não um valor fixo)
    max_rest_smo2 = [_v(l, 'max_smo2_est', 'max_smo2') for l in laps_descanso]
    min_rest_smo2 = [_v(l, 'min_smo2_est', 'min_smo2') for l in laps_descanso]
    max_rest_thb  = [_v(l, 'max_thb_est', 'max_thb') for l in laps_descanso]
    min_rest_thb  = [_v(l, 'min_thb_est', 'min_thb') for l in laps_descanso]

    if sum(v is not None for v in min_work_smo2) < min_laps_trabalho:
        return {'erro': 'dados de SmO2 insuficientes nos laps de trabalho '
                        '(precisa da métrica smo2 no ficheiro)'}

    tend_min_work_smo2 = _classificar_tendencia(min_work_smo2)
    tend_max_work_smo2 = _classificar_tendencia(max_work_smo2)
    tend_max_work_thb  = _classificar_tendencia(max_work_thb, limiar_ligeiro=0.1, limiar_claro=0.3)
    tend_min_work_thb  = _classificar_tendencia(min_work_thb, limiar_ligeiro=0.1, limiar_claro=0.3)
    tend_max_rest_smo2 = _classificar_tendencia(max_rest_smo2)
    tend_min_rest_smo2 = _classificar_tendencia(min_rest_smo2)
    tend_max_rest_thb  = _classificar_tendencia(max_rest_thb, limiar_ligeiro=0.1, limiar_claro=0.3)
    tend_min_rest_thb  = _classificar_tendencia(min_rest_thb, limiar_ligeiro=0.1, limiar_claro=0.3)
    # FC: usa só os ÚLTIMOS laps (perto da falha) — Peikon (NNOXX) descreve o
    # padrão como "FC faz patamar ACIMA de ~85% do esforço OU antes da
    # falha", ou seja, é um comportamento de FASE FINAL. Uma reta ajustada à
    # sessão inteira esconde um patamar tardio que vem depois de uma subida
    # inicial normal (mesmo problema que já vimos com o THb no caso do
    # fórum Moxy "5-1-5 decreasing peak SmO2").
    _n_final_hr = min(3, len(avg_work_hr))
    tend_hr_trabalho   = (_classificar_tendencia(avg_work_hr[-_n_final_hr:], limiar_ligeiro=3, limiar_claro=10)
                         if sum(v is not None for v in avg_work_hr[-_n_final_hr:]) >= min(3, min_laps_trabalho)
                         else ('dados_insuficientes', 0.0))

    ultimo_min_smo2  = next((v for v in reversed(min_work_smo2) if v is not None), None)
    primeiro_min_smo2 = next((v for v in min_work_smo2 if v is not None), None)

    sinais = []
    pontuacao = {'muscular': 0, 'cardiaco': 0, 'pulmonar': 0}

    # Muscular/Utilização — critério RELATIVO à própria sessão: quão pouco
    # o SmO2 mínimo desceu do 1º ao último lap de trabalho, face ao ponto de
    # partida. Substitui o corte absoluto (">60%") que o fórum confirma
    # variar demasiado com a posição do sensor.
    #
    # Calibração para ATLETA TREINADO (Andri Feldmann, perfil "andrifeldmann",
    # fórum Moxy): em atletas NÃO treinados, a fadiga muscular tende a fazer o
    # SmO2 DERIVAR PARA CIMA (menos capacidade de extrair, mesmo com a
    # intensidade a subir) — o que tornaria "SmO2 não desceu" um sinal
    # ambíguo (pode ser limitação de utilização OU só o efeito normal da
    # fadiga num atleta pouco treinado). Em atletas TREINADOS acontece o
    # oposto: a fadiga faz o SmO2 derivar para BAIXO (ficam menos eficientes,
    # a exigência central aumenta). Como o R tem anos de treino de endurance,
    # esse confundidor não se aplica aqui — "SmO2 não desceu apesar da
    # intensidade/fadiga a subir" é um sinal mais limpo e direto de limitação
    # de utilização, sem a ambiguidade que teria num atleta não treinado.
    if primeiro_min_smo2 and ultimo_min_smo2 is not None and primeiro_min_smo2 > 0:
        queda_relativa = (primeiro_min_smo2 - ultimo_min_smo2) / primeiro_min_smo2
        if queda_relativa < 0.15:
            pontuacao['muscular'] += 3.5 if atleta_treinado else 2.5
            sinais.append(
                f"SmO2 mínimo em trabalho quase não desceu entre o 1º "
                f"({primeiro_min_smo2:.0f}%) e o último lap ({ultimo_min_smo2:.0f}%) — "
                f"queda de só {queda_relativa*100:.0f}% apesar da intensidade a subir — "
                "sinal de limitação de utilização/capacidade oxidativa muscular"
                + (" (num atleta treinado, este sinal é mais fiável — a fadiga "
                   "normalmente fá-lo-ia descer, não ficar estável)" if atleta_treinado else ""))
        elif queda_relativa > 0.5:
            sinais.append(
                f"SmO2 mínimo desceu {queda_relativa*100:.0f}% do 1º ao último lap "
                f"({primeiro_min_smo2:.0f}%→{ultimo_min_smo2:.0f}%) — boa capacidade de "
                "extração, sem sinal de limitação de utilização")

    # Cardíaco — sinal PRINCIPAL vem do SmO2 (mais fiável): SmO2 máximo em
    # DESCANSO a descer entre laps sucessivos (recupera cada vez menos).
    # THb reforça mas conta MENOS (menos fiável para leitura individual —
    # ver aviso).
    if tend_max_rest_smo2[0] in ('clara_descida', 'ligeira_descida'):
        pts = 2 if tend_max_rest_smo2[0] == 'clara_descida' else 1
        pontuacao['cardiaco'] += pts
        sinais.append(
            f"SmO2 máximo em descanso desce entre laps sucessivos "
            f"({tend_max_rest_smo2[1]:+.1f} pontos) — cada descanso recupera "
            "menos do que o anterior; sugere vasoconstrição simpática "
            "progressiva (sinal cardíaco/delivery)")
    if tend_max_rest_thb[0] in ('clara_descida', 'ligeira_descida'):
        pts = 1 if tend_max_rest_thb[0] == 'clara_descida' else 0.5
        pontuacao['cardiaco'] += pts
        sinais.append(
            f"THb máximo em descanso também desce entre laps ({tend_max_rest_thb[1]:+.2f}) "
            "— reforça (com cautela — THb é menos fiável individualmente) o sinal "
            "de redistribuição de fluxo (cardíaco)")
    if tend_max_work_thb[0] in ('clara_descida', 'ligeira_descida'):
        pts = 0.5 if tend_max_work_thb[0] == 'clara_descida' else 0.25
        pontuacao['cardiaco'] += pts
        sinais.append(
            f"THb máximo em trabalho também desce entre laps ({tend_max_work_thb[1]:+.2f}) "
            "— reforço adicional, fraco, do sinal cardíaco")

    # Pulmonar — precisa da COMBINAÇÃO SmO2+THb (mais robusto do que THb
    # isolado): THb a subir enquanto o SmO2 de trabalho ainda desce.
    if (tend_max_work_thb[0] in ('clara_subida', 'ligeira_subida')
            and tend_min_work_smo2[0] in ('clara_descida', 'ligeira_descida')):
        pts = 2 if tend_max_work_thb[0] == 'clara_subida' else 1
        pontuacao['pulmonar'] += pts
        sinais.append(
            f"THb máximo em trabalho sobe ({tend_max_work_thb[1]:+.2f}) enquanto "
            f"o SmO2 mínimo desce ({tend_min_work_smo2[1]:+.1f} pontos) — sugere "
            "CO2 acumulado/vasodilatação apesar da queda de SmO2 (sinal pulmonar)")
    if tend_max_rest_thb[0] in ('clara_subida', 'ligeira_subida'):
        pts = 0.5 if tend_max_rest_thb[0] == 'clara_subida' else 0.25
        pontuacao['pulmonar'] += pts
        sinais.append(
            f"THb máximo em descanso também sobe entre laps ({tend_max_rest_thb[1]:+.2f}) "
            "— reforço adicional, fraco (o corpo continua a vasodilatar mesmo em "
            "repouso, sem parecer racionar fluxo)")

    # FC em patamar vs a continuar a subir — distingue Cardíaco de Respiratório
    # (Evan Peikon, NNOXX, "Identifying Physiological Limitations"): QUANDO o
    # SmO2 de trabalho já está a descer entre laps (limitação de entrega de
    # O2 already presente), a FC continuar a SUBIR até à falha aponta para
    # respiratório (o coração não é o travão — continua a tentar compensar);
    # a FC fazer PATAMAR aponta para cardíaco (débito cardíaco já no limite).
    # Só entra no score quando o SmO2 já mostra o padrão de "supply limitado";
    # sozinha, a FC não distingue nada (ver nota_fc no contexto).
    if tend_min_work_smo2[0] in ('clara_descida', 'ligeira_descida'):
        if tend_hr_trabalho[0] == 'estavel':
            pontuacao['cardiaco'] += 1.5
            sinais.append(
                "FC faz patamar entre laps sucessivos enquanto o SmO2 mínimo "
                "continua a descer — sugere débito cardíaco já perto do limite "
                "(sinal cardíaco, Peikon/NNOXX)")
        elif tend_hr_trabalho[0] in ('clara_subida', 'ligeira_subida'):
            pontuacao['pulmonar'] += 1
            sinais.append(
                "FC continua a subir entre laps sucessivos enquanto o SmO2 "
                "mínimo desce — o coração não parece ser o travão aqui, "
                "reforça um sinal respiratório em vez de cardíaco (Peikon/NNOXX)")

    limitador_provavel = (max(pontuacao, key=pontuacao.get)
                          if any(pontuacao.values()) else 'inconclusivo')

    # Contexto informativo — NUNCA pontuado, só para dar mais pistas ao
    # utilizador. NOTA sobre o sinal muscular/utilização: o critério de "SmO2
    # não desceu" pressupõe implicitamente que a fadiga faria o SmO2 DESCER
    # neste atleta — mas segundo o próprio Peikon (NNOXX), isto depende de
    # haver ou não substituição de recrutamento muscular possível naquele
    # movimento (ele próprio vê os dois padrões, dependendo da modalidade:
    # desce a pedalar, sobe a escalar). "atleta_treinado" é uma simplificação
    # de uma só variável para algo que na realidade depende também do
    # movimento/músculo monitorizado — sem 2º sensor não dá para confirmar
    # se houve substituição de recrutamento.
    contexto = {'tendencia_fc_trabalho': tend_hr_trabalho}
    if tend_hr_trabalho[0] == 'estavel' and tend_min_work_smo2[0] not in (
            'clara_descida', 'ligeira_descida'):
        contexto['nota_fc'] = (
            "FC pouco variou entre laps de trabalho, mas o SmO2 também não "
            "mostra um padrão claro de queda — isoladamente, FC estável pode "
            "ser SV já no limite (cardíaco) OU uma limitação de utilização "
            "local (visto em corredores a fazer arm-erg sem quase subir a "
            "FC); ambíguo sem mais contexto.")

    if df is not None and colunas is not None and laps_descanso:
        formas = []
        for l in laps_descanso:
            if '_t_ini' in l and '_t_fim' in l:
                formas.append(_forma_thb_inicio_descanso(df, colunas, l['_t_ini'], l['_t_fim']))
        formas_validas = [f for f in formas if f != 'indeterminado']
        if formas_validas:
            n_oclusao = formas_validas.count('possivel_oclusao')
            contexto['forma_thb_descanso'] = formas_validas
            if n_oclusao >= max(2, len(formas_validas) // 2):
                contexto['nota_thb_forma'] = (
                    f"Em {n_oclusao}/{len(formas_validas)} laps de descanso o THb desceu "
                    "antes de subir — pode ser saída de oclusão venosa (artefacto "
                    "mecânico/postural, ex. posição da perna), não necessariamente um "
                    "sinal fisiológico do limitador. Vale a pena rever a posição do "
                    "sensor/perna nos descansos.")

    return {
        'limitador_provavel': limitador_provavel,
        'pontuacao': pontuacao,
        'sinais': sinais,
        'contexto': contexto,
        'tendencias': {
            'min_smo2_trabalho':  tend_min_work_smo2,
            'max_smo2_trabalho':  tend_max_work_smo2,
            'max_thb_trabalho':   tend_max_work_thb,
            'min_thb_trabalho':   tend_min_work_thb,
            'max_smo2_descanso':  tend_max_rest_smo2,
            'min_smo2_descanso':  tend_min_rest_smo2,
            'max_thb_descanso':   tend_max_rest_thb,
            'min_thb_descanso':   tend_min_rest_thb,
        },
        'n_laps_trabalho': len(laps_trabalho),
        'n_laps_descanso': len(laps_descanso),
        'aviso': (
            "Classificação EXPLORATÓRIA de treino — identifica qual sistema "
            "fisiológico PODE estar a limitar a performance NESTE esforço, para "
            "ajudar a orientar a prescrição. NÃO é um diagnóstico médico de "
            "doença cardíaca, respiratória ou muscular, nem uma conclusão "
            "definitiva — mesmo investigadores de NIRS envolvidos neste tipo de "
            "análise são explícitos: observar um padrão não dá confiança para "
            "afirmar que ele é exclusiva ou predominantemente devido a um "
            "limitador específico. Os sinais de THb pesam menos no resultado do "
            "que os de SmO2, por serem menos fiáveis para leitura individual. "
            "Baseado só em SmO2/THb (sem sensor de NO/CO2) — não distingue uma "
            "limitação de 'extração' (hipocapnia por respiração superficial/"
            "rápida) das restantes; esse sinal, a existir, fica combinado no "
            "resultado muscular/pulmonar."
        ),
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


def _ajuste_triplo_linear(x, y, margem=0.12):
    """
    Ajusta um modelo de TRÊS rectas (dois pontos de quebra) — o análogo a
    detectar LT1 (1ª quebra de inclinação) e LT2 (2ª quebra/achatamento) na
    mesma série. Andri Feldmann (fórum de developers da Moxy, tópico "Moxy
    to control intensity"): "you will find two distinctive changes in your
    smo2 slope/rate. The first drop is LT1 and then the second is either a
    second clear drop or a flattening/attenuation, this is LT2."

    Generaliza _ajuste_double_linear (1 quebra) para 2 quebras: testa todos
    os pares de pontos candidatos (i < j) como pontos de quebra, ajusta uma
    recta em cada um dos 3 segmentos, e escolhe o par que minimiza a soma
    dos quadrados dos resíduos (SSE) — grid search O(n²), aceitável porque
    `x`/`y` já vêm sub-amostrados (dezenas de pontos, não milhares).

    margem : fracção dos extremos a ignorar como candidatos, tal como em
        _ajuste_double_linear (evita quebras sem significado fisiológico
        mesmo junto ao início/fim da série).

    Devolve dict com os dois breakpoints (bp1 < bp2 — LT1 e LT2), declives
    dos 3 segmentos, R² e os índices usados, ou None se não for possível
    ajustar (dados insuficientes para 3 segmentos com pontos suficientes).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 18:
        return None

    ini = max(4, int(n * margem))
    fim = min(n - 4, int(n * (1 - margem)))
    if fim - ini < 6:
        return None

    melhor = None
    sse_total = np.sum((y - y.mean()) ** 2)

    for i in range(ini, fim - 3):
        x1, y1 = x[:i], y[:i]
        if len(x1) < 3 or np.ptp(x1) < 1e-9:
            continue
        try:
            c1 = np.polyfit(x1, y1, 1)
        except Exception:
            continue
        sse1 = np.sum((y1 - np.polyval(c1, x1)) ** 2)

        for j in range(i + 3, fim):
            x2, y2 = x[i:j], y[i:j]
            x3, y3 = x[j:], y[j:]
            if len(x2) < 3 or len(x3) < 3:
                continue
            if np.ptp(x2) < 1e-9 or np.ptp(x3) < 1e-9:
                continue
            try:
                c2 = np.polyfit(x2, y2, 1)
                c3 = np.polyfit(x3, y3, 1)
            except Exception:
                continue
            sse = (sse1 + np.sum((y2 - np.polyval(c2, x2)) ** 2)
                        + np.sum((y3 - np.polyval(c3, x3)) ** 2))
            if melhor is None or sse < melhor['sse']:
                melhor = {'i': i, 'j': j, 'sse': sse, 'c1': c1, 'c2': c2, 'c3': c3}

    if melhor is None:
        return None

    c1, c2, c3 = melhor['c1'], melhor['c2'], melhor['c3']

    def _intersecao(ca, cb, x_candidato):
        if abs(ca[0] - cb[0]) > 1e-9:
            xi = (cb[1] - ca[1]) / (ca[0] - cb[0])
            if x.min() <= xi <= x.max():
                return float(xi)
        return float(x_candidato)

    bp1 = _intersecao(c1, c2, x[melhor['i']])
    bp2 = _intersecao(c2, c3, x[melhor['j']])
    if bp2 < bp1:  # segurança — por construção (i<j) já deveria vir ordenado
        bp1, bp2 = bp2, bp1

    r2 = 1 - melhor['sse'] / sse_total if sse_total > 0 else np.nan

    return {
        'breakpoint_lt1': bp1,
        'breakpoint_lt2': bp2,
        'idx_lt1': melhor['i'],
        'idx_lt2': melhor['j'],
        'slope_1': float(c1[0]),
        'slope_2': float(c2[0]),
        'slope_3': float(c3[0]),
        'coef_1': c1.tolist(),
        'coef_2': c2.tolist(),
        'coef_3': c3.tolist(),
        'r2': float(r2),
        'n_pontos': n,
    }


def breakpoint_smo2_continuo(df, colunas, lap_stats=None, janela_media=10,
                             usar_apenas_trabalho=True, so_estado_estacionario=True,
                             janela_estavel_s=90, protocolo=None, sinal='smo2'):
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
    if 'smo2' not in colunas and 'hhb' not in colunas:
        return None

    # ── Adaptação ao protocolo ───────────────────────────────────────────────
    # Numa RAMPA CONTÍNUA não existem "laps de trabalho" nem estado estacionário:
    # cada instante tem a sua própria intensidade, e é precisamente essa
    # continuidade que o método double-linear pressupõe. Restringir aos laps ou
    # ao fim de cada bloco destruiria a maior parte dos dados. Em DEGRAUS e
    # INTERVALOS, pelo contrário, as restrições são essenciais (ver docstring).
    if protocolo in ('rampa', 'continuo'):
        usar_apenas_trabalho = False
        so_estado_estacionario = False

    # `sinal` escolhe a métrica: 'smo2' (proporção) ou 'hhb' (quantidade
    # absoluta de hemoglobina desoxigenada — o que a literatura NIRS analisa).
    _chave = sinal if sinal in colunas else ('smo2' if 'smo2' in colunas else 'hhb')
    col_smo2 = colunas[_chave]
    nome_sinal = 'HHb' if _chave == 'hhb' else 'SmO₂'
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

    # Interpretação do padrão (recto femoral: espera-se aceleração da
    # desoxigenação). Em SmO2 isso é a queda a acelerar (declive mais negativo);
    # em HHb é a subida a acelerar (declive mais positivo).
    s1, s2 = res['slope_antes'], res['slope_depois']
    _acelerou = (s2 > s1) if _chave == 'hhb' else (s2 < s1)
    if _acelerou:
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
        'protocolo': protocolo,
        'sinal': nome_sinal,
        'chave_sinal': _chave,
        'usou_estado_estacionario': bool(so_estado_estacionario),
        'usou_apenas_trabalho': bool(usar_apenas_trabalho),
        'padrao': padrao,
        'coerente_recto_femoral': coerente,
    })
    return res


def breakpoint_smo2_lt1_lt2(df, colunas, lap_stats=None, janela_media=10,
                            usar_apenas_trabalho=True, so_estado_estacionario=True,
                            janela_estavel_s=90, protocolo=None, sinal='smo2'):
    """
    Deteta DOIS breakpoints de SmO₂ (aproximação a LT1 e LT2), em vez de um
    só — Andri Feldmann (fórum de developers da Moxy, "Moxy to control
    intensity"): "you will find two distinctive changes in your smo2
    slope/rate. The first drop is LT1 and then the second is either a
    second clear drop or a flattening/attenuation, this is LT2."

    Reutiliza exactamente o mesmo pré-processamento de breakpoint_smo2_continuo()
    (média móvel, amostragem, filtro de trabalho/estado-estacionário conforme o
    protocolo) — só a fase final do ajuste muda: em vez de _ajuste_double_linear
    (1 quebra), usa _ajuste_triplo_linear (2 quebras).

    LT1 é o breakpoint de MENOR intensidade (1ª mudança de declive — transição
    moderado→pesado); LT2 é o de MAIOR intensidade (2ª mudança — transição
    pesado→severo, tipicamente perto do que já calculamos como HRVT2/RCP via
    DFA-α1). Ter os dois a partir do MESMO sinal (SmO₂) dá um segundo ponto de
    vista, independente do DFA-α1, para cruzar com o HRVT1c existente.

    RESSALVA IMPORTANTE (Jem Arnold, investigador de NIRS, fórum Moxy,
    citando Caen et al. 2022): a margem de erro do breakpoint de NIRS entre
    sessões pode ser grande (~50W numa medição de 170W nalguns indivíduos) —
    trata isto como uma ESTIMATIVA aproximada, não um valor de precisão
    laboratorial, sobretudo para prescrição fina de treino.

    Parâmetros e devolução: mesma estrutura de breakpoint_smo2_continuo(),
    mas com 'breakpoint_lt1'/'breakpoint_lt2' (e declives '_1'/'_2'/'_3') em
    vez de um único 'breakpoint'. Devolve None nas mesmas condições (dados
    insuficientes) — precisa de mais pontos do que o caso de 1 quebra, já
    que agora há 3 segmentos em vez de 2.
    """
    if 'smo2' not in colunas and 'hhb' not in colunas:
        return None

    if protocolo in ('rampa', 'continuo'):
        usar_apenas_trabalho = False
        so_estado_estacionario = False

    _chave = sinal if sinal in colunas else ('smo2' if 'smo2' in colunas else 'hhb')
    col_smo2 = colunas[_chave]
    nome_sinal = 'HHb' if _chave == 'hhb' else 'SmO₂'
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

    if so_estado_estacionario and janela_estavel_s and janela_estavel_s > 0:
        partes = []
        for _, g in d.groupby('lap'):
            if len(g) == 0:
                continue
            t_fim = g['t'].max()
            sub = g[g['t'] >= t_fim - janela_estavel_s]
            if len(sub) < 10:
                sub = g.tail(max(10, len(g) // 2))
            partes.append(sub)
        if partes:
            d = pd.concat(partes, ignore_index=True)

    d = d.dropna(subset=['smo2', 'intensidade'])
    if len(d) < 40:
        return None

    d = d.sort_values('t').reset_index(drop=True)
    _mp = max(3, janela_media // 2)
    d['smo2_ma'] = d['smo2'].rolling(janela_media, min_periods=_mp).mean()
    d['int_ma'] = d['intensidade'].rolling(janela_media, min_periods=_mp).mean()
    amostra = d.iloc[::janela_media].dropna(subset=['smo2_ma', 'int_ma'])
    if len(amostra) < 18:  # precisa de mais pontos do que a versão de 1 quebra
        return None

    amostra = amostra.sort_values('int_ma').reset_index(drop=True)

    res = _ajuste_triplo_linear(amostra['int_ma'].values, amostra['smo2_ma'].values)
    if res is None:
        return None

    # Interpretação do padrão do 2º segmento->3º segmento (LT2), espelhando a
    # lógica já usada em breakpoint_smo2_continuo para o recto femoral
    s2, s3 = res['slope_2'], res['slope_3']
    _acelerou = (s3 > s2) if _chave == 'hhb' else (s3 < s2)
    if _acelerou:
        padrao_lt2 = 'aceleração da desoxigenação'
    elif abs(s3) < abs(s2) * 0.5:
        padrao_lt2 = 'plateau (padrão típico de vasto lateral)'
    else:
        padrao_lt2 = 'sem mudança clara de declive'

    res.update({
        'unidade': unidade,
        'pontos': amostra[['int_ma', 'smo2_ma', 't']].rename(
            columns={'int_ma': 'intensidade', 'smo2_ma': 'smo2', 't': 'tempo_s'}),
        'janela_media': janela_media,
        'protocolo': protocolo,
        'sinal': nome_sinal,
        'chave_sinal': _chave,
        'usou_estado_estacionario': bool(so_estado_estacionario),
        'usou_apenas_trabalho': bool(usar_apenas_trabalho),
        'padrao_lt2': padrao_lt2,
        'aviso': (
            'Estimativa aproximada — a margem de erro de breakpoints por NIRS '
            'entre sessões pode ser grande (dezenas de W); usar como referência '
            'de zona, não como valor de precisão laboratorial.'
        ),
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


def _agregar_dfa1_recalculado_por_lap(dfa1_serie, lap_stats, janela_final_s=60):
    """
    Agrega o DFA-α1 RECALCULADO a partir dos RR (calcular_dfa1_serie) por lap
    de trabalho — um ponto por lap, média sobre os últimos janela_final_s
    segundos (o mesmo critério de "estado estacionário" já usado nas outras
    métricas via estatisticas_por_lap).

    Pensado para protocolos de DEGRAUS/intervalos com descanso genuíno entre
    cada intensidade crescente: cada degrau vira um ponto único e limpo,
    já beneficiando de (a) correcção de artefactos do RR e (b) janelas que
    respeitam as fronteiras dos laps (ver respeitar_fases em
    calcular_dfa1_serie) — ao contrário do stream cru do dispositivo, que
    não passa por nenhuma das duas correcções.
    """
    if dfa1_serie is None or len(dfa1_serie) == 0:
        return pd.DataFrame()

    linhas = []
    for l in lap_stats:
        if l.get('phase') != 'work' or '_t_ini' not in l or '_t_fim' not in l:
            continue
        t_ini, t_fim = l['_t_ini'], l['_t_fim']
        t0_janela = max(t_ini, t_fim - janela_final_s) if janela_final_s > 0 else t_ini
        m = (dfa1_serie['tempo_s'] >= t0_janela) & (dfa1_serie['tempo_s'] <= t_fim)
        sub = dfa1_serie[m]
        if len(sub) < 2:
            continue
        linhas.append({
            'lap': l['lap_number'],
            'dfa1_recalculado': float(sub['dfa1'].mean()),
            'n_janelas': len(sub),
            'janela_efetiva_media_s': (float(sub['janela_efetiva_s'].mean())
                                       if 'janela_efetiva_s' in sub.columns else None),
        })
    return pd.DataFrame(linhas)


def limiar_dfa1_recalculado(dfa1_serie, lap_stats, colunas, alvos=(0.75, 0.70, 0.50),
                            janela_final_s=60):
    """
    Versão de limiar_dfa1() que usa o DFA-α1 RECALCULADO a partir dos
    intervalos RR, em vez do stream cru do dispositivo.

    Pensado especificamente para protocolos de DEGRAUS/intervalos com
    descanso entre cada degrau de intensidade crescente — um ponto por lap
    de trabalho, com janelas que já não misturam descanso com trabalho
    (ver respeitar_fases em calcular_dfa1_serie). Mesma lógica de
    regressão/solução que limiar_dfa1(); ver ali para a interpretação dos
    alvos (0.75/0.70/0.50).

    Devolve dict no mesmo formato de limiar_dfa1(), ou {'erro': ...}.
    """
    agg = _agregar_dfa1_recalculado_por_lap(dfa1_serie, lap_stats, janela_final_s)
    if len(agg) < 3:
        return {'erro': f'poucos laps de trabalho com α1 recalculado (n={len(agg)})'}

    intensidade = 'avg_power' if any('avg_power' in l for l in lap_stats) else 'avg_heart_rate'
    unidade = 'W' if intensidade == 'avg_power' else 'bpm'
    mapa_int = {l['lap_number']: l.get(intensidade) for l in lap_stats}
    agg['intensidade'] = agg['lap'].map(mapa_int)
    agg = agg.dropna(subset=['intensidade']).sort_values('intensidade').reset_index(drop=True)
    if len(agg) < 3:
        return {'erro': 'poucos laps com intensidade e α1 recalculado em conjunto'}

    x = agg['intensidade'].values.astype(float)
    y = agg['dfa1_recalculado'].values.astype(float)
    if np.ptp(x) < 1e-9:
        return {'erro': 'intensidade sem variação'}

    coef = np.polyfit(x, y, 1)
    y_pred = np.polyval(coef, x)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / sst if sst > 0 else np.nan

    limiares = {}
    for alvo in alvos:
        if abs(coef[0]) > 1e-12:
            xi = (alvo - coef[1]) / coef[0]
            dentro = x.min() - 0.1 * np.ptp(x) <= xi <= x.max() + 0.1 * np.ptp(x)
            _fisio_ok = True
            if intensidade == 'avg_heart_rate':
                _fisio_ok, _ = _checar_fc_plausivel(xi, fc_max_sessao=float(x.max()))
            limiares[alvo] = {'intensidade': float(xi), 'extrapolado': not dentro,
                              'fisiologicamente_plausivel': _fisio_ok}
        else:
            limiares[alvo] = None

    return {
        'limiares': limiares,
        'coef': coef.tolist(),
        'r2': float(r2),
        'unidade': unidade,
        'pontos': agg,
        'n_usados': len(agg),
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

def extrair_rr(file_bytes, rr_bruto=None):
    """
    Extrai os intervalos RR brutos das mensagens 'hrv' do ficheiro FIT.

    Muitos gravadores (Garmin, apps com Polar H10) guardam os RR além das
    métricas por segundo. Tê-los permite recalcular o DFA-alpha1 em vez de
    depender do valor pré-calculado pelo sensor.

    rr_bruto : lista já extraída por ler_fit() (ver preparar_fit()['rr_bruto']).
        Quando fornecida, esta função NÃO volta a abrir/percorrer o ficheiro —
        evita uma segunda passagem completa do fitdecode, que é cara em
        ficheiros com RR batimento-a-batimento (milhares de mensagens 'hrv').
        Se vier None, mantém o comportamento antigo (lê o ficheiro do zero) —
        para chamadas directas fora do fluxo preparar_fit()+analisar_completo.

    Devolve dict {'rr_ms': array, 'tempo_s': array (tempo cumulativo), 'n': int}
    ou None se o ficheiro não tiver RR.
    """
    if rr_bruto is not None:
        rr = list(rr_bruto)
    else:
        if not _TEM_FITDECODE:
            return None
        rr = []
        try:
            with fitdecode.FitReader(io.BytesIO(file_bytes), check_crc=fitdecode.CrcCheck.DISABLED) as fit:
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


def detrend_sp(rr_ms, lam=500):
    """
    Smoothness Priors detrending (Tarvainen et al. 2002, λ=500) — o método de
    detrending por DEFEITO do Kubios HRV, aplicado ao tacograma INTEIRO (não
    por janela) antes de qualquer cálculo de DFA-α1.

    Porquê isto importa (ver "DFA a1 and ChatGPT interview",
    muscleoxygentraining.com, ago/2025): aplicar Smoothness Priors DENTRO de
    cada janela de 2 min (em vez de à série completa) cria efeitos de
    fronteira que "empurram o α1 para baixo" e distorcem a recta α1×FC —
    foi exactamente essa diferença que fez o pipeline "à mão" divergir ~15-30
    bpm do Kubios num caso real analisado nesse artigo. A correcção foi
    aplicar o SP UMA VEZ à série toda, e só depois janelar.

    O que já estava implementado em calcular_dfa1_serie() (metodo_detrend=
    'local') é o DFA-1 clássico (Peng et al.), que faz o SEU PRÓPRIO
    detrending linear por escala DENTRO de cada janela — replica o script
    "Kubios vs Python" (dokato/dfa) já citado em _dfa_alpha(). Isso continua
    a ser calculado da mesma forma; o SP global é um passo ADICIONAL, antes,
    que remove a deriva lenta (troca de protocolo, deriva térmica, etc.) que
    de outra forma dominaria a fractal-scaling nas escalas mais longas.

    Implementação: resolve o sistema esparso
        tendência = (I + λ² · D2ᵀD2)⁻¹ · RR
    onde D2 é o operador de 2ª diferença. O resultado devolvido é
        RR − tendência + média(RR)
    (mantém a escala em ms, o que ajuda em qualquer plot de diagnóstico;
    para o DFA em si a constante é irrelevante, pois _dfa_alpha já subtrai
    a média internamente).

    Requer scipy.sparse. Sistema banda-estreita ⇒ O(n), viável mesmo para
    sessões de 1h+ (~10 mil batimentos).

    Devolve um array do mesmo comprimento que rr_ms, ou uma cópia sem
    alteração se a série for demasiado curta ou o sistema mal condicionado.
    """
    x = np.asarray(rr_ms, dtype=float)
    n = len(x)
    if n < 10:
        return x.copy()

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except ImportError:
        return x.copy()

    # Operador de 2ª diferença: (n-2) x n, cada linha [.., 1, -2, 1, ..]
    D2 = sparse.diags([1.0, -2.0, 1.0], offsets=[0, 1, 2],
                       shape=(n - 2, n), format='csc')
    I = sparse.eye(n, format='csc')
    A = (I + (lam ** 2) * (D2.T @ D2)).tocsc()

    try:
        tendencia = spsolve(A, x)
    except Exception:
        return x.copy()

    if not np.all(np.isfinite(tendencia)):
        return x.copy()

    return x - tendencia + np.mean(x)


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


def _lap_id_por_tempo(tempo_s_pontos, lap_stats):
    """
    Atribui a cada ponto temporal o número do lap onde cai, ou -1 se estiver
    fora de qualquer lap com fronteiras conhecidas ('_t_ini'/'_t_fim').

    Usado para impedir que uma janela de DFA-α1 misture batimentos de dois
    laps diferentes (ex.: fim da recuperação + início do trabalho seguinte),
    o que dilui exactamente o mergulho de α1 que se quer detectar em
    protocolos de intervalos curtos.
    """
    t = np.asarray(tempo_s_pontos, dtype=float)
    ids = np.full(len(t), -1, dtype=int)
    if not lap_stats:
        return ids
    for l in lap_stats:
        if '_t_ini' not in l or '_t_fim' not in l:
            continue
        m = (t >= l['_t_ini']) & (t <= l['_t_fim'])
        ids[m] = l['lap_number']
    return ids


def calcular_dfa1_serie(rr_ms, tempo_s, janela_s=120, passo_s=5,
                        n_min=4, n_max=16, metodo_detrend='local', lam_sp=500,
                        lap_stats=None, respeitar_fases=True):
    """
    Calcula o DFA-alpha1 ao longo do tempo, com janelas móveis.

    Parâmetros do estudo Murias 2023:
      "the DFA a1 was calculated over time using 2 min HRV measurement windows
       with a recalculation every 5 s"

    metodo_detrend:
      'local'     (default) — cada janela é detrendida linearmente por escala,
                  dentro do próprio DFA-1 clássico (ver _dfa_alpha). É o que
                  já estava implementado; replica a abordagem "Kubios vs
                  Python" (dokato/dfa) já usada neste ficheiro.
      'sp_global' — aplica Smoothness Priors (λ=lam_sp) UMA VEZ ao tacograma
                  inteiro antes de janelar (ver detrend_sp()) — replica o
                  pré-processamento por defeito do Kubios HRV.

    lap_stats, respeitar_fases : em protocolos de INTERVALOS (ex.: 3 min de
        trabalho + 1 min de descanso), um intervalo pode ser mais curto do
        que os 120s da janela-padrão. Sem cuidado, uma janela centrada
        pouco depois do início do trabalho ainda "olha para trás" 120s e
        acaba a incluir batimentos da recuperação anterior — o que dilui
        para cima exactamente o mergulho de α1 que se quer captar (a
        recuperação tem α1 mais alto). Com respeitar_fases=True (default,
        quando lap_stats é fornecido) cada janela é recortada para NUNCA
        atravessar a fronteira do lap onde está o seu centro: fica mais
        curta perto do início de cada lap (mas nunca abaixo do mínimo de
        batimentos exigido) e alcança os 120s completos assim que o lap for
        longo o suficiente. Para rampas/contínuo (um único lap efectivo)
        isto não faz diferença — passa respeitar_fases=False para desligar.

    Nota: a FC média de cada janela é SEMPRE calculada a partir do RR
    ORIGINAL (nunca do detrendido), já que o SP remove a escala absoluta
    do sinal — só o α1 usa a série processada por metodo_detrend.

    Devolve DataFrame com tempo_s, dfa1, n_batimentos, fc_media,
    janela_efetiva_s (quantos segundos de RR entraram realmente na janela —
    menos de janela_s indica um lap curto/início de intervalo).
    """
    rr = np.asarray(rr_ms, dtype=float)
    t = np.asarray(tempo_s, dtype=float)
    if len(rr) < 50:
        return pd.DataFrame()

    if metodo_detrend == 'sp_global':
        rr_dfa = detrend_sp(rr, lam=lam_sp)
    else:
        rr_dfa = rr  # o detrending acontece dentro de _dfa_alpha, por janela

    lap_ids = (_lap_id_por_tempo(t, lap_stats)
               if (respeitar_fases and lap_stats) else None)

    linhas = []
    t_fim = t[-1]
    t_ini = janela_s
    for centro in np.arange(t_ini, t_fim + 0.001, passo_s):
        m = (t > centro - janela_s) & (t <= centro)
        janela_efetiva = janela_s

        if lap_ids is not None:
            idx_centro = np.searchsorted(t, centro, side='right') - 1
            idx_centro = min(max(idx_centro, 0), len(t) - 1)
            lap_centro = lap_ids[idx_centro]
            if lap_centro != -1:
                m = m & (lap_ids == lap_centro)
                if m.sum() > 0:
                    janela_efetiva = float(centro - t[m].min())

        if m.sum() < n_max * 4:
            continue
        seg_dfa = rr_dfa[m]
        seg_fc = rr[m]
        a1 = _dfa_alpha(seg_dfa, n_min=n_min, n_max=n_max)
        if a1 is None:
            continue
        linhas.append({
            'tempo_s': float(centro),
            'dfa1': round(a1, 4),
            'n_batimentos': int(m.sum()),
            'fc_media': round(60000.0 / np.mean(seg_fc), 1),
            'janela_efetiva_s': round(janela_efetiva, 1),
        })

    df_out = pd.DataFrame(linhas)
    df_out.attrs['metodo_detrend'] = metodo_detrend
    df_out.attrs['respeitar_fases'] = bool(lap_ids is not None)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# HRVT2 — segundo limiar pelo DFA-alpha1 (Fleitas-Paniagua/Murias 2023)
# ══════════════════════════════════════════════════════════════════════════════

# Valores de referência do DFA-alpha1 na literatura
DFA1_HRVT2 = 0.50   # HRVT2 ≈ RCP / MLSS / limiar ALTO (Murias 2023, Rogers et al.)
DFA1_HRVT1 = 0.75   # aproximação do VT1 / limiar BAIXO (Gronwald, Rogers 2020)

# Limites de plausibilidade fisiológica para uma FC extrapolada/estimada por
# regressão. Sem isto, um declive quase-nulo (ex.: janela demasiado "achatada"
# depois de Smoothness Priors global remover a tendência lenta que era o
# próprio sinal de interesse) pode gerar uma FC "prevista" de centenas de bpm
# — matematicamente correcta pela recta, mas humanamente impossível. Os
# avisos de R²/extrapolação já cobrem MUITOS casos, mas não todos; este é o
# último guarda-redes, sempre aplicado.
#
# IMPORTANTE: 230 bpm é um tecto GENÉRICO populacional — não diz nada sobre
# ESTE atleta. Um valor de 220 bpm passaria neste teste sozinho, mas para um
# atleta cuja FC real nunca passou de ~165 bpm em toda a sessão, 220 bpm é
# tão implausível como 353 — só que mais difícil de notar à primeira vista.
# Por isso, sempre que houver FC medida na própria sessão (fc_max_sessao),
# o tecto usado é essa FC máxima real + uma margem, não o limite genérico.
FC_PLAUSIVEL_MIN = 30
FC_PLAUSIVEL_MAX = 230          # usado só quando não há FC de referência da sessão
FC_PLAUSIVEL_MARGEM_SESSAO = 15  # bpm acima do máximo realmente medido


def _checar_fc_plausivel(fc, fc_max_sessao=None):
    """
    Devolve (bool_plausivel, aviso_ou_None) para uma FC extrapolada.

    fc_max_sessao : FC máxima REALMENTE medida nesta sessão/atleta (não uma
        constante populacional). Quando fornecida, o tecto de plausibilidade
        passa a ser fc_max_sessao + FC_PLAUSIVEL_MARGEM_SESSAO — específico
        do atleta, em vez do limite genérico de 230 bpm. Segue o mesmo
        princípio já usado no resto do projecto: limiares sempre relativos
        à distribuição própria do atleta, nunca a normas populacionais.
    """
    if fc is None or not np.isfinite(fc):
        return False, 'FC não calculável (recta sem solução válida)'

    if fc_max_sessao is not None and np.isfinite(fc_max_sessao):
        teto = min(FC_PLAUSIVEL_MAX, fc_max_sessao + FC_PLAUSIVEL_MARGEM_SESSAO)
        ref = f'{fc_max_sessao:.0f} bpm medidos nesta sessão + {FC_PLAUSIVEL_MARGEM_SESSAO:.0f}'
    else:
        teto = FC_PLAUSIVEL_MAX
        ref = f'{FC_PLAUSIVEL_MAX:.0f} (limite genérico, sem referência da sessão)'

    if not (FC_PLAUSIVEL_MIN <= fc <= teto):
        return False, (f'FC extrapolada implausível para este atleta ({fc:.0f} bpm, '
                       f'acima de {ref} bpm) — a recta ficou demasiado achatada ou o '
                       'ajuste é dominado por ruído nesta janela; não usar este resultado')
    return True, None


def calcular_hrvt(serie_dfa1, df_metricas=None, colunas=None, alvo=0.50,
                  janela_ajuste=(0.4, 1.0), lap_stats=None, df_tempo=None,
                  so_trabalho=True, protocolo=None):
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
    if protocolo in ('rampa', 'continuo'):
        so_trabalho = False
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
    _fc_max_sessao = float(s['fc_media'].max()) if len(s) else None
    _fc_ok, _fc_aviso = _checar_fc_plausivel(fc_limiar, fc_max_sessao=_fc_max_sessao)
    if not _fc_ok:
        avisos.append(_fc_aviso)
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


# ══════════════════════════════════════════════════════════════════════════════
# DETEÇÃO DO TIPO DE PROTOCOLO
# As análises de limiares diferem consoante o teste seja uma RAMPA CONTÍNUA,
# DEGRAUS INCREMENTAIS ou INTERVALOS. Detectar isto automaticamente evita
# aplicar o método errado.
# ══════════════════════════════════════════════════════════════════════════════

def detectar_protocolo(df, colunas, lap_stats=None):
    """Classifica o protocolo. Ver docstring completa abaixo."""
    # ── Atalho: se os laps já dizem a estrutura, usar isso ───────────────────
    # Quando o utilizador (ou o ficheiro) define laps de trabalho separados por
    # recuperações, a estrutura é conhecida e não precisa de ser inferida do
    # sinal. Inferir pode falhar: por exemplo, ao excluir o aquecimento a série
    # restante pode parecer uma subida contínua e ser classificada como rampa,
    # levando a aplicar o método errado (usar todos os pontos em vez do estado
    # estacionário de cada degrau).
    if lap_stats:
        _w = [l for l in lap_stats if l.get('phase') == 'work']
        _r = [l for l in lap_stats if l.get('phase') == 'recovery']
        if len(_w) >= 3 and len(_r) >= 2:
            _pots = [l.get('avg_power') or l.get('avg_heart_rate') for l in _w]
            _pots = [p for p in _pots if p is not None]
            _sobe = (len(_pots) >= 3
                     and np.polyfit(range(len(_pots)), _pots, 1)[0] > 0)
            _tipo = 'degraus' if _sobe else 'intervalos'
            _u = 'W' if 'power' in colunas else 'bpm'
            return {
                'tipo': _tipo,
                'motivo': (f'{len(_w)} blocos de trabalho separados por '
                           f'{len(_r)} recuperações'
                           + (' com intensidade crescente' if _sobe else
                              ' à mesma intensidade')),
                'metodo_recomendado': (
                    'breakpoint sobre o estado estacionário de cada degrau'
                    if _tipo == 'degraus' else
                    'estabilidade do SmO₂ dentro de cada intervalo'),
                'duracao_min': round((df['time_seconds'].max()
                                      - df['time_seconds'].min()) / 60, 1),
                'origem': 'estrutura dos laps',
                'unidade': _u,
            }
    return _detectar_protocolo_sinal(df, colunas, lap_stats)


def _detectar_protocolo_sinal(df, colunas, lap_stats=None):
    """
    Classifica o tipo de sessão a partir do comportamento da intensidade:

      'rampa'      → intensidade sobe de forma quase monótona, sem recuperações
                     (ex.: +10-30 W/min contínuo). É o protocolo dos estudos.
      'degraus'    → patamares de intensidade constante separados por
                     recuperações ou por saltos (o teu caso habitual).
      'intervalos' → alternância trabalho/recuperação sem progressão clara de
                     intensidade (ex.: 5x3min à mesma potência).
      'continuo'   → intensidade estável do início ao fim (ex.: tempo run).
      'indefinido' → sem sinal utilizável.

    Devolve dict com o tipo, métricas de suporte e o método de limiar recomendado.
    """
    col = colunas.get('power') or colunas.get('heart_rate')
    if col is None:
        return {'tipo': 'indefinido', 'motivo': 'sem potência nem FC'}

    s = pd.to_numeric(df[col], errors='coerce')
    t = df['time_seconds']
    m = s.notna()
    if m.sum() < 120:
        return {'tipo': 'indefinido', 'motivo': 'poucos dados'}
    s, t = s[m].values, t[m].values

    # Suavizar para avaliar a forma geral, não o ruído
    ss = pd.Series(s).rolling(30, min_periods=1, center=True).mean().values

    dur_min = (t[-1] - t[0]) / 60.0
    amp = float(np.percentile(ss, 95) - np.percentile(ss, 5))
    nivel = float(np.median(ss))
    amp_rel = amp / nivel if nivel > 0 else 0

    # Tendência global: quanto da variação é explicada por uma subida linear?
    coef = np.polyfit(t, ss, 1)
    pred = np.polyval(coef, t)
    sst = np.sum((ss - ss.mean()) ** 2)
    r2_linear = float(1 - np.sum((ss - pred) ** 2) / sst) if sst > 0 else 0.0
    subida_por_min = float(coef[0] * 60)

    # Fracção do tempo em "recuperação" — só conta como recuperação se for uma
    # QUEDA a partir de intensidade alta, não o início baixo de uma rampa.
    # Por isso avalia-se em relação ao valor local anterior, não ao global.
    modo_alto = float(np.median(ss[ss >= np.median(ss)]))
    baixo = ss < modo_alto * 0.5
    # Ignorar o troço inicial contíguo abaixo do limiar (aquecimento/arranque)
    if baixo.size and baixo[0]:
        i = 0
        while i < len(baixo) and baixo[i]:
            baixo[i] = False
            i += 1
    frac_baixo = float(np.mean(baixo))

    # Monotonia: fracção do tempo em que o sinal suavizado sobe
    d = np.diff(ss)
    frac_sobe = float(np.mean(d > 0)) if len(d) else 0.0

    # "Escadaria": numa rampa a intensidade muda continuamente; em degraus há
    # patamares planos separados por saltos. Compara-se a variação dentro de cada
    # janela com a variação ESPERADA se fosse rampa (declive × duração da janela),
    # e não com a amplitude global — caso contrário uma rampa lenta pareceria
    # toda "plana".
    jan = 30
    n_jan = max(len(ss) // jan, 1)
    esperado_rampa = abs(coef[0]) * jan   # variação esperada numa janela, se rampa
    planos = 0
    for i in range(n_jan):
        w = ss[i * jan:(i + 1) * jan]
        if len(w) < 5:
            continue
        # É "plano" se variar muito menos do que uma rampa variaria
        if np.ptp(w) < max(esperado_rampa * 0.4, amp * 0.01):
            planos += 1
    frac_planos = planos / n_jan if n_jan else 0.0

    # Tendência dos BLOCOS DE TRABALHO. Num protocolo com pausas, a alternância
    # destrói o R² global — mas se cada bloco de trabalho for mais intenso que o
    # anterior, trata-se de degraus incrementais e não de intervalos repetidos.
    r2_trabalho, subida_trabalho = 0.0, 0.0
    trabalho = ss >= modo_alto * 0.6
    if trabalho.sum() > 60:
        tt, st_ = t[trabalho], ss[trabalho]
        if np.ptp(tt) > 0:
            ct = np.polyfit(tt, st_, 1)
            pt = np.polyval(ct, tt)
            sst_t = np.sum((st_ - st_.mean()) ** 2)
            r2_trabalho = float(1 - np.sum((st_ - pt) ** 2) / sst_t) if sst_t > 0 else 0.0
            subida_trabalho = float(ct[0] * 60)

    # ── Classificação ────────────────────────────────────────────────────────
    if amp_rel < 0.15:
        tipo = 'continuo'
        motivo = f'intensidade estável (amplitude {amp_rel*100:.0f}% do nível)'
    elif frac_baixo > 0.12:
        # Há quedas claras para intensidade baixa → protocolo com recuperações.
        # Se os blocos de trabalho sobem ao longo do tempo, são degraus
        # incrementais; caso contrário, intervalos repetidos à mesma intensidade.
        if r2_trabalho > 0.30 and subida_trabalho > 0:
            tipo = 'degraus'
            motivo = (f'{frac_baixo*100:.0f}% em recuperação, blocos de trabalho '
                      f'a subir {subida_trabalho:.1f}/min (R²={r2_trabalho:.2f})')
        else:
            tipo = 'intervalos'
            motivo = (f'{frac_baixo*100:.0f}% do tempo em recuperação, sem '
                      f'progressão clara entre blocos (R²={r2_trabalho:.2f})')
    elif r2_linear > 0.70 and subida_por_min > 0 and frac_planos < 0.35:
        tipo = 'rampa'
        motivo = (f'subida contínua de {subida_por_min:.1f}/min, '
                  f'R²={r2_linear:.2f}, sem patamares')
    elif r2_linear > 0.60 and subida_por_min > 0 and frac_planos >= 0.35:
        tipo = 'degraus'
        motivo = (f'subida em patamares ({frac_planos*100:.0f}% do tempo plano), '
                  f'{subida_por_min:.1f}/min')
    elif frac_sobe > 0.55 and amp_rel > 0.3 and frac_planos < 0.35:
        tipo = 'rampa'
        motivo = f'intensidade sobe em {frac_sobe*100:.0f}% do tempo'
    else:
        tipo = 'degraus'
        motivo = f'variação em patamares (R²={r2_linear:.2f})'

    # Método de limiar recomendado por tipo
    _METODOS = {
        'rampa': 'breakpoint contínuo (double-linear) sobre toda a rampa',
        'degraus': 'breakpoint sobre o estado estacionário de cada degrau',
        'intervalos': 'estabilidade do SmO₂ dentro de cada intervalo',
        'continuo': 'sem limiares — sessão de intensidade única',
        'indefinido': '—',
    }

    return {
        'tipo': tipo,
        'motivo': motivo,
        'metodo_recomendado': _METODOS[tipo],
        'duracao_min': round(dur_min, 1),
        'amplitude_rel': round(amp_rel, 2),
        'r2_linear': round(r2_linear, 2),
        'subida_por_min': round(subida_por_min, 1),
        'frac_recuperacao': round(frac_baixo, 2),
        'frac_planos': round(frac_planos, 2),
        'unidade': 'W' if col == colunas.get('power') else 'bpm',
    }


# ══════════════════════════════════════════════════════════════════════════════
# HRVT1c — ponto médio INDIVIDUAL (Rogers/Fleitas-Paniagua/Murias, IJSPP 2024)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_hrvt1c(serie_dfa1, df_metricas=None, colunas=None,
                    janela_inicial_frac=0.35, sd_limite=3.0):
    """
    HRVT1 "custom" — corrige o viés do alpha1=0.75 fixo usando o ponto médio
    INDIVIDUAL de cada atleta.

    Base (IJSPP 2024): o alpha1=0.75 assume que toda a gente parte de ~1.0 no
    início do esforço. Mas há quem comece em 1.5 ou mais. Nesses casos o
    "ponto médio entre bem correlacionado e não-correlacionado" não é 0.75 —
    é (max_inicial + 0.5) / 2.

    No estudo, o max inicial médio foi 1.52 e o alvo calculado 1.01. A correcção
    eliminou o viés face ao GET: de +16 bpm (α1=0.75) para +2 bpm (individual),
    e reduziu os limites de concordância de ±35 para ±26 bpm.

    janela_inicial_frac : fracção inicial do esforço onde procurar o alpha1 máximo.
    sd_limite : o máximo tem de estar dentro deste nº de SD da média local
        (evita apanhar um pico de artefacto).

    Devolve dict com o alvo individual, o limiar em FC/potência, e comparação
    com o método fixo — ou {'erro': ...}.
    """
    if serie_dfa1 is None or len(serie_dfa1) < 10:
        return {'erro': 'série DFA-α1 insuficiente'}

    s = serie_dfa1.dropna(subset=['dfa1', 'fc_media']).copy().sort_values('tempo_s')
    n_ini = max(int(len(s) * janela_inicial_frac), 5)
    inicial = s.head(n_ini)

    # Máximo do início, mas filtrado: tem de estar dentro de sd_limite da média
    # local (janela de ~45 s como no estudo), para não apanhar artefactos.
    _loc = inicial['dfa1'].rolling(9, min_periods=3, center=True).mean()
    _sd = inicial['dfa1'].rolling(9, min_periods=3, center=True).std()
    _ok = inicial['dfa1'] <= (_loc + sd_limite * _sd.fillna(0))
    cand = inicial[_ok] if _ok.sum() >= 3 else inicial
    max_inicial = float(cand['dfa1'].max())

    # Ponto médio individual entre o máximo inicial e 0.5 (não-correlacionado)
    alvo_c = (max_inicial + DFA1_HRVT2) / 2.0

    r_c = calcular_hrvt(serie_dfa1, df_metricas=df_metricas, colunas=colunas,
                        alvo=alvo_c, janela_ajuste=(DFA1_HRVT2, max(max_inicial, 1.0)))
    if 'erro' in r_c:
        return {'erro': r_c['erro'], 'max_inicial': max_inicial, 'alvo': alvo_c}

    # Para comparação: o método fixo de 0.75
    r_s = calcular_hrvt(serie_dfa1, df_metricas=df_metricas, colunas=colunas,
                        alvo=DFA1_HRVT1)

    r_c.update({
        'max_inicial_dfa1': round(max_inicial, 2),
        'alvo_individual': round(alvo_c, 2),
        'fc_metodo_fixo': (r_s.get('fc') if 'erro' not in r_s else None),
        'diferenca_vs_fixo': (round(r_c['fc'] - r_s['fc'], 1)
                              if 'erro' not in r_s and r_s.get('fc') else None),
        'metodo': 'HRVT1c (ponto médio individual)',
    })
    return r_c


# ══════════════════════════════════════════════════════════════════════════════
# HRVT2 SUBMÁXIMO — previsão sem chegar à exaustão (Rogers et al., JSCR 2025)
# ══════════════════════════════════════════════════════════════════════════════

def hrvt2_submaximo(serie_dfa1, df_metricas=None, colunas=None,
                    janela=(0.75, 1.5), min_pontos=8):
    """
    Prevê o HRVT2 usando APENAS dados submáximos, extrapolando a recta.

    Base (JSCR 2025): a trajectória do alpha1 é aproximadamente linear entre 1.5
    e 0.5. Basta ajustar a recta no troço 1.5→0.75 (que se atinge sem sair da
    zona 2) e extrapolar até 0.5 para prever o HRVT2 — sem ter de fazer uma rampa
    até à exaustão.

    Vantagem prática: o teste pode repetir-se com frequência, sem impacto no
    treino nem na recuperação.

    Cuidados do estudo (verificados aqui e devolvidos em 'avisos'):
      • a recta tem de ser inequívoca — se o alpha1 ondula (desce, sobe, desce),
        o resultado não é de confiança
      • o atleta tem de estar fresco; feito no dia seguinte a um esforço
        exaustivo, dá resultados errados por supressão autonómica

    Devolve dict com a previsão e o diagnóstico de qualidade.
    """
    if serie_dfa1 is None or len(serie_dfa1) < min_pontos:
        return {'erro': 'série DFA-α1 insuficiente'}

    s = serie_dfa1.dropna(subset=['dfa1', 'fc_media']).copy()
    lo, hi = janela
    sub = s[(s['dfa1'] >= lo) & (s['dfa1'] <= hi)]
    if len(sub) < min_pontos:
        return {'erro': f'poucos pontos na janela submáxima {lo}-{hi} (n={len(sub)})'}

    x = sub['fc_media'].values.astype(float)
    y = sub['dfa1'].values.astype(float)
    if np.ptp(x) < 5:
        return {'erro': 'variação de FC insuficiente na janela submáxima'}

    coef = np.polyfit(x, y, 1)
    y_pred = np.polyval(coef, x)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - y_pred) ** 2) / sst) if sst > 0 else np.nan
    if abs(coef[0]) < 1e-12:
        return {'erro': 'declive nulo'}

    fc_prev = float((DFA1_HRVT2 - coef[1]) / coef[0])

    # Quanto se extrapolou para lá dos dados
    extrapolacao_bpm = float(fc_prev - x.max())

    # Ondulação: o alpha1 deve descer de forma monótona. Contam-se as inversões
    # de sentido na série suavizada — muitas inversões = recta equívoca.
    _sm = s.sort_values('tempo_s')['dfa1'].rolling(5, min_periods=2, center=True).mean()
    _d = np.diff(_sm.dropna().values)
    inversoes = int(np.sum(np.diff(np.sign(_d[np.abs(_d) > 0.005])) != 0))
    ondulacao_pct = inversoes / max(len(_d), 1) * 100

    pot_prev = None
    if df_metricas is not None and colunas and 'power' in colunas and 'heart_rate' in colunas:
        try:
            dm = df_metricas[[colunas['heart_rate'], colunas['power']]].copy()
            dm.columns = ['fc', 'pot']
            dm = dm.apply(pd.to_numeric, errors='coerce').dropna()
            dm = dm[dm['pot'] > 0]
            if len(dm) > 30 and np.ptp(dm['fc'].values) > 5:
                pot_prev = float(np.polyval(np.polyfit(dm['fc'].values,
                                                      dm['pot'].values, 1), fc_prev))
        except Exception:
            pot_prev = None

    avisos = []
    if r2 < 0.7:
        avisos.append(f'recta pouco definida na zona submáxima (R²={r2:.2f})')
    if ondulacao_pct > 25:
        avisos.append(f'o α1 ondula ao longo do esforço ({ondulacao_pct:.0f}% de '
                      'inversões) — o estudo desaconselha confiar no resultado')
    if extrapolacao_bpm > 25:
        avisos.append(f'extrapolação longa ({extrapolacao_bpm:.0f} bpm acima do '
                      'medido) — quanto mais longe, menos fiável')
    _fc_max_sessao = float(s['fc_media'].max()) if len(s) else None
    _fc_ok, _fc_aviso = _checar_fc_plausivel(fc_prev, fc_max_sessao=_fc_max_sessao)
    if not _fc_ok:
        avisos.append(_fc_aviso)

    return {
        'fc': fc_prev,
        'potencia': pot_prev,
        'coef': coef.tolist(),
        'r2': r2,
        'n_pontos': len(sub),
        'janela': janela,
        'fc_max_medida': float(x.max()),
        'extrapolacao_bpm': round(extrapolacao_bpm, 1),
        'ondulacao_pct': round(ondulacao_pct, 1),
        'inversoes': inversoes,
        'avisos': avisos,
        'fiavel': len(avisos) == 0,
        'pontos': sub,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DURABILIDADE — deriva de HR, fB e DFA-alpha1 (Rogers et al., EJAP 2025)
# ══════════════════════════════════════════════════════════════════════════════

def analisar_durabilidade(df, colunas, serie_dfa1=None, lap_stats=None,
                          n_blocos=4):
    """
    Durabilidade / resiliência fisiológica: deterioração das características
    fisiológicas ao longo de uma sessão prolongada.

    Base (EJAP 2025): num esforço constante abaixo do MMSS, os marcadores
    metabólicos (VO2, lactato, glicose) estabilizam — mas **HR e frequência
    respiratória sobem** e o **DFA-alpha1 desce** progressivamente. Essa deriva é
    o sinal de perda de durabilidade, e é repetível entre sessões (ICC 0.73-0.94).

    O estudo usa o método "isotime": divide a sessão em quartos e compara as
    médias de cada quarto — o que normaliza sessões de duração diferente.

    Importante: os três sinais devem ser lidos em conjunto. Alguém pode ter pouca
    deriva de fB mas queda normal do alpha1, e olhar só para um marcador levaria a
    concluir erradamente que não houve degradação.

    Só faz sentido em sessões contínuas/longas; não em intervalos curtos.
    """
    d = df.copy()
    if lap_stats:
        excl = {l['lap_number'] for l in lap_stats if l.get('phase') == 'excluded'}
        if excl:
            d = d[~d['lap_number'].isin(excl)]
    if len(d) < 600:  # menos de 10 min não faz sentido
        return None

    t0, t1 = d['time_seconds'].min(), d['time_seconds'].max()
    dur_min = (t1 - t0) / 60.0
    bordas = np.linspace(t0, t1, n_blocos + 1)

    _METRICAS = [('heart_rate', 'FC', 'sobe'),
                 ('respiration', 'Respiração', 'sobe'),
                 ('resp_enhanced', 'Respiração enh.', 'sobe'),
                 ('power', 'Potência', 'estavel'),
                 ('smo2', 'SmO₂', 'estavel')]

    linhas = []
    for i in range(n_blocos):
        m = (d['time_seconds'] >= bordas[i]) & (d['time_seconds'] < bordas[i + 1])
        sub = d[m]
        if len(sub) < 30:
            continue
        linha = {'bloco': f'Q{i+1}', 'n': len(sub),
                 'inicio_min': round((bordas[i] - t0) / 60, 1)}
        for met, _, _ in _METRICAS:
            if met in colunas and colunas[met] in sub.columns:
                v = pd.to_numeric(sub[colunas[met]], errors='coerce').dropna()
                if len(v) > 10:
                    linha[met] = round(float(v.mean()), 1)
        # DFA-alpha1 do bloco (da série recalculada)
        if serie_dfa1 is not None and len(serie_dfa1) > 0:
            sd = serie_dfa1[(serie_dfa1['tempo_s'] >= bordas[i]) &
                            (serie_dfa1['tempo_s'] < bordas[i + 1])]
            if len(sd) >= 3:
                linha['dfa1'] = round(float(sd['dfa1'].mean()), 3)
        linhas.append(linha)

    if len(linhas) < 3:
        return None

    tabela = pd.DataFrame(linhas)

    # Deriva de cada marcador: variação do último bloco face ao primeiro
    derivas = {}
    for met, nome, esperado in _METRICAS + [('dfa1', 'DFA-α1', 'desce')]:
        if met not in tabela.columns:
            continue
        v = tabela[met].dropna()
        if len(v) < 3:
            continue
        delta = float(v.iloc[-1] - v.iloc[0])
        base = float(v.iloc[0])
        pct = (delta / abs(base) * 100) if abs(base) > 1e-9 else None
        derivas[met] = {
            'nome': nome,
            'inicio': base,
            'fim': float(v.iloc[-1]),
            'delta': round(delta, 2),
            'delta_pct': round(pct, 1) if pct is not None else None,
            'esperado': esperado,
        }

    # Veredicto: contam-se os sinais de degradação
    sinais = 0
    detalhe = []
    if 'heart_rate' in derivas and derivas['heart_rate']['delta'] > 3:
        sinais += 1
        detalhe.append(f"FC subiu {derivas['heart_rate']['delta']:.0f} bpm")
    for _r in ('respiration', 'resp_enhanced'):
        if _r in derivas and derivas[_r]['delta'] > 2:
            sinais += 1
            detalhe.append(f"respiração subiu {derivas[_r]['delta']:.0f} rpm")
            break
    if 'dfa1' in derivas and derivas['dfa1']['delta'] < -0.1:
        sinais += 1
        detalhe.append(f"DFA-α1 desceu {abs(derivas['dfa1']['delta']):.2f}")

    if sinais >= 2:
        veredicto, cor = 'Perda de durabilidade evidente', '#e74c3c'
    elif sinais == 1:
        veredicto, cor = 'Sinais ligeiros de deriva', '#f39c12'
    else:
        veredicto, cor = 'Durabilidade mantida', '#27ae60'

    return {
        'tabela': tabela,
        'derivas': derivas,
        'n_sinais': sinais,
        'detalhe': detalhe,
        'veredicto': veredicto,
        'cor': cor,
        'duracao_min': round(dur_min, 1),
        'n_blocos': len(linhas),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AVALIAÇÃO DE FIABILIDADE — critérios explícitos dos estudos publicados
# ══════════════════════════════════════════════════════════════════════════════

# Limites de concordância reportados na literatura (para calibrar expectativas)
LOA_LITERATURA = {
    'HRVT2_vs_RCP': '±15-21 bpm (MSSE 2024, IJSPP 2024)',
    'HRVT1c_vs_GET': '±26 bpm (IJSPP 2024)',
    'HRVT1_fixo_vs_GET': '±35 bpm, viés +16 bpm (MSSE 2024)',
    'NIRS_BP_vs_RCP': 'variável; combo reduz o erro individual (JSCR 2024)',
}


def avaliar_fiabilidade(resultado):
    """
    Avalia a fiabilidade dos limiares estimados, segundo os critérios explícitos
    da literatura. Devolve um semáforo global e a lista de critérios verificados.

    Critérios (com a fonte):
      • artefactos HRV ≤5%          — limite usado em todos os estudos do grupo
      • ajuste inequívoco (R²)      — JSCR 2025: "recta de regressão inequívoca"
      • sem ondulação do alpha1     — JSCR 2025: "se o α1 ondula, não confiar"
      • alpha1 atinge o alvo        — senão o valor é extrapolação
      • concordância entre métodos  — JSCR 2024: combo NIRS+HRV reduz erro
      • duração/estrutura adequadas — janelas de 2 min precisam de tempo

    O objectivo não é dar uma nota, é dizer ao utilizador **em que pode confiar**.
    """
    criterios = []

    def _add(nome, estado, detalhe, fonte=''):
        criterios.append({'criterio': nome, 'estado': estado,
                          'detalhe': detalhe, 'fonte': fonte})

    # 1. Qualidade do sinal HRV
    q = resultado.get('dfa1_qualidade')
    if q:
        pct = q.get('pct_artefactos', 0)
        if pct <= 5:
            _add('Artefactos HRV', 'ok', f'{pct:.1f}% (limite: 5%)',
                 'critério usado em todos os estudos do grupo Murias/Rogers')
        else:
            _add('Artefactos HRV', 'mau',
                 f'{pct:.1f}% — acima do limite de 5%',
                 'acima deste valor os estudos excluem o participante')
    elif resultado.get('rr_info') is None:
        _add('Intervalos RR', 'ausente',
             'o ficheiro não contém RR — DFA-α1 não pode ser recalculado', '')

    # 2. Ondulação e ajuste do alpha1
    sub = resultado.get('hrvt2_submax')
    if sub and 'erro' not in sub:
        if sub.get('ondulacao_pct', 0) <= 25:
            _add('Trajectória do α1', 'ok',
                 f"desce de forma consistente ({sub['ondulacao_pct']:.0f}% de inversões)",
                 'JSCR 2025: a recta deve ser inequívoca')
        else:
            _add('Trajectória do α1', 'mau',
                 f"ondula ao longo do esforço ({sub['ondulacao_pct']:.0f}% de inversões)",
                 'JSCR 2025: "se há ondulação, não confiar no teste"')

    # 3. HRVT2 atingido ou extrapolado
    h2 = resultado.get('hrvt2')
    if h2 and 'erro' not in h2:
        if h2.get('fiavel'):
            _add('HRVT2 (α1=0.50)', 'ok',
                 f"R²={h2.get('r2', 0):.2f}, {h2.get('pct_abaixo_alvo', 0):.0f}% "
                 "das janelas abaixo do alvo", 'MSSE 2024')
        else:
            _add('HRVT2 (α1=0.50)', 'mau',
                 '; '.join(h2.get('avisos', [])), 'MSSE 2024')

    # 3b. Amplitude do sinal NIRS
    # Um sensor bem colocado mostra quedas de 20-40 pontos de SmO2 num teste
    # incremental. Amplitudes pequenas indicam quase sempre má colocação
    # (tecido adiposo por cima, sensor solto, ou luz ambiente a entrar) — e o
    # breakpoint pode ter um R² alto mesmo assim, porque ajusta bem a uma recta
    # quase plana. Daí verificar a amplitude independentemente do ajuste.
    _w = [l for l in resultado.get('lap_stats', []) if l.get('phase') == 'work']
    _sm = [l['avg_smo2'] for l in _w if 'avg_smo2' in l]
    if len(_sm) >= 3:
        _amp = max(_sm) - min(_sm)
        if _amp >= 15:
            _add('Amplitude SmO₂', 'ok',
                 f'{_amp:.0f} pontos entre o degrau mais fácil e o mais duro',
                 'um sensor bem colocado mostra 20-40 pontos num incremental')
        elif _amp >= 8:
            _add('Amplitude SmO₂', 'aviso',
                 f'apenas {_amp:.0f} pontos — sinal com pouca dinâmica',
                 'verifica a colocação do sensor')
        else:
            _add('Amplitude SmO₂', 'mau',
                 f'apenas {_amp:.0f} pontos — o sensor pode estar mal colocado '
                 '(tecido adiposo, mal fixado, ou luz ambiente)',
                 'sem amplitude não há breakpoint fisiológico a detectar')

    # 4. Breakpoint NIRS
    bp = resultado.get('bp_continuo')
    if bp:
        if bp.get('r2', 0) >= 0.8 and bp.get('coerente_recto_femoral'):
            _add('Breakpoint SmO₂', 'ok',
                 f"R²={bp['r2']:.2f}, padrão coerente", 'JSCR 2024')
        elif bp.get('r2', 0) >= 0.8:
            _add('Breakpoint SmO₂', 'aviso',
                 f"R²={bp['r2']:.2f} mas padrão inesperado: {bp.get('padrao')}",
                 'JSCR 2024')
        else:
            _add('Breakpoint SmO₂', 'mau',
                 f"ajuste fraco (R²={bp.get('r2', 0):.2f})", 'JSCR 2024')

    # 5. Concordância entre métodos independentes
    cb = resultado.get('combo')
    if cb and cb.get('n_metodos', 0) >= 2:
        if cb['estado'] == 'concordante':
            _add('Concordância NIRS↔HRV', 'ok',
                 f"divergência de {cb['divergencia_pct']:.0f}%",
                 'JSCR 2024: métodos independentes que concordam dão mais confiança')
        else:
            _add('Concordância NIRS↔HRV', 'aviso',
                 f"divergência de {cb['divergencia_pct']:.0f}% entre métodos",
                 'JSCR 2024: divergência grande sugere problema num dos sinais')
    elif cb and cb.get('n_metodos') == 1:
        _add('Concordância NIRS↔HRV', 'aviso',
             'só um método disponível — sem validação cruzada',
             'JSCR 2024: o combo reduz o erro individual de ~25% para 14%')

    # 6. Estrutura do protocolo
    proto = resultado.get('protocolo')
    if proto:
        t = proto.get('tipo')
        if t in ('rampa', 'degraus'):
            _add('Protocolo', 'ok', f"{t} — adequado a estimativa de limiares", '')
        elif t == 'intervalos':
            _add('Protocolo', 'aviso',
                 'intervalos repetidos — os métodos de limiar assumem intensidade '
                 'progressiva', '')
        else:
            _add('Protocolo', 'aviso',
                 f'{t} — sem progressão de intensidade para estimar limiares', '')

    # ── Semáforo global ──────────────────────────────────────────────────────
    n_mau = sum(1 for c in criterios if c['estado'] == 'mau')
    n_aviso = sum(1 for c in criterios if c['estado'] == 'aviso')
    n_ok = sum(1 for c in criterios if c['estado'] == 'ok')

    if n_mau == 0 and n_aviso <= 1:
        nivel, cor, texto = ('alta', '#27ae60',
                             'Resultados fiáveis — os critérios da literatura estão cumpridos.')
    elif n_mau <= 1:
        nivel, cor, texto = ('média', '#f39c12',
                             'Fiabilidade moderada — usa os valores como orientação, '
                             'não como referência definitiva.')
    else:
        nivel, cor, texto = ('baixa', '#e74c3c',
                             'Fiabilidade baixa — vários critérios falharam. '
                             'Recomenda-se repetir o teste antes de usar estes números.')

    return {
        'nivel': nivel, 'cor': cor, 'texto': texto,
        'criterios': criterios,
        'n_ok': n_ok, 'n_aviso': n_aviso, 'n_mau': n_mau,
        'loa': LOA_LITERATURA,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HHb — hemoglobina desoxigenada derivada de SmO2 e THb
# É a métrica que a literatura NIRS usa (Murias et al.), não o SmO2 directamente.
# ══════════════════════════════════════════════════════════════════════════════

def derivar_hhb(df, colunas):
    """
    Deriva o HHb (hemoglobina desoxigenada) a partir do SmO2 e do THb.

    Relação: o SmO2 é a percentagem de hemoglobina oxigenada no volume medido,
    e o THb é a hemoglobina total. Logo:

        HHb = THb × (1 − SmO2/100)
        O2Hb = THb × (SmO2/100)

    PORQUÊ ISTO IMPORTA: os estudos de NIRS (Murias, Fleitas-Paniagua) analisam
    os breakpoints no **HHb**, não no SmO2. As duas métricas são inversamente
    relacionadas, mas não são equivalentes: o SmO2 é uma proporção (satura), o
    HHb é uma quantidade absoluta e mantém amplitude dinâmica útil a intensidades
    altas — precisamente onde o SmO2 começa a achatar.

    Se o THb não estiver disponível, devolve None (não se pode derivar).

    Devolve (df com colunas HHb/O2Hb, colunas actualizado).
    """
    if 'smo2' not in colunas or 'thb' not in colunas:
        return df, colunas

    d = df.copy()
    smo2 = pd.to_numeric(d[colunas['smo2']], errors='coerce')
    thb = pd.to_numeric(d[colunas['thb']], errors='coerce')
    if smo2.notna().sum() < 30 or thb.notna().sum() < 30:
        return df, colunas

    d['_HHb'] = thb * (1.0 - smo2 / 100.0)
    d['_O2Hb'] = thb * (smo2 / 100.0)
    cols = dict(colunas)
    cols['hhb'] = '_HHb'
    cols['o2hb'] = '_O2Hb'
    return d, cols


# ══════════════════════════════════════════════════════════════════════════════
# MLSS POR INTERVALOS LONGOS — método de comparação abaixo/acima
# (muscleoxygentraining.com 2019/03, baseado em Murias et al.)
# ══════════════════════════════════════════════════════════════════════════════

def mlss_intervalos_longos(df, colunas, lap_stats, ignorar_inicio_s=120,
                           limiar_slope_hhb=0.10, limiar_slope_smo2=-0.5,
                           min_dur_s=180):
    """
    Estima o MLSS comparando blocos de intensidade constante — o método que o
    autor considera MAIS FIÁVEL do que os breakpoints por rampa.

    Fundamento (artigo 2019/03 e Murias et al. 2018):
      • ABAIXO do MLSS: o HHb/SmO2 estabiliza após a transição inicial
      • ACIMA do MLSS: o HHb sobe continuamente (SmO2 desce continuamente),
        acompanhando a acumulação de lactato e a subida da ventilação
      • O MLSS fica ENTRE a intensidade mais alta estável e a mais baixa instável

    Porquê preferir isto a rampas: o artigo é explícito — "a maioria dos métodos
    para estimar a potência do MLSS tem um erro superior a 10 W", e o estudo de
    Murias mostrou que exercitar apenas +10 W acima do MLSS (~3-5%) já provoca
    subida progressiva do lactato e prejudica o desempenho posterior. Um erro de
    rampa maior que 10 W torna a estimativa pouco útil na prática.

    Diferença face a estabilidade_smo2_intervalos(): aqui a análise é
    COMPARATIVA entre blocos e produz um enquadramento do MLSS, em vez de
    classificar cada bloco isoladamente.

    ignorar_inicio_s : segundos iniciais a ignorar em cada bloco (a queda inicial
        é a transição da intensidade anterior, não o comportamento estacionário).
        O artigo observa que o SmO2 só estabiliza a partir do minuto ~2-3.
    min_dur_s : duração mínima de bloco a considerar. Blocos curtos não dão tempo
        para o padrão se manifestar.

    Devolve dict com a tabela por bloco, o enquadramento do MLSS e o diagnóstico.
    """
    # Preferir HHb (o que a literatura usa); cair para SmO2 se não houver THb.
    # Normalmente já vem derivado da preparação; deriva aqui se faltar.
    if 'hhb' not in colunas:
        df, colunas = derivar_hhb(df, colunas)
    usa_hhb = 'hhb' in colunas
    col_sinal = colunas.get('hhb') or colunas.get('smo2')
    if col_sinal is None:
        return None
    col_int = colunas.get('power') or colunas.get('heart_rate')
    if col_int is None:
        return None
    unidade = 'W' if col_int == colunas.get('power') else 'bpm'

    # Em HHb, "instável" = sobe; em SmO2, "instável" = desce
    limiar = limiar_slope_hhb if usa_hhb else limiar_slope_smo2
    nome_sinal = 'HHb' if usa_hhb else 'SmO₂'

    linhas = []
    for l in lap_stats:
        if l.get('phase') != 'work':
            continue
        if l.get('duration', 0) < min_dur_s:
            continue
        d = df[df['lap_number'] == l['lap_number']]
        if len(d) < 60:
            continue
        t0 = d['time_seconds'].iloc[0]
        d = d[d['time_seconds'] >= t0 + ignorar_inicio_s]
        if len(d) < 40:
            continue

        y = pd.to_numeric(d[col_sinal], errors='coerce')
        t = d['time_seconds']
        m = y.notna()
        if m.sum() < 30:
            continue
        y, t = y[m].values, t[m].values

        coef = np.polyfit(t, y, 1)
        slope_min = float(coef[0] * 60)          # unidades por minuto
        y_pred = np.polyval(coef, t)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum((y - y_pred) ** 2) / sst) if sst > 0 else np.nan

        # Um declive só é credível se a tendência for consistente. Em blocos
        # curtos ou com sinal ruidoso, o ajuste linear pode dar um declive
        # aparente que é apenas ruído — daí exigir um R² mínimo antes de
        # classificar o bloco como instável.
        tendencia_credivel = (not np.isnan(r2)) and r2 >= 0.25

        # Estável: o declive não excede o limiar no sentido "de instabilidade"
        if usa_hhb:
            excede = slope_min >= limiar          # HHb sobe → instável
        else:
            excede = slope_min <= limiar          # SmO2 desce → instável
        estavel = not (excede and tendencia_credivel)

        linhas.append({
            'lap': l['lap_number'],
            'intensidade': l.get('avg_power', l.get('avg_heart_rate')),
            'fc': l.get('avg_heart_rate'),
            f'{nome_sinal}_inicio': round(float(y[0]), 2),
            f'{nome_sinal}_fim': round(float(y[-1]), 2),
            'delta': round(float(y[-1] - y[0]), 2),
            'slope_por_min': round(slope_min, 3),
            'r2_tendencia': round(r2, 2) if not np.isnan(r2) else None,
            'comportamento': ('estável' if estavel else 'deriva contínua'),
            'tendencia_credivel': bool(tendencia_credivel),
            'estavel': estavel,
            'dur_analisada_s': int(t[-1] - t[0]),
        })

    if len(linhas) < 2:
        return None

    tabela = pd.DataFrame(linhas).sort_values('intensidade').reset_index(drop=True)
    estaveis = tabela[tabela['estavel']]
    instaveis = tabela[~tabela['estavel']]

    lim_inf = float(estaveis['intensidade'].max()) if len(estaveis) else None
    lim_sup = float(instaveis['intensidade'].min()) if len(instaveis) else None
    fc_inf = (float(estaveis.loc[estaveis['intensidade'].idxmax(), 'fc'])
              if len(estaveis) and estaveis['fc'].notna().any() else None)
    fc_sup = (float(instaveis.loc[instaveis['intensidade'].idxmin(), 'fc'])
              if len(instaveis) and instaveis['fc'].notna().any() else None)

    if lim_inf is not None and lim_sup is not None and lim_inf < lim_sup:
        estimativa = (lim_inf + lim_sup) / 2.0
        fc_est = ((fc_inf + fc_sup) / 2.0
                  if fc_inf is not None and fc_sup is not None else None)
        largura = lim_sup - lim_inf
        estado = 'enquadrado'
        # O artigo mostra que ±10 W já altera a resposta fisiológica
        precisao = ('boa' if largura <= 20 else
                    'moderada' if largura <= 40 else 'grosseira')
    elif lim_inf is not None and lim_sup is not None:
        estimativa, fc_est, largura, precisao = None, None, None, None
        estado = 'inconsistente'
    elif lim_sup is not None:
        estimativa, fc_est, largura, precisao = None, None, None, None
        estado = 'abaixo_do_testado'
    else:
        estimativa, fc_est, largura, precisao = None, None, None, None
        estado = 'acima_do_testado'

    return {
        'tabela': tabela,
        'sinal': nome_sinal,
        'usa_hhb': usa_hhb,
        'mlss_entre': (lim_inf, lim_sup),
        'mlss_estimado': estimativa,
        'mlss_fc': fc_est,
        'largura_janela': largura,
        'precisao': precisao,
        'estado': estado,
        'unidade': unidade,
        'n_blocos': len(tabela),
        'n_estaveis': len(estaveis),
        'n_instaveis': len(instaveis),
        'ignorar_inicio_s': ignorar_inicio_s,
        'limiar_slope': limiar,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSÃO POTÊNCIA ↔ FC
# A FC dos limiares é mais estável entre protocolos do que a potência
# (Physiological Reports 2023), por isso convém reportar ambas.
# ══════════════════════════════════════════════════════════════════════════════

def _relacao_pot_fc(df, colunas, lap_stats=None):
    """
    Ajusta a relação potência↔FC desta sessão, para converter limiares entre as
    duas unidades.

    Usa apenas os laps de trabalho (nas recuperações a FC desce com atraso e a
    relação distorce-se). Devolve dict com os coeficientes nos dois sentidos,
    ou None se não houver dados suficientes.
    """
    if 'power' not in colunas or 'heart_rate' not in colunas:
        return None

    # Preferir as MÉDIAS DE ESTADO ESTACIONÁRIO por lap (um ponto por degrau).
    # Usar todos os pontos a 1 Hz distorce a recta: no início de cada degrau a FC
    # ainda está a subir para o seu valor estacionário, o que achata a relação e
    # subestima a FC nas potências altas (erros de 5-13 bpm nos testes).
    usou_medias = False
    if lap_stats:
        pares = [(l['avg_power'], l['avg_heart_rate'])
                 for l in lap_stats
                 if l.get('phase') == 'work'
                 and l.get('avg_power') is not None
                 and l.get('avg_heart_rate') is not None
                 and l['avg_power'] > 0]
        if len(pares) >= 3:
            dm = pd.DataFrame(pares, columns=['pot', 'fc'])
            usou_medias = True

    if not usou_medias:
        d = df.copy()
        if lap_stats:
            laps_ok = {l['lap_number'] for l in lap_stats if l.get('phase') == 'work'}
            if laps_ok:
                d = d[d['lap_number'].isin(laps_ok)]
        dm = d[[colunas['power'], colunas['heart_rate']]].copy()
        dm.columns = ['pot', 'fc']
        dm = dm.apply(pd.to_numeric, errors='coerce').dropna()
        dm = dm[(dm['pot'] > 0) & (dm['fc'] > 30)]

    if len(dm) < 3 or np.ptp(dm['pot'].values) < 20 or np.ptp(dm['fc'].values) < 5:
        return None

    x, y = dm['pot'].values, dm['fc'].values
    c_pf = np.polyfit(x, y, 1)          # potência → FC
    c_fp = np.polyfit(y, x, 1)          # FC → potência
    y_pred = np.polyval(c_pf, x)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - y_pred) ** 2) / sst) if sst > 0 else np.nan

    return {
        'coef_pot_fc': c_pf.tolist(),
        'coef_fc_pot': c_fp.tolist(),
        'r2': r2,
        'n': len(dm),
        'usou_medias_por_lap': usou_medias,
        'pot_min': float(x.min()), 'pot_max': float(x.max()),
        'fc_min': float(y.min()), 'fc_max': float(y.max()),
    }


def pot_para_fc(potencia, relacao):
    """Converte potência em FC usando a relação da sessão."""
    if relacao is None or potencia is None:
        return None
    return float(np.polyval(relacao['coef_pot_fc'], potencia))


def fc_para_pot(fc, relacao):
    """Converte FC em potência usando a relação da sessão."""
    if relacao is None or fc is None:
        return None
    return float(np.polyval(relacao['coef_fc_pot'], fc))


def resumir_zonas(resultado):
    """
    Consolida todos os métodos numa proposta de zonas de treino, em FC e potência.

    Limiar BAIXO (Z1→Z2): prioridade ao HRVT1c (ponto médio individual), que a
    literatura mostra ter menos viés que o α1=0.75 fixo.

    Limiar ALTO (Z2→Z3): prioridade ao Combo (HRVT2 + NIRS), depois ao método dos
    intervalos longos, depois ao breakpoint isolado — pela ordem de fiabilidade
    que os estudos estabelecem.

    Devolve dict com os limiares nas duas unidades, a origem de cada um, e a
    lista de todas as estimativas disponíveis para comparação.
    """
    rel = resultado.get('relacao_pot_fc')

    def _par(pot=None, fc=None):
        """Completa o par (potência, FC) a partir do que existir."""
        if pot is None and fc is not None:
            pot = fc_para_pot(fc, rel)
        elif fc is None and pot is not None:
            fc = pot_para_fc(pot, rel)
        return pot, fc

    # ── Limiar baixo ─────────────────────────────────────────────────────────
    baixo = None
    h1c = resultado.get('hrvt1c')
    if h1c and 'erro' not in h1c and h1c.get('fiavel', True) and h1c.get('fc'):
        p, f = _par(h1c.get('potencia'), h1c['fc'])
        baixo = {'pot': p, 'fc': f, 'origem': 'HRVT1c (ponto médio individual)',
                 'fiavel': True}
    elif h1c and 'erro' not in h1c and h1c.get('fc'):
        p, f = _par(h1c.get('potencia'), h1c['fc'])
        baixo = {'pot': p, 'fc': f, 'origem': 'HRVT1c (com reservas)',
                 'fiavel': False}
    else:
        ld = resultado.get('limiar_dfa1')
        if ld and 'limiares' in ld:
            v = ld['limiares'].get(0.70)
            if v and not v.get('extrapolado'):
                if ld.get('unidade') == 'W':
                    p, f = _par(pot=v['intensidade'])
                else:
                    p, f = _par(fc=v['intensidade'])
                baixo = {'pot': p, 'fc': f,
                         'origem': 'DFA-α1 = 0.70 (método fixo)', 'fiavel': False}

    # ── Limiar alto ──────────────────────────────────────────────────────────
    alto = None
    alternativas = []

    cb = resultado.get('combo')
    if cb and cb.get('n_metodos', 0) >= 2 and cb.get('estado') == 'concordante':
        p, f = _par(pot=cb['combo'])
        alto = {'pot': p, 'fc': f, 'origem': 'Combo HRVT2 + NIRS', 'fiavel': True}

    mi = resultado.get('mlss_intervalos')
    if mi and mi.get('mlss_estimado'):
        p, f = _par(mi['mlss_estimado'], mi.get('mlss_fc'))
        alternativas.append({'pot': p, 'fc': f,
                             'origem': f"MLSS intervalos longos ({mi['sinal']})",
                             'fiavel': mi.get('precisao') == 'boa'})
        if alto is None:
            alto = dict(alternativas[-1])

    for _k, _lbl in (('bp_continuo', 'Breakpoint SmO₂'), ('bp_hhb', 'Breakpoint HHb')):
        bp = resultado.get(_k)
        if bp and bp.get('breakpoint'):
            if bp.get('unidade') == 'W':
                p, f = _par(pot=bp['breakpoint'])
            else:
                p, f = _par(fc=bp['breakpoint'])
            _fi = bp.get('r2', 0) >= 0.8
            alternativas.append({'pot': p, 'fc': f, 'origem': _lbl, 'fiavel': _fi})
            if alto is None:
                alto = dict(alternativas[-1])

    h2 = resultado.get('hrvt2')
    if h2 and 'erro' not in h2 and h2.get('fc'):
        p, f = _par(h2.get('potencia'), h2['fc'])
        alternativas.append({'pot': p, 'fc': f, 'origem': 'HRVT2 (DFA-α1 = 0.50)',
                             'fiavel': bool(h2.get('fiavel'))})
        if alto is None and h2.get('fiavel'):
            alto = dict(alternativas[-1])

    # Coerência: o limiar baixo tem de ficar abaixo do alto
    coerente = None
    if baixo and alto and baixo.get('fc') and alto.get('fc'):
        coerente = baixo['fc'] < alto['fc']

    return {
        'baixo': baixo,
        'alto': alto,
        'alternativas': alternativas,
        'relacao_pot_fc': rel,
        'coerente': coerente,
    }
