# 🎹 py_midi2song — Player MIDI da riga di comando (Python)

`py_midi2song` è un player **standalone** per file `.mid` / `.midi`, scritto in
**Python 3.9+**, che utilizza le librerie `mido` e `python-rtmidi` per
riprodurre eventi MIDI reali attraverso una porta MIDI OUT del sistema (es. su
Windows, "Microsoft GS Wavetable Synth"). Il progetto è ora impacchettato come
modulo Python installabile e fornisce un comando `py-midi2song` pronto all'uso.

---

## 🚀 Funzionalità principali

✅ Riproduce file MIDI reali su una porta MIDI OUT di sistema  
✅ Supporta tempo variabile (`set_tempo`) e tempo globale (`--tempo-scale`)  
✅ Supporta start offset e fine (`--start-at-seconds`, `--end-at-seconds`)  
✅ Filtro per tracce e canali (`--tracks`, `--channels`)  
✅ Controllo del gain sulla velocity (`--gain`)  
✅ Gestione corretta delle NoteOn/NoteOff, anche in caso di interruzione (`Ctrl+C`)  
✅ Visualizzazioni testuali opzionali (`--viz` con modalità *grid* o *lanes*)  
✅ Logging dettagliato (INFO / DEBUG)  
✅ Compatibile con **Windows**, **Linux** e **macOS**

---

## 📦 Installazione

Assicurati di avere **Python 3.9+** installato, poi esegui:

```bash
pip install py-midi2song
```

Il pacchetto installerà automaticamente `mido` e `python-rtmidi`. Su Windows è
consigliato attivare o installare un sintetizzatore MIDI come:

- **Microsoft GS Wavetable Synth** (già incluso in Windows)
- Oppure un sintetizzatore virtuale come **VirtualMIDISynth** o
  **loopMIDI + VST**

Per chi preferisce lavorare direttamente dal codice sorgente è disponibile anche
lo script di compatibilità:

```bash
python midi2song.py --help
```

---

## 🧩 Utilizzo

### Elenca le porte MIDI disponibili

```bash
python midi2song.py --list-ports
```

Esempio output:

```
[0] Microsoft GS Wavetable Synth
[1] loopMIDI Port 1
```

---

### Riproduci un file MIDI

```bash
python midi2song.py --midi song.mid
```

---

### Seleziona una porta specifica (per nome o indice)

```bash
python midi2song.py --midi song.mid --port "Microsoft GS"
# oppure
python midi2song.py --midi song.mid --port 0
```

---

### Riproduzione parziale o più lenta/veloce

```bash
python midi2song.py --midi song.mid --start-at-seconds 12.5 --tempo-scale 0.9
```

---

### Filtrare tracce e canali

```bash
# Solo tracce 0 e 2
python midi2song.py --midi song.mid --tracks include:0,2

# Tutti i canali tranne il 9 (percussioni)
python midi2song.py --midi song.mid --channels exclude:9
```

---

### Regolare la dinamica

```bash
# Gain 0.5 => più morbido
python midi2song.py --midi song.mid --gain 0.5
```

---

## ⚙️ Argomenti CLI completi

| Argomento | Tipo | Default | Descrizione |
|------------|-------|----------|-------------|
| `--midi PATH` | obbligatorio | — | Percorso del file MIDI da riprodurre |
| `--list-ports` | flag | — | Mostra le porte MIDI OUT disponibili e termina |
| `--port NAME/INDICE` | opzionale | auto | Seleziona la porta per nome o indice |
| `--start-at-seconds FLOAT` | float | 0.0 | Avvia la riproduzione da questo tempo |
| `--end-at-seconds FLOAT` | float | — | Interrompe la riproduzione a questo tempo |
| `--tempo-scale FLOAT` | float | 1.0 | Scala globale del tempo (0.8 = più lento) |
| `--gain FLOAT` | float | 1.0 | Moltiplicatore di velocity (clamp 1–127) |
| `--tracks all\|include:CSV\|exclude:CSV` | string | all | Filtra per indici di traccia (0-based) |
| `--channels all\|include:CSV\|exclude:CSV` | string | all | Filtra per canali MIDI (0–15) |
| `--ignore-meta` | flag | — | Ignora meta-eventi non necessari |
| `--log-level {INFO,DEBUG,WARNING}` | string | INFO | Livello di dettaglio log |
| `--viz` | flag | — | Abilita la visualizzazione testuale |
| `--viz-mode {grid,lanes}` | string | grid | Seleziona la modalità di visualizzazione |
| `--viz-refresh-hz FLOAT` | float | 30.0 | Frequenza di aggiornamento della visualizzazione |
| `--viz-base-note INT` | int | 60 | Nota base per *grid* e *lanes* |
| `--viz-lanes {5,7}` | int | 7 | Numero di corsie per la modalità *lanes* |
| `--viz-window-seconds FLOAT` | float | 4.0 | Finestra futura visualizzata nella modalità *lanes* |
| `--viz-height INT` | int | 20 | Altezza (righe) per la modalità *lanes* |
| `--viz-beats` | flag | — | Mostra la beat grid/metronomo nelle visualizzazioni |

---

## 🧠 Log e debug

Esempio log:

```
INFO: File: 'song.mid' | ticks_per_beat=480 | tracce=2 | eventi_canale=356 | meta=12 | durata=31.09s
INFO: Porta selezionata (auto): Microsoft GS Wavetable Synth
INFO: Riproduzione: start=0.000s
```

In modalità debug:
```
python py_midi2song.py --midi song.mid --log-level DEBUG
```

---

## 🧩 Architettura del codice

Struttura principale del pacchetto installabile:

```
py_midi2song.py
├── list_output_ports()
├── select_output_port()
├── build_timeline()
├── apply_filters()
├── seek_index()
├── play_events()
└── main()
```

È inoltre disponibile un wrapper di compatibilità (`py_midi2song.py`) per eseguire
la CLI direttamente dalla directory del repository senza installazione.

Ogni evento MIDI è rappresentato da una dataclass:

```python
@dataclass
class Event:
    time_s: float
    message: mido.Message
    track_index: int
    channel: Optional[int]
```

---

## ⚠️ Casi limite gestiti

- File con più `set_tempo`
- Tracce solo meta-eventi
- Canale 10 (percussioni)
- Note sovrapposte (NoteOn/NoteOff coerenti)
- MIDI Type 0 e Type 1
- Seek a metà nota (non invia NoteOff senza NoteOn post-seek)
- `Ctrl+C` → chiusura porta + `All Notes Off`

---

## 🧰 Esempi rapidi

```bash
python midi2song.py --midi test.mid
python midi2song.py --midi test.mid --tempo-scale 0.5
python midi2song.py --midi test.mid --gain 0.5
python midi2song.py --midi test.mid --start-at-seconds 10
python midi2song.py --midi test.mid --channels exclude:9
```

---

## 🧩 Compatibilità

- ✅ **Windows** (prioritario)
- ✅ **Linux**
- ✅ **macOS**

Su Windows, viene selezionata automaticamente la porta
**“Microsoft GS Wavetable Synth”** se disponibile.

---

## 🪶 Licenza

MIT License © 2025 — Autore originale: *Andrea Antonio Perrelli*

---

## 🧡 Contributi

Pull requests e issue sono benvenuti.
Ogni miglioramento al timing, alla gestione del seek o alla compatibilità multi-porta è apprezzato.

---

**Enjoy your MIDI playback! 🎶**
