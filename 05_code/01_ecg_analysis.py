#!/usr/bin/env python3
"""
ANALISI ECG BASE - PASSO 1
Calcola Φ cardiaco da dati ECG reali
"""

print("="*70)
print("ANALISI ECG: PASSO 1 - Φ CARDIACO")
print("="*70)

# 1. IMPORTAZIONI
print("\n1. 📦 Importazione librerie...")
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import os
import sys

print("   ✅ Librerie importate")

# 2. CONFIGURAZIONE - PERCORSI CORRETTI
print("\n2. ⚙️  Configurazione percorsi...")
# Siamo in: phi_f_project/05_code/
# I dati ECG sono in: phi_f_project/01_data_ecg/
ECG_DIR = "../../ecg_data"  # Torna indietro di 2 livelli e poi in ecg_data
# OPPURE percorso assoluto
HOME_DIR = os.path.expanduser("~")
ECG_DIR_ABS = os.path.join(HOME_DIR, "equity_study", "ecg_data")

ECG_FILE = "100"
OUTPUT_DIR = "../../phi_f_project/03_analysis"

print(f"   • Home directory: {HOME_DIR}")
print(f"   • Percorso relativo ECG: {ECG_DIR}")
print(f"   • Percorso assoluto ECG: {ECG_DIR_ABS}")
print(f"   • File ECG: {ECG_FILE}")
print(f"   • Output: {OUTPUT_DIR}")

# Crea directory output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. CARICAMENTO DATI - PROVA PIÙ PERCORSI
print("\n3. 📥 Caricamento dati ECG...")

percorsi_da_provare = [
    ECG_DIR_ABS,  # /home/ser/equity_study/ecg_data/
    "../../ecg_data",  # Relativo
    "../01_data_ecg",  # phi_f_project/01_data_ecg/
    "./01_data_ecg",   # Directory corrente/01_data_ecg
    os.path.join(HOME_DIR, "equity_study", "data_ecg"),  # Dove li hai scaricati prima
]

record = None
percorso_usato = None

for base_path in percorsi_da_provare:
    record_path = os.path.join(base_path, ECG_FILE)
    print(f"   🔍 Provo: {record_path}")
    
    if os.path.exists(record_path + ".hea") or os.path.exists(record_path + ".dat"):
        try:
            record = wfdb.rdrecord(record_path, sampto=10000)
            percorso_usato = record_path
            print(f"   ✅ TROVATO in: {base_path}")
            break
        except Exception as e:
            print(f"   ❌ Errore caricamento: {type(e).__name__}")
    else:
        print(f"   ❌ File .hea/.dat non trovati")

if record is None:
    print("\n   ❌❌❌ NESSUN FILE ECG TROVATO!")
    print("\n   📋 File esistenti in directory vicine:")
    for path in ["../../ecg_data", "../01_data_ecg", "../../data_ecg"]:
        if os.path.exists(path):
            print(f"\n   📁 {path}:")
            try:
                files = os.listdir(path)
                for f in files[:10]:  # Primi 10 file
                    print(f"      • {f}")
            except:
                print(f"      (non accessibile)")
    
    print("\n   🚀 SOLUZIONE: Scarica i dati con:")
    print("   mkdir -p ../../ecg_data")
    print("   cd ../../ecg_data && wget https://physionet.org/files/mitdb/1.0.0/100.dat")
    print("   wget https://physionet.org/files/mitdb/1.0.0/100.hea")
    sys.exit(1)

# 4. SE ARRIVIAMO QUI, ABBIAMO I DATI
signal_raw = record.p_signal[:, 0].flatten()
fs = record.fs

print(f"\n   ✅ SUCCESSO: {record.record_name}")
print(f"     • Percorso usato: {percorso_usato}")
print(f"     • Campioni: {len(signal_raw):,}")
print(f"     • Durata: {len(signal_raw)/fs:.1f} s")
print(f"     • Frequenza: {fs} Hz")
print(f"     • Range: [{signal_raw.min():.3f}, {signal_raw.max():.3f}] mV")

# 5. FILTRAGGIO (continua come prima...)
print("\n4. 🔧 Filtraggio segnale...")
def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=360, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

signal_filtered = bandpass_filter(signal_raw, lowcut=5.0, highcut=15.0, fs=fs)
print(f"   ✅ Segnale filtrato (5-15 Hz)")

# 6. RILEVAZIONE BATTITI
print("\n5. ❤️  Rilevazione battiti cardiaci...")
threshold = np.percentile(signal_filtered, 95)
peaks, properties = find_peaks(signal_filtered,
                              height=threshold,
                              distance=int(0.4 * fs),
                              prominence=0.1 * (signal_filtered.max() - signal_filtered.min()))

n_beats = len(peaks)
print(f"   • Battiti rilevati: {n_beats}")

if n_beats < 4:
    print(f"   ❌ BATTTI INSUFFICIENTI per analisi")
    sys.exit(1)

# 7. CALCOLO METRICHE
print("\n6. 📊 Calcolo metriche cardiache...")
rr_intervals = np.diff(peaks) / fs * 1000
hr_mean = 60 / (np.mean(rr_intervals) / 1000)
hr_std = np.std(60 / (rr_intervals / 1000))
hrv_sdnn = np.std(rr_intervals)
hrv_rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
cv_rr = hrv_sdnn / np.mean(rr_intervals)
phi_cardiac = 1 - cv_rr

print(f"   ✅ METRICHE CALCOLATE:")
print(f"     • Frequenza cardiaca: {hr_mean:.1f} ± {hr_std:.1f} bpm")
print(f"     • HRV (SDNN): {hrv_sdnn:.1f} ms")
print(f"     • HRV (RMSSD): {hrv_rmssd:.1f} ms")
print(f"     • CV(RR): {cv_rr:.3f}")
print(f"     • Φ CARDIACO: {phi_cardiac:.3f}")

# 8. SALVATAGGIO
print("\n7. 💾 Salvataggio risultati...")
results_df = pd.DataFrame({
    'patient_id': [ECG_FILE],
    'phi_cardiac': [phi_cardiac],
    'hr_bpm': [hr_mean],
    'hrv_ms': [hrv_sdnn],
    'cv_rr': [cv_rr]
})
csv_path = os.path.join(OUTPUT_DIR, "ecg_results_simple.csv")
results_df.to_csv(csv_path, index=False)
print(f"   ✅ Risultati salvati: {csv_path}")

# 9. GRAFICO SEMPLICE
print("\n8. 🖼️  Creazione grafico semplice...")
plt.figure(figsize=(10, 6))
time = np.arange(len(signal_raw[:int(5*fs)])) / fs
plt.plot(time, signal_raw[:int(5*fs)], 'b-', alpha=0.7, label='ECG grezzo')
peaks_in_range = peaks[peaks < len(time)]
plt.plot(time[peaks_in_range], signal_filtered[peaks_in_range], 'ro', 
         markersize=4, label=f'Battiti ({len(peaks_in_range)})')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza (mV)')
plt.title(f'ECG Paziente {ECG_FILE} - Φ = {phi_cardiac:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

fig_path = os.path.join(OUTPUT_DIR, "ecg_plot_simple.png")
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"   ✅ Grafico salvato: {fig_path}")

print("\n" + "="*70)
print("🎉 ANALISI ECG COMPLETATA!")
print("="*70)
print(f"\nΦ cardiaco calcolato: {phi_cardiac:.3f}")
print(f"File salvati in: {OUTPUT_DIR}/")

try:
    plt.show()
except:
    print("\nℹ️  Grafico salvato ma non visualizzabile")

print("\n✅ FINE PASSO 1")
