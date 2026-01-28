print("="*60)
print("PASSO 1: ANALISI ECG SINGOLO PAZIENTE")
print("="*60)

import wfdb
import numpy as np

print("1. Carico dati ECG...")
record = wfdb.rdrecord('./data_ecg/100', sampto=3000)
signal = record.p_signal[:, 0]

print(f"   Paziente: {record.record_name}")
print(f"   Campioni: {len(signal)}")
print(f"   OK!")
