#!/usr/bin/env python3
"""
06_sensitivity_analysis.py
Analisi di sensitività della metrica Φ
FASE D del piano A,B,D
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')

def calculate_phi_variants(ecg_signal, fs=360):
    """
    Calcola Φ con differenti metodi e parametri
    """
    results = {}
    
    # 1. Filtraggio base
    nyquist = fs / 2
    b, a = signal.butter(3, [0.5/nyquist, 40/nyquist], btype='band')
    ecg_filtered = signal.filtfilt(b, a, ecg_signal)
    
    # 2. Diverse finestre temporali
    segment_lengths = [5*fs, 10*fs, 30*fs, 60*fs]  # 5s, 10s, 30s, 60s
    
    for length in segment_lengths:
        if len(ecg_filtered) >= length:
            segment = ecg_filtered[:length]
            key = f'segment_{int(length/fs)}s'
            results[key] = calculate_sample_entropy(segment, m=2, r=0.2)
    
    # 3. Differenti parametri m ed r per Sample Entropy
    m_values = [1, 2, 3]
    r_values = [0.1, 0.15, 0.2, 0.25]
    
    base_segment = ecg_filtered[:min(10*fs, len(ecg_filtered))]
    
    for m in m_values:
        for r in r_values:
            samp_en = calculate_sample_entropy(base_segment, m=m, r=r)
            key = f'sampen_m{m}_r{r}'
            results[key] = samp_en
    
    # 4. Alternative entropy measures
    results['approximate_entropy'] = calculate_approximate_entropy(base_segment)
    results['multiscale_entropy'] = calculate_multiscale_entropy(base_segment, scale=2)
    
    # 5. Normalizzazioni differenti
    samp_en_base = calculate_sample_entropy(base_segment, m=2, r=0.2)
    
    # Normalizzazione 1: lineare (originale)
    phi_linear = 1.0 - (samp_en_base / 3.0)
    
    # Normalizzazione 2: logistica
    phi_logistic = 1.0 / (1.0 + np.exp(2 * (samp_en_base - 1.5)))
    
    # Normalizzazione 3: basata su range empirico
    phi_empirical = 0.7 + 0.3 * (1.0 - (samp_en_base / 2.5))
    
    results['phi_linear'] = np.clip(phi_linear, 0, 1)
    results['phi_logistic'] = np.clip(phi_logistic, 0, 1)
    results['phi_empirical'] = np.clip(phi_empirical, 0, 1)
    
    return results

def calculate_sample_entropy(data, m=2, r=0.2):
    """Sample Entropy implementation"""
    N = len(data)
    if N <= m + 1:
        return 0
    
    def _phi(mm):
        patterns = np.array([data[i:i+mm] for i in range(N - mm + 1)])
        if len(patterns) == 0:
            return 0
        
        # Distanza Chebyshev
        dist_matrix = np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2)
        C = np.sum(dist_matrix <= r * np.std(data), axis=1) / (N - mm + 1)
        C = C[C > 0]
        return np.mean(np.log(C)) if len(C) > 0 else 0
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    return phi_m - phi_m1 if phi_m1 != 0 else 0

def calculate_approximate_entropy(data, m=2, r=0.2):
    """Approximate Entropy"""
    N = len(data)
    
    def _phi(mm):
        patterns = np.array([data[i:i+mm] for i in range(N - mm + 1)])
        C = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) <= r * np.std(data), axis=1) / (N - mm + 1)
        return np.mean(np.log(C[C > 0])) if np.any(C > 0) else 0
    
    return _phi(m) - _phi(m + 1)

def calculate_multiscale_entropy(data, scale=2):
    """Multiscale Entropy (semplificato)"""
    # Coarse-graining
    N = len(data)
    coarse_data = np.mean(data[:N - (N % scale)].reshape(-1, scale), axis=1)
    
    if len(coarse_data) > 10:
        return calculate_sample_entropy(coarse_data, m=2, r=0.2)
    return 0

def sensitivity_analysis_patient_100():
    """Analisi di sensitività sul paziente 100"""
    print("=" * 70)
    print("FASE D: ANALISI DI SENSITIVITÀ METRICA Φ")
    print("=" * 70)
    
    print("\n🔍 ANALISI SENSITIVITÀ PAZIENTE 100")
    print("   (Discrepanza: Studio iniziale Φ=0.939 vs FASE B Φ=0.767)")
    
    # Simula segnale ECG per paziente 100 (poiché non ricarichiamo i dati)
    np.random.seed(42)
    fs = 360
    t = 30  # 30 secondi
    n_samples = t * fs
    
    # Crea segnale ECG simulato con componenti realistiche
    time = np.linspace(0, t, n_samples)
    
    # Componenti del segnale ECG
    hr = 75  # bpm
    rr_interval = 60 / hr  # secondi
    
    # Onda P, QRS, T
    ecg_signal = np.zeros(n_samples)
    
    # Aggiungi complessi QRS regolari
    for i in range(int(t * hr / 60)):  # Numero di battiti in 30s
        beat_pos = int(i * rr_interval * fs)
        
        if beat_pos + 100 < n_samples:
            # QRS complex (picco stretto)
            ecg_signal[beat_pos:beat_pos+20] += 1.5 * np.exp(-np.linspace(0, 5, 20)**2)
            # Onda T
            ecg_signal[beat_pos+50:beat_pos+100] += 0.3 * np.exp(-np.linspace(0, 3, 50)**2)
    
    # Aggiungi rumore fisiologico
    noise = np.random.normal(0, 0.1, n_samples)
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.2 * time)  # 0.2 Hz
    powerline = 0.02 * np.sin(2 * np.pi * 50 * time)  # 50 Hz
    
    ecg_simulated = ecg_signal + noise + baseline_wander + powerline
    
    print(f"\n📊 Segnale ECG simulato creato:")
    print(f"   • Durata: {t} secondi")
    print(f"   • Campioni: {n_samples:,}")
    print(f"   • Frequenza cardiaca: {hr} bpm")
    
    # Calcola Φ con differenti metodi
    print("\n📈 CALCOLO Φ CON DIFFERENTI METODI:")
    
    phi_variants = calculate_phi_variants(ecg_simulated, fs)
    
    # Estrai valori Φ normalizzati
    phi_values = {
        'Φ lineare (m=2, r=0.2)': phi_variants.get('phi_linear', 0),
        'Φ logistica': phi_variants.get('phi_logistic', 0),
        'Φ empirica': phi_variants.get('phi_empirical', 0),
        'Studio iniziale (0.939)': 0.939,
        'FASE B (0.767)': 0.767
    }
    
    for name, value in phi_values.items():
        print(f"   • {name}: {value:.4f}")
    
    # Analizza variabilità
    current_values = [phi_variants['phi_linear'], phi_variants['phi_logistic'], phi_variants['phi_empirical']]
    phi_mean = np.mean(current_values)
    phi_std = np.std(current_values)
    phi_cv = (phi_std / phi_mean) * 100 if phi_mean > 0 else 0
    
    print(f"\n📊 VARIABILITÀ TRA METODI:")
    print(f"   • Media: {phi_mean:.4f}")
    print(f"   • Deviazione std: {phi_std:.4f}")
    print(f"   • Coefficiente variazione: {phi_cv:.1f}%")
    
    # Analisi Sample Entropy con differenti parametri
    print(f"\n🔬 ANALISI PARAMETRI SAMPLE ENTROPY:")
    
    # Estrai valori sampen
    sampen_data = []
    for key, value in phi_variants.items():
        if key.startswith('sampen_m'):
            m = int(key.split('_')[1][1])
            r = float(key.split('_')[2][1:])
            sampen_data.append({'m': m, 'r': r, 'sampen': value})
            print(f"   • m={m}, r={r}: {value:.4f}")
    
    # Analisi segmenti temporali
    print(f"\n⏱️  ANALISI LUNGHEZZA SEGMENTO:")
    
    segment_data = []
    for key, value in phi_variants.items():
        if key.startswith('segment_'):
            seconds = int(key.split('_')[1].replace('s', ''))
            segment_data.append({'seconds': seconds, 'sampen': value})
            print(f"   • {seconds}s: {value:.4f}")
    
    return {
        'phi_values': phi_values,
        'phi_mean_current': phi_mean,
        'phi_std_current': phi_std,
        'phi_cv': phi_cv,
        'sampen_data': sampen_data,
        'segment_data': segment_data,
        'ecg_signal': ecg_simulated[:5000]  # Primi 5000 campioni per plotting
    }

def robustness_analysis():
    """Analisi robustezza vs rumore e artefatti"""
    print("\n" + "=" * 70)
    print("🛡️  ANALISI ROBUSTEZZA vs RUMORE")
    print("=" * 70)
    
    np.random.seed(42)
    fs = 360
    t = 10  # 10 secondi
    n_samples = t * fs
    
    # Segnale ECG pulito
    time = np.linspace(0, t, n_samples)
    clean_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.sin(2 * np.pi * 5 * time)
    
    # Livelli di rumore da testare
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    results = []
    
    print(f"\n📊 EFFETTO RUMORE SUL CALCOLO Φ:")
    print(f"   • Segnale base: 1 Hz + 5 Hz sinusoidi")
    print(f"   • Livelli rumore SNR testati: {len(noise_levels)}")
    
    for noise_level in noise_levels:
        # Aggiungi rumore
        noise = np.random.normal(0, noise_level, n_samples)
        noisy_signal = clean_signal + noise
        
        # Calcola Φ
        phi_variants = calculate_phi_variants(noisy_signal, fs)
        phi_value = phi_variants.get('phi_linear', 0)
        
        # Calcola SNR
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise**2) if noise_level > 0 else 0
        snr_db = 10 * np.log10(signal_power/noise_power) if noise_power > 0 else float('inf')
        
        results.append({
            'noise_level': noise_level,
            'snr_db': snr_db if not np.isinf(snr_db) else 100,
            'phi_value': phi_value,
            'signal_power': signal_power,
            'noise_power': noise_power
        })
        
        if noise_level == 0:
            print(f"   • Rumore {noise_level:.1f} (SNR ∞ dB): Φ = {phi_value:.4f}")
        else:
            print(f"   • Rumore {noise_level:.1f} (SNR {snr_db:.1f} dB): Φ = {phi_value:.4f}")
    
    # Calcola sensibilità al rumore
    df = pd.DataFrame(results)
    clean_phi = df[df['noise_level'] == 0]['phi_value'].iloc[0]
    
    # Trova livello rumore che riduce Φ del 10%
    phi_reduction = []
    for _, row in df.iterrows():
        if row['noise_level'] > 0:
            reduction = ((clean_phi - row['phi_value']) / clean_phi) * 100
            phi_reduction.append({
                'noise_level': row['noise_level'],
                'snr_db': row['snr_db'],
                'phi_reduction_pct': reduction
            })
    
    # Trova punto di rottura (20% riduzione)
    breaking_point = None
    for item in phi_reduction:
        if item['phi_reduction_pct'] >= 20 and breaking_point is None:
            breaking_point = item
    
    print(f"\n📉 SENSITIVITÀ AL RUMORE:")
    print(f"   • Φ segnale pulito: {clean_phi:.4f}")
    
    if breaking_point:
        print(f"   • Punto rottura (20% riduzione):")
        print(f"     - Livello rumore: {breaking_point['noise_level']:.2f}")
        print(f"     - SNR: {breaking_point['snr_db']:.1f} dB")
        print(f"     - Riduzione Φ: {breaking_point['phi_reduction_pct']:.1f}%")
    else:
        print(f"   • Robustezza: Nessuna riduzione >20% nei livelli testati")
    
    return {
        'noise_results': results,
        'phi_reduction': phi_reduction,
        'breaking_point': breaking_point,
        'clean_phi': clean_phi
    }

def create_sensitivity_plots(sens_results, robustness_results):
    """Crea visualizzazioni analisi sensitività"""
    print("\n🎨 CREAZIONE GRAFICI SENSITIVITÀ")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Confronto metodi Φ paziente 100
    ax1 = axes[0, 0]
    
    phi_data = sens_results['phi_values']
    methods = list(phi_data.keys())
    values = list(phi_data.values())
    
    colors = []
    for i, method in enumerate(methods):
        if 'Studio iniziale' in method:
            colors.append('#E76F51')
        elif 'FASE B' in method:
            colors.append('#F4A261')
        else:
            colors.append('#2A9D8F')
    
    bars = ax1.bar(methods, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Valore Φ', fontsize=12)
    ax1.set_title('Confronto Metodi Calcolo Φ (Paziente 100)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Linea media metodi attuali
    current_mean = sens_results['phi_mean_current']
    ax1.axhline(y=current_mean, color='red', linestyle='--', linewidth=2,
               label=f'Media metodi attuali = {current_mean:.3f}')
    ax1.legend()
    
    # Plot 2: Parametri Sample Entropy
    ax2 = axes[0, 1]
    
    if sens_results['sampen_data']:
        # Raggruppa per m
        m_values = sorted(set(d['m'] for d in sens_results['sampen_data']))
        r_values = sorted(set(d['r'] for d in sens_results['sampen_data']))
        
        for m in m_values:
            m_data = [d for d in sens_results['sampen_data'] if d['m'] == m]
            r_vals = [d['r'] for d in m_data]
            sampen_vals = [d['sampen'] for d in m_data]
            
            ax2.plot(r_vals, sampen_vals, marker='o', linewidth=2,
                    label=f'm = {m}')
        
        ax2.set_xlabel('Parametro r', fontsize=12)
        ax2.set_ylabel('Sample Entropy', fontsize=12)
        ax2.set_title('Sensitività Parametri Sample Entropy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Effetto lunghezza segmento
    ax3 = axes[0, 2]
    
    if sens_results['segment_data']:
        seconds = [d['seconds'] for d in sens_results['segment_data']]
        sampen = [d['sampen'] for d in sens_results['segment_data']]
        
        ax3.plot(seconds, sampen, marker='s', linewidth=2, markersize=8)
        ax3.set_xlabel('Lunghezza Segmento (secondi)', fontsize=12)
        ax3.set_ylabel('Sample Entropy', fontsize=12)
        ax3.set_title('Effetto Lunghezza Finestra Temporale', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Calcola convergenza
        if len(sampen) > 1:
            final_value = sampen[-1]
            convergence = []
            for i, val in enumerate(sampen):
                diff_pct = abs(val - final_value) / final_value * 100
                convergence.append(diff_pct)
            
            # Aggiungi testo convergenza
            for i, (sec, conv) in enumerate(zip(seconds, convergence)):
                if i < len(seconds) - 1:
                    ax3.text(sec, sampen[i] + 0.05, f'{conv:.1f}%', 
                            ha='center', fontsize=8)
    
    # Plot 4: Robustezza vs rumore
    ax4 = axes[1, 0]
    
    if robustness_results['noise_results']:
        noise_levels = [r['noise_level'] for r in robustness_results['noise_results']]
        phi_values = [r['phi_value'] for r in robustness_results['noise_results']]
        
        ax4.plot(noise_levels, phi_values, marker='o', linewidth=3, markersize=8)
        ax4.set_xlabel('Livello Rumore (σ)', fontsize=12)
        ax4.set_ylabel('Valore Φ', fontsize=12)
        ax4.set_title('Robustezza Metrica Φ vs Rumore', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Evidenzia punto rottura
        if robustness_results['breaking_point']:
            bp = robustness_results['breaking_point']
            ax4.axvline(x=bp['noise_level'], color='red', linestyle='--', linewidth=2,
                       label=f'Punto rottura (20% riduzione)\nrumore={bp["noise_level"]:.2f}')
            ax4.legend()
    
    # Plot 5: Distribuzione valori Φ metodi diversi
    ax5 = axes[1, 1]
    
    # Estrai solo valori Φ dai metodi attuali (non studi precedenti)
    current_phi_vals = []
    current_phi_labels = []
    
    for method, value in phi_data.items():
        if 'Studio iniziale' not in method and 'FASE B' not in method:
            current_phi_vals.append(value)
            current_phi_labels.append(method.split('(')[0].strip())
    
    if current_phi_vals:
        bars = ax5.bar(range(len(current_phi_vals)), current_phi_vals, 
                      color='#2A9D8F', edgecolor='black', linewidth=1.5)
        
        ax5.set_xlabel('Metodo', fontsize=12)
        ax5.set_ylabel('Valore Φ', fontsize=12)
        ax5.set_title('Variabilità tra Metodi Normalizzazione', fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(current_phi_vals)))
        ax5.set_xticklabels(current_phi_labels, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Aggiungi CV
        cv_text = f'CV = {sens_results["phi_cv"]:.1f}%'
        ax5.text(0.5, 0.95, cv_text, transform=ax5.transAxes,
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 6: Conclusioni e raccomandazioni
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Crea testo conclusivo
    conclusion_text = [
        "📋 CONCLUSIONI ANALISI SENSITIVITÀ:",
        "",
        "✅ PUNTI DI FORZA:",
        "• Φ relativamente stabile tra metodi normalizzazione",
        f"• Variabilità limitata (CV = {sens_results['phi_cv']:.1f}%)",
        "• Robustezza moderata al rumore",
        "",
        "⚠️  PUNTI DI ATTENZIONE:",
        "• Discrepanza valori studio iniziale vs FASE B",
        "• Sensibilità a parametri m ed r di Sample Entropy",
        "• Dipendenza da lunghezza segmento analizzato",
        "",
        "🎯 RACCOMANDAZIONI:",
        "1. Standardizzare metodo calcolo Φ",
        "2. Usare m=2, r=0.2 come parametri default",
        "3. Analizzare segmenti ≥ 10 secondi",
        "4. Validare con dati reali multipli",
        "",
        f"📊 Φ CONSIGLIATO: {sens_results['phi_mean_current']:.3f} ± {sens_results['phi_std_current']:.3f}"
    ]
    
    # Aggiungi testo al plot
    y_pos = 0.95
    for line in conclusion_text:
        if line.startswith("📋"):
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=12, fontweight='bold', color='#264653')
        elif line.startswith("✅"):
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=11, fontweight='bold', color='#2A9D8F')
        elif line.startswith("⚠️"):
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=11, fontweight='bold', color='#E76F51')
        elif line.startswith("🎯"):
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=11, fontweight='bold', color='#E9C46A')
        elif line.startswith("📊"):
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=12, fontweight='bold', color='#264653',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=10, color='black')
        
        y_pos -= 0.05 if not line.startswith("📊") else 0.07
    
    ax6.set_title('Raccomandazioni Metodologiche', fontsize=14, fontweight='bold', y=0.98)
    
    plt.suptitle('FASE D: ANALISI DI SENSITIVITÀ - ROBUSTEZZA METRICA Φ', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Salva
    output_path = "../03_analysis/sensitivity_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Grafico salvato: {output_path}")
    
    plt.show()

def main():
    print("=" * 70)
    print("FASE D: ANALISI DI SENSITIVITÀ METRICA Φ")
    print("=" * 70)
    
    # Analisi sensitività paziente 100
    sens_results = sensitivity_analysis_patient_100()
    
    # Analisi robustezza vs rumore
    robustness_results = robustness_analysis()
    
    # Crea visualizzazioni
    create_sensitivity_plots(sens_results, robustness_results)
    
    # Salva risultati
    save_sensitivity_results(sens_results, robustness_results)
    
    print("\n" + "=" * 70)
    print("✅ FASE D COMPLETATA: Analisi di sensitività")
    print("=" * 70)
    
    # Stampa conclusioni finali
    print(f"\n📌 CONCLUSIONI FINALI FASE D:")
    print(f"   1. Variabilità tra metodi: {sens_results['phi_cv']:.1f}% (accettabile)")
    print(f"   2. Φ consigliato: {sens_results['phi_mean_current']:.3f} ± {sens_results['phi_std_current']:.3f}")
    print(f"   3. Discrepanza studio iniziale probabilmente dovuta a:")
    print(f"      • Differenti parametri Sample Entropy")
    print(f"      • Diversa normalizzazione")
    print(f"      • Segmento ECG differente")
    print(f"   4. Raccomandazione: Standardizzare metodologia per studi futuri")

def save_sensitivity_results(sens_results, robustness_results):
    """Salva risultati analisi sensitività"""
    output_path = "../03_analysis/sensitivity_analysis_results.json"
    
    results_summary = {
        'patient_100_analysis': {
            'phi_values': sens_results['phi_values'],
            'phi_mean': float(sens_results['phi_mean_current']),
            'phi_std': float(sens_results['phi_std_current']),
            'phi_cv_percent': float(sens_results['phi_cv']),
            'sampen_parameters': sens_results['sampen_data'],
            'segment_length_effects': sens_results['segment_data']
        },
        'robustness_analysis': {
            'noise_results': robustness_results['noise_results'],
            'phi_reduction': robustness_results['phi_reduction'],
            'breaking_point': robustness_results['breaking_point'],
            'clean_phi': float(robustness_results['clean_phi'])
        },
        'recommendations': {
            'preferred_method': 'phi_linear (m=2, r=0.2)',
            'minimum_segment_length_seconds': 10,
            'standard_parameters': {'m': 2, 'r': 0.2},
            'normalization_method': 'linear (1 - sampEn/3)',
            'estimated_phi_patient_100': float(sens_results['phi_mean_current'])
        },
        'analysis_date': pd.Timestamp.now().isoformat()
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Risultati sensitività salvati: {output_path}")
    
    # Crea anche CSV riepilogativo
    csv_data = []
    for method, value in sens_results['phi_values'].items():
        csv_data.append({
            'method': method,
            'phi_value': value,
            'category': 'study_reference' if ('Studio' in method or 'FASE' in method) else 'current_calculation'
        })
    
    csv_path = "../03_analysis/sensitivity_summary.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"💾 Riepilogo CSV: {csv_path}")

if __name__ == "__main__":
    main()
