#!/usr/bin/env python3
"""
STEP 1: ANALISI Φ(t) SU DATI SINTETICI USA 1950-2024
Automatico - minimo intervento richiesto
"""

print("=" * 60)
print("ANALISI EQUITÀ EVOLUTIVA - STEP 1")
print("=" * 60)

# ============================================================================
# 1. IMPORTAZIONI
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, fft
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerie importate")

# ============================================================================
# 2. GENERAZIONE DATI SINTETICI (simulazione WID)
# ============================================================================
def generate_income_data(years=(1950, 2024)):
    """Genera dati reddito sintetici per test"""
    print("\n📊 Generazione dati sintetici USA 1950-2024...")
    
    year_range = np.arange(years[0], years[1] + 1)
    n_years = len(year_range)
    
    # Simula trend storico disuguaglianza USA
    base_gini = 0.35
    gini_trend = base_gini + 0.0015 * (year_range - 1950)
    gini_trend += 0.02 * np.sin(2 * np.pi * (year_range - 1950) / 25)
    
    # Crea dataframe
    data = pd.DataFrame(index=year_range, columns=range(1, 101))
    data.index.name = 'year'
    
    # Popola con distribuzioni log-normali
    for i, year in enumerate(year_range):
        gini = gini_trend[i]
        sigma = np.sqrt(2) * stats.norm.ppf((1 + gini) / 2)
        mu = -sigma**2 / 2
        
        percentiles = np.linspace(0.01, 0.99, 100)
        incomes = np.exp(mu + sigma * stats.norm.ppf(percentiles))
        
        base_income = 15000 * (1.02 ** (year - 1950))
        incomes = incomes / np.mean(incomes) * base_income
        
        data.loc[year] = incomes
    
    print(f"   Periodo: {years[0]}-{years[1]} ({n_years} anni)")
    print(f"   Percentili: 1-100 (distribuzione completa)")
    print(f"   Gini medio: {np.mean(gini_trend):.3f}")
    
    return data, gini_trend

# Genera i dati
income_data, gini_history = generate_income_data()

# ============================================================================
# 3. CALCOLO Φ(t) - METRICA EQUITÀ
# ============================================================================
def calculate_phi(income_data, elasticity=0.5):
    """Calcola Φ(t) = 1 - CV(utilità marginali)"""
    print("\n🧮 Calcolo Φ(t)...")
    
    incomes = income_data.values.astype(float)
    marginal_utilities = incomes ** (-elasticity)
    
    cv = np.std(marginal_utilities, axis=1) / np.mean(marginal_utilities, axis=1)
    phi = 1 - cv
    
    years = income_data.index.values
    n_years = len(years)
    
    print(f"   Φ(t) range: [{phi.min():.3f}, {phi.max():.3f}]")
    print(f"   Φ(t) medio: {phi.mean():.3f}")
    
    return years, phi, marginal_utilities

years, phi, marginal_utils = calculate_phi(income_data)

# ============================================================================
# 4. DECOMPOSIZIONE OSCILLAZIONI
# ============================================================================
def analyze_oscillations(years, phi):
    """Analizza trend e oscillazioni di Φ(t)"""
    print("\n📈 Analisi oscillazioni strutturate...")
    
    coeffs = np.polyfit(years, phi, 3)
    trend = np.polyval(coeffs, years)
    oscillations = phi - trend
    
    amplitude = np.sqrt(np.mean(oscillations**2))
    
    n = len(oscillations)
    if n > 10:
        fft_vals = np.abs(fft.rfft(oscillations))
        freqs = fft.rfftfreq(n, d=1)
        if len(fft_vals) > 1:
            dominant_idx = np.argmax(fft_vals[1:]) + 1
            period = 1 / freqs[dominant_idx] if freqs[dominant_idx] > 0 else np.nan
        else:
            period = np.nan
    else:
        period = np.nan
    
    snr = np.var(trend) / np.var(oscillations) if np.var(oscillations) > 0 else np.inf
    
    def simple_fractal_dim(series):
        n = len(series)
        if n < 10:
            return 1.0
        L = []
        for k in [2, 3, 4, 5]:
            Lk = 0
            for m in range(k):
                idx = np.arange(m, n, k)
                if len(idx) > 1:
                    Lkm = np.sum(np.abs(np.diff(series[idx])))
                    Lkm = Lkm * (n - 1) / (len(idx) * k)
                    Lk += Lkm
            L.append(np.log(Lk / k))
        k_vals = np.log(1.0 / np.array([2, 3, 4, 5]))
        slope = np.polyfit(k_vals, L, 1)[0]
        return -slope
    
    fractal_dim = simple_fractal_dim(oscillations)
    complexity = amplitude * fractal_dim
    
    metrics = {
        'ampiezza': amplitude,
        'periodo': period if not np.isnan(period) else "N/A",
        'SNR': snr,
        'dimensione_frattale': fractal_dim,
        'complessità': complexity
    }
    
    print("   Metriche M(Φ):")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"     {key}: {val:.4f}")
        else:
            print(f"     {key}: {val}")
    
    return trend, oscillations, metrics

trend, oscillations, metrics = analyze_oscillations(years, phi)

# ============================================================================
# 5. VISUALIZZAZIONE
# ============================================================================
def create_visualization(years, phi, trend, oscillations, metrics):
    """Crea grafico completo"""
    print("\n🎨 Generazione visualizzazione...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(years, phi, 'b-', linewidth=2, alpha=0.7, label='Φ(t)')
    axes[0, 0].plot(years, trend, 'r--', linewidth=2, label='Trend')
    axes[0, 0].fill_between(years, phi, trend, alpha=0.3, color='gray')
    axes[0, 0].set_xlabel('Anno', fontsize=11)
    axes[0, 0].set_ylabel('Φ(t)', fontsize=11)
    axes[0, 0].set_title('Dinamica Equità Φ(t) - USA 1950-2024', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(years, oscillations, 'g-', linewidth=2, alpha=0.7)
    axes[0, 1].fill_between(years, 0, oscillations, alpha=0.3, color='green')
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].set_xlabel('Anno', fontsize=11)
    axes[0, 1].set_ylabel('Φ̃(t) - Oscillazioni', fontsize=11)
    axes[0, 1].set_title('Componente Oscillatoria Strutturata', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(phi, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=phi.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Media: {phi.mean():.3f}')
    axes[1, 0].set_xlabel('Valore Φ', fontsize=11)
    axes[1, 0].set_ylabel('Frequenza', fontsize=11)
    axes[1, 0].set_title('Distribuzione Valori Φ(t)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    metric_names = ['ampiezza', 'SNR', 'dimensione_frattale', 'complessità']
    metric_values = [metrics.get(name, 0) for name in metric_names]
    
    max_val = max(abs(v) if isinstance(v, float) else 0 for v in metric_values)
    if max_val > 0:
        normalized_vals = [v/max_val if isinstance(v, float) else 0 for v in metric_values]
    else:
        normalized_vals = [0, 0, 0, 0]
    
    colors = ['blue', 'red', 'green', 'purple']
    bars = axes[1, 1].bar(metric_names, normalized_vals, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Valore Normalizzato', fontsize=11)
    axes[1, 1].set_title('Metrica M(Φ) - Proprietà Oscillazione', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars, metric_values):
        if isinstance(val, float):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = 'step1_phi_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   📁 Grafico salvato come: {output_file}")
    
    return fig

fig = create_visualization(years, phi, trend, oscillations, metrics)

# ============================================================================
# 6. RISULTATI E INTERPRETAZIONE
# ============================================================================
print("\n" + "=" * 60)
print("RISULTATI STEP 1 - INTERPRETAZIONE")
print("=" * 60)

trend_slope = (trend[-1] - trend[0]) / (years[-1] - years[0])
print(f"\n📈 TREND GENERALE Φ(t):")
print(f"   Valore iniziale (1950): {phi[0]:.3f}")
print(f"   Valore finale (2024): {phi[-1]:.3f}")
print(f"   Pendenza trend: {trend_slope:.5f} per anno")

if trend_slope > 0.0005:
    print("   → TENDENZA: Φ in AUMENTO (equità crescente)")
elif trend_slope < -0.0005:
    print("   → TENDENZA: Φ in DIMINUZIONE (disuguaglianza crescente)")
else:
    print("   → TENDENZA: Φ STABILE (equilibrio dinamico)")

print(f"\n🔄 PROPRIETÀ OSCILLATORIE:")
print(f"   Ampiezza oscillazioni: {metrics['ampiezza']:.4f}")
if isinstance(metrics['periodo'], float):
    print(f"   Periodo dominante: {metrics['periodo']:.1f} anni")
    if 5 < metrics['periodo'] < 15:
        print("     → CICLO: Business (8-12 anni)")
    elif 15 < metrics['periodo'] < 35:
        print("     → CICLO: Generazionale (20-30 anni)")
    elif metrics['periodo'] > 35:
        print("     → CICLO: Secolare/strutturale")

print(f"   SNR (segnale/rumore): {metrics['SNR']:.2f}")
if metrics['SNR'] > 1.0:
    print("     → OSCILLAZIONE: Strutturata (trend > rumore)")
else:
    print("     → OSCILLAZIONE: Rumore dominante")

print(f"   Dimensione frattale: {metrics['dimensione_frattale']:.3f}")
if metrics['dimensione_frattale'] > 1.3:
    print("     → COMPLESSITÀ: Alta (dinamica ricca)")
else:
    print("     → COMPLESSITÀ: Media/bassa")

print(f"\n🌍 IMPLICAZIONI EVOLUTIVE (Ipotesi Φ-F/Equity):")
print("   1. Φ(t) misura il 'regime dinamico' del sistema distributivo")
print("   2. Le oscillazioni strutturate indicano adattabilità")
print("   3. SNR alto → sistema mantiene equità attraverso perturbazioni")
print("   4. Trend negativo → possibile 'lock-in gerarchico' in sviluppo")

print(f"\n🔬 TEST EMPIRICI PROPOSTI:")
print("   A. Estendere analisi a dataset WID reali (100+ paesi)")
print("   B. Testare se Φ pre-crisi predice recovery post-crisi")
print("   C. Validare su Ostrom CPR (successo/fallimento commons)")
print("   D. Analisi cross-dominio: Bitcoin, Wikipedia, reti ecologiche")

print("\n" + "=" * 60)
print("STEP 1 COMPLETATO ✅")
print("File generati:")
print("   1. step1_phi_analysis.png (grafico)")
print("   2. Dati in memoria per step successivi")
print("=" * 60)

try:
    plt.show()
except:
    print("\n💡 Per vedere il grafico, apri il file: step1_phi_analysis.png")
