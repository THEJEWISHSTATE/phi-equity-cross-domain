#!/usr/bin/env python3
"""
STEP 2: ANALISI COMPARATIVA Φ(t) - 4 PAESI
Confronto USA, Germania, Svezia, Brasile 1950-2024
"""

print("=" * 70)
print("ANALISI COMPARATIVA EQUITÀ - STEP 2")
print("=" * 70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, fft
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerie importate")

# ============================================================================
# 1. DATI SINTETICI PER 4 PAESI (profili storici diversi)
# ============================================================================
def generate_country_data():
    """Genera dati per 4 paesi con pattern storici distinti"""
    
    countries = {
        'USA': {
            'base_gini': 0.35,
            'gini_trend': 0.0015,
            'cycle_amplitude': 0.02,
            'cycle_period': 25,
            'income_growth': 0.02
        },
        'GERMANY': {
            'base_gini': 0.28,
            'gini_trend': 0.0005,
            'cycle_amplitude': 0.015,
            'cycle_period': 30,
            'income_growth': 0.018
        },
        'SWEDEN': {
            'base_gini': 0.24,
            'gini_trend': 0.0002,
            'cycle_amplitude': 0.01,
            'cycle_period': 20,
            'income_growth': 0.015
        },
        'BRAZIL': {
            'base_gini': 0.55,
            'gini_trend': -0.0020,
            'cycle_amplitude': 0.03,
            'cycle_period': 15,
            'income_growth': 0.025
        }
    }
    
    years = np.arange(1950, 2025)
    all_data = {}
    gini_data = {}
    
    for country, params in countries.items():
        print(f"\n📊 Generazione dati {country}...")
        
        gini = (params['base_gini'] + 
                params['gini_trend'] * (years - 1950) +
                params['cycle_amplitude'] * np.sin(2 * np.pi * (years - 1950) / params['cycle_period']))
        
        data = pd.DataFrame(index=years, columns=range(1, 101))
        
        for i, year in enumerate(years):
            sigma = np.sqrt(2) * stats.norm.ppf((1 + gini[i]) / 2)
            mu = -sigma**2 / 2
            
            percentiles = np.linspace(0.01, 0.99, 100)
            incomes = np.exp(mu + sigma * stats.norm.ppf(percentiles))
            
            base_income = 10000 * (1 + params['income_growth']) ** (year - 1950)
            incomes = incomes / np.mean(incomes) * base_income
            
            data.loc[year] = incomes
        
        all_data[country] = data
        gini_data[country] = gini
        
        print(f"   Gini 1950: {gini[0]:.3f}, Gini 2024: {gini[-1]:.3f}")
        print(f"   ΔGini: {gini[-1]-gini[0]:.3f} ({'↑' if gini[-1]>gini[0] else '↓'})")
    
    return all_data, gini_data, years

country_data, country_gini, years = generate_country_data()

# ============================================================================
# 2. CALCOLO Φ(t) PER OGNI PAESE
# ============================================================================
def calculate_phi_for_countries(country_data, elasticity=0.5):
    """Calcola Φ(t) per tutti i paesi"""
    print("\n🧮 Calcolo Φ(t) per 4 paesi...")
    
    phi_results = {}
    metrics_results = {}
    
    for country, data in country_data.items():
        incomes = data.values.astype(float)
        marginal_utilities = incomes ** (-elasticity)
        
        cv = np.std(marginal_utilities, axis=1) / np.mean(marginal_utilities, axis=1)
        phi = 1 - cv
        
        coeffs = np.polyfit(years, phi, 2)
        trend = np.polyval(coeffs, years)
        oscillations = phi - trend
        
        amplitude = np.sqrt(np.mean(oscillations**2))
        snr = np.var(trend) / np.var(oscillations) if np.var(oscillations) > 0 else np.inf
        trend_slope = (trend[-1] - trend[0]) / (years[-1] - years[0])
        
        phi_results[country] = {
            'phi': phi,
            'trend': trend,
            'oscillations': oscillations,
            'mean': np.mean(phi),
            'trend_slope': trend_slope
        }
        
        metrics_results[country] = {
            'mean_phi': np.mean(phi),
            'trend_slope': trend_slope,
            'amplitude': amplitude,
            'SNR': snr,
            'phi_1950': phi[0],
            'phi_2024': phi[-1],
            'delta_phi': phi[-1] - phi[0]
        }
        
        print(f"   {country}:")
        print(f"     Φ medio: {np.mean(phi):.3f}  Trend: {trend_slope:.5f}/anno")
        print(f"     Φ 1950: {phi[0]:.3f}  Φ 2024: {phi[-1]:.3f}")
    
    return phi_results, metrics_results

phi_results, metrics_results = calculate_phi_for_countries(country_data)

# ============================================================================
# 3. VISUALIZZAZIONE COMPARATIVA (VERSIONE CORRETTA)
# ============================================================================
def create_comparison_plots(years, phi_results, metrics_results):
    """Crea grafici comparativi - versione corretta"""
    print("\n🎨 Generazione visualizzazioni comparative...")
    
    colors = {'USA': 'blue', 'GERMANY': 'green', 'SWEDEN': 'red', 'BRAZIL': 'orange'}
    countries = list(colors.keys())
    
    # Figura 1: Evoluzione Φ(t) nei 4 paesi
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('EVOLUZIONE Φ(t) - CONFRONTO 4 PAESI 1950-2024', fontsize=16, fontweight='bold')
    
    for idx, country in enumerate(countries):
        row = idx // 2
        col = idx % 2
        color = colors[country]
        
        phi = phi_results[country]['phi']
        trend = phi_results[country]['trend']
        
        axes1[row, col].plot(years, phi, color=color, linewidth=2, alpha=0.7, label=f'Φ(t)')
        axes1[row, col].plot(years, trend, 'k--', linewidth=1.5, label='Trend')
        axes1[row, col].fill_between(years, phi, trend, alpha=0.2, color=color)
        
        axes1[row, col].set_title(f'{country}', fontsize=14, fontweight='bold')
        axes1[row, col].set_xlabel('Anno')
        axes1[row, col].set_ylabel('Φ(t)')
        axes1[row, col].legend()
        axes1[row, col].grid(True, alpha=0.3)
        
        delta = phi[-1] - phi[0]
        trend_dir = "↑" if phi_results[country]['trend_slope'] > 0 else "↓"
        info_text = f"ΔΦ: {delta:+.3f}\nTrend: {trend_dir}"
        axes1[row, col].text(0.02, 0.95, info_text, transform=axes1[row, col].transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig1.savefig('step2_country_comparison.png', dpi=150, bbox_inches='tight')
    
    # Figura 2: Dashboard comparativa
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('DASHBOARD COMPARATIVA METRICHE EQUITÀ', fontsize=16, fontweight='bold')
    
    # 2.1: Φ medio per paese
    mean_phi_vals = [metrics_results[c]['mean_phi'] for c in countries]
    axes2[0, 0].bar(countries, mean_phi_vals, color=[colors[c] for c in countries])
    axes2[0, 0].set_title('Φ MEDIO 1950-2024')
    axes2[0, 0].set_ylabel('Φ medio')
    axes2[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_phi_vals):
        axes2[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 2.2: Trend (pendenza annua)
    trend_vals = [metrics_results[c]['trend_slope'] * 1000 for c in countries]
    axes2[0, 1].bar(countries, trend_vals, color=[colors[c] for c in countries])
    axes2[0, 1].set_title('TREND Φ (per 1000 anni)')
    axes2[0, 1].set_ylabel('ΔΦ/1000 anni')
    axes2[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes2[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(trend_vals):
        axes2[0, 1].text(i, v + (0.1 if v >= 0 else -0.15), f'{v:+.3f}', ha='center')
    
    # 2.3: Ampiezza oscillazioni
    amp_vals = [metrics_results[c]['amplitude'] for c in countries]
    axes2[0, 2].bar(countries, amp_vals, color=[colors[c] for c in countries])
    axes2[0, 2].set_title('AMPIEZZA OSCILLAZIONI')
    axes2[0, 2].set_ylabel('Ampiezza')
    axes2[0, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(amp_vals):
        axes2[0, 2].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    # 2.4: SNR - CORRETTO
    snr_vals = [min(metrics_results[c]['SNR'], 10) for c in countries]
    axes2[1, 0].bar(countries, snr_vals, color=[colors[c] for c in countries])
    axes2[1, 0].set_title('SEGNALE/RUMORE (SNR)')
    axes2[1, 0].set_ylabel('SNR (log)')
    axes2[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='SNR=1')
    axes2[1, 0].grid(True, alpha=0.3, axis='y')
    axes2[1, 0].legend()
    for i, (country, v) in enumerate(zip(countries, snr_vals)):
        axes2[1, 0].text(i, v + 0.2, f'{metrics_results[country]["SNR"]:.1f}', ha='center')
    
    # 2.5: Variazione Φ 1950-2024
    delta_vals = [metrics_results[c]['delta_phi'] for c in countries]
    axes2[1, 1].bar(countries, delta_vals, color=[colors[c] for c in countries])
    axes2[1, 1].set_title('VARIAZIONE Φ 1950-2024')
    axes2[1, 1].set_ylabel('ΔΦ (2024-1950)')
    axes2[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes2[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(delta_vals):
        axes2[1, 1].text(i, v + (0.01 if v >= 0 else -0.02), f'{v:+.3f}', ha='center')
    
    # 2.6: Evoluzione temporale comparata
    axes2[1, 2].set_title('EVOLUZIONE Φ(t) - COMPARATIVA')
    for country, color in colors.items():
        phi = phi_results[country]['phi']
        axes2[1, 2].plot(years, phi, color=color, linewidth=2, label=country, alpha=0.8)
    axes2[1, 2].set_xlabel('Anno')
    axes2[1, 2].set_ylabel('Φ(t)')
    axes2[1, 2].legend()
    axes2[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('step2_comparative_dashboard.png', dpi=150, bbox_inches='tight')
    
    print(f"   📁 Grafici salvati come:")
    print(f"     1. step2_country_comparison.png")
    print(f"     2. step2_comparative_dashboard.png")
    
    return fig1, fig2

fig1, fig2 = create_comparison_plots(years, phi_results, metrics_results)

# ============================================================================
# 4. ANALISI RISULTATI E PATTERN
# ============================================================================
print("\n" + "=" * 70)
print("ANALISI PATTERN COMPARATIVI")
print("=" * 70)

# Classifica paesi per diverse metriche
print("\n🏆 CLASSIFICHE COMPARATIVE:")

# Per Φ medio
print("\n1. CLASSIFICA Φ MEDIO (1950-2024):")
sorted_mean = sorted(metrics_results.items(), key=lambda x: x[1]['mean_phi'], reverse=True)
for rank, (country, data) in enumerate(sorted_mean, 1):
    print(f"   {rank}. {country}: {data['mean_phi']:.3f}")

# Per trend
print("\n2. CLASSIFICA TREND (crescita/declino Φ):")
sorted_trend = sorted(metrics_results.items(), key=lambda x: x[1]['trend_slope'], reverse=True)
for rank, (country, data) in enumerate(sorted_trend, 1):
    trend_dir = "↗" if data['trend_slope'] > 0.0002 else "↘" if data['trend_slope'] < -0.0002 else "→"
    print(f"   {rank}. {country}: {trend_dir} {data['trend_slope']:.5f}/anno")

# Per stabilità (SNR)
print("\n3. CLASSIFICA STABILITÀ (SNR - segnale/rumore):")
sorted_snr = sorted(metrics_results.items(), key=lambda x: x[1]['SNR'], reverse=True)
    countries = list(metrics_results.keys())
for rank, (country, data) in enumerate(sorted_snr, 1):
    stability = "ALTA" if data['SNR'] > 3 else "MEDIA" if data['SNR'] > 1 else "BASSA"
    print(f"   {rank}. {country}: SNR={data['SNR']:.1f} ({stability})")

# Pattern transizioni
print("\n📊 PATTERN TRANSIZIONI EQUITÀ:")

for country in countries:
    delta = metrics_results[country]['delta_phi']
    trend = metrics_results[country]['trend_slope']
    
    if delta > 0.05:
        pattern = "FORTE MIGLIORAMENTO"
    elif delta > 0.01:
        pattern = "MIGLIORAMENTO MODERATO"
    elif delta < -0.05:
        pattern = "FORTE PEGGIORAMENTO"
    elif delta < -0.01:
        pattern = "PEGGIORAMENTO MODERATO"
    else:
        pattern = "STABILE"
    
    print(f"   {country}:")
    print(f"     Φ 1950: {metrics_results[country]['phi_1950']:.3f}")
    print(f"     Φ 2024: {metrics_results[country]['phi_2024']:.3f}")
    print(f"     ΔΦ: {delta:+.3f} → {pattern}")
    print(f"     Trend annuo: {trend:+.5f}")

# ============================================================================
# 5. TEST IPOTESI Φ-F/EQUITY
# ============================================================================
print("\n" + "=" * 70)
print("TEST IPOTESI Φ-F/EQUITY SU DATI COMPARATIVI")
print("=" * 70)

print("\n🔍 IPOTESI 1: Φ(t) mostra oscillazioni strutturate in tutti i sistemi complessi")
snr_threshold = 1.0
countries_above_threshold = [c for c in countries if metrics_results[c]['SNR'] > snr_threshold]
print(f"   Paesi con SNR > {snr_threshold}: {len(countries_above_threshold)}/4")
print(f"   Risultato: {'✅ CONFERMATO' if len(countries_above_threshold) >= 3 else '⚠️ PARZIALE'}")

print("\n🔍 IPOTESI 2: La metrica M(Φ) è stabile e comparabile cross-paese")
amp_range = (0.005, 0.05)
countries_in_range = [c for c in countries if amp_range[0] < metrics_results[c]['amplitude'] < amp_range[1]]
print(f"   Paesi con ampiezza in {amp_range}: {len(countries_in_range)}/4")
print(f"   Risultato: {'✅ CONFERMATO' if len(countries_in_range) >= 3 else '⚠️ PARZIALE'}")

print("\n🔍 IPOTESI 3: Sistemi con Φ più alto mostrano maggior stabilità (SNR più alto)")
phi_means = [metrics_results[c]['mean_phi'] for c in countries]
snr_vals = [metrics_results[c]['SNR'] for c in countries]
corr_coef = np.corrcoef(phi_means, snr_vals)[0, 1]
print(f"   Correlazione Φ medio - SNR: {corr_coef:.3f}")
print(f"   Interpretazione: {'POSITIVA' if corr_coef > 0.3 else 'DEBOLE' if corr_coef > 0 else 'NEGATIVA'}")

print("\n🔍 IPOTESI 4: Trend negativo Φ → possibili indicatori di crisi futura")
negative_trend_countries = [c for c in countries if metrics_results[c]['trend_slope'] < -0.0005]
print(f"   Paesi con trend negativo marcato (< -0.0005/anno): {negative_trend_countries}")
if negative_trend_countries:
    print(f"   Allerta: Questi sistemi potrebbero avvicinarsi a 'lock-in gerarchico'")
else:
    print(f"   Nessun paese mostra trend negativo marcato")

# ============================================================================
# 6. RACCOMANDAZIONI PER STEP SUCCESSIVI
# ============================================================================
print("\n" + "=" * 70)
print("RACCOMANDAZIONI PER STEP 3")
print("=" * 70)

print("\n🎯 BASATO SUI RISULTATI STEP 2, SUGGERISCO:")

high_phi_countries = [c for c in countries if metrics_results[c]['mean_phi'] > 0.6]
stable_countries = [c for c in countries if metrics_results[c]['SNR'] > 2]

print("1. APPROFONDIMENTO PAESI AD ALTA Φ:")
if high_phi_countries:
    print(f"   → Analisi dettagliata: {', '.join(high_phi_countries)}")
else:
    print("   → Nessun paese con Φ particolarmente alto nei dati sintetici")

print("\n2. ANALISI CICLI TEMPORALI:")
print("   → Estrazione periodi dominanti per ogni paese")
print("   → Confronto cicli economici vs cicli equità")

print("\n3. VALIDAZIONE SU DATI REALI PRIORITARI:")
print("   a) World Inequality Database (WID.world)")
print("   b) Dataset Ostrom Common Pool Resources")
print("   c) Distribuzione Bitcoin/blockchain")

print("\n4. TEST PREDITTIVO:")
print("   → Usare Φ(t) pre-2008 per predire recovery post-crisi")

print("\n" + "=" * 70)
print("STEP 2 COMPLETATO ✅")
print("=" * 70)
print("File generati:")
print("   1. step2_country_comparison.png")
print("   2. step2_comparative_dashboard.png")
print("   3. Dati comparativi pronti per step 3")
print("=" * 70)

print("\n💡 Per vedere i grafici:")
print("   xdg-open step2_country_comparison.png")
print("   xdg-open step2_comparative_dashboard.png")
