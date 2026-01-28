#!/usr/bin/env python3
"""
03_final_comparative.py - Analisi Φ cardiaco vs economico
Dati corretti: phi_cardiac_results.csv contiene phi_cardiac, hr_bpm, hrv_ms, cv_rr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

def simple_regression(x, y):
    """Regressione lineare semplice"""
    # Normalizza x per stabilità numerica
    x_norm = (x - np.mean(x)) / np.std(x)
    A = np.vstack([x_norm, np.ones(len(x_norm))]).T
    m_norm, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Converti pendenza a scala originale
    m = m_norm / np.std(x)
    c = c - m * np.mean(x)
    
    # Calcola R²
    y_pred = m * x + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return m, c, r2, y_pred

def main():
    print("=" * 70)
    print("FASE 3: ANALISI COMPARATIVA Φ CARDIACO vs Φ ECONOMICO")
    print("=" * 70)
    
    # 1. CARICA DATI CARDIACI
    print("\n📊 1. DATI CARDIACI (MIT-BIH Paziente 100)")
    cardiac_path = "../03_analysis/phi_cardiac_results.csv"
    
    if os.path.exists(cardiac_path):
        cardiac_df = pd.read_csv(cardiac_path)
        print(f"   ✅ File trovato: {len(cardiac_df)} riga(e)")
        print(f"   📋 Colonne: {list(cardiac_df.columns)}")
        
        phi_cardiac = cardiac_df['phi_cardiac'].iloc[0]
        hr_bpm = cardiac_df['hr_bpm'].iloc[0]
        hrv_ms = cardiac_df['hrv_ms'].iloc[0]
        cv_rr = cardiac_df['cv_rr'].iloc[0]
        
        print(f"\n   📈 Metriche cardiache:")
        print(f"      • Φ cardiaco (complessità): {phi_cardiac:.4f}")
        print(f"      • Frequenza cardiaca: {hr_bpm:.1f} bpm")
        print(f"      • HRV (variabilità): {hrv_ms:.1f} ms")
        print(f"      • CV_RR (coeff. variazione): {cv_rr:.4f}")
        
        # Interpretazione Φ cardiaco
        if phi_cardiac > 0.9:
            phi_interp = "ALTISSIMA complessità"
        elif phi_cardiac > 0.8:
            phi_interp = "alta complessità"
        elif phi_cardiac > 0.7:
            phi_interp = "media complessità"
        else:
            phi_interp = "bassa complessità"
        
        print(f"      • Interpretazione: {phi_interp}")
        
    else:
        print("   ❌ File cardiaco non trovato")
        return
    
    # 2. CARICA DATI ECONOMICI
    print("\n💰 2. DATI ECONOMICI SIMULATI")
    economic_path = "../02_data_economic/economic_data.csv"
    
    if os.path.exists(economic_path):
        economic_df = pd.read_csv(economic_path)
        print(f"   ✅ File trovato: {len(economic_df)} individui")
        print(f"   📋 Colonne: {list(economic_df.columns)}")
        
        phi_economic_mean = economic_df['phi_economic'].mean()
        phi_economic_std = economic_df['phi_economic'].std()
        income_mean = economic_df['income'].mean()
        
        print(f"\n   📈 Statistiche economiche:")
        print(f"      • Φ economico medio: {phi_economic_mean:.4f} (±{phi_economic_std:.4f})")
        print(f"      • Reddito medio: ${income_mean:,.0f}")
        print(f"      • Range Φ economico: [{economic_df['phi_economic'].min():.3f}, {economic_df['phi_economic'].max():.3f}]")
        
    else:
        print("   ❌ File economico non trovato")
        return
    
    # 3. ANALISI COMPARATIVA
    print("\n" + "=" * 70)
    print("📈 3. ANALISI COMPARATIVA DETTAGLIATA")
    print("=" * 70)
    
    # Confronto diretto Φ
    diff_phi = phi_cardiac - phi_economic_mean
    diff_pct = (diff_phi / phi_economic_mean) * 100
    ratio = phi_cardiac / phi_economic_mean
    
    print(f"\n🔍 CONFRONTO SISTEMI COMPLESSI:")
    print(f"   • Sistema cardiaco:     Φ = {phi_cardiac:.4f}")
    print(f"   • Sistema economico:    Φ = {phi_economic_mean:.4f} (media)")
    print(f"   • ΔΦ (cardiaco - econ): {diff_phi:+.4f}")
    print(f"   • % differenza:         {diff_pct:+.1f}%")
    print(f"   • Rapporto:             {ratio:.3f} : 1")
    
    # Test statistico
    print(f"\n📊 ANALISI INFERENZIALE:")
    
    # Test t: il sistema cardiaco è diverso dalla popolazione economica?
    t_stat, p_value = stats.ttest_1samp(economic_df['phi_economic'], phi_cardiac)
    print(f"   • Test t (un campione):")
    print(f"     - t({len(economic_df)-1}) = {t_stat:.3f}")
    print(f"     - p-value = {p_value:.6f}")
    
    if p_value < 0.001:
        sig_stars = "***"
    elif p_value < 0.01:
        sig_stars = "**"
    elif p_value < 0.05:
        sig_stars = "*"
    else:
        sig_stars = "n.s."
    
    print(f"     - Significatività: {sig_stars}")
    
    if p_value < 0.05:
        print(f"     → CONCLUSIONE: Il sistema cardiaco ha complessità SIGNIFICATIVAMENTE diversa dal sistema economico")
    else:
        print(f"     → CONCLUSIONE: Nessuna differenza significativa nella complessità")
    
    # Correlazione economica
    print(f"\n🔗 CORRELAZIONI ECONOMICHE:")
    corr_income_phi, p_corr = stats.pearsonr(economic_df['income'], economic_df['phi_economic'])
    print(f"   • Reddito vs Φ economico:")
    print(f"     - r = {corr_income_phi:.3f}")
    print(f"     - p = {p_corr:.6f}")
    
    if abs(corr_income_phi) > 0.7:
        strength = "MOLTO FORTE"
    elif abs(corr_income_phi) > 0.5:
        strength = "FORTE"
    elif abs(corr_income_phi) > 0.3:
        strength = "MODERATA"
    elif abs(corr_income_phi) > 0.1:
        strength = "DEBOLE"
    else:
        strength = "TRASCURABILE"
    
    print(f"     - Forza: {strength}")
    
    # Regressione
    print(f"\n📉 REGRESSIONE LINEARE:")
    slope, intercept, r2, y_pred = simple_regression(economic_df['income'].values, 
                                                     economic_df['phi_economic'].values)
    
    print(f"   • Equazione: Φ_econ = {intercept:.6f} + {slope:.10f} * Income")
    print(f"   • R² (varianza spiegata): {r2:.4f} ({r2*100:.1f}%)")
    print(f"   • Interpretazione:")
    print(f"     - Per ogni $10,000: ΔΦ = {slope * 10000:+.6f}")
    print(f"     - Per ogni $50,000: ΔΦ = {slope * 50000:+.6f}")
    
    # 4. VISUALIZZAZIONI
    print("\n" + "=" * 70)
    print("🎨 4. VISUALIZZAZIONE RISULTATI")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Grafico 1: Confronto Φ
    ax1 = plt.subplot(2, 3, 1)
    systems = ['Cardiaco\n(Paziente 100)', 'Economico\n(Media)']
    phi_values = [phi_cardiac, phi_economic_mean]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(systems, phi_values, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Valore Φ (Complessità)', fontsize=12, fontweight='bold')
    ax1.set_title('Confronto Complessità Sistemica', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sulle barre
    for bar, val in zip(bars, phi_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Aggiungi linea di differenza
    ax1.plot([0, 1], [phi_cardiac, phi_cardiac], 'r--', alpha=0.5)
    ax1.text(0.5, phi_cardiac + 0.01, f'Δ = {diff_phi:+.3f}', 
             ha='center', fontweight='bold', color='red')
    
    # Grafico 2: Distribuzione Φ economico
    ax2 = plt.subplot(2, 3, 2)
    n_bins = 30
    hist_data = ax2.hist(economic_df['phi_economic'], bins=n_bins, 
                        alpha=0.7, color='#4ECDC4', edgecolor='black',
                        density=True)
    
    # Aggiungi linee di riferimento
    ax2.axvline(phi_economic_mean, color='green', linestyle='-', 
               linewidth=2, label=f'Media = {phi_economic_mean:.3f}')
    ax2.axvline(phi_cardiac, color='red', linestyle='--', 
               linewidth=3, label=f'Cardiaco = {phi_cardiac:.3f}')
    
    ax2.set_xlabel('Φ economico', fontsize=12)
    ax2.set_ylabel('Densità', fontsize=12)
    ax2.set_title('Distribuzione Complessità Economica', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Grafico 3: Scatter + regressione
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(economic_df['income']/1000, economic_df['phi_economic'], 
                         alpha=0.6, s=30, c=economic_df['phi_economic'], 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Linea di regressione
    x_vals = np.linspace(economic_df['income'].min(), economic_df['income'].max(), 100)
    y_vals = slope * x_vals + intercept
    ax3.plot(x_vals/1000, y_vals, 'r-', linewidth=3, 
            label=f'Regressione (R²={r2:.3f})')
    
    ax3.set_xlabel('Reddito (migliaia di $)', fontsize=12)
    ax3.set_ylabel('Φ economico', fontsize=12)
    ax3.set_title(f'Reddito vs Complessità Economica\nr = {corr_income_phi:.3f}', 
                 fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Valore Φ')
    
    # Grafico 4: Box plot comparativo
    ax4 = plt.subplot(2, 3, 4)
    
    # Prepara dati per box plot
    economic_sample = economic_df['phi_economic'].values[:1000]  # Campione
    
    box_data = [economic_sample]
    box_labels = ['Sistema Economico\n(n=1000 campioni)']
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    
    # Personalizza box
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][0].set_alpha(0.7)
    
    # Aggiungi punto cardiaco
    ax4.plot(1, phi_cardiac, 'r*', markersize=20, 
            label=f'Sistema Cardiaco\nΦ = {phi_cardiac:.3f}')
    
    ax4.set_ylabel('Valore Φ', fontsize=12)
    ax4.set_title('Confronto Distribuzioni', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    # Grafico 5: Metriche cardiache
    ax5 = plt.subplot(2, 3, 5)
    cardiac_metrics = ['Φ Cardiaco', 'FC (bpm)', 'HRV (ms)', 'CV_RR']
    cardiac_values = [phi_cardiac, hr_bpm, hrv_ms, cv_rr]
    cardiac_colors = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2']
    
    bars_c = ax5.bar(cardiac_metrics, cardiac_values, color=cardiac_colors, 
                    edgecolor='black', linewidth=1.5)
    
    ax5.set_ylabel('Valore', fontsize=12)
    ax5.set_title('Metriche Cardiache Dettagliate', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori
    for bar, val in zip(bars_c, cardiac_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + (0.02*max(cardiac_values)),
                f'{val:.3f}' if val < 1 else f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Grafico 6: Risultati test statistico
    ax6 = plt.subplot(2, 3, 6)
    
    # Crea una tabella visiva
    cell_text = [
        ['Metrica', 'Valore', 'Signif.'],
        ['Φ Cardiaco', f'{phi_cardiac:.4f}', ''],
        ['Φ Econ. Medio', f'{phi_economic_mean:.4f}', ''],
        ['ΔΦ', f'{diff_phi:+.4f}', f'{sig_stars}'],
        ['Test t', f'{t_stat:.3f}', f'p={p_value:.4f}'],
        ['Correlazione r', f'{corr_income_phi:.3f}', f'p={p_corr:.4f}'],
        ['R² regressione', f'{r2:.4f}', '']
    ]
    
    # Crea tabella
    table = ax6.table(cellText=cell_text, loc='center', cellLoc='left',
                     colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Colorazione celle
    for i in range(len(cell_text)):
        for j in range(len(cell_text[0])):
            if i == 0:  # Intestazione
                table[(i, j)].set_facecolor('#2D3047')
                table[(i, j)].set_text_props(weight='bold', color='white')
            elif 'p=0.000' in str(cell_text[i][j]):
                table[(i, j)].set_facecolor('#FF6B6B')
                table[(i, j)].set_text_props(weight='bold')
    
    ax6.axis('off')
    ax6.set_title('Riepilogo Statistico', fontsize=14, fontweight='bold', y=0.95)
    
    # Titolo generale
    plt.suptitle('STUDIO Φ-F/EQUITY: Analisi Comparativa Sistemi Complessi\nCardiaco vs Economico', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Salva
    output_path = "../03_analysis/final_comparative_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Grafico salvato: {output_path}")
    
    # 5. SALVA RISULTATI
    print("\n" + "=" * 70)
    print("💾 5. SALVATAGGIO RISULTATI FINALI")
    print("=" * 70)
    
    # Crea dataframe riepilogativo
    summary_data = {
        'timestamp': [pd.Timestamp.now()],
        'phi_cardiac': [phi_cardiac],
        'hr_bpm': [hr_bpm],
        'hrv_ms': [hrv_ms],
        'cv_rr': [cv_rr],
        'phi_economic_mean': [phi_economic_mean],
        'phi_economic_std': [phi_economic_std],
        'income_mean': [income_mean],
        'phi_difference': [diff_phi],
        'phi_difference_pct': [diff_pct],
        'phi_ratio': [ratio],
        't_statistic': [t_stat],
        'p_value': [p_value],
        'correlation_r': [corr_income_phi],
        'correlation_p': [p_corr],
        'regression_r2': [r2],
        'regression_slope': [slope],
        'regression_intercept': [intercept],
        'n_economic_samples': [len(economic_df)],
        'significance_stars': [sig_stars]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Salva in CSV
    csv_path = "../03_analysis/final_comparative_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"   ✅ Risultati CSV: {csv_path}")
    
    # Salva in JSON (più leggibile)
    json_path = "../03_analysis/final_comparative_results.json"
    summary_df.to_json(json_path, orient='records', indent=2)
    print(f"   ✅ Risultati JSON: {json_path}")
    
    # Stampa riepilogo finale
    print("\n" + "=" * 70)
    print("🏁 RIEPILOGO FINALE DELLO STUDIO Φ-F/EQUITY")
    print("=" * 70)
    
    print(f"\n📌 RISULTATI PRINCIPALI:")
    print(f"   1. Il sistema cardiaco mostra Φ = {phi_cardiac:.3f} (complessità {phi_interp})")
    print(f"   2. Il sistema economico mostra Φ medio = {phi_economic_mean:.3f}")
    print(f"   3. Differenza: {diff_phi:+.3f} ({diff_pct:+.1f}%)")
    print(f"   4. Significatività statistica: {sig_stars}")
    
    if diff_phi > 0 and p_value < 0.05:
        print(f"\n💡 INTERPRETAZIONE: Il sistema cardiaco è SIGNIFICATIVAMENTE PIÙ COMPLESSO")
        print(f"   del sistema economico analizzato (ΔΦ = {diff_phi:+.3f}, p < 0.05)")
    elif diff_phi < 0 and p_value < 0.05:
        print(f"\n💡 INTERPRETAZIONE: Il sistema economico è SIGNIFICATIVAMENTE PIÙ COMPLESSO")
    else:
        print(f"\n💡 INTERPRETAZIONE: Nessuna differenza significativa nella complessità")
        print(f"   tra i due sistemi (p = {p_value:.3f})")
    
    print(f"\n📈 CORRELAZIONE ECONOMICA:")
    print(f"   • Reddito vs complessità: r = {corr_income_phi:.3f} ({strength})")
    
    print(f"\n📊 METODOLOGIA:")
    print(f"   • Dati cardiaci: MIT-BIH Database (Paziente 100)")
    print(f"   • Dati economici: Simulazione (n={len(economic_df)})")
    print(f"   • Metrica Φ: Indice di complessità normalizzato [0,1]")
    
    print(f"\n✅ ANALISI COMPLETATA CON SUCCESSO!")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()
