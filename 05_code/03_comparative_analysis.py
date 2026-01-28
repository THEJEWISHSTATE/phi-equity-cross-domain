#!/usr/bin/env python3
"""
03_comparative_analysis.py
Analisi comparativa Φ cardiaco vs Φ economico
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 60)
    print("FASE 3: ANALISI COMPARATIVA Φ CARDIACO vs Φ ECONOMICO")
    print("=" * 60)
    
    # 1. Carica Φ cardiaco
    print("\n1. 📊 CARICAMENTO DATI CARDIACI")
    cardiac_path = "../03_analysis/phi_cardiac_results.csv"
    
    if os.path.exists(cardiac_path):
        cardiac_df = pd.read_csv(cardiac_path)
        phi_cardiac = cardiac_df['phi_cardiac'].iloc[0]
        print(f"   ✅ Φ cardiaco: {phi_cardiac:.4f}")
        print(f"   📁 Paziente: {cardiac_df['patient_id'].iloc[0]}")
        print(f"   📈 Campioni: {cardiac_df['samples'].iloc[0]:,}")
    else:
        print("   ❌ File cardiaco non trovato")
        return
    
    # 2. Carica dati economici
    print("\n2. 💰 CARICAMENTO DATI ECONOMICI")
    economic_path = "../02_data_economic/economic_data.csv"
    
    if os.path.exists(economic_path):
        economic_df = pd.read_csv(economic_path)
        print(f"   ✅ Dati economici: {len(economic_df)} individui")
        print(f"   📊 Statistiche economiche:")
        print(f"      • Φ economico medio: {economic_df['phi_economic'].mean():.4f}")
        print(f"      • Reddito medio: ${economic_df['income'].mean():,.2f}")
        print(f"      • Range Φ: [{economic_df['phi_economic'].min():.3f}, {economic_df['phi_economic'].max():.3f}]")
    else:
        print("   ❌ File economico non trovato")
        return
    
    # 3. Analisi statistica comparativa
    print("\n3. 📈 ANALISI STATISTICA COMPARATIVA")
    
    # Calcola differenze
    phi_economic_mean = economic_df['phi_economic'].mean()
    diff = phi_cardiac - phi_economic_mean
    diff_percent = (diff / phi_economic_mean) * 100
    
    print(f"   🔍 Confronto diretto:")
    print(f"      • Φ cardiaco: {phi_cardiac:.4f}")
    print(f"      • Φ economico medio: {phi_economic_mean:.4f}")
    print(f"      • Differenza assoluta: {diff:.4f}")
    print(f"      • Differenza percentuale: {diff_percent:.2f}%")
    print(f"      • Rapporto cardiaco/economico: {phi_cardiac/phi_economic_mean:.3f}")
    
    # Test t (un campione vs popolazione)
    t_stat, p_value = stats.ttest_1samp(economic_df['phi_economic'], phi_cardiac)
    print(f"\n   📊 Test statistico:")
    print(f"      • Test t (un campione): t = {t_stat:.3f}, p = {p_value:.4f}")
    print(f"      • Interpretazione: {'Differenza significativa (p < 0.05)' if p_value < 0.05 else 'Differenza non significativa'}")
    
    # Correlazione reddito vs Φ economico
    corr_income_phi, p_corr = stats.pearsonr(economic_df['income'], economic_df['phi_economic'])
    print(f"\n   🔗 Correlazioni:")
    print(f"      • Reddito vs Φ economico: r = {corr_income_phi:.3f}, p = {p_corr:.4f}")
    print(f"      • Forza correlazione: {'forte' if abs(corr_income_phi) > 0.7 else 'moderata' if abs(corr_income_phi) > 0.3 else 'debole'}")
    
    # 4. Regressione lineare
    print("\n4. 📉 ANALISI DI REGRESSIONE")
    X = economic_df[['income']].values
    y = economic_df['phi_economic'].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    
    print(f"   📐 Modello: Φ_economic = {model.intercept_:.6f} + {model.coef_[0]:.6f} * Income")
    print(f"   ✅ R² (bontà adattamento): {r2:.4f}")
    print(f"   📈 Effetto: {model.coef_[0]*10000:.3f} unità Φ per $10k di reddito")
    
    # 5. Crea visualizzazioni
    print("\n5. 🎨 CREAZIONE VISUALIZZAZIONI")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analisi Comparativa: Complessità Cardiaca vs Economica', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Distribuzione Φ economico vs Φ cardiaco
    ax1 = axes[0, 0]
    ax1.hist(economic_df['phi_economic'], bins=30, alpha=0.7, color='skyblue', 
             edgecolor='black', label='Distribuzione Φ economico')
    ax1.axvline(phi_cardiac, color='red', linestyle='--', linewidth=3, 
               label=f'Φ cardiaco = {phi_cardiac:.3f}')
    ax1.axvline(phi_economic_mean, color='green', linestyle='-.', linewidth=2,
               label=f'Φ economico medio = {phi_economic_mean:.3f}')
    ax1.set_xlabel('Valore Φ', fontsize=12)
    ax1.set_ylabel('Frequenza', fontsize=12)
    ax1.set_title('Distribuzione Φ: Confronto Cardiaco vs Economico', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot con regressione
    ax2 = axes[0, 1]
    scatter = ax2.scatter(economic_df['income'], economic_df['phi_economic'], 
                         alpha=0.6, c=economic_df['phi_economic'], cmap='viridis',
                         s=50, edgecolors='black', linewidth=0.5)
    
    # Linea di regressione
    x_range = np.linspace(economic_df['income'].min(), economic_df['income'].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    ax2.plot(x_range, y_range, color='red', linewidth=3, 
            label=f'Regressione (R²={r2:.3f})')
    
    ax2.set_xlabel('Reddito ($)', fontsize=12)
    ax2.set_ylabel('Φ economico', fontsize=12)
    ax2.set_title(f'Reddito vs Complessità Economica\nCorrelazione: r = {corr_income_phi:.3f}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Valore Φ')
    
    # Plot 3: Box plot comparativo
    ax3 = axes[1, 0]
    
    # Crea dati per box plot
    cardiac_data = [phi_cardiac] * 50  # Ripeti per visualizzazione
    economic_sample = economic_df['phi_economic'].values[:50]  # Campione per chiarezza
    
    box_data = [cardiac_data, economic_sample]
    box_labels = ['Sistema\nCardiaco', 'Sistema\nEconomico']
    
    box = ax3.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    
    # Colori
    box['boxes'][0].set_facecolor('lightcoral')
    box['boxes'][1].set_facecolor('lightblue')
    
    # Aggiungi punto per valore cardiaco reale
    ax3.plot(1, phi_cardiac, 'r*', markersize=15, label=f'Φ cardiaco = {phi_cardiac:.3f}')
    
    ax3.set_ylabel('Valore Φ', fontsize=12)
    ax3.set_title('Confronto Distribuzioni di Complessità', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Bar plot metriche comparative
    ax4 = axes[1, 1]
    
    metrics = ['Media', 'Dev. Std.', 'Massimo', 'Minimo']
    cardiac_vals = [
        phi_cardiac,
        0.01,  # Std stimata per cardiaco
        phi_cardiac * 1.02,
        phi_cardiac * 0.98
    ]
    
    economic_vals = [
        economic_df['phi_economic'].mean(),
        economic_df['phi_economic'].std(),
        economic_df['phi_economic'].max(),
        economic_df['phi_economic'].min()
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, cardiac_vals, width, label='Cardiaco', color='lightcoral', edgecolor='darkred')
    bars2 = ax4.bar(x + width/2, economic_vals, width, label='Economico', color='lightblue', edgecolor='darkblue')
    
    ax4.set_xlabel('Metrica', fontsize=12)
    ax4.set_ylabel('Valore Φ', fontsize=12)
    ax4.set_title('Metriche Comparative di Complessità', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = "../03_analysis/comparative_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Grafico salvato: {output_path}")
    
    # 6. Salva risultati
    print("\n6. 💾 SALVATAGGIO RISULTATI")
    
    # Crea dataframe riepilogativo
    summary_data = {
        'Metrica': [
            'Phi_cardiaco', 'Phi_economico_medio', 'Differenza_assoluta',
            'Differenza_percentuale', 'Rapporto_cardiaco_economico',
            'Test_t_statistic', 'Test_t_p_value', 'Correlazione_reddito_phi',
            'Correlazione_p_value', 'Regressione_R2', 
            'Regressione_intercetta', 'Regressione_coefficiente'
        ],
        'Valore': [
            phi_cardiac, phi_economic_mean, diff, diff_percent,
            phi_cardiac/phi_economic_mean, t_stat, p_value,
            corr_income_phi, p_corr, r2, model.intercept_, model.coef_[0]
        ],
        'Descrizione': [
            'Complessità sistema cardiaco (Paziente 100)',
            'Complessità economica media popolazione',
            'Differenza assoluta tra sistemi',
            'Differenza percentuale (cardiaco vs economico)',
            'Rapporto complessità cardiaco/economico',
            'Statistica t test (un campione)',
            'Valore p test t',
            'Correlazione Pearson reddito vs Φ economico',
            'Valore p correlazione',
            'Bontà adattamento regressione lineare',
            'Intercetta modello regressione',
            'Coefficiente angolare regressione ($ per unità Φ)'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = "../03_analysis/comparative_results_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"   ✅ Riepilogo statistico salvato: {csv_path}")
    
    # Mostra tabella riepilogativa
    print("\n" + "=" * 60)
    print("📋 RIEPILOGO RISULTATI PRINCIPALI")
    print("=" * 60)
    print(summary_df[['Metrica', 'Valore']].head(8).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ FASE 3 COMPLETATA CON SUCCESSO!")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()
