"""
Outlier Visualization Script for EMF Synthetic Data
This script creates comprehensive visualizations of outliers in the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the data
print("Loading EMF synthetic data...")
data = pd.read_csv('EMF_Synthetic_Data.csv')

print(f"Total samples: {len(data)}")
print(f"Outliers: {data['is_outlier'].sum()} ({data['is_outlier'].sum()/len(data)*100:.2f}%)")
print(f"Normal samples: {(~data['is_outlier']).sum()}")

# Separate normal and outlier data
normal_data = data[~data['is_outlier']]
outlier_data = data[data['is_outlier']]

# ============================================================================
# PLOT 1: Outlier Distribution Overview
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Outlier Distribution Across All Features', fontsize=16, fontweight='bold')

features = ['temperature_C', 'humidity_percent', 'load_current_A', 
            'distance_from_line_m', 'vertical_distance_m']
titles = ['Temperature (°C)', 'Humidity (%)', 'Load Current (A)', 
          'Distance from Line (m)', 'Vertical Distance (m)']

for idx, (feature, title) in enumerate(zip(features, titles)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Plot normal data
    ax.hist(normal_data[feature], bins=30, alpha=0.6, color='blue', 
            label=f'Normal (n={len(normal_data)})', edgecolor='black')
    
    # Plot outlier data
    ax.hist(outlier_data[feature], bins=15, alpha=0.8, color='red', 
            label=f'Outliers (n={len(outlier_data)})', edgecolor='black')
    
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# EMF fields comparison
ax = axes[1, 2]
x = np.arange(2)
width = 0.35

normal_means = [normal_data['E_field_V_m'].mean(), normal_data['H_field_A_m'].mean()]
outlier_means = [outlier_data['E_field_V_m'].mean(), outlier_data['H_field_A_m'].mean()]

ax.bar(x - width/2, normal_means, width, label='Normal', color='blue', alpha=0.7)
ax.bar(x + width/2, outlier_means, width, label='Outliers', color='red', alpha=0.7)
ax.set_ylabel('Mean Value')
ax.set_title('Mean E and H Fields Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['E Field (V/m)', 'H Field (A/m)'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outlier_distribution_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outlier_distribution_overview.png")
plt.close()

# ============================================================================
# PLOT 2: E Field Outliers - Detailed Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Electric Field (E) - Outlier Analysis', fontsize=16, fontweight='bold')

# 2.1: Box plot comparison
axes[0, 0].boxplot([normal_data['E_field_V_m'], outlier_data['E_field_V_m']], 
                     labels=['Normal', 'Outliers'],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='red', linewidth=2))
axes[0, 0].set_ylabel('E Field (V/m)')
axes[0, 0].set_title('E Field Box Plot Comparison')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2.2: Scatter plot - E field vs Distance
axes[0, 1].scatter(normal_data['distance_from_line_m'], normal_data['E_field_V_m'], 
                    alpha=0.4, s=20, c='blue', label='Normal')
axes[0, 1].scatter(outlier_data['distance_from_line_m'], outlier_data['E_field_V_m'], 
                    alpha=0.9, s=100, c='red', marker='X', edgecolors='darkred', 
                    linewidth=1.5, label='Outliers')
axes[0, 1].set_xlabel('Distance from Line (m)')
axes[0, 1].set_ylabel('E Field (V/m)')
axes[0, 1].set_title('E Field vs Distance (Outliers Highlighted)')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# 2.3: Histogram with outliers highlighted
axes[1, 0].hist(normal_data['E_field_V_m'], bins=50, alpha=0.6, color='blue', 
                 label='Normal', edgecolor='black', density=True)
axes[1, 0].hist(outlier_data['E_field_V_m'], bins=20, alpha=0.8, color='red', 
                 label='Outliers', edgecolor='darkred', density=True)
axes[1, 0].set_xlabel('E Field (V/m)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('E Field Distribution (Normalized)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 2.4: Outlier magnitude analysis
outlier_ratios = []
for idx in outlier_data.index:
    # Find similar normal samples (close distance)
    similar_mask = (np.abs(normal_data['distance_from_line_m'] - 
                           data.loc[idx, 'distance_from_line_m']) < 10)
    if similar_mask.any():
        normal_mean = normal_data[similar_mask]['E_field_V_m'].mean()
        ratio = data.loc[idx, 'E_field_V_m'] / normal_mean if normal_mean > 0 else 0
        outlier_ratios.append(ratio)

axes[1, 1].hist(outlier_ratios, bins=20, color='red', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=1, color='black', linestyle='--', linewidth=2, label='Normal level')
axes[1, 1].set_xlabel('Outlier Ratio (Outlier / Normal Mean)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('E Field Outlier Magnitude Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E_field_outlier_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: E_field_outlier_analysis.png")
plt.close()

# ============================================================================
# PLOT 3: H Field Outliers - Detailed Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Magnetic Field (H) - Outlier Analysis', fontsize=16, fontweight='bold')

# 3.1: Box plot comparison
axes[0, 0].boxplot([normal_data['H_field_A_m'], outlier_data['H_field_A_m']], 
                     labels=['Normal', 'Outliers'],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightgreen'),
                     medianprops=dict(color='red', linewidth=2))
axes[0, 0].set_ylabel('H Field (A/m)')
axes[0, 0].set_title('H Field Box Plot Comparison')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 3.2: Scatter plot - H field vs Load Current
axes[0, 1].scatter(normal_data['load_current_A'], normal_data['H_field_A_m'], 
                    alpha=0.4, s=20, c='green', label='Normal')
axes[0, 1].scatter(outlier_data['load_current_A'], outlier_data['H_field_A_m'], 
                    alpha=0.9, s=100, c='red', marker='X', edgecolors='darkred', 
                    linewidth=1.5, label='Outliers')
axes[0, 1].set_xlabel('Load Current (A)')
axes[0, 1].set_ylabel('H Field (A/m)')
axes[0, 1].set_title('H Field vs Load Current (Outliers Highlighted)')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# 3.3: Histogram with outliers highlighted
axes[1, 0].hist(normal_data['H_field_A_m'], bins=50, alpha=0.6, color='green', 
                 label='Normal', edgecolor='black', density=True)
axes[1, 0].hist(outlier_data['H_field_A_m'], bins=20, alpha=0.8, color='red', 
                 label='Outliers', edgecolor='darkred', density=True)
axes[1, 0].set_xlabel('H Field (A/m)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('H Field Distribution (Normalized)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3.4: Outlier statistics table
outlier_stats = pd.DataFrame({
    'Metric': ['Count', 'Mean E (V/m)', 'Mean H (A/m)', 'Std E', 'Std H', 
               'Max E', 'Max H', 'Min E', 'Min H'],
    'Normal': [
        len(normal_data),
        f"{normal_data['E_field_V_m'].mean():.2f}",
        f"{normal_data['H_field_A_m'].mean():.2f}",
        f"{normal_data['E_field_V_m'].std():.2f}",
        f"{normal_data['H_field_A_m'].std():.2f}",
        f"{normal_data['E_field_V_m'].max():.2f}",
        f"{normal_data['H_field_A_m'].max():.2f}",
        f"{normal_data['E_field_V_m'].min():.2f}",
        f"{normal_data['H_field_A_m'].min():.2f}"
    ],
    'Outliers': [
        len(outlier_data),
        f"{outlier_data['E_field_V_m'].mean():.2f}",
        f"{outlier_data['H_field_A_m'].mean():.2f}",
        f"{outlier_data['E_field_V_m'].std():.2f}",
        f"{outlier_data['H_field_A_m'].std():.2f}",
        f"{outlier_data['E_field_V_m'].max():.2f}",
        f"{outlier_data['H_field_A_m'].max():.2f}",
        f"{outlier_data['E_field_V_m'].min():.2f}",
        f"{outlier_data['H_field_A_m'].min():.2f}"
    ]
})

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=outlier_stats.values, 
                          colLabels=outlier_stats.columns,
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
axes[1, 1].set_title('Statistics Comparison Table', pad=20)

plt.tight_layout()
plt.savefig('H_field_outlier_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: H_field_outlier_analysis.png")
plt.close()

# ============================================================================
# PLOT 4: Outlier Locations in Feature Space
# ============================================================================
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Outlier Locations in Multi-Dimensional Feature Space', 
             fontsize=16, fontweight='bold')

# 4.1: Temperature vs Humidity
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(normal_data['temperature_C'], normal_data['humidity_percent'], 
            alpha=0.3, s=10, c='blue', label='Normal')
ax1.scatter(outlier_data['temperature_C'], outlier_data['humidity_percent'], 
            alpha=0.9, s=80, c='red', marker='*', edgecolors='darkred', 
            linewidth=1, label='Outliers')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Humidity (%)')
ax1.set_title('Weather Conditions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4.2: Distance vs Vertical Distance
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(normal_data['distance_from_line_m'], normal_data['vertical_distance_m'], 
            alpha=0.3, s=10, c='blue', label='Normal')
ax2.scatter(outlier_data['distance_from_line_m'], outlier_data['vertical_distance_m'], 
            alpha=0.9, s=80, c='red', marker='*', edgecolors='darkred', 
            linewidth=1, label='Outliers')
ax2.set_xlabel('Horizontal Distance (m)')
ax2.set_ylabel('Vertical Distance (m)')
ax2.set_title('Spatial Position')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 4.3: E field vs H field
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(normal_data['E_field_V_m'], normal_data['H_field_A_m'], 
            alpha=0.3, s=10, c='blue', label='Normal')
ax3.scatter(outlier_data['E_field_V_m'], outlier_data['H_field_A_m'], 
            alpha=0.9, s=80, c='red', marker='*', edgecolors='darkred', 
            linewidth=1, label='Outliers')
ax3.set_xlabel('E Field (V/m)')
ax3.set_ylabel('H Field (A/m)')
ax3.set_title('E vs H Fields')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4.4: Load Current vs E Field
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(normal_data['load_current_A'], normal_data['E_field_V_m'], 
            alpha=0.3, s=10, c='blue', label='Normal')
ax4.scatter(outlier_data['load_current_A'], outlier_data['E_field_V_m'], 
            alpha=0.9, s=80, c='red', marker='*', edgecolors='darkred', 
            linewidth=1, label='Outliers')
ax4.set_xlabel('Load Current (A)')
ax4.set_ylabel('E Field (V/m)')
ax4.set_title('Current vs E Field')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 4.5: Distance vs H Field
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(normal_data['distance_from_line_m'], normal_data['H_field_A_m'], 
            alpha=0.3, s=10, c='green', label='Normal')
ax5.scatter(outlier_data['distance_from_line_m'], outlier_data['H_field_A_m'], 
            alpha=0.9, s=80, c='red', marker='*', edgecolors='darkred', 
            linewidth=1, label='Outliers')
ax5.set_xlabel('Distance from Line (m)')
ax5.set_ylabel('H Field (A/m)')
ax5.set_title('Distance vs H Field')
ax5.set_yscale('log')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 4.6: Sample distribution timeline
ax6 = plt.subplot(2, 3, 6)
normal_indices = normal_data['sample_id'].values
outlier_indices = outlier_data['sample_id'].values
ax6.scatter(normal_indices, np.zeros_like(normal_indices), alpha=0.3, s=5, 
            c='blue', label='Normal', marker='|')
ax6.scatter(outlier_indices, np.zeros_like(outlier_indices), alpha=0.9, s=100, 
            c='red', label='Outliers', marker='X')
ax6.set_xlabel('Sample ID')
ax6.set_title('Outlier Distribution Across Dataset')
ax6.set_ylim(-0.5, 0.5)
ax6.set_yticks([])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='x')

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('outlier_feature_space.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outlier_feature_space.png")
plt.close()

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n" + "="*70)
print("OUTLIER ANALYSIS SUMMARY REPORT")
print("="*70)

print(f"\nDataset Overview:")
print(f"  Total Samples: {len(data)}")
print(f"  Normal Samples: {len(normal_data)} ({len(normal_data)/len(data)*100:.2f}%)")
print(f"  Outlier Samples: {len(outlier_data)} ({len(outlier_data)/len(data)*100:.2f}%)")

print(f"\nE Field Statistics:")
print(f"  Normal - Mean: {normal_data['E_field_V_m'].mean():.2f} V/m, "
      f"Std: {normal_data['E_field_V_m'].std():.2f}")
print(f"  Outliers - Mean: {outlier_data['E_field_V_m'].mean():.2f} V/m, "
      f"Std: {outlier_data['E_field_V_m'].std():.2f}")
print(f"  Outlier/Normal Ratio: {outlier_data['E_field_V_m'].mean()/normal_data['E_field_V_m'].mean():.2f}x")

print(f"\nH Field Statistics:")
print(f"  Normal - Mean: {normal_data['H_field_A_m'].mean():.2f} A/m, "
      f"Std: {normal_data['H_field_A_m'].std():.2f}")
print(f"  Outliers - Mean: {outlier_data['H_field_A_m'].mean():.2f} A/m, "
      f"Std: {outlier_data['H_field_A_m'].std():.2f}")
print(f"  Outlier/Normal Ratio: {outlier_data['H_field_A_m'].mean()/normal_data['H_field_A_m'].mean():.2f}x")

print(f"\nOutlier Extremes:")
print(f"  Maximum E Field (Outlier): {outlier_data['E_field_V_m'].max():.2f} V/m")
print(f"  Maximum H Field (Outlier): {outlier_data['H_field_A_m'].max():.2f} A/m")
print(f"  Minimum E Field (Outlier): {outlier_data['E_field_V_m'].min():.2f} V/m")
print(f"  Minimum H Field (Outlier): {outlier_data['H_field_A_m'].min():.2f} A/m")

print("\n" + "="*70)
print("Generated Files:")
print("  1. outlier_distribution_overview.png")
print("  2. E_field_outlier_analysis.png")
print("  3. H_field_outlier_analysis.png")
print("  4. outlier_feature_space.png")
print("="*70)
print("\nOutlier visualization complete!")
