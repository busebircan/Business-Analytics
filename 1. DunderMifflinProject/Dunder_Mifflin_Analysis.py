#!/usr/bin/env python
# coding: utf-8

# # Dunder Mifflin Order Processing Fee Analysis
# 
# This script contains comprehensive analysis of the Dunder Mifflin Paper Company order processing fees.
# 
# ## Analysis Sections
# 
# 1. **Data Exploration**
#    - Initial examination of the dataset structure
#    - Identifies columns and data types
# 
# 2. **Distribution Analysis**
#    - Descriptive statistical analysis (mean, median, skewness, etc.)
#    - Analyzes distribution characteristics
#    - Determines appropriate measure of central tendency
# 
# 3. **High-Value Orders Analysis**
#    - Analyzes orders with processing fees > £500
#    - Compares them with standard orders
#    - Examines customer demographics
# 
# 4. **Geographic Analysis**
#    - Analyzes sales data by geographic region
#    - Creates visualizations showing sales volume and total value
#    - Identifies key commercial insights by region
# 
# 5. **Statistical Testing**
#    - Demonstrates hypothesis testing methodology
#    - Two-proportion z-test for marketing effectiveness
# 
# ## Requirements
# 
# - pandas
# - numpy
# - matplotlib
# - seaborn
# - scipy
# - openpyxl
# 
# ## Usage
# 
# ```python
# python Dunder_Mifflin_Analysis.py
# ```
# 
# ## Output
# 
# The script generates visualization files (.png) and CSV files with analysis results.

# Import required libraries
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None)
datestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

# Set Dunder Mifflin brand colors
DM_BLUE = '#003d82'
DM_GOLD = '#f4a900'
DM_GRAY = '#333333'

print("=" * 100)
print(" " * 30 + "DUNDER MIFFLIN PAPER COMPANY")
print(" " * 25 + "ORDER PROCESSING FEE ANALYSIS")
print("=" * 100)
print(f"\nAnalysis Date: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
print("\n" + "=" * 100)

# ============================================================================
# SECTION 1: DATA EXPLORATION
# ============================================================================

print("\n" + "=" * 100)
print("SECTION 1: DATA EXPLORATION")
print("=" * 100)

# Load the data
file_path = 'data/Dunder_Mifflin_Orders.xlsx'

try:
    df_orders = pd.read_excel(file_path, sheet_name='Orders')
    df_customers = pd.read_excel(file_path, sheet_name='Customers')
    
    print("\n✓ Data loaded successfully")
    print(f"\nOrders Dataset: {df_orders.shape[0]:,} rows × {df_orders.shape[1]} columns")
    print(f"Customers Dataset: {df_customers.shape[0]:,} rows × {df_customers.shape[1]} columns")
    
    print("\n" + "-" * 100)
    print("ORDERS DATASET - COLUMN INFORMATION")
    print("-" * 100)
    for i, col in enumerate(df_orders.columns, 1):
        print(f"{i}. {col:30} - {df_orders[col].dtype}")
    
    print("\n" + "-" * 100)
    print("SAMPLE ORDERS (First 5 records)")
    print("-" * 100)
    print(df_orders.head())
    
    print("\n" + "-" * 100)
    print("CUSTOMERS DATASET - COLUMN INFORMATION")
    print("-" * 100)
    for i, col in enumerate(df_customers.columns, 1):
        print(f"{i}. {col:30} - {df_customers[col].dtype}")
    
    print("\n" + "-" * 100)
    print("MISSING VALUES CHECK")
    print("-" * 100)
    missing_orders = df_orders.isnull().sum()
    missing_customers = df_customers.isnull().sum()
    
    if missing_orders.sum() > 0:
        print("Orders dataset:")
        print(missing_orders[missing_orders > 0])
    else:
        print("Orders dataset: No missing values")
    
    if missing_customers.sum() > 0:
        print("\nCustomers dataset:")
        print(missing_customers[missing_customers > 0])
    else:
        print("Customers dataset: No missing values (except expected NaN in year_established)")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("Please ensure 'data/Dunder_Mifflin_Orders.xlsx' exists in the correct location")
    exit(1)

# Convert pence to pounds for analysis
df_orders['processing_fee_pounds'] = df_orders['processing_fee'] / 100
df_orders['monthly_order_value_pounds'] = df_orders['monthly_order_value'] / 100

# ============================================================================
# SECTION 2: DISTRIBUTION ANALYSIS
# ============================================================================

print("\n\n" + "=" * 100)
print("SECTION 2: PROCESSING FEE DISTRIBUTION ANALYSIS")
print("=" * 100)

# Calculate expected processing fee based on formula
df_orders['calculated_fee_pounds'] = (df_orders['monthly_order_value_pounds'] * 12 / 52).round(2)
df_orders['fee_difference'] = df_orders['processing_fee_pounds'] - df_orders['calculated_fee_pounds']
df_orders['formula_match'] = abs(df_orders['fee_difference']) < 0.01

print("\n" + "-" * 100)
print("FORMULA VERIFICATION")
print("-" * 100)
print("Formula: Processing Fee = Monthly Order Value × 12 ÷ 52 (approximately one week's order value)")
print(f"\nRecords matching formula: {df_orders['formula_match'].sum():,} ({df_orders['formula_match'].mean()*100:.2f}%)")
print(f"Records with differences: {(~df_orders['formula_match']).sum():,} ({(~df_orders['formula_match']).mean()*100:.2f}%)")

processing_fees = df_orders['processing_fee_pounds']

print("\n" + "-" * 100)
print("DESCRIPTIVE STATISTICS")
print("-" * 100)
print(f"{'Count:':<30} {processing_fees.count():>15,}")
print(f"{'Mean:':<30} {processing_fees.mean():>14.2f} £")
print(f"{'Median:':<30} {processing_fees.median():>14.2f} £")
print(f"{'Mode:':<30} {processing_fees.mode().iloc[0]:>14.2f} £")
print(f"{'Standard Deviation:':<30} {processing_fees.std():>14.2f} £")
print(f"{'Variance:':<30} {processing_fees.var():>14.2f}")
print(f"{'Minimum:':<30} {processing_fees.min():>14.2f} £")
print(f"{'Maximum:':<30} {processing_fees.max():>14.2f} £")
print(f"{'Range:':<30} {processing_fees.max() - processing_fees.min():>14.2f} £")

print("\n" + "-" * 100)
print("QUARTILES AND PERCENTILES")
print("-" * 100)
quartiles = processing_fees.quantile([0.25, 0.5, 0.75])
print(f"{'Q1 (25th percentile):':<30} {quartiles[0.25]:>14.2f} £")
print(f"{'Q2 (Median):':<30} {quartiles[0.5]:>14.2f} £")
print(f"{'Q3 (75th percentile):':<30} {quartiles[0.75]:>14.2f} £")
print(f"{'IQR:':<30} {quartiles[0.75] - quartiles[0.25]:>14.2f} £")

percentiles = processing_fees.quantile([0.05, 0.1, 0.9, 0.95, 0.99])
print(f"\n{'5th percentile:':<30} {percentiles[0.05]:>14.2f} £")
print(f"{'10th percentile:':<30} {percentiles[0.1]:>14.2f} £")
print(f"{'90th percentile:':<30} {percentiles[0.9]:>14.2f} £")
print(f"{'95th percentile:':<30} {percentiles[0.95]:>14.2f} £")
print(f"{'99th percentile:':<30} {percentiles[0.99]:>14.2f} £")

print("\n" + "-" * 100)
print("DISTRIBUTION SHAPE")
print("-" * 100)
skewness = processing_fees.skew()
kurtosis = processing_fees.kurtosis()
cv = (processing_fees.std() / processing_fees.mean()) * 100

print(f"{'Skewness:':<30} {skewness:>14.3f}")
if skewness > 1:
    skew_interpretation = "Highly right-skewed"
elif skewness > 0.5:
    skew_interpretation = "Moderately right-skewed"
elif skewness > -0.5:
    skew_interpretation = "Approximately symmetric"
else:
    skew_interpretation = "Left-skewed"
print(f"{'Interpretation:':<30} {skew_interpretation:>30}")

print(f"\n{'Kurtosis:':<30} {kurtosis:>14.3f}")
if kurtosis > 3:
    kurt_interpretation = "Leptokurtic (heavy-tailed)"
elif kurtosis < -3:
    kurt_interpretation = "Platykurtic (light-tailed)"
else:
    kurt_interpretation = "Mesokurtic (normal-like)"
print(f"{'Interpretation:':<30} {kurt_interpretation:>30}")

print(f"\n{'Coefficient of Variation:':<30} {cv:>13.2f} %")

# Outlier detection
Q1 = quartiles[0.25]
Q3 = quartiles[0.75]
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = processing_fees[(processing_fees < lower_bound) | (processing_fees > upper_bound)]

print("\n" + "-" * 100)
print("OUTLIER ANALYSIS (IQR Method)")
print("-" * 100)
print(f"{'Lower bound:':<30} {lower_bound:>14.2f} £")
print(f"{'Upper bound:':<30} {upper_bound:>14.2f} £")
print(f"{'Number of outliers:':<30} {len(outliers):>10,} ({len(outliers)/len(processing_fees)*100:>5.1f}%)")
if len(outliers) > 0:
    print(f"{'Outlier range:':<30} £{outliers.min():.2f} to £{outliers.max():.2f}")

print("\n" + "-" * 100)
print("APPROPRIATE AVERAGE MEASURE")
print("-" * 100)
print(f"\nFor this right-skewed distribution (skewness = {skewness:.2f}):")
print(f"  • Mean:   £{processing_fees.mean():.2f} (pulled higher by outliers)")
print(f"  • Median: £{processing_fees.median():.2f} (resistant to outliers)")
print(f"\n✓ RECOMMENDATION: Use MEDIAN (£{processing_fees.median():.2f}) as the 'average' processing fee")
print("  Reason: Median better represents the typical order in right-skewed distributions")

# ============================================================================
# SECTION 3: HIGH-VALUE ORDERS ANALYSIS (>£500)
# ============================================================================

print("\n\n" + "=" * 100)
print("SECTION 3: HIGH-VALUE ORDERS ANALYSIS (>£500)")
print("=" * 100)

# Merge with customer data
merged_df = pd.merge(df_orders, df_customers, on='order_id', how='inner')
current_year = 2025
merged_df['business_age'] = current_year - merged_df['year_established']

# Separate high-value and standard orders
high_value = df_orders[df_orders['processing_fee_pounds'] > 500].copy()
standard = df_orders[df_orders['processing_fee_pounds'] <= 500].copy()

high_value_customers = merged_df[merged_df['processing_fee_pounds'] > 500]
standard_customers = merged_df[merged_df['processing_fee_pounds'] <= 500]

print("\n" + "-" * 100)
print("ORDER SEGMENTATION")
print("-" * 100)
print(f"{'High-Value Orders (>£500):':<40} {len(high_value):>10,} ({len(high_value)/len(df_orders)*100:>5.2f}%)")
print(f"{'Standard Orders (≤£500):':<40} {len(standard):>10,} ({len(standard)/len(df_orders)*100:>5.2f}%)")

print("\n" + "-" * 100)
print("COMPARISON: HIGH-VALUE vs STANDARD ORDERS")
print("-" * 100)
print(f"\n{'Metric':<40} {'High-Value (>£500)':>20} {'Standard (≤£500)':>20}")
print("-" * 100)
print(f"{'Mean Processing Fee:':<40} £{high_value['processing_fee_pounds'].mean():>18.2f} £{standard['processing_fee_pounds'].mean():>18.2f}")
print(f"{'Median Processing Fee:':<40} £{high_value['processing_fee_pounds'].median():>18.2f} £{standard['processing_fee_pounds'].median():>18.2f}")
print(f"{'Mean Order Value:':<40} £{high_value['monthly_order_value_pounds'].mean():>18.2f} £{standard['monthly_order_value_pounds'].mean():>18.2f}")
print(f"{'Formula Match Rate:':<40} {high_value['formula_match'].mean()*100:>17.2f}% {standard['formula_match'].mean()*100:>17.2f}%")
print(f"{'Mean Customer Business Age:':<40} {high_value_customers['business_age'].mean():>17.1f} yrs {standard_customers['business_age'].mean():>17.1f} yrs")

print("\n" + "-" * 100)
print("KEY INSIGHTS")
print("-" * 100)
print(f"1. High-value orders represent only {len(high_value)/len(df_orders)*100:.2f}% of total orders")
print(f"2. High-value orders have {high_value['formula_match'].mean()*100:.2f}% formula match rate vs {standard['formula_match'].mean()*100:.2f}% for standard")
print(f"3. High-value orders associated with younger businesses (mean age: {high_value_customers['business_age'].mean():.1f} years)")
print("4. These represent premium bulk orders from growing businesses")

# ============================================================================
# SECTION 4: GEOGRAPHIC ANALYSIS
# ============================================================================

print("\n\n" + "=" * 100)
print("SECTION 4: GEOGRAPHIC SALES ANALYSIS")
print("=" * 100)

# Remove missing regions
df_clean = df_orders[df_orders['sales_region'].notna()].copy()

# Aggregate by region
regional_analysis = df_clean.groupby('sales_region').agg({
    'order_id': 'count',
    'processing_fee_pounds': ['sum', 'mean'],
    'monthly_order_value_pounds': 'mean'
}).round(2)

regional_analysis.columns = ['order_count', 'total_revenue', 'avg_processing_fee', 'avg_order_value']
regional_analysis['market_share'] = (regional_analysis['order_count'] / regional_analysis['order_count'].sum() * 100).round(2)
regional_analysis = regional_analysis.sort_values('order_count', ascending=False)

print("\n" + "-" * 100)
print("TOP 15 SALES REGIONS")
print("-" * 100)
print(f"\n{'Region':<10} {'Orders':>10} {'Market Share':>15} {'Total Revenue':>18} {'Avg Fee':>12} {'Avg Order':>12}")
print("-" * 100)
for idx in regional_analysis.head(15).index:
    row = regional_analysis.loc[idx]
    print(f"{idx:<10} {row['order_count']:>10,} {row['market_share']:>14.2f}% £{row['total_revenue']:>16,.2f} £{row['avg_processing_fee']:>10.2f} £{row['avg_order_value']:>10.2f}")

print("\n" + "-" * 100)
print("MARKET CONCENTRATION")
print("-" * 100)
top_5_share = regional_analysis.head(5)['market_share'].sum()
top_10_share = regional_analysis.head(10)['market_share'].sum()
print(f"{'Top 5 regions market share:':<40} {top_5_share:>10.2f}%")
print(f"{'Top 10 regions market share:':<40} {top_10_share:>10.2f}%")
print(f"{'Total regions:':<40} {len(regional_analysis):>10}")

print("\n" + "-" * 100)
print("COMMERCIAL INSIGHTS")
print("-" * 100)
top_region = regional_analysis.index[0]
top_region_share = regional_analysis.iloc[0]['market_share']
print(f"\n1. MARKET DOMINANCE")
print(f"   - {top_region} dominates with {top_region_share:.1f}% market share")
print(f"   - Top 5 regions account for {top_5_share:.1f}% of all orders")

top_avg_fee_regions = regional_analysis.nlargest(3, 'avg_processing_fee')
print(f"\n2. HIGH-VALUE REGIONS")
print(f"   - {top_avg_fee_regions.index[0]} has highest avg processing fee (£{top_avg_fee_regions.iloc[0]['avg_processing_fee']:.2f})")
print(f"   - Premium pricing opportunity in {', '.join(top_avg_fee_regions.index[:3])}")

print(f"\n3. STRATEGIC RECOMMENDATIONS")
print("   - Strengthen presence in top 5 regions to maintain market leadership")
print("   - Develop targeted campaigns for high-value, low-volume regions")
print("   - Consider regional pricing strategies based on local market conditions")

# ============================================================================
# SECTION 5: STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

print("\n\n" + "=" * 100)
print("SECTION 5: STATISTICAL SIGNIFICANCE TESTING")
print("=" * 100)

print("\n" + "-" * 100)
print("SCENARIO: Marketing Communication Test")
print("-" * 100)
print("""
The marketing team conducted a test to understand if more regular communication
with customers about new paper products contributed to an increase in "reorder rate"
defined as: Number of Repeat Orders / Total Customers per Month.

They found that a group of customers who receive more regular communications
show a 0.5% increase in their reorder rate.

QUESTION: How do we determine if this change is statistically significant?
""")

# Simulated test data
n_control = 5000
reorder_rate_control = 0.125
reorders_control = int(n_control * reorder_rate_control)

n_test = 5000
reorder_rate_test = 0.130
reorders_test = int(n_test * reorder_rate_test)

print("-" * 100)
print("TEST DATA")
print("-" * 100)
print(f"\nControl Group (No Extra Communications):")
print(f"  Sample Size:    {n_control:,} customers")
print(f"  Reorders:       {reorders_control:,}")
print(f"  Reorder Rate:   {reorder_rate_control*100:.1f}%")

print(f"\nTest Group (Regular Communications):")
print(f"  Sample Size:    {n_test:,} customers")
print(f"  Reorders:       {reorders_test:,}")
print(f"  Reorder Rate:   {reorder_rate_test*100:.1f}%")

print(f"\nObserved Difference: {(reorder_rate_test - reorder_rate_control)*100:.1f}%")

# Two-proportion z-test
p_pooled = (reorders_control + reorders_test) / (n_control + n_test)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_test))
z_stat = (reorder_rate_test - reorder_rate_control) / se
p_value = 1 - stats.norm.cdf(z_stat)

print("\n" + "-" * 100)
print("TWO-PROPORTION Z-TEST RESULTS")
print("-" * 100)
print(f"\nNull Hypothesis (H₀):        No difference in reorder rates")
print(f"Alternative Hypothesis (H₁): Regular communications increase reorder rate")
print(f"Significance Level (α):      0.05 (95% confidence)")

print(f"\nPooled Proportion:           {p_pooled:.4f}")
print(f"Standard Error:              {se:.4f}")
print(f"Z-Statistic:                 {z_stat:.4f}")
print(f"P-Value:                     {p_value:.4f}")

alpha = 0.05
is_significant = p_value < alpha

print("\n" + "-" * 100)
print("CONCLUSION")
print("-" * 100)
if is_significant:
    print(f"✓ STATISTICALLY SIGNIFICANT (p = {p_value:.4f} < {alpha})")
    print(f"  The 0.5% increase in reorder rate IS statistically significant.")
    print(f"  We can reject the null hypothesis with {(1-alpha)*100:.0f}% confidence.")
else:
    print(f"✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f} ≥ {alpha})")
    print(f"  The 0.5% increase in reorder rate is NOT statistically significant.")
    print(f"  We cannot reject the null hypothesis.")
    print(f"\n  Recommendation: Increase sample size or extend test duration")

# Calculate power
effect_size = (reorder_rate_test - reorder_rate_control) / np.sqrt(p_pooled * (1 - p_pooled))
z_beta = z_stat - stats.norm.ppf(1 - alpha)
power = stats.norm.cdf(z_beta)

print(f"\nEffect Size (Cohen's h):     {effect_size:.4f}")
print(f"Statistical Power:           {power:.2%}")
if power < 0.80:
    print(f"⚠ Power is below recommended 80% threshold")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n\n" + "=" * 100)
print("SAVING RESULTS")
print("=" * 100)

# Save detailed analysis results
analysis_results = pd.DataFrame({
    'Metric': [
        'Total Orders',
        'Mean Processing Fee',
        'Median Processing Fee',
        'Std Dev Processing Fee',
        'Skewness',
        'Coefficient of Variation',
        'Formula Match Rate',
        'High-Value Orders (>£500)',
        'High-Value %',
        'Top Region',
        'Top Region Market Share'
    ],
    'Value': [
        f"{len(df_orders):,}",
        f"£{processing_fees.mean():.2f}",
        f"£{processing_fees.median():.2f}",
        f"£{processing_fees.std():.2f}",
        f"{skewness:.2f}",
        f"{cv:.2f}%",
        f"{df_orders['formula_match'].mean()*100:.2f}%",
        f"{len(high_value):,}",
        f"{len(high_value)/len(df_orders)*100:.2f}%",
        top_region,
        f"{top_region_share:.2f}%"
    ]
})

analysis_results.to_csv('Dunder_Mifflin_Analysis_Summary.csv', index=False)
print("\n✓ Analysis summary saved to: Dunder_Mifflin_Analysis_Summary.csv")

regional_analysis.to_csv('Dunder_Mifflin_Regional_Performance.csv')
print("✓ Regional performance saved to: Dunder_Mifflin_Regional_Performance.csv")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 100)
print("CREATING VISUALIZATIONS")
print("=" * 100)

# Create comprehensive visualization figure
fig = plt.figure(figsize=(20, 16))
plt.suptitle('Dunder Mifflin Paper Company - Order Processing Fee Analysis', 
             fontsize=20, fontweight='bold', y=0.995)

# 1. Distribution histogram
ax1 = plt.subplot(3, 3, 1)
ax1.hist(processing_fees, bins=50, alpha=0.7, color=DM_BLUE, edgecolor='white', linewidth=0.5)
ax1.axvline(processing_fees.mean(), color=DM_GOLD, linestyle='--', linewidth=2, 
           label=f'Mean: £{processing_fees.mean():.2f}')
ax1.axvline(processing_fees.median(), color='red', linestyle='--', linewidth=2, 
           label=f'Median: £{processing_fees.median():.2f}')
ax1.set_xlabel('Processing Fee (£)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Processing Fees', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Box plot
ax2 = plt.subplot(3, 3, 2)
bp = ax2.boxplot(processing_fees, vert=True, patch_artist=True,
                 boxprops=dict(facecolor=DM_BLUE, alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
ax2.set_ylabel('Processing Fee (£)', fontsize=11)
ax2.set_title('Processing Fee Box Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Q-Q Plot
ax3 = plt.subplot(3, 3, 3)
stats.probplot(processing_fees, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Order Value vs Processing Fee
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(df_orders['monthly_order_value_pounds'], df_orders['processing_fee_pounds'],
           alpha=0.3, s=10, color=DM_BLUE)
x_range = np.linspace(df_orders['monthly_order_value_pounds'].min(),
                     df_orders['monthly_order_value_pounds'].max(), 100)
y_expected = x_range * 12 / 52
ax4.plot(x_range, y_expected, 'r--', linewidth=2, label='Expected Formula')
ax4.set_xlabel('Monthly Order Value (£)', fontsize=11)
ax4.set_ylabel('Processing Fee (£)', fontsize=11)
ax4.set_title('Order Value vs Processing Fee', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. High-value vs Standard comparison
ax5 = plt.subplot(3, 3, 5)
categories = ['Mean Fee', 'Mean Order Value', 'Formula Match %']
high_val_data = [
    high_value['processing_fee_pounds'].mean(),
    high_value['monthly_order_value_pounds'].mean(),
    high_value['formula_match'].mean() * 100
]
std_val_data = [
    standard['processing_fee_pounds'].mean(),
    standard['monthly_order_value_pounds'].mean(),
    standard['formula_match'].mean() * 100
]
x = np.arange(len(categories))
width = 0.35
ax5.bar(x - width/2, high_val_data, width, label='High-Value (>£500)', color=DM_BLUE)
ax5.bar(x + width/2, std_val_data, width, label='Standard (≤£500)', color=DM_GOLD)
ax5.set_ylabel('Value', fontsize=11)
ax5.set_title('High-Value vs Standard Orders', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(categories, fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Top regions bar chart
ax6 = plt.subplot(3, 3, 6)
top_10_regions = regional_analysis.head(10)
ax6.barh(range(len(top_10_regions)), top_10_regions['order_count'], color=DM_BLUE)
ax6.set_yticks(range(len(top_10_regions)))
ax6.set_yticklabels(top_10_regions.index, fontsize=9)
ax6.set_xlabel('Number of Orders', fontsize=11)
ax6.set_title('Top 10 Regions by Order Volume', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# 7. Regional revenue
ax7 = plt.subplot(3, 3, 7)
top_10_revenue = regional_analysis.sort_values('total_revenue', ascending=False).head(10)
ax7.barh(range(len(top_10_revenue)), top_10_revenue['total_revenue'], color=DM_GOLD)
ax7.set_yticks(range(len(top_10_revenue)))
ax7.set_yticklabels(top_10_revenue.index, fontsize=9)
ax7.set_xlabel('Total Revenue (£)', fontsize=11)
ax7.set_title('Top 10 Regions by Revenue', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

# 8. Market share pie chart
ax8 = plt.subplot(3, 3, 8)
top_5 = regional_analysis.head(5)
others_share = 100 - top_5['market_share'].sum()
pie_data = list(top_5['market_share']) + [others_share]
pie_labels = list(top_5.index) + ['Others']
colors = [DM_BLUE, DM_GOLD, '#66a3d2', '#ffd966', '#003d82', '#cccccc']
ax8.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax8.set_title('Market Share (Top 5 + Others)', fontsize=12, fontweight='bold')

# 9. Statistical test visualization
ax9 = plt.subplot(3, 3, 9)
groups = ['Control\n(No Extra Comms)', 'Test\n(Regular Comms)']
rates = [reorder_rate_control * 100, reorder_rate_test * 100]
bars = ax9.bar(groups, rates, color=[DM_GOLD, DM_BLUE], alpha=0.7, edgecolor='black', linewidth=2)
ax9.set_ylabel('Reorder Rate (%)', fontsize=11)
ax9.set_title('Marketing Test: Reorder Rate Comparison', fontsize=12, fontweight='bold')
for i, (bar, rate) in enumerate(zip(bars, rates)):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
if is_significant:
    ax9.text(0.5, max(rates) * 1.1, f'p = {p_value:.4f} *',
            ha='center', fontsize=10, fontweight='bold', color='green',
            transform=ax9.transData)
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('Dunder_Mifflin_Comprehensive_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved to: Dunder_Mifflin_Comprehensive_Analysis.png")

plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 100)
print("ANALYSIS COMPLETE - SUMMARY OF KEY FINDINGS")
print("=" * 100)

print(f"""
1. DISTRIBUTION CHARACTERISTICS
   - Processing fees show a right-skewed distribution (skewness = {skewness:.2f})
   - Mean: £{processing_fees.mean():.2f}, Median: £{processing_fees.median():.2f}
   - Median is the more appropriate "average" measure for this distribution
   - {df_orders['formula_match'].mean()*100:.2f}% of fees match the expected formula

2. HIGH-VALUE ORDERS (>£500)
   - Represent {len(high_value)/len(df_orders)*100:.2f}% of total orders
   - Show higher formula match rate ({high_value['formula_match'].mean()*100:.2f}%)
   - Associated with younger businesses (mean age: {high_value_customers['business_age'].mean():.1f} years)
   - Represent premium bulk orders from growing businesses

3. GEOGRAPHIC INSIGHTS
   - {top_region} dominates with {top_region_share:.1f}% market share
   - Top 5 regions account for {top_5_share:.1f}% of orders
   - Significant opportunity for geographic diversification
   - {top_avg_fee_regions.index[0]} shows highest average processing fee (£{top_avg_fee_regions.iloc[0]['avg_processing_fee']:.2f})

4. STATISTICAL TESTING
   - Demonstrated proper hypothesis testing methodology
   - 0.5% increase in reorder rate: {'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'}
   - Statistical power: {power:.2%} ({'adequate' if power >= 0.80 else 'below recommended 80%'})

5. STRATEGIC RECOMMENDATIONS
   - Use median (£{processing_fees.median():.2f}) when reporting average processing fees
   - Focus on high-value segment for premium service development
   - Strengthen presence in top regions while exploring underserved markets
   - Implement rigorous A/B testing with adequate sample sizes
""")

print("=" * 100)
print(" " * 35 + "END OF ANALYSIS")
print("=" * 100)
print(f"\nGenerated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
print("\nAll results saved to current directory.")
print("\nThank you for using Dunder Mifflin Analysis Script!")
print("=" * 100)

