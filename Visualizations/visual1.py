import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

zhvi_data = pd.read_csv("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
zori_data = pd.read_csv("Metro_zori_sm_month.csv")

# For simplicity, we'll focus on the most recent month's data for both indices, which is September 2023.

# Extract the most recent ZHVI and ZORI values for each region
zhvi_latest = zhvi_data[['RegionName', '2023-09-30']].rename(columns={'2023-09-30': 'ZHVI'})
zori_latest = zori_data[['RegionName', '2023-09-30']].rename(columns={'2023-09-30': 'ZORI'})

# Merge the datasets on the RegionName field
combined_zhvi_zori = pd.merge(zhvi_latest, zori_latest, on='RegionName')

# Calculate the correlation between ZHVI and ZORI for the merged data
correlation = combined_zhvi_zori[['ZHVI', 'ZORI']].corr()

# We will plot only the top 10 most populous regions for clarity in the visualization
top_regions = combined_zhvi_zori.nlargest(10, 'ZHVI')

# Create scatter plot
plt.figure(figsize=(12, 8))
sns.regplot(x='ZHVI', y='ZORI', data=top_regions)

plt.title('Scatter Plot of ZHVI vs ZORI for Top 10 Regions')
plt.xlabel('Zillow Home Value Index (ZHVI)')
plt.ylabel('Zillow Observed Rent Index (ZORI)')
plt.grid(True)
plt.show()

correlation

