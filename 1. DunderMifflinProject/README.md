
# Dunder Mifflin Paper Company - Order Processing Fee Analysis

## Project Overview

This portfolio project demonstrates comprehensive data analysis and statistical techniques applied to a fictional paper distribution company, Dunder Mifflin Paper Company. The analysis examines order processing fees, customer demographics, and geographic sales patterns to provide actionable business insights.

## Business Context

Dunder Mifflin Paper Company is a B2B paper distributor serving businesses across the United Kingdom. The company charges a processing fee for each order based on the customer's monthly order value, calculated as approximately one week's worth of their average monthly orders (Monthly Order Value × 12 ÷ 52).

## Dataset

- **Orders**: 26,833 order records with processing fees, order values, and sales regions
- **Customers**: 41,513 customer records with business establishment years
- **Time Period**: Orders from May 2023 to May 2024
- **Geographic Coverage**: 50+ UK sales regions

## Analysis Questions

1. **Distribution Analysis**: Analyze and visualize the distribution of processing fees using descriptive statistics
2. **Central Tendency**: Determine the most appropriate measure of average processing fee for business reporting
3. **High-Value Orders**: Analyze orders with processing fees > £500 and identify patterns
4. **Geographic Insights**: Visualize sales data by region and extract commercial insights
5. **Sales Investigation**: Develop a structured approach to investigate sudden sales declines
6. **Statistical Testing**: Determine statistical significance of marketing test results

## Key Findings

- Processing fees follow a right-skewed distribution (skewness = 2.18)
- Median processing fee (£214.62) is more representative than mean (£249.93)
- 95.28% of orders have processing fees ≤ £500
- Manchester region dominates with 18% of sales volume
- Younger businesses (0-7 years) are associated with higher-value bulk orders
- 69.16% of processing fees match the expected formula calculation

## Technical Skills Demonstrated

- **Statistical Analysis**: Descriptive statistics, distribution analysis, skewness, coefficient of variation
- **Data Visualization**: Histograms, scatter plots, bar charts, geographic analysis
- **Hypothesis Testing**: Two-proportion z-tests, statistical significance testing
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scipy
- **Business Intelligence**: KPI reporting, geographic segmentation, customer demographics
- **Presentation**: Professional slide deck with data-driven insights

## Tools & Technologies

- Python 3.11
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- scipy for statistical testing
- Excel for data storage
- HTML/CSS/JavaScript for presentation slides

## Project Structure

```
dunder-mifflin-analysis/
├── data/
│   ├── Dunder_Mifflin_Orders.xlsx
│   └── Dunder_Mifflin_Data_Dictionary.md
├── code/
│   ├── 01_data_exploration.py
│   ├── 02_distribution_analysis.py
│   ├── 03_high_value_analysis.py
│   ├── 04_geographic_analysis.py
│   └── 05_statistical_testing.py
├── visualizations/
│   ├── processing_fee_distribution.png
│   ├── geographic_analysis.png
│   └── customer_demographics.png
├── presentation/
│   └── Dunder_Mifflin_Analysis.pptx
└── README.md
```

## How to Run

1. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy openpyxl
   ```

2. Run the analysis scripts in order:
   ```bash
   python 01_data_exploration.py
   python 02_distribution_analysis.py
   python 03_high_value_analysis.py
   python 04_geographic_analysis.py
   python 05_statistical_testing.py
   ```

3. View the presentation slides for visual insights

## Author

Buse Bircan
Data Analytics Portfolio Project

## Disclaimer

This is a fictional portfolio project created for demonstration purposes. Dunder Mifflin Paper Company is a fictional company from the TV show "The Office." All data has been generated for educational and portfolio purposes only.

## License

This project is available for portfolio and educational purposes.
