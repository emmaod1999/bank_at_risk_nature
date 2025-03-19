import os.path as path

BASE_PATH = path.abspath(path.join(__name__, '../..'))

'''
#################################################################
VAR ANALYSIS PATHS
#################################################################
'''

GSIB_value_at_risk_sig_saving_path = BASE_PATH + '/Data/Value at Risk Figures/GSIB/Value at Risk Significance/'
value_at_risk_figure_saving_path = BASE_PATH +'/Data/Value at Risk Figures/GSIB/'

# sectoral overlap scores
scope1_overlap_mean_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope1_overlap_mean.csv'
scope1_overlap_min_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope1_overlap_min.csv'
scope1_overlap_max_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope1_overlap_max.csv'
scope3_overlap_mean_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope3_overlap_mean.csv'
scope3_overlap_min_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope3_overlap_min.csv'
scope3_overlap_max_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope3_overlap_max.csv'

# sectoral overlap rows scores
scope3_overlap_mean_rows_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope3_overlap_mean_rows.csv'
scope3_overlap_min_rows_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope3_overlap_min_rows.csv'
scope3_overlap_max_rows_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/scope3_overlap_max_rows.csv'

# NACE overlap scores
NACE_scope1_overlap_mean_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope1_overlap_mean.csv'
NACE_scope1_overlap_min_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope1_overlap_min.csv'
NACE_scope1_overlap_max_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope1_overlap_max.csv'
NACE_scope3_overlap_mean_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope3_overlap_mean.csv'
NACE_scope3_overlap_min_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope3_overlap_min.csv'
NACE_scope3_overlap_max_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope3_overlap_max.csv'
NACE_scope3_overlap_mean_rows_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope3_overlap_mean_rows.csv'
NACE_scope3_overlap_min_rows_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope3_overlap_min_rows.csv'
NACE_scope3_overlap_max_rows_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/NACE_scope3_overlap_max_rows.csv'

# sector overlap figure saving
sectoral_overlap_saving_path = BASE_PATH + '/Data/Value at Risk Figures/Sectoral Overlap/sectoral_overlap_heatmap'

# financial figures save
financial_bar_chart_path = BASE_PATH + '/Data/Value at Risk Figures/GSIB/Bloomberg Financial Data Bar Chart by Bank'
