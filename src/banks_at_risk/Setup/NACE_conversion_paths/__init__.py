import os.path as path

BASE_PATH = path.abspath(path.join(__name__, '../..'))

'''
#################################################################
CONVERSION PATHS
#################################################################
'''

NACE_to_EXIO_path = BASE_PATH + '/Data/exiobase_download_online/NACE2full_EXIOBASEp.xlsx'
finance_exio_region_path = BASE_PATH + '/Data/finance_exiobase_conversion/finance_exiobase_region.csv'
NACE_letters_path = BASE_PATH + '/Data/finance_exiobase_conversion/NACE_letter_sector.csv'


# saving path for the I_NACE and L_NACE
I_NACE_saving_path = BASE_PATH +'/Data/financial_data/I_NACE_df.csv'
L_NACE_saving_path = BASE_PATH +'/Data/financial_data/L_NACE_df.csv'
x_NACE_saving_path = BASE_PATH +'/Data/financial_data/x_NACE_df.csv'

# saving path for the scope 1 impact and depedency score NACE
scope1_impact_NACE_mean_path = BASE_PATH + '/Data/Impacts/Impact Scores/NACE/scope1_impact_NACE_score_mean.csv'
scope1_impact_NACE_min_path = BASE_PATH + '/Data/Impacts/Impact Scores/NACE/scope1_impact_NACE_score_min.csv'
scope1_impact_NACE_max_path = BASE_PATH + '/Data/Impacts/Impact Scores/NACE/scope1_impact_NACE_score_max.csv'

scope1_dependency_NACE_mean_path = BASE_PATH + '/Data/Dependencies/Dependency Scores/NACE/scope1_dependency_NACE_score_mean.csv'
scope1_dependency_NACE_min_path = BASE_PATH + '/Data/Dependencies/Dependency Scores/NACE/scope1_dependency_NACE_score_min.csv'
scope1_dependency_NACE_max_path = BASE_PATH + '/Data/Dependencies/Dependency Scores/NACE/scope1_dependency_NACE_score_max.csv'

