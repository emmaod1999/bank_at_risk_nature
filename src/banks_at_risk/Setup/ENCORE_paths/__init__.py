import os.path as path

BASE_PATH = path.abspath(path.join(__name__, '../..'))

'''
#################################################################
DEPENDENCY PATHS
#################################################################
'''

ENCORE_dep_path = BASE_PATH + '/Data/ENCORE_data/ENCORE_sector_dep_num.csv'
ENCORE_imp_path = BASE_PATH + '/Data/ENCORE_data/ENCORE_sector_imp_num.csv'
ENCORE_to_EXIO_path = BASE_PATH + '/Data/ENCORE_EXIOBASE_conversion/20201222_Benchmark-biodiv-systemic-risk-biodiversity_GICS-EXIOBASE-matching-table.xlsx'

ENCORE_imp_num_es_ass_path = BASE_PATH +'/Data/ENCORE_data/ENCORE_sector_imp_num_es_ass.csv'
ENCORE_imp_num_ass_driver_path = BASE_PATH +'/Data/ENCORE_data/ENCORE_sector_imp_num_ass_driver.csv'
# note this must be downloaded from repository and placed into the ENCORE_data file - not a part of the ENCORE data knowledge hub
ENCORE_imp_driver_driver_env_change_path = BASE_PATH + '/Data/ENCORE_data/asset_impact_driver_bind_driver_of_environmental_change_edited.csv'


