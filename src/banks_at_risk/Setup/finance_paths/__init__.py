import os.path as path

BASE_PATH = path.abspath(path.join(__name__, '../..'))


'''
#################################################################
FINANCE PATHS
#################################################################
'''

finance_data_path = BASE_PATH + '/Data/financial_data/finance_data_no_K.csv'
finance_data_anonymized_path = BASE_PATH + '/Data/financial_data/finance_data_no_K_anonymized.csv'

finance_exio_region_path = BASE_PATH + '/Data/finance_exiobase_conversion/finance_exiobase_region.csv'

GSIB_financial_data_path = BASE_PATH + '/Data/financial_data/Entities_alloc/'

GSIB_compiled_sheet_path = BASE_PATH + '/Data/financial_data/GSIB_financial_data.csv'

GSIB_bank_regions = BASE_PATH + '/Data/financial_data/bank_regions.csv'

GSIB_bank_names = BASE_PATH + '/Data/financial_data/GSIB_bank_names_short.csv'