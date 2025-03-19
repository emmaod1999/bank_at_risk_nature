import os.path as path


BASE_PATH = path.abspath(path.join(__name__, '../..'))


'''
############################################
EXIOBASE paths
############################################
'''
EXIO_file_path = BASE_PATH + '/Data/exiobase_download_online/IOT_2022_ixi.zip'
EXIO_categories_path = BASE_PATH + '/Data/exiobase_download_online/EXIOBASE20i_7sectors.txt'
NACE_categories_path = BASE_PATH + '/Data/finance_exiobase_conversion/NACE2full_EXIOBASEp.xlsx'
EXIO_production_perc_file_path = BASE_PATH + '/Data/exiobase_download_online/production_percentage.csv'
