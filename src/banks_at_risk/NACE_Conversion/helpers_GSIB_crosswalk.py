from banks_at_risk.Setup.finance_paths import GSIB_financial_data_path, GSIB_compiled_sheet_path
import os
import pandas as pd

def bank_one_sheet():
    """
    This function reads all the G-SIB portfolio information (stored in a single subfolder in the data repository) and
    stores it in a single dataframe and saved as a csv in the data repository
    :return: NA
    """
    # get all the files in the G-SIB financial data subfolder in the data repository (the names of all the sheets
    # describing the individual portfolios of G-SIB by sector-region exposure
    bank_sheets = os.listdir(GSIB_financial_data_path)
    # create a dataframe to store the information of all the banks in a single dataframe
    banks_df = pd.DataFrame(data=None, columns=['Bank', 'region', 'sector', 'production'])
    # loop through all the individual sheets containing the bank portfolio information
    for sheet in bank_sheets:
        # get the bank name
        bank_name = os.path.splitext(sheet)[0]
        # read the individual G-SIB portfolio information from CSV
        df = pd.read_csv(f'{GSIB_financial_data_path}{sheet}', header=[0, 1], index_col=[0])
        # Transpose the dataframe
        df = df.T
        # fill the Bank column with the bank name
        df['Bank'] = bank_name
        # fill the other columns with the relevant portfolio information for the bank
        banks_df = pd.concat([banks_df, df.reset_index()])

    # set the index of the compiled dataframe with all bank data portfolio information
    banks_df = banks_df.set_index(['Bank', 'region', 'sector'])
    # save the compiled sheet to the data repository
    banks_df.to_csv(GSIB_compiled_sheet_path)
    return None