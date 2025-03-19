import pandas as pd
from banks_at_risk.Value_at_Risk.helper_value_at_risk_finance import finance_to_NACE_EXIO_region, calculate_finance_var
from banks_at_risk.Value_at_Risk.helper_value_at_risk_GSIB import GSIB_calculate_finance_var, finance_GSIB_reformat, GSIB_calculate_finance_var_system
def calc_nVAR():
    """
    This funciton generates the endogenous risk of the portfolios for UK banks and Pillar 3 data
    :return: NA
    """
    finance_df = finance_to_NACE_EXIO_region()
    calculate_finance_var(finance_df, 'mean')
    calculate_finance_var(finance_df, 'min')
    calculate_finance_var(finance_df, 'max')

    return None

def calc_nVAR_GSIB():
    """
    This function generates the endodgenous risk at the bank portfolio- and system-level for the three methodological
    treatments (mean, max, min) and saves them to the data repository.
    :return: NA
    """
    # get the formatted financial data - with the proportion of portfolio exposed to each sector-region for each bank
    finance_df = finance_GSIB_reformat()
    # calculate the endogenous risk of the banks with each of the three methodological treatments (mean, max, min)
    GSIB_calculate_finance_var(finance_df, 'mean')
    GSIB_calculate_finance_var(finance_df, 'min')
    GSIB_calculate_finance_var(finance_df, 'max')

    # get system values
    # aggregate the financial data to the system level by combining the individual contributions by each bank
    system_finance_df = finance_df.reset_index().drop(columns =['Bank', 'Total Loan', 'Proportion of Loans']).groupby(['region', 'sector']).sum()
    # name the bank "System"
    system_finance_df['Bank'] = ['System'] * system_finance_df.shape[0]
    system_finance_df = system_finance_df.reset_index().set_index(['Bank', 'region', 'sector'])
    # calculate the total portfolio value of the system
    system_finance_df['Total Loan'] = [system_finance_df['EUR m adjusted'].sum()] * system_finance_df.shape[0]
    # calculate the propotion of the system exposed to each sector-region pair
    system_finance_df['Proportion of Loans'] = system_finance_df['EUR m adjusted'] / system_finance_df['Total Loan']

    # calculate the endogenous risk of the system with each of the three methodological treatments (mean, max, min)
    GSIB_calculate_finance_var_system(system_finance_df, 'mean')
    GSIB_calculate_finance_var_system(system_finance_df, 'min')
    GSIB_calculate_finance_var_system(system_finance_df, 'max')

    return None

if __name__ == "__main__":
    # calc_nVAR()
    calc_nVAR_GSIB()
 
    
