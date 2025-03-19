import pandas as pd
import numpy as np
from banks_at_risk.Value_at_Risk.helper_value_at_risk_GSIB import finance_GSIB_reformat, calc_L_min_I_full

def GSIB_var_calc_scope_1_sector(score, finance_data_df, type, folder):
    """
    :param score: a dataframe of the score that you want to use to calculate the value at risk
    :param finance_df: the finance of the bank you want to calculate the value at risk for
    :param folder: either 'Multipled/' for the combined multipled or '' for imp or dep
    :return: a list of dataframes that correspond to the finance_var and the rows for the finance_var plus the scores
    """
    if type != 'region_code' and type != 'code_only' and type != 'region_only':
        print('ERROR: Type must be "region_code" or "code_only" or "region_only"')
        return

    storing_list = []

    L_min_I = calc_L_min_I_full()

    banks = np.unique(finance_data_df.reset_index()['Bank'])
    services = score.columns
    # scope 1
    # for the storing the score-level scores
    imp_dep_compile_cols_scope_1_df = pd.DataFrame(index=L_min_I.index)
    # var values
    imp_dep_compile_cols_storing_var_finance_scope_1_df = pd.DataFrame(columns=services)

    # store score
    df = score.copy()
    score_name = score.name

    for bank in banks:
        # scope 1 value at risk with imp_dep scores
        # finance
        imp_dep_compile_cols_one_bank_var_finance_scope_1_df = imp_dep_compile_cols_scope_1_df.copy()
        imp_dep_compile_cols_one_bank_var_finance_scope_1_df['Bank'] = [f'{bank}'] * \
                                                                       imp_dep_compile_cols_one_bank_var_finance_scope_1_df.shape[
                                                                           0]

        # get finance values without converting the all to the bank values
        scope_1_score_one_bank = df.copy()

        # get financial data for Bank
        bank_data_df = finance_data_df.reset_index()[finance_data_df.reset_index()['Bank'] == bank].set_index(
            ['region', 'sector'])
        # bank_data_df = bank_data_df['Proportion of Loans']
        full_index = pd.DataFrame(index=L_min_I.index)
        bank_data_df = bank_data_df.merge(full_index, how='right', right_index=True, left_index=True)
        bank_data_df = bank_data_df.fillna(0.0).reset_index()
        # bank_data_df.drop(columns=['Bank', 'Total Loan', 'EUR m adjusted'], inplace=True)
        bank_data_dict = bank_data_df['Proportion of Loans'].to_dict()
        bank_data_absolute_dict = bank_data_df['EUR m adjusted'].to_dict()

        for service in services:
            # finance
            if (type == "region_code"):
                compiled_scope_1_df = bank_data_df.merge(
                    scope_1_score_one_bank[service].reset_index(), how='left', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
            if (type == "code_only"):
                compiled_scope_1_df = bank_data_df.merge(
                    scope_1_score_one_bank[service].reset_index(), how='left', left_on=['sector'],
                    right_on=['sector'])
            compiled_scope_1_df = compiled_scope_1_df.fillna(0.0).set_index(['region', 'sector'])
            imp_dep_compile_cols_one_bank_var_finance_scope_1_df[service] = compiled_scope_1_df['EUR m adjusted'] * \
                                                                            compiled_scope_1_df[service]

        # scope 1
        # finance
        imp_dep_compile_cols_storing_var_finance_scope_1_df = pd.concat(
            [imp_dep_compile_cols_storing_var_finance_scope_1_df.reset_index(),
             imp_dep_compile_cols_one_bank_var_finance_scope_1_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_dep_compile_cols_storing_var_finance_scope_1_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_var_finance_scope_1_df.drop(columns='index')
            imp_dep_compile_cols_storing_var_finance_scope_1_df = place_holder_df.copy()

    # load the services
    # # scope 1
    imp_dep_compile_cols_storing_var_finance_scope_1_df.to_csv(
        f'../Data/Value at Risk/Finance/GSIB/Both/Sectoral/{score_name} Finance VaR Scope 1.csv')
    storing_list.append(imp_dep_compile_cols_storing_var_finance_scope_1_df)

    return storing_list


