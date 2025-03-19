import pandas as pd
import numpy as np
from pathlib import Path
from banks_at_risk.Setup.finance_paths import GSIB_compiled_sheet_path
from banks_at_risk.Setup.NACE_conversion_paths import NACE_letters_path, L_NACE_saving_path, I_NACE_saving_path
from banks_at_risk.Setup.dependency_paths import scope1_dependency_max_path, scope1_dependency_mean_path, scope1_dependency_min_path
from banks_at_risk.Setup.impact_paths import scope1_impact_max_path, scope1_impact_mean_path, scope1_impact_min_path
from banks_at_risk.Utils.exio_ops import read_exio
from banks_at_risk.Setup.EXIO_paths import EXIO_file_path


def finance_GSIB_reformat():
    """
    This function formats the financial data for endogenous risk analysis by adding the proportion of the portfolio
    exposed to the sector-region pair
    :return: the compiled data for all G-SIBs formatting for endogenous risk analysis, including the portfolio proportion
    exposed to each sector-region
    """
    # get the sheet with all the G-SIB data
    finance_df = pd.read_csv(GSIB_compiled_sheet_path, index_col=[0, 1, 2], header=[0])
    finance_df = finance_df.rename(columns={'production':'EUR m adjusted'})
    # calculate the total portfolio value for each bank for each sector and region
    finance_data_NACE_region_grouped_df = finance_df.groupby(['Bank', 'sector', 'region']).sum()

    # calculate the total amount of loan in EUR for each bank
    finance_loan_total_by_bank_df = finance_data_NACE_region_grouped_df.groupby('Bank').sum()
    finance_loan_total_by_bank_df.rename(columns={"EUR m adjusted":"Total Loan"}, inplace=True)

    # add total loan values to the finance_data
    finance_data_NACE_region_grouped_w_total_df = pd.merge(finance_data_NACE_region_grouped_df,
                                                        finance_loan_total_by_bank_df,
                                                        left_index=True, right_index=True)
    # get proportion of loans for each bank and sector-region pair
    finance_data_NACE_region_grouped_w_total_df['Proportion of Loans'] = (
            finance_data_NACE_region_grouped_w_total_df['EUR m adjusted'] /
            finance_data_NACE_region_grouped_w_total_df['Total Loan'])

    return finance_data_NACE_region_grouped_w_total_df


def calc_L_min_I_full():
    """
    This function generates the Leontief matrix minus itself in the upstream supply chain needed for upstream supply
    chain analysis.
    :return: The Leontief matrix minus the identity matrix - the upstream supply chain without the sector-region pair
    included in its upstream supply chain
    """
    # read EXIOBASE MRIO wtih all required tables calculated
    EXIO3 = read_exio(EXIO_file_path)

    # get Leontief
    L_df = EXIO3.L

    # get Identity matrix
    I_full = np.eye(L_df.shape[0])
    I_df = L_df.copy()
    I_df.loc[:, :] = I_full

    # subtract the Leontief by the Identity matrix
    L_min_I = L_df - I_df

    return L_min_I

def GSIB_var_calc_scope_1_sector(score, finance_data_df, type, folder):
    """
    This function calculates the direct operations endogenous risk at the sector-level (meaning impact score is not
    aggregated to the regional level). The function saves the results into the appropriate folder in the data repository
    :param score: dataframe of the score that you want to use to calculate the endogenous risk (combined df)
    :param finance_df: the finance of the banks you want to calculate the endogenous risk for
    :param folder: specific subfolder for saving the results in data repository
    :param type: describes whether the score has sector and regions or just sectors or just regions
    :return: list of dataframes that correspond to the finance_var and the rows for the finance_var plus the scores
    """
    # check if the type corresponds to an accepted type
    if type != 'region_code' and type != 'code_only' and type != 'region_only':
        print('ERROR: Type must be "region_code" or "code_only" or "region_only"')
        return

    # create a list to store the results
    storing_list = []

    # generate the Leontief matrix without the sector itself in its own upstream supply chain for calculating upstream
    # endogenous risk
    L_min_I = calc_L_min_I_full()

    # get the list of banks
    banks = np.unique(finance_data_df.reset_index()['Bank'])
    # get list of ecosystem services
    services = score.columns

    # scope 1
    # for the storing the score-level scores
    imp_dep_compile_cols_scope_1_df = pd.DataFrame(index=L_min_I.index)
    # for storing the endogenous risk values
    imp_dep_compile_cols_storing_var_finance_scope_1_df = pd.DataFrame(columns=services)

    # store score and score name
    df = score.copy()
    score_name = score.name

    # loop through banks in the financial data
    for bank in banks:
        # scope 1 value at risk with imp_dep scores
        # finance
        # fill the Bank column with the corresponding bank name
        imp_dep_compile_cols_one_bank_var_finance_scope_1_df = imp_dep_compile_cols_scope_1_df.copy()
        imp_dep_compile_cols_one_bank_var_finance_scope_1_df['Bank'] = [f'{bank}'] * \
                                                                       imp_dep_compile_cols_one_bank_var_finance_scope_1_df.shape[
                                                                           0]

        # get finance values without converting the all to the bank values
        scope_1_score_one_bank = df.copy()

        # get financial data for Bank
        bank_data_df = finance_data_df.reset_index()[finance_data_df.reset_index()['Bank'] == bank].set_index(
            ['region', 'sector'])
        full_index = pd.DataFrame(index=L_min_I.index)
        bank_data_df = bank_data_df.merge(full_index, how='right', right_index=True, left_index=True)
        bank_data_df = bank_data_df.fillna(0.0).reset_index()
        bank_data_dict = bank_data_df['Proportion of Loans'].to_dict()
        bank_data_absolute_dict = bank_data_df['EUR m adjusted'].to_dict()

        # loop through the ecosystem services
        for service in services:
            # finance
            # check if the score has sector and regions
            if (type == "region_code"):
                # combine the bank portfolio data with the score data
                compiled_scope_1_df = bank_data_df.merge(
                    scope_1_score_one_bank[service].reset_index(), how='left', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
            # check if the score has only sectors
            if (type == "code_only"):
                # combine the bank portfolio data with the score data
                compiled_scope_1_df = bank_data_df.merge(
                    scope_1_score_one_bank[service].reset_index(), how='left', left_on=['sector'],
                    right_on=['sector'])
            # replace any NA values with zero values
            compiled_scope_1_df = compiled_scope_1_df.fillna(0.0).set_index(['region', 'sector'])
            # multiply the score values for each sector region by the portfolio exposure to that sector-region to get
            # the endogenous risk
            imp_dep_compile_cols_one_bank_var_finance_scope_1_df[service] = compiled_scope_1_df['EUR m adjusted'] * \
                                                                            compiled_scope_1_df[service]

        # scope 1
        # finance
        # add the values for this bank to the storing sheet for all the banks
        imp_dep_compile_cols_storing_var_finance_scope_1_df = pd.concat(
            [imp_dep_compile_cols_storing_var_finance_scope_1_df.reset_index(),
             imp_dep_compile_cols_one_bank_var_finance_scope_1_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        # format the dataframe in case of additional columns being added through the loop
        if 'index' in imp_dep_compile_cols_storing_var_finance_scope_1_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_var_finance_scope_1_df.drop(columns='index')
            imp_dep_compile_cols_storing_var_finance_scope_1_df = place_holder_df.copy()

    # save the endogenous risk to the appropriate path in the data repository
    # # scope 1
    imp_dep_compile_cols_storing_var_finance_scope_1_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}/{score_name} Finance VaR Scope 1.csv')
    storing_list.append(imp_dep_compile_cols_storing_var_finance_scope_1_df)

    return storing_list

def GSIB_var_calc_scope_1(impact_score_df, dependency_score_df, finance_data_df, type, folder):
    """
    This function calculates the direct operations endogenous risk at the portfolio-level (meaning impact score is
    aggregated to the regional level). The function saves the results into the appropriate folder in the data repository
    :param impact_score_df: dataframe of the impact score that you want to use to calculate the value at risk
    :param dependency_score_df: dataframe of the dependency score that you want to use to calculate the value at risk
    :param finance_df: the finance of the bank you want to calculate the value at risk for
    :param folder: subfolder for storing the risk exposure in the data repository
    :param type: describes whether the score has sector and regions or just sectors or just regions
    :return: a list of dataframes that correspond to the finance_var and the rows for the finance_var plus the scores
    """
    # check whether type is an accepted type
    if type != 'region_code' and type != 'code_only' and type != 'region_only':
        print('ERROR: Type must be "region_code" or "code_only" or "region_only"')
        return

    # create a list to store the endogenous risk exposure
    storing_list = []

    # generate the Leontief matrix without the sector itself in its own upstream supply chain for calculating upstream
    # endogenous risk
    L_min_I = calc_L_min_I_full()

    # get list of banks
    banks = np.unique(finance_data_df.reset_index()['Bank'])
    # get list of ecosystem services
    services = impact_score_df.columns

    # scope 1
    # for the storing the score-level scores
    imp_dep_compile_cols_scope_1_df = pd.DataFrame(index=L_min_I.index)
    # for storing the endogenous risk values
    imp_dep_compile_cols_storing_var_finance_scope_1_df = pd.DataFrame(columns=services)

    # store scores
    # impact score
    impact_df = impact_score_df.copy()
    impact_score_name = impact_score_df.name
    # dependency
    dependency_df = dependency_score_df.copy()
    dependency_score_name = dependency_score_df.name

    # loop through the banks
    for bank in banks:
        # scope 1 value at risk with imp_dep scores
        # finance
        # fill the Bank column with the corresponding bank name
        imp_dep_compile_cols_one_bank_var_finance_scope_1_df = imp_dep_compile_cols_scope_1_df.copy()
        imp_dep_compile_cols_one_bank_var_finance_scope_1_df['Bank'] = [f'{bank}'] * \
                                                                       imp_dep_compile_cols_one_bank_var_finance_scope_1_df.shape[
                                                                           0]

        # get finance values without converting the all to the bank values
        scope_1_dep_score_one_bank = dependency_df.copy()

        # get financial data for Bank
        bank_data_df = finance_data_df.reset_index()[finance_data_df.reset_index()['Bank'] == bank].set_index(
            ['region', 'sector'])
        full_index = pd.DataFrame(index=L_min_I.index)
        bank_data_df = bank_data_df.merge(full_index, how='right', right_index=True, left_index=True)
        bank_data_df = bank_data_df.fillna(0.0).reset_index()
        bank_data_dict = bank_data_df['Proportion of Loans'].to_dict()
        bank_data_absolute_dict = bank_data_df['EUR m adjusted'].to_dict()

        # combine the impact score with the financial data for the bank
        scope_1_imp_score_one_bank = pd.merge(impact_df.reset_index(), bank_data_df, right_on=['sector', 'region'], left_on=['sector', 'region'])
        scope_1_imp_score_one_bank = scope_1_imp_score_one_bank.drop(columns=['Total Loan', 'Proportion of Loans'])
        scope_1_imp_score_one_bank = scope_1_imp_score_one_bank.set_index(['region', 'sector'])

        # loop through the ecosystem services
        for service in services:
            # create a dataframe to save the impact score for one bank for one ecosystem service
            scope_1_imp_score_one_bank_one_service = pd.DataFrame(index=impact_df.index)
            # multiply the impact score for the sector-region by the portfolio exposure value for that sector-region
            scope_1_imp_score_one_bank_one_service[f'{service} region'] = scope_1_imp_score_one_bank[service] * scope_1_imp_score_one_bank['EUR m adjusted']
            # aggregate the impact and portfolio exposure to the regional level
            scope_1_imp_score_one_bank_one_service = scope_1_imp_score_one_bank_one_service.reset_index().drop(columns=['sector']).groupby(['region']).sum()
            # add the sector-region dependency score to the regional impact and portfolio exposure
            scope_1_imp_dep_score_one_bank_one_service = pd.merge(scope_1_imp_score_one_bank_one_service.reset_index(), scope_1_dep_score_one_bank[service].reset_index(), right_on=['region'], left_on=['region'])
            scope_1_imp_dep_score_one_bank_one_service = scope_1_imp_dep_score_one_bank_one_service.set_index(['region', 'sector'])
            # multiply the sector-region dependency score by the regional imapct and portfolio exposure
            scope_1_imp_dep_score_one_bank_one_service[f'{service} combined'] = scope_1_imp_dep_score_one_bank_one_service[service] * scope_1_imp_dep_score_one_bank_one_service[f'{service} region']
            scope_1_imp_dep_score_one_bank_one_service = scope_1_imp_dep_score_one_bank_one_service.drop(columns=[service, f'{service} region'])
            scope_1_imp_dep_score_one_bank_one_service = scope_1_imp_dep_score_one_bank_one_service.rename(columns={f'{service} combined':service})

            # calculate the proportion of sectors for each region
            region_totals = bank_data_df.drop(columns=['Total Loan']).groupby(['region']).sum()
            bank_data_w_region_df = pd.merge(bank_data_df.drop(columns=['Total Loan', 'sector', 'Bank', 'Proportion of Loans']).groupby(['region']).sum(), bank_data_df.drop(columns=['Total Loan', 'Proportion of Loans']), right_on=['region'], left_on=['region'])
            bank_data_w_region_df['region proportion'] = np.where(bank_data_w_region_df['EUR m adjusted_x'] == 0, 0, bank_data_w_region_df['EUR m adjusted_y'] / bank_data_w_region_df['EUR m adjusted_x'])

            # combine the bank data with sectoral proportion by region with the combined score
            # finance
            # if the score contains sector and region
            if (type == "region_code"):
                compiled_scope_1_df = bank_data_w_region_df.merge(
                    scope_1_imp_dep_score_one_bank_one_service[service].reset_index(), how='left', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
            # if the score contains only sector
            if (type == "code_only"):
                compiled_scope_1_df = bank_data_w_region_df.merge(
                    scope_1_imp_dep_score_one_bank_one_service[service].reset_index(), how='left', left_on=['sector'],
                    right_on=['sector'])
            # fill any NAs with 0s
            compiled_scope_1_df = compiled_scope_1_df.fillna(0.0).set_index(['region', 'sector'])
            # multiply the regional proportion by the combined score
            imp_dep_compile_cols_one_bank_var_finance_scope_1_df[service] = compiled_scope_1_df['region proportion'] * \
                                                                            compiled_scope_1_df[service]

        # scope 1
        # finance
        # add the values for this bank to the storing sheet for all the banks
        imp_dep_compile_cols_storing_var_finance_scope_1_df = pd.concat(
            [imp_dep_compile_cols_storing_var_finance_scope_1_df.reset_index(),
             imp_dep_compile_cols_one_bank_var_finance_scope_1_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_dep_compile_cols_storing_var_finance_scope_1_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_var_finance_scope_1_df.drop(columns='index')
            imp_dep_compile_cols_storing_var_finance_scope_1_df = place_holder_df.copy()

    # save the endogenous risk to the appropriate path in the data repository
    # load the services
    # # scope 1
    imp_dep_compile_cols_storing_var_finance_scope_1_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}/{impact_score_name} {dependency_score_name} Finance VaR Scope 1.csv')
    storing_list.append(imp_dep_compile_cols_storing_var_finance_scope_1_df)

    return storing_list



def finance_var_calc_scope_3_combined(imp_score_df, dep_score_df, finance_data_NACE_region_grouped_w_total_df, type, folder):
    """
    This function calculates the upstream supply chain endogenous risk at the bank portfolio-level and stores the
    results in the data repository. It also calculates the impact and depedency risk of the banks separately as well.
    :param imp_score_df: dataframe of the impact score that you want to use to calculate the value at risk
    :param dep_score_df: dataframe of the dependency score that you want to use to calculate the value at risk
    :param finance_data_NACE_region_grouped_w_total_df: the finance of the bank you want to calculate with (formatted)
    :param folder: subfolder for storing the risk exposure in the data repository
    :param type: describes whether the score has sector and regions or just sectors or just regions
    :return: list of dataframes that correspond to the finance_var and the rows for the finance_var plus the scores
    """
    # check if type is an accepted type
    if type != 'region_code' and type != 'code_only' and type != 'region_only':
        print('ERROR: Type must be "region_code" or "code_only" or "region_only"')
        return

    # create a list to store the results
    storing_list = []

    # calculate the relative weighting of sector-region pairs in upstream supply chains
    ### calculate overlined((L -1)), relative impact dependency matrix
    # generate the Leontief matrix without the sector itself in its own upstream supply chain
    L_min_I = calc_L_min_I_full()
    L_min_I_numpy = L_min_I.to_numpy(dtype=float)
    # get the column sums of the L_matrix without itself included in supply chain
    col_sums = np.sum(L_min_I, axis=0)
    col_sums = col_sums.to_numpy(dtype=float)
    # divide each element by its column sum to get the relative importance in the upstream supply chain
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_imp_array = np.where(col_sums == 0, 0, np.divide(L_min_I_numpy, col_sums[np.newaxis, :]))

    # get the weights for the contribution of each sector,region pair to the supply chain - formatting for next steps
    L_weights = pd.DataFrame(rel_imp_array, index=L_min_I.index, columns=L_min_I.columns)
    upstream_calc = L_weights.copy().reset_index()
    upstream_calc_format = upstream_calc.T.reset_index().T.rename(columns={0: 'region', 1: 'sector'})
    upstream_calc_format.loc['sector', 'region'] = 'region'
    upstream_calc_format.loc['sector', 'sector'] = 'sector'

    # list of banks
    banks = np.unique(finance_data_NACE_region_grouped_w_total_df.reset_index()['Bank'])
    # list of ecosystem services
    services = imp_score_df.columns

    # multiply the weighted sum for Leontief by the impact region score (based on investment location)
    # plus the dependency score for the sector region L_weighted * (impact region) * (dependency sector)

    # for the storing the score-level scores
    # both
    imp_dep_compile_cols_df = pd.DataFrame(index=L_min_I.index)
    imp_dep_compile_rows_df = pd.DataFrame(index=np.unique(L_min_I.reset_index()['region']))
    # imp
    imp_compile_cols_df = pd.DataFrame(index=L_min_I.index)
    imp_compile_rows_df = pd.DataFrame(index=L_min_I.index)
    # dep
    dep_compile_cols_df = pd.DataFrame(index=L_min_I.index)
    dep_compile_rows_df = pd.DataFrame(index=L_min_I.index)

    # storing endogenous risk throughout calculating
    # cols is for the source sector (the sector-region's score for its upstream supply chain)
    # rows is for the sector contributing in the value chain (the sector-region scores within the value chains of others
    # both - endogenous risk
    imp_dep_compile_cols_var_finance_df = pd.DataFrame(index=L_min_I.index)
    imp_dep_compile_rows_var_finance_df = pd.DataFrame(index=np.unique(L_min_I.reset_index()['region']))
    # impact only
    imp_compile_cols_var_finance_df = pd.DataFrame(index=L_min_I.index)
    imp_compile_rows_var_finance_df = pd.DataFrame(index=L_min_I.index)
    # dependency only
    dep_compile_cols_var_finance_df = pd.DataFrame(index=L_min_I.index)
    dep_compile_rows_var_finance_df = pd.DataFrame(index=L_min_I.index)

    # for storing the column sums for the score-level scores throughout calculating
    # overall endogenous risk for the ecosystem service from source sector's upstream supply chain
    # both - endogenous risk
    imp_dep_compile_cols_storing_df = pd.DataFrame(columns=services)
    # impact only
    imp_compile_cols_storing_df = pd.DataFrame(columns=services)
    # dependency only
    dep_compile_cols_storing_df = pd.DataFrame(columns=services)


    # for storing the row sums for the score-level scores throughout calculating
    # overall endogenous risk contributed by sector-region in the supply chain of others
    # both - endogenous risk
    imp_dep_compile_rows_storing_df = pd.DataFrame(columns=services)
    # impact only
    imp_compile_rows_storing_df = pd.DataFrame(columns=services)
    # dependency only
    dep_compile_rows_storing_df = pd.DataFrame(columns=services)

    # for storing the endogenous risk values - final
    # both - endogenous risk
    # value at risk finance - source sector
    imp_dep_compile_cols_storing_var_finance_df = pd.DataFrame(columns=services)
    # value at risk finance rows - value chain sector
    imp_dep_compile_rows_storing_var_finance_df = pd.DataFrame(columns=services)
    # impact only
    # value at risk finance - source sector
    imp_compile_cols_storing_var_finance_df = pd.DataFrame(columns=services)
    # value at risk finance rows - value chain sector
    imp_compile_rows_storing_var_finance_df = pd.DataFrame(columns=services)
    # dependency
    # value at risk finance - source sector
    dep_compile_cols_storing_var_finance_df = pd.DataFrame(columns=services)
    # value at risk finance rows - value chain sector
    dep_compile_rows_storing_var_finance_df = pd.DataFrame(columns=services)

    # store scores
    # impact scores
    imp_df = imp_score_df.copy()
    imp_score_name = imp_score_df.name
    # dependency
    dep_df = dep_score_df.copy()
    dep_score_name = dep_score_df.name

    # loop through banks
    for bank in banks:
        # fill the bank column with the name of the bank
        # source sectors
        # both - endogenous risk
        imp_dep_compile_cols_one_bank_df = imp_dep_compile_cols_df.copy()
        imp_dep_compile_cols_one_bank_df['Bank'] = [f'{bank}'] * imp_dep_compile_cols_df.shape[0]
        # impact only
        imp_compile_cols_one_bank_df = imp_compile_cols_df.copy()
        imp_compile_cols_one_bank_df['Bank'] = [f'{bank}'] * imp_compile_cols_df.shape[0]
        # dependency only
        dep_compile_cols_one_bank_df = dep_compile_cols_df.copy()
        dep_compile_cols_one_bank_df['Bank'] = [f'{bank}'] * dep_compile_cols_df.shape[0]

        # rows - value chain sectors
        # both - endogenous risk
        imp_dep_compile_rows_one_bank_df = imp_dep_compile_rows_df.copy()
        imp_dep_compile_rows_one_bank_df['Bank'] = [f'{bank}'] * imp_dep_compile_rows_df.shape[0]
        # impact only
        imp_compile_rows_one_bank_df = imp_compile_rows_df.copy()
        imp_compile_rows_one_bank_df['Bank'] = [f'{bank}'] * imp_compile_rows_df.shape[0]
        # dependency only
        dep_compile_rows_one_bank_df = dep_compile_rows_df.copy()
        dep_compile_rows_one_bank_df['Bank'] = [f'{bank}'] * dep_compile_rows_df.shape[0]

        # value at risk finance - source sectors
        # both - endogenous risk
        imp_dep_compile_cols_one_bank_var_finance_df = imp_dep_compile_cols_var_finance_df.copy()
        imp_dep_compile_cols_one_bank_var_finance_df['Bank'] = [f'{bank}'] * imp_dep_compile_cols_var_finance_df.shape[
            0]
        # dependency only
        dep_compile_cols_one_bank_var_finance_df = dep_compile_cols_var_finance_df.copy()
        dep_compile_cols_one_bank_var_finance_df['Bank'] = [f'{bank}'] * dep_compile_cols_var_finance_df.shape[
            0]
        # impact only
        imp_compile_cols_one_bank_var_finance_df = imp_compile_cols_var_finance_df.copy()
        imp_compile_cols_one_bank_var_finance_df['Bank'] = [f'{bank}'] * imp_compile_cols_var_finance_df.shape[
            0]

        # value at risk finance rows - value chian sectors
        # value at risk finance
        # both - enodgenous risk
        imp_dep_compile_rows_one_bank_var_finance_df = imp_dep_compile_rows_var_finance_df.copy()
        imp_dep_compile_rows_one_bank_var_finance_df['Bank'] = [f'{bank}'] * imp_dep_compile_rows_var_finance_df.shape[
            0]
        # impact only
        imp_compile_rows_one_bank_var_finance_df = imp_compile_rows_var_finance_df.copy()
        imp_compile_rows_one_bank_var_finance_df['Bank'] = [f'{bank}'] * imp_compile_rows_var_finance_df.shape[
            0]
        # dependency only
        dep_compile_rows_one_bank_var_finance_df = dep_compile_rows_var_finance_df.copy()
        dep_compile_rows_one_bank_var_finance_df['Bank'] = [f'{bank}'] * dep_compile_rows_var_finance_df.shape[
            0]

        # get financial data for Bank
        bank_data_df = finance_data_NACE_region_grouped_w_total_df.reset_index()[finance_data_NACE_region_grouped_w_total_df.reset_index()['Bank'] == bank].set_index(
            ['region', 'sector'])
        full_index = pd.DataFrame(index=L_min_I.index)
        bank_data_df = bank_data_df.merge(full_index, how='right', right_index=True, left_index=True)
        bank_data_df = bank_data_df.fillna(0.0)
        bank_data_dict = bank_data_df['Proportion of Loans'].to_dict()
        bank_data_absolute_dict = bank_data_df['EUR m adjusted'].to_dict()

        # loop through ecosystem services
        for service in services:
            # impact only - upstream supply chain score calculation
            # if score has sector and region
            if type == 'region_code':
                # get the scores and the weighted L in one DF
                compiled_imp_df = upstream_calc_format.merge(
                    imp_df[service].reset_index(), how='outer', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
                compiled_imp_df = compiled_imp_df.fillna(0.0)
            # if score has sector only
            if type == 'code_only':
                # get the scores and the weighted L in one DF
                compiled_imp_df = upstream_calc_format.merge(
                    imp_df[service].reset_index(), how='outer', left_on=['sector'],
                    right_on=['sector'])
                compiled_imp_df = compiled_imp_df.fillna(0.0)
            # if score has region only
            if type == 'region_only':
                # get the scores and the weighted L in one DF
                compiled_imp_df = upstream_calc_format.merge(
                    imp_df[service].reset_index(), how='outer', left_on=['region'],
                    right_on=['region'])
                compiled_imp_df = compiled_imp_df.fillna(0.0)

            # multiply the weighted average by the score
            compiled_imp_df = compiled_imp_df.set_index(['region', 'sector'])
            compiled_imp_df = compiled_imp_df.T.set_index(('region', 'sector')).T
            service_imp_df = compiled_imp_df[(0.0,0.0)]
            service_imp_df = service_imp_df[0:(L_min_I.shape[0])]
            service_imp_df = service_imp_df.astype(float)
            calc_imp_df = compiled_imp_df.drop(columns =(0.0,0.0))
            calc_imp_df = calc_imp_df.astype(float)
            multiplied_imp_df = np.multiply(calc_imp_df.to_numpy(), service_imp_df.to_numpy()[:, np.newaxis])
            imp_dep_compile_service_imp_df = pd.DataFrame(multiplied_imp_df, index=calc_imp_df.index, columns=calc_imp_df.columns)

            # dependency only - upstream supply chain score calculation
            # if score has sector and region
            if type == 'region_code':
                # get the scores and the weighted L in one DF
                compiled_dep_df = upstream_calc_format.merge(
                    dep_df[service].reset_index(), how='outer', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
                compiled_dep_df = compiled_dep_df.fillna(0.0)
            # if score has sector only
            if type == 'code_only':
                # get the scores and the weighted L in one DF
                compiled_dep_df = upstream_calc_format.merge(
                    dep_df[service].reset_index(), how='outer', left_on=['sector'],
                    right_on=['sector'])
                compiled_dep_df = compiled_dep_df.fillna(0.0)
            # if score has region only
            if type == 'region_only':
                # get the scores and the weighted L in one DF
                compiled_dep_df = upstream_calc_format.merge(
                    dep_df[service].reset_index(), how='outer', left_on=['region'],
                    right_on=['region'])
                compiled_dep_df = compiled_dep_df.fillna(0.0)

            # multiply the weighted average by the score
            compiled_dep_df = compiled_dep_df.set_index(['region', 'sector'])
            compiled_dep_df = compiled_dep_df.T.set_index(('region', 'sector')).T
            service_dep_df = compiled_dep_df[(0.0, 0.0)]
            service_dep_df = service_dep_df[0:(L_min_I.shape[0])]
            service_dep_df = service_dep_df.astype(float)
            calc_dep_df = compiled_dep_df.drop(columns=(0.0, 0.0))
            calc_dep_df = calc_dep_df.astype(float)
            multiplied_dep_df = np.multiply(calc_dep_df.to_numpy(), service_dep_df.to_numpy()[:, np.newaxis])
            imp_dep_compile_service_dep_df = pd.DataFrame(multiplied_dep_df, index=calc_dep_df.index,
                                                          columns=calc_dep_df.columns)


            # combine the impact and dependency scores
            # sum the rows by the region
            # both - endogenous risk
            imp_dep_compile_service_dep_region_df = imp_dep_compile_service_dep_df.reset_index().drop(columns='sector').groupby('region').sum()
            imp_dep_compile_service_imp_region_df = imp_dep_compile_service_imp_df.reset_index().drop(columns='sector').groupby('region').sum()
            imp_dep_compile_service_df = imp_dep_compile_service_dep_region_df * imp_dep_compile_service_imp_region_df

            # combine the impact and dependency score value at risks
            # multiply score by the absolute value of finance to sr to get endogenous risk
            imp_dep_compile_service_finance_VaR_df = imp_dep_compile_service_df
            imp_dep_compile_service_finance_VaR_df = imp_dep_compile_service_finance_VaR_df.mul(
                bank_data_absolute_dict, axis='columns')

            # multiply dependency/impact score separately by absolute value of finance to sr to get VaR
            # dependency only
            dep_compile_service_finance_VaR_df = imp_dep_compile_service_dep_df
            dep_compile_service_finance_VaR_df = imp_dep_compile_service_dep_df.mul(
                bank_data_absolute_dict, axis='columns')
            # impact only
            imp_compile_service_finance_VaR_df = imp_dep_compile_service_imp_df
            imp_compile_service_finance_VaR_df = imp_dep_compile_service_imp_df.mul(
                bank_data_absolute_dict, axis='columns')

            # both - endogenous risk
            # get the column sums for one bank for the scores
            imp_dep_compile_cols_one_bank_df[service] = imp_dep_compile_service_df.sum()
            # get the row sums for one bank and service into the greater the df
            imp_dep_compile_rows_one_bank_df[f'{service}'] = imp_dep_compile_service_df.T.sum()
            # dependency only
            # get the column sums for one bank for the scores
            dep_compile_cols_one_bank_df[service] = imp_dep_compile_service_dep_df.sum()
            # get the row sums for one bank and service into the greater the df
            dep_compile_rows_one_bank_df[f'{service}'] = imp_dep_compile_service_dep_df.T.sum()
            # impact only
            # get the column sums for one bank for the scores
            imp_compile_cols_one_bank_df[service] = imp_dep_compile_service_imp_df.sum()
            # get the row sums for one bank and service into the greater the df
            imp_compile_rows_one_bank_df[f'{service}'] = imp_dep_compile_service_imp_df.T.sum()


            # both - endogenous risk
            # get column sums for one bank for the value at risk finance
            imp_dep_compile_cols_one_bank_var_finance_df[service] = imp_dep_compile_service_finance_VaR_df.sum()
            # get the row sums for one bank and service into the greater the df
            imp_dep_compile_rows_one_bank_var_finance_df[f'{service}'] = imp_dep_compile_service_finance_VaR_df.T.sum()
            # impact only
            # get column sums for one bank for the value at risk finance
            imp_compile_cols_one_bank_var_finance_df[service] = imp_compile_service_finance_VaR_df.sum()
            # get the row sums for one bank and service into the greater the df
            imp_compile_rows_one_bank_var_finance_df[f'{service}'] = imp_compile_service_finance_VaR_df.T.sum()
            # dependency only
            # get column sums for one bank for the value at risk finance
            dep_compile_cols_one_bank_var_finance_df[service] = dep_compile_service_finance_VaR_df.sum()
            # get the row sums for one bank and service into the greater the df
            dep_compile_rows_one_bank_var_finance_df[f'{service}'] = dep_compile_service_finance_VaR_df.T.sum()

        # both - endogenous risk
        # load the services - into storage dataframe
        # cols - source sector
        imp_dep_compile_cols_storing_df = pd.concat(
            [imp_dep_compile_cols_storing_df.reset_index(), imp_dep_compile_cols_one_bank_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_dep_compile_cols_storing_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_df.drop(columns='index')
            imp_dep_compile_cols_storing_df = place_holder_df.copy()
        # rows - value chain sector
        # load the services - into storage dataframe
        imp_dep_compile_rows_storing_df = pd.concat(
            [imp_dep_compile_rows_storing_df.reset_index(), imp_dep_compile_rows_one_bank_df.reset_index().rename(columns={'index':'region'})]).set_index(
            ['Bank', 'region'])
        if 'index' in imp_dep_compile_rows_storing_df.columns:
            place_holder_df = imp_dep_compile_rows_storing_df.drop(columns='index')
            imp_dep_compile_rows_storing_df = place_holder_df.copy()
        # var finance - source sectors
        # load the services - into storage data frame
        imp_dep_compile_cols_storing_var_finance_df = pd.concat(
            [imp_dep_compile_cols_storing_var_finance_df.reset_index(),
             imp_dep_compile_cols_one_bank_var_finance_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_dep_compile_cols_storing_var_finance_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_var_finance_df.drop(columns='index')
            imp_dep_compile_cols_storing_var_finance_df = place_holder_df.copy()
        # rows - value chain sectors
        # var finance
        # load the services - into storage data frame
        imp_dep_compile_rows_storing_var_finance_df = pd.concat(
            [imp_dep_compile_rows_storing_var_finance_df.reset_index(),
             imp_dep_compile_rows_one_bank_var_finance_df.reset_index().rename(columns={'index':'region'})]).set_index(
            ['Bank', 'region'])
        if 'index' in imp_dep_compile_rows_storing_var_finance_df.columns:
            place_holder_df = imp_dep_compile_rows_storing_var_finance_df.drop(columns='index')
            imp_dep_compile_rows_storing_var_finance_df = place_holder_df.copy()

        # impact only
        # load the services - into storage dataframe
        # cols - source sector
        imp_compile_cols_storing_df = pd.concat(
            [imp_compile_cols_storing_df.reset_index(),
             imp_compile_cols_one_bank_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_compile_cols_storing_df.columns:
            place_holder_df = imp_compile_cols_storing_df.drop(columns='index')
            imp_compile_cols_storing_df = place_holder_df.copy()
        # rows - value chain sector
        # load the services - into storage dataframe
        imp_compile_rows_storing_df = pd.concat(
            [imp_compile_rows_storing_df.reset_index(),
             imp_compile_rows_one_bank_df.reset_index().rename(columns={'index': 'region'})]).set_index(
            ['Bank','region', 'sector'])
        if 'index' in imp_compile_rows_storing_df.columns:
            place_holder_df = imp_compile_rows_storing_df.drop(columns='index')
            imp_compile_rows_storing_df = place_holder_df.copy()
        # var finance - source sector
        # load the services - into storage dataframe
        imp_compile_cols_storing_var_finance_df = pd.concat(
            [imp_compile_cols_storing_var_finance_df.reset_index(),
             imp_compile_cols_one_bank_var_finance_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_compile_cols_storing_var_finance_df.columns:
            place_holder_df = imp_compile_cols_storing_var_finance_df.drop(columns='index')
            imp_compile_cols_storing_var_finance_df = place_holder_df.copy()
        # rows - value chain sector
        # var finance
        # load the services - into storage dataframe
        imp_compile_rows_storing_var_finance_df = pd.concat(
            [imp_compile_rows_storing_var_finance_df.reset_index(),
             imp_compile_rows_one_bank_var_finance_df.reset_index().rename(
                 columns={'index': 'region'})]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_compile_rows_storing_var_finance_df.columns:
            place_holder_df = imp_compile_rows_storing_var_finance_df.drop(columns='index')
            imp_compile_rows_storing_var_finance_df = place_holder_df.copy()

        # dependency
        # load the services - into storage dataframe
        # cols - source sector
        dep_compile_cols_storing_df = pd.concat(
            [dep_compile_cols_storing_df.reset_index(), dep_compile_cols_one_bank_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in dep_compile_cols_storing_df.columns:
            place_holder_df = dep_compile_cols_storing_df.drop(columns='index')
            dep_compile_cols_storing_df = place_holder_df.copy()
        # rows - value chain sector
        # load the services - into storage dataframe
        dep_compile_rows_storing_df = pd.concat(
            [dep_compile_rows_storing_df.reset_index(),
             dep_compile_rows_one_bank_df.reset_index().rename(columns={'index': 'region'})]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in dep_compile_rows_storing_df.columns:
            place_holder_df = dep_compile_rows_storing_df.drop(columns='index')
            dep_compile_rows_storing_df = place_holder_df.copy()
        # var finance
        # load the services - source sector
        dep_compile_cols_storing_var_finance_df = pd.concat(
            [dep_compile_cols_storing_var_finance_df.reset_index(),
             dep_compile_cols_one_bank_var_finance_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in dep_compile_cols_storing_var_finance_df.columns:
            place_holder_df = dep_compile_cols_storing_var_finance_df.drop(columns='index')
            dep_compile_cols_storing_var_finance_df = place_holder_df.copy()
        # rows - value chain sector
        # var finance
        # load the services - into storage dataframe
        dep_compile_rows_storing_var_finance_df = pd.concat(
            [dep_compile_rows_storing_var_finance_df.reset_index(),
             dep_compile_rows_one_bank_var_finance_df.reset_index().rename(columns={'index': 'region'})]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in dep_compile_rows_storing_var_finance_df.columns:
            place_holder_df = dep_compile_rows_storing_var_finance_df.drop(columns='index')
            dep_compile_rows_storing_var_finance_df = place_holder_df.copy()

    # save the scores to csv
    # both - endogenous risk
    # cols - source sector
    imp_dep_compile_cols_storing_var_finance_df.name = f'{imp_score_name} {dep_score_name} Both Finance VaR Source'
    imp_dep_compile_cols_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Both/{imp_dep_compile_cols_storing_var_finance_df.name}.csv')
    storing_list.append(imp_dep_compile_cols_storing_var_finance_df)
    # rows - value chain sector
    imp_dep_compile_rows_storing_var_finance_df.name = f'{imp_score_name} {dep_score_name} Both Finance VaR Value Chain'
    imp_dep_compile_rows_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Both/{imp_dep_compile_rows_storing_var_finance_df.name}.csv')
    storing_list.append(imp_dep_compile_rows_storing_var_finance_df)

    # impact only
    # cols - source sector
    imp_compile_cols_storing_var_finance_df.name = f'{imp_score_name} Finance VaR Source'
    imp_compile_cols_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Impact/{imp_compile_cols_storing_var_finance_df.name}.csv')
    storing_list.append(imp_compile_cols_storing_var_finance_df)
    # rows - value chain sector
    imp_compile_rows_storing_var_finance_df.name = f'{imp_score_name} Finance VaR Value Chain'
    imp_compile_rows_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Impact/{imp_compile_rows_storing_var_finance_df.name}.csv')
    storing_list.append(imp_compile_rows_storing_var_finance_df)

    # dependency only
    # cols - source sector
    dep_compile_cols_storing_var_finance_df.name = f'{dep_score_name} Finance VaR Source'
    dep_compile_cols_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Dependency/{dep_compile_cols_storing_var_finance_df.name}.csv')
    storing_list.append(dep_compile_cols_storing_var_finance_df)
    # rows - value chain sector
    dep_compile_rows_storing_var_finance_df.name = f'{dep_score_name} Finance VaR Value Chain'
    dep_compile_rows_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Dependency/{dep_compile_rows_storing_var_finance_df.name}.csv')
    storing_list.append(dep_compile_rows_storing_var_finance_df)

    return None


def finance_var_calc_scope_3_combined_sector(imp_score_df, dep_score_df, finance_data_NACE_region_grouped_w_total_df, type, folder):
    """
    This function calculates the upstream supply chain endogenous risk at the sector-level and stores the
    results in the data repository.
    :param imp_score_df: dataframe of the impact score that you want to use to calculate the value at risk
    :param dep_score_df: dataframe of the dependency score that you want to use to calculate the value at risk
    :param finance_data_NACE_region_grouped_w_total_df: the finance of the bank you want to calculate with (formatted)
    :param folder: subfolder for storing the risk exposure in the data repository
    :param type: describes whether the score has sector and regions or just sectors or just regions
    :return: list of dataframes that correspond to the finance_var and the rows for the finance_var plus the scores
    """
    # check if type is an accepted type
    if type != 'region_code' and type != 'code_only' and type != 'region_only':
        print('ERROR: Type must be "region_code" or "code_only" or "region_only"')
        return

    # create a list to store the results
    storing_list = []

    # calculate the relative weighting of sector-region pairs in upstream supply chains
    ### calculate overlined((L -1)), relative impact dependency matrix
    # generate the Leontief matrix without the sector itself in its own upstream supply chain
    L_min_I = calc_L_min_I_full()
    L_min_I_numpy = L_min_I.to_numpy(dtype=float)
    # get the column sums of the L_matrix without itself included in supply chain
    col_sums = np.sum(L_min_I, axis=0)
    col_sums = col_sums.to_numpy(dtype=float)
    # divide each element by its column sum to get the relative importance in the upstream supply chain
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_imp_array = np.where(col_sums == 0, 0, np.divide(L_min_I_numpy, col_sums[np.newaxis, :]))

    # get the weights for the contribution of each sector,region pair to the supply chain
    L_weights = pd.DataFrame(rel_imp_array, index=L_min_I.index, columns=L_min_I.columns)
    upstream_calc = L_weights.copy().reset_index()
    upstream_calc_format = upstream_calc.T.reset_index().T.rename(columns={0: 'region', 1: 'sector'})
    upstream_calc_format.loc['sector', 'region'] = 'region'
    upstream_calc_format.loc['sector', 'sector'] = 'sector'

    # list of banks
    banks = np.unique(finance_data_NACE_region_grouped_w_total_df.reset_index()['Bank'])
    # list of ecosystem services
    services = imp_score_df.columns

    # multiply the weighted sum for Leontief by the impact region score (based on investment location)
    # plus the dependency score for the sector region L_weighted * (impact region) * (dependency sector)

    # for the storing the score-level scores
    # both - endogenous risk
    imp_dep_compile_cols_df = pd.DataFrame(index=L_min_I.index)
    imp_dep_compile_rows_df = pd.DataFrame(index=np.unique(L_min_I.reset_index()['region']))
    # storing value at risk
    # both - endogenous risk
    imp_dep_compile_cols_var_finance_df = pd.DataFrame(index=L_min_I.index)
    imp_dep_compile_rows_var_finance_df = pd.DataFrame(index=np.unique(L_min_I.reset_index()['region']))
    # for storing the column sums for the score-level scores
    imp_dep_compile_cols_storing_df = pd.DataFrame(columns=services)
    # for storing the row sums for the score-level scores
    imp_dep_compile_rows_storing_df = pd.DataFrame(columns=services)
    # both - endogenous rsik
    # value at risk finance - source sector
    imp_dep_compile_cols_storing_var_finance_df = pd.DataFrame(columns=services)
    # value at risk finance rows - value chain sector
    imp_dep_compile_rows_storing_var_finance_df = pd.DataFrame(columns=services)

    # store scores
    # impact scores
    imp_df = imp_score_df.copy()
    imp_score_name = imp_score_df.name
    # dependency scores
    dep_df = dep_score_df.copy()
    dep_score_name = dep_score_df.name

    # loop through the banks
    for bank in banks:
        # fill in bank column with the name of the bank
        # source sector
        imp_dep_compile_cols_one_bank_df = imp_dep_compile_cols_df.copy()
        imp_dep_compile_cols_one_bank_df['Bank'] = [f'{bank}'] * imp_dep_compile_cols_df.shape[0]
        # rows - value chain sector
        imp_dep_compile_rows_one_bank_df = imp_dep_compile_rows_df.copy()
        imp_dep_compile_rows_one_bank_df['Bank'] = [f'{bank}'] * imp_dep_compile_rows_df.shape[0]
        # value at risk finance
        # both - endogenous risk - source sectors
        imp_dep_compile_cols_one_bank_var_finance_df = imp_dep_compile_cols_var_finance_df.copy()
        imp_dep_compile_cols_one_bank_var_finance_df['Bank'] = [f'{bank}'] * imp_dep_compile_cols_var_finance_df.shape[
            0]
        # value at risk finance rows
        # both - endogenous risk - value chain sectors
        imp_dep_compile_rows_one_bank_var_finance_df = imp_dep_compile_rows_var_finance_df.copy()
        imp_dep_compile_rows_one_bank_var_finance_df['Bank'] = [f'{bank}'] * imp_dep_compile_rows_var_finance_df.shape[
            0]

        # get financial data for Bank
        bank_data_df = finance_data_NACE_region_grouped_w_total_df.reset_index()[finance_data_NACE_region_grouped_w_total_df.reset_index()['Bank'] == bank].set_index(
            ['region', 'sector'])
        full_index = pd.DataFrame(index=L_min_I.index)
        bank_data_df = bank_data_df.merge(full_index, how='right', right_index=True, left_index=True)
        bank_data_df = bank_data_df.fillna(0.0)
        bank_data_dict = bank_data_df['Proportion of Loans'].to_dict()
        bank_data_absolute_dict = bank_data_df['EUR m adjusted'].to_dict()

        # loop through ecosysetm sergices
        for service in services:
            # impact score
            # if score has sector and region
            if type == 'region_code':
                # get the scores and the weighted L in one DF
                compiled_imp_df = upstream_calc_format.merge(
                    imp_df[service].reset_index(), how='outer', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
                compiled_imp_df = compiled_imp_df.fillna(0.0)
            # if score has sector only
            if type == 'code_only':
                # get the scores and the weighted L in one DF
                compiled_imp_df = upstream_calc_format.merge(
                    imp_df[service].reset_index(), how='outer', left_on=['sector'],
                    right_on=['sector'])
                compiled_imp_df = compiled_imp_df.fillna(0.0)
            # if score has region only
            if type == 'region_only':
                # get the scores and the weighted L in one DF
                compiled_imp_df = upstream_calc_format.merge(
                    imp_df[service].reset_index(), how='outer', left_on=['region'],
                    right_on=['region'])
                compiled_imp_df = compiled_imp_df.fillna(0.0)


            # multiply the weighted average by the score
            compiled_imp_df = compiled_imp_df.set_index(['region', 'sector'])
            compiled_imp_df = compiled_imp_df.T.set_index(('region', 'sector')).T
            service_imp_df = compiled_imp_df[(0.0,0.0)]
            service_imp_df = service_imp_df[0:(L_min_I.shape[0])]
            service_imp_df = service_imp_df.astype(float)
            calc_imp_df = compiled_imp_df.drop(columns =(0.0,0.0))
            calc_imp_df = calc_imp_df.astype(float)
            multiplied_imp_df = np.multiply(calc_imp_df.to_numpy(), service_imp_df.to_numpy()[:, np.newaxis])
            imp_dep_compile_service_imp_df = pd.DataFrame(multiplied_imp_df, index=calc_imp_df.index, columns=calc_imp_df.columns)

            # dependency score
            # if score has sector and region
            if type == 'region_code':
                # get the scores and the weighted L in one DF
                compiled_dep_df = upstream_calc_format.merge(
                    dep_df[service].reset_index(), how='outer', left_on=['region', 'sector'],
                    right_on=['region', 'sector'])
                compiled_dep_df = compiled_dep_df.fillna(0.0)
            # if score has sector only
            if type == 'code_only':
                # get the scores and the weighted L in one DF
                compiled_dep_df = upstream_calc_format.merge(
                    dep_df[service].reset_index(), how='outer', left_on=['sector'],
                    right_on=['sector'])
                compiled_dep_df = compiled_dep_df.fillna(0.0)
            # if score has region only
            if type == 'region_only':
                # get the scores and the weighted L in one DF
                compiled_dep_df = upstream_calc_format.merge(
                    dep_df[service].reset_index(), how='outer', left_on=['region'],
                    right_on=['region'])
                compiled_dep_df = compiled_dep_df.fillna(0.0)

            # multiply the weighted average by the score
            compiled_dep_df = compiled_dep_df.set_index(['region', 'sector'])
            compiled_dep_df = compiled_dep_df.T.set_index(('region', 'sector')).T
            service_dep_df = compiled_dep_df[(0.0, 0.0)]
            service_dep_df = service_dep_df[0:(L_min_I.shape[0])]
            service_dep_df = service_dep_df.astype(float)
            calc_dep_df = compiled_dep_df.drop(columns=(0.0, 0.0))
            calc_dep_df = calc_dep_df.astype(float)
            multiplied_dep_df = np.multiply(calc_dep_df.to_numpy(), service_dep_df.to_numpy()[:, np.newaxis])
            imp_dep_compile_service_dep_df = pd.DataFrame(multiplied_dep_df, index=calc_dep_df.index,
                                                          columns=calc_dep_df.columns)

            # combine the impact and dependency scores - at the sector level
            imp_dep_compile_service_df = imp_dep_compile_service_dep_df * imp_dep_compile_service_imp_df

            # combine the impact and dependency score value at risks
            # multiply score by the absolute value of finance to sr to get VaR
            imp_dep_compile_service_finance_VaR_df = imp_dep_compile_service_df
            imp_dep_compile_service_finance_VaR_df = imp_dep_compile_service_finance_VaR_df.mul(
                bank_data_absolute_dict, axis='columns')

            # both - endogenous risk
            # get the column sums for one bank for the scores
            imp_dep_compile_cols_one_bank_df[service] = imp_dep_compile_service_df.sum()
            # get the row sums for one bank and service into the greater the df
            imp_dep_compile_rows_one_bank_df[f'{service}'] = imp_dep_compile_service_df.T.sum()

            # both - endogenous risk
            # get column sums for one bank for the value at risk finance
            imp_dep_compile_cols_one_bank_var_finance_df[service] = imp_dep_compile_service_finance_VaR_df.sum()
            # get the row sums for one bank and service into the greater the df
            imp_dep_compile_rows_one_bank_var_finance_df[f'{service}'] = imp_dep_compile_service_finance_VaR_df.T.sum()

        # both - endogenous risk
        # load the services - into storage dataframe
        # cols - source sector
        imp_dep_compile_cols_storing_df = pd.concat(
            [imp_dep_compile_cols_storing_df.reset_index(), imp_dep_compile_cols_one_bank_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_dep_compile_cols_storing_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_df.drop(columns='index')
            imp_dep_compile_cols_storing_df = place_holder_df.copy()
        # rows - value chain sector
        # load the services - into storage dataframe
        imp_dep_compile_rows_storing_df = pd.concat(
            [imp_dep_compile_rows_storing_df.reset_index(), imp_dep_compile_rows_one_bank_df.reset_index().rename(columns={'index':'region'})]).set_index(
            ['Bank', 'region'])
        if 'index' in imp_dep_compile_rows_storing_df.columns:
            place_holder_df = imp_dep_compile_rows_storing_df.drop(columns='index')
            imp_dep_compile_rows_storing_df = place_holder_df.copy()
        # var finance - source sectors
        # load the services - into storage dataframe
        imp_dep_compile_cols_storing_var_finance_df = pd.concat(
            [imp_dep_compile_cols_storing_var_finance_df.reset_index(),
             imp_dep_compile_cols_one_bank_var_finance_df.reset_index()]).set_index(
            ['Bank', 'region', 'sector'])
        if 'index' in imp_dep_compile_cols_storing_var_finance_df.columns:
            place_holder_df = imp_dep_compile_cols_storing_var_finance_df.drop(columns='index')
            imp_dep_compile_cols_storing_var_finance_df = place_holder_df.copy()
        # rows - value chain sectors
        # var finance
        # load the services - into storage dataframe
        imp_dep_compile_rows_storing_var_finance_df = pd.concat(
            [imp_dep_compile_rows_storing_var_finance_df.reset_index(),
             imp_dep_compile_rows_one_bank_var_finance_df.reset_index().rename(columns={'index':'region'})]).set_index(
            ['Bank', 'region'])
        if 'index' in imp_dep_compile_rows_storing_var_finance_df.columns:
            place_holder_df = imp_dep_compile_rows_storing_var_finance_df.drop(columns='index')
            imp_dep_compile_rows_storing_var_finance_df = place_holder_df.copy()

    # save the scores to csv
    # both - endogenous risk
    # cols - source sector
    imp_dep_compile_cols_storing_var_finance_df.name = f'{imp_score_name} {dep_score_name} Both Finance VaR Source'
    imp_dep_compile_cols_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Both/Sectoral/{imp_dep_compile_cols_storing_var_finance_df.name}.csv')
    storing_list.append(imp_dep_compile_cols_storing_var_finance_df)
    # rows - value chain sector
    imp_dep_compile_rows_storing_var_finance_df.name = f'{imp_score_name} {dep_score_name} Both Finance VaR Value Chain Sectoral'
    imp_dep_compile_rows_storing_var_finance_df.to_csv(
        f'../Data/Value at Risk/Finance/{folder}Both/Sectoral/{imp_dep_compile_rows_storing_var_finance_df.name}.csv')
    storing_list.append(imp_dep_compile_rows_storing_var_finance_df)

    return None


def GSIB_calculate_finance_var(finance_df, score_type):
    """
    This function calculates the direct operations and upstream supply chain endogenous risk based on the financial data
    provided and stores these into their appropriate subfolders in the data repository. The functions calculates the
    endogenous risk for the bank portfolio-level and sector-level.
    :param finance_df: the financial data used as the exposures of the banks to sector-region pairs in their portfolios
    :param score_type: the methodological treatment of the score (mean, max, min)
    :return: NA
    """
    # get the appropriate score path from the data repository based the methodological treatment
    if score_type == 'mean':
        impact_path = scope1_impact_mean_path
        dependency_path = scope1_dependency_mean_path
    if score_type == 'min':
        impact_path = scope1_impact_min_path
        dependency_path = scope1_dependency_min_path
    if score_type == 'max':
        impact_path = scope1_impact_max_path
        dependency_path = scope1_dependency_max_path

    # read the scores from the appropriate paths and generate the combined impact and dependecy score used to calculate
    # the sector-level endogenous risk
    combined_df, impact_df, dependency_df = GSIB_calculate_direct_var(impact_path, dependency_path)

    # calculate the endogenous risk at the sector-level based on the provided financial data
    # combined is the sectoral score multiplying dependency and impact for each sector directly
    GSIB_var_calc_scope_1_sector(combined_df, finance_df, 'region_code', 'GSIB/Both/Sectoral')

    # calculate the direct operations endogneous risk at the bank portfolio-level
    GSIB_var_calc_scope_1(impact_df, dependency_df, finance_df,'region_code', 'GSIB/Both')
    # calculate the direct operations impact risk
    GSIB_var_calc_scope_1_sector(impact_df, finance_df, 'region_code', 'GSIB/Impact')
    # calculate the direct operations dependency risk
    GSIB_var_calc_scope_1_sector(dependency_df, finance_df, 'region_code', 'GSIB/Dependency')

    # calculate the upstream supply chain endogenous risk at the bank portfolio-level
    finance_var_calc_scope_3_combined(impact_df, dependency_df, finance_df, 'region_code', 'GSIB/')
    # calculate the upstream supply chain endogenous risk at the sector-level
    finance_var_calc_scope_3_combined_sector(impact_df, dependency_df, finance_df, 'region_code', 'GSIB/')

    return None

def GSIB_calculate_finance_var_system(finance_df, score_type):
    """
    This function calculates the direct operations and upstream supply chain endogenous risk for the system-level
    stores these into their appropriate subfolders in the data repository.
    :param finance_df: all the financial data aggregated into one bank to represent the system
    :param score_type: the methodological treatment of the score
    :return:
    """
    # get the appropriate score path from the data repository based the methodological treatment
    if score_type == 'mean':
        impact_path = scope1_impact_mean_path
        dependency_path = scope1_dependency_mean_path
    if score_type == 'min':
        impact_path = scope1_impact_min_path
        dependency_path = scope1_dependency_min_path
    if score_type == 'max':
        impact_path = scope1_impact_max_path
        dependency_path = scope1_dependency_max_path

    # read the scores from the appropriate paths and generate the combined impact and dependecy score used to calculate
    combined_df, impact_df, dependency_df = GSIB_calculate_direct_var(impact_path, dependency_path)

    # calculate the endogenous risk at the sector-level based on the provided financial data
    # combined is the sectoral score multiplying dependency and impact for each sector directly
    GSIB_var_calc_scope_1_sector(combined_df, finance_df, 'region_code', 'GSIB/System/Both/Sectoral')
    # calculate the direct operations endogneous risk at the system-level
    GSIB_var_calc_scope_1(impact_df, dependency_df, finance_df,'region_code', 'GSIB/System/Both')
    # calculate the direct operations impact risk for the system
    GSIB_var_calc_scope_1_sector(impact_df, finance_df, 'region_code', 'GSIB/System/Impact')
    # calculate the direct operations dependency risk for the system
    GSIB_var_calc_scope_1_sector(dependency_df, finance_df, 'region_code', 'GSIB/System/Dependency')

    # calculate the upstream supply chain endogenous risk at the system-level
    finance_var_calc_scope_3_combined(impact_df, dependency_df, finance_df, 'region_code', 'GSIB/System/')
    # calculate the upstream supply chain endogenous risk at the sector-level
    finance_var_calc_scope_3_combined_sector(impact_df, dependency_df, finance_df, 'region_code', 'GSIB/System/')

    return None



def GSIB_calculate_direct_var(scope1_impact_NACE_path, scope1_dependency_NACE_path):
    """
    This function reads the impact and dependency scores from the appropriate path and returns them as a dataframe.
    The function also calculates the combined score which is used to calculate the endogenous risk for direct operations
    at the sector-level
    :param scope1_impact_NACE_path: path to the appropriate impact score
    :param scope1_dependency_NACE_path: path to the appropriate dependency score
    :return: combined score that measures the endogenous risk at the sector-level, impact score, dependency score
    """
    # store the methodological treatment for the impact score
    if 'mean' in scope1_impact_NACE_path:
        imp_score_type = 'mean'
    if 'min' in scope1_impact_NACE_path:
        imp_score_type = 'min'
    if 'max' in scope1_impact_NACE_path:
        imp_score_type = 'max'

    # store the methodological treatment for the depedency score
    if 'mean' in scope1_dependency_NACE_path:
        dep_score_type = 'mean'
    if 'min' in scope1_dependency_NACE_path:
        dep_score_type = 'min'
    if 'max' in scope1_dependency_NACE_path:
        dep_score_type = 'max'

    # read and format the impact score from the appropriate path
    impact_df = pd.read_csv(scope1_impact_NACE_path, header=[0, 1], index_col=[0])
    impact_df = impact_df.T
    impact_df.name = f"Impact {imp_score_type}"

    # read and format the dependency score from the appropriate path
    dependency_df = pd.read_csv(scope1_dependency_NACE_path, header=[0, 1], index_col=[0])
    dependency_df = dependency_df.T
    dependency_df.name = f"Dependency {dep_score_type}"

    # calculate the combined score that measure the overlap between impact and dependency at the sector-level
    combined_df = impact_df * dependency_df
    combined_df.name = f"Impact {imp_score_type} and Dependency {dep_score_type}"

    return combined_df, impact_df, dependency_df