import re
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
# from banks_at_risk.Setup.var_paths import finance_var_both_scope1_mean, finance_var_imp_scope1_mean, \
#     finance_var_both_scope1_max, finance_var_both_scope1_min, finance_var_both_scope3_source_max, \
#     finance_var_both_scope3_source_max, finance_var_both_scope3_source_mean, finance_var_both_scope3_source_min, \
#     finance_var_both_scope3_value_chain_max, finance_var_both_scope3_value_chain_mean, \
#     finance_var_both_scope3_value_chain_min, finance_var_dep_scope1_max, finance_var_dep_scope1_mean, \
#     finance_var_dep_scope1_min, finance_var_dep_scope3_source_max, finance_var_dep_scope3_source_mean, \
#     finance_var_dep_scope3_source_min, finance_var_dep_scope3_value_chain_max, finance_var_dep_scope3_value_chain_mean, \
#     finance_var_dep_scope3_value_chain_min, finance_var_imp_scope1_max, finance_var_imp_scope1_min, \
#     finance_var_imp_scope3_source_max, finance_var_imp_scope3_source_mean, finance_var_imp_scope3_source_min, \
#     finance_var_imp_scope3_value_chain_max, finance_var_imp_scope3_value_chain_mean, \
#     finance_var_imp_scope3_value_chain_min
from banks_at_risk.Setup.var_paths import GSIB_finance_var_both_scope1_max, GSIB_finance_var_both_scope3_source_mean, \
    GSIB_finance_var_dep_scope3_source_max, GSIB_finance_var_imp_scope1_min, GSIB_finance_var_both_scope1_mean, \
    GSIB_finance_var_dep_scope1_max, GSIB_finance_var_both_scope3_source_min, GSIB_finance_var_both_scope1_min, \
    GSIB_finance_var_both_scope3_source_max, GSIB_finance_var_dep_scope3_source_mean, GSIB_finance_var_dep_scope1_mean, \
    GSIB_finance_var_imp_scope3_source_max, GSIB_finance_var_imp_scope1_max, GSIB_finance_var_dep_scope3_source_min, \
    GSIB_finance_var_dep_scope1_min, GSIB_finance_var_imp_scope3_source_mean, GSIB_finance_var_imp_scope1_mean, \
    GSIB_finance_var_both_scope3_value_chain_max, GSIB_finance_var_imp_scope3_source_min, \
    GSIB_finance_var_both_scope3_value_chain_mean, GSIB_finance_var_both_scope3_value_chain_min, \
    GSIB_finance_var_dep_scope3_value_chain_max, GSIB_finance_var_dep_scope3_value_chain_mean, \
    GSIB_finance_var_dep_scope3_value_chain_min, GSIB_finance_var_imp_scope3_value_chain_max, \
    GSIB_finance_var_imp_scope3_value_chain_mean, GSIB_finance_var_imp_scope3_value_chain_min, \
    GSIB_finance_var_both_scope1_sector_max, GSIB_finance_var_both_scope1_sector_mean, \
    GSIB_finance_var_both_scope1_sector_min, GSIB_finance_var_both_scope3_source_sector_mean, \
    GSIB_finance_var_both_scope3_source_sector_max, GSIB_finance_var_both_scope3_source_sector_min, \
    GSIB_system_var_dep_scope3_source_max, GSIB_system_var_both_scope3_source_sector_max, \
    GSIB_system_var_dep_scope3_value_chain_max, GSIB_system_var_imp_scope3_value_chain_max, \
    GSIB_system_var_dep_scope3_value_chain_mean, GSIB_system_var_dep_scope3_value_chain_min, \
    GSIB_system_var_both_scope1_max, GSIB_system_var_both_scope1_sector_max, GSIB_system_var_dep_scope1_mean, \
    GSIB_system_var_both_scope1_mean, GSIB_system_var_both_scope1_sector_mean, GSIB_system_var_dep_scope1_max, \
    GSIB_system_var_both_scope1_min, GSIB_system_var_both_scope1_sector_min, GSIB_system_var_imp_scope1_max, \
    GSIB_system_var_dep_scope1_min, GSIB_system_var_dep_scope3_source_min, GSIB_system_var_both_scope3_source_max, \
    GSIB_system_var_imp_scope1_mean, GSIB_system_var_imp_scope3_source_mean, GSIB_system_var_both_scope3_source_min, \
    GSIB_system_var_imp_scope1_min, GSIB_system_var_imp_scope3_source_min, GSIB_system_var_both_scope3_source_mean, \
    GSIB_system_var_dep_scope3_source_mean, GSIB_system_var_both_scope3_source_sector_mean, \
    GSIB_system_var_imp_scope3_source_max, GSIB_system_var_both_scope3_source_sector_min, \
    GSIB_system_var_both_scope3_value_chain_max, GSIB_system_var_imp_scope3_value_chain_mean, \
    GSIB_system_var_both_scope3_value_chain_mean, GSIB_system_var_imp_scope3_value_chain_min, \
    GSIB_system_var_both_scope3_value_chain_min
from banks_at_risk.Setup.finance_paths import finance_data_path, finance_exio_region_path, GSIB_bank_regions, GSIB_bank_names
from banks_at_risk.Value_at_Risk.helper_value_at_risk_GSIB import finance_GSIB_reformat
from banks_at_risk.Setup.var_plots_paths import GSIB_value_at_risk_sig_saving_path, value_at_risk_figure_saving_path, financial_bar_chart_path
from banks_at_risk.NACE_Conversion.helpers_NACE_conversion import  convert_EXIO_to_NACE


def get_var_scores(type_score, calc_type):
    """
    This function reads the scores based on score type (endogenous risk, impact or dependency) and methodological
    treatment (mean, max, min)
    type_score: 'Both', 'Impact' or 'Dependency' for each type
    calc_type: 'mean', 'min', 'max'
    return: list of score dataframes meeting the criteria provided
    """

    # check if calc_type is accepted
    if calc_type != 'mean' and calc_type != 'min' and calc_type != 'max':
        print("calc_type must be 'mean', 'min'or 'max'")
        return
    # check if type_score is accepted
    if type_score != 'Both' and type_score != 'Impact' and type_score != 'Dependency' and type_score != 'Sector' and type_score != 'System Impact' and type_score != 'System Both' and type_score != 'System Dependency' and type_score != 'System Sector':
        print("Type must be 'Both', 'Impact' or 'Dependency' or any 3 category preceeded by 'System'")
        return

    # load the appropriate score paths based on criteria
    if calc_type == 'mean':
        if type_score == 'Both':
            var_file_path_list = [GSIB_finance_var_both_scope1_mean, GSIB_finance_var_both_scope3_source_mean,
                                  GSIB_finance_var_both_scope3_value_chain_mean]
        if type_score == 'Impact':
            var_file_path_list = [GSIB_finance_var_imp_scope1_mean, GSIB_finance_var_imp_scope3_source_mean,
                                  GSIB_finance_var_imp_scope3_value_chain_mean]
        if type_score == 'Dependency':
            var_file_path_list = [GSIB_finance_var_dep_scope1_mean, GSIB_finance_var_dep_scope3_source_mean,
                                  GSIB_finance_var_dep_scope3_value_chain_mean]
        if type_score == 'Sector':
            var_file_path_list = [GSIB_finance_var_both_scope3_source_sector_mean, GSIB_finance_var_both_scope1_sector_mean]
    if calc_type == 'min':
        if type_score == 'Both':
            var_file_path_list = [GSIB_finance_var_both_scope1_min, GSIB_finance_var_both_scope3_source_min,
                                  GSIB_finance_var_both_scope3_value_chain_min]
        if type_score == 'Impact':
            var_file_path_list = [GSIB_finance_var_imp_scope1_min, GSIB_finance_var_imp_scope3_source_min,
                                  GSIB_finance_var_imp_scope3_value_chain_min]
        if type_score == 'Dependency':
            var_file_path_list = [GSIB_finance_var_dep_scope1_min, GSIB_finance_var_dep_scope3_source_min,
                                  GSIB_finance_var_dep_scope3_value_chain_min]
        if type_score == 'Sector':
            var_file_path_list = [GSIB_finance_var_both_scope3_source_sector_min, GSIB_finance_var_both_scope1_sector_min]
    if calc_type == 'max':
        if type_score == 'Both':
            var_file_path_list = [GSIB_finance_var_both_scope1_max, GSIB_finance_var_both_scope3_source_max,
                                  GSIB_finance_var_both_scope3_value_chain_max]
        if type_score == 'Impact':
            var_file_path_list = [GSIB_finance_var_imp_scope1_max, GSIB_finance_var_imp_scope3_source_max,
                                  GSIB_finance_var_imp_scope3_value_chain_max]
        if type_score == 'Dependency':
            var_file_path_list = [GSIB_finance_var_dep_scope1_max, GSIB_finance_var_dep_scope3_source_max,
                                  GSIB_finance_var_dep_scope3_value_chain_max]
        if type_score == 'Sector':
            var_file_path_list = [GSIB_finance_var_both_scope3_source_sector_max, GSIB_finance_var_both_scope1_sector_max]

    if 'System' in type_score:
        if 'Dependency' in type_score:
            if calc_type == 'mean':
                var_file_path_list = [GSIB_system_var_dep_scope1_mean, GSIB_system_var_dep_scope3_source_mean,
                                      GSIB_system_var_dep_scope3_value_chain_mean]
            if calc_type == 'max':
                var_file_path_list = [GSIB_system_var_dep_scope1_max, GSIB_system_var_dep_scope3_source_max,
                                      GSIB_system_var_dep_scope3_value_chain_max]
            if calc_type == 'min':
                var_file_path_list = [GSIB_system_var_dep_scope1_min, GSIB_system_var_dep_scope3_source_min,
                                      GSIB_system_var_dep_scope3_value_chain_min]
        if 'Impact' in type_score:
            if calc_type == 'mean':
                var_file_path_list = [GSIB_system_var_imp_scope1_mean, GSIB_system_var_imp_scope3_source_mean,
                                      GSIB_system_var_imp_scope3_value_chain_mean]
            if calc_type == 'max':
                var_file_path_list = [GSIB_system_var_imp_scope1_max, GSIB_system_var_imp_scope3_source_max,
                                      GSIB_system_var_imp_scope3_value_chain_max]
            if calc_type == 'min':
                var_file_path_list = [GSIB_system_var_imp_scope1_min, GSIB_system_var_imp_scope3_source_min,
                                      GSIB_system_var_imp_scope3_value_chain_min]
        if 'Both' in type_score:
            if calc_type == 'mean':
                var_file_path_list = [GSIB_system_var_both_scope1_mean, GSIB_system_var_both_scope3_source_mean,
                                      GSIB_system_var_both_scope3_value_chain_mean]
            if calc_type == 'max':
                var_file_path_list = [GSIB_system_var_both_scope1_max, GSIB_system_var_both_scope3_source_max,
                                      GSIB_system_var_both_scope3_value_chain_max]
            if calc_type == 'min':
                var_file_path_list = [GSIB_system_var_both_scope1_min, GSIB_system_var_both_scope3_source_min,
                                      GSIB_system_var_both_scope3_value_chain_min]
        if 'Sector' in type_score:
            if calc_type == 'mean':
                var_file_path_list = [GSIB_system_var_both_scope1_sector_mean, GSIB_system_var_both_scope3_source_sector_mean]
            if calc_type == 'max':
                var_file_path_list = [GSIB_system_var_both_scope1_sector_max, GSIB_system_var_both_scope3_source_sector_max]
            if calc_type == 'min':
                var_file_path_list = [GSIB_system_var_both_scope1_sector_min, GSIB_system_var_both_scope3_source_sector_min]

    # create a list to store the scores
    score_list = []
    # loop through paths matching the criteria
    for score_path in var_file_path_list:
        # read the csv file containing the results based on the path and appropriate type
        # append the dataframe to the score_list
        if 'Value Chain' in score_path:
            score_df = pd.read_csv(score_path, header=[0], index_col=[0, 1])
            score_df.name = f'{type_score} Value Chain {calc_type}'
            score_list.append(score_df)
            next
        if 'Source' in score_path:
            score_df = pd.read_csv(score_path, header=[0], index_col=[0, 1, 2])
            score_df.name = f'{type_score} Source {calc_type}'
            score_list.append(score_df)
            next
        if 'Scope 1' in score_path:
            score_df = pd.read_csv(score_path, header=[0], index_col=[0, 1, 2])
            score_df.name = f'{type_score} Scope 1 {calc_type}'
            score_list.append(score_df)

    # create a list containing the absolute result values
    absolute_list = score_list.copy()
    # loop through absolute result dataframes
    for score in score_list:
        # calculate the proporational score by dividing by bank total to get proportion of the portfolio exposed
        prop_score = proportion_transformation(score)
        prop_score.name = f'{score.name} Proportional'
        # append proportional scores to the storage absolute result list from above
        absolute_list.append(prop_score)

    return absolute_list


def proportion_transformation(score_df):
    """
    This function transforms the result from absolute endogenous risk to proportion of the portfolio exposed to
    endogenous risk by dividing the absolute results by the total portfolio value of the bank
    :param score_df: absolute results
    :return: proportional results
    """
    # get the % values for the difference - format bank data
    financial_data_df = finance_GSIB_reformat()
    # for system - get the proportion of the system
    if 'System' == score_df.reset_index()['Bank'].iloc[0]:
        system_total = financial_data_df['EUR m adjusted'].sum()
        df = score_df.copy()
        # divide by SYSTEM TOTAL
        proportional_score_df = df/system_total
        proportional_score_df = proportional_score_df * 100
        return proportional_score_df

    # the  total portfolio value for each bank
    total_value_banks_df = financial_data_df.reset_index().drop(
        columns=['sector', 'region', 'Total Loan', 'Proportion of Loans']).groupby(['Bank']).sum()

    # merge the total bank values with the absolute results values
    df = score_df.reset_index().copy()
    df_2 = total_value_banks_df.reset_index().copy()
    if 'sector' in df.columns:
        if 'region' in df.columns:
            finance_score_merge_df = df.merge(df_2, right_on=['Bank'], left_on=['Bank'], how='left').set_index(
                ['Bank', 'region', 'sector'])
        else:
            finance_score_merge_df = df.merge(df_2, right_on=['Bank'], left_on=['Bank'], how='left').set_index(
                ['Bank', 'sector'])
    else:
        finance_score_merge_df = df.merge(df_2, right_on=['Bank'], left_on=['Bank'], how='left').set_index(
            ['Bank', 'region'])

    # calculate the proportional score by dividing the absolute results by the total bank portoflio value
    proportional_score_df = finance_score_merge_df.iloc[:, 0:(finance_score_merge_df.shape[1] - 1)].div(
        finance_score_merge_df.iloc[:, (finance_score_merge_df.shape[1] - 1)], axis=0)
    # multiply by 100 to get the % of the portfolio exposed
    proportional_score_df = proportional_score_df * 100

    return proportional_score_df


def aggregate_to_region_service(score_list, how):
    """
    This function aggregates the list of scores to the region level.
    :param score_list: list of scores with sectors and regions
    :param how: determine whether to 'mean' or 'sum' for the aggregation
    :return: return a list of the score aggregated to the regional level
    """
    # create a list to store the aggregated scores
    return_list = []
    # check whether the score list is only one dataframe
    if isinstance(score_list, pd.DataFrame):
        # store the score
        df = score_list.copy()
        name = score_list.name
        # aggregate to the region by the appropriate method from 'how'
        if how == 'sum':
            # for portfolio-level
            if 'Bank' in df.reset_index().columns:
                df = df.reset_index().drop(columns='sector').groupby(['Bank', 'region']).sum()
            # for sector or system level
            else:
                df = df.reset_index().drop(columns='sector').groupby(['region']).sum()
        if how == 'mean':
            # for portfolio-level
            if 'Bank' in df.reset_index().columns:
                df = df.reset_index().drop(columns='sector').groupby(['Bank', 'region']).mean()
            # for sector or system level
            else:
                df = df.reset_index().drop(columns='sector').groupby(['region']).mean()

        df.name = name
        # in the case where there is only one dataframe in the list - return only the one dataframe aggregated
        return df

    # loop through scores in score_list
    for score in score_list:
        # store the score
        df = score.copy()
        name = score.name
        # aggregate to the region by the appropriate method from 'how'
        if how == 'sum':
            # for portfolio-level
            if 'Bank' in df.reset_index().columns:
                df = df.reset_index().drop(columns='sector').groupby(['Bank', 'region']).sum()
            # for sector or system level
            else:
                df = df.reset_index().drop(columns='sector').groupby(['region']).sum()
        if how == 'mean':
            # for portfolio-level
            if 'Bank' in df.reset_index().columns:
                df = df.reset_index().drop(columns='sector').groupby(['Bank', 'region']).mean()
            # for sector or system level
            else:
                df = df.reset_index().drop(columns='sector').groupby(['region']).mean()
        # name the aggregated dataframe
        df.name = name
        # append to the list for storing the aggregated dataframes
        return_list.append(df)

    return return_list


def anonymize_banks(score_list):
    """
    This function anonymizes the scores from the list.
    :param score_list: list of scores
    :return: list of scores with anonymized banks
    """
    # get the list of bank names
    bank_names = np.unique(score_list[0].reset_index()['Bank']).tolist()
    # create a list to store the anonymous bank names
    anonymized_names = []
    # generate the names of the banks Bank 1, 2, ..., n based on the number of banks and append to list
    for i in range(1, (len(bank_names) + 1)):
        anonymized_names.append(f'Bank {i}')

    # create a list to store the anonymized scores
    new_score_list = []
    # loop through the scores
    for score in score_list:
        i = 0
        # store the score
        df = score.reset_index()
        name = score.name
        # loop through banks and rename the banks based on the anonymized names
        for bank in bank_names:
            df = df.replace({f'{bank}': f'{anonymized_names[i]}'})
            i = i + 1
        if 'region' in df.columns:
            if 'sector' in df.columns:
                df = df.set_index(['Bank', 'region', 'sector'])
            else:
                df = df.set_index(['Bank', 'region'])
        else:
            df = df.set_index(['Bank', 'sector'])

        # append the anonymized score to the list
        df.name = name
        new_score_list.append(df)

    return new_score_list


def anonymize(score_list_1, score_list_2, score_list_3):
    """
    This function anonymizes multiple scores at a time from multiple lists.
    :param score_list_1: a non-anonymzied score list
    :param score_list_2: a non-anonymzied score list
    :param score_list_3: a non-anonymzied score list
    :return:
    """
    # anonymize the scores in these three lists
    anon_1 = anonymize_banks(score_list_1)
    anon_2 = anonymize_banks(score_list_2)
    anon_3 = anonymize_banks(score_list_3)

    return anon_1, anon_2, anon_3


def statistical_tests(score_list):
    """
    This function tests the difference between direct operations (scope 1) and upstream supply chain (scope 3).
    The function uses a wilcoxon test and stores the results in sheets in the data repository.
    :param score_list: a score list containing the scope 1 and scope 3 scores for a particular type and method
    :return: NA
    """
    # check if system-level score or portfolio-level score
    if 'System' == score_list[0].reset_index()['Bank'].iloc[0]:
        type = 'System'
    else:
        type = 'Finance'

    # get the direct operations and upstream supply chain proportional results from the score list
    for score in score_list:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_df = score
                scope_1_name = score.name
            if 'Source' in score.name:
                scope_3_df = score
                scope_3_name = score.name

    # aggregate to the regional-level
    scope_1_var_finance_region = aggregate_to_region_service(scope_1_df, 'sum')
    scope_3_var_finance_region = aggregate_to_region_service(scope_3_df, 'sum')

    # get banks and services
    # get a list of banks and services
    banks = np.unique(scope_3_df.reset_index()['Bank']).tolist()
    services = np.unique(scope_3_df.columns).tolist()

    # list of both directions of wilcoxon test
    stats_type = ["greater", "less"]
    # loop through both directions of wilcoxon test to test both ways
    for stat_type in stats_type:

        # create dataframes to hold the results
        index = pd.MultiIndex.from_product([banks, services], names=["Bank", "service"])
        # statistical values
        stats_test_df_scp1_vs_scp_3 = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
        # summary tables
        scope_1_vs_3_significance = pd.DataFrame(columns=services, index=banks)

        # loop through banks
        for bank in banks:
            # get values for one banks
            # upstream supply chain
            scope_3_combo_mean_one_bank = scope_3_var_finance_region.reset_index()[
                scope_3_var_finance_region.reset_index()['Bank'] == bank]
            # direct operations
            scope_1_combo_mean_one_bank = scope_1_var_finance_region.reset_index()[
                scope_1_var_finance_region.reset_index()['Bank'] == bank]

            # loop through ecosysetm services
            for service in services:
                # if there is no endogenous risk for either scope 1 or scope 3 continue to ecosystem service
                if (
                ((scope_3_combo_mean_one_bank[service].sum() == 0) and (scope_1_combo_mean_one_bank[service].sum() == 0))):
                    continue

                # compare scope 1 vs scope 3 with wilcoxon test with associated direction
                results = stats.wilcoxon(scope_1_combo_mean_one_bank[f'{service}'],
                                         scope_3_combo_mean_one_bank[f'{service}'],
                                         zero_method = "wilcox", alternative=f"{stat_type}", method="exact")

                # store the statistical values
                stats_test_df_scp1_vs_scp_3.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scp1_vs_scp_3.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # store the summary values in the summary tables
                if results[1] > 0.05:
                    scope_1_vs_3_significance[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_1_vs_3_significance[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_1_vs_3_significance[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_1_vs_3_significance[service][bank] = "***"

        # save the statistical value table and the summary table
        stats_test_df_scp1_vs_scp_3.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Scope 1 vs Scope 3/{stat_type}/GSIB {type} {scope_1_name} vs {scope_3_name} Statistics.csv')
        scope_1_vs_3_significance.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Scope 1 vs Scope 3/{stat_type}/GSIB {type} {scope_1_name} vs {scope_3_name} Significance.csv')

    return None


def statistical_tests_sector(score_list, sector_score_list, calc_type):
    """
    This function tests the portfolio- or system-level results with the sector-level results.
    :param score_list: portfolio- or system-level results list
    :param sector_score_list: sector-level results list
    :param calc_type: methodological treatment (mean, min, max)
    :return:
    """
    # check whether system-level or portfolio-level results
    if 'System' == score_list[0].reset_index()['Bank'].iloc[0]:
        type = 'System'
    else:
        type = 'Finance'

    # loop through scores in the portfolio- or system-level score list
    for score in score_list:
        # get the absolute results from the score list for direct operations, source sector and value chain
        if 'Proportional' not in score.name:
            if 'Scope 1' in score.name:
                scope_1_df = score.copy()
                scope_1_df.name = score.name
            if 'Source' in score.name:
                scope_3_source_df = score.copy()
                scope_3_source_df.name = score.name
            if 'Value Chain' in score.name:
                scope_3_value_chain_df = score.copy()
                scope_3_value_chain_df.name = score.name
    # loop through the scores in the sector-level score list
    for score in sector_score_list:
        # get the absolute results from the score list for direct operations, source sector and value chain
        if 'Proportional' not in score.name:
            if 'Scope 1' in score.name:
                scope_1_sector_df = score.copy()
                scope_1_sector_df.name = score.name
            if 'Source' in score.name:
                scope_3_source_sector_df = score.copy()
                scope_3_source_sector_df.name = score.name
            # if 'Value Chain' in score.name:
            #     scope_3_value_chain_sector_df = score.copy()
            #     scope_3_value_chain_sector_df.name = score.name

    # get banks and services
    # get a list of banks and services
    banks = np.unique(scope_3_source_df.reset_index()['Bank']).tolist()
    services = np.unique(scope_3_source_df.columns).tolist()

    # a list storing both directions of wilcoxon test
    stats_type = ['greater', 'less']
    # loop through both directions of wilcoxon test to do both
    for stat_type in stats_type:

        # test bank biting themselves compared to the economic activities biting themselves
        # create a dataframe to store the statistical values - for direct operations and upstream supply chain
        index = pd.MultiIndex.from_product([banks, services], names=["Bank", "service"])
        stats_test_df_scope_1_sector_vs_portfolio = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
        stats_test_df_scope_3_value_chain_sector_vs_portfolio = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
        stats_test_df_scope_3_source_sector_vs_portfolio = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)

        # create a dataframe to generate the summary tables - for direct operations and upstream supply chain
        scope_1_sector_vs_regular = pd.DataFrame(columns=services, index=banks)
        scope_3_source_sector_vs_regular = pd.DataFrame(columns=services, index=banks)
        scope_3_value_chain_sector_vs_regular = pd.DataFrame(columns=services, index=banks)

        # loop through the banks
        for bank in banks:
            # load sector scores for one bank
            scope_3_source_mean_sector_one_bank = scope_3_source_sector_df.reset_index()[
                scope_3_source_sector_df.reset_index()['Bank'] == bank]
            # scope_3_value_chain_mean_sector_one_bank = scope_3_value_chain_sector_df.reset_index()[
            #     scope_3_value_chain_sector_df.reset_index()['Bank'] == bank]
            scope_1_combo_mean_sector_one_bank = scope_1_sector_df.reset_index()[
                scope_1_sector_df.reset_index()['Bank'] == bank]

            # load regular scores - for one bank
            scope_3_source_mean_one_bank = scope_3_source_df.reset_index()[
                scope_3_source_df.reset_index()['Bank'] == bank]
            # scope_3_value_chain_mean_one_bank = scope_3_value_chain_df.reset_index()[
            #     scope_3_value_chain_df.reset_index()['Bank'] == bank]
            scope_1_combo_mean_one_bank = scope_1_df.reset_index()[
                scope_1_df.reset_index()['Bank'] == bank]

            # loop through ecosystem services
            for service in services:
                # scope 1 sector vs portfolio
                # if both are zero then skip
                if (
                        ((scope_1_combo_mean_one_bank[service].sum() == 0) and (
                                scope_1_combo_mean_sector_one_bank[service].sum() == 0))):
                    continue
                # scope 1 sector vs portfolio test
                results = stats.wilcoxon(scope_1_combo_mean_one_bank[f'{service}'],
                                         scope_1_combo_mean_sector_one_bank[f'{service}'],
                                         zero_method = "wilcox", alternative=f"{stat_type}", method="exact")

                # store statistical values
                stats_test_df_scope_1_sector_vs_portfolio.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scope_1_sector_vs_portfolio.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # generate summary table
                if results[1] > 0.05:
                    scope_1_sector_vs_regular[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_1_sector_vs_regular[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_1_sector_vs_regular[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_1_sector_vs_regular[service][bank] = "***"

                # scope 3 source sector vs portfolio
                # if both are zero then skip
                if (
                        ((scope_3_source_mean_one_bank[service].sum() == 0) and (
                                scope_3_source_mean_sector_one_bank[service].sum() == 0))):
                    continue

                # compare sector vs regular for upstream supply chain
                results = stats.wilcoxon(scope_3_source_mean_one_bank[f'{service}'],
                                         scope_3_source_mean_sector_one_bank[f'{service}'],
                                         zero_method = "wilcox", alternative=f"{stat_type}", method="exact")

                # store statistical values
                stats_test_df_scope_3_source_sector_vs_portfolio.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scope_3_source_sector_vs_portfolio.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # generate summary table
                if results[1] > 0.05:
                    scope_3_source_sector_vs_regular[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_3_source_sector_vs_regular[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_3_source_sector_vs_regular[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_3_source_sector_vs_regular[service][bank] = "***"

                # scope 3 value chain sector vs portfolio
                # scope 3 source sector vs portfolio
                # if (
                #         ((scope_3_value_chain_mean_one_bank[service].sum() == 0) and (
                #                 scope_3_value_chain_mean_sector_one_bank[service].sum() == 0))):
                #     continue
                #
                # # compare sector vs regular
                # results = stats.wilcoxon(scope_3_value_chain_mean_one_bank[f'{service}'],
                #                          scope_3_value_chain_mean_sector_one_bank[f'{service}'],
                #                          zero_method = "wilcox", alternative="less", method="exact")
                #
                # stats_test_df_scope_3_value_chain_sector_vs_portfolio.loc[bank, service]['statistic'] = results.statistic
                # stats_test_df_scope_3_value_chain_sector_vs_portfolio.loc[bank, service]['p-value'] = results.pvalue
                # # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic
                #
                # if results[1] > 0.05:
                #     scope_3_value_chain_sector_vs_regular[service][bank] = "NS"
                # if results[1] < 0.05 and results[1] > 0.01:
                #     scope_3_value_chain_sector_vs_regular[service][bank] = "*"
                # if results[1] <= 0.01 and results[1] > 0.005:
                #     scope_3_value_chain_sector_vs_regular[service][bank] = "**"
                # if results[1] <= 0.005:
                #     scope_3_value_chain_sector_vs_regular[service][bank] = "***"

        # save the summary table and statistical values for
        # direct operations
        stats_test_df_scope_1_sector_vs_portfolio.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Sector vs Portfolio/{stat_type}/GSIB {type} {calc_type} Scope 1 Sector vs Portfolio Statistics.csv')
        scope_1_sector_vs_regular.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Sector vs Portfolio/{stat_type}/GSIB {type} {calc_type} Scope 1 Sector vs Portfolio Significance.csv')
        # upstream supply chain
        stats_test_df_scope_3_source_sector_vs_portfolio.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Sector vs Portfolio/{stat_type}/GSIB {type} {calc_type} Scope 3 Source Sector vs Portfolio Statistics.csv')
        scope_3_source_sector_vs_regular.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Sector vs Portfolio/{stat_type}/GSIB {type} {calc_type} Scope 3 Source Sector vs Portfolio Significance.csv')

        # stats_test_df_scope_3_value_chain_sector_vs_portfolio.to_csv(
        #     f'{GSIB_value_at_risk_sig_saving_path}/GSIB {type} Scope 3 Value Chain Sector vs Portfolio Statistics.csv')
        # scope_3_value_chain_sector_vs_regular.T.to_csv(
        #     f'{GSIB_value_at_risk_sig_saving_path}/GSIB {type} Scope 3 Value Chain Sector vs Portfolio Significance.csv')

    return None


def sensitivity_calc_type(mean_list, min_list, max_list):
    """
    This function tests the difference between the different methodological treatments (mean, max, min)
    :param mean_list: scores calculated with the mean
    :param min_list: scores calculated with the minimum
    :param max_list: scores calculated with the maximum
    :return: NA
    """
    # determine whether it is a system or bank-level analysis
    if 'System' == mean_list[0].reset_index()['Bank'].iloc[0]:
        type = 'System'
    else:
        type = 'Finance'

    # get banks and services
    # get a list of banks and services
    banks = np.unique(mean_list[1].reset_index()['Bank']).tolist()
    services = np.unique(mean_list[1].columns).tolist()

    # create dataframes to store the statistical values and the summary tables
    index = pd.MultiIndex.from_product([banks, services], names=["Bank", "service"])
    # mean vs min
    stats_test_df_mean_vs_min = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
    significance_df_mean_vs_min = pd.DataFrame(columns=services, index=banks)
    # mean vs max
    stats_test_df_mean_vs_max = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
    significance_df_mean_vs_max = pd.DataFrame(columns=services, index=banks)

    # loop through the mean scores and determine the type of score (Both - endogenous risk, Impact or Dependency)
    for mean_score in mean_list:
        if 'Scope 1' in mean_score.name:
            scope_type = 'Scope 1'
            mean_score_df = mean_score.copy()
        if 'Source' in mean_score.name:
            scope_type = 'Source'
            mean_score_df = mean_score.copy()
        if 'Value Chain' in mean_score.name:
            scope_type = 'Value Chain'
            mean_score_df = mean_score.copy()
        if 'Both' not in mean_score.name:
            if 'Impact' in mean_score.name:
                score_type = 'Impact'
            if 'Dependency' in mean_score.name:
                score_type = 'Dependency'
        else:
            score_type = 'Both'

        # tests for mean vs min
        # loop through min scores
        for min_score in min_list:
            if scope_type not in min_score.name:
                continue
            min_score_df = min_score.copy()

            # loop through banks
            for bank in banks:
                # load mean score
                score_mean_sector_one_bank = mean_score_df.reset_index()[mean_score_df.reset_index()['Bank'] == bank]

                # load min score
                score_min_sector_one_bank = min_score_df.reset_index()[min_score_df.reset_index()['Bank'] == bank]

                # loop through the services
                for service in services:
                    # skip if neither score has any values
                    if (
                            ((score_mean_sector_one_bank[service].sum() == 0) and (
                                    score_min_sector_one_bank[service].sum() == 0))):
                        continue

                    # compare scope mean vs min
                    results = stats.wilcoxon(score_mean_sector_one_bank[f'{service}'],
                                             score_min_sector_one_bank[f'{service}'],
                                             zero_method = "wilcox", alternative="greater", method="exact")

                    # store statistical values into data frame
                    stats_test_df_mean_vs_min.loc[bank, service]['statistic'] = results.statistic
                    stats_test_df_mean_vs_min.loc[bank, service]['p-value'] = results.pvalue
                    # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                    # generate summary values and store into summary dataframe
                    if results[1] > 0.05:
                        significance_df_mean_vs_min[service][bank] = "NS"
                    if results[1] < 0.05 and results[1] > 0.01:
                        significance_df_mean_vs_min[service][bank] = "*"
                    if results[1] <= 0.01 and results[1] > 0.005:
                        significance_df_mean_vs_min[service][bank] = "**"
                    if results[1] <= 0.005:
                        significance_df_mean_vs_min[service][bank] = "***"

        # max vs mean
        # loop through max scores
        for max_score in max_list:
            if scope_type not in max_score.name:
                continue
            max_score_df = max_score.copy()

            # loop through banks
            for bank in banks:
                # load mean score
                score_mean_sector_one_bank = mean_score_df.reset_index()[mean_score_df.reset_index()['Bank'] == bank]

                # load min score
                score_max_sector_one_bank = max_score_df.reset_index()[max_score_df.reset_index()['Bank'] == bank]

                # loop through services
                for service in services:
                    # if netiher score has values for this ESS skip
                    if (
                            ((score_mean_sector_one_bank[service].sum() == 0) and (
                                    score_max_sector_one_bank[service].sum() == 0))):
                        continue

                    # compare mean vs max
                    results = stats.wilcoxon(score_mean_sector_one_bank[f'{service}'],
                                             score_max_sector_one_bank[f'{service}'],
                                             zero_method = "wilcox", alternative="less", method="exact")

                    # store statistical values into dataframe
                    stats_test_df_mean_vs_max.loc[bank, service]['statistic'] = results.statistic
                    stats_test_df_mean_vs_max.loc[bank, service]['p-value'] = results.pvalue
                    # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                    # generate summary values and store in summary table
                    if results[1] > 0.05:
                        significance_df_mean_vs_max[service][bank] = "NS"
                    if results[1] < 0.05 and results[1] > 0.01:
                        significance_df_mean_vs_max[service][bank] = "*"
                    if results[1] <= 0.01 and results[1] > 0.005:
                        significance_df_mean_vs_max[service][bank] = "**"
                    if results[1] <= 0.005:
                        significance_df_mean_vs_max[service][bank] = "***"


        # save mean vs min
        stats_test_df_mean_vs_min.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Mean vs max vs min/GSIB {type} {score_type} {scope_type} mean vs min Statistics.csv')
        significance_df_mean_vs_min.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Mean vs max vs min/GSIB {type} {score_type} {scope_type} mean vs min Significance.csv')

        # save mean vs max
        stats_test_df_mean_vs_max.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Mean vs max vs min/GSIB {type} {score_type} {scope_type} mean vs max Statistics.csv')
        significance_df_mean_vs_max.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Mean vs max vs min/GSIB {type} {score_type} {scope_type} mean vs max Significance.csv')

    return None


def calculate_significances(score_list_1, score_list_2, score_list_3):
    """
    This function generates the scope 1 vs scope 3 statistical tests for 3 lists of scores
    :param score_list_1: a list of scores
    :param score_list_2: a list of scores
    :param score_list_3: a list of scores
    :return:
    """
    # run the tests on each of the lists
    statistical_tests(score_list_1)
    statistical_tests(score_list_2)
    statistical_tests(score_list_3)
    return None


# calculate whether a sepcific bank has greater VaR than the average bank (summed across all)
def calculate_bank_vs_average(score_list):
    """
    This function tests whether an individual bank portfolio has greater risk than the average of all banks
    :param score_list: list of portfolio-level results
    :return: NA
    """
    # determine what kind of score (Both - endogenous risk, Impact or Dependnecy
    if 'Both' in score_list[0].name:
        folder = 'Both'
    else:
        if 'Impact' in score_list[0].name:
            folder = 'Impact'
        if 'Dependency' in score_list[0].name:
            folder = 'Dependency'

    # determine the methodological treatment of the score
    if 'mean' in score_list[0].name:
        score_type = 'mean'
    if 'min' in score_list[0].name:
        score_type = 'min'
    if 'max' in score_list[0].name:
        score_type = 'max'


    for score in score_list:
        # get the proportional score for each score type
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = f'{score.name}'
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = f'{score.name}'
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = f'{score.name}'
        # get the absolute score for each score type
        else:
            if 'Scope 1' in score.name:
                scope_1_score_abs = score.copy()
                scope_1_score_abs.name = f'{score.name}'
            if 'Source' in score.name:
                scope_3_score_source_abs = score.copy()
                scope_3_score_source_abs.name = f'{score.name}'
            if 'Value Chain' in score.name:
                scope_3_score_rows_abs = score.copy()
                scope_3_score_rows_abs.name = f'{score.name}'

    # get the bank-level financial data - formatted
    financial_data_df = finance_GSIB_reformat()
    # get system total
    system_total = financial_data_df['EUR m adjusted'].sum()

    # scope 1 - get the system-level average
    df = scope_1_score_abs.copy()
    score_name = scope_1_score_abs.name
    prop_df = df.reset_index().drop(columns=['Bank']).groupby(['sector', 'region']).sum()
    prop_df = (prop_df / system_total) * 100
    prop_df.name = score_name
    scope_1_score_abs_sector_prop = prop_df

    # scope 3 - get system level average
    df = scope_3_score_source_abs.copy()
    score_name = scope_3_score_source_abs.name
    prop_df = df.reset_index().drop(columns=['Bank']).groupby(['sector', 'region']).sum()
    prop_df = (prop_df / system_total) * 100
    prop_df.name = score_name
    scope_3_score_source_abs_sector_prop = prop_df

    # compare to see whether scope 1 bank 1 is greater than scope 1 for system (proportionally):

    # get banks and services
    # get a list of banks and services
    banks = np.unique(score_list[0].reset_index()['Bank']).tolist()
    services = np.unique(score_list[0].columns).tolist()

    # two directions of wilcoxon test
    stats_type = ["greater", "less"]
    # run the statistical tests in both directions
    for stat_type in stats_type:

        # create the dataframes to store the results
        index = pd.MultiIndex.from_product([banks, services], names=["Bank", "service"])
        # statistical values
        stats_test_df_scope_1 = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
        stats_test_df_scope_3 = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)
        # significance values
        scope_1_significance = pd.DataFrame(columns=services, index=banks)
        scope_3_significance = pd.DataFrame(columns=services, index=banks)

        # loop through the banks
        for bank in banks:
            # get the scope 3 values for the bank
            scope_3_combo_mean_one_bank = scope_3_score_source.reset_index()[
                scope_3_score_source.reset_index()['Bank'] == bank]
            # get the score 1 values for the bank
            scope_1_combo_mean_one_bank = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank]

            # loop through the services
            for service in services:
                # scope 1
                # if both scores have no non-zero values then skip
                if (
                        ((scope_1_score_abs_sector_prop[service].sum() == 0) and (
                                scope_1_combo_mean_one_bank[service].sum() == 0))):
                    continue

                # test the bank compared to the average
                results = stats.wilcoxon(scope_1_combo_mean_one_bank[f'{service}'],
                                         scope_1_score_abs_sector_prop[f'{service}'],
                                         zero_method="wilcox", alternative=f"{stat_type}", method="exact")

                # store the statistical values
                stats_test_df_scope_1.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scope_1.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # generate and store the summary values
                if results[1] > 0.05:
                    scope_1_significance[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_1_significance[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_1_significance[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_1_significance[service][bank] = "***"


                # scope 3
                # if netiher score has no non-zero values then skip
                if (
                        ((scope_3_score_source_abs_sector_prop[service].sum() == 0) and (
                                scope_3_combo_mean_one_bank[service].sum() == 0))):
                    continue

                # compare bank to average
                results = stats.wilcoxon(scope_3_combo_mean_one_bank[f'{service}'],
                                         scope_3_score_source_abs_sector_prop[f'{service}'],
                                         zero_method="wilcox", alternative=f"{stat_type}", method="exact")

                # store the statistical values
                stats_test_df_scope_3.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scope_3.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # generate and store the summary values
                if results[1] > 0.05:
                    scope_3_significance[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_3_significance[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_3_significance[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_3_significance[service][bank] = "***"

        # save the results - summary and statistical values
        stats_test_df_scope_1.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Bank vs Average/{stat_type}/GSIB Finance Scope 1 {folder} {score_type} Bank vs Average System Statistics.csv')
        scope_1_significance.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Bank vs Average/{stat_type}/GSIB Finance Scope 1 {folder} {score_type} Bank vs Average System Significance.csv')

        stats_test_df_scope_3.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Bank vs Average/{stat_type}/GSIB Finance Scope 3 Source {folder} {score_type} Bank vs Average System Statistics.csv')
        scope_3_significance.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/Bank vs Average/{stat_type}/GSIB Finance Scope 3 Source {folder} {score_type} Bank vs Average System Significance.csv')


    return None


def calculate_portfolio_vs_system(system_score, bank_level_score) :
    """
    This function compares the portfolio-level risk to the system-level risk
    :param system_score: the system-level risk values - generated by treating all banks as one portfolio (system)
    :param bank_level_score: bank portfolio-level risk values
    :return: NA
    """
    # determine the score type (Both - endogenous risk, impact, dependency)
    if 'Both' in bank_level_score[0].name:
        folder = 'Both'
    else:
        if 'Impact' in bank_level_score[0].name:
            folder = 'Impact'
        if 'Dependency' in bank_level_score[0].name:
            folder = 'Dependency'

    # determine the methodological treatment
    if 'mean' in bank_level_score[0].name:
        score_type = 'mean'
    if 'min' in bank_level_score[0].name:
        score_type = 'min'
    if 'max' in bank_level_score[0].name:
        score_type = 'max'

    # loop through system scores
    for score in system_score:
        # get proportional scores for each type
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                system_scope_1_score = score.copy()
                system_scope_1_score.name = f'System {score.name}'
            if 'Source' in score.name:
                system_scope_3_score_source = score.copy()
                system_scope_3_score_source.name = f'System {score.name}'
            if 'Value Chain' in score.name:
                system_scope_3_score_rows = score.copy()
                system_scope_3_score_rows.name = f'System {score.name}'
        # get absolute scores for each type
        else:
            if 'Scope 1' in score.name:
                system_scope_1_score_abs = score.copy()
                system_scope_1_score_abs.name = f'System {score.name}'
            if 'Source' in score.name:
                system_scope_3_score_source_abs = score.copy()
                system_scope_3_score_source_abs.name = f'System {score.name}'
            if 'Value Chain' in score.name:
                system_scope_3_score_rows_abs = score.copy()
                system_scope_3_score_rows_abs.name = f'System {score.name}'

    # loop through portfolio-level scores
    for score in bank_level_score:
        # get proportional scores for each type
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = f'{score.name}'
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = f'{score.name}'
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = f'{score.name}'
        # get absolute scores for each type
        else:
            if 'Scope 1' in score.name:
                scope_1_score_abs = score.copy()
                scope_1_score_abs.name = f'{score.name}'
            if 'Source' in score.name:
                scope_3_score_source_abs = score.copy()
                scope_3_score_source_abs.name = f'{score.name}'
            if 'Value Chain' in score.name:
                scope_3_score_rows_abs = score.copy()
                scope_3_score_rows_abs.name = f'{score.name}'

    # list of banks
    banks = np.unique(scope_3_score_source.reset_index()['Bank']).tolist()
    # create lists of relevant banks to test - adding portfolio system (average) less and greater to test both
    banks_for_storage = banks.copy()
    banks_for_storage.append('Portfolio System greater')
    banks_for_storage.append('Portfolio System less')
    banks.append('Portfolio System')

    # list of ecosystem services
    services = np.unique(scope_3_score_source.columns).tolist()

    # both directions of the wilcoxon test
    stats_type = ["greater", "less"]

    # run the tests in both directions by looping through both directions
    for stat_type in stats_type:
        # test bank biting themselves compared to the system biting itself

        # create dataframes to store the statistical values and summary tables
        index = pd.MultiIndex.from_product([banks_for_storage, services], names=["Bank", "service"])
        stats_test_df_scope_1_system_vs_portfolio = pd.DataFrame(columns=['statistic', 'p-value', 'z'], index=index)

        stats_test_df_scope_3_source_system_vs_portfolio = pd.DataFrame(columns=['statistic', 'p-value', 'z'],
                                                                        index=index)
        scope_1_system_vs_regular = pd.DataFrame(columns=services, index=banks_for_storage)
        scope_3_system_sector_vs_regular = pd.DataFrame(columns=services, index=banks_for_storage)

        # loop through the banks
        for bank in banks:
            # skip the portfolio system "bank"
            if bank == 'Portfolio System':
                continue

            # load bank portfolio scores
            scope_3_source_mean_score_one_bank = scope_3_score_source.reset_index()[
                scope_3_score_source.reset_index()['Bank'] == bank]

            # scope_3_value_chain_mean_sector_one_bank = scope_3_value_chain_sector_df.reset_index()[
            #     scope_3_value_chain_sector_df.reset_index()['Bank'] == bank]

            scope_1_combo_mean_score_one_bank = scope_3_score_source.reset_index()[
                scope_3_score_source.reset_index()['Bank'] == bank]

            # load system scores
            scope_3_source_mean_system = system_scope_3_score_source.reset_index()[
                system_scope_3_score_source.reset_index()['Bank'] == 'System']

            # scope_3_value_chain_mean_one_bank = scope_3_value_chain_df.reset_index()[
            #     scope_3_value_chain_df.reset_index()['Bank'] == bank]

            scope_1_combo_mean_system = system_scope_1_score.reset_index()[
                system_scope_1_score.reset_index()['Bank'] == 'System']

            # loop through ecosystem services
            for service in services:
                # scope 1 portfolio vs system
                # if neither score has no non-zero values then skip
                if (
                        ((scope_1_combo_mean_score_one_bank[service].sum() == 0) and (
                                scope_1_combo_mean_system[service].sum() == 0))):
                    continue

                # compare system vs portfolio
                results = stats.wilcoxon(scope_1_combo_mean_system[f'{service}'],
                                         scope_1_combo_mean_score_one_bank[f'{service}'],
                                         zero_method="wilcox", alternative=f"{stat_type}", method="exact")

                # store statistical values
                stats_test_df_scope_1_system_vs_portfolio.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scope_1_system_vs_portfolio.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # generate and store summary values
                if results[1] > 0.05:
                    scope_1_system_vs_regular[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_1_system_vs_regular[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_1_system_vs_regular[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_1_system_vs_regular[service][bank] = "***"

                # scope 3 source system vs portfolio
                # if netiher score has no non-zero values then skip
                if (
                        ((scope_3_source_mean_score_one_bank[service].sum() == 0) and (
                                scope_3_source_mean_system[service].sum() == 0))):
                    continue

                # compare system vs portfolio
                results = stats.wilcoxon(scope_3_source_mean_system[f'{service}'],
                                         scope_3_source_mean_score_one_bank[f'{service}'],
                                         zero_method="wilcox", alternative=f"{stat_type}", method="exact")

                # store statistical values
                stats_test_df_scope_3_source_system_vs_portfolio.loc[bank, service]['statistic'] = results.statistic
                stats_test_df_scope_3_source_system_vs_portfolio.loc[bank, service]['p-value'] = results.pvalue
                # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

                # generate and store summary values
                if results[1] > 0.05:
                    scope_3_system_sector_vs_regular[service][bank] = "NS"
                if results[1] < 0.05 and results[1] > 0.01:
                    scope_3_system_sector_vs_regular[service][bank] = "*"
                if results[1] <= 0.01 and results[1] > 0.005:
                    scope_3_system_sector_vs_regular[service][bank] = "**"
                if results[1] <= 0.005:
                    scope_3_system_sector_vs_regular[service][bank] = "***"

        # get financial data
        financial_data_df = finance_GSIB_reformat()
        # get system total
        system_total = financial_data_df['EUR m adjusted'].sum()

        # scope 1 - generate average bank portfolio (system portfolio)
        df = scope_1_score_abs.copy()
        score_name = scope_1_score_abs.name
        prop_df = df.reset_index().drop(columns=['Bank']).groupby(['sector', 'region']).sum()
        prop_df = (prop_df / system_total) * 100
        prop_df.name = score_name
        scope_1_score_abs_sector_prop = prop_df

        # scope 3 - generate average bank portfolio (system portfolio)
        df = scope_3_score_source_abs.copy()
        score_name = scope_3_score_source_abs.name
        prop_df = df.reset_index().drop(columns=['Bank']).groupby(['sector', 'region']).sum()
        prop_df = (prop_df / system_total) * 100
        prop_df.name = score_name
        scope_3_score_source_abs_sector_prop = prop_df

        # compare to see whether system is greater than for average bank (proportionally):
        # get banks and services
        # get a list of banks and services
        bank = 'Portfolio System'

        # loop through ecosystem services
        for service in services:
            # GREATER
            # scope 1
            # if neither has no non-zero values - skip
            if (
                    ((scope_1_score_abs_sector_prop[service].sum() == 0) and (
                            system_scope_1_score[service].sum() == 0))):
                continue

            # compare bank average vs system - greater
            results = stats.wilcoxon(system_scope_1_score[f'{service}'],
                                     scope_1_score_abs_sector_prop[f'{service}'],
                                     zero_method="wilcox", alternative="greater", method="exact")

            # store statistical values
            stats_test_df_scope_1_system_vs_portfolio.loc[f'{bank} greater', service]['statistic'] = results.statistic
            stats_test_df_scope_1_system_vs_portfolio.loc[f'{bank} greater', service]['p-value'] = results.pvalue
            # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

            # generate and store summary values
            if results[1] > 0.05:
                scope_1_system_vs_regular[service][f'{bank} greater'] = "NS"
            if results[1] < 0.05 and results[1] > 0.01:
                scope_1_system_vs_regular[service][f'{bank} greater'] = "*"
            if results[1] <= 0.01 and results[1] > 0.005:
                scope_1_system_vs_regular[service][f'{bank} greater'] = "**"
            if results[1] <= 0.005:
                scope_1_system_vs_regular[service][f'{bank} greater'] = "***"


            # scope 3
            # if neither has no non-zero values then skip
            if (
                    ((scope_3_score_source_abs_sector_prop[service].sum() == 0) and (
                            system_scope_3_score_source[service].sum() == 0))):
                continue

            # compare average bank to system - greater
            results = stats.wilcoxon(system_scope_3_score_source[f'{service}'],
                                     scope_3_score_source_abs_sector_prop[f'{service}'],
                                     zero_method="wilcox", alternative="greater", method="exact")

            # store statistical values
            stats_test_df_scope_3_source_system_vs_portfolio.loc[f'{bank} greater', service]['statistic'] = results.statistic
            stats_test_df_scope_3_source_system_vs_portfolio.loc[f'{bank} greater', service]['p-value'] = results.pvalue
            # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

            # generate and store summary values
            if results[1] > 0.05:
                scope_3_system_sector_vs_regular[service][f'{bank} greater'] = "NS"
            if results[1] < 0.05 and results[1] > 0.01:
                scope_3_system_sector_vs_regular[service][f'{bank} greater'] = "*"
            if results[1] <= 0.01 and results[1] > 0.005:
                scope_3_system_sector_vs_regular[service][f'{bank} greater'] = "**"
            if results[1] <= 0.005:
                scope_3_system_sector_vs_regular[service][f'{bank} greater'] = "***"

            # TEST IF LESS ####################################################################
            # scope 1
            # if neither has no non-zero values - skip
            if (
                    ((scope_1_score_abs_sector_prop[service].sum() == 0) and (
                            system_scope_1_score[service].sum() == 0))):
                continue

            # compare bank average to system - less
            results = stats.wilcoxon(system_scope_1_score[f'{service}'],
                                     scope_1_score_abs_sector_prop[f'{service}'],
                                     zero_method="wilcox", alternative="less", method="exact")

            # store statistical values
            stats_test_df_scope_1_system_vs_portfolio.loc[f'{bank} less', service]['statistic'] = results.statistic
            stats_test_df_scope_1_system_vs_portfolio.loc[f'{bank} less', service]['p-value'] = results.pvalue
            # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

            # generate and store summary values
            if results[1] > 0.05:
                scope_1_system_vs_regular[service][f'{bank} less'] = "NS"
            if results[1] < 0.05 and results[1] > 0.01:
                scope_1_system_vs_regular[service][f'{bank} less'] = "*"
            if results[1] <= 0.01 and results[1] > 0.005:
                scope_1_system_vs_regular[service][f'{bank} less'] = "**"
            if results[1] <= 0.005:
                scope_1_system_vs_regular[service][f'{bank} less'] = "***"

            # scope 3 - less
            # if neither score has no non-zero values - skip
            if (
                    ((scope_3_score_source_abs_sector_prop[service].sum() == 0) and (
                            system_scope_3_score_source[service].sum() == 0))):
                continue

            # compare bank average vs system - less
            results = stats.wilcoxon(system_scope_3_score_source[f'{service}'],
                                     scope_3_score_source_abs_sector_prop[f'{service}'],
                                     zero_method="wilcox", alternative="less", method="exact")

            # store statistical values
            stats_test_df_scope_3_source_system_vs_portfolio.loc[f'{bank} less', service][
                'statistic'] = results.statistic
            stats_test_df_scope_3_source_system_vs_portfolio.loc[f'{bank} less', service]['p-value'] = results.pvalue
            # stats_test_df_scp1_vs_scp_3.loc[bank, service]['z'] = results.zstatistic

            # generate and store summary values
            if results[1] > 0.05:
                scope_3_system_sector_vs_regular[service][f'{bank} less'] = "NS"
            if results[1] < 0.05 and results[1] > 0.01:
                scope_3_system_sector_vs_regular[service][f'{bank} less'] = "*"
            if results[1] <= 0.01 and results[1] > 0.005:
                scope_3_system_sector_vs_regular[service][f'{bank} less'] = "**"
            if results[1] <= 0.005:
                scope_3_system_sector_vs_regular[service][f'{bank} less'] = "***"


        # save results
        stats_test_df_scope_1_system_vs_portfolio.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/System vs Portfolio/{stat_type}/GSIB Scope 1 {folder} {score_type} Portfolio vs System Statistics.csv')
        scope_1_system_vs_regular.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/System vs Portfolio/{stat_type}/GSIB Scope 1 {folder} {score_type} Portfolio vs System Significance.csv')

        stats_test_df_scope_3_source_system_vs_portfolio.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/System vs Portfolio/{stat_type}/GSIB Scope 3 Source {folder} {score_type} Portfolio vs System Statistics.csv')
        scope_3_system_sector_vs_regular.T.to_csv(
            f'{GSIB_value_at_risk_sig_saving_path}/System vs Portfolio/{stat_type}/GSIB Scope 3 Source {folder} {score_type} Portfolio vs System Significance.csv')

    return None


def plot_bar_chart(both_mean, both_min, both_max):
    """
    This function plots bar charts of the direct operations and upstream supply chain results with error bars from the
    three methodological treatments (mean, min, max) for each bank.
    :param both_mean: score with mean methodological treatment
    :param both_min: score with min methodological treatment
    :param both_max: score with max methodological treatment
    :return: NA
    """
    # determine what type of score (both -endogenous risk, impact, dependency)
    if 'Both' in both_mean[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in both_mean[0].name:
            folder = 'Impact'
        if 'Dependency' in both_mean[0].name:
            folder = 'Dependency'

    # create lists to store the types of score
    scope_1_var_finance = []
    scope_3_var_finance = []
    scope_1_var_finance_abs = []
    scope_3_var_finance_abs = []
    # loop through mean scores and sort into respective lists
    for score in both_mean:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance_abs.append(score)
    # loop through min scores and sort into respective lists
    for score in both_min:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance_abs.append(score)
    # loop through max scores and sort into respective lists
    for score in both_max:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance_abs.append(score)

    # convert to region
    scope_1_var_finance_region = aggregate_to_region_service(scope_1_var_finance, 'sum')
    scope_3_var_finance_region = aggregate_to_region_service(scope_3_var_finance, 'sum')
    scope_1_var_finance_region_abs = aggregate_to_region_service(scope_1_var_finance_abs, 'sum')
    scope_3_var_finance_region_abs = aggregate_to_region_service(scope_3_var_finance_abs, 'sum')

    # get banks and services
    # get a list of banks and services
    banks = np.unique(scope_1_var_finance[0].reset_index()['Bank']).tolist()
    services = np.unique(scope_1_var_finance[0].columns).tolist()

    # create a plot of scope 1 and scope 3 value at risk for all the banks and all the portfolio
    # with the max and min values as the error bars
    score_types = ['mean', 'min', 'max']

    # get regions
    regions = ['North America', 'Europe', 'Asia']
    bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    # cycle through domicile regions for the banks
    for region in regions:
        # var finance
        plt.figure(figsize=(20, 20))

        # plt.subplots(4,2,sharey=True)
        num_banks = bank_regions_df['Region'].value_counts()[region]
        i = 0
        # loop through the banks
        for bank in banks:
            # if the bank is not in the current region - skip
            if not bank_regions_dict.get(bank) == region:
                continue
            # create the subplot so all banks share the same y and x axes
            i = i + 1
            if (i == 1):
                ax = plt.subplot(math.ceil(num_banks/2) + 1, 2, i)
            if (i != 1):
                ax = plt.subplot(math.ceil(num_banks/2) + 1, 2, i, sharey=ax, sharex=ax)
            # loop through the scope 1 scores (proportional)
            for sheet in scope_1_var_finance:
                df = sheet.copy()
                # get score for one bank
                one_bank_df = df.reset_index()[df.reset_index()['Bank'] == bank]
                mydict = {}
                # loop through ecosystem services
                for service in services:
                    # sum the total risk for the service for the bank
                    var = np.sum(one_bank_df[service])
                    mydict[service] = var
                # store the dict by methodological treatment
                if re.search('min', sheet.name):
                    scope_1_min_values = mydict
                if re.search('mean', sheet.name):
                    scope_1_mean_values = mydict
                if re.search('max', sheet.name):
                    scope_1_max_values = mydict

            # create the x axis ticks for the ecosystem services
            X_axis = np.arange(len(services))

            # scope 1 - store the values by methodological treatment for plotting
            scope_1_mean = np.array(list(scope_1_mean_values.values()))
            scope_1_min = np.array(list(scope_1_min_values.values()))
            scope_1_max = np.array(list(scope_1_max_values.values()))

            # calculate the error bars lengths with the min and max methodological treatments
            lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
            higher_err_scope_1 = (scope_1_max) - (scope_1_mean)

            # store the error bar values for plotting
            asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

            # plot the scope 1 bar in the chart based on whether it is endogenous (overlap) or impact/dependency
            if folder == 'Overlap':
                ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
            else:
                ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')

            # plot the error bars
            ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')


            # scope 3
            # loop through scope 3 scores
            for sheet in scope_3_var_finance:
                df = sheet.copy()
                # get the data for one bank
                one_bank_df = df.reset_index()[df.reset_index()['Bank'] == bank]
                mydict = {}
                # loop through the services
                for service in services:
                    # get the sum of the risk for the service for the bank
                    var = np.sum(one_bank_df[service])
                    mydict[service] = var
                # store the services total values according to the methodological treatment
                if re.search('min', sheet.name):
                    scope_3_min_values = mydict
                if re.search('mean', sheet.name):
                    scope_3_mean_values = mydict
                if re.search('max', sheet.name):
                    scope_3_max_values = mydict

            # scope 3 - store the values for plotting
            scope_3_mean = np.array(list(scope_3_mean_values.values()))
            scope_3_min = np.array(list(scope_3_min_values.values()))
            scope_3_max = np.array(list(scope_3_max_values.values()))

            # generate the values for the error bars
            lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
            higher_err_scope_3 = (scope_3_max) - (scope_3_mean)

            # format the values for the error bars for plotting
            asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

            # plot the scope 3 bar in the chart based on whether it is endogenous (overlap) or impact/dependency
            if folder == 'Overlap':
                ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Direct Operations')
            else:
                ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
            ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')
            # for item in ([ax.title, ax.xaxis.label] +
            #              ax.get_xticklabels() ):
            #     item.set_fontsize(20)

            # add legend
            ax.legend()

            # set title for the subplot based on score type
            if folder == 'Overlap':
                ax.set_title(f'{bank} \n Endogenous Risk Exposure for Direct Operations and Upstream Supply Chain')
            else:
                ax.set_title(f'{bank} \n {folder} Value at Risk for Direct Operations and Upstream Supply Chain')

            # set the x-ticks as ecosystem services
            ax.set_xticks(X_axis, services, rotation=45, ha='right')
            # only display the x labels for the bottom subplots
            if i < (num_banks):
                ax.tick_params(labelbottom=False)

        # compare with system-level
        # get the financial data
        financial_data_df = finance_GSIB_reformat()
        # get the system total
        system_total = financial_data_df['EUR m adjusted'].sum()
        # create lists to store the system (bank average) scores
        scope_1_var_finance_region_system = []
        scope_3_var_finance_region_system = []
        # scope 1
        # loop through the absolute score values
        for score in scope_1_var_finance_region_abs:
            df = score.copy()
            score_name = score.name
            # convert to proportion for the system (bank average)
            prop_df = df.reset_index().drop(columns='Bank').groupby('region').sum()
            prop_df = (prop_df / system_total) * 100
            prop_df.name = score_name
            # store the proportional score for bank average
            scope_1_var_finance_region_system.append(prop_df)
        # scope 3
        # loop through the absolute score values
        for score in scope_3_var_finance_abs:
            df = score.copy()
            score_name = score.name
            # convert to proportion for the system (bank average)
            prop_df = df.reset_index().drop(columns=['Bank', 'sector']).groupby('region').sum()
            prop_df = (prop_df / system_total) * 100
            prop_df.name = score_name
            # store the proportional score for bank average
            scope_3_var_finance_region_system.append(prop_df)

        # i = 0
        # create a new subplot for the bank average plot
        i = i + 1
        ax = plt.subplot(math.ceil(num_banks/2) + 1, 2, i, sharey=ax, sharex=ax)
        # loop through the scope 1 bank average scores
        for sheet in scope_1_var_finance_region_system:
            df = sheet.copy()
            one_bank_df = df.reset_index()
            mydict = {}
            # loop through ecosystem services
            for service in services:
                # get total risk for ecosystem service for bank average
                var = np.sum(one_bank_df[service])
                mydict[service] = var
            # store the values according to relevant methodological treatment
            if re.search('min', sheet.name):
                scope_1_min_values = mydict
            if re.search('mean', sheet.name):
                scope_1_mean_values = mydict
            if re.search('max', sheet.name):
                scope_1_max_values = mydict

        # create the x axis ticks for the ecosystem services
        X_axis = np.arange(len(services))

        # scope 1 - store the score values for plotting
        scope_1_mean = np.array(list(scope_1_mean_values.values()))
        scope_1_min = np.array(list(scope_1_min_values.values()))
        scope_1_max = np.array(list(scope_1_max_values.values()))

        # generate the error bar values
        lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
        higher_err_scope_1 = (scope_1_max) - (scope_1_mean)
        # store the error bar values for plotting
        asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

        # plot the bar chart according to score type (overlap- endogenous risk, impact, dependency)
        if folder == 'Overlap':
            ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
        else:
            ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
        # plot the error bars
        ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')

        # loop through the scope 3 bank average scores
        for sheet in scope_3_var_finance_region_system:
            df = sheet.copy()
            one_bank_df = df.reset_index()
            mydict = {}
            # loop through ecosystem services
            for service in services:
                # get the total risk for ecosystem service for bank average
                var = np.sum(one_bank_df[service])
                mydict[service] = var
            # store the values by methodological treatment
            if re.search('min', sheet.name):
                scope_3_min_values = mydict
            if re.search('mean', sheet.name):
                scope_3_mean_values = mydict
            if re.search('max', sheet.name):
                scope_3_max_values = mydict

        # scope 3 - store the scores for plotting
        scope_3_mean = np.array(list(scope_3_mean_values.values()))
        scope_3_min = np.array(list(scope_3_min_values.values()))
        scope_3_max = np.array(list(scope_3_max_values.values()))

        # generate the error bar values
        lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
        higher_err_scope_3 = (scope_3_max) - (scope_3_mean)
        # store the error bar value for plotting
        asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

        # plot the bar chart according to the score type (overlap-endogenous risk, impact, dependency)
        if folder == 'Overlap':
            ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
        else:
            ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
        # plot the error bars
        ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')

        # for item in ([ax.title, ax.xaxis.label] +
        #              ax.get_xticklabels()):
        #     item.set_fontsize(20)

        # add legend
        ax.legend()
        # set x-ticks the ecosystem services
        ax.set_xticks(X_axis, services, rotation=45, ha='right')
        ax.tick_params(axis='x', labelbottom=True)

        # title the subplot based on the score type
        if folder == 'Overlap':
            ax.set_title(f' System-level Endogenous Risk Exposure for Direction Operations and Upstream Supply Chain')
        else:
            ax.set_title(f' System-level {folder} Value at Risk for Direction Operations and Upstream Supply Chain')
        # adjust the plots to fit all the plots
        plt.tight_layout()
        # add enough space for the labels
        plt.subplots_adjust(hspace=0.2)
        # save the figure
        plt.savefig(
            f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank-level Finance {folder} Value at Risk for Banks with Error Bars Percentage')
        plt.close()

    return None


def plot_heatmap_system_level(score_list):
    """
    This function plots heatmaps at the system level for sector and region risk exposure for direct operations and
    upstream supply chain.
    :param score_list: list of scores with direct operations and upstream supply chains results (source and value chain)
    :return: NA
    """
    # determine which score type - both (endogenous risk), impact, dependency
    if 'Both' in score_list[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in score_list[0].name:
            folder = 'Impact'
        if 'Dependency' in score_list[0].name:
            folder = 'Dependency'

    # loop through the score list and get the relevant results
    for score in score_list:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = f'{score.name}'
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = f'{score.name}'
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = f'{score.name}'
        else:
            if 'Scope 1' in score.name:
                scope_1_score_abs = score.copy()
                scope_1_score_abs.name = f'{score.name}'
            if 'Source' in score.name:
                scope_3_score_source_abs = score.copy()
                scope_3_score_source_abs.name = f'{score.name}'
            if 'Value Chain' in score.name:
                scope_3_score_rows_abs = score.copy()
                scope_3_score_rows_abs.name = f'{score.name}'

    # get financial data
    financial_data_df = finance_GSIB_reformat()
    # get total value of portfolios across all banks
    system_total = financial_data_df['EUR m adjusted'].sum()

    # SECTOR
    # create the system-level (bank average) score by sector
    # scope 1 - direct operations
    df = scope_1_score_abs.copy()
    score_name = scope_1_score_abs.name
    prop_df = df.reset_index().drop(columns=['Bank', 'region']).groupby('sector').sum()
    prop_df = (prop_df / system_total) * 100
    prop_df.name = score_name
    scope_1_score_abs_sector_prop = prop_df

    # scope 3 - upstream supply cahin
    df = scope_3_score_source_abs.copy()
    score_name = scope_3_score_source_abs.name
    prop_df = df.reset_index().drop(columns=['Bank', 'region']).groupby('sector').sum()
    prop_df = (prop_df / system_total) * 100
    prop_df.name = score_name
    scope_3_score_source_abs_sector_prop = prop_df

    # create the color palette for plots
    colors = sns.color_palette("Reds", as_cmap=True)

    # create a list to the store the results
    scores_list = []

    # scope 1 - direct operations
    # format and store the score in list
    imp_dep_bank_scp_1 = scope_1_score_abs_sector_prop.reset_index().rename(columns={'sector': 'Sector'}).set_index(
        ['Sector'])
    imp_dep_bank_scp_1.name = 'System-level Direct Operations'
    scores_list.append(imp_dep_bank_scp_1)
    # scope 3 - upstream supply chain
    # format and store score in list
    imp_dep_bank_scp_3 = scope_3_score_source_abs_sector_prop.reset_index().rename(
        columns={'sector': 'Sector'}).set_index(['Sector'])
    imp_dep_bank_scp_3.name = 'System-level Upstream Supply Chain'
    scores_list.append(imp_dep_bank_scp_3)

    # get converter for NACE sector summarization
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])

    # create list to store the NACE converted scores
    NACE_score_list = []
    # loop through the scope 1 and scope 3 system-level scores
    for score in scores_list:
        name = score.name
        # merge the NACE converter and the score and group the sectors by their higher-level classification for
        # display purposes
        NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(['Code']).sum()
        NACE_score = NACE_score.rename(columns={'Code':'Sector'})
        NACE_score.name = name
        # append to storage list
        NACE_score_list.append(NACE_score)

    # plot a heatmap for each
    # create plot
    i = 1
    plt.figure(figsize=(20, 20))
    # loop through NACE converted list
    for score in NACE_score_list:
        # create subplot
        ax = plt.subplot(2, 2, i)
        # assign colors
        color_scheme = colors
        # plot the heatmap
        sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False)
        # set title and the label size
        ax.set_title(f'{score.name}')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        i = i + 1
    # plt.suptitle(f'System-level Impact Feedback Intensity at the Sectoral Level', y=0.99)
    # plt.tight_layout()
    # plt.savefig(f'{value_at_risk_figure_saving_path}/Sector/System Sector Percentage Value at Risk')
    # plt.show()

    # REGION
    # create a list to store the results
    score_list = []

    # scope 1 and scope 3
    # set the color scheme for the plots
    colors = sns.color_palette("Reds", as_cmap=True)

    # direct operations - generate the system-level proportional results for region
    df = scope_1_score_abs.copy()
    score_name = scope_1_score_abs.name
    prop_df = df.reset_index().drop(columns=['Bank', 'sector']).groupby('region').sum()
    prop_df = (prop_df / system_total) * 100
    prop_df.name = score_name
    scope_1_score_abs_region_prop = prop_df

    # upstream supply chain - generate the system-level proportional results for region
    df = scope_3_score_source_abs.copy()
    score_name = scope_3_score_source_abs.name
    prop_df = df.reset_index().drop(columns=['Bank', 'sector']).groupby('region').sum()
    prop_df = (prop_df / system_total) * 100
    prop_df.name = score_name
    scope_3_score_source_abs_region_prop = prop_df

    # format results for plotting
    scope_1_overlap_one_bank = scope_1_score_abs_region_prop.reset_index().rename(
        columns={'region': 'Region'}).set_index(
        ['Region'])
    scope_1_overlap_one_bank.name = 'System-level Direct Operations'
    scope_3_overlap_one_bank = scope_3_score_source_abs_region_prop.reset_index().rename(
        columns={'region': 'Region'}).set_index(
        ['Region'])
    scope_3_overlap_one_bank.name = 'System-level Upstream Supply Chain'
    # append regional level scores to list
    score_list.append(scope_1_overlap_one_bank)
    score_list.append(scope_3_overlap_one_bank)

    # plot the heatmaps
    # loop through the region scores
    for score in score_list:
        # create subplot
        ax = plt.subplot(2, 2, i)

        # ax.set_title(f'System {score.name}')
        # if (i < 4):
        #     sns.heatmap(score.T, cmap=colors, ax=ax, xticklabels=False)
        # else:
        # plot heatmap
        sns.heatmap(score, cmap=colors, ax=ax, xticklabels=True)
        i = i + 1
        # set label size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels()):
            item.set_fontsize(20)
    # plt.suptitle(f'System-level Impact Feedback Intensity at the Country and Sectoral Level', y=0.99)
    # adjust the layout to fit figures
    plt.tight_layout()
    # save figure
    plt.savefig(f'{value_at_risk_figure_saving_path}/{folder}/System VaR Sectoral and Region Level {folder} Heatmap')
    plt.close()

    return None


def plot_heatmap_bank_level(score_list):
    """
    This function plots heatmaps at the bank portfolio-level for region and sector for upstream supply chain and direct
    operations separated by domicile region for the banks
    :param score_list: score list containing the direct operations and upstream supply chain results
    :return: NA
    """
    # determine the score type
    if 'Both' in score_list[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in score_list[0].name:
            folder = 'Impact'
        if 'Dependency' in score_list[0].name:
            folder = 'Dependency'
    # store the relevant results from the list
    for score in score_list:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = score.name
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = score.name
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = score.name

    # create the color palette for the plot
    colors = sns.color_palette("Reds", as_cmap=True)

    # list of banks
    banks = np.unique(scope_1_score.reset_index()['Bank'])

    # get regions for domicile of the banks
    regions = ['North America', 'Europe', 'Asia']
    bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    # get converter for NACE sector summarization
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])

    # loop through bank domicile regions
    for region in regions:
        # var finance
        # create figure
        plt.figure(figsize=(20, 20))

        # get number of banks in region
        num_banks = bank_regions_df['Region'].value_counts()[region]

        i = 1
        # loop through the banks
        for bank in banks:
            # if the bank is not the region - skip
            if not bank_regions_dict.get(bank) == region:
                continue
            # create list to store the results
            scores_list = []
            # scope 1 - direct operations - get score for one bank and group by sector
            imp_dep_bank_scp_1 = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank].drop(columns=['Bank', 'region']).groupby(['sector']).sum()
            imp_dep_bank_scp_1 = imp_dep_bank_scp_1.reset_index().rename(columns={'sector': 'Sector'}).set_index(['Sector'])
            imp_dep_bank_scp_1.name = f'Direct Operations'
            scores_list.append(imp_dep_bank_scp_1)

            # scope 3 - upstream supply chain - get score for one bank and group by sector
            imp_dep_bank_scp_3 = scope_3_score_source.reset_index()[
                scope_3_score_source.reset_index()['Bank'] == bank].drop(columns=['Bank', 'region']).groupby(
                ['sector']).sum()
            imp_dep_bank_scp_3 = imp_dep_bank_scp_3.reset_index().rename(columns={'sector': 'Sector'}).set_index(['Sector'])
            imp_dep_bank_scp_3.name = f'Upstream Supply Chain'
            scores_list.append(imp_dep_bank_scp_3)

            # create list to store NACE converted scores
            NACE_score_list = []
            # loop through scores in list
            for score in scores_list:
                name = score.name
                # merge the NACE converted with score and group by the NACE high level code categories
                NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(['Code']).mean()
                NACE_score.name = name
                NACE_score_list.append(NACE_score)

            # plot a heatmap for each bank
            # loop through NACE converted scores
            for score in NACE_score_list:
                # create subplot
                ax = plt.subplot(num_banks, 2, i)
                color_scheme = colors
                # plot heatmap with x labels only for bottom two plots
                if i < ((num_banks*2) - 1):
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)
                else:
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)

                # set the subplot title and size label size
                bank_name = bank_names_dict.get(bank)
                if i == 1 or i == 2:
                    ax.set_title(f'{score.name} \n {bank_name}')
                else:
                    ax.set_title(f'{bank_name}')
                i = i + 1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                ax.set_ylabel('Sector')

        # plt.suptitle(f'Sector Impact Feedback Intensity Metric', y=0.99)
        # adjust the layout to fit all subplots
        plt.tight_layout()
        # save the figure - for each bank domicile region - North America, Europe, Asia
        plt.savefig(f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank-level Sector {folder} Percentage Value at Risk')
        plt.close()

    # REGION
    # loop through domicile regions
    for region in regions:
        # var finance
        # create plot
        plt.figure(figsize=(20, 20))

        # get number of banks in domicile region
        num_banks = bank_regions_df['Region'].value_counts()[region]

        i = 1
        # loop through banks
        for bank in banks:
            # if bank not in domicile region - skip
            if not bank_regions_dict.get(bank) == region:
                continue

            # create list to store scores
            score_list = []

            # get score for one bank and aggregate to region
            # direct operations
            scope_1_overlap_one_bank = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank].drop(columns=['Bank', 'sector']).groupby('region').sum()
            scope_1_overlap_one_bank = scope_1_overlap_one_bank.reset_index().rename(
                columns={'region': 'Region'}).set_index(
                ['Region'])
            scope_1_overlap_one_bank.name = f'Direct Operations'
            # upstream supply chain
            if folder == 'Overlap':
                scope_3_overlap_one_bank = scope_3_score_rows.reset_index()[
                    scope_3_score_rows.reset_index()['Bank'] == bank].drop(columns=['Bank']).groupby('region').sum()
            else:
                scope_3_overlap_one_bank = scope_3_score_rows.reset_index()[
                    scope_3_score_rows.reset_index()['Bank'] == bank].drop(columns=['Bank', 'sector']).groupby('region').sum()
            scope_3_overlap_one_bank = scope_3_overlap_one_bank.reset_index().rename(
                columns={'region': 'Region'}).set_index(
                ['Region'])
            scope_3_overlap_one_bank.name = f'Upstream Supply Chain'

            # append the region aggregated scores for the abnk to the list
            score_list.append(scope_1_overlap_one_bank)
            score_list.append(scope_3_overlap_one_bank)

            # loop through the scores in the list
            for score in score_list:
                # create subplot
                ax = plt.subplot(num_banks, 2, i)

                # plot the heatmap with x labels only on the bottom two plots
                if i < ((num_banks * 2) - 1):
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False)
                else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True)

                # set the subplot title and label sizes
                bank_name = bank_names_dict.get(bank)
                if i == 1 or i == 2:
                    ax.set_title(f'{score.name} \n {bank_name}')
                else:
                    ax.set_title(f'{bank_name}')
                i = i + 1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels()):
                    item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                # ax.set_ylabel(bank_name)
        # adjust the layout to fit all subplots
        plt.tight_layout()
        # save the figure
        plt.savefig(f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank Level VaR Region Level {folder} Heatmap')
        plt.close()

    return None


def plot_heatmap_bank_level_separated(score_list):
    """
    This function plots the bank portfolio -level risk by sector and region for direct operations and upstream supply
    chain - separated by bank domicile
    :param score_list: list of scores with upstream supply chain and direct operations results
    :return:
    """
    # determine the score type - both (endogenous), impact, dependency
    if 'Both' in score_list[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in score_list[0].name:
            folder = 'Impact'
        if 'Dependency' in score_list[0].name:
            folder = 'Dependency'

    # store the scores by type
    for score in score_list:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = score.name
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = score.name
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = score.name

    # create color palette for figures
    colors = sns.color_palette("Reds", as_cmap=True)

    # list of banks
    banks = np.unique(scope_1_score.reset_index()['Bank'])

    # get regions - domicile for banks
    regions = ['North America', 'Europe', 'Asia']
    bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    # get converter for NACE sector summarization
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])

    # loop through bank domicile regions
    for region in regions:
        # var finance
        # create plot
        plt.figure(figsize=(20, 20))

        # get the number of banks in the domicile
        num_banks = bank_regions_df['Region'].value_counts()[region]

        # Direct operations
        i = 1
        # loop through the banks
        for bank in banks:
            # if the bank is not the domicile region - skip
            if not bank_regions_dict.get(bank) == region:
                continue

            # create list to store scores
            scores_list = []

            # scope 1 - direct operations - get score for the bank by sector
            imp_dep_bank_scp_1 = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank].drop(columns=['Bank', 'region']).groupby(['sector']).sum()
            imp_dep_bank_scp_1 = imp_dep_bank_scp_1.reset_index().rename(columns={'sector': 'Sector'}).set_index(
                ['Sector'])
            imp_dep_bank_scp_1.name = f'Direct Operations'
            scores_list.append(imp_dep_bank_scp_1)

            # create list of scores
            NACE_score_list = []
            # loop through the scores to convert to NACE high level category
            for score in scores_list:
                name = score.name
                # merge the score and the NACE converter and group by NACE high level category
                NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(['Code']).sum()
                NACE_score.name = name
                NACE_score_list.append(NACE_score)

            # plot a heatmap for each
            # loop through NACE scores
            for score in NACE_score_list:
                # create subplot
                ax = plt.subplot(math.ceil(num_banks/2), 2, i)
                color_scheme = colors
                # plot heatmaps with xlabels only on the bottom figures
                if i < (num_banks - 1):
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)
                else:
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)
                # set the title of the subplot
                bank_name = bank_names_dict.get(bank)
                # if i == 1 or i == 2:
                #     ax.set_title(f'{score.name} \n {bank_name}')
                # else:
                ax.set_title(f'{bank_name}')

                i = i + 1
                # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                #              ax.get_xticklabels() + ax.get_yticklabels()):
                #     item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                ax.set_ylabel('Sector')
        # set the title of the figure
        if folder != 'Overlap':
            plt.suptitle(f'{region} Sector Direct Operations {folder}', y=0.99)
        else:
            plt.suptitle(f'{region} Sector Direct Operations Endogenous Risk Exposure', y=0.99)
        # adjust the layout to fit all subplots
        plt.tight_layout()
        # save figure
        plt.savefig(
            f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank-level Sector Direct Operations {folder} Percentage Value at Risk')
        plt.close()

        # Upstream Supply Chain
        # create figure
        plt.figure(figsize=(20, 20))
        i = 1
        # loop through the banks
        for bank in banks:
            # if not in the domicile region - skip
            if not bank_regions_dict.get(bank) == region:
                continue

            # create a score list
            scores_list = []

            # scope 3 - upstream supply chain - get the score for one bank and group by sector
            imp_dep_bank_scp_3 = scope_3_score_source.reset_index()[
                scope_3_score_source.reset_index()['Bank'] == bank].drop(columns=['Bank', 'region']).groupby(
                ['sector']).sum()
            imp_dep_bank_scp_3 = imp_dep_bank_scp_3.reset_index().rename(columns={'sector': 'Sector'}).set_index(
                ['Sector'])
            imp_dep_bank_scp_3.name = f'Upstream Supply Chain'
            scores_list.append(imp_dep_bank_scp_3)

            # convert the score to the NACE high level categories
            NACE_score_list = []
            for score in scores_list:
                name = score.name
                NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(['Code']).sum()
                NACE_score.name = name
                NACE_score_list.append(NACE_score)

            # plot a heatmap for each
            # loop through NACE scores
            for score in NACE_score_list:
                # create subplot
                ax = plt.subplot(math.ceil(num_banks / 2), 2, i)
                color_scheme = colors
                # plot heatmap with only bottom two plots having x labels
                if i < (num_banks - 1):
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)
                else:
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)
                # set title of subplot
                bank_name = bank_names_dict.get(bank)
                # if i == 1 or i == 2:
                #     ax.set_title(f' {bank_name}')
                # else:
                ax.set_title(f'{bank_name}')

                i = i + 1
                # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                #              ax.get_xticklabels() + ax.get_yticklabels()):
                #     item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                ax.set_ylabel('Sector')
        # set figure title
        if folder != 'Overlap':
            plt.suptitle(f'{region} Sector Upstream Supply Chain {folder}', y=0.99)
        else:
            plt.suptitle(f'{region} Sector Upstream Supply Chain Endogenous Risk Exposure', y=0.99)
        # adjust layout to fit all figures
        plt.tight_layout()
        # save figure
        plt.savefig(
            f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank-level Sector Upstream Supply Chain {folder} Percentage Value at Risk')
        plt.close()

        # REGION

        # create figure
        plt.figure(figsize=(20, 20))

        # get number of banks in the domicile region
        num_banks = bank_regions_df['Region'].value_counts()[region]

        # direct operations
        i = 1
        # loop through the banks
        for bank in banks:
            # if bank not in the domicile region - skip
            if not bank_regions_dict.get(bank) == region:
                continue

            # create list to store the scores
            score_list = []

            # scope 1
            # get the score for the bank - grouped by the region
            scope_1_overlap_one_bank = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank].drop(columns=['Bank', 'sector']).groupby('region').sum()
            scope_1_overlap_one_bank = scope_1_overlap_one_bank.reset_index().rename(
                columns={'region': 'Region'}).set_index(
                ['Region'])
            scope_1_overlap_one_bank.name = f'Direct Operations'
            score_list.append(scope_1_overlap_one_bank)

            # loop through the scores
            for score in score_list:
                # create subplot
                ax = plt.subplot(math.ceil(num_banks/2), 2, i)

                # plot the heatmap with only the last two having xtick labels
                if i < (num_banks - 1):
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False)
                else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True)

                # set subplot title and label sizes
                bank_name = bank_names_dict.get(bank)
                # if i == 1 or i == 2:
                #     ax.set_title(f'{score.name} \n {bank_name}')
                # else:
                ax.set_title(f'{bank_name}')
                i = i + 1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels()):
                    item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                # ax.set_ylabel(bank_name)
        # set figure title
        if folder != 'Overlap':
            plt.suptitle(f'{region} Region Direct Operations {folder}', y=0.99)
        else:
            plt.suptitle(f'{region} Region Direct Operations Endogenous Risk Exposure', y=0.99)
        # adjust the layout to fit all subplots
        plt.tight_layout()
        # save the figure
        plt.savefig(
            f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank Level VaR Region Level Direction Operations {folder} Heatmap')
        plt.close()


        # Upstream Supply Chain
        # create figure
        plt.figure(figsize=(20, 20))
        i = 1
        # loop through banks
        for bank in banks:
            # if bank not in domicile region - skip
            if not bank_regions_dict.get(bank) == region:
                continue

            # create list to store the scores
            score_list = []
            # scope 3
            # get the score for one bank and group by region
            if folder == 'Overlap':
                scope_3_overlap_one_bank = scope_3_score_rows.reset_index()[
                    scope_3_score_rows.reset_index()['Bank'] == bank].drop(columns=['Bank']).groupby('region').sum()
            else:
                scope_3_overlap_one_bank = scope_3_score_rows.reset_index()[
                    scope_3_score_rows.reset_index()['Bank'] == bank].drop(columns=['Bank', 'sector']).groupby(
                    'region').sum()
            scope_3_overlap_one_bank = scope_3_overlap_one_bank.reset_index().rename(
                columns={'region': 'Region'}).set_index(
                ['Region'])
            scope_3_overlap_one_bank.name = f'Upstream Supply Chain'
            score_list.append(scope_3_overlap_one_bank)

            # loop through region scores
            for score in score_list:
                # create subplot
                ax = plt.subplot(math.ceil(num_banks / 2), 2, i)
                # plot heatmap with only bottom two having xtick labels
                if i < (num_banks - 1):
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False)
                else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True)
                # set subplot title and set label size
                bank_name = bank_names_dict.get(bank)
                # if i == 1 or i == 2:
                #     ax.set_title(f'{score.name} \n {bank_name}')
                # else:
                ax.set_title(f'{bank_name}')
                i = i + 1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels()):
                    item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                # ax.set_ylabel(bank_name)

        # set figure title
        if folder != 'Overlap':
            plt.suptitle(f'{region} Region Upstream Supply Chain {folder}', y=0.99)
        else:
            plt.suptitle(f'{region} Region Upstream Supply Chain Endogenous Risk Exposure', y=0.99)
        # adjust layout to fit all subplots
        plt.tight_layout()
        # save figure
        plt.savefig(
            f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank Level VaR Region Level Upstream Supply Chain {folder} Heatmap')
        plt.close()

    return None


def plot_bar_chart_system(both_mean, both_min, both_max):
    """
    This function plots bar chart for the system-level based on the scores for direct operations and upstream supply
    chain
    :param both_mean: results with mean methoodological treatment
    :param both_min: results with min methoodological treatment
    :param both_max: results with max methoodological treatment
    :return:
    """
    # determine score type
    if 'Both' in both_mean[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in both_mean[0].name:
            folder = 'Impact'
        if 'Dependency' in both_mean[0].name:
            folder = 'Dependency'

    # store the scores in the appropriate list based on whether proportional or absolute and for direct operations,
    # or upstream supply chain for each methodological treatment
    scope_1_var_finance = []
    scope_3_var_finance = []
    scope_1_var_finance_abs = []
    scope_3_var_finance_abs = []
    for score in both_mean:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance_abs.append(score)
    for score in both_min:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance_abs.append(score)
    for score in both_max:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance_abs.append(score)

    # convert to region
    scope_1_var_finance_region = aggregate_to_region_service(scope_1_var_finance, 'sum')
    scope_3_var_finance_region = aggregate_to_region_service(scope_3_var_finance, 'sum')
    scope_1_var_finance_region_abs = aggregate_to_region_service(scope_1_var_finance_abs, 'sum')
    scope_3_var_finance_region_abs = aggregate_to_region_service(scope_3_var_finance_abs, 'sum')

    # get banks and services
    # get a list of banks and services
    banks = np.unique(scope_1_var_finance[0].reset_index()['Bank']).tolist()
    services = np.unique(scope_1_var_finance[0].columns).tolist()

    # generate the system-level proportion for direct operations and upstream supply chain using total portfolio value
    # across all banks
    financial_data_df = finance_GSIB_reformat()
    system_total = financial_data_df['EUR m adjusted'].sum()
    scope_1_var_finance_region_system = []
    scope_3_var_finance_region_system = []
    # scope 1
    for score in scope_1_var_finance_region_abs:
        df = score.copy()
        score_name = score.name
        prop_df = df.reset_index().drop(columns='Bank').groupby('region').sum()
        prop_df = (prop_df / system_total) * 100
        prop_df.name = score_name
        scope_1_var_finance_region_system.append(prop_df)
    # scope 3
    for score in scope_3_var_finance_region_abs:
        df = score.copy()
        score_name = score.name
        prop_df = df.reset_index().drop(columns=['Bank']).groupby('region').sum()
        prop_df = (prop_df / system_total) * 100
        prop_df.name = score_name
        scope_3_var_finance_region_system.append(prop_df)

    # create figure and subplot
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1)
    # loop through the system-level proportional scores - direct operations
    for sheet in scope_1_var_finance_region_system:
        df = sheet.copy()
        one_bank_df = df.reset_index()
        mydict = {}
        # loop through ecosystem services
        for service in services:
            # get the total risk for the ecosystem service
            var = np.sum(one_bank_df[service])
            mydict[service] = var
        # assign to the appropriate methodological treatment
        if re.search('min', sheet.name):
            scope_1_min_values = mydict
        if re.search('mean', sheet.name):
            scope_1_mean_values = mydict
        if re.search('max', sheet.name):
            scope_1_max_values = mydict
    # generate the xticks with the ecosystem services
    X_axis = np.arange(len(services))

    # scope 1 - direct operations - format values for plotting
    scope_1_mean = np.array(list(scope_1_mean_values.values()))
    scope_1_min = np.array(list(scope_1_min_values.values()))
    scope_1_max = np.array(list(scope_1_max_values.values()))

    # create values for error bars
    lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
    higher_err_scope_1 = (scope_1_max) - (scope_1_mean)
    # format error bar values for plotting
    asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

    # plot the bar chart direct operations bar
    if folder == 'Overlap':
        ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
    else:
        ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
    # plot the error bars
    ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')

    # loop through the system-level proportional scores - upstream supply chain
    for sheet in scope_3_var_finance_region_system:
        # if not (re.search('imp', sheet.name) and re.search('dep', sheet.name)):
        #     continue
        df = sheet.copy()
        one_bank_df = df.reset_index()
        mydict = {}
        # loop through ecosystem services
        for service in services:
            # get total risk for ecosystem service
            var = np.sum(one_bank_df[service])
            mydict[service] = var
        # assign to associated methodological treatment
        if re.search('min', sheet.name):
            scope_3_min_values = mydict
        if re.search('mean', sheet.name):
            scope_3_mean_values = mydict
        if re.search('max', sheet.name):
            scope_3_max_values = mydict

    # scope 3 - upstream supply chain - format for plotting
    scope_3_mean = np.array(list(scope_3_mean_values.values()))
    scope_3_min = np.array(list(scope_3_min_values.values()))
    scope_3_max = np.array(list(scope_3_max_values.values()))

    # generate error bars values
    lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
    higher_err_scope_3 = (scope_3_max) - (scope_3_mean)
    # format error bar values for plotting
    asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

    # plot bar chart upstream supply chain bar
    if folder == 'Overlap':
        ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
    else:
        ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
    # set error bars
    ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')
    # for item in ([ax.title, ax.xaxis.label] +
    #              ax.get_xticklabels()):
    #     item.set_fontsize(20)
    # generate the legend and xticks
    ax.legend()
    ax.set_xticks(X_axis, services, rotation=45, ha='right')
    # set the figure title
    if folder == 'Overlap':
        ax.set_title(f'System-level Endogenous Risk Exposure for Direction Operations and Upstream Supply Chain')
    else:
        ax.set_title(f' System-level {folder} Value at Risk for Direction Operations and Upstream Supply Chain')
    # adjust the layout
    plt.tight_layout()
    # save figure
    plt.savefig(
        f'{value_at_risk_figure_saving_path}/{folder}/System Finance {folder} Value at Risk for Banks with Error Bars Percentage')
    plt.close()

    return None


def plot_heatmap_bank_level_UK(score_list):
    """
    This function generates heatmaps for the UK banks only
    :param score_list: list of scores with upstream supply chain and direct operations results
    :return: NA
    """
    if 'Both' in score_list[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in score_list[0].name:
            folder = 'Impact'
        if 'Dependency' in score_list[0].name:
            folder = 'Dependency'

    for score in score_list:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = score.name
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = score.name
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = score.name

    colors = sns.color_palette("Reds", as_cmap=True)

    # banks = np.unique(scope_1_score.reset_index()['Bank'])
    banks = ['Barclays PLC', 'Banco Santander SA', 'Standard Chartered PLC', 'HSBC Holdings PLC']


    # plot combined region heatmap at country level and sectoral level
    # bank level

    # get regions
    regions = ['UK']
    # bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    # bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    # get converter for NACE sector summarization
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])

    for region in regions:
        # var finance
        plt.figure(figsize=(20, 20))

        # plt.subplots(4,2,sharey=True)
        # num_banks = bank_regions_df['Region'].value_counts()[region]
        num_banks= 4

        i = 1
        for bank in banks:

            # plot sector heatmap for scope 1 and scope 3 for impact, and dependency
            scores_list = []
            # scope 1
            imp_dep_bank_scp_1 = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank].drop(columns=['Bank', 'region']).groupby(['sector']).sum()
            imp_dep_bank_scp_1 = imp_dep_bank_scp_1.reset_index().rename(columns={'sector': 'Sector'}).set_index(['Sector'])
            imp_dep_bank_scp_1.name = f'Direct Operations'
            scores_list.append(imp_dep_bank_scp_1)

            # scope 3
            imp_dep_bank_scp_3 = scope_3_score_source.reset_index()[
                scope_3_score_source.reset_index()['Bank'] == bank].drop(columns=['Bank', 'region']).groupby(
                ['sector']).sum()
            imp_dep_bank_scp_3 = imp_dep_bank_scp_3.reset_index().rename(columns={'sector': 'Sector'}).set_index(['Sector'])
            imp_dep_bank_scp_3.name = f'Upstream Supply Chain'
            scores_list.append(imp_dep_bank_scp_3)

            NACE_score_list = []
            for score in scores_list:
                name = score.name
                NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(['Code']).mean()
                NACE_score.name = name
                NACE_score_list.append(NACE_score)

            # plot a heatmap for each
            for score in NACE_score_list:
                ax = plt.subplot(num_banks, 2, i)
                color_scheme = colors
                if i < ((num_banks*2) - 1):
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)
                else:
                    # if (i % 2 == 0):
                    #     sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=False)
                    # else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)

                bank_name = bank_names_dict.get(bank)
                if i == 1 or i == 2:
                    ax.set_title(f'{score.name} \n {bank_name}')
                else:
                    ax.set_title(f'{bank_name}')
                i = i + 1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                ax.set_ylabel('Sector')
        # plt.suptitle(f'Sector Impact Feedback Intensity Metric', y=0.99)
        plt.tight_layout()
        plt.savefig(f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank-level Sector {folder} Percentage Value at Risk')
        plt.close()


    for region in regions:
        # var finance
        plt.figure(figsize=(20, 20))

        # plt.subplots(4,2,sharey=True)
        # num_banks = bank_regions_df['Region'].value_counts()[region]

        i = 1
        for bank in banks:
            # if not bank_regions_dict.get(bank) == region:
            #     continue

            score_list = []
            # scope 1 and scope 3
            # overlap
            scope_1_overlap_one_bank = scope_1_score.reset_index()[
                scope_1_score.reset_index()['Bank'] == bank].drop(columns=['Bank', 'sector']).groupby('region').sum()
            scope_1_overlap_one_bank = scope_1_overlap_one_bank.reset_index().rename(
                columns={'region': 'Region'}).set_index(
                ['Region'])
            scope_1_overlap_one_bank.name = f'Direct Operations'
            if folder == 'Overlap':
                scope_3_overlap_one_bank = scope_3_score_rows.reset_index()[
                    scope_3_score_rows.reset_index()['Bank'] == bank].drop(columns=['Bank']).groupby('region').sum()
            else:
                scope_3_overlap_one_bank = scope_3_score_rows.reset_index()[
                    scope_3_score_rows.reset_index()['Bank'] == bank].drop(columns=['Bank', 'sector']).groupby('region').sum()
            scope_3_overlap_one_bank = scope_3_overlap_one_bank.reset_index().rename(
                columns={'region': 'Region'}).set_index(
                ['Region'])
            scope_3_overlap_one_bank.name = f'Upstream Supply Chain'

            score_list.append(scope_1_overlap_one_bank)
            score_list.append(scope_3_overlap_one_bank)

            for score in score_list:
                ax = plt.subplot(num_banks, 2, i)

                if i < ((num_banks * 2) - 1):
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=False)
                else:
                    sns.heatmap(score, ax=ax, cmap=color_scheme, xticklabels=True)

                bank_name = bank_names_dict.get(bank)
                if i == 1 or i == 2:
                    ax.set_title(f'{score.name} \n {bank_name}')
                else:
                    ax.set_title(f'{bank_name}')
                i = i + 1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels()):
                    item.set_fontsize(17)
                # bank_name = bank_names_dict.get(bank)
                # ax.set_ylabel(bank_name)
        plt.tight_layout()
        plt.savefig(f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank Level VaR Region Level {folder} Heatmap')
        plt.close()

    return None


def plot_bar_chart_UK(both_mean, both_min, both_max):
    """
    This function generates bar charts for the UK banks only
    :param both_mean: results with mean methoodological treatment
    :param both_min: results with min methoodological treatment
    :param both_max: results with max methoodological treatment
    :return: NA
    """
    if 'Both' in both_mean[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in both_mean[0].name:
            folder = 'Impact'
        if 'Dependency' in both_mean[0].name:
            folder = 'Dependency'

    scope_1_var_finance = []
    scope_3_var_finance = []
    scope_1_var_finance_abs = []
    scope_3_var_finance_abs = []
    for score in both_mean:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance_abs.append(score)
    for score in both_min:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance_abs.append(score)
    for score in both_max:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance_abs.append(score)

    # convert to region
    scope_1_var_finance_region = aggregate_to_region_service(scope_1_var_finance, 'sum')
    scope_3_var_finance_region = aggregate_to_region_service(scope_3_var_finance, 'sum')
    scope_1_var_finance_region_abs = aggregate_to_region_service(scope_1_var_finance_abs, 'sum')
    scope_3_var_finance_region_abs = aggregate_to_region_service(scope_3_var_finance_abs, 'sum')

    # get banks and services
    # get a list of banks and services
    # banks = np.unique(scope_1_var_finance[0].reset_index()['Bank']).tolist()
    banks = ['Barclays PLC', 'Banco Santander SA', 'Standard Chartered PLC', 'HSBC Holdings PLC']
    services = np.unique(scope_1_var_finance[0].columns).tolist()

    # create a plot of scope 1 and scope 3 value at risk for all the banks and all the portfolio
    # with the max and min values as the error bars
    score_types = ['mean', 'min', 'max']

    # get regions
    regions = ['UK']
    # bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    # bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    for region in regions:
        # var finance
        plt.figure(figsize=(20, 20))

        # plt.subplots(4,2,sharey=True)
        # num_banks = bank_regions_df['Region'].value_counts()[region]
        num_banks = 4
        i = 0
        for bank in banks:
            # if not bank_regions_dict.get(bank) == region:
            #     continue
            i = i + 1
            if (i == 1):
                ax = plt.subplot(math.ceil(num_banks/2) + 1, 2, i)
            if (i != 1):
                ax = plt.subplot(math.ceil(num_banks/2) + 1, 2, i, sharey=ax, sharex=ax)
            for sheet in scope_1_var_finance:
                df = sheet.copy()
                one_bank_df = df.reset_index()[df.reset_index()['Bank'] == bank]
                mydict = {}
                for service in services:
                    var = np.sum(one_bank_df[service])
                    mydict[service] = var
                if re.search('min', sheet.name):
                    scope_1_min_values = mydict
                if re.search('mean', sheet.name):
                    scope_1_mean_values = mydict
                if re.search('max', sheet.name):
                    scope_1_max_values = mydict

            X_axis = np.arange(len(services))

            # scope 1
            scope_1_mean = np.array(list(scope_1_mean_values.values()))
            scope_1_min = np.array(list(scope_1_min_values.values()))
            scope_1_max = np.array(list(scope_1_max_values.values()))

            lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
            higher_err_scope_1 = (scope_1_max) - (scope_1_mean)

            asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

            if folder == 'Overlap':
                ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
            else:
                ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')

            ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')

            for sheet in scope_3_var_finance:
                df = sheet.copy()
                one_bank_df = df.reset_index()[df.reset_index()['Bank'] == bank]
                mydict = {}
                for service in services:
                    var = np.sum(one_bank_df[service])
                    mydict[service] = var
                if re.search('min', sheet.name):
                    scope_3_min_values = mydict
                if re.search('mean', sheet.name):
                    scope_3_mean_values = mydict
                if re.search('max', sheet.name):
                    scope_3_max_values = mydict

            # scope 3
            scope_3_mean = np.array(list(scope_3_mean_values.values()))
            scope_3_min = np.array(list(scope_3_min_values.values()))
            scope_3_max = np.array(list(scope_3_max_values.values()))

            lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
            higher_err_scope_3 = (scope_3_max) - (scope_3_mean)

            asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

            if folder == 'Overlap':
                ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
            else:
                ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
            ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')
            # for item in ([ax.title, ax.xaxis.label] +
            #              ax.get_xticklabels() ):
            #     item.set_fontsize(20)
            ax.legend()
            if folder == 'Overlap':
                ax.set_title(f'{bank} \n Endogenous Risk Exposure for Direct Operations and Upstream Supply Chain')
            else:
                ax.set_title(f'{bank} \n {folder} Value at Risk for Direct Operations and Upstream Supply Chain')

            ax.set_xticks(X_axis, services, rotation=45, ha='right')
            if i < (num_banks):
                ax.tick_params(labelbottom=False)

        # compare with system-level --> see if any different --> that might be what you need for -->
        financial_data_df = finance_GSIB_reformat()
        system_total = financial_data_df['EUR m adjusted'].sum()
        scope_1_var_finance_region_system = []
        scope_3_var_finance_region_system = []
        # scope 1
        for score in scope_1_var_finance_region_abs:
            df = score.copy()
            score_name = score.name
            prop_df = df.reset_index().drop(columns='Bank').groupby('region').sum()
            prop_df = (prop_df / system_total) * 100
            prop_df.name = score_name
            scope_1_var_finance_region_system.append(prop_df)
        # scope 3
        for score in scope_3_var_finance_abs:
            df = score.copy()
            score_name = score.name
            prop_df = df.reset_index().drop(columns=['Bank', 'sector']).groupby('region').sum()
            prop_df = (prop_df / system_total) * 100
            prop_df.name = score_name
            scope_3_var_finance_region_system.append(prop_df)

        # i = 0
        i = i + 1
        ax = plt.subplot(math.ceil(num_banks/2) + 1, 2, i, sharey=ax, sharex=ax)
        for sheet in scope_1_var_finance_region_system:
            df = sheet.copy()
            one_bank_df = df.reset_index()
            mydict = {}
            for service in services:
                var = np.sum(one_bank_df[service])
                mydict[service] = var
            if re.search('min', sheet.name):
                scope_1_min_values = mydict
            if re.search('mean', sheet.name):
                scope_1_mean_values = mydict
            if re.search('max', sheet.name):
                scope_1_max_values = mydict

        X_axis = np.arange(len(services))

        # scope 1
        scope_1_mean = np.array(list(scope_1_mean_values.values()))
        scope_1_min = np.array(list(scope_1_min_values.values()))
        scope_1_max = np.array(list(scope_1_max_values.values()))

        lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
        higher_err_scope_1 = (scope_1_max) - (scope_1_mean)

        asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

        if folder == 'Overlap':
            ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
        else:
            ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
        ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')

        for sheet in scope_3_var_finance_region_system:
            # if not (re.search('imp', sheet.name) and re.search('dep', sheet.name)):
            #     continue
            df = sheet.copy()
            one_bank_df = df.reset_index()
            mydict = {}
            for service in services:
                var = np.sum(one_bank_df[service])
                mydict[service] = var
            if re.search('min', sheet.name):
                scope_3_min_values = mydict
            if re.search('mean', sheet.name):
                scope_3_mean_values = mydict
            if re.search('max', sheet.name):
                scope_3_max_values = mydict

        # scope 3
        scope_3_mean = np.array(list(scope_3_mean_values.values()))
        scope_3_min = np.array(list(scope_3_min_values.values()))
        scope_3_max = np.array(list(scope_3_max_values.values()))

        lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
        higher_err_scope_3 = (scope_3_max) - (scope_3_mean)

        asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

        if folder == 'Overlap':
            ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
        else:
            ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')

        ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')

        # for item in ([ax.title, ax.xaxis.label] +
        #              ax.get_xticklabels()):
        #     item.set_fontsize(20)

        ax.legend()
        ax.set_xticks(X_axis, services, rotation=45, ha='right')
        # ax.tick_params(axis='x', labelbottom=True)

        if folder == 'Overlap':
            ax.set_title(f' System-level Endogenous Risk Exposure for Direction Operations and Upstream Supply Chain')
        else:
            ax.set_title(f' System-level {folder} Value at Risk for Direction Operations and Upstream Supply Chain')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(
            f'{value_at_risk_figure_saving_path}/{folder}/{region} Bank-level Finance {folder} Value at Risk for Banks with Error Bars Percentage')
        plt.close()

    return None

def plot_finance_data():
    """
    This function generates plots displaying the financial portfolio construction. Specifically, it generates a bar
    chart of total portfolio value by bank and a heatmap of the region-sector exposure of the overall portfolio.
    :return: NA
    """
    # get the financial data
    financial_data_df = finance_GSIB_reformat()

    # list of banks
    banks = np.unique(financial_data_df.reset_index()['Bank']).tolist()

    # get total only for each bank
    total_only_df = financial_data_df.reset_index().drop(columns=['sector', 'region', 'EUR m adjusted', 'Proportion of Loans']).drop_duplicates().set_index(['Bank'])
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()
    total_only_short_name_df = pd.merge(total_only_df, bank_names_df, left_index=True, right_index=True)
    total_only_short_name_df = total_only_short_name_df.reset_index().drop(columns=['index'])
    total_only_short_name_df[['Bank', 'Country']] = total_only_short_name_df['Short Name'].str.split('(', expand=True)
    total_only_short_name_df[['Country', 'Cleaning']] = total_only_short_name_df['Country'].str.split(')', expand=True)
    total_country_bank_df = total_only_short_name_df.drop(columns='Cleaning')

    # sort banks within each country by total
    total_country_bank_df = total_country_bank_df.sort_values(by = ["Country", "Bank"], ascending=[True, False])
    country_order = total_country_bank_df.groupby("Country")["Total Loan"].sum().sort_values(ascending=False).index
    total_country_bank_df["Country"] = pd.Categorical(total_country_bank_df["Country"], categories=country_order, ordered=True)
    total_country_bank_df = total_country_bank_df.sort_values(by=["Country", "Total Loan"], ascending=[True, False])

    # create figure
    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(10, 6))
    # plot the bar of total portfolio value for each bank
    ax.bar(total_country_bank_df["Bank"], total_country_bank_df["Total Loan"])
    # plt.bar(x = total_country_bank_df["Bank"], height = total_country_bank_df['Total Loan'])
    plt.xticks(rotation=90)

    # Add country labels below the x-axis labels
    positions = range(len(total_country_bank_df))
    prev_country = None
    country_positions = []
    for pos, (bank, country) in enumerate(zip(total_country_bank_df["Bank"], total_country_bank_df["Country"])):
        if country != prev_country:
            country_positions.append((pos, country, total_country_bank_df["Country"].value_counts()[country]))
            prev_country = country

    # set the x ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(total_country_bank_df["Bank"], rotation=90, ha='center')
    # set the y label
    plt.ylabel("Portfolio Value in Sample (mEUR)")

    # Create a second x-axis for country labels
    sec = ax.secondary_xaxis('bottom')
    sec.set_xticks([pos + (count - 1) / 2 for pos, _, count in country_positions])
    sec.set_xticklabels([f'\n\n\n\n\n\n\n\n\n{country}' for _, country, _ in country_positions], fontsize=10, fontweight='bold')
    sec.spines['bottom'].set_visible(False)

    # Add separator lines with small gaps
    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks([pos + count - 1 + 0.5 for pos, _, count in country_positions], labels=[])
    sec2.tick_params('x', length=110, width=1.5)

    # set the figure title
    plt.title("Total Portfolio Value for GSIB Banks")
    plt.tight_layout()
    # save the figure
    plt.savefig(financial_bar_chart_path)
    plt.close()

    # get region and sector only for the portfolio (overall)
    region_sector_only_df = financial_data_df.reset_index().drop(columns=['Bank', 'Total Loan', 'Proportion of Loans']).groupby(['region', 'sector']).sum()

    # plot by sector and region for system-level
    # NACE_converter_df = generate_converter_sector()
    # read the NACE converted
    NACE_converter_df = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])
    # convert EXIOBASE sectors to high level NACE categories
    region_code_df = region_sector_only_df.reset_index().merge(NACE_converter_df.reset_index(), right_on=['sector'], left_on=['sector']).groupby(['region', 'Code']).sum().drop(columns=['sector'])
    # generate table of NACE high level sector x region for heat map
    region_code_pivot = region_code_df.reset_index().pivot(index='Code', columns = 'region', values = 'EUR m adjusted')

    # create figure
    plt.figure(figsize=(10, 15))
    colors = sns.color_palette("Reds", as_cmap=True)
    # plot heatmap of NACE code x region and save figure
    sns.heatmap(region_code_pivot.T, cmap=colors)
    plt.savefig(f'{financial_bar_chart_path} Heat Map')

    return None


def plot_heatmap_bank_level_regions_combined(score_list):
    """
    This function plots heatmaps of portfolio-level overlap for sector and region for direct operations and upstream
    supply chain aggregated to the domicile region
    :param score_list: list of scores with upstream supply chain and direct operations results
    :return: NA
    """
    # determine the score type
    if 'Both' in score_list[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in score_list[0].name:
            folder = 'Impact'
        if 'Dependency' in score_list[0].name:
            folder = 'Dependency'
    # get the relevant scores and store by direct operations, upstream supply chain (source and value chain)
    for score in score_list:
        if 'Proportional' not in score.name:
            if 'Scope 1' in score.name:
                scope_1_score = score.copy()
                scope_1_score.name = score.name
            if 'Source' in score.name:
                scope_3_score_source = score.copy()
                scope_3_score_source.name = score.name
            if 'Value Chain' in score.name:
                scope_3_score_rows = score.copy()
                scope_3_score_rows.name = score.name

    # create color palette for heatmaps
    colors = sns.color_palette("Reds", as_cmap=True)

    # list of banks
    banks = np.unique(scope_1_score.reset_index()['Bank'])

    # get the financial data
    financial_data = finance_GSIB_reformat()

    # get regions
    regions = ['North America', 'Europe', 'Asia']
    bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    # get the totals and proportions for portfolio values for banks in each domicile region
    financial_data_region = pd.merge(bank_regions_df.reset_index(), financial_data.reset_index(), right_on=['Bank'], left_on=['Bank']).drop(
        columns=['Bank', 'Total Loan', 'Proportion of Loans']).groupby(['Region', 'region', 'sector']).sum()
    regional_financial_total = pd.merge(bank_regions_df.reset_index(), financial_data.reset_index(), right_on=['Bank'], left_on=['Bank']).drop(columns=['Bank', 'Total Loan', 'Proportion of Loans', 'sector', 'region']).groupby(['Region']).sum().rename(columns={'EUR m adjusted':'Total Loan'})
    financial_data_region_w_total = pd.merge(regional_financial_total.reset_index(), financial_data_region.reset_index(), right_on=['Region'], left_on=['Region'])
    financial_data_region_w_total['Proportional'] = financial_data_region_w_total['EUR m adjusted'] / financial_data_region_w_total['Total Loan']

    # get converter for NACE sector summarization
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])

    # create list to store scores
    scores_list = []

    # aggregate the scores by the bank domicile region to get the average bank in the domicile region
    # direct operations
    scope_1_score_w_region = pd.merge(scope_1_score.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'],
                                      left_on=['Bank'])
    imp_dep_bank_scp_1 = scope_1_score_w_region.reset_index().drop(columns=['Bank']).groupby(
        ['Region', 'sector', 'region']).sum()
    imp_dep_bank_scp_1 = imp_dep_bank_scp_1.reset_index().rename(columns={'sector': 'Sector'}).set_index(
        ['Region', 'Sector', 'region'])
    imp_dep_bank_scp_1.name = f'Direct Operations'
    scores_list.append(imp_dep_bank_scp_1)
    # upstream supply chain - source
    scope_3_score_w_region = pd.merge(scope_3_score_source.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'], left_on=['Bank'])

    imp_dep_bank_scp_3 = scope_3_score_w_region.reset_index().drop(columns=['Bank']).groupby(
        ['Region', 'region', 'sector']).sum()

    imp_dep_bank_scp_3.name = f'Upstream Supply Chain'
    scores_list.append(imp_dep_bank_scp_3)

    # create subplots
    plt.subplots(4, 2,  figsize=(20,20))

    i = 1
    # loop through bank domicile regions
    for region in regions:
        # SECTOR
        # Direct operations
        # get values only for domicile region
        imp_dep_scope_1_region = imp_dep_bank_scp_1.reset_index()[imp_dep_bank_scp_1.reset_index()['Region'] == region].set_index(['Sector'])
        name = imp_dep_bank_scp_1.name
        # aggregate to the NACE high level classification for display purposes and calculate regional proportion
        NACE_score = pd.merge(imp_dep_scope_1_region, NACE_converter, right_index=True, left_index=True).reset_index().drop(columns=['index', 'region', 'Sector']).set_index(['Region']).groupby(['Code']).sum()
        NACE_score = NACE_score.reset_index().rename(columns={'Code': 'Sector'}).set_index(
            ['Sector'])
        NACE_score = (NACE_score / regional_financial_total.loc[region]['Total Loan'])* 100
        NACE_score.name = name

        # create subplot
        ax = plt.subplot(4, 2, i)
        color_scheme = colors

        # plot heatmap
        sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)
        # set heatmap title
        ax.set_title(f'G-SIB Direct Operations - {region}')

        i = i + 1
        # set label size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels()):
            item.set_fontsize(17)
        # bank_name = bank_names_dict.get(bank)
        ax.set_ylabel('Sector')
        # ax.tick_params(axis='y', labelsize=9)

        # upstream supply chain
        # get score only for domicile region
        imp_dep_scope_3_region = imp_dep_bank_scp_3.reset_index()[
            imp_dep_bank_scp_3.reset_index()['Region'] == region].set_index(['sector'])
        name = imp_dep_bank_scp_3.name
        # aggregate to the NACE high level classification for display purposes and calculate regional proportion
        NACE_score = pd.merge(imp_dep_scope_3_region, NACE_converter, right_index=True, left_index=True).reset_index().drop(columns=['index', 'region']).set_index(['Region', 'sector']).groupby(['Code']).sum()
        NACE_score = NACE_score.reset_index().rename(columns={'Code': 'Sector'}).set_index(
            ['Sector'])
        NACE_score = (NACE_score / regional_financial_total.loc[region]['Total Loan'])* 100
        NACE_score.name = name

        # create subplot
        ax = plt.subplot(4, 2, i)
        color_scheme = colors

        # plot heatmap
        sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)

        # set the subplot title
        ax.set_title(f'G-SIB Upstream Supply Chain - {region} ')

        i = i + 1

        # set label size and y label
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels()):
            item.set_fontsize(17)
        ax.set_ylabel('Sector')
        # ax.tick_params(axis='y', labelsize=9)

    # System-level (global bank average)
    # direct operations
    # aggregate to the sector-region for all banks and domiciles
    imp_dep_bank_scp_1_system_avg = imp_dep_bank_scp_1.reset_index().drop(columns = ['Region', 'index']).groupby(['Sector', 'region']).sum()
    imp_dep_bank_scp_1_system_avg = imp_dep_bank_scp_1_system_avg.reset_index().rename(columns={'Sector':'sector'}).set_index(['sector', 'region'])
    name = imp_dep_bank_scp_1.name
    # aggregate to the NACE high level classification for display purposes and calculate regional proportion
    NACE_score = pd.merge(imp_dep_bank_scp_1_system_avg, NACE_converter, right_index=True, left_index=True).reset_index().drop(columns = ['region', 'sector']).groupby(['Code']).sum()
    NACE_score = NACE_score.reset_index().rename(columns={'Code': 'Sector'}).set_index(
        ['Sector'])
    NACE_score = (NACE_score / regional_financial_total['Total Loan'].sum()) * 100
    NACE_score.name = name

    # create subplot
    ax = plt.subplot(4, 2, 7)
    color_scheme = colors
    # plot heatmap
    sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)
    # format heatmap
    ax.set_title(f'G-SIB Direct Operations - System')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels()):
        item.set_fontsize(17)

    ax.set_ylabel('Sector')
    # ax.tick_params(axis='y', labelsize=9)

    # upstream supply chain
    # aggregate to the sector-region for all banks and domiciles
    imp_dep_bank_scp_3_system_avg = imp_dep_bank_scp_3.reset_index().drop(
        columns=['Region', 'index']).groupby(['sector', 'region']).sum()
    name = imp_dep_bank_scp_3.name
    # aggregate to the NACE high level classification for display purposes and calculate regional proportion
    NACE_score = pd.merge(imp_dep_bank_scp_3_system_avg, NACE_converter, right_index=True, left_index=True).reset_index().drop(columns = ['region', 'sector']).groupby(['Code']).sum()
    NACE_score = NACE_score.reset_index().rename(columns={'Code': 'Sector'}).set_index(
        ['Sector'])
    NACE_score = (NACE_score / regional_financial_total['Total Loan'].sum())* 100
    NACE_score.name = name

    # create subplot
    ax = plt.subplot(4, 2, 8)
    color_scheme = colors
    # plot hatmap
    sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)
    # format heatmap
    ax.set_title(f'G-SIB Upstream Supply chain - System')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels()):
        item.set_fontsize(17)
    ax.set_ylabel('Sector')
    # ax.tick_params(axis='y', labelsize=9)

    # set heatmap title
    if folder != 'Overlap':
        plt.suptitle(f'Sector {folder} Risk Exposure', y=0.99,  fontsize=19)
    else:
        plt.suptitle(f'Sector Endogenous Risk Exposure', y=0.99,  fontsize=19)
    plt.tight_layout()
    # save figure
    plt.savefig(
        f'{value_at_risk_figure_saving_path}/{folder}/Region level Sector Upstream Supply Chain {folder} Percentage Value at Risk')
    plt.close()


    # REGION
    # aggregate the scores by the bank domicile region to get the average bank in the domicile region
    # direct operations
    scope_1_score_w_region = pd.merge(scope_1_score.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'],
                                      left_on=['Bank'])
    imp_dep_bank_scp_1 = scope_1_score_w_region.reset_index().drop(columns=['Bank']).groupby(
        ['Region', 'region', 'sector']).mean()
    imp_dep_bank_scp_1.name = f'Direct Operations'
    scores_list.append(imp_dep_bank_scp_1)
    # upsteam supply chain - source
    scope_3_score_w_region = pd.merge(scope_3_score_source.reset_index(), bank_regions_df.reset_index(),
                                      right_on=['Bank'], left_on=['Bank'])

    imp_dep_bank_scp_3 = scope_3_score_w_region.reset_index().drop(columns=['Bank']).groupby(
        ['Region', 'region', 'sector']).mean()
    imp_dep_bank_scp_3.name = f'Upstream Supply Chain'
    scores_list.append(imp_dep_bank_scp_3)

    # create subplots
    plt.subplots(4, 2, figsize=(20,20))
    i = 1

    # loop through bank domicile regions
    for region in regions:
        # REGION
        # Direct operations
        # get values only for domicile region
        imp_dep_scope_1_region = imp_dep_bank_scp_1.reset_index()[
            imp_dep_bank_scp_1.reset_index()['Region'] == region].set_index(['region'])
        imp_dep_scope_1_region = imp_dep_scope_1_region.reset_index().drop(columns=['Region', 'sector']).groupby(['region']).sum()
        name = imp_dep_bank_scp_1.name
        #  calculate regional proportion
        NACE_score = imp_dep_scope_1_region.drop(columns=['index'])
        NACE_score = (NACE_score / regional_financial_total.loc[region]['Total Loan'])* 100
        NACE_score.name = name

        # create subplot
        ax = plt.subplot(4, 2, i)
        color_scheme = colors
        # plot heatmap
        sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)
        # format heatmap
        ax.set_title(f'G-SIB Direct Operations - {region}')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels()):
            item.set_fontsize(17)
            # bank_name = bank_names_dict.get(bank)
        ax.set_ylabel('Region')
        ax.tick_params(axis='y', labelsize=7)

        i = i + 1

        # upstream supply chain
        # get values only for domicile region
        imp_dep_scope_3_region = imp_dep_bank_scp_3.reset_index()[
            imp_dep_bank_scp_3.reset_index()['Region'] == region].set_index(['region'])
        imp_dep_scope_3_region = imp_dep_scope_3_region.reset_index().drop(columns=['Region', 'sector']).groupby(['region']).sum()
        name = imp_dep_bank_scp_3.name
        #  calculate regional proportion
        NACE_score = imp_dep_scope_3_region.drop(columns=['index'])
        NACE_score = (NACE_score / regional_financial_total.loc[region]['Total Loan'])* 100
        NACE_score.name = name

        # create subplot
        ax = plt.subplot(4, 2, i)
        color_scheme = colors
        # plot heatmap
        sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=False, yticklabels=True)

        # format heatmap
        ax.set_title(f'G-SIB Upstream Supply Chain - {region}')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels()):
            item.set_fontsize(17)
        # bank_name = bank_names_dict.get(bank)
        ax.set_ylabel('Region')
        ax.tick_params(axis='y', labelsize=7)

        i = i + 1

    #system level - global bank average
    # direct operations
    # aggregate all banks to one and aggregate by sector - region
    imp_dep_bank_scp_1_system_avg = imp_dep_bank_scp_1.reset_index().drop(columns=['Region', 'index']).groupby(
        ['sector', 'region']).mean()
    imp_dep_bank_scp_1_system_avg = imp_dep_bank_scp_1_system_avg.reset_index().drop(columns=['sector']). groupby(['region']).sum()
    name = imp_dep_bank_scp_1.name
    #  calculate global proportion
    NACE_score = imp_dep_bank_scp_1_system_avg.copy()
    NACE_score = (NACE_score / regional_financial_total['Total Loan'].sum())* 100
    NACE_score.name = name

    # create subplot
    ax = plt.subplot(4, 2, 7)
    color_scheme = colors
    # plot heatmap
    sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)
    # format heatmap
    ax.set_title(f'G-SIB Direct Operations - System')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels()):
        item.set_fontsize(17)
    # bank_name = bank_names_dict.get(bank)
    ax.set_ylabel('Region')
    ax.tick_params(axis='y', labelsize=7)

    # upstream supply chain
    # aggregate all banks to one and aggregate by sector - region
    imp_dep_bank_scp_3_system_avg = imp_dep_bank_scp_3.reset_index().drop(
        columns=['Region', 'index']).groupby(['region', 'sector']).mean()
    imp_dep_bank_scp_3_system_avg = imp_dep_bank_scp_3_system_avg.reset_index().drop(columns=['sector']). groupby(['region']).sum()
    name = imp_dep_bank_scp_3.name
    #  calculate global proportion
    NACE_score = imp_dep_bank_scp_3_system_avg.copy()
    NACE_score = (NACE_score / regional_financial_total['Total Loan'].sum()) * 100
    NACE_score.name = name

    # create subplot
    ax = plt.subplot(4, 2, 8)
    color_scheme = colors
    # plot heatmap
    sns.heatmap(NACE_score, ax=ax, cmap=color_scheme, xticklabels=True, yticklabels=True)
    # format heatmap
    ax.set_title(f'G-SIB Upstream Supply chain - System')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels()):
        item.set_fontsize(17)
    # bank_name = bank_names_dict.get(bank)
    ax.set_ylabel('Region')
    ax.tick_params(axis='y', labelsize=7)

    # set title for figure
    if folder != 'Overlap':
        plt.suptitle(f'Region {folder} Risk Exposure', y=0.99, fontsize=17)
    else:
        plt.suptitle(f'Region Endogenous Risk Exposure', y=0.99,  fontsize=17)
    plt.tight_layout()
    # save figure
    plt.savefig(
        f'{value_at_risk_figure_saving_path}/{folder}/Region level Region {folder} Percentage Value at Risk')
    plt.close()


    return None


# regional Bar Charts
def plot_bar_chart_regional(both_mean, both_min, both_max):
    """
    This function plots the upstream supply chain and direct operations risk as bar charts aggregated by bank domicile
    region
    :param both_mean: results with mean methoodological treatment
    :param both_min: results with min methoodological treatment
    :param both_max: results with max methoodological treatment
    :return: NA
    """
    # determine the score type - both (endogenous), impact, dependency
    if 'Both' in both_mean[0].name:
        folder = 'Overlap'
    else:
        if 'Impact' in both_mean[0].name:
            folder = 'Impact'
        if 'Dependency' in both_mean[0].name:
            folder = 'Dependency'

    # store the scores by type - proportional vs absolute / direct operations, upstream supply chain
    scope_1_var_finance = []
    scope_3_var_finance = []
    scope_1_var_finance_abs = []
    scope_3_var_finance_abs = []
    for score in both_mean:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} mean'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} mean'
                scope_3_var_finance_abs.append(score)
    for score in both_min:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} min'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} min'
                scope_3_var_finance_abs.append(score)
    for score in both_max:
        if 'Proportional' in score.name:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance.append(score)
        else:
            if 'Scope 1' in score.name:
                score.name = f'{score.name} max'
                scope_1_var_finance_abs.append(score)
            if 'Source' in score.name:
                score.name = f'{score.name} max'
                scope_3_var_finance_abs.append(score)

    # get banks and services
    # get a list of banks and services
    banks = np.unique(scope_1_var_finance[0].reset_index()['Bank']).tolist()
    services = np.unique(scope_1_var_finance[0].columns).tolist()

    # create a plot of scope 1 and scope 3 value at risk for all the banks and all the portfolio
    # with the max and min values as the error bars
    score_types = ['mean', 'min', 'max']

    # get regions
    regions = ['North America', 'Europe', 'Asia']
    bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    bank_regions_dict = bank_regions_df['Region'].to_dict()

    # get shorter names
    bank_names_df = pd.read_csv(GSIB_bank_names, header=[0], index_col=[0])
    bank_names_dict = bank_names_df['Short Name'].to_dict()

    # get financial data
    financial_data = finance_GSIB_reformat()
    # get the regional total for domiciles and the regional proportion of portfolio value
    financial_data_region = pd.merge(bank_regions_df.reset_index(), financial_data.reset_index(), right_on=['Bank'],
                                     left_on=['Bank']).drop(
        columns=['Bank', 'Total Loan', 'Proportion of Loans']).groupby(['Region', 'region', 'sector']).sum()
    regional_financial_total = pd.merge(bank_regions_df.reset_index(), financial_data.reset_index(), right_on=['Bank'],
                                        left_on=['Bank']).drop(
        columns=['Bank', 'Total Loan', 'Proportion of Loans', 'sector', 'region']).groupby(['Region']).sum().rename(
        columns={'EUR m adjusted': 'Total Loan'})
    financial_data_region_w_total = pd.merge(regional_financial_total.reset_index(),
                                             financial_data_region.reset_index(), right_on=['Region'],
                                             left_on=['Region'])
    financial_data_region_w_total['Proportional'] = financial_data_region_w_total['EUR m adjusted'] / \
                                                    financial_data_region_w_total['Total Loan']

    # create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20), sharey=True)
    axs = axs.flatten()  # Flatten to 1D array for easier indexing

    i = 0
    # loop through domicile regions
    for region in regions:

        # aggregate the results by the domicile region and calculate the proportion of domicile portfolio exposed
        # direct operations
        scope_1_var_finance_region = []
        for score in scope_1_var_finance_abs:
            df = score.copy()
            score_name = score.name
            df_w_region = pd.merge(df.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'], left_on=['Bank']).set_index(['Region', 'Bank', 'region', 'sector'])
            prop_df = df_w_region.reset_index().drop(columns=['Bank', 'region', 'sector']).groupby('Region').sum()
            prop_df = (prop_df / regional_financial_total.loc[region]['Total Loan']) * 100
            prop_df.name = score_name
            scope_1_var_finance_region.append(prop_df)
        # upstream supply chain
        scope_3_var_finance_region = []
        for score in scope_3_var_finance_abs:
            df = score.copy()
            score_name = score.name
            df_w_region = pd.merge(df.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'],
                                   left_on=['Bank']).set_index(['Region', 'Bank', 'region', 'sector'])
            prop_df = df_w_region.reset_index().drop(columns=['Bank', 'region', 'sector']).groupby('Region').sum()
            prop_df = (prop_df / regional_financial_total.loc[region]['Total Loan']) * 100
            prop_df.name = score_name
            scope_3_var_finance_region.append(prop_df)

        # get number of banks in domicile region
        num_banks = bank_regions_df['Region'].value_counts()[region]

        # assign axis
        ax = axs[i]
        i = i + 1
        # loop through direct operations scores
        for sheet in scope_1_var_finance_region:
            df = sheet.copy()
            mydict = {}
            # loop through ecosystem services
            for service in services:
                # get total risk from ecosystem service
                var = np.sum(df[service])
                mydict[service] = var
            # assign to associated methodological treatment
            if re.search('min', sheet.name):
                scope_1_min_values = mydict
            if re.search('mean', sheet.name):
                scope_1_mean_values = mydict
            if re.search('max', sheet.name):
                scope_1_max_values = mydict

        # create x ticks for ecosystem services
        X_axis = np.arange(len(services))

        # direct operations - format values for calculating error bars
        scope_1_mean = np.array(list(scope_1_mean_values.values()))
        scope_1_min = np.array(list(scope_1_min_values.values()))
        scope_1_max = np.array(list(scope_1_max_values.values()))

        # create error bar values
        lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
        higher_err_scope_1 = (scope_1_max) - (scope_1_mean)
        # format error bar values
        asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

        # plot the bar for direct operations
        if folder == 'Overlap':
            ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations$')
        else:
            ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
        # plot the error bars
        ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')

        # loop through the upstream supply chain results
        for sheet in scope_3_var_finance_region:
            df = sheet.copy()
            mydict = {}
            # loop through ecosystem services
            for service in services:
                # get total risk for ecosystem service
                var = np.sum(df[service])
                mydict[service] = var
            # assign to associated methodological treatment
            if re.search('min', sheet.name):
                scope_3_min_values = mydict
            if re.search('mean', sheet.name):
                scope_3_mean_values = mydict
            if re.search('max', sheet.name):
                scope_3_max_values = mydict

        # upstream supply chain - format to calculate the error bars
        scope_3_mean = np.array(list(scope_3_mean_values.values()))
        scope_3_min = np.array(list(scope_3_min_values.values()))
        scope_3_max = np.array(list(scope_3_max_values.values()))
        # calculate the error bar values
        lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
        higher_err_scope_3 = (scope_3_max) - (scope_3_mean)
        # format the error bar values for plotting
        asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

        # plot the upstream supply chain bar
        if folder == 'Overlap':
            ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
        else:
            ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
        # plot the error bars
        ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')
        # for item in ([ax.title, ax.xaxis.label] +
        #              ax.get_xticklabels() ):
        #     item.set_fontsize(20)

        # format the chart
        ax.legend()
        if folder == 'Overlap':
            ax.set_title(f'{region} \n Endogenous Risk Exposure for Direct Operations and Upstream Supply Chain')
        else:
            ax.set_title(f'{region} \n {folder} Value at Risk for Direct Operations and Upstream Supply Chain')

        ax.set_xticks(X_axis, services, rotation=45, ha='right')

    # compare with system-level - global average for the banks
    # get the financial data and the system-level total
    financial_data_df = finance_GSIB_reformat()
    system_total = financial_data_df['EUR m adjusted'].sum()

    # aggregate the scores to global bank average by aggregating all the bansk
    # calculate the proportion of global portfolio exposed using system total
    scope_1_var_finance_region_system = []
    scope_3_var_finance_region_system = []
    # direct operations
    for score in scope_1_var_finance_abs:
        df = score.copy()
        score_name = score.name
        prop_df = df.reset_index().drop(columns=['Bank', 'sector']).groupby('region').sum()
        prop_df = (prop_df / system_total) * 100
        prop_df.name = score_name
        scope_1_var_finance_region_system.append(prop_df)
    # upstream supply chain
    for score in scope_3_var_finance_abs:
        df = score.copy()
        score_name = score.name
        prop_df = df.reset_index().drop(columns=['Bank', 'sector']).groupby('region').sum()
        prop_df = (prop_df / system_total) * 100
        prop_df.name = score_name
        scope_3_var_finance_region_system.append(prop_df)


    i = i + 1
    # assign the axis
    ax = axs[3]
    # direct operations
    for sheet in scope_1_var_finance_region_system:
        df = sheet.copy()
        one_bank_df = df.reset_index()
        mydict = {}
        # loop through ecosystem services
        for service in services:
            # get total risk for ecosystem service
            var = np.sum(one_bank_df[service])
            mydict[service] = var
        # assign to associated methodological treatment
        if re.search('min', sheet.name):
            scope_1_min_values = mydict
        if re.search('mean', sheet.name):
            scope_1_mean_values = mydict
        if re.search('max', sheet.name):
            scope_1_max_values = mydict

    # create x ticks for ecosystem services
    X_axis = np.arange(len(services))

    # direct operations - format to calcualte error bar values
    scope_1_mean = np.array(list(scope_1_mean_values.values()))
    scope_1_min = np.array(list(scope_1_min_values.values()))
    scope_1_max = np.array(list(scope_1_max_values.values()))
    # calcute the error bar values
    lower_err_scope_1 = (scope_1_mean) - (scope_1_min)
    higher_err_scope_1 = (scope_1_max) - (scope_1_mean)
    # format error bar values for plotting
    asymetric_error_scope_1 = np.array(list(zip(lower_err_scope_1, higher_err_scope_1))).T

    # plot bar for direct operations
    if folder == 'Overlap':
        ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
    else:
        ax.bar(X_axis - 0.225, scope_1_mean_values.values(), 0.45, label=r'Direct Operations')
    # plot bar chart
    ax.errorbar(X_axis - 0.225, scope_1_mean_values.values(), yerr=asymetric_error_scope_1, fmt='ro')

    # upstream supply chain
    for sheet in scope_3_var_finance_region_system:
        df = sheet.copy()
        one_bank_df = df.reset_index()
        mydict = {}
        # loop through ecosystem services
        for service in services:
            # get total risk for ecosystem service
            var = np.sum(one_bank_df[service])
            mydict[service] = var
        # assign to associated methodological treatment
        if re.search('min', sheet.name):
            scope_3_min_values = mydict
        if re.search('mean', sheet.name):
            scope_3_mean_values = mydict
        if re.search('max', sheet.name):
            scope_3_max_values = mydict

    # upstream supply chain - format for calculating error bar values
    scope_3_mean = np.array(list(scope_3_mean_values.values()))
    scope_3_min = np.array(list(scope_3_min_values.values()))
    scope_3_max = np.array(list(scope_3_max_values.values()))
    # calculate error bar values
    lower_err_scope_3 = (scope_3_mean) - (scope_3_min)
    higher_err_scope_3 = (scope_3_max) - (scope_3_mean)
    # format error bar values for plotting
    asymetric_error_scope_3 = np.array(list(zip(lower_err_scope_3, higher_err_scope_3))).T

    # plot upstream supply chain bar
    if folder == 'Overlap':
        ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
    else:
        ax.bar(X_axis + 0.225, scope_3_mean_values.values(), 0.45, label=r'Upstream Supply Chain')
    # plot error bars
    ax.errorbar(X_axis + 0.225, scope_3_mean_values.values(), yerr=asymetric_error_scope_3, fmt='ro')

    # for item in ([ax.title, ax.xaxis.label] +
    #              ax.get_xticklabels()):
    #     item.set_fontsize(20)

    # format the chart
    ax.legend()
    ax.set_xticks(X_axis, services, rotation=45, ha='right')
    # ax.tick_params(axis='x', labelbottom=True)

    # format chart
    if folder == 'Overlap':
        ax.set_title(f' System G-SIB Average Endogenous Risk Exposure for Direction Operations and Upstream Supply Chain')
    else:
        ax.set_title(f' System Average {folder} Value at Risk for Direction Operations and Upstream Supply Chain')
    plt.tight_layout()
    for ax in axs.flatten():
        ax.yaxis.set_tick_params(labelleft=True)  # Ensure left y-ticks are displayed

    plt.subplots_adjust(hspace=0.4)

    # save figure
    plt.savefig(
        f'{value_at_risk_figure_saving_path}/{folder}/Region-level Finance {folder} Value at Risk for Banks with Error Bars Percentage')
    plt.close()

    return None

