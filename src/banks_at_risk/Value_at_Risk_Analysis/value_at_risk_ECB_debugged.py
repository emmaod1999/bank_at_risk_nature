import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os.path
import geopandas as gpd
from matplotlib.transforms import blended_transform_factory
from banks_at_risk.Setup.var_paths import (ECB_bank_var_scope_1_both_mean_path, ECB_bank_var_scope_1_both_max_path,
                                           ECB_bank_var_scope_1_both_min_path, ECB_bank_var_scope_3_source_path,
                                           ECB_bank_var_scope_3_value_chain_path, GSIB_bank_var_scope_3_source_impact_path,
                                           GSIB_bank_var_scope_3_value_chain_impact_path, GSIB_bank_var_scope_3_source_dependency_path,
                                           GSIB_bank_var_scope_3_value_chain_dependency_path, GSIB_finance_var_dep_scope1_mean,
                                           GSIB_finance_var_dep_scope1_min, GSIB_finance_var_dep_scope1_max, GSIB_finance_var_imp_scope1_mean,
                                           GSIB_finance_var_imp_scope1_min, GSIB_finance_var_imp_scope1_max, GSIB_finance_var_both_scope1_mean,
                                           GSIB_finance_var_both_scope1_min, GSIB_finance_var_both_scope1_max, GSIB_finance_var_both_scope3_value_chain_mean,
                                           GSIB_finance_var_both_scope3_value_chain_min, GSIB_finance_var_both_scope3_value_chain_max,
                                           GSIB_finance_var_both_scope3_source_mean, GSIB_finance_var_both_scope3_source_min,
                                           GSIB_finance_var_both_scope3_source_max, GSIB_finance_var_imp_scope3_source_mean,
                                           GSIB_finance_var_imp_scope3_source_min, GSIB_finance_var_imp_scope3_source_max,
                                           GSIB_finance_var_imp_scope3_value_chain_mean, GSIB_finance_var_imp_scope3_value_chain_min,
                                           GSIB_finance_var_imp_scope3_value_chain_max, GSIB_system_var_both_scope1_mean,
                                           GSIB_system_var_both_scope1_min, GSIB_system_var_both_scope1_max, GSIB_system_var_both_scope3_source_mean,
                                           GSIB_system_var_both_scope3_source_min, GSIB_system_var_both_scope3_source_max,
                                           GSIB_system_var_both_scope3_value_chain_mean, GSIB_system_var_both_scope3_value_chain_min,
                                           GSIB_system_var_both_scope3_value_chain_max)
from banks_at_risk.Setup.var_plots_paths import value_at_risk_figure_saving_path, EXIO_regions_to_world_countries, natural_earth_w_countries
from banks_at_risk.Setup.finance_paths import GSIB_financial_data_path, GSIB_bank_regions, GSIB_bank_names
from banks_at_risk.Value_at_Risk.helper_value_at_risk_GSIB import finance_GSIB_reformat
from banks_at_risk.Setup.var_plots_paths import value_at_risk_figure_saving_path
from banks_at_risk.Setup.NACE_conversion_paths import NACE_converter_path, NACE_letters_path
from banks_at_risk.Setup.ENCORE_paths import ESS_types_path
from banks_at_risk.Value_at_Risk_Analysis.helper_value_at_risk_analysis_GSIB import aggregate_to_region_service

def aggregate_banks_scope_3(type_name, score_type, drop_banks):
    """
    Aggregate all the scope 3 scores to visualize the results for specificed type name (mean, max or min)
    :return:the combined dataframe
    # """
    if score_type == 'Both':
        scope_3_source_folder_path = ECB_bank_var_scope_3_source_path
        scope_3_value_chain_folder_path = ECB_bank_var_scope_3_value_chain_path
    if score_type == 'Impact':
        scope_3_source_folder_path = GSIB_bank_var_scope_3_source_impact_path
        scope_3_value_chain_folder_path = GSIB_bank_var_scope_3_value_chain_impact_path
    if score_type == 'Dependency':
        scope_3_source_folder_path = GSIB_bank_var_scope_3_source_dependency_path
        scope_3_value_chain_folder_path = GSIB_bank_var_scope_3_value_chain_dependency_path

    # source
    scope_3_source_csvs = os.listdir(scope_3_source_folder_path)

    scope_3_source_csvs_one_df = pd.read_csv(f'{scope_3_source_folder_path}/{scope_3_source_csvs[0]}', header=[0], index_col=[0,1,2])
    ESSs = scope_3_source_csvs_one_df.columns.tolist()
    if drop_banks == True:
        scope_3_source_csvs_one_reindex_df = scope_3_source_csvs_one_df.reset_index().drop(columns=['Bank']).set_index(['region', 'sector'])
    else:
        scope_3_source_csvs_one_reindex_df = scope_3_source_csvs_one_df.reset_index().set_index(['Bank','region', 'sector'])


    scope_3_source_csvs_one_sheet_df = pd.DataFrame(index=scope_3_source_csvs_one_reindex_df.index, columns=ESSs)

    for file in scope_3_source_csvs:
        bank_code = file.split()[0]
        score_df = pd.read_csv(f'{scope_3_source_folder_path}/{file}', index_col=[0,1,2], header=[0])
        if drop_banks == True:
            score_reindexed_df = score_df.reset_index().drop(columns=['Bank']).set_index(['region', 'sector'])
        else:
            score_reindexed_df = score_df.reset_index().set_index(['Bank','region', 'sector'])
        # add the score to the appropriate csv_df
        if type_name == 'mean':
            if 'mean' in file:
                scope_3_source_csvs_one_sheet_df = scope_3_source_csvs_one_sheet_df.add(score_reindexed_df, fill_value=0)
        if type_name == 'min':
            if 'min' in file:
                scope_3_source_csvs_one_sheet_df = scope_3_source_csvs_one_sheet_df.add(score_reindexed_df, fill_value=0)
        if type_name == 'max':
            if 'max' in file:
                scope_3_source_csvs_one_sheet_df = scope_3_source_csvs_one_sheet_df.add(score_reindexed_df, fill_value=0)

    # value chain
    scope_3_value_chain_csvs = os.listdir(scope_3_value_chain_folder_path)
    scope_3_vc_csvs_one_df = pd.read_csv(f'{scope_3_value_chain_folder_path}/{scope_3_value_chain_csvs[0]}', header=[0], index_col=[0,1])
    if drop_banks == True:
        scope_3_vc_csvs_one_reindex_df = scope_3_vc_csvs_one_df.reset_index().drop(columns=['Bank']).rename(columns={'level_0':'region'}).set_index(['region'])
    else:
        scope_3_vc_csvs_one_reindex_df = scope_3_vc_csvs_one_df.reset_index().rename(columns={'level_0':'region'}).set_index(['Bank','region'])
    #
    scope_3_value_chain_csvs_one_sheet_df = pd.DataFrame(index=scope_3_vc_csvs_one_reindex_df.index, columns=ESSs)
    #
    for file in scope_3_value_chain_csvs:
        # bank_code = file.split()[0]
        score_df = pd.read_csv(f'{scope_3_value_chain_folder_path}/{file}', index_col=[0, 1], header=[0])
        if drop_banks == True:
            if 'level_0' in score_df.reset_index().columns:
                score_reindexed_df = score_df.reset_index().rename(columns={'level_0':'region'}).drop(columns=['Bank']).set_index(['region'])
            else:
                score_reindexed_df = score_df.reset_index().drop(columns=['Bank']).set_index(['region'])
        else:
            if 'level_0' in score_df.reset_index().columns:
                score_reindexed_df = score_df.reset_index().rename(columns={'level_0':'region'}).set_index(['Bank','region'])
            else:
                score_reindexed_df = score_df.reset_index().set_index(['Bank','region'])
        if 'sector' in score_reindexed_df.columns:
            score_reindexed_df = score_df.reset_index().set_index(['Bank','region', 'sector'])

        # add the score to the appropriate csv_df
        if type_name == 'mean':
            if 'mean' in file:
                scope_3_value_chain_csvs_one_sheet_df = scope_3_value_chain_csvs_one_sheet_df.add(score_reindexed_df,
                                                                                        fill_value=0)
        if type_name == 'min':
            if 'min' in file:
                scope_3_value_chain_csvs_one_sheet_df = scope_3_value_chain_csvs_one_sheet_df.add(score_reindexed_df,
                                                                                        fill_value=0)
        if type_name == 'max':
            if 'max' in file:
                scope_3_value_chain_csvs_one_sheet_df = scope_3_value_chain_csvs_one_sheet_df.add(score_reindexed_df, fill_value=0)

    # scope_3_source_csvs_one_sheet_df.columns = [col.split(" - ")[-1] for col in scope_3_source_csvs_one_sheet_df.columns]
    # # Sort columns alphabetically (A-Z)
    # scope_3_source_csvs_one_sheet_df = scope_3_source_csvs_one_sheet_df.sort_index(axis=1)
    #
    # scope_3_value_chain_csvs_one_sheet_df.columns = [col.split(" - ")[-1] for col in
    #                                             scope_3_value_chain_csvs_one_sheet_df.columns]
    # # Sort columns alphabetically (A-Z)
    # scope_3_value_chain_csvs_one_sheet_df = scope_3_value_chain_csvs_one_sheet_df.sort_index(axis=1)

    return scope_3_source_csvs_one_sheet_df, scope_3_value_chain_csvs_one_sheet_df

def aggregate_scope_1(score_type, drop_banks):
    """
    Aggregate the scope 1 scores so all the banks have been summed
    :return: the mean, min and max aggregated scope 1 scores
    """
    if score_type == 'Both':
        mean_df = pd.read_csv(GSIB_finance_var_both_scope1_mean, header=[0])
        min_df = pd.read_csv(GSIB_finance_var_both_scope1_min, header=[0])
        max_df = pd.read_csv(GSIB_finance_var_both_scope1_max, header=[0])
    if score_type == 'Impact':
        mean_df = pd.read_csv(GSIB_finance_var_imp_scope1_mean, header=[0])
        min_df = pd.read_csv(GSIB_finance_var_imp_scope1_min, header=[0])
        max_df = pd.read_csv(GSIB_finance_var_imp_scope1_max, header=[0])
    if score_type == 'Dependency':
        mean_df = pd.read_csv(GSIB_finance_var_dep_scope1_mean, header=[0])
        min_df = pd.read_csv(GSIB_finance_var_dep_scope1_min, header=[0])
        max_df = pd.read_csv(GSIB_finance_var_dep_scope1_max, header=[0])

    if drop_banks == True:
        mean_grouped_df = mean_df.drop(columns=['Bank']).groupby(['region', 'sector']).sum()
        min_grouped_df = min_df.drop(columns=['Bank']).groupby(['region', 'sector']).sum()
        max_grouped_df = max_df.drop(columns=['Bank']).groupby(['region', 'sector']).sum()
    else:
        mean_grouped_df = mean_df.groupby(['Bank','region', 'sector']).sum()
        min_grouped_df = min_df.groupby(['Bank','region', 'sector']).sum()
        max_grouped_df = max_df.groupby(['Bank','region', 'sector']).sum()

    # clean the column names and sort them alphabetically
    # mean_grouped_df.columns = [col.split(" - ")[-1] for col in mean_grouped_df.columns]
    # mean_grouped_df = mean_grouped_df.sort_index(axis=1)
    #
    # min_grouped_df.columns = [col.split(" - ")[-1] for col in min_grouped_df.columns]
    # min_grouped_df = min_grouped_df.sort_index(axis=1)
    #
    # max_grouped_df.columns = [col.split(" - ")[-1] for col in max_grouped_df.columns]
    # max_grouped_df = max_grouped_df.sort_index(axis=1)



    return mean_grouped_df, min_grouped_df, max_grouped_df

def aggregate_finance():
    """
    aggregate finance across all the banks for proportional score calculation
    :return: aggregated finance dataframe
    """
    bank_files_list = os.listdir(GSIB_financial_data_path)

    bank_one_df = pd.read_csv(f'{GSIB_financial_data_path}/{bank_files_list[0]}', header=[0,1], index_col=[0])
    aggregated_df = pd.DataFrame(index=bank_one_df.index, columns=bank_one_df.columns)

    for file in bank_files_list:
        bank_one_file_df = pd.read_csv(f'{GSIB_financial_data_path}/{file}', header=[0, 1], index_col=[0])
        aggregated_df = aggregated_df.add(bank_one_file_df, fill_value=0)

    return aggregated_df

def proportional_score_converter(score_df, type):
    """
    convert score to percentage endogenous risk based on total amount of money in system vs amount at risk
    :return: proportional score
    """
    if type == 'total':
        finance_total_df = aggregate_finance()
        total = finance_total_df.iloc[0].sum(axis=0)

        prop_score_df = score_df.div(total)

        # prop_score_df = prop_score_df.multiply(100)
    elif type == 'by_bank':
        finance_df = finance_GSIB_reformat()
        finance_bank_total_df = finance_df.reset_index().drop(columns=['sector', 'region']).groupby(['Bank']).sum()
        score_finance_merged_df = pd.merge(score_df.reset_index(), finance_bank_total_df.reset_index(), left_on=['Bank'], right_on=['Bank'], how='left')
        for ESS in score_df.columns:
            score_finance_merged_df[ESS] = score_finance_merged_df[ESS].div(
                score_finance_merged_df['EUR m adjusted'].replace(0, np.nan)).fillna(0)
        # score_finance_merged_df = score_finance_merged_df.multiply(100)
        prop_score_df = score_finance_merged_df.drop(columns=['EUR m adjusted', 'Total Loan', 'Proportion of Loans'])

    return prop_score_df

def plot_bar_chart():
    """
    Plot scope 1 and scope 3 values as a bar chart
    :return: None
    """
    # get aggregated scope 1 and scope 3 source + value chain
    scope_3_source_mean_df, scope_3_value_chain_mean_df = aggregate_banks_scope_3('mean', 'Both', True)
    scope_1_mean_df, scope_1_min_df, scope_1_max_df = aggregate_scope_1('Both', True)

    # convert scores to proportional
    scope_3_source_prop_df = proportional_score_converter(scope_3_source_mean_df, 'total')
    scope_1_mean_prop_df = proportional_score_converter(scope_1_mean_df, 'total')

    # get the ESS
    services = np.unique(scope_1_mean_prop_df.columns).tolist()

    # create figure and subplot
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(1, 1, 1)
    # generate the xticks with the ecosystem services
    X_axis = np.arange(len(services))
    # scope 1
    mydict = {}
    for service in services:
        # get the total risk for the ecosystem service
        var = np.sum(scope_1_mean_prop_df[service])
        mydict[service] = var
    ax.bar(X_axis - 0.225, mydict.values(), 0.45, label=r'Direct Operations')

    figure_data_df = pd.DataFrame(mydict, index=['Scope 1']).T

    # scope 3
    mydict = {}
    for service in services:
        # get the total risk for the ecosystem service
        var = np.sum(scope_3_source_prop_df[service])
        mydict[service] = var
    ax.bar(X_axis + 0.225, mydict.values(), 0.45, label=r'Upstream Supply Chain')

    figure_data_df['Scope 3'] = pd.Series(mydict)
    figure_data_df.to_csv(f'{value_at_risk_figure_saving_path}/bar_chart_data.csv')

    # generate the legend and xticks
    ax.legend()
    ax.set_xticks(X_axis, services, rotation=90, ha='right')


    ax.set_title(f'Endogenous Risk Exposure for Direction Operations and Upstream Supply Chain')

    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(20)

    # adjust the layout
    plt.tight_layout()
    # save figure
    plt.savefig(
        f'{value_at_risk_figure_saving_path}/Endogenous Risk Exposure for Banks')
    plt.show()
    plt.close()

    return None

def plot_heatmaps():
    """
    Produce heatmaps that describe the region and sector breakdown of the endogenous risk exposure
    :return: None
    """
    # get aggregated scope 1 and scope 3 source + value chain
    scope_3_source_mean_df, scope_3_value_chain_mean_df = aggregate_banks_scope_3('mean', 'Both', True)
    scope_1_mean_df, scope_1_min_df, scope_1_max_df = aggregate_scope_1('Both', True)

    # convert scores to proportional
    scope_3_source_prop_df = proportional_score_converter(scope_3_source_mean_df, 'total')
    scope_3_value_chain_mean_prop_df = proportional_score_converter(scope_3_value_chain_mean_df, 'total')
    scope_1_mean_prop_df = proportional_score_converter(scope_1_mean_df, 'total')

    # get the ESS
    services = np.unique(scope_1_mean_prop_df.columns).tolist()

    # sector
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv(NACE_converter_path, header=[0], index_col=[0])

    # merge the NACE converter and the score and group the sectors by their higher-level classification for
    # display purposes
    # scope 1
    NACE_score = pd.merge(scope_1_mean_prop_df.reset_index().drop(columns=['region']).set_index(['sector']), NACE_converter, right_index=True, left_index=True).groupby(['Code']).sum()
    scope_1_NACE_mean_df = NACE_score.rename(columns={'Code': 'Sector'})

    # scope 3 - source
    NACE_score = pd.merge(scope_3_source_prop_df.reset_index().drop(columns=['region']).set_index(['sector']), NACE_converter, right_index=True, left_index=True).groupby(
        ['Code']).sum()
    scope_3_NACE_mean_df = NACE_score.rename(columns={'Code': 'Sector'})

    # plot a heatmap for each
    # create plot
    plt.figure(figsize=(20, 20))
    # create color scheme
    # create the color palette for plots
    colors = sns.color_palette("Reds", as_cmap=True)

    # scope 1
    # create subplot
    ax = plt.subplot(2, 1, 1)
    # plot the heatmap
    sns.heatmap(scope_1_NACE_mean_df.T, ax=ax, cmap=colors, xticklabels=True)
    scope_1_NACE_mean_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_1_sector_heatmap_data_sectoral.csv')
    # set title and the label size
    ax.set_title(f'Direct Operations')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(20)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # scope 3 - source
    # create subplot
    ax = plt.subplot(2, 1, 2)
    # assign colors
    color_scheme = colors
    # plot the heatmap
    sns.heatmap(scope_3_NACE_mean_df.T.astype(float), ax=ax, cmap=color_scheme, xticklabels=True)
    scope_3_NACE_mean_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_source_sector_heatmap_data.csv')
    # set title and the label size
    ax.set_title(f'Upstream Supply Chain (Source)')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(20)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # adjust the layout to fit figures
    plt.tight_layout()
    # save figure
    plt.savefig(f'{value_at_risk_figure_saving_path}/Endogenous Risk Sectoral Heatmap')
    plt.show()
    plt.close()

    # by region
    scope_3_source_region_df = scope_3_source_prop_df.reset_index().drop(columns=['sector']).groupby(['region']).sum()
    scope_3_value_chain_region_df = scope_3_value_chain_mean_prop_df.copy()
    scope_1_region_df = scope_1_mean_prop_df.reset_index().drop(columns=['sector']).groupby(['region']).sum()

    # plot a heatmap for each
    # create plot
    plt.figure(figsize=(20, 20))
    # create color scheme
    # create the color palette for plots
    colors = sns.color_palette("Reds", as_cmap=True)

    # scope 1
    # create subplot
    ax = plt.subplot(3, 1, 1)
    # assign colors
    color_scheme = colors
    # plot the heatmap
    sns.heatmap(scope_1_region_df.T.astype(float), ax=ax, cmap=color_scheme, xticklabels=True)
    scope_1_region_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_1_region_heatmap_data.csv')
    # set title and the label size
    ax.set_title(f'Direct Operations')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(20)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(rotation=90)

    # scope 3 - source
    # create subplot
    ax = plt.subplot(3, 1, 2)
    # assign colors
    color_scheme = colors
    # plot the heatmap
    sns.heatmap(scope_3_source_region_df.T.astype(float), ax=ax, cmap=color_scheme, xticklabels=True)
    scope_3_source_region_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_source_region_heatmap_data.csv')

    # set title and the label size
    ax.set_title(f'Upstream Supply Chain (Source)')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(20)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(rotation=90)

    # scope 3 - value chain
    # create subplot
    ax = plt.subplot(3, 1, 3)
    # assign colors
    color_scheme = colors
    # plot the heatmap
    sns.heatmap(scope_3_value_chain_region_df.T.astype(float), ax=ax, cmap=color_scheme, xticklabels=True)
    scope_3_value_chain_region_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_value_chain_region_heatmap_data.csv')

    # set title and the label size
    ax.set_title(f'Upstream Supply Chain (Value Chain)')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(20)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(rotation=90)

    # adjust the layout to fit figures
    plt.tight_layout()
    # save figure
    plt.savefig(f'{value_at_risk_figure_saving_path}/Endogenous Risk Regional Heatmap')
    plt.show()
    plt.close()

    return None

def plot_imp_dep_both_heatmaps(grouping, aggregation):
    """
    Create 6 heatmaps of impact, dependency and overlap for the bank portfolios
    :return:
    """
    # get aggregated scope 1 and scope 3 source + value chain
    # both
    scope_3_source_mean_both_df, scope_3_value_chain_mean_both_df = aggregate_banks_scope_3('mean', 'Both', False)
    scope_1_mean_both_df, scope_1_min_both_df, scope_1_max_both_df = aggregate_scope_1('Both', False)
    # impact
    scope_3_source_mean_imp_df, scope_3_value_chain_mean_imp_df = aggregate_banks_scope_3('mean', 'Impact', False)
    scope_1_mean_imp_df, scope_1_min_imp_df, scope_1_max_imp_df = aggregate_scope_1('Impact', False)
    # dependency
    scope_3_source_mean_dep_df, scope_3_value_chain_mean_dep_df = aggregate_banks_scope_3('mean', 'Dependency', False)
    scope_1_mean_dep_df, scope_1_min_dep_df, scope_1_max_dep_df = aggregate_scope_1('Dependency', False)


    # get the ESS
    services = np.unique(scope_1_mean_both_df.columns).tolist()

    # sector
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv(NACE_converter_path, header=[0], index_col=[0])

    # merge the NACE converter and the score and group the sectors by their higher-level classification for
    # display purposes
    # scope 1 both
    scope_1_NACE_mean_both_df = pd.merge(scope_1_mean_both_df.reset_index().drop(columns=['region']).set_index(['sector']), NACE_converter, right_index=True, left_index=True).groupby(['Code', 'Bank']).sum()
    # scope_1_NACE_mean_both_df = scope_1_NACE_mean_both_df.rename(columns={'Code': 'Sector'})
    # scope 1 impact
    scope_1_NACE_mean_imp_df = pd.merge(scope_1_mean_imp_df.reset_index().drop(columns=['region']).set_index(['sector']),
                          NACE_converter, right_index=True, left_index=True).groupby(['Code', 'Bank']).sum()
    # scope_1_NACE_mean_imp_df = scope_1_NACE_mean_imp_df.rename(columns={'Code': 'Sector'})
    # scope 1 dependency
    scope_1_NACE_mean_dep_df = pd.merge(scope_1_mean_dep_df.reset_index().drop(columns=['region']).set_index(['sector']),
                          NACE_converter, right_index=True, left_index=True).groupby(['Code', 'Bank']).sum()
    # scope_1_NACE_mean_dep_df = scope_1_NACE_mean_dep_df.rename(columns={'Code': 'Sector'})

    # scope 3 - source
    # both
    scope_3_NACE_mean_both_df = pd.merge(scope_3_source_mean_both_df.reset_index().drop(columns=['region']).set_index(['sector']), NACE_converter, right_index=True, left_index=True).groupby(
        ['Code', 'Bank']).sum()
    # scope_3_NACE_mean_both_df = scope_3_NACE_mean_both_df.rename(columns={'Code': 'Sector'})
    # impact
    scope_3_NACE_mean_imp_df = pd.merge(scope_3_source_mean_imp_df.reset_index().drop(columns=['region']).set_index(['sector']),
                          NACE_converter, right_index=True, left_index=True).groupby(
        ['Code', 'Bank']).sum()
    # scope_3_NACE_mean_imp_df = scope_3_NACE_mean_imp_df.rename(columns={'Code': 'Sector'})
    # dependency
    scope_3_NACE_mean_dep_df = pd.merge(scope_3_source_mean_dep_df.reset_index().drop(columns=['region']).set_index(['sector']),
                          NACE_converter, right_index=True, left_index=True).groupby(
        ['Code', 'Bank']).sum()
    # scope_3_NACE_mean_dep_df = scope_3_NACE_mean_dep_df.rename(columns={'Code': 'Sector'})


    # aggregate into 10 sectors
    if aggregation == 'SNA':
        #both
        scope_1_NACE_mean_both_df = aggregate_to_SNA(scope_1_NACE_mean_both_df)
        scope_3_NACE_mean_both_df = aggregate_to_SNA(scope_3_NACE_mean_both_df)

        # imp
        scope_1_NACE_mean_imp_df = aggregate_to_SNA(scope_1_NACE_mean_imp_df)
        scope_3_NACE_mean_imp_df = aggregate_to_SNA(scope_3_NACE_mean_imp_df)
        # dep
        scope_1_NACE_mean_dep_df = aggregate_to_SNA(scope_1_NACE_mean_dep_df)
        scope_3_NACE_mean_dep_df = aggregate_to_SNA(scope_3_NACE_mean_dep_df)

    if grouping == '':
        # convert scores to proportional
        # both
        scope_1_mean_prop_both_df = proportional_score_converter(scope_1_NACE_mean_both_df, 'by_bank')
        scope_3_source_prop_both_df = proportional_score_converter(scope_3_NACE_mean_both_df, 'by_bank')
        # imp
        scope_1_mean_prop_imp_df = proportional_score_converter(scope_1_NACE_mean_imp_df, 'by_bank')
        scope_3_source_prop_imp_df = proportional_score_converter(scope_3_NACE_mean_imp_df, 'by_bank')
        # dep
        scope_1_mean_prop_dep_df = proportional_score_converter(scope_1_NACE_mean_dep_df, 'by_bank')
        scope_3_source_prop_dep_df = proportional_score_converter(scope_3_NACE_mean_dep_df, 'by_bank')


    if grouping == 'sector' or grouping == 'region':
        # both
        scope_1_mean_prop_both_df = convert_to_percentage_by_sector_region(scope_1_NACE_mean_both_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_1_mean_prop_both_df = scope_1_mean_prop_both_df.reset_index().drop(columns='Bank').groupby(['Aggregation']).mean()
            else:
                scope_1_mean_prop_both_df = scope_1_mean_prop_both_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_1_mean_prop_both_df =  scope_1_mean_prop_both_df.reset_index().drop(columns='Bank').groupby(['region']).mean()

        scope_3_source_prop_both_df = convert_to_percentage_by_sector_region(scope_3_NACE_mean_both_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_3_source_prop_both_df = scope_3_source_prop_both_df.reset_index().drop(columns='Bank').groupby(
                    ['Aggregation']).mean()
            else:
                scope_3_source_prop_both_df = scope_3_source_prop_both_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_3_source_prop_both_df = scope_3_source_prop_both_df.reset_index().drop(columns='Bank').groupby(['region']).mean()

        # imp
        scope_1_mean_prop_imp_df = convert_to_percentage_by_sector_region(scope_1_NACE_mean_imp_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_1_mean_prop_imp_df = scope_1_mean_prop_imp_df.reset_index().drop(columns='Bank').groupby(
                    ['Aggregation']).mean()
            else:
                scope_1_mean_prop_imp_df = scope_1_mean_prop_imp_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_1_mean_prop_imp_df = scope_1_mean_prop_imp_df.reset_index().drop(columns='Bank').groupby(['region']).mean()

        scope_3_source_prop_imp_df = convert_to_percentage_by_sector_region(scope_3_NACE_mean_imp_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_3_source_prop_imp_df = scope_3_source_prop_imp_df.reset_index().drop(columns='Bank').groupby(
                    ['Aggregation']).mean()
            else:
                scope_3_source_prop_imp_df = scope_3_source_prop_imp_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_3_source_prop_imp_df = scope_3_source_prop_imp_df.reset_index().drop(columns='Bank').groupby(['region']).mean()

        # dep
        scope_1_mean_prop_dep_df = convert_to_percentage_by_sector_region(scope_1_NACE_mean_dep_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_1_mean_prop_dep_df = scope_1_mean_prop_dep_df.reset_index().drop(columns='Bank').groupby(
                    ['Aggregation']).mean()
            else:
                scope_1_mean_prop_dep_df = scope_1_mean_prop_dep_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_1_mean_prop_dep_df = scope_1_mean_prop_dep_df.reset_index().drop(columns='Bank').groupby(['region']).mean()

        scope_3_source_prop_dep_df = convert_to_percentage_by_sector_region(scope_3_NACE_mean_dep_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_3_source_prop_dep_df = scope_3_source_prop_dep_df.reset_index().drop(columns='Bank').groupby(
                    ['Aggregation']).mean()
            else:
                scope_3_source_prop_dep_df = scope_3_source_prop_dep_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_3_source_prop_dep_df = scope_3_source_prop_dep_df.reset_index().drop(columns='Bank').groupby(['region']).mean()


    # aggregate to ESS types
    # both
    # scope_1_mean_prop_both_df = aggregate_ESS_types(scope_1_mean_prop_both_df)
    # scope_3_source_prop_both_df = aggregate_ESS_types(scope_3_source_prop_both_df)
    # # imp
    # scope_1_mean_prop_imp_df = aggregate_ESS_types(scope_1_mean_prop_imp_df)
    # scope_3_source_prop_imp_df = aggregate_ESS_types(scope_3_source_prop_imp_df)
    # # dep
    # scope_1_mean_prop_dep_df = aggregate_ESS_types(scope_1_mean_prop_dep_df)
    # scope_3_source_prop_dep_df = aggregate_ESS_types(scope_3_source_prop_dep_df)
    # get the ESS type

    # drop the cultural services and order by ecosystem type
    scope_1_mean_prop_both_df = order_ESS_types(scope_1_mean_prop_both_df)
    scope_3_source_prop_both_df = order_ESS_types(scope_3_source_prop_both_df)
    # # imp
    scope_1_mean_prop_imp_df = order_ESS_types(scope_1_mean_prop_imp_df)
    scope_3_source_prop_imp_df = order_ESS_types(scope_3_source_prop_imp_df)
    # # dep
    scope_1_mean_prop_dep_df = order_ESS_types(scope_1_mean_prop_dep_df)
    scope_3_source_prop_dep_df = order_ESS_types(scope_3_source_prop_dep_df)


    # plot a heatmap for each score
    # create plot
    plt.figure(figsize=(20, 20))
    # create color scheme
    # create the color palette for plots
    imp_colors = sns.color_palette("Reds", as_cmap=True)
    dep_colors = sns.color_palette("Blues", as_cmap=True)
    both_colors = sns.color_palette("Purples", as_cmap=True)

    # scope 1
    # create subplot - imp scope 1
    ax = plt.subplot(3, 2, 1)
    # plot the heatmap
    numeric_df = scope_1_mean_prop_imp_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=imp_colors, xticklabels=False, yticklabels=True)
    scope_1_mean_prop_imp_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_1_sector_heatmap_imp_data.csv')
    # set title and the label size
    ax.set_title(f'Direct Operations - Impact')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.setp(ax.get_yticklabels(),
             rotation=0,  # horizontal
             ha='right',  # align labels to the right of the tick
             rotation_mode='anchor')    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)
    # create subplot - both scope 1
    ax = plt.subplot(3, 2, 3)
    # plot the heatmap
    numeric_df = scope_1_mean_prop_dep_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=dep_colors, xticklabels=False, yticklabels=True)
    scope_1_mean_prop_dep_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_1_sector_heatmap_dep_data.csv')
    # set title and the label size
    ax.set_title(f'Direct Operations - Dependency')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # create subplot - both scope 1
    ax = plt.subplot(3, 2, 5)
    # plot the heatmap
    numeric_df = scope_1_mean_prop_both_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=both_colors, xticklabels=False, yticklabels=True)
    scope_1_mean_prop_both_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_1_sector_heatmap_both_data.csv')
    # set title and the label size
    ax.set_title(f'Direct Operations - Endogenous Risk')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # --- Subgroup labels (bottom row labels) ---
    ax.set_xticks(np.arange(len(numeric_df.columns)) + 0.5)
    ax.set_xticklabels(
        numeric_df.columns.get_level_values(0),
        rotation=90,
        ha="right"
    )

    # Move subgroup labels slightly down
    ax.tick_params(axis='x', pad=15)

    # --- Category labels (second row further down) ---
    groups = numeric_df.columns.get_level_values(1)
    n_cols = len(numeric_df.columns)

    for cat in groups.unique():
        idx = np.where(groups == cat)[0]
        center = idx.mean()
        x_axes = (center + 0.5) / n_cols  # normalized axes position

        # Nudge "Provisioning" label to the left
        if cat == "Provisioning":
            x_axes -= 0.05  # increase this value to move further left

        ax.text(
            x_axes,
            -1.9,
            cat,
            ha='center',
            va='top',
            fontsize=17,
            fontweight='bold',
            transform=ax.transAxes
        )


    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # create subplot - imp scope 3
    ax = plt.subplot(3, 2, 2)
    # plot the heatmap
    numeric_df = scope_3_source_prop_imp_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=imp_colors, xticklabels=False, yticklabels=False)
    scope_3_source_prop_imp_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_sector_heatmap_imp_data.csv')
    # set title and the label size
    ax.set_title(f'Upstream Supply Chain - Impact')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.setp(ax.get_yticklabels(),
             rotation=0,  # horizontal
             ha='right',  # align labels to the right of the tick
             rotation_mode='anchor')    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # create subplot - dep scope 3
    ax = plt.subplot(3, 2, 4)
    # plot the heatmap
    numeric_df = scope_3_source_prop_dep_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=dep_colors, xticklabels=False, yticklabels=False)
    scope_3_source_prop_dep_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_sector_heatmap_dep_data.csv')
    # set title and the label size
    ax.set_title(f'Upstream Supply Chain - Dependency')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)


    # create subplot - both scope 3
    ax = plt.subplot(3, 2, 6)
    # plot the heatmap
    numeric_df = scope_3_source_prop_both_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=both_colors, xticklabels=False, yticklabels=False)
    scope_3_source_prop_both_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_sector_heatmap_both_data.csv')
    # set title and the label size
    ax.set_title(f'Upstream Supply Chain - Endogenous Risk')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # --- Subgroup labels (bottom row labels) ---
    ax.set_xticks(np.arange(len(numeric_df.columns)) + 0.5)
    ax.set_xticklabels(
        numeric_df.columns.get_level_values(0),
        rotation=90,
        ha="right"
    )

    # Move subgroup labels slightly down
    ax.tick_params(axis='x', pad=15)

    # --- Category labels (second row further down) ---
    groups = numeric_df.columns.get_level_values(1)
    n_cols = len(numeric_df.columns)

    for cat in groups.unique():
        idx = np.where(groups == cat)[0]
        center = idx.mean()
        x_axes = (center + 0.5) / n_cols  # normalized axes position
        # Nudge "Provisioning" label to the left
        if cat == "Provisioning":
            x_axes -= 0.05  # increase this value to move further left

        ax.text(
            x_axes,
            -1.9,
            cat,
            ha='center',
            va='top',
            fontsize=17,
            fontweight='bold',
            transform=ax.transAxes
        )

    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)



    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # adjust the layout to fit figures
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.25)
    # save figure
    plt.savefig(f'{value_at_risk_figure_saving_path}/Impact Dependency Endogenous Risk Sector Heatmap Flipped')
    plt.show()
    plt.close()

    return None


def plot_both_heatmaps_system(grouping, aggregation):
    """
    Create 6 heatmaps of overlap for the system
    :return:
    """
    # get aggregated scope 1 and scope 3 source + value chain
    # both
    scope_3_source_mean_both_df = pd.read_csv(GSIB_system_var_both_scope3_source_mean, index_col = [0, 1, 2], header=[0])
    scope_1_mean_both_df = pd.read_csv(GSIB_system_var_both_scope1_mean, index_col = [0, 1, 2], header=[0])

    # get the ESS
    services = np.unique(scope_1_mean_both_df.columns).tolist()

    # sector
    # NACE_converter = generate_converter_sector()
    NACE_converter = pd.read_csv(NACE_converter_path, header=[0], index_col=[0])

    # merge the NACE converter and the score and group the sectors by their higher-level classification for
    # display purposes
    # scope 1 both
    scope_1_NACE_mean_both_df = pd.merge(scope_1_mean_both_df.reset_index().drop(columns=['region']).set_index(['sector']), NACE_converter, right_index=True, left_index=True).groupby(['Code', 'Bank']).sum()
    # scope_1_NACE_mean_both_df = scope_1_NACE_mean_both_df.rename(columns={'Code': 'Sector'})


    # scope 3 - source
    # both
    scope_3_NACE_mean_both_df = pd.merge(scope_3_source_mean_both_df.reset_index().drop(columns=['region']).set_index(['sector']), NACE_converter, right_index=True, left_index=True).groupby(
        ['Code', 'Bank']).sum()
    # scope_3_NACE_mean_both_df = scope_3_NACE_mean_both_df.rename(columns={'Code': 'Sector'})



    # aggregate into 10 sectors
    if aggregation == 'SNA':
        #both
        scope_1_NACE_mean_both_df = aggregate_to_SNA(scope_1_NACE_mean_both_df)
        scope_3_NACE_mean_both_df = aggregate_to_SNA(scope_3_NACE_mean_both_df)


    if grouping == '':
        # convert scores to proportional
        # both
        scope_1_mean_prop_both_df = proportional_score_converter(scope_1_NACE_mean_both_df, 'total')
        scope_3_source_prop_both_df = proportional_score_converter(scope_3_NACE_mean_both_df, 'total')

    if grouping == 'sector' or grouping == 'region':
        # both
        scope_1_mean_prop_both_df = convert_to_percentage_by_sector_region(scope_1_NACE_mean_both_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_1_mean_prop_both_df = scope_1_mean_prop_both_df.reset_index().drop(columns='Bank').groupby(['Aggregation']).mean()
            else:
                scope_1_mean_prop_both_df = scope_1_mean_prop_both_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_1_mean_prop_both_df =  scope_1_mean_prop_both_df.reset_index().drop(columns='Bank').groupby(['region']).mean()

        scope_3_source_prop_both_df = convert_to_percentage_by_sector_region(scope_3_NACE_mean_both_df, grouping)
        # average across banks
        if grouping == 'sector':
            if aggregation == 'SNA':
                scope_3_source_prop_both_df = scope_3_source_prop_both_df.reset_index().drop(columns='Bank').groupby(
                    ['Aggregation']).mean()
            else:
                scope_3_source_prop_both_df = scope_3_source_prop_both_df.reset_index().drop(columns='Bank').groupby(['Code']).mean()
        elif grouping == 'region':
            scope_3_source_prop_both_df = scope_3_source_prop_both_df.reset_index().drop(columns='Bank').groupby(['region']).mean()


    # aggregate to ESS types
    # both
    # scope_1_mean_prop_both_df = aggregate_ESS_types(scope_1_mean_prop_both_df)
    # scope_3_source_prop_both_df = aggregate_ESS_types(scope_3_source_prop_both_df)
    # # imp
    # scope_1_mean_prop_imp_df = aggregate_ESS_types(scope_1_mean_prop_imp_df)
    # scope_3_source_prop_imp_df = aggregate_ESS_types(scope_3_source_prop_imp_df)
    # # dep
    # scope_1_mean_prop_dep_df = aggregate_ESS_types(scope_1_mean_prop_dep_df)
    # scope_3_source_prop_dep_df = aggregate_ESS_types(scope_3_source_prop_dep_df)
    # get the ESS type

    # drop the cultural services and order by ecosystem type
    scope_1_mean_prop_both_df = order_ESS_types(scope_1_mean_prop_both_df)
    scope_3_source_prop_both_df = order_ESS_types(scope_3_source_prop_both_df)


    # plot a heatmap for each score
    # create plot
    plt.figure(figsize=(20, 20))
    # create color scheme
    # create the color palette for plots
    both_colors = sns.color_palette("Purples", as_cmap=True)

    # scope 1
    # create subplot - imp scope 1
    ax = plt.subplot(3, 2, 5)
    # plot the heatmap
    numeric_df = scope_1_mean_prop_both_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=both_colors, xticklabels=False, yticklabels=True)
    scope_1_mean_prop_both_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_1_sector_heatmap_both_data_system.csv')
    # set title and the label size
    ax.set_title(f'Direct Operations - Endogenous Risk')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # --- Subgroup labels (bottom row labels) ---
    ax.set_xticks(np.arange(len(numeric_df.columns)) + 0.5)
    ax.set_xticklabels(
        numeric_df.columns.get_level_values(0),
        rotation=90,
        ha="right"
    )

    # Move subgroup labels slightly down
    ax.tick_params(axis='x', pad=15)

    # --- Category labels (second row further down) ---
    groups = numeric_df.columns.get_level_values(1)
    n_cols = len(numeric_df.columns)

    for cat in groups.unique():
        idx = np.where(groups == cat)[0]
        center = idx.mean()
        x_axes = (center + 0.5) / n_cols  # normalized axes position

        # Nudge "Provisioning" label to the left
        if cat == "Provisioning":
            x_axes -= 0.05  # increase this value to move further left

        ax.text(
            x_axes,
            -1.9,
            cat,
            ha='center',
            va='top',
            fontsize=17,
            fontweight='bold',
            transform=ax.transAxes
        )


    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # create subplot - both scope 3
    ax = plt.subplot(3, 2, 6)
    # plot the heatmap
    numeric_df = scope_3_source_prop_both_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df, ax=ax, cmap=both_colors, xticklabels=False, yticklabels=False)
    scope_3_source_prop_both_df.T.to_csv(f'{value_at_risk_figure_saving_path}/scope_3_sector_heatmap_both_data_system.csv')
    # set title and the label size
    ax.set_title(f'Upstream Supply Chain - Endogenous Risk')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # --- Subgroup labels (bottom row labels) ---
    ax.set_xticks(np.arange(len(numeric_df.columns)) + 0.5)
    ax.set_xticklabels(
        numeric_df.columns.get_level_values(0),
        rotation=90,
        ha="right"
    )

    # Move subgroup labels slightly down
    ax.tick_params(axis='x', pad=15)

    # --- Category labels (second row further down) ---
    groups = numeric_df.columns.get_level_values(1)
    n_cols = len(numeric_df.columns)

    for cat in groups.unique():
        idx = np.where(groups == cat)[0]
        center = idx.mean()
        x_axes = (center + 0.5) / n_cols  # normalized axes position
        # Nudge "Provisioning" label to the left
        if cat == "Provisioning":
            x_axes -= 0.05  # increase this value to move further left

        ax.text(
            x_axes,
            -1.9,
            cat,
            ha='center',
            va='top',
            fontsize=17,
            fontweight='bold',
            transform=ax.transAxes
        )

    # set label size
    items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if ax.legend_:
        items += ax.legend_.get_texts()  # Get legend text elements
    for item in items:
        item.set_fontsize(17)



    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Increase tick size
    cbar.ax.tick_params(labelsize=20)

    # adjust the layout to fit figures
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.25)
    # save figure
    plt.savefig(f'{value_at_risk_figure_saving_path}/System Endogenous Risk Sector Heatmap Flipped')
    plt.show()
    plt.close()

    return None


def ecosystem_maps_by_region(service, grouping):
    """
    A function that plots the regional spread of select ecosystem service separated by region.
    :param both_mean: the endogenous risk score for portfolio-level
    :param service: the ecosystem service you wish to map
    :return: NA
    """

    # get aggregated scope 1 and scope 3 source + value chain
    # both
    scope_3_source_mean_both_df, scope_3_value_chain_mean_both_df = aggregate_banks_scope_3('mean', 'Both', False)
    scope_1_mean_both_df, scope_1_min_both_df, scope_1_max_both_df = aggregate_scope_1('Both', False)


    # get world map as a basemap
    worldmap = gpd.read_file(natural_earth_w_countries)
    # EXIO to country
    EXIO_to_countries_df = pd.read_csv(EXIO_regions_to_world_countries, index_col=0)
    EXIO_to_countries_df.drop(columns=['usa_state_code', 'usa_state_latitude', 'usa_state_longitude', 'usa_state'],
                      inplace=True)
    EXIO_to_countries_df = EXIO_to_countries_df.dropna()

    # get regions
    regions = ['North America', 'Europe', 'Asia']
    bank_regions_df = pd.read_csv(GSIB_bank_regions, header=[0], index_col=[0])
    bank_regions_dict = bank_regions_df['Region'].to_dict()

    if grouping == '':
        # organize scores
        scope_1_var_finance = scope_1_mean_both_df.copy()
        scope_1_var_finance.name = f'Scope 1 Overlap mean'
        scope_3_var_finance = scope_3_source_mean_both_df.copy()
        scope_3_var_finance.name = f'Scope 3 Overlap Source mean'
        scope_3_var_finance_value_chain= scope_3_value_chain_mean_both_df.copy()
        scope_3_var_finance_value_chain.name = f'Scope 3 Overlap Value Chain mean'


        # get financial data
        financial_data = finance_GSIB_reformat()
        # get the regional total for domiciles and the regional proportion of portfolio value
        financial_data_region = pd.merge(bank_regions_df.reset_index(), financial_data.reset_index(), right_on=['Bank'],
                                         left_on=['Bank']).drop(
            columns=['Bank', 'Total Loan', 'Proportion of Loans']).groupby(['Region', 'region', 'sector']).sum()
        regional_financial_total = pd.merge(bank_regions_df.reset_index(), financial_data.reset_index(),
                                            right_on=['Bank'],
                                            left_on=['Bank']).drop(
            columns=['Bank', 'Total Loan', 'Proportion of Loans', 'sector', 'region']).groupby(['Region']).sum().rename(
            columns={'EUR m adjusted': 'Total Loan'})
        financial_data_region_w_total = pd.merge(regional_financial_total.reset_index(),
                                                 financial_data_region.reset_index(), right_on=['Region'],
                                                 left_on=['Region'])
        financial_data_region_w_total['Proportional'] = financial_data_region_w_total['EUR m adjusted'] / \
                                                        financial_data_region_w_total['Total Loan']

    elif grouping == 'region':
        scope_1_var_finance = convert_to_percentage_by_sector_region(scope_1_mean_both_df, grouping)
        scope_1_var_finance = pd.merge(bank_regions_df.reset_index(), scope_1_var_finance.reset_index(),
                                              right_on=['Bank'],
                                              left_on=['Bank'])
        scope_1_var_finance.name = 'Scope 1 Overlap mean'
        scope_3_var_finance_value_chain = convert_to_percentage_by_sector_region(scope_3_source_mean_both_df, grouping)
        scope_3_var_finance_value_chain = pd.merge(bank_regions_df.reset_index(), scope_3_var_finance_value_chain.reset_index(),
                                              right_on=['Bank'],
                                              left_on=['Bank'])
        scope_3_var_finance_value_chain.name = 'Scope 3 Overlap Value Chain mean'





    # create subplots
    plt.subplots(3, 2, figsize=(20, 20), sharey=True)

    i = 1
    # loop through domicile regions
    for region in regions:

        if grouping == '':
            # aggregate the results by the domicile region and calculate the proportion of domicile portfolio exposed
            # direct operations
            df = scope_1_var_finance.copy()
            score_name = scope_1_var_finance.name
            df_w_region = pd.merge(df.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'],
                                   left_on=['Bank']).set_index(['Region', 'Bank', 'region', 'sector'])
            prop_df = df_w_region.reset_index().drop(columns=['Bank', 'sector']).groupby(['Region', 'region']).sum()
            prop_df = prop_df.reset_index()[prop_df.reset_index()['Region'] == region].set_index(['Region', 'region'])
            prop_df = (prop_df / regional_financial_total.loc[region]['Total Loan'])
            prop_df.name = score_name
            scope_1_var_finance_region = prop_df.copy()

            # upstream supply chain - value chain
            df = scope_3_var_finance_value_chain.copy()
            score_name = scope_3_var_finance_value_chain.name
            df_w_region = pd.merge(df.reset_index(), bank_regions_df.reset_index(), right_on=['Bank'],
                                   left_on=['Bank']).set_index(['Region', 'Bank', 'region'])
            prop_df = df_w_region.reset_index().drop(columns=['Bank']).groupby(['Region', 'region']).sum()
            prop_df = prop_df.reset_index()[prop_df.reset_index()['Region'] == region].set_index(['Region', 'region'])
            prop_df = (prop_df / regional_financial_total.loc[region]['Total Loan'])
            prop_df.name = score_name
            scope_3_var_finance_region = prop_df.copy()

        if grouping == 'region':
            # scope 1
            df = scope_1_var_finance.copy()
            df = df.drop(columns=['Bank']).groupby(['Region', 'region']).mean()
            scope_1_var_finance_region = df.copy()
            # scope 3
            df = scope_3_var_finance_value_chain.copy()
            df = df.drop(columns=['Bank']).groupby(['Region', 'region']).mean()
            scope_3_var_finance_region = df.copy()

        # merge the country name
        # scope 3
        country_df_merge_scp3 = pd.merge(scope_3_var_finance_region.reset_index(), EXIO_to_countries_df, left_on='region',
                                         right_on='country_code')
        country_df_merge_scp3.replace(["United States"], ["United States of America"], inplace=True)
        country_df_merge_scp3 = worldmap.merge(country_df_merge_scp3, left_on='NAME', right_on='country')

        # scope 1
        country_df_merge_scp1 = pd.merge(scope_1_var_finance_region.reset_index(), EXIO_to_countries_df, right_on='country_code',
                                         left_on='region')
        country_df_merge_scp1.replace(["United States"], ["United States of America"], inplace=True)
        country_df_merge_scp1 = worldmap.merge(country_df_merge_scp1, left_on='NAME', right_on='country')

        # Creating axes and plotting world map
        # scope 1
        ax = plt.subplot(3, 2, i)
        i = i + 1
        vmin, vmax, vcenter = country_df_merge_scp1[service].min(), country_df_merge_scp1[service].max(), 0
        worldmap.plot(color="lightgrey", ax=ax)
        country_df_merge_scp1.plot(column=service, cmap="Purples", linewidth=0.4, ax=ax, legend=True)
        # worldmap.boundary.plot(zorder=1)
        ax.set_title(f'GSIB Direct Operations - {region} \n {service}')
        ax.title.set_size(20)

        items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.legend_:
            items += ax.legend_.get_texts()  # Get legend text elements
        for item in items:
            item.set_fontsize(20)
        # Access the colorbar and adjust its font size
        cbar = ax.get_figure().axes[-1]  # Get the last axis (colorbar)
        cbar.tick_params(labelsize=20)  # Set tick label size
        # cbar.set_ylabel(service, fontsize=20)  # Set colorbar label size

        plt.xlim([-180, 180])
        plt.ylim([-90, 90])

        # scope 3
        ax = plt.subplot(3, 2, i)
        i = i + 1
        worldmap.plot(color="lightgrey", ax=ax)
        country_df_merge_scp3.plot(column=service, cmap="Purples", linewidth=0.4, ax=ax, legend=True)
        # worldmap.boundary.plot(zorder=1)
        ax.set_title(f'GSIB Upstream Supply Chain - {region} \n {service}')
        ax.title.set_size(20)
        plt.tight_layout()
        plt.xlim([-180, 180])
        plt.ylim([-90, 90])

        items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.legend_:
            items += ax.legend_.get_texts()  # Get legend text elements
        for item in items:
            item.set_fontsize(20)
        # Access the colorbar and adjust its font size
        cbar = ax.get_figure().axes[-1]  # Get the last axis (colorbar)
        cbar.tick_params(labelsize=20)  # Set tick label size
        # cbar.set_ylabel(service, fontsize=20)  # Set colorbar label size

    plt.savefig(f'{value_at_risk_figure_saving_path}/Overlap/Portfolio-level Endogenous risk Maps by region {service}')
    plt.show()
    plt.close()

    return None


def ecosystem_maps_system(service_list, grouping):
    """
    A function that plots the regional spread of select ecosystem service separated by region.
    :param both_mean: the endogenous risk score for system-level
    :param service: the ecosystem service you wish to map
    :return: NA
    """
    # get aggregated scope 1 and scope 3 source + value chain
    # both
    # scope_3_source_mean_both_df, scope_3_value_chain_mean_both_df = aggregate_banks_scope_3('mean', 'Both', False)
    # scope_1_mean_both_df, scope_1_min_both_df, scope_1_max_both_df = aggregate_scope_1('Both', False)
    scope_3_source_mean_both_df = pd.read_csv(GSIB_system_var_both_scope3_source_mean, index_col=[0, 1, 2], header=[0])
    scope_1_mean_both_df = pd.read_csv(GSIB_system_var_both_scope1_mean, index_col=[0, 1, 2], header=[0])
    scope_3_value_chain_mean_both_df = pd.read_csv(GSIB_system_var_both_scope3_value_chain_mean, index_col=[0, 1], header=[0])


    if grouping == '':
        # convert scores to proportional
        scope_3_source_prop_df = proportional_score_converter(scope_3_source_mean_both_df, 'total')
        scope_3_value_chain_prop_df = proportional_score_converter(scope_3_value_chain_mean_both_df, 'total')
        scope_1_mean_prop_df = proportional_score_converter(scope_1_mean_both_df, 'total')

        scope_1_var_finance = scope_1_mean_prop_df.copy()
        scope_1_var_finance.name = 'Scope 1 Overlap mean'
        scope_3_var_finance_value_chain = scope_3_value_chain_prop_df.copy()
        scope_3_var_finance_value_chain.name = 'Scope 3 Overlap Value Chain mean'

        scope_1_var_finance_region = aggregate_to_region_service(scope_1_var_finance, 'sum')
        # scope_3_var_finance_value_chain_region = aggregate_to_region_service(scope_3_var_finance_value_chain, 'sum')
    elif grouping == 'region':
        scope_1_var_finance_region = convert_to_percentage_by_sector_region(scope_1_mean_both_df, grouping)
        # scope_1_var_finance_region = scope_1_var_finance_region.reset_index().drop(columns='Bank').groupby(['region']).mean()
        scope_1_var_finance_region.name = 'Scope 1 Overlap mean'
        scope_3_var_finance_value_chain = convert_to_percentage_by_sector_region(scope_3_source_mean_both_df, grouping)
        # scope_3_var_finance_value_chain = scope_3_var_finance_value_chain.reset_index().drop(columns='Bank').groupby(['region']).mean()
        scope_3_var_finance_value_chain.name = 'Scope 3 Overlap Value Chain mean'


    # get world map as a basemap
    worldmap = gpd.read_file(natural_earth_w_countries)
    # EXIO to country
    EXIO_to_countries_df = pd.read_csv(EXIO_regions_to_world_countries, index_col=0)
    EXIO_to_countries_df.drop(columns=['usa_state_code', 'usa_state_latitude', 'usa_state_longitude', 'usa_state'],
                      inplace=True)
    EXIO_to_countries_df = EXIO_to_countries_df.dropna()




    num_service = len(service_list)

    # create subplots
    plt.subplots(num_service, 2, figsize=(20, 20), sharey=True)

    i = 1
    for service in service_list:
        # merge the country name
        # scope 3
        country_df_merge_scp3 = pd.merge(scope_3_var_finance_value_chain.reset_index(), EXIO_to_countries_df, left_on='region',
                                         right_on='country_code')
        country_df_merge_scp3.replace(["United States"], ["United States of America"], inplace=True)
        country_df_merge_scp3 = worldmap.merge(country_df_merge_scp3, left_on='NAME', right_on='country')

        # scope 1
        country_df_merge_scp1 = pd.merge(scope_1_var_finance_region.reset_index(), EXIO_to_countries_df, right_on='country_code',
                                         left_on='region')
        country_df_merge_scp1.replace(["United States"], ["United States of America"], inplace=True)
        country_df_merge_scp1 = worldmap.merge(country_df_merge_scp1, left_on='NAME', right_on='country')

        # Creating axes and plotting world map
        # scope 1
        ax = plt.subplot(num_service, 2, i)
        i = i + 1
        # vmin, vmax, vcenter = country_df_merge_scp1[service].min(), country_df_merge_scp1[service].max(), 0
        worldmap.plot(color="lightgrey", ax=ax)
        country_df_merge_scp1.plot(column=service, cmap="Purples", linewidth=0.4, ax=ax, legend=True)
        # worldmap.boundary.plot(zorder=1)
        ax.set_title(f'G-SIB Portfolio Direct Operations \n {service}')
        ax.title.set_size(20)

        items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.legend_:
            items += ax.legend_.get_texts()  # Get legend text elements
        for item in items:
            item.set_fontsize(20)
        # Access the colorbar and adjust its font size
        cbar = ax.get_figure().axes[-1]  # Get the last axis (colorbar)
        cbar.tick_params(labelsize=20)  # Set tick label size
        # cbar.set_ylabel(service, fontsize=20)  # Set colorbar label size

        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        # Get the bounding box of the map plot
        pos = ax.get_position().bounds  # (left, bottom, width, height)

        # scope 3
        ax = plt.subplot(num_service, 2, i)
        i = i + 1
        worldmap.plot(color="lightgrey", ax=ax)
        country_df_merge_scp3.plot(column=service, cmap="Purples", linewidth=0.4, ax=ax, legend=True)
        # worldmap.boundary.plot(zorder=1)
        ax.set_title(f'G-SIB Portfolio Upstream Supply Chain \n {service}')
        ax.title.set_size(20)
        plt.tight_layout()
        plt.xlim([-180, 180])
        plt.ylim([-90, 90])


        items = [ax.title, ax.yaxis.label, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.legend_:
            items += ax.legend_.get_texts()  # Get legend text elements
        for item in items:
            item.set_fontsize(20)
        # Access the colorbar and adjust its font size
        cbar = ax.get_figure().axes[-1]  # Get the last axis (colorbar)
        cbar.tick_params(labelsize=20)  # Set tick label size
        # cbar.set_ylabel(service, fontsize=20)  # Set colorbar label size

    plt.savefig(f'{value_at_risk_figure_saving_path}/Overlap/System-level Endogenous risk Maps by region {service}')
    plt.show()
    plt.close()

    return None


def convert_to_percentage_by_sector_region(score_df, grouping):
    """
    Takes a score and converts it to the percentage lost by sector rather than by bank portfolios
    :param score: a score with either region and sector or just sector that can be converted into sector percentage
    :return: a score that is in sector percentage
    """
    if grouping == 'sector':
        drop_col = 'region'
    elif grouping == 'region':
        drop_col = 'sector'

    # get the % values for the difference - format bank data
    financial_data_df = finance_GSIB_reformat()
    financial_data_clean_df = financial_data_df.reset_index().drop(
        columns=['Total Loan', 'Proportion of Loans'])

    # get NACE converter
    NACE_converter = pd.read_csv(NACE_converter_path, header=[0], index_col=[0])


    # get the NACE sector aggregation
    NACE_sector_agg_df = pd.read_csv(NACE_letters_path, header=[0], index_col=[0]).drop(columns=['Description'])
    order = NACE_sector_agg_df.reset_index()['Aggregation'].drop_duplicates()
    NACE_sector_agg_df['Aggregation'] = pd.Categorical(NACE_sector_agg_df['Aggregation'], categories=order,
                                                       ordered=True)

    # aggregate the financial data into appropriate aggregation for sector
    if grouping == 'sector':
        financial_data_clean_df = pd.merge(
            financial_data_clean_df.reset_index().drop(columns=['region']).set_index(['sector']),
            NACE_converter, right_index=True, left_index=True).groupby(['Code', 'Bank']).sum()
        if 'Aggregation' in score_df.reset_index().columns:
            financial_data_clean_df = aggregate_to_SNA(financial_data_clean_df)
            financial_data_clean_df = financial_data_clean_df.drop(columns='index')
            if grouping == 'sector':
                grouping = 'Aggregation'

    # for system - get the proportion of the system
    if 'System' == score_df.reset_index()['Bank'].iloc[0]:
        df = score_df.copy()
        financial_data_clean_df = financial_data_clean_df.drop(columns=['Bank']).groupby(['sector', 'region']).sum()
        sector_df = pd.merge(df.reset_index(), financial_data_clean_df.reset_index(), left_on=['sector', 'region'], right_on=['sector', 'region'])
        sector_df = sector_df.drop(columns=[drop_col, 'Bank']).groupby([grouping]).sum()
        # divide by SYSTEM TOTAL
        for ESS in score_df.columns:
            sector_df[ESS] = sector_df[ESS]/sector_df['EUR m adjusted']
        sector_df = sector_df.drop(columns='EUR m adjusted')
        # sector_df = sector_df * 100
        return sector_df


    # merge the total bank values with the absolute results values
    df = score_df.reset_index().copy()
    df_2 = financial_data_clean_df.copy()
    if 'region' in df.columns and 'sector' in df.columns:
        finance_score_merge_df = df.merge(df_2, left_on=['Bank', grouping, drop_col], right_on=['Bank', grouping, drop_col],
                                          how='left').drop(columns=drop_col).groupby(['Bank', grouping]).sum()
    elif 'region' not in df.columns and (grouping == 'sector' or grouping == 'Aggregation'):
        df_2 = df_2.groupby(['Bank', grouping]).sum()
        finance_score_merge_df = df.merge(df_2, left_on=['Bank', grouping],
                                          right_on=['Bank', grouping],
                                          how='left').set_index(['Bank', grouping])
    elif 'region' not in df.columns and grouping == 'region':
        return None
    elif 'region' in df.columns and 'sector' not in df.columns and grouping == 'region':
        finance_score_merge_df = df.merge(df_2, left_on=['Bank', grouping],
                                          right_on=['Bank', grouping],
                                          how='left').drop(columns=drop_col).groupby(['Bank', grouping]).sum()

    for ESS in score_df.columns:
        finance_score_merge_df[ESS] = finance_score_merge_df[ESS].div(finance_score_merge_df['EUR m adjusted'].replace(0, np.nan)).fillna(0)

    # multiply by 100 to get the % of the portfolio exposed
    finance_score_merge_df = finance_score_merge_df.drop(columns='EUR m adjusted')

    return finance_score_merge_df



def aggregate_to_SNA(score_df):
    # get the NACE sector aggregation
    NACE_sector_agg_df = pd.read_csv(NACE_letters_path, header=[0], index_col=[0]).drop(columns=['Description'])
    order = NACE_sector_agg_df.reset_index()['Aggregation'].drop_duplicates()
    NACE_sector_agg_df['Aggregation'] = pd.Categorical(NACE_sector_agg_df['Aggregation'], categories=order,
                                                       ordered=True)

    df = score_df.copy()

    # aggregate sectors
    if 'region' not in df.T.reset_index().columns:
        score_agg_df = pd.merge(df.reset_index(), NACE_sector_agg_df.reset_index(), left_on=['Code'],
                             right_on=['Code'], how='left').drop(columns=['Code']).groupby(['Aggregation', 'Bank']).mean()

    elif 'region' in df.T.reset_index().columns:
        df = df.T
        score_agg_df = pd.merge(df.reset_index(), NACE_sector_agg_df.reset_index(), left_on=['Code'],
                             right_on=['Code'], how='left').drop(columns=['Code']).groupby(['Aggregation', 'region', 'Bank']).mean()

        # score_agg_df.name = score_df.name

    return score_agg_df

def aggregate_ESS_types(score_df):
    # get the ESS type
    ESS_types_df = pd.read_csv(ESS_types_path, header=[0], index_col=[0])

    df = score_df.copy()
    if 'region' not in df.T.reset_index().columns:
        merged_long_df = pd.melt(df.reset_index(), id_vars='Aggregation', value_vars=df.columns.tolist(),
                                 var_name='Ecosystem Services', value_name='score')
        # aggregate ecosystem services
        merged_ess_df = pd.merge(merged_long_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                 right_on=['Ecosystem Services'], how='left').drop(
            columns='Ecosystem Services').groupby(['Ecosystem Services Type', 'Aggregation']).mean()
        score_agg_df = merged_ess_df.reset_index().pivot(index='Aggregation', columns='Ecosystem Services Type',
                                                         values='score')
    elif 'region' in df.T.reset_index().columns:
        merged_long_df = pd.melt(df.reset_index(), id_vars=['Aggregation', 'region'],
                                 value_vars=df.columns.tolist(),
                                 var_name='Ecosystem Services', value_name='score')
        # aggregate ecosystem services
        merged_ess_df = pd.merge(merged_long_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                 right_on=['Ecosystem Services'], how='left').drop(
            columns='Ecosystem Services').groupby(['Ecosystem Services Type', 'Aggregation', 'region']).mean()
        score_agg_df = merged_ess_df.reset_index().pivot(index=['Aggregation', 'region'],
                                                     columns='Ecosystem Services Type',
                                                     values='score')

    return score_agg_df

def order_ESS_types(score_df):
    # get the ESS type
    ESS_types_df = pd.read_csv(ESS_types_path, header=[0], index_col=[0])
    ESS_order = ['Provisioning', 'Regulating and maintenance']
    df = score_df.copy()
    if 'Bank' not in df.columns:
        if 'region' not in df.T.reset_index().columns:
            merged_long_df = pd.melt(df.reset_index(), id_vars='Aggregation', value_vars=df.columns.tolist(),
                                     var_name='Ecosystem Services', value_name='score')
            # order ecosystem services
            merged_ess_df = pd.merge(merged_long_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                     right_on=['Ecosystem Services'], how='left')
            merged_ess_no_culture_df = merged_ess_df[merged_ess_df['Ecosystem Services Type'] != 'Cultural']
            merged_ess_no_culture_df['Ecosystem Services Type'] = pd.Categorical(
                merged_ess_no_culture_df['Ecosystem Services Type'],
                categories=ESS_order,
                ordered=True
            )
            merged_ess_no_culture_df = merged_ess_no_culture_df.sort_values('Ecosystem Services Type')
            # merged_ess_no_culture_df = merged_ess_no_culture_df.drop(columns='Ecosystem Services Type')
            score_agg_df = merged_ess_no_culture_df.pivot(index=['Aggregation'], columns=['Ecosystem Services', 'Ecosystem Services Type'], values='score')
        elif 'region' in df.T.reset_index().columns:
            merged_long_df = pd.melt(df.reset_index(), id_vars=['Aggregation', 'region'],
                                     value_vars=df.columns.tolist(),
                                     var_name='Ecosystem Services', value_name='score')
            # aggregate ecosystem services
            merged_ess_df = pd.merge(merged_long_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                     right_on=['Ecosystem Services'], how='left')

            # order ecosystem services
            merged_ess_df = pd.merge(merged_ess_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                     right_on=['Ecosystem Services'], how='left')
            merged_ess_no_culture_df = merged_ess_df[merged_ess_df['Ecosystem Services Type'] != 'Cultural']
            merged_ess_no_culture_df['Ecosystem Services Type'] = pd.Categorical(
                merged_ess_no_culture_df['Ecosystem Services Type'],
                categories=ESS_order,
                ordered=True
            )
            merged_ess_no_culture_df = merged_ess_no_culture_df.sort_values('Ecosystem Services Type')
            # merged_ess_no_culture_df = merged_ess_no_culture_df.drop(columns='Ecosystem Services Type')
            score_agg_df = merged_ess_no_culture_df.pivot(index=['Aggregation', 'region'],
                                                                        columns=['Ecosystem Services', 'Ecosystem Services Type'],
                                                                        values='score')
    elif 'Bank' in df.columns:
        if 'region' not in df.T.reset_index().columns:
            merged_long_df = pd.melt(df.reset_index(), id_vars=['Aggregation', 'Bank'], value_vars=df.columns.tolist(),
                                     var_name='Ecosystem Services', value_name='score')
            # order ecosystem services
            merged_ess_df = pd.merge(merged_long_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                     right_on=['Ecosystem Services'], how='left')
            merged_ess_no_culture_df = merged_ess_df[merged_ess_df['Ecosystem Services Type'] != 'Cultural']
            merged_ess_no_culture_df['Ecosystem Services Type'] = pd.Categorical(
                merged_ess_no_culture_df['Ecosystem Services Type'],
                categories=ESS_order,
                ordered=True
            )
            merged_ess_no_culture_df = merged_ess_no_culture_df.sort_values('Ecosystem Services Type')
            # merged_ess_no_culture_df = merged_ess_no_culture_df.drop(columns='Ecosystem Services Type')
            score_agg_df = merged_ess_no_culture_df.pivot(index=['Aggregation', 'Bank'],
                                                          columns=['Ecosystem Services', 'Ecosystem Services Type'],
                                                          values='score')
            score_agg_df = score_agg_df.reset_index().drop(columns='Bank').groupby(['Aggregation']).mean()
        elif 'region' in df.T.reset_index().columns:
            merged_long_df = pd.melt(df.reset_index(), id_vars=['Aggregation', 'region', 'Bank'],
                                     value_vars=df.columns.tolist(),
                                     var_name='Ecosystem Services', value_name='score')
            # aggregate ecosystem services
            merged_ess_df = pd.merge(merged_long_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                     right_on=['Ecosystem Services'], how='left')

            # order ecosystem services
            merged_ess_df = pd.merge(merged_ess_df, ESS_types_df.reset_index(), left_on=['Ecosystem Services'],
                                     right_on=['Ecosystem Services'], how='left')
            merged_ess_no_culture_df = merged_ess_df[merged_ess_df['Ecosystem Services Type'] != 'Cultural']
            merged_ess_no_culture_df['Ecosystem Services Type'] = pd.Categorical(
                merged_ess_no_culture_df['Ecosystem Services Type'],
                categories=ESS_order,
                ordered=True
            )
            merged_ess_no_culture_df = merged_ess_no_culture_df.sort_values('Ecosystem Services Type')
            # merged_ess_no_culture_df = merged_ess_no_culture_df.drop(columns='Ecosystem Services Type')
            score_agg_df = merged_ess_no_culture_df.pivot(index=['Aggregation', 'region', 'Bank'],
                                                          columns=['Ecosystem Services', 'Ecosystem Services Type'],
                                                          values='score')
            score_agg_df = score_agg_df.reset_index().drop(columns='Bank').groupby(['Aggregation', 'region']).mean()

    score_agg_df.columns = score_agg_df.columns.map(lambda col: tuple((c.split(" - ")[-1] if isinstance(c, str) else c) for c in col))

    return score_agg_df



if __name__ == "__main__":
    # plot_bar_chart()
    # plot_both_heatmaps_system('', 'SNA')
    # plot_heatmaps()
    # plot_imp_dep_both_heatmaps('', 'SNA')
    # ecosystem_maps_by_region('Flood control','')
    # ecosystem_maps_by_region('Local (micro and meso) climate regulation','')
    # ecosystem_maps_by_region('Rainfall pattern regulation','')
    ecosystem_maps_system(['Flood control', 'Rainfall pattern regulation', 'Local (micro and meso) climate regulation'], '')
    scope_3_source_overlap_mean_df, scope_3_value_chain_overlap_mean_df = aggregate_banks_scope_3('mean','Impact',  False)
    # scope_3_source_overlap_min_df, scope_3_value_chain_overlap_min_df = aggregate_banks_scope_3('min', 'Impact', False)
    # scope_3_source_overlap_max_df, scope_3_value_chain_overlap_max_df = aggregate_banks_scope_3('max','Impact',  False)
    #
    # # clear the column names
    # # source
    # # scope_3_source_overlap_mean_df.columns = [col.split(" - ")[-1] for col in scope_3_source_overlap_mean_df.columns]
    # # scope_3_source_overlap_min_df.columns = [col.split(" - ")[-1] for col in scope_3_source_overlap_min_df.columns]
    # # scope_3_source_overlap_max_df.columns = [col.split(" - ")[-1] for col in scope_3_source_overlap_max_df.columns]
    # # # value chain
    # # scope_3_value_chain_overlap_mean_df.columns = [col.split(" - ")[-1] for col in scope_3_value_chain_overlap_mean_df.columns]
    # # scope_3_value_chain_overlap_min_df.columns = [col.split(" - ")[-1] for col in scope_3_value_chain_overlap_min_df.columns]
    # # scope_3_value_chain_overlap_max_df.columns = [col.split(" - ")[-1] for col in scope_3_value_chain_overlap_max_df.columns]
    #
    # # to csv
    # # source
    # scope_3_source_overlap_mean_df.to_csv(GSIB_finance_var_imp_scope3_source_mean)
    # scope_3_source_overlap_min_df.to_csv(GSIB_finance_var_imp_scope3_source_min)
    # scope_3_source_overlap_max_df.to_csv(GSIB_finance_var_imp_scope3_source_max)
    # # value chain
    # scope_3_value_chain_overlap_mean_df.to_csv(GSIB_finance_var_imp_scope3_value_chain_mean)
    # scope_3_value_chain_overlap_min_df.to_csv(GSIB_finance_var_imp_scope3_value_chain_min)
    # scope_3_value_chain_overlap_max_df.to_csv(GSIB_finance_var_imp_scope3_value_chain_max)