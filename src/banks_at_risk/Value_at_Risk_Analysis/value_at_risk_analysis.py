# from banks_at_risk.Value_at_Risk_Analysis.helper_value_at_risk_analysis import get_var_scores, anonymize, calculate_significances, plot_bar_chart, plot_heatmap_system_level, plot_heatmap_bank_level, plot_bar_chart_system
from banks_at_risk.Value_at_Risk_Analysis.helper_value_at_risk_analysis_GSIB import get_var_scores, anonymize, calculate_significances, plot_bar_chart, plot_heatmap_system_level, plot_heatmap_bank_level, plot_bar_chart_system, plot_heatmap_bank_level_separated, plot_heatmap_bank_level_UK, plot_bar_chart_UK, statistical_tests_sector, sensitivity_calc_type, calculate_bank_vs_average, calculate_portfolio_vs_system, plot_finance_data, plot_heatmap_bank_level_regions_combined, plot_bar_chart_regional

def analyze_nvar(anonymize):
    """
    This function reads the endogenous, impact and dependency risk from the sector, bank portfolio and system level and
    conducts statistical significance tests and generates figures to display the data.
    :param: anonymize - true if you want banks anonymized or false if you do not
    :return: NA
    """
    # plot financial data
    plot_finance_data()

    # get the impact, dependency and endogenous risk results with the mean methodological treatment
    impact_nvars_mean = get_var_scores('Impact', 'mean')
    dependency_nvars_mean = get_var_scores('Dependency', 'mean')
    both_nvars_mean = get_var_scores('Both', 'mean')

    # get the impact, dependency and endogenous risk results with the min methodological treatment
    impact_nvars_min = get_var_scores('Impact', 'min')
    dependency_nvars_min = get_var_scores('Dependency', 'min')
    both_nvars_min = get_var_scores('Both', 'min')

    # get the impact, dependency and endogenous risk results with the max methodological treatment
    impact_nvars_max = get_var_scores('Impact', 'max')
    dependency_nvars_max = get_var_scores('Dependency', 'max')
    both_nvars_max = get_var_scores('Both', 'max')

    if anonymize == True:
        # anonymize the banks in these sheets
        impact_nvars_mean, dependency_nvars_mean, both_nvars_mean = anonymize(impact_nvars_mean, dependency_nvars_mean, both_nvars_mean)
        impact_nvars_min, dependency_nvars_min, both_nvars_min = anonymize(impact_nvars_min, dependency_nvars_min, both_nvars_min)
        impact_nvars_max, dependency_nvars_max, both_nvars_max = anonymize(impact_nvars_max, dependency_nvars_max, both_nvars_max)

    # get the sectoral overlap scores for the three methodological treatments
    sector_scores_mean = get_var_scores('Sector', 'mean')
    sector_scores_min = get_var_scores('Sector', 'min')
    sector_scores_max = get_var_scores('Sector', 'max')

    # get the system-level endongeous risk scores for each of the methodological treatments
    system_both_scores_mean = get_var_scores('System Both', 'mean')
    system_both_scores_min = get_var_scores('System Both', 'min')
    system_both_scores_max = get_var_scores('System Both', 'max')
    # get the sectoral overlap system-level endogenous risk scores for each of the methodological treatments
    system_sector_both_scores_mean = get_var_scores('System Sector', 'mean')
    system_sector_both_scores_min = get_var_scores('System Sector', 'min')
    system_sector_both_scores_max = get_var_scores('System Sector', 'max')

    # calculate the significance of trends
    # portfolio level

    # calculate the statistical significance of difference between direct operations and upstream
    # supply chain - for each methodological treatment
    calculate_significances(impact_nvars_mean, dependency_nvars_mean, both_nvars_mean)
    calculate_significances(impact_nvars_min, dependency_nvars_min, both_nvars_min)
    calculate_significances(impact_nvars_max, dependency_nvars_max, both_nvars_max)

    # sector vs portfolio - calculate the statistical significance of difference between sector and portfolio endogenous
    # risk - for each methodological treatment
    statistical_tests_sector(both_nvars_mean, sector_scores_mean, 'mean')
    statistical_tests_sector(both_nvars_min, sector_scores_min, 'min')
    statistical_tests_sector(both_nvars_max, sector_scores_max, 'max')

    # mean vs min vs max - calculate the statistical significance of differences between each methodological treatment
    # endogenous risk
    sensitivity_calc_type(both_nvars_mean, both_nvars_min, both_nvars_max)
    # impact only
    sensitivity_calc_type(impact_nvars_mean, impact_nvars_min, impact_nvars_max)
    # dependency only
    sensitivity_calc_type(dependency_nvars_mean, dependency_nvars_min, dependency_nvars_max)

    # portfolio-level - calculate the difference between each bank portfolio endogenous risk and the average bank
    # portfolio endogenous risk to see what banks are more exposed than most - for each methodological treatment
    calculate_bank_vs_average(both_nvars_mean)
    calculate_bank_vs_average(both_nvars_min)
    calculate_bank_vs_average(both_nvars_max)


    # system level
    # calculate the statistical significance of difference between direct operations and upstream
    # supply chain - for each methodological treatment
    calculate_significances(system_both_scores_mean, system_both_scores_min, system_both_scores_max)

    # sector vs portfolio - calculate the statistical significance of difference between sector and portfolio endogenous
    # risk - for each methodological treatment
    statistical_tests_sector(system_both_scores_mean, system_sector_both_scores_mean, 'mean')
    statistical_tests_sector(system_both_scores_min, system_sector_both_scores_min, 'min')
    statistical_tests_sector(system_both_scores_max, system_sector_both_scores_max, 'max')

    # mean vs min vs max - calculate the statistical significance of differences between each methodological treatment
    # endogenous risk
    sensitivity_calc_type(system_both_scores_mean, system_both_scores_min, system_both_scores_max)

    # portfolio-level - calculate the difference between each bank portfolio endogenous risk and the average bank
    # portfolio endogenous risk to see what banks are more exposed than most - for each methodological treatment
    calculate_portfolio_vs_system(system_both_scores_mean, both_nvars_mean)
    calculate_portfolio_vs_system(system_both_scores_min, both_nvars_min)
    calculate_portfolio_vs_system(system_both_scores_max, both_nvars_max)

    # get figures
    # plot bar charts of all bank portfolios' endogenous risk for the direct operations and upstream supply chain
    plot_bar_chart(both_nvars_mean, both_nvars_min, both_nvars_max)
    # impact risk
    plot_bar_chart(impact_nvars_mean, impact_nvars_min, impact_nvars_max)
    # dependency risk
    plot_bar_chart(dependency_nvars_mean, dependency_nvars_min, dependency_nvars_max)
    # plot bar charts of all bank portfolios' endogenous risk for the direct operations and upstream supply chain
    # aggregated by bank domicile region
    plot_bar_chart_regional(both_nvars_mean, both_nvars_min, both_nvars_max)

    # plot heatmaps of all bank portfolios' endogneous risk by sector and region for direct operations and upstream
    # supply chain
    plot_heatmap_bank_level(both_nvars_mean)
    # impact risk
    plot_heatmap_bank_level(impact_nvars_mean)
    # dependency risk
    plot_heatmap_bank_level(dependency_nvars_mean)
    # plot heatmaps of bank portfolios' endogenous risk by sector and region for direct operations and upstream supply
    # chain - split by bank domicile region
    plot_heatmap_bank_level_separated(both_nvars_mean)
    # impact risk
    plot_heatmap_bank_level_separated(impact_nvars_mean)
    # dependency risk
    plot_heatmap_bank_level_separated(dependency_nvars_mean)
    # plot heatmaps of all bank portfolios' endogneous risk by sector and region for direct operations and upstream
    # supply chain - aggregated by bank domicile region
    plot_heatmap_bank_level_regions_combined(both_nvars_mean)

    return None

def analyze_nvar_system():

    # get the system-level impact, dependency and endogenous risk results with the mean methodological treatment
    impact_nvars_mean = get_var_scores('Impact', 'mean')
    dependency_nvars_mean = get_var_scores('Dependency', 'mean')
    both_nvars_mean = get_var_scores('Both', 'mean')

    # get the system-level impact, dependency and endogenous risk results with the min methodological treatment
    impact_nvars_min = get_var_scores('Impact', 'min')
    dependency_nvars_min = get_var_scores('Dependency', 'min')
    both_nvars_min = get_var_scores('Both', 'min')

    # get the system-level impact, dependency and endogenous risk results with the max methodological treatment
    impact_nvars_max = get_var_scores('Impact', 'max')
    dependency_nvars_max = get_var_scores('Dependency', 'max')
    both_nvars_max = get_var_scores('Both', 'max')

    # anonymize the banks in these sheets
    # impact_nvars_mean, dependency_nvars_mean, both_nvars_mean = anonymize(impact_nvars_mean, dependency_nvars_mean, both_nvars_mean)
    # impact_nvars_min, dependency_nvars_min, both_nvars_min = anonymize(impact_nvars_min, dependency_nvars_min, both_nvars_min)
    # impact_nvars_max, dependency_nvars_max, both_nvars_max = anonymize(impact_nvars_max, dependency_nvars_max, both_nvars_max)

    # get figures
    # plot bar charts of system endogenous risk for the direct operations and upstream supply chain
    plot_bar_chart_system(both_nvars_mean, both_nvars_min, both_nvars_max)
    # impact risk
    plot_bar_chart_system(impact_nvars_mean, impact_nvars_min, impact_nvars_max)
    # dependency risk
    plot_bar_chart_system(dependency_nvars_mean, dependency_nvars_min, dependency_nvars_max)

    # plot heatmaps of system endogneous risk by sector and region for direct operations and upstream supply chain
    plot_heatmap_system_level(both_nvars_mean)
    # impact risk
    plot_heatmap_system_level(impact_nvars_mean)
    # dependency risk
    plot_heatmap_system_level(dependency_nvars_mean)


if __name__ == "__main__":
    analyze_nvar(anonymize=True)
    analyze_nvar_system()