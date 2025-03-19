from banks_at_risk.Setup.impact_paths import scope1_impact_mean_path, scope1_impact_max_path, scope1_impact_min_path
from banks_at_risk.Setup.dependency_paths import scope1_dependency_max_path, scope1_dependency_min_path, scope1_dependency_mean_path
from banks_at_risk.Value_at_Risk.helper_value_at_risk_GSIB import calc_L_min_I_full
from banks_at_risk.Setup.var_plots_paths import scope1_overlap_max_path, scope1_overlap_min_path, scope1_overlap_mean_path, scope3_overlap_max_path, scope3_overlap_min_path, scope3_overlap_mean_path, \
    NACE_scope3_overlap_max_path, NACE_scope1_overlap_max_path, NACE_scope1_overlap_mean_path, NACE_scope1_overlap_min_path, NACE_scope3_overlap_mean_path, NACE_scope3_overlap_min_path, sectoral_overlap_saving_path,  \
    scope3_overlap_max_rows_path, scope3_overlap_mean_rows_path, scope3_overlap_min_rows_path, NACE_scope3_overlap_min_rows_path, NACE_scope3_overlap_max_rows_path, NACE_scope3_overlap_mean_rows_path
from banks_at_risk.NACE_Conversion.helpers_NACE_conversion import generate_converter_sector

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# calculate the overlap for direct operations
def calc_ENCORE_overlap(impact_df, dependency_df):
    # scope 1
    overlap_sector_df = impact_df * dependency_df

    overlap_sector_df.name = f'{impact_df.name} {dependency_df.name}'

    return overlap_sector_df

def generate_overlap_scores():
    imp_score_list = []
    # impact
    scope1_impact_mean_df = pd.read_csv(scope1_impact_mean_path, header=[0, 1], index_col=[0])
    scope1_impact_mean_df.name = 'scope1_impact_mean'
    imp_score_list.append(scope1_impact_mean_df)
    scope1_impact_min_df = pd.read_csv(scope1_impact_min_path, header=[0, 1], index_col=[0])
    scope1_impact_min_df.name = 'scope1_impact_min'
    imp_score_list.append(scope1_impact_min_df)
    scope1_impact_max_df = pd.read_csv(scope1_impact_max_path, header=[0, 1], index_col=[0])
    scope1_impact_max_df.name = 'scope1_impact_max'
    imp_score_list.append(scope1_impact_max_df)

    dep_score_list = []
    # dependency
    scope1_dependency_mean_df = pd.read_csv(scope1_dependency_mean_path, header=[0, 1], index_col=[0])
    scope1_dependency_mean_df.name = 'scope1_dependency_mean'
    dep_score_list.append(scope1_dependency_mean_df)
    scope1_dependency_min_df = pd.read_csv(scope1_dependency_min_path, header=[0, 1], index_col=[0])
    scope1_dependency_min_df.name = 'scope1_dependency_min'
    dep_score_list.append(scope1_dependency_min_df)
    scope1_dependency_max_df = pd.read_csv(scope1_dependency_max_path, header=[0, 1], index_col=[0])
    scope1_dependency_max_df.name = 'scope1_dependency_max'
    dep_score_list.append(scope1_dependency_max_df)

    scope1_overlap_scores_list = []
    overlap_mean_df = calc_ENCORE_overlap(scope1_impact_mean_df, scope1_dependency_mean_df)
    scope1_overlap_scores_list.append(overlap_mean_df)
    overlap_min_df = calc_ENCORE_overlap(scope1_impact_min_df, scope1_dependency_min_df)
    scope1_overlap_scores_list.append(overlap_min_df)
    overlap_max_df = calc_ENCORE_overlap(scope1_impact_max_df, scope1_dependency_max_df)
    scope1_overlap_scores_list.append(overlap_max_df)

    ### calculate overlined((L -1)), relative impact dependency matrix
    L_min_I = calc_L_min_I_full()
    L_min_I_numpy = L_min_I.to_numpy(dtype=float)
    col_sums = np.sum(L_min_I, axis=0)
    col_sums = col_sums.to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_imp_array = np.where(col_sums == 0, 0, np.divide(L_min_I_numpy, col_sums[np.newaxis, :]))

    # get the weights for the contribution of each sector,region pair to the supply chain
    L_weights = pd.DataFrame(rel_imp_array, index=L_min_I.index, columns=L_min_I.columns)
    upstream_calc = L_weights.copy().reset_index()

    upstream_calc_format = upstream_calc.T.reset_index().T.rename(columns={0: 'region', 1: 'sector'})
    upstream_calc_format.loc['sector', 'region'] = 'region'
    upstream_calc_format.loc['sector', 'sector'] = 'sector'

    scope3_overlap_scores_list = []
    scope3_overlap_scores_rows_list = []
    for score in scope1_overlap_scores_list:
        scope3_overlap_cols_df, scope3_overlap_rows_df = get_scope3_scores(score, upstream_calc_format, L_min_I)
        scope3_overlap_scores_list.append(scope3_overlap_cols_df)
        scope3_overlap_scores_rows_list.append(scope3_overlap_rows_df)

    services = scope3_overlap_scores_rows_list[0].columns.tolist()

    scope3_overlap_scores_rows_adjusted_list = []

    for score in scope3_overlap_scores_rows_list:
        df = score.copy()
        score_name = score.name
        for service in services:
            df[service] = df[service] / L_min_I.T.sum()
            df.name = score_name
        scope3_overlap_scores_rows_adjusted_list.append(df)


    for score in scope1_overlap_scores_list:
        if 'mean' in score.name:
            score.to_csv(scope1_overlap_mean_path)
        if 'max' in score.name:
            score.to_csv(scope1_overlap_max_path)
        if 'min' in score.name:
            score.to_csv(scope1_overlap_min_path)

    for score in scope3_overlap_scores_list:
        if 'mean' in score.name:
            score.to_csv(scope3_overlap_mean_path)
        if 'max' in score.name:
            score.to_csv(scope3_overlap_max_path)
        if 'min' in score.name:
            score.to_csv(scope3_overlap_min_path)

    for score in scope3_overlap_scores_rows_adjusted_list:
        if 'mean' in score.name:
            score.to_csv(scope3_overlap_mean_rows_path)
        if 'max' in score.name:
            score.to_csv(scope3_overlap_max_rows_path)
        if 'min' in score.name:
            score.to_csv(scope3_overlap_min_rows_path)

        # convert to NACE

    # convert to NACE sectors
    NACE_converter = pd.read_csv('/Users/emmao/banks_at_risk/NACE_converter.csv', header=[0], index_col=[0])

    NACE_score_list = []
    for score in scope1_overlap_scores_list:
        df = score.copy()
        score_name = score.name
        NACE_score = pd.merge(score.T, NACE_converter, right_index=True, left_index=True).groupby(
                ['Code']).sum()
        if 'mean' in score_name:
            NACE_score.to_csv(NACE_scope1_overlap_mean_path)
        if 'max' in score_name:
            NACE_score.to_csv(NACE_scope1_overlap_max_path)
        if 'min' in score_name:
            NACE_score.to_csv(NACE_scope1_overlap_min_path)
        NACE_score_list.append(NACE_score)

    for score in scope3_overlap_scores_list:
        score_name = score.name
        NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(
                ['Code', 'region']).sum()
        if 'mean' in score_name:
             NACE_score.to_csv(NACE_scope3_overlap_mean_path)
        if 'max' in score_name:
             NACE_score.to_csv(NACE_scope3_overlap_max_path)
        if 'min' in score_name:
             NACE_score.to_csv(NACE_scope3_overlap_min_path)
        NACE_score_list.append(NACE_score)

    for score in scope3_overlap_scores_rows_adjusted_list:
        score_name = score.name
        NACE_score = pd.merge(score, NACE_converter, right_index=True, left_index=True).groupby(
                ['Code', 'region']).sum()
        if 'mean' in score_name:
             NACE_score.to_csv(NACE_scope3_overlap_mean_rows_path)
        if 'max' in score_name:
             NACE_score.to_csv(NACE_scope3_overlap_max_rows_path)
        if 'min' in score_name:
             NACE_score.to_csv(NACE_scope3_overlap_min_rows_path)
        NACE_score_list.append(NACE_score)

    return scope1_overlap_scores_list, scope3_overlap_scores_list, scope3_overlap_scores_rows_list, NACE_score_list

def get_scope3_scores(score, upstream_calc_format, L_min_I):

    services = score.index.tolist()

    score_name = score.name

    df = score.copy()

    overlap_score_storing_df = pd.DataFrame(index=L_min_I.index)
    imp_dep_compile_rows_df = pd.DataFrame(index=L_min_I.index)

    # for storing the column sums for the score-level scores
    imp_dep_compile_cols_storing_df = pd.DataFrame(columns=services)
    # for storing the row sums for the score-level scores
    imp_dep_compile_rows_storing_df = pd.DataFrame(columns=services)

    for service in services:

        scope3_score_df = upstream_calc_format.merge(
            df.T[service].reset_index(), how='outer', left_on=['region', 'sector'],
            right_on=['region', 'sector'])
        scope3_score_df = scope3_score_df.fillna(0.0)

        compiled_imp_df = scope3_score_df.set_index(['region', 'sector'])
        compiled_imp_df = compiled_imp_df.T.set_index(('region', 'sector')).T
        service_imp_df = compiled_imp_df[(0.0, 0.0)]
        service_imp_df = service_imp_df[0:(L_min_I.shape[0])]
        service_imp_df = service_imp_df.astype(float)
        # calc_df = compiled_df.drop(columns=service).iloc[0:(L_min_I.shape[0]), 2:(L_min_I.shape[0] + 2)]
        # calc_df = calc_df.astype(float)
        calc_imp_df = compiled_imp_df.drop(columns=(0.0, 0.0))
        calc_imp_df = calc_imp_df.astype(float)
        multiplied_imp_df = np.multiply(calc_imp_df.to_numpy(), service_imp_df.to_numpy()[:, np.newaxis])
        imp_dep_compile_service_imp_df = pd.DataFrame(multiplied_imp_df, index=calc_imp_df.index,
                                                      columns=calc_imp_df.columns)


        # get the column sums for one bank for the scores
        overlap_score_storing_df[service] = imp_dep_compile_service_imp_df.sum()
        # # get the row sums for one bank and service into the greater the df
        imp_dep_compile_rows_df[f'{service}'] = imp_dep_compile_service_imp_df.T.sum()


    overlap_score_storing_df.name = f'scope3 {score_name}'
    imp_dep_compile_rows_df.name = f'scope3 {score_name} rows'

    return overlap_score_storing_df, imp_dep_compile_rows_df

def generate_overlap_figures():
    # scope1_scores, scope3_scores = generate_overlap_scores()
    score_list = []

    scope1_overlap_mean_df = pd.read_csv(scope1_overlap_mean_path, index_col=[0], header=[0, 1])
    scope1_overlap_mean_df.name = 'scope1_overlap_mean_df'
    score_list.append(scope1_overlap_mean_df)
    scope1_overlap_min_df = pd.read_csv(scope1_overlap_min_path, index_col=[0], header=[0, 1])
    scope1_overlap_min_df.name = 'scope1_overlap_min_df'
    score_list.append(scope1_overlap_min_df)
    scope1_overlap_max_df = pd.read_csv(scope1_overlap_max_path, index_col=[0], header=[0, 1])
    scope1_overlap_max_df.name = 'scope1_overlap_max_df'
    score_list.append(scope1_overlap_max_df)

    scope3_overlap_mean_df = pd.read_csv(scope3_overlap_mean_path, index_col=[0, 1], header=[0])
    scope3_overlap_mean_df = scope3_overlap_mean_df.T
    scope3_overlap_mean_df.name = 'scope3_overlap_mean_df'
    score_list.append(scope3_overlap_mean_df)
    scope3_overlap_min_df = pd.read_csv(scope3_overlap_min_path, index_col=[0, 1], header=[0])
    scope3_overlap_min_df = scope3_overlap_min_df.T
    scope3_overlap_min_df.name = 'scope3_overlap_min_df'
    score_list.append(scope3_overlap_min_df)
    scope3_overlap_max_df = pd.read_csv(scope3_overlap_max_path, index_col=[0, 1], header=[0])
    scope3_overlap_max_df = scope3_overlap_max_df.T
    scope3_overlap_max_df.name = 'scope3_overlap_max_df'
    score_list.append(scope3_overlap_max_df)

    NACE_score_list = []
    NACE_scope1_overlap_mean_df = pd.read_csv(NACE_scope1_overlap_mean_path, index_col=[0], header=[0])
    NACE_scope1_overlap_mean_df.name = 'NACE_scope1_overlap_mean_df'
    NACE_score_list.append(NACE_scope1_overlap_mean_df)
    NACE_scope1_overlap_min_df = pd.read_csv(NACE_scope1_overlap_min_path, index_col=[0], header=[0])
    NACE_scope1_overlap_min_df.name = 'NACE_scope1_overlap_min_df'
    NACE_score_list.append(NACE_scope1_overlap_min_df)
    NACE_scope1_overlap_max_df = pd.read_csv(NACE_scope1_overlap_max_path, index_col=[0], header=[0])
    NACE_scope1_overlap_max_df.name = 'NACE_scope1_overlap_max_df'
    NACE_score_list.append(NACE_scope1_overlap_max_df)

    NACE_scope3_overlap_mean_df = pd.read_csv(NACE_scope3_overlap_mean_path, index_col=[0, 1], header=[0])
    NACE_scope3_overlap_mean_df = NACE_scope3_overlap_mean_df.T
    NACE_scope3_overlap_mean_df.name = 'NACE_scope3_overlap_mean_df'
    NACE_score_list.append(NACE_scope3_overlap_mean_df)
    NACE_scope3_overlap_min_df = pd.read_csv(NACE_scope3_overlap_min_path, index_col=[0, 1], header=[0])
    NACE_scope3_overlap_min_df = NACE_scope3_overlap_min_df.T
    NACE_scope3_overlap_min_df.name = 'NACE_scope3_overlap_min_df'
    NACE_score_list.append(NACE_scope3_overlap_min_df)
    NACE_scope3_overlap_max_df = pd.read_csv(NACE_scope3_overlap_max_path, index_col=[0, 1], header=[0])
    NACE_scope3_overlap_max_df = NACE_scope3_overlap_max_df.T
    NACE_scope3_overlap_max_df.name = 'NACE_scope3_overlap_max_df'
    NACE_score_list.append(NACE_scope3_overlap_max_df)

    NACE_scope3_overlap_mean_rows_df = pd.read_csv(NACE_scope3_overlap_mean_rows_path, index_col=[0, 1], header=[0])
    NACE_scope3_overlap_mean_rows_df = NACE_scope3_overlap_mean_rows_df.T
    NACE_scope3_overlap_mean_rows_df.name = 'NACE_scope3_overlap_mean_rows_df'
    NACE_score_list.append(NACE_scope3_overlap_mean_rows_df)
    NACE_scope3_overlap_min_rows_df = pd.read_csv(NACE_scope3_overlap_min_rows_path, index_col=[0, 1], header=[0])
    NACE_scope3_overlap_min_rows_df = NACE_scope3_overlap_min_rows_df.T
    NACE_scope3_overlap_min_rows_df.name = 'NACE_scope3_overlap_min_rows_df'
    NACE_score_list.append(NACE_scope3_overlap_min_rows_df)
    NACE_scope3_overlap_max_rows_df = pd.read_csv(NACE_scope3_overlap_max_rows_path, index_col=[0, 1], header=[0])
    NACE_scope3_overlap_max_rows_df = NACE_scope3_overlap_max_rows_df.T
    NACE_scope3_overlap_max_rows_df.name = 'NACE_scope3_overlap_max_rows_df'
    NACE_score_list.append(NACE_scope3_overlap_max_rows_df)



    colors = sns.color_palette("Reds", as_cmap=True)
    fig, axs = plt.subplots(3, 1, figsize =(15,15))

    for score in NACE_score_list:
        if 'mean' not in score.name:
            continue
        df = score.copy()
        if 'scope1' in score.name:
            sns.heatmap(score.T, cmap=colors, ax=axs[0])
            axs[0].set_title('Direct Operations Endogenous Risk Exposure by Sector')
            axs[0].set_xlabel('Sector')

        else:
            if 'rows' in score.name:
                rows_df = df.T.reset_index().drop(columns=['region']).groupby(['Code']).sum()
                sns.heatmap(rows_df.T, cmap=colors, ax=axs[2])
                axs[2].set_title('Upstream Supply Chain Endogenous Risk Exposure by Value Chain Sector')
                axs[1].set_xlabel('Sector')

            else:
                sector_df = df.T.reset_index().drop(columns=['region']).groupby(['Code']).sum()
                sns.heatmap(sector_df.T, cmap=colors, ax=axs[1])
                axs[1].set_title('Upstream Supply Chain Endogenous Risk Exposure by Source Sector')
                axs[1].set_xlabel('Sector')

            plt.tight_layout()
            plt.savefig(sectoral_overlap_saving_path)
    return None




if __name__ == '__main__':
    generate_overlap_scores()
    generate_overlap_figures()