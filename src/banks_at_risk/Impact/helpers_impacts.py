import pandas as pd
import numpy as np
from banks_at_risk.Setup.ENCORE_paths import ENCORE_imp_path, ENCORE_to_EXIO_path, ENCORE_imp_num_es_ass_path, ENCORE_imp_num_ass_driver_path, ENCORE_imp_driver_driver_env_change_path
from banks_at_risk.Setup.impact_paths import scope1_impact_max_path, scope1_impact_mean_path, scope1_impact_min_path


def read_encore_imp():
    """
    Function reads the impact materiality ratings from the ENCORE Knowledge base
    :return: ENCORE impact materiality ratings for impact drivers
    """
    # list of required columns
    usecols = ['Production process', 'Disturbances', 'Freshwater ecosystem use', 'GHG emissions', 'Marine ecosystem use','Non-GHG air pollutants', 'Other resource use', 'Soil pollutants', 'Solid waste', 'Terrestrial ecosystem use', 'Water pollutants', 'Water use']
    # read ENCORE impact materiality for impact drivers ratings with required columns
    ENCORE_imp_df = pd.read_csv(ENCORE_imp_path, index_col=[0], header=0, usecols=usecols)
    return ENCORE_imp_df


def create_impacts_on_ess(EXIO3, EXIO_imp_es_df):
    """
    This function formats the ENCORE impact scores for direct operations for EXIOBASE sectors by replicating each of
    the sectors for all the regions (as it does not mediate by country for direct operations).
    :param EXIO3: EXIOBASE MRIO with all required tables calculated
    :param EXIO_imp_es_df: ENCORE impact materiality ratings for EXIOBASE sectors for ecosystem services
    :return: Impact score for ecosystem sercies for EXIOBASE sectors and regions
    """
    # get EXIOBASE regions
    EXIO_regions = EXIO3.get_regions().to_list()
    # get included EXIOBASE sectors
    EXIO_sectors = EXIO_imp_es_df.columns.values
    # create a multi-index with the included EXIOBASE sector-region pairs
    EXIO_columns = pd.MultiIndex.from_product([EXIO_regions, EXIO_sectors]).set_names(['region', 'sector'])
    # get a list of ecosystem services
    ESSs = EXIO_imp_es_df.index.values
    # get the impact scores for all EXIOBASE sector-region pairs by replicating the sector scores for every region
    # becuase they do not change by region for direct operations
    impacts_df = pd.DataFrame(np.tile(EXIO_imp_es_df.to_numpy(), (1, len(EXIO3.get_regions().tolist()))),
                                   columns=EXIO_columns, index=ESSs)

    return impacts_df

def general_impacts(EXIO3, EXIO_to_ENCORE_df, ENCORE_imp_restricted_df, EXIO_included_sectors):
    """
    The function calculates the impacts scores for each EXIOBASE sector for ecosystem services
    :param EXIO3: EXIOBASE3 MRIO with all required tables calculated
    :param EXIO_to_ENCORE_df: finalized ENCORE-EXIOBASE concordance table
    :param ENCORE_imp_restricted_df: ENCORE impact materiality ratings for impact drivers for production processes
     included in the analysis
    :param EXIO_included_sectors: EXIOBASE sectors included in the analysis
    :return: The impact score for each EXIOBASE sector for ecosystem services calculated with 3 methodological treatments
    """
    # get list of impact drivers
    ENCORE_imp_drivers = ENCORE_imp_restricted_df.columns.values
    # create dataframes to store the scores for each impact drivers for each EXIOBASE sector for 3 methodological
    # treatments (max, mean, min)
    EXIO_imp_mean_df = pd.DataFrame(0, index=ENCORE_imp_drivers, columns=EXIO3.get_sectors().to_list())
    EXIO_imp_min_df = pd.DataFrame(0, index=ENCORE_imp_drivers, columns=EXIO3.get_sectors().to_list())
    EXIO_imp_max_df = pd.DataFrame(0, index=ENCORE_imp_drivers, columns=EXIO3.get_sectors().to_list())

    # get the ENCORE production processes included in the analysis
    ENCORE_imp_restricted_unique_df = ENCORE_imp_restricted_df.reset_index().drop_duplicates(keep='first').set_index(['Production process'])

    # calculate the impact score for each impact driver
    for driver in ENCORE_imp_drivers:
        # loop over EXIOBASE_sectors
        for sector_EXIO in EXIO_included_sectors:
            # create an array to store scores for each EXIOBASE sector for this impact driver
            driver_sector_array = []

            # get ENCORE production processes associated with this EXIOBASE sector
            associated_ENCORE_df = EXIO_to_ENCORE_df.loc[sector_EXIO, :]

            # loop over associated ENCORE production processes
            for sector_ENCORE in associated_ENCORE_df.index.values:
                # check if the impact driver has an impact materiality rating for this production process
                if driver in ENCORE_imp_restricted_unique_df.loc[sector_ENCORE, :].index.values:
                    # get the ENCORE-EXIOBASE conversion multiplier
                    poid = EXIO_to_ENCORE_df.loc[(sector_EXIO, sector_ENCORE), 'Poids']
                    # get the materiality rating
                    rating = ENCORE_imp_restricted_unique_df.loc[sector_ENCORE, driver]
                    # mediate the rating by the ENCORE-EXIOBASE multiplier and append to the storing array
                    driver_sector_array.append(poid * rating)
                    exit
                else:
                    # if the impact driver does not have a materiality rating for the sector, append zero to the
                    # storage array to assume it is not material
                    driver_sector_array.append(0)

            # store the impact score for the EXIOBASE sector and impact driver into the dataframe for each
            # methodological treatment (mean, max, min)
            EXIO_imp_mean_df.loc[driver, sector_EXIO] = np.mean(np.array(driver_sector_array)).astype(float)
            EXIO_imp_min_df.loc[driver, sector_EXIO] = np.min(np.array(driver_sector_array)).astype(float)
            EXIO_imp_max_df.loc[driver, sector_EXIO] = np.max(np.array(driver_sector_array)).astype(float)

    # name the df after the three methodological treatments to distinguish
    EXIO_imp_mean_df.name = 'mean'
    EXIO_imp_min_df.name = 'min'
    EXIO_imp_max_df.name = 'max'

    # generate the impact score for EXIOBASE sectors for ecosystem services
    EXIO_imp_mean_es_df= general_impacts_on_ess(EXIO_imp_mean_df)
    EXIO_imp_min_es_df= general_impacts_on_ess(EXIO_imp_min_df)
    EXIO_imp_max_es_df= general_impacts_on_ess(EXIO_imp_max_df)

    return EXIO_imp_mean_es_df, EXIO_imp_min_es_df, EXIO_imp_max_es_df

def generate_id_to_ess():
    """
    This function creates a dataframe that provides a connection from impact driver impact scores to ecosystem service
    impact scores by using the influence of impact drivers on natural capital assets and importance of natural capital
    assets to ecoystem service provision.
    :return: a dataframe that links ecosystem services to impact drivers
    """
    # join the importance of the asset to the ecosystem asset driver of environment change influence
    # load ENCORE ecosystem asset importance
    ENCORE_imp_es_ass_df = pd.read_csv(ENCORE_imp_num_es_ass_path, index_col=[0], header=0)
    ENCORE_imp_es_ass_df.reset_index(inplace=True)

    # load ENCORE ecosystem driver of environmental change influence
    ENCORE_imp_ass_dr_df = pd.read_csv(ENCORE_imp_num_ass_driver_path, index_col=[0], header=0)
    ENCORE_imp_ass_dr_df.reset_index(inplace=True)
    ENCORE_imp_ass_dr_df.rename(columns={"Ecosystem service": "Ecosystem Service"}, inplace=True)

    # join the ecosystem service, asset, driver influence with the ecosystem service and asset materiality
    ENCORE_es_asset_driver_join = pd.merge(ENCORE_imp_ass_dr_df,ENCORE_imp_es_ass_df,how='left',
                                       left_on=['Ecosystem Service','Asset'],right_on=['Ecosystem Service','Asset'])

    # multiply the materiality (influence) by the importance
    ENCORE_es_asset_driver_join['Driver Influence on Ecosystem Service'] = (ENCORE_es_asset_driver_join.Influence *
                                                                        ENCORE_es_asset_driver_join.Materiality)
    ENCORE_es_asset_driver_join_clean = ENCORE_es_asset_driver_join.drop(columns = ['Influence', 'Materiality'])


    # connect impact drivers to drivers of environmental change
    # load the impact drivers
    ENCORE_imp_dr_dr_df = pd.read_csv(ENCORE_imp_driver_driver_env_change_path, index_col=[0], header=0, usecols = ['Impact Driver', 'Driver', 'Asset'])
    #reset the index
    ENCORE_imp_dr_dr_df.reset_index(inplace=True)

    # join the ecosystem service asset driver
    ENCORE_id_to_ess_df = pd.merge(ENCORE_es_asset_driver_join_clean,ENCORE_imp_dr_dr_df,how='left',left_on=['Driver','Asset'],right_on=['Driver','Asset'])
    ENCORE_id_to_ess_df.drop(columns=['Asset'], inplace=True)
    ENCORE_id_to_ess_df.drop_duplicates(inplace=True)

    return ENCORE_id_to_ess_df


def general_impacts_on_ess(EXIO_imp_mean_df):
    """
    This function take ths impact scores for EXIOBASE sectors for impact drivers and converts it to the imapct score for
    EXIOBASE sector for ecosystem services
    :param EXIO_imp_mean_df: the impact score for EXIOBASE sectors for impact drivers
    :return: the impact score for EXIOBASE sectors for ecosysetm services
    """
    score_name = EXIO_imp_mean_df.name

    # create id to es - a dataframe that links ecosystem services to impact drivers
    ENCORE_id_to_ess_df = generate_id_to_ess()
    
    # get ecosystem services 
    ENCORE_ESSs = ENCORE_id_to_ess_df['Ecosystem Service'].unique().tolist()
    # get EXIO sectors
    EXIO_sectors = EXIO_imp_mean_df.columns.tolist()

    # right join EXIO_imp_df (impact score for impact drivers) and ENCORE_id_to_ess_df (dataframe linking ecosystem
    # services to impact drivers)
    EXIO_imp_df_mean_no_index = EXIO_imp_mean_df.reset_index()
    EXIO_imp_df_mean_no_index.rename(columns={"index": "Impact Driver"}, inplace=True)
    EXIO_imp_df_mean_w_dr_es = pd.merge(ENCORE_id_to_ess_df, EXIO_imp_df_mean_no_index, how='inner',
                                    left_on=['Impact Driver'], right_on=['Impact Driver'])
    EXIO_imp_df_mean_w_dr_es_filtered = EXIO_imp_df_mean_w_dr_es.copy()


    # multiply the impact scores by the driver influence on ES
    # loop through EXIOBASE sectors included in the analysis
    for sector in EXIO_imp_mean_df.columns.tolist():
        # loop through the impact scores
        for row in range(len(EXIO_imp_df_mean_w_dr_es)):
            # mean
            EXIO_imp_df_mean_w_dr_es_filtered.loc[row, sector] = \
            EXIO_imp_df_mean_w_dr_es['Driver Influence on Ecosystem Service'].iloc[row] * EXIO_imp_df_mean_w_dr_es.loc[
                row, sector]
            
    
    # creater a dataframe with ecosystem services as index and sectors as columns
    EXIO_imp_es_df = pd.DataFrame(0, index=ENCORE_ESSs, columns=EXIO_sectors)

    # connect the driver to impact driver
    # loop through EXIOBASE sectors included in analysis
    for sector in EXIO_sectors:
        # loop through ecosystem services
        for ec_service in ENCORE_ESSs:
            # create arra to store the imapct scores for each ecosystem service for this sector
            es_sector_array = []
            # loop through impact scores
            for row in range(len(EXIO_imp_df_mean_w_dr_es)):
                # find the rows associated with this ecosystem service
                if (EXIO_imp_df_mean_w_dr_es.loc[row, 'Ecosystem Service'] == ec_service):
                    # append the impact driver to ecosystem service converstion multiplier for this sector
                    es_sector_array.append(EXIO_imp_df_mean_w_dr_es.loc[row, sector])
            # if there are no conversion multipliers for this sector for this ecosystem service/impact driver -
            # assume not material and set impact score to zero
            if (len(es_sector_array) == 0):
                EXIO_imp_es_df[ec_service, sector] = 0
            # otherwise take the average of the impact scores to get the final ecsosystem service impact score
            else:
                EXIO_imp_es_df.loc[ec_service, sector] = np.mean(np.array(es_sector_array))
    # name the score with original score indicating methodological treatment of impact score
    EXIO_imp_es_df.name = score_name
    
    return EXIO_imp_es_df


def compute_impacts_scope1(EXIO3, EXIO_imp_es_df):
    """
    This function formats and stores the impact scores for ecosystem services in the data repository.
    :param EXIO3: EXIOBASE MRIO with all required tables calculated
    :param EXIO_imp_es_df: The impact score for ecosystem services dataframe
    :return:
    """
    # format the impact scores for EXIOBASE sectors and regions
    scope_1_impacts_df = create_impacts_on_ess(EXIO3, EXIO_imp_es_df)

    # get the appropriate path in the data repository for this score and store the dataframe as csv
    if EXIO_imp_es_df.name == 'mean':
        scope_1_impacts_df.to_csv(scope1_impact_mean_path, index=True, header=True)
    if EXIO_imp_es_df.name == 'min':
        scope_1_impacts_df.to_csv(scope1_impact_min_path, index=True, header=True)
    if EXIO_imp_es_df.name == 'max':
        scope_1_impacts_df.to_csv(scope1_impact_max_path, index=True, header=True)

    return None