import pandas as pd
import numpy as np
from banks_at_risk.Setup.ENCORE_paths import ENCORE_dep_path, ENCORE_to_EXIO_path
from banks_at_risk.Setup.dependency_paths import scope1_dependency_max_path, scope1_dependency_mean_path, scope1_dependency_min_path

def read_encore_dep():
    """
    Function reads the dependency materiality ratings from the ENCORE Knowledge base
    :return: ENCORE dependency materiality ratings
    """
    # list of required columns
    usecols = ['Process', 'Ecosystem Service', 'Rating Num']
    # read ENCORE dependency materiality ratings with required columns
    ENCORE_dep_df = pd.read_csv(ENCORE_dep_path, index_col=[0, 1], header=0, usecols=usecols)

    return ENCORE_dep_df


def create_dependencies_df(EXIO3, EXIO_dep_df):
    """
    This function formats the ENCORE dependency scores for direct operations for EXIOBASE sectors by replicating each of
    the sectors for all the regions (as it does not mediate by country for direct operations).
    :param EXIO3: EXIOBASE3 MRIO with all required tables calculated
    :param EXIO_dep_df: ENCORE dependency materiality ratings for EXIOBASE sectors
    :return: Depedendency score for EXIOBASE sectors and regions
    """
    # get list of EXIOBASE regions
    EXIO_regions = EXIO3.get_regions().to_list()
    # get EXIOBASE sectors included in the analysis
    EXIO_sectors = EXIO_dep_df.columns.values
    # create a multi-index with the included EXIOBASE sector-region pairs
    EXIO_columns = pd.MultiIndex.from_product([EXIO_regions, EXIO_sectors]).set_names(['region', 'sector'])
    # get a list of ecosystem services
    ESSs = EXIO_dep_df.index.values
    # get the dependency scores for all EXIOBASE sector-region pairs by replicating the sector scores for every region
    # becuase they do not change by region for direct operations
    dependencies_df = pd.DataFrame(np.tile(EXIO_dep_df.to_numpy(), (1, len(EXIO3.get_regions().tolist()))),
                                   columns=EXIO_columns, index=ESSs)

    return dependencies_df


def general_dependencies(EXIO3, EXIO_to_ENCORE_df, ENCORE_dep_restricted_df, EXIO_included_sectors):
    """
    This function calculates the dependency score for each EXIOBASE sector
    :param EXIO3: EXIOBASE3 MRIO with all required tables calculated
    :param EXIO_to_ENCORE_df: finalized ENCORE-EXIOBASE concordance table
    :param ENCORE_dep_restricted_df: ENCORE dependency materiality ratings for productions processes included in the
    analysis
    :param EXIO_included_sectors: EXIOBASE sectors included in the analysis
    :return: The dependency score for each EXIOBASE sector calculated with 3 methodological treatments (mean, max, min)
    """
    # get list of ecosystem services
    ENCORE_ecosys_serv = np.unique(ENCORE_dep_restricted_df.index.get_level_values(1))
    # create dataframes to store the scores for each ecosystem services for each EXIOBASE sector for 3 methodological
    # treatments (max, mean, min)
    EXIO_dep_mean_df = pd.DataFrame(0, index=ENCORE_ecosys_serv, columns=EXIO3.get_sectors().to_list())
    EXIO_dep_min_df = pd.DataFrame(0, index=ENCORE_ecosys_serv, columns=EXIO3.get_sectors().to_list())
    EXIO_dep_max_df = pd.DataFrame(0, index=ENCORE_ecosys_serv, columns=EXIO3.get_sectors().to_list())

    # calculate the dependency score for each ecosystem service
    for service in ENCORE_ecosys_serv:
        # for each included EXIOBASE sector
        for sector_EXIO in EXIO_included_sectors:
            # an array to store the scores for each sector for this ecosystem service
            service_sector_array = []
            # get production processes associated with this EXIOBASE sector
            associated_ENCORE_df = EXIO_to_ENCORE_df.loc[sector_EXIO, :]

            # for the production process associated with this EXIOBASE sector
            for sector_ENCORE in associated_ENCORE_df.index.values:
                # check if ecosystem service has a dependency materiality rating for that production process
                if service in ENCORE_dep_restricted_df.loc[sector_ENCORE, :].index.values:
                    # get the ENCORE-EXIOBASE conversion multiplier
                    poid = EXIO_to_ENCORE_df.loc[(sector_EXIO, sector_ENCORE), 'Poids']
                    # get the materiality rating
                    rating = ENCORE_dep_restricted_df.loc[(sector_ENCORE, service), 'Rating Num']
                    # mediate the rating by the ENCORE-EXIOBASE multiplier and append to the storing array
                    service_sector_array.append(poid * rating)
                    exit
                else:
                    # if the ecosystem service does not have a materiality rating for the sector, append zero to the
                    # storage array to assume it is not material
                    service_sector_array.append(0)

            # store the dependency score for the EXIOBASE sector and ecosystem service into the dataframe for each
            # methodological treatment (mean, max, min)
            EXIO_dep_mean_df.loc[service, sector_EXIO] = np.mean(np.array(service_sector_array))
            EXIO_dep_max_df.loc[service, sector_EXIO] = np.max(np.array(service_sector_array))
            EXIO_dep_min_df.loc[service, sector_EXIO] = np.min(np.array(service_sector_array))

        # name the df after the three methodological treatments to distinguish
        EXIO_dep_mean_df.name = 'mean'
        EXIO_dep_max_df.name = 'max'
        EXIO_dep_min_df.name = 'min'

    return EXIO_dep_mean_df, EXIO_dep_max_df, EXIO_dep_min_df



def compute_dependencies_scope1(EXIO3, EXIO_dep_df):
    """
    This function formats and stores the dependency scores in the data repository.
    :param EXIO3: EXIOBASE MRIO with all required tables calculated
    :param EXIO_dep_df: Dependency score dataframe
    :return: NA
    """
    # format the depdendency scores for EXIOBASE sectors and regions
    scope_1_dependencies_df = create_dependencies_df(EXIO3, EXIO_dep_df)

    # get the appropriate path in the data repository for this score
    if EXIO_dep_df.name == 'mean':
        path = scope1_dependency_mean_path
    if EXIO_dep_df.name == 'max':
        path = scope1_dependency_max_path
    if EXIO_dep_df.name == 'min':
        path = scope1_dependency_min_path

    # store the score at the appropriate path in the data repository
    scope_1_dependencies_df.to_csv(path, index=True, header=True)

    return None

    
