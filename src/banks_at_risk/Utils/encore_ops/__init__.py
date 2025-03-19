import pandas as pd
import numpy as np
from banks_at_risk.Setup.ENCORE_paths import ENCORE_to_EXIO_path


def read_encore_to_exio():
    """
    Function reads the ENCORE-EXIOBASE concordance tables from the excel file in the data repository
    :return: ENCORE-EXIOBASE concordance table with relevant information as a dataframe
    """
    # read the concordance table
    ENCORE_to_EXIO_df = pd.read_excel(ENCORE_to_EXIO_path, index_col=[5], header=[0], sheet_name='table_correspondance')
    # remove unnecessary columns
    ENCORE_to_EXIO_df_restricted = ENCORE_to_EXIO_df.drop(labels=['Grandes catégories', 'Sector', 'Subindustry',
                                                                  'Industry_group_benchmark',
                                                                  'ID_Industry_group_benchmark',
                                                                  'Exiobase_industry_group'], axis=1)
    ENCORE_to_EXIO_df_restricted.sort_index(inplace=True)

    return ENCORE_to_EXIO_df_restricted

def get_incl_sectors(EXIO3, ENCORE_to_EXIO_df_restricted):
    """
    Function generates a list of EXIOBASE sectors with appropriate ENCORE production process analogues
    :param EXIO3: EXIOBASE3 MRIO
    :param ENCORE_to_EXIO_df_restricted: ENCORE-EXIOBASE concordance table
    :return: list of sectors to include from EXIOBASE
    """
    # get a list of all EXIOBASE sectors
    EXIO_sectors = EXIO3.get_sectors().to_list()
    # get EXIOBASE sectors with no ENCORE equivalent
    EXIO_not_included_sectors = set(EXIO_sectors) - set(ENCORE_to_EXIO_df_restricted.loc[:, 'IndustryTypeName'])
    # remove the sectors with no ENCORE equivalent
    EXIO_included_sectors = set(EXIO_sectors) - EXIO_not_included_sectors

    return EXIO_included_sectors


def get_restricted_ENCORE(ENCORE_dep_df, ENCORE_to_EXIO_df_restricted):
    """
    Function generates the ENCORE sectors with appropriate EXIOBASE analogues.
    :param ENCORE_dep_df: ENCORE dependency scores
    :param ENCORE_to_EXIO_df_restricted: ENCORE-EXIOBASE concordance table
    :return: dataframe with index that includes production processes to include from ENCORE
    """
    ENCORE_dep_restricted_df = ENCORE_dep_df.loc[np.unique(ENCORE_to_EXIO_df_restricted.index.values), :]
    ENCORE_dep_restricted_df.sort_index(inplace=True)

    return ENCORE_dep_restricted_df


def aggregate_poids(ENCORE_to_EXIO_df_restricted):
    """
    Function aggregates the multiplier for the production processes that correspond to multiple EXIOBASE sectors and
    vice versa. This is necessary to convert the scores from ENCORE production processes to EXIOBASE sectors.
    :param ENCORE_to_EXIO_df_restricted: ENCORE-EXIOBASE concordance table
    :return: ENCORE-EXIOBASE concordance table with the appropriate multipliers for conversion to EXIOBASE.
    """

    EXIO_to_ENCORE_df = ENCORE_to_EXIO_df_restricted.reset_index()
    # get the number of ENCORE production processes (Process) corresponding to each EXIOBASE sector (IndustryTypeName)
    EXIO_to_ENCORE_df['Count'] = EXIO_to_ENCORE_df.groupby(['IndustryTypeName', 'Process']).transform('count')
    # multiply the POIDs (which represents the number of EXIOBASE sectors that correspond to each Production process)
    # by the number of production processes corresponding to each EXIOBASE sector
    EXIO_to_ENCORE_df['Poids'] = EXIO_to_ENCORE_df['Poids'] * EXIO_to_ENCORE_df['Count']
    # drop the duplicate entries
    EXIO_to_ENCORE_df.drop_duplicates(subset=['IndustryTypeName', 'Process'], keep='first', inplace=True)
    # set the index to EXIOBASE sector, ENCORE Production Process
    EXIO_to_ENCORE_df.set_index(['IndustryTypeName', 'Process'], inplace=True)
    # remove unnecessary columns
    EXIO_to_ENCORE_df.drop(columns=['Count'], inplace=True)
    EXIO_to_ENCORE_df.sort_index(inplace=True)

    return EXIO_to_ENCORE_df

