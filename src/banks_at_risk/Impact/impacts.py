import pandas as pd
import numpy as np
from banks_at_risk.Setup.EXIO_paths import EXIO_file_path
from banks_at_risk.Utils.encore_ops import read_encore_to_exio, get_restricted_ENCORE, get_incl_sectors, aggregate_poids
from banks_at_risk.Utils.exio_ops import read_exio
from banks_at_risk.Impact.helpers_impacts import read_encore_imp, general_impacts, general_impacts_on_ess, compute_impacts_scope1, generate_id_to_ess

def compute_impacts():
    """
    This function generates the impact scores for direct operations for the EXIOBASE sectors and saves them into
    the data repository
    :return: NA
    """
    # read EXIOBASE from the EXIOBASE data repository
    EXIO3 = read_exio(EXIO_file_path)

    # read the ENCORE impact materiality ratings
    ENCORE_imp_df = read_encore_imp()

    # get the conversion from ENCORE production processes to EXIOBASE sectors
    # read the ENCORE-EXIOBASE concordance table
    ENCORE_to_EXIO_df_restricted = read_encore_to_exio()
    # generate multipliers to convert the production processes to EXIOBASE sectors
    EXIO_to_ENCORE_df = aggregate_poids(ENCORE_to_EXIO_df_restricted)
    # remove duplicates to get the ENCORE production processes with appropriate EXIOBASE analogues
    ENCORE_imp_restricted_df = get_restricted_ENCORE(ENCORE_imp_df, ENCORE_to_EXIO_df_restricted)
    # get EXIOBASE sectors with appropriate ENCORE production process analogues
    EXIO_included_sectors = get_incl_sectors(EXIO3, ENCORE_to_EXIO_df_restricted)

    # calculate the impact scores for EXIOBASE sectors with the three methodological treatments (mean, max, min)
    EXIO_imp_mean_df, EXIO_imp_min_df, EXIO_imp_max_df = general_impacts(EXIO3, EXIO_to_ENCORE_df, ENCORE_imp_restricted_df, EXIO_included_sectors)

    # format the impact scores with EXIOBASE sectors and regions and store them in the data repository
    compute_impacts_scope1(EXIO3, EXIO_imp_mean_df)
    compute_impacts_scope1(EXIO3, EXIO_imp_min_df)
    compute_impacts_scope1(EXIO3, EXIO_imp_max_df)

    return None


if __name__ == "__main__":
    compute_impacts()





