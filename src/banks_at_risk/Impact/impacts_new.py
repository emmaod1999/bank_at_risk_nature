import pandas as pd
import numpy as np
from banks_at_risk.Setup.EXIO_paths import EXIO_file_path
from banks_at_risk.Utils.encore_ops import ISIC_to_EXIO
from banks_at_risk.Utils.exio_ops import read_exio
from banks_at_risk.Impact.helpers_impacts_new import read_encore_imp, general_impacts
from banks_at_risk.Impact.helpers_impacts import general_impacts_on_ess, compute_impacts_scope1, generate_id_to_ess

def compute_impacts():
    """
    This function generates the impact scores for direct operations for the EXIOBASE sectors and saves them into
    the data repository
    :return: NA
    """


    # read the ENCORE impact materiality ratings
    ENCORE_imp_df = read_encore_imp()

    # calculate the impact scores for EXIOBASE sectors with the three methodological treatments (mean, max, min)
    EXIO_imp_mean_df, EXIO_imp_min_df, EXIO_imp_max_df = general_impacts(ENCORE_imp_df)

    # read EXIOBASE from the EXIOBASE data repository
    EXIO3 = read_exio(EXIO_file_path)

    # format the impact scores with EXIOBASE sectors and regions and store them in the data repository
    compute_impacts_scope1(EXIO3, EXIO_imp_mean_df)
    compute_impacts_scope1(EXIO3, EXIO_imp_min_df)
    compute_impacts_scope1(EXIO3, EXIO_imp_max_df)

    return None


if __name__ == "__main__":
    compute_impacts()