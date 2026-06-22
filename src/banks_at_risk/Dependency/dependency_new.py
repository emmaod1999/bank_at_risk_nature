import pandas as pd
import numpy as np
from banks_at_risk.Utils.exio_ops import read_exio
from banks_at_risk.Setup.EXIO_paths import EXIO_file_path
from banks_at_risk.Dependency.helpers_dependency_new import read_encore_dep, general_dependencies
from banks_at_risk.Dependency.helpers_dependency import compute_dependencies_scope1
def compute_dependencies():
    """
    This function generates the dependency scores for direct operations for the EXIOBASE sectors and saves them into
    the data repository
    :return: NA
    """

    # read the ENCORE dependency materiality ratings file from the ENCORE data repository
    ENCORE_dep_df = read_encore_dep()

    # # calculate the dependency scores for EXIOBASE sectors with the three methodological treatments (mean, max, min)
    EXIO_dep_mean_df, EXIO_dep_max_df, EXIO_dep_min_df = general_dependencies(ENCORE_dep_df)

    # read EXIOBASE from the EXIOBASE data repository
    EXIO3 = read_exio(EXIO_file_path)

    # # format the dependency scores with EXIOBASE sectors and regions and store them in the data repository
    compute_dependencies_scope1(EXIO3, EXIO_dep_mean_df)
    compute_dependencies_scope1(EXIO3, EXIO_dep_max_df)
    compute_dependencies_scope1(EXIO3, EXIO_dep_min_df)

    return None

if __name__ == "__main__":
    compute_dependencies()