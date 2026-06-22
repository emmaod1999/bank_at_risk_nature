import pandas as pd
from banks_at_risk.Setup.ENCORE_paths import Updated_ENCORE_dep_mat_path
from banks_at_risk.Utils.encore_ops import ISIC_to_EXIO
from banks_at_risk.Dependency.helpers_dependency import create_dependencies_df


def read_encore_dep():
    """
    Function reads the dependency materiality ratings from the ENCORE Knowledge base
    :return: ENCORE dependency materiality ratings
    """
    # read ENCORE dependency materiality ratings with required columns
    ENCORE_dep_df = pd.read_csv(Updated_ENCORE_dep_mat_path, index_col=[0, 1, 2, 3, 4, 5], header=0, skiprows=[0,1])

    # create dictionary of numerical equivalents for ENCORE ratings
    rating_nums_dict = {'VH':1, 'H':0.8, 'M':0.6, 'L':0.4, 'VL':0.2, 'N/A':0, 'ND':0}

    # get list of ecosystem services
    services = ENCORE_dep_df.columns.tolist()

    for service in services:
        # replace materiality ratings with numbers
        ENCORE_dep_df = ENCORE_dep_df.replace({f"{service}": rating_nums_dict})

    ENCORE_dep_df = ENCORE_dep_df.fillna(0.0)

    return ENCORE_dep_df

def general_dependencies(dep_df):
    """
    Calculates dependencies in EXIOBASE format for the three calculation types
    :param dep_df:
    :return:
    """
    # convert to exiobase sectors with appropriate calculation type
    dep_mean_EXIO_df = ISIC_to_EXIO(dep_df, "mean")
    dep_max_EXIO_df = ISIC_to_EXIO(dep_df, "max")
    dep_min_EXIO_df = ISIC_to_EXIO(dep_df, "min")

    # name the df after the three methodological treatments to distinguish
    dep_mean_EXIO_df.name = 'mean'
    dep_max_EXIO_df.name = 'max'
    dep_min_EXIO_df.name = 'min'



    return dep_mean_EXIO_df, dep_max_EXIO_df, dep_min_EXIO_df
