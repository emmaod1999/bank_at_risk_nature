import pymrio
import numpy as np
import pymrio


def read_exio(EXIO_file_path):
    """
    Function imports the EXIOBASE3 MRIO and calculates all required tables
    :param EXIO_file_path: File path where EXIOBASE files are stored
    :return: EXIOBASE3 MRIO with all required tables calculated
    """
    # read EXIOBASE3 from files
    EXIO3 = pymrio.parse_exiobase3(path=EXIO_file_path)
    # calculate all required tables
    EXIO3.calc_all()

    return EXIO3

def exio_supply_shock(EXIO3):
    """
    Generates the relative importances of each sector-region pair in the upstream supply chain of each sector-region
    pair using the Leontief (nxn) matrix.
    :param EXIO3: EXIOBASE MRIO with all required tables calculated
    :return: Relative importance matrix (nxn)
    """
    # get the Leontief from EXIOBASE and concert to a matrix
    L_matrix = EXIO3.L.to_numpy()
    # remove the sector-region from its own upstream supply chain
    L_min_I = L_matrix - np.eye(L_matrix.shape[0])
    # calculate the column sums for each column (with itself removed)
    col_sums = np.sum(L_min_I, axis=0)
    # divide the L matrix without the sector-region pair in the upstream supply chain (L_min_I) by the column sum to get
    # the relative importance of each sector-region pair in the upstream supply chain of the column sector-region pair
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_imp_array = np.where(col_sums == 0, 0, np.divide(L_min_I, col_sums[None, :]))

    return rel_imp_array