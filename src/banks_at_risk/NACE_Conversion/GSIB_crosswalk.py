from banks_at_risk.NACE_Conversion.helpers_GSIB_crosswalk import bank_one_sheet

def GSIB_format():
    """
    This function combines all the G-SIB portfolio information into a single dataframe and saves this into the data
    repository.
    :return: NA
    """
    # compile all the G-SIB portfolio information into one dataframe
    bank_one_sheet()
    return None

if __name__ == '__main__':
    GSIB_format()
