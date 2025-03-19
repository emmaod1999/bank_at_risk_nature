from banks_at_risk.Anonymize_banks.anonymize_banks import anonymize_banks
from banks_at_risk.Dependency.dependency import compute_dependencies
from banks_at_risk.Impact.impacts import compute_impacts
from banks_at_risk.NACE_Conversion.NACE_conversion import convert_score_to_NACE
from banks_at_risk.NACE_Conversion.GSIB_crosswalk import GSIB_format
from banks_at_risk.Value_at_Risk.value_at_risk_finance import calc_nVAR, calc_nVAR_GSIB
from banks_at_risk.Value_at_Risk_Analysis.value_at_risk_analysis import analyze_nvar, analyze_nvar_system

def run_nvar_analysis():
    """
    This function runs all the analysis.
    :return: NA
    """
    # anonymize_banks()
    compute_dependencies()
    compute_impacts()
    convert_score_to_NACE()
    calc_nVAR()
    # displays figures at the (anonymized) bank level
    analyze_nvar()
    # displays figures aggregated to the system level
    analyze_nvar_system()
    return None

def run_nvar_analysis_GSIB():
    """
    This function runs all the functions required to calculate the endogenous risk and generates the figures and analysis
    :return: NA
    """
    # computes the dependency score
    compute_dependencies()
    # computes the impact score
    compute_impacts()
    # foramts the portfolio data
    GSIB_format()
    # calculates the endogenous risk of the portfolios
    calc_nVAR_GSIB()
    # displays figures at the bank level
    analyze_nvar(anonymize=False)
    # displays figures aggregated to the system level
    analyze_nvar_system()
    return None

if __name__ == '__main__':
   # run_nvar_analysis()
   run_nvar_analysis_GSIB()