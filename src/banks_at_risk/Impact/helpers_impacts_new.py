import pandas as pd
from banks_at_risk.Setup.ENCORE_paths import Updated_ENCORE_imp_mat_path, Updated_ENCORE_imp_pressure_component_path, Updated_ENCORE_imp_ESS_component_path
from banks_at_risk.Utils.encore_ops import ISIC_to_EXIO
def read_encore_imp():

    # get the relevant sheets from ENCORE
    # read ENCORE dependency materiality ratings with required columns
    pressure_mats_df = pd.read_csv(Updated_ENCORE_imp_mat_path, index_col=[1, 2, 3, 4, 5, 6], header=0, skiprows=[0, 1]).drop(columns='#')
    pressure_comps_df = pd.read_csv(Updated_ENCORE_imp_pressure_component_path, index_col=[], header=[0])
    ESS_comps_df = pd.read_csv(Updated_ENCORE_imp_ESS_component_path)

    # merge pressure to components and ESS and components
    pressure_comps_ESS_df = pd.merge(pressure_comps_df.set_index(['Ecosystem component']), ESS_comps_df.set_index(['Ecosystem components']), right_index=True, left_index=True, how="outer")

    pressure_mats_long_df = pressure_mats_df.reset_index().melt(
                        id_vars=["ISIC Unique code", "ISIC Section", "ISIC Division", "ISIC Group", "ISIC Class", "ISIC level used for analysis"],
                        value_vars=["Disturbances (e.g noise, light)", "Area of freshwater use", "Emissions of GHG", "Area of seabed use", "Emissions of non-GHG air pollutants", "Other biotic resource extraction (e.g. fish, timber)", "Other abiotic resource extraction", "Emissions of toxic soil and water pollutants", "Emissions of nutrient soil and water pollutants", "Generation and release of solid waste", "Area of land use", "Introduction of invasive species", "Volume of water use"],
                        value_name="Pressure Materiality",
                        var_name="Pressure").fillna(0.0)


    # convert both into numbers based on respective scoring
    mat_scores = {'Materialities': ['VL', 'L', 'M', 'H', 'VH'], 'Score': [0.2, 0.4, 0.6, 0.8, 1.0]}
    mat_scores_df = pd.DataFrame(data=mat_scores)
    mat_scores_dict = mat_scores_df.set_index('Materialities')['Score']
    pressure_mats_long_df['Pressure Materiality'] = pressure_mats_long_df['Pressure Materiality'].map(mat_scores_dict)

    imp_scores = {'Importance':['G', 'A', 'R'], 'Score':[0.333333333, 0.666666668, 1]}
    imp_scores_df = pd.DataFrame(data=imp_scores)
    imp_scores_dict = imp_scores_df.set_index('Importance')['Score']
    pressure_comps_ESS_df['Rating'] = pressure_comps_ESS_df['Rating'].map(imp_scores_dict)
    pressure_comps_ESS_df['Pressures(/Impact drivers)'] = pressure_comps_ESS_df['Pressures(/Impact drivers)'].replace('Disturbances (e.g noise, light) - Light', 'Disturbances (e.g noise, light)')
    pressure_comps_ESS_df['Pressures(/Impact drivers)'] = pressure_comps_ESS_df['Pressures(/Impact drivers)'].replace('Disturbances (e.g noise, light) - Noise', 'Disturbances (e.g noise, light)')

    # merge importance and materiality together
    sector_pressure_comps_ESS_df = pd.merge(pressure_comps_ESS_df.reset_index().set_index(['Pressures(/Impact drivers)']), pressure_mats_long_df.set_index(['Pressure']), left_index=True, right_index=True, how='outer')
    sector_pressure_comps_ESS_df = sector_pressure_comps_ESS_df.fillna(0.0)

    sector_pressure_comps_ESS_df = sector_pressure_comps_ESS_df.reset_index().set_index(['ISIC Unique code', 'ISIC Section', 'ISIC Division', 'ISIC Group', 'ISIC Class', 'ISIC level used for analysis'])

    sector_pressure_comps_ESS_df = sector_pressure_comps_ESS_df.drop(columns=['Link between pressures and mechanisms', 'Timescale', 'Direct vs. indirect', 'Spatial characteristics', 'Reference', 'Long reference', 'Article link', 'Link assessment', 'Link_assessment', 'Justification'])
    sector_pressure_comps_ESS_df['Pressure to ESS Materiality'] = sector_pressure_comps_ESS_df['Rating'] * sector_pressure_comps_ESS_df['Pressure Materiality']

    sector_pressure_ESS_df = sector_pressure_comps_ESS_df.drop(columns=['Mechanism causing state change', 'Rating', 'Ecosystem types', 'Ecosystem component', 'Pressure Materiality', 'Pressures(/Impact drivers)']).reset_index().groupby(['ISIC Unique code', 'ISIC Section', 'ISIC Division', 'ISIC Group',  'ISIC Class', 'ISIC level used for analysis', 'Ecosystem services']).mean()

    sector_pressure_ESS_df = sector_pressure_ESS_df.reset_index()[
        sector_pressure_ESS_df.reset_index()['ISIC Unique code'] != 0]
    sector_pressure_ESS_df = sector_pressure_ESS_df[sector_pressure_ESS_df['Ecosystem services'] != 0].set_index(
        ['ISIC Unique code', 'ISIC Section', 'ISIC Division', 'ISIC Group', 'ISIC Class',
         'ISIC level used for analysis', 'Ecosystem services'])

    imp_ESS_df = sector_pressure_ESS_df.reset_index().pivot(index=['ISIC Unique code', 'ISIC Section', 'ISIC Division', 'ISIC Group', 'ISIC Class',
         'ISIC level used for analysis'], columns='Ecosystem services', values='Pressure to ESS Materiality')


    return imp_ESS_df

def general_impacts(imp_df):
    """
    Converts from ISIC to EXIOBASE with each calculation type
    :param imp_df:
    :return: EXIOBASE sectors impact materiality for ESS
    """
    # convert to exiobase sectors with appropriate calculation type
    imp_mean_EXIO_df = ISIC_to_EXIO(imp_df, "mean")
    imp_max_EXIO_df = ISIC_to_EXIO(imp_df, "max")
    imp_min_EXIO_df = ISIC_to_EXIO(imp_df, "min")

    # name the df after the three methodological treatments to distinguish
    imp_mean_EXIO_df.name = 'mean'
    imp_max_EXIO_df.name = 'max'
    imp_min_EXIO_df.name = 'min'


    return imp_mean_EXIO_df, imp_max_EXIO_df, imp_min_EXIO_df