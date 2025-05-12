# Banks at Risk 

A Package that will take sector-region data for banks and produce the nVAR metric. 

Code and Publicly Available Data for "Banks at Risk: Quantifying Endogenous Nature Risk Exposure of G-SIBs" paper

This is the code for the research presented in "Banks at Risk: Quantifying Endogenous Nature Risk Exposure of G-SIBs"

The code was written on a Windows 11 Home Operating System (OS Build: 22631.3880) using JetBrains PyCharm 2023.2.4 and Python 3.9.13. No non-standard hardware is required. Install time on a "normal" computer should be minimal.
The template for the format of the data is in the financial_data folder in the financial_data_no_K.csv file and the file should populated with relevant Bank data. A demo file is included with example bank data to run the analysis.

## Files 

- env.yml: Specifies conda environment to run in micromamba:
$ micromamba create -f env.yml -y
To run python scripts in this environment, first run:
$ micromamba activate banks_at_risk

## Run
- run_all_analysis.py: will run all the code and produce figures
- if you wish to anonymize the banks, edit the run_all_analysis.py script so that anonymize=True in line 29. 

## Required Data Inputs
- Input bank portfolio data into Data/financial_data/Entities_alloc with each bank portfolio as a separate csv named as the bank 
- An empty sheet has been included for formatting purposes
- The EXIOBASE depository is also not included and can be downloaded (IOT_2022_ixi.zip) here: https://zenodo.org/records/5589597
- Put the EXIOBASE zipfile in the exiobase_download_online folder in the Data depository (/src/banks_at_risk/Data/exiobase_download_online/)

## Expected output
- The results figures in the main text should be generated in "Data/Value at Risk Figures/"
- the bank-level sector heatmap figure available in the supplementary materials should be generated in "Data/Value at Risk Figures/Sector"
- the significance values should be generated in "Data/Value at Risk Figures/Value at Risk Significance" as excel files
