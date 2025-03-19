# Banks at Risk 

A Package that will take sector-region data for banks and produce the nVAR metric. 

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
