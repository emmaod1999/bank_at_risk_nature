import pandas as pd
import numpy as np
from banks_at_risk.Setup.finance_paths import finance_data_path, finance_data_anonymized_path

def anonymize_banks():
    finance_df = pd.read_csv(finance_data_path, header=[0])

    bank_names = np.unique(finance_df['Bank']).tolist()
    anonymized_names = []
    for i in range(1, (len(bank_names)+1)):
        anonymized_names.append(f'Bank {i}')
    
    i = 0
    for bank in bank_names:
        finance_df = finance_df.replace({f'{bank}':f'{anonymized_names[i]}'})
        i = i + 1
    
    finance_df = finance_df.set_index('Bank')

    finance_df.to_csv(f'{finance_data_anonymized_path}')

    return None


if __name__ == "__main__":
    anonymize_banks()