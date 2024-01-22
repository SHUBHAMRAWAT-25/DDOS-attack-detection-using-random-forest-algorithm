import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import gc
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

chunk = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', chunksize=1000000, index_col=0)
df = pd.concat(chunk)

df.drop_duplicates(inplace=True)
print("After dropping duplicates, shape:", df.shape)

num_rows = len(df)
for col in df.columns:
    cnts = df[col].value_counts(dropna=False)
    top_pct = (cnts/num_rows).iloc[0]
    
    if top_pct > 0.98:
        print('{0}: {1:.2f}%'.format(col, top_pct*100))
        print(cnts)
        print()

colsToDrop = np.array([' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags','FIN Flag Count',' URG Flag Count','Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate'])
missing = df.isna().sum()
missing = pd.DataFrame({'count': missing, '% of total': missing/len(df)*100}, index=df.columns)
print(missing)

colsToDrop = np.union1d(colsToDrop, missing[missing['% of total'] >= 50].index.values)
dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values
df['Flow Bytes/s'].replace(np.inf, np.nan, inplace=True)
df[' Flow Packets/s'].replace(np.inf, np.nan, inplace=True)
dropnaCols = np.union1d(dropnaCols, ['Flow Bytes/s', ' Flow Packets/s'])
print("Columns to drop:", colsToDrop)
print("Columns with missing values to drop:", dropnaCols)

initial_shape = df.shape
df.drop(columns=colsToDrop, inplace=True)
print("After dropping columns with high missing values, shape:", df.shape)

df.dropna(subset=dropnaCols, inplace=True)
print("After dropping rows with missing values in specific columns, shape:", df.shape)

gc.collect()

