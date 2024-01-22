import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

chunk = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', chunksize=1000000, index_col=0)
df = pd.concat(chunk)

num_cols = df.select_dtypes(exclude=['object']).columns
fwd_cols = [col for col in num_cols if 'Fwd' in col]
bwd_cols = [col for col in num_cols if 'Bwd' in col]

def get_correlated_features(corr, threshold=0.95):
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                correlated_features.add(corr.columns[i])
    return correlated_features

def plot_heatmap(corr, title):
    mask = np.triu(np.ones_like(corr, dtype=np.bool_))
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, mask=mask)
    plt.title(title)
    plt.show()

corr_fwd = df[fwd_cols].corr()
plot_heatmap(corr_fwd, 'Correlation Heatmap - Fwd Columns')
correlated_features_fwd = get_correlated_features(corr_fwd)
print("Correlated Features in Fwd Columns:", correlated_features_fwd)

corr_bwd = df[bwd_cols].corr()
plot_heatmap(corr_bwd, 'Correlation Heatmap - Bwd Columns')
correlated_features_bwd = get_correlated_features(corr_bwd)
print("Correlated Features in Bwd Columns:", correlated_features_bwd)

correlated_features = correlated_features_fwd | correlated_features_bwd
df.drop(columns=correlated_features, inplace=True)
dropped_count = len(correlated_features)
print("Total Correlated Features Dropped:", dropped_count)

del num_cols, fwd_cols, bwd_cols
gc.collect()

df.shape

df.head()

styled_df = (
    df.describe()
    .drop("count", axis=0)
    .style.background_gradient(axis=0, cmap="magma")
    .set_properties(**{"text-align": "center"})
    .set_table_styles([{"selector": "th", "props": [("background-color", "k")]}])
    .set_caption("Summary Statistics")
)

styled_df

from sklearn import preprocessing 
for f in df.columns: 
    if df[f].dtype=='object': 
        label = preprocessing.LabelEncoder() 
        label.fit(list(df[f].values)) 
        df[f] = label.transform(list(df[f].values))

mb = df.memory_usage().sum() / 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(mb))

df=df.astype('float32')

mb = df.memory_usage().sum() / 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(mb))

df.head()

df.tail()

