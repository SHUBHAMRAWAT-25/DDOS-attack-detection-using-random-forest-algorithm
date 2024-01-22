import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gc
import preprocess
import feature
import trainmodel

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

chunk = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', chunksize=1000000, index_col=0)
df = pd.concat(chunk)


print("Balanced Dataset shape:")
print(df.shape)
total = len(df) * 1.

plt.figure(figsize=(8, 6))
ax = sns.countplot(x=" Label", data=df)  # Make sure there's a space before 'Label' in the column name
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100 * p.get_height() / total), (p.get_x() + 0.1, p.get_height() + 5))

ax.yaxis.set_ticks(np.linspace(0, total, 2))
ax.set_yticklabels(map('{:.1f}%'.format, 100 * ax.yaxis.get_majorticklocs() / total))
plt.title('Label Distribution (Balanced Dataset)')
plt.show()

styled_df = (
    df.describe()
    .drop("count", axis=0)
    .style.background_gradient(axis=0, cmap="magma")
    .set_properties(**{"text-align": "center"})
    .set_table_styles([{"selector": "th", "props": [("background-color", "k")]}])
    .set_caption("Summary Statistics")
)
gc.collect()


with open("styled_summary.html", "w") as f:
    f.write(styled_df.to_html())








