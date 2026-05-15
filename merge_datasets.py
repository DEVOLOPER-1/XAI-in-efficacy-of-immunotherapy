import polars as pl

import pandas as pd

df1 = pl.DataFrame(
    pd.read_csv(
        "data/cbioportal_tabular_downloads/coad_tcga_gdc/data_clinical_patient.txt",
        sep="\t",
        comment="#",
    )
)
df2 = pl.DataFrame(
    pd.read_csv(
        "data/cbioportal_tabular_downloads/coad_tcga_gdc/data_clinical_sample.txt",
        sep="\t",
        comment="#",
    )
)

print(df1.shape, df2.shape)
print(df1.columns)
print(df2.columns)

final_df = df1.join(df2, on="PATIENT_ID", how="inner")
final_df.write_csv("data/clinical.csv")
print(final_df.shape , final_df.columns)
