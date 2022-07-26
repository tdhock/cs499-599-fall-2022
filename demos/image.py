import pandas as pd
zip_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/zip.test.gz",
    header=None,
    sep=" ")

# index on columns.
zip_df[0] # first column.

# subset on rows (everything but the first).
zip_df[1:] # still 257 columns.

# subset on columns using index (maybe named).
zip_df.loc[:,1:]

# subset on columns using column numbers (always int).
zip_df.iloc[:,1:]

# list all attributes/methods.
dir(zip_df)

zip_mat = zip_df.iloc[:,1:].to_numpy()
zip_mat.shape

import numpy as np

# BAD 16 repeated
index_vec = np.arange(16)
np.repeat(index_vec, 16)

# GOOD
n_pixels = 16
index_vec = np.arange(n_pixels)
image_index = 0
zip_mat[image_index,:] 
one_image_df = pd.DataFrame({
    "row":np.tile(index_vec, n_pixels),
    "col":np.repeat(index_vec, n_pixels),
    "intensity":zip_mat[image_index,:]
})
one_image_df

import seaborn.objects as so
plot_obj = so.Plot().add(
    mark=THERE IS NO EQUIVALENT OF GEOM_RECT YET https://seaborn.pydata.org/nextgen/
    data=one_image_df, 
    x="col", y="row", fill="intensity")
so.Plot(mpg, x="displ", y="hwy", color="drv").add(so.Line(), so.Agg()).add(so.Scatter())

import plotnine as p9

gg = p9.ggplot()+\
    p9.geom_raster(
        p9.aes(
            x="col",
            y="col",#should be row eventually.
            fill="intensity",
            ),
        data=one_image_df)
gg.save("2022_01_13.png")
