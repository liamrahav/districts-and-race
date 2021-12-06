# Preprocessing Steps
The following steps can reproduce the dataset at `preprocessed/dataset.pickle.bz2`:

1. Run `shapefiles.py` to download the raw shapefiles for all Congressional districts in the project period.
2. Run `score_shapefiles.R` to augment the shapefiles with their compactness scores
   - To do this, you will need to have `mandeR` installed. See the base directory README for more information.
3. Run `preprocess.py` to generate the final dataset from the raw datasets in `raw/`, along with the shapefile information gathered above. The output is a pickle of a Pandas `DataFrame`.

You should run all of the above files from the base directory of the repository. You may need to run `export PYTHONPATH=$(pwd)` for Python to correctly register the directory.