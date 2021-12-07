########################################################################
# NOTE: Make sure you have installed mandeR before running this file
# You can do this via the install_mandeR.R script in the base directory
#
# Make sure you run the file from the base directory in this repository
# Use command `Rscript dataset/score_shapefiles.R`
#
# Make sure you have run `python dataset/shapefiles.py` before running 
# this script to download + clean the needed shapefiles.
########################################################################
library(stringr)

base_dir = "dataset/preprocessed/shapefiles/"

congress_nums <- 93:113
for (c_num in congress_nums) {
    to_append = "/districtShapes/districtShapes.shp"
    shapefile <- paste(paste(base_dir, c_num, sep=""), to_append, sep="")
    
    print(paste(paste("Augmenting", shapefile), "..."))
    tryCatch({
        # Add all scores for now, we will pick which one to use later
        mandeR::augmentShapefileWithScores(shapefile,scores=c('all'))
    }, error=function(e){
        print(paste("ERROR OCURRED:", conditionMessage(e)))
    })
    
}

print("Done!")
