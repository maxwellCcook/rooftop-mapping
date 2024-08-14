
# Spatially balanced random sampling from OSM-ZTRAX footprints

library(tidyverse)
library(sf)
library(spsurvey)

set.seed(123)

getwd()

# Load the data

df <- st_read("data/spatial/raw/dc_data/ocm_w_ztrax_11001_matched.gpkg")
# df <- st_read("data/spatial/raw/denver_data/ocm_w_ztrax_08031_matched.gpkg")
glimpse(df)


# Tidy the data frame 

df <- df %>%
  select(OBJECTID,area,RoofCoverStndCode,NoOfBuildings,BuildingAreaSqFt) %>%
  # Create a cleaned class code attribute
  mutate(class_code = as.factor(RoofCoverStndCode)) %>%
  # Remove the "bad" classes
  filter(!class_code %in% c('', 'BU', 'OT')) %>%
  st_centroid()
summary(df$class_code)

# Also remove codes with too small a sample size
df <- df %>% filter(class_code != "CN" & class_code != "WD")
summary(df$class_code)

# Create the spatially balanced random sample using the
# Generalized Random Tessellation Stratified (GRTS) algorithm
# Equal stratified between 'class_codes' and weighted by area
# Write the results out to a point vector

strata_n <- c(CS=50,ME=50,SH=50,SL=50,TL=50,UR=50,WS=50)
rand.samp <- spsurvey::grts(df, n_base = strata_n, stratum_var = "class_code", aux_var = "area")
rand.samp
st_write(rand.samp$sites_base,"data/spatial/mod/dc_data/ocm_w_ztrax_11001_matched_randSamp50.gpkg")
