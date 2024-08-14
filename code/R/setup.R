
# Setup script for the OPP project

library(tidyverse)
library(sf)
library(viridis)

getwd()

# Read in the sampled roof materials from the PlanetScope SuperDove over D.C.
sample <- read_csv('../../data/tabular/mod/dc_data/training/dc_data_reference_sampled.csv') %>%
  rename(rid = ...1)
sample.mnf <- read_csv('../../data/tabular/mod/dc_data/training/dc_data_reference_sampled_mnf.csv') %>%
  rename(rid = ...1)