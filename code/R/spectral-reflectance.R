
library(tidyverse)
library(sf)
library(viridis)
library(patchwork)

# load the roofprint reflectance
dc.sample <- read_csv('data/tabular/mod/dc_data/dc_spectral_response.csv') %>%
 mutate(region = "Washington, D.C.",
        class_code = str_extract(uid, "[A-Za-z]+$"))
denver.sample <- read_csv('data/tabular/mod/denver_data/denver_spectral_response.csv') %>%
 mutate(region = "Denver, Colorado",
        class_code = str_extract(uid, "[A-Za-z]+$"))
glimpse(dc.sample)

# load the material lookup table
lookup = read_csv('data/tabular/raw/variable_lookup/RoofCoverStndCode_encoding.csv') %>%
 rename(class_code = Code,
        material = Description) %>%
 select(c(class_code, material))

# merge the two study areas and join the description
sample <- dc.sample %>%
 bind_rows(denver.sample) %>%
 left_join(., lookup, by="class_code")
glimpse(sample)


#============Original PSB.SD Bands============#

# Set up a dictionary for the band/wavelength combinations
psband <- c("coastal_blue","blue","green_i","green","yellow","red","rededge","nir")
wavelength <- c(443,490,531,565,610,665,705,865)
band_wave <- data.frame(psband,wavelength)
rm(psband,wavelength)

# Pivot the table and tidy
sample_p <- sample %>%
 select(c(region, class_code, material, coastal_blue, blue, green_i, green, yellow, red, rededge, nir)) %>%
 pivot_longer(
  cols = -c(region, class_code, material),
  names_to = "psband",
  values_to = "reflectance"
 ) %>%
 left_join(., band_wave, by = "psband")

# Set up a vector for the colors
cols <- c("#74a9cf", "#034e7b", "#a1d99b", "#006d2c", "#DCC95B", "#96534F", "#D29D9D", "#826D42")
bands <- c("Coastal Blue","Blue","Green(I)","Green","Yellow","Red","Red-edge","NIR")
materials <- c('Asphalt','Concrete','Composition Shingle', 
               'Metal','Shingle','Slate',
               'Tar and gravel','Tile',
               'Urethane','Wood shake/shingle')
# Set up a data frame for the rectangles
rects <- data.frame(
 ymin = -Inf, ymax = Inf, 
 xmin = c(438,485,526,560,605,660,700,860), 
 xmax = c(448,495,536,570,615,670,710,870), 
 band = bands,
 col = cols
)

# Normalize reflectance within each region and class
df <- sample_p %>%
 group_by(region, class_code, material, psband) %>%
 summarize_all(.funs = mean) %>%
 group_by(region, material) %>%
 mutate(
  refl_n = (reflectance - min(reflectance)) / (max(reflectance) - min(reflectance))
 )

# Define custom color mapping for class codes
roof_colors <- c(
 'Asphalt' = '#f67088', 'Concrete' = '#db8831', 'Composition Shingle' = '#ad9c31', 
 'Metal' = '#77aa31', 'Shingle' = '#33b07a', 'Slate' = '#35aca4', 
 'Tar and gravel' = '#38a8c5', 'Tile' = '#6e9af4', 
 'Urethane' = '#cc79f4', 'Wood shake/shingle' = '#f565cc'
)

# Plot with faceting by region
spectral_response <- ggplot() +
 geom_rect(data = rects, 
           aes(xmin = xmin, xmax = xmax, 
               ymin = ymin, ymax = ymax, 
               fill = band), alpha = 0.2) +
 scale_fill_manual(
  values = c("#74a9cf", "#034e7b", "#a1d99b", "#006d2c", "#DCC95B", "#96534F", "#D29D9D", "#826D42"), 
  labels = c("Coastal Blue", "Blue", "Green(I)", "Green", "Yellow", "Red", "Red-edge", "NIR"),
  breaks = c("Coastal Blue", "Blue", "Green(I)", "Green", "Yellow", "Red", "Red-edge", "NIR")) +
 geom_line(data = df, aes(x = wavelength, y = reflectance, color = material)) +
 scale_color_manual(
  values = roof_colors, 
  name = "Material"
 ) +
 facet_wrap(~region, ncol = 2) +  # Facet side-by-side by region
 guides(fill = guide_legend(ncol = 2), color = guide_legend(ncol = 1)) +
 theme_bw(base_size = 8) +
 labs(fill = "Band Center", color = "Material",
      x = "Wavelength (nm)", y = "Reflectance")
spectral_response

# Save it out
ggsave(
 spectral_response, 
 file = "figures/PSScene8Band_RoofprintSpectralResponse.png",
 dpi = 500, bg="white", width=8, height=4.25) # adjust dpi accordingly


#================Spectral indices===================#

# Pivot the table and tidy
sample_si <- sample %>%
 select(c(region, class_code, material, NDRE, VgNIRBI, VrNIRBI, 
          NDBIbg, NDBIrg, NISI, mnf1)) %>%
 pivot_longer(
  cols = -c(region, class_code, material),
  names_to = "index",
  values_to = "reflectance"
 ) %>%
 group_by(region, index) %>%
 mutate(
  refl_norm = 
   (reflectance - min(reflectance)) / 
   (max(reflectance) - min(reflectance))
 )

# make the plot
index_response <- ggplot() +
 geom_boxplot(
  data = sample_si,
  aes(y = refl_norm, fill = factor(material)),
  outlier.size = 0.1
 ) +
 scale_fill_manual(
  values = roof_colors,  
  name = "Material"
 ) +
 facet_grid(region ~ index) +  # Facet by region and spectral index
 labs(fill = "Material") +
 theme_bw(base_size = 8) +
 theme(legend.position = "bottom",
       axis.text.x = element_blank())

index_response

# Save it out
ggsave(
 index_response, 
 file = "figures/PSScene_SI_RoofprintSpectralResponse.png",
 dpi = 500, bg="white", width=8, height=4.25) # adjust dpi accordingly



##############################
# Version 2 #

# Pivot the table and tidy
sample_p2 <- sample %>%
 select(c(region, class_code, material, coastal_blue, blue, 
          green_i, green, yellow, red, rededge, nir)) %>%
 pivot_longer(
  cols = -c(region, class_code, material),
  names_to = "index",
  values_to = "reflectance"
 ) %>%
 group_by(region, index) %>%
 mutate(
  refl_norm = 
   (reflectance - min(reflectance)) / 
   (max(reflectance) - min(reflectance))
 )

# make the plot
spectral_response2 <- ggplot() +
 geom_boxplot(
  data = sample_p2,
  aes(y = refl_norm, fill = factor(material)),
  outlier.size = 0.1
 ) +
 scale_fill_manual(
  values = roof_colors,  
  name = "Material"
 ) +
 facet_grid(region ~ index) +  # Facet by region and spectral index
 labs(fill = "Material") +
 theme_bw(base_size = 8) +
 theme(legend.position = "bottom",
       axis.text.x = element_blank())

spectral_response2
