################################################################################
# Copyright (C) 2025  Michiel van Boven
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

# packages
library(ggplot2)
library(dplyr)
library(readr)
library(cowplot)
library(grid)

source("functions/format_PLOS.R")

#########################################
# Figure S1. Age-specific contact matrix
#########################################

# import and make contact matrix
contact_matrix_name <- "./data/contact_matrix_data_120.txt"
contact_data <- readr::read_delim(contact_matrix_name,
                                  delim = "\t",
                                  show_col_types = FALSE)

# filter data (0–119)
filtered_data <- contact_data %>%
  filter(part_age < 120, cont_age < 120)

# create and fill matrix
age_intervals <- 120
contact_matrix <- matrix(0, nrow = age_intervals, ncol = age_intervals)

for (i in seq_len(nrow(filtered_data))) {
  part_idx <- as.integer(filtered_data$part_age[i]) + 1
  cont_idx <- as.integer(filtered_data$cont_age[i]) + 1
  contact_matrix[part_idx, cont_idx] <- filtered_data$c[i]
}

# symmetrise the matrix
contact_matrix <- (contact_matrix + t(contact_matrix)) / 2

# normalise so dominant eigenvalue is 1
eig <- eigen(contact_matrix)
dominant_eigenvalue <- max(Re(eig$values))
contact_matrix <- contact_matrix / dominant_eigenvalue

# log-transform (avoid -Inf)
log_contact_matrix <- log(contact_matrix)
log_contact_matrix[!is.finite(log_contact_matrix)] <- NA_real_

# prepare data for ggplot, map indices -> ages (0–119)
heatmap_data <- expand.grid(
  i = seq_len(age_intervals),
  j = seq_len(age_intervals)
) %>%
  mutate(
    age_x       = i - 1,
    age_y       = j - 1,
    log_contact = as.vector(log_contact_matrix)
  )

# colour limits
z_min <- min(heatmap_data$log_contact, na.rm = TRUE)
z_max <- max(heatmap_data$log_contact, na.rm = TRUE)

heatmap_plot <- ggplot(heatmap_data, aes(x = age_x, y = age_y, fill = log_contact)) +
  #geom_tile(color=NA, linewidth=0) +
  geom_raster(interpolate = FALSE) +
  scale_fill_gradientn(
    colours = c("#56B4E9","white","#E69F00"),
    limits  = c(z_min, z_max),
    name    = "log C(a, a')"
  ) +
  scale_x_continuous(
    "Age (yr)",
    limits = c(0, 120),
    breaks = seq(0, 120, 20),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    "Age (yr)",
    limits = c(0, 120),
    breaks = seq(0, 120, 20),
    expand = c(0, 0)
  ) +
  coord_fixed() +
  labs(title = "Person-to-person contact rates") +
  theme_pub +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.grid       = element_blank(),
    #axis.title       = element_text(size = BASE_PT),
    #axis.text        = element_text(size = BASE_PT - 1),
    #plot.title       = element_text(hjust = 0.5, size = BASE_PT + 1),
    legend.position  = "bottom",
    #legend.title     = element_text(size = BASE_PT),
    #legend.text      = element_text(size = BASE_PT - 1),
    plot.margin      = margin(6, 6, 6, 6)
  )

heatmap_plot

# save as single-column supplementary figure
save_plos(
  plot     = heatmap_plot,
  filename = "figures/figureS1"
)
