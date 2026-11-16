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
packages <- c(
  "ggplot2", "dplyr", "tidyr", "readr", "cowplot",
  "grid"
)

# Install only missing packages (grid comes with R).
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0L) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Load all packages.
lapply(packages, library, character.only = TRUE)

source("functions/format_PLOS.R")

sci_fmt <- function(x) {
  fmt <- format(x, scientific = TRUE)
  gsub("e\\+0?", "e", fmt)
}

#################################################
# Figure 4. Time plots for influenza A scenario #
#################################################

#df_ts <- read_csv("data/time_series_influenza_2025_20251117.csv")
#df_ta <- read_csv("data/time_age_influenza_2025_20251117.csv")
output_tag <- "figure4_baseline"
first_age_s <- 60  # must match the Julia time-series simulation
fva_tag <- paste0("fva", gsub(".", "p", as.character(first_age_s), fixed = TRUE))
df_ts <- read_csv(paste0("output/time_series_", output_tag, "_", fva_tag, ".csv"))
df_ta <- read_csv(paste0("output/time_age_", output_tag, "_", fva_tag, ".csv"))

t_ref <- 150

df_ts <- df_ts %>%
  mutate(t_rel = t - t_ref)

# colours
col_low    <- "#56B4E9"
col_mid    <- "#E69F00"
col_high   <- "red"
col_low_w  <- "white" # "#c7e4fb"

# time-series theme

remove_x_all <- theme(
  axis.title.x = element_blank(),
  axis.text.x  = element_blank(),
  axis.ticks.x = element_blank()
)

# top panel: compartments over time
df_top <- df_ts %>%
  select(t_rel, S, I, R, V) %>%
  pivot_longer(-t_rel, names_to = "compartment", values_to = "value") %>%
  mutate(
    compartment = factor(compartment, levels = c("S", "I", "R", "V"))
  )

p_top <- ggplot() +
  geom_line(
    data = dplyr::filter(df_top, compartment != "I"),
    aes(x = t_rel, y = value, colour = compartment),
    linewidth = 0.4
  ) +
  geom_line(
    data = dplyr::filter(df_top, compartment == "I"),
    aes(x = t_rel, y = value, colour = compartment),
    linewidth = 0.4
  ) +
  scale_colour_manual(
    values = c(
      S = col_low,
      I = col_mid,
      R = "grey20",
      V = "grey70"
    ),
    breaks = c("S", "I", "R", "V"),  # <- force legend order
    name = NULL
  ) +
  scale_x_continuous(
    limits = c(0, 100),
    expand = c(0, 0),
    breaks = seq(0, 100, 20)
  ) +
  scale_y_continuous(labels = sci_fmt) +
  labs(
    x = NULL,
    y = "Number"
  ) +
  theme_pub +
  remove_x_all +
  theme(
    legend.position      = c(0.02, 0.97),
    legend.justification = c(0, 1),
    legend.background    = element_rect(fill = "white", colour = "grey80"),
    legend.margin        = margin(t = 1, r = 5, b = 2, l = 2),
    legend.key.width     = unit(0.6, "lines"),
    legend.key.height    = unit(0.6, "lines")
  )

# middle panel: age-time incidence heatmap
df_mid <- df_ta %>%
  mutate(t_rel = t - t_ref) %>%
  pivot_longer(
    cols = starts_with("age_"),
    names_to = "age_label",
    values_to = "I"
  ) %>%
  mutate(
    age = as.numeric(sub("age_", "", age_label))
  ) %>%
  arrange(t_rel, age)

t_vals   <- sort(unique(df_mid$t_rel))
dt_vals  <- diff(t_vals)
dt       <- max(dt_vals)

age_vals <- sort(unique(df_mid$age))
da_vals  <- diff(age_vals)
da       <- max(da_vals)

I_min <- min(df_mid$I, na.rm = TRUE)
I_max <- max(df_mid$I, na.rm = TRUE)

# warning: large file. try rasterisation
p_mid <- ggplot(df_mid, aes(x = t_rel, y = age, fill = I)) +
#  ggrastr::rasterise(
#    geom_tile(width = dt * 1.02, height = da * 1.02),
#    dpi = 150   
#  ) +
  geom_tile(width = dt * 1.02, height = da * 1.02) +
  annotate(
    "text",
    x = 98,
    y = 115,
    label = "italic(I)(t,a)",
    parse = TRUE,
    family = BASE_FAM,
    size = FONT_PT[2] / (72.27 / 25.4),
    hjust = 1,
    vjust = 1
  ) +
  scale_y_continuous(
    name   = "Age (yr)",
    expand = c(0, 0),
    limits = range(df_mid$age),
    breaks = seq(0, 120, 20)
  ) +
  scale_x_continuous(
    name   = NULL,
    expand = c(0, 0),
    limits = c(0, 100),
    breaks = seq(0, 100, 20)
  ) +
  scale_fill_gradient2(
    low      = col_low_w,
    mid      = col_mid,
    high     = col_high,
    midpoint = (I_min + I_max) / 2,
    limits   = c(I_min, I_max),
    name     = NULL,
    labels   = sci_fmt,
    guide    = guide_colorbar(
      title          = NULL,
      barheight      = unit(2.0, "cm"),
      barwidth       = unit(0.25, "cm"),
      ticks          = TRUE,
      label.position = "right"
    )
  ) +
  theme_pub +
  remove_x_all +
  theme(
    legend.position      = c(0.02, 0.97),
    legend.justification = c(0, 1),
    legend.background    = element_rect(fill = "white", colour = "grey80"),
  )

# bottom panel: mortality over time
p_bottom <- ggplot(df_ts, aes(x = t_rel, y = M_inc)) +
  geom_line(linewidth = 0.4, colour = col_high) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 10),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    breaks = c(0, 1e4, 2e4),
    labels = sci_fmt
  ) +
  labs(
    x = "Time (yr)",
    y = expression(plain(Mort) ~ plain("(") * yr^{-1} * plain(")"))
  ) +
  theme_pub +
  theme(
    legend.position = "none",
    plot.margin     = margin(t = 6, r = 10, b = 2, l = 6)
  )

figure_4 <- cowplot::plot_grid(
  p_top,
  p_mid,
  p_bottom,
  ncol             = 1,
  labels           = c("A", "B", "C"),
  label_fontface   = "bold",
  label_fontfamily = BASE_FAM,
  label_size       = FONT_PT[4],
  label_x          = 0.015,
  label_y          = 0.985,
  align            = "v",
  axis             = "l",
  rel_heights      = c(1, 1, 1)
)

figure_4

# Same 2-point outer white border as Figure 2
figure_export <- ggplotGrob(figure_4)
figure_export <- gtable::gtable_add_rows(figure_export, unit(2, "pt"), pos = 0)
figure_export <- gtable::gtable_add_rows(figure_export, unit(2, "pt"), pos = -1)
figure_export <- gtable::gtable_add_cols(figure_export, unit(2, "pt"), pos = 0)
figure_export <- gtable::gtable_add_cols(figure_export, unit(2, "pt"), pos = -1)

dir.create("figures", showWarnings = FALSE, recursive = TRUE)

save_plos(
  plot     = figure_export,
  filename = "figures/figure4"
)
