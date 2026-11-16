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
  "grid", "latex2exp", "showtext", "sysfonts"
)

# Install only missing packages (grid comes with R).
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0L) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Load all packages.
lapply(packages, library, character.only = TRUE)

source("functions/format_PLOS.R")

##########################################
# Figure 6. Influenza mortality heatmaps #
##########################################

# mortality results for influenza A
df_mort <- read_csv("output/mortality_influenza_2025_v50_f005_coverage30_20261020.csv") %>%
  rename(vac_start = first_age, mortality_abs = total_mortality) %>%
  filter(vac_start >= 40)

# prep function (relative mortality)
prep_for_plot_nu <- function(df, value_col) {
  keep_imm <- c(5, 2, 1, 0.5)   # vaccine-induced immunity (years)

  df %>%
    mutate(
      R0            = as.numeric(R0),
      vac_start_num = as.numeric(vac_start),
      imm_years     = as.numeric(nu_duration_years),
      is_no_vacc    = vac_start_num >= 130
    ) %>%
    filter(imm_years %in% keep_imm) %>%
    group_by(imm_years, R0) %>%
    mutate(
      baseline = if (any(is_no_vacc)) .data[[value_col]][which(is_no_vacc)[1]] else NA_real_
    ) %>%
    ungroup() %>%
    filter(!is_no_vacc, dplyr::between(vac_start_num, 0, 100)) %>%
    tidyr::drop_na(baseline) %>%
    mutate(
      rel       = .data[[value_col]] / baseline,
      R0_factor = factor(sprintf("%g", R0),
                         levels = sprintf("%g", sort(unique(R0)))),
      imm_label = factor(paste0("Immunity: ", imm_years, " yr"),
                         levels = paste0("Immunity: ", rev(keep_imm), " yr"))
    )
}

mortality_df <- prep_for_plot_nu(df_mort, "mortality_abs")

# Shared colour scale across all four panels
rel_limits <- range(mortality_df$rel, na.rm = TRUE)

rel_breaks <- sort(unique(c(
  rel_limits,
  if (rel_limits[1] < 1 && rel_limits[2] > 1) 1
)))

rel_labels <- formatC(
  rel_breaks,
  format = "f",
  digits = 2,
  drop0trailing = TRUE
)

# R0 values retained, as in the original Figure 6 script
r0_keep <- c(1.5, 2, 3, 4, 5)
r0_ticks_show <- sprintf("%g", r0_keep)
x_lab_R0 <- "\u211B\u2080"

panel_plot_factor <- function(df_sub) {
  stopifnot(is.factor(df_sub$R0_factor))
  
  ggplot(df_sub, aes(x = R0_factor, y = vac_start_num, fill = rel)) +
    geom_raster(interpolate = TRUE) +
    scale_fill_gradientn(
      colours = c("black","#08306B", "#56B4E9", "white", "#E69F00"),
      values = scales::rescale(
        c(0, 0.4, 0.8, 1, ceiling(rel_limits[2]*10)/10),
        from = c(0, ceiling(rel_limits[2]*10)/10)
      ),
      limits = c(0, ceiling(rel_limits[2]*10)/10),
      breaks = sort(unique(c(0, 0.8, 1, ceiling(rel_limits[2]*10)/10))),
      labels = function(x) {
        formatC(x, format = "f", digits = 2, drop0trailing = TRUE)
      },
      name = "Relative\nmortality"
    ) +
    scale_x_discrete(
      breaks = r0_ticks_show,
      labels = r0_ticks_show,
      limits = r0_ticks_show,
      expand = c(0, 0),
      drop   = FALSE
    ) +
    scale_y_continuous(
      breaks = seq(40, 100, 10),
      expand = c(0, 0),
      name   = "First vaccination (yr)"
    ) +
    labs(
      x     = x_lab_R0,
      title = as.character(unique(df_sub$imm_label)[1])
    ) +
    theme_pub +
    theme(aspect.ratio = 1) +
    guides(fill = guide_colorbar(
      title.position = "top",
      title.hjust    = 0.5,
      ticks          = TRUE,
      barwidth       = unit(0.35, "cm"),
      barheight      = unit(5, "cm")
    )) +
    coord_cartesian(ylim = c(40, 100), clip = "on")
}

# Build panels: 0.5, 1, 2, and 5 years of vaccine-induced immunity
get_panel <- function(years) {
  panel_plot_factor(
    dplyr::filter(mortality_df, dplyr::near(imm_years, years))
  )
}

pA <- get_panel(0.5)
pB <- get_panel(1.0)
pC <- get_panel(2.0)
pD <- get_panel(5.0)

# Extract the shared legend before hiding panel legends
shared_legend <- cowplot::get_legend(pA)

# Hide labels and titles, retaining tick marks
hide_x_labels <- function(p) {
  p + theme(
    axis.text.x  = element_blank(),
    axis.title.x = element_blank()
  )
}

hide_y_labels <- function(p) {
  p + theme(
    axis.text.y  = element_blank(),
    axis.title.y = element_blank()
  )
}

pA <- hide_x_labels(pA)
pB <- hide_x_labels(hide_y_labels(pB))
pD <- hide_y_labels(pD)

# Tight inner margins: top, right, bottom, left (points)
pA <- pA + theme(
  legend.position = "none",
  plot.margin = margin(2, 0, 0, 2)
)

pB <- pB + theme(
  legend.position = "none",
  plot.margin = margin(2, 2, 0, 0)
)

pC <- pC + theme(
  legend.position = "none",
  plot.margin = margin(0, 0, 2, 2)
)

pD <- pD + theme(
  legend.position = "none",
  plot.margin = margin(0, 2, 2, 0)
)

# Assemble panels
panels <- cowplot::plot_grid(
  pA, pB, pC, pD,
  ncol = 2,
  nrow = 2,
  labels = c("A", "B", "C", "D"),
  label_fontface = "bold",
  label_fontfamily = BASE_FAM,
  label_size = FONT_PT[4],
  label_x = 0,
  label_y = 0.98,
  align = "hv",
  axis = "tblr",
  rel_heights = c(0.88, 1)
)

# Shared legend centered vertically on the right
figure_6 <- cowplot::plot_grid(
  panels,
  shared_legend,
  ncol = 2,
  rel_widths = c(1, 0.18)
)

figure_6

# Recommended 2-point outer white border, without changing text sizes.
figure_export <- ggplotGrob(figure_6)
figure_export <- gtable::gtable_add_rows(figure_export, unit(2, "pt"), pos = 0)
figure_export <- gtable::gtable_add_rows(figure_export, unit(2, "pt"), pos = -1)
figure_export <- gtable::gtable_add_cols(figure_export, unit(2, "pt"), pos = 0)
figure_export <- gtable::gtable_add_cols(figure_export, unit(2, "pt"), pos = -1)

dir.create("figures", showWarnings = FALSE, recursive = TRUE)

save_plos(
  plot     = figure_export,
  filename = "figures/figure6"
)