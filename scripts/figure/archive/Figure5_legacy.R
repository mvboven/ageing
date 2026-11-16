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
library(tidyr)
library(readr)
library(cowplot)
library(grid)
library(sysfonts)
library(showtext)

# fonts (for calligraphic ℛ)
sysfonts::font_add_google("STIX Two Text", "STIX2")
showtext::showtext_auto(TRUE)

# sizes
FIG_W_CM   <- 8.7        # PNAS single-column width
FIG_ASPECT <- 1.2        # 2×2 panels + legend
BASE_PT    <- 9*2          # base font size (pt)
BASE_FAM   <- "CMUSerif" # main serif font

cm_to_in <- function(x) x / 2.54

# simple scientific format (if needed elsewhere)
sci_fmt <- function(x) {
  fmt <- format(x, scientific = TRUE)
  gsub("e\\+0?", "e", fmt)
}

# global theme
theme_pub <- theme_minimal(base_size = BASE_PT, base_family = BASE_FAM) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.25),
    axis.ticks       = element_line(linewidth = 0.25),
    axis.title       = element_text(margin = margin(t = 2, r = 2, b = 2, l = 2)),
    plot.title       = element_text(face = "bold"),
    plot.margin      = margin(4, 4, 4, 4),
    aspect.ratio     = FIG_ASPECT
  )
theme_set(theme_pub)

# geom defaults
update_geom_defaults("line",    list(linewidth = 0.4))
update_geom_defaults("segment", list(linewidth = 0.4))
update_geom_defaults("point",   list(size = 1.4))

# pdf saver (PNAS style)
save_pdf <- function(plot, file_pdf,
                     width_cm = FIG_W_CM,
                     aspect   = FIG_ASPECT,
                     dpi      = 300) {
  w_in <- cm_to_in(width_cm)
  h_in <- w_in * aspect
  ggplot2::ggsave(
    filename  = file_pdf,
    plot      = plot,
    width     = w_in,
    height    = h_in,
    units     = "in",
    dpi       = dpi,
    device    = cairo_pdf,
    limitsize = FALSE
  )
  message("Wrote: ", file_pdf)
}

##########################################
# Figure 5. Influenza mortality heatmaps #
##########################################

# mortality results for influenza A
df_mort <- read_csv("output/mortality_influenza_2025_v50_f005_coverage80_20261020.csv") %>%
#df_mort <- read_csv("./data/mortality_influenza_2025_f005_20251215.csv") %>%
  rename(vac_start = first_age, mortality_abs = total_mortality) %>% #mean_age_infection
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

# single-panel function
panel_plot_factor <- function(df_sub,
                              inset_xy = c(0.96, 0.965),
                              bar_w_cm = 0.24,
                              bar_h_cm = 1.10) {
  stopifnot(is.factor(df_sub$R0_factor))
  
  r0_keep   <- c(1.5, 2, 3, 4, 5)
  r0_breaks <- sprintf("%g", r0_keep)
  
  rel_min <- min(df_sub$rel, na.rm = TRUE)
  rel_max <- max(df_sub$rel, na.rm = TRUE)
  brk_num <- unique(c(rel_min, if (rel_min < 1 && rel_max > 1) 1, rel_max))
  fmt     <- function(x) formatC(x, format = "f", digits = 2, drop0trailing = TRUE)
  
  ggplot(df_sub, aes(x = R0_factor, y = vac_start_num, fill = rel)) +
    geom_raster(interpolate = TRUE) +
    scale_fill_gradient2(
      low = "#56B4E9", mid = "white", high = "red", #  "#E69F00"
      midpoint = 1, limits = c(rel_min, rel_max),
      breaks = brk_num, labels = fmt(brk_num), name = NULL
    ) +
    scale_x_discrete(
      breaks = r0_breaks,
      labels = r0_breaks,
      limits = r0_breaks,
      expand = c(0, 0),
      drop   = FALSE
    ) +
    scale_y_continuous(
      breaks = seq(40, 100, 10),
      limits = c(40, 100),
      expand = c(0, 0),
      name   = "First vaccination (yr)"
    ) +
    labs(
      x     = "\u211B\u2080",  # calligraphic R0 via STIX2
      title = as.character(unique(df_sub$imm_label)[1])
    ) +
    theme_bw(base_size = BASE_PT, base_family = BASE_FAM) +
    theme(
      axis.text        = element_text(size = BASE_PT - 1),
      axis.title.x     = element_text(
        size   = BASE_PT,
        margin = margin(t = 4),
        family = "STIX2"         
      ),
      axis.title.y     = element_text(
        size   = BASE_PT,
        margin = margin(t = 2, r = 4, b = 2, l = 2)  
      ),
      plot.title       = element_text(face = "bold", hjust = 0, size = BASE_PT + 1,
                                      margin = margin(t = 3, r = 4, b = 2)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3),
      aspect.ratio     = 1,
      legend.position      = inset_xy,
      legend.justification = c(1, 1),
      legend.background    = element_rect(fill = "white", colour = "grey80"),
      legend.text          = element_text(size = BASE_PT - 5),
      plot.margin          = margin(t = 4, r = 10, b = 4, l = 4)
    ) +
    guides(fill = guide_colorbar(
      title = NULL, ticks = TRUE, label = TRUE,
      barwidth  = unit(bar_w_cm, "cm"),
      barheight = unit(bar_h_cm, "cm")
    )) +
    coord_cartesian(clip = "off")
}

# choose immunity durations
imm_levels <- c(0.5, 1.0, 2.0, 5.0)

get_panel <- function(y) {
  panel_plot_factor(dplyr::filter(mortality_df, imm_years == y))
}

pA <- get_panel(0.5)   # top-left
pB <- get_panel(1.0)   # top-right
pC <- get_panel(2.0)   # bottom-left
pD <- get_panel(5.0)   # bottom-right

# remove redundant axis titles
remove_x <- function(p) p + theme(axis.title.x = element_blank())
remove_y <- function(p) p + theme(axis.title.y = element_blank())

pA <- remove_x(pA)
pB <- remove_x(remove_y(pB))
pD <- remove_y(pD)

# assemble 2×2 layout
figure_5 <- cowplot::plot_grid(
  pA, pB, pC, pD,
  ncol = 2, nrow = 2,
  labels = c("A","B","C","D"),
  label_fontface = "bold", label_size = BASE_PT + 1,
  label_x = 0.02, label_y = 0.98, align = "hv"
)

figure_5

# save as PNAS-style single-column square figure
save_pdf(
  plot     = figure_5,
  file_pdf = "figures/figure5b.pdf",
  width_cm = 20,
  aspect   = 1.0
)
