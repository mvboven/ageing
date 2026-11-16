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

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(cowplot)
library(grid)
library(latex2exp)
library(showtext)
library(sysfonts)

# sizes
FIG_W_CM   <- 8.7        # PNAS single-column width
FIG_ASPECT <- 1.2        # 2×2 panels + legend
BASE_PT    <- 9*2          # base font size (pt)
BASE_FAM   <- "CMUSerif" # main serif font

cm_to_in <- function(x) x / 2.54

# STIX for ℛ₀ axis title
sysfonts::font_add_google("STIX Two Text", "STIX2")
showtext::showtext_auto(TRUE)

# global theme (mostly overridden by panel themes, but keeps things consistent)
theme_pub <- theme_minimal(base_size = BASE_PT, base_family = BASE_FAM) +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.25),
    axis.ticks       = element_line(linewidth = 0.25),
    axis.title       = element_text(margin = margin(t = 2, r = 2, b = 2, l = 2)),
    plot.title       = element_text(face = "bold"),
    plot.margin      = margin(4, 4, 4, 4),
    aspect.ratio     = 1
  )
theme_set(theme_pub)

# geom defaults
update_geom_defaults("line",    list(linewidth = 0.4))
update_geom_defaults("segment", list(linewidth = 0.4))
update_geom_defaults("point",   list(size = 1.4))

# pdf saver (PNAS style, cairo_pdf, 300 dpi)
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

#########################
# Data for 2025 and 2050
#########################

# 2025 sweep
df_mort_2025 <- read_csv("./output/mortality_2025_nuhalf_20261002.csv") %>%
  rename(vac_start = vac_age,
         mortality_abs = total_mortality) %>%
  dplyr::filter(beta <= 500)

df_yll_2025  <- read_csv("./output/yll_2025_nuhalf_20261002.csv") %>%
  rename(vac_start = vac_age,
         YLL_abs = total_yll) %>%
  dplyr::filter(beta <= 500)

# 2050 sweep
df_mort_2050 <- read_csv("./output/mortality_2050_nuhalf_20261001.csv") %>%
  rename(vac_start = vac_age,
         mortality_abs = total_mortality) %>%
  dplyr::filter(beta <= 500)

df_yll_2050  <- read_csv("./output/yll_2050_nuhalf_20261001.csv") %>%
  rename(vac_start = vac_age,
         YLL_abs = total_yll) %>%
  dplyr::filter(beta <= 500)
# Common factor levels (R0 and immune period) across all four datasets
r0_vals <- sort(unique(round(
  c(df_mort_2025$beta, df_yll_2025$beta,
    df_mort_2050$beta, df_yll_2050$beta) / 50,
  2
)))

immune_vals <- sort(unique(round(
  c(1 / df_mort_2025$alpha, 1 / df_yll_2025$alpha,
    1 / df_mort_2050$alpha, 1 / df_yll_2050$alpha),
  2
)))

###############################
# Optimal vaccination per grid
###############################

# helper: compute optimal vac_start per (beta, alpha)
make_opt_df <- function(df, value_col) {
  df %>%
    group_by(beta, alpha) %>%
    slice_min(order_by = .data[[value_col]], n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
      beta   = as.numeric(beta),
      alpha  = as.numeric(alpha),
      R0     = factor(round(beta / 50, 2), levels = r0_vals),
      immune = factor(round(1 / alpha, 2),  levels = immune_vals) 
    )
}

df_mort_2025_opt <- make_opt_df(df_mort_2025, "mortality_abs")
df_yll_2025_opt  <- make_opt_df(df_yll_2025,  "YLL_abs")
df_mort_2050_opt <- make_opt_df(df_mort_2050, "mortality_abs")
df_yll_2050_opt  <- make_opt_df(df_yll_2050,  "YLL_abs")

# shared fill range across all four panels
fill_range <- range(
  c(df_mort_2025_opt$vac_start,
    df_yll_2025_opt$vac_start,
    df_mort_2050_opt$vac_start,
    df_yll_2050_opt$vac_start),
  na.rm = TRUE
)

######################
# Scales and panel fn
######################

x_lab_R0   <- "\u211B\u2080"         # ℛ₀ (calligraphic) as in figs 2 & 5
y_lab_text <- "Immune period (yr)"

fill_scale <- scale_fill_gradientn(
  colors = c("#A6CEE3", "#FDBF6F"),
  limits = fill_range,
  name   = latex2exp::TeX("\\textbf{Optimal vaccination age (yr)}"),
  guide  = guide_colorbar(
    title.position = "top", title.hjust = 0.5,
    barwidth       = unit(120, "pt"),
    barheight      = unit(5,  "pt"),
    ticks          = FALSE,
    label.position = "bottom"
  )
)

# Only show tick labels at these R0 values
r0_ticks_show <- c("1.5", "2", "3", "4", "5", "10")

panel_plot_sq <- function(df_sub, title_text) {
  ggplot(df_sub, aes(R0, immune, fill = vac_start)) +
    geom_tile(color = "white", linewidth = 0.25) +
    geom_text(
      aes(label = sprintf("%d", round(vac_start))),
      size  = 1.4*2,          # small numbers in cells
      color = "grey30"
    ) +
    fill_scale +
    scale_x_discrete(
      breaks = r0_ticks_show,   # only label main R0 values
      labels = r0_ticks_show,
      drop   = FALSE
    ) +
    labs(
      x     = x_lab_R0,
      y     = y_lab_text,
      title = title_text
    ) +
    theme_bw(base_size = BASE_PT, base_family = BASE_FAM) +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3),
      axis.text.x      = element_text(
        angle = 45, hjust = 1,
        size  = BASE_PT - 1       # slightly smaller tick labels
      ),
      axis.title.y     = element_text(
        size   = BASE_PT,
        margin = margin(t = 2, r = 4, b = 2, l = 2)  # <-- increase r
      ),
      legend.position  = "none",
      plot.title       = element_text(
        hjust  = 0.5,
        face   = "bold",
        size   = BASE_PT + 1,
        margin = margin(b = 2)
      ),
      plot.margin      = margin(4, 6, 4, 4),
      axis.title.x     = element_text(family = "STIX2", size = BASE_PT),
      aspect.ratio     = 1
    )
}

################################
# Build the four panels (A–D)
################################

# A: 2025 – mortality  (top-left)
pA <- panel_plot_sq(df_mort_2025_opt,
                    "Mortality in 2025")

# B: 2025 – YLL       (top-right)
pB <- panel_plot_sq(df_yll_2025_opt,
                    "YLL in 2025")

# C: 2050 – mortality (bottom-left)
pC <- panel_plot_sq(df_mort_2050_opt,
                    "Mortality in 2050")

# D: 2050 – YLL       (bottom-right)
pD <- panel_plot_sq(df_yll_2050_opt,
                    "YLL in 2050")

# Remove redundant axis titles: top row no x-title, right column no y-title
remove_x <- function(p) p + theme(axis.title.x = element_blank())
remove_y <- function(p) p + theme(axis.title.y = element_blank())

pA <- remove_x(pA)
pB <- remove_x(remove_y(pB))
pD <- remove_y(pD)
# pC keeps both x and y titles

##########################
# Shared legend (bottom)
##########################

get_legend <- function(plot) {
  g <- ggplotGrob(plot)
  g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
}

legend <- get_legend(
  ggplot(df_mort_2025_opt, aes(R0, immune, fill = vac_start)) +
    geom_tile() +
    fill_scale +
    theme_minimal(base_size = BASE_PT) +
    theme(
      legend.position   = "bottom",
      legend.title      = element_text(margin = margin(b = 3)),
      legend.key.height = unit(5, "pt"),
      legend.key.width  = unit(16, "pt"),
      legend.margin     = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )
)

######################
# Assemble Figure 3
######################

figure_3_panels <- cowplot::plot_grid(
  pA, pB, pC, pD,
  ncol = 2, nrow = 2,
  labels = c("A", "B", "C", "D"),
  label_fontface = "bold",
  label_size     = BASE_PT + 1,
  label_x        = 0.02,
  label_y        = 0.98,
  align          = "hv"
)

figure_3 <- cowplot::plot_grid(
  figure_3_panels, legend,
  ncol = 1,
  rel_heights = c(1, 0.14)   # legend close but not cramped
)

figure_3  # view in RStudio

# save as PNAS-style single-column 4-panel figure
save_pdf(
  plot     = figure_3,
  file_pdf = "figures/figure3.pdf",
  width_cm = 20,
  aspect   = FIG_ASPECT
)
