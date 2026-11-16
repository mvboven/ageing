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

# sizes
FIG_W_CM   <- 8.7        # PNAS single-column width
FIG_ASPECT <- 1.0        # square-ish by default
BASE_PT    <- 9*1.2          # base font size (pt)
BASE_FAM   <- "CMUSerif" # main serif font

cm_to_in <- function(x) x / 2.54

# optional: STIX Two Text for math labels (used for ℛ₀)
sysfonts::font_add_google("STIX Two Text", "STIX2")
showtext::showtext_auto(TRUE)

# global theme (white backgrounds, consistent with other figs)
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
    aspect.ratio     = FIG_ASPECT
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

###########################################
# Figure 2. Relative mortality (no demog.)
###########################################

# sweep results (mortality)
# old: _da01_20251023.csv; new: _nu02_20251119.csv
df_mort <- read_csv("./output/mortality_2025_nuhalf_20261002.csv") %>%
  rename(vac_start = vac_age, mortality_abs = total_mortality)

# helper: prepare mortality data (relative to no-vacc baseline)
prep_for_plot <- function(df, value_col) {
  keep_alphas <- c(0.02, 0.05, 0.1, 0.2)   # waning rates (1 / immune years)
  
  df %>%
    mutate(
      alpha         = as.numeric(alpha),
      beta          = as.numeric(beta),
      vac_start_num = as.numeric(vac_start)
    ) %>%
    filter(alpha %in% keep_alphas) %>%
    mutate(
      R0    = factor(beta / 50, levels = sort(unique(beta / 50))),
      is_no = vac_start_num >= 130      # sentinel for "no vaccination"
    ) %>%
    group_by(alpha, R0) %>%
    mutate(
      baseline = if (any(is_no)) .data[[value_col]][which(is_no)[1]] else NA_real_
    ) %>%
    ungroup() %>%
    # now allow up to 100 yr for the axis
    filter(!is_no, dplyr::between(vac_start_num, 0, 100)) %>%
    tidyr::drop_na(baseline) %>%
    mutate(
      imm_years = 1 / alpha,
      imm_label = factor(
        paste0("Immunity: ", imm_years, " yr"),
        levels = c("Immunity: 5 yr",
                   "Immunity: 10 yr",
                   "Immunity: 20 yr",
                   "Immunity: 50 yr")
      )
    )
}

mort_df <- prep_for_plot(df_mort, "mortality_abs") %>%
  mutate(rel = mortality_abs / baseline)

# R0 ticks to *show* (others may be present for interpolation)
r0_ticks_show <- c("1.5", "2", "3", "4", "5", "10", "15", "20")

# ℛ₀ label (same as in Figure 5)
x_lab_R0 <- "\u211B\u2080"  

# smaller bar, top-right; extra right margin for long titles
panel_plot_factor <- function(df_sub,
                              inset_xy  = c(0.965, 0.965),  # nearer top-right (inside)
                              bar_w_cm  = 0.24,
                              bar_h_cm  = 1.10) {
  stopifnot(is.factor(df_sub$R0))
  
  rel_min <- min(df_sub$rel, na.rm = TRUE)
  rel_max <- max(df_sub$rel, na.rm = TRUE)
  brk_num <- unique(c(rel_min, if (rel_min < 1 && rel_max > 1) 1, rel_max))
  fmt     <- function(x) formatC(x, format = "f", digits = 2, drop0trailing = TRUE)
  
  ggplot(df_sub, aes(R0, vac_start_num, fill = rel)) +
    geom_raster(interpolate = TRUE) +
    scale_fill_gradient2(
      low      = "#56B4E9",
      mid      =  "white", # "#E69F00",
      high     = "red",
      midpoint = 1,
      limits   = c(rel_min, rel_max),
      breaks   = brk_num,
      labels   = fmt(brk_num),
      name     = NULL
    ) +
    scale_x_discrete(
      breaks = r0_ticks_show,
      labels = r0_ticks_show,
      drop   = FALSE
    ) +
    scale_y_continuous(
      breaks = seq(0, 100, 10),
      limits = c(0, 100),
      expand = c(0, 0),
      name   = "Vaccination age (yr)"
    ) +
    labs(
      x     = x_lab_R0,
      title = as.character(unique(df_sub$imm_label)[1])
    ) +
    theme_bw(base_size = BASE_PT, base_family = BASE_FAM) +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      axis.text        = element_text(size = BASE_PT - 1),
      axis.title.x     = element_text(
        size   = BASE_PT,
        margin = margin(t = 4),
        family = "STIX2",
      ),
      axis.title.y     = element_text(
        size   = BASE_PT,
        margin = margin(t = 2, r = 4, b = 2, l = 2) 
      ),
      plot.title       = element_text(
        face   = "bold",
        hjust  = 0,
        size   = BASE_PT + 1,
        margin = margin(t = 4, r = 10, b = 2)
      ),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3),
      aspect.ratio     = 1,
      legend.position      = inset_xy,
      legend.justification = c(1, 1),
      legend.background    = element_rect(fill = "white", colour = "grey80"),
      legend.text          = element_text(size = BASE_PT - 5),
      plot.margin          = margin(t = 4, r = 6, b = 4, l = 4)
    ) +
    guides(fill = guide_colorbar(
      title     = NULL,
      ticks     = TRUE,
      label     = TRUE,
      barwidth  = unit(bar_w_cm, "cm"),
      barheight = unit(bar_h_cm, "cm")
    )) +
    coord_cartesian(clip = "off")
}

# build panels (sorted alpha; right column nudged a bit for legend)
alphas <- sort(unique(mort_df$alpha))

pA <- panel_plot_factor(
  dplyr::filter(mort_df, alpha == alphas[4]),
  inset_xy = c(0.965, 0.965)
)
pB <- panel_plot_factor(
  dplyr::filter(mort_df, alpha == alphas[3]),
  inset_xy = c(0.955, 0.965)
)
pC <- panel_plot_factor(
  dplyr::filter(mort_df, alpha == alphas[2]),
  inset_xy = c(0.965, 0.965)
)
pD <- panel_plot_factor(
  dplyr::filter(mort_df, alpha == alphas[1]),
  inset_xy = c(0.955, 0.965)
)

# remove redundant axis titles for interior panels
remove_x <- function(p) p + theme(axis.title.x = element_blank())
remove_y <- function(p) p + theme(axis.title.y = element_blank())

pA <- remove_x(pA)               # top-left: no x
pB <- remove_x(remove_y(pB))     # top-right: no x, no y
pD <- remove_y(pD)               # bottom-right: no y

# assemble 2×2 layout (tight, PNAS single column)
figure_2 <- cowplot::plot_grid(
  pA, pB, pC, pD,
  ncol  = 2, nrow = 2,
  labels = c("A", "B", "C", "D"),
  label_fontface = "bold",
  label_size     = BASE_PT + 1,
  label_x        = 0.02,
  label_y        = 0.98,
  align          = "hv"
)

figure_2  # for inspection in RStudio

# save as PNAS-style single-column square figure
save_pdf(
  plot     = figure_2,
  file_pdf = "figures/figure2.pdf",
  width_cm = 20,
  aspect   = 1.0
)

