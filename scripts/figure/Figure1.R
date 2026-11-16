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
packages <- c("ggplot2", "dplyr", "tidyr", "readr", "cowplot", "grid", "readxl")

# Install only missing packages (grid comes with R).
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0L) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Load all packages.
lapply(packages, library, character.only = TRUE)

source("functions/format_PLOS.R")

##########################################
# Data: AG2022 mortality (qx male/female)
##########################################

qx_female_wide <- read_excel("./data/AG2022DefinitiefGevalideerd.xlsx", sheet = "qx vrouwen 2022")
qx_male_wide   <- read_excel("./data/AG2022DefinitiefGevalideerd.xlsx", sheet = "qx mannen 2022")

qx_female_long <- qx_female_wide %>%
  pivot_longer(!x, names_to = "year", values_to = "rate") %>%
  rename(age = x) %>%
  mutate(across(where(is.character), as.numeric))

qx_male_long <- qx_male_wide %>%
  pivot_longer(!x, names_to = "year", values_to = "rate") %>%
  rename(age = x) %>%
  mutate(across(where(is.character), as.numeric))

# overall mortality (simple mean male/female)
qx_overall <- (qx_male_long + qx_female_long) / 2

# log-mortality and year offset
qx_overall <- qx_overall %>%
  mutate(
    log_rate    = log10(rate),
    year_offset = year - 2020
  )

# fit log10(mu) ~ age + (year - 2020) for ages >= 15
qx_overall_filtered <- qx_overall %>%
  filter(age >= 15)

linear_model <- lm(log_rate ~ age + year_offset, data = qx_overall_filtered)

# coefficients
a0 <- coef(linear_model)["(Intercept)"]
a1 <- coef(linear_model)["year_offset"]
a2 <- coef(linear_model)["age"]

##########################################
# Predicted mortality and survival
##########################################

ages  <- 0:120
years <- 2020:2100

combined_data <- tidyr::expand_grid(age = ages, year = years) %>%
  arrange(year, age) %>%
  group_by(year) %>%
  mutate(
    log_predicted_rate = a0 + a1 * (year - 2020) + a2 * age,
    mortality_rate     = 10^log_predicted_rate,
    dx                 = c(0, diff(age)),
    cumulative_mort    = cumsum(mortality_rate * dx),
    survival_prob      = exp(-cumulative_mort)
  ) %>%
  ungroup()

##########################################
# Figure 1: panels + legend
##########################################

# shared colour scale
color_scale <- scale_colour_gradientn(
  colors = c("#E69F00", "#56B4E9"),
  breaks = c(2020, 2060, 2100),
  labels = c("2020", "2060", "2100"),
  name   = "Year",
  guide = guide_colorbar(
    title.position = "top",
    title.hjust    = 0.5,
    direction      = "vertical",
    barwidth       = unit(10, "pt"),
    barheight      = unit(100, "pt"),
    ticks          = FALSE,
    label.position = "right"
  )
)

ylab_mort_expr <- expression(log[10](mu(t, a)) ~ yr^{-1})
ylab_surv_expr <- expression(Pr(alive))

# mortality panel
mortality_plot <- ggplot(qx_overall %>% filter(year <= 2100)) +
  geom_line(aes(age, log_rate, group = year, colour = year), alpha = 0.18) +
  geom_line(
    data = combined_data,
    aes(age, log_predicted_rate, group = year, colour = year),
    linewidth = 0.3, alpha = 0.9
  ) +
  color_scale +
  scale_x_continuous(
    "Age (years)",
    limits = c(0, 120), expand = c(0, 0),
    breaks = c(0, 25, 50, 75, 100, 120)
  ) +
  scale_y_continuous(
    limits = c(-5, 0), expand = c(0, 0),
    breaks = -5:0
  ) +
  labs(y = ylab_mort_expr, title = "Mortality") +
  theme_pub +
  theme(
    aspect.ratio = 1,
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.3),
    legend.position  = "none",
    plot.margin      = margin(8, 8, 6, 8)
  )

# survival panel
survival_plot <- ggplot(combined_data, aes(age, survival_prob, group = year, colour = year)) +
  geom_line(alpha = 0.18) +
  color_scale +
  scale_x_continuous(
    "Age (years)",
    limits = c(0, 120), expand = c(0, 0),
    breaks = c(0, 25, 50, 75, 100, 120)
  ) +
  scale_y_continuous(
    limits = c(0, 1), expand = c(0, 0),
    breaks = seq(0, 1, by = 0.2)
  ) +
  labs(y = ylab_surv_expr, title = "Survival") +
  theme_pub +
  theme(
    aspect.ratio = 1,
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.3),
    legend.position  = "right",
    legend.title     = element_text(size = FONT_PT[5]),
    legend.text      = element_text(size = FONT_PT[4]),
    plot.margin      = margin(8, 8, 6, 8)
  )

# Panel C: infection/case fatality rate 
# IFR COVID, Levin et al., 2020 https://pmc.ncbi.nlm.nih.gov/articles/PMC7721859/#_ad93_
ifr_data_covid_loglinear <- data.frame(age = 0:120, pathogen_and_fatality_type = "SARS-CoV-2, IFR") %>%
  mutate(
    log10_ifr = (-3.27 + 0.0524 * age)-log10(100) #convert from % to proportion
  ) 

# IFR COVID, O'Driscoll et al. 2020 https://www.nature.com/articles/s41586-020-2918-0
ifr_data_covid <- data.frame(
  age_group_min = seq(0, 80, by = 5),
  ifr_percent = c(
    0.003, 0.001, 0.001, 0.003, 0.006,
    0.013, 0.024, 0.040, 0.075, 0.121,
    0.207, 0.323, 0.456, 1.075, 1.674,
    3.203, 8.292
  ),
  ifr_lower_percent = c(
    0.002, 0.000, 0.001, 0.002, 0.005,
    0.011, 0.021, 0.034, 0.064, 0.104,
    0.177, 0.277, 0.392, 0.921, 1.435,
    2.744, 7.105
  ),
  ifr_upper_percent = c(
    0.003, 0.001, 0.001, 0.003, 0.008,
    0.015, 0.028, 0.047, 0.087, 0.140,
    0.239, 0.373, 0.527, 1.244, 1.937,
    3.705, 9.593
  )
) %>%
  mutate(
    age_group_max = lead(age_group_min, default = Inf),
    representative_age = if_else(
      is.infinite(age_group_max),
      82.5,
      (age_group_min + age_group_max) / 2
    ),
    ifr = ifr_percent / 100,
    log10_ifr = log10(ifr)
  )

# CFR Influenza, McDonald et al., 2023 https://onlinelibrary.wiley.com/doi/10.1111/irv.13146
influenza_age_data <- data.frame(
  age_group_min = c(
    0, 1, 5, 10, 15, 20, 25, 30, 35, 40,
    45, 50, 55, 60, 65, 70, 75, 80, 85
  ),
    age_group_max = c(
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, Inf
  ),
  symptomatic_cases = c(
    104300, 431200, 392300, 417500, 250000,
    261200, 262700, 251300, 249700, 277200,
    310200, 306500, 282100, 256400, 203000,
    152300, 111900, 79540, 70680
  ),
  influenza_deaths = c(
    9, 8, 4, 4, 4, 5, 5, 5, 11, 17,
    23, 23, 81, 100, 220, 418, 546, 1105, 2999
  )
)

cfr_data_influenza <- influenza_age_data %>%
  arrange(age_group_min) %>%
  mutate(
    age_group_max = lead(age_group_min, default = Inf),
    representative_age = if_else(
      is.infinite(age_group_max),
      87.5,
      (age_group_min + age_group_max) / 2
    ),
    cfr = influenza_deaths / symptomatic_cases,
    log_cfr = log10(cfr)
  )

grDevices::palette.colors(palette = "Okabe-Ito")

pathogen_fatality_plot <- ggplot() +
  geom_line(data = ifr_data_covid_loglinear, aes(x = age, y=log10_ifr, color = "SARS-CoV-2 (Levin), IFR"), linewidth = 1, alpha = 0.9) +
    geom_point(data = ifr_data_covid, aes(x=representative_age, y=log10_ifr, color = "SARS-CoV-2 (O'Driscoll), IFR"), size = 1, alpha = 0.9) +
  geom_line(data = ifr_data_covid, aes(x=representative_age, y=log10_ifr, color = "SARS-CoV-2 (O'Driscoll), IFR"), linewidth = .5, alpha = 0.9) +
  geom_point(data = cfr_data_influenza, aes(x=representative_age, y=log_cfr, color = "Influenza, CFR"), size = 1, alpha = 0.9) +
  geom_line(data = cfr_data_influenza, aes(x=representative_age, y=log_cfr, color = "Influenza, CFR"), linewidth = .5, alpha = 0.9) +
  scale_x_continuous(
    "Age (years)",
    limits = c(0, 120), expand = c(0, 0),
    breaks = c(0, 25, 50, 75, 100, 120)
  ) +
  scale_y_continuous(
    expression(log[10](FR)),
    limits = c(-5.5, 0), expand = c(0, 0),
    breaks = seq(-5.5, 0, by = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "SARS-CoV-2 (Levin), IFR" = "#0072B2",
      "SARS-CoV-2 (O'Driscoll), IFR" = "#56B4E9",
      "Influenza, CFR" = "#009E73"
    ),
    name   = "Pathogen",
    guide  = guide_legend(
      title.position = "top",
      title.hjust    = 0.5,
      ncol = 1,
      title.theme    = element_text(size = FONT_PT[2]),
      label.position = "right",
      label.theme    = element_text(size = FONT_PT[1])
    )
  ) +
  labs(title = "Fatality rate") +
  theme_pub +
  theme(
    aspect.ratio = 1,
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.3),
    legend.position = "right",
    legend.justification = "center",
    plot.margin      = margin(8, 8, 6, 8)
  )


# Top row: A has no legend; B carries the year legend for both panels.
plots_row <- cowplot::plot_grid(
  mortality_plot,
  survival_plot,
  ncol = 2,
  rel_widths = c(1, 1.25),
  align = "h",
  axis = "tb",
  labels = c("A", "B"),
  label_size = FONT_PT[4],
  label_fontfamily = BASE_FAM,
  label_fontface = "bold",
  label_x = 0,
  label_y = 0.98
)

# Bottom row: C retains its own legend on the right.
panel_c <- cowplot::plot_grid(
  pathogen_fatality_plot,
  labels = "C",
  label_size = FONT_PT[4],
  label_fontfamily = BASE_FAM,
  label_fontface = "bold",
  label_x = 0,
  label_y = 0.98
)

bottom_row <- cowplot::plot_grid(
  panel_c,
  NULL,
  ncol = 2,
  rel_widths = c(3, 1)
)

figure_1 <- cowplot::plot_grid(
  plots_row,
  bottom_row,
  ncol = 1,
  rel_heights = c(1, 1)
)

# save_plos() adds the outer whitespace once around the whole figure.
save_plos(
  plot = figure_1,
  filename = "figures/figure1",
  aspect = 0.95
)
