# Regression Prediction Problem - Round 3
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load testing and fitted data 
load(here("results/round_3/final_fit.rda")) # Baseline Boosted Tree Version 1 
load(here("results/round_3/final_fit_2.rda")) # Baseline Random Forest Version 1
load(here("data/model_data/round_3/airbnb_test.rda"))
test_regression <- read_rds(here("data/test_regression.rds"))

# assessing models performance 

# Baseline Boosted Tree Version 1
airbnb_test_res <- bind_cols(airbnb_test, predict(final_fit, airbnb_test)) %>%
  select(id, price, .pred)

airbnb_model_metrics <- mae(airbnb_test_res, price, .pred) %>% rename(metric = .metric, estimate = .estimate) %>% select(-.estimator) 

airbnb_price_plot <- ggplot(airbnb_test_res, aes(x = price, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Price (log10)", x = "Actual Price (log10)", title = "Actual Airbnb Price (log10) vs. Predicted Airbnb Price (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred() + theme_minimal()

# Baseline Random Forest Version 1 
airbnb_test_res_2 <- bind_cols(airbnb_test, predict(final_fit_2, airbnb_test)) %>%
  select(id, price, .pred)

airbnb_model_metrics_2 <- mae(airbnb_test_res_2, price, .pred) %>% rename(metric = .metric, estimate = .estimate) %>% select(-.estimator) 

airbnb_price_plot_2 <- ggplot(airbnb_test_res_2, aes(x = price, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Price (log10)", x = "Actual Price (log10)", title = "Actual Airbnb Price (log10) vs. Predicted Airbnb Price (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred() + theme_minimal()

# assessing models performance (test_regression)
# Baseline Boosted Tree Version 1 
airbnb_submission_3_3 <- bind_cols(test_regression, predict(final_fit, test_regression)) %>%
  select(id, .pred) %>% mutate(
    .pred = 10^(.pred)
  ) %>%
  rename(
    predicted = .pred
  )


# Baseline Random Forest Version 1 
airbnb_submission_3_4 <- bind_cols(test_regression, predict(final_fit_2, test_regression)) %>%
  select(id, .pred) %>% mutate(
    .pred = 10^(.pred)
  ) %>%
  rename(
    predicted = .pred
  )

# save out results (plot, table)
ggsave(filename = here("results/round_3/airbnb_price_plot.png"), airbnb_price_plot)
write_rds(airbnb_model_metrics, file = here("results/round_3/airbnb_model_metrics.rds"))
write_csv(airbnb_submission_3_3, file = here("submissions/airbnb_submission_3_3.csv"))
ggsave(filename = here("results/round_3/airbnb_price_plot_2.png"), airbnb_price_plot_2)
write_rds(airbnb_model_metrics_2, file = here("results/round_3/airbnb_model_metrics_2.rds"))
write_csv(airbnb_submission_3_4, file = here("submissions/airbnb_submission_3_4.csv"))
