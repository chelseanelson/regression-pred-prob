# Regression Prediction Problem - Round 1 
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load testing and fitted data 
load(here("results/round_1/final_fit.rda"))
load(here("data/model_data/round_1/airbnb_test.rda"))
test_regression <- read_rds(here("data/test_regression.rds"))

# assessing models performance 
airbnb_test_res <- bind_cols(airbnb_test, predict(final_fit, airbnb_test)) %>%
  select(id, price_log10, .pred)

airbnb_model_metrics <- mae(airbnb_test_res, price_log10, .pred)

airbnb_price_plot <- ggplot(airbnb_test_res, aes(x = price_log10, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Price (log10)", x = "Actual Price (log10)", title = "Actual Airbnb Price (log10) vs. Predicted Airbnb Price (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()

# assessing models performance (test_regression)
airbnb_submission_1_2 <- bind_cols(test_regression, predict(final_fit, test_regression)) %>%
  select(id, .pred) %>% mutate(
    .pred = 10^(.pred)
  ) %>%
  rename(
    predicted = .pred
  )

# save out results (plot, table)
ggsave(filename = here("results/round_1/airbnb_price_plot.png"), airbnb_price_plot)
write_rds(airbnb_model_metrics, file = here("results/round_1/airbnb_model_metrics.rds"))
write_csv(airbnb_submission_1_2, file = here("submissions/airbnb_submission_1_2.csv"))
