# Regression Prediction Problem - Round 2
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load testing and fitted data 
load(here("results/round_2/final_fit.rda"))
load(here("data/model_data/round_2/airbnb_test.rda"))
test_regression <- read_rds(here("data/test_regression.rds"))

# assessing models performance 
airbnb_test_res <- bind_cols(airbnb_test, predict(final_fit, airbnb_test)) %>%
  select(id, price, .pred)

airbnb_model_metrics <- mae(airbnb_test_res, price, .pred)

airbnb_price_plot <- ggplot(airbnb_test_res, aes(x = price, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Price", x = "Actual Price", title = "Actual Airbnb Price vs. Predicted Airbnb Price") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()

# assessing models performance (test_regression)
airbnb_submission_2 <- bind_cols(test_regression, predict(final_fit, test_regression)) %>%
  select(id, .pred) %>%
  rename(
    predicted = .pred
  )

# save out results (plot, table)
ggsave(filename = here("results/round_2/airbnb_price_plot.png"), airbnb_price_plot)
write_rds(airbnb_model_metrics, file = here("results/round_2/airbnb_model_metrics.rds"))
write_csv(airbnb_submission_2, file = here("submissions/airbnb_submission_2.csv"))
