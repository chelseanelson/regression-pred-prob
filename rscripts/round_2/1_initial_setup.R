# Regression Prediction Problem - Round 2
# Initial data checks, data splitting, & data folding 
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load in dataset 
airbnb_data <- read_rds(here("data/train_regression.rds"))

# skim the data
airbnb_data %>% skimr::skim_without_charts()
# there is no missing values within our response variable 

# looking at `price` variable 
price_plot <- airbnb_data %>% ggplot(aes(price)) + geom_histogram(color = "white", bins = 75) + 
  labs(
    title = "Distribution of Price",
    x = "Price",
    y = "Count"
  ) + theme_minimal()

airbnb_data %>% skimr::skim_without_charts(price) 
# there seem to be large outlier within the data, thus I will remove it from the data
airbnb_data <- airbnb_data %>% filter(price < 99998)

price_plot_2 <- airbnb_data %>% ggplot(aes(price)) + geom_histogram(color = "white", bins = 75) + 
  labs(
    title = "Distribution of Price",
    x = "Price",
    y = "Count"
  ) + theme_minimal()

## set seed for random split 
set.seed(6173233)

# initial split of the data ----
airbnb_split <- airbnb_data %>% 
  initial_split(prop = 0.75, strata = price)

airbnb_train <- airbnb_split %>% training()
airbnb_test <- airbnb_split %>% testing()

dim(airbnb_train)
dim(airbnb_test)

# folding data (resamples) ----
# set seed 
set.seed(12389)
airbnb_folds <- vfold_cv(airbnb_train, v = 5, repeats = 3, strata = price)

# making metric set 
metric_sets <- metric_set(mae, rmse)

# write out split, train, test and folds, metric set
save(airbnb_split, file = here("data/model_data/round_2/airbnb_split.rda"))
save(airbnb_folds, file = here("data/model_data/round_2/airbnb_folds.rda"))
save(airbnb_test, file = here("data/model_data/round_2/airbnb_test.rda"))
save(airbnb_train, file = here("data/model_data/round_2/airbnb_train.rda"))
save(metric_sets, file = here("results/round_2/fitted_tuned_models/metric_sets.rda"))
