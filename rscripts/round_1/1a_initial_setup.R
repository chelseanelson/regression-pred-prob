# Regression Prediction Problem - Round 1 
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
# there seem to be large outliers within the data 
# thus a log-10 transformation could help to solve for this

# looking at the outliers more deeply 
airbnb_data %>% select(price) %>% arrange(desc(price))

# adding a new variable - price log10 transformed
# update: have to just change price to be log10 because it works better for how I am conducting my recipes
airbnb_data <- airbnb_data %>%
  mutate(
    price_log10 = log10(price)
  ) %>% 
  select(-price)

# creating a plot of this 
price_log10_plot <- airbnb_data %>% ggplot(aes(price_log10)) + geom_histogram(color = "white", bins = 75) + 
  labs(
    title = "Distribution of Price (Log10 Scale)",
    x = "Price (Log10 Scale)",
    y = "Count"
  ) + theme_minimal()

# this made the plot more normal, providing better scaling 
# saving figures
ggsave(here("figures/figure-1.png"), price_plot)
ggsave(here("figures/figure-2.png"), price_log10_plot)

## set seed for random split 
set.seed(6029)

# initial split of the data ----
airbnb_split <- airbnb_data %>% 
  initial_split(prop = 0.75, strata = price_log10)

airbnb_train <- airbnb_split %>% training()
airbnb_test <- airbnb_split %>% testing()

dim(airbnb_train)
dim(airbnb_test)

# folding data (resamples) ----
# set seed 
set.seed(12389)
airbnb_folds <- vfold_cv(airbnb_train, v = 5, repeats = 3, strata = price_log10)

# making metric set 
metric_sets <- metric_set(mae, rmse)

# write out split, train, test and folds, metric set
save(airbnb_split, file = here("data/model_data/round_1/airbnb_split.rda"))
save(airbnb_folds, file = here("data/model_data/round_1/airbnb_folds.rda"))
save(airbnb_test, file = here("data/model_data/round_1/airbnb_test.rda"))
save(airbnb_train, file = here("data/model_data/round_1/airbnb_train.rda"))
save(metric_sets, file = here("results/round_1/fitted_tuned_models/metric_sets.rda"))
