# Regression Prediction Problem - Round 2
# Define, fit and tune baseline Boosted Trees model 
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data/model_data/round_2/airbnb_folds.rda"))

# load pre-processing/feature engineering recipe 
load(here("recipes/round_2/baseline_nonpara_rec.rda"))
load(here("results/round_2/fitted_tuned_models/metric_sets.rda"))

# model specifications ----
bt_model <-
  boost_tree(
    mode = "regression",
    mtry = tune(),
    min_n = tune(),
    learn_rate = tune(),
    trees = tune()
  ) %>%
  set_engine("xgboost")

# define workflows ---
bt_wflow <-
  workflow() %>%
  add_model(bt_model) %>%
  add_recipe(baseline_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(bt_model)

# change hyperparameter ranges 
bt_params <- extract_parameter_set_dials(bt_model) %>%
  update(
    mtry = mtry(c(1,11)),
    learn_rate = learn_rate(range = c(-5, -0.2)),
    trees = trees(range = c(500, 1000))
  )

# build tuning grid 
bt_grid <- grid_regular(bt_params, 
                        levels = c(mtry = 5, learn_rate = 8, min_n = 4, trees = 6))
# tune workflows/models ----
# set seed 
set.seed(112234)

tuned_bt_1 <-
  bt_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = bt_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_bt_1, file = here("results/round_2/fitted_tuned_models/tuned_bt_1.rda")) 