# Regression Prediction Problem - Round 1
# Define, fit and tune feature engineered Boosted Trees model 
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
load(here("data/model_data/round_1/airbnb_folds.rda"))

# load pre-processing/feature engineering recipe 
load(here("recipes/round_1/fe_tree_rec.rda"))
load(here("results/round_1/fitted_tuned_models/metric_sets.rda"))

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
  add_recipe(fe_tree_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(bt_model)

# change hyperparameter ranges 
bt_params <- extract_parameter_set_dials(bt_model) %>%
  update(
    mtry = mtry(c(1,15)),
    learn_rate = learn_rate(range = c(-4, -0.1)),
    min_n = min_n(c(1,20)),
    trees = tress(1000, 1500)
  )

rec_params <- extract_parameter_set_dials(fe_tree_rec) %>%
  update(
    num_comp = num_comp(range = c(10,20))
  )

all_params <- bind_rows(bt_params, rec_params)

# build tuning grid 
bt_grid <- grid_regular(all_params, levels = 10)
# tune workflows/models ----
# set seed 
set.seed(35681)

tuned_bt_2 <-
  bt_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = bt_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_bt_2, file = here("results/round_1/fitted_tuned_models/tuned_bt_2.rda")) 