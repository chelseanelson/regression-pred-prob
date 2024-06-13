# Regression Prediction Problem
# Define, fit and tune feature engineered Random Forest model 
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
load(here("recipes/round_1/fe_nonpara_rec.rda"))
load(here("results/round_1/fitted_tuned_models/metric_sets.rda"))

# model specifications ----
rf_model <-
  rand_forest(
    mode = "regression",
    mtry = tune(),
    min_n = tune(),
    trees = 1000
  ) %>%
  set_engine("ranger")

# define workflows ---
rf_wflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(fe_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(rf_model)

# change hyperparameter ranges 
rf_params <- extract_parameter_set_dials(rf_model) %>%
  update(
    mtry = mtry(c(1,15)),
    min_n = min_n(c(1,20))
  )

# build tuning grid 
rf_grid <- grid_regular(rf_params, levels = 5)

# tune workflows/models ----
# set seed 
set.seed(958310)

tuned_rf_2 <-
  rf_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = rf_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_rf_2, file = here("results/round_1/fitted_tuned_models/tuned_rf_2.rda")) 