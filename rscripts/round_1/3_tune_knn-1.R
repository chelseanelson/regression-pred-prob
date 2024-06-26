# Regression Prediction Problem - Round 1 
# Define, fit and tune baseline k-nearest neighbors model 
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
load(here("recipes/round_1/baseline_nonpara_rec.rda"))
load(here("results/round_1/fitted_tuned_models/metric_sets.rda"))

# model specifications ----
knn_model <-
  nearest_neighbor(
    mode = "regression",
    neighbors = tune()
  ) %>%
  set_engine("kknn")

# define workflows ---
knn_wflow <-
  workflow() %>%
  add_model(knn_model) %>%
  add_recipe(baseline_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(knn_model)

# change hyperparameter ranges 
knn_params <- extract_parameter_set_dials(knn_model) %>%
  update(
    neighbors = neighbors(c(1,20L))
  )

# build tuning grid 
knn_grid <- grid_regular(knn_params, levels = 10)

# tune workflows/models ----
# set seed 
set.seed(3876)

tuned_knn_1 <-
  knn_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = knn_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_knn_1, file = here("results/round_1/fitted_tuned_models/tuned_knn_1.rda")) 