# Regression Prediction Problem - Round 1
# Define, fit and tune feature engineered K-nearest Neighbors model 
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
  add_recipe(fe_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(knn_model)

# change hyperparameter ranges 
knn_params <- extract_parameter_set_dials(knn_model) %>%
  update(
    neighbors = neighbors(c(13, 23))
  )

rec_params <- extract_parameter_set_dials(fe_nonpara_rec) %>%
  update(
    num_comp = num_comp(range = c(10,20))
  )

all_params <- bind_rows(knn_params, rec_params)

# build tuning grid 
knn_grid <- grid_regular(all_params, levels = 10)

# tune workflows/models ----
# set seed 
set.seed(60193)

tuned_knn_2 <-
  knn_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = knn_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_knn_2, file = here("results/round_1/fitted_tuned_models/tuned_knn_2.rda")) 