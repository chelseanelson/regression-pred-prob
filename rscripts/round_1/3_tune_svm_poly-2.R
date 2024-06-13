# Regression Prediction Problem - Round 1
# Define, fit and tune feature engineered SVM Polynomial model 
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
load(here("recipes/round_1/fe_rec.rda"))
load(here("results/round_1/fitted_tuned_models/metric_sets.rda"))

# model specifications ----
svm_poly_model <-
  svm_poly(
    mode = "regression",
    cost = tune(),
    degree = tune(),
    scale_factor = tune()
  ) %>%
  set_engine("kernlab")

# define workflows ---
svm_poly_wflow <-
  workflow() %>%
  add_model(svm_poly_model) %>%
  add_recipe(fe_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(svm_poly_model)

# change hyperparameter ranges 
svm_poly_params <- extract_parameter_set_dials(svm_poly_model) %>%
  update(
    degree = degree(c(3, 8))
  )

rec_params <- extract_parameter_set_dials(fe_rec) %>%
  update(
    num_comp = num_comp(range = c(10,20))
  )

all_params <- bind_rows(svm_poly_params, rec_params)

# build tuning grid 
svm_poly_grid <- grid_regular(all_params, levels = 6)

# tune workflows/models ----
# set seed 
set.seed(6544)

tuned_svm_poly_2 <-
  svm_poly_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = svm_poly_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_svm_poly_2, file = here("results/round_1/fitted_tuned_models/tuned_svm_poly_2.rda")) 