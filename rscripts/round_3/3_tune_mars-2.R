# Regression Prediction Problem - Round 3 
# Define, fit and tune baseline MARS model 2
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
load(here("data/model_data/round_3/airbnb_folds.rda"))

# load pre-processing/feature engineering recipe 
load(here("recipes/round_3/baseline_rec.rda"))
load(here("results/round_3/fitted_tuned_models/metric_sets.rda"))

# model specifications ----
mars_model <-
  mars(
    mode = "regression",
    num_terms = tune(),
    prod_degree = tune()
  ) %>%
  set_engine("earth")

# define workflows ---
mars_wflow <-
  workflow() %>%
  add_model(mars_model) %>%
  add_recipe(baseline_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(mars_model)

# change hyperparameter ranges 
mars_params <- extract_parameter_set_dials(mars_model) %>%
  update(
    num_terms = num_terms(range = c(40,60)),
    prod_degree = prod_degree(range = c(3, 5))
  )

# build tuning grid 
mars_grid <- grid_regular(mars_params, levels = 21)

# tune workflows/models ----
# set seed 
set.seed(6411)

tuned_mars_2 <-
  mars_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = mars_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_mars_2, file = here("results/round_3/fitted_tuned_models/tuned_mars_2.rda")) 