# Regression Prediction Problem - Round 1
# Define, fit and tune feature engineered MARS model 
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
  add_recipe(fe_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(mars_model)

# change hyperparameter ranges 
mars_params <- extract_parameter_set_dials(mars_model) %>%
  update(
    num_terms = num_terms(range = c(15,35)),
    prod_degree = prod_degree(range = c(3, 8))
  )

rec_params <- extract_parameter_set_dials(fe_rec) %>%
  update(
    num_comp = num_comp(range = c(10,20))
  )


all_params <- bind_rows(mars_params, rec_params)

# build tuning grid 
mars_grid <- grid_regular(all_params, levels = 26)

# tune workflows/models ----
# set seed 
set.seed(6590)

tuned_mars_2 <-
  mars_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = mars_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_mars_2, file = here("results/round_1/fitted_tuned_models/tuned_mars_2.rda")) 