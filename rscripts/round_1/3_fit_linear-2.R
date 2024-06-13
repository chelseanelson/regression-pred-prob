# Regression Prediction Problem - Round 1
# Define and fit feature engineered linear model 

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
linear_model <- 
  linear_reg() %>%
  set_engine("lm")

# define workflows ---
linear_wflow <- 
  workflow() %>%
  add_model(linear_model) %>%
  add_recipe(fe_rec)

# hyperparameters ----
rec_params <- extract_parameter_set_dials(fe_rec) %>%
  update(
    num_comp = num_comp(range = c(1,10))
  )

# build tuning grid 
linear_grid <- grid_regular(rec_params, levels = 10)

# fit workflows/models
tuned_linear_2 <- linear_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = linear_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(tuned_linear_2, file = here("results/round_1/fitted_tuned_models/tuned_linear_2.rda"))