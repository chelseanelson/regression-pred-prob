# Regression Prediction Problem - Round 1
# Define and fit null model 

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
load(here("recipes/round_1/baseline_rec.rda"))
load(here("results/round_1/fitted_tuned_models/metric_sets.rda"))

# model specifications ----
null_spec <- 
  null_model(mode = "regression") %>%
  set_engine("parsnip")

# define workflows ---
null_wflow <- 
  workflow() %>%
  add_model(null_spec) %>%
  add_recipe(baseline_rec)

# fit workflows/models
fit_null <- null_wflow %>%
  fit_resamples(
    resamples = airbnb_folds,
    control = control_resamples(save_workflow = TRUE),
    metrics = metric_sets
  )

# write out results (fitted/trained workflows) ----
save(fit_null, file = here("results/round_1/fitted_tuned_models/fit_null.rda"))