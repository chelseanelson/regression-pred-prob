# Regression Prediction Problem - Round 3
# Train final model
# Best Model: Baseline Random Forest 1
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

# best model: Baseline Random Forest 

# load tuned and training data 
load(here("data/model_data/round_3/airbnb_train.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_rf_1.rda"))

# finalize workflow ----
final_wflow <-
  tuned_rf_1 %>%
  extract_workflow() %>%
  finalize_workflow(select_best(tuned_rf_1, metric = "mae"))

# train final model ----
# set seed
set.seed(393)
final_fit_2 <- fit(final_wflow, airbnb_train)

# write out fitted data 
save(final_fit_2, file = here("results/round_3/final_fit_2.rda"))