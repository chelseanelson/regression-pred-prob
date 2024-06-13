# Regression Prediction Problem - Round 2
# Setup pre-processing/recipes

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts 
tidymodels_prefer()

## load in training data 
load(here("data/model_data/round_2/airbnb_train.rda"))

# build recipes ----

## recipe 1 (baseline) ----

### variation 1 (parametric)
baseline_rec <- recipe(price ~., data = airbnb_train) %>%
  update_role(id, new_role = "id") %>%
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

### variation 2 (non-parametric)
baseline_nonpara_rec <- recipe(price ~., data = airbnb_train) %>%
  update_role(id, new_role = "id") %>%
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

# check recipes 
baseline_rec %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

baseline_nonpara_rec %>%
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

## recipe 2 (feature engineering) ----

### variation 1 (parametric)
fe_rec <- recipe(price ~., data = airbnb_train) %>%
  update_role(id, new_role = "id") %>%
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_knn(host_location, host_response_time, host_response_rate, host_acceptance_rate,
                  host_neighbourhood, bathrooms_text, beds, review_scores_rating, review_scores_accuracy,
                  review_scores_cleanliness, review_scores_checkin, review_scores_communication, 
                  review_scores_location, review_scores_value, reviews_per_month) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.25) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

### variation 2 (non-parametric)
fe_nonpara_rec <- recipe(price ~., data = airbnb_train) %>%
  update_role(id, new_role = "id") %>%
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_knn(host_location, host_response_time, host_response_rate, host_acceptance_rate,
                  host_neighbourhood, bathrooms_text, beds, review_scores_rating, review_scores_accuracy,
                  review_scores_cleanliness, review_scores_checkin, review_scores_communication, 
                  review_scores_location, review_scores_value, reviews_per_month) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.25) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

# check recipes 
fe_rec %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

fe_nonpara_rec %>%
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

# write out recipes
save(baseline_rec, file = here("recipes/round_2/baseline_rec.rda"))
save(baseline_nonpara_rec, file = here("recipes/round_2/baseline_nonpara_rec.rda"))
save(fe_rec, file = here("recipes/round_2/fe_rec.rda"))
save(fe_nonpara_rec, file = here("recipes/round_2/fe_nonpara_rec.rda"))

