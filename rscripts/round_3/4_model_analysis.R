# Regression Prediction Problem - Round 3
# Analysis of tuned and trained models (comparison)
# Main Assessment Metric : MAE (Mean Absolute Error)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load in tuned models 
load(here("results/round_3/fitted_tuned_models/tuned_bt_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_svm_poly_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_svm_radial_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_mars_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_rf_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_bt_2.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_svm_radial_2.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_mars_2.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_rf_2.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_bt_3.rda"))

# comparing sub-models ----
## SVM Polynomial
svm_poly_1_plot <- tuned_svm_poly_1 %>% autoplot(metric = "mae")
svm_poly_best_1 <- show_best(tuned_svm_poly_1, metric = "mae")

## SVM Radial
svm_radial_1_plot <- tuned_svm_radial_1 %>% autoplot(metric = "mae") # cost between 1 and 32 best
svm_radial_2_plot <- tuned_svm_radial_2 %>% autoplot(metric = "mae")
svm_radial_best_1 <- show_best(tuned_svm_radial_1, metric = "mae") # got worse
svm_radial_best_2 <- show_best(tuned_svm_radial_2, metric = "mae")

## Boosted Trees
bt_1_plot <- tuned_bt_1 %>% autoplot(metric = "mae") # 18 - 25 works well here, trees 1300-1500, 0.0245 learn rate good - .0464 
bt_2_plot <- tuned_bt_2 %>% autoplot(metric = "mae")
bt_3_plot <- tuned_bt_3 %>% autoplot(metric = "mae")
bt_best_1 <- show_best(tuned_bt_1, metric = "mae") 
bt_best_2 <- show_best(tuned_bt_2, metric = "mae")
bt_best_3 <- show_best(tuned_bt_3, metric = "mae")

## Random Forests 
rf_1_plot <- tuned_rf_1 %>% autoplot(metric = "mae") # same trees, look at mtry between 30 and 50 
rf_2_plot <- tuned_rf_2 %>% autoplot(metric = "mae")
rf_best_1 <- show_best(tuned_rf_1, metric = "mae")
rf_best_2 <- show_best(tuned_rf_2, metric = "mae") 

## MARS
mars_1_plot <- tuned_mars_1 %>% autoplot(metric = "mae") # prod_degree 3 and 4 work best, num_terms should be around 40-60 
mars_2_plot <- tuned_mars_2 %>% autoplot(metric = "mae")
mars_best_1 <- show_best(tuned_mars_1, metric = "mae") 
mars_best_2 <- show_best(tuned_mars_2, metric = "mae")


model_results_baseline <- as_workflow_set(mars_1 = tuned_mars_1,
                                          svm_poly_1 = tuned_svm_poly_1,
                                          svm_radial_1 = tuned_svm_radial_1,
                                          rf_1 = tuned_rf_1,
                                          bt_1 = tuned_bt_1
)

# Best Model Currently: Baseline Boosted Tree 1

model_results <- as_workflow_set(mars_1 = tuned_mars_1,
                                 svm_poly_1 = tuned_svm_poly_1,
                                 svm_radial_1 = tuned_svm_radial_1,
                                 bt_1 = tuned_bt_1,
                                 rf_1 = tuned_rf_1,
                                 mars_2 = tuned_mars_2,
                                 svm_radial_2 = tuned_svm_radial_2,
                                 bt_2 = tuned_bt_2,
                                 rf_2 = tuned_rf_2,
                                 bt_3 = tuned_bt_3
                        
)

model_accuracy_comparison_baseline <- model_results_baseline %>%
  collect_metrics() %>%
  filter(.metric == "mae") %>%
  slice_min(mean, by = wflow_id) %>% 
  arrange(std_err) %>%
  arrange(mean) %>% 
  select(wflow_id, .metric, mean, std_err, n) %>% 
  rename(metric = .metric)

model_accuracy_comparison <- model_results %>%
  collect_metrics() %>%
  filter(.metric == "mae") %>%
  slice_min(mean, by = wflow_id) %>% 
  arrange(std_err) %>%
  arrange(mean) %>% 
  select(wflow_id, .metric, mean, std_err, n) %>% 
  rename(metric = .metric)

# Best Model Overall:  Baseline Boosted Trees 1

# write out results (plots, tables)
write_rds(model_accuracy_comparison, file = here("results/round_3/fitted_tuned_models/model_accuracy_comparison.rds"))
