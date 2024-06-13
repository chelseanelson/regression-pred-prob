# Regression Prediction Problem - Round 2
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
load(here("results/round_2/fitted_tuned_models/tuned_bt_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_poly_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_radial_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_mars_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_rf_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_knn_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_nnet_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_bt_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_poly_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_radial_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_mars_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_rf_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_knn_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_nnet_2.rda"))

# comparing sub-models ----
## KNN 
knn_1_plot <- tuned_knn_1 %>% autoplot(metric = "mae") # we should look at neighbors more deeply between 8 and 12
knn_2_plot <- tuned_knn_2 %>% autoplot(metric = "mae")
knn_best_1 <- show_best(tuned_knn_1, metric = "mae")
knn_best_2 <- show_best(tuned_knn_2, metric = "mae")

## NNET 
nnet_1_plot <- tuned_nnet_1 %>% autoplot(metric = "mae") # make my levels larger for hidden units, same with penalty
nnet_2_plot <- tuned_nnet_2 %>% autoplot(metric = "mae")
nnet_best_1 <- show_best(tuned_nnet_1, metric = "mae")
nnet_best_2 <- show_best(tuned_nnet_2, metric = "mae")

## SVM Polynomial
svm_poly_1_plot <- tuned_svm_poly_1 %>% autoplot(metric = "mae") # cost should be around 2 and up, degree of interaction 2 and up, scale factor should be .00562 or .1, try more levels on this as well 
svm_poly_2_plot <- tuned_svm_poly_2 %>% autoplot(metric = "mae")
svm_poly_best_1 <- show_best(tuned_svm_poly_1, metric = "mae")
svm_poly_best_2 <- show_best(tuned_svm_poly_2, metric = "mae")

## SVM Radial
svm_radial_1_plot <- tuned_svm_radial_1 %>% autoplot(metric = "mae") # cost should be 2 and up, rbf should be 0.00316 and up, try more levels on this 
svm_radial_2_plot <- tuned_svm_radial_2 %>% autoplot(metric = "mae")
svm_radial_best_1 <- show_best(tuned_svm_radial_1, metric = "mae")
svm_radial_best_2 <- show_best(tuned_svm_radial_2, metric = "mae")

## Boosted Trees
bt_1_plot <- tuned_bt_1 %>% autoplot(metric = "mae") # leave trees at 1000, min_n of 1 or 2, mtry could try higher, change learn rate from -5 to (-4 to -0.1) make more levels 
bt_2_plot <- tuned_bt_2 %>% autoplot(metric = "mae")
bt_best_1 <- show_best(tuned_bt_1, metric = "mae")
bt_best_2 <- show_best(tuned_bt_2, metric = "mae")

## Random Forests 
rf_1_plot <- tuned_rf_1 %>% autoplot(metric = "mae") # min_n should be 1, mtry should be higher 
rf_2_plot <- tuned_rf_2 %>% autoplot(metric = "mae")
rf_best_1 <- show_best(tuned_rf_1, metric = "mae")
rf_best_2 <- show_best(tuned_rf_2, metric = "mae")

## MARS
mars_1_plot <- tuned_mars_1 %>% autoplot(metric = "mae") # model terms should be higher 25 to 50, and then # degree of interaction could be 2 and higher 
mars_2_plot <- tuned_mars_2 %>% autoplot(metric = "mae")
mars_best_1 <- show_best(tuned_mars_1, metric = "mae") 
mars_best_2 <- show_best(tuned_mars_2, metric = "mae")


model_results_baseline <- as_workflow_set(mars_1 = tuned_mars_1,
                                          svm_poly_1 = tuned_svm_poly_1,
                                          svm_radial_1 = tuned_svm_radial_1,
                                          rf_1 = tuned_rf_1,
                                          knn_1 = tuned_knn_1,
                                          nnet_1 = tuned_nnet_1,
                                          bt_1 = tuned_bt_1
)

# Best Model Currently: Baseline Boosted Tree

model_results <- as_workflow_set(mars_1 = tuned_mars_1,
                                 svm_poly_1 = tuned_svm_poly_1,
                                 svm_radial_1 = tuned_svm_radial_1,
                                 bt_1 = tuned_bt_1,
                                 rf_1 = tuned_rf_1,
                                 knn_1 = tuned_knn_1,
                                 nnet_1 = tuned_nnet_1,
                                 mars_2 = tuned_mars_2,
                                 svm_poly_2 = tuned_svm_poly_2,
                                 svm_radial_2 = tuned_svm_radial_2,
                                 bt_2 = tuned_bt_2,
                                 rf_2 = tuned_rf_2,
                                 knn_2 = tuned_knn_2,
                                 nnet_2 = tuned_nnet_2
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

# Best Model Overall: Baseline Boosted Tree Model 

# write out results (plots, tables)
write_rds(model_accuracy_comparison, file = here("results/round_2/fitted_tuned_models/model_accuracy_comparison.rds"))
