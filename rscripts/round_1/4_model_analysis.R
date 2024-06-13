# Regression Prediction Problem - Round 1 
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
load(here("results/round_1/fitted_tuned_models/fit_null.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_bt_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_svm_poly_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_svm_radial_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_mars_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_rf_1.rda"))
load(here("results/round_1/fitted_tuned_models/fit_linear_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_knn_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_nnet_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_bt_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_svm_poly_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_svm_radial_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_mars_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_rf_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_linear_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_knn_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_nnet_2.rda"))

# comparing sub-models ----
## Null Model
null_best <- show_best(fit_null, metric = "mae")

## Linear Regression
linear_best_1 <- show_best(fit_linear_1, metric = "mae")
linear_2_plot <- tuned_linear_2 %>% autoplot(metric = "mae")
linear_best_2 <- show_best(tuned_linear_2, metric = "mae")


## KNN 
knn_1_plot <- tuned_knn_1 %>% autoplot(metric = "mae") # neighbors between 13,15,17 the best however should try more than 20 also?? 
knn_2_plot <- tuned_knn_2 %>% autoplot(metric = "mae") # knn got worse, num_comp = 15, and less than 15 neighbors seems to be the best here 
knn_best_1 <- show_best(tuned_knn_1, metric = "mae") 
knn_best_2 <- show_best(tuned_knn_2, metric = "mae")

## NNET 
nnet_1_plot <- tuned_nnet_1 %>% autoplot(metric = "mae") # hidden unit should be > 5 
nnet_2_plot <- tuned_nnet_2 %>% autoplot(metric = "mae") # got worse, the more num_comp the better, hidden units should be > 15
nnet_best_1 <- show_best(tuned_nnet_1, metric = "mae")
nnet_best_2 <- show_best(tuned_nnet_2, metric = "mae")

## SVM Polynomial
svm_poly_1_plot <- tuned_svm_poly_1 %>% autoplot(metric = "mae") # higher degrees 3 and up might be good to look at
svm_poly_2_plot <- tuned_svm_poly_2 %>% autoplot(metric = "mae")
svm_poly_best_1 <- show_best(tuned_svm_poly_1, metric = "mae")
svm_poly_best_2 <- show_best(tuned_svm_poly_2, metric = "mae")

## SVM Radial
svm_radial_1_plot <- tuned_svm_radial_1 %>% autoplot(metric = "mae") # don't know how to manipulate, maybe recipe changes with help 
svm_radial_2_plot <- tuned_svm_radial_2 %>% autoplot(metric = "mae")
svm_radial_best_1 <- show_best(tuned_svm_radial_1, metric = "mae")
svm_radial_best_2 <- show_best(tuned_svm_radial_2, metric = "mae")

## Boosted Trees
bt_1_plot <- tuned_bt_1 %>% autoplot(metric = "mae") # min_n should be 1, mtry should be 15 and 45, tress 1000 to 1500
bt_2_plot <- tuned_bt_2 %>% autoplot(metric = "mae")
bt_best_1 <- show_best(tuned_bt_1, metric = "mae")
bt_best_2 <- show_best(tuned_bt_2, metric = "mae")

## Random Forests 
rf_1_plot <- tuned_rf_1 %>% autoplot(metric = "mae") # min_n should be 1, and mtry should be 15 and 45, tune trees
rf_2_plot <- tuned_rf_2 %>% autoplot(metric = "mae")
rf_best_1 <- show_best(tuned_rf_1, metric = "mae")
rf_best_2 <- show_best(tuned_rf_2, metric = "mae")

## MARS
mars_1_plot <- tuned_mars_1 %>% autoplot(metric = "mae") # look at model terms between 15 and 35, prod_degree should be > 3
mars_2_plot <- tuned_mars_2 %>% autoplot(metric = "mae") # got worse, the more num_comps the better and num_terms shoudl also be pretty high, prod_degree of 3 works best 
mars_best_1 <- show_best(tuned_mars_1, metric = "mae") 
mars_best_2 <- show_best(tuned_mars_2, metric = "mae")


model_results_baseline <- as_workflow_set(null = fit_null,
                                          linear_1 = fit_linear_1,
                                          mars_1 = tuned_mars_1,
                                          svm_poly_1 = tuned_svm_poly_1,
                                          svm_radial_1 = tuned_svm_radial_1,
                                          rf_1 = tuned_rf_1,
                                          knn_1 = tuned_knn_1,
                                          nnet_1 = tuned_nnet_1,
                                          bt_1 = tuned_bt_1
)

# Best Model Currently: Baseline Boosted Trees Model

model_results <- as_workflow_set(null = fit_null,
                                 linear_1 = fit_linear_1,
                                 linear_2 = tuned_linear_2,
                                 mars_1 = tuned_mars_1,
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
write_rds(model_accuracy_comparison, file = here("results/round_1/fitted_tuned_models/model_accuracy_comparison.rds"))
