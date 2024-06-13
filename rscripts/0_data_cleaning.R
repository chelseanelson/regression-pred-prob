# Regression Prediction Problem
# Initial data cleaning to load in correctly

# load packages
library(tidyverse)
library(here)

train_regression <- read_csv(here("data/train_regression.csv"),
                                 col_types = cols(id = col_character())) %>%
  mutate(
    price = parse_number(price),
    across(where(is.logical), factor),
    across(where(is.character), factor),
    host_response_rate = as.numeric(str_remove(host_response_rate, "%")),
    host_acceptance_rate = as.numeric(str_remove(host_acceptance_rate, "%")),
    number_vertifications = as.factor(str_count(host_verifications, "\\b\\w+\\b")),
    in_illinois = if_else(
      str_detect(host_location, "IL") | str_detect(host_location, "Illinois"),
      "Yes",
      "No"
    )
  )

test_regression <- read_csv(here("data/test_regression.csv"),
                                col_types = cols(id = col_character())) %>%
  mutate(
    across(where(is.logical), factor),
    across(where(is.character), factor),
    host_response_rate = as.numeric(str_remove(host_response_rate, "%")),
    host_acceptance_rate = as.numeric(str_remove(host_acceptance_rate, "%")),
    number_vertifications = as.factor(str_count(host_verifications, "\\b\\w+\\b")),
    in_illinois = if_else(
      str_detect(host_location, "IL") | str_detect(host_location, "Illinois"),
      "Yes",
      "No"
    )
  )

# write out rds
write_rds(train_regression, file = here("data/train_regression.rds"))
write_rds(test_regression, file = here("data/test_regression.rds"))
