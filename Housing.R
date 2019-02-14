library(keras)
library(tidyverse)
library(DataExplorer)
library(rsample)
library(recipes)

attach(housing)
myData = housing
glimpse(myData)

myData = myData %>% select(median_house_value,everything())
glimpse(myData)

set.seed(100)
train_test_split = initial_split(myData, prop = 0.8)
train_test_split

train_tbl = training(train_test_split)
test_tbl = testing(train_test_split)

dim(train_tbl)
str(train_tbl)

dim(test_tbl)
str(test_tbl)

plot_missing(train_tbl)
plot_density(train_tbl)

rec_obj = recipe(median_house_value ~ ., data=train_tbl) %>% 
  step_bagimpute(all_predictors(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data=train_tbl)

train_x = bake(rec_obj, new_data=train_tbl) %>% select(-median_house_value)
test_x = bake(rec_obj, new_data=test_tbl) %>% select(-median_house_value)

train_y = train_tbl[,1]
test_y = test_tbl[,1]

model = keras_model_sequential() %>%
  layer_dense(units = 32, activation = "selu", initializer_he_normal() ,input_shape = c(ncol(train_x)))%>%
  layer_dense(units = 64, activation = "selu", initializer_he_normal()) %>%
  layer_dense(units = 128, activation = "selu", initializer_he_normal()) %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  optimizer = optimizer_adadelta(lr=1),
  loss = "mse",
  metrics = c("mae")
)

model %>% fit(as.matrix(train_x), train_y, epochs=10, batch_size=150, validation_split=.20)

metric = model %>% evaluate(as.matrix(test_x), test_y)
metric
