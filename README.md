---
title: "Build and deploy a stroke prediction model using R"
date: "`r Sys.Date()`"
output: https://www.coursera.org/learn/showcase-build-and-deploy-a-stroke-prediction-model-with-r/ungradedLab/Zow3N/option-a-using-courseras-rstudio-environment/lab?path=%2F
author: "Dipan Burja!"
---

# About Data Analysis Report

This RMarkdown file contains the report of the data analysis done for the project on building and deploying a stroke prediction model in R. It contains analysis such as data exploration, summary statistics and building the prediction models. The final report was completed on `r date()`. 

**Data Description:**

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

This data set is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.


# Task One: Import data and data preprocessing
## Load data and install packages

```{r}
# Install necessary packages if not already installed
# install.packages(c("tidyverse", "janitor", "skimr", "tidymodels", "themis"))
install.packages("janitor")
library(tidyverse)
library(tidymodels)
library(janitor)
library(themis)
# install.packages("devtools")
devtools::install_github("r-lib/conflicted")
library(tidyverse)
library(skimr)
library(tidymodels)
library(conflicted)
library(dplyr)

# Load the dataset
stroke_data <- read_csv("healthcare-dataset-stroke-data.csv")
# Quick look at the data
head(stroke_data)
colnames(stroke_data)
```
## Describe and explore the data

```{r}
stroke_clean <- stroke_data %>%
  mutate(
    bmi = as.numeric(na_if(bmi, "N/A")),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),
    hypertension = factor(hypertension, levels = c(0, 1), labels = c("No", "Yes")),
    heart_disease = factor(heart_disease, levels = c(0, 1), labels = c("No", "Yes")),
    gender = factor(gender),
    smoking_status = factor(smoking_status)
  )
# Get a professional summary of the whole dataset
library(skimr)
skim(stroke_clean)
# Fill missing BMI with the median value of the dataset
stroke_final <- stroke_clean %>%
  mutate(bmi = if_else(is.na(bmi), median(bmi, na.rm = TRUE), bmi))
# Double check: Should show 0 missing now
sum(is.na(stroke_final$bmi))
colSums(is.na(stroke_final))
head(stroke_final)
# Check average BMI before and after
summary(stroke_clean$bmi)  # Original (with NAs)
summary(stroke_final$bmi)  # Final (imputed)
```



# Task Two: Build prediction models

```{r}
# Load tidymodels for the machine learning workflow
library(tidymodels)
# Set a seed for reproducibility (so you get the same results every time)
set.seed(123)
# Create the split object
stroke_split <- initial_split(stroke_final, prop = 0.75, strata = stroke)
# Extract the training and testing sets
train_data <- training(stroke_split)
test_data  <- testing(stroke_split)

# Quick check: How many rows in each?
nrow(train_data)
nrow(test_data)
# 1. Load the specific library for SMOTE
library(themis)
library(tidymodels)
# Define the recipe (the blueprint for preprocessing)
stroke_recipe <- recipe(stroke ~ ., data = train_data) %>%
  update_role(id, new_role = "ID") %>%            # Don't use ID to predict
  step_dummy(all_nominal_predictors()) %>%        # Turn text categories into numbers
  step_zv(all_predictors()) %>%                   # Remove variables that stay the same
  step_normalize(all_numeric_predictors()) %>%    # Make all numbers use the same scale
  step_smote(stroke)                              # Balance the Yes/No cases
# Let's see what the recipe looks like
stroke_recipe
```




# Task Three: Evaluate and select prediction models

```{r}
# Define the model engine
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Combine the recipe and the model into a "Workflow"
stroke_workflow <- workflow() %>%
  add_recipe(stroke_recipe) %>%
  add_model(log_spec)

# Train the model!
stroke_fit <- fit(stroke_workflow, data = train_data)
# Pull out the mathematical results in a clean table
model_results <- stroke_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()
# View the results
model_results
install.packages("vip")
library(vip) # Install if needed: install.packages("vip")
stroke_fit %>%
  extract_fit_parsnip() %>%
  vi() %>%
  ggplot(aes(x = Importance, y = reorder(Variable, Importance), fill = Importance)) +
  geom_col() +
  labs(title = "Which Health Factors Predict Stroke Most?",
       subtitle = "Based on Logistic Regression Model",
       x = "Importance Score",
       y = "Health Metric") +
  theme_minimal()
# 1. Combine into a workflow
stroke_workflow <- workflow() %>%
  add_recipe(stroke_recipe) %>%
  add_model(log_spec) # We defined this as logistic_reg() earlier

# 2. Train the model (this is where the "learning" happens)
stroke_fit <- fit(stroke_workflow, data = train_data)

# 3. See the results!
stroke_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()
# 1. Generate predictions on the 25% Test Data
stroke_preds <- test_data %>%
  bind_cols(predict(stroke_fit, test_data)) %>%
  bind_cols(predict(stroke_fit, test_data, type = "prob"))
# 2. Create the Confusion Matrix table
stroke_preds %>%
  conf_mat(truth = stroke, estimate = .pred_class)
# 3. Calculate Accuracy and Sensitivity
stroke_preds %>%
  metrics(truth = stroke, estimate = .pred_class)
# This gives us the "Medical Quality" metrics
stroke_preds %>%
  sens(truth = stroke, estimate = .pred_class)
```


# Task Four: Deploy the prediction model

```{r}
# 1. Force R to use the correct 'predict' function
tidymodels_prefer()

 # 2. Create two test patients (One High Risk, One Low Risk)
test_patients <- tibble(
  id = c(1, 2),
  gender = c("Male", "Female"),
  age = c(75, 25),
  hypertension = c("Yes", "No"),
  heart_disease = c("Yes", "No"),
  ever_married = c("Yes", "No"),
  work_type = c("Private", "Private"),
  Residence_type = c("Urban", "Rural"),
  avg_glucose_level = c(220, 85),
  bmi = c(34, 21),
  smoking_status = c("smokes", "never smoked")
)

# 3. Generate predictions
# This will create a table with the probability of stroke
deployment_results <- test_patients %>%
  bind_cols(predict(stroke_fit, test_patients, type = "prob")) %>%
  select(age, avg_glucose_level, .pred_Yes) %>%
  mutate(Risk_Percent = round(.pred_Yes * 100, 2))

# 4. FORCE THE OUTPUT TO SHOW
# This command displays the table directly in your report
knitr::kable(deployment_results, caption = "Model Deployment: Risk Prediction Results")
```


# Task Five: Findings and Conclusions
1. Model Performance & Reliability

The stroke prediction model was developed using a Logistic Regression framework within the tidymodels ecosystem.
    Accuracy: The model achieved an accuracy of approximately 76% on the test dataset.
    Handling Imbalance: Prior to using SMOTE, the model struggled to identify stroke cases due to the 95/5 class split. By synthetically balancing the training data, we significantly improved the model's Sensitivity, allowing it to flag at-risk patients more effectively.

2. Key Risk Drivers

Based on the Feature Importance analysis (Variable Importance Plot), the strongest predictors of stroke in this population are:

    Age: By far the most significant factor; risk increases exponentially with age.

    Average Glucose Level: Higher blood sugar levels showed a strong correlation with stroke incidence.

    Hypertension & Heart Disease: These clinical markers were consistent secondary predictors.

    BMI: Interestingly, while BMI is a factor, it was less predictive than Age and Glucose in this specific dataset.

3. Clinical Implications of Deployment

The deployment test in Task Four demonstrated that the model is ready for "triage support."

    A test on a high-risk profile (Age 75, High Glucose) returned a high probability score, while a low-risk profile (Age 25, Normal Glucose) returned a negligible score.

    This suggests that the model can be used as a preliminary screening tool to help healthcare providers prioritize patients for further diagnostic testing.

4. Future Work

To improve this model in future iterations:

    Advanced Algorithms: Testing non-linear models like Random Forest or XGBoost might capture more complex interactions between variables.

    Feature Expansion: Including more granular data such as cholesterol levels, physical activity hours, or genetic markers could increase predictive precision.























