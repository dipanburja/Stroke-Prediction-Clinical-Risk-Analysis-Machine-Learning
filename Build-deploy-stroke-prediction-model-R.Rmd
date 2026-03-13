# ============================================================
# STROKE PREDICTION MODEL - COMPLETE ADDITIONS
# Adds: EDA Visuals, Cross-Validation, More Models, ROC Curve
# Author: Dipan Burja
# ============================================================

# --- SETUP: Install & Load All Required Packages ---
packages <- c("tidyverse", "tidymodels", "themis", "janitor",
              "skimr", "vip", "pROC", "ggcorrplot",
              "ranger", "xgboost", "patchwork")

installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!pkg %in% installed) install.packages(pkg)
}

library(tidyverse)
library(tidymodels)
library(themis)
library(janitor)
library(skimr)
library(vip)
library(pROC)
library(ggcorrplot)
library(ranger)
library(xgboost)
library(patchwork)

tidymodels_prefer()

# --- LOAD & CLEAN DATA (same as original project) ---
stroke_data <- read_csv("healthcare-dataset-stroke-data.csv")

stroke_clean <- stroke_data %>%
  mutate(
    bmi = as.numeric(na_if(bmi, "N/A")),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),
    hypertension = factor(hypertension, levels = c(0, 1), labels = c("No", "Yes")),
    heart_disease = factor(heart_disease, levels = c(0, 1), labels = c("No", "Yes")),
    gender = factor(gender),
    smoking_status = factor(smoking_status)
  )

stroke_final <- stroke_clean %>%
  mutate(bmi = if_else(is.na(bmi), median(bmi, na.rm = TRUE), bmi))


# ============================================================
# ADDITION 1: EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ============================================================
cat("\n--- TASK: EDA VISUALIZATIONS ---\n")

# --- Plot 1: Age Distribution by Stroke Status ---
p1 <- ggplot(stroke_final, aes(x = age, fill = stroke)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  labs(
    title = "Age Distribution by Stroke Status",
    subtitle = "Stroke cases are heavily skewed toward older patients",
    x = "Age", y = "Density", fill = "Had Stroke?"
  ) +
  theme_minimal(base_size = 13)

# --- Plot 2: Glucose Level by Stroke Status ---
p2 <- ggplot(stroke_final, aes(x = stroke, y = avg_glucose_level, fill = stroke)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21) +
  scale_fill_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  labs(
    title = "Glucose Levels vs Stroke",
    subtitle = "Higher glucose is associated with stroke",
    x = "Had Stroke?", y = "Average Glucose Level"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

# --- Plot 3: Stroke Rate by Smoking Status ---
p3 <- stroke_final %>%
  group_by(smoking_status) %>%
  summarise(stroke_rate = mean(stroke == "Yes") * 100) %>%
  ggplot(aes(x = reorder(smoking_status, stroke_rate), y = stroke_rate, fill = stroke_rate)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#90CAF9", high = "#B71C1C") +
  coord_flip() +
  labs(
    title = "Stroke Rate by Smoking Status",
    x = "Smoking Status", y = "Stroke Rate (%)"
  ) +
  theme_minimal(base_size = 13)

# --- Plot 4: Stroke Rate by Hypertension & Heart Disease ---
p4 <- stroke_final %>%
  group_by(hypertension, heart_disease) %>%
  summarise(stroke_rate = mean(stroke == "Yes") * 100, .groups = "drop") %>%
  ggplot(aes(x = hypertension, y = stroke_rate, fill = heart_disease)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = c("No" = "#64B5F6", "Yes" = "#EF5350")) +
  labs(
    title = "Stroke Rate: Hypertension × Heart Disease",
    subtitle = "Patients with BOTH conditions have the highest risk",
    x = "Hypertension", y = "Stroke Rate (%)", fill = "Heart Disease?"
  ) +
  theme_minimal(base_size = 13)

# --- Combine all 4 EDA plots into one figure ---
eda_combined <- (p1 | p2) / (p3 | p4)
print(eda_combined)
ggsave("eda_visualizations.png", eda_combined, width = 14, height = 10, dpi = 150)
cat("✅ EDA plots saved as 'eda_visualizations.png'\n")


# ============================================================
# TRAIN/TEST SPLIT + RECIPE (shared by all models)
# ============================================================
set.seed(123)
stroke_split <- initial_split(stroke_final, prop = 0.75, strata = stroke)
train_data   <- training(stroke_split)
test_data    <- testing(stroke_split)

# Shared preprocessing recipe
stroke_recipe <- recipe(stroke ~ ., data = train_data) %>%
  update_role(id, new_role = "ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(stroke)


# ============================================================
# ADDITION 2: CROSS-VALIDATION (10-Fold CV)
# ============================================================
cat("\n--- TASK: 10-FOLD CROSS-VALIDATION ---\n")

# Create 10-fold cross-validation folds (stratified on stroke)
set.seed(456)
cv_folds <- vfold_cv(train_data, v = 10, strata = stroke)

# Define Logistic Regression model
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Workflow for CV
log_workflow <- workflow() %>%
  add_recipe(stroke_recipe) %>%
  add_model(log_spec)

# Run Cross-Validation
cv_metrics <- log_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics   = metric_set(accuracy, roc_auc, sens, spec),
    control   = control_resamples(save_pred = TRUE)
  )

# Summarise CV results
cv_summary <- collect_metrics(cv_folds_results <- cv_metrics)
print(cv_summary)

# Plot CV metric distribution across folds
cv_plot <- cv_metrics %>%
  collect_metrics(summarize = FALSE) %>%
  ggplot(aes(x = .metric, y = .estimate, fill = .metric)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "10-Fold Cross-Validation Results (Logistic Regression)",
    subtitle = "Each dot = one fold. Boxplot shows stability across folds.",
    x = "Metric", y = "Score"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

print(cv_plot)
ggsave("cross_validation_results.png", cv_plot, width = 9, height = 6, dpi = 150)
cat("✅ Cross-validation plot saved as 'cross_validation_results.png'\n")


# ============================================================
# ADDITION 3: COMPARE THREE MODELS
# ============================================================
cat("\n--- TASK: MODEL COMPARISON (Logistic vs Random Forest vs XGBoost) ---\n")

# --- Model 1: Logistic Regression (already defined above) ---

# --- Model 2: Random Forest ---
rf_spec <- rand_forest(trees = 500) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(stroke_recipe) %>%
  add_model(rf_spec)

# --- Model 3: XGBoost ---
xgb_spec <- boost_tree(trees = 500, learn_rate = 0.05) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_workflow <- workflow() %>%
  add_recipe(stroke_recipe) %>%
  add_model(xgb_spec)

# --- Fit all three models on training data ---
cat("Training Logistic Regression...\n")
log_fit <- fit(log_workflow, data = train_data)

cat("Training Random Forest...\n")
rf_fit  <- fit(rf_workflow,  data = train_data)

cat("Training XGBoost...\n")
xgb_fit <- fit(xgb_workflow, data = train_data)

# --- Generate predictions on test data ---
get_preds <- function(model_fit, model_name) {
  test_data %>%
    bind_cols(predict(model_fit, test_data)) %>%
    bind_cols(predict(model_fit, test_data, type = "prob")) %>%
    mutate(model = model_name)
}

log_preds <- get_preds(log_fit, "Logistic Regression")
rf_preds  <- get_preds(rf_fit,  "Random Forest")
xgb_preds <- get_preds(xgb_fit, "XGBoost")

all_preds <- bind_rows(log_preds, rf_preds, xgb_preds)

# --- Compare metrics across all three models ---
comparison_metrics <- all_preds %>%
  group_by(model) %>%
  summarise(
    Accuracy    = accuracy_vec(truth = stroke, estimate = .pred_class),
    ROC_AUC     = roc_auc_vec(truth = stroke, estimate = .pred_Yes),
    Sensitivity = sens_vec(truth = stroke, estimate = .pred_class),
    Specificity = spec_vec(truth = stroke, estimate = .pred_class),
    .groups = "drop"
  )

cat("\n📊 Model Comparison Table:\n")
print(comparison_metrics)

# Visualise comparison
comparison_long <- comparison_metrics %>%
  pivot_longer(-model, names_to = "Metric", values_to = "Score")

comparison_plot <- ggplot(comparison_long,
                          aes(x = Metric, y = Score, fill = model)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_text(aes(label = round(Score, 3)),
            position = position_dodge(width = 0.9),
            vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c(
    "Logistic Regression" = "#42A5F5",
    "Random Forest"       = "#66BB6A",
    "XGBoost"             = "#FFA726"
  )) +
  ylim(0, 1.05) +
  labs(
    title    = "Model Comparison: Logistic Regression vs Random Forest vs XGBoost",
    subtitle = "Higher is better for all metrics",
    x = NULL, y = "Score", fill = "Model"
  ) +
  theme_minimal(base_size = 13)

print(comparison_plot)
ggsave("model_comparison.png", comparison_plot, width = 12, height = 7, dpi = 150)
cat("✅ Model comparison plot saved as 'model_comparison.png'\n")

# --- Feature Importance for best model (Random Forest) ---
rf_importance_plot <- rf_fit %>%
  extract_fit_parsnip() %>%
  vi() %>%
  slice_max(Importance, n = 12) %>%
  ggplot(aes(x = Importance, y = reorder(Variable, Importance), fill = Importance)) +
  geom_col() +
  scale_fill_gradient(low = "#90CAF9", high = "#1565C0") +
  labs(
    title    = "Random Forest: Top 12 Most Important Features",
    subtitle = "Based on Gini Impurity reduction",
    x = "Importance Score", y = "Feature"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

print(rf_importance_plot)
ggsave("rf_feature_importance.png", rf_importance_plot, width = 9, height = 6, dpi = 150)
cat("✅ Feature importance plot saved as 'rf_feature_importance.png'\n")


# ============================================================
# ADDITION 4: ROC CURVES (All Three Models)
# ============================================================
cat("\n--- TASK: ROC CURVES ---\n")

# Build ROC curve data for each model
build_roc_data <- function(preds_df, model_name) {
  roc_obj <- roc(
    response  = preds_df$stroke,
    predictor = preds_df$.pred_Yes,
    levels    = c("No", "Yes"),
    direction = "<"
  )
  auc_val <- round(as.numeric(auc(roc_obj)), 4)
  tibble(
    FPR   = 1 - roc_obj$specificities,
    TPR   = roc_obj$sensitivities,
    model = paste0(model_name, " (AUC = ", auc_val, ")")
  )
}

roc_data <- bind_rows(
  build_roc_data(log_preds, "Logistic Regression"),
  build_roc_data(rf_preds,  "Random Forest"),
  build_roc_data(xgb_preds, "XGBoost")
)

roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR, color = model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey50", linewidth = 0.8) +
  scale_color_manual(values = c(
    "Logistic Regression" = "#42A5F5",
    "Random Forest"       = "#66BB6A",
    "XGBoost"             = "#FFA726"
  ) %>% setNames(unique(roc_data$model))) +
  labs(
    title    = "ROC Curves: All Three Models",
    subtitle = "Closer to top-left corner = better model. Dashed line = random chance.",
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Sensitivity)",
    color    = "Model (AUC)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = c(0.65, 0.25),
        legend.background = element_rect(fill = "white", color = "grey85"))

print(roc_plot)
ggsave("roc_curves.png", roc_plot, width = 9, height = 7, dpi = 150)
cat("✅ ROC curve plot saved as 'roc_curves.png'\n")

# ============================================================
# FINAL SUMMARY
# ============================================================
cat("\n")
cat("============================================================\n")
cat("  PROJECT COMPLETE — FILES GENERATED:\n")
cat("============================================================\n")
cat("  📊 eda_visualizations.png      → Task 1 (EDA)\n")
cat("  📈 cross_validation_results.png → Task 2 (CV)\n")
cat("  🏆 model_comparison.png         → Task 3 (3 Models)\n")
cat("  🌲 rf_feature_importance.png    → Task 3 (RF Importance)\n")
cat("  📉 roc_curves.png               → Task 4 (ROC/AUC)\n")
cat("============================================================\n")
cat("\n  Best Model Summary:\n")
print(comparison_metrics %>% arrange(desc(ROC_AUC)))
