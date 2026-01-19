#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# GLMM Analysis for BEAT-120 Experiment
# Implements the pre-registered statistical model from analysis_plan.md §2.1
#
# Model specification:
# logit(P_reversal) = β₀ + β₁·Reputation + β₂·Evidence + β₃·Framing + 
#                      β₄·Temperature + β₅·TopP + 
#                      β₆·(Reputation × Evidence) + 
#                      (1 | Model) + (1 | Question)

library(lme4)
library(emmeans)
library(car)
library(broom.mixed)
library(dplyr)
library(readr)
library(jsonlite)

# Suppress startup messages
suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
})

#' Fit Primary GLMM Model
#'
#' @param df Data frame with trial records
#' @return Fitted glmer model object
fit_primary_glmm <- function(df) {
  cat("Fitting primary GLMM model...\n")
  
  # Ensure factors are properly coded
  df <- df %>%
    mutate(
      reputation_factor = factor(reputation_factor, 
                                 levels = c("A0", "A1", "A2", "A3")),
      evidence_factor = factor(evidence_factor,
                              levels = c("B0", "B1", "B2", "B3")),
      framing_factor = factor(framing_factor,
                             levels = c("C1", "C2")),
      model_id = factor(model_id),
      question_id = factor(question_id)
    )
  
  # Fit model with random intercepts
  model <- glmer(
    reversal ~ reputation_factor + evidence_factor + framing_factor +
               temperature + top_p +
               reputation_factor:evidence_factor +
               (1 | model_id) + (1 | question_id),
    data = df,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
  )
  
  return(model)
}

#' Fit Random-Slope Model (Robustness Check)
#'
#' @param df Data frame with trial records
#' @return List with fitted models or NULL if non-convergent
fit_random_slope_models <- function(df) {
  cat("Fitting random-slope models (robustness check)...\n")
  
  models <- list()
  
  # Try (1 + Reputation | Model)
  tryCatch({
    model_rep <- glmer(
      reversal ~ reputation_factor + evidence_factor + framing_factor +
                 temperature + top_p +
                 reputation_factor:evidence_factor +
                 (1 + reputation_factor | model_id) + (1 | question_id),
      data = df,
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
    )
    models$reputation_slope <- model_rep
    cat("Random-slope model (Reputation) converged\n")
  }, error = function(e) {
    cat("✗ Random-slope model (Reputation) failed to converge\n")
    models$reputation_slope <- NULL
  })
  
  # Try (1 + Evidence | Model)
  tryCatch({
    model_evid <- glmer(
      reversal ~ reputation_factor + evidence_factor + framing_factor +
                 temperature + top_p +
                 reputation_factor:evidence_factor +
                 (1 | model_id) + (1 + evidence_factor | model_id) + (1 | question_id),
      data = df,
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
    )
    models$evidence_slope <- model_evid
    cat("Random-slope model (Evidence) converged\n")
  }, error = function(e) {
    cat("✗ Random-slope model (Evidence) failed to converge\n")
    models$evidence_slope <- NULL
  })
  
  return(models)
}

#' Compute Drift-Adjusted Effects
#'
#' @param model Fitted GLMM model
#' @param df Original data
#' @return Data frame with drift-adjusted estimates
compute_drift_adjusted_effects <- function(model, df) {
  cat("Computing drift-adjusted effects...\n")
  
  # Get estimated marginal means for each reputation level
  emm <- emmeans(model, ~ reputation_factor, type = "response")
  
  # Convert to data frame
  emm_df <- as.data.frame(emm)
  
  # Get A0 (drift baseline) probability
  drift_prob <- emm_df$prob[emm_df$reputation_factor == "A0"]
  
  # Compute drift-adjusted effects
  emm_df <- emm_df %>%
    mutate(
      drift_adjusted = prob - drift_prob,
      drift_baseline = drift_prob
    )
  
  return(emm_df)
}

#' Extract Model Coefficients with 95% CI
#'
#' @param model Fitted GLMM model
#' @return Data frame with coefficients, OR, and 95% CI
extract_coefficients <- function(model) {
  cat("Extracting coefficients...\n")
  
  # Get coefficients with CI
  coef_df <- broom.mixed::tidy(model, conf.int = TRUE, conf.level = 0.95) %>%
    filter(effect == "fixed") %>%
    mutate(
      OR = exp(estimate),
      OR_lower = exp(conf.low),
      OR_upper = exp(conf.high)
    ) %>%
    select(term, estimate, std.error, statistic, p.value, OR, OR_lower, OR_upper)
  
  return(coef_df)
}

#' Compute Estimated Marginal Means for Key Comparisons
#'
#' @param model Fitted GLMM model
#' @return List of emmeans results
compute_emmeans <- function(model) {
  cat("Computing estimated marginal means...\n")
  
  results <- list()
  
  # Main effect of Reputation
  results$reputation <- emmeans(model, ~ reputation_factor, type = "response")
  
  # Main effect of Evidence
  results$evidence <- emmeans(model, ~ evidence_factor, type = "response")
  
  # Reputation × Evidence interaction
  results$reputation_x_evidence <- emmeans(model, ~ reputation_factor * evidence_factor, 
                                          type = "response")
  
  # Pairwise comparisons (key contrasts)
  results$contrasts_reputation <- pairs(results$reputation, adjust = "none")
  results$contrasts_evidence <- pairs(results$evidence, adjust = "none")
  
  # Specific contrasts for SSI
  # A3 vs A1 at B0
  results$contrast_a3_vs_a1_b0 <- contrast(
    results$reputation_x_evidence,
    list("A3_vs_A1_at_B0" = c(0, 1, 0, -1, rep(0, 12)))  # Adjust indices based on factor levels
  )
  
  return(results)
}

#' Model Diagnostics and Goodness-of-Fit
#'
#' @param model Fitted GLMM model
#' @return List with diagnostic statistics
model_diagnostics <- function(model) {
  cat("Running model diagnostics...\n")
  
  diagnostics <- list()
  
  # Variance components
  diagnostics$variance_components <- as.data.frame(VarCorr(model))
  
  # ICC (Intraclass Correlation Coefficient)
  vc <- VarCorr(model)
  var_model <- attr(vc$model_id, "stddev")^2
  var_question <- attr(vc$question_id, "stddev")^2
  var_residual <- pi^2 / 3  # For logistic regression
  
  diagnostics$icc_model <- var_model / (var_model + var_question + var_residual)
  diagnostics$icc_question <- var_question / (var_model + var_question + var_residual)
  
  # AIC and BIC
  diagnostics$aic <- AIC(model)
  diagnostics$bic <- BIC(model)
  
  # Log-likelihood
  diagnostics$loglik <- logLik(model)
  
  # Number of observations
  diagnostics$n_obs <- nobs(model)
  
  return(diagnostics)
}

#' Save Results to Output Directory
#'
#' @param results List of analysis results
#' @param output_dir Output directory path
save_results <- function(results, output_dir) {
  cat("Saving results to", output_dir, "...\n")
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save coefficients (Table 2)
  write.csv(results$coefficients, 
            file.path(output_dir, "glmm_coefficients.csv"),
            row.names = FALSE)
  
  # Save drift-adjusted effects
  write.csv(results$drift_adjusted,
            file.path(output_dir, "glmm_drift_adjusted.csv"),
            row.names = FALSE)
  
  # Save estimated marginal means
  write.csv(as.data.frame(results$emmeans$reputation),
            file.path(output_dir, "emmeans_reputation.csv"),
            row.names = FALSE)
  
  write.csv(as.data.frame(results$emmeans$evidence),
            file.path(output_dir, "emmeans_evidence.csv"),
            row.names = FALSE)
  
  write.csv(as.data.frame(results$emmeans$reputation_x_evidence),
            file.path(output_dir, "emmeans_reputation_x_evidence.csv"),
            row.names = FALSE)
  
  # Save contrasts
  write.csv(as.data.frame(results$emmeans$contrasts_reputation),
            file.path(output_dir, "contrasts_reputation.csv"),
            row.names = FALSE)
  
  write.csv(as.data.frame(results$emmeans$contrasts_evidence),
            file.path(output_dir, "contrasts_evidence.csv"),
            row.names = FALSE)
  
  # Save diagnostics
  write.csv(results$diagnostics$variance_components,
            file.path(output_dir, "variance_components.csv"),
            row.names = FALSE)
  
  # Save model summary as text
  sink(file.path(output_dir, "glmm_summary.txt"))
  print(summary(results$model))
  sink()
  
  # Save diagnostics as JSON
  diag_minimal <- list(
    icc_model = results$diagnostics$icc_model,
    icc_question = results$diagnostics$icc_question,
    aic = results$diagnostics$aic,
    bic = results$diagnostics$bic,
    n_obs = results$diagnostics$n_obs
  )
  writeLines(toJSON(diag_minimal, pretty = TRUE),
             file.path(output_dir, "diagnostics.json"))
  
  cat("All results saved\n")
}

#' Main Analysis Function
#'
#' @param trials_file Path to trials.jsonl or trials.csv
#' @param output_dir Output directory for results
main <- function(trials_file, output_dir = "results/glmm") {
  cat("=" %R% rep("=", 60) %R% "\n")
  cat("GLMM Analysis for BEAT-120\n")
  cat("=" %R% rep("=", 60) %R% "\n\n")
  
  # Load data
  cat("Loading trial data from:", trials_file, "\n")
  
  if (grepl("\\.csv$", trials_file)) {
    df <- read_csv(trials_file, show_col_types = FALSE)
  } else if (grepl("\\.jsonl$", trials_file)) {
    # Read JSONL
    lines <- readLines(trials_file)
    df <- bind_rows(lapply(lines, fromJSON))
  } else {
    stop("Unsupported file format. Use .csv or .jsonl")
  }
  
  cat("Loaded", nrow(df), "trials\n")
  
  # Filter to valid trials (no truncation/format violations)
  df_valid <- df %>%
    filter(
      !pass1_truncated,
      !pass2_truncated,
      !pass1_format_violation,
      !pass2_format_violation
    )
  
  cat("Valid trials:", nrow(df_valid), 
      sprintf("(%.1f%%)\n", 100 * nrow(df_valid) / nrow(df)))
  
  # Fit primary model
  model <- fit_primary_glmm(df_valid)
  
  # Extract results
  results <- list()
  results$model <- model
  results$coefficients <- extract_coefficients(model)
  results$drift_adjusted <- compute_drift_adjusted_effects(model, df_valid)
  results$emmeans <- compute_emmeans(model)
  results$diagnostics <- model_diagnostics(model)
  
  # Try random-slope models
  results$random_slope_models <- fit_random_slope_models(df_valid)
  
  # Save results
  save_results(results, output_dir)
  
  cat("\n" %R% rep("=", 60) %R% "\n")
  cat("GLMM Analysis Complete\n")
  cat("=" %R% rep("=", 60) %R% "\n")
  cat("\nKey Results:\n")
  cat("  • AIC:", results$diagnostics$aic, "\n")
  cat("  • BIC:", results$diagnostics$bic, "\n")
  cat("  • ICC (Model):", sprintf("%.3f", results$diagnostics$icc_model), "\n")
  cat("  • ICC (Question):", sprintf("%.3f", results$diagnostics$icc_question), "\n")
  cat("\nOutput saved to:", output_dir, "\n")
  cat("  • glmm_coefficients.csv (Table 2)\n")
  cat("  • emmeans_*.csv (marginal means)\n")
  cat("  • glmm_summary.txt (full model output)\n")
  cat("=" %R% rep("=", 60) %R% "\n")
  
  return(results)
}

# Command-line interface
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript analyze_glmm.R <trials_file> [output_dir]\n")
    cat("Example: Rscript analyze_glmm.R results/trials.jsonl results/glmm\n")
    quit(status = 1)
  }
  
  trials_file <- args[1]
  output_dir <- ifelse(length(args) >= 2, args[2], "results/glmm")
  
  results <- main(trials_file, output_dir)
}

