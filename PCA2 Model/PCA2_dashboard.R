# Dashboard helpers. Uses base R only; called automatically by PCA2model.R.
json_num <- function(x) {
  if (length(x) != 1L || !is.finite(x)) "null" else
    format(x, digits = 12, scientific = FALSE, trim = TRUE)
}
json_str <- function(x) encodeString(as.character(x), quote = '"')

dashboard_data <- function(result) {
  X <- result$X
  n <- nrow(X)
  fitted_origin <- as.numeric(predict(result$m1, newdata = X))
  fitted_target <- c(NA_real_, head(fitted_origin, -1L))
  residual_target <- X$trim - fitted_target
  # Fit and residual plots show only rows actually used in the regression.
  used_origin <- is.finite(X$trim_next) & is.finite(X$trim) & is.finite(X$PC1)
  fitted_target[!c(FALSE, head(used_origin, -1L))] <- NA_real_
  residual_target[!is.finite(fitted_target)] <- NA_real_
  latest <- tail(which(is.finite(X$trim) & is.finite(X$PC1)), 1)
  interval <- predict(result$m1, newdata = X[latest, ],
                      interval = "prediction", level = 0.80)
  observations <- vapply(seq_len(n), function(i) {
    vals <- c(trim = X$trim[i], pc1 = X$PC1[i],
              fitted = fitted_target[i], residual = residual_target[i])
    fields <- paste(sprintf('"%s":%s', names(vals),
                             vapply(vals, json_num, character(1))), collapse = ",")
    paste0('{"date":', json_str(X$q[i]), ",", fields, "}")
  }, character(1))
  labels <- c(ippi_yy = "Industrial product prices", rmpi_yy = "Raw materials",
              rmpi_oil_yy = "Crude energy", cpi_food_yy = "Food CPI",
              cpi_gas_yy = "Gasoline CPI", wage_yy = "Hourly wages")
  loadings <- vapply(names(result$loadings), function(nm) {
    paste0('{"name":', json_str(labels[[nm]]), ',"value":',
           json_num(result$loadings[[nm]]), "}")
  }, character(1))
  ct <- coef(summary(result$m1))
  coefs <- vapply(seq_len(nrow(ct)), function(i) {
    paste0('{"name":', json_str(rownames(ct)[i]),
           ',"estimate":', json_num(ct[i, 1]),
           ',"se":', json_num(ct[i, 2]), "}")
  }, character(1))
  forecast <- paste0('{"date":', json_str(result$forecast$forecast_quarter),
    ',"value":', json_num(interval[1, "fit"]),
    ',"lower":', json_num(interval[1, "lwr"]),
    ',"upper":', json_num(interval[1, "upr"]), "}")
  paste0('{"generated":', json_str(format(Sys.time(), "%Y-%m-%d %H:%M")),
    ',"latest":', json_str(X$q[latest]),
    ',"variance":', json_num(result$variance_share),
    ',"trainingN":', nrow(result$pc$x),
    ',"regressionN":', nobs(result$m1),
    ',"rSquared":', json_num(summary(result$m1)$r.squared),
    ',"rmse":', json_num(sqrt(mean(residuals(result$m1)^2))),
    ',"benchmarkRmse":', json_num(sqrt(mean(residuals(result$benchmark)^2))),
    ',"forecast":', forecast,
    ',"loadings":[', paste(loadings, collapse = ","), "]",
    ',"coefficients":[', paste(coefs, collapse = ","), "]",
    ',"rows":[', paste(observations, collapse = ","), "]}")
}

draw_pca2_plots <- function(result) {
  X <- result$X
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mfrow = c(2, 2), mar = c(4.5, 4, 3, 1), oma = c(1, 0, 0, 0),
      col.axis = "#536174", col.lab = "#243449",
      col.main = "#172C45", bg = "white")
  plot(X$q, X$PC1, type = "l", col = "#245781", lwd = 2,
       xlab = "", ylab = "Training-window SD", main = "Inflation-pressure index")
  abline(h = 0, col = "#DCE2E8")
  abline(h = 1, col = "#AF5339", lty = 2)
  fitted_origin <- as.numeric(predict(result$m1, newdata = X))
  fit <- c(NA_real_, head(fitted_origin, -1))
  fit[!c(FALSE, head(is.finite(X$trim_next) & is.finite(X$PC1) &
                    is.finite(X$trim), -1))] <- NA_real_
  fq <- result$forecast$forecast_quarter
  yy <- range(c(X$trim, fit, result$forecast$trim_forecast), na.rm = TRUE)
  plot(X$q, X$trim, type = "l", col = "#172C45", lwd = 2, ylim = yy,
       xlim = range(c(X$q, fq)), xlab = "", ylab = "Year-on-year %",
       main = "Core inflation: actual and in-sample fit")
  lines(X$q, fit, col = "#2D7D79", lty = 2, lwd = 1.5)
  points(fq, result$forecast$trim_forecast, pch = 19, col = "#AF5339")
  legend("topleft", c("Actual CPI-trim", "In-sample fitted", "Next-quarter forecast"),
         col = c("#172C45", "#2D7D79", "#AF5339"), lty = c(1, 2, NA),
         pch = c(NA, NA, 19), bty = "n", cex = 0.8)
  par(mar = c(4.5, 8, 3, 1))
  labs <- c("IPPI", "RMPI", "Crude energy", "Food CPI", "Gasoline CPI", "Wages")
  barplot(rev(result$loadings), names.arg = rev(labs), horiz = TRUE, las = 1,
          col = ifelse(rev(result$loadings) >= 0, "#245781", "#AF5339"),
          border = NA, xlim = c(-0.6, 0.6), xlab = "Signed PC1 loading",
          main = "What drives the index?")
  abline(v = 0, col = "#536174")
  par(mar = c(4.5, 4, 3, 1))
  plot(X$q, X$trim - fit, type = "h", col = "#536F88", xlab = "",
       ylab = "Percentage points", main = "In-sample residuals")
  abline(h = 0, col = "#AF5339", lty = 2)
}

write_pca2_dashboard <- function(result) {
  template_file <- file.path(project_dir, "PCA2_dashboard_template.html")
  if (!file.exists(template_file)) stop("Missing dashboard template: ", template_file)
  html <- paste(readLines(template_file, warn = FALSE, encoding = "UTF-8"),
                collapse = "\n")
  if (!grepl("__PCA2_JSON__", html, fixed = TRUE)) stop("Missing dashboard data placeholder.")
  html <- sub("__PCA2_JSON__", dashboard_data(result), html, fixed = TRUE)
  destination <- file.path(project_dir, "PCA2_dashboard.html")
  writeLines(enc2utf8(html), destination, useBytes = TRUE)
  png(file.path(project_dir, "PCA2_dashboard.png"), width = 1600, height = 1000,
      res = 120)
  tryCatch(draw_pca2_plots(result), finally = dev.off())
  # Also populate the RStudio Plots pane without opening a browser during batch runs.
  if (interactive()) {
    draw_pca2_plots(result)
    browseURL(destination)
  }
  cat("\nDashboard saved to:", destination, "\n")
  invisible(destination)
}
