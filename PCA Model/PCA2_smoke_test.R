# Lightweight tests; synthetic values are NOT economic results.
options(pca2.functions_only = TRUE)
source("C:/Users/jacki/Desktop/School/Govenor's Challenge/PCA2 Model/PCA2model.R")
t <- valet("CPI_TRIM")
b <- read_fred(file.path(project_dir, "DCOILBRENTEU.csv"))
stopifnot(nrow(t) > 300, nrow(b) > 40, !anyDuplicated(b$m))
cat("PASS: Valet metadata reader and FRED monthly reader\n")
set.seed(21)
dates <- seq(as.Date("1997-01-01"), as.Date("2026-09-01"), by = "month")
f <- cumsum(rnorm(length(dates), 0.15, 0.3))
can <- data.frame(m = dates)
for (nm in names(vectors)) {
  can[[nm]] <- exp(4 + 0.02 * f + rnorm(length(dates), 0, 0.01))
}
tr <- data.frame(m = dates, value = 2 + sin(seq_along(dates) / 20))
invisible(capture.output(ans <- build_model(can, tr)))
train <- ans$X$q >= as.Date("2000-01-01") & ans$X$q <= as.Date("2019-10-01")
stopifnot(abs(mean(ans$X$PC1[train])) < 1e-10,
          abs(sd(ans$X$PC1[train]) - 1) < 1e-10,
          nrow(ans$X) == 119)
can2 <- can
after <- can2$m >= as.Date("2020-01-01")
can2[after, names(vectors)] <- can2[after, names(vectors)] * 1.5
invisible(capture.output(ans2 <- build_model(can2, tr)))
stopifnot(isTRUE(all.equal(ans$loadings, ans2$loadings)),
          isTRUE(all.equal(ans$X$PC1[train], ans2$X$PC1[train])))
# An incomplete final quarter must not receive a score.
invisible(capture.output(partial <- build_model(head(can, -1), head(tr, -1))))
stopifnot(is.na(tail(partial$X$PC1, 1)))
cat("PASS: quarterly aggregation, training-only PCA scaling, forecast regression\n")
cat("PASS: future-data invariance and partial-quarter exclusion\n")
