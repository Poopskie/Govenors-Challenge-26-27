# PCA2: simple Canadian inflation-pressure index and one-quarter forecast
# Open the PCA2 Model RStudio project, then Source this file.
# Graphs and dashboard use base R; only live StatCan pulls need cansim.
# If needed, install it yourself once with install.packages("cansim").
# This is an exploratory forecasting model, NOT a causal spillover estimate.
# Data: https://www.bankofcanada.ca/valet/docs
# StatCan R API: https://mountainmath.github.io/cansim/reference/get_cansim_vector.html

project_dir <- "C:/Users/jacki/Desktop/School/Govenor's Challenge/PCA2 Model"

month_date <- function(x) {
  as.Date(paste0(substr(as.character(x), 1, 7), "-01"))
}
quarter_date <- function(x) {
  y <- as.integer(format(x, "%Y"))
  m <- as.integer(format(x, "%m"))
  as.Date(sprintf("%04d-%02d-01", y, 1L + 3L * ((m - 1L) %/% 3L)))
}

# Valet CSVs include metadata. Locate OBSERVATIONS, not the first CSV row.
read_valet <- function(path, series) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  marker <- which(trimws(gsub('"', "", lines, fixed = TRUE)) == "OBSERVATIONS")
  if (!length(marker)) stop("No OBSERVATIONS section in: ", path)
  z <- read.csv(text = paste(lines[(marker[1] + 1L):length(lines)],
                             collapse = "\n"),
                stringsAsFactors = FALSE, check.names = FALSE,
                na.strings = c("", "NA", "."))
  if (!all(c("date", series) %in% names(z))) {
    stop("Expected date and ", series, "; found: ", paste(names(z), collapse = ", "))
  }
  data.frame(m = month_date(z$date), value = as.numeric(z[[series]]))
}

valet <- function(series, start = "1997-01-01") {
  local_file <- file.path(project_dir, paste0(series, ".csv"))
  if (file.exists(local_file)) {
    z <- read_valet(local_file, series)
  } else {
    url <- paste0("https://www.bankofcanada.ca/valet/observations/",
                  series, "/csv?start_date=", start)
    z <- read_valet(url, series)
  }
  z[z$m >= as.Date(start), , drop = FALSE]
}

# FRED headers can be DATE or observation_date; "." means missing.
# Average daily observations to MONTHLY before joining other monthly series.
read_fred <- function(path) {
  lines <- readLines(path, warn = FALSE)
  clean <- tolower(gsub('"', "", sub("^\ufeff", "", lines)))
  header <- grep("^(date|observation_date),", clean)
  if (!length(header)) stop("Cannot locate FRED date header in ", path)
  z <- read.csv(text = paste(lines[header[1]:length(lines)], collapse = "\n"),
                stringsAsFactors = FALSE, check.names = FALSE,
                na.strings = c(".", "", "NA"))
  if (!"DCOILBRENTEU" %in% names(z)) stop("Missing DCOILBRENTEU column.")
  d <- data.frame(m = month_date(z[[1]]),
                  brent = as.numeric(z$DCOILBRENTEU))
  aggregate(brent ~ m, data = d, FUN = mean)
}

vectors <- c(
  cpi_all   = "v41690973",
  cpi_food  = "v41690974",
  cpi_store = "v41690975",
  cpi_gas   = "v41691136",
  ippi      = "v1230995983",
  rmpi      = "v1230998135",
  rmpi_oil  = "v1230998136",
  wage      = "v2132579"
)

pull_statcan <- function() {
  if (!requireNamespace("cansim", quietly = TRUE)) {
    stop('Install the download package in RStudio first: install.packages("cansim")')
  }
  # Full date required; unname() avoids version-specific named-vector behaviour.
  sc <- cansim::get_cansim_vector(unname(vectors),
                                  start_time = "1997-01-01")
  if (is.null(sc) || !nrow(sc)) stop("StatCan returned no observations. Try again later.")
  # Use original VALUE: all chosen series are indices or dollars in units.
  required <- c("REF_DATE", "VECTOR", "VALUE")
  if (!all(required %in% names(sc))) {
    stop("Unexpected StatCan columns: ", paste(names(sc), collapse = ", "))
  }
  id <- paste0("v", sub("^v", "", as.character(sc$VECTOR)))
  parts <- lapply(names(vectors), function(nm) {
    rows <- id == vectors[[nm]]
    z <- data.frame(m = month_date(sc$REF_DATE[rows]),
                    value = as.numeric(sc$VALUE[rows]))
    z <- unique(z)
    if (!nrow(z) || anyDuplicated(z$m)) {
      stop("Missing or duplicate monthly observations for ", nm)
    }
    names(z)[2] <- nm
    z
  })
  Reduce(function(a, b) merge(a, b, by = "m", all = TRUE), parts)
}

build_model <- function(can, trim, brent = NULL) {
  names(trim)[2] <- "trim"
  monthly <- merge(can, trim, by = "m", all = TRUE)
  if (!is.null(brent)) monthly <- merge(monthly, brent, by = "m", all = TRUE)
  if (anyDuplicated(monthly$m)) stop("Monthly join created duplicate dates.")
  # Full calendar prevents lag(4) from silently jumping over missing quarters.
  calendar <- data.frame(m = seq(min(monthly$m), max(monthly$m), by = "month"))
  monthly <- merge(calendar, monthly, by = "m", all.x = TRUE, sort = TRUE)
  monthly$q <- quarter_date(monthly$m)
  cols <- setdiff(names(monthly), c("m", "q"))
  Q <- data.frame(q = sort(unique(monthly$q)))
  # Require three finite MONTHLY observations for every quarterly value.
  # Thus an unfinished quarter cannot be compared with complete quarters.
  for (nm in cols) {
    Q[[nm]] <- vapply(Q$q, function(q) {
      x <- monthly[[nm]][monthly$q == q]
      if (length(x) == 3L && all(is.finite(x))) mean(x) else NA_real_
    }, numeric(1))
  }
  level_cols <- intersect(c(names(vectors), "brent"), names(Q))
  for (nm in level_cols) {
    x <- Q[[nm]]
    previous_year <- c(rep(NA_real_, 4L), head(x, -4L))
    Q[[paste0(nm, "_yy")]] <- 100 * (x / previous_year - 1)
  }
  predictors <- c("ippi_yy", "rmpi_yy", "rmpi_oil_yy",
                  "cpi_food_yy", "cpi_gas_yy", "wage_yy")
  X <- Q[c("q", "trim", predictors)]
  A <- as.matrix(X[predictors])
  finite_rows <- rowSums(!is.finite(A)) == 0L
  train_rows <- X$q >= as.Date("2000-01-01") &
                X$q <= as.Date("2019-10-01") & finite_rows
  if (sum(train_rows) < 20L) stop("Too few complete 2000-2019 training quarters.")
  # PCA learns centring and scaling ONLY from the training window.
  pc <- prcomp(A[train_rows, , drop = FALSE], center = TRUE, scale. = TRUE)
  loadings <- pc$rotation[, 1]
  # A factor's sign is arbitrary; orient its average loading positively.
  direction <- if (sum(loadings) < 0) -1 else 1
  train_scores <- pc$x[, 1] * direction
  index <- rep(NA_real_, nrow(X))
  new_scores <- predict(pc, newdata = A[finite_rows, , drop = FALSE])[, 1]
  index[finite_rows] <- (new_scores * direction - mean(train_scores)) /
                       sd(train_scores)
  X$PC1 <- index
  X$warn <- X$PC1 > 1
  # Create next-quarter target BEFORE dropping missing rows.
  X$trim_next <- c(tail(X$trim, -1L), NA_real_)
  fit_rows <- is.finite(X$PC1) & is.finite(X$trim) & is.finite(X$trim_next)
  m1 <- lm(trim_next ~ PC1 + trim, data = X[fit_rows, ])
  benchmark <- lm(trim_next ~ trim, data = X[fit_rows, ])
  variance_share <- pc$sdev[1]^2 / sum(pc$sdev^2)
  cat("\nTraining quarters:", sum(train_rows), "\nPC1 loadings:\n")
  print(round(loadings * direction, 3))
  cat("\nPC1 variance explained:", round(100 * variance_share, 1), "%\n")
  print(summary(m1))
  latest <- tail(which(is.finite(X$PC1) & is.finite(X$trim)), 1L)
  forecast <- data.frame(
    information_quarter = X$q[latest],
    forecast_quarter = seq(X$q[latest], by = "quarter", length.out = 2L)[2],
    trim_forecast = as.numeric(predict(m1, newdata = X[latest, ]))
  )
  cat("\nExploratory next-quarter forecast (year-on-year %, not q/q):\n")
  print(forecast)
  cat("\nIn-sample regression only; predictive value is not yet validated out of sample.\n")
  list(monthly = monthly, Q = Q, X = X, pc = pc, m1 = m1,
       benchmark = benchmark, loadings = loadings * direction,
       variance_share = variance_share, forecast = forecast)
}

run_pca2 <- function() {
  old_timeout <- getOption("timeout")
  options(timeout = max(300, old_timeout))
  on.exit(options(timeout = old_timeout), add = TRUE)
  trim <- valet("CPI_TRIM")
  cache <- file.path(project_dir, "statcan_monthly_cache.csv")
  # Set options(pca2.refresh = TRUE) before sourcing to refresh the live pull.
  if (file.exists(cache) && !isTRUE(getOption("pca2.refresh", FALSE))) {
    message("Using cached StatCan data. Set options(pca2.refresh = TRUE) to update.")
    can <- read.csv(cache, stringsAsFactors = FALSE)
    can$m <- as.Date(can$m)
  } else {
    can <- pull_statcan()
    write.csv(can, cache, row.names = FALSE)
  }
  fred_file <- file.path(project_dir, "DCOILBRENTEU.csv")
  brent <- if (file.exists(fred_file)) read_fred(fred_file) else NULL
  # Brent is optional and not a PCA input in this first-pass model.
  result <- build_model(can, trim, brent)
  write.csv(result$X, file.path(project_dir, "PCA2_quarterly_panel.csv"), row.names = FALSE)
  png(file.path(project_dir, "PCA2_index.png"), width = 1200, height = 650)
  tryCatch({
    plot(result$X$q, result$X$PC1, type = "l", xlab = "", ylab = "Training-window SD",
         main = "Exploratory inflation-pressure index (PC1)")
    abline(h = 1, lty = 2, col = "red")
  }, finally = dev.off())
  # Create the interactive browser dashboard plus a four-chart PNG.
  # No new package installation or web server is required.
  source(file.path(project_dir, "PCA2_dashboard.R"), local = TRUE)
  write_pca2_dashboard(result)
  result
}

# Keep functions importable for tests without downloading data or writing outputs.
if (!isTRUE(getOption("pca2.functions_only", FALSE))) {
  result <- run_pca2()
  X <- result$X
  Q <- result$Q
  pc <- result$pc
  m1 <- result$m1
}
