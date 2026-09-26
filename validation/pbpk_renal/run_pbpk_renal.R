#!/usr/bin/env Rscript
# =============================================================================
# TrialSense Module 2 - PBPK ground-truth validation of renal impairment
# =============================================================================
#
# Loads PK-Sim .pkml simulations exported at several eGFR levels, runs each one,
# computes AUC(0-infinity) for the PARENT drug in peripheral venous plasma, and
# expresses each impaired level as an AUC ratio against the eGFR 90 reference.
#
# SCIENTIFIC CONTRACT
#   * This script NEVER writes a model parameter. The only mutation it performs
#     on a loaded simulation is the OUTPUT SELECTION (which curve to record) via
#     clearOutputs()/setOutputs(). Renal impairment comes exclusively from the
#     "Chronic kidney disease" individual created in the PK-Sim desktop app.
#     There is no call to setParameterValuesByPath / setQuantityValuesByPath
#     anywhere below - grep the file to confirm.
#   * Nothing is imputed. A model that fails to load, fails to run, or yields no
#     AUC_inf is written to the CSV with auc = NA and a `status` + `note`
#     explaining why. No value is ever guessed or carried over.
#
# PK-SIM CKD RANGE LIMIT  (read from PKSimDB.sqlite, 2026-09-26)
#   The "Chronic Kidney Disease - No Dialysis (Malik et al, 2020)" disease state
#   accepts eGFR in [0, 60] mL/min/1.73m^2 only. The database stores the upper
#   bound as 0.00034683 l/min/dm^2, which is exactly 60.0 mL/min/1.73m^2. That
#   matches the clinical definition of CKD stage 3+ (eGFR < 60).
#   Consequences for this study:
#     * eGFR 90 (normal) CANNOT be built with the CKD option. The reference arm
#       is instead a HEALTHY individual, matched on age/weight/height/gender/
#       population. That is also what renal-impairment studies use as control.
#     * eGFR 75 CANNOT be produced at all through the CKD option, and is left
#       as a documented gap rather than faked by editing GFR by hand.
#     * eGFR 45 / 22 / 10 are within range and use the CKD disease state.
#   Each row carries `egfr_actual`, read back out of the simulation, so the
#   nominal label is never taken on trust.
#
# EXPECTED INPUT
#   One .pkml per (drug, eGFR) in PKML_DIR, named:   <Drug>_eGFR<value>.pkml
#   e.g. Fluconazole_eGFR90.pkml (built HEALTHY), Fluconazole_eGFR22.pkml (CKD)
#
# ENVIRONMENT (verified 2026-09-26 on this machine)
#   R 4.6.1 | ospsuite 12.4.5 | rSharp 1.2.3 (.NET 8) | PK-Sim 12.3.173
#   rSharp MUST stay <= 1.2.3. rSharp 2.0.0 targets .NET 10 and breaks
#   loadSimulation() on .pkml files when paired with ospsuite 12.4.x.
#
# USAGE
#   "C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe" run_pbpk_renal.R
#   optional: Rscript run_pbpk_renal.R <pkml_dir> <out_csv>
# =============================================================================

# ---- library path (user library created during setup) -----------------------
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.6")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

suppressPackageStartupMessages(library(ospsuite))

# =============================================================================
# CONFIGURATION
# =============================================================================

script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", a, value = TRUE)
  if (length(m)) dirname(normalizePath(sub("^--file=", "", m[1]))) else getwd()
})

args     <- commandArgs(trailingOnly = TRUE)
PKML_DIR <- if (length(args) >= 1) args[1] else file.path(script_dir, "pkml")
OUT_CSV  <- if (length(args) >= 2) args[2] else file.path(script_dir, "pksim_renal_predictions.csv")

EGFR_LEVELS <- c(90, 75, 45, 22, 10)   # mL/min/1.73m^2
REF_EGFR    <- 90                      # normal-function reference for the ratio

# Upper bound of PK-Sim's CKD disease state, in mL/min/1.73m^2.
# Verified against PKSimDB.sqlite (tab_container_parameter_rates, CKD/eGFR):
# max_value 0.00034683 l/min/dm^2 * 173000 = 60.0.
CKD_MAX_EGFR <- 60

# Levels above the CKD ceiling. REF_EGFR is buildable as a Healthy control;
# any other above-ceiling level is simply not producible via the CKD option.
ABOVE_CKD_CEILING <- EGFR_LEVELS[EGFR_LEVELS > CKD_MAX_EGFR]

# Extrapolated tail beyond which AUC_inf is flagged as poorly supported.
# 20% is the conventional non-compartmental acceptance limit.
EXTRAP_WARN <- 0.20

# ---- observation window -----------------------------------------------------
# Renal impairment lengthens half-life several-fold, so a window chosen for a
# healthy subject often stops before the terminal phase at eGFR 10, leaving
# AUC_inf resting on extrapolation instead of on simulated data.
#
# When AUTO_EXTEND is TRUE the simulation is first run exactly as published. If
# that leaves more than EXTRAP_WARN of AUC_inf extrapolated (or yields no
# AUC_inf at all), it is re-run once with a longer observation window and the
# better-supported result is kept. Both the decision and the window used are
# recorded per row, so this is never silent.
#
# This changes only WHEN the solver is observed, never WHAT is being solved:
# setOutputInterval() rewrites the output schema, not a model parameter. The
# administration protocol still governs dosing, so a longer window cannot add
# doses. Verified on ospsuite's Aciclovir example: extending 1440 -> 20000 min
# moved AUC_inf by 0.04% (4072.63 -> 4073.14) while the extrapolated tail fell
# from 0.21% to 0.00% - i.e. AUC_inf is window-insensitive once the terminal
# phase is captured, which is exactly the property that makes this safe.
AUTO_EXTEND    <- TRUE
EXTEND_FACTOR  <- 10      # multiple of the model's own end time to extend to
EXTEND_POINTS  <- 3000    # target number of output points over the new window

# Provenance: exact commit pinned at download time (2026-09-26).
# Frozen here on purpose so the CSV always cites the precise model revision.
MODEL_SOURCES <- c(
  Fluconazole  = "https://github.com/Open-Systems-Pharmacology/Fluconazole-Model/blob/fa23aa5a5865d544b9c2aeba6ba4629df31de0ac/Fluconazole-Model.json",
  Metformin    = "https://github.com/Open-Systems-Pharmacology/Metformin-Model/blob/5586d01d42ffa688a0fc751187f437b67ad8e013/Metformin-Model.json",
  Methotrexate = "https://github.com/Open-Systems-Pharmacology/Methotrexate-Model/blob/6cd01494129888e4bca14e79bfca7e2d48853624/Methotrexate-Model.json",
  Omeprazole   = "https://github.com/Open-Systems-Pharmacology/Omeprazole-Model/blob/c8b0f70a40c8513146a32a350957a81d01737431/Omeprazole-Model.json",
  Atorvastatin = "https://github.com/Open-Systems-Pharmacology/Atorvastatin-Model/blob/bf3befea6c1269e48e9c193cd4a08aad7dfe314d/Atorvastatin-Model.json",
  Ketoconazole = "https://github.com/Open-Systems-Pharmacology/Ketoconazole-Model/blob/bb8d0029bffdec2961eaab41d3f6dda149ff30ef/Ketoconazole-Model.json",
  Theophylline = "https://github.com/Open-Systems-Pharmacology/Theophylline-Model/blob/0610b095ab3eb7adfa73df6298796ee7ad37368c/Theophylline-Model.pksim5"
)
DRUGS <- names(MODEL_SOURCES)

# SOURCES.tsv (written by the download step) overrides the frozen table above
# when present, so re-downloading models keeps the CSV provenance truthful.
sources_tsv <- file.path(script_dir, "models", "SOURCES.tsv")
if (file.exists(sources_tsv)) {
  st <- tryCatch(
    utils::read.delim(sources_tsv, stringsAsFactors = FALSE),
    error = function(e) NULL
  )
  if (!is.null(st) && all(c("drug", "permalink") %in% names(st))) {
    # one row per drug: prefer the snapshot .json over the .pksim5
    st$rank <- ifelse(grepl("[.]json$", st$file), 1L, 2L)
    st <- st[order(st$drug, st$rank), ]
    st <- st[!duplicated(st$drug), ]
    for (i in seq_len(nrow(st))) {
      if (st$drug[i] %in% names(MODEL_SOURCES)) {
        MODEL_SOURCES[[st$drug[i]]] <- st$permalink[i]
      }
    }
    message("Provenance URLs refreshed from models/SOURCES.tsv")
  }
}

# =============================================================================
# HELPERS
# =============================================================================

#' PK-Sim build version stamped inside the .pkml by the exporting application.
#' Returns the most specific pKSimVersion attribute found (e.g. "12.3.173").
#' The attribute occurs many times: the Individual carries a short form
#' ("12.3") while the PK-Sim Module carries the full build ("12.3.173"), often
#' thousands of lines in. Scan in bounded chunks and stop as soon as a full
#' three-component build number is seen, so large .pkml files stay cheap.
read_pksim_version <- function(pkml_path, chunk = 5000L, max_lines = 400000L) {
  out <- tryCatch({
    con <- file(pkml_path, open = "r", encoding = "UTF-8")
    on.exit(close(con), add = TRUE)
    seen <- character(0)
    read_so_far <- 0L
    repeat {
      lines <- readLines(con, n = chunk, warn = FALSE, skipNul = TRUE)
      if (!length(lines)) break
      read_so_far <- read_so_far + length(lines)
      hits <- regmatches(lines, gregexpr('pKSimVersion="[^"]*"', lines))
      hits <- unlist(hits, use.names = FALSE)
      if (length(hits)) {
        seen <- unique(c(seen, gsub('pKSimVersion="|"', "", hits)))
        nparts <- vapply(strsplit(seen, ".", fixed = TRUE), length, 1L)
        if (any(nparts >= 3)) return(seen[which.max(nparts)])
      }
      if (read_so_far >= max_lines) break
    }
    if (!length(seen)) return(NA_character_)
    seen[which.max(vapply(strsplit(seen, ".", fixed = TRUE), length, 1L))]
  }, error = function(e) NA_character_)
  if (length(out) != 1 || is.na(out) || !nzchar(out)) NA_character_ else out
}

#' Locate the peripheral-venous-plasma observer belonging to the PARENT drug.
#' Metabolites also publish this observer, so the molecule segment of the path
#' must match the drug name exactly - otherwise we would silently integrate a
#' metabolite curve and report it as parent exposure.
pick_parent_output <- function(sim, drug) {
  obs <- getAllObserverPathsIn(sim)
  if (!length(obs)) {
    return(list(path = NA_character_, note = "simulation exposes no observers"))
  }

  plasma <- obs[
    grepl("PeripheralVenousBlood", obs, fixed = TRUE) &
    grepl("Plasma (Peripheral Venous Blood)", obs, fixed = TRUE)
  ]
  if (!length(plasma)) {
    return(list(
      path = NA_character_,
      note = "no 'Plasma (Peripheral Venous Blood)' observer in simulation"
    ))
  }

  # molecule name is the segment immediately before the observer name
  molecule_of <- function(p) {
    seg <- toPathArray(p)
    if (length(seg) >= 2) seg[length(seg) - 1] else NA_character_
  }
  mols <- vapply(plasma, molecule_of, character(1), USE.NAMES = FALSE)

  exact <- which(tolower(mols) == tolower(drug))
  if (length(exact) == 1) {
    return(list(path = plasma[exact], note = NA_character_))
  }
  if (length(exact) > 1) {
    return(list(path = plasma[exact[1]],
                note = paste0("ambiguous parent observer; used first of: ",
                              paste(plasma[exact], collapse = " ; "))))
  }
  if (length(plasma) == 1) {
    return(list(path = plasma[1],
                note = paste0("no observer molecule matched '", drug,
                              "'; used sole plasma observer for molecule '",
                              mols[1], "'")))
  }
  list(path = NA_character_,
       note = paste0("parent '", drug, "' not among plasma observers: ",
                     paste(unique(mols), collapse = ", ")))
}

#' Latest end time of a simulation's output schema, in hours.
#' ospsuite stores schedule times in minutes.
end_hours_of <- function(simulation) {
  tryCatch({
    iv <- simulation$outputSchema$intervals
    if (!length(iv)) return(NA_real_)
    max(vapply(iv, function(x) as.numeric(x$endTime$value), numeric(1))) / 60
  }, error = function(e) NA_real_)
}

#' Load -> select parent output -> run -> AUC_inf. Never throws.
evaluate_pkml <- function(pkml_path, drug) {
  res <- list(
    auc = NA_real_, auc_unit = NA_character_, frac_extrap = NA_real_,
    output_path = NA_character_, pksim_version = NA_character_,
    egfr_actual = NA_real_, sim_end_h = NA_real_,
    status = NA_character_, note = NA_character_
  )
  res$pksim_version <- read_pksim_version(pkml_path)

  sim <- tryCatch(loadSimulation(pkml_path), error = function(e) e)
  if (inherits(sim, "error")) {
    res$status <- "load_failed"
    res$note   <- paste("loadSimulation() error:", conditionMessage(sim))
    return(res)
  }

  # Read the individual's realised eGFR straight out of the simulation, so the
  # filename label is verified rather than believed. ospsuite returns base
  # units (l/min/dm^2); convert with the package's own unit table.
  res$egfr_actual <- tryCatch({
    p <- getParameter("Organism|Kidney|eGFR", sim, stopIfNotFound = FALSE)
    if (is.null(p)) NA_real_ else as.numeric(toUnit(p, p$value, "ml/min/1.73m²"))
  }, error = function(e) NA_real_)

  sel <- pick_parent_output(sim, drug)
  if (is.na(sel$path)) {
    res$status <- "no_parent_output"
    res$note   <- sel$note
    return(res)
  }
  res$output_path <- sel$path
  carry_note <- sel$note

  # ---- OUTPUT SELECTION ONLY - no model parameter is touched ----------------
  ok <- tryCatch({
    clearOutputs(sim)
    setOutputs(quantitiesOrPaths = sel$path, simulation = sim)
    TRUE
  }, error = function(e) e)
  if (inherits(ok, "error")) {
    res$status <- "output_selection_failed"
    res$note   <- paste("setOutputs() error:", conditionMessage(ok))
    return(res)
  }

  # ---- one run + PK analysis over whatever window `sim` currently has -------
  run_once <- function(simulation) {
    sr <- tryCatch(runSimulations(simulation)[[1]], error = function(e) e)
    if (inherits(sr, "error")) {
      return(list(err = "run_failed",
                  msg = paste("runSimulations() error:", conditionMessage(sr))))
    }
    pk_df <- tryCatch({
      as.data.frame(pkAnalysesToDataFrame(calculatePKAnalyses(sr)))
    }, error = function(e) e)
    if (inherits(pk_df, "error")) {
      return(list(err = "pk_analysis_failed",
                  msg = paste("calculatePKAnalyses() error:",
                              conditionMessage(pk_df))))
    }
    mine <- pk_df[pk_df$QuantityPath == sel$path, , drop = FALSE]
    arow <- mine[mine$Parameter == "AUC_inf", , drop = FALSE]
    if (!nrow(arow)) {
      return(list(err = "auc_inf_absent",
                  msg = paste("AUC_inf not returned for", sel$path)))
    }
    frow <- mine[mine$Parameter == "FractionAucLastToInf", , drop = FALSE]
    list(
      err   = NULL,
      auc   = suppressWarnings(as.numeric(arow$Value[1])),
      unit  = as.character(arow$Unit[1]),
      frac  = if (nrow(frow)) suppressWarnings(as.numeric(frow$Value[1])) else NA_real_,
      end_h = end_hours_of(simulation)
    )
  }

  first <- run_once(sim)
  if (!is.null(first$err)) {
    res$status <- first$err
    res$note   <- first$msg
    return(res)
  }

  best        <- first
  extend_note <- NA_character_

  # Re-run once with a longer window if the terminal phase was not captured.
  needs_more <- is.na(first$auc) ||
                (!is.na(first$frac) && first$frac > EXTRAP_WARN)

  if (AUTO_EXTEND && needs_more) {
    new_end_h <- (if (is.na(first$end_h)) 24 else first$end_h) * EXTEND_FACTOR
    retry <- tryCatch({
      sim2 <- loadSimulation(pkml_path)
      clearOutputs(sim2)
      setOutputs(quantitiesOrPaths = sel$path, simulation = sim2)
      setOutputInterval(simulation = sim2,
                        startTime  = 0,
                        endTime    = new_end_h * 60,          # minutes
                        resolution = EXTEND_POINTS / (new_end_h * 60))
      run_once(sim2)
    }, error = function(e) list(err = "extend_failed", msg = conditionMessage(e)))

    if (is.null(retry$err) && !is.na(retry$auc) &&
        (is.na(first$frac) || is.na(retry$frac) || retry$frac <= first$frac)) {
      best <- retry
      extend_note <- sprintf(
        paste0("observation window extended %.4g h -> %.4g h to capture the ",
               "terminal phase (extrapolated tail %s -> %s); dosing unchanged"),
        first$end_h, retry$end_h,
        if (is.na(first$frac)) "NA" else sprintf("%.1f%%", 100 * first$frac),
        if (is.na(retry$frac)) "NA" else sprintf("%.1f%%", 100 * retry$frac))
    } else {
      extend_note <- sprintf(
        "window extension to %.4g h did not improve the estimate; kept the model's own %.4g h window",
        new_end_h, first$end_h)
    }
  }

  res$auc         <- best$auc
  res$auc_unit    <- best$unit
  res$frac_extrap <- best$frac
  res$sim_end_h   <- best$end_h

  if (is.na(res$auc)) {
    res$status <- "auc_inf_na"
    res$note   <- paste(c(
      "AUC_inf returned NA - terminal slope not estimable over the simulated window",
      extend_note), collapse = " | ")
    return(res)
  }

  res$status <- "ok"
  notes <- c(carry_note, extend_note)
  # FractionAucLastToInf is the fraction of AUC_inf contributed by the
  # extrapolated tail. A large tail means AUC_inf leans on extrapolation
  # rather than on simulated data.
  if (!is.na(res$frac_extrap) && res$frac_extrap > EXTRAP_WARN) {
    notes <- c(notes, sprintf(
      "extrapolated tail %.1f%% of AUC_inf (> %.0f%%) - treat this AUC as poorly supported",
      100 * res$frac_extrap, 100 * EXTRAP_WARN))
  }
  notes <- notes[!is.na(notes)]
  if (length(notes)) res$note <- paste(notes, collapse = " | ")
  res
}

# =============================================================================
# MAIN
# =============================================================================

cat("=============================================================\n")
cat("TrialSense Module 2 - PBPK renal validation\n")
cat("=============================================================\n")
cat("pkml dir :", PKML_DIR, "\n")
cat("out csv  :", OUT_CSV, "\n")
cat("ospsuite :", as.character(packageVersion("ospsuite")), "\n")
cat("rSharp   :", as.character(packageVersion("rSharp")), "\n\n")

if (!dir.exists(PKML_DIR)) {
  stop("PKML_DIR does not exist: ", PKML_DIR,
       "\nExport the .pkml files from PK-Sim into that folder first.")
}

available <- list.files(PKML_DIR, pattern = "[.]pkml$", ignore.case = TRUE)
cat("found", length(available), ".pkml file(s)\n\n")

rows <- list()
for (drug in DRUGS) {
  for (egfr in EGFR_LEVELS) {

    expected <- paste0(drug, "_eGFR", egfr, ".pkml")
    hit <- available[tolower(available) == tolower(expected)]

    base <- data.frame(
      drug = drug, egfr = egfr,
      auc = NA_real_, auc_ratio = NA_real_,
      model_source_url = unname(MODEL_SOURCES[[drug]]),
      pksim_version = NA_character_,
      status = NA_character_, note = NA_character_,
      auc_unit = NA_character_, egfr_actual = NA_real_,
      frac_auc_extrapolated = NA_real_, sim_end_hours = NA_real_,
      output_path = NA_character_, pkml_file = NA_character_,
      stringsAsFactors = FALSE
    )

    if (!length(hit)) {
      if (egfr > CKD_MAX_EGFR && egfr != REF_EGFR) {
        # Not a missing export - it cannot be built in the first place.
        base$status <- "ckd_range_exceeded"
        base$note <- paste0(
          "eGFR ", egfr, " exceeds the PK-Sim CKD maximum of ", CKD_MAX_EGFR,
          " mL/min/1.73m2 (Malik et al 2020 parameterisation covers CKD ",
          "stage 3+ only), so it cannot be produced via the CKD disease ",
          "state; not simulated rather than fabricated by editing GFR")
        cat(sprintf("  %-13s eGFR %3d  NOT PRODUCIBLE (CKD max %d)\n",
                    drug, egfr, CKD_MAX_EGFR))
      } else {
        base$status <- "missing_pkml"
        base$note <- paste0("expected file '", expected,
                            "' not found in ", PKML_DIR,
                            " - simulation was not exported from PK-Sim")
        if (egfr == REF_EGFR) {
          base$note <- paste0(base$note,
            "; build this arm as a HEALTHY individual (Disease State = Healthy)",
            " - the CKD option cannot reach eGFR ", REF_EGFR)
        }
        cat(sprintf("  %-13s eGFR %3d  MISSING\n", drug, egfr))
      }
      rows[[length(rows) + 1]] <- base
      next
    }

    path <- file.path(PKML_DIR, hit[1])
    base$pkml_file <- hit[1]
    cat(sprintf("  %-13s eGFR %3d  ... ", drug, egfr)); flush.console()

    r <- evaluate_pkml(path, drug)
    base$auc                   <- r$auc
    base$auc_unit              <- r$auc_unit
    base$egfr_actual           <- r$egfr_actual
    base$frac_auc_extrapolated <- r$frac_extrap
    base$sim_end_hours         <- r$sim_end_h
    base$output_path           <- r$output_path
    base$pksim_version         <- r$pksim_version
    base$status                <- r$status
    base$note                  <- r$note

    # Cross-check the filename label against the simulation's realised eGFR.
    # The reference arm is deliberately Healthy, so its eGFR is whatever the
    # matched healthy physiology gives (typically ~100) - report, don't warn.
    if (!is.na(r$egfr_actual)) {
      lab <- if (egfr == REF_EGFR) {
        sprintf("reference arm built as Healthy control; realised eGFR %.1f mL/min/1.73m2",
                r$egfr_actual)
      } else if (abs(r$egfr_actual - egfr) > 1) {
        sprintf("LABEL MISMATCH: filename says eGFR %d but simulation reports %.1f mL/min/1.73m2",
                egfr, r$egfr_actual)
      } else NA_character_
      if (!is.na(lab)) {
        base$note <- if (is.na(base$note)) lab else paste(base$note, lab, sep = " | ")
      }
    }

    cat(if (identical(r$status, "ok"))
          sprintf("AUC_inf = %.4g %s\n", r$auc, r$auc_unit)
        else paste0(r$status, "\n"))

    rows[[length(rows) + 1]] <- base
  }
}

results <- do.call(rbind, rows)

# ---- AUC ratio vs the eGFR 90 reference, within each drug -------------------
for (drug in DRUGS) {
  idx <- which(results$drug == drug)
  ref <- results$auc[results$drug == drug & results$egfr == REF_EGFR]
  ref <- if (length(ref)) ref[1] else NA_real_

  if (is.na(ref) || !is.finite(ref) || ref <= 0) {
    for (i in idx) {
      msg <- paste0("no AUC ratio: eGFR ", REF_EGFR,
                    " reference unavailable for ", drug)
      results$note[i] <- if (is.na(results$note[i])) msg
                         else paste(results$note[i], msg, sep = " | ")
    }
    next
  }
  results$auc_ratio[idx] <- results$auc[idx] / ref
}

# ---- run metadata -----------------------------------------------------------
results$ospsuite_version <- as.character(packageVersion("ospsuite"))
results$run_timestamp    <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")

# Required column order first, diagnostics after.
col_order <- c("drug", "egfr", "auc", "auc_ratio",
               "model_source_url", "pksim_version",
               "status", "note", "auc_unit", "egfr_actual",
               "frac_auc_extrapolated", "sim_end_hours",
               "output_path", "pkml_file", "ospsuite_version", "run_timestamp")
results <- results[, col_order]
results <- results[order(match(results$drug, DRUGS),
                         -results$egfr), ]

dir.create(dirname(OUT_CSV), showWarnings = FALSE, recursive = TRUE)
utils::write.csv(results, OUT_CSV, row.names = FALSE, na = "")

# ---- summary ----------------------------------------------------------------
cat("\n-------------------------------------------------------------\n")
cat("wrote:", OUT_CSV, "\n")
cat("rows :", nrow(results), "\n\n")
cat("status breakdown:\n")
print(table(results$status, useNA = "ifany"))

ok <- results[results$status == "ok", ]
if (nrow(ok)) {
  cat("\nAUC ratio vs eGFR ", REF_EGFR, ":\n", sep = "")
  wide <- reshape(ok[, c("drug", "egfr", "auc_ratio")],
                  idvar = "drug", timevar = "egfr", direction = "wide")
  print(wide, row.names = FALSE)
} else {
  cat("\nNo simulation produced an AUC. Nothing to compare.\n")
}
cat("-------------------------------------------------------------\n")
