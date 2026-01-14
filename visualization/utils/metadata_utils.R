load_pig_metadata <- function(metadata_path = file.path("..", "data", "PigGTEx_v0.MetaTable.xlsx")) {
  candidate_paths <- unique(c(
    metadata_path,
    file.path("data", "PigGTEx_v0.MetaTable.xlsx"),
    file.path("..", "data", "PigGTEx_v0.MetaTable.xlsx")
  ))

  existing_path <- candidate_paths[file.exists(candidate_paths)][1]
  if (is.na(existing_path)) {
    stop("Metadata file not found: ", metadata_path)
  }

  if (!requireNamespace("readxl", quietly = TRUE)) {
    stop("Package 'readxl' is required to load metadata.")
  }

  raw <- readxl::read_excel(existing_path)

  parse_age_days <- function(age_value) {
    if (is.na(age_value)) {
      return(NA_real_)
    }

    if (is.numeric(age_value)) {
      return(as.numeric(age_value))
    }

    age_str <- tolower(trimws(as.character(age_value)))
    if (age_str == "" || grepl("unknown", age_str)) {
      return(NA_real_)
    }

    value <- suppressWarnings(as.numeric(gsub("[^0-9.]+", "", age_str)))
    if (is.na(value)) {
      return(NA_real_)
    }

    if (grepl("day", age_str)) {
      return(value)
    }
    if (grepl("week", age_str)) {
      return(value * 7)
    }
    if (grepl("month", age_str)) {
      return(value * 30)
    }
    if (grepl("year", age_str)) {
      return(value * 365)
    }

    NA_real_
  }

  assign_stage <- function(age_days) {
    if (is.na(age_days)) {
      return(NA_character_)
    }
    if (age_days <= 20) {
      return("Infant_0_20d")
    }
    if (age_days <= 59) {
      return("Early childhood_21_59d")
    }
    if (age_days <= 149) {
      return("Pre_pubertal_60_149d")
    }
    if (age_days <= 365) {
      return("Post_pubertal_150_365d")
    }
    "Adult_>365d"
  }

  metadata <- raw %>%
    dplyr::rename(
      Sample_ID = BioSample,
      Tissue = `Main categories`,
      Tissue_detail = `Sub categories`,
      Tissue_class = `Tissue class`,
      Sex = Sex,
      Age_raw = Age
    ) %>%
    dplyr::mutate(
      Age_days = vapply(Age_raw, parse_age_days, numeric(1)),
      Stage = vapply(Age_days, assign_stage, character(1))
    ) %>%
    dplyr::select(Sample_ID, Tissue, Tissue_detail, Tissue_class, Sex, Age_raw, Age_days, Stage)

  metadata
}
