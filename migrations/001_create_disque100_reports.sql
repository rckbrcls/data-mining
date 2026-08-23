\set ON_ERROR_STOP on

BEGIN;

DROP TABLE IF EXISTS public.disque100_raw;

CREATE TABLE public.disque100_reports (
    id bigint generated always as identity primary key,
    source_hash varchar(64) not null constraint disque100_reports_source_hash_check check (source_hash ~ '^(?:[0-9A-F]{32}|[0-9A-F]{64})$'),
    registered_at timestamp(3) without time zone not null,
    service_channel text,
    emergency_status text,
    reporter_type text,
    violation_setting text,
    country text,
    state text,
    municipality text,
    frequency text,
    violation_start_period text,
    victim_count smallint not null constraint disque100_reports_victim_count_check check (victim_count > 0),
    vulnerable_group text,
    motivation text,
    victim_suspect_relationship text,
    victim_entity_type text,
    victim_gender text,
    victim_sexual_orientation text,
    victim_age_group text,
    victim_nationality text,
    victim_birthplace text,
    victim_naturalized_state text,
    victim_naturalized_municipality text,
    victim_disability text,
    victim_rare_disease text,
    victim_disability_related_to_rare_disease text,
    victim_incarceration_status text,
    victim_country text,
    victim_state text,
    victim_municipality text,
    victim_occupation text,
    victim_education_level text,
    victim_religion text,
    victim_race_color text,
    victim_ethnicity text,
    victim_income_range text,
    suspect_legal_nature text,
    suspect_gender text,
    suspect_sexual_orientation text,
    suspect_age_group text,
    suspect_nationality text,
    suspect_birthplace text,
    suspect_naturalized_state text,
    suspect_naturalized_municipality text,
    suspect_disability text,
    suspect_rare_disease text,
    suspect_disability_related_to_rare_disease text,
    suspect_incarceration_status text,
    suspect_country text,
    suspect_state text,
    suspect_municipality text,
    suspect_occupation text,
    suspect_education_level text,
    suspect_religion text,
    suspect_race_color text,
    suspect_ethnicity text,
    suspect_income_range text,
    suspect_organization_relationship text,
    suspect_business_sector text,
    suspect_ethnicity_details text,
    victim_ethnicity_details text,
    violation text
);

\copy public.disque100_reports (source_hash, registered_at, service_channel, emergency_status, reporter_type, violation_setting, country, state, municipality, frequency, violation_start_period, victim_count, vulnerable_group, motivation, victim_suspect_relationship, victim_entity_type, victim_gender, victim_sexual_orientation, victim_age_group, victim_nationality, victim_birthplace, victim_naturalized_state, victim_naturalized_municipality, victim_disability, victim_rare_disease, victim_disability_related_to_rare_disease, victim_incarceration_status, victim_country, victim_state, victim_municipality, victim_occupation, victim_education_level, victim_religion, victim_race_color, victim_ethnicity, victim_income_range, suspect_legal_nature, suspect_gender, suspect_sexual_orientation, suspect_age_group, suspect_nationality, suspect_birthplace, suspect_naturalized_state, suspect_naturalized_municipality, suspect_disability, suspect_rare_disease, suspect_disability_related_to_rare_disease, suspect_incarceration_status, suspect_country, suspect_state, suspect_municipality, suspect_occupation, suspect_education_level, suspect_religion, suspect_race_color, suspect_ethnicity, suspect_income_range, suspect_organization_relationship, suspect_business_sector, suspect_ethnicity_details, victim_ethnicity_details, violation) FROM 'data/disque100-primeiro-semestre-2026.csv' WITH (FORMAT csv, HEADER true, DELIMITER ';', ENCODING 'UTF8', NULL 'NULL');

UPDATE public.disque100_reports
SET
    service_channel = CASE WHEN TRIM(service_channel) = '' THEN NULL ELSE service_channel END,
    emergency_status = CASE WHEN TRIM(emergency_status) = '' THEN NULL ELSE emergency_status END,
    reporter_type = CASE WHEN TRIM(reporter_type) = '' THEN NULL ELSE reporter_type END,
    violation_setting = CASE WHEN TRIM(violation_setting) = '' THEN NULL ELSE violation_setting END,
    country = CASE WHEN TRIM(country) = '' THEN NULL ELSE country END,
    state = CASE WHEN TRIM(state) = '' THEN NULL ELSE state END,
    municipality = CASE WHEN TRIM(municipality) = '' THEN NULL ELSE municipality END,
    frequency = CASE WHEN TRIM(frequency) = '' THEN NULL ELSE frequency END,
    violation_start_period = CASE WHEN TRIM(violation_start_period) = '' THEN NULL ELSE violation_start_period END,
    vulnerable_group = CASE WHEN TRIM(vulnerable_group) = '' THEN NULL ELSE vulnerable_group END,
    motivation = CASE WHEN TRIM(motivation) = '' THEN NULL ELSE motivation END,
    victim_suspect_relationship = CASE WHEN TRIM(victim_suspect_relationship) = '' THEN NULL ELSE victim_suspect_relationship END,
    victim_entity_type = CASE WHEN TRIM(victim_entity_type) = '' THEN NULL ELSE victim_entity_type END,
    victim_gender = CASE WHEN TRIM(victim_gender) = '' THEN NULL ELSE victim_gender END,
    victim_sexual_orientation = CASE WHEN TRIM(victim_sexual_orientation) = '' THEN NULL ELSE victim_sexual_orientation END,
    victim_age_group = CASE WHEN TRIM(victim_age_group) = '' THEN NULL ELSE victim_age_group END,
    victim_nationality = CASE WHEN TRIM(victim_nationality) = '' THEN NULL ELSE victim_nationality END,
    victim_birthplace = CASE WHEN TRIM(victim_birthplace) = '' THEN NULL ELSE victim_birthplace END,
    victim_naturalized_state = CASE WHEN TRIM(victim_naturalized_state) = '' THEN NULL ELSE victim_naturalized_state END,
    victim_naturalized_municipality = CASE WHEN TRIM(victim_naturalized_municipality) = '' THEN NULL ELSE victim_naturalized_municipality END,
    victim_disability = CASE WHEN TRIM(victim_disability) = '' THEN NULL ELSE victim_disability END,
    victim_rare_disease = CASE WHEN TRIM(victim_rare_disease) = '' THEN NULL ELSE victim_rare_disease END,
    victim_disability_related_to_rare_disease = CASE WHEN TRIM(victim_disability_related_to_rare_disease) = '' THEN NULL ELSE victim_disability_related_to_rare_disease END,
    victim_incarceration_status = CASE WHEN TRIM(victim_incarceration_status) = '' THEN NULL ELSE victim_incarceration_status END,
    victim_country = CASE WHEN TRIM(victim_country) = '' THEN NULL ELSE victim_country END,
    victim_state = CASE WHEN TRIM(victim_state) = '' THEN NULL ELSE victim_state END,
    victim_municipality = CASE WHEN TRIM(victim_municipality) = '' THEN NULL ELSE victim_municipality END,
    victim_occupation = CASE WHEN TRIM(victim_occupation) = '' THEN NULL ELSE victim_occupation END,
    victim_education_level = CASE WHEN TRIM(victim_education_level) = '' THEN NULL ELSE victim_education_level END,
    victim_religion = CASE WHEN TRIM(victim_religion) = '' THEN NULL ELSE victim_religion END,
    victim_race_color = CASE WHEN TRIM(victim_race_color) = '' THEN NULL ELSE victim_race_color END,
    victim_ethnicity = CASE WHEN TRIM(victim_ethnicity) = '' THEN NULL ELSE victim_ethnicity END,
    victim_income_range = CASE WHEN TRIM(victim_income_range) = '' THEN NULL ELSE victim_income_range END,
    suspect_legal_nature = CASE WHEN TRIM(suspect_legal_nature) = '' THEN NULL ELSE suspect_legal_nature END,
    suspect_gender = CASE WHEN TRIM(suspect_gender) = '' THEN NULL ELSE suspect_gender END,
    suspect_sexual_orientation = CASE WHEN TRIM(suspect_sexual_orientation) = '' THEN NULL ELSE suspect_sexual_orientation END,
    suspect_age_group = CASE WHEN TRIM(suspect_age_group) = '' THEN NULL ELSE suspect_age_group END,
    suspect_nationality = CASE WHEN TRIM(suspect_nationality) = '' THEN NULL ELSE suspect_nationality END,
    suspect_birthplace = CASE WHEN TRIM(suspect_birthplace) = '' THEN NULL ELSE suspect_birthplace END,
    suspect_naturalized_state = CASE WHEN TRIM(suspect_naturalized_state) = '' THEN NULL ELSE suspect_naturalized_state END,
    suspect_naturalized_municipality = CASE WHEN TRIM(suspect_naturalized_municipality) = '' THEN NULL ELSE suspect_naturalized_municipality END,
    suspect_disability = CASE WHEN TRIM(suspect_disability) = '' THEN NULL ELSE suspect_disability END,
    suspect_rare_disease = CASE WHEN TRIM(suspect_rare_disease) = '' THEN NULL ELSE suspect_rare_disease END,
    suspect_disability_related_to_rare_disease = CASE WHEN TRIM(suspect_disability_related_to_rare_disease) = '' THEN NULL ELSE suspect_disability_related_to_rare_disease END,
    suspect_incarceration_status = CASE WHEN TRIM(suspect_incarceration_status) = '' THEN NULL ELSE suspect_incarceration_status END,
    suspect_country = CASE WHEN TRIM(suspect_country) = '' THEN NULL ELSE suspect_country END,
    suspect_state = CASE WHEN TRIM(suspect_state) = '' THEN NULL ELSE suspect_state END,
    suspect_municipality = CASE WHEN TRIM(suspect_municipality) = '' THEN NULL ELSE suspect_municipality END,
    suspect_occupation = CASE WHEN TRIM(suspect_occupation) = '' THEN NULL ELSE suspect_occupation END,
    suspect_education_level = CASE WHEN TRIM(suspect_education_level) = '' THEN NULL ELSE suspect_education_level END,
    suspect_religion = CASE WHEN TRIM(suspect_religion) = '' THEN NULL ELSE suspect_religion END,
    suspect_race_color = CASE WHEN TRIM(suspect_race_color) = '' THEN NULL ELSE suspect_race_color END,
    suspect_ethnicity = CASE WHEN TRIM(suspect_ethnicity) = '' THEN NULL ELSE suspect_ethnicity END,
    suspect_income_range = CASE WHEN TRIM(suspect_income_range) = '' THEN NULL ELSE suspect_income_range END,
    suspect_organization_relationship = CASE WHEN TRIM(suspect_organization_relationship) = '' THEN NULL ELSE suspect_organization_relationship END,
    suspect_business_sector = CASE WHEN TRIM(suspect_business_sector) = '' THEN NULL ELSE suspect_business_sector END,
    suspect_ethnicity_details = CASE WHEN TRIM(suspect_ethnicity_details) = '' THEN NULL ELSE suspect_ethnicity_details END,
    victim_ethnicity_details = CASE WHEN TRIM(victim_ethnicity_details) = '' THEN NULL ELSE victim_ethnicity_details END,
    violation = CASE WHEN TRIM(violation) = '' THEN NULL ELSE violation END
WHERE
    (service_channel IS NOT NULL AND TRIM(service_channel) = '') OR
    (emergency_status IS NOT NULL AND TRIM(emergency_status) = '') OR
    (reporter_type IS NOT NULL AND TRIM(reporter_type) = '') OR
    (violation_setting IS NOT NULL AND TRIM(violation_setting) = '') OR
    (country IS NOT NULL AND TRIM(country) = '') OR
    (state IS NOT NULL AND TRIM(state) = '') OR
    (municipality IS NOT NULL AND TRIM(municipality) = '') OR
    (frequency IS NOT NULL AND TRIM(frequency) = '') OR
    (violation_start_period IS NOT NULL AND TRIM(violation_start_period) = '') OR
    (vulnerable_group IS NOT NULL AND TRIM(vulnerable_group) = '') OR
    (motivation IS NOT NULL AND TRIM(motivation) = '') OR
    (victim_suspect_relationship IS NOT NULL AND TRIM(victim_suspect_relationship) = '') OR
    (victim_entity_type IS NOT NULL AND TRIM(victim_entity_type) = '') OR
    (victim_gender IS NOT NULL AND TRIM(victim_gender) = '') OR
    (victim_sexual_orientation IS NOT NULL AND TRIM(victim_sexual_orientation) = '') OR
    (victim_age_group IS NOT NULL AND TRIM(victim_age_group) = '') OR
    (victim_nationality IS NOT NULL AND TRIM(victim_nationality) = '') OR
    (victim_birthplace IS NOT NULL AND TRIM(victim_birthplace) = '') OR
    (victim_naturalized_state IS NOT NULL AND TRIM(victim_naturalized_state) = '') OR
    (victim_naturalized_municipality IS NOT NULL AND TRIM(victim_naturalized_municipality) = '') OR
    (victim_disability IS NOT NULL AND TRIM(victim_disability) = '') OR
    (victim_rare_disease IS NOT NULL AND TRIM(victim_rare_disease) = '') OR
    (victim_disability_related_to_rare_disease IS NOT NULL AND TRIM(victim_disability_related_to_rare_disease) = '') OR
    (victim_incarceration_status IS NOT NULL AND TRIM(victim_incarceration_status) = '') OR
    (victim_country IS NOT NULL AND TRIM(victim_country) = '') OR
    (victim_state IS NOT NULL AND TRIM(victim_state) = '') OR
    (victim_municipality IS NOT NULL AND TRIM(victim_municipality) = '') OR
    (victim_occupation IS NOT NULL AND TRIM(victim_occupation) = '') OR
    (victim_education_level IS NOT NULL AND TRIM(victim_education_level) = '') OR
    (victim_religion IS NOT NULL AND TRIM(victim_religion) = '') OR
    (victim_race_color IS NOT NULL AND TRIM(victim_race_color) = '') OR
    (victim_ethnicity IS NOT NULL AND TRIM(victim_ethnicity) = '') OR
    (victim_income_range IS NOT NULL AND TRIM(victim_income_range) = '') OR
    (suspect_legal_nature IS NOT NULL AND TRIM(suspect_legal_nature) = '') OR
    (suspect_gender IS NOT NULL AND TRIM(suspect_gender) = '') OR
    (suspect_sexual_orientation IS NOT NULL AND TRIM(suspect_sexual_orientation) = '') OR
    (suspect_age_group IS NOT NULL AND TRIM(suspect_age_group) = '') OR
    (suspect_nationality IS NOT NULL AND TRIM(suspect_nationality) = '') OR
    (suspect_birthplace IS NOT NULL AND TRIM(suspect_birthplace) = '') OR
    (suspect_naturalized_state IS NOT NULL AND TRIM(suspect_naturalized_state) = '') OR
    (suspect_naturalized_municipality IS NOT NULL AND TRIM(suspect_naturalized_municipality) = '') OR
    (suspect_disability IS NOT NULL AND TRIM(suspect_disability) = '') OR
    (suspect_rare_disease IS NOT NULL AND TRIM(suspect_rare_disease) = '') OR
    (suspect_disability_related_to_rare_disease IS NOT NULL AND TRIM(suspect_disability_related_to_rare_disease) = '') OR
    (suspect_incarceration_status IS NOT NULL AND TRIM(suspect_incarceration_status) = '') OR
    (suspect_country IS NOT NULL AND TRIM(suspect_country) = '') OR
    (suspect_state IS NOT NULL AND TRIM(suspect_state) = '') OR
    (suspect_municipality IS NOT NULL AND TRIM(suspect_municipality) = '') OR
    (suspect_occupation IS NOT NULL AND TRIM(suspect_occupation) = '') OR
    (suspect_education_level IS NOT NULL AND TRIM(suspect_education_level) = '') OR
    (suspect_religion IS NOT NULL AND TRIM(suspect_religion) = '') OR
    (suspect_race_color IS NOT NULL AND TRIM(suspect_race_color) = '') OR
    (suspect_ethnicity IS NOT NULL AND TRIM(suspect_ethnicity) = '') OR
    (suspect_income_range IS NOT NULL AND TRIM(suspect_income_range) = '') OR
    (suspect_organization_relationship IS NOT NULL AND TRIM(suspect_organization_relationship) = '') OR
    (suspect_business_sector IS NOT NULL AND TRIM(suspect_business_sector) = '') OR
    (suspect_ethnicity_details IS NOT NULL AND TRIM(suspect_ethnicity_details) = '') OR
    (victim_ethnicity_details IS NOT NULL AND TRIM(victim_ethnicity_details) = '') OR
    (violation IS NOT NULL AND TRIM(violation) = '');

CREATE INDEX idx_disque100_source_hash ON public.disque100_reports (source_hash);
CREATE INDEX idx_disque100_registered_at ON public.disque100_reports (registered_at);
CREATE INDEX idx_disque100_state_hash ON public.disque100_reports (state, source_hash);
CREATE INDEX idx_disque100_vulnerable_hash ON public.disque100_reports (vulnerable_group, source_hash);
CREATE INDEX idx_disque100_violation_hash ON public.disque100_reports (violation, source_hash);

DO $$
DECLARE
  v_count bigint;
  v_distinct_hash bigint;
  v_min_date timestamp;
  v_max_date timestamp;
  v_invalid_hashes bigint;
  v_null_dates bigint;
  v_non_positive_victims bigint;
  v_table_count int;
BEGIN
  SELECT count(*), count(distinct source_hash), min(registered_at), max(registered_at),
         count(*) filter (where source_hash is null or source_hash !~ '^[0-9A-F]{64}$'),
         count(*) filter (where registered_at is null),
         count(*) filter (where victim_count is null or victim_count <= 0)
  INTO v_count, v_distinct_hash, v_min_date, v_max_date, v_invalid_hashes, v_null_dates, v_non_positive_victims
  FROM public.disque100_reports;

  IF v_count != 2825614 THEN RAISE EXCEPTION 'Invalid count: %', v_count; END IF;
  IF v_distinct_hash != 371117 THEN RAISE EXCEPTION 'Invalid distinct hashes: %', v_distinct_hash; END IF;
  IF v_min_date != '2026-01-01 00:02:44.687' THEN RAISE EXCEPTION 'Invalid min date: %', v_min_date; END IF;
  IF v_max_date != '2026-06-30 23:59:03.497' THEN RAISE EXCEPTION 'Invalid max date: %', v_max_date; END IF;
  IF v_invalid_hashes > 0 THEN RAISE EXCEPTION 'Invalid hashes found'; END IF;
  IF v_null_dates > 0 THEN RAISE EXCEPTION 'Null dates found'; END IF;
  IF v_non_positive_victims > 0 THEN RAISE EXCEPTION 'Non positive victims found'; END IF;

  SELECT count(*) INTO v_table_count FROM pg_tables WHERE schemaname = 'public';
  IF v_table_count != 1 THEN RAISE EXCEPTION 'Expected exactly 1 public table, got %', v_table_count; END IF;
END $$;

ANALYZE public.disque100_reports;

COMMIT;
