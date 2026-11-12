-- Draft only: not executed against a database.
-- Run in a selected project database after coordinating table ownership with A/B.
-- Large images and DEM pixels remain external; store source URIs and metadata.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS agri_c;

CREATE TABLE IF NOT EXISTS agri_c.weather_daily (
    site_id text NOT NULL,
    observation_date date NOT NULL,
    nasa_tmean_c double precision,
    nasa_tmax_c double precision,
    nasa_tmin_c double precision,
    nasa_rh_pct double precision CHECK (nasa_rh_pct BETWEEN 0 AND 100),
    nasa_precip_mm_day double precision CHECK (nasa_precip_mm_day >= 0),
    nasa_wind2m_ms double precision CHECK (nasa_wind2m_ms >= 0),
    nasa_solar_mj_m2_day double precision CHECK (nasa_solar_mj_m2_day >= 0),
    era5_sm_0_7_m3_m3_daily_mean double precision CHECK (era5_sm_0_7_m3_m3_daily_mean BETWEEN 0 AND 1),
    era5_sm_7_28_m3_m3_daily_mean double precision CHECK (era5_sm_7_28_m3_m3_daily_mean BETWEEN 0 AND 1),
    era5_temperature_c_daily_mean double precision,
    chirps_selected_precip_mm_day double precision CHECK (chirps_selected_precip_mm_day >= 0),
    source_metadata jsonb NOT NULL,
    PRIMARY KEY (site_id, observation_date)
);
CREATE TABLE IF NOT EXISTS agri_c.risk_daily (
    site_id text NOT NULL, observation_date date NOT NULL,
    event_type text NOT NULL, rule_version text NOT NULL,
    screen_level smallint CHECK (screen_level IN (0,1,2)),
    status text NOT NULL CHECK (status IN ('not_triggered','candidate','insufficient_data')),
    indicator_values jsonb NOT NULL, thresholds jsonb NOT NULL,
    missing_inputs jsonb NOT NULL, phase text NOT NULL,
    PRIMARY KEY (site_id, observation_date, event_type, rule_version),
    CHECK ((status='insufficient_data')=(screen_level IS NULL)),
    CHECK (status<>'candidate' OR screen_level IN (1,2)),
    CHECK (status<>'not_triggered' OR screen_level=0),
    FOREIGN KEY (site_id, observation_date) REFERENCES agri_c.weather_daily
);
CREATE TABLE IF NOT EXISTS agri_c.plot_reference (
    plot_id text PRIMARY KEY, site_id text NOT NULL,
    geom geometry(MultiPolygon,4326) NOT NULL,
    crs_verified boolean NOT NULL DEFAULT false,
    boundary_verified boolean NOT NULL DEFAULT false,
    boundary_source text NOT NULL,
    terrain_drainage_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_synthetic boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS plot_reference_geom_idx ON agri_c.plot_reference USING gist (geom);
CREATE TABLE IF NOT EXISTS agri_c.image_observation (
    observation_id text PRIMARY KEY,
    plot_id text REFERENCES agri_c.plot_reference,
    captured_at timestamptz,
    source_reference text NOT NULL,
    a_response jsonb NOT NULL,
    association_metadata jsonb NOT NULL,
    acceptance_status text NOT NULL CHECK (acceptance_status IN ('accepted','quarantined')),
    quarantine_reasons jsonb NOT NULL DEFAULT '[]'::jsonb
);
CREATE TABLE IF NOT EXISTS agri_c.inspection_request (
    request_id text PRIMARY KEY,
    plot_id text NOT NULL REFERENCES agri_c.plot_reference,
    as_of_date date NOT NULL,
    priority text NOT NULL CHECK (priority IN ('P1','P2')),
    evidence jsonb NOT NULL,
    execution_status text NOT NULL DEFAULT 'awaiting_B_route_and_site_checks',
    is_synthetic boolean NOT NULL DEFAULT false
);
