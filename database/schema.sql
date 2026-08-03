-- ===========================================================================
-- Dynamic Pricing Intelligence Platform - Star Schema
-- PostgreSQL 16
--
-- Run with:
--     python src/load_to_postgres.py     (executes this file, then loads data)
--
-- Design notes are inline. The two that matter most:
--   1. dim_date is ROLE-PLAYING: the booking fact joins to it twice, once as
--      the stay date and once as the booking date.
--   2. Fact grain is (hotel, room_type, stay_date, booking_date) - the
--      booking curve, not a daily sales row.
-- ===========================================================================

DROP SCHEMA IF EXISTS pricing CASCADE;
CREATE SCHEMA pricing;
SET search_path TO pricing;


-- ===========================================================================
-- DIMENSIONS
-- ===========================================================================

-- Natural keys ('HTL001') are used as primary keys rather than generated
-- surrogate integers. For a static, small dimension with stable business
-- keys this is readable and perfectly defensible. A production warehouse
-- tracking history (Slowly Changing Dimensions) would use surrogates.
CREATE TABLE dim_hotel (
    hotel_id        VARCHAR(10)  PRIMARY KEY,
    hotel_name      VARCHAR(60)  NOT NULL,
    city            VARCHAR(40)  NOT NULL,
    star_rating     SMALLINT     NOT NULL CHECK (star_rating BETWEEN 1 AND 5),
    total_rooms     INTEGER      NOT NULL CHECK (total_rooms > 0),
    leisure_share   NUMERIC(4,3) NOT NULL CHECK (leisure_share BETWEEN 0 AND 1),
    base_rate       NUMERIC(8,2) NOT NULL CHECK (base_rate > 0),
    segment         VARCHAR(20)  NOT NULL
);

CREATE TABLE dim_room_type (
    room_type_id    VARCHAR(10)  PRIMARY KEY,
    room_type_name  VARCHAR(30)  NOT NULL,
    rate_index      NUMERIC(5,3) NOT NULL CHECK (rate_index > 0),
    room_share      NUMERIC(4,3) NOT NULL
);

-- date_id is a "smart key" in YYYYMMDD form. Standard warehouse practice:
-- human-readable in query results, sorts chronologically, and joins as a
-- compact 4-byte integer.
CREATE TABLE dim_date (
    date_id             INTEGER      PRIMARY KEY,
    calendar_date       DATE         NOT NULL UNIQUE,
    is_stay_date        BOOLEAN      NOT NULL,
    year                SMALLINT     NOT NULL,
    quarter             SMALLINT     NOT NULL,
    month               SMALLINT     NOT NULL,
    week_of_year        SMALLINT     NOT NULL,
    day_of_week         SMALLINT     NOT NULL,
    day_name            VARCHAR(10)  NOT NULL,
    is_weekend          BOOLEAN      NOT NULL,
    dow_mult_business   NUMERIC(6,4) NOT NULL,
    dow_mult_leisure    NUMERIC(6,4) NOT NULL,
    season_mult         NUMERIC(6,4) NOT NULL,
    is_holiday          BOOLEAN      NOT NULL,
    holiday_name        VARCHAR(60),
    days_until_holiday  SMALLINT     NOT NULL,
    holiday_mult        NUMERIC(6,4) NOT NULL
);

-- Physical inventory per hotel/room type. This is what turns rooms sold
-- into occupancy, and it is the hard constraint the optimizer must respect.
CREATE TABLE dim_capacity (
    hotel_id         VARCHAR(10)  NOT NULL REFERENCES dim_hotel(hotel_id),
    room_type_id     VARCHAR(10)  NOT NULL REFERENCES dim_room_type(room_type_id),
    rooms_available  INTEGER      NOT NULL CHECK (rooms_available > 0),
    base_rate        NUMERIC(8,2) NOT NULL CHECK (base_rate > 0),
    PRIMARY KEY (hotel_id, room_type_id)
);

-- Events are city-scoped and span a date RANGE, so they cannot live in
-- dim_date without breaking its one-row-per-day grain.
CREATE TABLE dim_event (
    event_id           VARCHAR(10)  PRIMARY KEY,
    city               VARCHAR(40)  NOT NULL,
    event_name         VARCHAR(60)  NOT NULL,
    event_start        DATE         NOT NULL,
    event_end          DATE         NOT NULL,
    duration_nights    SMALLINT     NOT NULL,
    demand_multiplier  NUMERIC(6,3) NOT NULL,
    CHECK (event_end >= event_start)
);


-- ===========================================================================
-- FACTS
-- ===========================================================================

-- GRAIN: one row per (hotel, room_type, stay_date, booking_date).
-- "How many rooms were picked up for this future night, on this day,
--  at the rate we were showing."
CREATE TABLE fact_booking_pickup (
    stay_date_id     INTEGER      NOT NULL REFERENCES dim_date(date_id),
    booking_date_id  INTEGER      NOT NULL REFERENCES dim_date(date_id),
    hotel_id         VARCHAR(10)  NOT NULL REFERENCES dim_hotel(hotel_id),
    room_type_id     VARCHAR(10)  NOT NULL REFERENCES dim_room_type(room_type_id),
    lead_time        SMALLINT     NOT NULL CHECK (lead_time >= 0),
    rate             NUMERIC(9,2) NOT NULL CHECK (rate > 0),
    rooms_booked     SMALLINT     NOT NULL CHECK (rooms_booked >= 0),
    room_revenue     NUMERIC(12,2) NOT NULL CHECK (room_revenue >= 0),
    rooms_on_books   SMALLINT     NOT NULL CHECK (rooms_on_books >= 0),
    is_available     BOOLEAN      NOT NULL,
    is_rate_test     BOOLEAN      NOT NULL,
    PRIMARY KEY (hotel_id, room_type_id, stay_date_id, booking_date_id)
);

-- GRAIN: one row per (hotel, competitor, stay_date, booking_date).
-- Separate table because competitor rates are quoted at HOTEL level -
-- forcing them into the booking fact would silently duplicate them across
-- three room types and corrupt any average.
CREATE TABLE fact_competitor_rate (
    stay_date_id     INTEGER      NOT NULL REFERENCES dim_date(date_id),
    booking_date_id  INTEGER      NOT NULL REFERENCES dim_date(date_id),
    hotel_id         VARCHAR(10)  NOT NULL REFERENCES dim_hotel(hotel_id),
    competitor_name  VARCHAR(20)  NOT NULL,
    lead_time        SMALLINT     NOT NULL CHECK (lead_time >= 0),
    competitor_rate  NUMERIC(9,2) NOT NULL CHECK (competitor_rate > 0),
    PRIMARY KEY (hotel_id, competitor_name, stay_date_id, booking_date_id)
);


-- ===========================================================================
-- ROLE-PLAYING DIMENSION VIEWS
-- ===========================================================================
-- dim_date is joined TWICE by the booking fact. Aliasing it through views
-- makes queries self-documenting and prevents the classic mistake of
-- joining the wrong date and silently analysing the wrong thing.

CREATE VIEW dim_stay_date AS
SELECT date_id            AS stay_date_id,
       calendar_date      AS stay_date,
       year               AS stay_year,
       quarter            AS stay_quarter,
       month              AS stay_month,
       day_name           AS stay_day_name,
       is_weekend         AS stay_is_weekend,
       is_holiday         AS stay_is_holiday,
       holiday_name       AS stay_holiday_name,
       days_until_holiday AS stay_days_until_holiday,
       season_mult, holiday_mult, dow_mult_business, dow_mult_leisure
FROM dim_date;

CREATE VIEW dim_booking_date AS
SELECT date_id       AS booking_date_id,
       calendar_date AS booking_date,
       year          AS booking_year,
       month         AS booking_month,
       day_name      AS booking_day_name,
       is_weekend    AS booking_is_weekend
FROM dim_date;


-- ===========================================================================
-- INDEXES
-- ===========================================================================
-- The primary keys already cover lookups that lead with hotel_id. These
-- support the access patterns analysis actually uses: slicing by stay date,
-- and filtering to the randomized rate tests.

CREATE INDEX idx_booking_stay_date   ON fact_booking_pickup (stay_date_id);
CREATE INDEX idx_booking_booking_date ON fact_booking_pickup (booking_date_id);
CREATE INDEX idx_booking_lead_time   ON fact_booking_pickup (lead_time);
CREATE INDEX idx_booking_rate_test   ON fact_booking_pickup (is_rate_test)
    WHERE is_rate_test;                       -- partial index: only ~12% of rows
CREATE INDEX idx_comp_stay_date      ON fact_competitor_rate (stay_date_id);
CREATE INDEX idx_event_city_dates    ON dim_event (city, event_start, event_end);


-- ===========================================================================
-- ANALYSIS VIEW
-- ===========================================================================
-- Flattens the star into the shape Steps 3-5 actually want, including
-- occupancy and the average competitor rate. Written as a view so the
-- join logic lives in one place instead of being re-typed in every notebook.

CREATE VIEW vw_booking_analysis AS
SELECT
    f.hotel_id,
    h.hotel_name,
    h.city,
    h.segment,
    f.room_type_id,
    rt.room_type_name,
    sd.stay_date,
    bd.booking_date,
    f.lead_time,
    f.rate,
    f.rooms_booked,
    f.room_revenue,
    f.rooms_on_books,
    c.rooms_available,
    ROUND(f.rooms_on_books::NUMERIC / c.rooms_available, 4) AS occupancy_on_books,
    f.is_available,
    f.is_rate_test,
    sd.stay_day_name,
    sd.stay_is_weekend,
    sd.stay_is_holiday,
    sd.stay_days_until_holiday,
    comp.avg_competitor_rate,
    ROUND(f.rate - comp.avg_competitor_rate, 2) AS rate_vs_comp
FROM fact_booking_pickup f
JOIN dim_hotel        h  ON h.hotel_id = f.hotel_id
JOIN dim_room_type    rt ON rt.room_type_id = f.room_type_id
JOIN dim_capacity     c  ON c.hotel_id = f.hotel_id
                        AND c.room_type_id = f.room_type_id
JOIN dim_stay_date    sd ON sd.stay_date_id = f.stay_date_id
JOIN dim_booking_date bd ON bd.booking_date_id = f.booking_date_id
LEFT JOIN (
    SELECT hotel_id, stay_date_id, booking_date_id,
           ROUND(AVG(competitor_rate), 2) AS avg_competitor_rate
    FROM fact_competitor_rate
    GROUP BY hotel_id, stay_date_id, booking_date_id
) comp ON comp.hotel_id = f.hotel_id
      AND comp.stay_date_id = f.stay_date_id
      AND comp.booking_date_id = f.booking_date_id;
