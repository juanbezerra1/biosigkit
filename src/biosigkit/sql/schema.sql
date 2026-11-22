-- biosigkit schema (SQLite).
-- Features live in a long table (one row per segment x channel x feature) so new features or
-- channels never require a schema change; see queries/ for analytical examples.

CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS recordings (
    recording_id INTEGER PRIMARY KEY,
    subject_id   TEXT NOT NULL REFERENCES subjects (subject_id),
    session      TEXT NOT NULL DEFAULT 'S1',
    fs_hz        REAL,
    preprocess   TEXT,              -- JSON: provenance of the preprocessing chain
    UNIQUE (subject_id, session)
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id   INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings (recording_id),
    kind         TEXT NOT NULL CHECK (kind IN ('window', 'trial')),
    t_start      REAL NOT NULL,
    t_stop       REAL NOT NULL,
    label        TEXT,
    CHECK (t_stop > t_start)
);

CREATE TABLE IF NOT EXISTS features (
    segment_id INTEGER NOT NULL REFERENCES segments (segment_id),
    channel    TEXT NOT NULL,
    feature    TEXT NOT NULL,
    value      REAL,                -- NULL when the feature could not be computed
    PRIMARY KEY (segment_id, channel, feature)
);

CREATE INDEX IF NOT EXISTS idx_segments_recording ON segments (recording_id, kind, t_start);
CREATE INDEX IF NOT EXISTS idx_features_feature ON features (feature, channel);
