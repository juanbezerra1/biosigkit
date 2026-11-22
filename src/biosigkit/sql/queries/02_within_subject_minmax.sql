-- Within-subject min-max normalisation and percentile rank of RMS, entirely in SQL.
-- The same idea as biosigkit.ml.within_group_zscore: remove between-subject scale differences.
-- Shows: CTE, window functions (MIN/MAX/PERCENT_RANK OVER), a named WINDOW, NULLIF to avoid /0.
WITH rms AS (
    SELECT r.subject_id, s.segment_id, s.label, f.channel, f.value AS rms
    FROM features   AS f
    JOIN segments   AS s ON s.segment_id   = f.segment_id
    JOIN recordings AS r ON r.recording_id = s.recording_id
    WHERE s.kind = 'trial'
      AND f.feature = 'rms'
)
SELECT subject_id,
       segment_id,
       label,
       channel,
       ROUND(rms, 4) AS rms,
       ROUND((rms - MIN(rms) OVER by_subject)
             / NULLIF(MAX(rms) OVER by_subject - MIN(rms) OVER by_subject, 0), 4) AS rms_minmax,
       ROUND(PERCENT_RANK() OVER (PARTITION BY subject_id, channel ORDER BY rms), 4) AS rms_pct_rank
FROM rms
WINDOW by_subject AS (PARTITION BY subject_id, channel)
ORDER BY subject_id, channel, segment_id;
