-- The three strongest trials (by RMS) of every subject and channel.
-- Shows: ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) and filtering on its result via a CTE
-- (window functions cannot be used directly in WHERE).
WITH ranked AS (
    SELECT r.subject_id,
           f.channel,
           s.segment_id,
           s.label,
           f.value AS rms,
           ROW_NUMBER() OVER (
               PARTITION BY r.subject_id, f.channel
               ORDER BY f.value DESC
           ) AS rank_in_subject
    FROM features   AS f
    JOIN segments   AS s ON s.segment_id   = f.segment_id
    JOIN recordings AS r ON r.recording_id = s.recording_id
    WHERE s.kind = 'trial'
      AND f.feature = 'rms'
)
SELECT subject_id, channel, rank_in_subject, segment_id, label, ROUND(rms, 4) AS rms
FROM ranked
WHERE rank_in_subject <= 3
ORDER BY subject_id, channel, rank_in_subject;
