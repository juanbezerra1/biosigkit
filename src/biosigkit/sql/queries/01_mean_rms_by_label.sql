-- Mean, min and max RMS per subject, force level and channel (trial-level features).
-- Shows: JOIN across three tables, GROUP BY, aggregate functions.
SELECT r.subject_id,
       s.label,
       f.channel,
       COUNT(*)               AS n_trials,
       ROUND(AVG(f.value), 4) AS mean_rms,
       ROUND(MIN(f.value), 4) AS min_rms,
       ROUND(MAX(f.value), 4) AS max_rms
FROM features   AS f
JOIN segments   AS s ON s.segment_id   = f.segment_id
JOIN recordings AS r ON r.recording_id = s.recording_id
WHERE s.kind = 'trial'
  AND f.feature = 'rms'
GROUP BY r.subject_id, s.label, f.channel
ORDER BY r.subject_id, s.label, f.channel;
