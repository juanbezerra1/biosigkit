-- Trial-balance report per subject. Flags subjects whose rarest level has fewer than half as many
-- trials as their most frequent level (a common source of biased classifiers).
-- Shows: CTE, GROUP BY on an aggregate of an aggregate, CASE expression.
WITH per_label AS (
    SELECT r.subject_id, s.label, COUNT(*) AS n_trials
    FROM segments   AS s
    JOIN recordings AS r ON r.recording_id = s.recording_id
    WHERE s.kind = 'trial'
    GROUP BY r.subject_id, s.label
)
SELECT subject_id,
       SUM(n_trials) AS total_trials,
       COUNT(*)      AS n_levels,
       MIN(n_trials) AS min_per_level,
       MAX(n_trials) AS max_per_level,
       CASE WHEN MIN(n_trials) < 0.5 * MAX(n_trials) THEN 'IMBALANCED' ELSE 'ok' END AS balance_flag
FROM per_label
GROUP BY subject_id
ORDER BY subject_id;
