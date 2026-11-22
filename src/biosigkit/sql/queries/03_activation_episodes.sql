-- Collapse consecutive labelled windows into "episodes" (one row per contraction).
-- Classic gaps-and-islands: for each recording, ROW_NUMBER over all windows minus ROW_NUMBER over
-- windows with the same label is constant inside an unbroken run and jumps between runs.
-- Needs window-level features (kind = 'window'); rest windows are skipped.
-- Shows: two ROW_NUMBER() windows, CTE chaining, GROUP BY on a derived key.
WITH ordered AS (
    SELECT s.recording_id,
           s.label,
           s.t_start,
           s.t_stop,
           ROW_NUMBER() OVER (PARTITION BY s.recording_id ORDER BY s.t_start)          AS rn_all,
           ROW_NUMBER() OVER (PARTITION BY s.recording_id, s.label ORDER BY s.t_start) AS rn_label
    FROM segments AS s
    WHERE s.kind = 'window'
),
islands AS (
    SELECT *, rn_all - rn_label AS island
    FROM ordered
)
SELECT r.subject_id,
       i.label,
       ROUND(MIN(i.t_start), 3)             AS episode_start,
       ROUND(MAX(i.t_stop), 3)              AS episode_stop,
       ROUND(MAX(i.t_stop) - MIN(i.t_start), 3) AS duration_s,
       COUNT(*)                             AS n_windows
FROM islands    AS i
JOIN recordings AS r ON r.recording_id = i.recording_id
WHERE i.label <> 'rest'
GROUP BY i.recording_id, i.label, i.island
ORDER BY r.subject_id, episode_start;
