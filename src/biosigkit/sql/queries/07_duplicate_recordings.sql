-- Recordings that hold the same samples under different subject / session names.
-- Recordings registered with a content hash (biosigkit.duplicates.signal_hash) are grouped by it;
-- a hash shared by two or more recordings means one is a copy of the other, which leaks the test
-- subject into the training set of a leave-one-subject-out split.
-- Shows: HAVING, GROUP_CONCAT over an ordered subquery, ignoring NULLs on purpose.
SELECT SUBSTR(content_sha256, 1, 12) AS hash,
       COUNT(*)                      AS n_recordings,
       GROUP_CONCAT(name, ', ')      AS recordings
FROM (
    SELECT content_sha256, subject_id || '/' || session AS name
    FROM recordings
    WHERE content_sha256 IS NOT NULL
    ORDER BY subject_id, session
)
GROUP BY content_sha256
HAVING COUNT(*) > 1
ORDER BY n_recordings DESC, hash;
