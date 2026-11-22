-- Share of missing (NULL) feature values per feature and channel.
-- Shows: boolean arithmetic in aggregates (SUM(value IS NULL)) and a computed percentage.
SELECT f.feature,
       f.channel,
       COUNT(*)                                          AS n_values,
       SUM(f.value IS NULL)                              AS n_null,
       ROUND(100.0 * SUM(f.value IS NULL) / COUNT(*), 2) AS pct_null
FROM features AS f
GROUP BY f.feature, f.channel
ORDER BY pct_null DESC, f.feature, f.channel;
