-- Customer velocity features, point-in-time: each row only sees the customer's EARLIER
-- transactions, so nothing from the future leaks into the features.
--
-- Written in standard SQL that runs unchanged on BigQuery (production) and DuckDB
-- (the automated tests). {table} is replaced by the calling script.
--
-- Time windows use TransactionDT in seconds:
--   RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING  =  the hour before this transaction

WITH w AS (
  SELECT
    TransactionID,
    uid_complete,
    TransactionAmt,

    COUNT(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                   RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)   AS n_1h,
    COUNT(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                   RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING)  AS n_24h,
    COUNT(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                   RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING) AS n_7d,
    SUM(TransactionAmt) OVER (PARTITION BY uid ORDER BY TransactionDT
                   RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING)  AS amt_24h,

    AVG(TransactionAmt) OVER (PARTITION BY uid ORDER BY TransactionDT, TransactionID
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS amt_mean_prior,
    TransactionDT - LAG(TransactionDT) OVER (PARTITION BY uid ORDER BY TransactionDT, TransactionID)
                                                                   AS secs_since_prev,
    ROW_NUMBER() OVER (PARTITION BY uid ORDER BY TransactionDT, TransactionID) AS txn_number,

    CASE WHEN DeviceInfo IS NOT NULL THEN
      ROW_NUMBER() OVER (PARTITION BY uid, DeviceInfo ORDER BY TransactionDT, TransactionID) END
                                                                   AS device_seen_n,
    CASE WHEN P_emaildomain IS NOT NULL THEN
      ROW_NUMBER() OVER (PARTITION BY uid, P_emaildomain ORDER BY TransactionDT, TransactionID) END
                                                                   AS email_seen_n
  FROM {table}
)

SELECT
  TransactionID,
  -- Features only exist when we can identify the customer.
  CASE WHEN uid_complete = 1 THEN txn_number END                     AS uid_txn_number,
  CASE WHEN uid_complete = 1 THEN n_1h END                           AS uid_n_1h,
  CASE WHEN uid_complete = 1 THEN n_24h END                          AS uid_n_24h,
  CASE WHEN uid_complete = 1 THEN n_7d END                           AS uid_n_7d,
  CASE WHEN uid_complete = 1 THEN COALESCE(amt_24h, 0) END           AS uid_amt_24h,
  CASE WHEN uid_complete = 1 THEN amt_mean_prior END                 AS uid_amt_mean_prior,
  CASE WHEN uid_complete = 1 THEN TransactionAmt / NULLIF(amt_mean_prior, 0) END
                                                                     AS amt_to_uid_mean,
  CASE WHEN uid_complete = 1 THEN secs_since_prev END                AS uid_secs_since_prev,
  -- 1 = this customer has history and this device / email is new for them.
  CASE WHEN uid_complete = 1 AND txn_number > 1 AND device_seen_n IS NOT NULL
       THEN CASE WHEN device_seen_n = 1 THEN 1 ELSE 0 END END        AS new_device_for_uid,
  CASE WHEN uid_complete = 1 AND txn_number > 1 AND email_seen_n IS NOT NULL
       THEN CASE WHEN email_seen_n = 1 THEN 1 ELSE 0 END END         AS new_email_for_uid
FROM w
ORDER BY TransactionID
