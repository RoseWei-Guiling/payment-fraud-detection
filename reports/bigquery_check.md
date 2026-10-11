# Velocity features: BigQuery vs pandas

Table `fraud.transactions` in location EU, 590,540 rows, 0.03 GB processed.

**All features match: True**

| feature             |   max_abs_diff |   nan_mismatches | match   |
|:--------------------|---------------:|-----------------:|:--------|
| uid_txn_number      |              0 |                0 | True    |
| uid_n_1h            |              0 |                0 | True    |
| uid_n_24h           |              0 |                0 | True    |
| uid_n_7d            |              0 |                0 | True    |
| uid_amt_24h         |              0 |                0 | True    |
| uid_amt_mean_prior  |              0 |                0 | True    |
| amt_to_uid_mean     |              0 |                0 | True    |
| uid_secs_since_prev |              0 |                0 | True    |
| new_device_for_uid  |              0 |                0 | True    |
| new_email_for_uid   |              0 |                0 | True    |
