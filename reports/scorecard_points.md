# Scorecard points

600 points = odds of 50 genuine payments to 1 fraud; every 20 points doubles the odds. Higher total = safer. Each payment's score is the sum of its points across features.

| Variable             | Bin                                                                            |   Points |
|:---------------------|:-------------------------------------------------------------------------------|---------:|
| uid_n_7d             | (-inf, 0.5000)                                                                 |       69 |
| uid_n_7d             | [0.5000, 1.5000)                                                               |       59 |
| uid_n_7d             | [1.5000, 2.5000)                                                               |       52 |
| uid_n_7d             | [2.5000, 4.5000)                                                               |       45 |
| uid_n_7d             | [4.5000, inf)                                                                  |       37 |
| uid_n_7d             | Special                                                                        |       53 |
| uid_n_7d             | Missing                                                                        |       53 |
| R_email_provider     | <ArrowStringArray>                                                             |       55 |
|                      | [         'twc', 'servicios-ta',          'mac',     'scranton',               |          |
|                      |          'juno',   'suddenlink',   'rocketmail',          'gmx',               |          |
|                      |       'verizon',  'frontiernet',     'frontier',      'netzero',               |          |
|                      |    'embarqmail',           'sc',          'ptd',   'windstream',               |          |
|                      |           'cfl',  'centurylink',     'cableone',    'bellsouth',               |          |
|                      |           'att',            'q',   'roadrunner',          'web',               |          |
|                      |           'msn',    'sbcglobal',           'me',      'prodigy',               |          |
|                      |     'optonline',      'comcast',          'cox',    'anonymous',               |          |
|                      |          'live',          'aol',          'aim',        'ymail',               |          |
|                      |       'charter',        'yahoo']                                               |          |
|                      | Length: 38, dtype: str                                                         |          |
| R_email_provider     | <ArrowStringArray>                                                             |       41 |
|                      | ['hotmail']                                                                    |          |
|                      | Length: 1, dtype: str                                                          |          |
| R_email_provider     | <ArrowStringArray>                                                             |       34 |
|                      | ['earthlink', 'gmail', 'icloud', 'outlook', 'mail', 'protonmail']              |          |
|                      | Length: 6, dtype: str                                                          |          |
| R_email_provider     | Special                                                                        |       53 |
| R_email_provider     | Missing                                                                        |       53 |
| browser_family       | <ArrowStringArray>                                                             |       53 |
|                      | [     'line',  'palemoon',     'nokia', 'seamonkey', 'microsoft',   'maxthon', |          |
|                      |          'm',  'waterfox',        'lg',    'puffin',      'iron',      'silk', |          |
|                      |   'cyberfox',    'comodo',  'chromium',       'blu',       'aol',      'inco', |          |
|                      |         'ie',    'safari',      'edge',   'firefox']                           |          |
|                      | Length: 22, dtype: str                                                         |          |
| browser_family       | <ArrowStringArray>                                                             |       52 |
|                      | ['mobile']                                                                     |          |
|                      | Length: 1, dtype: str                                                          |          |
| browser_family       | <ArrowStringArray>                                                             |       52 |
|                      | [ 'chrome', 'samsung', 'generic',   'other',     'zte', 'android',   'opera',  |          |
|                      |  'mozilla',   'lanix']                                                         |          |
|                      | Length: 9, dtype: str                                                          |          |
| browser_family       | Special                                                                        |       53 |
| browser_family       | Missing                                                                        |       53 |
| uids_per_fingerprint | (-inf, 7.5000)                                                                 |       45 |
| uids_per_fingerprint | [7.5000, 266.5000)                                                             |       49 |
| uids_per_fingerprint | [266.5000, inf)                                                                |       53 |
| uids_per_fingerprint | Special                                                                        |       53 |
| uids_per_fingerprint | Missing                                                                        |       53 |
| uid_txn_number       | (-inf, 1.5000)                                                                 |       58 |
| uid_txn_number       | [1.5000, 2.5000)                                                               |       57 |
| uid_txn_number       | [2.5000, 3.5000)                                                               |       56 |
| uid_txn_number       | [3.5000, 5.5000)                                                               |       56 |
| uid_txn_number       | [5.5000, 10.5000)                                                              |       54 |
| uid_txn_number       | [10.5000, inf)                                                                 |       50 |
| uid_txn_number       | Special                                                                        |       53 |
| uid_txn_number       | Missing                                                                        |       53 |
| n_missing_d          | (-inf, 6.5000)                                                                 |       42 |
| n_missing_d          | [6.5000, 8.5000)                                                               |       55 |
| n_missing_d          | [8.5000, 9.5000)                                                               |       56 |
| n_missing_d          | [9.5000, 10.5000)                                                              |       58 |
| n_missing_d          | [10.5000, 11.5000)                                                             |       54 |
| n_missing_d          | [11.5000, inf)                                                                 |       52 |
| n_missing_d          | Special                                                                        |       53 |
| n_missing_d          | Missing                                                                        |       53 |
| amt_cents            | (-inf, 0.0005)                                                                 |       53 |
| amt_cents            | [0.0005, 0.5115)                                                               |       48 |
| amt_cents            | [0.5115, 0.9475)                                                               |       44 |
| amt_cents            | [0.9475, inf)                                                                  |       64 |
| amt_cents            | Special                                                                        |       53 |
| amt_cents            | Missing                                                                        |       53 |
| uid_secs_since_prev  | (-inf, 371.5000)                                                               |       46 |
| uid_secs_since_prev  | [371.5000, 2755.5000)                                                          |       51 |
| uid_secs_since_prev  | [2755.5000, 253779.5000)                                                       |       52 |
| uid_secs_since_prev  | [253779.5000, 765923.5000)                                                     |       60 |
| uid_secs_since_prev  | [765923.5000, 2375122.5000)                                                    |       67 |
| uid_secs_since_prev  | [2375122.5000, inf)                                                            |       70 |
| uid_secs_since_prev  | Special                                                                        |       53 |
| uid_secs_since_prev  | Missing                                                                        |       53 |
| card6                | <ArrowStringArray>                                                             |       60 |
|                      | ['charge card', 'debit or credit', 'debit']                                    |          |
|                      | Length: 3, dtype: str                                                          |          |
| card6                | <ArrowStringArray>                                                             |       40 |
|                      | ['credit']                                                                     |          |
|                      | Length: 1, dtype: str                                                          |          |
| card6                | Special                                                                        |       53 |
| card6                | Missing                                                                        |       53 |
| device_family_te     | (-inf, 0.0260)                                                                 |       54 |
| device_family_te     | [0.0260, 0.0265)                                                               |       54 |
| device_family_te     | [0.0265, 0.0376)                                                               |       55 |
| device_family_te     | [0.0376, 0.0621)                                                               |       50 |
| device_family_te     | [0.0621, inf)                                                                  |       47 |
| device_family_te     | Special                                                                        |       53 |
| device_family_te     | Missing                                                                        |       53 |
| TransactionAmt       | (-inf, 23.9145)                                                                |       47 |
| TransactionAmt       | [23.9145, 47.7400)                                                             |       53 |
| TransactionAmt       | [47.7400, 106.5360)                                                            |       55 |
| TransactionAmt       | [106.5360, 108.9850)                                                           |       63 |
| TransactionAmt       | [108.9850, 136.0330)                                                           |       54 |
| TransactionAmt       | [136.0330, inf)                                                                |       51 |
| TransactionAmt       | Special                                                                        |       53 |
| TransactionAmt       | Missing                                                                        |       53 |
