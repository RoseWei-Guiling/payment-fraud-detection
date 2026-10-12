# Scorecard points

600 points = odds of 50 genuine payments to 1 fraud; every 20 points doubles the odds. Higher total = safer. Each payment's score is the sum of its points across features.

| Variable                 | Bin                                                                            |   Points |
|:-------------------------|:-------------------------------------------------------------------------------|---------:|
| uid_n_7d                 | (-inf, 0.5000)                                                                 |       36 |
| uid_n_7d                 | [0.5000, 1.5000)                                                               |       25 |
| uid_n_7d                 | [1.5000, 2.5000)                                                               |       19 |
| uid_n_7d                 | [2.5000, 4.5000)                                                               |       11 |
| uid_n_7d                 | [4.5000, inf)                                                                  |        3 |
| uid_n_7d                 | Special                                                                        |       19 |
| uid_n_7d                 | Missing                                                                        |       19 |
| card_type_and_product_te | (-inf, 0.0141)                                                                 |       34 |
| card_type_and_product_te | [0.0141, 0.0239)                                                               |       30 |
| card_type_and_product_te | [0.0239, 0.0435)                                                               |       16 |
| card_type_and_product_te | [0.0435, 0.0616)                                                               |       15 |
| card_type_and_product_te | [0.0616, 0.0879)                                                               |        8 |
| card_type_and_product_te | [0.0879, inf)                                                                  |       -5 |
| card_type_and_product_te | Special                                                                        |       19 |
| card_type_and_product_te | Missing                                                                        |       19 |
| uid_amt_24h              | (-inf, 0.2120)                                                                 |       27 |
| uid_amt_24h              | [0.2120, 75.6330)                                                              |       21 |
| uid_amt_24h              | [75.6330, 269.9950)                                                            |       17 |
| uid_amt_24h              | [269.9950, inf)                                                                |       13 |
| uid_amt_24h              | Special                                                                        |       19 |
| uid_amt_24h              | Missing                                                                        |       19 |
| uid_n_24h                | (-inf, 0.5000)                                                                 |        6 |
| uid_n_24h                | [0.5000, 1.5000)                                                               |       21 |
| uid_n_24h                | [1.5000, inf)                                                                  |       26 |
| uid_n_24h                | Special                                                                        |       19 |
| uid_n_24h                | Missing                                                                        |       19 |
| R_emaildomain_te         | (-inf, 0.0212)                                                                 |       30 |
| R_emaildomain_te         | [0.0212, 0.0218)                                                               |       33 |
| R_emaildomain_te         | [0.0218, 0.0282)                                                               |       33 |
| R_emaildomain_te         | [0.0282, 0.0728)                                                               |       18 |
| R_emaildomain_te         | [0.0728, 0.1006)                                                               |        1 |
| R_emaildomain_te         | [0.1006, inf)                                                                  |       -9 |
| R_emaildomain_te         | Special                                                                        |       19 |
| R_emaildomain_te         | Missing                                                                        |       19 |
| R_email_provider         | <ArrowStringArray>                                                             |       19 |
|                          | [         'twc', 'servicios-ta',          'mac',     'scranton',               |          |
|                          |          'juno',   'suddenlink',   'rocketmail',          'gmx',               |          |
|                          |       'verizon',  'frontiernet',     'frontier',      'netzero',               |          |
|                          |    'embarqmail',           'sc',          'ptd',   'windstream',               |          |
|                          |           'cfl',  'centurylink',     'cableone',    'bellsouth',               |          |
|                          |           'att',            'q',   'roadrunner',          'web',               |          |
|                          |           'msn',    'sbcglobal',           'me',      'prodigy',               |          |
|                          |     'optonline',      'comcast',          'cox',    'anonymous',               |          |
|                          |          'live',          'aol',          'aim',        'ymail',               |          |
|                          |       'charter',        'yahoo']                                               |          |
|                          | Length: 38, dtype: str                                                         |          |
| R_email_provider         | <ArrowStringArray>                                                             |       18 |
|                          | ['hotmail']                                                                    |          |
|                          | Length: 1, dtype: str                                                          |          |
| R_email_provider         | <ArrowStringArray>                                                             |       17 |
|                          | ['earthlink', 'gmail', 'icloud', 'outlook', 'mail', 'protonmail']              |          |
|                          | Length: 6, dtype: str                                                          |          |
| R_email_provider         | Special                                                                        |       19 |
| R_email_provider         | Missing                                                                        |       19 |
| ProductCD                | <ArrowStringArray>                                                             |        3 |
|                          | ['W']                                                                          |          |
|                          | Length: 1, dtype: str                                                          |          |
| ProductCD                | <ArrowStringArray>                                                             |       19 |
|                          | ['R']                                                                          |          |
|                          | Length: 1, dtype: str                                                          |          |
| ProductCD                | <ArrowStringArray>                                                             |       29 |
|                          | ['H', 'S']                                                                     |          |
|                          | Length: 2, dtype: str                                                          |          |
| ProductCD                | <ArrowStringArray>                                                             |       56 |
|                          | ['C']                                                                          |          |
|                          | Length: 1, dtype: str                                                          |          |
| ProductCD                | Special                                                                        |       19 |
| ProductCD                | Missing                                                                        |       19 |
| uid_n_1h                 | (-inf, 0.5000)                                                                 |       21 |
| uid_n_1h                 | [0.5000, inf)                                                                  |       18 |
| uid_n_1h                 | Special                                                                        |       19 |
| uid_n_1h                 | Missing                                                                        |       19 |
| browser_family           | <ArrowStringArray>                                                             |       20 |
|                          | [     'line',  'palemoon',     'nokia', 'seamonkey', 'microsoft',   'maxthon', |          |
|                          |          'm',  'waterfox',        'lg',    'puffin',      'iron',      'silk', |          |
|                          |   'cyberfox',    'comodo',  'chromium',       'blu',       'aol',      'inco', |          |
|                          |         'ie',    'safari',      'edge',   'firefox']                           |          |
|                          | Length: 22, dtype: str                                                         |          |
| browser_family           | <ArrowStringArray>                                                             |       10 |
|                          | ['mobile']                                                                     |          |
|                          | Length: 1, dtype: str                                                          |          |
| browser_family           | <ArrowStringArray>                                                             |        7 |
|                          | [ 'chrome', 'samsung', 'generic',   'other',     'zte', 'android',   'opera',  |          |
|                          |  'mozilla',   'lanix']                                                         |          |
|                          | Length: 9, dtype: str                                                          |          |
| browser_family           | Special                                                                        |       19 |
| browser_family           | Missing                                                                        |       19 |
| uids_per_fingerprint     | (-inf, 7.5000)                                                                 |        4 |
| uids_per_fingerprint     | [7.5000, 266.5000)                                                             |       12 |
| uids_per_fingerprint     | [266.5000, inf)                                                                |       19 |
| uids_per_fingerprint     | Special                                                                        |       19 |
| uids_per_fingerprint     | Missing                                                                        |       19 |
| uid_txn_number           | (-inf, 1.5000)                                                                 |       25 |
| uid_txn_number           | [1.5000, 2.5000)                                                               |       23 |
| uid_txn_number           | [2.5000, 3.5000)                                                               |       23 |
| uid_txn_number           | [3.5000, 5.5000)                                                               |       22 |
| uid_txn_number           | [5.5000, 10.5000)                                                              |       21 |
| uid_txn_number           | [10.5000, inf)                                                                 |       16 |
| uid_txn_number           | Special                                                                        |       19 |
| uid_txn_number           | Missing                                                                        |       19 |
| addr1_te                 | (-inf, 0.0150)                                                                 |       36 |
| addr1_te                 | [0.0150, 0.0216)                                                               |       27 |
| addr1_te                 | [0.0216, 0.0241)                                                               |       26 |
| addr1_te                 | [0.0241, 0.0372)                                                               |       24 |
| addr1_te                 | [0.0372, 0.1071)                                                               |       21 |
| addr1_te                 | [0.1071, inf)                                                                  |       -3 |
| addr1_te                 | Special                                                                        |       19 |
| addr1_te                 | Missing                                                                        |       19 |
| n_missing_d              | (-inf, 6.5000)                                                                 |        2 |
| n_missing_d              | [6.5000, 8.5000)                                                               |       22 |
| n_missing_d              | [8.5000, 9.5000)                                                               |       24 |
| n_missing_d              | [9.5000, 10.5000)                                                              |       27 |
| n_missing_d              | [10.5000, 11.5000)                                                             |       22 |
| n_missing_d              | [11.5000, inf)                                                                 |       18 |
| n_missing_d              | Special                                                                        |       19 |
| n_missing_d              | Missing                                                                        |       19 |
| email_P_equals_R         | (-inf, inf)                                                                    |       38 |
| email_P_equals_R         | Special                                                                        |       19 |
| email_P_equals_R         | Missing                                                                        |       19 |
| amt_cents                | (-inf, 0.0005)                                                                 |       19 |
| amt_cents                | [0.0005, 0.5115)                                                               |       14 |
| amt_cents                | [0.5115, 0.9475)                                                               |        9 |
| amt_cents                | [0.9475, inf)                                                                  |       32 |
| amt_cents                | Special                                                                        |       19 |
| amt_cents                | Missing                                                                        |       19 |
| n_missing_v              | (-inf, 82.0000)                                                                |       30 |
| n_missing_v              | [82.0000, 143.5000)                                                            |       23 |
| n_missing_v              | [143.5000, 163.0000)                                                           |       14 |
| n_missing_v              | [163.0000, 180.5000)                                                           |       15 |
| n_missing_v              | [180.5000, inf)                                                                |       17 |
| n_missing_v              | Special                                                                        |       19 |
| n_missing_v              | Missing                                                                        |       19 |
| DeviceType               | <ArrowStringArray>                                                             |       16 |
|                          | ['desktop']                                                                    |          |
|                          | Length: 1, dtype: str                                                          |          |
| DeviceType               | <ArrowStringArray>                                                             |       13 |
|                          | ['mobile']                                                                     |          |
|                          | Length: 1, dtype: str                                                          |          |
| DeviceType               | Special                                                                        |       19 |
| DeviceType               | Missing                                                                        |       19 |
| uid_complete             | (-inf, 0.5000)                                                                 |       48 |
| uid_complete             | [0.5000, inf)                                                                  |       11 |
| uid_complete             | Special                                                                        |       19 |
| uid_complete             | Missing                                                                        |       19 |
| n_missing_id             | (-inf, 14.5000)                                                                |       15 |
| n_missing_id             | [14.5000, 17.5000)                                                             |        7 |
| n_missing_id             | [17.5000, 20.5000)                                                             |       12 |
| n_missing_id             | [20.5000, inf)                                                                 |       23 |
| n_missing_id             | Special                                                                        |       19 |
| n_missing_id             | Missing                                                                        |       19 |
| has_identity             | (-inf, 0.5000)                                                                 |        4 |
| has_identity             | [0.5000, inf)                                                                  |       42 |
| has_identity             | Special                                                                        |       19 |
| has_identity             | Missing                                                                        |       19 |
| n_missing_m              | (-inf, 0.5000)                                                                 |       20 |
| n_missing_m              | [0.5000, 2.5000)                                                               |       39 |
| n_missing_m              | [2.5000, 5.5000)                                                               |       26 |
| n_missing_m              | [5.5000, 7.5000)                                                               |       19 |
| n_missing_m              | [7.5000, inf)                                                                  |       14 |
| n_missing_m              | Special                                                                        |       19 |
| n_missing_m              | Missing                                                                        |       19 |
| addr1_freq               | (-inf, 527.0000)                                                               |        1 |
| addr1_freq               | [527.0000, 2905.5000)                                                          |       20 |
| addr1_freq               | [2905.5000, 6729.5000)                                                         |       24 |
| addr1_freq               | [6729.5000, 10326.5000)                                                        |       29 |
| addr1_freq               | [10326.5000, 28245.5000)                                                       |       26 |
| addr1_freq               | [28245.5000, inf)                                                              |       24 |
| addr1_freq               | Special                                                                        |       19 |
| addr1_freq               | Missing                                                                        |       19 |
| device_family            | <ArrowStringArray>                                                             |       19 |
|                          | [       'lm',         'n',       'nem', 'nethelper',       'nxa',       'nyx', |          |
|                          |          'o',       'one',   'oneplus',     'orbis',                           |          |
|                          |  ...                                                                           |          |
|                          |          'c',     'pixel',     'ilium',       'vtr',       'mot',         'p', |          |
|                          |     'mddrjs',   'android',      'lgls',       'ios']                           |          |
|                          | Length: 175, dtype: str                                                        |          |
| device_family            | <ArrowStringArray>                                                             |       28 |
|                          | ['windows',     'bll',     'ale',      'rv',  'kfdowi',      'lg',       'm',  |          |
|                          |  'samsung',     'rct',       'h',  'lenovo',     'rne',     'htc',      'gt',  |          |
|                          |      'mya',      'es',  'huawei',  'kfauwi',     'mha',       'e',      'xt',  |          |
|                          |  'hisense',     'cro',      'sm',   'build',    'lgmp',   'redmi',   'nexus',  |          |
|                          |  'alcatel',   'motog',   'motoe',    'lgms',    'moto',      'mi',   'blade',  |          |
|                          |      'was',       'f',     'trt',      'nx',     'zte',      'en',       'y',  |          |
|                          |        'a',      'vs',  'neffos',  'qtasun',       'z',      'ta',     'rex',  |          |
|                          |      'chc',      'le', 'fractal',     'tab',      'lt',  'kffowi',      'tr',  |          |
|                          |       'hi',    'beat',  'iphone',    'zeia',   'nokia',    'lglk']             |          |
|                          | Length: 62, dtype: str                                                         |          |
| device_family            | Special                                                                        |       19 |
| device_family            | Missing                                                                        |       19 |
| uid_secs_since_prev      | (-inf, 371.5000)                                                               |       11 |
| uid_secs_since_prev      | [371.5000, 2755.5000)                                                          |       17 |
| uid_secs_since_prev      | [2755.5000, 253779.5000)                                                       |       18 |
| uid_secs_since_prev      | [253779.5000, 765923.5000)                                                     |       29 |
| uid_secs_since_prev      | [765923.5000, 2375122.5000)                                                    |       38 |
| uid_secs_since_prev      | [2375122.5000, inf)                                                            |       42 |
| uid_secs_since_prev      | Special                                                                        |       19 |
| uid_secs_since_prev      | Missing                                                                        |       19 |
| m_true_count             | (-inf, 0.5000)                                                                 |       23 |
| m_true_count             | [0.5000, 2.5000)                                                               |       17 |
| m_true_count             | [2.5000, 3.5000)                                                               |       15 |
| m_true_count             | [3.5000, 4.5000)                                                               |       15 |
| m_true_count             | [4.5000, inf)                                                                  |       14 |
| m_true_count             | Special                                                                        |       19 |
| m_true_count             | Missing                                                                        |       19 |
| card6                    | <ArrowStringArray>                                                             |       24 |
|                          | ['charge card', 'debit or credit', 'debit']                                    |          |
|                          | Length: 3, dtype: str                                                          |          |
| card6                    | <ArrowStringArray>                                                             |       11 |
|                          | ['credit']                                                                     |          |
|                          | Length: 1, dtype: str                                                          |          |
| card6                    | Special                                                                        |       19 |
| card6                    | Missing                                                                        |       19 |
| device_family_te         | (-inf, 0.0260)                                                                 |       22 |
| device_family_te         | [0.0260, 0.0265)                                                               |       23 |
| device_family_te         | [0.0265, 0.0376)                                                               |       24 |
| device_family_te         | [0.0376, 0.0621)                                                               |       12 |
| device_family_te         | [0.0621, inf)                                                                  |        4 |
| device_family_te         | Special                                                                        |       19 |
| device_family_te         | Missing                                                                        |       19 |
| id_31_freq               | (-inf, 1.5000)                                                                 |       19 |
| id_31_freq               | [1.5000, 2635.0000)                                                            |       19 |
| id_31_freq               | [2635.0000, 6775.0000)                                                         |       20 |
| id_31_freq               | [6775.0000, inf)                                                               |       19 |
| id_31_freq               | Special                                                                        |       19 |
| id_31_freq               | Missing                                                                        |       19 |
| fingerprint_freq         | (-inf, 2.5000)                                                                 |       26 |
| fingerprint_freq         | [2.5000, 66.5000)                                                              |        3 |
| fingerprint_freq         | [66.5000, inf)                                                                 |       10 |
| fingerprint_freq         | Special                                                                        |       19 |
| fingerprint_freq         | Missing                                                                        |       19 |
| TransactionAmt           | (-inf, 23.9145)                                                                |       14 |
| TransactionAmt           | [23.9145, 47.7400)                                                             |       20 |
| TransactionAmt           | [47.7400, 106.5360)                                                            |       21 |
| TransactionAmt           | [106.5360, 108.9850)                                                           |       28 |
| TransactionAmt           | [108.9850, 136.0330)                                                           |       21 |
| TransactionAmt           | [136.0330, inf)                                                                |       17 |
| TransactionAmt           | Special                                                                        |       19 |
| TransactionAmt           | Missing                                                                        |       19 |
