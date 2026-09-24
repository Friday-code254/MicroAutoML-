import sqlite3
import pandas as pd
conn = sqlite3.connect('runs/RUN_btrain_clean_20260924_202555_9C7E/database.sqlite')
df = pd.read_sql("SELECT experiment_id, json_extract(spec_json, '$.model_name') as model, json_extract(result_json, '$.cv_score_mean') as score FROM experiments", conn)
print(df.to_string())
