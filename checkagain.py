import dagshub
dagshub.init(repo_owner='atikul-islam-sajib', repo_name='Advanced-Software-Engineering', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)