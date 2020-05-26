# -*- coding: utf-8 -*-
"""
Created on Mon May 25 06:00:28 2020

@author: Aksam
"""

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
repo.remotes.origin

def getGitInfos():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    try:
        remoteurl = repo.remotes.origin.url
    except AttributeError:
        remoteurl = ""
    return sha, remoteurl

def logMlflow(model,data,output_folder="mlflow_out", param=dict(),metrics=dict(),features=None, tags=dict(),run_name=None):
    # Imports
    from sklearn.externals import joblib
    import mlflow
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get some general information
    type = model.__module__.split(".")[0]
    modelname = model.__class__.__name__
    sha, remoteurl = getGitInfos()
    
    # Start actual logging
    mlflow.set_experiment(experiment_name="demo")
    if not run_name:
        run_name = modelname
    with mlflow.start_run(source_name=remoteurl,source_version=sha, run_name=run_name):
        
        # Log Parameters
        for k,v in param.items():
            mlflow.log_param(k, v)

        # Track dependencies
        import pkg_resources
        with open("{}/dependencies.txt".format(output_folder), "w+") as f: 
            for d in pkg_resources.working_set:
                f.write("{}={}\n".format(d.project_name,d.version))
        mlflow.log_artifact("{}/dependencies.txt".format(output_folder))
        
        # Track data
        data.to_csv("{}/data".format(output_folder))
        mlflow.log_artifact("{}/data".format(output_folder))
        
        if type=="sklearn":
            _ = joblib.dump(model,"{}/sklearn".format(output_folder))
            mlflow.log_artifact("{}/sklearn".format(output_folder))
        if type=="lgb":
            model.save_model("{}/lghtgbm.txt".format(output_folder))
            mlflow.log_artifact("{}/lghtgbm.txt".format(output_folder))
        
        # Log metrics
        for k,v in metrics.items():
            mlflow.log_metric(k,v)

        # plot Feature importances if avaible
        featurePlot = plotFeatureImportances(model, features, type)
        if featurePlot:
            mlflow.log_artifact("{}.png".format(featurePlot))
            
        # Set some tags to identify the experiment
        mlflow.set_tag("model",modelname)
        for tag, v in tags.items():
            mlflow.set_tag(t,v)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Lasso
    # We need to import utils here, since it is an own script and the execution environment has no access to the jupyter execution environment
    from utils import *

    # Do a train_test_split
    from sklearn.datasets import load_boston
    boston = load_boston()
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    data = pd.DataFrame(boston.data,columns=boston.feature_names)
    data['target'] = pd.Series(boston.target)
    data.sample(5)
    data = getData()
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=10, random_state=42)

    # Define the details of our run
    params=dict(alpha=0.4)
    clf = Lasso(**params)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    metrics = eval_metrics(y_test, predictions)
    logMlflow(model=clf,params,metrics,features=x_test.columns.values)
