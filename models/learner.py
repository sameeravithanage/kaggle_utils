from sklearn.model_selection import StratifiedKFold 
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import numpy as np
import pandas as pd
from .tree_model_diagnostics import TreeModelDiagnostics


class Learner:
    
    def __init__(self, df_train, df_test, target, model, selected_features, note, metric, random_state=42, preprocessor=None, needs_proba=False, target_class_idx=1, sampler=None):
        self.X_train = df_train[selected_features]
        self.Y_train = df_train[target]
        self.X_test = df_test[selected_features]
        self.note = note
        self.metric = metric
        self.random_state = random_state
        self.model = model
        self.feature_cols = selected_features
        self.needs_proba = needs_proba
        self.proba_class_idx = target_class_idx
        self.sampler = sampler

        if preprocessor:
            preprocessor.set_output(transform='pandas')
        self.preprocessor = preprocessor

        self.diagnostics = TreeModelDiagnostics(
            metric=self.metric, 
            preprocessor=self.preprocessor, 
            feature_names=self.feature_cols,
            needs_proba=self.needs_proba,
            target_class_idx=self.proba_class_idx
        )
    
    def _get_predictions(self, pipeline, data):
        """Helper to extract correct prediction format based on needs_proba"""
        if self.needs_proba:
            preds = pipeline.predict_proba(data)
            
            if self.proba_class_idx is not None:
                if hasattr(preds, 'iloc'):
                    return preds.iloc[:, self.proba_class_idx]
                return preds[:, self.proba_class_idx]
                
            return preds
            
        return pipeline.predict(data)

    def _get_estimator(self, pipeline):
        return (
            pipeline.named_steps["pred_model"]
            if hasattr(pipeline, "named_steps")
            else pipeline
        )


    def _prepare_fold_data(self, pipeline, x_train, y_train, x_valid):
        x_train_processed = x_train
        x_valid_processed = x_valid
    
        if self.preprocessor:
            preprocessor = pipeline.named_steps["preprocessor"]
            x_train_processed = preprocessor.fit_transform(x_train, y_train)
            x_valid_processed = preprocessor.transform(x_valid)
    
        if self.sampler:
            sampler = pipeline.named_steps["sampler"]
            x_train_processed, y_train = sampler.fit_resample(
                x_train_processed,
                y_train,
            )
    
        return x_train_processed, y_train, x_valid_processed


    def _fit_with_early_stopping( self, pipeline, x_train, y_train, x_valid, y_valid,):
        model = self._get_estimator(pipeline)

        (x_train_processed,y_train_processed,x_valid_processed) = self._prepare_fold_data(pipeline,x_train,y_train,x_valid)

        module_name = model.__class__.__module__

        if module_name.startswith("xgboost."):
            eval_set = [(x_train_processed, y_train_processed), (x_valid_processed, y_valid)]
        elif module_name.startswith("catboost."):
            eval_set = (x_valid_processed, y_valid)
        else:
            raise TypeError(
                            "Early stopping is supported only for XGBoost and CatBoost."
                        )

        if eval_set is not None:
            model.fit(
                x_train_processed,
                y_train_processed,
                eval_set=eval_set,
                verbose=False,
            )

        return pipeline

    def fit(self, n_folds=5, early_stopping_enabled=False, save_folds_results=False,fold_path=".",params=None, plotting={"fi":False, "score_vs_trees":False, "tree_depth":False}):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        self.fold_accuracy = []
        self.train_accuracy = []
        self.test_preds_list = []
        
        with mlflow.start_run(run_name=self.note):
            model_name = self.model.__name__
            mlflow.log_param("model", model_name)
            if self.preprocessor:
                mlflow.log_param("preprocessor", self.preprocessor.__class__.__name__)
            if self.sampler:
                mlflow.log_param("sampler", self.sampler.__class__.__name__)
            if params:
                mlflow.log_params(params)
                  
            for i, (train_index, test_index) in enumerate(skf.split(self.X_train, self.Y_train)):
                print(f"Training fold {i+1}...")
                x_train_fold, x_test_fold = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                y_train_fold, y_test_fold = self.Y_train.iloc[train_index], self.Y_train.iloc[test_index]
            
                model_params = dict(params or {})
                if not params:
                    model_params['random_state'] = self.random_state


                model_instance = self.model(**model_params)

                # Setting up the pipeline with preprocessor and sampler if they exist
                steps = []
                if self.preprocessor:
                    steps.append(('preprocessor', self.preprocessor))
                if self.sampler:
                    steps.append(('sampler', self.sampler))
                steps.append(('pred_model', model_instance))
                if len(steps) > 1:
                    pipeline = ImbPipeline(steps)
                else:
                    pipeline = model_instance
                self.pipeline = pipeline    

                # Fit the model with or without early stopping
                if early_stopping_enabled:
                    self._fit_with_early_stopping(pipeline,x_train_fold,y_train_fold,x_test_fold,y_test_fold)                                        
                else:
                    pipeline.fit(x_train_fold, y_train_fold)
                
                # Dynamic Evaluation
                model_oof_preds = self._get_predictions(pipeline, x_test_fold)
                OOFaccScore = self.metric(y_test_fold, model_oof_preds)
                self.fold_accuracy.append(OOFaccScore)

                train_preds = self._get_predictions(pipeline, x_train_fold)
                train_score = self.metric(y_train_fold, train_preds)
                self.train_accuracy.append(train_score)
                
                mlflow.log_metric(f"val_acc", OOFaccScore, step=i+1)
                mlflow.log_metric(f"train_acc", train_score, step=i+1)
                print(f"Fold {i+1} ==> OOF score: {OOFaccScore:.5f}")

                if save_folds_results:
                    fold_results_df = pd.DataFrame({
                        "original_index": y_test_fold.index,
                        "true_label": y_test_fold.values,
                        "prediction": model_oof_preds
                    })
                
                    csv_filename = f"{fold_path}/{self.note}_fold_{i+1}_predictions.csv"
                    fold_results_df.to_csv(csv_filename, index=False)
                
                raw_estimator = self._get_estimator(pipeline)
                if model_name in ["XGBClassifier", "XGBRegressor"]:
                    best_iteration = getattr(raw_estimator, "best_iteration", None)
                elif model_name in ["CatBoostClassifier", "CatBoostRegressor"]:
                    best_iteration = getattr(raw_estimator, "best_iteration_", None)
                else:
                    best_iteration = None

                if early_stopping_enabled:
                    mlflow.log_metric(
                        "best_iteration",
                        best_iteration,
                        step=i + 1,
                    )
                self.diagnostics.log_oob_score(raw_estimator, i+1)
                self.diagnostics.plot_feature_importances(pipeline, i+1) if plotting.get("fi", False) else None
                if plotting.get("score_vs_trees", False):
                    if model_name in ["XGBClassifier", "XGBRegressor", "CatBoostClassifier", "CatBoostRegressor"]:
                        self.diagnostics.plot_boosting_learning_curve(pipeline, i+1)
                    else:
                        self.diagnostics.plot_score_vs_trees(pipeline, x_test_fold, y_test_fold, i+1)
                self.diagnostics.plot_tree_depth_distribution(pipeline, i+1) if plotting.get("tree_depth", False) else None
                    
                # Always capture full probas for the test set to allow flexible final aggregation
                fold_test_probs = pipeline.predict_proba(self.X_test)
                self.test_preds_list.append(fold_test_probs)
        
            avgAcc = np.mean(self.fold_accuracy)
            stdAcc = np.std(self.fold_accuracy)
            avgTrainAcc = np.mean(self.train_accuracy)
            stdTrainAcc = np.std(self.train_accuracy)

            mlflow.log_metric("cv_mean_val_acc", avgAcc)
            mlflow.log_metric("cv_std_val_acc", stdAcc)
            mlflow.log_metric("cv_mean_train_acc", avgTrainAcc)
            mlflow.log_metric("cv_std_train_acc", stdTrainAcc)
            
            print(f"CV Mean Score: {avgAcc:.4f} | Std: {stdAcc:.4f}")
            print('---------------------------------------------------------------')
            
            
            mean_test_probs = np.mean(self.test_preds_list, axis=0)
            if self.needs_proba:
                # Return probabilities (sliced if necessary)
                self.final_test_predictions = mean_test_probs[:, self.proba_class_idx] if self.proba_class_idx is not None else mean_test_probs
            else:
                # Return hard labels via majority vote
                self.final_test_predictions = np.argmax(mean_test_probs, axis=1)

            return avgAcc