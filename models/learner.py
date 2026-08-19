from sklearn.model_selection import StratifiedKFold 
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import numpy as np
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

    def fit(self, params=None, plotting={"fi":False, "score_vs_trees":False, "tree_depth":False}):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.fold_accuracy = []
        self.train_accuracy = []
        self.test_preds_list = []
        
        run_name = f"{self.note}"
        with mlflow.start_run(run_name=run_name):
            
            mlflow.log_param("model", self.model.__name__)
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
            
                model_instance = self.model(**params) if params else self.model(random_state=self.random_state)

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

                
                raw_estimator = pipeline.named_steps['pred_model'] if self.preprocessor else pipeline
                self.diagnostics.log_oob_score(raw_estimator, i+1)
                self.diagnostics.plot_feature_importances(pipeline, i+1) if plotting.get("fi", False) else None
                self.diagnostics.plot_score_vs_trees(pipeline, x_test_fold, y_test_fold, i+1) if plotting.get("score_vs_trees", False) else None
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