# my_ds_lib/tuning/gpu_tuner.py

import optuna
import mlflow
import numpy as np
import cupy as cp
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler


class GPUTuner:
    def __init__(self, learner, random_state=42):
        """
        Initializes the GPUTuner with a Learner instance and a random state.
        """
        self.learner = learner
        self.random_state = random_state

    def fast_gpu_tune(self, study_name, n_trials, get_params_func, db_path="optuna_study.db"):
        
        # 1. Safely extract raw data from the Learner
        X_train_raw = self.learner.train[self.learner.feature_cols]
        y_train_raw = self.learner.train[self.learner.target]

        # 2. Preprocess using the Learner's preprocessor (if it exists)
        if self.learner.preprocessor:
            X_processed_cpu = self.learner.preprocessor.fit_transform(X_train_raw)
        else:
            X_processed_cpu = X_train_raw.copy()
            
        X_processed_cpu = np.array(X_processed_cpu, dtype='float32')
        y_cpu = y_train_raw.to_numpy().astype('int32')

        # 3. Initialize Optuna
        self.tuning_study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///{db_path}', 
            load_if_exists=True
        )
        
        def objective(trial):
            params = get_params_func(trial)
            
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                mlflow.log_param("trial_no", trial.number)
                mlflow.log_params(params)

                ros = RandomOverSampler(random_state=self.random_state)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                
                fold_accuracy = []
                train_accuracy = []

                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_processed_cpu, y_cpu)):
                    # CPU Splitting
                    X_train_fold_cpu = X_processed_cpu[train_idx]
                    y_train_fold_cpu = y_cpu[train_idx]

                    # CPU Resampling
                    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_fold_cpu, y_train_fold_cpu)
    
                    # Move to GPU VRAM
                    X_train_gpu = cp.array(X_train_resampled)
                    y_train_gpu = cp.array(y_train_resampled)
                    X_val_gpu = cp.array(X_processed_cpu[val_idx])

                    # Extract Model Class from Learner and instantiate
                    model_instance = self.learner.model_class(**params)
                    model_instance.fit(X_train_gpu, y_train_gpu)
                    
                    # Predict and pull back to CPU for scoring
                    val_preds_gpu = model_instance.predict(X_val_gpu)
                    val_preds_cpu = val_preds_gpu.get() 

                    train_preds_gpu = model_instance.predict(X_train_gpu)  
                    train_preds_cpu = train_preds_gpu.get()
                    
                    # Score using the Learner's metric function
                    OOF_val_score = self.learner.metric(y_cpu[val_idx], val_preds_cpu)
                    train_score = self.learner.metric(y_train_resampled, train_preds_cpu)
                    
                    fold_accuracy.append(OOF_val_score)
                    train_accuracy.append(train_score)
                    
                    mlflow.log_metric("val_acc", OOF_val_score, step=fold_idx + 1)
                    mlflow.log_metric("train_acc", train_score, step=fold_idx + 1)

                # Aggregate Metrics
                mean_val_acc = np.mean(fold_accuracy)
                mean_train_acc = np.mean(train_accuracy)

                mlflow.log_metric("cv_mean_val_acc", mean_val_acc)
                mlflow.log_metric("cv_std_val_acc", np.std(fold_accuracy))
                mlflow.log_metric("cv_mean_train_acc", mean_train_acc)
                mlflow.log_metric("cv_std_train_acc", np.std(train_accuracy))
                
            return mean_val_acc

        # Wrap the whole study in a parent MLflow run using Learner's experiment name
        run_name = f"{self.learner.experiment_name}_Fast_GPU_Tune"
        with mlflow.start_run(run_name=run_name):
            
            self.tuning_study.optimize(
                objective, 
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            mlflow.log_params(self.tuning_study.best_params)
            mlflow.log_metric("best_cv_mean_val_acc", self.tuning_study.best_value)
        
        print(f"Best Params: {self.tuning_study.best_params}")
        return self.tuning_study.best_params