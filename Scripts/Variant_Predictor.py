
import numpy as np
import itertools

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
from sklearn.svm import SVR


N_SPLITS = 5
MAX_ITER = 200000

class Variant_Predictor:
    def __init__(self, pred_model) -> None:
        self.pred_model = pred_model.lower()

        self.current_seed = None
            
    def get_predictor(self, params=None):
        if self.pred_model == 'lasso':
            if params == None:
                return Lasso(max_iter=MAX_ITER)
            else:
                return Lasso(**params, max_iter=MAX_ITER)
        elif self.pred_model == 'rf':
            if params == None:
                return RandomForestRegressor(random_state=42)
            else:
                return RandomForestRegressor(**params, random_state=42)
        elif self.pred_model == 'svr':
            if params == None:
                return SVR()
            else:
                return SVR(**params)
    
    def get_param_grid(self):
        if self.pred_model == 'lasso':
            return {"alpha": [float(f"1e{n}") for n in range(-3, -1)]}
        elif self.pred_model == 'rf':
            return {
                'n_estimators': [1500], #[500, 700, 1000, 1200, 1400, 1600],
                'max_features': ['sqrt'],
                'max_depth' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 60],
            }
        elif self.pred_model == 'svr':
            return [
                {'kernel': ['rbf'], 'gamma': ['scale', 'auto'], 'C': [0.5, 1, 3, 5, 6, 8, 10]},
                # {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto'], 'C': [0.1, 3, 5, 6, 8, 10]}
            ]
    
    def get_params_str(self, params):
        if self.pred_model == 'lasso':
            return f"alpha: {params['alpha']}"
        elif self.pred_model == 'rf':
            return f"n_tree: {params['n_estimators']}, max_feat: {params['max_features']}, max_depth: {params['max_depth']}"
        elif self.pred_model == 'svr':
            return ", ".join([f"{p}: {params[p]}" for p in params])
    
    def grid_search(self, X, y, n_jobs):
        param_grid = self.get_param_grid()
        predictorr = self.get_predictor()
        cros_valid = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        lasso_grid = GridSearchCV(estimator=predictorr, param_grid=param_grid, cv=cros_valid, n_jobs=n_jobs)
        return lasso_grid.fit(X, y).best_params_
    
    def fit_predict(self, X_train, X_test, y_train, params):
        predictor = self.get_predictor(params)
        predictor.fit(X_train, y_train)  
        return predictor.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        mse  = mean_squared_error(y_true, y_pred)
        r, _ = spearmanr(y_true, y_pred)
        return mse, r
