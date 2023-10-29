from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def classify(X, y):
    # Handle any missing values in the data
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # sure all the features are on a similar scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # balance data to give each observation equal importance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # start analysis with a Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.5], 'max_depth': [3, 5, 7]}
    grid_search_gb = GridSearchCV(gb_clf, param_grid_gb, cv=StratifiedKFold(5), scoring='accuracy')
    grid_search_gb.fit(X_train, y_train)
    best_gb = grid_search_gb.best_estimator_
    y_pred_gb = best_gb.predict(X_test)
    print(f"Gradient Boosting gave us an accuracy of {accuracy_score(y_test, y_pred_gb) * 100:.2f}%")

    # try XGBoost
    xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    param_grid_xgb = {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.5], 'max_depth': [3, 5, 7]}
    grid_search_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=StratifiedKFold(5), scoring='accuracy')
    grid_search_xgb.fit(X_train, y_train)
    best_xgb = grid_search_xgb.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    print(f"XGBoost gave us an accuracy of {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")


    # SVC
    svc = SVC(random_state=42, probability=True)
    param_grid_svc = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['rbf']}
    grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=StratifiedKFold(5), scoring='accuracy')
    grid_search_svc.fit(X_train, y_train)
    best_svc = grid_search_svc.best_estimator_
    y_pred_svc = best_svc.predict(X_test)
    print(f"Support Vector Classifier gave us an accuracy of {accuracy_score(y_test, y_pred_svc) * 100:.2f}%")

    # combine our best models into an ensemble
    voting_clf = VotingClassifier(estimators=[('gb', best_gb), ('xgb', best_xgb), ('svc', best_svc)], voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred_vote = voting_clf.predict(X_test)
    print(f"Our Ensemble Model gave us an accuracy of {accuracy_score(y_test, y_pred_vote) * 100:.2f}%")

    # return
    return voting_clf
