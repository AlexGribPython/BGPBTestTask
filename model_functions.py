import pickle
import constants
import pandas as pd
import os
from loguru import logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier


def load_and_process_data():
    df = pd.read_excel(f'{os.getcwd()}/data/data.xlsx', header=1)
    logger.info(f'The data was loaded, size: {df.shape}')
    df.columns = [column.replace(' ', '_').replace('_Д', 'Д').replace('рпописки', 'прописки') for column in df.columns]
    df.drop(['Номер_заявки', 'Мобильный_телефон', 'Кредитов_всего', 'Совпадение_адреса_прописки_и_регистрации',
             'Домашний_телефон', 'Страна_рождения'], axis=1, inplace=True)

    x = df.drop('Маркер', axis=1)
    y = df['Маркер']
    y = y.apply(lambda target: 1 if target == 'bad' else 0)

    x_train_boost, x_test_boost, y_train, y_test = train_test_split(x, y,
                                                                    test_size=constants.TEST_SIZE,
                                                                    stratify=y,
                                                                    random_state=constants.RANDOM_STATE)
    return x_train_boost, x_test_boost, y_train, y_test


def fit_and_save_model():
    x_train, x_test, y_train, y_test = load_and_process_data()

    data_preprocessor_tree = ColumnTransformer(
        [('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
          x_train.select_dtypes(include=['object']).columns),
         ('numerical', MinMaxScaler(), x_train.select_dtypes(exclude=['object']).columns)
         ],
        remainder='passthrough'
    )

    pipe_tree = Pipeline([
        ('preprocessor', data_preprocessor_tree),
        ('feature_selector', SelectKBest(score_func=chi2, k=10)),
        ('models', None)
    ])

    param_grid_tree = [
        {
            'models': [RandomForestClassifier(random_state=constants.RANDOM_STATE)],
            'models__max_depth': range(5, 8),
            'models__n_estimators': [100, 150],
            'feature_selector__k': [i for i in range(20, 36)] + ['all']

        }]

    gs = GridSearchCV(pipe_tree,
                      param_grid_tree,
                      cv=3,
                      n_jobs=-1,
                      scoring='roc_auc',
                      verbose=10)
    gs.fit(x_train, y_train)

    logger.info(f'Best model was founded, params: {gs.best_params_}, best AUC-ROC: {gs.best_score_:.4f}.')

    predicted_proba = gs.predict_proba(x_test)[:, 1:]
    roc_auc_test = roc_auc_score(y_test, predicted_proba)

    logger.info(f'AUC-ROC on test data: {roc_auc_test:.4f}.')

    if not os.path.isdir(f'{os.getcwd()}/models'):
        os.mkdir(f'{os.getcwd()}/models')

    with open(f'{os.getcwd()}/models/model.pkl', 'wb') as output:
        pickle.dump(gs.best_estimator_, output)
    logger.info('The model was trained and saved.')

    return {
        'Status': f'The model was trained and saved. AUC-ROC on CV = {gs.best_score_:.4f}, '
                  f'on test = {roc_auc_test:.4f}.'}


def predict(data):
    try:
        with open(f'{os.getcwd()}/models/model.pkl', 'rb') as pkl_file:
            model = pickle.load(pkl_file)
        logger.info(f'Pre-trained model was downloaded from the source: {os.getcwd()}/models/')
    except FileNotFoundError:
        logger.warning("""
        This is the first launch of the application - there aren't any model files in the app directory.
        You have to run the '/train_model' endpoint first of all to get pre-trained models to use.
        """)

        return {'Warning':
                """
                This is the first launch of the application - there aren't any model files in the app directory.
                You have to run the '/train_model' endpoint first of all to get pre-trained models to use.
                """}

    data = pd.DataFrame(data, index=[0])[constants.ORDER_COLUMNS]
    predicted_label = model.predict(data)[0]
    predicted_proba = model.predict_proba(data)[:, 1][0]

    return {'label': f'{predicted_label}', 'proba': f'{predicted_proba}'}
