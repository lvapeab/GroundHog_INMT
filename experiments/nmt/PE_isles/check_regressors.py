from sknn.backend import lasagne
from sknn.mlp import Layer
from sknn.mlp import Regressor as MLP_Regressor
import numpy as np
from sklearn import linear_model, ensemble, gaussian_process
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import NuSVR
from sklearn.cross_decomposition import PLSRegression
import argparse
import cPickle
import logging
import math
from sklearn.externals import joblib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y_%H:%M:%S: ')
logger = logging.getLogger(__name__)
rng = np.random.RandomState(1)



def get_regressors ():
        return [
                #('MLP_Regressor',MLP_Regressor(
                #    layers=[
                #        Layer('Rectifier', units=20),
                #        Layer('Linear')],
                #    learning_rate=0.001,
                #    n_iter=100)),
                ('Linear', linear_model.SGDRegressor()),
                ('Decision Tree', DecisionTreeRegressor(max_depth=4)),
                ('AdaBoost', AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                              n_estimators=500, random_state=rng))
                #('Gaussian Process', gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)),
                #('Gradient Boosting', ensemble.GradientBoostingRegressor(n_estimators=1000,
                #                               max_depth = 6,
                #                               min_samples_split =1,
                #                               learning_rate =0.01,
                #                               loss= 'ls')),
                # ('PLS Decomposition', PLSRegression(n_components=2)),
                # ('Random Forest', RandomForestRegressor(n_estimators=10, warm_start=True,
                #                                         n_jobs=4)),
                # ('Ridge', linear_model.Ridge(alpha=0.1)),
                # ('Lasso', linear_model.Lasso(alpha=0.01)),
                # ('LassoLars', linear_model.LassoLars(alpha=.1)),
                # ('BayesianRidge', linear_model.BayesianRidge()),
                # ('Perceptron', linear_model.Perceptron(penalty='l2',
                #                                        n_jobs=4,
                #                                        warm_start=True)),
                # ('LassoLars', linear_model.LassoLars(alpha=.1)),
                # ("Passive-Aggressive I", linear_model.PassiveAggressiveRegressor(loss='epsilon_insensitive',
                #                                                                  C=1.0)),
                # ("Passive-Aggressive II", linear_model.PassiveAggressiveClassifier(loss='squared_epsilon_insensitive',
                #                                                                  C = 1.0)),
                # ("SAG", linear_model.LogisticRegression(solver='sag', tol=1e-1, C=1.e4 /5000)),
                # ('OLS', linear_model.LinearRegression())
                # ('Theil-Sen', linear_model.TheilSenRegressor(random_state=42)),
                # ('RANSAC', linear_model.RANSACRegressor(linear_model.LinearRegression()))
        ]


def test_regressor(regressor, name, X_train, y_train, X_dev, y_dev):

    best_model = None
    clf = regressor

    clf.fit(X_train, y_train)
    logger.info('Results of %s:'%name)

    # The mean square error
    logger.info('\tResidual sum of squares: %.2f'
          % np.mean((clf.predict(X_dev).flatten() - y_dev) ** 2))
    # Explained variance score: 1 is perfect prediction
    logger.info('\tVariance score: %.2f' % clf.score(X_dev, y_dev))

    # predicted = cross_validation.cross_val_predict(clf, X_dev, y_dev, cv=10)
    predicted = clf.predict(X_dev)
    predicted = map(int, predicted)
    acc = accuracy_score(y_dev, predicted)
    mse = mean_squared_error(y_dev, predicted)
    r2 = r2_score(y_dev, predicted, multioutput='variance_weighted')
    logger.info( '\tR2 score: %.4f', r2)
    logger.info( '\tAccuracy: %.4f', acc)
    logger.info( '\tMSE: %.4f', mse)
    return r2


def main(train_data, dev_data, save_path):

    best_model = None
    best_name = ''
    best_r2 = 999

    X_train = train_data['embeddings']
    y_train = train_data['classes']

    X_dev = dev_data['embeddings']
    y_dev = dev_data['classes']


    for name, regressor in get_regressors():
        r2 = test_regressor(regressor, name, X_train, y_train, X_dev, y_dev)
        if abs(1-r2)  < best_r2:
            best_r2 = abs(1-r2)
            best_model = regressor
            best_name = name
    print "Best regressor obtained: %s with a R2 value of: " \
          "%4f and an accuracy of:" \
          " %4f" % (best_name,
                    r2_score(y_dev, map(int,  best_model.predict(X_dev)), multioutput='variance_weighted'),
                    accuracy_score(y_dev, map(int,  best_model.predict(X_dev))))
    joblib.dump(best_model, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--train",
                        required=True, help="Training data to use")
    parser.add_argument("--dev",
                        required=True, help="Development data to use")
    parser.add_argument("--output", default= 'models/best_accuracy',
                         help="Output path data to use")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logger.info( 'Loading data')
    train_data = np.load(args.train)
    dev_data = np.load(args.dev)
    save_path = args.output

    main(train_data, dev_data, save_path)
