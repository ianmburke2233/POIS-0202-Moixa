import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from missingpy import MissForest
from statsmodels.imputation import mice
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import metrics
import math
import time
from sklearn.decomposition import PCA
from pathlib import Path

file_names_dict = {'raw_data_train':'TrainingData.csv',
                    'raw_data_test': 'DevelopmentData.csv',
                    'impute_train': 'ImpTrain.csv'}

RAW_DATA_PATH = Path('../Data')
BASE_EXPERIMENT_PATH = Path('.')


def col_filter(col_dict, dataframe):
    filter_df = dataframe[col_dict['imputation']]
    ignore_df = dataframe[col_dict['ignore']]
    return filter_df, ignore_df

class Metadata:

    def __init__(self, path_dict, label_dict):
        self.path_dict = path_dict
        self.label_dict = label_dict

    def get_path(self, label, phase):
        # find file name in phase metatdata pertaining to a given label
        pass

    def is_computed(self, label, phase):
        path = self.get_path(label, phase)
        return path.exists()


class ScenarioScorer:

    SCENARIOS = ['Scenario1_1', 'Scenario1_2', 'Scenario1_3', 'Scenario1_4', 'Scenario1_5', 'Scenario1_6',
                 'Scenario1_7', 'Scenario1_8',
                 'Scenario2_1', 'Scenario2_2', 'Scenario2_3', 'Scenario2_4', 'Scenario2_5', 'Scenario2_6',
                 'Scenario2_7', 'Scenario2_8']

    def __init__(self, score_fn, method):
        self.method = method
        self.score_fn = score_fn

    def score(self, file_dict, update=False):
        # if update=False, then if file already exists, report and return.
        for type in file_dict.keys():
            infile = file_dict[type]
            data = pd.read_csv(RAW_DATA_PATH / infile)
            scored_scenarios = [self.score_helper(data, col_name) for col_name in self.SCENARIOS]
            scored_df = pd.concat(scored_scenarios, axis=1)
            scored_df.to_csv(BASE_EXPERIMENT_PATH / f'feature_gen/scenario_scores/{self.method}/{type}.csv',
                             index=False)

    def score_helper(self, data, scenario):
        df = pd.DataFrame(data[scenario])
        mode = int(df.mode().iloc[0][0])
        df['{}_score'.format(scenario)] = df[scenario].apply(lambda x: self.score_fn(x, mode))
        return pd.DataFrame(df['{}_score'.format(scenario)])

    @staticmethod
    def factory(method):
        if method == 'linear':
            fn = lambda x, mode: abs(x - mode)
        elif method == 'quadratic':
            fn = lambda x, mode: abs(x - mode) * abs(x - mode)
        elif method == 'exponential':
            fn = lambda x, mode: math.exp((abs(x - mode)))
        elif method == 'binary':
            fn = lambda x, mode: 1 if x == mode else 0
        return ScenarioScorer(fn, method)

obj = ScenarioScorer.factory('linear')
obj.score({'train':'TrainingData.csv', 'dev':'DevelopmentData.csv'})

pd.set_option('mode.chained_assignment', None)


def feature_generation(infile, imp_method, scoring_methods):
    data = pd.read_csv(infile)
    ids = pd.DataFrame(data[list(data.columns)[0]])
    imputer = imp_method
    imp_ids = data[list(data.columns)[:8]]
    imp_data = data[list(data.columns)[9:-1]]
    feature_vectors_filled = pd.DataFrame(imputer.fit_transform(imp_data), columns=list(imp_data.columns))
    feature_vectors_filled = pd.DataFrame(feature_vectors_filled, columns=list(imp_data.columns))
    data = imp_ids.join(feature_vectors_filled)
    SJTimes = data.filter(regex="SJ_Time.*", axis=1)
    scenarioTimes = data.filter(regex="Scenario.*._Time.*", axis=1)
    bioData = data.filter(regex="Bio.*", axis=1)

    scenarios = ['Scenario1_1', 'Scenario1_2', 'Scenario1_3', 'Scenario1_4', 'Scenario1_5', 'Scenario1_6',
                 'Scenario1_7', 'Scenario1_8',
                 'Scenario2_1', 'Scenario2_2', 'Scenario2_3', 'Scenario2_4', 'Scenario2_5', 'Scenario2_6',
                 'Scenario2_7', 'Scenario2_8']

    def scenarioScoring(scenario, raw, method):
        if method == 'linear':
            df = pd.DataFrame(raw[scenario])
            mode = int(df.mode().iloc[0][0])
            df['{}_score'.format(scenario)] = df[scenario].apply(lambda x: abs(x - mode))
            return pd.DataFrame(df['{}_score'.format(scenario)])
        elif method == 'quadratic':
            df = pd.DataFrame(raw[scenario])
            mode = int(df.mode().iloc[0][0])
            df['{}_score'.format(scenario)] = df[scenario].apply(lambda x: (abs(x - mode) * abs(x - mode)))
            return pd.DataFrame(df['{}_score'.format(scenario)])
        elif method == 'exponential':
            df = pd.DataFrame(raw[scenario])
            mode = int(df.mode().iloc[0][0])
            df['{}_score'.format(scenario)] = df[scenario].apply(lambda x: math.exp((abs(x - mode))))
            return pd.DataFrame(df['{}_score'.format(scenario)])
        elif method == 'binary':
            df = pd.DataFrame(raw[scenario])
            mode = int(df.mode().iloc[0][0])
            df['{}_score'.format(scenario)] = df[scenario].apply(lambda x: 1 if x == mode else 0)
            return pd.DataFrame(df['{}_score'.format(scenario)])

    for obj in scenarios:
        df1 = scenarioScoring(obj, data, scoring_methods['SS'])
        if scenarios.index(obj) == 0:
            scenarioScores = df1
        else:
            scenarioScores = scenarioScores.join(df1)

    def scorer(x, best, worst):
        if np.isnan(x):
            return np.nan
        if x == best:
            return 1
        if x == worst:
            return -1
        else:
            return 0

    def judgementScoring(sjItem, raw, method):
        if method == 'best_worst_full':
            df = pd.DataFrame(raw[sjItem])
            counts = pd.DataFrame(df.value_counts())
            counts = counts.reset_index()
            best = counts.iloc[0][0]
            worst = counts.iloc[-1][0]
            df['{}_score'.format(sjItem)] = df[sjItem].apply(lambda x: scorer(x, best, worst))
            return df['{}_score'.format(sjItem)]
        if method == 'best_worst_retained':
            df = pd.DataFrame(raw[sjItem])
            retained = pd.DataFrame(raw[raw['Retained'] == 1]['sjItem'])
            counts = pd.DataFrame(retained.value_counts())
            counts = counts.reset_index()
            best = counts.iloc[0][0]
            worst = counts.iloc[-1][0]
            df['{}_score'.format(sjItem)] = df[sjItem].apply(lambda x: scorer(x, best, worst))
            return df['{}_score'.format(sjItem)]
        if method == 'weights_full':
            df = pd.DataFrame(raw[sjItem])
            counts = pd.DataFrame(df.value_counts(normalize=True))
            df['{}_score'.format(sjItem)] = df[sjItem].apply(lambda x: counts.loc[x] if x in counts.index else 0)
            return df['{}_score'.format(sjItem)]
        if method == 'weights_retained':
            df = pd.DataFrame(raw[sjItem])
            retained = pd.DataFrame(raw[raw['Retained'] == 1]['sjItem'])
            counts = pd.DataFrame(retained.value_counts(normalize=True))
            df['{}_score'.format(sjItem)] = df[sjItem].apply(lambda x: counts.loc[x] if x in counts.index else 0)
            return df['{}_score'.format(sjItem)]

    mosts = data.filter(regex="SJ_Most.*", axis=1)

    for column in list(mosts.columns):
        df1 = pd.DataFrame(judgementScoring(column, mosts, scoring_methods['SJ']))
        if list(mosts.columns).index(column) == 0:
            SJMostScores = df1
        else:
            SJMostScores = SJMostScores.join(df1)

    leasts = data.filter(regex="SJ_Least.*", axis=1)

    for column in list(leasts.columns):
        df2 = pd.DataFrame(judgementScoring(column, leasts, scoring_methods['SJ']))
        if list(leasts.columns).index(column) == 0:
            SJLeastScores = df2
        else:
            SJLeastScores = SJLeastScores.join(df2)

    scale1 = data[['PScale01_Q1', 'PScale01_Q2', 'PScale01_Q3', 'PScale01_Q4']]
    scale1['PScale01_Q2_Rev'] = scale1['PScale01_Q2'].apply(lambda x: 5 - x)
    scale1['PScale01_Q3_Rev'] = scale1['PScale01_Q3'].apply(lambda x: 5 - x)
    scale1['average_S1'] = (scale1['PScale01_Q1'] + scale1['PScale01_Q2_Rev'] + scale1['PScale01_Q3_Rev'] + scale1[
        'PScale01_Q4']) / 4

    scale2 = data[['PScale02_Q1', 'PScale02_Q2', 'PScale02_Q3', 'PScale02_Q4']]
    scale2['PScale02_Q2_Rev'] = scale2['PScale02_Q2'].apply(lambda x: 5 - x)
    scale2['average_S2'] = (scale2['PScale02_Q1'] + scale2['PScale02_Q2_Rev'] + scale2['PScale02_Q3'] + scale2[
        'PScale02_Q4']) / 4

    scale3 = data[['PScale03_Q1', 'PScale03_Q2', 'PScale03_Q3', 'PScale03_Q4']]
    scale3['PScale03_Q1_Rev'] = scale3['PScale03_Q1'].apply(lambda x: 5 - x)
    scale3['average_S3'] = (scale3['PScale03_Q1_Rev'] + scale3['PScale03_Q2'] + scale3['PScale03_Q3'] + scale3[
        'PScale03_Q4']) / 4

    scale4 = data[['PScale04_Q1', 'PScale04_Q2', 'PScale04_Q3', 'PScale04_Q4']]
    scale4['PScale04_Q1_Rev'] = scale4['PScale04_Q1'].apply(lambda x: 5 - x)
    scale4['average_S4'] = (scale4['PScale04_Q1_Rev'] + scale4['PScale04_Q2'] + scale4['PScale04_Q3'] + scale4[
        'PScale04_Q4']) / 4

    scale5 = data[['PScale05_Q1', 'PScale05_Q2', 'PScale05_Q3', 'PScale05_Q4']]
    scale5['average_S5'] = (scale5['PScale05_Q1'] + scale5['PScale05_Q2'] + scale5['PScale05_Q3'] + scale5[
        'PScale05_Q4']) / 4

    scale6 = data[['PScale06_Q1', 'PScale06_Q2', 'PScale06_Q3', 'PScale06_Q4', 'PScale06_Q5', 'PScale06_Q6']]
    scale6['PScale06_Q3_Rev'] = scale6['PScale06_Q3'].apply(lambda x: 5 - x)
    scale6['PScale06_Q6_Rev'] = scale6['PScale06_Q6'].apply(lambda x: 5 - x)
    scale6['average_S6'] = (scale6['PScale06_Q1'] + scale6['PScale06_Q2'] +
                            scale6['PScale06_Q3_Rev'] + scale6['PScale06_Q4'] +
                            scale6['PScale06_Q5'] + scale6['PScale06_Q6_Rev']) / 4

    scale7 = data[['PScale07_Q1', 'PScale07_Q2', 'PScale07_Q3', 'PScale07_Q4']]
    scale7['PScale07_Q2_Rev'] = scale7['PScale07_Q2'].apply(lambda x: 5 - x)
    scale7['average_S7'] = (scale7['PScale07_Q1'] + scale7['PScale07_Q2_Rev'] + scale7['PScale07_Q3'] + scale7[
        'PScale07_Q4']) / 4

    scale8 = data[['PScale08_Q1', 'PScale08_Q2', 'PScale08_Q3', 'PScale08_Q4']]
    scale8['PScale08_Q1_Rev'] = scale8['PScale08_Q1'].apply(lambda x: 5 - x)
    scale8['PScale08_Q3_Rev'] = scale8['PScale08_Q3'].apply(lambda x: 5 - x)
    scale8['average_S8'] = (scale8['PScale08_Q1_Rev'] + scale8['PScale08_Q2']
                            + scale8['PScale08_Q3_Rev'] + scale8['PScale08_Q4']) / 4

    scale9 = data[['PScale09_Q1', 'PScale09_Q2', 'PScale09_Q3', 'PScale09_Q4']]
    scale9['PScale09_Q2_Rev'] = scale9['PScale09_Q2'].apply(lambda x: 5 - x)
    scale9['average_S9'] = (scale9['PScale09_Q1'] + scale9['PScale09_Q2_Rev']
                            + scale9['PScale09_Q3'] + scale9['PScale09_Q4']) / 4

    scale10 = data[['PScale10_Q1', 'PScale10_Q2', 'PScale10_Q3', 'PScale10_Q4']]
    scale10['PScale10_Q3_Rev'] = scale10['PScale10_Q3'].apply(lambda x: 5 - x)
    scale10['PScale10_Q4_Rev'] = scale10['PScale10_Q4'].apply(lambda x: 5 - x)
    scale10['average_S10'] = (scale10['PScale10_Q1'] + scale10['PScale10_Q2']
                              + scale10['PScale10_Q3_Rev'] + scale10['PScale10_Q4_Rev']) / 4

    scale11 = data[['PScale11_Q1', 'PScale11_Q2', 'PScale11_Q3', 'PScale11_Q4']]
    scale11['PScale11_Q2_Rev'] = scale11['PScale11_Q2'].apply(lambda x: 5 - x)
    scale11['PScale11_Q3_Rev'] = scale11['PScale11_Q3'].apply(lambda x: 5 - x)
    scale11['average_S11'] = (scale11['PScale11_Q1'] + scale11['PScale11_Q2_Rev']
                              + scale11['PScale11_Q3_Rev'] + scale11['PScale11_Q4']) / 4

    scale12 = data[['PScale12_Q1', 'PScale12_Q2', 'PScale12_Q3', 'PScale12_Q4']]
    scale12['PScale12_Q3_Rev'] = scale12['PScale12_Q3'].apply(lambda x: 5 - x)
    scale12['PScale12_Q4_Rev'] = scale12['PScale12_Q4'].apply(lambda x: 5 - x)
    scale12['average_S12'] = (scale12['PScale12_Q1'] + scale12['PScale12_Q2']
                              + scale12['PScale12_Q3_Rev'] + scale12['PScale12_Q4_Rev']) / 4

    scale13 = data[['PScale13_Q1', 'PScale13_Q2', 'PScale13_Q3', 'PScale13_Q4', 'PScale13_Q5']]
    scale13['PScale13_Q3_Rev'] = scale13['PScale13_Q3'].apply(lambda x: 5 - x)
    scale13['PScale13_Q4_Rev'] = scale13['PScale13_Q4'].apply(lambda x: 5 - x)
    scale13['average_S13'] = (scale13['PScale13_Q1'] + scale13['PScale13_Q2']
                              + scale13['PScale13_Q3_Rev'] + scale13['PScale13_Q4_Rev'] +
                              scale13['PScale13_Q5']) / 4

    PScaleScores = pd.concat([scale1['average_S1'], scale2['average_S2'], scale3['average_S3'], scale4['average_S4'],
                              scale5['average_S5'], scale6['average_S6'], scale7['average_S7'], scale8['average_S8'],
                              scale9['average_S9'], scale10['average_S10'], scale11['average_S11'],
                              scale12['average_S12'],
                              scale13['average_S13']], axis=1)

    if scoring_methods['P'] == 'PCA':
        pca = PCA(n_components=5)
        pca.fit(data)
        print(pca.explained_variance_)
        pca_data = pca.transform(data)
        feature_vectors = bioData.join([SJTimes, SJMostScores, SJLeastScores, scenarioTimes, scenarioScores, pca_data])
    else:
        feature_vectors = bioData.join([SJTimes, SJMostScores, SJLeastScores, scenarioTimes, scenarioScores,
                                        PScaleScores])
    final = ids.join(feature_vectors)

    i = 1
    while i < 10:
        final['SJ_Total_{}_score'.format(i)] = final['SJ_Most_{}_score'.format(i)] + final[
            'SJ_Least_{}_score'.format(i)]
        i += 1
    final['SJ_Sum'] = final.filter(regex='SJ_Total.*').sum(axis=1)

    def hasPerformance(q):
        if math.isnan(q):
            return 0
        else:
            return 1

    final['Overall_Rating'] = np.nan
    final['hasPerf'] = final['Overall_Rating'].apply(lambda q: hasPerformance(q))

    timestamp = time.time()

    outfile = open('Functions/Phase 1 Metatdata.csv', 'a')
    outfile.write('{},{},{},{},{},{}'.format(timestamp, data, imp_method, scoring_methods['SS'],
                                             scoring_methods['SJ'], scoring_methods['P']))
    final.to_csv('Functions/Phase 1/{}'.format(timestamp), index=False)


# Multiple imputed files should be the default
# https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html

class MissingDataImputer:

    def __init__(self, label, impute_fn, col_dict):
        self.label = label
        self.impute_fn = impute_fn
        self.col_dict = col_dict
        self.folder_path = BASE_EXPERIMENT_PATH / f'md_impute/md_{self.label}'
        try:
            self.folder_path.mkdir()
        except FileExistsError:
            val = input(f'Replace Directory {self.folder_path} (Y/N): ')
            if val.lower() == 'y':
                self.folder_path.mkdir(exist_ok=True)
            else:
                raise

    def impute(self, file_dict):
        for type in file_dict.keys():
            df = pd.read_csv(file_dict[type])
            filtered_df, ignored_df = col_filter(self.col_dict, df)
            imputed_df = self.impute_fn(filtered_df)
            final_df = pd.concat([ignored_df, imputed_df])
            final_df.to_csv(self.folder_path/f'{type}_md_imp_file_{0}.csv', index=False)
            ## TO DO: Add code to save metadata to file

            # for each imputation :

    @staticmethod
    def factory(method, label, cols, **kwargs):
        if method =="knn":
            imputer = KNNImputer(**kwargs)
            fn = lambda df: imputer.fit_transform(df)
        elif method == "missforest":
            imputer = MissForest(**kwargs)
            fn = lambda df: imputer.fit_transform(df)
        # elif method == "mice":
            # Need to get proper design pattern for this
            # imputer = mice.MICEData(**kwargs)
            # fn = lambda df: imputer.fit_transform(df)
        return MissingDataImputer(label, fn, cols)


class HPImputer:

    def __init__(self, label, fname, impute_fn, col_dict, normalize, oversample):
        self.label = label
        self.folder_path = BASE_EXPERIMENT_PATH / f'hp_impute/hp_{self.label}'
        self.data = pd.read_csv(RAW_DATA_PATH / fname)
        self.impute_fn = impute_fn
        self.col_dict = col_dict
        self.features = None
        self.normalize = normalize
        self.oversample = oversample
        try:
            self.folder_path.mkdir()
        except FileExistsError:
            val = input(f'Replace Directory {self.folder_path} (Y/N): ')
            if val.lower() == 'y':
                self.folder_path.mkdir(exist_ok=True)
            else:
                raise

    def preprocess(self, normalize, oversample):
        if normalize:
            self.features = scale(self.features)
        if oversample.lower() == 'ros':
            ros = RandomOverSampler()
            self.features, self.target = ros.fit_resample(self.features, self.target)
        if oversample.lower() == 'smote':
            smote = SMOTE()
            self.features, self.target = smote.fit_resample(self.features, self.target)
        if oversample.lower() == 'adasyn':
            adasyn = ADASYN()
            self.features, self.target = adasyn.fit_resample(self.features, self.target)

    def model(self):
        self.features, ignored_df = col_filter(self.col_dict, self.data)
        self.preprocess(self.normalize, self.oversample)
        fitted_model = self.impute_fn(self.features, self.features['High Performer'])
        predictions = fitted_model.predict(self.features)
        self.data = self.data.append(predictions, axis=1)
        self.data.to_csv(self.folder_path/'hp_imp_file.csv', index=False)

        ## TO DO: Add code to save metadata to file

        # for each imputation :

    @staticmethod
    def factory(method, filename, cols, label, normalize, oversample, **kwargs):
        if method == 'RF':
            clf = RandomForestClassifier(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'GB':
            clf = GradientBoostingClassifier(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'ADA':
            clf = AdaBoostClassifier(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'SVM':
        if method == 'LR':
        if method == 'EN':
        if method == 'MLP':
        return HPImputer(label, filename, fn, cols, normalize, oversample)


class Modeler:

    def __init__(self, label, fname, model_fn, col_dict):
        self.label = label
        self.folder_path = BASE_EXPERIMENT_PATH / f'model/{self.label}'
        self.data = pd.read_csv(RAW_DATA_PATH / fname)
        self.model_fn = model_fn
        self.col_dict = col_dict
        self.features = None
        try:
            self.folder_path.mkdir()
        except FileExistsError:
            val = input(f'Replace Directory {self.folder_path} (Y/N): ')
            if val.lower() == 'y':
                self.folder_path.mkdir(exist_ok=True)
            else:
                raise

    def model_eval(self, df, target, test_size):
        scorers = {'Accuracy': 'accuracy', 'ROC': 'roc_auc'}
        data_train, data_test, target_train, target_test = train_test_split(df, target, test_size=test_size)
        clf = self.model_fn(df)
        scores = cross_validate(estimator=clf, X=data_train, y=target_train, scoring=scorers, cv=5)
        scores_Acc = scores['test_Accuracy']
        scores_AUC = scores['ROC']
        clf.fit(data_train, target_train)
        pred = clf.predict(data_test)
        cm = metrics.confusion_matrix(target_test, pred, normalize='true')
        out = pd.DataFrame({'metric': ['ACC', 'AUC', 'TP', 'TN', 'FP', 'FN'],
                            'mean': [scores_Acc.mean(), scores_AUC.mean(), cm[1][1], cm[0][0], cm[0][1], cm[1][0]],
                            'std': [scores_Acc.std(), scores_AUC.std(), 'NA', 'NA', 'NA', 'NA']})
        out.to_csv(self.folder_path/'performance.csv', index=False)

    def model_predict(self):
        filtered_df, ignored_df = col_filter(self.col_dict, self.data)
        fitted_model = self.model_fn(filtered_df)
        predictions = fitted_model.predict(filtered_df)
        self.data = self.data.append(predictions, axis=1)
        self.data.to_csv(self.folder_path/'prediction.csv', index=False)

        ## TO DO: Add code to save metadata to file


    @staticmethod
    def factory(method, filename, cols, label, **kwargs):
        if method == 'RF':
            clf = RandomForestClassifier(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'GB':
            clf = GradientBoostingClassifier(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'ADA':
            clf = AdaBoostClassifier(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'SVM':
        if method == 'LR':
        if method == 'EN':
        if method == 'MLP':
        return Modeler(label, filename, fn, cols)


class Experiment:

    def __init__(self, function_dict, exp_path):
        if 'missing_data_fn' in function_dict:
            self.missing_data_fn = function_dict['missing_data_fn']
        if 'feature_gen_fn' in function_dict:
            self.feature_gen_fn = function_dict['feature_gen_fn']
        if 'hp_impute_fn' in function_dict:
            self.feature_gen_fn = function_dict['hp_impute_fn']
        if 'model_fn' in function_dict:
            self.feature_gen_fn = function_dict['model_fn']


    def generate_features(self):


    def HP_impute(self):