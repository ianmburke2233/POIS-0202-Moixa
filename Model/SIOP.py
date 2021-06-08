import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from missingpy import MissForest
from statsmodels.imputation import mice
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import metrics
import math
import time
from sklearn.decomposition import PCA
from pathlib import Path
from shutil import rmtree
import json

pd.set_option('mode.chained_assignment', None)

RAW_DATA_PATH = Path('../Data')
BASE_EXPERIMENT_PATH = Path('.')
COLUMNS = pd.read_csv(BASE_EXPERIMENT_PATH / 'columns.csv')


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False


def col_filter(included_groups, dataframe, type):
    included_features = list(COLUMNS[COLUMNS['Feature Group'].isin(included_groups)].Column.values)
    excluded_features = list(COLUMNS[~COLUMNS['Feature Group'].isin(included_groups)].Column.values)
    filter_df = dataframe[included_features]
    if type == 'Train':
        ignore_df = dataframe[excluded_features]
    else:
        ignore_df = pd.DataFrame(dataframe['UNIQUE_ID'])
    return filter_df, ignore_df


def combiner(fpaths, datatype):
    if datatype == 'Train':
        performance = pd.read_csv(BASE_EXPERIMENT_PATH / 'base_data/performance.csv')
        combined = performance
        for fpath in fpaths:
            df = pd.read_csv(BASE_EXPERIMENT_PATH / f'{fpath}')
            combined = combined.merge(df, left_on=['UNIQUE_ID'], right_on=['UNIQUE_ID'])
    else:
        for fpath in fpaths:
            df = pd.read_csv(BASE_EXPERIMENT_PATH / f'{fpath}')
            if fpaths.index(fpath) == 0:
                combined = df
            else:
                combined = combined.merge(df, left_on=['UNIQUE_ID'], right_on=['UNIQUE_ID'])
    return combined


class ScenarioScorer:

    SCENARIOS = ['Scenario1_1', 'Scenario1_2', 'Scenario1_3', 'Scenario1_4', 'Scenario1_5', 'Scenario1_6',
                 'Scenario1_7', 'Scenario1_8',
                 'Scenario2_1', 'Scenario2_2', 'Scenario2_3', 'Scenario2_4', 'Scenario2_5', 'Scenario2_6',
                 'Scenario2_7', 'Scenario2_8']

    def __init__(self, score_fn, method):
        self.method = method
        self.score_fn = score_fn

    def score(self, file_dict, update=False):
        fname_paths = {}
        file_path = BASE_EXPERIMENT_PATH / f'feature_gen/scenario_scores/{self.method}'
        file_path.mkdir(exist_ok=True)
        # if update=False, then if file already exists, report and return.
        for type in file_dict.keys():
            fname_path = BASE_EXPERIMENT_PATH / f'feature_gen/scenario_scores/{self.method}/{type}.csv'
            if not fname_path.is_file():
                infile = file_dict[type]
                data = pd.read_csv(RAW_DATA_PATH / infile)
                ids = pd.DataFrame(data['UNIQUE_ID'])
                scored_scenarios = [self.score_helper(data, col_name) for col_name in self.SCENARIOS]
                scored_df = ids
                for item in scored_scenarios:
                    scored_df = pd.concat([scored_df, item], axis=1)
                scored_df.to_csv(fname_path, index=False)
            fname_paths[type] = fname_path
        return fname_paths

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


class JudgementScorer:

    QUESTIONS = ['SJ_Most_1', 'SJ_Most_2', 'SJ_Most_3', 'SJ_Most_4', 'SJ_Most_5', 'SJ_Most_6', 'SJ_Most_7',
                 'SJ_Most_8', 'SJ_Most_9', 'SJ_Least_1', 'SJ_Least_2', 'SJ_Least_3', 'SJ_Least_4', 'SJ_Least_5',
                 'SJ_Least_6', 'SJ_Least_7', 'SJ_Least_8', 'SJ_Least_9']

    def __init__(self, method, retained_only, proportion):
        self.retained_only = retained_only
        self.proportion = proportion
        self.method = method

    def bw_scorer(self, x):
        if np.isnan(x):
            return np.nan
        if x == self.best_answer:
            return 1
        if x == self.worst_answer:
            return -1
        else:
            return 0

    def proportion_scorer(self, x, col):
        if np.isnan(x):
            return np.nan
        else:
            return self.score_matrix.at[x, col][0]

    def score(self, file_dict, update=False):
        # if update=False, then if file already exists, report and return.
        fname_paths = {}
        file_path = BASE_EXPERIMENT_PATH / f'feature_gen/judgement_scores/{self.method}'
        file_path.mkdir(exist_ok=True)
        for type in file_dict.keys():
            fname_path = BASE_EXPERIMENT_PATH / f'feature_gen/judgement_scores/{self.method}/{type}.csv'
            if not fname_path.is_file():
                infile = file_dict[type]
                data = pd.read_csv(RAW_DATA_PATH / infile)
                ids = pd.DataFrame(data['UNIQUE_ID'])
                scored_judgement = [self.score_helper(data, col_name, type) for col_name in self.QUESTIONS]
                scored_df = ids
                for item in scored_judgement:
                    scored_df = pd.concat([scored_df, item], axis=1)
                for i in range(1, 9):
                    scored_df['SJ_Total_{}_score'.format(i)] = scored_df['SJ_Most_{}_score'.format(i)] + scored_df[
                        'SJ_Least_{}_score'.format(i)]
                scored_df['SJ_Sum'] = scored_df.filter(regex='SJ_Total.*').sum(axis=1)
                scored_df.to_csv(fname_path, index=False)
            fname_paths[type] = fname_path
        return fname_paths

    def score_helper(self, data, question, type):
        if self.retained_only and type == 'Train':
            judgement_data = data[data['Retained'] == 1]
        else:
            judgement_data = data
        df = pd.DataFrame(judgement_data[question])
        if self.proportion:
            self.score_matrix = pd.DataFrame(df.value_counts(normalize=True), columns=[question])
            df['{}_score'.format(question)] = df[question].apply(lambda x: self.proportion_scorer(x, question))
        else:
            self.score_matrix = pd.DataFrame(df.value_counts()).reset_index()
            self.best_answer = self.score_matrix.iloc[0][question]
            self.worst_answer = self.score_matrix.iloc[len(self.score_matrix)-1][question]
            df['{}_score'.format(question)] = df[question].apply(lambda x: self.bw_scorer(x))
        return pd.DataFrame(df['{}_score'.format(question)])

    @staticmethod
    def factory(method):
        if method == 'best_worst_full':
            retained_only = False
            proportion = False
        elif method == 'best_worst_retained':
            retained_only = True
            proportion = False
        elif method == 'weights_full':
            retained_only = False
            proportion = True
        elif method == 'weights_retained':
            retained_only = True
            proportion = True
        else:
            retained_only = False
            proportion = False
        return JudgementScorer(method, retained_only, proportion)


class PersonalityScorer:

        def __init__(self, method):
            self.method = method

        def pca(self, df):
            df = df.fillna(0)
            pca = PCA(n_components=self.method['n_components'])
            pca.fit(df)
            pca_data = pca.transform(df)
            return pca_data

        def score(self, file_dict, update=False):
            fname_paths = {}
            file_path = BASE_EXPERIMENT_PATH / f'feature_gen/personality_scores/{self.method["pca"]}'
            file_path.mkdir(exist_ok=True)
            for type in file_dict.keys():
                fname_path = BASE_EXPERIMENT_PATH / f'feature_gen/personality_scores/'f'{self.method["pca"]}/{type}.csv'
                if not fname_path.is_file():
                    infile = file_dict[type]
                    data = pd.read_csv(RAW_DATA_PATH / infile)
                    ids = pd.DataFrame(data['UNIQUE_ID'])
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
                    if str_to_bool(self.method['pca']):
                        PScaleScores = pd.DataFrame(self.pca(PScaleScores))
                    scored_df = pd.concat([ids, PScaleScores], axis=1)
                    scored_df.to_csv(fname_path, index=False)
                fname_paths[type] = fname_path
            return fname_paths


class BioGrouper:

    def __init__(self, method, grouping_fn):
        self.method = method
        self.grouping_fn = grouping_fn

    def score(self, file_dict, update=False):
        # if update=False, then if file already exists, report and return.
        fname_paths = {}
        file_path = BASE_EXPERIMENT_PATH / f'feature_gen/bio_groups/{self.method}'
        file_path.mkdir(exist_ok=True)
        for type in file_dict.keys():
            fname_path = BASE_EXPERIMENT_PATH / f'feature_gen/bio_groups/{self.method}/{type}.csv'
            if not fname_path.is_file():
                infile = file_dict[type]
                data = pd.read_csv(RAW_DATA_PATH / infile).fillna(0)
                ids = pd.DataFrame(data['UNIQUE_ID'])
                bio_data = data.filter(regex="Biodata.*", axis=1)
                transformed_bio_data = pd.DataFrame(self.grouping_fn(bio_data))
                scored_df = pd.concat([ids, transformed_bio_data], axis=1)
                scored_df.to_csv(fname_path, index=False)
            fname_paths[type] = fname_path
        return fname_paths

    @staticmethod
    def factory(method, **kwargs):
        if method == 'Kmeans':
            cluster = KMeans(**kwargs)
            fn = lambda df: cluster.fit_transform(df)
        if method == 'None':
            fn = lambda df: df
        return BioGrouper(method, fn)


# Multiple imputed files should be the default
# https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html


class MissingDataImputer:

    def __init__(self, label, impute_fn, col_groups, method):
        self.label = label
        self.impute_fn = impute_fn
        self.col_groups = col_groups
        self.method = method
        self.folder_path = BASE_EXPERIMENT_PATH / f'md_impute/{self.method}'
        self.data_imputed = False
        try:
            self.folder_path.mkdir()
        except FileExistsError:
            val = input(f'Smite Directory {self.folder_path} (Y/N):')
            if val.lower() == 'y':
                rmtree(self.folder_path.absolute())
                self.folder_path.mkdir()
                print('Directory Smote')
            else:
                self.data_imputed = True
        self.fpaths = {}
        for type in ['Train', 'Test']:
            fpath = self.folder_path/f'{type}.csv'
            self.fpaths[type] = fpath

    def impute(self, data_dict):
        for type in data_dict.keys():
            if not self.fpaths[type].is_file():
                df = data_dict[type]
                filtered_df, ignored_df = col_filter(self.col_groups, df, type)
                print('Imputing...')
                md_impute_start = time.time()
                imputed_df = pd.DataFrame(self.impute_fn(filtered_df), columns=list(filtered_df.columns))
                md_impute_end = time.time()
                print(f'Imputation complete in {round((md_impute_end - md_impute_start), 2)} seconds')
                final_df = pd.concat([ignored_df, imputed_df], axis=1)
                final_df.to_csv(self.fpaths[type], index=False)

    @staticmethod
    def factory(method, label, col_groups, kwargs):
        if method == "mean":
            imputer = KNNImputer(**kwargs)
            fn = lambda df: imputer.fit_transform(df)
        if method == "knn":
            imputer = KNNImputer(**kwargs)
            fn = lambda df: imputer.fit_transform(df)
        elif method == "missforest":
            imputer = MissForest(**kwargs)
            fn = lambda df: imputer.fit_transform(df)
        # elif method == "mice":
            # Need to get proper design pattern for this
            # imputer = mice.MICEData(**kwargs)
            # fn = lambda df: imputer.fit_transform(df)
        return MissingDataImputer(label, fn, col_groups, method)


class HPImputer:

    def __init__(self, label, impute_fn, col_groups, normalize, oversample, method):
        self.label = label
        self.impute_fn = impute_fn
        self.col_groups = col_groups
        self.features = list(COLUMNS.loc[COLUMNS['Feature Group'].isin(col_groups)]['Column'].values)
        self.normalize = str_to_bool(normalize)
        self.oversample = oversample
        self.method = method
        self.folder_path = BASE_EXPERIMENT_PATH / f'hp_impute/{self.method}/{self.oversample}'
        self.data_imputed = False
        try:
            self.folder_path.mkdir(parents=True)
        except FileExistsError:
            val = input(f'Smite Directory {self.folder_path} (Y/N):')
            if val.lower() == 'y':
                rmtree(self.folder_path.absolute())
                self.folder_path.mkdir()
                print('Directory Smote')
            else:
                self.data_imputed = True
        self.fpath = self.folder_path / 'hp_imputed_train.csv'

    def get_only_hp(self, fname):
        self.data = pd.read_csv(BASE_EXPERIMENT_PATH / fname)
        self.hp_data = self.data.loc[~self.data['High_Performer'].isna()]
        self.train_data = self.hp_data[self.features]
        self.train_target = self.hp_data['High_Performer']
        self.test_data = self.data[self.features]

    def preprocess(self):
        if self.normalize:
            self.features = scale(self.train_data)
        if self.oversample.lower() == 'ros':
            ros = RandomOverSampler()
            self.features, self.target = ros.fit_resample(self.train_data, self.target)
        elif self.oversample.lower() == 'smote':
            smote = SMOTE()
            self.features, self.target = smote.fit_resample(self.train_data, self.target)
        elif self.oversample.lower() == 'adasyn':
            adasyn = ADASYN()
            self.train_data, self.train_target = adasyn.fit_resample(self.train_data, self.target)
        else:
            self.train_data, self.train_target = self.train_data, self.train_target

    def model(self):
        if self.impute_fn == 'NA':
            self.data.to_csv(self.fpath, index=False)
            return
        self.preprocess()
        fitted_model = self.impute_fn(self.train_data, self.train_target)
        predictions = pd.DataFrame(fitted_model.predict(self.test_data), columns=['Predicted_HP'])
        self.data = pd.concat([self.data, predictions], axis=1)
        self.data['Filled_High_Performer'] = self.data['Predicted_HP']
        self.data['Filled_High_Performer'][self.data['High_Performer'].notna()] = self.data['High_Performer'][
            self.data['High_Performer'].notna()]
        self.data['High_Performer'] = self.data['Filled_High_Performer']
        self.data.drop('Filled_High_Performer', axis=1)
        self.data.to_csv(self.fpath, index=False)

    @staticmethod
    def factory(method, col_groups, label, normalize, oversample, **kwargs):
        if method == 'None':
            clf = 'NA'
            fn = 'NA'
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
            clf = SVC(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'LR':
            clf = LogisticRegression(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        if method == 'EN':
            clf = ElasticNet(**kwargs)
            fn = lambda df, target: clf.fit(df, target)
        return HPImputer(label, fn, col_groups, normalize, oversample, method)


class Modeler:

    def __init__(self, label, model_fn, col_groups, method, target):
        self.label = label
        self.model_fn = model_fn
        self.col_groups = col_groups
        self.features = list(COLUMNS[COLUMNS['Feature Group'].isin(col_groups)].Column.values)
        self.method = method
        self.target = target
        self.folder_path = BASE_EXPERIMENT_PATH / f'model/{self.target}/{self.method}'
        try:
            self.folder_path.mkdir(parents=True)
        except FileExistsError:
            rmtree(self.folder_path.absolute())
            self.folder_path.mkdir()
            print('Directory Smote')

    def get_data(self, train_fname, test_fname):
        self.train_data = pd.read_csv(BASE_EXPERIMENT_PATH / train_fname)
        self.train_data['HP_Retained'] = np.floor((self.train_data['High_Performer'] + self.train_data['Retained'])/2)
        self.train_data['HP_Retained'] = self.train_data['HP_Retained'].fillna(0)
        self.train_data['Retained'] = self.train_data['Retained'].fillna(0)
        self.train_data['Protected_Group'] = self.train_data['Protected_Group'].fillna(0)
        self.train_features = self.train_data[self.features]
        self.test_data = pd.read_csv(BASE_EXPERIMENT_PATH / test_fname)
        self.test_features = self.test_data[self.features]

    def get_only_protected(self, train_fname, test_fname):
        self.train_data = pd.read_csv(BASE_EXPERIMENT_PATH / train_fname)
        self.train_data = self.train_data.loc[(self.train_data['Protected_Group'] == 1) &
                                              (~self.train_data['High_Performer'].isna())]
        self.train_data['Retained'] = self.train_data['Retained'].fillna(0)
        self.train_features = self.train_data[self.features]
        self.test_data = pd.read_csv(BASE_EXPERIMENT_PATH / test_fname)
        self.test_features = self.test_data[self.features]

    def get_only_not_null_hp(self, train_fname, test_fname):
        self.train_data = pd.read_csv(BASE_EXPERIMENT_PATH / train_fname)
        self.train_data = self.train_data.loc[~self.train_data['High_Performer'].isna()]
        self.train_data['HP_Retained'] = np.floor((self.train_data['High_Performer'] + self.train_data['Retained'])/2)
        self.train_data['HP_Retained'] = self.train_data['HP_Retained'].fillna(0)
        self.train_data['Retained'] = self.train_data['Retained'].fillna(0)
        self.train_data['Protected_Group'] = self.train_data['Protected_Group'].fillna(0)
        self.train_features = self.train_data[self.features]
        self.test_data = pd.read_csv(BASE_EXPERIMENT_PATH / test_fname)
        self.test_features = self.test_data[self.features]

    def model_eval(self, test_size):
        scorers = {'Accuracy': 'accuracy', 'ROC': 'roc_auc'}
        data_train, data_test, target_train, target_test = train_test_split(self.train_data[self.features],
                                                                            self.train_data[self.target],
                                                                            test_size=test_size)
        clf = self.model_fn
        scores = cross_validate(estimator=clf, X=data_train, y=target_train, scoring=scorers, cv=5)
        scores_Acc = scores['test_Accuracy']
        scores_AUC = scores['test_ROC']
        clf.fit(data_train, target_train)
        pred = clf.predict(data_test)
        cm = metrics.confusion_matrix(target_test, pred, normalize='true')
        out = pd.DataFrame({'metric': ['ACC', 'AUC', 'TP', 'TN', 'FP', 'FN'],
                            'mean': [scores_Acc.mean(), scores_AUC.mean(), cm[1][1], cm[0][0], cm[0][1], cm[1][0]],
                            'std': [scores_Acc.std(), scores_AUC.std(), 'NA', 'NA', 'NA', 'NA']})
        out.to_csv(self.folder_path / 'performance.csv', index=False)
        return out

    def model_predict(self):
        filtered_df, ignored_df = col_filter(self.col_groups, self.test_data, 'Test')
        clf = self.model_fn
        fitted_model = clf.fit(self.train_data[self.features], self.train_data[self.target])
        predictions = pd.DataFrame(fitted_model.predict_proba(filtered_df), columns=['Prob_Not_Target', 'predicted_Y'])
        self.predictions = pd.concat([ignored_df, filtered_df, predictions['predicted_Y']], axis=1)
        fpath = self.folder_path/'prediction.csv'
        self.predictions.to_csv(fpath, index=False)
        return fpath

        ## TO DO: Add code to save metadata to file

    @staticmethod
    def factory(method, target, col_groups, label, kwargs):
        if method == 'RF':
            clf = RandomForestClassifier(**kwargs)
        if method == 'GB':
            clf = GradientBoostingClassifier(**kwargs)
        if method == 'ADA':
            clf = AdaBoostClassifier(**kwargs)
        if method == 'SVM':
            clf = SVC(**kwargs)
        if method == 'LR':
            clf = LogisticRegression(**kwargs)
        if method == 'EN':
            clf = ElasticNet(**kwargs)
        return Modeler(label, clf, col_groups, method, target)


class Experiment:

    def __init__(self, experiment_json):
        self.timestamp = time.time()
        self.json = open(experiment_json)
        self.experiment = json.load(self.json)
        self.label = self.experiment['Label']
        self.file_dict = self.experiment['File_Dict']
        self.scenario_scoring = self.experiment['FeatureGeneration']['ScenarioScoring']
        self.judgement_scoring = self.experiment['FeatureGeneration']['JudgementScoring']
        self.personality_scoring = self.experiment['FeatureGeneration']['PersonalityScoring']
        self.bio_grouping = self.experiment['FeatureGeneration']['BioGrouping']
        self.md_imputing = self.experiment['MDImpute']
        self.hp_imputing = self.experiment['HPImpute']

    def setup(self):
        self.scenario_scorer = ScenarioScorer.factory(self.scenario_scoring['Method'])
        self.judgement_scorer = JudgementScorer.factory(self.judgement_scoring['Method'])
        self.personality_scorer = PersonalityScorer(self.personality_scoring['Method'])
        self.bio_grouper = BioGrouper.factory(self.bio_grouping['Method'])

        self.md_imputer = MissingDataImputer.factory(self.md_imputing['Method'], self.label,
                                                     self.md_imputing['Impute_Col_Groups'], self.md_imputing['Args'])
        self.hp_imputer = HPImputer.factory(self.hp_imputing['Method'], self.hp_imputing['Impute_Col_Groups'],
                                            self.hp_imputing['Args'], self.hp_imputing['Normalize'],
                                            self.hp_imputing['Oversample'])
        self.modeler = Modeler.factory(self.modeling['Method'], self.modeling['Target'],
                                       self.modeling['Model_Col_Groups'], self.label, self.modeling['Args'])

    def execute(self):
        ss_path = self.scenario_scorer.score(self.file_dict, False)
        js_path = self.judgement_scorer.score(self.file_dict, False)
        ps_path = self.personality_scorer.score(self.file_dict, False)
        bio_path = self.bio_grouper.score(self.file_dict, False)

        train_data_paths = [ss_path['Train'], js_path['Train'], ps_path['Train'], bio_path['Train']]
        test_data_paths = [ss_path['Test'], js_path['Test'], ps_path['Test'], bio_path['Test']]
        combined_training_file = combiner(train_data_paths, 'Train')
        combined_test_file = combiner(test_data_paths, 'Test')

        combined_files = {'Train': combined_training_file, 'Test': combined_test_file}

        if not self.md_imputer.data_imputed:
            self.md_imputer.impute(combined_files)
        if not self.hp_imputer.data_imputed:
            self.hp_imputer.get_data(self.md_imputer.fpaths['Train'])
            self.hp_imputer.preprocess()
            self.hp_imputer.model()
        self.modeler.get_data(self.hp_imputer.fpath, self.md_imputer.fpaths['Test'])
        self.scores = self.modeler.model_eval(self.modeling['TestSize'])
        self.modeler.model_predict(self.modeling['Target'])

    def record_metadata(self, scores):
        self.metadata = self.experiment
        self.metadata['Timestamp'] = self.timestamp
        self.metadata['Performance'] = scores.to_dict()
        with open(BASE_EXPERIMENT_PATH/'metadata.json', 'a+') as f:
            json.dump(self.metadata, f, indent=4)


class ExperimentThreeTier(Experiment):
    def setup(self):
        self.scenario_scorer = ScenarioScorer.factory(self.scenario_scoring['Method'])
        self.judgement_scorer = JudgementScorer.factory(self.judgement_scoring['Method'])
        self.personality_scorer = PersonalityScorer(self.personality_scoring['Method'])
        self.bio_grouper = BioGrouper.factory(self.bio_grouping['Method'])

        self.md_imputer = MissingDataImputer.factory(self.md_imputing['Method'], self.label,
                                                     self.md_imputing['Impute_Col_Groups'], self.md_imputing['Args'])
        self.hp_imputer = HPImputer.factory(self.hp_imputing['Method'], self.hp_imputing['Impute_Col_Groups'],
                                            self.hp_imputing['Args'], self.hp_imputing['Normalize'],
                                            self.hp_imputing['Oversample'])
        self.modeling1 = self.experiment['Model1']
        self.modeling2 = self.experiment['Model2']
        self.modeling3 = self.experiment['Model3']
        self.modeling4 = self.experiment['Model4']
        self.modeler1 = Modeler.factory(self.modeling1['Method'], self.modeling1['Target'],
                                       self.modeling1['Model_Col_Groups'], self.label, self.modeling1['Args'])
        self.modeler2 = Modeler.factory(self.modeling2['Method'], self.modeling2['Target'],
                                       self.modeling2['Model_Col_Groups'], self.label, self.modeling2['Args'])
        self.modeler3 = Modeler.factory(self.modeling3['Method'], self.modeling3['Target'],
                                       self.modeling3['Model_Col_Groups'], self.label, self.modeling3['Args'])
        self.modeler4 = Modeler.factory(self.modeling4['Method'], self.modeling4['Target'],
                                       self.modeling4['Model_Col_Groups'], self.label, self.modeling4['Args'])

    def execute(self):
        ss_path = self.scenario_scorer.score(self.file_dict, False)
        js_path = self.judgement_scorer.score(self.file_dict, False)
        ps_path = self.personality_scorer.score(self.file_dict, False)
        bio_path = self.bio_grouper.score(self.file_dict, False)

        train_data_paths = [ss_path['Train'], js_path['Train'], ps_path['Train'], bio_path['Train']]
        test_data_paths = [ss_path['Test'], js_path['Test'], ps_path['Test'], bio_path['Test']]
        combined_training_file = combiner(train_data_paths, 'Train')
        combined_test_file = combiner(test_data_paths, 'Test')

        combined_files = {'Train': combined_training_file, 'Test': combined_test_file}

        if not self.md_imputer.data_imputed:
            self.md_imputer.impute(combined_files)
        if not self.hp_imputer.data_imputed:
            self.hp_imputer.get_only_hp(self.md_imputer.fpaths['Train'])
            self.hp_imputer.preprocess()
            self.hp_imputer.model()

        # self.modeler1.get_data(self.hp_imputer.fpath, self.md_imputer.fpaths['Test'])
        self.modeler1.get_only_not_null_hp(self.hp_imputer.fpath, self.md_imputer.fpaths['Test'])
        self.scores1 = self.modeler1.model_eval(self.modeling1['TestSize'])
        model1preds = self.modeler1.model_predict()
        model1filter = pd.read_csv(model1preds)
        model1filter = model1filter.sort_values(by='predicted_Y', ascending=False).reset_index(drop=True)
        model1_out_fname = self.modeler1.folder_path / 'High Performer Probabilities.csv'
        model1filter.to_csv(model1_out_fname, index=False)
        self.record_metadata(self.scores1)

        self.modeler2.get_data(self.hp_imputer.fpath, self.md_imputer.fpaths['Test'])
        self.scores2 = self.modeler2.model_eval(self.modeling2['TestSize'])
        model2preds = self.modeler2.model_predict()
        model2filter = pd.read_csv(model2preds)
        model2filter = model2filter.sort_values(by='predicted_Y', ascending=False).reset_index(drop=True)
        model2_out_fname = self.modeler2.folder_path / 'Retained Probabilities.csv'
        model2filter.to_csv(model2_out_fname, index=False)
        self.record_metadata(self.scores2)

        self.modeler3.get_data(self.hp_imputer.fpath, self.md_imputer.fpaths['Test'])
        self.scores3 = self.modeler3.model_eval(self.modeling3['TestSize'])
        model3preds = self.modeler3.model_predict()
        model3filter = pd.read_csv(model3preds)
        model3filter = model3filter.sort_values(by='predicted_Y', ascending=False).reset_index(drop=True)
        model3_out_fname = self.modeler3.folder_path / 'Protected Group Probabilities.csv'
        model3filter.to_csv(model3_out_fname, index=False)
        self.record_metadata(self.scores3)

        self.modeler4.get_only_protected(self.hp_imputer.fpath, self.md_imputer.fpaths['Test'])
        self.scores4 = self.modeler4.model_eval(self.modeling4['TestSize'])
        model4preds = self.modeler4.model_predict()
        model4filter = pd.read_csv(model4preds)
        model4filter = model4filter.sort_values(by='predicted_Y', ascending=False).reset_index(drop=True)
        model4_out_fname = self.modeler4.folder_path / 'Protected High Performer Probabilities.csv'
        model4filter.to_csv(model4_out_fname, index=False)
        self.record_metadata(self.scores4)

        final_selections = model1filter[['UNIQUE_ID', 'predicted_Y']].merge(model2filter[['UNIQUE_ID', 'predicted_Y']],
                                                                          left_on='UNIQUE_ID', right_on='UNIQUE_ID')
        final_selections = final_selections.merge(model3filter[['UNIQUE_ID', 'predicted_Y']],
                                                  left_on='UNIQUE_ID', right_on='UNIQUE_ID')
        final_selections = final_selections.merge(model4filter[['UNIQUE_ID', 'predicted_Y']],
                                                  left_on='UNIQUE_ID', right_on='UNIQUE_ID')
        final_selections.columns = ['UNIQUE_ID', 'High_Performer', 'Retained', 'Protected', 'Protected_High_Performers']

        final_selections.to_csv('probabilities/ADA.csv')


exp = ExperimentThreeTier(BASE_EXPERIMENT_PATH / 'experiment.json')
exp.setup()
exp.execute()
