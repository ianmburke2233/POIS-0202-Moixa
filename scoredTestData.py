import pandas as pd

pd.set_option('mode.chained_assignment', None)


def scenarioScoring(scenario, raw):
    df = pd.DataFrame(raw[scenario])
    mode = int(df.mode().iloc[0][0])
    df['{}_score'.format(scenario)] = df[scenario].apply(lambda x: abs(x - mode))
    return pd.DataFrame(df['{}_score'.format(scenario)])


def scorer(x, best, worst):
    if x == best:
        return 1
    if x == worst:
        return -1
    else:
        return 0


def judgementScoring(sjItem, raw):
    df = raw[sjItem]
    counts = pd.DataFrame(df.value_counts())
    df = pd.DataFrame(df)
    counts = counts.reset_index()
    best = counts.iloc[0][0]
    worst = counts.iloc[-1][0]
    df['{}_score'.format(sjItem)] = df[sjItem].apply(lambda x: scorer(x, best, worst))
    return df['{}_score'.format(sjItem)]


x = pd.read_csv("C:/Users/Ian/Box Sync/Client/Baker Tilly/201901 ProdDev Initiative/"
                "2 - Work in Process/POIS-0202-Moixa/Data/TrainingData.csv")

ids = x[list(x.columns)[:8]]
SJTimes = x.filter(regex="SJ_Time.*", axis=1)
scenarioTimes = x.filter(regex="Scenario.*._Time.*", axis=1)
bioData = x.filter(regex="Bio.*", axis=1)

scenarios = ['Scenario1_1', 'Scenario1_2', 'Scenario1_3', 'Scenario1_4', 'Scenario1_5', 'Scenario1_6', 'Scenario1_7',
             'Scenario1_8',
             'Scenario2_1', 'Scenario2_2', 'Scenario2_3', 'Scenario2_4', 'Scenario2_5', 'Scenario2_6', 'Scenario2_7',
             'Scenario2_8']

for obj in scenarios:
    df1 = scenarioScoring(obj, x)
    if scenarios.index(obj) == 0:
        scenarioScores = df1
    else:
        scenarioScores = scenarioScores.join(df1)

mosts = x.filter(regex="SJ_Most.*", axis=1)

for column in list(mosts.columns):
    df1 = pd.DataFrame(judgementScoring(column, mosts))
    if list(mosts.columns).index(column) == 0:
        SJMostScores = df1
    else:
        SJMostScores = SJMostScores.join(df1)

leasts = x.filter(regex="SJ_Least.*", axis=1)

for column in list(leasts.columns):
    df2 = pd.DataFrame(judgementScoring(column, leasts))
    if list(leasts.columns).index(column) == 0:
        SJLeastScores = df2
    else:
        SJLeastScores = SJLeastScores.join(df2)

scale1 = x[['PScale01_Q1', 'PScale01_Q2', 'PScale01_Q3', 'PScale01_Q4']]
scale1['PScale01_Q2_Rev'] = scale1['PScale01_Q2'].apply(lambda x: 5 - x)
scale1['PScale01_Q3_Rev'] = scale1['PScale01_Q3'].apply(lambda x: 5 - x)
scale1['average_S1'] = (scale1['PScale01_Q1'] + scale1['PScale01_Q2_Rev'] + scale1['PScale01_Q3_Rev'] + scale1[
    'PScale01_Q4']) / 4

scale2 = x[['PScale02_Q1', 'PScale02_Q2', 'PScale02_Q3', 'PScale02_Q4']]
scale2['PScale02_Q2_Rev'] = scale2['PScale02_Q2'].apply(lambda x: 5 - x)
scale2['average_S2'] = (scale2['PScale02_Q1'] + scale2['PScale02_Q2_Rev'] + scale2['PScale02_Q3'] + scale2[
    'PScale02_Q4']) / 4

scale3 = x[['PScale03_Q1', 'PScale03_Q2', 'PScale03_Q3', 'PScale03_Q4']]
scale3['PScale03_Q1_Rev'] = scale3['PScale03_Q1'].apply(lambda x: 5 - x)
scale3['average_S3'] = (scale3['PScale03_Q1_Rev'] + scale3['PScale03_Q2'] + scale3['PScale03_Q3'] + scale3[
    'PScale03_Q4']) / 4

scale4 = x[['PScale04_Q1', 'PScale04_Q2', 'PScale04_Q3', 'PScale04_Q4']]
scale4['PScale04_Q1_Rev'] = scale4['PScale04_Q1'].apply(lambda x: 5 - x)
scale4['average_S4'] = (scale4['PScale04_Q1_Rev'] + scale4['PScale04_Q2'] + scale4['PScale04_Q3'] + scale4[
    'PScale04_Q4']) / 4

scale5 = x[['PScale05_Q1', 'PScale05_Q2', 'PScale05_Q3', 'PScale05_Q4']]
scale5['average_S5'] = (scale5['PScale05_Q1'] + scale5['PScale05_Q2'] + scale5['PScale05_Q3'] + scale5[
    'PScale05_Q4']) / 4

scale6 = x[['PScale06_Q1', 'PScale06_Q2', 'PScale06_Q3', 'PScale06_Q4', 'PScale06_Q5', 'PScale06_Q6']]
scale6['PScale06_Q3_Rev'] = scale6['PScale06_Q3'].apply(lambda x: 5 - x)
scale6['PScale06_Q6_Rev'] = scale6['PScale06_Q6'].apply(lambda x: 5 - x)
scale6['average_S6'] = (scale6['PScale06_Q1'] + scale6['PScale06_Q2'] +
                        scale6['PScale06_Q3_Rev'] + scale6['PScale06_Q4'] +
                        scale6['PScale06_Q5'] + scale6['PScale06_Q6_Rev']) / 4

scale7 = x[['PScale07_Q1', 'PScale07_Q2', 'PScale07_Q3', 'PScale07_Q4']]
scale7['PScale07_Q2_Rev'] = scale7['PScale07_Q2'].apply(lambda x: 5 - x)
scale7['average_S7'] = (scale7['PScale07_Q1'] + scale7['PScale07_Q2_Rev'] + scale7['PScale07_Q3'] + scale7[
    'PScale07_Q4']) / 4

scale8 = x[['PScale08_Q1', 'PScale08_Q2', 'PScale08_Q3', 'PScale08_Q4']]
scale8['PScale08_Q1_Rev'] = scale8['PScale08_Q1'].apply(lambda x: 5 - x)
scale8['PScale08_Q3_Rev'] = scale8['PScale08_Q3'].apply(lambda x: 5 - x)
scale8['average_S8'] = (scale8['PScale08_Q1_Rev'] + scale8['PScale08_Q2']
                        + scale8['PScale08_Q3_Rev'] + scale8['PScale08_Q4']) / 4

scale9 = x[['PScale09_Q1', 'PScale09_Q2', 'PScale09_Q3', 'PScale09_Q4']]
scale9['PScale09_Q2_Rev'] = scale9['PScale09_Q2'].apply(lambda x: 5 - x)
scale9['average_S9'] = (scale9['PScale09_Q1'] + scale9['PScale09_Q2_Rev']
                        + scale9['PScale09_Q3'] + scale9['PScale09_Q4']) / 4

scale10 = x[['PScale10_Q1', 'PScale10_Q2', 'PScale10_Q3', 'PScale10_Q4']]
scale10['PScale10_Q3_Rev'] = scale10['PScale10_Q3'].apply(lambda x: 5 - x)
scale10['PScale10_Q4_Rev'] = scale10['PScale10_Q4'].apply(lambda x: 5 - x)
scale10['average_S10'] = (scale10['PScale10_Q1'] + scale10['PScale10_Q2']
                          + scale10['PScale10_Q3_Rev'] + scale10['PScale10_Q4_Rev']) / 4

scale11 = x[['PScale11_Q1', 'PScale11_Q2', 'PScale11_Q3', 'PScale11_Q4']]
scale11['PScale11_Q2_Rev'] = scale11['PScale11_Q2'].apply(lambda x: 5 - x)
scale11['PScale11_Q3_Rev'] = scale11['PScale11_Q3'].apply(lambda x: 5 - x)
scale11['average_S11'] = (scale11['PScale11_Q1'] + scale11['PScale11_Q2_Rev']
                          + scale11['PScale11_Q3_Rev'] + scale11['PScale11_Q4']) / 4

scale12 = x[['PScale12_Q1', 'PScale12_Q2', 'PScale12_Q3', 'PScale12_Q4']]
scale12['PScale12_Q3_Rev'] = scale12['PScale12_Q3'].apply(lambda x: 5 - x)
scale12['PScale12_Q4_Rev'] = scale12['PScale12_Q4'].apply(lambda x: 5 - x)
scale12['average_S12'] = (scale12['PScale12_Q1'] + scale12['PScale12_Q2']
                          + scale12['PScale12_Q3_Rev'] + scale12['PScale12_Q4_Rev']) / 4

scale13 = x[['PScale13_Q1', 'PScale13_Q2', 'PScale13_Q3', 'PScale13_Q4', 'PScale13_Q5']]
scale13['PScale13_Q3_Rev'] = scale13['PScale13_Q3'].apply(lambda x: 5 - x)
scale13['PScale13_Q4_Rev'] = scale13['PScale13_Q4'].apply(lambda x: 5 - x)
scale13['average_S13'] = (scale13['PScale13_Q1'] + scale13['PScale13_Q2']
                          + scale13['PScale13_Q3_Rev'] + scale13['PScale13_Q4_Rev'] +
                          scale13['PScale13_Q5']) / 4

PScaleScores = pd.concat([scale1['average_S1'], scale2['average_S2'], scale3['average_S3'], scale4['average_S4'],
                          scale5['average_S5'], scale6['average_S6'], scale7['average_S7'], scale8['average_S8'],
                          scale9['average_S9'], scale10['average_S10'], scale11['average_S11'], scale12['average_S12'],
                          scale13['average_S13']], axis=1)

final = ids.join([bioData, SJTimes, SJMostScores, SJLeastScores, scenarioTimes, scenarioScores, PScaleScores])

final.to_csv('Data/scoredTrainingData.csv', index=False)
