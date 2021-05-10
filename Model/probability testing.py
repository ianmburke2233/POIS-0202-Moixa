import pandas as pd
import numpy as np


def prob_selection(df, selection_method):
    prob_df = pd.DataFrame()
    prob_df['Retained_Probability'] = selection_method['Retained'](df)
    prob_df['High_Performer_Probability'] = selection_method['High_Performer'](df)
    prob_df['Protected_Probability'] = selection_method['Protected'](df)
    prob_df.insert(0, 'UNIQUE_ID', RAW_DATA['UNIQUE_ID'])
    return prob_df


def hire(prob_df, hired_list, unhired_list):
    if len(hired_list) == 1125:
        final_score = score(prob_df, hired_list)
        return hired_list, final_score
    else:
        best_score = -999999
        best_id = -1
        for unique_id in unhired_list:
            new_hired_list = hired_list + [unique_id]
            id_score = score(prob_df, new_hired_list)
            if id_score > best_score:
                best_score = id_score
                best_id = unique_id
        hired_list.append(best_id)
        unhired_list.remove(best_id)
        print(f'{round(len(hired_list) / 1125, 4) * 100}%, Best Score: {best_score}')
        return hire(prob_df, hired_list, unhired_list)


def score(df, hire_list):
    hired_df = df.loc[df['UNIQUE_ID'].isin(hire_list)]
    retained_score = hired_df['Retained_Probability'].sum() / len(hired_df)
    high_performer_score = hired_df['High_Performer_Probability'].sum() / len(hired_df)
    hp_retained_score = np.sum(hired_df['Retained_Probability'] * hired_df['High_Performer_Probability']) / len(hired_df)
    protected_selection_ratio = hired_df['Protected_Probability'].sum() / df['Protected_Probability'].sum()
    unprotected_selection_ratio = (1 - hired_df['Protected_Probability'].sum()) / \
                                  (1 - df['Protected_Probability'].sum())
    AIR = protected_selection_ratio / unprotected_selection_ratio
    AIR_score = abs(1-AIR) * 100
    expected_score = (retained_score * 25) + (high_performer_score * 25) + (hp_retained_score * 50) - AIR_score
    return expected_score


RAW_DATA = pd.read_csv('test_glmnet_all_probs.csv')

yes_cols = RAW_DATA.filter(regex=r'yes_*')

selection_method = {'Retained': lambda df: df['yes_ret_yhp_en'],
                    'High_Performer': lambda df: df['yes_hp_en'],
                    'Protected': lambda df: df['yes_pg_yhp_en']}

probs = prob_selection(RAW_DATA, selection_method)
# hire_list, final_score = hire(probs, hired_list=[], unhired_list=list(probs.UNIQUE_ID.values))

hires = RAW_DATA.sort_values(by='yes_hp_en').reset_index(drop=True)

hire_list = list(hires.UNIQUE_ID.loc[:1124].values)

best = pd.read_csv('best.csv')
mar5 = pd.read_csv('mar 5.csv')
mar6 = pd.read_csv('mar 6.csv')
mar12 = pd.read_csv('mar 12.csv')

compare = best.merge(mar5, left_on='UNIQUE_ID', right_on='UNIQUE_ID')
compare = compare.merge(mar6, left_on='UNIQUE_ID', right_on='UNIQUE_ID')
compare = compare.merge(mar12, left_on='UNIQUE_ID', right_on='UNIQUE_ID')
compare.columns = ['UNIQUE_ID', 'Best', 'Mar5', 'Mar6', 'Mar12']
compare['Number Selected'] = compare['Best'] + compare['Mar5'] + compare['Mar6'] + compare['Mar12']

pre_hired = list(compare.loc[compare['Number Selected'] > 2].UNIQUE_ID.values)
edge_cases = list(compare.loc[compare['Number Selected'] == 2].UNIQUE_ID.values)

hired, final_score = hire(probs, pre_hired, edge_cases)

compare['Hire'] = compare['UNIQUE_ID'].apply(lambda x: 1 if x in hired else 0)

compare[['UNIQUE_ID', 'Hire']].to_csv('final_selections/ensemble_v6.csv')
