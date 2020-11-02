import pandas as pd


def hire(mid, x):
    if x > mid:
        decision = 1
    else:
        decision = 0
    return decision


fname = '2020-10-30 hp-RF ret-GB Combined (ADA)'
protected_splits = 0
print_results = 1

if protected_splits == 1:
    prot_hp = pd.read_csv('Selections/2020-09-18 Random Forest (Protected Group, High Performers Only).csv')
    prot_r = pd.read_csv('Selections/2020-09-18 Random Forest (Protected Group, Retained Only).csv')
    nprot_hp = pd.read_csv('Selections/2020-09-18 Random Forest (Non-Protected Group, High Performers Only).csv')
    nprot_r = pd.read_csv('Selections/2020-09-18 Random Forest (Non-Protected Group, Retained Only).csv')
    prot_classification = pd.read_csv('Data/Predicted Protected Status.csv')

    prot = prot_hp.merge(prot_r, left_on='UNIQUE_ID', right_on='UNIQUE_ID', suffixes={'_hp', '_r'})
    nprot = nprot_hp.merge(nprot_r, left_on='UNIQUE_ID', right_on='UNIQUE_ID', suffixes={'_hp', '_r'})

    prot['product'] = prot['Class 2 Prob_r'] * prot['Class 2 Prob_hp']
    nprot['product'] = nprot['Class 2 Prob_r'] * nprot['Class 2 Prob_hp']

    protected = prot_classification[prot_classification['Protected'] == 1].\
        merge(prot, left_on='UNIQUE_ID', right_on='UNIQUE_ID')
    non_protected = prot_classification[prot_classification['Protected'] == 0].\
        merge(nprot, left_on='UNIQUE_ID', right_on='UNIQUE_ID')

    prot_mid = protected['product'].median()
    nprot_mid = non_protected['product'].median()

    protected['Hire'] = protected['product'].apply(lambda x: hire(prot_mid, x))
    protected = protected.sort_values(by='product').reset_index(drop=True)
    protected.at[562, 'Hire'] = 1
    non_protected['Hire'] = non_protected['product'].apply(lambda x: hire(nprot_mid, x))

    out = protected[['UNIQUE_ID', 'Hire']].append(non_protected[['UNIQUE_ID', 'Hire']])
else:
    hp = pd.read_csv('Selections/2020-10-02 Random Forest (Full Pop, High Performer).csv')
    retained = pd.read_csv('Selections/2020-10-23 GB (Full Pop, Retained).csv')
    hybrid = hp.merge(retained, left_on='UNIQUE_ID', right_on='UNIQUE_ID', suffixes={'_hp', '_r'})
    hybrid['product'] = hybrid['Class 2 Prob_r'] * hybrid['Class 2 Prob_hp']
    median = hybrid['product'].median()
    hybrid['Hire'] = hybrid['product'].apply(lambda x: hire(median, x))
    counts = hybrid['Hire'].value_counts()
    if counts[0] < counts[1]:
        hybrid.at[562, 'Hire'] = 1
    elif counts[0] > counts[1]:
        hybrid.at[563, 'Hire'] = 1
    out = hybrid[['UNIQUE_ID', 'Hire']]

if print_results == 1:
    out.to_csv('Selections/{}.csv'.format(fname), index=False)


