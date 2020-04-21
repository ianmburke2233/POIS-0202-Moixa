from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from urllib.parse import quote_plus
import json
import pypyodbc as odbc
from sqlalchemy import create_engine

# Config
settings = json.load(open(
    'C:/Users/Ian/Box Sync/Client/Baker Tilly/201901 ProdDev Initiative/2 - Work in Process/IB Working/Scoring Model/config/secret.json'))
username = settings['username']
password = settings['password']

connStr = (r'Driver={ODBC Driver 13 for SQL Server};'
           r'Server=tcp:axiompd1.database.windows.net,1433;'
           r'Database=PD_BTVK;'
           r'Uid=' + username + ';'
                                r'Pwd=' + password + ';'
                                                     r'Encrypt=yes;'
                                                     r'TrustServerCertificate=no;')
quoted = quote_plus(connStr)
new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
engine = create_engine(new_con, fast_executemany=True)
conn = odbc.connect("Driver={ODBC Driver 13 for SQL Server};"
                    "Server=tcp:axiompd1.database.windows.net,1433;"
                    "Database=PD_BTVK;"
                    "Uid=" + username + ";"
                                        "Pwd=" + password + ";"
                                                            "Encrypt=yes;")


def Corr():
    data = pd.read_csv("C:/Users/Ian/Downloads/Grouped Data.csv")
    data = data.fillna(data.mean())
    df1 = pd.get_dummies(data['Grouping'], prefix="group")
    data = pd.concat([data, df1], axis=1)
    columns = list(data.columns)
    removalList = ['UNIQUE_ID',
                   'High_Performer',
                   'Protected_Group',
                   'Retained',
                   'Split',
                   'Grouping']
    num_columns = []
    for col in columns:
        if col not in removalList:
            num_columns.append(col)
    df4 = data[num_columns]
    df5 = data[num_columns]
    coeffmat = np.zeros((df4.shape[1], df5.shape[1]))
    pvalmat = np.zeros((df4.shape[1], df5.shape[1]))
    for i in range(df4.shape[1]):
        for j in range(df5.shape[1]):
            corrtest = pearsonr(df4[df4.columns[i]], df5[df5.columns[j]])

            coeffmat[i, j] = corrtest[0]
            pvalmat[i, j] = corrtest[1]

    dfcoeff = pd.DataFrame(coeffmat, columns=df5.columns, index=df4.columns)
    dfcoeff.insert(0, 'features', num_columns)

    results = dfcoeff[['features',
                       'group_1',
                       'group_2',
                       'group_3',
                       'group_4',
                       'group_5',
                       'group_6',
                       'group_7',
                       'group_8']]

    results.to_csv("groupCorrelations.csv", index=False)


Corr()
