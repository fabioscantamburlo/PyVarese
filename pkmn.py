import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/Pokemon.csv').rename(columns={'#': 'Number', 'Sp. Atk': 'Spatk',
                                                    'Sp. Def': 'Spdef', 'Type 1': 'Type1',
                                                    'Type 2': 'Type2'})

simple_imputer = SimpleImputer(missing_values='nan', strategy='constant', fill_value='Pure')
onehot_type1 = OneHotEncoder(sparse_output=False)
onehot_type2 = OneHotEncoder(sparse_output=False)
sc = StandardScaler()
le_tgt = LabelEncoder()

df['Type1'] = df['Type1'].astype(str)
df['Type2'] = df['Type2'].astype(str)
df['Type2'] = simple_imputer.fit_transform(df[['Type2']]).ravel()

df_encoded_type1 = pd.DataFrame(onehot_type1.fit_transform(df[['Type1']]),
                                columns=onehot_type1.get_feature_names_out(['Type1']))

df_encoded_type2 = pd.DataFrame(onehot_type2.fit_transform(df[['Type2']]),
                                columns=onehot_type2.get_feature_names_out(['Type2']))
df_enc = pd.concat([df_encoded_type1, df_encoded_type2], axis=1)

to_std_features = ['Total', 'HP', 'Attack', 'Defense', 'Spatk', 'Spdef', 'Speed']
num_features = ['Total', 'HP', 'Attack', 'Defense', 'Spatk', 'Spdef', 'Speed']
features = num_features + ['Type1', 'Type2']
columns_encoded = list(df_enc.columns)

df = pd.concat([df, df_enc], axis=1)

df = df.assign(y_enc = le_tgt.fit_transform(df['Legendary'].values.ravel()))

X_train, X_test, y_train, y_test = train_test_split(df[features+columns_encoded],
                                                    df[['y_enc']], test_size=0.1, random_state=42)
tr = HistGradientBoostingClassifier()
tr = tr.fit(X=X_train, y=y_train.to_numpy().ravel())
tr.predict(X_test)
