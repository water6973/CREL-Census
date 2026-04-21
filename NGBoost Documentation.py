import pandas as pd
from ngboost import NGBClassifier
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv("/Users/parthhalani/Downloads/POI_data_quarterly.csv")

# Sort by time to ensure chronological order
df = df.sort_values('CURRENT_QUARTER_START')

# Time-based train/test split (80% train, 20% test)
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# Define features and target
drop_cols = ['CLOSED_NEXT_QUARTER', 'NAME', 'BRAND', 'OPENED_ON', 
             'CLOSED_ON', 'ADDRESS', 'CURRENT_QUARTER', 
             'CURRENT_QUARTER_START', 'COUNTY', 'NAICS_CODE', 
             'NAICS_CODE_2022']
X_train = train.drop(columns=drop_cols)
Y_train = train['CLOSED_NEXT_QUARTER']
X_test = test.drop(columns=drop_cols)
Y_test = test['CLOSED_NEXT_QUARTER']

# Train the model
# Calculate how imbalanced the data is
n_open = (Y_train == 0).sum()
n_closed = (Y_train == 1).sum()
ratio = n_open / n_closed

# Tell the model to weight closures more heavily
from sklearn.utils import compute_sample_weight
weights = compute_sample_weight(class_weight='balanced', y=Y_train)
ngb = NGBClassifier().fit(X_train, Y_train, sample_weight=weights)

# Get predictions
Y_pred = ngb.predict(X_test)

# Measure precision and recall 
print(classification_report(Y_test, Y_pred))

# Get probabilities for each business
probabilities = ngb.predict_proba(X_test)
print(probabilities)