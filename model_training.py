from asyncio.windows_events import NULL
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import os
import matplotlib.pyplot as plt
import seaborn as sns

# create confusion matrix and save as png
def create_confusion_matrix(name):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(name, dpi=150)

# create feature importance dataframe
def create_feature_importance_df():
    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

# Drop difficulty level column (not a feature)
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

df_full = pd.concat([df_train, df_test])

cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

target_encoder = LabelEncoder()
df_full['category_encoded'] = target_encoder.fit_transform(df_full['category'])

train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category', 'category_encoded'])
y_train = df_train_processed['category_encoded']

X_test = df_test_processed.drop(columns=['category', 'category_encoded'])
y_test = df_test_processed['category_encoded']

print(f"\nFeatures: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

class_names = target_encoder.classes_

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================
#
# You now have:
#   X_train, y_train  — training features and labels (5 categories)
#   X_test, y_test    — test features and labels (5 categories)
#
# Your task:
#   1. Train one or more models on X_train / y_train
#   2. Predict on X_test
#   3. Evaluate using macro F1-score
#
# Useful imports for evaluation:
#   from sklearn.metrics import classification_report, confusion_matrix, f1_score
#
# To compute macro F1:
#   f1_score(y_test, y_pred, average='macro')

# ==============================================================
# 1. CLASS IMBALANCE HANDLING
# ==============================================================

# SMOTEENN was determined to be the most effective data resampling technique (gave the highest f1 score)
smoteenn = SMOTEENN(random_state=42, n_jobs=-1)
X_train_resampled_smoteenn, y_train_resampled_smoteenn = smoteenn.fit_resample(X_train, y_train)

# ==============================================================
# 2. BASELINE TEST TO DETERMINE UNIMPORTANT FEATURES AND REMOVE THEM
# ==============================================================

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_resampled_smoteenn, y_train_resampled_smoteenn)

y_pred = xgb_model.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average='macro')

feature_importance_df = create_feature_importance_df()

# if a feature has a threshold of < 0.01 (or 1%), then it is removed from the data set.
threshold = 0.01
important_features = feature_importance_df[feature_importance_df["importance"] >= threshold]["feature"].tolist()
X_train_resampled_smoteenn = X_train_resampled_smoteenn[important_features]
X_test = X_test[important_features]

# ==============================================================
# 4. HYPERPARAMETER TUNING AND RandomSearchCV
# ==============================================================

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 1, 2, 3],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 2, 5],
    "reg_alpha": [0, 1, 2]
}

xgb_model = xgb.XGBClassifier(random_state=42)
grid_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    scoring="f1_macro",
    cv=3,
    verbose=1,
    n_iter=20,
    random_state=42
)

grid_search.fit(
    X_train_resampled_smoteenn, y_train_resampled_smoteenn
)

# ==============================================================
# 5. FINALIST MODEL FITTING
# ==============================================================

final_model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(
    X_train_resampled_smoteenn, y_train_resampled_smoteenn
)

# ==============================================================
# 6. FINALIST MODEL EVALUATION
# ==============================================================

y_pred = final_model.predict(X_test)
final_f1 = f1_score(y_test, y_pred, average='macro')

output_file = open("classification_report.txt", "w")

print("Classification Report:", file=output_file)
print(classification_report(y_test, y_pred, target_names=class_names), file=output_file)
print(f"Cross-Validation F1-Score: {grid_search.best_score_:.4f}", file=output_file)
print(f"Macro F1-Score: {final_f1:.4f}", file=output_file)
print(f"\nBest Parameters: {grid_search.best_params_}", file=output_file)

output_file.close()

create_confusion_matrix(f"confusion_matrix.png")

