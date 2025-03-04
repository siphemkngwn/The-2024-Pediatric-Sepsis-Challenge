mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix

rl = 'https://raw.githubusercontent.com/Kamaleswaran-Lab/The-2024-Pediatric-Sepsis-Challenge/refs/heads/main/SyntheticData_Training.csv'
df = pd.read_csv(url)

# Data Preprocessing
df = df.dropna(subset=['momagefirstpreg_adm'])

# Feature Engineering
age_bins = [0, 12, 36, 60, np.inf]
age_labels = ['infant', 'toddler', 'preschool', 'schoolgoing']
df['agecat'] = pd.cut(df['agecalc_adm'], bins=age_bins, labels=age_labels)

sc = ['height_cm_adm', 'muac_mm_adm', 'glucose_mmolpl_adm', 'lengthadm', 'rr_brpm_app_adm', 'weight_kg_adm', 'diasbp_mmhg_adm', 'sqi1_perc_oxi_adm', 'bcsverbal_adm', 'temp_c_adm', 'lactate_mmolpl_adm', 'hematocrit_gpdl_adm', 'bcsmotor_adm', 'sysbp_mmhg_adm', 'inhospital_mortality']
df = df[sc]

muac_bins = [0, 115, 125, np.inf]
muac_labels = ['sam', 'mam', 'normal']
df['muac_cat'] = pd.cut(df['muac_mm_adm'], bins=muac_bins, labels=muac_labels)

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'inhospital_mortality']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

numerical_features = df.select_dtypes(include=['number']).columns
imputer = SimpleImputer(strategy='mean')
df[numerical_features] = imputer.fit_transform(df[numerical_features])

x = df.drop('inhospital_mortality', axis=1)
y = df['inhospital_mortality']

# Model Training and Evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train_resampled, y_train_resampled)

y_pred = rf_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
