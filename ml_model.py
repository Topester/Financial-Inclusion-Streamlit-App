import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pickle

df = pd.read_csv('data/Financial_inclusion_dataset.csv')
print(df.head())

def clean_input(value):
    if isinstance(value, str):  # Check if the value is a string
        return value.strip()  # Remove leading/trailing spaces from strings
    return value  # Return the value as-is if it's not a string

df = df.applymap(clean_input)

# Function to clean column names by replacing spaces with underscores
def clean_column_names(df):
    df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores in column names
    return df


df = clean_column_names(df)

df.info()
#df['bank_account'] = df['bank_account'].map({
#    'no': 0,
#   'yes': 1,
 #   })

df.drop(columns=['year','uniqueid'],axis=1,inplace=True)
#df = pd.get_dummies(df[['country','location_type','cellphone_access','household_size','age_of_respondent','gender_of_respondent','relationship_with_head','marital_status','education_level','job_type']],dtype=int)
df = pd.get_dummies(df,drop_first=True,dtype = int)
y = df['bank_account_Yes']
X = df.drop(columns=['bank_account_Yes'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


y_pred = model.predict(X_test)
print(y_pred)

score = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {score}")

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully.")

try:
    with open("model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    print("Pickle file not found!")


# Save the trained model
#joblib.dump(model, "model/model.pkl")

# Load the trained model (for testing purposes)
#loaded_model = joblib.load("model/model.pkl")

#if not os.path.exists("model"):
 #   os.makedirs("model")
#joblib.dump(model, "model/model.pkl")

# Load the model for testing
#loaded_model = joblib.load("model/model.pkl")
#test_pred = loaded_model.predict(X_test)
#print(f"Loaded model accuracy: {accuracy_score(y_test, test_pred)}")