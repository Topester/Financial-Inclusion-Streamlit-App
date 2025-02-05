import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.title("Financial Inclusion Classifier")
st.write(
    "This app predicts whether an individual is likely to have a bank account. Fill in the form below to get started."
)



file_path = r"C:\Users\ADMIN\PycharmProjects\PythonProject_StreamlitApps\Financial_Inclusion_Chackpoint\model.pkl"

try:
    with open(file_path, "rb") as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at: {file_path}")
    st.stop()


filepath = r"C:/Users/ADMIN/PycharmProjects/PythonProject_StreamlitApps/Financial_Inclusion_Chackpoint/data/Financial_inclusion_dataset.csv"
data = pd.read_csv(filepath)

data['bank_account'] = data['bank_account'].map({'Yes': 1, 'No': 0})

# Count of people with and without bank accounts per education level
education_counts = data.groupby('education_level')['bank_account'].value_counts().unstack()


fig, ax = plt.subplots(figsize=(10, 6))
education_counts.plot(kind='bar', stacked=False, color=['blue', 'yellow'], ax=ax)
plt.xlabel("Level of Education")
plt.ylabel("Number of People")
plt.title("Bank Account Ownership by Education Level")
plt.legend(["No Bank Account", "Has Bank Account"])
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)
st.write("Number of people with and without bank accounts per education level:")
st.write(education_counts)

#data['bank_account'] = data['bank_account'].map({'Yes': 1, 'No': 0})

# Count of people with and without bank accounts per education level
employment_counts = data.groupby('job_type')['bank_account'].value_counts().unstack()


fig, ax = plt.subplots(figsize=(10, 6))
employment_counts.plot(kind='bar', stacked=False, color=['blue', 'yellow'], ax=ax)
plt.xlabel("Job Type")
plt.ylabel("Number of People")
plt.title("Bank Account Ownership by Job Type")
plt.legend(["No Bank Account", "Has Bank Account"])
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)
st.write("Number of people with and without bank accounts per job type education level:")
st.write(employment_counts)



st.title("Bank Account Ownership by Country and Job Type")

# Encode 'Yes' and 'No' as 1 and 0 for numerical analysis
#data['bank_account'] = data['bank_account'].map({'Yes': 1, 'No': 0})

# Group data by country and job type
country_counts = data.groupby(['country', 'job_type'])['bank_account'].value_counts().unstack(fill_value=0)

# Rename columns to avoid KeyError
country_counts.columns = ['No Bank Account', 'Has Bank Account']

# Select country from dropdown
selected_country = st.selectbox("Select a country:", data['country'].unique())

# Filter data for the selected country
filtered_data = country_counts.loc[selected_country]

# Plot pie chart
fig, ax = plt.subplots(figsize=(8, 6))
filtered_data.sum().plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'yellow'], ax=ax)
plt.ylabel("")  # Hide y-label
plt.title(f"Bank Account Ownership in {selected_country}")

# Show plot in Streamlit
st.pyplot(fig)

# Display data table
st.write(f"Bank Account Ownership Data for {selected_country}:")
st.write(filtered_data)
st.info("As exhibited in the data above, a huge portion of the population do not have bank accounts in the countries they come from, their level of education or employment type")


# User input section
st.header("User Input Parameters")


def user_input_features():
    # Country
    country = st.selectbox("Country", ['Rwanda', 'Tanzania', 'Uganda', 'Kenya'])
    country_Rwanda = 1 if country == 'Rwanda' else 0
    country_Tanzania = 1 if country == 'Tanzania' else 0
    country_Uganda = 1 if country == 'Uganda' else 0
    country_Kenya = 1 if country == 'Kenya' else 0

    # Year (assuming it's relevant for prediction)
    year = st.slider("Year", 2010, 2019)

    # Location Type
    location_type = st.selectbox("Location Type", ['Rural', 'Urban'])
    location_type_Urban = 1 if location_type == 'Urban' else 0
    location_type_Rural = 1 if location_type == 'Rural' else 0

    # Cellphone Access
    cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])
    cellphone_access_Yes = 1 if cellphone_access == 'Yes' else 0
    cellphone_access_No = 1 if cellphone_access == 'No' else 0

    # Household Size
    household_size = st.slider("Household Size", 1, 20)

    # Age of Respondent
    age_of_respondent = st.slider("Age of Respondent", 18, 100)

    # Gender
    gender = st.selectbox("Gender", ['Male', 'Female'])
    gender_of_respondent_Male = 1 if gender == 'Male' else 0
    gender_of_respondent_Female = 1 if gender == 'Female' else 0

    # Relationship with Head
    relationship_with_head = st.selectbox(
        "Relationship with Head",
        ['Child', 'Head of Household', 'Other non-relatives',
         'Other relative', 'Parent', 'Spouse']
    )
    relationship_with_head_Child = 1 if relationship_with_head == 'Child' else 0
    relationship_with_head_Head_of_Household = 1 if relationship_with_head == 'Head of Household' else 0
    relationship_with_head_Other_non_relatives = 1 if relationship_with_head == 'Other non-relatives' else 0
    relationship_with_head_Other_relative = 1 if relationship_with_head == 'Other relative' else 0
    relationship_with_head_Parent = 1 if relationship_with_head == 'Parent' else 0
    relationship_with_head_Spouse = 1 if relationship_with_head == 'Spouse' else 0

    # Marital Status
    marital_status = st.selectbox(
        "Marital Status",
        ['Don’t know', 'Married/Living together', 'Single/Never Married',
         'Widowed', 'Divorced/Separated']
    )
    marital_status_Dont_know = 1 if marital_status == 'Don’t know' else 0
    marital_status_Married_Living_together = 1 if marital_status == 'Married/Living together' else 0
    marital_status_Single_Never_Married = 1 if marital_status == 'Single/Never Married' else 0
    marital_status_Widowed = 1 if marital_status == 'Widowed' else 0
    marital_status_Divorced_Separated = 1 if marital_status == 'Divorced/Separated' else 0

    # Education Level
    education_level = st.selectbox(
        "Education Level",
        ['Other/Don’t know/RTA', 'Primary education', 'Secondary education',
         'Tertiary education', 'Vocational/Specialised training', 'No formal education']
    )
    education_level_Other_Dont_know_RTA = 1 if education_level == 'Other/Don’t know/RTA' else 0
    education_level_Primary_education = 1 if education_level == 'Primary education' else 0
    education_level_Secondary_education = 1 if education_level == 'Secondary education' else 0
    education_level_Tertiary_education = 1 if education_level == 'Tertiary education' else 0
    education_level_Vocational_Specialised_training = 1 if education_level == 'Vocational/Specialised training' else 0
    education_level_No_formal_education = 1 if education_level == 'No formal education' else 0

    # Job Type
    job_type = st.selectbox(
        "Job Type",
        ['Farming and Fishing', 'Formally employed Government', 'Formally employed Private',
         'Government Dependent', 'Informally employed', 'No Income',
         'Other Income', 'Remittance Dependent', 'Self employed']
    )
    job_type_Farming_and_Fishing = 1 if job_type == 'Farming and Fishing' else 0
    job_type_Formally_employed_Government = 1 if job_type == 'Formally employed Government' else 0
    job_type_Formally_employed_Private = 1 if job_type == 'Formally employed Private' else 0
    job_type_Government_Dependent = 1 if job_type == 'Government Dependent' else 0
    job_type_Informally_employed = 1 if job_type == 'Informally employed' else 0
    job_type_No_Income = 1 if job_type == 'No Income' else 0
    job_type_Other_Income = 1 if job_type == 'Other Income' else 0
    job_type_Remittance_Dependent = 1 if job_type == 'Remittance Dependent' else 0
    job_type_Self_employed = 1 if job_type == 'Self employed' else 0

    # Combine into a DataFrame
    features = pd.DataFrame([{
        'household_size': household_size,
        'age_of_respondent': age_of_respondent,
        'country_Rwanda': country_Rwanda,
        'country_Tanzania': country_Tanzania,
        'country_Uganda': country_Uganda,
        'country_Kenya': country_Kenya,
        'location_type_Urban': location_type_Urban,
        'location_type_Rural': location_type_Rural,
        'cellphone_access_Yes': cellphone_access_Yes,
        'cellphone_access_No': cellphone_access_No,
        'gender_of_respondent_Male': gender_of_respondent_Male,
        'gender_of_respondent_Female': gender_of_respondent_Female,
        'relationship_with_head_Child': relationship_with_head_Child,
        'relationship_with_head_Head_of_Household': relationship_with_head_Head_of_Household,
        'relationship_with_head_Other_non_relatives': relationship_with_head_Other_non_relatives,
        'relationship_with_head_Other_relative': relationship_with_head_Other_relative,
        'relationship_with_head_Parent': relationship_with_head_Parent,
        'relationship_with_head_Spouse': relationship_with_head_Spouse,
        'marital_status_Dont_know': marital_status_Dont_know,
        'marital_status_Married_Living_together': marital_status_Married_Living_together,
        'marital_status_Single_Never_Married': marital_status_Single_Never_Married,
        'marital_status_Widowed': marital_status_Widowed,
        'marital_status_Divorced_Separated': marital_status_Divorced_Separated,
        'education_level_Other_Dont_know_RTA': education_level_Other_Dont_know_RTA,
        'education_level_Primary_education': education_level_Primary_education,
        'education_level_Secondary_education': education_level_Secondary_education,
        'education_level_Tertiary_education': education_level_Tertiary_education,
        'education_level_Vocational_Specialised_training': education_level_Vocational_Specialised_training,
        'education_level_No_formal_education': education_level_No_formal_education,
        'job_type_Farming_and_Fishing': job_type_Farming_and_Fishing,
        'job_type_Formally_employed_Government': job_type_Formally_employed_Government,
        'job_type_Formally_employed_Private': job_type_Formally_employed_Private,
        'job_type_Government_Dependent': job_type_Government_Dependent,
        'job_type_Informally_employed': job_type_Informally_employed,
        'job_type_No_Income': job_type_No_Income,
        'job_type_Other_Income': job_type_Other_Income,
        'job_type_Remittance_Dependent': job_type_Remittance_Dependent,
        'job_type_Self_employed': job_type_Self_employed
    }])

    return features



# Get user input
input_data = user_input_features()

# Encode features dynamically
encoded_data = pd.get_dummies(input_data, drop_first=True)

# Align columns with model input
model_columns = loaded_model.feature_names_in_
encoded_data = encoded_data.reindex(columns=model_columns, fill_value=0)

# Prediction
prediction = loaded_model.predict(encoded_data)[0]
prediction_text = "Yes" if prediction == 1 else "No"

if st.button("Submit"):
    # Predict only when the button is clicked
    prediction = loaded_model.predict(encoded_data)[0]
    st.write("Prediction:")
    st.write(prediction)



# Display the prediction result
st.write(f"The prediction is: {prediction_text}")
