import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the golf dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
            'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\n")

# Calculate prior probabilities P(Play=Yes) and P(Play=No)
play_counts = df['Play'].value_counts()
total_instances = len(df)
p_yes = play_counts['Yes'] / total_instances
p_no = play_counts['No'] / total_instances

print(f"Total instances: {total_instances}")
print(f"Number of 'Yes': {play_counts['Yes']}")
print(f"Number of 'No': {play_counts['No']}")
print(f"P(Play=Yes) = {play_counts['Yes']}/{total_instances} = {p_yes:.4f}")
print(f"P(Play=No) = {play_counts['No']}/{total_instances} = {p_no:.4f}")
print("\n")

# Calculate likelihood tables for each feature given class
def calculate_likelihood_table(df, feature):
    feature_counts = df.groupby([feature, 'Play']).size().unstack(fill_value=0)
    
    yes_total = play_counts['Yes']
    no_total = play_counts['No']
    
    # Add formula-style strings
    likelihood_table = pd.DataFrame({
        'P(x|Yes)': [f"{feature_counts.at[idx, 'Yes']}/{yes_total} = {feature_counts.at[idx, 'Yes']/yes_total:.4f}" 
                     for idx in feature_counts.index],
        'P(x|No)': [f"{feature_counts.at[idx, 'No']}/{no_total} = {feature_counts.at[idx, 'No']/no_total:.4f}" 
                    for idx in feature_counts.index],
        'Total': [f"{(feature_counts.at[idx, 'Yes'] + feature_counts.at[idx, 'No'])}/{total_instances} = {(feature_counts.at[idx, 'Yes'] + feature_counts.at[idx, 'No'])/total_instances:.4f}" 
                  for idx in feature_counts.index],
        'Count|Yes': feature_counts['Yes'],
        'Count|No': feature_counts['No'],
        'Total Count': feature_counts['Yes'] + feature_counts['No']
    }, index=feature_counts.index)
    
    return likelihood_table

# Generate and display likelihood tables for all features
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
likelihood_tables = {}

for feature in features:
    likelihood_tables[feature] = calculate_likelihood_table(df, feature)
    print(f"Likelihood Table for {feature}:")
    print(likelihood_tables[feature])
    print("\n")

# Function to predict using Naive Bayes
def naive_bayes_predict(new_instance, likelihood_tables, p_yes, p_no):
    p_x_given_yes = 1.0
    p_x_given_no = 1.0
    
    yes_steps = []
    no_steps = []
    
    for feature, value in new_instance.items():
        if value in likelihood_tables[feature].index:
            yes_formula = likelihood_tables[feature].loc[value, 'P(x|Yes)']
            no_formula = likelihood_tables[feature].loc[value, 'P(x|No)']
            
            # Extract the numerical value from the formula (after '=' sign)
            yes_prob = float(yes_formula.split('=')[-1].strip())
            no_prob = float(no_formula.split('=')[-1].strip())
            
            p_x_given_yes *= yes_prob
            p_x_given_no *= no_prob
            
            yes_steps.append(f"P({feature}={value}|Yes) = {yes_formula}")
            no_steps.append(f"P({feature}={value}|No) = {no_formula}")
        else:
            # Handle unseen values
            p_x_given_yes = 0
            p_x_given_no = 0

    p_yes_given_x_unnormalized = p_x_given_yes * p_yes
    p_no_given_x_unnormalized = p_x_given_no * p_no
    
    p_x = p_yes_given_x_unnormalized + p_no_given_x_unnormalized
    
    p_yes_given_x = p_yes_given_x_unnormalized / p_x if p_x > 0 else 0
    p_no_given_x = p_no_given_x_unnormalized / p_x if p_x > 0 else 0
    
    # Print detailed calculation steps
    print("\nDetailed Calculations:")
    
    print("P(X|Yes) = ", end="")
    for step in yes_steps:
        print(f"{step} × ", end="")
    print(f"= {p_x_given_yes:.6f}")
    
    print(f"P(Yes|X) = P(X|Yes) × P(Yes) = {p_x_given_yes:.6f} × {p_yes:.6f} = {p_yes_given_x_unnormalized:.6f}")
    
    print("P(X|No) = ", end="")
    for step in no_steps:
        print(f"{step} × ", end="")
    print(f"= {p_x_given_no:.6f}")
    
    print(f"P(No|X) = P(X|No) × P(No) = {p_x_given_no:.6f} × {p_no:.6f} = {p_no_given_x_unnormalized:.6f}")
    
    print(f"P(X) = {p_yes_given_x_unnormalized:.6f} + {p_no_given_x_unnormalized:.6f} = {p_x:.6f}")
    
    print(f"P(Yes|X) = {p_yes_given_x_unnormalized:.6f} / {p_x:.6f} = {p_yes_given_x:.6f}")
    print(f"P(No|X) = {p_no_given_x_unnormalized:.6f} / {p_x:.6f} = {p_no_given_x:.6f}")
    
    prediction = "Yes" if p_yes_given_x > p_no_given_x else "No"
    
    return {
        "P(Yes|X)": p_yes_given_x,
        "P(No|X)": p_no_given_x,
        "Prediction": prediction
    }

# Test with the example
new_instance = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

print("Predicting for new instance:")
for feature, value in new_instance.items():
    print(f"{feature} = {value}")
print("\n")

# Make prediction
prediction_results = naive_bayes_predict(new_instance, likelihood_tables, p_yes, p_no)
print("\nFinal Results:")
for key, value in prediction_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")
