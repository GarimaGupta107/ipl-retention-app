import streamlit as st
import pandas as pd
import joblib

model=joblib.load("rf_retention_model.pkl")
data=pd.read_csv("test_data.csv")

features = [
    "Age", "IsExperienced", "Batting_Impact", "Bowling_Impact",
    "WasRetainedConsistently", "WasRetainedBefore", "RunsPerAge",
    "PerformancePerAge", "RecentWeightedRuns", "RetentionTrend",
    "TeamStrength", "RecentWeightedWickets"
]
st.title("🏏 IPL Player Retention Predictor 2026")
st.write("Select a player and their team to predict if they will be retained.")

player_name=st.selectbox("Select Player",sorted(data['Player'].unique()))
team_name=st.selectbox("Select Team",sorted(data['Current_Team'].unique()))

if st.button("Predict Retention"):
   row=data[(data['Player']==player_name)&(data['Current_Team']==team_name)]

   if row.empty:
      st.error("⚠️ Player not found with selected team.")
   else:
       X = row[features].values
       prediction = model.predict(X)[0]
       probability = model.predict_proba(X)[0][1]
        
       if prediction == 1:
            st.success(f"✅ {player_name} from {team_name} WILL be retained (Probability: {probability:.2f})")
       else:
            st.error(f"❌ {player_name} from {team_name} will NOT be retained (Probability: {probability:.2f})")   

