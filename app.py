import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Teams and Cities
# -----------------------------
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# -----------------------------
# Load Model
# -----------------------------
pipe = pickle.load(open('pipe.pkl', 'rb'))

# -----------------------------
# UI
# -----------------------------
st.title('🏏 IPL Win Predictor')

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

# Validation
if batting_team == bowling_team:
    st.error("Batting and Bowling teams must be different!")

# City
selected_city = st.selectbox('Select Host City', sorted(cities))

# Target
target = st.number_input('Target', min_value=1)

# Match situation
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets Out', min_value=0, max_value=10)

# -----------------------------
# Prediction
# -----------------------------
if st.button('Predict Probability'):

    if batting_team != bowling_team:

        # Convert overs → balls correctly
        over_int = int(overs)
        balls = int((overs - over_int) * 10)

        # Validation for wrong input like 10.7
        if balls > 5:
            st.error("Invalid input! After decimal, balls should be between 0–5.")
            st.stop()

        balls_bowled = over_int * 6 + balls
        balls_left = 120 - balls_bowled

        # Edge case
        if balls_left <= 0:
            st.error("Match is already over!")
            st.stop()

        # Calculations
        runs_left = max(target - score, 0)
        wickets = 10 - wickets_out

        crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Input DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Prediction
        result = pipe.predict_proba(input_df)

        loss = result[0][0]
        win = result[0][1]

        # Output
        st.subheader("Winning Probability")
        st.success(f"{batting_team}: {round(win * 100)}%")
        st.error(f"{bowling_team}: {round(loss * 100)}%")

        # Progress bar (nice UX)
        st.progress(int(win * 100))

    else:
        st.warning("Please select different teams!")