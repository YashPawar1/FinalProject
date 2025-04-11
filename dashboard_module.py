# dashboard_module.py
import streamlit as st
import pandas as pd

def show_dashboard():
    st.markdown("""
        <style>
        .dashboard-card {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = []

    st.markdown("<h3 class='bordered-header'>📊 Workout Summary Dashboard</h3>", unsafe_allow_html=True)

    refresh = st.button("🔄 Refresh Dashboard")

    if refresh or ("session_data" in st.session_state and st.session_state["session_data"]):
        # st.write("Raw session data:", st.session_state["session_data"])
        df = pd.DataFrame(st.session_state["session_data"])
        if "exercise" not in df.columns:
            st.warning("No 'exercise' data found in the session. Please complete a workout.")
            return


        with st.container():
            st.dataframe(df, use_container_width=True)

        st.markdown("### Total Reps per Exercise")
        st.bar_chart(df.groupby("exercise")["reps"].sum())

        st.markdown("### Calories Burned per Exercise")
        st.bar_chart(df.groupby("exercise")["calories"].sum())
    else:
        st.info("No workout data to display yet. Complete a workout session to see analytics.")


def add_to_session_data(exercise, reps, calories):
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = []

    st.session_state["session_data"].append({
        "exercise": exercise,
        "reps": reps,
        "calories": round(calories, 2)
    })
