# dashboard_module.py
import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd

import plotly.express as px

import plotly.graph_objects as go
import datetime

def show_dashboard():
    st.markdown("""
        <style>
        .dashboard-metrics {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            flex: 1;
            background: linear-gradient(145deg, #1e1e1e, #2c2c2c);
            border-radius: 16px;
            padding: 20px;
            color: #fff;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            text-align: center;
        }
        .metric-card h3 {
            margin: 0;
            font-size: 18px;
            color: #bbb;
        }
        .metric-card h1 {
            margin: 10px 0 0 0;
            font-size: 32px;
            color: #fff;
        }
        .chart-header {
            font-size: 20px;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 10px;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:white;'>📊 Workout Summary</h2>", unsafe_allow_html=True)

    if "session_data" not in st.session_state or not st.session_state["session_data"]:
        st.info("No workout data yet. Do a session to see stats.")
        return

    df = pd.DataFrame(st.session_state["session_data"])

    # ===== METRICS =====
    total_reps = int(df["reps"].sum())
    total_cals = round(df["calories"].sum(), 2)
    total_types = df["exercise"].nunique()

    st.markdown('<div class="dashboard-metrics">', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="metric-card">
            <h3>Total Exercises</h3>
            <h1>{total_types}</h1>
        </div>
        <div class="metric-card">
            <h3>Total Reps</h3>
            <h1>{total_reps}</h1>
        </div>
        <div class="metric-card">
            <h3>Calories Burned</h3>
            <h1>{total_cals} kcal</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # ========== DAILY PROGRESS RING ==========
    st.markdown("<h3 style='color:white;'>🎯 Daily Progress Goal</h3>", unsafe_allow_html=True)
    # Let user set their goal (default = 50)
    goal_reps = st.slider("Set your daily rep goal", min_value=10, max_value=200, value=50, step=10)
    # Assume current session is for today
    today = datetime.date.today().strftime('%Y-%m-%d')
    df["date"] = today  # Temporary, since we aren't tracking per date yet
    total_reps = int(df["reps"].sum())
    progress = min(total_reps / goal_reps, 1.0)
    fig = go.Figure(go.Pie(
        values=[progress, 1 - progress],
        hole=0.7,
        marker_colors=["#00cc96", "#222"],
        textinfo='none',
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        annotations=[dict(text=f"{int(progress * 100)}%", font_size=24, showarrow=False, font_color="white")]
    )
    st.plotly_chart(fig, use_container_width=False)

    # ========== ACHIEVEMENT BANNER ==========
    st.markdown("<h3 style='color:white;'>🏅 Achievements Unlocked</h3>", unsafe_allow_html=True)

    badges = []

    if total_reps >= 10:
        badges.append("🥉 10 Reps Badge")
    if total_reps >= 25:
        badges.append("🥈 25 Reps Badge")
    if total_reps >= 50:
        badges.append("🥇 50 Reps Badge")
    if df["calories"].sum() >= 100:
        badges.append("🔥 100 Calories Burned")
    if df["exercise"].nunique() >= 3:
        badges.append("💪 Multi-Exercise Warrior")

    if badges:
        st.markdown(
            "<div style='display: flex; gap: 10px; flex-wrap: wrap;'>"
            + "".join([f"<div style='background: #333; color: #fff; padding: 10px 20px; border-radius: 20px; font-weight: bold;'>{badge}</div>" for badge in badges])
            + "</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No achievements unlocked yet — keep going! 🚀")

    # ===== CHARTS =====
    st.markdown("<div class='chart-header'>📈 Reps per Exercise</div>", unsafe_allow_html=True)
    reps_data = df.groupby("exercise")["reps"].sum().reset_index()
    rep_chart = px.bar(
        reps_data,
        x="exercise",
        y="reps",
        color="exercise",
        color_discrete_map={
            "squat": "#1f77b4",
            "pushup": "#ff7f0e",
            "curl": "#2ca02c",
            "lunge": "#9467bd",
            "press": "#d62728"
        },
        text="reps",
    )
    rep_chart.update_layout(showlegend=False, plot_bgcolor="#111", paper_bgcolor="#111", font_color="white")
    st.plotly_chart(rep_chart, use_container_width=True)

    st.markdown("<div class='chart-header'>🔥 Calories Burned</div>", unsafe_allow_html=True)
    cal_data = df.groupby("exercise")["calories"].sum().reset_index()
    cal_chart = px.bar(
        cal_data,
        x="exercise",
        y="calories",
        color="exercise",
        color_discrete_map={
            "squat": "#1f77b4",
            "pushup": "#ff7f0e",
            "curl": "#2ca02c",
            "lunge": "#9467bd",
            "press": "#d62728"
        },
        text="calories"
    )
    cal_chart.update_layout(showlegend=False, plot_bgcolor="#111", paper_bgcolor="#111", font_color="white")
    st.plotly_chart(cal_chart, use_container_width=True)

    # ===== TABLE + EXPORT =====
    with st.expander("📋 Show Raw Data Table"):
        st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Workout History", csv, "workout_summary.csv", "text/csv")


def add_to_session_data(exercise, reps, calories):
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = []

    st.session_state["session_data"].append({
        "exercise": exercise,
        "reps": reps,
        "calories": round(calories, 2)
    })