import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from faker import Faker

# OR-Tools
try:
    from ortools.sat.python import cp_model
except Exception:
    cp_model = None

fake = Faker()
st.set_page_config(page_title="AI Workforce Management", layout="wide")
st.title("AI Workforce Management")

DAILY_TARGET_DEFAULT = 295

# ----------------- Utilities: leave synthesis & availability model -----------------
def synthesize_leave_history(employees_df, historical_days=30, base_leave_prob=0.05):
    records = []
    start_date = datetime.today().date() - timedelta(days=historical_days)
    for _, emp in employees_df.iterrows():
        emp_id = emp["employee_id"]
        role = str(emp.get("role", "TM"))
        if "SME" in role.upper():
            p = base_leave_prob * 0.6
        elif "QC" in role.upper():
            p = base_leave_prob * 1.3
        else:
            p = base_leave_prob
        weekday_bias = np.random.uniform(-0.02, 0.03, size=7)
        for d in range(historical_days):
            date = start_date + timedelta(days=d)
            dow = date.weekday()
            prob_leave = float(np.clip(p + weekday_bias[dow], 0.0, 0.6))
            available = 0 if random.random() < prob_leave else 1
            records.append({"employee_id": emp_id, "date": date, "dow": dow, "available": available})
    return pd.DataFrame(records)

def train_availability_model(leave_history_df):
    emp_rates = leave_history_df.groupby("employee_id").apply(lambda df: 1.0 - df["available"].mean()).to_dict()
    df = leave_history_df.copy()
    df["emp_abs_rate"] = df["employee_id"].map(emp_rates)
    X = df[["emp_abs_rate", "dow"]].values
    y = df["available"].values
    model = LogisticRegression(max_iter=300).fit(X, y)
    return model, emp_rates

def predict_availability_next_days(employees_df, model, emp_rates, forecast_days=7):
    start = datetime.today().date()
    rows = []
    for _, emp in employees_df.iterrows():
        emp_id = emp["employee_id"]
        abs_rate = emp_rates.get(emp_id, 0.1)
        for d in range(forecast_days):
            date = start + timedelta(days=d)
            dow = date.weekday()
            X = np.array([[abs_rate, dow]])
            prob = float(model.predict_proba(X)[0][1])
            pred = 1 if prob >= 0.5 else 0
            rows.append({"employee_id": emp_id, "date": date, "prob_available": prob, "predicted_available": int(pred)})
    return pd.DataFrame(rows)

# ----------------- Demand generator -----------------
def generate_challenging_demand(total_days=14):
    start_date = datetime.today().date()
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    base = np.linspace(260000, 360000, total_days)
    zigzag = 700 * np.sin(np.linspace(0, 5 * np.pi, total_days))
    noise = np.random.randint(-450, 450, size=total_days)
    demand = (base + zigzag + noise).astype(int)
    demand = np.maximum(demand, 500)
    return pd.DataFrame({"date": dates, "forecast_demand": demand})

# ----------------- Shift assignment helpers -----------------
def assign_shifts_to_teams(teams, morning_count, day_count, night_count, team_avg_capacity=None):
    total_requested = morning_count + day_count + night_count
    if total_requested != len(teams):
        raise ValueError("Sum of shift counts must equal number of teams in this group")
    if team_avg_capacity:
        teams_sorted = sorted(teams, key=lambda t: team_avg_capacity.get(t, 0), reverse=True)
    else:
        teams_sorted = teams.copy()
        random.shuffle(teams_sorted)
    assignment = {}
    idx = 0
    for _ in range(morning_count):
        assignment[teams_sorted[idx]] = "M"; idx += 1
    for _ in range(day_count):
        assignment[teams_sorted[idx]] = "D"; idx += 1
    for _ in range(night_count):
        assignment[teams_sorted[idx]] = "N"; idx += 1
    return assignment

# ----------------- OR-Tools optimizer with fairness options -----------------
def optimize_offdays_multiobjective(teams, team_day_capacity, demand_series,
                                    fairness_type="offday", fairness_weight=1, short_weight=1000,
                                    time_limit_seconds=12):
    if cp_model is None:
        raise RuntimeError("OR-Tools not installed. `pip install ortools`")

    D = len(demand_series)
    T = len(teams)
    cap = {t: team_day_capacity[t] for t in teams}

    model = cp_model.CpModel()

    # x[t_idx,d] = 1 if team t works on day d (0 = off)
    x = {}
    for t_idx, t in enumerate(teams):
        for d in range(D):
            x[(t_idx, d)] = model.NewBoolVar(f"x_t{t_idx}_d{d}")

    # Each team works exactly D-2 days
    for t_idx, t in enumerate(teams):
        model.Add(sum(x[(t_idx, d)] for d in range(D)) == (D - 2))

    # shortage variables
    max_short = int(max(demand_series) + sum(sum(cap[t]) for t in teams))
    s = {d: model.NewIntVar(0, max_short, f"s_{d}") for d in range(D)}

    # capacity constraints: sum_t cap_t_d * x_t_d + s_d >= demand_d
    for d in range(D):
        terms = []
        for t_idx, t in enumerate(teams):
            ctd = int(cap[t][d])
            if ctd > 0:
                terms.append(ctd * x[(t_idx, d)])
        model.Add(sum(terms) + s[d] >= int(demand_series[d]))

    # Fairness variables and constraints
    fairness_devs = []
    if fairness_type == "offday":
        for d in range(D):
            dev_pos = model.NewIntVar(0, T * D, f"devpos_off_d{d}")
            dev_neg = model.NewIntVar(0, T * D, f"devneg_off_d{d}")
            fairness_devs.extend([dev_pos, dev_neg])
            sum_x = sum(x[(t_idx, d)] for t_idx in range(T))
            model.Add(D * (T - sum_x) - 2 * T == dev_pos - dev_neg)
    elif fairness_type == "per_shift":
        if "SHIFT_GROUPS" not in globals() or not globals()["SHIFT_GROUPS"]:
            st.warning("SHIFT_GROUPS not found; falling back to off-day fairness.")
            for d in range(D):
                dev_pos = model.NewIntVar(0, T * D, f"devpos_off_d{d}")
                dev_neg = model.NewIntVar(0, T * D, f"devneg_off_d{d}")
                fairness_devs.extend([dev_pos, dev_neg])
                sum_x = sum(x[(t_idx, d)] for t_idx in range(T))
                model.Add(D * (T - sum_x) - 2 * T == dev_pos - dev_neg)
        else:
            shift_groups = globals()["SHIFT_GROUPS"] 
            for shift, shift_team_list in shift_groups.items():
                num_shift_teams = len(shift_team_list)
                if num_shift_teams == 0:
                    continue
                t_idx_map = {t: teams.index(t) for t in shift_team_list if t in teams}
                for d in range(D):
                    dev_pos = model.NewIntVar(0, num_shift_teams * D, f"devpos_{shift}_d{d}")
                    dev_neg = model.NewIntVar(0, num_shift_teams * D, f"devneg_{shift}_d{d}")
                    fairness_devs.extend([dev_pos, dev_neg])
                    sum_x_shift = sum(x[(t_idx_map[t], d)] for t in t_idx_map)
                    model.Add(D * (num_shift_teams - sum_x_shift) - 2 * num_shift_teams == dev_pos - dev_neg)
    else:
        # no fairness
        pass
    objective_terms = []
    # shortages
    for d in range(D):
        objective_terms.append(short_weight * s[d])
    # fairness
    for dev in fairness_devs:
        objective_terms.append(fairness_weight * dev)

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        st.warning("OR-Tools did not find a solution; falling back to random off-days.")
        return {t: random.sample(range(D), 2) for t in teams}

    weekoffs_idx = {}
    for t_idx, t in enumerate(teams):
        offs = []
        for d in range(D):
            if solver.Value(x[(t_idx, d)]) == 0:
                offs.append(d)
        # ensure 2
        if len(offs) != 2:
            if len(offs) > 2:
                offs = offs[:2]
            else:
                for cand in range(D):
                    if cand not in offs:
                        offs.append(cand)
                    if len(offs) == 2:
                        break
        weekoffs_idx[t] = offs

    return weekoffs_idx

# ----------------- Calendar builder -----------------
def build_team_calendar(teams, dates, team_shifts, team_weekoffs):
    cal = pd.DataFrame(index=teams, columns=dates)
    for team in teams:
        for i, dt in enumerate(dates):
            cal.at[team, dt] = "Off" if i in team_weekoffs.get(team, []) else team_shifts.get(team, "M")
    return cal

# ----------------- UI -----------------
st.sidebar.header("Upload & Settings")
teams_file = st.sidebar.file_uploader("Upload Team Database CSV (employee_id,name,team,role)", type=["csv"])
forecast_mode = st.sidebar.radio("Demand Forecast Source", ["Generate Synthetic", "Upload CSV", "Edit After Generate"], index=0)
uploaded_forecast = None
if forecast_mode == "Upload CSV":
    uploaded_forecast = st.sidebar.file_uploader("Upload forecast CSV (date,forecast_demand)", type=["csv"], key="upload_forecast")

# shift counts per group
st.sidebar.markdown("### Shift counts (Production teams)")
prod_morning = st.sidebar.number_input("Prod teams in Morning (M)", min_value=0, value=3, step=1)
prod_day = st.sidebar.number_input("Prod teams in Day (D)", min_value=0, value=3, step=1)
prod_night = st.sidebar.number_input("Prod teams in Night (N)", min_value=0, value=2, step=1)

st.sidebar.markdown("### Shift counts (QC teams)")
qc_morning = st.sidebar.number_input("QC teams in Morning (M)", min_value=0, value=1, step=1)
qc_day = st.sidebar.number_input("QC teams in Day (D)", min_value=0, value=1, step=1)
qc_night = st.sidebar.number_input("QC teams in Night (N)", min_value=0, value=1, step=1)

daily_target = st.sidebar.number_input("Daily target per employee", min_value=1, value=DAILY_TARGET_DEFAULT, step=1)
historical_days = st.sidebar.number_input("Historical days for leave model", min_value=7, max_value=90, value=30, step=1)
forecast_total_days = st.sidebar.number_input("Total forecast days to generate (context)", min_value=7, max_value=28, value=14, step=1)

st.sidebar.markdown("### Optimization settings")
fairness_type = st.sidebar.radio("Secondary objective (fairness)", ["Off-day fairness", "Per-shift off-day fairness"], index=0)
fairness_weight = st.sidebar.slider("Fairness weight (higher = fairer)", min_value=0, max_value=1000, value=50, step=10)
short_weight = st.sidebar.slider("Shortage weight (higher = prioritize demand coverage)", min_value=1, max_value=10000, value=1000, step=10)
time_limit = st.sidebar.number_input("Solver time limit (seconds)", min_value=1, max_value=60, value=12, step=1)

run_button = st.sidebar.button("Generate 7-Day Roster (OR-Tools Multi-Objective)")

if teams_file is None:
    st.info("Please upload Team Database CSV to proceed.")
    st.stop()

employees = pd.read_csv(teams_file)
if "employee_id" not in employees.columns or "team" not in employees.columns or "role" not in employees.columns:
    st.error("Team CSV must include 'employee_id', 'team', and 'role' columns.")
    st.stop()

# separate teams into production and QC based on role composition
teams_all = sorted(employees["team"].unique())
team_roles = employees.groupby("team")["role"].apply(list).to_dict()
qc_teams = []
prod_teams = []
for t, roles in team_roles.items():
    qc_count = sum(1 for r in roles if "QC" in str(r).upper())
    if qc_count >= 0.5 * len(roles):
        qc_teams.append(t)
    else:
        prod_teams.append(t)

# validate shift counts
if len(prod_teams) > 0 and (prod_morning + prod_day + prod_night != len(prod_teams)):
    st.warning(f"Production shift counts sum ({prod_morning+prod_day+prod_night}) must equal production teams ({len(prod_teams)}).")
if len(qc_teams) > 0 and (qc_morning + qc_day + qc_night != len(qc_teams)):
    st.warning(f"QC shift counts sum ({qc_morning+qc_day+qc_night}) must equal QC teams ({len(qc_teams)}).")

# base forecast
if forecast_mode == "Generate Synthetic":
    base_forecast_df = generate_challenging_demand(total_days=forecast_total_days)
elif forecast_mode == "Upload CSV":
    if uploaded_forecast is None:
        st.info("Please upload a forecast CSV in the sidebar.")
        st.stop()
    base_forecast_df = pd.read_csv(uploaded_forecast, parse_dates=["date"])
    base_forecast_df["date"] = base_forecast_df["date"].dt.date
else:
    base_forecast_df = generate_challenging_demand(total_days=forecast_total_days)

st.subheader("Forecast Data (editable)")
editable_forecast_df = st.data_editor(base_forecast_df.copy(), num_rows="dynamic")
if editable_forecast_df["date"].dtype != object:
    editable_forecast_df["date"] = pd.to_datetime(editable_forecast_df["date"]).dt.date

if run_button:
    # checks
    if len(editable_forecast_df) < 7:
        st.error("Editable forecast must have at least 7 rows/dates.")
        st.stop()
    if len(prod_teams) > 0 and (prod_morning + prod_day + prod_night != len(prod_teams)):
        st.error("Production shift counts must sum to number of production teams.")
        st.stop()
    if len(qc_teams) > 0 and (qc_morning + qc_day + qc_night != len(qc_teams)):
        st.error("QC shift counts must sum to number of QC teams.")
        st.stop()

    with st.spinner("Training model and preparing data..."):
        leave_history = synthesize_leave_history(employees, historical_days=historical_days, base_leave_prob=0.06)
        avail_model, emp_rates = train_availability_model(leave_history)
        avail_pred = predict_availability_next_days(employees, avail_model, emp_rates, forecast_days=7)
        avail_pred["date"] = pd.to_datetime(avail_pred["date"]).dt.date
        team_avail = avail_pred.merge(employees[["employee_id","team"]], on="employee_id")
        team_counts = team_avail.groupby(["team","date"])["predicted_available"].sum().unstack(fill_value=0)

    roster_forecast = editable_forecast_df.iloc[:7].reset_index(drop=True)
    roster_dates = roster_forecast["date"].tolist()
    demand_values = roster_forecast["forecast_demand"].values.astype(int)

    all_teams = prod_teams + qc_teams
    team_day_capacity = {}
    team_avg_capacity = {}
    for t in all_teams:
        counts = []
        for dt in roster_dates:
            cnt = int(team_counts.loc[t, dt]) if (t in team_counts.index and dt in team_counts.columns) else 0
            counts.append(cnt * daily_target)
        team_day_capacity[t] = counts
        team_avg_capacity[t] = int(np.mean(counts))

    # assign shifts per group
    team_shifts = {}
    if len(prod_teams) > 0:
        prod_assignment = assign_shifts_to_teams(prod_teams.copy(), prod_morning, prod_day, prod_night, team_avg_capacity)
        team_shifts.update(prod_assignment)
    if len(qc_teams) > 0:
        qc_assignment = assign_shifts_to_teams(qc_teams.copy(), qc_morning, qc_day, qc_night, team_avg_capacity)
        team_shifts.update(qc_assignment)

    # Prepare SHIFT_GROUPS global if needed for per-shift fairness
    SHIFT_GROUPS = {"M": [], "D": [], "N": []}
    for t, s in team_shifts.items():
        SHIFT_GROUPS[s].append(t)
    globals()["SHIFT_GROUPS"] = SHIFT_GROUPS

    # Select fairness_type param for optimizer
    fairness_key = "offday" if fairness_type == "Off-day fairness" else "per_shift"

    # Optimize with OR-Tools (multi-objective weighted)
    try:
        weekoffs_idx = optimize_offdays_multiobjective(all_teams, team_day_capacity, demand_values,
                                                      fairness_type=fairness_key,
                                                      fairness_weight=fairness_weight,
                                                      short_weight=short_weight,
                                                      time_limit_seconds=time_limit)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # Build calendars for prod and qc
    prod_calendar = pd.DataFrame()
    qc_calendar = pd.DataFrame()
    if len(prod_teams) > 0:
        prod_calendar = build_team_calendar(prod_teams, roster_dates, team_shifts, weekoffs_idx)
    if len(qc_teams) > 0:
        qc_calendar = build_team_calendar(qc_teams, roster_dates, team_shifts, weekoffs_idx)

    # compute totals and shortages
    total_capacity_per_day = np.zeros(7, dtype=int)
    for t in all_teams:
        for d_idx in range(7):
            if d_idx in weekoffs_idx[t]:
                continue
            total_capacity_per_day[d_idx] += team_day_capacity[t][d_idx]
    shortage = np.maximum(demand_values - total_capacity_per_day, 0)

    team_counts_display = pd.DataFrame(team_day_capacity, index=[d.strftime("%Y-%m-%d") for d in roster_dates]).T
    team_headcounts = (team_counts_display / daily_target).astype(int)

    # display
    colA, colB = st.columns([1.0, 1.2])
    with colA:
        st.subheader("Editable Forecast (7 days used)")
        st.dataframe(roster_forecast.rename(columns={"date":"Date","forecast_demand":"Demand"}))
        st.subheader("Capacity vs Demand")
        cap_df = pd.DataFrame({
            "date":[d.strftime("%Y-%m-%d") for d in roster_dates],
            "demand": demand_values,
            "capacity": total_capacity_per_day,
            "shortage": shortage
        }).set_index("date")
        st.dataframe(cap_df)
        st.line_chart(cap_df[["demand","capacity"]])
        st.subheader("Team Predicted Headcount (per day)")
        st.dataframe(team_headcounts)

    with colB:
        if len(prod_teams) > 0:
            st.subheader("Production Teams Calendar")
            st.dataframe(prod_calendar.style.applymap(lambda v: "background-color: lightcoral; color:white" if v=="Off" else ("background-color: lightgreen" if v=="M" else ("background-color: lightblue" if v=="D" else "background-color: plum"))), height=420)
        if len(qc_teams) > 0:
            st.subheader("QC Teams Calendar")
            st.dataframe(qc_calendar.style.applymap(lambda v: "background-color: lightcoral; color:white" if v=="Off" else ("background-color: lightgreen" if v=="M" else ("background-color: lightblue" if v=="D" else "background-color: plum"))), height=300)

    # exports
    st.markdown("---")
    st.subheader("Exports")
    if not prod_calendar.empty:
        st.download_button("Download Production Calendar CSV", data=prod_calendar.to_csv().encode("utf-8"),
                           file_name="production_calendar.csv", mime="text/csv")
    if not qc_calendar.empty:
        st.download_button("Download QC Calendar CSV", data=qc_calendar.to_csv().encode("utf-8"),
                           file_name="qc_calendar.csv", mime="text/csv")
    st.download_button("Download Editable Forecast CSV", data=editable_forecast_df.to_csv(index=False).encode("utf-8"),
                       file_name="editable_forecast.csv", mime="text/csv")
    st.download_button("Download Availability Predictions (employee-level)", data=avail_pred.to_csv(index=False).encode("utf-8"),
                       file_name="availability_predictions.csv", mime="text/csv")

    st.success("7-day roster optimized (shortage + fairness) and calendars generated.")

# preview
st.subheader("Team Database Preview")
st.dataframe(employees.head(200))

st.subheader("Detected Team Groups")
st.markdown(f"- Total teams: **{len(teams_all)}**")
st.markdown(f"- Production teams: **{len(prod_teams)}** — {prod_teams}")
st.markdown(f"- QC teams: **{len(qc_teams)}** — {qc_teams}")

