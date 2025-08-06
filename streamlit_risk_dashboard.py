import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🔍 ESG Integrated Process Design & Technologies Tool")

# --- Section 1: Risk Input ---
st.header("1️⃣ Risk Entry & Ratings")
num_risks = st.number_input("How many risks would you like to enter?", min_value=1, step=1, key="num_risks")

risks = []
if num_risks:
    for i in range(num_risks):
        with st.expander(f"Risk {i+1}"):
            risk_name = st.text_input(f"Risk Name", key=f"name_{i}")
            probability = st.slider(f"Probability (0-1)", 0.01, 1.0, step=0.01, key=f"prob_{i}")
            severity = st.slider(f"Severity (1-10)", 1, 10, step=1, key=f"sev_{i}")
            if risk_name:
                rating = probability * severity
                risks.append({'Risk': risk_name, 'Probability': probability, 'Severity': severity, 'Rating': rating})

if risks:
    risk_df = pd.DataFrame(risks)
    st.subheader("📊 Risk Ratings Table")
    st.dataframe(risk_df)

    # Heatmap
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(risk_df.set_index('Risk')[['Probability', 'Severity', 'Rating']], annot=True, cmap='YlOrRd', fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    # Barplot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=risk_df, x='Risk', y='Rating', palette='viridis', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

    # Bubble plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(risk_df['Probability'], risk_df['Severity'], s=risk_df['Rating']*50, alpha=0.6, edgecolors='w')
    for i in range(len(risk_df)):
        ax3.text(risk_df['Probability'][i]+0.01, risk_df['Severity'][i]+0.1, risk_df['Risk'][i], fontsize=9)
    ax3.set_xlabel("Probability")
    ax3.set_ylabel("Severity")
    ax3.set_title("Risk Bubble Chart (P vs S with Rating Size)")
    ax3.grid(True)
    st.pyplot(fig3)

# --- Section 2: ESGFP Input & Scenario Analysis ---
st.header("2️⃣ ESGFP Technology Evaluation")
esgfp = {
    "Environmental": ["GHG Emissions", "Water Use", "Waste Management"],
    "Social": ["Labor Safety", "Community Impact"],
    "Governance": ["Compliance", "Transparency"],
    "Financial": ["CAPEX", "OPEX", "ROI"],
    "Process": ["Efficiency", "Flexibility", "Scalability"]
}
exposure_map = {"high": 1.0, "moderate": 0.75, "moderate to lower": 0.5, "lower": 0.25}

def get_score(sub, tech_key):
    score = st.slider(f"{sub} (1-9)", 1, 9, step=1, key=f"{sub}_{tech_key}")
    exposure = st.selectbox(f"Geographic Exposure for {sub}", list(exposure_map.keys()), key=f"exp_{sub}_{tech_key}")
    final_score = score + score * exposure_map[exposure]
    return round(final_score, 2)

tech_scores = {}
for tech in ["Technology A", "Technology B"]:
    st.subheader(f"🧪 {tech}")
    scores = {}
    for pillar, subs in esgfp.items():
        st.markdown(f"**{pillar}**")
        for sub in subs:
            scores[f"{pillar}:{sub}"] = get_score(sub, f"{tech}_{sub}")
    tech_scores[tech] = scores

# --- Process Tech Scores to DataFrame ---
df = pd.DataFrame([tech_scores["Technology A"], tech_scores["Technology B"]], index=["Technology A", "Technology B"]).T
df['Pillar'] = df.index.map(lambda x: x.split(":")[0])
pillar_avgs = df.groupby("Pillar").mean()

# --- Section 3: Scenario Analysis ---
st.header("3️⃣ Scenario-Based Comparison")
scenario_count = st.number_input("How many decision scenarios to compare?", min_value=1, step=1)
scenarios = {}

for s in range(scenario_count):
    with st.expander(f"Scenario {s+1} - Weight Distribution"):
        weights = {}
        total = 0
        for pillar in esgfp.keys():
            weight = st.slider(f"{pillar} (%)", 0, 100, key=f"w_{pillar}_{s}")
            weights[pillar] = weight
        total = sum(weights.values())
        if total != 100:
            st.warning(f"Total = {total}. It must equal 100%.")
        else:
            scenario_result = {}
            for tech in ["Technology A", "Technology B"]:
                total_score = sum(pillar_avgs.loc[p, tech] * (weights[p]/100) for p in weights)
                scenario_result[tech] = round(total_score, 3)
            scenarios[f"Scenario {s+1}"] = scenario_result

if scenarios:
    summary_df = pd.DataFrame(scenarios).T
    st.subheader("📈 Final ESGFP Scores Across Scenarios")
    st.dataframe(summary_df)

    # Bar chart
    st.subheader("📊 ESGFP Comparison Bar Chart")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    summary_df.plot(kind='bar', ax=ax_bar)
    ax_bar.set_ylabel("Weighted ESGFP Score")
    ax_bar.set_title("Technology Comparison Across Scenarios")
    ax_bar.grid(True)
    st.pyplot(fig_bar)

    # Line chart
    st.subheader("📉 ESGFP Score Trends")
    fig_line, ax_line = plt.subplots(figsize=(10, 6))
    summary_df.plot(marker='o', ax=ax_line)
    ax_line.set_ylabel("Score")
    ax_line.set_title("Trend of ESGFP Scores Across Scenarios")
    ax_line.grid(True)
    st.pyplot(fig_line)

    # Radar Chart
    st.subheader("📡 Combined Radar Chart (ESGFP Pillars)")
    labels = pillar_avgs.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for tech in pillar_avgs.columns:
        values = pillar_avgs[tech].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, label=tech, marker='o')
        ax_radar.fill(angles, values, alpha=0.1)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title("Radar Chart: ESGFP Pillars")
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig_radar)