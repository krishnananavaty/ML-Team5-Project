
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
st.set_page_config(layout="wide")

st.markdown("### <h1 style='text-align: center;'>Income Census - Univariate Analysis Dashboard</h1>", unsafe_allow_html=True)
#st.title("Income Census - Univariate Analysis Dashboard")
df = pd.read_csv("df_train_test.csv") 

st.markdown("""<style> div[data-baseweb="tab-list"] {
        justify-content: center !important;}
    </style>""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EDA - Introduction",
    "Demographics",
    "Work & Education",
    "Other Features",
    "Income-Target Feature"
])



with tab1:
    st.markdown("### <h4 style='text-align: center;'>Introduction : This dashboard provides univariate analysis for the Adult Income Census dataset. We can explore distributions, counts, and summary statistics for each feature.</h4>", unsafe_allow_html=True)
with tab2:    
    col1, col2, col3 = st.columns(3)
    with col1:
        
        st.subheader("Age Distribution by Age Groups")
        
        # Histogram
        bins = [15, 20, 30, 40, 50, 60, 70, 80, 90]
        labels = ['15-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90']
        
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        
        fig = px.bar(
            df['age_group'].value_counts().sort_index())
        fig.update_traces(hovertemplate='Age Group: %{x}<br>Count: %{y}')
        fig.update_layout(
            showlegend=False,             # remove legend
            yaxis=dict(showgrid=False),   # remove horizontal gridlines
            # Add border around chart
            shapes=[dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='black', width=1),
                layer='above'
            )]
        )
        # Streamlit display
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(""" The dataset is heavily concentrated in the 21–50 age range, with the highest representation in the 31–40 group. Both younger (15–20) and older (70+) age groups are underrepresented, indicating that the dataset primarily reflects working‑age adults. """)
    with col2:
        st.markdown("### <center>Age Statistical Distribution</center>", unsafe_allow_html=True)
        # Box plot
        fig = px.box(df, y='age')
        fig.update_layout(yaxis=dict(showgrid=False),
                          shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,line=dict(color='black', width=1),layer='above')]
                         )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(""" The age distribution is centered around 37 years, with most individuals between 28 and 48. The data shows a slight right skew due to older individuals (70–90), which appear as outliers. Overall, the distribution is clean, realistic, and consistent with a working‑age population.""")
        
    with col3:
        st.markdown("### <center>Age Density</center>", unsafe_allow_html=True)
        # Density plot
        fig, ax = plt.subplots(figsize=(6,7))
        sns.kdeplot(df['age'], fill=True, ax=ax)
        st.pyplot(fig)
        ax.grid(False)
        st.markdown(""" The age density plot shows a smooth, slightly right‑skewed distribution with the highest concentration of individuals between ages 30 and 40. Density declines steadily after age 50, with very few individuals above 70, indicating a typical working‑age population. """)

    st.markdown("<hr style='height:1px;border:none;background-color:black;'>", unsafe_allow_html=True)
    # =========================
    # ROW 2 → GENDER
    # =========================
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Gender Distribution")
        col = "sex"
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        
        fig, ax = plt.subplots(figsize=(6,5))
        
        sns.countplot(
            data=df,
            x=col,
            order=counts.index,
            palette="Accent",
            ax=ax
        )
        ax.grid(False)
        
        for p in ax.patches:
            count = int(p.get_height())
            percent = (count / len(df)) * 100
        
            ax.annotate(
                f'{percent:.1f}%',
                (p.get_x() + p.get_width() / 2, count),
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""The dataset is male‑dominant, with males representing nearly double the number of females.This imbalance may influence any analysis or modeling involving sex as a feature.""")
    with col5:
        st.markdown("### <center>Race Distribution</center>", unsafe_allow_html=True)
        col = "race"

        counts = df[col].value_counts()
        
        fig, ax = plt.subplots(figsize=(6,4))
        
        sns.countplot(
            data=df,
            x=col,
            order=counts.index,
            palette="Set3",
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(False)
        for p in ax.patches:
            percent = (p.get_height() / len(df)) * 100
            ax.annotate(
                f'{percent:.1f}%',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""White individuals make up over 86%. and this will storngly influence any analysis involving race.Black individuals represent 10%, making them the only sizable minority group in the dataset. Small but distinct groups such as Asian-Pac-Islander, Amer-Indian-Eskimo and Other may need careful handling in modeling and can be combined depending on the analysis.""")

    st.markdown("<hr style='height:1px;border:none;background-color:black;'>", unsafe_allow_html=True)
    # =========================
    # ROW 3 → marital_status and relationship
    # =========================
    col6, col7 = st.columns(2)
    with col6:
        st.markdown("### <center>Relationship Distribution</center>", unsafe_allow_html=True)
        col = "relationship"

        counts = df[col].value_counts()
        total = len(df[col])
        max_count = counts.max()

        # Create figure
        fig, ax = plt.subplots(figsize=(10,6))
        
        sns.countplot(
            data=df,
            y=col,
            order=counts.index,
            palette="Set2",  # unique colors for bars
            ax=ax
        )
        ax.grid(False)
       
        # Expand x-axis so labels fit
        ax.set_xlim(0, max_count * 1.15)
        
        # Add count + percentage labels INSIDE the bars
        for p in ax.patches:
            count = int(p.get_width())
            percent = (count / total) * 100
        
            ax.annotate(
                f'{count} ({percent:.0f}%)',
                (count * 0.98, p.get_y() + p.get_height() / 2),
                ha='right',
                va='center',
                color='white',
                fontsize=9,
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("""The dataset is strong male- headed household signal with husband as the largest catgory(40%). 'Not-in-family is another major category with 26%. This variable is highly imbalanced.So for anlaysis or modeling, have to combine small categories and can explore interactions with marital status or sex.""")

    with col7:
        st.markdown("### <center>Distribution of Marital Status</center>", unsafe_allow_html=True)
        col = "marital_status"

        # Count values
        counts = df[col].value_counts()
        total = len(df[col])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10,7))
        
        sns.countplot(data=df, x=col, order=counts.index,palette="Set2", ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(False)
        # Add count + percentage labels above bars
        for p in ax.patches:
            count = int(p.get_height())
            percent = (count / total) * 100
            ax.annotate(
                f'{count} ({percent:.0f}%)',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("""Strong Skew Toward Married Individuals, Never-Married Is the Only Other large Group. Imbalance is significant — any model using marital status will be heavily influenced by the Married-civ-spouse group.Small categories may need grouping or special handling requires in analysis and modeling.""")

with tab3:    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### <center>Grouped Education Distribution</center>", unsafe_allow_html=True)
        df['education_grouped'] = df['education'].replace({
            'Preschool': 'Lower-Education',
            '1st-4th': 'Lower-Education',
            '5th-6th': 'Lower-Education',
            '7th-8th': 'Lower-Education',
            '9th': 'Lower-Education',
            '10th': 'Lower-Education',
            '11th': 'Lower-Education',
            '12th': 'Lower-Education',
            'HS-grad': 'High-School',
            'Some-college': 'Some-College',
            'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
            'Bachelors': 'Bachelors',
            'Masters': 'Advanced-Degree',
            'Doctorate': 'Advanced-Degree',
            'Prof-school': 'Advanced-Degree'
        })
        edu_grouped = df['education_grouped'].value_counts().reset_index()
        edu_grouped.columns = ['education_grouped', 'count']
        
        # Create Plotly bar chart
        fig = px.bar(
            edu_grouped,
            x='education_grouped',
            y='count',
            color='education_grouped',
            labels={'education_grouped': 'Education Level', 'count': 'Count'},
            text='count'  # show count on top of bars
        )
        
        # Optional: customize hover
        fig.update_traces(hovertemplate='Education: %{x}<br>Count: %{y}')
        # Remove legend
        fig.update_layout(showlegend=False,yaxis=dict(showgrid=False),
                         shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,line=dict(color='black', width=1),layer='above')]
                         )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### <center>Education_num Distribution</center>", unsafe_allow_html=True)
        # Create histogram
        fig = px.histogram(
            df,
            x='education_num',
            nbins=16,
            #color='education_num',
            labels={'education_num': 'Education Level (Numeric)'},
            text_auto=True  # optional: shows count on top of bars
        )
        
        # Optional: customize hover
        fig.update_traces(hovertemplate='Education_num: %{x}<br>Count: %{y}')
        # Remove legend (histograms don't usually need one)
        fig.update_layout(showlegend=False, yaxis=dict(showgrid=False),
                         shapes=[dict(type='rect', xref='paper', yref='paper',x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1),layer='above')])
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        st.markdown("### <center>Education_num statistical Plot</center>", unsafe_allow_html=True)
        fig = px.box(
        df,
        y='education_num',
        labels={'education_num': 'Education Level (Numeric)'})
        fig.update_traces(hovertemplate='Education_num: %{y}')

        fig.update_layout(width=600, height=450, yaxis=dict(showgrid=False),
                        shapes=[dict(type='rect', xref='paper', yref='paper',x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1),layer='above')]     
        )
        st.plotly_chart(fig)
    st.markdown("""The distribution of education levels shows that most individuals fall into mid‑level categories, with High‑School being the largest group, followed by Some‑College and Bachelors. Lower‑Education and Advanced‑Degree groups appear in moderate numbers, while Associates is the smallest category. This pattern reflects a population where basic and intermediate education levels dominate, and higher academic qualifications are comparatively less common.""")
    st.markdown("<hr style='height:1px;border:none;background-color:black;'>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("### <center>Grouped Workclass Distribution</center>", unsafe_allow_html=True)
        df['workclass_grouped'] = df['workclass'].replace({
            'Federal-gov': 'Government',
            'State-gov': 'Government',
            'Local-gov': 'Government',
            'Self-emp-inc': 'Self-Employed',
            'Self-emp-not-inc': 'Self-Employed',
            'Without-pay': 'Not-Working',
            'Never-worked': 'Not-Working',
            'Unknown': 'Unknown',
            'Private': 'Private'
        })
        wc_grouped = df['workclass_grouped'].value_counts().reset_index()
        wc_grouped.columns = ['workclass_grouped', 'count']
        
        fig = px.bar(
            wc_grouped,
            x='workclass_grouped',
            y='count',
            color='workclass_grouped',
            labels={'workclass_grouped': 'Workclass', 'count': 'Count'}
        )
        # Remove legend
        fig.update_layout(width=400, height=550,showlegend=False, yaxis=dict(showgrid=False),
                         shapes=[dict(type='rect', xref='paper', yref='paper',x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1),layer='above')]
                         )
        fig.update_traces(hovertemplate='Workclass: %{x}<br>Count: %{y}')
        st.plotly_chart(fig)
        st.markdown("""The workclass is heavily dominated by Private‑sector workers, with Government and Self‑Employed forming mid‑sized groups, while Unknown and Not‑Working categories are very small and require careful handling in analysis and modeling.""")
    with col5:
        st.markdown("### <center>Distribution of Occupation</center>", unsafe_allow_html=True)
        counts = df["occupation"].value_counts().reset_index()
        counts.columns = ["occupation", "count"]
        
        fig = px.bar(
            counts,
            x="count",
            y="occupation",
            orientation="h",
            text="count",
            color="occupation",
            color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(width=550, height=550,showlegend=False, yaxis=dict(showgrid=False),
                          shapes=[dict(type='rect', xref='paper', yref='paper',x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1),layer='above')]
                         )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        
        st.plotly_chart(fig)
        st.markdown("""The occupation variable has three nearly equal leading categories, making it more balanced.Prof-specialty,Craft-repair, Exec-managerial together, they are 37% of all individuals.""")
    
    st.markdown("<hr style='height:1px;border:none;background-color:black;'>", unsafe_allow_html=True)
    col6,  = st.columns(1)
    with col6:
        st.markdown("### <center>Hours Per Week</center>", unsafe_allow_html=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Single subplot

        # Full distribution
        sns.histplot(df['hours_per_week'], bins=20, kde=True, color='#4c72b0', ax=ax)
        counts, bins = np.histogram(df['hours_per_week'], bins=20)
        # Add labels on top of each bar
        for i in range(len(counts)):
            ax.text(
                (bins[i] + bins[i+1]) / 2,  # x position (center of bin)
                counts[i] + counts[i]*0.05, # y position: 3% above bar height
                str(int(counts[i])),         # label text
                ha='center', 
                va='bottom', 
                fontsize=10
            )
    
        ax.set_xlabel('Hours per Week')
        ax.set_ylabel('Count')
        ax.grid(False)
        plt.tight_layout()
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(""" Distribution centers tightly around standard 40-hour workweek (mean 40.4, median 40), with a peak at exactly 40 hours. Clear right skew shows most work full-time hours, with smaller tails of part-time (<30) and overtime (>60) workers.""")
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### <center>Capital Gain Non-Zero Values</center>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        non_zero_gain = df[df['capital_gain'] > 0]['capital_gain']
        sns.histplot(non_zero_gain, bins=30, kde=True, ax=ax, color='#4c72b0')
        
        ax.set_xlabel('capital_gain (non-zero)')
        ax.set_ylabel('Count')
        plt.tight_layout()
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(""" The distribution is heavily right‑skewed. Capital gains are zero for the majority of individuals. Among positive capital gains, amounts are dispersed with a few very large values, indicating an extremely right‑skewed distribution. """)
    with col2:
        st.markdown("### <center>Capital Loss Non-Zero Values</center>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Non‑zero values
        non_zero_loss = df[df['capital_loss'] > 0]['capital_loss']
        
        sns.histplot(non_zero_loss, bins=30, kde=True, ax=ax, color='#4c72b0')
        ax.set_xlabel('capital_gain (non-zero)')
        ax.set_ylabel('Count')

        plt.tight_layout()
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""Capital loss is zero for 95.33% of individuals, Positive losses show a right-skewed distribution with peak around 2000 and a maximum of 4356, though much less extreme than capital gains. """)

    st.markdown("<hr style='height:1px;border:none;background-color:black;'>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### <center>Final weight Distribution by Groups</center>", unsafe_allow_html=True)
        bins = [0, 100000, 200000, 300000, 400000, 600000, 800000, 1500000]
        labels = [
            '0-100K', '100K-200K', '200K-300K', '300K-400K',
            '400K-600K', '600K-800K', '800K-1.5M'
            ]
        
        df['fnlwgt_group'] = pd.cut(df['fnlwgt'], bins=bins, labels=labels, include_lowest=True)
        
        fig = px.bar(
            df['fnlwgt_group'].value_counts().sort_index())
        fig.update_traces(hovertemplate='Group: %{x}<br>Count: %{y}')
        fig.update_layout(showlegend=False,yaxis=dict(showgrid=False),
                         shapes=[dict(type='rect', xref='paper', yref='paper',x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1),layer='above')]
                         )
        st.plotly_chart(fig)
    with col4:
        st.markdown("### <center>Final Weight Box</center>", unsafe_allow_html=True)
        fig = px.box(df, y='fnlwgt')
        fig.update_traces(hovertemplate='fnlwgt: %{y}')
        fig.update_layout(yaxis=dict(showgrid=False),
                         shapes=[dict(type='rect', xref='paper', yref='paper',x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1),layer='above')]
                         )
        st.plotly_chart(fig)
    st.markdown("""The fnlwgt variable shows a strongly right‑skewed distribution, where most individuals fall within moderate weight ranges and a small portion carry exceptionally large sampling weights. The majority of values cluster between 100K and 300K, with frequencies dropping sharply as the weight increases. This pattern reveals substantial high‑end outliers and an uneven spread, suggesting that fnlwgt may need transformation, binning, or careful handling during modeling.""")
    st.markdown("<hr style='height:1px;border:none;background-color:black;'>", unsafe_allow_html=True)
    col5, = st.columns(1)
    with col5:
        st.markdown("### <center>Native Country Distribution</center>", unsafe_allow_html=True)
        

        fig, ax = plt.subplots(figsize=(12, 8))
        country_counts = df['native_country'].value_counts()
        country_counts.plot(kind='barh', ax=ax)
        
        ax.set_xlabel("Count")
        ax.set_ylabel("Native Country")
        ax.invert_yaxis()  # Highest values at top
        ax.xaxis.grid(False)
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3)
        
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""The native_country variable is highly imbalanced, with the majority of individuals originating from the United-States, while other countries have very low representation.""")
with tab5:
    
    col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is 2x wider
    with col2:
        st.markdown("### <center>Income Distribution</center>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        income_counts = df['income'].value_counts()
        income_counts.plot(kind='bar', color=['skyblue', 'pink'], ax=ax)
    
        ax.set_xlabel("Income Category")
        ax.set_ylabel("Count")
        ax.set_xticklabels(income_counts.index, rotation=0)
        ax.grid(False)
    
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3)
    
        st.pyplot(fig)
        st.markdown("""The income variable is imbalanced, with approximately 37,000 individuals earning ≤50K and around 11,700 earning >50K, showing a clear majority in the lower income category.""")

        st.markdown(""" 
        **Conclusion:**  
        The dataset primarily represents working-age individuals in private sector jobs, 
        with noticeable demographic dominance across gender and race. Income is moderately 
        imbalanced, and several numerical variables show strong right-skewness. 
        Overall, the data reflects structured employment and demographic concentration patterns.

        **While univariate analysis highlights distributional patterns within individual variables, comprehensive understanding demands bivariate and multivariate exploration to uncover interdependencies and predictive relationships.**
        """)

        st.markdown(
            """
            <p style='text-align:center; color:#4c72b0; font-size:16px; font-weight:bold;'>
            Thank you! Univariate Analysis Completed. Next target: Bivariate & Multivariate Analysis.
            </p>
            """,unsafe_allow_html=True
            )

