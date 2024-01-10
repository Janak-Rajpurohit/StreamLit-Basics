import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

st.title("NBA Player Explorer")

st.markdown("""
This app performs simple web scraping for NBA player stats data.
* **Python Libraries:** base64, pandas, streamlit.
* **Data Source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header("Input Parameters")
selected_year = st.sidebar.selectbox("Year",list(reversed(range(1950,2020))))


# webscraping for nba players
@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_"+str(year)+"_per_game.html"
    html = pd.read_html(url,header=0)
    df = html[0]
    raw = df.drop(df[df.Age=='Age'].index)   #delete repeatition 
    raw = raw.fillna(0)

    playerstats = raw.drop(['Rk'],axis=1)
    return playerstats

playerstats = load_data(selected_year)

# sidebar Team Selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect("Select Teams",list(sorted_unique_team),default=list(sorted_unique_team))

unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect("Select Position",unique_pos,default=[unique_pos[2],unique_pos[4]])

# filtering data
df_filtered_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header("Display Player Stats of Selected Team(s)")
st.write("Data Dimension: "+str(df_filtered_team.shape[0])+" rows and "+str(df_filtered_team.shape[1])+" columns.")
st.dataframe(df_filtered_team)

def fileDownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  #string bytes conversion
    href = f"<a href = 'data:file/csv;base64,{b64}' download='playerstats.csv'> Download CSV File </a>"
    return href

st.markdown(fileDownload(df_filtered_team),unsafe_allow_html=True)

# if button is clicked (returns True)
if st.button("Heatmap"):
    st.header("Intercorrelation Matrix Heatmap")

    df_filtered_team.to_csv("output.csv",index=False)
    df=pd.read_csv('output.csv')
    # including those who are numerical dtype
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    corr = df[numeric_columns].corr()
    # corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)]=True
    with sn.axes_style("darkgrid"):
        fig,ax=plt.subplots(figsize=(7,5))
        ax = sn.heatmap(corr,mask=mask,vmax=1,square=True)
    st.pyplot(fig)


