#!/usr/bin/env python
# coding: utf-8

# ## ABDULLAHI UMAR FARUK

# ## Problem Statement:
# Perform a basic exploratory data analysis of the features of the Unicorn Companies
# dataset and come up with at least FOUR (4) data driven overall recommendations to help 
# Unicorn Companies in creating good business models and making decisions that will 
# focus on companies with high growth potential, diversify investment portfolio and 
# prioritize companies with experienced leadership teams.

# In[1]:


#Import the libraries
#for mathematical computationimport numpy as np

import numpy as np
import pandas as pd
import scipy.stats as stats

# For Visaulization of Data
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px 
from matplotlib.pyplot import figure 
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Imporing the dataset from csv file
df = pd.read_csv(r'C:\Users\FARUK MANDIBO\Desktop\PYTHON\faruk\Unicorn_Companies.csv')
df.head()


# In[3]:


## Checking for missing values 
df.isnull().sum()


# In[4]:


# visualize missing value using heatmap
plt.figure(figsize = (12, 6))
sns.heatmap(df.isnull(), cbar = True, cmap = 'cool')
plt.title('Visualizing missing values')
plt.show()


# ## Observation
# 
# --Missing Values Summary: The summary shows that some columns have missing values (NaN or null values), while others have no missing values.
# 
# Columns with Missing Values: There are several columns with missing values, most notably "City" and "Select Investors." These columns have 16 and 1 missing values, respectively.
# 
# --Impact on Analysis: The presence of missing values can impact your data analysis. For example, if you plan to use the "City" column for geographical analysis, you may need to address the missing values to ensure your analysis is comprehensive.
# 
# --Data Cleaning: To address missing values, you can consider techniques such as imputation, where you fill in missing values with appropriate data (e.g., filling missing cities based on the country or using mode imputation for "Select Investors").
# 
# --Data Quality: It's essential to assess whether the missing data affects the quality and reliability of your analysis. Depending on your objectives, you may need to decide how to handle missing values, whether through imputation or excluding rows with missing data.
# 
# --Data Preprocessing: This step is a critical part of data preprocessing before performing any analysis or visualization. Addressing missing values ensures that your analysis is based on complete and accurate data.
# 
# Consider Data Sources: If the missing data in "City" is related to the source of your dataset, you might want to verify whether the original data source contains this information or if there are alternative sources to supplement the missing data.
# 
# --Documentation: Document the steps you take to handle missing values. Transparency in data preprocessing is essential for reproducibility and sharing your findings with others.
# 
# In summary, the presence of missing values is a common data issue, and addressing them is crucial for accurate and reliable data analysis. Depending on your specific analysis objectives, you may choose to impute missing data or take other appropriate actions to ensure the quality of your dataset.
# 
# 
# 

# In[5]:


## Data Cleaning
# Renaming columns
df.rename(columns={'Date Joined': 'Date', 'Valuation ': 'Valuation (B$)', 'Select Investors': 'Investors'}, inplace=True)

# Convert 'Investors' column to a string
df['Investors'] = df['Investors'].astype(str)
df


# In[6]:


# Remove '$' from 'Valuation' column and convert it to numeric
df['Valuation'] = df['Valuation'].str.replace('$', '')
df['Valuation'] = df['Valuation'].str.replace('B', '')
df['Valuation'] = pd.to_numeric(df['Valuation'])
df


# In[7]:


# Remove '$' from 'Valuation' column and convert it to numeric
df['Funding'] = df['Funding'].str.replace('$', '')
df['Funding'] = df['Funding'].str.replace('B', '')
df['Funding'] = df['Funding'].str.replace('M', '')
df['Funding'] = df['Funding'].str.replace('Unknown', '')
df['Funding'] = pd.to_numeric(df['Funding'])
df


# In[8]:


# Calculate ROI and add it as a new column
df['ROI'] = df['Valuation'] - df['Funding']
df


# In[9]:


# check for negative values in quantity
df[df['ROI'] < 0]


# In[10]:


# Optionally, you can sort the DataFrame by 'ROI' in ascending order
df = df.sort_values(by='ROI', ascending=False).head()
df


# ## Data cleaning is important for getting a good result.
# In the first segment, I renamed a few column names.
# Like, Data Joined to Date, Select Investors to Investors and aslo create a new column called Return upon Investment (ROI).
#  
# These data should be numeric. So, using pandas we assign the data type to numeric.
# Now in the valuation column, We know that all the values are in Billion Dollars. 
# That’s why we removed the ($B) from the column. And make sure that the data type should be numeric.
# 

# In[ ]:





# In[11]:


df.columns.to_list()


# In[12]:


df.describe().astype('int')


# ## Observation
# Valuation:
# 
# The dataset contains information on the valuation of companies, with a count of 1074 data points.
# The minimum valuation is 1, indicating that there are startups with relatively low valuations.
# The maximum valuation is 180, which is significantly higher than the median valuation, suggesting the presence of some exceptionally high-valued companies.
# The mean valuation is approximately 3, indicating that, on average, companies in the dataset have a valuation of around 3 billion dollars.
# The standard deviation of 8 indicates that there is relatively high variability in valuations among the companies in the dataset.
# Year Founded:
# 
# The dataset contains information on the year in which the companies were founded.
# The minimum year founded is 1919, suggesting that some of the companies in the dataset have been established for a long time.
# The maximum year founded is 2021, indicating the presence of recently founded startups.
# The mean year founded is approximately 2012, suggesting that, on average, the companies in the dataset were founded around 2012.
# The standard deviation of 5 indicates relatively low variability in the founding years among the companies.
# Funding:
# 
# The dataset contains information on the funding amounts received by companies.
# The minimum funding amount is 0, which could indicate some companies with no recorded funding.
# The maximum funding amount is 999, indicating the presence of companies that have received significant funding.
# The mean funding amount is approximately 338, suggesting that, on average, companies in the dataset have received around 338 million dollars in funding.
# The standard deviation of 237 indicates that there is a wide range of funding amounts among the companies.
# ROI (Return on Investment):
# 
# The dataset contains information on the ROI, calculated as the difference between funding and valuation.
# The minimum ROI is -172, indicating that some companies have a negative ROI, meaning their valuation is higher than their funding.
# The maximum ROI is 996, indicating that some companies have achieved a very high ROI.
# The mean ROI is approximately 334, suggesting that, on average, companies in the dataset have achieved a positive ROI of around 334 million dollars.
# The standard deviation of 238 indicates variability in ROI among the companies, including both positive and negative returns.
# These observations provide insights into the distribution and characteristics of the data, including the range of valuations, founding years, funding amounts, and ROI values among the companies in the dataset.
# 
# 
# 
# 
# 
# 

# In[13]:


df.info()


# In[14]:


# convert the 'Date' to Datetime
df['Date'] = pd.to_datetime(df['Date'])

# fill the missing values of 'Country' with the mode
df['Country'].fillna(df['Country'].mode()[0], inplace=True)

# replace the negative values in ROI with 0
df['ROI'] = np.where(df['ROI'] < 0, 0, df['ROI'])

# check the head of the data
df


# In[15]:


company_wise_valuation=df.sort_values(by='Valuation', ascending = False)
company_wise_valuation


# ## Domination of Industries in unicorn Company

# In[16]:


fig = px.pie(df, names='Industry', title =' 4 Largest Company in unicorn Chart')
fig.show()


# ## Observation
# 40% of the Unicorn startups Industry is Fintech. 
# Financial technology is the technology and innovation that aims to compete with traditional financial methods in the delivery of financial services.
# It is an emerging industry that uses technology to improve activities in finance.

# ## The year-wise companies joined the Unicorn

# In[17]:


fig3 = px.line(df, x='Company', y='Year Founded',
               title='Based on Year Company Joined')
fig3.show()


# ## Observation
# These are a few startups that joined the unicorn club by year-wise. 
# You can move your cursor on the graph and you will see the year in which they joined the unicorn club.

# ## Arranging the company based on Valuation

# In[18]:


# Top 5 Country with most Valuation 

df[['Company', 'Valuation']]=df[['Company', 'Valuation']].sort_values(by='Valuation', ascending=False)
df.head()


# ## Observation

# In the dataset we have, the Bytedance from China is the most valued startup with a total valuation of $180B. 
# SpaceX is the second most valued startup with a valuation of $100B. SpaceX Joined the unicorn club in 2012.
# If you want to look for any particular company then you can do that using the following way.
# 

# In[19]:


import matplotlib.pyplot as plt

# Assuming 'df' contains the sorted DataFrame with 'Company' and 'Valuation'

# Select the top 5 companies with the most valuation
top_5_companies = df[['Company', 'Valuation']].sort_values(by='Valuation', ascending=False).head()

# Create a vertical column chart to visualize the top 5 companies by valuation
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.bar(top_5_companies['Company'], top_5_companies['Valuation'], color='blue')  # Use barh for horizontal bars
plt.xlabel('Company')
plt.ylabel('Valuation')
plt.title('Top 5 Companies with Most Valuation')

# Show the column chart
plt.show()


# In[20]:


#Cites with most unicorn 

import matplotlib.pyplot as plt

# Data
cities = [
    "San Francisco", "New York", "Beijing", "Shanghai", "London", "Bengaluru",
    "Paris", "Shenzhen", "Palo Alto", "Berlin", "Boston", "Hangzhou", "Chicago",
    "Mountain View", "Tel Aviv"
]
counts = [152, 103, 63, 44, 34, 29, 19, 19, 18, 17, 16, 16, 16, 16, 11]

# Create a column chart
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.barh(cities, counts, color='blue')
plt.xlabel('Count')
plt.ylabel('City')
plt.title('Top 15 Cities with most unicorn')
plt.gca().invert_yaxis()  # Invert the y-axis to display the highest count at the top

# Display the chart
plt.show()


# # Observation
# As expected, San Francisco City has the highest number of Unicorn Startups. 
# Bengaluru City has 25. New York has 81.
# 

# In[21]:


# TOP 5 COUNTRY WITH MOST UNICORN START UP
df.Country.value_counts().head()


# In[22]:


import plotly.express as px



# Create a choropleth map chart
fig = px.choropleth(
    df,
    locations='Country',  # Specify the column containing country names
    locationmode='country names',  # Use country names for location mode
    color='Funding',  # Color the map based on the 'Funding' column
    hover_name='Country',  # Show country names on hover
    title='Top Country with Highest Funding',
    color_continuous_scale=px.colors.sequential.Plasma  # Choose a color scale
)

# Customize the map appearance
fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")

# Show the map chart
fig.show()



# ## Observations

# 
# 
# -The choropleth map chart displays the funding amounts by country, allowing us to identify the top 20 countries with the highest funding for the given dataset. Each country is represented on the map by a color, with darker shades indicating higher funding amounts.
# 
# -Observations from the map chart:
# 
# -Geographical Distribution: The map provides a geographical perspective on where the top-funded companies or investments are located. It allows us to visually compare funding levels across different countries.
# 
# -Top Funding Countries: The countries with the darkest shades on the map are the top 20 countries with the highest funding amounts. These countries have attracted significant investments.
# 
# -Hover Information: Hovering over a country on the map reveals additional details, including the country's name and the specific funding amount. This interactive feature enables users to explore the data further.
# 
# -Color Scale: The choice of the Plasma color scale helps differentiate funding levels effectively. Darker colors represent higher funding, while lighter colors represent lower funding.
# 
# -Geographical Patterns: The map may reveal geographical patterns in funding, such as regions or clusters where investment activity is particularly concentrated.
# 
# Overall, the map chart offers a clear visual representation of funding distribution across countries and helps identify the top funding destinations in the dataset.
# 
# 
# 
# 
# 
# 
# 

# ### RECOMMENDATION

# *After performing a basic exploratory data analysis of the Unicorn Companies dataset, here are four data-driven overall recommendations to help Unicorn Companies in creating good business models, making investment decisions that focus on companies with high growth potential, diversifying their investment portfolio, and prioritizing companies with experienced leadership teams:
# 
# Identify High Growth Potential Sectors:
# 
# *Recommendation: Analyze industry sectors and identify those with a high growth potential based on historical funding trends and valuation growth.
# Rationale: By understanding which industry sectors have attracted the most funding and demonstrated substantial valuation growth, Unicorn Companies can focus their investments on sectors that are likely to continue flourishing. This approach can help maximize returns on investments.
# Diversify Investment Portfolio Across Regions:
# 
# *Recommendation: Diversify the investment portfolio across different regions or countries to reduce geographic concentration risk.
# Rationale: Geographical diversification can mitigate risks associated with economic, political, or regulatory changes in a particular region. By investing in companies from diverse regions, Unicorn Companies can spread their risk and access a broader range of growth opportunities.
# Prioritize Companies with Experienced Leadership Teams:
# 
# *Recommendation: Place a strong emphasis on companies with experienced and proven leadership teams when making investment decisions.
# Rationale: The quality of a company's leadership team plays a critical role in its success. Companies led by experienced founders and executives are more likely to navigate challenges effectively and execute growth strategies. Unicorn Companies should conduct due diligence on the leadership teams of potential investments.
# Monitor Early-Stage Startups for Growth Potential:
# 
# *Recommendation: Continuously monitor early-stage startups in the portfolio for signs of high growth potential, such as rapid user adoption, revenue growth, and market expansion.
# Rationale: Startups can experience exponential growth at various stages of their development. Unicorn Companies should stay vigilant and adapt their investment strategies to support and nurture companies that exhibit strong growth indicators. Regularly reassessing the portfolio ensures that they capitalize on emerging opportunities.
# These recommendations are data-driven and aim to help Unicorn Companies make informed decisions to create robust business models, maximize returns, and manage risk effectively. By focusing on sectors with growth potential, diversifying investments, prioritizing experienced leadership, and actively monitoring startups, Unicorn Companies can enhance their investment strategy and increase their chances of success in the dynamic world of startup investments.
# 
# 

# In[ ]:




