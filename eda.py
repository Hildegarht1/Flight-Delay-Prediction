import plotly.express as px
import plotly.io as pio
from etl import load_data, clean_data

# Render charts in notebook or browser
pio.renderers.default = "notebook"

df = clean_data(load_data())

# Total Distance Flown by Airline
fig1 = px.bar(df.groupby('Airline')['Distance'].sum().reset_index(),
              x='Airline', y='Distance', color='Airline',
              title='Total Distance Flown by Airline')
fig1.show()