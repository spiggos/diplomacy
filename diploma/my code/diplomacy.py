import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import networkx as nx
import osmnx as ox
import osmnx.distance as distance
from datetime import timedelta
import geopy.distance
from geopy.distance import distance
from geopy.distance import great_circle
from osmnx.distance import nearest_nodes
import geopandas as gpd
import folium
import webbrowser
from folium.plugins import MarkerCluster
import sklearn.neighbors
import sklearn
import requests
import geopy.distance
from geopy.distance import distance, Point
from geopy.geocoders import Nominatim
from geopy import distance
import folium
import webbrowser
from folium.plugins import MarkerCluster
import requests
import json
import googlemaps


df=pd.read_excel('excel files/hpeiros_new.xlsx')
hpeirosdata=df.loc[:,['longtitude','latitude']]
hpeirosdata=hpeirosdata.drop_duplicates()
print(hpeirosdata)
print(len(hpeirosdata['longtitude']))
print(len(hpeirosdata['latitude']))


hpeiros_all_centers=pd.read_excel('excel files/hpeiros_all_centers.xlsx')
print(len(hpeiros_all_centers['all_centers_long']))
print(hpeiros_all_centers['all_centers_long'], hpeiros_all_centers['all_centers_lat'])



long_list=[]
lat_list=[]
for i in hpeirosdata['longtitude']:
    digit_counts = len(str(i))
    if digit_counts==10:
        long = i*pow(10,-8)
        long_list.append(long)
    if digit_counts==9:
        long = i*pow(10,-7)
        long_list.append(long)
    if digit_counts==8:
        long = i*pow(10,-6)
        long_list.append(long)

for j in hpeirosdata['latitude']:
    digit_counts2 = len(str(j))
    if digit_counts2==10:
        lat = j*pow(10,-8)
        lat_list.append(lat)
    if digit_counts2==9:
        lat = j*pow(10,-7)
        lat_list.append(lat)
    if digit_counts2==8:
        lat = j*pow(10,-6)
        lat_list.append(lat)

print(long_list)
print(lat_list)
print(len(long_list))
print(len(lat_list))

hpeiros= pd.DataFrame({'longtitude':long_list,'latitude':lat_list})
print(hpeiros)
hpeiros.to_excel("hpeiros_geodata.xlsx")

fixed6 = hpeiros.dropna()

# Dhmiourgia zeugaria shmeiwn gia thn euresh apostasewn meta3u shmeiwn zhthshs kai upopsifiwn shmeiwn(domes ugeias)
points_list=[]
for i, row1 in fixed6.iterrows():
    for i, row2 in hpeiros_all_centers.iterrows():
        long1=row1['longtitude']
        lat1=row1['latitude']
        points1=(lat1, long1)
        long2=row2['all_centers_long']
        lat2=row2['all_centers_lat']
        points2=(lat2, long2)
        print("points1:",points1)
        print("points2:",points2)
        pair=[points1,points2]
        points_list.append(pair)
print(points_list)

print(len(points_list))

# Function to calculate Haversine distance
def haversine_distance(coord1, coord2):
    return geopy.distance.geodesic(coord1, coord2).km


distances = []

for start, end in points_list:
    distance_km = haversine_distance(start, end)
    distances.append(distance_km)

print(distances)
print(len(distances))


# dhmiourgia listas me ta shmeia zhthshs(perioxes pou menoun oi hlikiwmenoi)
df=pd.read_excel('excel files/hpeiros_new.xlsx')
areas=df.loc[:,['house']]
areas=areas.drop_duplicates()
print(areas)

non_median=[]

for i in areas['house']:
    non_median.append(i)
print(non_median)
print(len(non_median))


#dhmiourgia listas me tis ypopsifies 8eseis(kentra ygeias)
df=pd.read_excel('excel files/hpeiros_all_centers.xlsx')
all_centers=df.loc[:,['all_centers']]
all_centers=all_centers.drop_duplicates()
print(all_centers)

candidate_location=[]

for i in all_centers['all_centers']:
    candidate_location.append(i)
print(candidate_location)
print(len(candidate_location))

#dhmiourgia Dij matrix me apostaseis
D = np.reshape(distances, (40, 12))
print(D)



#----FACTORS OF DEMAND----
#1) People_over_65_Population_COUNTY
#2) Average age of elder people
#3) GDP_per_capita_COUNTY
#4) Euro_per_inhabitant_EU27


# Kanonikopoihsh paragontwn
df=pd.read_excel('excel files/hpeiros_new.xlsx')
#---FACTOR 1 OF DEMAND----
elder_people = []
areas = df['house'].unique()
print(len(areas))

for a in areas:
    filtered_df = df[df['house'] == a]
    people_over_65 = filtered_df['People_over_65_Population_COUNTY'].unique()
    if len(people_over_65) > 0:
        people_over_65 = people_over_65[0] # Take the first value
        convert=float(people_over_65)
        elder_people.append(convert)
        print(f"Area: {a}, People over 65: {people_over_65}")


print(elder_people)
print(len(elder_people))


value1=max(elder_people)
value2=min(elder_people)

print(value1)
print(value2)

normalized_factor1=[]
for i in elder_people:
    normalized_people_over_65=(i-value2)/(value1-value2)
    normalized_factor1.append(normalized_people_over_65)
print(f"normalized_factor1: {normalized_factor1}")



#---FACTOR 2 OF DEMAND---
averages=[]
average=df.groupby('house')['age'].mean()

for i in average:
    avg=int(i)
    averages.append(avg)

print(averages)
print(len(averages))

value1=max(averages)
value2=min(averages)

print(value1)
print(value2)

normalized_factor2=[]
for i in averages:
    normalized_average=(i-value2)/(value1-value2)
    normalized_factor2.append(normalized_average)
print(f"normalized_factor2: {normalized_factor2}")



#---FACTOR 3 OF DEMAND----
gdp=[]

for a in areas:
    filtered_df = df[df['house'] == a]
    gdp_per_capita = filtered_df['GDP_per_capita_COUNTY'].unique()
    if len(gdp_per_capita) > 0:
        gdp_per_capita = gdp_per_capita[0] # Take the first value
        convert=float(gdp_per_capita)
        gdp.append(convert)
        print(f"Area: {a}, GDP_per_capita: {gdp_per_capita}")


print(gdp)
print(len(gdp))


value1=max(gdp)
value2=min(gdp)

print(value1)
print(value2)

normalized_factor3=[]
for i in gdp:
    normalized_gdp_per_capita=(i-value2)/(value1-value2)
    normalized_factor3.append(normalized_gdp_per_capita)
print(f"normalized_factor3: {normalized_factor3}")



#---FACTOR 4 OF DEMAND----
euro = []

for a in areas:
    filtered_df = df[df['house'] == a]
    euro_per_inhabitant = filtered_df['Euro_per_inhabitant_EU27'].unique()
    if len(euro_per_inhabitant) > 0:
        euro_per_inhabitant = euro_per_inhabitant[0] # Take the first value
        convert=float(euro_per_inhabitant)
        euro.append(convert)
        print(f"Area: {a}, Euro_per_inhabitant: {euro_per_inhabitant}")


print(euro)
print(len(euro))


value1=max(euro)
value2=min(euro)

print(value1)
print(value2)

normalized_factor4=[]
for i in euro:
    normalized_euro_per_inhabitant=(i-value2)/(value1-value2)
    normalized_factor4.append(normalized_euro_per_inhabitant)
print(f"normalized_factor4: {normalized_factor4}")


#Weights and demand calculation
wi1=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] #demand array all ones (test)

w1=0.3
w2=0.3
w3=0.2
w4=0.2
a1=[]
a2=[]
a3=[]
a4=[]
wi2=[]
for i in normalized_factor1:
    multi1= i*w1
    a1.append(multi1)

print(a1)
print(len(a1))

for i in normalized_factor2:
    multi2= i*w2
    a2.append(multi2)

print(a2)
print(len(a2))
for i in normalized_factor3:
    multi3= i*w3
    a3.append(multi3)

print(a3)
print(len(a3))

for i in normalized_factor4:
    multi4= i*w4
    a4.append(multi4)

print(a4)
print(len(a4))

for item1,item2,item3,item4 in zip(a1,a2,a3,a4):
    sum=item1+item2+item3+item4
    wi2.append(sum)

print(wi2)
print(len(wi2))


from pulp import *
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#-----------P-median----------------
p=5
#DECISION VARIABLES
X = LpVariable.dicts('X',(candidate_location),0,1,LpBinary)
print("\nX:\n",X)

Y = LpVariable.dicts('Y',
[(i,j) for i in non_median
       for j in candidate_location],0,1,LpBinary)
print(type(Y))
print("\nY:\n",Y)
print("\nY_list:\n",Y.values())
allocation = np.array(list(Y.values())).reshape(40,12)
print("\nallocation:\n",allocation)

demand=np.array(wi2).reshape(40,1)
print("\ndemand:\n",demand)

#FORMULATION
# MODEL: MINIMIZATION problem
model = LpProblem('P Median', LpMinimize)

#OBJECTIVE FUNCTION
obj_func = lpSum([demand[i]*lpDot(D[i], allocation[i]) for i in range(40)])
print("\nOBJECTIVE FUNCTION:\n",obj_func)
model += obj_func

#CONSTRAINTS
model += lpSum(X[j] for j in candidate_location) == p
for i in non_median:
    model += lpSum(Y[i,j] for j in candidate_location) == 1

for i in non_median:
    for j in candidate_location:
        model +=  Y[i,j] <= X[j]

output_path = 'p-median.lp'

#model.writeLP(output_path)

with open(output_path, 'w', encoding='utf-8') as file:
    # Write the LP model to the file
    file.write(str(model))
               
print("\nModel:\n",model)
model.solve()

#FORMAT OUTPUT
print("Objective: ",value(model.objective))
print(' ')

for v in model.variables():
    subV = v.name.split('_')

    if subV[0] == "X" and v.varValue == 1:
        print('p-Median Node: ', subV[1])

print(' ')
for v in model.variables():
    subV = v.name.split('_')
    if subV[0] == "Y" and v.varValue == 1:
        print(subV[1], ' is connected to', subV[2])

        #--APOTYPWSH APOTELESMATWN SE GRAFO--
#--endexomenws na mporesw na apotupwsw ta apotelesmata kai se xarth--
import matplotlib.pyplot as plt
# Create an empty graph
G = nx.Graph()

# Add nodes
for name in non_median:
    G.add_node(name)

for name2 in candidate_location:
    G.add_node(name2)



edges_list = []

for v in model.variables():
    subV = v.name.split('_')
    if subV[0] == "Y" and v.varValue == 1:
        print(subV[1], ' is connected to', subV[2])
        edge_tuple = (subV[1], subV[2])
        edges_list.append(edge_tuple)

# Add edges to the graph
G.add_edges_from(edges_list)


plt.figure(figsize=(15,15))

# Draw the graph
nx.draw(G, with_labels=True, node_color=['lightgrey' if node in non_median else 'lightgreen' for node in G.nodes()], node_size=800, font_size=10, font_color='black')


# Show the graph"""
plt.show()


#-----Map only Elder People-------
map1 = folium.Map(
    location=[38.2745,23.8103],
    tiles='openstreetmap',
    zoom_start=7,
)

for index, row in fixed6.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longtitude']],
        radius=5,
        icon=folium.Icon(color='green', icon=''),
        popup="Elder People",
        fill = True,
        fill_color='green',
        fill_opacity=0.6
    ).add_to(map1)

map1.save('elder_people.html')
webbrowser.open('elder_people.html')


#-----Map only Centers-------
map2 = folium.Map(
    location=[38.2745,23.8103],
    tiles='openstreetmap',
    zoom_start=7,
)

for index, row in hpeiros_all_centers.iterrows():
    folium.Marker(
        location=[row['all_centers_lat'], row['all_centers_long']],
        icon=folium.Icon(color='red', icon=''),
        popup="Centers",
        fill = True,
        fill_color='red',
        fill_opacity=0.6
    ).add_to(map2)

map2.save('centers.html')
webbrowser.open('centers.html')