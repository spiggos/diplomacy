 #---------------TASKS---------------
#να φτιαξω το google maps
#να φτιαξω ενα function για να διαβάζει τα excel
#να βάλω τις τιμές των rural areas

import itertools
import math
import io
import os
import webbrowser
import geopy
from geopy.distance import distance
import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import networkx as nx
import folium

try:
    file_path1 = os.path.join("excel files", "hpeiros_new.xlsx")
    df1 = pd.read_excel(file_path1)
    names = df1.loc[:,['CODENAME']].drop_duplicates()
    people_coords = df1.loc[:,['longtitude','latitude']].drop_duplicates()
except FileNotFoundError:
    print("Excel file not found.")
except pd.errors.ParserError:
    print("Invalid excel file format.")

try:
    file_path2 = os.path.join("excel files", "hpeiros_all_centers.xlsx")
    df2 = pd.read_excel(file_path2)
    center_coords = df2.loc[:,['all_centers_long','all_centers_lat']].drop_duplicates()
except FileNotFoundError:
    print("Excel file not found.")
except pd.errors.ParserError:
    print("Invalid excel file format.")

try:
    file_path3 = os.path.join("excel files", "geocode synopsis EG 06_12.xlsx")
    df3 = pd.read_excel(file_path3)
    people_ari = df3.loc[:,['CODENAME','ARI(gr)']].drop_duplicates().dropna() 
except FileNotFoundError:
    print("Excel file not found.")
except pd.errors.ParserError:
    print("Invalid excel file format.")



#μετατροπη συντεταγμένων longtitude και latitude 
DIGIT_COUNT_MAPPING = {
    10: -8,
    9: -7,
    8: -6
}

long_list = []

for longtitude in people_coords['longtitude']:
    digit_counts = math.floor(math.log10(longtitude)) + 1
    if digit_counts in DIGIT_COUNT_MAPPING:
        long = longtitude * pow(10, DIGIT_COUNT_MAPPING[digit_counts])
        long_list.append(long)

lat_list = [] 

for latitude in people_coords['latitude']:
    digit_counts2 = math.floor(math.log10(latitude)) + 1
    if digit_counts2 in DIGIT_COUNT_MAPPING:
        lat = latitude * pow(10, DIGIT_COUNT_MAPPING[digit_counts])
        lat_list.append(lat)

#δημιουργεια excel απο το lat_list και long_list 
people_geo= pd.DataFrame({'longtitude':long_list,'latitude':lat_list})
people_geo.to_excel("excel files/people_geodata.xlsx", index=False)
fixed6 = people_geo.dropna()

#δημιουργεια ζευγων (tuple) για το people_coords
people_tuple = list(fixed6.itertuples(index=False, name=None))

#δημιουργεια ζευγων (tuple) για το centers_coords
fixed6 =center_coords.dropna()
centers_tuple = list(fixed6.itertuples(index=False, name=None))

#δημιουργεία (tuple1,tuple2)
pair = list(itertools.product(people_tuple, centers_tuple))
        
#δημιουργεια excel απο το pair(tuple1,tuple2)
pair_df = pd.DataFrame(pair, columns=['people_coords', 'centers_coords'])
#pair_df.to_excel("excel files/coords.xlsx", index=False)

def haversine_distance(coord1, coord2):
    return geopy.distance.geodesic(coord1, coord2).km      

#δημιουργεία αποστάσεων pairs
distance = list()

for start, end in pair:
    distance_km = haversine_distance(start, end)
    distance.append(distance_km)

#δημιουργεία distance excel
km_df = pd.DataFrame(distance,columns=['km'])
#km_df.to_excel("excel files/distance.xlsx",index=False)


#δημιουργεία ενός ενωμένου excel
merged_df = pd.merge(pair_df, km_df, left_index=True, right_index=True)
merged_df.to_excel('excel files/coords_and_distance .xlsx', index=False)


#δημιουργεια λιστας για τις περιοχές
areas = df1.loc[:,['house']].drop_duplicates()
non_median = [area for area in areas['house']]

 
#δημιουργεία excel με τις περιοχές
non_median_df = pd.DataFrame(non_median,columns=['non_median'])
non_median_df.to_excel('excel files/only_non_median.xlsx', index=False)

#δημιουργεία λίστας για τα ιατρικά κέντρα
centers = df2.loc[:,['all_centers']].drop_duplicates()
candidate_location = [center for center in centers['all_centers']]

#δημιουργεία excel με τα ιατρικά κέντρα 
centers_df = pd.DataFrame(candidate_location,columns=['centers'])
non_median_df.to_excel('excel files/only_centers.xlsx', index=False)

#δημιουργεία excel με τα ονόματα και τα ari's τους
name_ari_df = names.merge(people_ari, on='CODENAME', how='inner')
name_ari_df.to_excel('excel files/names_and_ari.xlsx', index=False)

#----FACTORS OF DEMAND----
#1) People_over_65_Population_COUNTY
#2) Average age of elder people
#3) GDP_per_capita_COUNTY
#4) Euro_per_inhabitant_EU27
#5) Accessibility/remoteness in (gr)

#1)People_over_65_Population_COUNTY
elder_people =[]
areas = df1['house'].unique()
print(areas)
for area in areas : 
    filtered_df = df1[df1['house'] == area]
    people_over_65 = filtered_df['People_over_65_Population_COUNTY'].unique()
    if len(people_over_65) > 0:
        people_over_65 = people_over_65[0] # sπάρε τη πρώτη τιμή
        convert=float(people_over_65)
        elder_people.append(convert)
        #print(f"Area: {area}, People over 65: {people_over_65}")

#min max των ηλικιωμένων
MAX_ELDER = df1['People_over_65_Population_COUNTY'].max()
MIN_ELDER = df1['People_over_65_Population_COUNTY'].min()

#normalized factor
normalized_factor1 = []
if MAX_ELDER != MIN_ELDER :
    if elder_people:
        normalized_factor1 = [(elder-MIN_ELDER)/(MAX_ELDER-MIN_ELDER) for elder in elder_people]
    #print(f"normalized_factor1: {normalized_factor1}")

#2)Μέσος όρος ηλικίας των ηλικιωμένων
averages = []
mean_age = df1.groupby('house')['age'].mean()
averages = [int(average) for average in mean_age]

#min max των μ.ο ηλικιών
MAX_AVERAGE = max(averages)
MIN_AVERAGE = min(averages)

#normalized factor
normalized_factor2 = []
if MAX_AVERAGE != MIN_AVERAGE :
    if averages:
        normalized_factor2 = [(avg-MIN_AVERAGE)/(MAX_AVERAGE-MIN_AVERAGE) for avg in averages]
    #print(f"normalized_factor2: {normalized_factor2}")

#3)GDP
gdps = []

for area in areas :
    filtered_df = df1[df1['house'] == area]
    gdp_per_capita = filtered_df['GDP_per_capita_COUNTY'].unique()
    if len(gdp_per_capita) > 0 : 
        gdp_per_capita = gdp_per_capita[0] #πάρε τη πρώτη τιμή
        convert=float(gdp_per_capita)
        gdps.append(convert)
        #print(f"Area: {area}, GDP_per_capita: {gdp_per_capita}")

#min max gdp
MAX_GDP = df1['GDP_per_capita_COUNTY'].max()
MIN_GDP = df1['GDP_per_capita_COUNTY'].min()

#normalized factor
normalized_factor3 = []
if MAX_GDP != MIN_GDP :
    if gdps:
        normalized_factor3 = [(gdp-MIN_GDP)/(MAX_GDP-MIN_GDP) for gdp in gdps]
    #print(f"normalized_factor2: {normalized_factor3}")

#Euros
euros = []

for area in areas:
    filtered_df = df1[df1['house'] == area]
    euro_per_inhabitant = filtered_df['Euro_per_inhabitant_EU27'].unique()
    if len(euro_per_inhabitant) > 0:
        euro_per_inhabitant = euro_per_inhabitant[0] # Take the first value
        convert=float(euro_per_inhabitant)
        euros.append(convert)
       # print(f"Area: {area}, Euro_per_inhabitant: {euro_per_inhabitant}")

#min max euro
MAX_EURO = df1['Euro_per_inhabitant_EU27'].max()
MIN_EURO = df1['Euro_per_inhabitant_EU27'].min()

#normalized factor
normalized_factor4=[]
if MAX_EURO != MIN_EURO :
    if euros:
        normalized_factor4 = [(euro-MIN_EURO)/(MAX_EURO-MIN_EURO) for euro in euros]
    #print(f"normalized_factor2: {normalized_factor4}")
        
#ARI(gr)
ari_list = []
aris = name_ari_df['ARI(gr)'].unique()
ari_list = [float(ari) for ari in aris]

MAX_ARI = max(ari_list)
MIN_ARI = min(ari_list)
if MAX_ARI != MIN_ARI :
    if ari_list:
        normalized_factor5 = [(ari-MIN_ARI)/(MAX_ARI-MIN_ARI) for ari in ari_list]
    #print(f"normalized_factor5: {normalized_factor5}")

#δημιουργεία ενός ενωμένου excel
factor1_df = pd.DataFrame(normalized_factor1,columns=['over 65'])
factor2_df = pd.DataFrame(normalized_factor2,columns=['avg_age'])
factor3_df = pd.DataFrame(normalized_factor3,columns=['gdp'])
factor4_df = pd.DataFrame(normalized_factor4,columns=['Euro'])
factor5_df = pd.DataFrame(normalized_factor5,columns=['Ari'])    
merged_df = pd.merge(factor1_df, factor2_df, on=None, left_index=True, right_index=True)
merged_df = pd.merge(merged_df, factor3_df, on=None, left_index=True, right_index=True)
merged_df = pd.merge(merged_df, factor4_df, on=None, left_index=True, right_index=True)
merged_df = pd.merge(merged_df, factor5_df, on=None, left_index=True, right_index=True)
merged_df.to_excel('excel files/normalized_factors .xlsx', index=False)


#Υπολογισμός βαρών για κάθε κανονικοποιημένη συνάρτηση

#len(wi1) = 40 οσα και τα άτομα στην περιοχή
demand_array = [1] * len(people_coords)

weights = [0.2, 0.2, 0.2, 0.2,0.2]

a1 =[number*weights[0] for number in normalized_factor1]
a2 =[number*weights[1] for number in normalized_factor2]
a3 =[number*weights[2] for number in normalized_factor3]
a4 =[number*weights[3] for number in normalized_factor4]
a5 =[number*weights[4] for number in normalized_factor5]

result = [number1+number2+number3+number4+number5 for number1,number2,number3,number4,number5 in zip(a1,a2,a3,a4,a5)]

#δημιουργεία ενός result excel
result_df =pd.DataFrame(result, columns = ['result'])
result_df.to_excel('excel files/weight_and_factors_result.xlsx', index=False)


                   #-----------P-median----------------
def solve_p_median(p):
    # Set the working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    X = LpVariable.dicts('X',(candidate_location),0,1,LpBinary)
    Y = LpVariable.dicts('Y',[(area,centers) for area in non_median for centers in candidate_location],0,1,LpBinary)

    #2D
    allocation = np.array(list(Y.values())).reshape(40,12)
    D = np.reshape(distance, (40,12))
    demand=np.array(result).reshape(40,1)

    #FORMULATION
    # MODEL: MINIMIZATION problem
    model = LpProblem('P Median', LpMinimize)

    #OBJECTIVE FUNCTION
    obj_func = lpSum([demand[i]*lpDot(D[i], allocation[i]) for i in range(40)])
    model += obj_func

    #CONSTRAINTS
    model += lpSum(X[j] for j in candidate_location) == p
    for i in non_median:
        model += lpSum(Y[i,j] for j in candidate_location) == 1

    for i in non_median:
        for j in candidate_location:
            model +=  Y[i,j] <= X[j]

    #WRITE THE LP MODEL TO A FILE
    output_folder = 'lp_files'
    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'p-median.lp'

    output_path = os.path.join(output_folder, output_filename)
    with io.open(output_path, 'w', encoding='utf-8') as file:
        file.write(str(model))

    model.solve()

    #FORMAT OUTPUT
    print("Objective: ",value(model.objective))
    print(' ')

    return model

solved_model = solve_p_median(4)


#Δημιουργεία γράφου
G = nx.Graph()

#Προσθήκη κομβων και ακμών
edges_list = []
for v in solved_model.variables():
        subV = v.name.split('_')
        if subV[0] == "Y" and v.varValue is not None and v.varValue == 1:
            edge_tuple = (subV[1], subV[2])
            edges_list.append(edge_tuple)
G.add_edges_from(edges_list)

#Χρώματα στα non_median και στα centers 
subV1_nodes = [edge[0] for edge in edges_list]
subV2_nodes = [edge[1] for edge in edges_list]

node_colors = []
for node in G.nodes():
    if node in subV1_nodes:
        node_colors.append('lightgrey')
    elif node in subV2_nodes:
        node_colors.append('green')
    else:
        node_colors.append('black')  # or any default color

plt.figure(figsize=(12,15))
nx.draw_networkx(G, with_labels=True, node_color=node_colors, node_size=800, font_size=10, font_color='black')
plt.show()


#δημιουργεία χάρτη
map = folium.Map(
    location=[38.2745,23.8103],
    tiles='openstreetmap',
    zoom_start=7,
)

for index, row in people_geo.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longtitude']],
        radius=5,
        icon=folium.Icon(color='green', icon=''),
        popup="Elder People",
        fill = True,
        fill_color='green',
        fill_opacity=0.6
    ).add_to(map)

    
for index, row in center_coords.iterrows():
    folium.Marker(
        location=[row['all_centers_lat'], row['all_centers_long']],
        icon=folium.Icon(color='red', icon=''),
        popup="Centers",
        fill = True,
        fill_color='red',
        fill_opacity=0.6
    ).add_to(map)    

#for edge in edges_list:
#    coord1 = (people_geo.loc[edge[0], 'latitude'], people_geo.loc[edge[0], 'longtitude'])
#    coord2 = (center_coords.loc[edge[1], 'all_centers_lat'], center_coords.loc[edge[1], 'all_centers_long'])
#    folium.PolyLine([coord1, coord2], color='blue', weight=2.5, opacity=1).add_to(map)
# Display the map
map.save('map.html')
#webbrowser.open('map.html')

               