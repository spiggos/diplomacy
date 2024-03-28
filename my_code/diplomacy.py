import io
import os
import webbrowser
import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import networkx as nx
import folium
import googlemaps
from polyline import decode as poly_decode


try:
    file_path1 = os.path.join("excel files", "hpeiros_new.xlsx")
    df1 = pd.read_excel(file_path1)
    names = df1.loc[:,['CODENAME']].drop_duplicates()
    people_coords = df1.loc[:,['longtitude','latitude']].drop_duplicates()
    info = df1.loc[:,['house']].drop_duplicates()
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
long_list= list()
lat_list= list()
for logntitude in people_coords['longtitude']:
    digit_counts = len(str(logntitude))
    if digit_counts==10:
        long = logntitude*pow(10,-8)
        long_list.append(long)
    if digit_counts==9:
        long = logntitude*pow(10,-7)
        long_list.append(long)
    if digit_counts==8:
        long = logntitude*pow(10,-6)
        long_list.append(long)

for latitude in people_coords['latitude']:
    digit_counts2 = len(str(latitude))
    if digit_counts2==10:
        lat = latitude*pow(10,-8)
        lat_list.append(lat)
    if digit_counts2==9:
        lat = latitude*pow(10,-7)
        lat_list.append(lat)
    if digit_counts2==8:
        lat = latitude*pow(10,-6)
        lat_list.append(lat)

#Δημιουργεία excel απο το lat_list και long_list 
people_info = pd.DataFrame({'house':info['house'],'longtitude':long_list,'latitude':lat_list})
people_geo= pd.DataFrame({'longtitude':long_list,'latitude':lat_list})
people_geo.to_excel("excel files/people_geodata.xlsx", index=False)
fixed6 = people_geo.dropna()

#Δημιουργεία pairs
pair= list()
for i, row1 in fixed6.iterrows():
    for i, row2 in df2.iterrows():
        long1=row1['longtitude']
        lat1=row1['latitude']
        points1=(lat1, long1)
        long2=row2['all_centers_long']
        lat2=row2['all_centers_lat']
        points2=(lat2, long2)
        points_list=[points1,points2]
        pair.append(points_list)

#Δημιουργεία excel για το pair
pair_df = pd.DataFrame(pair, columns=['people_coords', 'centers_coords'])
pair_df.to_excel("excel files/coords.xlsx", index=False)

#Υπολογισμός του distance 
def get_distance(api_key, start, end):
    gmaps = googlemaps.Client(key=api_key)
    # Request directions
    directions_result = gmaps.directions(start, end, mode="driving")
    polyline = directions_result[0]['overview_polyline']['points']
    decoded_polyline = poly_decode(polyline)
    # Extract the distance
    distance = directions_result[0]['legs'][0]['distance']['text']
    distance = distance.replace(' km', '')
    distance = float(distance)

    return distance, decoded_polyline


api_key = 'AIzaSyDTrLnYWUUgIylmNX5RUZlDaHfQx_MFrW8'

distance = list()
     
for start, end in pair:
    distance_km, decoded_polyline = get_distance(api_key, start, end)
    if distance_km is not None:
        distance.append(distance_km)


#Δημιουργεία distance excel
km_df = pd.DataFrame(distance,columns=['km'])   
km_df.to_excel("excel files/distance.xlsx",index=False)
#Δημιουργεία excel ditance και pairs
cd_df = pd.merge(pair_df, km_df, left_index=True, right_index=True)
cd_df.to_excel('excel files/coords_and_distance .xlsx', index=False)


#Δημιουργεία list για non-median
areas = df1.loc[:,['house']].drop_duplicates()
non_median = [area for area in areas['house']]
#Δημιουργεία excel για non-median
non_median_df = pd.DataFrame(non_median,columns=['non_median'])
non_median_df.to_excel('excel files/only_non_median.xlsx', index=False)


#Δημιουργεία list για τα centers
centers = df2.loc[:,['all_centers']].drop_duplicates()
candidate_location = [center for center in centers['all_centers']]
#Δημιουργεία excel με τα centers
centers_df = pd.DataFrame(candidate_location,columns=['centers'])
non_median_df.to_excel('excel files/only_centers.xlsx', index=False)


#Δημιουργεία excel με τα names και τα ari's τους
name_ari_df = names.merge(people_ari, on='CODENAME', how='inner')
name_ari_df.to_excel('excel files/names_and_ari.xlsx', index=False)


#----FACTORS OF DEMAND----
#1) People_over_65_Population_COUNTY
#2) Average age of elder people
#3) GDP_per_capita_COUNTY
#4) Euro_per_inhabitant_EU27
#5) Accessibility/remoteness in (gr)

#1)People_over_65_Population_COUNTY
elder_people = list()
areas = df1['house'].unique()

for area in areas : 
    filtered_df = df1[df1['house'] == area]
    people_over_65 = filtered_df['People_over_65_Population_COUNTY'].unique()
    if len(people_over_65) > 0:
        people_over_65 = people_over_65[0] #πάρε τη πρώτη τιμή
        convert=float(people_over_65)
        elder_people.append(convert)
        #print(f"Area: {area}, People over 65: {people_over_65}")

#min max των ηλικιωμένων
MAX_ELDER = df1['People_over_65_Population_COUNTY'].max()
MIN_ELDER = df1['People_over_65_Population_COUNTY'].min()

#normalized factor
normalized_factor1 = list()
if MAX_ELDER != MIN_ELDER :
    if elder_people:
        normalized_factor1 = [(elder-MIN_ELDER)/(MAX_ELDER-MIN_ELDER) for elder in elder_people]
    #print(f"normalized_factor1: {normalized_factor1}")

#2)Μέσος όρος ηλικίας των ηλικιωμένων
averages = list()
mean_age = df1.groupby('house')['age'].mean()
averages = [int(average) for average in mean_age]

#min max των μ.ο ηλικιών
MAX_AVERAGE = max(averages)
MIN_AVERAGE = min(averages)

#normalized factor
normalized_factor2 = list()
if MAX_AVERAGE != MIN_AVERAGE :
    if averages:
        normalized_factor2 = [(avg-MIN_AVERAGE)/(MAX_AVERAGE-MIN_AVERAGE) for avg in averages]
    #print(f"normalized_factor2: {normalized_factor2}")

#3)GDP
gdps = list()

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
normalized_factor3 = list()
if MAX_GDP != MIN_GDP :
    if gdps:
        normalized_factor3 = [(gdp-MIN_GDP)/(MAX_GDP-MIN_GDP) for gdp in gdps]
    #print(f"normalized_factor2: {normalized_factor3}")

#Euros
euros = list()

for area in areas:
    filtered_df = df1[df1['house'] == area]
    euro_per_inhabitant = filtered_df['Euro_per_inhabitant_EU27'].unique()
    if len(euro_per_inhabitant) > 0:
        euro_per_inhabitant = euro_per_inhabitant[0] #πάρε τη πρώτη τιμή
        convert=float(euro_per_inhabitant)
        euros.append(convert)
       #print(f"Area: {area}, Euro_per_inhabitant: {euro_per_inhabitant}")

#min max euro
MAX_EURO = df1['Euro_per_inhabitant_EU27'].max()
MIN_EURO = df1['Euro_per_inhabitant_EU27'].min()

#normalized factor
normalized_factor4= list()
if MAX_EURO != MIN_EURO :
    if euros:
        normalized_factor4 = [(euro-MIN_EURO)/(MAX_EURO-MIN_EURO) for euro in euros]
    #print(f"normalized_factor2: {normalized_factor4}")
        
#ARI(gr)
ari_list = list()
aris = name_ari_df['ARI(gr)'].unique()
ari_list = [float(ari) for ari in aris]

MAX_ARI = max(ari_list)
MIN_ARI = min(ari_list)
if MAX_ARI != MIN_ARI :
    if ari_list:
        normalized_factor5 = [(ari-MIN_ARI)/(MAX_ARI-MIN_ARI) for ari in ari_list]
    #print(f"normalized_factor5: {normalized_factor5}")

#Δημιουργεία ενός ενωμένου excel
factor1_df = pd.DataFrame(normalized_factor1,columns=['over 65'])
factor2_df = pd.DataFrame(normalized_factor2,columns=['avg_age'])
factor3_df = pd.DataFrame(normalized_factor3,columns=['gdp'])
factor4_df = pd.DataFrame(normalized_factor4,columns=['Euro'])
factor5_df = pd.DataFrame(normalized_factor5,columns=['Ari'])    
factors_df = pd.merge(factor1_df, factor2_df, on=None, left_index=True, right_index=True)
factors_df = pd.merge(factors_df, factor3_df, on=None, left_index=True, right_index=True)
factors_df = pd.merge(factors_df, factor4_df, on=None, left_index=True, right_index=True)
factors_df = pd.merge(factors_df, factor5_df, on=None, left_index=True, right_index=True)
factors_df.to_excel('excel files/normalized_factors .xlsx', index=False)


#Υπολογισμός weights για κάθε normalized function

#weights = [0.2, 0.2, 0.3, 0.3]
#weights = [0.2, 0.2, 0.2, 0.2,0.2]
#weights = [0.2, 0.1, 0.2, 0.2,0.3]
weights = [0.025, 0.025, 0.025, 0.025,0.9]
#weights = [0.025, 0.025, 0.9, 0.025,0.025]

a1 =[number*weights[0] for number in normalized_factor1]
a2 =[number*weights[1] for number in normalized_factor2]
a3 =[number*weights[2] for number in normalized_factor3]
a4 =[number*weights[3] for number in normalized_factor4]
a5 =[number*weights[4] for number in normalized_factor5]

#result = [number1+number2+number3+number4 for number1,number2,number3,number4 in zip(a1,a2,a3,a4)]
result = [number1+number2+number3+number4+number5 for number1,number2,number3,number4,number5 in zip(a1,a2,a3,a4,a5)]


#Δημιουργεία ενός result excel
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

    #Δημιουργεία ενός file για το LP-model 
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

solved_model = solve_p_median(5)


#Δημιουργεία graph
G = nx.Graph()

#Προσθήκη nodes και edges
edges_list = list()
for v in solved_model.variables():
        subV = v.name.split('_')
        if subV[0] == "Y" and v.varValue is not None and v.varValue == 1:
            node_info = ' '.join(subV[1:])
            edge_tuple = tuple(node_info.split(','))
            edges_list.append(edge_tuple)


cleaned_edges_list = [(edge[0].replace("(", "").replace("'", "").replace(",",""), edge[1].replace("'", "").replace(")","")) for edge in edges_list]
G.add_edges_from(cleaned_edges_list)

#Χρώματα στα non_median και στα centers 
subV1_nodes = [edge[0] for edge in cleaned_edges_list]
subV2_nodes = [edge[1] for edge in cleaned_edges_list]

node_colors = list()
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
plt.savefig('plot.png')


#Δημιουργεία map
map = folium.Map(
    location=[38.2745,23.8103],
    tiles='openstreetmap',
    zoom_start=7,
)

#Δημιουργεια edges excel    
cleaned_df = pd.DataFrame(cleaned_edges_list, columns = ['non_median','centers'])
cleaned_df.to_excel('excel files/edges.xlsx', index = False)


#Δημιουργεία ενός dictionary για κάθε center στα coords του
center_coords_dict = dict(zip(df2['all_centers'], 
                              zip(df2['all_centers_lat'], 
                                  df2['all_centers_long'])))
#Δημιουργεία ενός dictionary για κάθε people στα coords του
people_info_dict = dict(zip(people_info['house'],
                             zip(people_info['latitude'],
                                  people_info['longtitude'])))

for index, edge in enumerate(cleaned_edges_list):
    non_median = edge[0]
    center = edge[1].strip()

    non_median_coords = people_info_dict.get(non_median)
    cecoords = center_coords_dict.get(center)
     
    #Non-median
    folium.Marker(
        location=[non_median_coords[0], non_median_coords[1]],
        popup=f"Non Median: {non_median}",
        icon=folium.Icon(prefix="fa",icon ="home")
    ).add_to(map)

    if center in [center for edge in cleaned_edges_list]:
                
                distance_km, decoded_polyline = get_distance(api_key,non_median_coords, cecoords)
                #Centers
                folium.Marker(  
                    location=[cecoords[0], cecoords[1]],
                    popup=f"Center: {center}",
                    icon=folium.Icon(color='green',icon='medkit',prefix="fa")
                ).add_to(map) 
                #Distance 
                folium.PolyLine(
                    locations=decoded_polyline,
                    color='blue',
                    weight=2,
                    popup=f"{non_median} connects to {center} th {distance_km} km" if distance_km else  None
                ).add_to(map)


# Display the map
map.save('map.html')
webbrowser.open('map.html')
