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
    file_path3 = os.path.join("excel files", "geocode synopsis EG 21_12.xlsx")
    synopsis = pd.read_excel(file_path3, sheet_name='data_full')
    people_ari = synopsis.loc[:,['CODENAME','ARI(gr)']].drop_duplicates().dropna()
    name_house = df1.loc[:,['CODENAME','house']].drop_duplicates()
    name_visit = synopsis.loc[:,['CODENAME','visit']].drop_duplicates()
    name_neuro = synopsis.loc[:,['CODENAME','neuropsychiatric disorders']].drop_duplicates()
    name_dd = synopsis.loc[:,['CODENAME','dementia or depression prior']].drop_duplicates()
    name_od = synopsis.loc[:,['CODENAME','other diseases (pneumon, endocr, hematol, urol, pain, gi']].drop_duplicates()
    people_visit = pd.merge(name_house, name_visit, on='CODENAME', how='inner')
    people_neuro = pd.merge(name_house, name_neuro, on='CODENAME', how='inner')
    people_dd = pd.merge(name_house, name_dd, on='CODENAME', how='inner')
    people_od = pd.merge(name_house, name_od, on='CODENAME', how='inner')
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

people_info = pd.DataFrame({'house':info['house'],'longtitude':long_list,'latitude':lat_list})
people_geo= pd.DataFrame({'longtitude':long_list,'latitude':lat_list})
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

#Δημιουργεία list για non-median
areas = df1.loc[:,['house']].drop_duplicates()

non_median = [area for area in areas['house']]
non_median_df = pd.DataFrame(non_median,columns=['non_median'])

#Δημιουργεία list για τα centers
centers = df2.loc[:,['all_centers']].drop_duplicates()
candidate_location = [center for center in centers['all_centers']]
centers_df = pd.DataFrame(candidate_location,columns=['centers'])

name_ari_df = names.merge(people_ari, on='CODENAME', how='inner')


#----FACTORS OF DEMAND----
#1)visit
#2)neuropsychiatic disorders
#3)euro_per_inhabitant
#4)dementia or depression prior
#5)urol
#6)age


# Visits
visit_list = list()
mean_visit = people_visit.groupby('house')['visit'].mean()
visit_list = [visit for visit in mean_visit]
#min max visit 
MAX_VISIT = max(visit_list)
MIN_VISIT = min(visit_list)
#normalized factor
normalized_factor1 = list()
if MAX_VISIT != MIN_VISIT :
    if visit_list:
        normalized_factor1 = [(visit-MIN_VISIT)/(MAX_VISIT-MIN_VISIT) for visit in visit_list]
    #print(f"normalized_factor1: {normalized_factor1}")

# neuropsychiatic disorders
neuro_list = list()
mean_neuro = people_neuro.groupby('house')['neuropsychiatric disorders'].mean()
neuro_list = [neuro for neuro in mean_neuro]
#min max neuro 
MAX_NEURO = max(neuro_list)
MIN_NEURO = min(neuro_list)
#normalized factor
normalized_factor2 = list()
if MAX_NEURO != MIN_NEURO :
    if neuro_list:
        normalized_factor2 = [(neuro-MIN_NEURO)/(MAX_NEURO-MIN_NEURO) for neuro in neuro_list]
    #print(f"normalized_factor2: {normalized_factor2}")

#euro_per_inhabitant
eu27_list = list()    
areas = df1['house'].unique()
for area in areas :
    filtered_df = df1[df1['house'] == area]
    euro_per_inhabitant = filtered_df['Euro_per_inhabitant_EU27'].unique()
    if len(euro_per_inhabitant) > 0 : 
        euro_per_inhabitant = euro_per_inhabitant[0] #πάρε τη πρώτη τιμή
        convert=float(euro_per_inhabitant)
        eu27_list.append(convert)
#min max eu27
MAX_EU27 = df1['Euro_per_inhabitant_EU27'].max()
MIN_EU27 = df1['Euro_per_inhabitant_EU27'].min()
#normalized factor
normalized_factor3 = list()
if MAX_EU27 != MIN_EU27 :
    if eu27_list:
        normalized_factor3 = [(eu27-MIN_EU27)/(MAX_EU27-MIN_EU27) for eu27 in eu27_list]
    #print(f"normalized_factor3: {normalized_factor3}")


#dementia or depression prior
dd_list = list()
mean_dd = people_dd.groupby('house')['dementia or depression prior'].mean()
dd_list = [dd for dd in mean_dd]

MAX_DD = max(dd_list)
MIN_DD = min(dd_list)
#normalized factor
normalized_factor4 = list()
if MAX_DD != MIN_DD :
    if dd_list:
        normalized_factor4 = [(dd-MIN_DD)/(MAX_DD-MIN_DD) for dd in dd_list]
    #print(f"normalized_factor4: {normalized_factor4}")

#urol
#other diseases (pneumon, endocr, hematol, urol, pain, gi
od_list = list()
mean_od = people_od.groupby('house')['other diseases (pneumon, endocr, hematol, urol, pain, gi'].mean()
od_list = [od for od in mean_od]

MAX_OD = max(od_list)
MIN_OD = min(od_list)
#normalized factor
normalized_factor5 = list()
if MAX_OD != MIN_OD :
    if od_list:
        normalized_factor5 = [(od-MIN_OD)/(MAX_OD-MIN_OD) for od in od_list]
    #print(f"normalized_factor5: {normalized_factor5}")

#age 
averages = list()
mean_age = df1.groupby('house')['age'].mean()
averages = [average for average in mean_age]

#min max των μ.ο ηλικιών
MAX_AVERAGE = max(averages)
MIN_AVERAGE = min(averages)

#normalized factor
normalized_factor6 = list()
if MAX_AVERAGE != MIN_AVERAGE :
    if averages:
        normalized_factor6 = [(avg-MIN_AVERAGE)/(MAX_AVERAGE-MIN_AVERAGE) for avg in averages]
    #print(f"normalized_factor6: {normalized_factor6}")

#----FACTORS OF DEMAND----
#1)visit
#2)neuropsychiatic disorders
#3)euro_per_inhabitant
#4)dementia or depression prior
#5)urol
#6)age

#weights = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
#weights = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
weights = [0.025, 0.025, 0.025, 0.025, 0.4, 0.5]
a1 =[number*weights[0] for number in normalized_factor1]
a2 =[number*weights[1] for number in normalized_factor2]
a3 =[number*weights[2] for number in normalized_factor3]
a4 =[number*weights[3] for number in normalized_factor4]
a5 =[number*weights[4] for number in normalized_factor5]
a6 =[number*weights[5] for number in normalized_factor6]

result = [number1+number2+number3+number4+number5+number6 for number1,number2,number3,number4,number5,number6 in zip(a1,a2,a3,a4,a5,a6)]

result_df =pd.DataFrame(result, columns = ['result'])

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
plt.savefig('plot1.png')

#Δημιουργεία map
map = folium.Map(
    location=[38.2745,23.8103],
    tiles='openstreetmap',
    zoom_start=7,
)

#Δημιουργεια edges excel    
cleaned_df = pd.DataFrame(cleaned_edges_list, columns = ['non_median','centers'])
cleaned_df.to_excel('excel files/edges_visit.xlsx', index = False)