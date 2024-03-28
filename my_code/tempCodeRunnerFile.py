
                   #-----------P-median----------------import itertools
import math
import io
import os
import webbrowser
import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import networkx as nx
import folium
import requests
import json
import googlemaps
import polyline
from polyline import decode as poly_decode


try:
    file_path1 = os.path.join("excel files", "hpeiros_new.xlsx")
    df1 = pd.read_excel(file_path1)
    names = df1.loc[:,['CODENAME']].drop_duplicates()
    info = df1.loc[:,['house']].drop_duplicates()
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
print(len(info))
print(len(people_coords))

long_list=[]
lat_list=[]
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

center_coords_dict = dict(zip(df2['all_centers'], 
                              zip(df2['all_centers_lat'], 
                                  df2['all_centers_long'])))
people_info_dict = dict(zip(people_info['house'],
                             zip(people_info['latitude'],
                                  people_info['longtitude'])))

#Δημιουργεία χάρτη
map = folium.Map(
    location=[38.2745,23.8103],
    tiles='openstreetmap',
    zoom_start=7,
)    

file_path4 = os.path.join("excel files", "edges.xlsx")
edges = pd.read_excel(file_path4)
print(edges)

my_list =list()
for index, row in edges.iterrows():
    # Extract values from the columns and create a tuple
    value_tuple = (row['non_median'], row['centers'])
    # Append the tuple to my_list
    my_list.append(value_tuple)
        

for index, edge in enumerate(my_list):
    non_median = edge[0]
    center = edge[1].strip()

    non_median_coords = people_info_dict.get(non_median)
    cecoords = center_coords_dict.get(center)

# Add marker for non_median location
    folium.Marker(
        location=[non_median_coords[0], non_median_coords[1]],
        popup=f"Non Median: {non_median}",
        icon=folium.Icon(prefix="fa",icon ="home")
    ).add_to(map)

    folium.Marker(  
                    location=[cecoords[0], cecoords[1]],
                    popup=f"Center: {center}",
                    icon=folium.Icon(color='green',icon='medkit',prefix="fa")
                ).add_to(map) 
    #if center in [center for edge in my_list]:
    #            distance_km, decoded_polyline = get_distance(api_key, (non_median_coords['latitude'], non_median_coords['longtitude']), cecoords)
                
                #distance_row = cd_df[
                #    (cd_df['people_coords'] == (non_median_coords['latitude'], non_median_coords['longtitude'])) & 
                #    (cd_df['centers_coords'] == cecoords)]
                #distance = distance_row['km'].values[0] if not distance_row.empty else None
                
                 
                
                #ένωση non_median με centers
                #folium.PolyLine(
                #    locations=decoded_polyline,
                #    color='blue',
                #    weight=2,
                #    popup=f"{non_median} connects to {center} with {distance_km} km" if distance_km else  None
                #).add_to(map)

# Display the map
map.save('map.html')
webbrowser.open('map.html')
