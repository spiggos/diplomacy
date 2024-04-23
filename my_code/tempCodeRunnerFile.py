
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
import seaborn as sns
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
    file_path3 = os.path.join("excel files", "geocode synopsis EG 21_12.xlsx")
    synopsis = pd.read_excel(file_path3, sheet_name='data_full')
    visit = pd.read_excel(file_path3, sheet_name='visit')
    visited_df = names.merge(visit, on='CODENAME', how='inner').drop_duplicates()
    no_visit = pd.read_excel(file_path3, sheet_name='no_visit')
    no_visited_df = names.merge(no_visit, on='CODENAME', how='inner')
    data = names.merge(synopsis, on='CODENAME' , how ='inner')
except FileNotFoundError:
    print("Excel file not found.")
except pd.errors.ParserError:
    print("Invalid excel file format.")

visits = list()
mean_visit = df1.groupby('house')['visit'].mean()
visits = [int(visit) for visit in mean_visit]
print(visits)


arigr = visited_df.loc[:,['ARI(gr)']]
ari =arigr['ARI(gr)'].unique()

visitgr = data.loc[:,['visit']]

numeric_columns = data.select_dtypes(include=[np.number])
correlation_with_visit = numeric_columns.corrwith(data['ARI(gr)']).sort_values(ascending=False)
correlation_with_visit.to_excel('correlation.xlsx')