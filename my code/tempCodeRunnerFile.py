import pandas as pd
import os
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


name_ari_df = names.merge(people_ari, on='CODENAME', how='inner')

#normalized factor
#normalized_factor1 = []
ari_list = []
aris = name_ari_df['ARI(gr)'].unique()
ari_list = [float(ari) for ari in aris]

MAX_ARI = max(ari_list)
MIN_ARI = min(ari_list)
if MAX_ARI != MIN_ARI :
    if ari_list:
        normalized_factor5 = [(ari-MIN_ARI)/(MAX_ARI-MIN_ARI) for ari in ari_list]
    #print(f"normalized_factor5: {normalized_factor5}")

