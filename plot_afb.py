
"""
Notes: This script will automatically generate XX plots
- First: histogram plots of question response for fine/coarse locations (each question has a plot)
- Second: percentage of fine locations (each round has one plot)
- Third: absolute difference for response at fine/coarse level (each question has a plot)
- Forth: distribution of geocoded dataset at fine level (each question has a plot and a raster)



"""



import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from adjustText import adjust_text, repel_text
from mpl_toolkits.basemap import Basemap,cm
import plotly.graph_objs as go
import plotly.plotly as py
from affine import Affine
from distancerasters import rasterize
import geopandas as gpd
from shapely.geometry import Point
import numpy as np


# -*- coding: utf-8 -*-

file = r"/Users/miranda/Documents/AidData/projects/datasets/AFB/rasterization/Merged Stacked Datasets_v3.dta"
basemap = r"/Users/miranda/Documents/AidData/projects/datasets/AFB/africa.geojson"

# ------------------------------------
# create output directory

dirc = os.path.dirname(file)

for rd in range(1, 7):
    folder = "round_" + str(rd)
    fpath = os.path.join(dirc, folder)
    if not os.path.isdir(fpath):
        os.mkdir(fpath)

dta = pd.read_stata(file)

# ------------------------------------------
# Set filters: each questions per round
questions = ["trust_pres", "trust_police", "trust_court", "trust_electcom",
            "trust_party", "trust_oppart"]

questionsname = {"trust_pres":"Trust the President",
                 "trust_police": "Trust the Police",
                 "trust_court": "Trust Courts of Law",
                 "trust_electcom": "Trust the Independent Electoral Commission",
                 "trust_party": "Trust the Ruling Party",
                 "trust_oppart": "Trust Opposition Political Parties"}

# ---------------------------------
# calculate percentage rate of fine locations within a given round

def percentage_fine(gdf, question):
    gdf['country_name'] = gdf['geoname_adm_name'].apply(lambda x: x.split('|')[2])
    countrynames = list(gdf['country_name'].unique())
    categorylist = ["fine", "coarse"]

    wiredcase = {"Republic of Mauritius": "Mauritius", "Western Sahara": "Morocco"}

    gdf['country_name'] = gdf['country_name'].apply(
        lambda x: wiredcase[x] if x.decode('utf-8') in wiredcase.keys() else x)

    # gdf['country_name'] = gdf[gdf['country_name']=='Republic of Mauritius']
    gdf_group = gdf.groupby(['country_name', 'category'])

    dta_dict = dict()
    dta_dict["country_name"] = list()
    dta_dict["fine_count"] = list()
    dta_dict["coarse_count"] = list()
    dta_dict["fine_mean"] = list()
    dta_dict["coarse_mean"] = list()

    for countryname in countrynames:

        dta_dict["country_name"].append(countryname)

        for category in categorylist:

            count_name = category + "_count"
            mean_name = category + "_mean"

            if (countryname, category) in gdf_group.groups.keys():

                dta_dict[count_name].append(len(gdf_group.get_group((countryname, category))))
                dta_dict[mean_name].append(gdf_group.get_group((countryname, category))[question].mean())
            else:
                dta_dict[count_name].append(0)
                dta_dict[mean_name].append(0)

    newdf = pd.DataFrame.from_dict(dta_dict)

    newdf['fine_count_percent'] = newdf['fine_count'] / (newdf['fine_count'] + newdf['coarse_count'])
    newdf['coarse_count_percent'] = newdf['coarse_count'] / (newdf['fine_count'] + newdf['coarse_count'])
    newdf['abs_mean_diff'] = newdf['fine_mean'] - newdf['coarse_mean']
    #newdf['abs_mean_diff'] = abs(newdf['fine_mean'] - newdf['coarse_mean'])

    return newdf


def plot_map(dta, mapvar, maptitle, legendtitle, outfile):

    """
    :param dta: dataframe that summarizes mean/count of question response at fine level
    :param mapvar: the attribute to map, eg: fine_count_percent (percentage of fine locations)
    :param maptitle: image title
    :param legendtitle: legend title
    :param outfile: output directory
    :return: an image saved
    """
    try:
        py.sign_in('mirandalv', 'TMvHv1KBIfeambeSCXwt')
    except:
        time.sleep(60)
        py.sign_in('mirandalv', 'TMvHv1KBIfeambeSCXwt')

    """
    if maps=='abs':
        colorscale = [
            [0, "rgb(209, 190, 90)"],
            [0.20, "rgb(177, 173, 42)"],
            [0.35, "rgb(95, 144, 11)"],
            [0.50,"rgb(57, 129, 27)"],
            [0.75, "rgb(40, 122, 33)"],
            [1, "rgb(13, 86, 44)"]
        ]
    elif maps=='percent':
        colorscale = [
            [0, "rgb(209, 190, 90)"],
            [0.20, "rgb(177, 173, 42)"],
            [0.35, "rgb(95, 144, 11)"],
            [0.50,"rgb(57, 129, 27)"],
            [0.75, "rgb(40, 122, 33)"],
            [1, "rgb(13, 86, 44)"]
        ]
    """
    max = dta[mapvar].max()
    min = dta[mapvar].min()

    colorscale=[
        [
          0,
          "rgb(209, 190, 90)"
        ],
        [
          0.35,
          "rgb(177, 173, 42)"
        ],
        [
          0.5,
          "rgb(95, 144, 11)"
        ],
        [
          0.6,
          "rgb(57, 129, 27)"
        ],
        [
          0.7,
          "rgb(40, 122, 33)"
        ],
        [
          1,
          "rgb(13, 86, 44)"
        ]
      ]

    data = [go.Choropleth(
        type='choropleth',
        autocolorscale=False,
        colorscale=colorscale,
        zauto=False,
        zmax=max,
        zmin=min,
        reversescale=False,
        locations=dta['country_name'],
        z=dta[mapvar].astype(float),
        locationmode='country names',
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
        colorbar=dict(
            title=legendtitle,
            ticklen=3,
            len=0.5,
            thickness=10)
    )]

    layout = dict(
        title=maptitle,
        geo=dict(
            scope='africa',
            projection=dict(type='Mercator'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'),
        showlegend=True
    )

    fig = dict(data=data, layout=layout)
    # py.plot(fig, filename='d3-cloropleth-map.html')
    py.image.save_as(fig, filename=outfile, scale=10)


def scatter_label(df, outf, rd):
    x = df['coarse_count_percent']
    y = df['fine_count_percent']

    labels = df['country_name']

    plt.scatter(x, y, s=2)

    texts = list()

    for label, x, y in zip(labels, x, y):
        # plt.annotate(label.decode('utf-8'), xy=(x, y))
        label = label.decode('utf-8')
        texts.append(plt.text(x, y, label))

    # force_text: r1 = 1.0; r2=1.0; r3=1.0; r4=; r5=1.6; r6=

    adjust_text(texts, force_text=0.8, autoalign='y', precision=1,
                arrowprops=dict(arrowstyle="-|>", color='r', alpha=0.3, lw=0.5))

    plt.xlabel('Coarse location percentage')
    plt.ylabel('Fine location percentage')

    title = "Location accuracy of each country in round: " + str(rd)
    plt.title(title)
    plt.savefig(outf, dpi=900)


def percentageplot(df, question):
    count = dict()
    dta_gp = df.groupby(question)

    for name, group in dta_gp:
        count[name] = group['trust_pres'].count()

    return count


# --------------------------------------------
# define precision categories

is_precise = ((dta["location_class"] == 2) | (dta["location_class"] == 3) | (
(dta["location_class"] == 1) & (dta["location_type_code"].isin(["ADM3", "ADM4", "ADM4H", "ADM5"]))))
dta['category'] = None

dta.loc[is_precise, 'category'] = 'fine'
dta.loc[~is_precise, 'category'] = 'coarse'

gdf = dta.copy(deep=True)

# ------ rasterization setting ---------
#minx = gdf['longitude'].min()
#maxx = gdf['longitude'].max()
#miny = gdf['latitude'].min()
#maxy = gdf['latitude'].max()

minx=-20
miny=-40
maxx=60
maxy=40

pixel_size = 0.5

out_shape = (int((maxy-miny) / pixel_size), int((maxx-minx) / pixel_size))

affine = Affine(pixel_size, 0, -20,
                0, -pixel_size, 40)

nan_val = 255

# -------------------------------------

for rd in range(1, 7, 1):

    print "Start working on Round: ", str(rd)

    # set path
    folder = "round_" + str(rd)
    fpath = os.path.join(dirc, folder)

    dta_gdf = gdf.loc[gdf['round'] == rd]

    for question in questions:

        if not dta_gdf[question].isnull().all():
            print "Working on Round " + str(rd) + " Question " + question
            print "---------------------------------------------------"

            # --------------------------------- Start plot -----------------------------------
            dta_asub = dta_gdf.loc[dta_gdf['category'] == 'fine']
            dta_bsub = dta_gdf.loc[dta_gdf['category'] == 'coarse']
            summary_df = percentage_fine(dta_gdf, question)
            summary_df.to_csv('delete.csv')


            # ------------------------------------------------------------------------------
            # map 1: category histogram
            print "Working on histogram..........."
            fig, ax = plt.subplots()
            percent_fine = percentageplot(dta_asub, question)
            percent_coarse = percentageplot(dta_bsub, question)

            count = dta_gdf.shape[0]
            catval1 = [val/float(count) for val in percent_fine.values()]
            catval2 = [val/float(count) for val in percent_coarse.values()]

            cat1 = ax.bar(np.asarray(percent_fine.keys()), catval1,0.25,color='g',label='fine',alpha=0.8)
            cat2 = ax.bar(np.asarray(percent_coarse.keys())+0.25, catval2, 0.25, color='y', label='coarse',
                           alpha=0.8)

            name = "Round " + str(rd) + " " + questionsname[question]
            ax.set_ylabel('Percent')
            ax.set_xlabel('Response')
            ax.set_title(name)
            ax.set_xticks(np.asarray(percent_fine.keys())+0.125)
            ax.set_xticklabels([int(val) for val in percent_fine.keys()])

            ax.legend((cat1[0], cat2[0]), ('Fine', 'Coarse'))
            plt.savefig(os.path.join(fpath, name))
            plt.clf()
            #plt.close()


            # -------------------------------------------------------------------------

            # map 2: percentage of fine locations (this is identical for different question
            # Only one plot each round)

            print "Working on percentage of fine locations..........."

            maptitle1 = "Percentage of coding at fine level - " + "Round " + str(rd)
            legendtitle1 = "values scale [0,1]"
            outname1 = "round_" + str(rd) + "fine_percent.png"
            outfile1 = os.path.join(fpath, outname1)
            mapvar1 = 'fine_count_percent'

            if not os.path.exists(outfile1):
                plot_map(summary_df, mapvar1, maptitle1, legendtitle1, outfile1)


            # ------------------------------------------------------------------------------------
            # map 3: Absolute difference of mean(A) & mean(B)
            print "Working on mean difference..........."
            plt.clf()

            maptitle3 = "Round - " + str(rd) + " - Absolute difference between fine/coarse coding - " + question
            legendtitle3 = "values scale [-3,3]" #+ dta_gdf[question].max()
            outname3 = "round_" + str(rd) + "absolute difference " + question + ".png"
            outfile3 = os.path.join(fpath, outname3)
            mapvar3 = 'abs_mean_diff'
            plot_map(summary_df, mapvar3, maptitle3, legendtitle3, outfile3)

            # ------------------------------------------------------------------------------------
            # map 4: Distribution of geocoded locations at fine level
            # visualize the fine level survey counts (Results are raster)
            print "Working on rasterization ..........."
            round_accuracy = int(1/pixel_size)

            dta_asub['latitude'] = dta_asub['latitude'].apply(lambda x: round(x * round_accuracy) / round_accuracy)
            dta_asub['longitude'] = dta_asub['longitude'].apply(lambda x: round(x * round_accuracy) / round_accuracy)
            dta_gp = dta_asub.groupby(('longitude', 'latitude'))

            dta_dict = dict()
            dta_dict['latitude'] = list()
            dta_dict['longitude'] = list()
            dta_dict['count'] = list()

            for name, group in dta_gp:
                count = group[question].count()
                dta_dict['latitude'].append(name[1])
                dta_dict['longitude'].append(name[0])
                dta_dict['count'].append(count)

            newdf = pd.DataFrame.from_dict(dta_dict)
            newdf["geometry"] = newdf.apply(lambda x: Point(x["longitude"], x["latitude"]), axis=1)
            newgdf = gpd.GeoDataFrame(newdf)
            newgdf.to_csv("delete.csv")

            cat_raster, _ = rasterize(newgdf, affine=affine, shape=out_shape, attribute='count', nodata=0,
                                      fill=0) #, output=output


            cat_raster = np.flipud(cat_raster)

            x_center = (minx + maxx) / 2
            y_center = (miny + maxy) / 2

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            m = Basemap(llcrnrlon=-20, llcrnrlat=-40, urcrnrlon=60, urcrnrlat=40,
                        resolution='h', ellps='WGS84')  # projection='merc',lat_0 = y_center, lon_0 = x_center,

            # draw map boundary, meridians, longitudes
            m.drawmapboundary(fill_color='white', zorder=-1)
            m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 1], dashes=[1, 1], linewidth=0.25, color='0.5',
                            fontsize=14)
            m.drawmeridians(np.arange(0., 360., 60.), labels=[1, 0, 0, 1], dashes=[1, 1], linewidth=0.25, color='0.5',
                            fontsize=14)

            # draw coastlines, state and country boundaries, edge of map
            m.drawcountries()
            m.drawcoastlines(linewidth=0.5)  # color='0.6', linewidth=1

            ny = int((maxy - miny) / pixel_size)
            nx = int((maxx - minx) / pixel_size)
            lons, lats = m.makegrid(nx, ny)  # get lat/lons of ny by nx evenly space grid.
            x, y = m(lons, lats)  # compute map proj coordinates.

            # draw filled contours.
            #cmap = LinearSegmentedColormap.from_list("my_colormap", colors, N=len(levels), gamma=1.0)

            clevs = [1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250,300,350,400]
            # clevs = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,240,260,300,350,400]
            cs = m.contourf(x, y, cat_raster, clevs, cmap=cm.s3pcpn)#cm.s3pcpn #cm.s3pcpn

            # add colorbar.
            cbar = m.colorbar(cs, location='bottom', pad="5%")
            cbar.set_label('Survey count at fine level')

            # add title
            title = "Round_" + str(rd) + questionsname[question]
            plt.title(title)
            output = os.path.join(fpath, title)

            plt.savefig(output, dpi=900)
            plt.clf()
            plt.close()











