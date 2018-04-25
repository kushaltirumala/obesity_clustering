import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
odata = pd.read_csv("percentObesityAllcCountriesRegions.csv")
odatat = odata.transpose()

xcells = [1, 2, 5, 10, 15, 20, 30, 40]
ycells = [-0.4, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005, -0.002, -0.001, \
          0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]

directory = 'test_binary/'

def get_country(odata,country):
    x = odata['location']
    for i in range(len(x)):  # This is Country in some other files
        if x[i] == country: return i
        
#print get_country(odata,'Australia')

def hist2dc(dframe,country,xcells,ycells,paxes, directory):
    cname = get_country(dframe,country)
#i = 10 # 1st country
    years = list(map(int,odata.dtypes.index[2:]))
#years = list(map(int, years))
    vals = odatat[cname][2:].convert_objects(convert_numeric=True)
    binvals = []
    binyears = []
    for i in range(len(years)):
        for j in range(i+1,len(years)):
            binyears.append(years[j]-years[i])
            binvals.append(vals[j]-vals[i])
    dvals = np.digitize(binvals,ycells)
    dyears = np.digitize(binyears,xcells)
    plt.clf()
    fig, ax = plt.subplots()
    h = ax.hist2d(dyears,dvals,bins=[range(len(xcells)),range(len(ycells))])
    if paxes == 0:
        plt.hist2d(dyears,dvals,bins=[range(len(xcells)),range(len(ycells))], cmap=plt.get_cmap("binary"))
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            labelleft='off',
            labelbottom='off') # labels along the bottom edge are off
    else:
        plt.colorbar(h[3], ax=ax)
        plt.xlabel('dYears')
        plt.ylabel('dChange')
        plt.title(country)
        plt.xticks(range(len(xcells)),xcells)
        plt.yticks(range(len(ycells)),ycells)

    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + country + '2.jpg')


# hist2dc(odata,'Hungary',xcells,ycells,1) 

for country in odata['location']:
    print("Creating image for: " + country)
    hist2dc(odata,country,xcells,ycells,0, directory)





