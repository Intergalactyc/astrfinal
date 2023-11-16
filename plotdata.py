import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column

path = "./dataproducts/data.csv"

data = pd.read_csv(path)

data = data[data["BAD"] == False]
gdata = data[data["FILT"]=="g'"]
idata = data[data["FILT"]=="i'"]
rdata = data[data["FILT"]=="r'"]
days = [["Oct 05 2023",2],["Oct 08 2023",3],["Oct 13 2023",3]]
poor = data["NREF"]<=3

def weightedaverage(dat,errs):
    data = np.array(dat)
    errors = np.array(errs)
    N = len(data)
    if len(data) != len(errors):
        print("Error: data and errors must be of same size.")
        return 0,0
    reserr = 1/np.sum(1/(errors**2))
    wavg = np.sum(data/(errors**2))*reserr
    wstd = np.sqrt(N*reserr*np.sum((data-wavg)**2/(errors**2))/(N-1))
    return wavg, wstd

fig, axs = plt.subplots(3,1,constrained_layout=True,figsize=(6,8.75),sharey=True)

plt.xlabel("Time (JD)",fontsize=12)
plt.ylabel("Magnitude",fontsize=12)

axs[0].set_ylim([12.4,14.6])
axs[0].invert_yaxis()
axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))

runavgs = Table(names=("DATE","FILT","RUN","JD","T_MAG","T_MAG_STD"),dtype=("U16","U8","U8","f8","f8","f8"))

for i, daypair in enumerate(days):
    day,runs = daypair
    dayloc = data["DATE"]==day
    gday,rday,iday = gdata.loc[dayloc],rdata.loc[dayloc],idata.loc[dayloc]
    filts = [[gday,"g'"],[rday,"r'"],[iday,"i'"]]

    for filt in filts:
        flist,fname = filt
        rnums = flist["RUN"].str.slice(stop=4)
        for rnum in np.unique(rnums):
            mags,errs,times = [],[],[]
            for index,row in flist.iterrows():
                if row["RUN"][0:4] == rnum:
                    mags.append(row["T_MAG"])
                    errs.append(row["T_MAG_ERR"])
                    times.append(row["JD"])
            wavg,wstd = weightedaverage(mags,errs)
            mtime = (min(times)+max(times))/2
            runavgs.add_row((day,fname,rnum,mtime,wavg,wstd))
    
    axs[i].set_title(day,x=0.11)
    axs[i].errorbar(gday["JD"],gday["T_MAG"]-0.6,gday["T_MAG_ERR"],None,"b-o",label="g'-0.6",linewidth=0.75)
    axs[i].errorbar(rday["JD"],rday["T_MAG"],rday["T_MAG_ERR"],None,"g-o",label="r'",linewidth=0.75)
    axs[i].errorbar(iday["JD"],iday["T_MAG"]+0.3,iday["T_MAG_ERR"],None,"r-o",label="i'+0.3",linewidth=0.75)
    axs[i].scatter(gday.loc[poor]["JD"],gday.loc[poor]["T_MAG"]-0.6,facecolors="none",edgecolors="black",marker="o",zorder=5,s=40)
    axs[i].scatter(rday.loc[poor]["JD"],rday.loc[poor]["T_MAG"],facecolors="none",edgecolors="black",marker="o",zorder=5,s=40)
    axs[i].scatter(iday.loc[poor]["JD"],iday.loc[poor]["T_MAG"]+0.3,facecolors="none",edgecolors="black",marker="o",zorder=5,s=40)

axs[0].legend(loc="upper center",ncol=3,bbox_to_anchor=(0.735,1.165))

plt.savefig("./dataproducts/lightcurves.png",bbox_inches="tight")
runavgs.write("./dataproducts/avgmagnitudes.csv",format="csv",overwrite=True)