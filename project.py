import os
import glob
import warnings
import time

from pathlib import PurePath
from tqdm import tqdm
from math import ceil

import astroalign as aa
import numpy as np

from astropy.table import Table, Column
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats

from astroquery.astrometry_net import AstrometryNet
from astroquery.sdss import SDSS

from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, ApertureStats, CircularAnnulus, CircularAperture, SkyCircularAperture

import matplotlib.pyplot as plt

warnings.simplefilter('ignore', category=np.exceptions.VisibleDeprecationWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)

target_sky = SkyCoord("22h02m43.26s +42d16m39.65s")

FWHM = 8.0
WIDTH = 3216
HEIGHT = 2208
SRSIZE = 30
GAIN = 2.45
ANALOGDIGITALERROR = np.sqrt(1/12)
READNOISE = 2.5
MAXATTEMPTS = 10
WAITTIME = 5

ast = AstrometryNet()
ast.api_key = "vxbrustekypatepi"

directory = "./data"
subfolders = []
subpipes = ["masters","images","plots"]
subout = ["outdata","plots"]

scan = os.scandir(directory)
for item in scan: 
    if item.is_dir() and item.name not in subout: subfolders.append(item.name)

for so in subout:
    path = os.path.join(directory,so)
    os.makedirs(path,exist_ok=True)

outtable = Table(names = ("JD","DATE","FILT","RUN","ZP","ZP_ERR","MAG","MAG_ERR","NREF","SNR"),dtype=("f8","U16","U8","U8","f8","f8","f8","f8","i4","f4"))
disappointments = []
for folder in subfolders:
    # Begin by sorting data
    base_path = os.path.join(directory,folder)
    pipeout = os.path.join(base_path,"pipelineout")
    for sp in subpipes:
        path = os.path.join(pipeout,sp)
        os.makedirs(path, exist_ok = True)
    files = glob.glob(base_path+"/**/*.fit", recursive=True)
    files.extend(glob.glob(base_path+"/**/*.fits",recursive=True))
    biases = []
    darks = []
    flats = {}
    science = {}
    for file in tqdm(files, desc="Sorting data from " + base_path):
        filepath = PurePath(file)
        if "bias" in filepath.parent.name.lower():
            biases.append(file)
        elif "dark" in filepath.parent.name.lower():
            darks.append(file)
        elif "flat" in filepath.parent.parent.name.lower():
            filt = filepath.parent.name
            if filt not in flats.keys(): flats[filt]=[]
            flats[filt].append(file)
        elif "science" in filepath.parent.parent.parent.name.lower():
            filt = filepath.parent.parent.name
            run = filepath.parent.name
            if filt not in science.keys(): science[filt]={}
            if run not in science[filt].keys(): science[filt][run]=[]
            science[filt][run].append(file)
        else: print(file+" could not be classified.")
    print("Data sorted.")

    # Form master bias by median
    print("Forming master bias for " + base_path)
    bias_stack = [fits.getdata(bias) for bias in biases]
    master_bias = np.median(bias_stack, axis=0).astype("float32")
    fits.PrimaryHDU(master_bias).writeto(pipeout+"\\masters\\master_bias.fit",overwrite=True)
    print(str(len(biases)) + " biases combined.")

    # Debias darks, and form master dark by median
    print("Forming master dark for " + base_path)
    dark_stack = [fits.getdata(dark) - master_bias for dark in darks]
    master_dark = np.median(dark_stack, axis=0).astype("float32")
    std_dark = np.std(dark_stack-master_bias, axis=0)
    dark_err = np.mean(std_dark)
    fits.PrimaryHDU(master_dark).writeto(pipeout+"\\masters\\master_dark.fit",overwrite=True)
    print(str(len(darks)) + " darks combined.")

    # Debias flats in each filter, and form master flat by median (dark current ignored because of extremely short exposure time)
    for filt in flats.keys():
        print("Forming master flat for filter " + filt + " in " + base_path)
        flat_stack = [fits.getdata(flat) - master_bias for flat in flats[filt]]
        median_flat = np.median(flat_stack, axis=0).astype("float32")
        mflatmean = np.mean(median_flat)
        master_flat = median_flat / mflatmean
        fits.PrimaryHDU(master_flat).writeto(pipeout+"\\masters\\master_flat_"+filt+".fit",overwrite=True)
        print(str(len(flats[filt])) + " " + filt + " flats combined")

    for filt in science.keys():
        master_flat = fits.getdata(pipeout+"\\masters\\master_flat_"+filt+".fit")
        for run in science[filt].keys():
            subruns = ceil(len(science[filt][run])/SRSIZE)
            for subrun in range(subruns):
                try:
                    print(f"Calibrating, aligning, and stacking frames in {run}.{subrun} for filter {filt} in {base_path}")
                    target_fits = fits.open(science[filt][run][subrun*SRSIZE])
                    target_ccd = target_fits[0].data
                    JD_OBS = target_fits[0].header["JD"]
                    DATE_OBS = folder
                    target_cal = (target_ccd - master_bias - master_dark)/master_flat
                    cal_stack = [((fits.getdata(frame) - master_bias - master_dark)/master_flat, frame) for frame in science[filt][run][subrun*SRSIZE+1:(subrun+1)*SRSIZE]]
                    alg_stack = [target_ccd]
                    translations = [[],[]]
                    for i in tqdm(range(len(cal_stack)),desc="Aligning..."):
                        try:
                            ccd = cal_stack[i][0]
                            transform = aa.find_transform(ccd,target_cal)
                            translations[0].append(transform[0].translation[0])
                            translations[1].append(transform[0].translation[1])
                            aligned = aa.apply_transform(transform[0],ccd,target_cal)[0]
                            alg_stack.append(aligned)
                        except Exception:
                            disappointments.append(cal_stack[i][1])
                    rightcrop = ceil(0.5 * (max(translations[0])+abs(max(translations[0]))))+FWHM
                    leftcrop = ceil(0.5 * (min(translations[0])-abs(min(translations[0]))))+FWHM
                    topcrop = ceil(0.5 * (max(translations[1])+abs(max(translations[1]))))+FWHM
                    botcrop = ceil(0.5 * (min(translations[1])-abs(min(translations[1]))))+FWHM
                    final_image = np.sum(alg_stack, axis=0).astype("float32")

                    print("Finding sources...")
                    daofind = DAOStarFinder(fwhm=FWHM, threshold=4.*np.std(final_image))
                    sources = daofind(final_image - np.median(final_image))
                    ogsources = len(sources)
                    for source in enumerate(sources):
                        if not (leftcrop < source[1]["xcentroid"] < WIDTH - rightcrop and botcrop < source[1]["ycentroid"] < HEIGHT - topcrop):
                            sources.remove_row(source[0])
                    netsources = len(sources)
                    sources.sort("flux")
                    sources.reverse()
                    print(str(netsources) + " sources found after eliminating " + str(ogsources-netsources) + " out-of-bounds. Querying astrometry.net for astrometric solution")

                    attempts = 0
                    while(attempts < MAXATTEMPTS):
                        try:
                            wcs_header = ast.solve_from_source_list(sources["xcentroid"], sources["ycentroid"],WIDTH,HEIGHT,solve_timeout=60,center_ra=330.23,center_dec=42.28,radius=0.5)
                        except TimeoutError:
                            print("Timed out before solution found.")
                            attempts = MAXATTEMPTS - 1 # only retry once if timed out
                        except Exception as e:
                            print(f"Failed. Reason: {e}")
                            attempts += 1
                            if attempts >= MAXATTEMPTS:
                                raise Exception("Out of tries. Moving on.")
                            else:
                                print(f"Retrying in {WAITTIME} seconds.")
                                time.sleep(WAITTIME)
                        else:
                            attempts = MAXATTEMPTS
                        if wcs_header:
                            print("Astrometric solution found!")

                    fits.PrimaryHDU(final_image,wcs_header).writeto(pipeout+"\\images\\final_"+filt+"_"+run+".fit",overwrite=True)
                    
                    wcs = WCS(wcs_header)
                    target_pos = wcs.world_to_pixel(target_sky)
                    sourcepositions = [(source["xcentroid"],source["ycentroid"]) for source in sources]
                    sourcelist = sources["xcentroid","ycentroid"]
                    skycoords = np.transpose(wcs.pixel_to_world_values(sourcepositions))
                    sourcelist.add_columns([skycoords[0],skycoords[1],0.,0.],names=["ra","dec","ref_mag","ref_mag_err"])

                    t_inst_mag = 0
                    zero_point = 0
                    failed = True
                    sourceid = None

                    print("Querying for SDSS catalog cross-matches...")
                    attempts = 0
                    while(attempts < MAXATTEMPTS):
                        try:
                            result = SDSS.query_crossid(wcs.pixel_to_world(sources["xcentroid"],sources["ycentroid"]))
                        except Exception as e:
                            print(f"Failed. Reason: {e}")
                            attempts += 1
                            if attempts >= MAXATTEMPTS:
                                raise Exception("Out of tries. Moving on.")
                            else:
                                print(f"Retrying in {WAITTIME} seconds.")
                                time.sleep(WAITTIME)
                        else:
                            attempts = MAXATTEMPTS
                            print("Cross-match returned " + str(len(result)) + " matches.")    

                    for row in enumerate(sourcelist):
                        source = row[1]
                        if abs(source["xcentroid"]-target_pos[0])+abs(source["ycentroid"]-target_pos[1]) < 3:
                            source["ref_mag"] = 1000
                            failed = False
                        else:
                            for cross in result:
                                err = abs(cross["ra"]-source["ra"])+abs(cross["dec"]-source["dec"])
                                if err < 0.001:
                                    source["ref_mag"] = cross["psfMag_"+filt[0]]
                                    source["ref_mag_err"] = cross["psfMagerr_"+filt[0]]
                    
                    sourcelist = sourcelist[sourcelist["ref_mag"] != 0]
                    references = len(sourcelist) - 1 + int(failed)
                    print(str(references)+" matches confirmed.")
                    if failed: print("Failed to find target in source list.")
                    else: print("Target found in source list.")

                    print("Performing aperture photometry...")
                    positions = [(source["xcentroid"],source["ycentroid"]) for source in sourcelist]
                    aperture = CircularAperture(positions, r=FWHM*1.75)
                    annulus = CircularAnnulus(positions, r_in=FWHM*2., r_out=FWHM*3.)
                    sclip = SigmaClip(sigma=3.0, maxiters=5)
                    obj_stats = ApertureStats(final_image, aperture, sigma_clip=None)
                    bkg_raw_stats = ApertureStats(final_image, annulus, sigma_clip=None)
                    bkg_stats = ApertureStats(final_image, annulus, sigma_clip=sclip)
                    total_bkg = bkg_stats.median * obj_stats.sum_aper_area.value
                    bkgsub = obj_stats.sum - total_bkg
                    inst_mag = -2.5 * np.log10(bkgsub)
                    snr = bkgsub/np.sqrt(bkgsub+obj_stats.sum_aper_area.value*(1+obj_stats.sum_aper_area.value/bkg_stats.sum_aper_area.value)*(bkg_raw_stats.std+SRSIZE*dark_err+SRSIZE*READNOISE**2+SRSIZE*(GAIN*ANALOGDIGITALERROR)**2))
                    inst_mag_err = 2.5/(snr*np.log(10))
                    med_snr = np.median(snr)
                    sourcelist.add_columns([inst_mag,inst_mag_err],names=["inst_mag","inst_mag_err"])
                    reflist = sourcelist[sourcelist["ref_mag"] != 1000]

                    blist = [source["ref_mag"]-source["inst_mag"] for source in reflist]

                    iters = 0
                    zero_point_std = 1
                    while(zero_point_std > 0.1):
                        iters += 1
                        bstats = sigma_clipped_stats(blist,sigma=1,maxiters=iters)
                        zero_point = bstats[1]
                        zero_point_std = bstats[2]
                        
                    bclip = sigma_clip(blist,sigma=1,maxiters=iters)
                    reflist.add_column(bclip.mask,name="outlier")
                    inlist = reflist[reflist["outlier"]==False]
                    outlist = reflist[reflist["outlier"]==True]
                    nref = len(inlist)
                    zero_point_err = zero_point_std / np.sqrt(nref) # standard error estimated as std divided by sqrt of sample size.
                    
                    if failed: 
                        print("Zero-point computed as " + str(zero_point) + " with error " + str(zero_point_err))
                    else: 
                        t_inst_mag, t_inst_mag_err = sourcelist[sourcelist["ref_mag"]==1000]["inst_mag","inst_mag_err"][0]
                        t_mag = t_inst_mag + zero_point
                        t_mag_err = np.sqrt(zero_point_err**2 + t_inst_mag_err**2)
                        print(f"Target magnitude computed as {t_mag} with error {t_mag_err}.")

                        fig_zp = plt.figure()
                        plt.xlabel("Instrumental magnitude")
                        plt.ylabel("Reference magnitude")
                        plt.title(f"Night of {folder}, {run}.{subrun} in filter {filt}")
                        plt.suptitle("Zero point " + str("{:.4f}".format(zero_point)) + r"$\pm$" + str("{:.4f}".format(zero_point_err)) + f" with {iters} iterations")
                        xlist_zp = np.linspace(np.min(reflist["inst_mag"]-0.1),np.max(reflist["inst_mag"]+0.1))
                        ylist_zp = xlist_zp + zero_point
                        plt.plot(xlist_zp,ylist_zp,c="royalblue",label="Fit line")
                        plt.scatter(inlist["inst_mag"],inlist["ref_mag"],marker="o",c="dodgerblue",label="Reference stars")
                        plt.scatter(outlist["inst_mag"],outlist["ref_mag"],marker="x",c="orangered",label="Rejected outliers")
                        plt.scatter([t_inst_mag],[t_mag],marker="D",c="seagreen",label="Target")
                        plt.legend()
                        plt.savefig(pipeout+f"\\plots\\{filt}_{run}-{subrun}_zeropoint.png",bbox_inches="tight")
                        outtable.add_row((JD_OBS,DATE_OBS,filt,run+"."+str(subrun),zero_point,zero_point_err,t_mag,t_mag_err,nref,med_snr))
                except Exception as e:
                    print(f"Failure in {run}.{subrun} for filter {filt} in {base_path}. Reason: {e}")

print("Analysis completed! Writing files.")
outtable.write(directory+"\\outdata\\data.csv",format="csv",overwrite=True)
with open(directory+"\\outdata\\failures.txt","w") as f: f.write("\n".join(disappointments))
print("Complete.")

# Last step is to plot the final data.