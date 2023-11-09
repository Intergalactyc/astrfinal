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

from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.aperture import aperture_photometry, ApertureStats, CircularAnnulus, CircularAperture, SkyCircularAperture

import matplotlib.pyplot as plt

start_time = time.time()

warnings.simplefilter('ignore', category=np.exceptions.VisibleDeprecationWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)

target_sky = SkyCoord("22h02m43.26s +42d16m39.65s") # BL Lac coordinates

CENTER_RA, CENTER_DEC = 330.23, 42.28 # RA and Dec for center of search, in degrees
FWHM_LIST = [7.0,10.0,9.0] # Different nights have different FWHMs
FOLDER_ID = 0
R_AP = 1.75
R_IN = 2.0
R_OUT = 3.25
WIDTH = 3216
HEIGHT = 2208
SRSIZE = 5
GAIN = 2.45 # Gain of specific camera used on medium-gain readout mode
ANALOGDIGITALERROR = np.sqrt(1/12) # Error in analog-digital conversion
READNOISE = 2.5 # From the specific camera used on medium-gain readout mode
MAXATTEMPTS = 30
MAXALIGNS = 10
WAITTIME = 10

ast = AstrometryNet()
ast.api_key = "vxbrustekypatepi"

directory = ".\\data"
subfolders = []
subpipes = ["masters","images","plots"]
subout = ["outdata"]

scan = os.scandir(directory)
for item in scan: 
    if item.is_dir() and item.name not in subout: subfolders.append(item.name)

for so in subout:
    path = os.path.join(directory,so)
    os.makedirs(path,exist_ok=True)

outtable = Table(names = ("JD","DATE","FILT","RUN","ZP","ZP_ERR","T_MAG","T_MAG_ERR","FAIL","NREF","TGT_SNR","T_INST_MAG_ERR","BAD"),dtype=("f8","U16","U8","U8","f8","f8","f8","f8","i4","i4","f4","f4","?"))
disappointments = []
for folder in subfolders:
    FWHM = FWHM_LIST[FOLDER_ID]
    try:
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
        med_dark = np.median(master_dark)
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

        # Now for the real processing
        for filt in science.keys():
            master_flat = fits.getdata(pipeout+"\\masters\\master_flat_"+filt+".fit")
            for run in science[filt].keys():
                subruns = ceil(len(science[filt][run])/SRSIZE)
                fig_zp = plt.figure()
                # Iterate over subruns of size defined by SRSIZE
                for subrun in range(subruns):
                    try:

                        






                        print(f"Calibrating, aligning, and stacking frames in {run}.{subrun} for filter {filt} in {base_path}")
                        target_fits = fits.open(science[filt][run][subrun*SRSIZE])
                        target_ccd = target_fits[0].data
                        JD_OBS = target_fits[0].header["JD"]
                        DATE_OBS = folder
                        target_cal = (target_ccd - master_bias - master_dark)/master_flat
                        considered_frames = science[filt][run][subrun*SRSIZE+1:(subrun+1)*SRSIZE]
                        cal_stack = [((fits.getdata(frame) - master_bias - master_dark)/master_flat, frame) for frame in considered_frames]
                        alg_stack = [target_ccd]
                        translations = [[],[]]
                        failed_aligns = 0
                        # Run alignment. If a frame fails to align to the target frame, attempt to align it to the frame following the target frame, and so on up to MAXALIGNS times, after which it will be labeled a "disappointment" and discarded
                        for i in tqdm(range(len(cal_stack)),desc="Aligning..."):
                            align_attempts = 0
                            align_target = target_cal
                            while(align_attempts <= min([MAXALIGNS,i-failed_aligns])):
                                if align_attempts > 0:
                                    align_target = alg_stack[align_attempts]
                                try:
                                    ccd = cal_stack[i][0]
                                    transform = aa.find_transform(ccd,align_target)
                                except Exception:
                                    align_attempts += 1
                                else:
                                    align_attempts = 100
                                    translations[0].append(transform[0].translation[0])
                                    translations[1].append(transform[0].translation[1])
                                    aligned = aa.apply_transform(transform[0],ccd,align_target)[0]
                                    alg_stack.append(aligned)
                            if align_attempts != 100:
                                failed_aligns += 1 
                                disappointments.append(cal_stack[i][1])
                        
                        # Find border regions not common to all frames
                        rightcrop = ceil(0.5 * (max(translations[0])+abs(max(translations[0]))))+FWHM
                        leftcrop = ceil(0.5 * (min(translations[0])-abs(min(translations[0]))))+FWHM
                        topcrop = ceil(0.5 * (max(translations[1])+abs(max(translations[1]))))+FWHM
                        botcrop = ceil(0.5 * (min(translations[1])-abs(min(translations[1]))))+FWHM

                        # Is there a better way to do the above? Having to find_transform and then apply_transform seems to take significantly longer than just registering

                        # Stack aligned frames
                        final_image = np.sum(alg_stack, axis=0).astype("float32")
                        FRAMES = len(alg_stack)

                        # Run a DAOStarFinder source finding algorithm with median sky subtraction
                        print("Finding sources...")
                        daofind = DAOStarFinder(fwhm=FWHM, threshold=3.*np.std(final_image))
                        sources = daofind(final_image - np.median(final_image))
                        ogsources = len(sources)
                        for source in enumerate(sources):
                            # If the source is outside of the determined borders, discard it
                            if not (leftcrop < source[1]["xcentroid"] < WIDTH - rightcrop and botcrop < source[1]["ycentroid"] < HEIGHT - topcrop):
                                sources.remove_row(source[0])
                        netsources = len(sources)
                        sources.sort("flux")
                        sources.reverse()
                        print(str(netsources) + " sources found after eliminating " + str(ogsources-netsources) + " out-of-bounds. Querying astrometry.net for astrometric solution")

                        # Query Astrometry.net for an astrometric solution. Try-catch set up, mostly for the sake of retrying on a connection failure
                        attempts = 0
                        while(attempts < MAXATTEMPTS):
                            try:
                                wcs_header = ast.solve_from_source_list(sources["xcentroid"], sources["ycentroid"],WIDTH,HEIGHT,solve_timeout=120,center_ra=CENTER_RA,center_dec=CENTER_DEC,radius=0.5)
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
                                else:
                                    print("Ultimately failed to find astrometric solution, but did not throw error.")
                        
                        # Write out astrometrically solved stacked image
                        fits.PrimaryHDU(final_image,wcs_header).writeto(pipeout+f"\\images\\final_{filt}_{run}-{subrun}.fit",overwrite=True)
                        
                        wcs = WCS(wcs_header)
                        target_pos = wcs.world_to_pixel(target_sky)
                        sourcepositions = [(source["xcentroid"],source["ycentroid"]) for source in sources]
                        sourcelist = sources["xcentroid","ycentroid"]
                        skycoords = np.transpose(wcs.pixel_to_world_values(sourcepositions))
                        sourcelist.add_columns([skycoords[0],skycoords[1],0.],names=["ra","dec","ref_mag"])

                        failed = True

                        # Use Astroquery to check the SDSS catalog for source matches in order to estimate zero-point for photometric calibration. Again a try-catch is set up to retry in case of connection failure
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

                        # Instead of this I could possibly run a cone search once, and then here just perform a list pairing instead of querying every single time. Possibly even use the same reference stars (within bounds) each time - independent of source list????

                        # Confirm matches and pair lists
                        for source in sourcelist:
                            if abs(source["xcentroid"]-target_pos[0])+abs(source["ycentroid"]-target_pos[1]) < 3:
                                # Mark target with a reference magnitude "tag" of 1000
                                source["ref_mag"] = 1000
                                failed = False
                            else:
                                for cross in result:
                                    err = abs(cross["ra"]-source["ra"])+abs(cross["dec"]-source["dec"])
                                    if err < 0.001:
                                        source["ref_mag"] = cross["psfMag_"+filt[0]]
                        # Remove sources too close to eachother / sources without a catalog cross-match
                        sourcelist = sourcelist[sourcelist["ref_mag"] != 0] 
                        keepers = list(range(len(sourcelist)))
                        for first in enumerate(sourcelist):
                            for second in enumerate(sourcelist):
                                if (first != second) and (np.sqrt((first[1]["xcentroid"]-second[1]["xcentroid"])**2+(first[1]["ycentroid"]-second[1]["ycentroid"])**2) < FWHM * (R_OUT + R_AP)):
                                    if not(first[1]["ref_mag"] == 1000 or second[1]["ref_mag"] == 1000):
                                        if first[0] in keepers:
                                            keepers.remove(first[0])
                                        if second[0] in keepers:
                                            keepers.remove(second[0])
                        sourcelist = sourcelist[keepers]
                        references = len(sourcelist) - 2 + int(failed)
                        print(str(references)+" matches confirmed.")
                        if failed: 
                            print("Failed to find target in source list. Using WCS-derived position instead.")
                            sourcelist.add_row((target_pos[0],target_pos[1],0,0,1000))
                        else: 
                            print("Target found in source list.")

                        # Aperture photometry: circular apertures; sigma-clipped median background from annulus
                        print("Performing aperture photometry...")
                        positions = [(source["xcentroid"],source["ycentroid"]) for source in sourcelist]
                        aperture = CircularAperture(positions, r=FWHM*R_AP)
                        annulus = CircularAnnulus(positions, r_in=FWHM*R_IN, r_out=FWHM*R_OUT)
                        sclip = SigmaClip(sigma=3.0, maxiters=5)
                        obj_stats = ApertureStats(final_image, aperture, sigma_clip=None)
                        bkg_stats = ApertureStats(final_image, aperture, sigma_clip=sclip)
                        bkg_perpixel = bkg_stats.median
                        total_bkg = bkg_perpixel * obj_stats.sum_aper_area.value
                        bkgsub = obj_stats.sum - total_bkg
                        # Tag bad objects such that their instrumental magnitude is computed as exactly -40
                        bkgsub[bkgsub<=0] = 10**16
                        inst_mag = -2.5 * np.log10(bkgsub)
                        # Estimate S/N ratio using the modified CCD equation.
                        snr = bkgsub/np.sqrt(bkgsub+obj_stats.sum_aper_area.value*(1+obj_stats.sum_aper_area.value/bkg_stats.sum_aper_area.value)*(bkg_perpixel+FRAMES*med_dark+FRAMES*READNOISE**2+FRAMES*(GAIN*ANALOGDIGITALERROR)**2))
                        # Find error in instrumental magnitude from S/N
                        inst_mag_err = 2.5/(snr*np.log(10))
                        sourcelist.add_columns([inst_mag,inst_mag_err,snr],names=["inst_mag","inst_mag_err","snr"])
                        
                        # Remove bad objects as previously tagged
                        sourcelist = sourcelist[sourcelist["inst_mag"] != -40]
                        med_inst_mag_err = np.median(sourcelist["inst_mag_err"])
                        # Don't use target as a zp reference; only use references within 2 magnitudes of target
                        t_inst_mag, t_inst_mag_err, t_snr = sourcelist[sourcelist["ref_mag"]==1000]["inst_mag","inst_mag_err","snr"][0]
                        reflist = sourcelist[sourcelist["ref_mag"] != 1000]
                        reflist = reflist[abs(reflist["inst_mag"]-t_inst_mag)<2]
                        if len(reflist) < 5:
                            print("Very few (n<5) reference stars found. Data quality likely poor.")
                            bad = True
                        # Estimated list of zero-points for each reference star
                        blist = [source["ref_mag"]-source["inst_mag"] for source in reflist]
                        bad = False
                        # Perform sigma-clipping to determine zero-point without outliers
                        bstats = sigma_clipped_stats(blist,sigma=1,maxiters=2,cenfunc="median") # Want a better way of detecting outliers. Cluster analysis?
                        zero_point = bstats[1]
                        zero_point_std = bstats[2]
                        # Repeat the same sigma-clipping but now in order to determine WHICH points are outliers
                        bclip = sigma_clip(blist,sigma=2,maxiters=2,cenfunc="median") 
                        reflist.add_column(bclip.mask,name="outlier")
                        inlist = reflist[reflist["outlier"]==False]
                        outlist = reflist[reflist["outlier"]==True]
                        nref = len(inlist)
                        # Standard error on the mean estimated as sample standard deviation divided by square root of sample size
                        zero_point_err = 1.253*zero_point_std / np.sqrt(nref) 
                        if zero_point_err > 0.2:
                            print("Unexpected zero point error. Data quality likely poor.")
                            bad = True
                        print(f"Zero point computed as {zero_point} with error {zero_point_err}.")
                        t_mag = t_inst_mag + zero_point

                        # Zero point error and instrumental magnitude error combine to make the net error on the target magnitude
                        t_mag_err = np.sqrt(zero_point_err**2 + t_inst_mag_err**2)
                        print(f"Target S/N ratio: {t_snr}")
                        print(f"Target magnitude computed as {t_mag} with error {t_mag_err}.")
                        if np.isnan(t_mag) or t_snr < 0:
                            print("Nonphysical values obtained. Data quality likely poor.")
                            bad = True

                        # Plot data regarding zero-point calculation
                        plt.xlabel("Instrumental magnitude")
                        plt.title(f"Night of {folder}, {run}.{subrun} in filter {filt}")
                        plt.suptitle("Zero point " + str("{:.4f}".format(zero_point)) + r"$\pm$" + str("{:.4f}".format(zero_point_err)))
                        xlist_zp = [np.min(inlist["inst_mag"])-0.1,np.max(inlist["inst_mag"])+0.1]
                        ylist_zp = [np.min(inlist["inst_mag"])-0.1+zero_point,np.max(inlist["inst_mag"])+0.1+zero_point]
                        plt.plot(xlist_zp,ylist_zp,c="royalblue",label="Fit line")
                        plt.scatter(inlist["inst_mag"],inlist["ref_mag"],marker="o",c="dodgerblue",label="Reference stars")
                        plt.scatter(outlist["inst_mag"],outlist["ref_mag"],marker="x",c="orangered",label="Rejected outliers")
                        plt.scatter([t_inst_mag],[t_mag],marker="D",c="mediumorchid",label="Target")
                        plt.legend()
                        plt.savefig(pipeout+f"\\plots\\{filt}_{run}-{subrun}_zeropoint.png",bbox_inches="tight")
                        plt.cla()

                        # Append data to output table
                        outtable.add_row((JD_OBS,DATE_OBS,filt,run+"."+str(subrun),zero_point,zero_point_err,t_mag,t_mag_err,failed_aligns,nref,t_snr,t_inst_mag_err,bad))
                    except Exception as e:
                        print(f"Failure in {run}.{subrun} for filter {filt} in {base_path}. Reason: {e}")
                plt.close(fig_zp)
    except Exception as e:
        print(f"Failure in {folder}. Reason: {e}")
    FOLDER_ID += 1

print("Analysis completed! Writing files.")
outtable.write(directory+"\\outdata\\data.csv",format="csv",overwrite=True)
with open(directory+"\\outdata\\failures.txt","w") as f: f.write("\n".join(disappointments))
print(f"Complete. Execution time: {time.time()-start_time} seconds.")