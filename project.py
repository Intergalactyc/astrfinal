import os
import glob

from pathlib import PurePath
from tqdm import tqdm
from math import ceil

import astroalign as aa
import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from astroquery.astrometry_net import AstrometryNet

from astroquery.sdss import SDSS
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, ApertureStats, CircularAnnulus, CircularAperture, SkyCircularAperture


target_sky = SkyCoord("22h02m43.26s +42d16m39.65s")

FWHM = 8.0
WIDTH = 3216
HEIGHT = 2208

ast = AstrometryNet()
ast.api_key = "vxbrustekypatepi"

directory = "./testdata1"
subfolders = []
subpipes = ["masters","images"]

scan = os.scandir(directory)
for item in scan: 
    if item.is_dir(): subfolders.append(item.name)

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

    disappointments = []
    for filt in science.keys():
        master_flat = fits.getdata(pipeout+"\\masters\\master_flat_"+filt+".fit")
        for run in science[filt].keys():
            try:
                print("Calibrating, aligning, and stacking frames in " + run + " for filter " + filt + " in " + base_path)
                target_ccd = fits.getdata(science[filt][run][0])
                target_cal = (target_ccd - master_bias - master_dark)/master_flat
                cal_stack = [[(fits.getdata(frame) - master_bias - master_dark)/master_flat, frame] for frame in science[filt][run]]
                alg_stack = []
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

                print("Stacking...")
                final_image = np.sum(alg_stack, axis=0).astype("float32")

                print("Finding sources...")
                daofind = DAOStarFinder(fwhm=FWHM, threshold=4.*np.std(final_image))
                sources = daofind(final_image - np.median(final_image))
                ogsources = len(sources)
                for source in enumerate(sources):
                    if not (leftcrop < source[1]["xcentroid"] < WIDTH - rightcrop and botcrop < source[1]["ycentroid"] < HEIGHT - topcrop):
                        sources.remove_row(source[0])
                netsources = len(sources)
                print(str(netsources) + " sources found after eliminating " + str(ogsources-netsources) + " out-of-bounds. Querying astrometry.net for astrometric solution")
                sources.sort("flux")
                sources.reverse()

                try:
                    wcs_header = ast.solve_from_source_list(sources["xcentroid"], sources["ycentroid"],WIDTH,HEIGHT,solve_timeout=60,center_ra=330.23,center_dec=42.28,radius=0.5)
                except TimeoutError:
                    print("Timed out before solution found.")
                if wcs_header:
                    print("Astrometric solution found!")
                else:
                    print("Failed to find astrometric solution.")

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
                result = SDSS.query_crossid(wcs.pixel_to_world(sources["xcentroid"],sources["ycentroid"]))
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
                references = len(sourcelist) - int(failed)
                print(str(references)+" matches confirmed.")
                if failed: print("Failed to find target in source list.")
                else: print("Target found in source list.")

                print("Performing aperture photometry...")
                positions = [(source["xcentroid"],source["ycentroid"]) for source in sourcelist]
                aperture = CircularAperture(positions, r=FWHM*1.5)
                annulus = CircularAnnulus(positions, r_in=FWHM*2., r_out=FWHM*3.)
                sclip = SigmaClip(sigma=3.0, maxiters=10)
                obj_stats = ApertureStats(final_image, aperture, sigma_clip=None)
                bkg_stats = ApertureStats(final_image, annulus, sigma_clip=sclip)
                total_bkg = bkg_stats.median * obj_stats.sum_aper_area.value
                bkgsub = obj_stats.sum - total_bkg
                inst_mag = -2.5 * np.log10(bkgsub)
                bkgerr = bkg_stats.std * obj_stats.sum_aper_area.value
                inst_mag_err = -2.5 * np.log10(bkgerr)
                sourcelist.add_columns([inst_mag,inst_mag_err],names=["inst_mag","inst_mag_err"])

                blist = [source["ref_mag"]-source["inst_mag"] for source in sourcelist]
                bstats = sigma_clipped_stats(blist,sigma=1)
                zero_point = bstats[1]
                zero_point_err = bstats[2]
                
                if failed: 
                    print("Zero-point computed as " + str(zero_point) + " with error " + str(zero_point_err))
                else: 
                    t_inst_mag = sourcelist[sourcelist["ref_mag"]==1000]["inst_mag"][0]
                    t_mag = t_inst_mag + zero_point
                    print("Target magnitude computed as " + str(t_mag) + " with zero-point error " + str(zero_point_err))
            except Exception as e:
                print(f"Failure in {run} for filter {filt} in {base_path}. Reason: {e}")
            
    print(disappointments)


# Want 2 files as an overall result (across all three subfolders):
    # Text file with list of "disappointments"
    # CSV file with the following columns:
        # filt / Filter out of {g',r',i'}
        # date / Date of observation
        # run / Intranight run number
        # MJD_obs / MJD of observation beginning
        # zero_point /
        # zero_point_error / Error in zero-point, as std of sigma-clipped zero-point list
        # mag / BL_Lac magnitude (t_mag) in the given filter
        # mag_err / Estimated error in mag
        # references / Number of cross-matched stars used as references