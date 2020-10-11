import pandas as pd
import tarfile

import astroquery
from astroquery.alma import Alma
from astropy.table import Table
alma = Alma()
alma.archive_url = 'https://almascience.nrao.edu'

def get_mous(science_keyword, save_file_path = None):
    """Function to get all mous IDs for a given science keyword

    science_keyword: string search keyword
    save_file_path: optional path to save csv of results
    """



def download_all_fits(fits_links, cache_location = '', unzip=True, clean_up=True):
    """Function to download and unzip all FITS files from a list of mous IDs

    mous_list_loc: path to csv with mous IDs for a given search term
    cache_location: path to save downloaded files to (default current working directory)
    unzip: binary whether or not to unzip tar files
    clean_up: if true, remove tar files after unzipping and move fits files out of nested subpath
    """
