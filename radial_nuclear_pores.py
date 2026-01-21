import tifffile
import numpy as np
import csv
from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.stats import norm
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def plot_profiles(properties, working_dir, c_idx, pxl_size, max_img):
    if working_dir is None:
        return
    name = f"profile_centroid_{str(c_idx).zfill(4)}.png"
    path = working_dir / name
    plt.figure(figsize=(10, 6))
    
    plt.scatter(properties['x_ori'] * pxl_size, properties['y_ori'], label="original", marker='o')
    plt.plot(properties['x_inter'] * pxl_size, properties['y_inter'], label="cubic", linestyle='--')
    plt.plot(properties['x_inter'] * pxl_size, properties['y_base'], label="initial guess", linestyle='-.')
    plt.plot(properties['x_inter'] * pxl_size, properties['y_theor'], label="fit", linestyle=':')
    plt.axvline(x=properties['x1'] * pxl_size, color='red', linestyle='--', alpha=0.7, label='x1')
    plt.axvline(x=properties['x2'] * pxl_size, color='green', linestyle='--', alpha=0.7, label='x2')
    plt.ylim(0, max_img)
    plt.title(f"Radial Profile Centroid {c_idx}")
    plt.xlabel(f"x ({pxl_size} µm/pxl)")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    textstr = "\n".join((
        f"factor = x{properties['factor']:.2f}",
        f"Ioffset = {properties['i_offset']:.2f}",
        f"x1 = {pxl_size * properties['x1']:.2f}µm",
        f"x2 = {pxl_size * properties['x2']:.2f}µm",
        f"i1 = {properties['i1']:.2f}",
        f"i2 = {properties['i2']:.2f}",
        f"fwhm1 = {pxl_size * properties['fwhm1']:.2f}µm",
        f"fwhm2 = {pxl_size * properties['fwhm2']:.2f}µm",
        f"shrink = x{properties['shrink']:.2f}"
    ))
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def double_gaussian(x, offset, x1, x2, i1, i2, fwhm1, fwhm2):
    g1 = i1 * np.exp(-4.0 * np.log(2) * ((x - x1) / fwhm1) ** 2)
    g2 = i2 * np.exp(-4.0 * np.log(2) * ((x - x2) / fwhm2) ** 2)
    return offset + g1 + g2

def analyze_profile_gaussian(profile, pxl_size, shrink, factor, max_img):
    y_ori = np.array(profile)
    x_ori = np.arange(0, factor * len(profile), factor)

    spl = CubicSpline(x_ori, y_ori)
    x_inter = np.arange(0, factor * len(profile), 1)
    y_inter = spl(x_inter)

    i_offset = np.min(y_inter[len(y_inter)//2-2:len(y_inter)//2+2])
    x1 = int(len(y_inter) / 4)
    x2 = int(3 * len(y_inter) / 4)
    inter = (x2 - x1) * shrink
    x1 = (x1 + x2) / 2 - inter / 2
    x2 = (x1 + x2) / 2 + inter / 2
    i1 = np.max(y_inter[:len(y_inter)//2]) - i_offset
    i2 = np.max(y_inter[len(y_inter)//2:]) - i_offset
    fwhm1 = len(y_inter) / 8
    fwhm2 = len(y_inter) / 8

    y_base = double_gaussian(x_inter, i_offset, x1, x2, i1, i2, fwhm1, fwhm2)
    try:
        popt, _ = curve_fit(
            double_gaussian, x_inter, y_inter, p0=[i_offset, x1, x2, i1, i2, fwhm1, fwhm2], maxfev=5000
        )
    except RuntimeError:
        return None

    i_offset_fit, x1_fit, x2_fit, i1_fit, i2_fit, fwhm1_fit, fwhm2_fit = popt
    y_theor = double_gaussian(x_inter, *popt)

    if (x2_fit - x1_fit) <= 0:
        return None
    
    prominence_left = (double_gaussian(x1_fit, *popt) - i_offset_fit) / (max_img - i_offset_fit)
    prominence_right = (double_gaussian(x2_fit, *popt) - i_offset_fit) / (max_img - i_offset_fit)

    return {
        'x_ori'   : x_ori,
        'y_ori'   : y_ori,
        'x_inter' : x_inter,
        'y_inter' : y_inter,
        'y_base'  : y_base,
        'y_theor' : y_theor,
        'i_offset': i_offset_fit,
        'x1'      : x1_fit,
        'x2'      : x2_fit,
        'i1'      : i1_fit,
        'i2'      : i2_fit,
        'fwhm1'   : fwhm1_fit,
        'fwhm2'   : fwhm2_fit,
        'factor'  : factor,
        'inter'   : "cubic",
        'peak_l'  : double_gaussian(x1_fit, *popt),
        'peak_r'  : double_gaussian(x2_fit, *popt),
        'minima'  : double_gaussian((x1_fit + x2_fit)/2, *popt),
        'shrink'  : shrink,
        'prom_l'  : prominence_left,
        'prom_r'  : prominence_right
    }


def make_profile(image, center, diameter_px, angle):
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    t = np.linspace(-diameter_px/2, diameter_px/2, int(diameter_px))
    y_coords = center[0] + t * dy
    x_coords = center[1] + t * dx
    coords = np.array([y_coords, x_coords])
    profile = map_coordinates(image, coords, order=1, mode='constant', cval=0)
    return profile

def get_results_keys():
    return [
        "Index", "Factor", "X1", "X2", "I1", "I2", "FWHM1", "FWHM2",
        "IOffset", "Left Peak Value", "Right Peak Value", "Minima Value",
        "Gap Length (µm)", "StdDev Ratio", "Intensity Ratio", "Prominence Left", 
        "Prominence Right"
    ]

def add_to_results(properties, pxl_size, c_idx, results):
    if results is None:
        return
    entry    = [0] * len(get_results_keys())
    entry[0] = c_idx
    entry[1] = properties['factor']
    entry[2] = properties['x1'] / properties['factor'] * pxl_size
    entry[3] = properties['x2'] / properties['factor'] * pxl_size
    entry[4] = properties['i1']
    entry[5] = properties['i2']
    entry[6] = properties['fwhm1'] / properties['factor'] * pxl_size
    entry[7] = properties['fwhm2'] / properties['factor'] * pxl_size
    entry[8] = properties['i_offset']
    entry[9] = properties['peak_l']
    entry[10]= properties['peak_r']
    entry[11]= properties['minima']
    entry[12]= (properties['x2'] - properties['x1']) / properties['factor'] * pxl_size
    entry[13]= properties['stddev_ratio']
    entry[14]= properties['intensity_ratio']
    entry[15]= properties['prom_l']
    entry[16]= properties['prom_r']
    results.append(entry)

def smooth_profile(profile, window_size=2):
    if window_size < 1:
        return profile
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(profile, window, mode='same')
    return smoothed

def intensities_ratio(profiles, max_img):
    ratios = (np.mean(np.max(profiles, axis=1)) - np.mean(np.min(profiles, axis=1))) / max_img
    return np.min(ratios)

def radial_profiles(image, centroids, diameter_px, pxl_size, n_steps, factor, shrink=1.0, working_dir=None):
    angles = np.linspace(0, 180, n_steps, endpoint=False)
    results = []
    max_img = image.max()
    for c_idx, center in tqdm(enumerate(centroids), total=len(centroids), desc="Analyzing radial profiles"):
        profiles = []
        for angle in angles:
            profile = make_profile(image, center, diameter_px, angle)
            profiles.append(profile)
        profiles = np.array(profiles)
        stddevs = np.std(profiles, axis=0)
        std_ratio = np.min(stddevs) / (np.max(stddevs) + 1e-6)
        int_ratio = intensities_ratio(profiles, max_img)
        avg_profile = np.median(profiles, axis=0)
        avg_profile = smooth_profile(avg_profile, window_size=2)
        properties = analyze_profile_gaussian(avg_profile, pxl_size, shrink, factor, max_img)
        if properties is None:
            results.append(None)
            continue
        properties['stddev_ratio'] = std_ratio
        properties['intensity_ratio'] = int_ratio
        plot_profiles(properties, working_dir, c_idx, pxl_size, max_img)
        add_to_results(properties, pxl_size, c_idx, results)
    return results

def export_as_csv(results, working_dir, filename="radial_profiles_results.csv"):
    columns = get_results_keys()
    file_path = working_dir / filename
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for idx, result in enumerate(results):
            if result is None:
                continue
            writer.writerow(list(result))

def as_pxls(value, pxl_size):
    return np.ceil(value / pxl_size)

def import_points_from_csv(file_path):
    import csv
    points = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            y, x = float(row[0]), float(row[1])
            points.append((y, x))
    return np.array(points)

if __name__ == "__main__":
    folder_path = Path("/home/clement/Documents/projects/measures-orestis/2025-10-27/STED-pores-nucleaires")
    image_name  = "0v17b.tif"
    image_path = folder_path / image_name
    points_path = folder_path / "detected_nuclear_pores.csv"
    image_data = tifffile.imread(image_path)
    points = import_points_from_csv(points_path)
    print(len(points), "points imported.")
    results = radial_profiles(
        image_data, 
        points, 
        diameter_px=as_pxls(0.25, 0.018), 
        pxl_size=0.018, 
        n_steps=36,
        factor=10.0,
        shrink=0.8,
        working_dir=Path("/tmp/orestis")
    )
    export_as_csv(results, Path("/tmp/orestis"))