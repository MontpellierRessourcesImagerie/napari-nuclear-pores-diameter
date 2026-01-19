import tifffile
import numpy as np
import csv
from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def plot_profiles(profiles, median_minima_idx, pl_idx, pr_idx, working_dir, c_idx):
    if working_dir is None:
        return
    name = f"profile_centroid_{str(c_idx).zfill(4)}.png"
    path = working_dir / name
    plt.figure(figsize=(10, 6))
    for i, profile in enumerate(profiles):
        plt.plot(profile, alpha=0.6, label=f'Profile {i+1}', marker='o', markersize=3)
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Intensity')
    plt.title('Radial Profiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(median_minima_idx, color='red', linestyle='--', label='Median Minima')
    print(f"Left Peak Index: {pl_idx}, Right Peak Index: {pr_idx}")
    plt.axvline(pl_idx, color='green', linestyle='--', label='Left Peak')
    plt.axvline(pr_idx, color='blue', linestyle='--', label='Right Peak')
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def smooth_profile(profile, window_size=2):
    if window_size < 1:
        return profile
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(profile, window, mode='same')
    return smoothed

def find_minima(smoothed_profile, derivative):
    minima_idx = len(smoothed_profile) // 2
    while True:
        if minima_idx <= 0 or minima_idx >= len(smoothed_profile) - 1:
            return None
        m1 = derivative[minima_idx]
        m2 = derivative[minima_idx + 1]
        same_sign = (m1 * m2) > 0
        if same_sign:
            if m1 < 0:
                minima_idx += 1
            else:
                minima_idx -= 1
        else:
            break
    return minima_idx

def find_peak(smoothed_profile, derivative, start_idx, direction):
    peak_idx = start_idx
    while True:
        if peak_idx <= 0 or peak_idx >= len(smoothed_profile) - 1:
            return None
        m1 = derivative[peak_idx]
        m2 = derivative[peak_idx + 1]
        is_peak = (m1 * m2) < 0 and (m1 > 0)
        if not is_peak:
            if m1 < 0 and direction == 'left':
                peak_idx -= 1
            elif m1 < 0 and direction == 'right':
                peak_idx += 1
            elif m1 > 0 and direction == 'left':
                peak_idx -= 1
            elif m1 > 0 and direction == 'right':
                peak_idx += 1
            else:
                break
        else:
            break
    return peak_idx

def analyze_profile_peaks(profile, pxl_size):
    smoothed = smooth_profile(profile)
    derivative = np.gradient(smoothed)
    minima_idx = find_minima(smoothed, derivative)
    if minima_idx is None:
        return None, None, None
    pl_idx = find_peak(smoothed, derivative, minima_idx, 'left')
    pr_idx = find_peak(smoothed, derivative, minima_idx, 'right')
    if pl_idx is None or pr_idx is None:
        return None, None, None
    return minima_idx, pl_idx, pr_idx

def gaussian_center(y):
    """
    Estimate the center µ of a 1D Gaussian-like signal y[i].
    Returns a float µ, potentially between indices.
    """
    if len(y) <= 5:
        return None
    
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)

    # Gaussian model
    def gaussian(x, A, mu, sigma, offset):
        return offset + A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Initial guesses
    A0 = np.max(y) - np.min(y)
    mu0 = np.argmax(y)       # peak index
    sigma0 = len(y) / 10     # loose guess
    offset0 = np.min(y)

    p0 = [A0, mu0, sigma0, offset0]

    # Fit
    popt, _ = curve_fit(
        gaussian, x, y, p0=p0, maxfev=5000
    )

    A, mu, sigma, offset = popt
    return float(mu)

def analyze_profile_gaussian(profile, pxl_size):
    smoothed = smooth_profile(profile)
    derivative = np.gradient(smoothed)
    minima_idx = find_minima(smoothed, derivative)
    if minima_idx is None:
        return None, None, None
    try:
        pl_idx = gaussian_center(smoothed[:minima_idx])
        pr_idx = gaussian_center(smoothed[minima_idx:])
    except RuntimeError:
        return None, None, None
    if pl_idx is None or pr_idx is None:
        return None, None, None
    pr_idx += minima_idx
    if int(pl_idx) <= 0 or int(pr_idx) >= len(smoothed) - 1:
        return None, None, None
    if int(pl_idx) >= len(smoothed) or int(pr_idx) <= 0:
        return None, None, None
    print(f"µ1: {pl_idx}, µ2: {pr_idx}")
    return minima_idx, pl_idx, pr_idx

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

def extract_values(pl_idx, m_idx, pr_idx, profile, pxl_size):
    left_peak_value = profile[int(pl_idx)]
    if int(pl_idx) < len(profile) - 1:
        left_peak_value = max(left_peak_value, profile[int(pl_idx)+1])
    right_peak_value = profile[int(pr_idx)]
    if int(pr_idx) < len(profile) - 1:
        right_peak_value = max(right_peak_value, profile[int(pr_idx)+1])
    minima_value = profile[m_idx]
    gap_length = (pr_idx - pl_idx) * pxl_size
    prominence_min = min(left_peak_value - minima_value, right_peak_value - minima_value)
    prominence_max = max(left_peak_value - minima_value, right_peak_value - minima_value)
    return left_peak_value, right_peak_value, minima_value, gap_length, prominence_min, prominence_max

def radial_profiles(image, centroids, diameter_px, pxl_size, n_steps, mode='peaks', working_dir=None):
    angles = np.linspace(0, 180, n_steps, endpoint=False)
    results = []
    analyze_profile = analyze_profile_peaks if mode == 'peaks' else analyze_profile_gaussian
    for c_idx, center in enumerate(centroids):
        profiles = []
        for angle in angles:
            profile = make_profile(image, center, diameter_px, angle)
            profiles.append(profile)
        profiles = np.array(profiles)
        avg_profile = np.median(profiles, axis=0)
        minima_idx, pl_idx, pr_idx = analyze_profile(avg_profile, pxl_size)
        if minima_idx is None or pl_idx is None or pr_idx is None:
            print("No minima, left peak or right peak found for centroid at:", center)
            results.append(None)
            continue
        plot_profiles([avg_profile], minima_idx, pl_idx, pr_idx, working_dir, c_idx)
        results.append(extract_values(pl_idx, minima_idx, pr_idx, avg_profile, pxl_size))
    return results


def export_as_csv(results, working_dir, filename="radial_profiles_results.csv"):
    columns = ["Index", "Left Peak Value", "Right Peak Value", "Minima Value", "Gap Length (µm)", "Prominence Min", "Prominence Max"]
    file_path = working_dir / filename
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for idx, result in enumerate(results):
            if result is None:
                continue
            writer.writerow([idx] + list(result))


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
    radial_profiles(
        image_data, 
        points, 
        diameter_px=as_pxls(0.36, 0.018), 
        pxl_size=0.018, 
        n_steps=36, 
        mode='gaussian', 
        working_dir=folder_path
    )