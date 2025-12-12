import tifffile
import numpy as np
import napari
import csv
from pathlib import Path
from scipy.ndimage import map_coordinates
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

def analyze_profile(profile, pxl_size):
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
    left_peak_value = profile[pl_idx]
    right_peak_value = profile[pr_idx]
    minima_value = profile[m_idx]
    gap_length = (pr_idx - pl_idx) * pxl_size
    prominence_min = min(left_peak_value - minima_value, right_peak_value - minima_value)
    prominence_max = max(left_peak_value - minima_value, right_peak_value - minima_value)
    return left_peak_value, right_peak_value, minima_value, gap_length, prominence_min, prominence_max

def radial_profiles(image, centroids, diameter_px, pxl_size, n_steps, working_dir=None):
    angles = np.linspace(0, 180, n_steps, endpoint=False)
    results = []
    for c_idx, center in enumerate(centroids):
        profiles = []
        for angle in angles:
            profile = make_profile(image, center, diameter_px, angle)
            profiles.append(profile)
        profiles = np.array(profiles)
        avg_profile = np.mean(profiles, axis=0)
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

if __name__ == "__main__":
    image_path = "/home/clement/Documents/projects/measures-orestis/2025-10-27/STED-pores-nucleaires/0v17b.tif"
    image_data = tifffile.imread(image_path)
    spots_positions = np.array([
        [  9.5       ,  12.        ],
        [ 10.6       , 164.8       ],
        [ 11.63636364, 461.09090909],
        [ 13.125     , 278.125     ],
        [ 14.5       , 168.        ],
        [ 15.        , 182.        ],
        [ 16.66666667, 360.75      ],
        [ 19.5       , 300.5       ],
        [ 23.25      , 212.33333333],
        [ 25.        , 155.5       ],
        [ 25.875     , 324.6875    ],
        [ 26.7       , 181.3       ],
        [ 26.71428571, 406.5       ],
        [ 32.09090909, 300.27272727],
        [ 36.25      , 231.66666667],
        [ 36.5       , 343.        ],
        [ 36.57142857, 142.85714286],
        [ 36.875     , 214.125     ],
        [ 38.38461538, 259.38461538],
        [ 40.5       , 160.71428571],
        [ 40.        , 275.        ],
        [ 40.16666667, 411.16666667],
        [ 45.875     , 206.125     ],
        [ 47.07142857, 291.85714286],
        [ 47.78571429, 183.78571429],
        [ 47.6       , 312.8       ],
        [ 48.        , 419.5       ],
        [ 50.33333333, 360.88888889],
        [ 54.25      , 227.625     ],
        [ 54.63636364, 389.90909091],
        [ 55.66666667, 159.5       ],
        [ 55.5       , 263.71428571],
        [ 55.07142857, 330.85714286],
        [ 61.94444444, 247.22222222],
        [ 62.15384615, 285.53846154],
        [ 63.4       , 359.6       ],
        [ 69.35714286, 136.21428571],
        [ 70.38461538, 303.38461538],
        [ 72.5       , 158.28571429],
        [ 74.71428571, 411.5       ],
        [ 75.5       , 175.22222222],
        [ 76.35714286, 247.21428571],
        [ 77.125     , 378.0625    ],
        [ 77.46153846, 425.84615385],
        [ 77.        , 453.2       ],
        [ 79.        , 193.5       ],
        [ 81.        , 116.58333333],
        [ 81.13333333, 327.53333333],
        [ 82.5       , 445.5       ],
        [ 84.91666667, 146.33333333],
        [ 86.5       , 211.        ],
        [ 88.4       , 347.6       ],
        [ 91.5       , 224.        ],
        [ 94.61538462, 420.61538462],
        [ 97.53333333, 197.86666667],
        [ 97.        , 298.28571429],
        [ 98.64285714, 319.78571429],
        [ 99.125     , 213.125     ],
        [ 99.7       , 287.7       ],
        [100.5       , 122.5       ],
        [102.125     , 158.125     ],
        [101.5       , 167.5       ],
        [104.5       , 358.5       ],
        [104.        , 368.5       ],
        [107.        , 131.5       ],
        [109.        , 214.5       ],
        [111.4375    , 255.375     ],
        [111.        , 354.5       ],
        [112.93333333, 311.86666667],
        [113.63636364,  30.90909091],
        [116.38461538, 173.61538462],
        [116.46153846, 392.84615385],
        [117.875     , 447.125     ],
        [121.5       , 147.71428571],
        [123.28571429, 333.5       ],
        [124.        ,  99.        ],
        [126.        , 126.        ],
        [128.09090909, 111.36363636],
        [128.07142857, 375.85714286],
        [130.35714286, 245.21428571],
        [132.11764706, 215.58823529],
        [132.15384615, 451.        ],
        [132.5       , 187.        ],
        [133.8       , 156.6       ],
        [134.66666667, 294.83333333],
        [136.        , 200.        ],
        [135.91666667, 135.83333333],
        [135.        , 315.        ],
        [139.70588235,  87.05882353],
        [138.6       , 162.8       ],
        [142.53333333, 352.33333333],
        [143.71428571, 104.5       ],
        [149.14285714, 125.42857143],
        [149.5       , 184.28571429],
        [149.5       , 248.        ],
        [151.21428571, 229.35714286],
        [154.15789474, 424.84210526],
        [155.11111111,  79.66666667],
        [155.16666667, 363.83333333],
        [157.61538462, 284.61538462],
        [157.66666667, 141.33333333],
        [160.        , 162.83333333],
        [161.        , 211.5       ],
        [161.42857143, 273.71428571],
        [166.23076923, 249.76923077],
        [167.        ,  82.        ],
        [168.        , 407.8       ],
        [169.125     ,  76.875     ],
        [169.53333333, 309.86666667],
        [170.28571429, 355.5       ],
        [171.6       , 106.06666667],
        [171.        ,  19.5       ],
        [172.61538462, 442.61538462],
        [172.5       ,  48.5       ],
        [174.5       , 288.71428571],
        [179.        , 404.        ],
        [180.5       , 229.        ],
        [181.        , 416.5       ],
        [183.5       ,  70.66666667],
        [183.5       , 182.        ],
        [183.        , 457.        ],
        [184.        , 303.5       ],
        [184.2       , 368.6       ],
        [186.07142857, 116.78571429],
        [186.61538462,  88.38461538],
        [188.05555556, 243.22222222],
        [188.5       , 150.5       ],
        [188.        , 405.        ],
        [190.5       , 277.        ],
        [191.41666667, 459.25      ],
        [191.5       , 330.        ],
        [193.8       , 198.4       ],
        [194.36363636, 425.54545455],
        [195.18181818, 299.18181818],
        [196.84615385, 168.76923077],
        [197.        , 136.25      ],
        [199.        ,  61.5       ],
        [199.5       , 442.71428571],
        [199.5       , 315.5       ],
        [203.5       , 368.        ],
        [203.6       , 195.2       ],
        [204.71428571,  88.        ],
        [204.83333333, 136.83333333],
        [205.70588235, 239.88235294],
        [206.15384615, 417.53846154],
        [206.5       , 400.5       ],
        [208.        ,  94.        ],
        [209.        , 336.2       ],
        [213.30769231, 352.        ],
        [212.66666667, 201.66666667],
        [216.        , 327.75      ],
        [216.5       ,  72.5       ],
        [217.55555556,  54.77777778],
        [218.28571429, 184.5       ],
        [220.18181818,  41.81818182],
        [222.5       , 215.        ],
        [225.2       , 152.2       ],
        [224.70588235, 378.88235294],
        [229.66666667,  57.11111111],
        [230.5       ,  92.        ],
        [231.        , 113.625     ],
        [232.11111111, 239.33333333],
        [231.71428571, 338.        ],
        [232.90909091, 130.36363636],
        [233.53333333, 282.86666667],
        [232.4       , 317.2       ],
        [234.2       , 250.        ],
        [234.81818182, 428.81818182],
        [235.5       , 348.5       ],
        [238.75      , 199.41666667],
        [241.30769231, 403.23076923],
        [240.875     , 310.125     ],
        [244.        , 266.83333333],
        [244.09090909,  83.36363636],
        [247.875     , 162.625     ],
        [248.71428571, 111.5       ],
        [249.08333333, 452.16666667],
        [250.91666667,  55.66666667],
        [250.38461538, 226.38461538],
        [251.66666667,  72.11111111],
        [253.5       ,  37.        ],
        [252.        , 386.        ],
        [253.5       , 242.        ],
        [256.63636364,  85.90909091],
        [256.88235294, 406.58823529],
        [259.33333333, 142.66666667],
        [260.5       , 100.5       ],
        [260.5       , 327.        ],
        [261.73333333, 433.66666667],
        [262.        , 270.5       ],
        [264.        ,  51.        ],
        [265.28571429, 312.5       ],
        [267.21428571, 180.28571429],
        [265.        , 379.        ],
        [267.5       , 353.71428571],
        [267.5       , 247.        ],
        [268.83333333,  30.        ],
        [271.3       , 280.3       ],
        [272.5       ,  60.5       ],
        [271.6       , 145.8       ],
        [272.        , 294.        ],
        [272.46153846, 419.15384615],
        [276.06666667, 256.4       ],
        [276.61538462, 332.38461538],
        [277.28571429,  87.57142857],
        [276.        , 154.        ],
        [278.        , 376.28571429],
        [280.25      , 118.41666667],
        [279.5       , 213.5       ],
        [282.11111111, 304.66666667],
        [284.4       ,  37.06666667],
        [284.16666667, 366.83333333],
        [286.21428571, 393.64285714],
        [285.33333333, 146.66666667],
        [286.71428571, 225.5       ],
        [286.5       , 100.5       ],
        [287.875     , 188.625     ],
        [290.        , 243.35294118],
        [291.5       ,  78.71428571],
        [290.        , 359.        ],
        [292.2       ,  48.4       ],
        [292.        , 461.5       ],
        [295.5       , 347.        ],
        [296.66666667, 105.11111111],
        [297.        , 172.5       ],
        [301.13333333, 328.        ],
        [301.09090909,  59.36363636],
        [304.23529412, 153.        ],
        [302.57142857, 382.28571429],
        [303.        ,  92.        ],
        [303.875     , 391.875     ],
        [304.08333333, 433.83333333],
        [308.27777778, 450.11111111],
        [307.90909091, 207.63636364],
        [309.125     ,  30.875     ],
        [310.64285714, 241.78571429],
        [310.        , 272.        ],
        [313.4       , 137.9       ],
        [314.        ,  14.5       ],
        [315.        , 258.        ],
        [316.        ,  65.5       ],
        [318.35294118, 350.52941176],
        [318.875     , 225.125     ],
        [319.125     , 309.875     ],
        [319.11111111, 416.66666667],
        [321.        ,  78.5       ],
        [320.71428571, 427.        ],
        [321.5       , 134.5       ],
        [326.16666667, 191.72222222],
        [327.        , 173.5       ],
        [326.35714286, 213.21428571],
        [327.        , 386.5       ],
        [327.        , 231.        ],
        [328.5       ,  35.5       ],
        [329.38461538, 103.38461538],
        [330.42857143, 248.64285714],
        [330.        , 326.5       ],
        [333.        , 155.33333333],
        [333.36363636, 422.54545455],
        [336.66666667,  30.66666667],
        [339.        ,  90.71428571],
        [341.5       ,  45.5       ],
        [342.94444444, 128.77777778],
        [343.5       , 172.        ],
        [344.4       , 385.6       ],
        [344.        , 461.        ],
        [347.21428571, 211.35714286],
        [346.90909091, 343.63636364],
        [347.8       ,  11.6       ],
        [347.4       ,  85.2       ],
        [349.4       , 196.6       ],
        [349.66666667, 276.25      ],
        [351.38461538,  68.61538462],
        [351.71428571, 308.5       ],
        [351.625     , 437.75      ],
        [354.9375    , 411.875     ],
        [355.53333333, 108.86666667],
        [361.6       ,  88.4       ],
        [365.11764706, 157.58823529],
        [365.69230769,  29.30769231],
        [369.        ,  43.5       ],
        [370.73333333,  75.06666667],
        [370.6       , 122.4       ],
        [372.5       , 239.5       ],
        [378.11764706, 206.41176471],
        [379.9       , 181.05      ],
        [381.0625    , 221.875     ],
        [381.58823529,  60.11764706],
        [383.09090909,  28.36363636],
        [387.91666667, 135.66666667],
        [386.54545455, 245.72727273],
        [387.53333333, 430.86666667],
        [388.38461538,  11.38461538],
        [388.5       ,  95.        ],
        [390.5       , 399.66666667],
        [392.18181818, 147.81818182],
        [393.2       , 303.8       ],
        [393.66666667, 447.33333333],
        [396.        ,  29.5       ],
        [398.4       ,  69.6       ],
        [399.        ,  99.8       ],
        [399.625     , 171.25      ],
        [401.625     , 418.75      ],
        [405.125     ,  25.125     ],
        [406.5       , 451.5       ],
        [408.5       , 458.        ],
        [410.875     , 118.125     ],
        [413.5       , 237.5       ],
        [413.5       , 272.71428571],
        [417.875     , 121.125     ],
        [418.        , 179.        ],
        [418.5       , 223.71428571],
        [422.83333333, 143.83333333],
        [424.92307692,  18.15384615],
        [424.8       , 189.        ],
        [424.875     , 206.0625    ],
        [428.13333333,  56.53333333],
        [429.71428571,  33.5       ],
        [428.42857143, 136.28571429],
        [429.63636364, 258.90909091],
        [429.5       , 355.5       ],
        [430.5       , 367.        ],
        [431.        , 157.        ]
    ])
    pixel_size = 0.018 # µm
    diameter = 0.36 # µm
    n_steps = 36
    working_dir = Path("/home/clement/Documents/projects/measures-orestis/2025-10-27/STED-pores-nucleaires/plots")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    diameter_px = as_pxls(diameter, pixel_size)
    results = radial_profiles(image_data, spots_positions, diameter_px, pixel_size, n_steps, working_dir=None)
    export_as_csv(results, working_dir)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # viewer = napari.Viewer()
    # napari.run()