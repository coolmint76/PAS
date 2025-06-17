import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Constants for ppm scale
WATER_FREQ_PPM = 0.0  # Water resonance at 0 ppm
FAT_FREQ_PPM = -3.4   # Fat resonance at -3.4 ppm relative to water

# Create figure with subplots for general shim issues
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('MRI Shim Spectrum Examples: Critical Features (Field-Independent ppm Scale)', 
              fontsize=16, fontweight='bold')

# Define ppm range
ppm = np.linspace(-6, 6, 1000)  # ppm scale

# Function to create Lorentzian peak
def lorentzian(x, x0, gamma, amplitude):
    return amplitude * (gamma**2 / ((x - x0)**2 + gamma**2))

# Function to create Gaussian peak
def gaussian(x, x0, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - x0) / sigma)**2)

# 1. Good Shim Spectrum (top left)
ax1 = axes1[0, 0]
# Convert FWHM from Hz to ppm (assuming ~20 Hz at 3T = 0.16 ppm)
water_good = lorentzian(ppm, 0, 0.16, 1.0)  # Narrow water peak at 0 ppm
fat_good = lorentzian(ppm, -3.4, 0.12, 0.05)  # Small fat peak at -3.4 ppm
spectrum_good = water_good + fat_good + np.random.normal(0, 0.005, len(ppm))

ax1.plot(ppm, spectrum_good, 'b-', linewidth=2)
ax1.set_title('A) Good Shim - Ideal Spectrum', fontsize=12, fontweight='bold')
ax1.set_xlabel('Chemical Shift (ppm)')
ax1.set_ylabel('Signal Intensity')
ax1.grid(True, alpha=0.3)
# Don't invert - negative values on left for Canon convention

# Annotations for good spectrum
ax1.annotate('Narrow water peak\n(FWHM ~0.16 ppm)', xy=(0, 1.0), xytext=(2, 0.8),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold')

# Fix symmetry annotation
ax1.annotate('Symmetric\nLorentzian shape', xy=(0.3, 0.5), xytext=(2.5, 0.4),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5, linestyle='dashed'),
            fontsize=10, color='green')
ax1.annotate('', xy=(-0.3, 0.5), xytext=(-2.5, 0.4),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5, linestyle='dashed'))
ax1.text(0, 0.3, 'Symmetry', ha='center', fontsize=9, color='green')

ax1.text(-3.4, 0.15, 'Fat peak\n(-3.4 ppm)', ha='center', fontsize=9, color='gray')

# Add proper FWHM indicator
half_max = 0.5
fwhm_indices = np.where(np.abs(water_good - half_max) < 0.02)[0]
if len(fwhm_indices) > 1:
    left_idx = fwhm_indices[0]
    right_idx = fwhm_indices[-1]
    # Horizontal line at half maximum
    ax1.hlines(half_max, ppm[left_idx], ppm[right_idx], 
               colors='green', linestyles='solid', linewidth=2)
    # Add arrow pointing to FWHM measurement
    ax1.annotate('FWHM measured here\n(at half maximum)', 
                xy=(ppm[right_idx]/2, half_max), xytext=(1.5, 0.65),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, color='green', fontweight='bold')
    # Add vertical indicators
    ax1.vlines(ppm[left_idx], 0, half_max, colors='green', linestyles='dotted', alpha=0.5)
    ax1.vlines(ppm[right_idx], 0, half_max, colors='green', linestyles='dotted', alpha=0.5)

# 2. Shifted Peak (top right)
ax2 = axes1[0, 1]
# Shift of ~0.3 ppm
water_shifted = lorentzian(ppm, 0.3, 0.12, 1.0)  
fat_shifted = lorentzian(ppm, -3.1, 0.16, 0.08)
spectrum_shifted = water_shifted + fat_shifted + np.random.normal(0, 0.005, len(ppm))

ax2.plot(ppm, spectrum_shifted, 'r-', linewidth=2)
ax2.set_title('B) Frequency Offset Problem', fontsize=12, fontweight='bold')
ax2.set_xlabel('Chemical Shift (ppm)')
ax2.set_ylabel('Signal Intensity')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Expected position')

# Annotations for shifted spectrum
ax2.annotate('Peak shifted by 0.3 ppm\n(Poor center frequency)', 
            xy=(0.3, 1.0), xytext=(2, 0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')
ax2.annotate('Fat peak also shifted', xy=(-3.1, 0.08), xytext=(-1, 0.3),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=9, color='orange')

# 3. Broad Peak with Shoulders (bottom left)
ax3 = axes1[1, 0]
# Create broad peak with shoulders (FWHM ~0.6 ppm)
water_broad = lorentzian(ppm, 0, 0.6, 0.8)  
shoulder1 = gaussian(ppm, -0.4, 0.16, 0.3)  # Left shoulder
shoulder2 = gaussian(ppm, 0.35, 0.2, 0.25)  # Right shoulder
fat_broad = lorentzian(ppm, -3.4, 0.27, 0.15)  # Broader fat peak
spectrum_broad = water_broad + shoulder1 + shoulder2 + fat_broad + np.random.normal(0, 0.005, len(ppm))

ax3.plot(ppm, spectrum_broad, 'purple', linewidth=2)
ax3.set_title('C) Poor Shim - Broad Peak with Shoulders', fontsize=12, fontweight='bold')
ax3.set_xlabel('Chemical Shift (ppm)')
ax3.set_ylabel('Signal Intensity')
ax3.grid(True, alpha=0.3)

# Annotations for broad spectrum
ax3.annotate('Very broad peak\n(FWHM ~0.6 ppm)', xy=(0, 0.8), xytext=(2, 0.6),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')
ax3.annotate('Left shoulder\n(B0 inhomogeneity)', xy=(-0.4, 0.3), xytext=(-2, 0.5),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=9, color='orange')
ax3.annotate('Right shoulder', xy=(0.35, 0.25), xytext=(2.5, 0.3),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=9, color='orange')
ax3.annotate('Wide fat peak', xy=(-3.4, 0.15), xytext=(-5, 0.3),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2),
            fontsize=9, color='purple')

# 4. Multiple Issues (bottom right)
ax4 = axes1[1, 1]
# Create spectrum with multiple problems
water_multi = lorentzian(ppm, 0.12, 0.45, 0.7)  # Shifted and broad
water_split = lorentzian(ppm, -0.15, 0.38, 0.4)  # Split peak
baseline_drift = 0.1 * np.exp(-ppm**2 / 8)  # Broad baseline
fat_multi = lorentzian(ppm, -3.2, 0.6, 0.2)  # Very broad fat
sidebands = 0.1 * np.cos(2 * np.pi * ppm / 1.5)  # Oscillating sidebands
spectrum_multi = water_multi + water_split + baseline_drift + fat_multi + \
                 0.05 * sidebands + np.random.normal(0, 0.01, len(ppm))

ax4.plot(ppm, spectrum_multi, 'darkred', linewidth=2)
ax4.set_title('D) Severe Shim Problems - Multiple Issues', fontsize=12, fontweight='bold')
ax4.set_xlabel('Chemical Shift (ppm)')
ax4.set_ylabel('Signal Intensity')
ax4.grid(True, alpha=0.3)

# Annotations for multiple issues
ax4.annotate('Split/distorted peak', xy=(0, 0.65), xytext=(2, 0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')
ax4.annotate('Broad baseline\n(severe inhomogeneity)', xy=(1.5, 0.08), xytext=(3.5, 0.3),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=9, color='orange')
ax4.annotate('Very wide fat', xy=(-3.2, 0.2), xytext=(-5, 0.4),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2),
            fontsize=9, color='purple')

# Add a rectangle to highlight baseline issues
baseline_rect = Rectangle((-6, -0.05), 12, 0.15, 
                         linewidth=2, edgecolor='orange', 
                         facecolor='orange', alpha=0.2)
ax4.add_patch(baseline_rect)

# Add common information box
textstr = 'Key Quality Indicators (Field-Independent):\n' \
          '• Good shim: FWHM < 0.2 ppm\n' \
          '• Peak should be at 0 ppm (water)\n' \
          '• Fat peak at -3.4 ppm (left side)\n' \
          '• Symmetric Lorentzian shape\n' \
          '• Minimal shoulders or sidebands'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig1.text(0.98, 0.02, textstr, transform=fig1.transFigure, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.05)
plt.show()

# Create second figure for anatomy-specific examples
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Shim Spectra by Anatomical Region: Expected Patterns', 
              fontsize=16, fontweight='bold')

# Brain - Excellent shimming possible
ax_brain = axes2[0, 0]
brain_water = lorentzian(ppm, 0, 0.12, 1.0)  # Very narrow
brain_fat = lorentzian(ppm, -3.4, 0.1, 0.03)  # Minimal scalp fat
brain_spectrum = brain_water + brain_fat + np.random.normal(0, 0.003, len(ppm))

ax_brain.plot(ppm, brain_spectrum, 'darkgreen', linewidth=2)
ax_brain.set_title('Brain (Central)', fontsize=12, fontweight='bold', color='darkgreen')
ax_brain.text(0.5, 0.95, 'Excellent shimming\nFWHM < 0.15 ppm', 
             transform=ax_brain.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax_brain.set_xlabel('Chemical Shift (ppm)')
ax_brain.set_ylabel('Signal Intensity')
ax_brain.grid(True, alpha=0.3)

# Knee - Moderate difficulty
ax_knee = axes2[0, 1]
knee_water = lorentzian(ppm, 0, 0.35, 0.9)  # Broader due to susceptibility
knee_fat = lorentzian(ppm, -3.4, 0.25, 0.4)  # Significant subcutaneous fat
knee_synovial = gaussian(ppm, 0.2, 0.15, 0.2)  # Synovial fluid component
knee_spectrum = knee_water + knee_fat + knee_synovial + np.random.normal(0, 0.005, len(ppm))

ax_knee.plot(ppm, knee_spectrum, 'darkblue', linewidth=2)
ax_knee.set_title('Knee Joint', fontsize=12, fontweight='bold', color='darkblue')
ax_knee.text(0.5, 0.95, 'Moderate difficulty\nBone/tissue interfaces\nFWHM ~0.3-0.4 ppm', 
             transform=ax_knee.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax_knee.set_xlabel('Chemical Shift (ppm)')
ax_knee.set_ylabel('Signal Intensity')
ax_knee.grid(True, alpha=0.3)

# Abdomen - Challenging
ax_abdomen = axes2[0, 2]
abd_water = lorentzian(ppm, 0.1, 0.5, 0.8)  # Broad and shifted
abd_fat = lorentzian(ppm, -3.3, 0.4, 0.6)  # Very prominent
abd_shoulder = gaussian(ppm, -0.5, 0.3, 0.3)  # Air/tissue interfaces
abd_spectrum = abd_water + abd_fat + abd_shoulder + 0.05 * np.sin(5 * ppm) + np.random.normal(0, 0.008, len(ppm))

ax_abdomen.plot(ppm, abd_spectrum, 'darkorange', linewidth=2)
ax_abdomen.set_title('Abdomen', fontsize=12, fontweight='bold', color='darkorange')
ax_abdomen.text(0.5, 0.95, 'Challenging\nAir/tissue interfaces\nFWHM > 0.5 ppm', 
             transform=ax_abdomen.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax_abdomen.set_xlabel('Chemical Shift (ppm)')
ax_abdomen.set_ylabel('Signal Intensity')
ax_abdomen.grid(True, alpha=0.3)

# Obese Patient - Large FOV (NEW)
ax_obese = axes2[1, 0]
obese_water = lorentzian(ppm, 0.05, 0.45, 0.6)  # Broad water peak
obese_fat = lorentzian(ppm, -3.35, 0.8, 1.0)  # VERY broad and dominant fat peak
obese_spectrum = obese_water + obese_fat + 0.02 * np.cos(3 * ppm) + np.random.normal(0, 0.01, len(ppm))

ax_obese.plot(ppm, obese_spectrum, 'brown', linewidth=2)
ax_obese.set_title('Large FOV - Obese Patient', fontsize=12, fontweight='bold', color='brown')
ax_obese.text(0.5, 0.95, 'Extreme fat content\nVery broad fat peak\nFWHM > 0.8 ppm', 
             transform=ax_obese.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax_obese.set_xlabel('Chemical Shift (ppm)')
ax_obese.set_ylabel('Signal Intensity')
ax_obese.grid(True, alpha=0.3)
# Add annotation for dominant fat
ax_obese.annotate('Fat peak dominates\nwater signal', xy=(-3.35, 1.0), xytext=(-5.5, 0.8),
                 arrowprops=dict(arrowstyle='->', color='brown', lw=2),
                 fontsize=9, color='brown', fontweight='bold')

# Cardiac - Motion Effects (NEW)
ax_cardiac = axes2[1, 1]
# Create time-varying spectrum to simulate motion
cardiac_base = lorentzian(ppm, 0, 0.3, 0.8)
# Add motion-induced variations
motion_component1 = 0.3 * gaussian(ppm, -0.2, 0.2, 1) * np.sin(2 * np.pi * 0.5)
motion_component2 = 0.2 * gaussian(ppm, 0.3, 0.25, 1) * np.cos(2 * np.pi * 0.8)
# Multiple fat peaks from different regions
cardiac_fat1 = lorentzian(ppm, -3.4, 0.3, 0.4)  # Epicardial fat
cardiac_fat2 = gaussian(ppm, -3.2, 0.2, 0.2)  # Motion-shifted fat
# Blood flow artifact
flow_artifact = 0.15 * np.sin(10 * ppm) * np.exp(-np.abs(ppm))
cardiac_spectrum = cardiac_base + motion_component1 + motion_component2 + \
                  cardiac_fat1 + cardiac_fat2 + flow_artifact + np.random.normal(0, 0.02, len(ppm))

ax_cardiac.plot(ppm, cardiac_spectrum, 'darkred', linewidth=2)
ax_cardiac.set_title('Cardiac (Motion Effects)', fontsize=12, fontweight='bold', color='darkred')
ax_cardiac.text(0.5, 0.95, 'Heartbeat + breathing\nMultiple components\nFlow artifacts', 
               transform=ax_cardiac.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))
ax_cardiac.set_xlabel('Chemical Shift (ppm)')
ax_cardiac.set_ylabel('Signal Intensity')
ax_cardiac.grid(True, alpha=0.3)
# Add motion annotations
ax_cardiac.annotate('Motion-induced\nasymmetry', xy=(0.3, 0.6), xytext=(2, 0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=9, color='red')
ax_cardiac.annotate('Flow artifacts', xy=(1, 0.15), xytext=(3, 0.3),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                   fontsize=9, color='orange')

# Breast - Fat/water separation important
ax_breast = axes2[1, 2]
breast_water = lorentzian(ppm, 0, 0.2, 0.7)  # Good shim possible
breast_fat = lorentzian(ppm, -3.4, 0.18, 0.8)  # Prominent fat peak
breast_spectrum = breast_water + breast_fat + np.random.normal(0, 0.004, len(ppm))

ax_breast.plot(ppm, breast_spectrum, 'magenta', linewidth=2)
ax_breast.set_title('Breast', fontsize=12, fontweight='bold', color='magenta')
ax_breast.text(0.5, 0.95, 'Fat suppression critical\nGood separation needed\nFWHM ~0.2 ppm', 
             transform=ax_breast.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lavenderblush', alpha=0.7))
ax_breast.set_xlabel('Chemical Shift (ppm)')
ax_breast.set_ylabel('Signal Intensity')
ax_breast.grid(True, alpha=0.3)

# Add field strength conversion box
conversion_text = 'Field Strength Conversions:\n' \
                 '• 1.5T: 1 ppm ≈ 64 Hz\n' \
                 '• 3.0T: 1 ppm ≈ 128 Hz\n' \
                 '• 7.0T: 1 ppm ≈ 298 Hz\n\n' \
                 'Target FWHM:\n' \
                 '• Spectroscopy: < 0.1 ppm\n' \
                 '• Clinical imaging: < 0.3 ppm'

props2 = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
fig2.text(0.98, 0.02, conversion_text, transform=fig2.transFigure, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props2)

plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.08)
plt.show()

# Create a detailed FWHM measurement demonstration
fig3, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_title('Detailed FWHM Measurement Guide', fontsize=14, fontweight='bold')

# Create a single peak for demonstration
demo_freq = np.linspace(-2, 2, 500)
demo_peak = lorentzian(demo_freq, 0, 0.2, 1.0)
ax.plot(demo_freq, demo_peak, 'b-', linewidth=3, label='Water peak')

# Mark key points
ax.hlines(1.0, -0.5, 0.5, colors='green', linestyles='dotted', label='Peak maximum (100%)')
ax.hlines(0.5, -0.5, 0.5, colors='red', linestyles='solid', linewidth=3, label='Half maximum (50%)')

# Find and mark FWHM points
half_max_indices = np.where(np.abs(demo_peak - 0.5) < 0.01)[0]
if len(half_max_indices) > 1:
    left_point = demo_freq[half_max_indices[0]]
    right_point = demo_freq[half_max_indices[-1]]
    
    # Vertical lines at FWHM boundaries
    ax.vlines(left_point, 0, 0.5, colors='red', linestyles='dashed', linewidth=2)
    ax.vlines(right_point, 0, 0.5, colors='red', linestyles='dashed', linewidth=2)
    
    # FWHM measurement arrow
    arrow = FancyArrowPatch((left_point, 0.5), (right_point, 0.5),
                           connectionstyle="arc3,rad=0", color='red', 
                           arrowstyle='<->', mutation_scale=20, linewidth=3)
    ax.add_patch(arrow)
    ax.text(0, 0.52, f'FWHM = {right_point - left_point:.2f} ppm', 
            ha='center', fontsize=14, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add measurement annotations
    ax.annotate('Left FWHM\nboundary', xy=(left_point, 0.5), xytext=(-1.5, 0.7),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red')
    ax.annotate('Right FWHM\nboundary', xy=(right_point, 0.5), xytext=(1.5, 0.7),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red')

# Add grid and labels
ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
ax.set_ylabel('Signal Intensity', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-0.1, 1.2)

# Add step-by-step instructions
instructions = 'FWHM Measurement Steps:\n' \
               '1. Find peak maximum (100% intensity)\n' \
               '2. Calculate half maximum (50% intensity)\n' \
               '3. Find where spectrum crosses 50% line\n' \
               '4. Measure width between crossing points\n' \
               '5. Report in ppm for field independence'

ax.text(0.02, 0.02, instructions, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

# Create a fourth figure showing typical shimming challenges
fig4, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title('Common Shimming Challenges by Anatomy', fontsize=14, fontweight='bold')

# Create a complexity chart
anatomies = ['Brain\n(central)', 'Extremities', 'Spine', 'Breast', 'Knee/Joints', 
             'Heart', 'Liver', 'Abdomen', 'Lung/Chest', 'Near Metal']
difficulty = [1, 2, 3, 3, 4, 5, 6, 7, 8, 9]
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(anatomies)))

bars = ax.barh(anatomies, difficulty, color=colors, edgecolor='black', linewidth=1.5)

# Add difficulty labels
for i, (bar, diff) in enumerate(zip(bars, difficulty)):
    if diff <= 3:
        label = 'Easy'
        detail = 'FWHM < 0.2 ppm'
    elif diff <= 6:
        label = 'Moderate'
        detail = 'FWHM 0.2-0.5 ppm'
    else:
        label = 'Challenging'
        detail = 'FWHM > 0.5 ppm'
    
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{label}\n({detail})', fontsize=9, va='center')

ax.set_xlabel('Shimming Difficulty Score', fontsize=12)
ax.set_xlim(0, 11)
ax.grid(True, axis='x', alpha=0.3)
ax.set_axisbelow(True)

# Add notes
notes_text = 'Factors affecting shimming difficulty:\n' \
             '• Tissue interfaces (air/tissue, bone/soft tissue)\n' \
             '• Magnetic susceptibility differences\n' \
             '• Motion (breathing, cardiac, flow)\n' \
             '• Field of view size\n' \
             '• Distance from isocenter'

ax.text(0.98, 0.02, notes_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()
