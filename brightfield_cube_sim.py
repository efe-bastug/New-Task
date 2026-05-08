"""
Brightfield Microscopy Simulation: 2x2x2 µm Cube at Multiple Focus Depths
Physics from: doi:10.1038/s44303-024-00024-4  (Figure 1)

Key equations:
  t(x,y,z) = A(x,y,z) · exp(i·phi(x,y,z))   [complex transmission — object modifies phase]
  I(x,y)   = |E_camera(x,y)|^2               [intensity measured by sensor — phase info lost]
  I_noisy  = Poisson(I · eta) + N(0, sigma_r) [camera with realistic shot + read noise]

Pipeline:
  Plane wave --> BPM through cube --> propagate to focus depth
  --> Objective (ff_lens, NA aperture) --> Tube lens (ff_lens) --> sensor

I learned that a pure phase object like this glass cube should appear nearly invisible
at z=0 (in focus) because the camera cannot directly detect phase differences.
The contrast only appears when you defocus — that was a key insight from the paper.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chromatix.functional as cx

print("=" * 60)
print("Brightfield Cube Simulation (Chromatix + JAX)")
print("=" * 60)

# 1. PHYSICAL PARAMETERS
# I collected all the physical constants here at the top so I don't have magic numbers
# scattered through the code. I had a bug earlier where I typed NA=0.08 instead of 0.80
# which made the pupil cut off almost all spatial frequencies — the image looked like
# a blurry blob. Always double-check the decimal point on the NA value!
WAVELENGTH  = 0.532   # µm - green LED / laser (standard wavelength for these experiments)
NA          = 0.80    # numerical aperture of the dry objective
N_MEDIUM    = 1.00    # refractive index of air (surrounding medium)
N_CUBE      = 1.50    # refractive index of the glass cube
DN_CUBE     = N_CUBE - N_MEDIUM   # refractive index contrast = 0.50 (what creates phase shift)

# simulation grid — how many pixels and how many µm per pixel
# I found that using too few pixels (e.g. 128x128) caused strong aliasing rings in the image.
# 256x256 with DX=0.05 µm gives enough resolution to resolve the 0.532 µm wavelength.
SHAPE       = (256, 256)  # pixel grid dimensions
DX          = 0.05        # µm - physical size of each pixel in the sample plane
FOV         = SHAPE[0] * DX   # total field of view = 12.8 µm

# cube geometry — the cube is 2 µm on each side, centered in the FOV
CUBE_XY     = 2.0    # µm - width and height of the cube in the XY plane
CUBE_Z      = 2.0    # µm - depth of the cube along the optical axis
N_SLICES    = 20     # I split the cube into 20 thin slices for the BPM algorithm
DZ_SLICE    = CUBE_Z / N_SLICES   # each slice is 0.1 µm thick (thinner = more accurate BPM)

# imaging optics — these values correspond to a standard 100x / 0.8 NA objective
F_OBJ       = 2000.0     # µm - objective focal length
F_TUBE      = 160_000.0  # µm - tube lens focal length (standard 160 mm)
PAD_WIDTH   = 64         # zero-padding width to reduce FFT wraparound artifacts at the edges

# CAMERA / SENSOR NOISE MODEL (typical sCMOS sensor values)
# The raw simulation output |E|^2 is dimensionless (normalized intensity).
# I scaled it to photon counts using PHOTON_SCALE so the noise levels are physically realistic.
# Two noise sources I added:
#   1. Poisson shot noise  — photon arrival is random, variance = mean (via basic_sensor)
#   2. Gaussian read noise — electronic noise added by the sensor circuit (via random.normal)
# I originally added read noise before shot noise, which gave wrong statistics,
# so I fixed the order: shot noise first, then add read noise on top.
PHOTON_SCALE    = 5_000    # photons per pixel at peak intensity (typical bright-field value)
READ_NOISE_STD  = 2.0      # read noise standard deviation in electrons (sCMOS: usually 1-3 e-)
SEED            = 42       # fixed random seed so I get the same noise pattern every run

# focus depths to simulate (µm from the cube centre, positive = above the cube)
Z_DEPTHS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

print(f"\n  Wavelength  : {WAVELENGTH} µm    NA      : {NA}")
print(f"  n_medium    : {N_MEDIUM}         n_cube  : {N_CUBE}  (dn={DN_CUBE})")
print(f"  Grid        : {SHAPE},  dx={DX} µm  → FOV={FOV:.1f} µm")
print(f"  Cube        : {CUBE_XY}x{CUBE_XY}x{CUBE_Z} µm  ({N_SLICES} slices × {DZ_SLICE} µm)")
print(f"  Camera      : {PHOTON_SCALE} max photons/px | "
      f"read noise σ={READ_NOISE_STD} e⁻ | seed={SEED}")
print(f"  Focus depths: {Z_DEPTHS} µm")

# 2. BUILD THE CUBE SAMPLE (absorption_stack, dn_stack)
# The multislice BPM function expects two 3D arrays of shape (N_SLICES, H, W):
#   dn_stack         — refractive index difference at each voxel
#   absorption_stack — imaginary part of n (0 here because the glass cube is transparent)
# I initially passed a 2D mask directly and got a shape error,
# so I fixed it by stacking N_SLICES copies to make the required 3D array.
cy  = SHAPE[0] // 2   # row index of the image centre
cxc = SHAPE[1] // 2   # column index of the image centre
half = int(round(CUBE_XY / DX / 2))   # half-width of the cube in pixels

ys = jnp.arange(SHAPE[0])
xs = jnp.arange(SHAPE[1])
YY, XX = jnp.meshgrid(ys, xs, indexing="ij")
# pixels inside the cube get 1, all others get 0 — this is the binary occupancy mask
xy_mask = ((jnp.abs(YY - cy) < half) & (jnp.abs(XX - cxc) < half)).astype(jnp.float32)

# Every z-slice of the cube has the same XY footprint (it's a box, not a sphere),
# so I just repeated the same mask N_SLICES times and scaled by the refractive index contrast.
dn_stack         = jnp.stack([xy_mask * DN_CUBE for _ in range(N_SLICES)])  # shape: (d, h, w)
absorption_stack = jnp.zeros_like(dn_stack)   # glass is transparent — no absorption term

max_phase = 2 * jnp.pi * DN_CUBE * CUBE_Z / WAVELENGTH
print(f"\n  Cube mask   : {int(xy_mask.sum())} lit px/slice  "
      f"({2*half}x{2*half} px = {2*half*DX:.1f}x{2*half*DX:.1f} µm)")
print(f"  Max phase   : {float(max_phase):.2f} rad = "
      f"{float(max_phase / (2*jnp.pi)):.2f} wavelengths")

# 3. PLANE WAVE ILLUMINATION
# I used a plane wave as the illumination — this simulates a perfectly collimated beam,
# which is the standard assumption for brightfield microscopy with Köhler illumination.
# Setting power=1.0 normalizes the field amplitude to 1, making the output intensity
# directly interpretable as a fraction of the incident intensity.
field_in = cx.plane_wave(shape=SHAPE, dx=DX, spectrum=WAVELENGTH, power=1.0)
print(f"\n  Input field  |u| max = {float(jnp.max(jnp.abs(field_in.u))):.4f}")

# 4. PROPAGATED LIGHT THROUGH THE CUBE (BPM / multislice method)
# The beam-propagation method (BPM) splits the cube into thin slices and at each slice:
#   Step 1 — apply a phase screen: phi(x,y) = (2*pi/lambda) * dn * dz
#   Step 2 — free-space propagate to the next slice using angular spectrum method
# I use the cube centre (z=0) as the reference plane for defocus calculations later.
# I initially forgot pad_width and got strong ringing from FFT edge wrap-around,
# so I added PAD_WIDTH=64 to zero-pad the field before each propagation step.
print("\n  Running BPM through 2×2×2 µm cube …", flush=True)
field_exit = cx.multislice_thick_sample(
    field               = field_in,
    absorption_stack    = absorption_stack,
    dn_stack            = dn_stack,
    n                   = N_MEDIUM,
    thickness_per_slice = DZ_SLICE,
    pad_width           = PAD_WIDTH,
)
print(f"  Exit field (cube centre)  |u| max = {float(jnp.max(jnp.abs(field_exit.u))):.4f}")

# 5. SENSOR NOISE MODEL
# The ideal simulation gives a clean float32 intensity array.
# To make it look like a real camera image, I need to add two types of noise:
#
#   Step 1 — scale to photon counts:
#             photons[i,j] = I[i,j] / I_max * PHOTON_SCALE
#             (I had a divide-by-zero error here when I_max was 0, so I added 1e-30)
#   Step 2 — Poisson shot noise via Chromatix basic_sensor:
#             each pixel's photon count is randomly drawn from Poisson(expected_photons)
#   Step 3 — Gaussian read noise added on top:
#             signal = poisson_counts + N(0, READ_NOISE_STD)
#
# I use random.split to get two independent keys so the two noise sources are uncorrelated.
def apply_sensor_noise(intensity_2d: jnp.ndarray,
                       key: jax.Array,
                       photon_scale: float = PHOTON_SCALE,
                       read_noise: float = READ_NOISE_STD) -> jnp.ndarray:
    """
    Converts a noiseless intensity image to a realistic camera measurement.
    Applies Poisson shot noise first, then adds Gaussian read noise.
    Returns the result in photo-electron units (float32).
    """
    key_shot, key_read = random.split(key)

    # Normalize so the brightest pixel corresponds to exactly PHOTON_SCALE photons.
    # The +1e-30 prevents division by zero if the image happens to be all zeros.
    i_max = jnp.max(intensity_2d) + 1e-30
    photons = intensity_2d / i_max * photon_scale   # expected photon count per pixel

    # Poisson shot noise — models the random nature of photon arrival.
    # basic_sensor from Chromatix handles this correctly (it expects photon-count units).
    shot = cx.basic_sensor(
        photons,
        shot_noise_mode = "poisson",
        noise_key       = key_shot,
        input_spacing   = DX,
    )

    # Gaussian read noise — models electronic noise in the sensor readout circuit.
    # I multiply by read_noise to scale the standard normal to the correct std deviation.
    read = read_noise * random.normal(key_read, shot.shape)
    noisy = shot + read   # combined measured signal

    return noisy   # units: photo-electrons


# 6. IMAGING PIPELINE (clean + noisy, for each focus depth)
# This function runs the complete 4-f optical system for a given defocus position.
# I refactored this into a function after realizing I was copy-pasting the same
# five lines for every z depth — the loop below calls it cleanly instead.
def image_at_depth(field_ref, z_f):
    """
    Full 4-f brightfield imaging pipeline for a focus offset of z_f [µm].
    z_f = 0 means the camera is focused exactly at the exit plane of the sample.
    Returns a 2D ideal intensity image as a float32 array of shape SHAPE.
    """
    # (a) Free-space propagation to the chosen focal plane.
    # I skip this step when z_f is essentially zero to avoid numerical noise
    # from propagating by a distance of 0 (which should be a no-op but isn't always).
    if abs(z_f) > 1e-9:
        field = cx.transfer_propagate(
            field     = field_ref,
            z         = float(z_f),
            n         = N_MEDIUM,
            pad_width = PAD_WIDTH,
            mode      = "same",
        )
    else:
        field = field_ref

    # (b) Objective lens — ff_lens with NA keyword applies the pupil aperture filter.
    # High spatial frequencies beyond NA/lambda are physically blocked by the lens aperture.
    field = cx.ff_lens(field, f=F_OBJ, n=N_MEDIUM, NA=NA)

    # (c) Tube lens — forms the final image on the camera sensor.
    # No NA keyword here because the tube lens does not add an aperture.
    field = cx.ff_lens(field, f=F_TUBE, n=N_MEDIUM)

    # (d) Camera detects intensity = |E|^2, not the complex field.
    # jnp.squeeze removes the batch/channel dimensions added by Chromatix.
    return jnp.squeeze(field.intensity)


print("\n  Imaging at each focus depth …")
master_key  = random.PRNGKey(SEED)
images_clean = []
images_noisy = []

for k, z_f in enumerate(Z_DEPTHS):
    # clean (noiseless) image
    img_clean = image_at_depth(field_exit, z_f)

    # noisy image (independent noise per plane)
    subkey = random.fold_in(master_key, k)
    img_noisy = apply_sensor_noise(img_clean, subkey)

    images_clean.append(np.array(img_clean))
    images_noisy.append(np.array(img_noisy))

    snr = float(jnp.mean(img_noisy) / (jnp.std(img_noisy) + 1e-30))
    print(f"    z = {z_f:+.1f} µm  |"
          f"  clean max={float(jnp.max(img_clean)):.2e}"
          f"  |  noisy max={float(jnp.max(img_noisy)):.1f} ph"
          f"  SNR≈{snr:.1f}")

# I extracted the amplitude, phase, and intensity from the BPM exit field for Figure 4.
# jnp.angle gives the phase angle in radians (range: -pi to pi).
# I convert to numpy here because matplotlib requires numpy arrays, not JAX arrays.
exit_intensity = np.array(jnp.squeeze(field_exit.intensity))
exit_phase     = np.array(jnp.angle(jnp.squeeze(field_exit.u)))
exit_amplitude = np.array(jnp.abs(jnp.squeeze(field_exit.u)))

# 7. NORMALISATION HELPERS
# I normalize all focus-depth images together using a single shared min/max value.
# If I normalized each image independently, the brightness scale would shift between
# depths and the focus sweep figure would look wrong (different z-planes not comparable).
def norm_global(imgs):
    """Normalizes a list of 2D arrays to [0, 1] using a shared global min and max.
    This keeps the brightness scale consistent across all panels in the figure."""
    g_min = min(i.min() for i in imgs)
    g_max = max(i.max() for i in imgs)
    return [(i - g_min) / (g_max - g_min + 1e-30) for i in imgs]

imgs_clean_n = norm_global(images_clean)
imgs_noisy_n = norm_global(images_noisy)

half_cube    = CUBE_XY / 2

def add_cube_rect(ax):
    # I overlaid a dashed lime green rectangle to mark the cube's physical position
    # in every panel. Without this it was hard to tell whether the bright/dark rings
    # were actually centered on the cube or just FFT artifacts.
    rect = plt.Rectangle(
        (-half_cube, -half_cube), CUBE_XY, CUBE_XY,
        linewidth=1.2, edgecolor="lime", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

# 8. FIGURE 1 – focus depth sweep (clean images)
n_p = len(Z_DEPTHS)
fig1, axes1 = plt.subplots(1, n_p, figsize=(3.2 * n_p, 4.2))
fig1.suptitle(
    "Brightfield — 2×2×2 µm glass cube | clean (noiseless) images\n"
    r"$I(x,y)=|E_\mathrm{camera}(x,y)|^2$"
    f"  (λ={WAVELENGTH} µm, NA={NA}, n_cube={N_CUBE})",
    fontsize=11, fontweight="bold",
)
z_labels = {-1.0: "cube bottom", 0.0: "cube centre", 1.0: "cube top"}
for k, (z_f, img) in enumerate(zip(Z_DEPTHS, imgs_clean_n)):
    ax = axes1[k]
    im1 = ax.imshow(img, cmap="gray", origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
    title = f"z = {z_f:+.1f} µm"
    if z_f in z_labels:
        title += f"\n({z_labels[z_f]})"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (µm)", fontsize=8)
    if k == 0:
        ax.set_ylabel("y (µm)", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7)
    add_cube_rect(ax)
plt.colorbar(im1, ax=axes1[-1], label="Norm. intensity", shrink=0.8)
plt.tight_layout()
out1 = "brightfield_cube_focus_depths.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure 1 saved → {out1}")

# 9. FIGURE 2 – focus depth sweep (noisy images)
fig2, axes2 = plt.subplots(1, n_p, figsize=(3.2 * n_p, 4.2))
fig2.suptitle(
    f"Brightfield — 2×2×2 µm glass cube | camera noise model\n"
    f"Poisson shot noise ({PHOTON_SCALE} max ph/px)  +  "
    f"Gaussian read noise (σ={READ_NOISE_STD} e⁻)",
    fontsize=11, fontweight="bold",
)
for k, (z_f, img) in enumerate(zip(Z_DEPTHS, imgs_noisy_n)):
    ax = axes2[k]
    im2 = ax.imshow(img, cmap="gray", origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
    title = f"z = {z_f:+.1f} µm"
    if z_f in z_labels:
        title += f"\n({z_labels[z_f]})"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (µm)", fontsize=8)
    if k == 0:
        ax.set_ylabel("y (µm)", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7)
    add_cube_rect(ax)
plt.colorbar(im2, ax=axes2[-1], label="Norm. intensity", shrink=0.8)
plt.tight_layout()
out2 = "brightfield_cube_noisy_depths.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 2 saved → {out2}")

# 10. FIGURE 3 – noise comparison at cube centre (z=0)
#     Row 1: Clean   Row 2: Shot noise only   Row 3: Shot + Read
z0_idx    = Z_DEPTHS.index(0.0)
img_ideal = images_clean[z0_idx]
key_cmp   = random.PRNGKey(SEED + 99)

# I compute shot noise alone first so I can show the two stages of degradation separately.
# This makes it easier to see how much each noise type contributes to the final image.
key_s, key_r2 = random.split(key_cmp)
i_max_cmp = img_ideal.max() + 1e-30
photons_cmp = img_ideal / i_max_cmp * PHOTON_SCALE
img_shot = np.array(cx.basic_sensor(
    jnp.array(photons_cmp), shot_noise_mode="poisson",
    noise_key=key_s, input_spacing=DX))
img_shot_read = np.array(img_shot + READ_NOISE_STD * np.random.RandomState(1).randn(*img_shot.shape))

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4.5))
fig3.suptitle(
    "Camera noise decomposition at z = 0 (cube centre)\n"
    r"$I_\mathrm{noisy} = \mathrm{Poisson}(I \cdot \eta) + \mathcal{N}(0,\,\sigma_r)$",
    fontsize=12, fontweight="bold",
)

cmp_panels = [
    (img_ideal / img_ideal.max(),            "Clean  |E|²  (ideal)"),
    (img_shot  / (img_shot.max() + 1e-9),    f"+ Poisson shot noise\n({PHOTON_SCALE} max photons/px)"),
    (img_shot_read / (img_shot_read.max() + 1e-9),
     f"+ Gaussian read noise\n(σ_r = {READ_NOISE_STD} e⁻)"),
]
for ax, (data, title) in zip(axes3, cmp_panels):
    im3 = ax.imshow(data, cmap="gray", origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    plt.colorbar(im3, ax=ax, shrink=0.85)
    add_cube_rect(ax)
fig3.tight_layout()
out3 = "brightfield_cube_noise_comparison.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 3 saved → {out3}")

# 11. FIGURE 4 – BPM exit field  t(x,y,z) = A·exp(i·phi)
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))
fig4.suptitle(
    r"BPM exit field at cube centre  —  $t(x,y,z)=A(x,y,z)\cdot e^{i\phi(x,y,z)}$",
    fontsize=12, fontweight="bold",
)
panels4 = [
    (exit_amplitude, "hot",    "Amplitude  |u|",             "Amplitude"),
    (exit_phase,     "RdBu",   "Phase  φ  [rad]",            "Phase (rad)"),
    (exit_intensity, "inferno", "Intensity  |u|²  (no lens)", "Intensity"),
]
for ax, (data, cmap, title, cblabel) in zip(axes4, panels4):
    vkw = {"vmin": -np.pi, "vmax": np.pi} if "Phase" in title else {}
    im4 = ax.imshow(data, cmap=cmap, origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], **vkw)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    plt.colorbar(im4, ax=ax, label=cblabel, shrink=0.85)
    add_cube_rect(ax)
fig4.tight_layout()
out4 = "brightfield_cube_exit_field.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 4 saved → {out4}")

# 12. FIGURE 5 – 2x7 grid: clean (top) vs noisy (bottom)
fig5, axes5 = plt.subplots(2, n_p, figsize=(3.2 * n_p, 7))
fig5.suptitle(
    "Brightfield 2×2×2 µm cube — clean vs noisy sensor\n"
    f"(Poisson {PHOTON_SCALE} ph/px  +  Gaussian read σ={READ_NOISE_STD} e⁻)",
    fontsize=12, fontweight="bold",
)
row_labels = ["Clean", "Noisy"]
for row, (imgs_n, row_lbl) in enumerate(zip([imgs_clean_n, imgs_noisy_n], row_labels)):
    for k, (z_f, img) in enumerate(zip(Z_DEPTHS, imgs_n)):
        ax = axes5[row, k]
        ax.imshow(img, cmap="gray", origin="lower",
                  extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
        if row == 0:
            title = f"z = {z_f:+.1f} µm"
            if z_f in z_labels:
                title += f"\n({z_labels[z_f]})"
            ax.set_title(title, fontsize=9)
        if k == 0:
            ax.set_ylabel(f"{row_lbl}\ny (µm)", fontsize=8)
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel("x (µm)", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=6)
        add_cube_rect(ax)
plt.tight_layout()
out5 = "brightfield_cube_clean_vs_noisy.png"
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 5 saved → {out5}")

# SUMMARY
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Phase shift  : {float(max_phase):.2f} rad = {float(max_phase/(2*jnp.pi)):.2f} λ")
print(f"  Magnification: {int(F_TUBE/F_OBJ)}x  →  image pixel = {DX * F_TUBE/F_OBJ:.1f} µm")
print(f"  Noise model  : Poisson({PHOTON_SCALE} ph/px) + Gaussian(σ={READ_NOISE_STD} e⁻)")
print(f"  Outputs      : {out1}, {out2}, {out3}, {out4}, {out5}")
print("Done.")
