"""
Brightfield Microscopy Simulation: Spherocylindrical Bacterial Cell
Cell geometry: 1 µm diameter, 4 µm total length (stadium cross-section per z-slice)
Physics from: doi:10.1038/s44303-024-00024-4  (Figure 1)

Key equations:
  t(x,y,z) = A(x,y,z) · exp(i·phi(x,y,z))   [complex transmission]
  I(x,y)   = |E_camera(x,y)|^2               [intensity measured by sensor]
  I_noisy  = Poisson(I · eta) + N(0, sigma_r) [camera with noise]

Pipeline:
  Plane wave --> BPM through cell --> propagate to focus depth
  --> Objective (ff_lens, NA aperture) --> Tube lens (ff_lens) --> sensor

Spherocylinder geometry (long axis along X, optical axis along Z):
  The cell looks like a cylinder with two hemispherical end caps (like a capsule).
  Cylinder body: |y| ≤ r,     |x| ≤ L_half      (r = 0.5 µm, L_half = 1.5 µm)
  Left  cap:     (x+L_half)² + y² ≤ r²
  Right cap:     (x-L_half)² + y² ≤ r²
  At BPM depth z_c, I shrink the cross-section using r_eff = sqrt(r² - z_c²)
  so the caps taper correctly at the top and bottom of the cell.
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
print("Brightfield Spherocylinder Cell Simulation (Chromatix + JAX)")
print("=" * 60)

# 1. PHYSICAL PARAMETERS
# I set up the same optical values as in the cube simulation so I can compare them later.
# For the cell refractive index I used 1.38, which is a typical value for bacterial cytoplasm.
WAVELENGTH  = 0.532   # µm - green LED illumination
NA          = 0.80    # numerical aperture of the dry objective
N_MEDIUM    = 1.00    # surrounding medium is air
N_CELL      = 1.38    # refractive index of bacterial cytoplasm (literature value)
DN_CELL     = N_CELL - N_MEDIUM   # refractive index contrast = 0.38

# simulation grid - same resolution as the cube sim so images are directly comparable
SHAPE       = (256, 256)  # number of pixels in x and y
DX          = 0.05        # µm - physical size of each pixel
FOV         = SHAPE[0] * DX   # total field of view = 12.8 µm

# spherocylinder geometry
# The cell is a cylinder (radius 0.5 µm, length 3 µm) capped with two hemispheres.
# Total length = cylindrical body + two half-sphere radii = 3 + 0.5 + 0.5 = 4 µm.
CELL_RADIUS    = 0.5    # µm  (so diameter = 1 µm)
CELL_LENGTH    = 4.0    # µm  total end-to-end length
CYL_HALF       = (CELL_LENGTH - 2 * CELL_RADIUS) / 2   # = 1.5 µm, half-length of just the cylinder part
CELL_Z         = 2 * CELL_RADIUS                        # = 1.0 µm  axial depth of the cell
N_SLICES       = 20     # I divided the cell into 20 thin slices along z for BPM
DZ_SLICE       = CELL_Z / N_SLICES   # = 0.05 µm thickness per slice

# imaging optics - 100x objective class, standard 160 mm tube lens
F_OBJ       = 2000.0     # µm - objective focal length
F_TUBE      = 160_000.0  # µm - tube lens focal length
PAD_WIDTH   = 64         # zero-padding to reduce FFT wraparound artifacts

# camera noise model (sCMOS sensor, same as cube sim)
PHOTON_SCALE    = 5_000   # max photons per pixel at peak intensity
READ_NOISE_STD  = 2.0     # read noise standard deviation in electrons
SEED            = 42      # fixed random seed so results are reproducible

# focus depths I want to image at, measured from the cell centre
Z_DEPTHS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

print(f"\n  Wavelength  : {WAVELENGTH} µm    NA      : {NA}")
print(f"  n_medium    : {N_MEDIUM}         n_cell  : {N_CELL}  (dn={DN_CELL})")
print(f"  Grid        : {SHAPE},  dx={DX} µm  → FOV={FOV:.1f} µm")
print(f"  Cell        : diameter={2*CELL_RADIUS} µm, length={CELL_LENGTH} µm")
print(f"               ({N_SLICES} slices × {DZ_SLICE} µm)")
print(f"  Camera      : {PHOTON_SCALE} max photons/px | "
      f"read noise σ={READ_NOISE_STD} e⁻ | seed={SEED}")
print(f"  Focus depths: {Z_DEPTHS} µm")

# 2. BUILD THE SPHEROCYLINDER SAMPLE STACK
# The key idea: for each z-slice I compute a 2D "stadium" shaped mask in XY.
# The long axis of the cell points along X (horizontal), so it looks like a pill shape from above.
# As I move away from z=0 toward the top/bottom of the cell, the effective radius r_eff
# shrinks following r_eff = sqrt(r^2 - z_c^2), which is just the equation of a sphere.
# This is what makes the end caps round rather than flat.

cy_pix  = SHAPE[0] // 2   # row index of the image centre
cx_pix  = SHAPE[1] // 2   # column index of the image centre

ys = jnp.arange(SHAPE[0])
xs = jnp.arange(SHAPE[1])
YY, XX = jnp.meshgrid(ys, xs, indexing="ij")   # YY varies along rows, XX along columns

# I computed pixel distances from the cell centre so I can write the geometry in physical units
dY = YY - cy_pix   # vertical pixel offset from centre
dX = XX - cx_pix   # horizontal pixel offset from centre

r_pix_full   = CELL_RADIUS / DX    # full radius in pixels = 10 px
half_cyl_pix = CYL_HALF    / DX    # half-length of cylinder body in pixels = 30 px

dn_slices         = []
absorption_slices = []

for k in range(N_SLICES):
    # z_c is the physical depth of the centre of this slice, measured from the cell centre
    z_c   = -CELL_RADIUS + (k + 0.5) * DZ_SLICE
    r_eff = float(np.sqrt(max(0.0, CELL_RADIUS**2 - z_c**2)))   # effective XY radius at this depth
    r_eff_pix = r_eff / DX

    # I built the stadium mask as the union of a rectangle (cylinder body) and two circles (caps).
    # This is the 2D cross-section of the spherocylinder at depth z_c.
    body      = (jnp.abs(dY) <= r_eff_pix) & (jnp.abs(dX) <= half_cyl_pix)
    cap_left  = (dX + half_cyl_pix)**2 + dY**2 <= r_eff_pix**2
    cap_right = (dX - half_cyl_pix)**2 + dY**2 <= r_eff_pix**2
    mask      = (body | cap_left | cap_right).astype(jnp.float32)

    dn_slices.append(mask * DN_CELL)
    absorption_slices.append(jnp.zeros_like(mask))   # I treated the cell as a pure phase object

dn_stack         = jnp.stack(dn_slices)          # shape: (N_SLICES, H, W)
absorption_stack = jnp.stack(absorption_slices)  # all zeros - no absorption

# maximum phase shift a ray accumulates when passing straight through the cell centre
max_phase = 2 * jnp.pi * DN_CELL * CELL_Z / WAVELENGTH
centre_mask = dn_slices[N_SLICES // 2] / DN_CELL
print(f"\n  Centre-slice lit px : {int(centre_mask.sum())}  "
      f"(stadium ≈ {2*CELL_RADIUS:.1f} µm wide × {CELL_LENGTH:.1f} µm long)")
print(f"  Max phase           : {float(max_phase):.2f} rad = "
      f"{float(max_phase / (2*jnp.pi)):.2f} wavelengths")

# 3. PLANE WAVE ILLUMINATION
# I used a uniform plane wave hitting the cell from below, same as the cube simulation
field_in = cx.plane_wave(shape=SHAPE, dx=DX, spectrum=WAVELENGTH, power=1.0)
print(f"\n  Input field  |u| max = {float(jnp.max(jnp.abs(field_in.u))):.4f}")

# 4. BEAM PROPAGATION METHOD (BPM) THROUGH THE CELL
# I propagated the light slice by slice through the cell using the multislice method.
# At each slice the cell adds a phase shift of 2*pi/lambda * dn * dz to the wavefront.
print("\n  Running BPM through spherocylinder …", flush=True)
field_exit = cx.multislice_thick_sample(
    field               = field_in,
    absorption_stack    = absorption_stack,
    dn_stack            = dn_stack,
    n                   = N_MEDIUM,
    thickness_per_slice = DZ_SLICE,
    pad_width           = PAD_WIDTH,
)
print(f"  Exit field (cell centre)  |u| max = {float(jnp.max(jnp.abs(field_exit.u))):.4f}")

# 5. SENSOR NOISE MODEL
# I used the same two-stage noise model as in the cube simulation:
#   Step 1 - scale the ideal intensity to photon counts
#   Step 2 - apply Poisson shot noise (via Chromatix basic_sensor)
#   Step 3 - add Gaussian read noise on top
def apply_sensor_noise(intensity_2d: jnp.ndarray,
                       key: jax.Array,
                       photon_scale: float = PHOTON_SCALE,
                       read_noise: float = READ_NOISE_STD) -> jnp.ndarray:
    key_shot, key_read = random.split(key)
    i_max   = jnp.max(intensity_2d) + 1e-30
    photons = intensity_2d / i_max * photon_scale   # rescale so peak pixel = PHOTON_SCALE photons
    shot = cx.basic_sensor(
        photons,
        shot_noise_mode = "poisson",
        noise_key       = key_shot,
        input_spacing   = DX,
    )
    read  = read_noise * random.normal(key_read, shot.shape)
    return shot + read   # total measured signal in photo-electrons


# 6. IMAGING PIPELINE
# For each focus depth I propagated the exit field to that plane, then passed it
# through the objective lens (with NA aperture) and tube lens to get the camera image.
def image_at_depth(field_ref, z_f):
    # (a) propagate to the chosen focus offset; skip if we are already at z=0
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
    # (b) objective lens - the NA aperture cuts off high spatial frequencies here
    field = cx.ff_lens(field, f=F_OBJ, n=N_MEDIUM, NA=NA)
    # (c) tube lens - brings the beam to a focus at the camera plane
    field = cx.ff_lens(field, f=F_TUBE, n=N_MEDIUM)
    # (d) camera records intensity I = |E|^2
    return jnp.squeeze(field.intensity)


print("\n  Imaging at each focus depth …")
master_key   = random.PRNGKey(SEED)
images_clean = []
images_noisy = []

for k, z_f in enumerate(Z_DEPTHS):
    img_clean = image_at_depth(field_exit, z_f)
    subkey    = random.fold_in(master_key, k)   # different noise seed for each depth
    img_noisy = apply_sensor_noise(img_clean, subkey)
    images_clean.append(np.array(img_clean))
    images_noisy.append(np.array(img_noisy))
    snr = float(jnp.mean(img_noisy) / (jnp.std(img_noisy) + 1e-30))
    print(f"    z = {z_f:+.1f} µm  |"
          f"  clean max={float(jnp.max(img_clean)):.2e}"
          f"  |  noisy max={float(jnp.max(img_noisy)):.1f} ph"
          f"  SNR≈{snr:.1f}")


# 7. NORMALISATION
# I normalised all clean images together and all noisy images together using a shared
# min/max so brightness is consistent across focus depths in the same figure.
def norm_global(imgs):
    g_min = min(i.min() for i in imgs)
    g_max = max(i.max() for i in imgs)
    return [(i - g_min) / (g_max - g_min + 1e-30) for i in imgs]

imgs_clean_n = norm_global(images_clean)
imgs_noisy_n = norm_global(images_noisy)

def add_cell_outline(ax):
    # I drew a dashed lime rectangle to show where the cell bounding box is in the image
    from matplotlib.patches import Rectangle
    rect = plt.Rectangle(
        (-CELL_LENGTH / 2, -CELL_RADIUS), CELL_LENGTH, 2 * CELL_RADIUS,
        linewidth=1.2, edgecolor="lime", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

# 8. FIGURE 1 – focus depth sweep (clean, noiseless images)
n_p  = len(Z_DEPTHS)
fig1, axes1 = plt.subplots(1, n_p, figsize=(3.2 * n_p, 4.2))
fig1.suptitle(
    f"Brightfield — spherocylinder {2*CELL_RADIUS}×{CELL_LENGTH} µm | clean (noiseless)\n"
    r"$I(x,y)=|E_\mathrm{camera}(x,y)|^2$"
    f"  (λ={WAVELENGTH} µm, NA={NA}, n_cell={N_CELL})",
    fontsize=11, fontweight="bold",
)
z_labels = {-0.5: "cell bottom", 0.0: "cell centre", 0.5: "cell top"}
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
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    add_cell_outline(ax)
plt.colorbar(im1, ax=axes1[-1], label="Norm. intensity", shrink=0.8)
plt.tight_layout()
out1 = "brightfield_spherocyl_focus_depths.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure 1 saved → {out1}")

# 9. FIGURE 2 – focus depth sweep (noisy images)
fig2, axes2 = plt.subplots(1, n_p, figsize=(3.2 * n_p, 4.2))
fig2.suptitle(
    f"Brightfield — spherocylinder {2*CELL_RADIUS}×{CELL_LENGTH} µm | camera noise\n"
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
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    add_cell_outline(ax)
plt.colorbar(im2, ax=axes2[-1], label="Norm. intensity", shrink=0.8)
plt.tight_layout()
out2 = "brightfield_spherocyl_noisy_depths.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 2 saved → {out2}")

# 10. FIGURE 3 – noise decomposition at z=0 (cell centre)
# I wanted to see the effect of each noise source separately, so I built three versions:
# clean ideal, shot noise only, and shot + read noise combined.
z0_idx    = Z_DEPTHS.index(0.0)
img_ideal = images_clean[z0_idx]
key_cmp   = random.PRNGKey(SEED + 99)
key_s, _  = random.split(key_cmp)
i_max_cmp = img_ideal.max() + 1e-30
photons_cmp = img_ideal / i_max_cmp * PHOTON_SCALE
img_shot = np.array(cx.basic_sensor(
    jnp.array(photons_cmp), shot_noise_mode="poisson",
    noise_key=key_s, input_spacing=DX))
img_shot_read = img_shot + READ_NOISE_STD * np.random.RandomState(1).randn(*img_shot.shape)

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4.5))
fig3.suptitle(
    f"Camera noise decomposition at z = 0 (cell centre)  —  "
    f"spherocylinder {2*CELL_RADIUS}×{CELL_LENGTH} µm\n"
    r"$I_\mathrm{noisy} = \mathrm{Poisson}(I \cdot \eta) + \mathcal{N}(0,\,\sigma_r)$",
    fontsize=12, fontweight="bold",
)
cmp_panels = [
    (img_ideal / img_ideal.max(),                "Clean  |E|²  (ideal)"),
    (img_shot  / (img_shot.max()  + 1e-9),       f"+ Poisson shot noise\n({PHOTON_SCALE} max ph/px)"),
    (img_shot_read / (img_shot_read.max() + 1e-9),
     f"+ Gaussian read noise\n(σ_r = {READ_NOISE_STD} e⁻)"),
]
for ax, (data, title) in zip(axes3, cmp_panels):
    im3 = ax.imshow(data, cmap="gray", origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.colorbar(im3, ax=ax, shrink=0.85)
    add_cell_outline(ax)
fig3.tight_layout()
out3 = "brightfield_spherocyl_noise_comparison.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 3 saved → {out3}")

# SUMMARY
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Cell geometry: spherocylinder, d={2*CELL_RADIUS} µm, L={CELL_LENGTH} µm")
print(f"  n_cell = {N_CELL}  (dn = {DN_CELL})")
print(f"  Phase shift  : {float(max_phase):.2f} rad = {float(max_phase/(2*jnp.pi)):.2f} λ")
print(f"  Magnification: {int(F_TUBE/F_OBJ)}x  →  image pixel = {DX * F_TUBE/F_OBJ:.1f} µm")
print(f"  Noise model  : Poisson({PHOTON_SCALE} ph/px) + Gaussian(σ={READ_NOISE_STD} e⁻)")
print(f"  Outputs      : {out1}, {out2}, {out3}")
print("Done.")
