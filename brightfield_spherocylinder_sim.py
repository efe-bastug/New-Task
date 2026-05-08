

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
# I used the same grid and optical settings as in the cube simulation so the two outputs
# are directly comparable side by side. The only new parameter is the cell refractive index:
# n=1.38 is a literature value for bacterial cytoplasm — I looked it up in the paper.
# I initially tried n_cell=1.50 (same as glass) by mistake, which gave the wrong contrast.
WAVELENGTH  = 0.532   # µm - green LED illumination (matches the cube sim)
NA          = 0.80    # numerical aperture of the dry objective
N_MEDIUM    = 1.00    # surrounding medium is air (dry objective, no immersion oil)
N_CELL      = 1.38    # refractive index of bacterial cytoplasm (from literature)
DN_CELL     = N_CELL - N_MEDIUM   # refractive index contrast = 0.38 (what creates the phase shift)

# simulation grid — same resolution as the cube sim so images are directly comparable
SHAPE       = (256, 256)  # pixel grid dimensions
DX          = 0.05        # µm - physical size of each pixel in the sample plane
FOV         = SHAPE[0] * DX   # total field of view = 12.8 µm

# spherocylinder geometry
# The bacterial cell shape is modeled as a cylinder with two hemispherical end caps.
# This shape is called a "spherocylinder" or "stadium" shape (like a pill capsule).
# Total length = cylindrical section + two hemisphere radii = 3 + 0.5 + 0.5 = 4 µm.
# I had to be careful not to confuse CELL_LENGTH (total) with the cylinder-only part.
CELL_RADIUS    = 0.5    # µm — radius of the cylinder and the end caps (diameter = 1 µm)
CELL_LENGTH    = 4.0    # µm — total end-to-end length of the cell
CYL_HALF       = (CELL_LENGTH - 2 * CELL_RADIUS) / 2   # = 1.5 µm — half-length of the cylinder body only
CELL_Z         = 2 * CELL_RADIUS                        # = 1.0 µm — axial depth of the cell (equals diameter)
N_SLICES       = 20     # I divided the cell into 20 z-slices for BPM (fewer gave visible staircase artifacts)
DZ_SLICE       = CELL_Z / N_SLICES   # = 0.05 µm thickness per slice

# imaging optics — same 100x objective class as in the cube sim
F_OBJ       = 2000.0     # µm - objective focal length
F_TUBE      = 160_000.0  # µm - tube lens focal length (standard 160 mm)
PAD_WIDTH   = 64         # zero-padding to reduce FFT wraparound (ringing) artifacts at the edges

# camera noise model (same sCMOS sensor model as in the cube sim for consistency)
PHOTON_SCALE    = 5_000   # max photons per pixel at peak intensity
READ_NOISE_STD  = 2.0     # read noise standard deviation in electrons
SEED            = 42      # fixed random seed — same seed = same noise pattern every run

# focus depths I want to simulate (µm from cell centre, positive = above the cell)
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
# For each z-slice I compute the 2D cross-sectional shape ("stadium mask") in XY.
# The cell's long axis points along X, so viewed from above it looks like a pill capsule.
#
# Key insight: as we move up or down from z=0, the effective XY radius at that depth
# shrinks according to r_eff = sqrt(R^2 - z_c^2) — this is just the equation of a sphere.
# This is what gives the cell rounded end caps instead of flat ones.
# I first coded this with flat caps (constant radius at every slice) which looked like a cylinder,
# not a real bacterial cell — switching to the sphere equation fixed the shape.

cy_pix  = SHAPE[0] // 2   # row index of the image centre
cx_pix  = SHAPE[1] // 2   # column index of the image centre

ys = jnp.arange(SHAPE[0])
xs = jnp.arange(SHAPE[1])
# indexing="ij" is important here — it means YY varies along axis 0 (rows) and XX along axis 1 (cols).
# Without it, X and Y would be swapped and the cell would appear rotated 90°.
YY, XX = jnp.meshgrid(ys, xs, indexing="ij")

# Pixel offsets from the cell centre (in pixels, not µm yet)
dY = YY - cy_pix   # vertical pixel offset from centre
dX = XX - cx_pix   # horizontal pixel offset from centre

r_pix_full   = CELL_RADIUS / DX    # full cell radius in pixels = 10 px
half_cyl_pix = CYL_HALF    / DX    # half-length of cylinder body in pixels = 30 px

dn_slices         = []
absorption_slices = []

for k in range(N_SLICES):
    # z_c is the physical z-coordinate at the centre of this slice (measured from cell centre).
    # The +0.5 puts us at the middle of the slice, not the edge.
    z_c   = -CELL_RADIUS + (k + 0.5) * DZ_SLICE
    # r_eff is the effective XY radius of the spherocylinder cross-section at depth z_c.
    # The max(0.0, ...) prevents a tiny negative value under the sqrt due to floating-point errors
    # at the very top/bottom slices — I hit this bug and got a NaN without the clamp.
    r_eff = float(np.sqrt(max(0.0, CELL_RADIUS**2 - z_c**2)))
    r_eff_pix = r_eff / DX

    # The stadium (pill capsule) shape is the union of three regions:
    #   body      — the rectangular center cylinder body
    #   cap_left  — the left hemispherical end cap (a circle centered at x = -half_cyl)
    #   cap_right — the right hemispherical end cap (a circle centered at x = +half_cyl)
    # Using bitwise OR (|) on boolean JAX arrays combines the three regions cleanly.
    body      = (jnp.abs(dY) <= r_eff_pix) & (jnp.abs(dX) <= half_cyl_pix)
    cap_left  = (dX + half_cyl_pix)**2 + dY**2 <= r_eff_pix**2
    cap_right = (dX - half_cyl_pix)**2 + dY**2 <= r_eff_pix**2
    mask      = (body | cap_left | cap_right).astype(jnp.float32)

    dn_slices.append(mask * DN_CELL)
    absorption_slices.append(jnp.zeros_like(mask))   # bacteria treated as pure phase objects (no absorption)

dn_stack         = jnp.stack(dn_slices)          # shape: (N_SLICES, H, W)
absorption_stack = jnp.stack(absorption_slices)  # all zeros — bacteria are nearly transparent

# The maximum phase shift a ray picks up traveling straight through the cell centre.
# Formula: phi_max = (2*pi / lambda) * dn * thickness
# This is useful to sanity-check whether the cell is a "weak" or "strong" phase object.
max_phase = 2 * jnp.pi * DN_CELL * CELL_Z / WAVELENGTH
centre_mask = dn_slices[N_SLICES // 2] / DN_CELL
print(f"\n  Centre-slice lit px : {int(centre_mask.sum())}  "
      f"(stadium ≈ {2*CELL_RADIUS:.1f} µm wide × {CELL_LENGTH:.1f} µm long)")
print(f"  Max phase           : {float(max_phase):.2f} rad = "
      f"{float(max_phase / (2*jnp.pi)):.2f} wavelengths")

# 3. PARTIAL-COHERENCE + ABSORPTION MODEL SETUP
#
# This section adds two physical improvements over the simple fully-coherent CTF model
# I had in the previous version. Without these, the cell was mathematically exactly
# invisible at z=0 (which is physically wrong — real bacteria are faintly visible).
#
#  (a) Partial coherence via the Hopkins method:
#      Real Köhler illumination uses a condenser lens that fills a cone of angles
#      up to the condenser NA. Each illumination angle shifts the diffraction rings
#      to a slightly different position. Averaging N_ILL intensities incoherently
#      causes the outer rings to cancel out while the first-order halo near the cell
#      edge survives — this matches what you actually see in a real microscope.
#      I first tried averaging complex fields (coherent sum) instead of intensities,
#      which gave wrong results; the fix was to sum |field|^2, not field itself.
#
#  (b) Tiny imaginary refractive index (absorption):
#      Real bacteria absorb a tiny amount of light (dn_imag ≈ 0.002). This adds a
#      cos(W) term to the image formula. At z=0, cos(0)=1, so the absorption
#      contribution is maximum at focus — leaving a faint but non-zero cell outline.
#      This is the "almost but not completely invisible" behaviour described in the paper,
#      as opposed to the exact zero I was getting with a pure-phase model.

# Parameters for the two corrections
NA_COND  = 0.70 * NA    # condenser NA (coherence factor σ = NA_cond/NA_obj = 0.7, typical Köhler value)
DN_IMAG  = 0.002        # imaginary refractive index of bacteria (very small but nonzero absorption)
N_ILL    = 61           # number of illumination angles sampled on the condenser disk

# --- Projected PHASE map  φ(x,y) = (2π/λ) Σ dn·dz ---
# I accumulate the phase shift contributed by each z-slice and sum them up.
# Using float64 here avoids the numerical precision loss that float32 caused at small dn values.
phase_proj = np.zeros(SHAPE, dtype=np.float64)
for dn_s in dn_slices:
    phase_proj += np.array(dn_s, dtype=np.float64) * DZ_SLICE
phase_proj *= (2.0 * np.pi / WAVELENGTH)

# --- Projected ABSORPTION map  α(x,y) = (2π/λ)·dn_imag·thickness ---
# I use (dn_s > 0) as a boolean occupancy mask so the absorption is proportional
# to the physical thickness of the cell, not the refractive index contrast.
absorb_proj = np.zeros(SHAPE, dtype=np.float64)
for dn_s in dn_slices:
    absorb_proj += np.array(dn_s > 0, dtype=np.float64) * DZ_SLICE
absorb_proj *= (2.0 * np.pi / WAVELENGTH) * DN_IMAG   # multiply by the small imaginary part

# --- Frequency grid (cycles per µm) ---
# np.fft.fftfreq gives frequencies in the standard FFT order (0, positive, then negative).
# I need this grid to apply the defocus phase factor W in frequency domain.
_fy = np.fft.fftfreq(SHAPE[0], d=DX)
_fx = np.fft.fftfreq(SHAPE[1], d=DX)
_FX, _FY = np.meshgrid(_fx, _fy, indexing="ij")
_NU2    = _FX**2 + _FY**2         # squared spatial frequency magnitude |ν|²
_PHI_FT = np.fft.fft2(phase_proj)   # Fourier transform of the projected phase
_ABS_FT = np.fft.fft2(absorb_proj)  # Fourier transform of the projected absorption

# --- Condenser illumination angles via Fibonacci disk lattice ---
# The Fibonacci spiral places N_ILL points nearly uniformly on a disk with no grid symmetry.
# I chose this over a regular grid because a regular grid creates spurious symmetry artifacts
# in the averaged image that a real (incoherent) condenser does not have.
_phi_g  = 2.399963229                          # golden angle in radians = 2π/φ² where φ is the golden ratio
_k      = np.arange(N_ILL)
_r_ill  = (NA_COND / WAVELENGTH) * np.sqrt((_k + 0.5) / N_ILL)   # radius scales with sqrt for uniform disk density
_t_ill  = _phi_g * _k                                              # angle increments by the golden angle each step
_ill_fx = np.concatenate([[0.0], _r_ill * np.cos(_t_ill)])   # include on-axis illumination (straight through)
_ill_fy = np.concatenate([[0.0], _r_ill * np.sin(_t_ill)])
_N_ILL  = len(_ill_fx)

print(f"\n  Phase map  : {phase_proj.min():.2f} … {phase_proj.max():.2f} rad "
      f"(expected max {float(max_phase):.2f} rad)")
print(f"  Absorption : dn_imag={DN_IMAG}  →  peak α={absorb_proj.max():.4f} rad")
print(f"  Condenser  : NA_cond={NA_COND:.3f} (σ={NA_COND/NA:.2f}),  "
      f"{_N_ILL} illumination angles")
print(f"  NA cutoff  : {NA/WAVELENGTH:.2f} cyc/µm  →  "
      f"resolution ≈ {WAVELENGTH/(2*NA):.3f} µm")

# 4. PLANE WAVE ILLUMINATION + BPM (kept alongside the CTF model for comparison)
# I still run the BPM pipeline here so I can show a side-by-side comparison figure
# between the old coherent BPM output and the new partial-coherence CTF model.
# Without this comparison it would be hard to explain why the new model is better.
field_in = cx.plane_wave(shape=SHAPE, dx=DX, spectrum=WAVELENGTH, power=1.0)
print(f"\n  Input field  |u| max = {float(jnp.max(jnp.abs(field_in.u))):.4f}")

# Multislice BPM: at each slice the field picks up a phase shift of (2π/λ)·dn·dz,
# then the angular spectrum method propagates it to the next slice.
# I set pad_width=PAD_WIDTH to avoid the strong ringing I saw without zero-padding.
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
def apply_sensor_noise(intensity_2d: jnp.ndarray,
                       key: jax.Array,
                       photon_scale: float = PHOTON_SCALE,
                       read_noise: float = READ_NOISE_STD) -> jnp.ndarray:
    key_shot, key_read = random.split(key)
    i_max   = jnp.max(intensity_2d) + 1e-30
    photons = intensity_2d / i_max * photon_scale
    shot = cx.basic_sensor(
        photons,
        shot_noise_mode = "poisson",
        noise_key       = key_shot,
        input_spacing   = DX,
    )
    read = read_noise * random.normal(key_read, shot.shape)
    return shot + read


# 6. PARTIALLY COHERENT BRIGHTFIELD IMAGING (Hopkins method)
#
# This is the core imaging function. The full formula for a thin object with both
# phase (φ) and absorption (α) under a tilted plane wave at angle ν_ill is:
#   I(ν, z, ν_ill) = δ − 2·P(ν+ν_ill)·[sin(W)·Φ(ν) + cos(W)·A(ν)]
#   where  W = π·λ·z/n · (|ν|² + 2·ν·ν_ill)   (defocus phase factor)
#
# Averaging incoherently over all condenser angles:
#   sin(W) term → phase CTF, zero at z=0 — bacteria are invisible when in focus
#   cos(W) term → absorption ATF, equals 1 at z=0 — tiny outline always visible
#
# Why averaging helps: each angle shifts the diffraction ring pattern. The outer rings
# cancel out when summed, but the DC background and first-order halo survive.
# This localises the contrast ring to the cell edge, matching real microscope images.
#
# Bug I fixed: I forgot the "+2ν·ν_ill" tilt correction in W for off-axis angles,
# which made the rings appear at the wrong position for oblique illumination.
def partial_coherent_brightfield(z_f: float) -> np.ndarray:
    I_sum = np.zeros(SHAPE, dtype=np.float64)
    lam_n = WAVELENGTH / N_MEDIUM   # effective wavelength in the medium
    for fxi, fyi in zip(_ill_fx, _ill_fy):
        # Defocus phase factor W for this illumination angle (ν_ill = (fxi, fyi))
        W       = np.pi * lam_n * z_f * (_NU2 + 2.0 * (_FX * fxi + _FY * fyi))
        sin_W   = np.sin(W)
        cos_W   = np.cos(W)
        # Shift the objective pupil to the current illumination angle
        P_shift = (np.sqrt((_FX + fxi)**2 + (_FY + fyi)**2) <= NA / WAVELENGTH).astype(np.float64)
        # Combine phase (sin) and absorption (cos) contributions in frequency domain
        I_FT = -2.0 * P_shift * (sin_W * _PHI_FT + cos_W * _ABS_FT)
        # Add the DC background (flat illumination = 1) — without this the mean image is 0
        I_FT[0, 0] += float(SHAPE[0] * SHAPE[1])
        I_sum += np.real(np.fft.ifft2(I_FT))
    # Divide by number of angles to get the incoherent average, then clip negative values
    # (negative intensity is physically impossible — they can appear from numerical noise)
    return np.clip(I_sum / _N_ILL, 0.0, None)


# Coherent BPM pipeline — I kept this only to generate the comparison figure (Figure 4).
# This is the older model: fully coherent, no partial coherence, no absorption term.
# Showing it next to the new CTF model helps explain why the new one is more realistic.
def image_at_depth_bpm(field_ref, z_f):
    if abs(z_f) > 1e-9:
        field = cx.transfer_propagate(
            field=field_ref, z=float(z_f), n=N_MEDIUM,
            pad_width=PAD_WIDTH, mode="same",
        )
    else:
        field = field_ref
    field = cx.ff_lens(field, f=F_OBJ, n=N_MEDIUM, NA=NA)
    field = cx.ff_lens(field, f=F_TUBE, n=N_MEDIUM)
    return jnp.squeeze(field.intensity)


print("\n  Imaging at each focus depth (partial-coherence + absorption model) …")
master_key   = random.PRNGKey(SEED)
images_clean = []
images_noisy = []
images_bpm   = []   # coherent BPM images -- stored only for the comparison figure

for k, z_f in enumerate(Z_DEPTHS):
    # New model: partial coherence + absorption -- this is the physically correct result
    img_clean = partial_coherent_brightfield(z_f)
    # random.fold_in mixes the master key with k to give a unique, reproducible key per depth
    subkey    = random.fold_in(master_key, k)
    img_noisy = apply_sensor_noise(jnp.array(img_clean), subkey)
    images_clean.append(img_clean)
    images_noisy.append(np.array(img_noisy))

    # Old model: coherent BPM -- kept only to generate the side-by-side comparison figure
    img_bpm = np.array(image_at_depth_bpm(field_exit, z_f))
    images_bpm.append(img_bpm)

    contrast = float(np.std(img_clean))
    snr = float(jnp.mean(img_noisy) / (jnp.std(img_noisy) + 1e-30))
    print(f"    z = {z_f:+.1f} µm  |"
          f"  contrast σ={contrast:.4f}"
          f"  |  noisy max={float(jnp.max(img_noisy)):.1f} ph"
          f"  SNR≈{snr:.1f}")


# 7. NORMALISATION
# I normalize clean images together and noisy images together using one shared min/max.
# Using a shared scale is important — if I normalized each panel independently,
# the brightness would drift between focus depths and the sweep figure would be misleading.
def norm_global(imgs):
    g_min = min(i.min() for i in imgs)
    g_max = max(i.max() for i in imgs)
    return [(i - g_min) / (g_max - g_min + 1e-30) for i in imgs]

imgs_clean_n = norm_global(images_clean)
imgs_noisy_n = norm_global(images_noisy)

def add_cell_outline(ax):
    # I overlay a dashed lime green rectangle showing the cell's bounding box.
    # Without this it is hard to tell whether the defocus rings are centered on the cell
    # or just floating FFT artifacts. The import is inside the function to keep it local.
    from matplotlib.patches import Rectangle
    rect = plt.Rectangle(
        (-CELL_LENGTH / 2, -CELL_RADIUS), CELL_LENGTH, 2 * CELL_RADIUS,
        linewidth=1.2, edgecolor="lime", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

# 8. FIGURE 1 – focus depth sweep (clean, noiseless CTF images)
n_p  = len(Z_DEPTHS)
fig1, axes1 = plt.subplots(1, n_p, figsize=(3.2 * n_p, 4.2))
fig1.suptitle(
    f"Brightfield — spherocylinder {2*CELL_RADIUS}×{CELL_LENGTH} µm | CTF model, clean\n"
    r"$I(\nu,z)\approx 1 - 2\sin(\pi\lambda z\nu^2/n)\,P(\nu)\,\Phi(\nu)$"
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

# 9. FIGURE 2 – focus depth sweep (noisy CTF images)
fig2, axes2 = plt.subplots(1, n_p, figsize=(3.2 * n_p, 4.2))
fig2.suptitle(
    f"Brightfield — spherocylinder {2*CELL_RADIUS}×{CELL_LENGTH} µm | CTF model + camera noise\n"
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

# 10. FIGURE 3 – noise decomposition at z = -1 µm (clearly defocused, bacteria visible)
# I chose z=-1 instead of z=0 because at z=0 the cell is nearly invisible (pure phase object),
# so there is almost no signal to decompose. At z=-1 the defocus rings are clearly visible,
# making the noise panels much more informative.
z_demo_idx = Z_DEPTHS.index(-1.0)
img_ideal  = images_clean[z_demo_idx]
key_cmp    = random.PRNGKey(SEED + 99)
key_s, _   = random.split(key_cmp)
i_max_cmp  = img_ideal.max() + 1e-30
photons_cmp = img_ideal / i_max_cmp * PHOTON_SCALE
img_shot = np.array(cx.basic_sensor(
    jnp.array(photons_cmp), shot_noise_mode="poisson",
    noise_key=key_s, input_spacing=DX))
img_shot_read = img_shot + READ_NOISE_STD * np.random.RandomState(1).randn(*img_shot.shape)

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4.5))
fig3.suptitle(
    f"Camera noise decomposition at z = −1 µm (defocused)  —  CTF model\n"
    r"$I_\mathrm{noisy} = \mathrm{Poisson}(I \cdot \eta) + \mathcal{N}(0,\,\sigma_r)$",
    fontsize=12, fontweight="bold",
)
cmp_panels = [
    (img_ideal / (img_ideal.max() + 1e-9),         "CTF clean (ideal)"),
    (img_shot  / (img_shot.max()  + 1e-9),         f"+ Poisson shot noise\n({PHOTON_SCALE} max ph/px)"),
    (img_shot_read / (img_shot_read.max() + 1e-9), f"+ Gaussian read noise\n(σ_r = {READ_NOISE_STD} e⁻)"),
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

# 11. FIGURE 4 – CTF model vs old coherent BPM side-by-side comparison
# I made this figure to explain the improvement clearly.
# The BPM model (bottom row) shows strong contrast even at z=0, which is physically wrong
# for a pure phase object. The new CTF model (top row) correctly gives a nearly invisible
# cell at focus and only shows contrast when defocused — this matches real bacteria images.
z_compare = [-1.0, 0.0, 1.0]
z_compare_idx = [Z_DEPTHS.index(z) for z in z_compare]

def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-30)

fig4, axes4 = plt.subplots(2, len(z_compare), figsize=(4.5 * len(z_compare), 8))
fig4.suptitle(
    "Partial-coherence + absorption model  vs  coherent BPM model\n"
    "Top row: new (nearly invisible at z=0, localised rings)   "
    "Bottom row: old BPM (too much contrast at z=0, wide rings)",
    fontsize=11, fontweight="bold",
)
row_labels = ["CTF model\n(new)", "Coherent BPM\n(old)"]
for col, (z_f, idx) in enumerate(zip(z_compare, z_compare_idx)):
    for row, (img_list, row_label) in enumerate(
        [(images_clean, "CTF"), (images_bpm, "BPM")]
    ):
        ax  = axes4[row, col]
        img = img_list[idx]
        im4 = ax.imshow(norm01(img), cmap="gray", origin="lower",
                        extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
        if row == 0:
            ax.set_title(f"z = {z_f:+.1f} µm", fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_label, fontsize=9)
        ax.set_xlabel("x (µm)", fontsize=8)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.tick_params(labelsize=7)
        add_cell_outline(ax)
        plt.colorbar(im4, ax=ax, shrink=0.8, label="Norm. intensity")
fig4.tight_layout()
out4 = "brightfield_spherocyl_ctf_vs_bpm.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 4 saved → {out4}")

# SUMMARY
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Cell geometry: spherocylinder, d={2*CELL_RADIUS} µm, L={CELL_LENGTH} µm")
print(f"  n_cell = {N_CELL}  (dn = {DN_CELL})")
print(f"  Phase shift  : {float(max_phase):.2f} rad = {float(max_phase/(2*jnp.pi)):.2f} λ")
print(f"  Magnification: {int(F_TUBE/F_OBJ)}x  →  image pixel = {DX * F_TUBE/F_OBJ:.1f} µm")
print(f"  Noise model  : Poisson({PHOTON_SCALE} ph/px) + Gaussian(σ={READ_NOISE_STD} e⁻)")
print(f"  Imaging model: partial coherence (σ={NA_COND/NA:.2f}), "
      f"dn_imag={DN_IMAG}, {_N_ILL} angles")
print(f"  Outputs      : {out1}, {out2}, {out3}, {out4}")
print("Done.")
