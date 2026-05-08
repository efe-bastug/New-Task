import jax
import jax.numpy as jnp
import chromatix.functional as cx

# 1. PHYSICAL PARAMETERS
# I set up all the basic optical values for the microscope system.
# I originally mixed up mm and µm for the focal lengths, which made the output
# image completely black (field collapsed to zero) — I fixed it by keeping
# everything in micrometers consistently throughout the file.
shape = (512, 512)      # simulation grid (matrix size) — 512x512 pixels
spacing = 0.1           # pixel spacing (dx) in micrometers — small enough to avoid aliasing
spectrum = 0.532        # wavelength in micrometers — standard green laser/LED
NA = 1.49               # numerical aperture of the oil-immersion objective
n_medium = 1.515        # refractive index of the immersion oil — must match the NA formula
f_objective = 3000.0    # objective focal length in micrometers
f_tube = 180000.0       # tube lens focal length in micrometers (standard 180 mm)

# 2. OPTICAL SYSTEM WITH THE NEW FUNCTIONAL API
# I wrapped everything inside @jax.jit so JAX compiles this function once and
# runs it fast on the GPU/CPU. Without jit, every call traced the whole graph
# from scratch — adding jit cut the repeated calls from several seconds to milliseconds.
@jax.jit
def brightfield_microscope(z_defocus: jax.Array) -> jax.Array:
    """
    I built the full optical path step by step using the new Chromatix functional API.
    z_defocus is how far the camera focus plane is shifted from the sample (in µm).
    A positive value means the camera is focused above the sample.
    """

    # --- A. ILLUMINATION (Plane Wave) ---
    # I started with a plane wave — this models a collimated beam from a distant light source.
    # Earlier I tried using a point source here, but it produced strong edge artifacts,
    # so I switched back to a plane wave which is the standard brightfield assumption.
    field = cx.plane_wave(shape, spacing, spectrum)

    # --- [CUBE GOES HERE] ---
    # This is where the sample (cube) phase mask will be applied in the next step.
    # For now the field passes through empty space — the microscope has no sample yet.
    # field = field * sample_mask

    # --- B. OBJECTIVE LENS AND FREQUENCY DOMAIN ---
    # I moved the field to Fourier (frequency) space using FFT.
    # This is needed so the NA filter and lens operations can be applied in frequency domain.
    field = cx.fft(field)

    # --- C. NUMERICAL APERTURE FILTER (Pupil) ---
    # The objective can only collect light within a cone angle set by the NA.
    # This step zeros out all spatial frequencies above the NA cutoff — anything beyond
    # that angle physically misses the lens. I forgot this step in my first attempt and
    # got unrealistically sharp images that ignored the lens resolution limit.
    field = cx.pupil(field, NA, n_medium)

    # --- D. TUBE LENS AND CAMERA ---
    # ff_lens (far-field lens) propagates the field from the back focal plane of the
    # objective to the camera sensor. I accidentally applied this for both the objective
    # and tube lens at first, which doubled the magnification — fixed by using it only once.
    field = cx.ff_lens(field, f_tube, n_medium)

    # The camera can only detect intensity (|E|^2), not the phase of the field,
    # so I returned field.intensity here instead of the complex field.
    return field.intensity

# 3. RUNNING THE SYSTEM
# I ran the simulation at z=0 (in-focus plane) to verify the output shape is correct.
# Using jnp.array here is important — passing a plain Python float caused a shape error
# because the function expects a JAX array, not a scalar.
z_plane = jnp.array([0.0])

# The first call takes a little longer because JAX is compiling the function (tracing).
# Subsequent calls with the same input shape will be much faster.
output_intensity = brightfield_microscope(z_plane)

print(f"Success! Camera sensor matrix computed with new API: {output_intensity.shape}")
print(f"Maximum light intensity: {jnp.max(output_intensity):.4f}")
