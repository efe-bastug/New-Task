import jax
import jax.numpy as jnp
import chromatix.functional as cx

# 1. PHYSICAL PARAMETERS
# I defined the basic optical values for the microscope system
shape = (512, 512)      # simulation grid (matrix size)
spacing = 0.1           # pixel spacing (dx) in micrometers
spectrum = 0.532        # wavelength in micrometers - green light
NA = 1.49               # numerical aperture of the objective
n_medium = 1.515        # refractive index of the medium (immersion oil)
f_objective = 3000.0    # objective focal length in micrometers
f_tube = 180000.0       # tube lens focal length in micrometers

# 2. OPTICAL SYSTEM WITH THE NEW FUNCTIONAL API
# I used @jax.jit so this function gets compiled to run fast on GPU/CPU
@jax.jit
def brightfield_microscope(z_defocus: jax.Array) -> jax.Array:
    """
    I defined the optical path using the new Chromatix API.
    z_defocus represents the focus depth of the camera.
    """

    # --- A. ILLUMINATION (Plane Wave) ---
    # I started the system with a plane wave as the input field
    field = cx.plane_wave(shape, spacing, spectrum)

    # --- [CUBE GOES HERE] ---
    # I will add the cube's refractive index as a mask here in the next step
    # field = field * sample_mask

    # --- B. OBJECTIVE LENS AND FREQUENCY DOMAIN ---
    # I moved the field to Fourier (frequency) space
    field = cx.fft(field)

    # --- C. NUMERICAL APERTURE FILTER (Pupil) ---
    # I limited how much light the objective can collect using the NA
    field = cx.pupil(field, NA, n_medium)

    # --- D. TUBE LENS AND CAMERA ---
    # I used ff_lens (Fourier-Fourier Lens) to focus the light back onto the sensor plane
    field = cx.ff_lens(field, f_tube, n_medium)

    # I returned the intensity of the light hitting the camera
    return field.intensity

# 3. RUNNING THE SYSTEM
# I ran the simulation at the in-focus plane (z=0)
z_plane = jnp.array([0.0])

# I computed the model output
output_intensity = brightfield_microscope(z_plane)

print(f"Success! Camera sensor matrix computed with new API: {output_intensity.shape}")
print(f"Maximum light intensity: {jnp.max(output_intensity):.4f}")
