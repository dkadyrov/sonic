# %%
from sonicdb import audio, utilities
import numpy as np

# file = utilities.metadata("examples/sonicdb.wav")

a = audio.Audio("examples/sonicdb.wav")
# %%
fig, ax = a.plot_spectrogram(
    window_size=128,
    nfft=128,
    noverlap=0,
    nperseg=128,
    zmin=-100,
    zmax=-50,
    time_format="samples",
)
ax.set_xlim(0, max(a.data.index))

# %%
# split the data into 5 chunks
for i in np.linspace(0, len(a.data), 5, endpoint=False):
    n = a.trim(time_format="samples", start=i, end=i + len(a.data) / 5)

    fig, ax = n.plot_spectrogram(
        window_size=128,
        nfft=128,
        noverlap=0,
        nperseg=128,
        zmin=-100,
        zmax=-50,
        time_format="samples",
    )


# %%
