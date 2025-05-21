from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, buttord
import os
from settings.config import PATH_TO_PLOT, BUTTERWORTH_PARAMS
from matplotlib.patches import Circle



def butterworth(save=False):
    """
    Illustratie van het effect van een butterworth filter
    :return:
    """
    t = np.linspace(0, 1, 1000, False)  # 1 second
    sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) #2 signalen, 1 van 10 en 20Hz genereren.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    ax1.plot(t, sig)
    ax1.set_title('10Hz en 20Hz sinussen')
    ax1.axis([0, 1, -2, 2])
    params = BUTTERWORTH_PARAMS
    b, a = butter(params["order"], 15, fs=1000, btype="low", analog=False)
    sig10 = np.sin(2*np.pi*10*t)
    filtered = signal.filtfilt(b, a, sig)
    ax2.plot(t, filtered)
    ax2.plot(t, sig10, "g--", label="10Hz")
    ax2.set_title('Na 15Hz low-pass filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [s]')
    ax2.legend()
    plt.tight_layout()
    if save:
        plt.savefig("filtering_illustrative.png")
    plt.show()

def aliasing(save=False):
    """
    Illustreert aliasing. Een proces met een hogere frequentie dan nyquist frequentie zal een incoherent signaal geveen.
    :param save: of de figuur al dan niet opgeslaan worden
    :return:
    """
    n_days = 14
    seconds_per_day = 86400
    t1 = np.linspace(0, (n_days - 1) * seconds_per_day, 10000, endpoint=False)
    t2 = np.linspace(0, (n_days - 1) * seconds_per_day, n_days, False)
    sig1 = np.sin(2 * np.pi * 17.3*10**(-6) * t1) #16u
    sig2 = np.sin(2 * np.pi * 17.3*10**(-6) * t2)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8)) #combineert beide figuren ineen
    ax1.plot(t1, sig1, "g--",label="Fysisch proces (17.3 µHz, 1/16u)")
    ax1.plot(t2, sig2, "o", label="Gemeten signaal (11.6 µHz, 1/24u)")
    ax1.legend()
    ax1.set_title("Signaal met periode 16u, meting elke 24u")


    ax2.plot(t2, sig2, 'ro', label="Resultaat metingen")
    ax2.plot(t2, sig2, 'r--', linewidth=1, alpha=0.6)  # Connect them
    ax2.set_title('Bekomen signaal')
    ax2.legend()
    if save:
        fname = os.path.join(PATH_TO_PLOT, "aliasing_example" + '.png')  # locatie en naam waar naartoe geschreven wordt
        plt.savefig(fname, dpi=200)
    plt.show()


def density_plot_digital(save=False):
    """
    Deze grafiek toont de respons van de butterworth filter in het z-domein.
    De z-as is de logaritmische 'amplification' van het signaal.
    Een signaal wordt voorgesteld op de eenheidscirkel (np.linspace(1,1) zou dit zijn, deze is uitgebreid voor de duidelijkheid)
    :param save:
    :return:
    """
    #1butterworth filter definiëren
    params = BUTTERWORTH_PARAMS
    cutoff = params["cutoff"] * np.pi * 2
    b, a = butter(N=params["order"], Wn=params["cutoff"], btype='low', analog=False)
    #2 een grid maken op het z domein
    theta = np.linspace(-np.pi, np.pi, 400)
    radius = np.linspace(0.8, 1.2, 400)
    R, T = np.meshgrid(radius, theta)
    Z = R * np.exp(1j * T) #j is de imaginaire eenheid (i) in python

    #3 de transferfunctie H(s) evalueren in het z domein
    H_z = np.polyval(b, Z ** -1) / np.polyval(a, Z ** -1) #bilineaire transformatie
    H_mag_log = np.log10(np.abs(H_z))

    #4 3D figuur maken
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.real(Z), np.imag(Z), H_mag_log, cmap='viridis')


    #5 frequenties van 0 tot 6 Hz displayen, alsook de cutoff frequentie
    f_micro = np.arange(0, 7, 1) * 1e-6  # in Hz
    f_micro = np.append(f_micro, cutoff*1e-6)
    omega_digital = 2 * np.pi * f_micro / params["fs"]  # in radians/sample
    z_points = np.exp(1j * omega_digital)

    #6 magnitude berekenen op elke frequentie
    H_points = np.polyval(b, z_points ** -1) / np.polyval(a, z_points ** -1)
    H_points_log = np.log10(np.abs(H_points))

    #7 punten plotten op de 3D figuur
    ax.scatter(
        np.real(z_points),
        np.imag(z_points),
        H_points_log,
        color='red',
        s=50,
        label="Specific μHz points"
    )
    for i, f in enumerate(f_micro):
        label = f"{f * 1e6:.1f}μHz"
        ax.text(
            np.real(z_points[i]),
            np.imag(z_points[i]),
            H_points_log[i] + 1.5,  # Offset a bit above the point
            label,
            color='black',
            fontsize=8
        )
    ax.set_xlabel('Re(z) = σ')
    ax.set_ylabel('Im(z) = ω')
    ax.set_zlabel('log₁₀|H(z)|')
    ax.legend()
    ax.set_title('Log-grootte magnitude respons van een {}e orde butterworth filter met cutoff frequentie {:.2f}μHz'.format(params["order"], cutoff))
    plt.tight_layout()
    if save:
        fname = os.path.join(PATH_TO_PLOT, "density_plot_with_bode" + f"_ord{params["order"]}" '.png')
        plt.savefig(fname, dpi=300)
    plt.show()

def pole_zero_plot(save=False):
    """
    Plot dat de polen en nullen toont van de plot in het z-domein
    :param save: al dan niet figuur opslaan
    :return:
    """
    params = BUTTERWORTH_PARAMS
    #zpk zijn de Zeroes, Polen en gain (geen idee van waar de k komt)
    z, p, k = butter(N=params["order"], Wn=params["cutoff"], btype="low", analog=False, output="zpk")
    t = np.linspace(0, 2 * np.pi, 500)
    plt.plot(np.cos(t), np.sin(t), 'k--', linewidth=1, label='Eenheidscirkel z=1')
    # nullen en polen plotten
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='b', label='Nullen')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Polen')

    # cirkelvormige achtergrond om polair stelsel duidelijk te maken
    plt.title("Filter design in z-domein")
    plt.xlabel("Reëel")
    plt.ylabel("Imaginair")
    plt.axis("equal")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    # concentrische lijnen maken
    for r in np.linspace(0.2, 1.0, 5):
        circle = Circle((0, 0), r, color='lightgray', fill=False, linestyle='--', linewidth=0.5)
        plt.gca().add_patch(circle)

    # andere assen tekenen
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)

    plt.legend()
    if save:
        fname = os.path.join(PATH_TO_PLOT, "pole_plot" + f"_ord{params["order"]}" '.png')
        plt.savefig(fname, dpi=300)
    plt.show()
