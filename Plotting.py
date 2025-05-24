import matplotlib.pyplot as plt
import time
from settings.config import BUTTERWORTH_PARAMS, WINDOW_LENGTH, PATH_TO_DATA, PATH_TO_PLOT
from Help.Helper_functions import nan_helper, standardise, big_number_removal, shift, export
import numpy as np
import pandas
import os
from matplotlib.patches import Circle

from scipy.signal import freqz, butter

from Filter import Filtering


def plot_missing_values(values, index, filtering_instance):
    """
    :param values: de waarden van de metingen
    :param index: welke metingen worden bekeken
    :param filtering_instance: een instantie van de Class filtering
    :return: 2 lijsten, 1 met de data van de lege waarden, eentje met de 'lege' waarden geïnterpoleerd
    """
    empty_dates_values, empty_dates = [], []
    interpolated_top, dates = filtering_instance.interpolate()
    for meas_id in range(len(index)):
        nans, x = nan_helper(values.iloc[:, meas_id].to_numpy())
        interpolated = interpolated_top[:, meas_id]
        for j in x(nans):
            empty_dates.append(dates[j])
            empty_dates_values.append(interpolated[j])
    return empty_dates, empty_dates_values

class Plotting:
    """
    Er wordt een class gemaakt zodat niet alle berekeningen telkens opnieuw moeten
    De inhoud van deze class can grotendeels gecopy paste wordeni bij Filter.py geloof ik,
    maar de tijdswinst lijkt me minimaal (max 1s) voor de duidelijkheid die het met zich mee brengt.
    """
    def __init__(self, path_to_data, index):
        self.path_to_data = path_to_data # data uitlezen
        self.dataframe = pandas.read_csv(path_to_data)
        self.dates = self.dataframe.iloc[0:,0].dropna()  # datums; iloc zoekt bepaalde delen op; dropna laat lege waarden weg;
        self.values = self.dataframe.iloc[0:, shift(index)].map(big_number_removal)  # alle metingen excl header en datums
        if isinstance(index, int):
            self.index = [index]
        else:
            self.index = index

    def plot_moving_median(self, extension, save=False):
        """
        Een plot die het effect van de moving median aangeeft.
        :param index: de index van de meting waarop je de median wilt doen
        :param save: als dit ingevuld wordt, zal de grafiek opgeslaan worden naar die locatie
        :return:
        """
        index = self.index
        values = standardise(self.values, index) #omdat er altijd met 2D arrays moet gewerkt worden
        filtering_instance = Filtering(self.path_to_data, index)
        values_filtered, dates_filtered = filtering_instance.moving_window()

        # ontbrekende data aanduiden
        empty_dates, empty_dates_values = plot_missing_values(values, index, filtering_instance)

        # plot maken
        plt.plot(self.dates, values, linewidth=1, label="raw data")
        plt.plot(dates_filtered, values_filtered[:], linewidth=1, label="filtered data")
        plt.plot(empty_dates, empty_dates_values, marker="x", linestyle="", color="red", markersize=8)
        plt.xlabel('Date')
        plt.xticks(rotation=15, ticks=dates_filtered[::50])
        plt.grid(True)
        plt.legend(['raw', 'filtered'])
        if save:
            export("moving median plot" + str(filtering_instance.window) + str(index), extension=extension)
        plt.show()
        plt.close()

    def plot_butterworth(self, extension, phase_shift = False, interpolation = False, save = False, moving_window = False, residuals = False, compare = False):
        dates, index = self.dates, self.index
        values = standardise(self.values, index)

        filtering_instance = Filtering(self.path_to_data, index)
        if moving_window:
            filtering_instance.values, _ = filtering_instance.moving_window()

        values_filtered, dates_filtered = filtering_instance.butterworth(phase_shift, interpolation)

        #plotsize kiezen
        n_rows = 2 if residuals else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 if n_rows == 1 else 8))  # adjust size for subplots
        if residuals:
            ax1, ax2 = axes
        else:
            ax1 = axes


        params = BUTTERWORTH_PARAMS
        cutoff = params["cutoff"] / float(2) * params["fs"] * float(10 ** 6)  # terug omzetten van relatieve cutoff naar echte

        #titelgeving grafiek
        suffix = ""
        if phase_shift:
            suffix += ", met faseverschuiving"
        if moving_window:
            suffix += ", met moving window van {} dagen".format(WINDOW_LENGTH)
        if interpolation:
            suffix += ", met geïnterpoleerde waarden"
        ax1.set_title("Plot datapunt {} met Butterworth filter".format(str(index), cutoff, float(1)/(cutoff*10**(-6)*3600*24), suffix))

        # plotting
        ax1.plot(dates, values, "b--", linewidth=1, label="raw data")
        ax1.plot(dates_filtered, values_filtered, "r", linewidth=1, label="filtered data")
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('Weerstand (Ω)')
        ax1.set_xticks(dates_filtered[::50])
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid('on')
        legend_ax1 = [f"ruwe data", f"gefilterde data {params["days"]}d"]

        if not interpolation:
            empty_dates, empty_dates_values = plot_missing_values(values, index, filtering_instance)
            ax1.plot(empty_dates, empty_dates_values, marker="x", linestyle="", color="red", markersize=5)
            legend_ax1.append("ontbrekende waarden")

        if compare:
            filtering_instance_compare = Filtering(self.path_to_data, index)
            days_compare = 14  # Comparison filter length
            assert days_compare > 2, "tijdseenheid niet boven Nyquist frequentie"
            fs_compare = 1 / (3600 * 24)
            cutoff_compare = 1 / (3600 * 24 * 2) * 2 / days_compare

            compare_params = {
                "days": days_compare,
                "order": 2,
                "fs": fs_compare,
                "cutoff": cutoff_compare / (0.5 * fs_compare)
            }
            filtering_instance_compare.params = compare_params
            filtering_instance_compare.values = standardise(self.values, index)

            values_filtered_compared, dates_filtered_compared = filtering_instance_compare.butterworth(phase_shift, interpolation)
            residual_values_compared = np.abs(values_filtered_compared - values[:len(values_filtered_compared)])

            ax1.plot(dates_filtered_compared, values_filtered_compared, color="green")
            legend_ax1.append(f"gefilterde data {days_compare}d")

        if residuals:
            residual_values = np.abs(values_filtered - values[:len(values_filtered)])
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.scatter(dates_filtered, residual_values, color="red", s=1)
            ax2.set_title("Residuen (|gefiltered - raw|)")
            ax2.set_xlabel("Datum")
            ax2.set_xticks(dates_filtered[::50])
            ax2.set_ylabel("Restwaarde (Ω)")
            legend_ax2 = [f"restwaarden {BUTTERWORTH_PARAMS["days"]}d"]
            if compare:
                ax2.scatter(dates_filtered_compared, residual_values_compared, s=1, color="green")
                legend_ax2.append(f"restwaarden {days_compare}d")
            ax2.legend(legend_ax2)
            ax2.grid('on')

        ax1.legend(legend_ax1)
        plt.tight_layout()

        if save:
            #zodat men in de naam kan zien wat er allemaal is toegepast op de data
            suffix = ""
            if interpolation:
                suffix += "_interp"
            if moving_window:
                suffix += "_mw{}".format(WINDOW_LENGTH)
            if phase_shift:
                suffix += "_shifted"
            if residuals:
                suffix += "_res"
            export(f"butterworth{index}{suffix}_{cutoff:.2f}μHz", extension=extension)
        plt.show()


def butterworth_frequency_response(log=False, poles=False, save=False) -> np.ndarray:
    """
    Frequency response is hoeveel elke frequentie wordt gedempt. Geen enkele filter is perfect, er zal altijd wat door komen
    Net zoals er altijd wat water in de kom pasta blijft zitten na het afgieten :))
    :param log: of de y-as logaritmisch moet zin of niet
    :param save: hierin zet je of er opgeslagen moet worden
    :return: een plot van de frequency response
    """
    params = BUTTERWORTH_PARAMS
    b, a = butter(N=params["order"], Wn=params["cutoff"], btype="low", analog=False)
    z, p, k = butter(N=params["order"], Wn=params["cutoff"], btype="low", analog=False, output="zpk")

    cutoff = params["cutoff"] / float(2) * params["fs"] * float(10 ** 6)
    #frequency response
    w, h = freqz(b, a, worN=8000) #steekt 8000 frequenties in de filter en bekijkt de respons ervan

    #scaling
    xf = (params["fs"] / (2 * np.pi)) * w #w wordt gegeven in radialen, dus moet het aangepast worden naar frequentiedomein.
    yf = np.abs(h)
    is_log = ""
    if log is True:
        is_log = "Logarithmic"
        yf = np.log(yf)
    if poles:
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        t = np.linspace(0, 2 * np.pi, 500)
        ax2.plot(np.cos(t), np.sin(t), 'k--', linewidth=1, label='Eenheidscirkel z=1')

        # plotten nullen en polen
        ax2.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='b', label='Nullen')
        ax2.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Polen')

        # Microhertz frequenties erop zetten
        f_micro = np.arange(0, 6, 1) * 1e-6  # in Hz
        omega_digital = 2 * np.pi * f_micro / params["fs"]  # radians/sample
        z_points = np.exp(1j * omega_digital)

        # Punten plotten
        ax2.scatter(np.real(z_points), np.imag(z_points), color='g', s=50, label='μHz punten')

        # Labels
        for i, zpt in enumerate(z_points):
            ax2.text(np.real(zpt) + 0.06, np.imag(zpt) + 0.02, f"{f_micro[i] * 1e6:.0f}μHz", fontsize=8, color='green')

        # Cirkelvormige achtergrond
        ax2.set_title("Filter design in z-domein")
        ax2.set_xlabel("Reëel")
        ax2.set_ylabel("Imaginair")
        ax2.axis("equal")
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        for r in np.linspace(0.2, 1.0, 5):
            circle = Circle((0, 0), r, color='lightgray', fill=False, linestyle='--', linewidth=0.5)
            ax2.add_patch(circle)

        # Andere assen
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.axvline(0, color='gray', linewidth=0.5)

        ax2.legend()
        fig.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the frequency response
    ax1.plot(xf, yf)
    ax1.grid()
    ax1.set_xlabel("Frequentie (μHz)")
    if log:
        ax1.set_ylabel("Gain (dB)")
    else:
        ax1.set_ylabel("Fractie doorgelaten")
    ax1.set_title("Butterworth {:.2f}μHz Low-pass Filter {} Frequency Response".format(cutoff, is_log))

    if save:
        suffix = ""
        if poles:
            suffix = "_wPoles"
        export(f"butterworth_frequency_response_{cutoff:.2f}μHz{suffix}")
    return np.array([xf, yf])




if __name__ == "__main__": #zal uitgevoerd worden indien het vanuit deze file gerund wordt
    start_time = time.time()
    plot_instance = Plotting(PATH_TO_DATA, 0)  # Create an instance of the class
    #plot_instance.plot_moving_median(extension = "svg", save=True)  # Call the method properly
    #plot_instance.plot_butterworth(path_to_plot=PATH_TO_PLOT)  # Call the method properly
    print("--- %s seconds for data ---" % (time.time() - start_time))
    #plot_instance.plot_butterworth(extension="svg", phase_shift=False, moving_window=True, interpolation=False, save=True, residuals=False, compare=True)



