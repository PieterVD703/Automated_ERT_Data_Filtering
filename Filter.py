import pandas
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, lfilter, freqz, medfilt, filtfilt, lfilter_zi
from settings.config import PATH_TO_DATA, BASE_DIR, PATH_TO_PLOT
from settings.config import BUTTERWORTH_PARAMS, WINDOW_LENGTH
from Help.Helper_functions import nan_helper, shift, big_number_removal, count_selected_columns, standardise
import os

class Filtering:
    def __init__(self, path_to_data, index):
        self.path_to_data = path_to_data # data uitlezen
        self.dataframe = pandas.read_csv(path_to_data)
        self.dates = self.dataframe.iloc[0:,0].dropna()  # datums; iloc zoekt bepaalde delen op; dropna laat lege waarden weg;
        values = self.dataframe.iloc[0:, shift(index)].map(big_number_removal) # alle metingen excl header en datums, incl nan data
        self.values = standardise(values, index)
        if isinstance(index, int):
            self.index = [index]
        else:
            self.index = index
        #voor vergelijkingen mogelijk te maken
        self.params = BUTTERWORTH_PARAMS
        self.window = WINDOW_LENGTH
    def moving_window(self):
        """
        moving window average eerst, grote gemiddeldes zouden interpolatie verstoren
        !let er wel op dat de Nan data niet als 0 gezien wordt, maar genegeerd wordt.!
        1D median filter -> moving window filter
        zet window_length waarden gesorteerd op een rij, vervangt middelste value met de mediaan, zo worden uitschieters verwijderd
        :return:
        """
        data = self.values
        window = self.window
        values_filtered = np.empty_like(data)

        for meas_id in range(len(self.index)):
            data_column = data.iloc[:, meas_id]
            nan_mask = np.isnan(data_column)  #Nan posities vinden
            #Filter enkel op NaN waarden
            filtered_values = medfilt(data_column[~nan_mask], kernel_size=window)
            #gefilterde waarden terugplaatsen; NaN blijft io plek
            data_column[~nan_mask] = filtered_values
            values_filtered[:, meas_id] = data_column  #terug assignen

        return values_filtered, self.dates

    def interpolate(self):
       """
       :param dates: de datums
       :param values: de resistiviteitsmetingen
       :param indices: welke data er gebruikt wordt
       :return: intergepoleerde waarden voor ontbrekende datapunten
       """
       values_filtered = np.empty_like(self.values) #x,y  van matrix

       for meas_id in range(len(self.index)):
           #checken op instance omdat deze code ook door plotting wordt gebruikt, waardat self.values een numpy array is
           if isinstance(self.values, pandas.DataFrame):
                y = self.values.iloc[:, meas_id].to_numpy()
           elif isinstance(self.values, np.ndarray):
               y = self.values.reshape(-1)
           if any(np.isnan(y)) and not all(np.isnan(y)): #als er ontbrekende data zijn
               nans, x = nan_helper(y)
               y[nans] = np.interp(x=x(nans), xp=x(~nans), fp=y[~nans]) #(~nans) zijn de niet Nan waarden
               values_filtered[:, meas_id] = y
           else:
               return print("no missing data")

       return values_filtered, self.dates

    def butterworth(self, phase_shift = False, interpolation = False): #smoother
        """
        butterworth filtering, is applied to one measurement at a time
        filter can be applied both ways to prevent a phase shift.
        :return: butterworth gefilterde data
        """
        def butter_filter(values, phase_shift):
            """Filtering toepassen, voorwaarts of beide richtingen"""
            params = self.params
            b, a = butter(N=params["order"], Wn=params["cutoff"], btype="low", analog=False)
            if phase_shift:
                zi = lfilter_zi(b, a) * values[0]  # Herschalen naar eerste waarde van x
                filtered_values,_ =  lfilter(b, a, values, zi=zi)
            else:
                filtered_values = filtfilt(b, a, values)
            return filtered_values

        def restore_nans(original_values, filtered_values):
            """Restores NaNs in their original positions after filtering."""
            nan_mask = np.isnan(original_values)
            result = np.full_like(original_values, np.nan)
            result[~nan_mask] = filtered_values
            return result

        if interpolation:
            values, _ = Filtering(self.path_to_data, self.index).interpolate()
        elif not isinstance(self.values, np.ndarray):
            values = self.values.to_numpy()
        else:
            values = self.values

        all_values = np.empty_like(values) #lege array initialiseren

        for meas_id in range(values.shape[1]):
            column_values = values[:, meas_id]
            non_nan_values = column_values[~np.isnan(column_values)] #enkel de gevulde waarden er uit halen
            filtered_values = butter_filter(non_nan_values, phase_shift) #filter toepassen
            all_values[:, meas_id] = restore_nans(column_values, filtered_values) #nans terughalen

        return all_values, self.dates

    def filteren(self, moving_window = False, interpolate = False, butterworth = False, export = False):
        """
        Deze functie voegt alles tesamen
        :param moving_window:
        :param interpolate:
        :param butterworth:
        :param export: of er een csv file met de geëxporteerde data moet gemaakt worden of niet
        :return: gefilterde values en hun data
        """
        suffix = ""
        params = BUTTERWORTH_PARAMS
        cutoff = params["cutoff"] / float(2) * params["fs"] * float(10 ** 6)  # terug omzetten van relatieve cutoff naar echte
        path_to_plot = PATH_TO_PLOT
        if moving_window is True:
            self.values, self.dates  = self.moving_window()
            suffix += "_mw"
        if interpolate is True:
            self.values, self.dates = self.interpolate()
            suffix += "_interp"
        if butterworth is True:
            self.values, self.dates = self.butterworth()
            suffix += f"_butter{cutoff:.2f}μHz"
        if export is True: #exporteren naar nieuwe csv
            fname = os.path.join(
                path_to_plot,
                f"data{self.index}{suffix}.csv"
            )
            # locatie en naam waar naartoe geschreven wordt
            header = "Date," + ",".join(map(str, self.index))
            data_to_save = np.column_stack((self.dates.to_numpy(), self.values))

            np.savetxt(fname, data_to_save, delimiter=",", header=header, comments="", fmt="%s")
        else:
            return self.dates, self.values,


if __name__ == "__main__":
    start_time = time.time()
    filtering_instance = Filtering(PATH_TO_DATA, 0)  # Create an instance
    #filtered_values, dates = filtering_instance.moving_window()
    #filtered_values, dates = filtering_instance.interpolate()
    #filtered_values, dates = filtering_instance.butterworth(phase_shift=False, interpolation=False)  # Call the method
    #np.savetxt("raw", filtered_values, delimiter=",")
    filtering_instance.filteren(moving_window=False, interpolate=False, butterworth=False, export=True)
    print("--- %s seconds for data ---" % (time.time() - start_time))

