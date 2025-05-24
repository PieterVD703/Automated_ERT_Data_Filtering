import os
"""
hierin alle variabelen aanpassen.
"""
#paden apart definiëren voor netheid
#huidige directory aangeven
current = os.path.dirname(os.path.abspath(__file__))  #directory van config.py
BASE_DIR = os.path.dirname(current)  #een stapje omhoog

#locaties van de data
PATH_TO_DATA = os.path.join(BASE_DIR, "data", "R_1yr.csv")
PATH_TO_QUADRUPOLES = os.path.join(BASE_DIR, "data", "quadrupolen.csv")
PATH_TO_PLOT = os.path.join(BASE_DIR, "Figuren")
PATH_TO_PICKLE = os.path.join(BASE_DIR, "pickles")
PATH_TO_INVERSION_OUTPUT = os.path.join(BASE_DIR, "inversion_output")

#inversie parameters, voor toekomst
INVERSION_PARAMS = {
    "max_iterations": 1000,
    "tolerance": 1e-6,
    "method": "least_squares"
}

"""
FILTERING
"""
#moving median
WINDOW_LENGTH: int = 3
days = 3 #voor het gemak; hoeveel dagen de cutoff frequentie is, minimum 2!
assert days > 2, "tijdseenheid niet boven Nyquist frequentie"
cutoff = 1/(3600*24*2)*2/days  # desired cutoff frequency of the filter, Hz, smaller means more smoothing
fs = 1 / (3600*24) # sampling frequency, Hz
assert 0 < cutoff/(0.5*fs) <= 1, "invalid cutoff"
#butterworth
BUTTERWORTH_PARAMS = {
    "days": days,
    "order": 2 ,
    "fs": fs,    # sample rate, Hz
    "cutoff": cutoff/(0.5*fs)  # desired cutoff frequency of the filter, Hz
    #meetfrequentie is 1/3600, 3.667 / 30 -> 30 is nyquist frequency, 3.667 is 12% ervan
}

# Pickle filename
#pickling is het proces waarbij data wordt opgeslaan naar een file, toekomst
PICKLE_NAME = "inversion_results.pkl"

