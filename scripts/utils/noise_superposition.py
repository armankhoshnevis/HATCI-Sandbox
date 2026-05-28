import os
import json
import numpy as np
from pathlib import Path

import librosa
import soundfile as sf

from natsort import natsort_keygen

class noise_superposition:
    def __init__(self, path, fs=16000, condition=None):
        # Initialize the class
        self.__path = path
        self.__fs = fs
        info_file = os.path.join(self.__path, 'info.json')
        with open(info_file, 'r') as f:
            info_dict = json.load(f)
        self.__in_dict = info_dict
        self.__make = info_dict['make']
        self.__model = info_dict['model']
        self.__year = info_dict['year']

        self.__mic_setups = self.__find_folders(condition)

        # Load the available noise and ventilation conditions for each microphone setup
        self.__noises = {}
        self.__ventilation = {}
        natsort_key = natsort_keygen(key=lambda y: y.lower())
        for mic_setup in self.__mic_setups:
            noise_list = os.listdir(os.path.join(self.__path, mic_setup, 'noise'))
            ventilation_list = os.listdir(os.path.join(self.__path, mic_setup, 'ventilation'))

            self.__noises[mic_setup] = sorted([wav[:-4] for wav in noise_list], key=natsort_key)
            self.__ventilation[mic_setup] = sorted([wav[:-4] for wav in ventilation_list], key=natsort_key)
        
        # Load the correction gains for all microphones
        # gains_file = os.path.join('supplementary_material', 'correction_gains', 'gains.json')
        scripts_dir = Path(__file__).resolve().parents[1]
        gains_file = scripts_dir / "supplementary_material" / "correction_gains" / "gains.json"
        # TODO: Update gains_file to its simpler version in future
        with open(gains_file, 'r') as f:
            self.__correction_gains = json.load(f)
    
    def __repr__(self):
        return f'Car(path={self.__path!r}, info_dict={self.__in_dict!r})'
    
    def __str__(self):
        return f'{self.make}, {self.model}, {self.year}'
    
    # Define properties/attributes for the class
    @property
    def fs(self):
        """Returns the sampling frequency."""
        return self.__fs
    
    @fs.setter
    def fs(self, value):
        """Sets the sampling frequency."""
        self.__fs = value
    
    @property
    def make(self):
        """Returns a string containing the make of the car."""
        return self.__make
    
    @make.setter
    def make(self, value):
        """Prevents setting the make of the car."""
        raise AttributeError('Cannot set make.')
    
    @property
    def model(self):
        """Returns a string containing  the model of the car."""
        return self.__model
    
    @model.setter
    def model(self, value):
        """Prevents setting the model of the car."""
        raise AttributeError('Cannot set model.')
    
    @property
    def year(self):
        """Returns an integer with the year of the car."""
        return self.__year
    
    @year.setter
    def year(self, value):
        """Prevents setting the year of the car."""
        raise AttributeError('Cannot set year.')

    @property
    def mic_setups(self):
        """Returns a list of available microphone configurations."""
        return self.__mic_setups
    
    @mic_setups.setter
    def mic_setups(self, value):
        """Prevents setting the microphone configurations."""
        raise AttributeError('Cannot set mic_setups.')
    
    @property
    def noise_recordings(self):
        """Returns a dictionary of available noise condition recordings per microphone configuration."""
        return self.__noises
    
    @noise_recordings.setter
    def noise_recordings(self, value):
        """Prevents setting the noise recordings."""
        raise AttributeError('Cannot set noise_recordings.')
    
    @property
    def ventilation_recordings(self):
        """Returns a dictionary of available ventilation conditions recordings per microphone configuration."""
        return self.__ventilation
    
    @ventilation_recordings.setter
    def ventilation_recordings(self, value):
        """Prevents setting the ventilation recordings."""
        raise AttributeError('Cannot set ventilation_recordings.')
    
    @property
    def correction_gains(self):
        """Returns a dictionary with the correction gains of all microphones."""
        return self.__correction_gains
    
    @correction_gains.setter
    def correction_gains(self, value):
        """Prevents setting the correction gains."""
        raise AttributeError('Cannot set correction_gains.')
    
    # Private method
    def __find_folders(self, mic_setup=None):
        """
        This function returns a list of folders in self.__path that match a given condition.

        Parameters:
        mic_setup (str, optional): The microphone setup that folder names must match. If no mic_setup is provided, all folders in the path are returned.

        Returns:
        list: A list of folder names that match the mic_setup. If no folders match the mic_setup, a message is printed and None is returned.
        """
        items = [f for f in os.listdir(self.__path) if os.path.isdir(os.path.join(self.__path, f))]
        if not mic_setup:
            return items
        
        cond_items = [item for item in items if os.path.isdir(os.path.join(self.__path, item)) and mic_setup in item]
        if cond_items:
            return cond_items
        
        else:
            print(f"No folders found with {mic_setup} in their names.")
            return None
    
    # Instance methods
    def load_noise(self, mic_setup, condition, mic_range=list(range(8))):  # TODO: Change to 4 for the HATCI project
        """
        Loads the noise recording channels for a given microphone setup and noise condition.

        Args:
            mic_setup (str): The microphone setup to load the noise recording for.
            condition (str): The specific noise condition to load ("speed condition_window condition").
            mic_range (list, optional): The range of microphone channels to load. Defaults to list(range(8)).

        Returns:
            tuple: A tuple containing the noise data as a NumPy array (N_samples x M_channels) and the sampling frequency of noise recording.

        Raises:
            ValueError: If the given noise condition is not available for the given microphone setup.
        """
        noise_path = os.path.join(self.__path, mic_setup, 'noise', condition + '.wav')

        noise, fs_noise = sf.read(noise_path)
        
        # Resample
        if fs_noise != self.fs:
            noise = librosa.resample(noise, orig_sr=fs_noise, target_sr=self.fs, axis=0)
            fs_noise = self.fs
        
        return noise[:, mic_range], fs_noise
    
    def load_ventilation(self, mic_setup, condition, mic_range=list(range(8))):
        """
        Loads the ventilation recording for a given microphone setup and condition.
        
        Args:
            mic_setup (str): The microphone setup to load the ventilation recording for.
            condition (str): The specific ventilation condition to load ("ventilation level_window condition").
            mic_range (list, optional): The range of microphone channels to load. Defaults to list(range(8)).
        
        Returns:
            tuple: A tuple containing the ventilation data as a NumPy array (N_samples x M_channels) and the sampling frequency.
        
        Raises:
            ValueError: If the given ventilation condition is not available for the given microphone setup.
        """
        ventilation_path = os.path.join(self.__path, mic_setup, 'ventilation', condition + '.wav')  

        ventilation, fs_ventilation = sf.read(ventilation_path)

        # Resample
        if fs_ventilation != self.fs:
            ventilation = librosa.resample(ventilation, orig_sr=fs_ventilation, target_sr=self.fs, axis=0)
            fs_ventilation = self.fs
        
        return ventilation[:, mic_range], fs_ventilation
    
    def get_noise(self, mic_setup:str, speed:int, window:int, version:str=None, mics=None, use_correction_gains=True):
        """
        Retrieves the in-motion noise recording for a given microphone setup, condition, and microphone index.
        
        Args:
            mic_setup (str): The microphone setup to use.
            speed (int): The speed condition.
            window (int): The window condition.
            version (str, optional): The version of the noise recording in case there are multiple versions. Defaults to None. Must be "ver1", "ver2", etc or "coarse". 
            mics (int or list of int, optional): The microphone index or a list of microphone indices to use. Defaults to None. If mics is None, all microphones are used.
            use_correction_gains (bool, optional): A boolean indicating whether to use the correction gains. Defaults to True.
        
        Returns:
            numpy.ndarray: The processed noise signal.
        
        Raises:
            ValueError: If the specified microphone setup is not available.
            ValueError: If mics is not an integer or a list of integers.
        """
        if mic_setup not in self.mic_setups:
            raise ValueError(f"Microphone setup {mic_setup} is not available.")
        
        if not (isinstance(mics, list) and all(isinstance(item, int) for item in mics)) and not isinstance(mics, int) and mics is not None:
            raise ValueError(f"mics must be an integer or a list of integers.")
        
        if window not in [0, 1, 2, 3]:
            raise ValueError(f"Window condition in condition must be 0, 1, 2 or 3.")
        
        condition = f's{speed}_w{window}'
        if version:
            condition += f'_{version}'
        
        noise, _ = self.load_noise(mic_setup, condition)

        if mics is None:
            mics = list(range(noise.shape[1]))
        if not isinstance(mics, list):
            mics = [mics]
        noise = noise[:, mics]
        
        # Apply correction gains
        if use_correction_gains:
            gains = [self.correction_gains[str(mic)] for mic in mics]
            noise = noise * np.array(gains)
        
        return noise

    def get_ventilation(self, mic_setup: str, level: int, window:int, version:str=None, mics=None, use_correction_gains=True):
        """
        Retrieves and processes the ventilation recording for a given microphone setup, condition, and ventilation level.
        
        Args:
            mic_setup (str): The microphone setup to use.
            window (int): The window condition.
            level (int): The ventilation level (must be 1, 2, or 3).
            version (str, optional): The version of the ventilation recording in case there are multiple versions. Defaults to None. Must be "ver1", "ver2". 
            mics (int or list of int, optional): The microphone index or a list of microphone indices to use. Defaults to None. If mics is None, all microphones are used.
            use_correction_gains (bool, optional): A boolean indicating whether to use the correction gains. Defaults to True.
        
        Returns:
        numpy.ndarray: The processed ventilation signal for the specified microphones.

        Raises:
            ValueError: If the microphone setup is not available.
            ValueError: If the ventilation level is not 1, 2, or 3.
            ValueError: If the window condition is invalid.
            ValueError: If mics is not an integer or a list of integers.
        """
        if mic_setup not in self.mic_setups:
            raise ValueError(f"Microphone setup {mic_setup} is not available.")
        
        if level not in [1, 2, 3]:
            raise ValueError(f"Ventilation level must be 1, 2 or 3.")
        
        if window not in [0, 1, 2, 3]:
            raise ValueError(f"Window condition must be 0, 1, 2 or 3.")
        
        if not (isinstance(mics, list) and all(isinstance(item, int) for item in mics)) and not isinstance(mics, int) and mics is not None:
            raise ValueError(f"mics must be an integer or a list of integers.")
        
        ventilation_condition = f'v{level}_w{window}'
        if version:
            ventilation_condition += f'_{version}'
        ventilation, _ = self.load_ventilation(mic_setup, ventilation_condition)
        
        if mics is None:
            mics = list(range(ventilation.shape[1]))
        if not isinstance(mics, list):
            mics = [mics]
        ventilation = ventilation[:, mics]
        
        if use_correction_gains:
            gains = [self.correction_gains[str(mic)] for mic in mics]
            ventilation = ventilation * np.array(gains)
        return ventilation
    
    @classmethod
    def match_duration(cls, audio_list: list, fs):
        """
        Matches the duration of elements in the list `audio_list` to the duration of the first element.

        This method adjusts the duration of each element in the list `audio_list` to match the duration 
        of the first element in the list. If an element is longer than the first element, it is 
        truncated. If an element is shorter, it is looped using crossfading to match the duration.

        Args:
            audio_list (list): A list of numpy arrays where each array represents a signal.
            fs (int): The sampling frequency of the signals.

        Returns:
            list: A list of numpy arrays with matched durations.

        Notes:
            - If the list `audio_list` contains only one element, it is returned as is.
            - Crossfading is used to loop shorter elements. The crossfade duration is set to 1 second.
            - The crossfade is achieved using the first quarter of sine and cosine functions.
        """
        if len(audio_list) == 1:
            return audio_list
        
        # Check if all elements have the same number of columns (channels)
        num_columns = audio_list[0].shape[1] if len(audio_list[0].shape) > 1 else 1
        for audio in audio_list:
            if (len(audio.shape) > 1 and audio.shape[1] != num_columns) or (len(audio.shape) == 1 and num_columns != 1):
                raise ValueError("All components must have the same number of columns.")
        
        def create_sine_cosine_masks(period):
            '''Create the first quarter of sine and cosine functions for crossfading. 
            Period must be 4*crossfade_seconds.
            '''
            f = 1 / period
            samples = np.arange(period * fs) / fs
            sine = np.sin(2 * np.pi * f * samples)
            cos = np.cos(2 * np.pi * f * samples)
            
            # Return only the first quarter of the sine and cosine functions
            return sine[:int(len(samples)/4)], cos[:int(len(samples)/4)]
        
        first_audio = audio_list.pop(0)
        if num_columns == 1:
            for i, element in enumerate(audio_list):
                if len(element) >= len(first_audio):
                    audio_list[i] = element[:len(first_audio)]
                elif len(element) < len(first_audio):
                    crossfade_seconds = 1
                    crossfade_samples = int(crossfade_seconds*fs)

                    # Create the sine and cosine masks for crossfading
                    sine, cos = create_sine_cosine_masks(4*crossfade_seconds)

                    start = element[0:crossfade_samples]
                    middle = element[crossfade_samples:len(element)-crossfade_samples]
                    end = element[len(element)-crossfade_samples:]
                    cf = start * sine + end * cos

                    result = np.concatenate((start, middle, cf))
                    while len(result) < len(first_audio):
                        result = np.concatenate((result, middle))
                        if len(result) + len(cf) > len(first_audio):
                            result = np.concatenate((result, end))
                        else:
                            result = np.concatenate((result, cf))
                    
                    # Cut result to match the length of first_audio
                    audio_list[i] = result[:len(first_audio)]
            audio_list.insert(0, first_audio)
            return audio_list
        
        else:
            reference_len = len(first_audio[:, 0])
            updated_audio_list = []
            for i, audio in enumerate(audio_list):
                if len(audio[:, 0]) >= reference_len:
                    updated_audio_list.append(audio[:reference_len, :])
                
                elif len(audio[:, 0]) < reference_len:
                    crossfade_seconds = 1
                    crossfade_samples = int(crossfade_seconds*fs)
                    sine, cos = create_sine_cosine_masks(4*crossfade_seconds)
                    
                    crossfaded_mic = np.zeros((reference_len, num_columns))
                    for j in range(num_columns):
                        start = audio[0:crossfade_samples, j]
                        middle = audio[crossfade_samples:len(audio)-crossfade_samples, j]
                        end = audio[len(audio)-crossfade_samples:, j]
                        cf = start * sine + end * cos 
                    
                        result = np.concatenate((start, middle, cf))
                        while len(result) < reference_len:
                            result = np.concatenate((result, middle), axis=0)
                            if len(result) + len(cf) > len(first_audio):
                                result = np.concatenate((result, end), axis=0)
                            else:
                                result = np.concatenate((result, cf), axis=0)
                        
                        crossfaded_mic[:, j] = result[:reference_len]
                    updated_audio_list.append(crossfaded_mic)
            updated_audio_list.insert(0, first_audio)
            return updated_audio_list
