import os
import re
import json
import numpy as np
from pathlib import Path

import librosa
import soundfile as sf

from natsort import natsort_keygen

class noise_superposition:

    NOISE_WINDOWS = (0, 1, 2, 3)
    NOISE_SPEEDS = (0, 25, 45, 70)

    VENTILATION_WINDOWS = (0, 1, 2, 3)
    VENTILATION_LEVELS = (0, 2, 4)

    def __init__(self, path, fs=16000):
        self.__path = Path(path)
        self.__fs = fs

        self.__noises = self.__index_noise_recordings()
        self.__ventilation = self.__index_ventilation_recordings()
        # self.__irs = self.__index_irs()

        scripts_dir = Path(__file__).resolve().parents[1]
        gains_file = scripts_dir / "supplementary_material" / "correction_gains" / "gains.json"
        with open(gains_file, "r") as f:
            self.__correction_gains = json.load(f)
    
    def __repr__(self):
        return f"noise_superposition(path={str(self.__path)!r}, fs={self.__fs!r})"
    
    def __str__(self):
        return f"noise_superposition dataset at {self.__path}"
    
    # Define properties/attributes for the class
    @property
    def fs(self):
        """Returns the target sampling frequency."""
        return self.__fs
    
    @fs.setter
    def fs(self, value):
        """Sets the target sampling frequency."""
        self.__fs = value
    
    @property
    def noise_recordings(self):
        """Returns available moving noise recordings keyed by window and speed."""
        return self.__noises
    
    @noise_recordings.setter
    def noise_recordings(self, value):
        raise AttributeError("Cannot set noise_recordings.")
    
    @property
    def ventilation_recordings(self):
        """Returns available stationary ventilation recordings keyed by window and level."""
        return self.__ventilation
    
    @ventilation_recordings.setter
    def ventilation_recordings(self, value):
        raise AttributeError("Cannot set ventilation_recordings.")
    
    @property
    def irs(self):
        """Returns available IR recordings, if an IRs folder is present."""
        return self.__irs
    
    @irs.setter
    def irs(self, value):
        raise AttributeError("Cannot set irs.")

    @property
    def correction_gains(self):
        """Returns correction gains keyed by zero-based microphone index."""
        return self.__correction_gains
    
    @correction_gains.setter
    def correction_gains(self, value):
        raise AttributeError("Cannot set correction_gains.")
    
    def __index_noise_recordings(self):
        """
        Indexes files under:
            Moving/Window_{window}/{speed}MPH/*_chN.wav

        Returns:
            dict: {window: {speed: [ch1_path, ch2_path, ...]}}
        """
        noise_root = self.__path / "Moving"
        recordings = {}

        for window in self.NOISE_WINDOWS:
            recordings[window] = {}

            for speed in self.NOISE_SPEEDS:
                speed_dir = noise_root / f"Window_{window}" / f"{speed}MPH"

                channel_files = self.__find_channel_files(speed_dir, pattern="*_ch*.wav")
                if channel_files:
                    recordings[window][speed] = channel_files

        return recordings
    
    def __index_ventilation_recordings(self):
        """
        Indexes files under:
            Stationary/<date_folder>/W{window}_L{level}_*/ride_output_chN.wav

        Returns:
            dict: {window: {level: [ch1_path, ch2_path, ...]}}
        """
        stationary_root = self.__path / "Stationary"
        recordings = {}

        for window in self.VENTILATION_WINDOWS:
            recordings[window] = {}

            for level in self.VENTILATION_LEVELS:
                session_pattern = f"*/W{window}_L{level}_*"

                matching_sessions = [
                    path for path in stationary_root.glob(session_pattern)
                    if path.is_dir()
                ]

                channel_files = self.__find_channel_files(
                    matching_sessions[0],
                    pattern="ride_output_ch*.wav",
                )

                if channel_files:
                    recordings[window][level] = channel_files

        return recordings
    
    # def __index_irs(self):
    #     """
    #     Indexes future IR files if an IRs folder is later added under the data root.

    #     Expected flexible layout:
    #         IRs/*.wav
    #         or IRs/<condition>/*_chN.wav

    #     Returns:
    #         dict: discovered IR file paths.
    #     """
    #     irs_root = self.__path / "IRs"
    #     irs = {}

    #     wav_files = sorted(irs_root.glob("*.wav")) #, key=self.__path_sort_key)
    #     for wav_file in wav_files:
    #         irs[wav_file.stem] = wav_file

    #     for condition_dir in sorted(irs_root.iterdir()): #, key=self.__path_sort_key):
    #         if not condition_dir.is_dir():
    #             continue

    #         channel_files = self.__find_channel_files(condition_dir, pattern="*_ch*.wav")
    #         if channel_files:
    #             irs[condition_dir.name] = channel_files

    #     return irs
    
    def __find_channel_files(self, folder, pattern):
        files = [path for path in folder.glob(pattern) if path.is_file()]
        return sorted(files) # key=self.__channel_sort_key
    
    # def __path_sort_key(self, path):
    #     natsort_key = natsort_keygen(key=lambda value: value.lower())
    #     return natsort_key(path.name)

    # def __channel_sort_key(self, path):
    #     channel_match = re.search(r"_ch(\d+)\.wav$", path.name)
    #     if channel_match:
    #         return int(channel_match.group(1))
    #     return self.__path_sort_key(path)

    def __normalize_mics(self, mics, n_channels):
        if mics is None:
            return list(range(n_channels))

        if isinstance(mics, int):
            mics = [mics]

        if not isinstance(mics, list) or not all(isinstance(mic, int) for mic in mics):
            raise ValueError("mics must be an integer, a list of integers, or None.")

        if any(mic < 0 or mic >= n_channels for mic in mics):
            raise ValueError(f"mics must be between 0 and {n_channels - 1}.")

        return mics

    def __load_channel_files(self, files, mics=None):
        """
        Loads separate mono channel files and stacks them into one array.

        Assumes all selected channel files have the same duration and sampling frequency.

        Args:
            files (list[Path]): Channel files sorted by channel number.
            mics (int, list[int], optional): Zero-based microphone indices.

        Returns:
            tuple: audio array with shape (N_samples, M_channels), sampling frequency.
        """
        files = sorted(files) # key=self.__channel_sort_key
        selected_mics = self.__normalize_mics(mics, len(files))

        signals = []

        for mic in selected_mics:
            signal, fs_signal = sf.read(files[mic])

            # if signal.ndim > 1:
            #     signal = np.mean(signal, axis=1)

            # if fs_audio is None:
            #     fs_audio = fs_signal

            signals.append(signal)

        audio = np.column_stack(signals)

        if fs_signal != self.fs:
            audio = librosa.resample(audio, orig_sr=fs_signal, target_sr=self.fs, axis=0)
            fs_signal = self.fs

        return audio, fs_signal

    def __apply_correction_gains(self, audio, mics):
        if mics is None:
            mics = list(range(audio.shape[1]))
        elif isinstance(mics, int):
            mics = [mics]

        gains = [self.correction_gains[str(mic)] for mic in mics]
        return audio * np.array(gains)
    
    def load_noise(self, speed, window, mics=None):
        """
        Loads moving noise for a speed/window condition.

        Args:
            speed (int): Speed condition, e.g. 0, 25, 45, 70.
            window (int): Window condition, e.g. 0, 1, 2, 3.
            mics (int, list[int], optional): Zero-based microphone indices.

        Returns:
            tuple: audio array with shape (N_samples, M_channels), sampling frequency.
        """
        if window not in self.__noises:
            raise ValueError(f"Window condition {window} is not available for noise.")

        if speed not in self.__noises[window]:
            raise ValueError(f"Speed condition {speed} is not available for noise with window {window}.")

        return self.__load_channel_files(self.__noises[window][speed], mics=mics)
    
    def load_ventilation(self, level, window, mics=None):
        """
        Loads stationary ventilation for a level/window condition.

        Args:
            level (int): Ventilation level, e.g. 0, 2, 4.
            window (int): Window condition, e.g. 0, 1, 2, 3.
            mics (int, list[int], optional): Zero-based microphone indices.

        Returns:
            tuple: audio array with shape (N_samples, M_channels), sampling frequency.
        """
        if window not in self.__ventilation:
            raise ValueError(f"Window condition {window} is not available for ventilation.")

        if level not in self.__ventilation[window]:
            raise ValueError(f"Ventilation level {level} is not available for ventilation with window {window}.")

        return self.__load_channel_files(self.__ventilation[window][level], mics=mics)

    # def load_ir(self, condition, mics=None):
    #     """
    #     Loads an IR condition if future IR data is added under data/IRs.

    #     Args:
    #         condition (str): IR condition key.
    #         mics (int, list[int], optional): Zero-based microphone indices.

    #     Returns:
    #         tuple: audio array, sampling frequency.
    #     """
    #     if condition not in self.__irs:
    #         raise ValueError(f"IR condition {condition} is not available.")

    #     ir_entry = self.__irs[condition]

    #     if isinstance(ir_entry, list):
    #         return self.__load_channel_files(ir_entry, mics=mics)

    #     ir, fs_ir = sf.read(ir_entry)

    #     if ir.ndim == 1:
    #         ir = ir[:, np.newaxis]

    #     selected_mics = self.__normalize_mics(mics, ir.shape[1])
    #     ir = ir[:, selected_mics]

    #     if fs_ir != self.fs:
    #         ir = librosa.resample(ir, orig_sr=fs_ir, target_sr=self.fs, axis=0)
    #         fs_ir = self.fs

    #     return ir, fs_ir
    
    def get_noise(self, speed, window, mics=None, use_correction_gains=False):
        noise, _ = self.load_noise(speed=speed, window=window, mics=mics)

        if use_correction_gains:
            noise = self.__apply_correction_gains(noise, mics)

        return noise

    def get_ventilation(self, level, window, mics=None, use_correction_gains=False):
        ventilation, _ = self.load_ventilation(level=level, window=window, mics=mics)

        if use_correction_gains:
            ventilation = self.__apply_correction_gains(ventilation, mics)

        return ventilation

    @classmethod
    def match_duration(cls, audio_list, fs):
        """
        Matches all arrays in audio_list to the duration of the first array.

        Longer arrays are truncated. Shorter arrays are looped with a 1-second crossfade.
        Does not mutate the caller's list.
        """
        if len(audio_list) == 1:
            return list(audio_list)

        audio_list = list(audio_list)

        num_columns = audio_list[0].shape[1] if audio_list[0].ndim > 1 else 1
        for audio in audio_list:
            audio_columns = audio.shape[1] if audio.ndim > 1 else 1
            if audio_columns != num_columns:
                raise ValueError("All components must have the same number of columns.")

        def create_sine_cosine_masks(period):
            f = 1 / period
            samples = np.arange(period * fs) / fs
            sine = np.sin(2 * np.pi * f * samples)
            cos = np.cos(2 * np.pi * f * samples)
            return sine[: int(len(samples) / 4)], cos[: int(len(samples) / 4)]

        def match_one_channel(audio, reference_len):
            if len(audio) >= reference_len:
                return audio[:reference_len]

            crossfade_seconds = 1
            crossfade_samples = int(crossfade_seconds * fs)

            if len(audio) <= 2 * crossfade_samples:
                repeats = int(np.ceil(reference_len / len(audio)))
                return np.tile(audio, repeats)[:reference_len]

            sine, cos = create_sine_cosine_masks(4 * crossfade_seconds)

            start = audio[:crossfade_samples]
            middle = audio[crossfade_samples : len(audio) - crossfade_samples]
            end = audio[len(audio) - crossfade_samples :]
            crossfade = start * sine + end * cos

            result = np.concatenate((start, middle, crossfade))
            while len(result) < reference_len:
                result = np.concatenate((result, middle))
                if len(result) + len(crossfade) > reference_len:
                    result = np.concatenate((result, end))
                else:
                    result = np.concatenate((result, crossfade))

            return result[:reference_len]

        first_audio = audio_list[0]
        reference_len = first_audio.shape[0]

        matched_audio = [first_audio[:reference_len]]

        if num_columns == 1:
            for audio in audio_list[1:]:
                matched_audio.append(match_one_channel(audio, reference_len))
            return matched_audio

        for audio in audio_list[1:]:
            matched = np.zeros((reference_len, num_columns))
            for channel_idx in range(num_columns):
                matched[:, channel_idx] = match_one_channel(audio[:, channel_idx], reference_len)
            matched_audio.append(matched)

        return matched_audio
    