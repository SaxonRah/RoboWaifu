import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import blackman
from scipy.signal import butter, filtfilt
import time
import json
import os
from datetime import datetime


class ToneGenerator:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.stream = None

    def generate_tone(self, frequency, duration=0.5):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        envelope = np.ones_like(t)
        attack = int(0.005 * self.sample_rate)
        decay = int(0.01 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)
        tone = 0.3 * np.sin(2 * np.pi * frequency * t) * envelope
        return tone.astype(np.float32)

    def play_sequence(self, frequencies):
        with sd.OutputStream(channels=1, samplerate=self.sample_rate) as stream:
            for freq in frequencies:
                tone = self.generate_tone(freq)
                stream.write(tone)
                time.sleep(0.1)


class BassRecorder:
    def __init__(self):
        self.last_detection_time = 0
        self.last_detected_note = None
        # self.detection_cooldown = 0.125  # Increased cooldown
        self.detection_cooldown = 0.05125  # Increased cooldown
        self.recorded_notes = []
        self.recording = False
        self.min_note_duration = 0.1  # Minimum duration for a valid note
        self.noise_floor = 0.02  # Minimum amplitude threshold

        # Create bandpass filter for bass frequency range
        self.nyquist = SAMPLING_RATE / 2
        self.lowcut = 35  # Hz, just below lowest bass note
        self.highcut = 450  # Hz, just above highest bass note
        self.filter_order = 4

        # Calculate harmonic tolerance windows
        self.harmonic_windows = self._calculate_harmonic_windows()

    def _calculate_harmonic_windows(self):
        """Calculate frequency windows for fundamental and harmonics of each note."""
        windows = {}
        for note, freq in FREQUENCIES.items():
            # Store fundamental and first two harmonics with tolerance ranges
            windows[note] = [
                (freq * i - FREQ_TOLERANCE, freq * i + FREQ_TOLERANCE)
                for i in range(1, 4)  # Fundamental and 2 harmonics
            ]
        return windows

    def _design_bandpass_filter(self):
        """Design butterworth bandpass filter coefficients."""
        b, a = butter(self.filter_order,
                        [self.lowcut / self.nyquist, self.highcut / self.nyquist],
                        btype='band',
                        output='ba')
        return b, a

    @staticmethod
    def get_closest_frequency(frequency, min_confidence=0.8):
        """Get the closest bass note frequency with confidence check."""
        closest_note = None
        min_diff = float('inf')
        confidence = 0

        for note, freq in FREQUENCIES.items():
            diff = abs(freq - frequency)
            rel_diff = diff / freq  # Relative difference

            if rel_diff < min_diff:
                min_diff = rel_diff
                closest_note = note
                # Calculate confidence based on relative difference
                confidence = max(0, 1 - (rel_diff * 10))

        if confidence < min_confidence:
            return None, None

        return closest_note, FREQUENCIES[closest_note]

    def _check_harmonics(self, fft_freqs, fft_magnitude, fundamental_freq, note):
        """Verify note by checking presence of harmonics."""
        windows = self.harmonic_windows[note]
        harmonic_strength = []

        for low, high in windows:
            mask = (fft_freqs >= low) & (fft_freqs <= high)
            if np.any(mask):
                peak_in_window = np.max(fft_magnitude[mask])
                harmonic_strength.append(peak_in_window)

        # Verify harmonic relationships
        if len(harmonic_strength) >= 2:
            # First harmonic should be strong relative to fundamental
            h1_ratio = harmonic_strength[1] / harmonic_strength[0]
            if 0.1 <= h1_ratio <= 1.5:  # Typical range for bass guitar
                return True
        return False

    def process_audio_block(self, indata):
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            return None

        # Check if signal is above noise floor
        if np.max(np.abs(indata)) < self.noise_floor:
            return None

        # Apply bandpass filter
        b, a = self._design_bandpass_filter()
        audio_data = filtfilt(b, a, indata[:, 0])

        # Apply window function
        audio_data = audio_data * blackman(len(audio_data))

        # Compute FFT with zero padding for better frequency resolution
        fft_size = BLOCK_SIZE * 4  # Increased FFT size for better resolution
        fft_data = np.fft.rfft(audio_data, n=fft_size)
        fft_freqs = np.fft.rfftfreq(fft_size, d=1 / SAMPLING_RATE)
        fft_magnitude = np.abs(fft_data)

        # Normalize magnitude
        fft_magnitude = fft_magnitude / np.max(fft_magnitude)

        # Find peaks in the bass frequency range
        bass_range_mask = (fft_freqs >= self.lowcut) & (fft_freqs <= self.highcut)
        peaks, properties = find_peaks(
            fft_magnitude[bass_range_mask],
            height=THRESHOLD,
            distance=int(fft_size / 256),  # Increased minimum peak distance
            prominence=0.1  # Add prominence requirement
        )

        if peaks.size == 0:
            return None

        # Get frequencies and magnitudes of peaks
        peak_freqs = fft_freqs[bass_range_mask][peaks]
        peak_magnitudes = properties['peak_heights']

        # Find the strongest peak
        max_peak_idx = np.argmax(peak_magnitudes)
        dominant_freq = peak_freqs[max_peak_idx]

        # Get closest note with confidence check
        note, freq = self.get_closest_frequency(dominant_freq, min_confidence=0.8)

        if note is None:
            return None

        # Verify harmonics
        if not self._check_harmonics(fft_freqs, fft_magnitude, freq, note):
            return None

        # Additional validation passed, update detection
        if note != self.last_detected_note:
            self.last_detection_time = current_time
            self.last_detected_note = note
            if self.recording:
                self.recorded_notes.append({
                    'note': note,
                    'frequency': freq,
                    'timestamp': time.time()
                })
            return {'note': note, 'frequency': freq}
        return None

    def save_recording(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bass_recording_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.recorded_notes, f)
        return filename


calculated_frequencies = {
    'E0': 41.203, 'E1': 43.653057876886045, 'E2': 46.2488037764911, 'E3': 48.998900759457115, 'E4': 51.912527018818466,
    'E5': 54.99940651136793, 'E6': 58.269841410458646, 'E7': 61.734746488549916, 'E8': 65.40568554424573,
    'E9': 69.2949099953976, 'E10': 73.4153997670728, 'E11': 77.78090661084764, 'E12': 82.406,
    'E13': 87.30611575377209, 'E14': 92.4976075529822, 'E15': 97.99780151891423, 'E16': 103.82505403763693,
    'E17': 109.99881302273586, 'E18': 116.53968282091729, 'E19': 123.46949297709983, 'E20': 130.81137108849146,
    'E21': 138.5898199907952, 'E22': 146.8307995341456, 'E23': 155.56181322169527, 'E24': 164.812,

    'A0': 55.0, 'A1': 58.27047018976124, 'A2': 61.735412657015516, 'A3': 65.40639132514966, 'A4': 69.29565774421802,
    'A5': 73.4161919793519, 'A6': 77.78174593052023, 'A7': 82.40688922821748, 'A8': 87.30705785825097,
    'A9': 92.4986056779086, 'A10': 97.99885899543732, 'A11': 103.82617439498628, 'A12': 110.0,
    'A13': 116.54094037952248, 'A14': 123.47082531403103, 'A15': 130.8127826502993, 'A16': 138.59131548843604,
    'A17': 146.8323839587038, 'A18': 155.56349186104046, 'A19': 164.81377845643496, 'A20': 174.61411571650194,
    'A21': 184.9972113558172, 'A22': 195.99771799087463, 'A23': 207.65234878997256, 'A24': 220.0,

    'D0': 73.42, 'D1': 77.78578038785946, 'D2': 82.41116358687417, 'D3': 87.31158638349979, 'D4': 92.50340348328159,
    'D5': 98.00394209316393, 'D6': 103.83155974943264, 'D7': 110.00570558428596, 'D8': 116.5469852355052,
    'D9': 123.47722961585544, 'D10': 130.81956777172744, 'D11': 138.59850407417989, 'D12': 146.84,
    'D13': 155.57156077571892, 'D14': 164.82232717374833, 'D15': 174.62317276699957, 'D16': 185.00680696656318,
    'D17': 196.00788418632786, 'D18': 207.66311949886529, 'D19': 220.01141116857193, 'D20': 233.09397047101044,
    'D21': 246.95445923171087, 'D22': 261.63913554345487, 'D23': 277.19700814835977, 'D24': 293.68,

    'G0': 98.0, 'G1': 103.82738324721095, 'G2': 110.00128073431856, 'G3': 116.54229727026666, 'G4': 123.47226288969757,
    'G5': 130.81430570866337, 'G6': 138.59292911256333, 'G7': 146.8340935339148, 'G8': 155.56530309288354,
    'G9': 164.81569738972803, 'G10': 174.6161487555065, 'G11': 184.99936528561193, 'G12': 196.0,
    'G13': 207.6547664944219, 'G14': 220.0025614686371, 'G15': 233.08459454053332, 'G16': 246.94452577939515,
    'G17': 261.62861141732674, 'G18': 277.18585822512665, 'G19': 293.6681870678296, 'G20': 311.13060618576714,
    'G21': 329.63139477945606, 'G22': 349.232297511013, 'G23': 369.99873057122386, 'G24': 392.0
 }


def calculate_fret_frequencies():
    base_frequencies = {
        "E1": 41.203,
        "A1": 55.000,
        "D2": 73.420,
        "G2": 98.000
    }

    frequencies = {}
    for string, base_freq in base_frequencies.items():
        for fret in range(25):
            freq = base_freq * (2 ** (fret / 12))
            note_name = f"{string[0]}{fret}"
            frequencies[note_name] = freq
    print(frequencies)
    return frequencies


# Audio parameters
# FREQUENCIES = calculate_fret_frequencies()
FREQUENCIES = calculated_frequencies
SAMPLING_RATE = 48000
# BLOCK_SIZE = 4096
BLOCK_SIZE = 2048
THRESHOLD = 0.1
FREQ_TOLERANCE = 1.5


def record_bass():
    """Start recording bass notes."""
    recorder = BassRecorder()
    recorder.recording = True

    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        result = recorder.process_audio_block(indata)
        if result:
            print(f"Detected: {result['note']} ({result['frequency']:.1f} Hz)")

    print("\nRecording... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=SAMPLING_RATE,
                blocksize=BLOCK_SIZE
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        filename = recorder.save_recording()
        print(f"\nRecording saved to: {filename}")


def play_recording(filename):
    """Play back a recorded sequence."""
    try:
        with open(filename, 'r') as f:
            recording = json.load(f)

        frequencies = [note['frequency'] for note in recording]
        generator = ToneGenerator()
        print(f"\nPlaying recording from {filename}...")
        generator.play_sequence(frequencies)
        print("Playback complete.")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not a valid recording file.")


def list_recordings():
    """List all available recordings."""
    recordings = [f for f in os.listdir('.') if f.startswith('bass_recording_') and f.endswith('.json')]
    if not recordings:
        print("\nNo recordings found.")
        return None

    print("\nAvailable recordings:")
    for i, recording in enumerate(recordings, 1):
        print(f"{i}. {recording}")

    return recordings


def main_menu():
    while True:
        print("\nBass Guitar Note System")
        print("1. Start Recording")
        print("2. Play Recording")
        print("3. Exit")

        choice = input("Choose an option (1-3): ")

        if choice == '1':
            record_bass()
        elif choice == '2':
            recordings = list_recordings()
            if recordings:
                try:
                    idx = int(input("\nEnter recording number to play: ")) - 1
                    if 0 <= idx < len(recordings):
                        play_recording(recordings[idx])
                    else:
                        print("Invalid recording number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main_menu()
