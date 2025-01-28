import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt, savgol_filter
import time
import pyttsx3


class VocalSynthesizer:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.stream = None

        # Vibrato parameters
        self.vibrato_rate = 5.0  # Hz
        self.vibrato_depth = 0.03  # Pitch variation amount

        # Prosody parameters
        self.pitch_range = (80, 400)  # Hz
        self.sentence_pitch_contour = {
            'statement': [(0, 1.0), (0.5, 0.9), (1.0, 0.7)],  # Falling intonation
            'question': [(0, 0.8), (0.5, 1.0), (1.0, 1.2)],  # Rising intonation
        }

        # Enhanced formant transition time
        self.transition_time = 0.03  # seconds

        # Glottal pulse model parameters
        self.open_quotient = 0.6  # Proportion of cycle vocal folds are open
        self.return_quotient = 0.35  # Speed of return phase

        # Comprehensive formant frequencies for all vowels (in Hz)
        self.vowel_formants = {
            # Basic vowels
            '': [(0, 0), (0, 0), (0, 0)],
            'a': [(800, 1200), (1200, 2500), (2500, 3500)],  # 'ah' as in father
            'e': [(400, 2300), (2300, 2800), (2800, 3400)],  # 'ee' as in see
            'i': [(300, 2200), (2200, 2700), (2700, 3300)],  # 'ih' as in bit
            'o': [(450, 800), (800, 1200), (2200, 3000)],  # 'oh' as in go
            'u': [(300, 800), (800, 1200), (2200, 3000)],  # 'oo' as in boot

            # Additional vowels
            'ɑ': [(750, 1200), (1200, 2500), (2500, 3500)],  # 'ah'
            'æ': [(660, 1700), (1700, 2400), (2400, 3300)],  # 'ae' as in cat
            'ʌ': [(600, 1200), (1200, 2000), (2000, 2800)],  # 'uh' as in cup
            'ɔ': [(500, 900), (900, 1500), (2200, 3000)],  # 'aw' as in law
            'ə': [(500, 1500), (1500, 2500), (2500, 3200)],  # schwa as in about
            'ɛ': [(550, 1800), (1800, 2500), (2500, 3300)],  # 'eh' as in bed
            'ɪ': [(400, 1900), (1900, 2500), (2500, 3300)],  # 'ih' as in bit
            'ʊ': [(450, 1000), (1000, 1600), (2200, 3000)],  # 'uu' as in foot

            # Diphthongs
            'aɪ': [(800, 1200), (1200, 2500), (2500, 3500)],  # 'ai' as in price
            'aʊ': [(700, 1100), (1100, 2300), (2300, 3300)],  # 'ow' as in mouth
            'eɪ': [(450, 2300), (2300, 2800), (2800, 3400)],  # 'ay' as in face
            'oʊ': [(450, 800), (800, 1200), (2200, 3000)],  # 'oh' as in goat
            'ɔɪ': [(500, 900), (900, 1500), (2200, 3000)],  # 'oy' as in choice

            # R-colored vowels
            'ɑr': [(750, 1100), (1100, 2300), (2300, 3300)],  # 'ar' as in start
            'ɛr': [(550, 1800), (1800, 2500), (2500, 3300)],  # 'er' as in ferry
            'ɪr': [(400, 1900), (1900, 2500), (2500, 3300)],  # 'ir' as in near
            'ɔr': [(500, 900), (900, 1500), (2200, 3000)],  # 'or' as in north
            'ʊr': [(450, 1000), (1000, 1600), (2200, 3000)],  # 'ur' as in cure
        }

        # Enhanced consonant characteristics
        self.consonant_params = {
            'pause': {'noise_band': (10, 100), 'duration': 0.1, 'type': 'noise'},
            # Fricatives (complete coverage)
            's': {'noise_band': (4000, 8000), 'duration': 0.1, 'type': 'noise'},
            'z': {'noise_band': (4000, 8000), 'duration': 0.1, 'type': 'noise', 'voiced': True},
            'f': {'noise_band': (3000, 7000), 'duration': 0.08, 'type': 'noise'},
            'v': {'noise_band': (3000, 7000), 'duration': 0.08, 'type': 'noise', 'voiced': True},
            'h': {'noise_band': (1000, 4000), 'duration': 0.08, 'type': 'noise'},
            'sh': {'noise_band': (2500, 6000), 'duration': 0.1, 'type': 'noise'},
            'zh': {'noise_band': (2500, 6000), 'duration': 0.1, 'type': 'noise', 'voiced': True},
            'th': {'noise_band': (1500, 5000), 'duration': 0.09, 'type': 'noise'},
            'dh': {'noise_band': (1500, 5000), 'duration': 0.09, 'type': 'noise', 'voiced': True},
            'c': {'noise_band': (2000, 6000), 'duration': 0.08, 'type': 'noise'},  # 'c' as in city
            'x': {'noise_band': (2500, 6500), 'duration': 0.09, 'type': 'noise'},  # 'x' as in fix
            'q': {'noise_band': (1500, 4000), 'duration': 0.05, 'type': 'plosive'},  # 'q' as in queen

            # Plosives (expanded)
            't': {'noise_band': (2000, 6000), 'duration': 0.05, 'type': 'plosive'},
            'd': {'noise_band': (2000, 6000), 'duration': 0.05, 'type': 'plosive', 'voiced': True},
            'k': {'noise_band': (1500, 4000), 'duration': 0.05, 'type': 'plosive'},
            'g': {'noise_band': (1500, 4000), 'duration': 0.05, 'type': 'plosive', 'voiced': True},
            'p': {'noise_band': (800, 2000), 'duration': 0.05, 'type': 'plosive'},
            'b': {'noise_band': (800, 2000), 'duration': 0.05, 'type': 'plosive', 'voiced': True},

            # Approximants (expanded)
            'l': {'formants': [(300, 1200), (1200, 2500)], 'duration': 0.08, 'type': 'resonant'},
            'r': {'formants': [(400, 1300), (1300, 2000)], 'duration': 0.08, 'type': 'resonant'},
            'w': {'formants': [(200, 800), (800, 1500)], 'duration': 0.08, 'type': 'resonant'},
            'y': {'formants': [(250, 900), (2300, 3000)], 'duration': 0.08, 'type': 'resonant'},
            'j': {'formants': [(250, 900), (2300, 3000)], 'duration': 0.08, 'type': 'resonant'},

            # Nasals (expanded)
            'm': {'formants': [(250, 800), (1000, 2000)], 'duration': 0.08, 'type': 'resonant'},
            'n': {'formants': [(300, 900), (1500, 2500)], 'duration': 0.08, 'type': 'resonant'},
            'ng': {'formants': [(350, 950), (1700, 2700)], 'duration': 0.08, 'type': 'resonant'},

            # Special combinations (expanded)
            'ch': {'noise_band': (2000, 5500), 'duration': 0.09, 'type': 'affricate'},
            'dj': {'noise_band': (2000, 5500), 'duration': 0.09, 'type': 'affricate', 'voiced': True},
            'ts': {'noise_band': (3000, 7000), 'duration': 0.09, 'type': 'affricate'},
            'dz': {'noise_band': (3000, 7000), 'duration': 0.09, 'type': 'affricate', 'voiced': True},
        }

        # Enhanced vowel patterns
        self.vowel_patterns = {
            'ee': 'i',  # as in see
            'ea': 'i',  # as in beat
            'ei': 'i',  # as in receive
            'ey': 'i',  # as in key
            'ie': 'i',  # as in piece
            'oo': 'u',  # as in boot
            'ue': 'u',  # as in blue
            'eu': 'u',  # as in neutral
            'ui': 'u',  # as in fruit
            'ou': 'o',  # as in ground
            'ow': 'o',  # as in low
            'oa': 'o',  # as in boat
            'ai': 'e',  # as in rain
            'ay': 'e',  # as in say
            'ae': 'æ',  # as in aesthetic
            'au': 'ɔ',  # as in cause
            'aw': 'ɔ',  # as in law
            'a_e': 'e',  # as in make
            'i_e': 'i',  # as in bite
            'o_e': 'o',  # as in note
            'u_e': 'u',  # as in cube
            'igh': 'i',  # as in high
            'eigh': 'e',  # as in eight
            'ough': 'o',  # as in though
            'augh': 'ɔ',  # as in caught
            'ar': 'ɑ',  # as in car
            'er': 'ə',  # as in worker
            'ir': 'ə',  # as in bird
            'or': 'ɔ',  # as in for
            'ur': 'ə',  # as in turn
        }

        # Enhanced consonant patterns
        self.consonant_patterns = {
            'ch': 'ch',  # as in chair
            'tch': 'ch',  # as in catch
            'sh': 'sh',  # as in ship
            'th': 'th',  # as in think
            'ph': 'f',  # as in phone
            'wh': 'w',  # as in what
            'ck': 'k',  # as in back
            'gh': 'g',  # as in ghost
            'ng': 'ng',  # as in sing
            'nk': 'ngk',  # as in think
            'dge': 'dj',  # as in bridge
            'ti': 'sh',  # as in nation
            'si': 'zh',  # as in vision
            'ci': 'sh',  # as in special
            'kn': 'n',  # as in knife
            'gn': 'n',  # as in gnome
            'wr': 'r',  # as in write
            'mb': 'm',  # as in lamb
            'rh': 'r',  # as in rhythm
            'ps': 's',  # as in psychology
            'pn': 'n',  # as in pneumonia
            'sc': 's',  # as in science
        }

        # Enhanced syllable break patterns
        self.syllable_breaks = [
            'tion', 'sion', 'ture', 'dle', 'ble', 'ple', 'gle', 'cle',
            'ing', 'ers', 'ness', 'ment', 'ly', 'ful', 'age', 'est',
            'ence', 'ance', 'able', 'ible', 'ity', 'ary', 'ery', 'ory',
            'ious', 'eous', 'uous', 'ial', 'ual', 'ian', 'ean', 'ize',
            'ate', 'ify', 'phy', 'ogy', 'ism', 'ist', 'ite', 'ive',
            'hood', 'ship', 'less', 'ling', 'ward', 'wise', 'fold', 'most'
        ]

        # Rules for handling doubled letters
        self.double_letter_rules = {
            # Double vowels
            'oo': {'sound': 'u', 'type': 'vowel'},  # as in boot
            'ee': {'sound': 'i', 'type': 'vowel'},  # as in see
            'aa': {'sound': 'a', 'type': 'vowel'},  # as in bazaar
            'ai': {'sound': 'e', 'type': 'vowel'},  # as in rain

            # Double consonants (treat as single)
            'tt': {'sound': 't', 'type': 'consonant'},
            'dd': {'sound': 'd', 'type': 'consonant'},
            'ss': {'sound': 's', 'type': 'consonant'},
            'ff': {'sound': 'f', 'type': 'consonant'},
            'll': {'sound': 'l', 'type': 'consonant'},
            'mm': {'sound': 'm', 'type': 'consonant'},
            'nn': {'sound': 'n', 'type': 'consonant'},
            'pp': {'sound': 'p', 'type': 'consonant'},
            'rr': {'sound': 'r', 'type': 'consonant'},
            'bb': {'sound': 'b', 'type': 'consonant'},
            'cc': {'sound': 'k', 'type': 'consonant'},  # as in accept
            'gg': {'sound': 'g', 'type': 'consonant'},
        }

        # Add punctuation handling
        self.punctuation_marks = {
            ',': 0.1,  # Short pause
            '.': 0.3,  # Long pause
            '!': 0.3,  # Long pause
            '?': 0.3,  # Long pause
            ';': 0.2,  # Medium pause
            ':': 0.2,  # Medium pause
        }

    def clean_text(self, text):
        """Clean and normalize input text."""
        cleaned = ""
        last_char_was_space = False

        for char in text:
            if char.isalnum() or char == "'":  # Keep letters, numbers, and apostrophes
                cleaned += char.lower()
                last_char_was_space = False
            elif char in self.punctuation_marks:
                if not last_char_was_space:
                    cleaned += " "  # Add space before punctuation if not already there
                cleaned += char
                last_char_was_space = True
            elif char.isspace():
                if not last_char_was_space:
                    cleaned += " "
                    last_char_was_space = True

        return cleaned.strip()

    def text_to_phonemes(self, text):
        """Convert text to phonemes, handling word boundaries and punctuation."""
        segments = []
        current_word = []

        # Clean and normalize the text first
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()

        for word in words:
            if word in self.punctuation_marks:
                segments.append(('pause', self.punctuation_marks[word]))
            else:
                word_phonemes = self._word_to_phonemes(word)
                segments.extend(word_phonemes)
                segments.append(None)  # Word boundary

        return segments[:-1]  # Remove the last word boundary

    def generate_pause(self, duration):
        """Generate a pause of specified duration."""
        return np.zeros(int(self.sample_rate * duration))

    def apply_envelope(self, audio, attack=0.02, decay=0.1, sustain=0.5, release=0.1):
        """Apply ADSR envelope to audio."""
        total_samples = len(audio)
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)

        envelope = np.ones(total_samples)

        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Decay
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)
        # Release
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)

        return audio * envelope

    def generate_noise(self, duration, band, amplitude=0.1):
        """Generate filtered noise for consonants."""
        noise = np.random.normal(0, 1, int(self.sample_rate * duration))
        nyquist = self.sample_rate / 2
        b, a = butter(4, [band[0] / nyquist, band[1] / nyquist], btype='band')
        filtered_noise = filtfilt(b, a, noise)
        return amplitude * filtered_noise

    def generate_formant(self, frequency, formant_freq, duration, amplitude=0.3):
        """Generate a single formant."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        carrier = np.sin(2 * np.pi * frequency * t)
        formant = np.sin(2 * np.pi * formant_freq * t)
        return amplitude * carrier * formant

    def generate_vowel(self, fundamental_freq, vowel, duration=0.3):
        """Generate an improved vowel sound with glottal source."""
        if vowel not in self.vowel_formants:
            raise ValueError(f"Unsupported vowel: {vowel}")

        # Generate glottal source
        glottal = self.generate_glottal_pulse(duration, fundamental_freq)

        # Apply formants
        audio = np.zeros(int(self.sample_rate * duration))
        for formant_range in self.vowel_formants[vowel]:
            formant = self.generate_formant(fundamental_freq, np.mean(formant_range), duration)
            # audio += formant + glottal
            audio += formant

        # Apply vibrato
        audio = self.apply_vibrato(audio, fundamental_freq)

        # Apply enhanced envelope
        audio = self.apply_enhanced_envelope(audio, 'vowel')

        # audio += audio + glottal

        return audio

    def generate_consonant(self, consonant):
        """Generate a consonant sound."""
        if consonant not in self.consonant_params:
            raise ValueError(f"Unsupported consonant: {consonant}")

        params = self.consonant_params[consonant]

        if params['type'] == 'noise':
            # Generate noise-based consonants (fricatives)
            audio = self.generate_noise(params['duration'], params['noise_band'])
            return self.apply_envelope(audio, attack=0.01, decay=0.02, sustain=0.3, release=0.02)

        elif params['type'] == 'plosive':
            # Generate plosive consonants with a sharp attack
            audio = self.generate_noise(params['duration'], params['noise_band'])
            return self.apply_envelope(audio, attack=0.001, decay=0.03, sustain=0.1, release=0.01)

        elif params['type'] == 'resonant':
            # Generate resonant consonants (approximants, nasals) using formants
            audio = np.zeros(int(self.sample_rate * params['duration']))
            base_freq = 120  # Base frequency for resonant consonants

            for formant_range in params['formants']:
                formant = self.generate_formant(base_freq,
                                                np.mean(formant_range),
                                                params['duration'],
                                                amplitude=0.2)
                audio += formant

            return self.apply_envelope(audio, attack=0.02, decay=0.05, sustain=0.4, release=0.03)

        return np.zeros(int(self.sample_rate * 0.05))  # Fallback to silence

    def generate_syllable(self, consonant, vowel, fundamental_freq=120):
        """Generate a complete syllable (consonant + vowel)."""
        if consonant:
            cons_sound = self.generate_consonant(consonant)
        else:
            cons_sound = np.zeros(0)

        vowel_sound = self.generate_vowel(fundamental_freq, vowel)

        # Combine consonant and vowel
        syllable = np.concatenate([cons_sound, vowel_sound])
        return syllable

    def play_syllable(self, consonant, vowel, fundamental_freq=120):
        """Play a generated syllable."""
        audio = self.generate_syllable(consonant, vowel, fundamental_freq)
        sd.play(audio, self.sample_rate)
        sd.wait()

    def _normalize_doubled_letters(self, word):
        """Handle doubled letters in a word."""
        i = 0
        normalized = ""
        while i < len(word):
            # Check for doubled letters
            if i + 1 < len(word) and word[i] == word[i + 1]:
                double = word[i:i + 2]
                if double in self.double_letter_rules:
                    # Replace with normalized sound
                    normalized += self.double_letter_rules[double]['sound']
                    i += 2
                else:
                    # Default: treat as single letter
                    normalized += word[i]
                    i += 2
            else:
                normalized += word[i]
                i += 1
        return normalized

    def _word_to_phonemes(self, word):
        """Convert a word to phonemes with advanced schwa and diphthong handling."""
        phonemes = []
        i = 0
        word = word.lower().strip()
        previous_phoneme = None  # Track the last phoneme to avoid duplicates

        print(f"Processing word: {word}")

        while i < len(word):
            # Check for consonant patterns first
            found_pattern = False
            for pattern, sound in sorted(self.consonant_patterns.items(), key=lambda x: len(x[0]), reverse=True):
                if word[i:].startswith(pattern):
                    next_vowel = self._find_next_vowel(word[i + len(pattern):])
                    current_phoneme = (sound, next_vowel)

                    if current_phoneme != previous_phoneme:
                        phonemes.append(current_phoneme)
                        previous_phoneme = current_phoneme

                    i += len(pattern)
                    found_pattern = True
                    break

            if found_pattern:
                continue

            # Handle single consonants
            if word[i] in self.consonant_params:
                next_vowel = self._find_next_vowel(word[i + 1:])
                current_phoneme = (word[i], next_vowel)

                if current_phoneme != previous_phoneme:
                    phonemes.append(current_phoneme)
                    previous_phoneme = current_phoneme

                i += 1
                continue

            # Handle diphthongs and vowels
            vowel_found = False
            for pattern, sound in sorted(self.vowel_patterns.items(), key=lambda x: len(x[0]), reverse=True):
                if word[i:].startswith(pattern):
                    if not phonemes or phonemes[-1][1] != sound:
                        current_phoneme = ('', sound)
                        if current_phoneme != previous_phoneme:
                            phonemes.append(current_phoneme)
                            previous_phoneme = current_phoneme

                    i += len(pattern)
                    vowel_found = True
                    break

            if not vowel_found and word[i] in 'aeiou':
                vowel_map = {'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u'}
                sound = vowel_map.get(word[i], 'ə')

                if not phonemes or phonemes[-1][1] != sound:
                    current_phoneme = ('', sound)
                    if current_phoneme != previous_phoneme:
                        phonemes.append(current_phoneme)
                        previous_phoneme = current_phoneme

                i += 1
                continue

            # Skip any other characters
            i += 1

        # Remove trailing schwa if unnecessary
        if phonemes and phonemes[-1][1] == 'ə' and len(phonemes) > 1:
            phonemes.pop()

        print(f"Phonemes generated for {word}: {phonemes}")
        return phonemes

    @staticmethod
    def _find_syllable_boundary(consonants):
        """Find the natural syllable boundary in a consonant cluster."""
        if len(consonants) <= 1:
            return len(consonants)

        # Common English syllable patterns
        for i in range(len(consonants) - 1):
            pair = consonants[i:i + 2]
            # Common consonant pairs that typically split between syllables
            splits = ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr',
                      'pl', 'pr', 'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st',
                      'str', 'sw', 'tr', 'tw']

            if pair in splits:
                return i

        # Default to splitting before the last consonant
        return len(consonants) - 1

    def _find_vowel_groups(self, word):
        """Find all vowel groups in a word."""
        vowel_groups = []
        i = 0
        while i < len(word):
            # Check for vowel patterns first
            found_pattern = False
            for pattern in self.vowel_patterns:
                if word[i:].startswith(pattern):
                    vowel_groups.append((i, i + len(pattern)))
                    i += len(pattern)
                    found_pattern = True
                    break

            if not found_pattern:
                # Check for single vowels
                if word[i] in 'aeiou':
                    vowel_groups.append((i, i + 1))
                i += 1

        return vowel_groups

    def _find_syllable_break(self, consonants):
        """Find the best place to break a consonant cluster."""
        if len(consonants) <= 1:
            return len(consonants)

        # Check for common consonant patterns
        for i in range(len(consonants) - 1):
            if consonants[i:i + 2] in self.consonant_patterns:
                return i

        # Default to breaking before the last consonant
        return len(consonants) - 1

    def _find_next_vowel(self, text):
        """Find the next valid vowel sound in text."""
        if not text:
            return 'ə'  # Default to schwa if no text

        # Check for vowel patterns first
        for pattern, phoneme in self.vowel_patterns.items():
            if text.startswith(pattern) and phoneme in self.vowel_formants:
                return phoneme

        # Check for single vowels
        if text[0] in 'aeiou':
            vowel_map = {'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u'}
            return vowel_map.get(text[0], 'ə')

        # If no vowel found, try the next character
        if len(text) > 1:
            return self._find_next_vowel(text[1:])

        return 'ə'  # Default to schwa if no vowel found

    def generate_word_sequence(self, phoneme_sequence):
        """Generate audio for a sequence of phonemes, including pauses."""
        word_audio = []

        for item in phoneme_sequence:
            if item is None:
                # Word boundary - add short pause
                word_audio.append(self.generate_pause(0.1))
            elif isinstance(item, tuple):
                if item[0] == 'pause':
                    # Handle punctuation pauses
                    word_audio.append(self.generate_pause(item[1]))
                else:
                    # Handle regular phoneme tuple (consonant, vowel)
                    consonant, vowel = item
                    if vowel:  # If there's a vowel, generate the syllable
                        syllable = self.generate_syllable(consonant, vowel)
                        word_audio.append(syllable)
                    elif consonant:  # Consonant-only case
                        cons_sound = self.generate_consonant(consonant)
                        word_audio.append(cons_sound)

        if word_audio:
            return np.concatenate(word_audio)
        return np.array([])

    def generate_glottal_pulse(self, duration, frequency):
        """Generate a more realistic glottal pulse waveform."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create basic pulse train
        period = 1.0 / frequency
        phase = t / period - np.floor(t / period)

        # Shape the pulse using LF (Liljencrants-Fant) model
        pulse = np.zeros_like(phase)

        # Opening phase
        mask_open = phase < self.open_quotient
        pulse[mask_open] = 0.5 * (1 - np.cos(np.pi * phase[mask_open] / self.open_quotient))

        # Return phase
        mask_return = (phase >= self.open_quotient) & (phase < self.open_quotient + self.return_quotient)
        return_phase = (phase[mask_return] - self.open_quotient) / self.return_quotient
        pulse[mask_return] = np.cos(np.pi * return_phase / 2)

        return pulse

    def apply_vibrato(self, audio, base_freq):
        """Apply natural vibrato to the audio."""
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        vibrato = 1.0 + self.vibrato_depth * np.sin(2 * np.pi * self.vibrato_rate * t)

        # Generate frequency-modulated signal
        phase = 2 * np.pi * base_freq * np.cumsum(vibrato) / self.sample_rate
        modulated = np.sin(phase)

        return audio * modulated

    def get_pitch_contour(self, duration, given_type='statement'):
        """Generate natural pitch contour for the utterance."""
        t = np.linspace(0, 1, int(duration * self.sample_rate))
        contour = np.interp(t,
                            [point[0] for point in self.sentence_pitch_contour[given_type]],
                            [point[1] for point in self.sentence_pitch_contour[given_type]])

        # Smooth the contour
        window = min(101, len(contour) - 1 if len(contour) % 2 == 0 else len(contour))
        if window > 3:
            contour = savgol_filter(contour, window, 3)

        return contour

    def generate_formant_transitions(self, from_formants, to_formants, duration):
        """Generate smooth transitions between formants."""
        transition_samples = int(self.transition_time * self.sample_rate)
        t = np.linspace(0, 1, transition_samples)

        # Use sigmoid function for smooth transition
        transition = 1 / (1 + np.exp(-10 * (t - 0.5)))

        transitions = []
        for (from_f, to_f) in zip(from_formants, to_formants):
            freq_transition = from_f + (to_f - from_f) * transition
            transitions.append(freq_transition)

        return transitions

    def apply_enhanced_envelope(self, audio, given_type='vowel'):
        """Apply more sophisticated ADSR envelope with natural curves."""
        total_samples = len(audio)

        if given_type == 'vowel':
            attack = int(0.03 * self.sample_rate)
            decay = int(0.1 * self.sample_rate)
            sustain_level = 0.8
            release = int(0.15 * self.sample_rate)
        else:  # consonant
            attack = int(0.01 * self.sample_rate)
            decay = int(0.05 * self.sample_rate)
            sustain_level = 0.6
            release = int(0.08 * self.sample_rate)

        envelope = np.ones(total_samples)

        # Attack (exponential curve)
        t_attack = np.linspace(0, 1, attack)
        envelope[:attack] = 1 - np.exp(-5 * t_attack)

        # Decay (exponential curve)
        t_decay = np.linspace(0, 1, decay)
        decay_curve = np.exp(-3 * t_decay) * (1 - sustain_level) + sustain_level
        envelope[attack:attack + decay] = decay_curve

        # Sustain
        envelope[attack + decay:-release] = sustain_level

        # Release (exponential curve)
        t_release = np.linspace(0, 1, release)
        envelope[-release:] = sustain_level * np.exp(-5 * t_release)

        return audio * envelope

    def speak(self, text, base_freq=120):
        """Speak text with improved prosody and pitch handling."""
        try:
            print(f"Speaking text: {text}")

            # Detect if the text is a question
            is_question = text.strip().endswith('?')
            contour_type = 'question' if is_question else 'statement'

            phoneme_sequence = self.text_to_phonemes(text)
            word_audio = []

            # Count actual phonemes (excluding pauses and None)
            actual_phonemes = [p for p in phoneme_sequence if isinstance(p, tuple) and p[0] != 'pause']
            total_duration = len(actual_phonemes) * 0.2
            pitch_contour = self.get_pitch_contour(total_duration, contour_type)

            pitch_idx = 0

            for phoneme in phoneme_sequence:
                if phoneme is None:
                    # Word boundary - add short pause
                    word_audio.append(self.generate_pause(0.1))
                elif isinstance(phoneme, tuple):
                    if phoneme[0] == 'pause':
                        # Handle punctuation pause
                        pause_duration = phoneme[1]
                        word_audio.append(self.generate_pause(pause_duration))
                    else:
                        # Handle actual phoneme (consonant-vowel pair)
                        consonant, vowel = phoneme
                        if vowel in self.vowel_formants:  # Verify vowel is supported
                            # Convert pitch_contour value to float
                            pitch_idx_bounded = min(pitch_idx, len(pitch_contour) - 1)
                            current_pitch = float(base_freq * pitch_contour[pitch_idx_bounded])
                            syllable = self.generate_syllable(consonant, vowel, current_pitch)
                            word_audio.append(syllable)
                            pitch_idx += 1
                        else:
                            print(f"Warning: Skipping unsupported vowel: {vowel}")

            if word_audio:
                audio = np.concatenate(word_audio)
                # Normalize audio
                audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
                sd.play(audio, self.sample_rate)
                sd.wait()

        except Exception as e:
            print(f"Error during speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()


# Example usage
if __name__ == "__main__":
    synth = VocalSynthesizer()

    # Test individual syllables
    # print("\nTesting individual syllables:")
    # syllables = [
    #     ('t', 'a'),
    #     ('k', 'e'),
    #     ('p', 'i'),
    #     ('s', 'o'),
    #     ('', 'u'),  # Just the vowel
    # ]
    #
    # for cons, vowel in syllables:
    #     print(f"Playing syllable: {cons + vowel if cons else vowel}")
    #     synth.play_syllable(cons, vowel)
    #     time.sleep(0.5)

    # Test speaking full sentences
    print("\nTesting full sentences:")

    # Test speaking with improved prosody
    test_sentences = [
        "Hello, how are you?",
        "I am doing well.",
        "This is a test of improved speech synthesis.",
        "This is a really difficult problem.",
        "Probably impossible.",
    ]

    engine = pyttsx3.init()

    from TTS.api import TTS
    # Initialize the TTS model
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")

    for sentence in test_sentences:
        print(f"\nSpeaking: {sentence}")
        synth.speak(sentence)
        time.sleep(1.0)

        engine.say(sentence)
        engine.runAndWait()
        time.sleep(1.0)

        # Generate audio data as a NumPy array
        speech_wav = tts.tts(sentence, speaker=None)
        sd.play(speech_wav, samplerate=22050)  # Use the appropriate sampling rate (22050 Hz for Tacotron 2)
        sd.wait()  # Wait until playback finishes
