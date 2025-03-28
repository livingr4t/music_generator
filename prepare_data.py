import os
import pretty_midi
import numpy as np

# Folder z plikami MIDI
midi_folder = "midi_files"
notes = []

# Przetwarzanie wszystkich plików MIDI w folderze
for file_name in os.listdir(midi_folder):
    if file_name.endswith(".mid") or file_name.endswith(".midi"):
        midi_path = os.path.join(midi_folder, file_name)
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append(note.pitch)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Konwersja nut na sekwencje
X_train = np.array([notes[i:i+100] for i in range(len(notes) - 100)])
y_train = np.array([notes[i + 100] for i in range(len(notes) - 100)])

# Zapisywanie danych
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
