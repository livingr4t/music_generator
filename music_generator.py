from flask import Flask, request, render_template, send_file
import numpy as np
from tensorflow.keras.models import load_model
from utils import process_prompt
import pretty_midi
import os
import random

app = Flask(__name__)

# Wczytanie wytrenowanego modelu
try:
    model = load_model("music_model.h5")
except Exception as e:
    print(f"Nie udało się załadować modelu: {e}")
    model = None

# Funkcja do regulacji temperatury
def apply_temperature(predictions, temperature=1.0):
    predictions = np.log(predictions + 1e-10) / temperature
    predictions = np.exp(predictions)
    return predictions / np.sum(predictions)

# Funkcja do zapisu nut do pliku MIDI
def save_to_midi(notes, filename="static/generated_music.mid", note_duration=0.5):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Domyślnie: fortepian

    # Dodawanie nut
    start_time = 0
    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=100,
            pitch=int(note),
            start=start_time,
            end=start_time + note_duration
        )
        instrument.notes.append(midi_note)
        start_time += note_duration

    midi.instruments.append(instrument)
    midi.write(filename)
    print(f"Plik MIDI zapisany jako {filename}")

# Generowanie muzyki na podstawie prompta
def generate_music_based_on_prompt(mood, tempo, num_notes=100):
    if not model:
        return {"error": "Model nie został poprawnie załadowany."}

    seed_length = 5
    seed = [random.randint(50, 80) for _ in range(seed_length)]

    while len(seed) < 100:
        seed.append(random.randint(50, 80))

    generated = list(seed)
    num_classes = model.output_shape[-1]

    try:
        for _ in range(num_notes):
            if len(generated) < 100:
                while len(generated) < 100:
                    generated.append(random.randint(50, 80))

            input_sequence = np.array(generated[-100:]).reshape(1, 100, 1)
            predictions = model.predict(input_sequence).flatten()
            predictions = apply_temperature(predictions, temperature=1.2)
            next_note = np.random.choice(range(num_classes), p=predictions)
            next_note = max(0, min(127, next_note))
            generated.append(next_note)
    except Exception as e:
        print(f"Błąd podczas generowania muzyki: {e}")
        return {"error": f"Błąd podczas generowania muzyki: {e}"}

    return generated

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form.get("prompt", "neutral medium")
        duration_seconds = int(request.form.get("duration_seconds", 30))
        note_duration = 0.5
        num_notes = int(duration_seconds / note_duration)

        mood, tempo = process_prompt(prompt)
        generated_notes = generate_music_based_on_prompt(mood, tempo, num_notes)

        if isinstance(generated_notes, dict):
            return render_template("index.html", error=generated_notes["error"])

        save_to_midi(generated_notes, note_duration=note_duration)

        return render_template("index.html", generated=True)
    return render_template("index.html")

@app.route("/download")
def download():
    file_path = "static/generated_music.mid"
    response = send_file(file_path, as_attachment=True)
    try:
        os.remove(file_path)
        print(f"Usunięto plik: {file_path}")
    except Exception as e:
        print(f"Nie udało się usunąć pliku: {e}")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
