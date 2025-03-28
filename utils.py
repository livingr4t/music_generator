def process_prompt(prompt):
    # Rozpoznawanie tempa i nastroju na podstawie prompta
    if "wesoła" in prompt.lower():
        mood = "happy"
    elif "smutna" in prompt.lower():
        mood = "sad"
    else:
        mood = "neutral"

    if "szybka" in prompt.lower():
        tempo = "fast"
    elif "wolna" in prompt.lower():
        tempo = "slow"
    else:
        tempo = "medium"

    return mood, tempo
