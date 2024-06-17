import subprocess

# NO_ANIMAL, RAT, CAT, DOG, WOLF, JAGUAR, TIGER, LION, ELEPHANT = range(9)
animals = range(1,9)
name_of_animal = {
    1: "RAT",
    2: "CAT",
    3: "DOG",
    4: "WOLF", 
    5: "JAGUAR",
    6: "TIGER",
    7: "LION", 
    8: "ELEPHANT"
}

def check(result, animal, trybe, phase):
    if result.returncode == 0:
        print(f"pahse {phase} done with {animal} in trybe {trybe}")
    else: 
        print(f"pahse {phase} faliure with {animal} in trybe {trybe}")
        print("error:", result.stderr)

def main():
    # obsługa jednego zwierzęcia
    for animal in animals:
        for trybe in range(2):
            # jeśli tryb 0 to przeciwnik robi losowe ruchy
            # jeśli tryb 1 to przeciwnik preferuje ruchy do przodu
        
            # najpierw generujemy bota dla gracza
            result = subprocess.run(['python3', 'bot_generator.py', name_of_animal[animal], trybe])
            check(result, animal, trybe, "generacja bota")
            
            # bot gotowy, czas go przetestować
            result = subprocess.run(['python3', 'stats_generator.py', name_of_animal[animal], trybe])
            check(result, animal, trybe, "generacja statystyk")
            
            # statystyki gotowe
            result = subprocess.run(['python3', 'chart_generator.py', name_of_animal[animal], trybe])
            check(result, animal, trybe, "generacja wykresów")
        
    
if __name__ == '__main__':
    main()