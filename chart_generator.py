import matplotlib.pyplot as plt
import sys

score = None 
sum_length, sum_kils, sum_deaths = 0, 0, 0
vis = [[0] * 7 for _ in range(9)]

animal_arg = sys.argv[1]
trybe_arg = int(sys.argv[2])

with open(f'stats/{animal_arg}{trybe_arg}.txt') as file:
    score = list(map(int, file.readline().split()))
    sum_length, sum_kils, sum_deaths = map(int, file.readline().split())
    for i in range(9):
        for j in range(7):
            vis[i][j] = int(file.readline())
            
# wykresiki
# heatmap
path = f'plots/{animal_arg}{trybe_arg}'
plt.figure(figsize=(10, 8))
plt.imshow(vis, cmap='hot', interpolation='nearest')
plt.colorbar()

# Dodanie tytułu i etykiet osi
plt.title('Heatmap of vis')
plt.xlabel('Column')
plt.ylabel('Row')
plt.savefig(path + 'heatmap.png')

# pie
plt.figure(figsize=(8, 8))
plt.pie(score, labels=['Score 1', 'Score 2'], autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Score')
plt.savefig(path + 'pie.png')
plt.close()

# bar 
labels = ['Sum Length', 'Sum Kills', 'Sum Deads']
values = [sum_length, sum_kils, sum_deaths]

plt.figure(figsize=(10, 8))
plt.bar(labels, values, color=['blue', 'orange', 'green'])
plt.title('Bar Chart of Sum Length, Sum Kills, Sum Deads')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig(path + 'bar.png')
plt.close()
    
    