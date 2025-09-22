import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Generuojame dvimačius duomenis, vienas duomenų rinkinys yra iš intervalo [1,5],
#kitas iš intervalo [6,10].
np.random.seed(763)
points1 = np.random.uniform(0,5, (10,2))
points2 = np.random.uniform(5,10, (10,2))

#Kiekvienam duomenų taškui priskiriame klasę (0 arba 1)
data = np.vstack((
    np.column_stack((points1, np.zeros(10))),
    np.column_stack((points2, np.ones(10))),
))

#Sudedame duomenis į dataframe'ą
df = pd.DataFrame(data, columns=['x', 'y', 'class'])
print(df)

#Atvaizudojame duomenis grafike
plt.figure(figsize=(8, 8))
plt.scatter(df[df['class'] == 0]['x'], df[df['class'] == 0]['y'],
            color='#ffbe0b', label='Klasė 0')
plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'],
            color='#3a86ff', label='Klasė 1')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sugeneruoti duomenų taškai")
plt.legend(loc = 'upper left')
plt.grid(True)
plt.savefig("generated_data.png", dpi=300, bbox_inches="tight") #Išsaugom paveiksliuką
plt.show()

#Dirbtinio neurono konstravimas

#Apibrėžiame slenkstinę aktyvacijos funkciją
def threshold(a):
    return np.where(a > 0, 1, 0)

#Apibrėžiame sigmoidinę aktyvacijos funckiją
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

#Apisibrėžiame dirbtinio neurono funkciją
def neuron(df, w1, w2, b, activation_function):
    a = w1 * df[0] + w2 * df[1] + b #Panaudojame funkciją iš skaidrių
    if activation_function == threshold:
        return threshold(a)
    else:
        return round(sigmoid(a), 0) #Suapvalinam, nes sigmoidinė funkcija grąžina skaičių tarp 0 ir 1

#Ieškome svorių ir poslinkio
def find_weights(df, activation_function):
    solutions = [] #Sprendinių sąrašas
    while len(solutions) < 3: #Kartojam tol, kol randam tris sprendinius
        w1 = round(np.random.uniform(-1, 1), 2)
        w2 = round(np.random.uniform(-1, 1), 2)
        b = round(np.random.uniform(-1, 1), 2)
        #Sprendin5 pridedam į sąrašą tik tada, jeigu jis visus taškus suklasifikuoja teisingai
        if all(neuron(row[:2], w1, w2, b, activation_function) == row[2] for row in df.values):
            solutions.append((w1, w2, b))
    return solutions

# print(find_weights(df, threshold))
# print(find_weights(df, sigmoid))

solutions_threshold = find_weights(df, threshold)
solutions_sigmoid = find_weights(df, sigmoid)

#Funkcija, kuri nubrėžia klases skiriančias tieses
def plot_decision_boundaries(df, solutions, activation_function):
    plt.figure(figsize=(8, 8))
    # Nubraižom taškus
    plt.scatter(df[df['class'] == 0]['x'], df[df['class'] == 0]['y'],
                color='#ffbe0b', label='Klasė 0')
    plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'],
                color='#3a86ff', label='Klasė 1')

    #Spalvytes
    colors = ['#fb5607', '#ff006e', '#8338ec']
    x_vals = np.linspace(df['x'].min()-1, df['x'].max()+1, 100)

    # Einam per rastus svorius ir poslinkius
    for (w1, w2, b), c in zip(solutions, colors):
        if w2 != 0:
            y_vals = -(w1/w2)*x_vals - b/w2 #Tiesės lygtis išsireiškus y per x
            plt.plot(x_vals, y_vals, color=c, label=f"{w1:.2f}x_1+{w2:.2f}x_2+{b:.2f}=0")

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc = 'upper left')
    plt.grid(True)
    #Galimybė pasirinkti norimą aktyvacijos funkciją
    if activation_function == threshold:
        plt.title("Klases skiriančios tiesės\nnaudojant slenkstinę aktyvavimo funckiją")
        #plt.savefig("treshold_fig.png", dpi=300, bbox_inches="tight")
    else:
        plt.title("Klases skiriančios tiesės\nnaudojant sigmoidinę aktyvavimo funckiją")
        #plt.savefig("sigmoid_fig.png", dpi=300, bbox_inches="tight")
    plt.show()

#Funkcija, kuri nubrėžia klases skiriančias tieses ir svorio vektorius
def plot_decision_boundaries_with_vector(df, solutions, activation_function):
    plt.figure(figsize=(8, 8))
    # Nubraižom taškus
    plt.scatter(df[df['class'] == 0]['x'], df[df['class'] == 0]['y'],
                color='#ffbe0b', label='Klasė 0')
    plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'],
                color='#3a86ff', label='Klasė 1')

    #Spalvytes
    colors = ['#fb5607', '#ff006e', '#8338ec']
    x_vals = np.linspace(df['x'].min()-1, df['x'].max()+1, 100)

    #Nusistatom vektoriaus ilgį
    # Einam per rastus svorius
    for (w1, w2, b), c in zip(solutions, colors):
        if w2 != 0:
            y_vals = -(w1/w2)*x_vals - b/w2
            plt.plot(x_vals, y_vals, color=c, label=f"{w1:.2f}x_1+{w2:.2f}x_2+{b:.2f}=0")

        # Pasirenkam pradinį tašką vektoriui
        x0 = np.clip(5, 1, 10)
        y0 = -(w1 / w2) * x0 - (b / w2)

        # Vektoriaus normalizavimas
        norm = np.sqrt(w1**2 + w2**2)
        w1_norm = w1 / norm
        w2_norm = w2 / norm

        # Nubraižom svorio vektorių (jei telpa į ribas)
        # if 0 <= x0 + w1_norm <= 11 and 0 <= y0 + w2_norm <= 11:
        plt.quiver(x0, y0, w1_norm, w2_norm,
                       color=c, angles='xy', scale_units='xy',
                       scale=1, width=0.005)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("su tiesem cia")
    plt.legend(loc = 'upper left')
    plt.grid(True)
    # Galimybė pasirinkti norimą aktyvacijos funkciją
    if activation_function == threshold:
        plt.title("Klases skiriančios tiesės ir vektoriai\nnaudojant slenkstinę aktyvavimo funckiją")
        #plt.savefig("treshold_fig_vectors.png", dpi=300, bbox_inches="tight")
    else:
        plt.title("Klases skiriančios tiesės ir vektoriai\nnaudojant sigmoidinę aktyvavimo funckiją")
        #plt.savefig("sigmoid_fig_vectors.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_decision_boundaries(df, solutions_threshold, threshold)
plot_decision_boundaries(df, solutions_sigmoid, sigmoid)

plot_decision_boundaries_with_vector(df, solutions_threshold, threshold)
plot_decision_boundaries_with_vector(df, solutions_sigmoid, sigmoid)

#Slenkstinė ir sigmoidinė aktyvacijos funkcijos grafikai
x = np.linspace(-5, 5, 1000)
y = threshold(x)

plt.figure(figsize=(8, 8))
plt.plot(x, y, color="black")
plt.title("Slenkstinė aktyvacijos funkcija")
plt.xlabel("a")
plt.ylabel("f(a)")
plt.savefig("sleknstine.png", dpi=300, bbox_inches="tight")
plt.grid(True)
plt.show()

y = sigmoid(x)
plt.figure(figsize=(8, 8))
plt.plot(x, y, color="black")
plt.title("Sigmoidinė aktyvacijos funkcija")
plt.xlabel("a")
plt.ylabel("f(a)")
plt.savefig("sigmoidine.png", dpi=300, bbox_inches="tight")
plt.grid(True)
plt.show()