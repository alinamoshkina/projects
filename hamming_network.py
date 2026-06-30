
# Нейросеть Хемминга


# Эталоны

etalon_matrices = [
    [
        "########",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        "...##..."
    ],

    [
        "..####..",
        ".######.",
        "##....##",
        "##....##",
        "##....##",
        "##....##",
        ".######.",
        "..####.."
    ],

    [
        "##....##",
        ".##..##.",
        "..####..",
        "...##...",
        "...##...",
        "..####..",
        ".##..##.",
        "##....##"
    ],

    [
        "..####..",
        ".######.",
        "##....##",
        "##....##",
        "########",
        "##....##",
        "##....##",
        "##....##"
    ]
]

# Искаженные матрицы

distorted_matrices = [
    [
        "#######.",
        "...##...",
        "...##...",
        "...#....",
        "...##...",
        "...##...",
        "....#...",
        "...##..."
    ],

    [
        "..###...",
        ".######.",
        "##....##",
        "#.....##",
        "##....##",
        "##.....#",
        ".######.",
        "...###.."
    ],

    [
        "##....#.",
        ".##..##.",
        "..###...",
        "...##...",
        "...#....",
        "..####..",
        ".##...#.",
        "#.....##"
    ],

    [
        "..###...",
        ".######.",
        "##....##",
        "##.....#",
        "########",
        "##....##",
        "#.....##",
        "##....##"
    ]
]

def char_to_number(ch):
    if ch == "#":
        return 1
    return -1


def number_to_char(x):
    if x == 1:
        return "#"
    return "."

def matrix_to_vector(matrix):
    vector = []

    for row in matrix:
        for ch in row:
            vector.append(char_to_number(ch))

    return vector

def vector_to_matrix(vector):
    matrix = []
    index = 0

    for i in range(8):
        row = []

        for j in range(8):
            row.append(vector[index])
            index += 1

        matrix.append(row)

    return matrix

def print_matrix_from_vector(vector):
    matrix = vector_to_matrix(vector)

    for row in matrix:
        for value in row:
            print(number_to_char(value), end="")
        print()

# Матрица весов первого слоя

def create_first_layer_weights(etalon_vectors):
    weights = []

    for etalon in etalon_vectors:
        row = []

        for value in etalon:
            row.append(value / 2)

        weights.append(row)

    return weights

# Пороги первого слоя
def create_first_layer_thresholds():
    thresholds = []
    for i in range(4):
        thresholds.append(64 / 2)

    return thresholds

# Работа первого слоя

def first_layer(input_vector, first_layer_weights, thresholds):
    outputs = []

    for k in range(4):
        s = 0
        for i in range(64):
            s += first_layer_weights[k][i] * input_vector[i]

        y = thresholds[k] + s
        outputs.append(y)

    return outputs

# Матрица весов второго слоя

def create_second_layer_weights(epsilon):
    weights = []

    for i in range(4):
        row = []
        for j in range(4):
            if i == j:
                row.append(1)
            else:
                row.append(-epsilon)

        weights.append(row)

    return weights

# Функция активации второго слоя
def activation_second_layer(value):
    if value < 0:
        return 0
    return value


# Работа второго слоя

def second_layer(first_layer_outputs, second_layer_weights, max_iterations=50):
    current = first_layer_outputs[:]

    for iteration in range(1, max_iterations + 1):
        new_outputs = []

        for j in range(4):
            s = 0

            for i in range(4):
                s += second_layer_weights[j][i] * current[i]

            y = activation_second_layer(s)
            new_outputs.append(y)

        if new_outputs == current:
            return new_outputs, iteration - 1

        current = new_outputs

        active_neurons = 0

        for value in current:
            if value > 0:
                active_neurons += 1

        if active_neurons <= 1:
            return current, iteration

    return current, max_iterations


# Расстояние Хемминга
def hamming_distance(vector1, vector2):
    distance = 0

    for i in range(64):
        if vector1[i] != vector2[i]:
            distance += 1

    return distance


# Степень схожести

def similarity_percent(matches):
    return matches * 100 / 64


# Поиск нейрона победителя
def find_winner(outputs):
    best_index = 0
    best_value = outputs[0]

    for i in range(1, len(outputs)):
        if outputs[i] > best_value:
            best_index = i
            best_value = outputs[i]

    return best_index, best_value


# MAIN

print("Нейросеть Хемминга")

etalon_vectors = []
for matrix in etalon_matrices:
    vector = matrix_to_vector(matrix)
    etalon_vectors.append(vector)

distorted_vectors = []
for matrix in distorted_matrices:
    vector = matrix_to_vector(matrix)
    distorted_vectors.append(vector)

first_layer_weights = create_first_layer_weights(etalon_vectors)
thresholds = create_first_layer_thresholds()
epsilon = 0.2
second_layer_weights = create_second_layer_weights(epsilon)



for i in range(4):
    print()
    print("Искаженная матрица", i + 1)

    print("Исходный вид:")
    print_matrix_from_vector(distorted_vectors[i])


    first_layer_outputs = first_layer(
        distorted_vectors[i],
        first_layer_weights,
        thresholds
    )

    for j in range(4):
        matches = int(first_layer_outputs[j])
        distance = hamming_distance(distorted_vectors[i], etalon_vectors[j])
        percent = similarity_percent(matches)


    second_layer_outputs, iterations = second_layer(
        first_layer_outputs,
        second_layer_weights
    )

    winner_index, winner_value = find_winner(second_layer_outputs)

    print()
    print("Количество итераций:", iterations)
    print("Победивший эталон:", winner_index + 1)
    final_matches = int(first_layer_outputs[winner_index])
    final_distance = hamming_distance(
        distorted_vectors[i],
        etalon_vectors[winner_index]
    )
    final_percent = similarity_percent(final_matches)

    print("Итоговое количество совпадений:", final_matches, "из", 64)
    print("Итоговое расстояние Хемминга:", final_distance)
    print("Итоговая схожесть:", round(final_percent, 2), "%")