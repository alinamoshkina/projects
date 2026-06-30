# Нейросеть Хопфилда


# 1. Эталоны

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

# 2. Искаженные матрицы

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


# 3. Обучение сети

def train_hopfield(vectors):
    weights = []

    for i in range(64):
        row = []
        for j in range(64):
            if i == j:
                row.append(0)
            else:
                s = 0
                for p in vectors:
                    s += p[i] * p[j]

                row.append(s)

        weights.append(row)

    return weights


# 4. Восстановление образа

def restore_image(start_vector, weights, max_iterations=50):
    current = start_vector[:]

    for iteration in range(1, max_iterations + 1):
        new_vector = []

        for i in range(64):
            s = 0
            for j in range(64):
                s += weights[i][j] * current[j]
            if s >= 0:
                new_vector.append(1)
            else:
                new_vector.append(-1)

        if new_vector == current:
            return new_vector, iteration - 1

        current = new_vector

    return current, max_iterations


# 5. Определение ближайшего эталона

def count_matches(vector1, vector2):
    matches = 0

    for i in range(64):
        if vector1[i] == vector2[i]:
            matches += 1

    return matches


def find_best_etalon(vector, etalons):
    best_index = 0
    best_matches = count_matches(vector, etalons[0])

    for i in range(1, len(etalons)):
        matches = count_matches(vector, etalons[i])

        if matches > best_matches:
            best_matches = matches
            best_index = i

    return best_index, best_matches


# 6. MAIN
etalon_vectors = []

for matrix in etalon_matrices:
    vector = matrix_to_vector(matrix)
    etalon_vectors.append(vector)


distorted_vectors = []

for matrix in distorted_matrices:
    vector = matrix_to_vector(matrix)
    distorted_vectors.append(vector)


weights = train_hopfield(etalon_vectors)



for i in range(4):
    print()
    print("Искаженная матрица", i + 1)
    print("Исходный вид:")

    print_matrix_from_vector(distorted_vectors[i])

    restored_vector, iterations = restore_image(distorted_vectors[i], weights)

    best_index, matches = find_best_etalon(restored_vector, etalon_vectors)

    print()
    print("Вид после восстановления:")

    print_matrix_from_vector(restored_vector)

    print()
    print("Кол-во итераций:", iterations)
    print("Ближе всего к эталонной матрице:", best_index + 1)
    print("Количество совпавших клеток:", matches, "из", 64)
