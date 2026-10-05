import random
import sys

SEED = 2024

TO_SELECT = 5

total_rows = int(sys.argv[1])

random.seed(SEED)

rows = random.sample(range(1, total_rows + 1), TO_SELECT)

print("Selected rows:", rows)
