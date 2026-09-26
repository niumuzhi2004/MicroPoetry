"""
Reformat weights to 32-bit entries to fit FPGA's RAMB36 units
"""

def reformat_wte():
    byte_count = 0
    bank_count = 0

    n_banks = 47
    bank_depth = 1024

    wte_rom = []

    for i in range(n_banks):
        wte_rom.append("")

    buf = []

    with open("./Data/wte_weight.hex", 'r') as file:
        for line in file:
            buf.append(line.strip())
            byte_count = byte_count + 1
            if (byte_count % 4 == 0):
                wte_rom[bank_count] += "".join(reversed(buf)) + "\n"
                buf = []
            if (byte_count == bank_depth * 4):
                bank_count += 1
                byte_count = 0

    for i in range(n_banks):
        n_words = wte_rom[i].count("\n")
        if n_words < bank_depth:
            for j in range(bank_depth - n_words):
                wte_rom[i] = wte_rom[i] + "00000000\n"

        with open(f"./Data/wte_rom/wte_weight_{i}.hex", 'w') as file:
            file.write(wte_rom[i])


def reformat_mlp_fc1():
    n_banks = 16
    n_layers = 4

    rows_per_bank = 16
    words_per_row = 16
    rows_per_layer = n_banks * rows_per_bank
    bytes_per_row = words_per_row * 4

    mlp_fc1_rom = []

    for i in range(n_banks):
        mlp_fc1_rom.append("")

    raw_rom = []

    with open("./Data/mlp_fc1_weight.hex", 'r') as file:
        for line in file:
            raw_rom.append(line.strip())

    for i in range(n_banks):
        for j in range(n_layers):
            for k in range(rows_per_bank):
                for m in range(words_per_row):
                    index = j * (rows_per_layer * bytes_per_row) + (i * rows_per_bank + k) * bytes_per_row + m * 4
                    buf = [raw_rom[index], raw_rom[index + 1], raw_rom[index + 2], raw_rom[index + 3]]
                    mlp_fc1_rom[i] = mlp_fc1_rom[i] + "".join(reversed(buf)) + "\n"

        with open(f"./Data/mlp_fc1_rom/mlp_fc1_weight_{i}.hex", 'w') as file:
            file.write(mlp_fc1_rom[i])


def reformat_mlp_fc2():
    n_banks = 16
    n_layers = 4

    rows_per_bank = 4
    words_per_row = 64
    rows_per_layer = n_banks * rows_per_bank
    bytes_per_row = words_per_row * 4

    mlp_fc2_rom = []

    for i in range(n_banks):
        mlp_fc2_rom.append("")

    raw_rom = []

    with open("./Data/mlp_fc2_weight.hex", 'r') as file:
        for line in file:
            raw_rom.append(line.strip())

    for i in range(n_banks):
        for j in range(n_layers):
            for k in range(rows_per_bank):
                for m in range(words_per_row):
                    index = j * (rows_per_layer * bytes_per_row) + (i * rows_per_bank + k) * bytes_per_row + m * 4
                    buf = [raw_rom[index], raw_rom[index + 1], raw_rom[index + 2], raw_rom[index + 3]]
                    mlp_fc2_rom[i] = mlp_fc2_rom[i] + "".join(reversed(buf)) + "\n"

        with open(f"./Data/mlp_fc2_rom/mlp_fc2_weight_{i}.hex", 'w') as file:
            file.write(mlp_fc2_rom[i])

def reformat_attn_w(weight_name):
    n_banks = 4
    n_layers = 4

    rows_per_bank = 16
    words_per_row = 16
    rows_per_layer = n_banks * rows_per_bank
    bytes_per_row = words_per_row * 4

    attn_rom = []

    for i in range(n_banks):
        attn_rom.append("")

    raw_rom = []

    with open(f"./Data/attn_w{weight_name}_weight.hex", 'r') as file:
        for line in file:
            raw_rom.append(line.strip())

    for i in range(n_banks):
        for j in range(n_layers):
            for k in range(rows_per_bank):
                for m in range(words_per_row):
                    index = j * (rows_per_layer * bytes_per_row) + (i * rows_per_bank + k) * bytes_per_row + m * 4
                    buf = [raw_rom[index], raw_rom[index + 1], raw_rom[index + 2], raw_rom[index + 3]]
                    attn_rom[i] = attn_rom[i] + "".join(reversed(buf)) + "\n"

        with open(f"./Data/attn_w{weight_name}_rom/attn_w{weight_name}_weight_{i}.hex", 'w') as file:
            file.write(attn_rom[i])


if __name__ == "__main__":
    # reformat_wte()
    # reformat_mlp_fc1()
    reformat_mlp_fc2()
    reformat_attn_w("k")
    reformat_attn_w("v")
    reformat_attn_w("q")
    reformat_attn_w("o")
