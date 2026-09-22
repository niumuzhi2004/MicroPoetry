"""
Reformat wte_weight to 32-bit entries to fit FPGA's RAMB36 units
"""

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
