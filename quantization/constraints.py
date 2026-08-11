import json

with open("../model/Data/token_id.json", 'r', encoding="utf-8") as file:
    token_id_map = json.load(file)

# turn templates into hex format
with open("../model/Data/templates.json", 'r', encoding="utf-8") as file:
    templates = json.load(file)

with open("./Data/templates.hex", 'w') as file:
    for template in templates.values():
        for char in template:
            if char == "无":
                val = 0
            elif char == "平":
                val = 1
            elif char == "仄":
                val = 2
            else:
                print("Error!")
            file.write(f"{val:01x}\n")


# turn tone table into hex format - one list for ping characters, one for ze characters
with open("../model/Data/tone_table.json", 'r', encoding="utf-8") as file:
    tone_table = json.load(file)

with open("./Data/tone_ping.hex", 'w') as file:
    for char, tone in tone_table.items():
        if tone == "平":
            token_id = token_id_map.index(char)
            file.write(f"{token_id:03x}\n")

with open("./Data/tone_ze.hex", 'w') as file:
    for char, tone in tone_table.items():
        if tone == "仄":
            token_id = token_id_map.index(char)
            file.write(f"{token_id:03x}\n")


# turn rhyme table into hex format - each character maps to its rhyme group id
with open("../model/Data/rhyme_table.json", 'r', encoding="utf-8") as file:
    rhyme_table = json.load(file)

unique_rhyme_groups = list(set(rhyme_table.values()))
print(f"There are {len(unique_rhyme_groups)} unique rhyme groups.")
ze_rhyme_group_id = len(unique_rhyme_groups)
special_token_id  = len(unique_rhyme_groups)

with open("./Data/rhyme.hex", 'w') as file:
    for char in token_id_map:
        if char in ["<BOS>", "<EOS>", "<UNK>", "<PAD>", "<SEP>"]:
            file.write(f"{special_token_id:02x}\n")
        elif tone_table[char] == "仄":
            file.write(f"{ze_rhyme_group_id:02x}\n")
        elif tone_table[char] == "平":
            rhyme_group = rhyme_table[char]
            rhyme_group_id = unique_rhyme_groups.index(rhyme_group)
            file.write(f"{rhyme_group_id:02x}\n")
        else:
            print("Error!")
