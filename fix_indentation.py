with open(
    "ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py", "r", encoding="utf-8"
) as f:
    lines = f.readlines()

# Find the last "    return False"
last_return_index = None
for i, line in enumerate(lines):
    if line.strip() == "    return False":
        last_return_index = i

if last_return_index is not None:
    for i in range(last_return_index + 1, len(lines)):
        if lines[i].startswith("    "):
            lines[i] = lines[i][4:]

# Write back
with open(
    "ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio.py", "w", encoding="utf-8"
) as f:
    f.writelines(lines)

print("Indentation fixed")
