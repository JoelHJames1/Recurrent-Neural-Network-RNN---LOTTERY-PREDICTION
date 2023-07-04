# Read input from the text file
with open('input.txt', 'r') as file:
    lines = file.readlines()

# Reverse the order of the lines
reversed_lines = reversed(lines)

# Write the reversed lines to a new text file
with open('output.txt', 'w') as file:
    file.writelines(reversed_lines)
