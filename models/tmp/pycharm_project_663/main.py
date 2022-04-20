a = 'flag{this_is_different_from_hg}'
b = ''
dic = {'{':'openbracket','}':'closebracket',"_":"underline"}
with open("sb") as file:
    for line in file:
        m = line.split(' ', 1)
        dic[m[0].swapcase()] = m[1].swapcase()
for i in a:
    b += dic[i]
print(b)
