import sys

if len(sys.argv) > 1:
    #from sys arguments
    file = open(sys.argv[1], 'r', encoding='UTF-8')
else:
    #User input
    filename = input('filename: ')
    file = open(filename, 'r', encoding='UTF-8')

content = file.read()
file.close

print('Running on platform {pc.platform}'.format(pc = sys))
print(content)
#print(content.encode('UTF-8'))
#print(content.encode('UTF-8').decode('UTF-8'))
numbers = []
for number in range(20):
    numbers.append(str(number))
print(", ".join(numbers))

print(", ".join([str(number) for number in range(20)]))
