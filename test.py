a = 'KNXMNSLKWJXMBFYJWGJSIXFIRNYXBTWIKNXMWFSITAJWMJQRNSLFSDIFD'
for i in range (26):
    result = ''
    for char in a:
        temp = ord(char) - i
        if(temp>ord('A')):
            result += chr(temp-1)
        else:
            temp = ord('Z') - (ord('A')-temp) 
            result += chr(temp)
    print((i,result))