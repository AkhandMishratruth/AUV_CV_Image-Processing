import socket

host='192.168.43.12'
port=7000

server=socket.socket()
server.connect((host,port))
while True:
    data = server.recv(1024)
    print data
server.close()
