import socket, json

ip = socket.gethostbyname(socket.gethostname())  # LAN IP (e.g., 192.168.x.x)
with open("screens/ip.json", "w") as f:
    json.dump({"ip": ip}, f)

print("Local IP written to ip.json:", ip)
