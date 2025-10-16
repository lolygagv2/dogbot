import serial
import time

ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
time.sleep(2)

# Set volume
ser.write(b'AT+VOL=25\r\n')
time.sleep(2)
print('Volume set response:', ser.read(100))

#TURN OFF LED
ser.write(b'AT+LED=OFF\r\n')

# Play modes: 1 is repeat one song, 2 is repeart all, 3 play one song and pause, 4 randoml lay 5 repeat all in folder, ? query mode
ser.write(b'AT+PLAYMODE=5\r\n')
time.sleep(2)

# Play first file
ser.write(b'\r\n')
time.sleep(5)
print('Play response:', ser.read(100))

print("Should be playing audio now - do you hear anything?")
time.sleep(5)

# Stop
ser.write(b'AT+PLAY=PP\r\n')

ser.close()

#FOR PROMPT ON OFF
#ser.write('AT+PROMPT=ON\r\n')

#FOR AMP ON OFF
#ser.write('AT+AMP=ON\r\n')

#FOR PLAY FILE BY NAME  
#ser.write('AT+PLAYFILE=/FOLDER/filename.MP3\r\n')

#PLAY NEXT SONG
#ser.write('AT+PLAY=NEXT\r\n')

#VOL control  use-n, +n, or just n.  0-30 is the range
#ser.write('AT+VOL=25\r\n')

# AT just AT tests the connection
#ser.write('AT\r\n')