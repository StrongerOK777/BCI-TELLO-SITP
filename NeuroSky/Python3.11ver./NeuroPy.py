import serial
import time
import _thread

print("DEBUG: Using local NeuroPy.py from current directory")

class NeuroPy(object):
    def __init__(self, port, baudRate=57600):
        self.__port, self.__baudRate = port, baudRate
        self.attention = 0
        self.meditation = 0
        self.rawValue = 0
        self.delta = 0
        self.theta = 0
        self.lowAlpha = 0
        self.highAlpha = 0
        self.lowBeta = 0
        self.highBeta = 0
        self.lowGamma = 0
        self.midGamma = 0
        self.poorSignal = 0
        self.blinkStrength = 0
        self.srl = None
        self.threadRun = False

    def start(self):
        self.threadRun = True 
        try:
            self.srl = serial.Serial(self.__port, self.__baudRate, timeout=0.1)
            self.srl.flushInput()  # 清空输入缓冲区，防止读取旧数据
            self.srl.flushOutput() # 清空输出缓冲区
            _thread.start_new_thread(self.__packetParser, (self.srl,))
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.threadRun = False

    def stop(self):
        self.threadRun = False
        if self.srl:
            self.srl.close()

    def __packetParser(self, srl):
        while self.threadRun:
            # Sync
            b = srl.read(1)
            if not b or b[0] != 0xAA: continue
            b = srl.read(1)
            if not b or b[0] != 0xAA: continue

            # Length
            b = srl.read(1)
            if not b: continue
            payloadLength = b[0]
            if payloadLength > 169: continue

            # Payload
            payload = srl.read(payloadLength)
            if len(payload) < payloadLength: continue

            # Checksum
            checksum_received = srl.read(1)
            if not checksum_received: continue
            
            checksum_calculated = sum(payload) & 0xFF
            checksum_calculated = (~checksum_calculated) & 0xFF

            if checksum_calculated != checksum_received[0]:
                continue

            # Parse Payload
            i = 0
            while i < payloadLength:
                code = payload[i]
                if code == 0x02: # poorSignal
                    i += 1
                    self.poorSignal = payload[i]
                    if self.poorSignal > 0:
                        print(f"Signal Quality Warning: {self.poorSignal}")
                    i += 1
                elif code == 0x04: # attention
                    i += 1
                    self.attention = payload[i]
                    if self.attention > 0:
                        print(f"--- GOT ATTENTION: {self.attention} ---")
                    i += 1
                elif code == 0x05: # meditation
                    i += 1
                    self.meditation = payload[i]
                    if self.meditation > 0:
                        print(f"--- GOT MEDITATION: {self.meditation} ---")
                    i += 1
                elif code == 0x16: # blink
                    i += 1
                    self.blinkStrength = payload[i]
                    i += 1
                elif code == 0x80: # raw
                    i += 1 # skip code
                    data_len = payload[i]
                    i += 1 # skip length
                    val = payload[i] * 256 + payload[i+1]
                    if val > 32768: val -= 65536
                    self.rawValue = val
                    i += data_len
                elif code == 0x83: # EEG Power
                    i += 1 # skip code
                    data_len = payload[i]
                    i += 1 # skip length
                    self.delta = (payload[i] << 16) | (payload[i+1] << 8) | payload[i+2]
                    self.theta = (payload[i+3] << 16) | (payload[i+4] << 8) | payload[i+5]
                    self.lowAlpha = (payload[i+6] << 16) | (payload[i+7] << 8) | payload[i+8]
                    self.highAlpha = (payload[i+9] << 16) | (payload[i+10] << 8) | payload[i+11]
                    self.lowBeta = (payload[i+12] << 16) | (payload[i+13] << 8) | payload[i+14]
                    self.highBeta = (payload[i+15] << 16) | (payload[i+16] << 8) | payload[i+17]
                    self.lowGamma = (payload[i+18] << 16) | (payload[i+19] << 8) | payload[i+20]
                    self.midGamma = (payload[i+21] << 16) | (payload[i+22] << 8) | payload[i+23]
                    i += data_len
                else:
                    i += 1
