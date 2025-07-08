import serial
import threading
import time
from typing import Optional

class SendTTL:
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.running = False
        
    def connect(self) -> bool:
        """Establish connection to the serial device."""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            return True
        except Exception as e:
            print(f"Error connecting to device: {e}")
            return False
            
    def disconnect(self):
        """Close the serial connection."""
        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None
            
    def _send_pulse(self, ttl_line: bytes):
        """Send a single TTL pulse."""
        if self.serial_connection:
            try:
                self.serial_connection.write(ttl_line)
            except Exception as e:
                print(f"Error sending pulse: {e}")

    def start_background(self, ttl_line: bytes = b'\x01'):
        """
        Start sending TTL pulses at 90Hz in background thread.
        
        Args:
            ttl_line: The command to send for high state (default: b'\x01')
        """
        if not self.serial_connection:
            if not self.connect():
                raise RuntimeError("Failed to connect to device")
                
        self.running = True
        threading.Thread(target=self._run_loop, args=(ttl_line,), daemon=True).start()
        
    def stop(self):
        """Stop sending TTL pulses."""
        self.running = False
        
    def _run_loop(self, ttl_line: bytes):
        """
        Main loop for sending TTL pulses at 90Hz.
        
        Period calculation:
        90 Hz = 90 cycles per second
        Each cycle duration = 1 / 90 ≈ 0.011111 seconds
        High pulse width = 50% duty cycle = 0.005555 seconds
        Low pulse width = 0.005555 seconds
        """
        period = 1/90  # Total cycle duration
        high_duration = period / 2  # 50% duty cycle
        
        while self.running:
            start_time = time.perf_counter()
            
            # Send high pulse
            self._send_pulse(ttl_line)
            time.sleep(high_duration)
            
            # Send low pulse
            self._send_pulse(b'\x00')
            time.sleep(period - high_duration * 2)
            
            # Adjust timing to maintain precise frequency
            elapsed = time.perf_counter() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

if __name__ == "__main__":
    ttl_sender = SendTTL(port='COM3')  # Replace 'COM3' with your port
    try:
        ttl_sender.start_background()
        print("Sending TTL pulses at 90Hz. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping TTL pulses.")
    finally:
        ttl_sender.stop()
        ttl_sender.disconnect()
        print("Disconnected.")