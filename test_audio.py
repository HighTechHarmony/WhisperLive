#!/usr/bin/env python3
import pyaudio
import numpy as np

def main():
    p = pyaudio.PyAudio()
    # Open the exact format Whisper expects
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                    input=True, frames_per_buffer=2048)
    
    print("[*] PyAudio is listening. Make some noise! Press Ctrl+C to stop.")
    try:
        while True:
            # exception_on_overflow=False prevents crashes if PipeWire glitches
            data = stream.read(2048, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Calculate a rough volume level
            vol = np.abs(audio_array).mean()
            bars = "=" * int(vol / 20)
            
            # Print over the same line
            print(f"Vol: {vol:05.1f} | {bars}".ljust(80), end='\r')
            
    except KeyboardInterrupt:
        print("\n[*] Exiting.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()