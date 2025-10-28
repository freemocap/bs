import cv2
import numpy as np
import time
from datetime import datetime

def test_codec(codec_name, fourcc, file_extension):
    width = 1024
    height = 1024
    frame = np.random.randint(0, 256, size=(width, height), dtype=np.uint8)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{codec_name}.{file_extension}"
    writer = cv2.VideoWriter(
        filename,
        fourcc=fourcc,
        fps=90,
        frameSize=(width, height),
        isColor=False
    )
    
    # Test writing 10000 frames
    frames_written = 0
    start_time = time.time()
    
    while frames_written < 10000:
        success = writer.write(frame)
        if not success:
            break
        frames_written += 1
    
    elapsed_time = time.time() - start_time
    fps = frames_written / elapsed_time
    print(f"{codec_name}: {fps} FPS")
    print(elapsed_time)
    
    return fps


if __name__=="__main__":
    # Test different codecs
    codecs = [
        ("H264", cv2.VideoWriter_fourcc(*'H264'), "mkv"),
        ("MJPG", cv2.VideoWriter_fourcc(*'MJPG'), "avi"),
        ("MP4V", cv2.VideoWriter_fourcc(*'mp4v'), "mp4")
    ]

    for name, fourcc, file_extension in codecs:
        test_codec(name, fourcc, file_extension)