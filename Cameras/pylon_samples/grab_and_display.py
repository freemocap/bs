from pypylon import pylon
import cv2

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
# new_width = camera.Width.Value - camera.Width.Inc
# if new_width >= camera.Width.Min:
#     camera.Width.Value = new_width

numberOfImagesToGrab = 60
camera.StartGrabbingMax(numberOfImagesToGrab)

frame = -1

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    frame += 1

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        print(f"Current frame: {frame}")
        img = grabResult.Array
        cv2.imshow(f"frame", img)

        if cv2.waitKey(67) & 0xFF == ord("q"):
            break
        

    grabResult.Release()

cv2.destroyAllWindows()
camera.Close()