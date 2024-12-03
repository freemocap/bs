import pypylon.pylon as py


print("\nCamera Information:")
tl_factory = py.TlFactory.GetInstance() 
for dev in tl_factory.EnumerateDevices():
    dev: py.DeviceInfo
    print(dev.GetFriendlyName())
    try:
        camera = py.InstantCamera(tl_factory.CreateDevice(dev))
        camera.Open()
        print(camera.DeviceFirmwareVersion.Value)
        camera.Close()
    except (py.LogicalErrorException, py.RuntimeException) as error:
        print(f"Error reading camera info: {error}")

print("\nRuntime Information:")
import sys, pypylon.pylon, platform
print(f'python: {sys.version}')
print(f'platform: {sys.platform}/{platform.machine()}/{platform.release()}')
print(f'pypylon: {pypylon.pylon.__version__} / {".".join([str(i) for i in pypylon.pylon.GetPylonVersion()])}')