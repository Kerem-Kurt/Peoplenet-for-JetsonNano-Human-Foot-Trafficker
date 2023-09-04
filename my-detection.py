from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import csv
import datetime

net = detectNet("peoplenet-pruned", threshold=0.4)
camera = videoSource("csi://0")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file
net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.3)
centersDict = {}
trackList = []
ingoers = 0
outgoers = 0



while display.IsStreaming():
    img = camera.Capture()

    if img is None: # capture timeout
        continue

    detections = net.Detect(img)
	
    for detection in detections:
        if detection.ClassID != 1:
            detections.remove(detection)
            continue
        if detection.TrackID not in trackList:
            trackList.append(detection.TrackID)
            centersDict[detection.TrackID] = []

    for detection in detections:
        if detection.TrackStatus >= 0:
            if detection.TrackID in trackList:
                if detection.TrackID == -1:
                    continue
                centersDict[detection.TrackID].append((detection.Left + detection.Right)/2)
            print(f"object {detection.TrackID} at ({(detection.Left + detection.Right)/2}, {(detection.Top + detection.Bottom)/2}) has been tracked for {detection.TrackFrames}")
        else:
            print(f"object {detection.TrackID} has lost tracking")
            if detection.ClassID == 1:
                if detection.TrackID not in centersDict.keys(): continue
                del centersDict[detection.TrackID]
                trackList.remove(detection.TrackID)
    
    for key in centersDict.keys():
        if len(centersDict[key]) > 2:
            if centersDict[key][-2] < 640:
                if centersDict[key][-1] >= 640:
                    ingoers += 1
                    with open('my_file.csv', 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow(["IN",datetime.date.today(),datetime.datetime.now().time()])
                        file.close()
            elif centersDict[key][-1] < 640:
                outgoers += 1
                with open('my_file.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(["OUT",datetime.datetime.now().time()])
                    file.close()
    print("outgoers==" + str(outgoers))
    print("ingoers==" + str(ingoers))
    print(centersDict)
        
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
