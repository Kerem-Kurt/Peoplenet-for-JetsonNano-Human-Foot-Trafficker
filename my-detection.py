from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import csv
import datetime

# Initialize the object detection network (peoplenet-pruned) with a confidence threshold of 0.4
net = detectNet("peoplenet-pruned", threshold=0.4)

# Initialize the camera as the video source (CSI camera) and display as the video output
camera = videoSource("csi://0")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file

# Enable object tracking with specific parameters
net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.3)

# Initialize dictionaries and lists for tracking objects and their positions
centersDict = {}
trackList = []

# Initialize counters for ingoing and outgoing people
ingoers = 0
outgoers = 0

# Main processing loop
while display.IsStreaming():
    # Capture a frame from the camera
    img = camera.Capture()

    if img is None: # Capture timeout
        continue

    # Detect objects in the frame using the object detection network
    detections = net.Detect(img)

    # Iterate through detected objects
    for detection in detections:
        if detection.ClassID != 1:
            # Filter out objects that are not classified as people
            detections.remove(detection)
            continue

        if detection.TrackID not in trackList:
            # Track new objects by adding them to the tracking list and initializing their position history
            trackList.append(detection.TrackID)
            centersDict[detection.TrackID] = []

    # Update the position history of tracked objects and print tracking information
    for detection in detections:
        if detection.TrackStatus >= 0:
            if detection.TrackID in trackList:
                if detection.TrackID == -1:
                    continue
                # Add the X-coordinate of the object's center to its position history
                centersDict[detection.TrackID].append((detection.Left + detection.Right) / 2)
            print(f"object {detection.TrackID} at ({(detection.Left + detection.Right) / 2}, {(detection.Top + detection.Bottom) / 2}) has been tracked for {detection.TrackFrames}")
        else:
            print(f"object {detection.TrackID} has lost tracking")
            if detection.ClassID == 1:
                if detection.TrackID not in centersDict.keys():
                    continue
                # Remove lost objects from tracking lists
                del centersDict[detection.TrackID]
                trackList.remove(detection.TrackID)

    # Check if a tracked object has crossed an entry or exit threshold and update the counters accordingly
    for key in centersDict.keys():
        if len(centersDict[key]) > 2:
            if centersDict[key][-2] < 640:
                if centersDict[key][-1] >= 640:
                    # Increment the ingoers count and log the entry event to a CSV file
                    ingoers += 1
                    with open('my_file.csv', 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow(["IN", datetime.date.today(), datetime.datetime.now().time()])
            elif centersDict[key][-1] < 640:
                # Increment the outgoers count and log the exit event to a CSV file
                outgoers += 1
                with open('my_file.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(["OUT", datetime.datetime.now().time()])

    # Print the counts and position history for debugging
    print("outgoers==" + str(outgoers))
    print("ingoers==" + str(ingoers))
    print(centersDict)

    # Render the frame with object bounding boxes and tracking information
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

# Exit the loop and close the display
