# Social-Distancing-Detector
A software for detecting people who are not mainiating proper social distance.

# Workflow of the application:
First the user need to select four points on the static image of the camera being showed as reference points for perspective transformation. The points needs to be the four corners of a any object whose opposites are parallel to each other(parallelogram). The user needs to select the point only in the following order : Top right, top left, bottom right, bottom left.  After selecting the points hit ESC.

After that a live feed window will open with red color bounding boxes on the defaulters. Also the mandatory minimum distance between people to be maintained can be adjusted by the GUI slider that opens up simultaneously with the video frame.
