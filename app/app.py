import os
import datetime
import cv2
from flask import Flask
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv

# Linking the database
load_dotenv()
url: str = os.getenv("DATABASE_URL")
key: str = os.getenv("DATABASE_KEY")
db_client: Client = create_client(url, key)

# Importing the models
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
helmet = YOLO(os.path.join(__location__, 'last.pt'))
bike = YOLO(os.path.join(__location__, 'yolov9e.pt'))

# Creating Flask object
app = Flask(__name__)

# Creating APIs
@app.route('/')
def index():
    return "Title: Real-time helmet detection \n Date Created: 26-07-2024"

@app.route('/processImage', methods = ['POST', 'GET'])
def processImage():
    img = cv2.imread("E:/Code/CV/images/bike_1.jpg")

    # first we call predict on image
    location_camera = "HOME"
    time = datetime.now()
    time_now = time.strftime('%H:%M:%S')
    date = datetime.date.today()
    count_tuple = predict(img)
    count_helmet, count_no_helmet = count_tuple[0], count_tuple[1]

    # then we commit insights to database
    flag_insert = storeCount(time_now, location_camera, date, count_helmet, count_no_helmet)

    # perform logging
    pass

    # return 200 unless error then 500
    return "DONE"

def predict(img):
    results_helmet = helmet.predict(img, imgsz = 1500, conf = 0.2)
    results_bike = bike.predict(img, imgsz = 1280, conf = 0.2)
    helmets_list = []
    bikes_list = []
    for result_bike in results_bike:
        names = result_bike.names
        for box in result_bike.boxes:
            name = names[int(box.cls[0])]
            if name != 'motorcycle':
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bikes_list.append((x1, y1, x2, y2))
    for result_helmet in results_helmet:
        names = result_helmet.names
        for box in result_helmet.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            helmets_list.append((x1, y1, x2, y2))
    count_tuple = count(bikes_list, helmets_list)
    count_helmet, count_no_helmet = count_tuple[0], count_tuple[1]
    return (count_helmet, count_no_helmet)

def count(bikes, helmets_in, expanding_factor = 0.30):
    """
    Count the number of bikes with and without helmets based on their bounding boxes.

    This function attempts to pair each bike with a helmet by checking if a helmet's
    bounding box is within an expanded area above the bike's bounding box. The expanded
    area is determined by the bike's dimensions and the expanding_factor.

    Parameters:
    -----------
    bikes : list of tuple
        A list of bike bounding boxes. Each box is represented as a tuple of four 
        values (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) 
        is the bottom-right corner of the bounding box.

    helmets_in : list of tuple
        A list of helmet bounding boxes, in the same format as the bikes list.

    expanding_factor : float, optional (default=0.30)
        A factor to expand the search area for helmets around each bike. This allows
        for some flexibility in the relative positioning of helmets to bikes.

    Returns:
    --------
    tuple of int
        A tuple containing two values:
        - count_helmet: The number of bikes with a detected helmet.
        - count_no_helmet: The number of bikes without a detected helmet.

    Notes:
    ------
    - The function modifies a copy of the helmets_in list, removing helmets as they
      are matched to bikes to prevent double-counting.
    - A helmet is considered matched to a bike if it's within the expanded area above
      the bike, which is defined as:
      * Horizontally: within the bike's width, expanded by expanding_factor.
      * Vertically: from slightly below the top of the bike to one bike height above,
        with both limits expanded by expanding_factor.
    - The function stops searching for a helmet once it finds a match for a bike.
    - Bounding box coordinates are adjusted to ensure x1 < x2 and y1 < y2.

    Example:
    --------
    >>> bikes = [(100, 100, 200, 200), (300, 300, 400, 400)]
    >>> helmets = [(120, 50, 180, 90), (320, 250, 380, 290)]
    >>> count_helmet(bikes, helmets)
    (2, 0)
    """ 
    # the logic is to find a helmet that is directly above the bike, within bike_height of the top of the bike
    # essentially, if a helmet is found in an area that is the same as the bike_box, but on top of it, then we do + helmet count. 
    # else + no_helmet count
    # no need to think about "the closest helmet", as there is no metric on which we can decide a helmet to be "closer" in this method of consideration.
    # A helmet is either in the zone and then thus accounted for and deleted, or it is not in the zone and skipped.
    # And to consider the cases in which helmets may be "partially" in the area, the "allowable" area has been made flexible by an input "expanding_factor"

    helmets = helmets_in.copy()
    count_helmet = 0
    count_no_helmet = 0
    for bike in bikes:
        f = False
        for helmet in helmets:
            bx1, by1, bx2, by2 = bike
            hx1, hy1, hx2, hy2 = helmet

            # ensure bx1 is min x, and by1 is min y
            t1, t2, t3, t4 = bx1, by1, bx2, by2
            bx1 = min(t1, t3)
            bx2 = max(t1, t3)
            by1 = min(t2, t4)
            by2 = max(t2, t4)

            # do the same for hx1, ...
            t1, t2, t3, t4 = hx1, hy1, hx2, hy2
            hx1 = min(t1, t3)
            hx2 = max(t1, t3)
            hy1 = min(t2, t4)
            hy2 = max(t2, t4)
            # check if the helmet is horizontally placed on "top" of the bike box
            bike_box_width = bx2 - bx1
            if hx1 < bx1 - (bike_box_width * expanding_factor) or hx2 > bx2 + (bike_box_width * expanding_factor):
                continue        # the helmet is outside of the region

            # check if the helmet is in the vertical range of bike box height now
            bike_box_height = by2 - by1
            if hy1 < by2 - (bike_box_height * expanding_factor) or hy2 > by2 + bike_box_height + (bike_box_height * expanding_factor):
                continue

            count_helmet += 1
            helmets.remove(helmet)
            f = True
            break

        # if no helmet was ever found, then count no helmet
        if not f:  
            count_no_helmet += 1

    return (count_helmet, count_no_helmet)

def storeCount(time_now, location_camera, date, count_helmet, count_no_helmet):
    response = (
        db_client.table("HelmetDetection")
        .insert({"time": time_now, "location":  location_camera, "date": date, "count_helmet": count_helmet, "count_no_helmet": count_no_helmet})
        .execute()
    )
    print(response)
    return "0"

def display():
    print("Printing table")
    response = db_client.table("HelmetDetection").select("*").execute()
    return response

# Running the app
if __name__ == "__main__":
    #app.run(debug = True