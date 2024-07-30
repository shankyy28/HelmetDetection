import os
import datetime
import cv2
from flask import Flask
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv
from flask import request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import logging
import json


from my_helpers import check_request


# loading env
load_dotenv()


# configuring logger
log_file_path = os.getenv('LOG_FILE_PATH')

if not log_file_path:
    raise AssertionError("log file path not defined in environment")

logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

logger = logging.getLogger(__name__)

# camera_timestamp (iso time), camera_address (string), log_entry (string). 

def log_error(timestamp, address, error_json):
    try:
        log_entry = {
                'camera_timestamp': timestamp,
                'camera_address': address,
                'log_entry': json.dumps(error_json)
            }
        response = (
                db_client.table("ErrorLogs")
                .insert(log_entry)
                .execute()
            )
        if response.error:
            logger.error(f"Failed to log to Supabase: {response.error.message}")
    except Exception as e:
        logger.error(f"Exception occurred while logging to Supabase: {str(e)}")
        raise e


# Linking the database
url = os.getenv("DATABASE_URL")
key = os.getenv("DATABASE_KEY")

if not url:
    raise AssertionError("database URL not defined in environment")

if not key:
    raise AssertionError("database API key not defined in environment")

db_client = create_client(url, key)



# Get model paths from environment variables
helmet_model_path = os.getenv('HELMET_MODEL_PATH')
bike_model_path = os.getenv('BIKE_MODEL_PATH')

# Check if the environment variables are set
if not helmet_model_path or not bike_model_path:
    raise AssertionError("database API key not defined in environment")


# Importing the models
helmet = YOLO(helmet_model_path)
bike = YOLO(bike_model_path)




####### Creating Flask object
app = Flask(__name__)

# Creating APIs
@app.route('/')
def index():
    return "Title: Real-time helmet detection \n Date Created: 26-07-2024"


@app.route('/processImage', methods = ['POST'])
def processImage():
    outcome = None 
    # Check if the request has the required fields (as of comment required is time, address, image file)
    result = check_request(log_error)

    if isinstance(result, tuple):  # This means an error was returned
        return result
    
    # If we get here, all required fields were provided
    timestamp = result['timestamp']
    camera_location = result['address']
    file = result['image']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        outcome = jsonify({'error': 'No selected file'}), 400
        log_error(
            timestamp=timestamp,
            address=camera_location,
            error_json=outcome
        )
        return outcome
    
    if file:
        # Secure the filename to prevent malicious input
        filename = secure_filename(file.filename)
        # Save the file
        file.save(os.path.join(app.config['FLASK_UPLOAD_FOLDER'], filename))
    

    # Process the image, timestamp, and address here
    img = cv2.imread(file)

    # first we call predict on image
    predict_result = predict(img)
    if isinstance(predict_result, tuple):
        return predict_result
     
    count_helmet, count_no_helmet = predict_result['count_helmet'], predict_result['count_no_helmet']


    # then we commit insights to database
    flag_insert = storeCount(timestamp, camera_location ,count_helmet, count_no_helmet)

    # if there was an error during insertion then log the error
    if flag_insert != "0":
        outcome = jsonify({'error': 'Some error occured in saving data'}), 500
        return outcome

    # return 200 unless error then 500
    return "DONE"

def predict(img, timestamp, camera_location):
    try:
        bikes = []
        helmets = []
        results_helmet = helmet.predict(img, imgsz = 1500, conf = 0.2)
        results_bike = bike.predict(img, imgsz = 1280, conf = 0.2)
        for result_bike in results_bike:
            names = result_bike.names
            for box in result_bike.boxes:
                name = names[int(box.cls[0])]
                if name != 'motorcycle':
                    continue
                # fetch whatever is need to count
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                name = names[cls]
                if name != 'motorcycle':
                    continue
                
                bikes.append([x1, y1, x2, y2])
                # testing
                count_bike += 1


        for result_helmet in results_helmet:
            names = result_helmet.names
            for box in result_helmet.boxes:
                # fetch whatever is need to count
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                name = names[cls]
                if name == "with helmet":
                    helmets.append([x1, y1, x2, y2])

        # call count function
        count_helmet, count_no_helmet = count_helmet(bikes, helmets)


        return {
        'count_helmet': count_helmet,
        'count_no_helmet': count_no_helmet
        }
    except Exception as e:
        log_error(timestamp, camera_location, str(e))
        return jsonify({'error': 'Error during model inference'}), 500


def count_helmet(bikes, helmets_in, expanding_factor = 0.30):
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

    # Function implementation goes here
    count_helmet, count_no_helmet = 0, 0 
    # the logic is to find a helmet that is directly above the bike, within bike_height of the top of the bike
    # essentially, if a helmet is found in an area that is the same as the bike_box, but on top of it, then we do + helmet count. 
    # else + no_helmet count
    # no need to think about "the closest helmet", as there is no metric on which we can decide a helmet to be "closer" in this method of consideration.
    # A helmet is either in the zone and then thus accounted for and deleted, or it is not in the zone and skipped.
    # And to consider the cases in which helmets may be "partially" in the area, the "allowable" area has been made flexible by an input "expanding_factor"

    helmets = helmets_in.copy()
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

def storeCount(timestamp, camera_location, count_helmet, count_no_helmet):
    try:
        response = (
            db_client.table("HelmetDetection")
            .insert({"time": timestamp, "location": camera_location, "count_helmet": count_helmet, "count_no_helmet": count_no_helmet})
            .execute()
        )
        if response.error:
            log_error(timestamp, camera_location, response.error)
            return response.error.message
        return "0"
    except Exception as e:
        log_error(timestamp, camera_location, str(e))
        return str(e)

def display():
    print("Printing table")
    response = db_client.table("HelmetDetection").select("*").execute()
    return response

# Running the app
if __name__ == "__main__":
    app.run(debug = True)
