from ultralytics import YOLO
import os
import shutil
import arabic_reshaper
from moviepy.editor import VideoFileClip

# Set the working directory
os.chdir(r"C:\Users\moham\OneDrive\سطح المكتب\test\signlanguagetest")

# Delete the 'done' folder if it exists
for folder in os.listdir('C:/Users/moham/OneDrive/سطح المكتب/test/signlanguagetest'):
    if folder == "done":
        shutil.rmtree("C:/Users/moham/OneDrive/سطح المكتب/test//signlanguagetest/done")
    else:
        continue

# Function to get label
labels = {'ALF': "ألف", 'BA': "باء", 'THAA': "تاء", 'THA': "ثاء", 'GEEM': "جيم", 'HAA': "حاء", 'KHA': "خاء", 'DAL': "دال", 'ZAL': "ذال", 'RAA': "راء",
          'ZAIN': "زين", 'SEEN': "سين", 'SHEEN': "شين", 'SAAD': "صاض", 'DAAD': "ضاض", 'TAH': "طاء", 'ZAH': "ظاء", 'AEN': "عين", 'GHEN': "غين", 'FAA': "فاء",
          'KAAF': "قاف", 'CAAF': "كاف", 'LAM': "لام", 'MEEM': "ميم", 'NOON': "نون", 'HA': "هاء", "WAW": "واو","YAA":"ياء","ONE":"واحد","TOW":"اثنين","THREE":"ثلاثه","FOUR":"أربعه",
          "FIVE":"خمسه","SIX":"سته","SEVEN":"سبعه","EIGHT":"ثمانيه","NINE":"تسعه","TEN":"عشرة","ADOU":"عدو","AML GEAD":"عمل جيد","ANA ASIF":"أنا أسف","ANTA":"انت",
          "ATAMANA LAK HAYAH SAEIDAH":"أتمني لك حياه سعيده","HAZA RAHIB":"هذا رهيب","HUSAN":"حصان","MANZIL":"منزل","MUTHALATH":"مثلث",
          "NAJAAR":"نجار","MUDIR":"مدير","OUHEBK GEDN":"أحبك جدا","SADEEK":"صديق","ZAWAJ":"زواج","YAWM ALAHAD":"يوم الأحد"}


# Function to get label
list_of_labels = []
def get_label(label):
    for key, value in labels.items():
        if key == label:
            return value
    return label


# Function to get video length
def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    duration_seconds = clip.duration
    clip.close()
    return duration_seconds


# Provide the path to your video file
input_path = 'C:/Users/moham/OneDrive/سطح المكتب/test//signlanguagetest/test2.mp4'

# Calculate the video length
length = get_video_length(input_path)
length = int(length * 2)

# Initialize YOLO model
model = YOLO('best.pt')

# Perform prediction
results = model.predict(source=input_path, conf=0.5, stream=True, save=True, vid_stride=length, show=True, save_crop=True, project='C:/Users/moham/OneDrive/سطح المكتب/test/signlanguagetest', name='done')

# Iterate over inference results
for r in results:
    # Access inference results for the current frame
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

# Get output of predet from this pass
for folder in os.listdir('C:/Users/moham/OneDrive/سطح المكتب/test/signlanguagetest/done/crops'):
    list_of_labels.append(folder)

# Convert English output to Arabic language
last_output = [get_label(label) for label in list_of_labels]

# Print the items of the list of outputs
for output in last_output:
    print(output)