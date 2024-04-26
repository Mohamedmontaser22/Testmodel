import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import shutil
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# Set the working directory
os.chdir(r"C:\Users\moham\OneDrive\سطح المكتب\test\signlanguagetest")

try:
    # Initialize the YOLO model
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error initializing YOLO model: {e}")
    model = None

# Dictionary to map English labels to Arabic labels
labels = {
     'ALF': "ألف", 'BA': "باء", 'THAA': "تاء", 'THA': "ثاء", 'GEEM': "جيم", 'HAA': "حاء", 'KHA': "خاء", 'DAL': "دال",
    'ZAL': "ذال", 'RAA': "راء", 'ZAIN': "زين", 'SEEN': "سين", 'SHEEN': "شين", 'SAAD': "صاض", 'DAAD': "ضاض",
    'TAH': "طاء", 'ZAH': "ظاء", 'AEN': "عين", 'GHEN': "غين", 'FAA': "فاء", 'KAAF': "قاف", 'CAAF': "كاف",
    'LAM': "لام", 'MEEM': "ميم", 'NOON': "نون", 'HA': "هاء", "WAW": "واو", "YAA": "ياء", "ONE": "واحد",
    "TOW": "اثنين", "THREE": "ثلاثه", "FOUR": "أربعه", "FIVE": "خمسه", "SIX": "سته", "SEVEN": "سبعه",
    "EIGHT": "ثمانيه", "NINE": "تسعه", "TEN": "عشرة", "ADOU": "عدو", "AML GEAD": "عمل جيد", "ANA ASIF": "أنا أسف",
    "ANTA": "انت", "ATAMANA LAK HAYAH SAEIDAH": "أتمني لك حياه سعيده", "HAZA RAHIB": "هذا رهيب", "HUSAN": "حصان",
    "MANZIL": "منزل", "MUTHALATH": "مثلث", "NAJAAR": "نجار", "MUDIR": "مدير", "OUHEBK GEDN": "أحبك جدا",
    "SADEEK": "صديق", "ZAWAJ": "زواج", "YAWM ALAHAD": "يوم الأحد"
}

# Function to get Arabic label
def get_label(label):
    return labels.get(label, label)

# Function to get video length
def get_video_length(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration_seconds = clip.duration
        clip.close()
        return duration_seconds
    except Exception as e:
        print(f"Error getting video length: {e}")
        return None
    
# Define the route for processing video
@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file part in the request.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file.'}), 400

        if file:
            os.makedirs('uploads', exist_ok=True)
            filename = file.filename
            file_path = os.path.join('uploads', file.filename)
            # Create the 'uploads' directory if it doesn't exist
            file.save(file_path)

            # Calculate the video length
            length = get_video_length(file_path)
            if length is None:
                return jsonify({'message': 'Error getting video length.'}), 500
            length = int(length * 2)

            if model:
                # Perform prediction
                results = model.predict(
                    source=file_path,
                    conf=0.5,
                    stream=True,
                    save=True,
                    vid_stride=length,
                    show=True,
                    save_crop=True,
                    project='C:/Users/moham/OneDrive/سطح المكتب/test/signlanguagetest',
                    name='done'
                )

                # Iterate over inference results
                for r in results:
                    # Access inference results for the current frame
                    boxes = r.boxes  # Boxes object for bbox outputs
                    masks = r.masks  # Masks object for segment masks outputs
                    probs = r.probs  # Class probabilities for classification outputs

                # Get output of predet from this pass
                list_of_labels = []

                # Ensure the directory exists before accessing its contents
                done_crops_path = 'C:/Users/moham/OneDrive/سطح المكتب/test/signlanguagetest/done/crops'
                if os.path.exists(done_crops_path):
                    for folder in os.listdir(done_crops_path):
                        list_of_labels.append(folder)

                # Convert English output to Arabic language
                last_output = [get_label(label) for label in list_of_labels]

                # Remove the 'done' folder if it exists
                done_folder_path = os.path.join('C:/Users/moham/OneDrive/سطح المكتب/test/signlanguagetest', 'done')
                if os.path.exists(done_folder_path):
                    shutil.rmtree(done_folder_path)
                if last_output:
                       return jsonify(last_output), 200
                else:
                        return jsonify({'message': ' No label found. '}), 200

    except Exception as e:
        return jsonify({'message': f'An error occurred: {e}'}), 500

# Run the Flask application
if __name__ == '__main__':
     app.run(debug=True)

