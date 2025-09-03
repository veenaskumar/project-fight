<<<<<<< HEAD
# nightshield-ai
=======
>>>>>>> 0799432be215e26229531ac13deeaf92ce2cf829
# Project-Fight: Violence Detection with YOLOv8
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/veenaskumar/project-fight)

This project provides a real-time violence detection system using a YOLOv8 model. The application is built with Streamlit and can process images, video files, and live RTSP streams to identify and log instances of violence.

## Features
*   **Multi-Source Detection:** Analyze static images (`.jpg`, `.png`), pre-recorded videos (`.mp4`, `.avi`, `.mov`), and live RTSP camera feeds.
*   **Web Interface:** A user-friendly interface built with Streamlit for easy interaction and monitoring.
*   **Incident Logging:** Detected events are automatically logged with a timestamp, source, and confidence score into `violence_detection_log.json`. Logs older than 24 hours are automatically removed.
*   **Incident Review:** A dedicated "Incidents" page displays a table of all detections from the last 24 hours for easy review.
*   **Adjustable Confidence:** Users can set a custom confidence threshold via a slider to tune the sensitivity of the detection model.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/veenaskumar/project-fight.git
    cd project-fight
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    Execute the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

2.  **Access the web interface:**
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Using the Application:**
    *   **Detection Page:**
        *   **File Upload:** Use the file uploader to select an image or video. The application will process the file and display the output with bounding boxes around any detected violence.
        *   **RTSP Stream:** In the sidebar, enter the URL of an RTSP stream, adjust the confidence threshold, and click "Start RTSP Stream" to begin real-time analysis.
    *   **Incidents Page:**
        *   Select "Incidents" from the sidebar navigation to view a log of all detected violent events within the last 24 hours.

## Project Files
*   `app.py`: The core Streamlit application logic for the UI, file/stream processing, and model inference.
*   `violence-detection-through-cctv_step3.pt`: The final YOLOv8 model used for violence detection.
*   `requirements.txt`: A list of Python dependencies required to run the project.
*   `violence_detection_log.json`: A log file that stores details of detected incidents.
*   `rwf-2000_step1.pt` & `fight_detection-m9aq1_step2 (1).pt`: Additional model files, likely from intermediate training steps.
