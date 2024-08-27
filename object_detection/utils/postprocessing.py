# Load the configuration
import json


with open('config.json') as config_file:
    config = json.load(config_file)
    
threshold = config['threshold']
COLORS = config['colors']
CLASSES = config['classes']

def process_torch_serve_response(response):
    """
    Processes the JSON response from TorchServe and organizes detections by class.
    """
    detection_results = response.json()

    # Initialize response data structure
    response_data = {
        'detected_classes': {class_name: {'present': False, 'detections': []} for class_name in CLASSES if class_name != '__background__'}
    }

    # Process detections and organize by class
    for box, score, label in zip(detection_results['boxes'], detection_results['scores'], detection_results['labels']):
        class_name = CLASSES[label]
        if score >= threshold:
            response_data['detected_classes'][class_name]['present'] = True
            response_data['detected_classes'][class_name]['detections'].append({
                'box': box,
                'score': score
            })

    return response_data

def aggregate_results(file_result, pdf_results):
    """
    Combine results from all pages of a PDF into a single JSON object.
    Each detection class is considered present if detected on any page.
    """
    detected_classes = {class_name: {'present': False, 'detections': []} for class_name in CLASSES if class_name != '__background__'}
    
    for result in pdf_results:
        for class_name, data in result.get('detected_classes', {}).items():
            if data['present']:
                detected_classes[class_name]['present'] = True
                detected_classes[class_name]['detections'].extend(data['detections'])

    file_result['detected_classes'] = detected_classes