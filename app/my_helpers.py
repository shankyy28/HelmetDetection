from flask import request, jsonify

def check_request(log_error):
    required_fields = {
        'timestamp': request.form,
        'address': request.form,
        'image': request.files
    }
    timestamp, address = None, None
    for field, source in required_fields.items():
        if field not in source:
            outcome = jsonify({'error': f'No {field} provided'}), 400
            log_error(timestamp, address, outcome)
            return outcome

    return {
        'timestamp': timestamp,
        'address': address,
        'image': request.files['image']
    }