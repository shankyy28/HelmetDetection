from flask import request, jsonify

def check_request(log_error, db_client, form, files):
    required_fields = {
        'timestamp': form,
        'address': form,
        'image': files
    }
    timestamp, address = None, None
    for field, source in required_fields.items():
        if field not in source:
            outcome = jsonify({'error': f'No {field} provided'}), 400
            log_error(
                timestamp=timestamp, 
                camera_location=address, 
                error_json= outcome,
                severity=3,
                db_client=db_client
            )
            return outcome

    return {
        'timestamp': timestamp,
        'address': address,
        'image': request.files['image']
    }