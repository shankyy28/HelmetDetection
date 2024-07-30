from flask import request, jsonify
from datetime import datetime

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
                address=address, 
                error_json=outcome,
                severity=3,
                db_client=db_client
            )
            return outcome
        
        if field == 'timestamp':
            timestamp = source[field]
            if not validate_timestamp(timestamp):
                outcome = jsonify({'error': 'Invalid timestamp format. Use ISO 8601 format.'}), 400
                log_error(
                    timestamp=timestamp, 
                    address=address, 
                    error_json=outcome,
                    severity=3,
                    db_client=db_client
                )
                return outcome
        elif field == 'address':
            address = source[field]

    return {
        'timestamp': timestamp,
        'address': address,
        'image': files['image']
    }

def validate_timestamp(timestamp_str):
    try:
        # Attempt to parse the timestamp string
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False