from rest_framework.views import exception_handler


def standardized_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is None:
        return response

    payload = {
        'error': {
            'code': 'api_error',
            'message': 'Request processing failed.',
            'details': response.data,
        }
    }

    if response.status_code == 400:
        payload['error']['code'] = 'validation_error'
        payload['error']['message'] = 'Input validation failed.'
    elif response.status_code == 401:
        payload['error']['code'] = 'unauthorized'
        payload['error']['message'] = 'Authentication credentials were missing or invalid.'
    elif response.status_code == 403:
        payload['error']['code'] = 'forbidden'
        payload['error']['message'] = 'You do not have permission to access this resource.'
    elif response.status_code == 404:
        payload['error']['code'] = 'not_found'
        payload['error']['message'] = 'Requested resource was not found.'
    elif response.status_code == 429:
        payload['error']['code'] = 'rate_limited'
        payload['error']['message'] = 'Request rate limit exceeded.'

    response.data = payload
    return response
