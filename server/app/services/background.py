import logging

logger = logging.getLogger(__name__)


def run_task_with_fallback(task_func, sync_callable, *args, **kwargs):
    """Try async task dispatch and fallback to sync execution if dispatch fails."""
    try:
        task_func.delay(*args, **kwargs)
        return {'mode': 'async'}
    except Exception as exc:
        logger.warning('Async task dispatch failed, running sync fallback: %s', exc)
        result = sync_callable(*args, **kwargs)
        return {'mode': 'sync_fallback', 'result': result}
