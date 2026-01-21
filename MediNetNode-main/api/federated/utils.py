import torch
from datetime import datetime
from opacus.validators import ModuleValidator

try:
    import django
    django.setup()
    from trainings.models import TrainingRound
    from django.contrib.auth import get_user_model
    DJANGO_AVAILABLE = True
    User = get_user_model()
    
except ImportError as e:
    print(f"Warning: Django models not available for training tracking: {e}")
    DJANGO_AVAILABLE = False
    

def flatten_with_prefix(config, prefix="", delimiter="__"):
    """
    Flattens a nested dictionary and adds a prefix or suffix to keys for context.

    Args:
        config (dict): The nested dictionary to flatten.
        prefix (str, optional): The prefix to add to keys. Defaults to "".
        delimiter (str, optional): The delimiter to use between prefix and key. Defaults to "__".

    Returns:
        dict: A flattened dictionary with prefixed keys.
    """
    flat_config = {}
    for key, value in config.items():
        new_key = f"{prefix}{delimiter}{key}" if prefix else key
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat_config.update(flatten_with_prefix(value, prefix=new_key, delimiter=delimiter))
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to strings
            flat_config[new_key] = str(value)
        else:
            flat_config[new_key] = value
    return flat_config

def unflatten_with_prefix(flat_config, delimiter="__"):
    """
    Reconstructs a nested dictionary from a flattened dictionary with prefixed keys.

    Args:
        flat_config (dict): The flattened dictionary with prefixed keys.
        delimiter (str, optional): The delimiter used between prefix and key. Defaults to "__".

    Returns:
        dict: A nested dictionary reconstructed from the flattened dictionary.
    """
    nested_config = {}

    for key, value in flat_config.items():
        parts = key.split(delimiter)
        current_level = nested_config

        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        
        if isinstance(value, str):
            try:
                value = eval(value)
            except (SyntaxError, NameError):
                pass
        current_level[parts[-1]] = value
    
    return nested_config

def check_model(net:torch.nn.Module):
    """
    Validates and fixes a PyTorch model using Opacus ModuleValidator.

    Args:
        net (torch.nn.Module): The PyTorch model to validate and fix.

    Returns:
        torch.nn.Module: The validated and fixed PyTorch model.
    """
    errors = ModuleValidator.validate(net, strict=False)
    print(f"Model validated with {len(errors)} errors")
    if len(errors) > 0:
        print("Fixing model")
        net = ModuleValidator.fix(net)
        errors = ModuleValidator.validate(net, strict=False)
        print("Model errors now after fixing: ", len(errors))
        print("Errors in model: \n", errors)
    return net


def update_training_progress(training_session,round_number, current_process, metrics=None):
    """Update training progress and create round record."""
    
    if not DJANGO_AVAILABLE or not training_session:
        return
    
    try:
        # Update resource usage
        if current_process:
            try:
                cpu_percent = current_process.cpu_percent()
                memory_info = current_process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                training_session.cpu_usage = cpu_percent
                training_session.memory_usage = memory_mb
            except:
                pass
        
        # Update progress and status (first call marks as ACTIVE)
        training_session.current_round = round_number
        if training_session.status == 'STARTING':
            training_session.status = 'ACTIVE'
            # Training session activated
        elif training_session.status != 'ACTIVE':
            training_session.status = 'ACTIVE'
        
        if training_session.total_rounds > 0:
            training_session.progress_percentage = (round_number / training_session.total_rounds) * 100
        
        # Save current round state for persistence across Flower client restarts
        training_session.save(update_fields=['current_round', 'status', 'progress_percentage', 'cpu_usage', 'memory_usage'])
                
        # Create round record if metrics provided
        if metrics:
            round_record = TrainingRound(
                session=training_session,
                round_number=round_number,
                loss=metrics.get('loss'),
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1')
            )
            
            # Add resource usage to round
            if current_process:
                try:
                    round_record.cpu_usage = current_process.cpu_percent()
                    memory_info = current_process.memory_info()
                    round_record.memory_usage = memory_info.rss / 1024 / 1024
                except:
                    pass
            
            round_record.save()
            round_record.complete_round(**metrics)
            
            print(f"[INFO] Round {round_number} completed - Loss: {metrics.get('loss', 'N/A'):.4f}, Acc: {metrics.get('accuracy', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
        
    except Exception as e:
        # Error updating progress
        raise e

def complete_training_session(training_session, final_metrics=None):
    """Mark training session as completed with final metrics."""
    
    if not DJANGO_AVAILABLE or not training_session:
        return
    
    try:
        if final_metrics:
            training_session.mark_completed(
                accuracy=final_metrics.get('accuracy'),
                loss=final_metrics.get('loss'),
                precision=final_metrics.get('precision'),
                recall=final_metrics.get('recall'),
                f1=final_metrics.get('f1')
            )
        else:
            training_session.status = 'COMPLETED'
            training_session.completed_at = datetime.now()
            training_session.save()
        
        # Training session completed
        
    except Exception as e:
        # Error completing session
        raise e

def fail_training_session(training_session, error_message, traceback=None):
    """Mark training session as failed with error details."""
    
    if not DJANGO_AVAILABLE or not training_session:
        return
    
    try:
        training_session.mark_failed(error_message, traceback)
        print(f"[ERROR] Training session failed: {training_session.session_id} - {error_message}")
        
    except Exception as e:
        print(f"[ERROR] Error marking training session as failed: {e}")

