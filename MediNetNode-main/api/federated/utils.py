from __future__ import annotations

import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import django
    django.setup()
    from trainings.models import TrainingRound
    from django.contrib.auth import get_user_model
    from dataset.models import DatasetPrivacyPolicy, ResearcherEpsilonBudget
    DJANGO_AVAILABLE = True
    User = get_user_model()

except ImportError as e:
    print(f"Warning: Django models not available for training tracking: {e}")
    DJANGO_AVAILABLE = False
    DatasetPrivacyPolicy = None
    ResearcherEpsilonBudget = None
    

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
            flat_config.update(flatten_with_prefix(value, prefix=new_key, delimiter=delimiter))
        elif isinstance(value, (list, tuple)):
            flat_config[new_key] = str(value)
        else:
            flat_config[new_key] = value
    return flat_config

# DEPRECATED - LEGACY CODE - DO NOT USE
# This function contained a critical security vulnerability (eval() usage)
# and has been removed from active use. Kept for reference only.
#
# SECURITY ISSUE: Used eval() which allows arbitrary code execution
# If you need similar functionality, use ast.literal_eval() instead
#
# def unflatten_with_prefix(flat_config, delimiter="__"):
#     """
#     Reconstructs a nested dictionary from a flattened dictionary with prefixed keys.
#
#     Args:
#         flat_config (dict): The flattened dictionary with prefixed keys.
#         delimiter (str, optional): The delimiter used between prefix and key. Defaults to "__".
#
#     Returns:
#         dict: A nested dictionary reconstructed from the flattened dictionary.
#     """
#     nested_config = {}
#
#     for key, value in flat_config.items():
#         parts = key.split(delimiter)
#         current_level = nested_config
#
#         for part in parts[:-1]:
#             if part not in current_level:
#                 current_level[part] = {}
#             current_level = current_level[part]
#
#         if isinstance(value, str):
#             try:
#                 value = eval(value)  # SECURITY VULNERABILITY - DO NOT USE
#             except (SyntaxError, NameError):
#                 pass
#         current_level[parts[-1]] = value
#
#     return nested_config

def check_model(net: "torch.nn.Module"):
    """
    Validates and fixes a PyTorch model using Opacus ModuleValidator.

    Args:
        net (torch.nn.Module): The PyTorch model to validate and fix.

    Returns:
        torch.nn.Module: The validated and fixed PyTorch model.

    Note:
        torch/opacus are imported lazily here so the pure DB-accounting paths
        in this module (e.g. _record_privacy_spend) can be imported and tested
        without the heavy ML stack installed.
    """
    from opacus.validators import ModuleValidator

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
        elif training_session.status != 'ACTIVE':
            training_session.status = 'ACTIVE'
        
        if training_session.total_rounds > 0:
            training_session.progress_percentage = (round_number / training_session.total_rounds) * 100
        
        # Save current round state for persistence across Flower client restarts
        training_session.save(update_fields=['current_round', 'status', 'progress_percentage', 'cpu_usage', 'memory_usage'])
                
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
        raise e

def _record_privacy_spend(training_session) -> None:
    """Record actual ε spent in DatasetPrivacyPolicy after training completes.

    Reads the highest-numbered TrainingRound's metrics to obtain the accumulated
    privacy_epsilon, then delegates to DatasetPrivacyPolicy.record_spent() which
    applies an atomic conditional DB update to prevent budget overruns.

    This function never raises — recording failure must not affect training status.
    """
    try:
        # Experimental sessions run on the small subset without budget accounting.
        # The use_experiment flag is stored on the session so this check survives
        # subprocess restarts where the original model_json is unavailable.
        if getattr(training_session, 'use_experiment', False):
            logger.info(
                "[DP] Skipping epsilon budget record for experimental session %s",
                getattr(training_session, 'session_id', '?'),
            )
            return

        dataset_id = getattr(training_session, 'dataset_id', None)
        if not dataset_id:
            logger.warning(
                "[DP] training_session %s has no dataset_id — privacy spend not recorded",
                getattr(training_session, 'session_id', '?'),
            )
            return

        try:
            policy = DatasetPrivacyPolicy.objects.get(dataset_id=dataset_id)
        except DatasetPrivacyPolicy.DoesNotExist:
            logger.info(
                "[DP] No DatasetPrivacyPolicy for dataset %s — no spend to record",
                dataset_id,
            )
            return

        rounds = list(training_session.rounds.order_by('round_number'))
        if not rounds:
            logger.warning(
                "[DP] No TrainingRound records for session %s — privacy spend not recorded",
                training_session.session_id,
            )
            return

        per_round = []
        for r in rounds:
            raw_eps = (r.metrics or {}).get('privacy_epsilon')
            if raw_eps is None:
                continue
            try:
                eps = float(raw_eps)
            except (TypeError, ValueError):
                logger.error(
                    "[DP] Non-numeric privacy_epsilon (%r) in round %s of session %s",
                    raw_eps, r.round_number, training_session.session_id,
                )
                continue
            if math.isfinite(eps) and eps > 0.0:
                per_round.append(eps)

        if not per_round:
            logger.warning(
                "[DP] No valid privacy_epsilon across %d round(s) of session %s — "
                "DP may not have been active for this job",
                len(rounds), training_session.session_id,
            )
            return

        actual_epsilon = float(sum(per_round))

        policy.record_spent(actual_epsilon)
        logger.info(
            "[DP] Recorded ε=%.6f for dataset %s (session %s, composed over %d round(s): %s)",
            actual_epsilon, dataset_id, training_session.session_id,
            len(per_round), [round(e, 4) for e in per_round],
        )

        # Update per-researcher epsilon budget. Delegate to the model's
        # record_spent() so the overrun-protected conditional update lives in a
        # single place (parity with DatasetPrivacyPolicy above).
        try:
            researcher_id = getattr(training_session, 'user_id', None)
            if researcher_id is not None:
                try:
                    researcher_budget = ResearcherEpsilonBudget.objects.get(
                        dataset_id=dataset_id,
                        researcher_id=researcher_id,
                    )
                except ResearcherEpsilonBudget.DoesNotExist:
                    logger.info(
                        "No ResearcherEpsilonBudget for researcher=%s dataset=%s — "
                        "researcher spend not recorded",
                        researcher_id, dataset_id,
                    )
                else:
                    researcher_budget.record_spent(actual_epsilon)
                    logger.info(
                        "ResearcherEpsilonBudget updated: researcher=%s dataset=%s +epsilon=%.4f",
                        researcher_id, dataset_id, actual_epsilon,
                    )
        except Exception as exc:
            logger.error(
                "Error updating ResearcherEpsilonBudget (researcher=%s): %s",
                getattr(training_session, 'user_id', None), exc,
            )

    except Exception as exc:
        logger.error(
            "[DP] Unexpected error recording privacy spend for session %s: %s",
            getattr(training_session, 'session_id', '?'), exc,
        )


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

        # Record differential-privacy spend now that training has finished.
        # _record_privacy_spend never raises, so it cannot affect session status.
        _record_privacy_spend(training_session)

    except Exception as e:
        raise e

def fail_training_session(training_session, error_message, traceback=None):
    """Mark training session as failed with error details."""

    if not DJANGO_AVAILABLE or not training_session:
        return

    # Record any privacy budget consumed by completed rounds before marking failed.
    # A Hub that deliberately crashes training on the last round must still be debited
    # for the rounds that did run. _record_privacy_spend never raises.
    _record_privacy_spend(training_session)

    try:
        training_session.mark_failed(error_message, traceback)
        print(f"[ERROR] Training session failed: {training_session.session_id} - {error_message}")

    except Exception as e:
        print(f"[ERROR] Error marking training session as failed: {e}")

