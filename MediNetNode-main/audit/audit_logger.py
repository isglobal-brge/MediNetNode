"""
Advanced audit logging system with automatic risk scoring and categorization.
"""
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from django.conf import settings
from django.utils import timezone
from django.contrib.auth import get_user_model
from .models import AuditEvent, DataAccessLog, SecurityIncident

User = get_user_model()


class AuditLogger:
    """Advanced audit logger with automatic risk scoring and incident detection."""
    
    # Base risk scores by category
    BASE_RISK_SCORES = {
        'AUTH': 15,
        'DATA_ACCESS': 25,
        'USER_MGMT': 20,
        'DATASET_MGMT': 30,
        'TRAINING': 10,
        'API': 15,
        'SYSTEM': 5,
    }
    
    # Action modifiers (added to base score)
    ACTION_MODIFIERS = {
        'DELETE': 30,
        'FAILED': 20,
        'UNAUTHORIZED': 40,
        'ADMIN': 25,
        'EXPORT': 35,
        'DOWNLOAD': 25,
        'BULK': 20,
        'CREATE': 10,
        'UPDATE': 15,
        'VIEW': 5,
        'LOGIN': 10,
        'LOGOUT': 0,
    }
    
    # High-risk patterns that trigger automatic incidents
    HIGH_RISK_PATTERNS = [
        r'(?i).*failed.*login.*',
        r'(?i).*unauthorized.*access.*',
        r'(?i).*data.*breach.*',
        r'(?i).*privilege.*escalation.*',
        r'(?i).*suspicious.*activity.*',
    ]
    
    # Review threshold
    REVIEW_THRESHOLD = 70
    INCIDENT_THRESHOLD = 80

    @classmethod
    def log_event(
        cls,
        action: str,
        user=None,
        resource: str = '',
        ip_address: str = '',
        success: bool = True,
        details: Dict[str, Any] = None,
        session_id: str = '',
        user_agent: str = '',
        request_size: Optional[int] = None,
        request_duration_ms: Optional[int] = None,
        medical_domain: str = '',
        patient_count: int = 0,
        records_accessed: int = 0,
        columns_accessed: List[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event with automatic categorization and risk scoring.
        
        Args:
            action: The action being performed
            user: User performing the action
            resource: Resource being accessed
            ip_address: IP address of the request
            success: Whether the action was successful
            details: Additional details as JSON
            session_id: Session ID
            user_agent: User agent string
            request_size: Size of the request in bytes
            request_duration_ms: Request duration in milliseconds
            medical_domain: Medical domain for data access events
            patient_count: Number of patients accessed
            records_accessed: Number of records accessed
            columns_accessed: List of columns accessed
            
        Returns:
            Created AuditEvent instance
        """
        if details is None:
            details = {}
        if columns_accessed is None:
            columns_accessed = []
            
        # Determine category and severity
        category = cls._determine_category(action, resource)
        severity = cls._determine_severity(action, success, details)
        
        # Calculate risk score
        risk_score = cls._calculate_risk_score(
            category, action, success, details, user, ip_address
        )
        
        # Set requires_review based on threshold
        requires_review = risk_score >= cls.REVIEW_THRESHOLD
        
        # Create audit event
        audit_event = AuditEvent.objects.create(
            user=user,
            action=action,
            resource=resource,
            ip_address=ip_address,
            success=success,
            details=details,
            category=category,
            severity=severity,
            risk_score=risk_score,
            session_id=session_id,
            user_agent=user_agent,
            request_size=request_size,
            request_duration_ms=request_duration_ms,
            requires_review=requires_review,
        )
        
        # Create DataAccessLog for data access events
        if category == 'DATA_ACCESS' and (medical_domain or patient_count or records_accessed):
            data_sensitivity_level = cls._calculate_data_sensitivity(
                medical_domain, patient_count, columns_accessed
            )
            
            DataAccessLog.objects.create(
                audit_event=audit_event,
                medical_domain=medical_domain,
                patient_count_accessed=patient_count,
                data_sensitivity_level=data_sensitivity_level,
                columns_accessed=columns_accessed,
                records_accessed=records_accessed,
                query_hash=cls._generate_query_hash(resource, columns_accessed),
            )
        
        # Check for security incidents
        if risk_score >= cls.INCIDENT_THRESHOLD:
            cls._create_security_incident(audit_event)
            
        return audit_event

    @classmethod
    def _determine_category(cls, action: str, resource: str) -> str:
        """Determine event category based on action and resource."""
        action_lower = action.lower()
        resource_lower = resource.lower()
        
        # Authentication events
        if any(keyword in action_lower for keyword in ['login', 'logout', 'auth', 'password']):
            return 'AUTH'
        
        # API events (check before data access to avoid conflicts)
        if '/api/' in resource_lower or 'api' in action_lower:
            return 'API'
        
        # Data access events
        if any(keyword in resource_lower for keyword in ['dataset', 'data', 'query', 'export', 'download']):
            return 'DATA_ACCESS'
        
        # User management events
        if any(keyword in resource_lower for keyword in ['user', 'profile', 'account']):
            return 'USER_MGMT'
        
        # Dataset management events
        if any(keyword in action_lower for keyword in ['create_dataset', 'delete_dataset', 'update_dataset']):
            return 'DATASET_MGMT'
        
        # Federated training events
        if any(keyword in action_lower for keyword in ['train', 'federated', 'model', 'epoch']):
            return 'TRAINING'
        
        # API access events
        if any(keyword in resource_lower for keyword in ['api', '/api/']):
            return 'API'
        
        # Default to SYSTEM
        return 'SYSTEM'

    @classmethod
    def _determine_severity(cls, action: str, success: bool, details: Dict[str, Any]) -> str:
        """Determine event severity based on action and outcome."""
        action_lower = action.lower()
        
        # Security events
        if any(keyword in action_lower for keyword in ['unauthorized', 'breach', 'attack']):
            return 'SECURITY'
        
        # Critical events
        if any(keyword in action_lower for keyword in ['delete', 'drop', 'truncate', 'admin']):
            return 'CRITICAL'
        
        # Failed actions are at least warnings
        if not success:
            if any(keyword in action_lower for keyword in ['login', 'auth', 'access']):
                return 'WARNING'
            return 'ERROR'
        
        # Bulk operations are warnings
        if any(keyword in action_lower for keyword in ['bulk', 'batch', 'mass']):
            return 'WARNING'
        
        # Default to INFO for successful operations
        return 'INFO'

    @classmethod
    def _calculate_risk_score(
        cls,
        category: str,
        action: str,
        success: bool,
        details: Dict[str, Any],
        user=None,
        ip_address: str = '',
    ) -> int:
        """
        Calculate risk score (0-100) based on multiple factors.
        
        Factors considered:
        - Base score by category
        - Action type modifiers
        - Success/failure status
        - Time of day (night access is riskier)
        - User context
        - Historical patterns
        """
        # Start with base score for category
        base_score = cls.BASE_RISK_SCORES.get(category, 5)
        
        # Apply action modifiers
        action_upper = action.upper()
        action_modifier = 0
        for keyword, modifier in cls.ACTION_MODIFIERS.items():
            if keyword in action_upper:
                action_modifier = max(action_modifier, modifier)
        
        # Calculate total score
        risk_score = base_score + action_modifier
        
        # Failure increases risk
        if not success:
            risk_score += 15
        
        # Night access (22:00-06:00) increases risk
        current_hour = timezone.now().hour
        if current_hour >= 22 or current_hour <= 6:
            risk_score += 10
        
        # Weekend access increases risk
        if timezone.now().weekday() >= 5:  # Saturday=5, Sunday=6
            risk_score += 5
        
        # Check for suspicious patterns in recent history
        if user and ip_address:
            suspicious_modifier = cls._check_suspicious_patterns(user, ip_address)
            risk_score += suspicious_modifier
        
        # High-risk pattern detection
        action_details = f"{action} {details}"
        for pattern in cls.HIGH_RISK_PATTERNS:
            if re.match(pattern, action_details):
                risk_score += 25
                break
        
        # Cap at 100
        return min(risk_score, 100)

    @classmethod
    def _check_suspicious_patterns(cls, user, ip_address: str) -> int:
        """Check for suspicious patterns in recent user activity."""
        modifier = 0
        now = timezone.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Check for multiple failed attempts in last hour
        recent_failures = AuditEvent.objects.filter(
            user=user,
            timestamp__gte=last_hour,
            success=False,
        ).count()
        
        if recent_failures >= 5:
            modifier += 20
        elif recent_failures >= 3:
            modifier += 10
        
        # Check for access from multiple IPs in short time
        if user:
            recent_ips = AuditEvent.objects.filter(
                user=user,
                timestamp__gte=last_hour,
            ).values_list('ip_address', flat=True).distinct()
            
            unique_ips = len([ip for ip in recent_ips if ip])
            if unique_ips >= 3:
                modifier += 15
        
        # Check for unusual volume of activity
        recent_events = AuditEvent.objects.filter(
            user=user,
            timestamp__gte=last_hour,
        ).count()
        
        if recent_events >= 50:
            modifier += 10
        elif recent_events >= 25:
            modifier += 5
        
        return modifier

    @classmethod
    def _calculate_data_sensitivity(
        cls,
        medical_domain: str,
        patient_count: int,
        columns_accessed: List[str]
    ) -> int:
        """Calculate data sensitivity level (1-5) based on accessed data."""
        sensitivity_level = 1
        
        # High-sensitivity medical domains
        high_sensitivity_domains = [
            'psychiatry', 'oncology', 'genetics', 'cardiology',
            'reproductive', 'infectious_diseases'
        ]
        
        if medical_domain.lower() in high_sensitivity_domains:
            sensitivity_level = 4
        elif medical_domain:
            sensitivity_level = 3
        
        # Large patient counts increase sensitivity
        if patient_count >= 1000:
            sensitivity_level = min(sensitivity_level + 1, 5)
        
        # Sensitive columns increase sensitivity
        sensitive_columns = ['ssn', 'dob', 'address', 'phone', 'email', 'name']
        if any(col.lower() in sensitive_columns for col in columns_accessed):
            sensitivity_level = min(sensitivity_level + 1, 5)
        
        return sensitivity_level

    @classmethod
    def _generate_query_hash(cls, resource: str, columns_accessed: List[str]) -> str:
        """Generate a hash for the query/resource access pattern."""
        query_pattern = f"{resource}:{','.join(sorted(columns_accessed))}"
        return hashlib.sha256(query_pattern.encode()).hexdigest()

    @classmethod
    def _create_security_incident(cls, audit_event: AuditEvent) -> Optional[SecurityIncident]:
        """Create security incident for high-risk events."""
        # Determine incident type based on event characteristics
        incident_type = cls._determine_incident_type(audit_event)
        
        # Determine severity (1-4) based on risk score
        if audit_event.risk_score >= 95:
            severity = 4  # Critical
        elif audit_event.risk_score >= 90:
            severity = 3  # High
        elif audit_event.risk_score >= 80:
            severity = 2  # Medium
        else:
            severity = 1  # Low
        
        # Generate description
        description = cls._generate_incident_description(audit_event)
        
        # Create incident
        incident = SecurityIncident.objects.create(
            incident_type=incident_type,
            severity=severity,
            description=description,
            risk_score=audit_event.risk_score,
        )
        
        # Link the triggering event
        incident.related_events.add(audit_event)
        
        return incident

    @classmethod
    def _determine_incident_type(cls, audit_event: AuditEvent) -> str:
        """Determine incident type based on audit event characteristics."""
        action_lower = audit_event.action.lower()
        
        if not audit_event.success and 'auth' in action_lower:
            return 'MULTIPLE_FAILED_AUTH'
        elif 'unauthorized' in action_lower:
            return 'UNAUTHORIZED_ACCESS'
        elif 'breach' in action_lower or 'leak' in action_lower:
            return 'DATA_BREACH_ATTEMPT'
        elif audit_event.category == 'DATA_ACCESS' and audit_event.risk_score >= 90:
            return 'DATA_BREACH_ATTEMPT'
        elif 'admin' in action_lower and audit_event.user and not audit_event.user.is_staff:
            return 'PRIVILEGE_ESCALATION'
        else:
            return 'SUSPICIOUS_ACTIVITY'

    @classmethod
    def _generate_incident_description(cls, audit_event: AuditEvent) -> str:
        """Generate human-readable incident description."""
        user_info = f"User: {audit_event.user.username}" if audit_event.user else "Unknown user"
        action_info = f"Action: {audit_event.action}"
        resource_info = f"Resource: {audit_event.resource}" if audit_event.resource else "No resource"
        ip_info = f"IP: {audit_event.ip_address}" if audit_event.ip_address else "Unknown IP"
        risk_info = f"Risk Score: {audit_event.risk_score}"
        
        return f"""
High-risk security event detected:
{user_info}
{action_info}
{resource_info}
{ip_info}
{risk_info}
Category: {audit_event.category}
Severity: {audit_event.severity}
Success: {'Yes' if audit_event.success else 'No'}
Timestamp: {audit_event.timestamp}
        """.strip()

    @classmethod
    def log_data_access(
        cls,
        dataset_name: str,
        user,
        medical_domain: str,
        records_accessed: int,
        columns_accessed: List[str],
        patient_count: int = 0,
        ip_address: str = '',
        **kwargs
    ) -> AuditEvent:
        """Convenience method for logging data access events."""
        return cls.log_event(
            action='DATA_ACCESS',
            user=user,
            resource=f"dataset:{dataset_name}",
            ip_address=ip_address,
            success=True,
            medical_domain=medical_domain,
            patient_count=patient_count,
            records_accessed=records_accessed,
            columns_accessed=columns_accessed,
            **kwargs
        )

    @classmethod
    def log_authentication(
        cls,
        action: str,
        user,
        success: bool,
        ip_address: str = '',
        details: Dict[str, Any] = None,
        **kwargs
    ) -> AuditEvent:
        """Convenience method for logging authentication events."""
        return cls.log_event(
            action=action,
            user=user,
            resource='authentication',
            ip_address=ip_address,
            success=success,
            details=details or {},
            **kwargs
        )

    @classmethod
    def log_api_access(
        cls,
        endpoint: str,
        user,
        method: str = 'GET',
        success: bool = True,
        ip_address: str = '',
        request_size: Optional[int] = None,
        request_duration_ms: Optional[int] = None,
        **kwargs
    ) -> AuditEvent:
        """Convenience method for logging API access events."""
        return cls.log_event(
            action=f'API_{method}',
            user=user,
            resource=f"api:{endpoint}",
            ip_address=ip_address,
            success=success,
            request_size=request_size,
            request_duration_ms=request_duration_ms,
            **kwargs
        )