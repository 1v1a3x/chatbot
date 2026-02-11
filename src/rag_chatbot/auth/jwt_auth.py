"""Authentication and authorization."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps
import hashlib
import secrets
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger

logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthError(Exception):
    """Authentication error."""

    pass


class User:
    """User model."""

    def __init__(
        self,
        id: str,
        email: str,
        hashed_password: str,
        is_active: bool = True,
        is_admin: bool = False,
        api_key: Optional[str] = None,
    ):
        self.id = id
        self.email = email
        self.hashed_password = hashed_password
        self.is_active = is_active
        self.is_admin = is_admin
        self.api_key = api_key


class UserManager:
    """Manage users and authentication."""

    def __init__(self):
        """Initialize user manager."""
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id

    def create_user(
        self,
        email: str,
        password: str,
        is_admin: bool = False,
    ) -> User:
        """Create a new user.

        Args:
            email: User email
            password: Plain text password
            is_admin: Admin flag

        Returns:
            Created user
        """
        user_id = str(uuid.uuid4())
        hashed_password = pwd_context.hash(password)

        user = User(
            id=user_id,
            email=email,
            hashed_password=hashed_password,
            is_admin=is_admin,
        )

        self._users[user_id] = user
        logger.info(f"Created user: {email}")

        return user

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User if authenticated, None otherwise
        """
        user = self.get_user_by_email(email)
        if not user:
            return None

        if not pwd_context.verify(password, user.hashed_password):
            return None

        if not user.is_active:
            return None

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        user_id = self._api_keys.get(api_key)
        if user_id:
            return self._users.get(user_id)
        return None

    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user.

        Args:
            user_id: User ID

        Returns:
            Generated API key
        """
        api_key = f"rag_{secrets.token_urlsafe(32)}"
        self._api_keys[api_key] = user_id

        user = self._users.get(user_id)
        if user:
            user.api_key = api_key

        logger.info(f"Generated API key for user: {user_id}")
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if revoked, False otherwise
        """
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            logger.info("API key revoked")
            return True
        return False


class JWTManager:
    """Manage JWT tokens."""

    def __init__(self):
        """Initialize JWT manager."""
        settings = get_settings()
        self.secret_key = settings.openai_api_key[:32]  # Use part of API key as secret

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token.

        Args:
            data: Token data
            expires_delta: Expiration time

        Returns:
            JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})

        return jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})

        return jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token.

        Args:
            token: JWT token

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None


# Global instances
user_manager = UserManager()
jwt_manager = JWTManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get current user from JWT or API key.

    Args:
        credentials: Authorization credentials

        Returns:
            Current user

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not credentials:
        raise credentials_exception

    token = credentials.credentials

    # Try API key first
    user = user_manager.get_user_by_api_key(token)
    if user:
        return user

    # Try JWT
    payload = jwt_manager.decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = user_manager.get_user(user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current admin user."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


def require_auth(func):
    """Decorator to require authentication."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This is a simplified version
        # In real use, integrate with FastAPI dependency injection
        return await func(*args, **kwargs)

    return wrapper
