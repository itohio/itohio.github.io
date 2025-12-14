---
title: "FastAPI with OpenAPI"
date: 2024-12-12
draft: false
category: "python"
tags: ["python-knowhow", "fastapi", "openapi", "api", "rest"]
---


FastAPI with automatic OpenAPI documentation using Pydantic models and docstrings.

---

## Installation

```bash
pip install fastapi uvicorn[standard]
pip install python-multipart  # For form data
```

---

## Basic FastAPI App

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="API documentation",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json"
)

@app.get("/")
def read_root():
    """
    Root endpoint
    
    Returns a welcome message.
    """
    return {"message": "Hello World"}

# Run with: uvicorn main:app --reload
```

---

## Using Pydantic Models

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="User API", version="1.0.0")

# Request models
class UserCreate(BaseModel):
    """User creation request"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, description="Full name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "secretpassword",
                "full_name": "John Doe"
            }
        }

class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[str] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")

# Response models
class User(BaseModel):
    """User response"""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_active: bool = Field(True, description="Is user active")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "johndoe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "created_at": "2024-12-12T10:00:00",
                "is_active": True
            }
        }

# Fake database
users_db: List[User] = []
user_id_counter = 1

@app.post(
    "/users/",
    response_model=User,
    status_code=201,
    tags=["users"],
    summary="Create a new user",
    description="Create a new user with the provided information"
)
def create_user(user: UserCreate):
    """
    Create a new user.
    
    Args:
        user: User creation data
    
    Returns:
        Created user
    
    Raises:
        HTTPException: If username already exists
    """
    global user_id_counter
    
    # Check if username exists
    if any(u.username == user.username for u in users_db):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    new_user = User(
        id=user_id_counter,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        created_at=datetime.now(),
        is_active=True
    )
    users_db.append(new_user)
    user_id_counter += 1
    
    return new_user

@app.get(
    "/users/",
    response_model=List[User],
    tags=["users"],
    summary="List all users",
    description="Get a list of all users"
)
def list_users(
    skip: int = Field(0, ge=0, description="Number of records to skip"),
    limit: int = Field(10, ge=1, le=100, description="Maximum number of records to return")
):
    """
    List all users with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of users
    """
    return users_db[skip:skip + limit]

@app.get(
    "/users/{user_id}",
    response_model=User,
    tags=["users"],
    summary="Get user by ID",
    description="Get a specific user by their ID"
)
def get_user(user_id: int = Field(..., description="User ID")):
    """
    Get a user by ID.
    
    Args:
        user_id: User ID
    
    Returns:
        User object
    
    Raises:
        HTTPException: If user not found
    """
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

@app.put(
    "/users/{user_id}",
    response_model=User,
    tags=["users"],
    summary="Update user",
    description="Update user information"
)
def update_user(
    user_id: int = Field(..., description="User ID"),
    user_update: UserUpdate = None
):
    """
    Update a user.
    
    Args:
        user_id: User ID
        user_update: User update data
    
    Returns:
        Updated user
    
    Raises:
        HTTPException: If user not found
    """
    for user in users_db:
        if user.id == user_id:
            if user_update.email is not None:
                user.email = user_update.email
            if user_update.full_name is not None:
                user.full_name = user_update.full_name
            return user
    raise HTTPException(status_code=404, detail="User not found")

@app.delete(
    "/users/{user_id}",
    status_code=204,
    tags=["users"],
    summary="Delete user",
    description="Delete a user by ID"
)
def delete_user(user_id: int = Field(..., description="User ID")):
    """
    Delete a user.
    
    Args:
        user_id: User ID
    
    Raises:
        HTTPException: If user not found
    """
    for i, user in enumerate(users_db):
        if user.id == user_id:
            users_db.pop(i)
            return
    raise HTTPException(status_code=404, detail="User not found")
```

---

## Using Dataclasses

```python
from fastapi import FastAPI, HTTPException
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="Product API", version="1.0.0")

@dataclass
class Product:
    """Product model"""
    id: int
    name: str
    description: str
    price: float
    quantity: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")

@dataclass
class ProductCreate:
    """Product creation request"""
    name: str
    description: str
    price: float
    quantity: int

# Fake database
products_db: List[Product] = []
product_id_counter = 1

@app.post("/products/", response_model=Product, status_code=201, tags=["products"])
def create_product(product: ProductCreate):
    """
    Create a new product.
    
    Args:
        product: Product creation data
            - name: Product name (required)
            - description: Product description (required)
            - price: Product price (required, must be >= 0)
            - quantity: Product quantity (required, must be >= 0)
    
    Returns:
        Created product with ID and timestamp
    
    Raises:
        HTTPException: If validation fails
    """
    global product_id_counter
    
    try:
        new_product = Product(
            id=product_id_counter,
            name=product.name,
            description=product.description,
            price=product.price,
            quantity=product.quantity
        )
        products_db.append(new_product)
        product_id_counter += 1
        return new_product
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/products/", response_model=List[Product], tags=["products"])
def list_products():
    """
    List all products.
    
    Returns:
        List of all products in the database
    """
    return products_db

@app.get("/products/{product_id}", response_model=Product, tags=["products"])
def get_product(product_id: int):
    """
    Get a product by ID.
    
    Args:
        product_id: Product ID
    
    Returns:
        Product object
    
    Raises:
        HTTPException: If product not found (404)
    """
    for product in products_db:
        if product.id == product_id:
            return product
    raise HTTPException(status_code=404, detail="Product not found")
```

---

## Response Models and Status Codes

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: Optional[str] = None

class SuccessResponse(BaseModel):
    """Success response"""
    message: str
    data: Optional[dict] = None

@app.get(
    "/items/{item_id}",
    response_model=SuccessResponse,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"message": "Item found", "data": {"id": 1, "name": "Item"}}
                }
            }
        },
        404: {
            "description": "Item not found",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Item not found", "error_code": "ITEM_NOT_FOUND"}
                }
            }
        }
    },
    tags=["items"]
)
def get_item(item_id: int):
    """
    Get an item by ID.
    
    This endpoint retrieves an item from the database.
    
    Args:
        item_id: The ID of the item to retrieve
    
    Returns:
        SuccessResponse with item data
    
    Raises:
        HTTPException: 404 if item not found
    """
    if item_id == 1:
        return SuccessResponse(
            message="Item found",
            data={"id": 1, "name": "Sample Item"}
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Item not found"
    )
```

---

## Path Operations with Tags and Metadata

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="My API",
    description="Comprehensive API documentation",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "users",
            "description": "Operations with users. The **login** logic is also here.",
        },
        {
            "name": "items",
            "description": "Manage items. So _fancy_ they have their own docs.",
            "externalDocs": {
                "description": "Items external docs",
                "url": "https://example.com/items",
            },
        },
    ]
)

class Item(BaseModel):
    name: str
    description: str

@app.post(
    "/items/",
    tags=["items"],
    summary="Create an item",
    description="Create an item with all the information",
    response_description="The created item"
)
def create_item(item: Item):
    """
    Create an item with all the information:
    
    - **name**: each item must have a name
    - **description**: a long description
    """
    return item

@app.get(
    "/items/",
    tags=["items"],
    summary="List items",
    deprecated=False
)
def list_items():
    """List all items in the database."""
    return [{"name": "Item 1"}, {"name": "Item 2"}]
```

---

## Query Parameters and Validation

```python
from fastapi import FastAPI, Query
from typing import Optional, List
from pydantic import BaseModel

app = FastAPI()

class SearchResults(BaseModel):
    """Search results"""
    query: str
    results: List[dict]
    total: int

@app.get("/search/", response_model=SearchResults, tags=["search"])
def search(
    q: str = Query(
        ...,
        min_length=3,
        max_length=50,
        description="Search query",
        example="python"
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number",
        example=1
    ),
    size: int = Query(
        10,
        ge=1,
        le=100,
        description="Page size",
        example=10
    ),
    sort: Optional[str] = Query(
        None,
        regex="^(asc|desc)$",
        description="Sort order",
        example="asc"
    ),
    tags: Optional[List[str]] = Query(
        None,
        description="Filter by tags",
        example=["python", "fastapi"]
    )
):
    """
    Search for items.
    
    Args:
        q: Search query (3-50 characters)
        page: Page number (>= 1)
        size: Page size (1-100)
        sort: Sort order (asc or desc)
        tags: Filter by tags
    
    Returns:
        Search results with pagination
    """
    return SearchResults(
        query=q,
        results=[{"id": 1, "name": "Result 1"}],
        total=1
    )
```

---

## Request Body Examples

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str = Field(..., example="Widget")
    description: Optional[str] = Field(None, example="A useful widget")
    price: float = Field(..., gt=0, example=9.99)
    tax: Optional[float] = Field(None, ge=0, example=0.99)

@app.post("/items/", tags=["items"])
def create_item(
    item: Item = Body(
        ...,
        examples={
            "normal": {
                "summary": "A normal example",
                "description": "A **normal** item works correctly.",
                "value": {
                    "name": "Widget",
                    "description": "A very nice widget",
                    "price": 35.4,
                    "tax": 3.2,
                },
            },
            "minimal": {
                "summary": "Minimal example",
                "value": {
                    "name": "Widget",
                    "price": 35.4,
                },
            },
            "invalid": {
                "summary": "Invalid data is rejected",
                "value": {
                    "name": "Widget",
                    "price": -10,  # Invalid: negative price
                },
            },
        },
    )
):
    """
    Create an item.
    
    The request body should include:
    - **name**: Item name (required)
    - **description**: Item description (optional)
    - **price**: Item price (required, must be > 0)
    - **tax**: Tax amount (optional, must be >= 0)
    """
    return item
```

---

## Dependencies and Security

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

app = FastAPI()
security = HTTPBearer()

class User(BaseModel):
    username: str
    email: str

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get current authenticated user.
    
    Args:
        credentials: Bearer token
    
    Returns:
        Current user
    
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    # Validate token (simplified)
    if token != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(username="johndoe", email="john@example.com")

@app.get(
    "/users/me",
    response_model=User,
    tags=["users"],
    summary="Get current user",
    description="Get the current authenticated user's information"
)
def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Requires authentication via Bearer token.
    
    Returns:
        Current user object
    """
    return current_user
```

---

## File Upload

```python
from fastapi import FastAPI, File, UploadFile
from typing import List

app = FastAPI()

@app.post("/upload/", tags=["files"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single file.
    
    Args:
        file: File to upload
    
    Returns:
        File information
    """
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(await file.read())
    }

@app.post("/uploadfiles/", tags=["files"])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files.
    
    Args:
        files: List of files to upload
    
    Returns:
        List of file information
    """
    return [
        {
            "filename": file.filename,
            "content_type": file.content_type,
        }
        for file in files
    ]
```

---

## Custom OpenAPI Schema

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Custom API",
        version="2.5.0",
        description="This is a custom OpenAPI schema",
        routes=app.routes,
    )
    
    # Add custom fields
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Hello World"}
```

---

## Run Application

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## Access Documentation

```
# Swagger UI (interactive)
http://localhost:8000/docs

# ReDoc (alternative)
http://localhost:8000/redoc

# OpenAPI JSON schema
http://localhost:8000/openapi.json
```

---