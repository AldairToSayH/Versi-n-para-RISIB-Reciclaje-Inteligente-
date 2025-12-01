from fastapi import FastAPI, APIRouter, HTTPException, Depends
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import random

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Helper function to convert ObjectId to string
def str_object_id(obj):
    if isinstance(obj, dict):
        obj['_id'] = str(obj['_id'])
    return obj

# ========== MODELS ==========

class PointsHistoryItem(BaseModel):
    date: datetime = Field(default_factory=datetime.utcnow)
    points: int
    description: str
    type: str  # 'earned' or 'redeemed'

class User(BaseModel):
    studentId: str
    name: str
    email: str
    password: str = "password123"  # Mock password
    avatar: str = ""  # Avatar initials or URL
    points: int = 0
    category: str = "Clásico"
    pointsHistory: List[PointsHistoryItem] = []
    recycledKg: float = 0.0
    createdAt: datetime = Field(default_factory=datetime.utcnow)

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    studentId: str
    name: str
    email: str
    avatar: str
    points: int
    category: str
    pointsHistory: List[PointsHistoryItem]
    recycledKg: float
    nextCategory: Optional[dict] = None

class Container(BaseModel):
    name: str
    location: dict  # {x: float, y: float}
    status: str  # 'operational' or 'maintenance'
    address: str
    lastMaintenance: datetime = Field(default_factory=datetime.utcnow)
    type: str = "mixed"  # type of recycling

class ContainerResponse(BaseModel):
    id: str
    name: str
    location: dict
    status: str
    address: str
    lastMaintenance: datetime
    type: str

class Reward(BaseModel):
    title: str
    description: str
    pointsCost: int
    category: str  # Minimum category required
    available: bool = True
    image: str
    location: str  # e.g., "Cafetín", "Librería"

class RewardResponse(BaseModel):
    id: str
    title: str
    description: str
    pointsCost: int
    category: str
    available: bool
    image: str
    location: str

class QRScan(BaseModel):
    userId: str
    containerId: str
    pointsEarned: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ScanQRRequest(BaseModel):
    userId: str
    qrCode: str

class RedeemRequest(BaseModel):
    userId: str
    rewardId: str

# ========== CATEGORY LOGIC ==========

def get_category_from_points(points: int) -> str:
    if points >= 1000:
        return "Black"
    elif points >= 600:
        return "Diamante"
    elif points >= 300:
        return "Oro"
    elif points >= 100:
        return "Plata"
    else:
        return "Clásico"

def get_next_category_info(points: int) -> Optional[dict]:
    thresholds = [
        (100, "Plata"),
        (300, "Oro"),
        (600, "Diamante"),
        (1000, "Black")
    ]
    
    for threshold, category in thresholds:
        if points < threshold:
            return {
                "name": category,
                "pointsNeeded": threshold - points,
                "currentPoints": points,
                "totalPoints": threshold,
                "progress": (points / threshold) * 100
            }
    
    return None  # Max category reached

def get_category_requirements(category: str) -> int:
    categories = {
        "Clásico": 0,
        "Plata": 100,
        "Oro": 300,
        "Diamante": 600,
        "Black": 1000
    }
    return categories.get(category, 0)

# ========== AUTH ENDPOINTS ==========

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    # Mock authentication - accept any credentials
    user = await db.users.find_one({"email": credentials.email})
    
    if not user:
        # Create a new user
        new_user = User(
            studentId=f"ST{random.randint(10000, 99999)}",
            name=credentials.email.split("@")[0].title(),
            email=credentials.email,
            password=credentials.password,
            avatar=credentials.email[0].upper()
        )
        result = await db.users.insert_one(new_user.dict())
        user = await db.users.find_one({"_id": result.inserted_id})
    
    user = str_object_id(user)
    return {
        "success": True,
        "user": {
            "id": user["_id"],
            "studentId": user["studentId"],
            "name": user["name"],
            "email": user["email"],
            "avatar": user["avatar"],
            "points": user["points"],
            "category": user["category"]
        },
        "token": "mock_token_123"
    }

@api_router.post("/auth/register")
async def register(credentials: UserLogin):
    # Check if user exists
    existing = await db.users.find_one({"email": credentials.email})
    if existing:
        raise HTTPException(status_code=400, detail="Usuario ya existe")
    
    new_user = User(
        studentId=f"ST{random.randint(10000, 99999)}",
        name=credentials.email.split("@")[0].title(),
        email=credentials.email,
        password=credentials.password,
        avatar=credentials.email[0].upper()
    )
    result = await db.users.insert_one(new_user.dict())
    user = await db.users.find_one({"_id": result.inserted_id})
    user = str_object_id(user)
    
    return {
        "success": True,
        "user": {
            "id": user["_id"],
            "studentId": user["studentId"],
            "name": user["name"],
            "email": user["email"],
            "avatar": user["avatar"],
            "points": user["points"],
            "category": user["category"]
        },
        "token": "mock_token_123"
    }

# ========== USER ENDPOINTS ==========

@api_router.get("/users/{user_id}")
async def get_user(user_id: str):
    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        user = str_object_id(user)
        next_cat = get_next_category_info(user["points"])
        
        return {
            "id": user["_id"],
            "studentId": user["studentId"],
            "name": user["name"],
            "email": user["email"],
            "avatar": user["avatar"],
            "points": user["points"],
            "category": user["category"],
            "pointsHistory": user.get("pointsHistory", []),
            "recycledKg": user.get("recycledKg", 0.0),
            "nextCategory": next_cat
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========== CONTAINERS ENDPOINTS ==========

@api_router.get("/containers")
async def get_containers():
    containers = await db.containers.find().to_list(100)
    return [{**str_object_id(c), "id": str(c["_id"])} for c in containers]

@api_router.get("/containers/{container_id}")
async def get_container(container_id: str):
    try:
        container = await db.containers.find_one({"_id": ObjectId(container_id)})
        if not container:
            raise HTTPException(status_code=404, detail="Contenedor no encontrado")
        return {**str_object_id(container), "id": str(container["_id"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========== REWARDS ENDPOINTS ==========

@api_router.get("/rewards")
async def get_rewards(category: Optional[str] = None):
    query = {}
    if category:
        # Get rewards available for this category
        cat_order = ["Clásico", "Plata", "Oro", "Diamante", "Black"]
        if category in cat_order:
            idx = cat_order.index(category)
            available_cats = cat_order[:idx+1]
            query["category"] = {"$in": available_cats}
    
    rewards = await db.rewards.find(query).to_list(100)
    return [{**str_object_id(r), "id": str(r["_id"])} for r in rewards]

@api_router.get("/rewards/{reward_id}")
async def get_reward(reward_id: str):
    try:
        reward = await db.rewards.find_one({"_id": ObjectId(reward_id)})
        if not reward:
            raise HTTPException(status_code=404, detail="Recompensa no encontrada")
        return {**str_object_id(reward), "id": str(reward["_id"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/rewards/redeem")
async def redeem_reward(request: RedeemRequest):
    try:
        # Get user
        user = await db.users.find_one({"_id": ObjectId(request.userId)})
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Get reward
        reward = await db.rewards.find_one({"_id": ObjectId(request.rewardId)})
        if not reward:
            raise HTTPException(status_code=404, detail="Recompensa no encontrada")
        
        # Check if user has enough points
        if user["points"] < reward["pointsCost"]:
            raise HTTPException(status_code=400, detail="Puntos insuficientes")
        
        # Check category requirement
        cat_order = ["Clásico", "Plata", "Oro", "Diamante", "Black"]
        user_cat_idx = cat_order.index(user["category"])
        reward_cat_idx = cat_order.index(reward["category"])
        
        if user_cat_idx < reward_cat_idx:
            raise HTTPException(status_code=400, detail="Categoría insuficiente")
        
        # Deduct points
        new_points = user["points"] - reward["pointsCost"]
        new_category = get_category_from_points(new_points)
        
        # Add to history
        history_item = {
            "date": datetime.utcnow(),
            "points": -reward["pointsCost"],
            "description": f"Canjeado: {reward['title']}",
            "type": "redeemed"
        }
        
        await db.users.update_one(
            {"_id": ObjectId(request.userId)},
            {
                "$set": {
                    "points": new_points,
                    "category": new_category
                },
                "$push": {"pointsHistory": history_item}
            }
        )
        
        return {
            "success": True,
            "message": "Recompensa canjeada exitosamente",
            "newPoints": new_points,
            "newCategory": new_category
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========== QR SCAN ENDPOINTS ==========

@api_router.post("/scan")
async def scan_qr(request: ScanQRRequest):
    try:
        # Validate QR code format (should be container ID)
        qr_data = request.qrCode
        
        # Try to find container by ID or by special code
        try:
            container = await db.containers.find_one({"_id": ObjectId(qr_data)})
        except:
            # If not valid ObjectId, try to find by name or code
            container = await db.containers.find_one({"name": qr_data})
        
        if not container:
            raise HTTPException(status_code=404, detail="Contenedor no válido")
        
        if container["status"] != "operational":
            raise HTTPException(status_code=400, detail="Contenedor en mantenimiento")
        
        # Get user
        user = await db.users.find_one({"_id": ObjectId(request.userId)})
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Generate random points (10-50)
        points_earned = random.randint(10, 50)
        
        # Update user points
        new_points = user["points"] + points_earned
        new_category = get_category_from_points(new_points)
        new_kg = user.get("recycledKg", 0.0) + (points_earned * 0.1)  # 1 point = 0.1 kg
        
        # Add to history
        history_item = {
            "date": datetime.utcnow(),
            "points": points_earned,
            "description": f"Reciclaje en {container['name']}",
            "type": "earned"
        }
        
        await db.users.update_one(
            {"_id": ObjectId(request.userId)},
            {
                "$set": {
                    "points": new_points,
                    "category": new_category,
                    "recycledKg": new_kg
                },
                "$push": {"pointsHistory": history_item}
            }
        )
        
        # Save scan record
        scan_record = QRScan(
            userId=request.userId,
            containerId=str(container["_id"]),
            pointsEarned=points_earned
        )
        await db.scans.insert_one(scan_record.dict())
        
        return {
            "success": True,
            "pointsEarned": points_earned,
            "newPoints": new_points,
            "newCategory": new_category,
            "containerName": container["name"],
            "kgRecycled": points_earned * 0.1
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========== INIT DATA ENDPOINT ==========

@api_router.post("/init-data")
async def init_data():
    # Check if data already exists
    existing_containers = await db.containers.count_documents({})
    if existing_containers > 0:
        return {"message": "Datos ya inicializados"}
    
    # Create sample containers
    containers_data = [
        Container(
            name="Contenedor A - Biblioteca",
            location={"x": 30, "y": 40},
            status="operational",
            address="Edificio de Biblioteca, Piso 1",
            type="mixed"
        ),
        Container(
            name="Contenedor B - Cafetería",
            location={"x": 60, "y": 30},
            status="operational",
            address="Cafetería Central",
            type="plastic"
        ),
        Container(
            name="Contenedor C - Laboratorios",
            location={"x": 45, "y": 70},
            status="maintenance",
            address="Edificio de Laboratorios, Entrada Principal",
            type="paper"
        ),
        Container(
            name="Contenedor D - Gimnasio",
            location={"x": 75, "y": 60},
            status="operational",
            address="Centro Deportivo",
            type="mixed"
        ),
        Container(
            name="Contenedor E - Estacionamiento",
            location={"x": 20, "y": 20},
            status="operational",
            address="Estacionamiento Norte",
            type="mixed"
        )
    ]
    
    for container in containers_data:
        await db.containers.insert_one(container.dict())
    
    # Create sample rewards
    rewards_data = [
        Reward(
            title="Café Gratis",
            description="1 café americano o capuchino gratis en la cafetería",
            pointsCost=50,
            category="Clásico",
            image="☕",
            location="Cafetín"
        ),
        Reward(
            title="Descuento 10% Librería",
            description="10% de descuento en cualquier producto de la librería",
            pointsCost=80,
            category="Clásico",
            image="📚",
            location="Librería"
        ),
        Reward(
            title="Snack Gratis",
            description="Elige un snack gratis en la cafetería",
            pointsCost=30,
            category="Clásico",
            image="🍪",
            location="Cafetín"
        ),
        Reward(
            title="Cuaderno Universitario",
            description="Cuaderno de 100 hojas tamaño universitario",
            pointsCost=120,
            category="Plata",
            image="📓",
            location="Librería"
        ),
        Reward(
            title="Vale S/10 Cafetería",
            description="Vale por S/10 para usar en la cafetería",
            pointsCost=150,
            category="Plata",
            image="🎫",
            location="Cafetín"
        ),
        Reward(
            title="Set de Lapiceros",
            description="Set de 5 lapiceros de colores",
            pointsCost=100,
            category="Plata",
            image="🖊️",
            location="Librería"
        ),
        Reward(
            title="Vale S/20 Librería",
            description="Vale por S/20 para usar en la librería",
            pointsCost=300,
            category="Oro",
            image="🎁",
            location="Librería"
        ),
        Reward(
            title="Mochila Ecológica",
            description="Mochila de material reciclado con logo RISIB",
            pointsCost=400,
            category="Oro",
            image="🎒",
            location="Tienda RISIB"
        ),
        Reward(
            title="Almuerzo Gratis",
            description="Menú completo gratis en la cafetería",
            pointsCost=250,
            category="Oro",
            image="🍽️",
            location="Cafetín"
        ),
        Reward(
            title="Vale S/50 Multiuso",
            description="Vale por S/50 para usar en cafetería o librería",
            pointsCost=700,
            category="Diamante",
            image="💎",
            location="Multiuso"
        ),
        Reward(
            title="Laptop Cooling Pad",
            description="Base refrigerante para laptop",
            pointsCost=800,
            category="Diamante",
            image="💻",
            location="Tienda Tech"
        ),
        Reward(
            title="Vale S/100 Premium",
            description="Vale por S/100 para usar en cualquier establecimiento del campus",
            pointsCost=1200,
            category="Black",
            image="🏆",
            location="Premium"
        ),
        Reward(
            title="Tablet Ecológica",
            description="Tablet para tomar notas digitales",
            pointsCost=1500,
            category="Black",
            image="📱",
            location="Tienda Tech"
        )
    ]
    
    for reward in rewards_data:
        await db.rewards.insert_one(reward.dict())
    
    return {
        "message": "Datos inicializados correctamente",
        "containers": len(containers_data),
        "rewards": len(rewards_data)
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
