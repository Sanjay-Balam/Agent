"""
System Design Knowledge Base - Contains templates, patterns, and examples for Low Level Design (LLD) and High Level Design (HLD).
This extends the system to support System Design topics while keeping existing functionality intact.
"""

# Low Level Design Patterns
LLD_PATTERNS = {
    "singleton": {
        "definition": "Ensure a class has only one instance and provide global access to it",
        "code_template": """
class Singleton:
    \"\"\"
    Singleton Design Pattern
    Ensures only one instance of the class exists
    \"\"\"
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # Initialize your singleton here
            self.data = "Singleton Instance"
    
    def get_data(self):
        return self.data

# Thread-safe Singleton
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ThreadSafeSingleton, cls).__new__(cls)
        return cls._instance

# Example usage
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - same instance
""",
        "pros": ["Controlled access to sole instance", "Reduced memory usage", "Global access point"],
        "cons": ["Difficult to unit test", "Violates single responsibility", "Can create tight coupling"],
        "use_cases": ["Database connections", "Logging", "Configuration settings", "Cache"]
    },
    
    "factory": {
        "definition": "Create objects without specifying their exact classes",
        "code_template": """
from abc import ABC, abstractmethod

# Product interface
class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

# Concrete products
class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class Bird(Animal):
    def make_sound(self):
        return "Tweet!"

# Factory
class AnimalFactory:
    \"\"\"
    Factory Pattern
    Creates objects without specifying their exact classes
    \"\"\"
    
    @staticmethod
    def create_animal(animal_type):
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        elif animal_type.lower() == "bird":
            return Bird()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Example usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")

print(dog.make_sound())  # Woof!
print(cat.make_sound())  # Meow!
""",
        "pros": ["Loose coupling", "Easy to extend", "Centralized object creation"],
        "cons": ["Can become complex", "Additional abstraction layer"],
        "use_cases": ["Object creation based on configuration", "Plugin systems", "Different implementations"]
    },
    
    "observer": {
        "definition": "Define a one-to-many dependency between objects",
        "code_template": """
from abc import ABC, abstractmethod
from typing import List

# Observer interface
class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

# Subject interface
class Subject(ABC):
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

# Concrete Subject
class WeatherStation(Subject):
    \"\"\"
    Observer Pattern Example - Weather Station
    \"\"\"
    def __init__(self):
        super().__init__()
        self._temperature = 0
        self._humidity = 0
        self._pressure = 0
    
    def set_measurements(self, temperature, humidity, pressure):
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.notify()
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def humidity(self):
        return self._humidity
    
    @property
    def pressure(self):
        return self._pressure

# Concrete Observers
class CurrentConditionsDisplay(Observer):
    def update(self, weather_station):
        print(f"Current: {weather_station.temperature}°F, "
              f"{weather_station.humidity}% humidity, "
              f"{weather_station.pressure} inHg")

class StatisticsDisplay(Observer):
    def __init__(self):
        self.temperatures = []
    
    def update(self, weather_station):
        self.temperatures.append(weather_station.temperature)
        avg_temp = sum(self.temperatures) / len(self.temperatures)
        print(f"Avg temperature: {avg_temp:.1f}°F")

# Example usage
weather_station = WeatherStation()
current_display = CurrentConditionsDisplay()
stats_display = StatisticsDisplay()

weather_station.attach(current_display)
weather_station.attach(stats_display)

weather_station.set_measurements(80, 65, 30.4)
weather_station.set_measurements(82, 70, 29.2)
""",
        "pros": ["Loose coupling", "Dynamic relationships", "Broadcast communication"],
        "cons": ["Memory leaks if not detached", "Unexpected updates", "Complex debugging"],
        "use_cases": ["Event handling", "Model-View architectures", "Publish-subscribe systems"]
    },
    
    "strategy": {
        "definition": "Define a family of algorithms and make them interchangeable",
        "code_template": """
from abc import ABC, abstractmethod

# Strategy interface
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

# Concrete strategies
class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number, cvv):
        self.card_number = card_number
        self.cvv = cvv
    
    def pay(self, amount):
        return f"Paid ${amount} using Credit Card ending in {self.card_number[-4:]}"

class PayPalPayment(PaymentStrategy):
    def __init__(self, email):
        self.email = email
    
    def pay(self, amount):
        return f"Paid ${amount} using PayPal account {self.email}"

class BitcoinPayment(PaymentStrategy):
    def __init__(self, wallet_address):
        self.wallet_address = wallet_address
    
    def pay(self, amount):
        return f"Paid ${amount} using Bitcoin wallet {self.wallet_address[:10]}..."

# Context
class ShoppingCart:
    \"\"\"
    Strategy Pattern Example - Payment Processing
    \"\"\"
    def __init__(self):
        self.items = []
        self.payment_strategy = None
    
    def add_item(self, item, price):
        self.items.append((item, price))
    
    def set_payment_strategy(self, strategy: PaymentStrategy):
        self.payment_strategy = strategy
    
    def checkout(self):
        total = sum(price for item, price in self.items)
        if self.payment_strategy:
            return self.payment_strategy.pay(total)
        else:
            return "No payment method selected"

# Example usage
cart = ShoppingCart()
cart.add_item("Laptop", 999.99)
cart.add_item("Mouse", 29.99)

# Pay with credit card
credit_card = CreditCardPayment("1234-5678-9012-3456", "123")
cart.set_payment_strategy(credit_card)
print(cart.checkout())

# Pay with PayPal
paypal = PayPalPayment("user@example.com")
cart.set_payment_strategy(paypal)
print(cart.checkout())
""",
        "pros": ["Algorithm flexibility", "Easy to extend", "Eliminates conditionals"],
        "cons": ["Increased number of classes", "Client must be aware of strategies"],
        "use_cases": ["Payment processing", "Sorting algorithms", "Compression algorithms"]
    }
}

# High Level Design Components
HLD_COMPONENTS = {
    "load_balancer": {
        "definition": "Distributes incoming requests across multiple servers",
        "types": {
            "layer_4": "Transport layer load balancing (TCP/UDP)",
            "layer_7": "Application layer load balancing (HTTP/HTTPS)"
        },
        "algorithms": {
            "round_robin": "Requests distributed equally in rotation",
            "least_connections": "Route to server with fewest active connections",
            "weighted_round_robin": "Assign weights to servers based on capacity",
            "ip_hash": "Route based on client IP hash"
        },
        "code_example": """
import random
from typing import List

class Server:
    def __init__(self, name, capacity=100):
        self.name = name
        self.capacity = capacity
        self.current_load = 0
    
    def handle_request(self):
        if self.current_load < self.capacity:
            self.current_load += 1
            return f"Request handled by {self.name}"
        return f"{self.name} is overloaded"
    
    def finish_request(self):
        if self.current_load > 0:
            self.current_load -= 1

class LoadBalancer:
    def __init__(self):
        self.servers: List[Server] = []
        self.current_server = 0
    
    def add_server(self, server: Server):
        self.servers.append(server)
    
    def round_robin(self):
        \"\"\"Round Robin Load Balancing\"\"\"
        if not self.servers:
            return "No servers available"
        
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)
        return server.handle_request()
    
    def least_connections(self):
        \"\"\"Least Connections Load Balancing\"\"\"
        if not self.servers:
            return "No servers available"
        
        server = min(self.servers, key=lambda s: s.current_load)
        return server.handle_request()
""",
        "benefits": ["High availability", "Scalability", "Performance optimization"],
        "use_cases": ["Web applications", "API gateways", "Database clusters"]
    },
    
    "database": {
        "definition": "Data storage and retrieval system",
        "types": {
            "relational": {
                "description": "ACID compliance, structured data, SQL",
                "examples": ["PostgreSQL", "MySQL", "Oracle"],
                "use_cases": ["Financial systems", "E-commerce", "CRM"]
            },
            "nosql": {
                "document": {
                    "description": "Document-oriented, flexible schema",
                    "examples": ["MongoDB", "CouchDB"],
                    "use_cases": ["Content management", "Catalogs"]
                },
                "key_value": {
                    "description": "Simple key-value pairs, high performance",
                    "examples": ["Redis", "DynamoDB"],
                    "use_cases": ["Caching", "Session storage"]
                },
                "column_family": {
                    "description": "Column-oriented, big data",
                    "examples": ["Cassandra", "HBase"],
                    "use_cases": ["Time-series data", "Analytics"]
                },
                "graph": {
                    "description": "Nodes and relationships",
                    "examples": ["Neo4j", "Amazon Neptune"],
                    "use_cases": ["Social networks", "Recommendations"]
                }
            }
        },
        "scaling_strategies": {
            "vertical": "Scale up - more powerful hardware",
            "horizontal": "Scale out - more servers",
            "replication": "Master-slave, master-master configurations",
            "sharding": "Partition data across multiple databases"
        }
    },
    
    "caching": {
        "definition": "Store frequently accessed data for faster retrieval",
        "levels": {
            "browser_cache": "Client-side caching in web browsers",
            "cdn": "Content Delivery Network for static assets",
            "reverse_proxy": "Nginx, Apache for caching responses",
            "application_cache": "In-memory caching within application",
            "database_cache": "Query result caching"
        },
        "strategies": {
            "cache_aside": "Application manages cache explicitly",
            "write_through": "Write to cache and database simultaneously",
            "write_behind": "Write to cache first, database later",
            "refresh_ahead": "Proactively refresh cache before expiration"
        },
        "code_example": """
import time
from typing import Any, Optional

class Cache:
    \"\"\"Simple in-memory cache with TTL\"\"\"
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        \"\"\"Get value from cache with TTL check\"\"\"
        if key in self._cache:
            if time.time() - self._timestamps[key] < ttl:
                return self._cache[key]
            else:
                # Expired, remove from cache
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        \"\"\"Set value in cache\"\"\"
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def delete(self, key: str):
        \"\"\"Delete key from cache\"\"\"
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self):
        \"\"\"Clear all cache\"\"\"
        self._cache.clear()
        self._timestamps.clear()

# Cache-aside pattern example
class UserService:
    def __init__(self):
        self.cache = Cache()
        self.database = {}  # Simulated database
    
    def get_user(self, user_id: str):
        # Try cache first
        cached_user = self.cache.get(f"user:{user_id}")
        if cached_user:
            return cached_user
        
        # Cache miss, get from database
        user = self.database.get(user_id)
        if user:
            # Store in cache for future requests
            self.cache.set(f"user:{user_id}", user)
        
        return user
""",
        "benefits": ["Reduced latency", "Lower database load", "Improved user experience"]
    },
    
    "message_queue": {
        "definition": "Asynchronous communication between services",
        "patterns": {
            "point_to_point": "One producer, one consumer",
            "publish_subscribe": "One producer, multiple consumers",
            "request_reply": "Synchronous-like communication over async"
        },
        "implementations": {
            "rabbitmq": "Feature-rich, AMQP protocol",
            "apache_kafka": "High-throughput, distributed streaming",
            "amazon_sqs": "Managed queue service",
            "redis_pub_sub": "Simple pub/sub with Redis"
        },
        "code_example": """
import queue
import threading
from typing import Callable, Any

class MessageQueue:
    \"\"\"Simple message queue implementation\"\"\"
    
    def __init__(self):
        self._queue = queue.Queue()
        self._subscribers = {}
        self._running = True
    
    def publish(self, topic: str, message: Any):
        \"\"\"Publish message to topic\"\"\"
        self._queue.put((topic, message))
    
    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        \"\"\"Subscribe to topic with callback\"\"\"
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
    
    def start_consuming(self):
        \"\"\"Start consuming messages\"\"\"
        def consume():
            while self._running:
                try:
                    topic, message = self._queue.get(timeout=1)
                    if topic in self._subscribers:
                        for callback in self._subscribers[topic]:
                            try:
                                callback(message)
                            except Exception as e:
                                print(f"Error in callback: {e}")
                    self._queue.task_done()
                except queue.Empty:
                    continue
        
        thread = threading.Thread(target=consume)
        thread.daemon = True
        thread.start()
        return thread
    
    def stop(self):
        self._running = False

# Example usage
mq = MessageQueue()

def user_registered_handler(message):
    print(f"Send welcome email to {message['email']}")

def user_registered_analytics(message):
    print(f"Track user registration: {message['user_id']}")

mq.subscribe("user.registered", user_registered_handler)
mq.subscribe("user.registered", user_registered_analytics)

consumer_thread = mq.start_consuming()

# Publish event
mq.publish("user.registered", {"user_id": "123", "email": "user@example.com"})
""",
        "benefits": ["Decoupling", "Scalability", "Reliability", "Asynchronous processing"]
    }
}

# System Design Patterns
SYSTEM_DESIGN_PATTERNS = {
    "microservices": {
        "definition": "Architecture pattern that structures application as collection of loosely coupled services",
        "characteristics": [
            "Business capability focused",
            "Decentralized governance",
            "Failure isolation",
            "Technology diversity"
        ],
        "benefits": ["Independent deployment", "Technology flexibility", "Team autonomy", "Fault isolation"],
        "challenges": ["Network complexity", "Data consistency", "Testing complexity", "Operational overhead"],
        "when_to_use": [
            "Large, complex applications",
            "Multiple teams working independently",
            "Need for different technologies per service",
            "Requirement for independent scaling"
        ]
    },
    
    "event_sourcing": {
        "definition": "Store all changes to application state as sequence of events",
        "benefits": ["Complete audit trail", "Temporal queries", "Event replay", "Scalability"],
        "challenges": ["Complexity", "Storage overhead", "Eventual consistency"],
        "use_cases": ["Financial systems", "Collaborative applications", "Analytics"]
    },
    
    "cqrs": {
        "definition": "Command Query Responsibility Segregation - separate read and write operations",
        "benefits": ["Optimized read/write models", "Scalability", "Security", "Complex business logic"],
        "challenges": ["Complexity", "Eventual consistency", "Code duplication"],
        "use_cases": ["Complex domain logic", "High read/write ratios", "Event sourcing"]
    },
    
    "circuit_breaker": {
        "definition": "Prevent cascading failures by monitoring for failures and blocking requests when threshold is reached",
        "states": {
            "closed": "Normal operation, requests flow through",
            "open": "Failure threshold reached, requests fail fast",
            "half_open": "Testing if service has recovered"
        },
        "code_example": """
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = 1
    OPEN = 2
    HALF_OPEN = 3

class CircuitBreaker:
    \"\"\"Circuit Breaker Pattern Implementation\"\"\"
    
    def __init__(self, failure_threshold=5, timeout=60, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        \"\"\"Execute function with circuit breaker protection\"\"\"
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        \"\"\"Reset circuit breaker on successful call\"\"\"
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        \"\"\"Handle failure and potentially open circuit\"\"\"
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Example usage
def unreliable_service():
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Service failure")
    return "Success"

cb = CircuitBreaker(failure_threshold=3)

for i in range(10):
    try:
        result = cb.call(unreliable_service)
        print(f"Call {i}: {result}")
    except Exception as e:
        print(f"Call {i}: {e}")
""",
        "benefits": ["Prevents cascading failures", "Fast failure", "System stability"],
        "use_cases": ["Service-to-service communication", "External API calls", "Database connections"]
    }
}

# Scalability Patterns
SCALABILITY_PATTERNS = {
    "horizontal_scaling": {
        "definition": "Scale out by adding more machines",
        "techniques": ["Load balancing", "Sharding", "Replication", "Microservices"],
        "benefits": ["Linear scaling", "Fault tolerance", "Cost effective for large scale"],
        "challenges": ["Data consistency", "Network complexity", "Management overhead"]
    },
    
    "vertical_scaling": {
        "definition": "Scale up by adding more power to existing machines",
        "techniques": ["CPU upgrade", "RAM increase", "SSD storage", "Network bandwidth"],
        "benefits": ["Simple implementation", "No code changes", "Strong consistency"],
        "challenges": ["Hardware limits", "Single point of failure", "Expensive"]
    },
    
    "database_scaling": {
        "read_replicas": "Separate read and write operations",
        "sharding": "Partition data across multiple databases",
        "federation": "Split databases by function",
        "denormalization": "Trade-off consistency for performance"
    }
}

def get_lld_pattern_info(pattern_name):
    """Get information about an LLD pattern."""
    return LLD_PATTERNS.get(pattern_name.lower())

def get_hld_component_info(component_name):
    """Get information about an HLD component."""
    return HLD_COMPONENTS.get(component_name.lower())

def get_system_pattern_info(pattern_name):
    """Get information about a system design pattern."""
    return SYSTEM_DESIGN_PATTERNS.get(pattern_name.lower())

def get_scalability_info(pattern_name):
    """Get information about scalability patterns."""
    return SCALABILITY_PATTERNS.get(pattern_name.lower())

def list_system_design_topics():
    """List all available system design topics."""
    return {
        "lld_patterns": list(LLD_PATTERNS.keys()),
        "hld_components": list(HLD_COMPONENTS.keys()),
        "system_patterns": list(SYSTEM_DESIGN_PATTERNS.keys()),
        "scalability_patterns": list(SCALABILITY_PATTERNS.keys())
    }