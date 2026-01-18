"""Models package"""
from models.base import Base

# Import tất cả models để đảm bảo chúng được đăng ký với Base
from models.product import Product
from models.business import Business
from models.intent import Intent
from models.business_intent import BusinessIntent

__all__ = ['Base', 'Product', 'Business', 'Intent', 'BusinessIntent']
