"""
Service để query và quản lý Intent từ database
"""
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from models.business_intent import BusinessIntent
from models.intent import Intent
import logging

logger = logging.getLogger(__name__)


class IntentService:
    """Service để làm việc với Intent"""
    
    @staticmethod
    def get_active_intents_by_business(db: Session, business_id: int) -> List[Dict]:
        """
        Lấy danh sách các intent đang bật của business
        
        Args:
            db: Database session
            business_id: ID của business
            
        Returns:
            List[Dict]: Danh sách intent với thông tin type và description
        """
        try:
            # Query business_intent với status = 1 và business_id
            business_intents = db.query(BusinessIntent).filter(
                BusinessIntent.business_id == business_id,
                BusinessIntent.status == 1
            ).all()

            print(str(business_intents))  # Nó sẽ in ra câu SELECT ... WHERE ...
            
            # Lấy thông tin intent tương ứng
            intents_info = []
            for bi in business_intents:
                if bi.intent_id:
                    intent = db.query(Intent).filter(
                        Intent.id == bi.intent_id,
                        Intent.status == 1
                    ).first()
                    
                    if intent:
                        intents_info.append({
                            'id': intent.id,
                            'name': intent.name,
                            'type': intent.type,
                            'description': intent.description,
                            'template': bi.template_override if bi.template_override else intent.template
                        })
            
            logger.info(f"Tìm thấy {len(intents_info)} intent đang bật cho business_id={business_id}")
            return intents_info
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy intent cho business_id={business_id}: {str(e)}")
            return []
    
    @staticmethod
    def get_intent_by_type(db: Session, intent_type: str) -> Optional[Intent]:
        """
        Lấy intent theo type
        
        Args:
            db: Database session
            intent_type: Type của intent
            
        Returns:
            Intent hoặc None
        """
        try:
            intent = db.query(Intent).filter(
                Intent.type == intent_type,
                Intent.status == 1
            ).first()
            return intent
        except Exception as e:
            logger.error(f"Lỗi khi lấy intent theo type={intent_type}: {str(e)}")
            return None



