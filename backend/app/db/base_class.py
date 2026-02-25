from typing import Any
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.declarative import declared_attr

class Base(DeclarativeBase):
    """
    SQLAlchemy 声明式基类 (Declarative Base)
    所有数据库模型都应继承此类。
    """
    id: Any
    __name__: str
    
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        """
        自动生成表名 (Auto-generate Table Name)
        默认使用类名的小写形式作为表名。
        """
        return cls.__name__.lower()
