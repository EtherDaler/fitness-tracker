from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, DATETIME, Boolean, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base

class Transactions(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True, unique=True)
    price = Column(Float)                  # Цена
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    user_id = Column(Integer, index=True, nullable=True)
    tax = Column(String, nullable=True)
    eMail = Column(String, nullable=True)
    paymentMethod = Column(String, nullable=True) # Метод оплаты
    incCurrLabel = Column(String, nullable=True)  # Валюта оплаты
    datetime = Column(TIMESTAMP, nullable=True)  # Дата и время совершения операции
    status = Column(String, default="Создан", nullable=True) # Статус платежа
    finished = Column(Boolean, default=False) # Статус завершения платежа

    def __repr__(self):
        return f"Transaction(id={self.id}, price={self.price}), description={self.description})"