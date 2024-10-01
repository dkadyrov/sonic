from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from .base import Base


class Subject(Base):  # type: ignore
    """Subject object"""

    __tablename__ = "subject"
    id = Column(Integer, primary_key=True)
    """int: Subject object database ID"""

    name = Column(String)
    """str: Subject name"""

    samples = relationship(
        "sonic.database.sample.Sample",
        back_populates="subject",
        enable_typechecks=False,
    )
    """list: list of Sample objects featuring the subject"""

    # events = relationship("sonic.database.event.EventSubject", back_populates="subject", enable_typechecks=False)
    events = relationship(
        "sonic.database.event.Event", back_populates="subject", enable_typechecks=False
    )
    """list: list of Run objects featuring the subject"""

    __mapper_args__ = {"polymorphic_identity": "subject"}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Subject: {self.name}"
